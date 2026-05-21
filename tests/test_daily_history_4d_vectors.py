"""Tests for #982 — 4D-pipeline persistence in daily_history.

``hourly_log`` trims after retention (default 90 days) but ``daily_history``
is unbounded; without dedicated 4D fields on the daily aggregate, the 4D
pipeline's per-hour DNI/DHI and normalization delta would be lost forever
on the trim, making future Track B/C retraining against 4D coefficients
impossible.  These tests pin the persistence shape and — critically —
the missing-key handling: an hour where 4D didn't fire must produce
``None`` in the per-hour vector, not a phantom 0 that would bias daily
averages and sums.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from custom_components.heating_analytics.daily_processor import DailyProcessor


def _make_coord(*, solar_enabled=True, balance_point=15.0):
    coord = MagicMock()
    coord.solar_enabled = solar_enabled
    coord.balance_point = balance_point
    coord.hourly_solar_impact_kwh = MagicMock(return_value=0.0)
    return coord


def _make_log(hour: int, **overrides) -> dict:
    """Build a minimal hourly_log entry for the given hour."""
    base = {
        "timestamp": f"2026-05-17T{hour:02d}:00:00",
        "hour": hour,
        "temp": 10.0,
        "effective_wind": 3.0,
        "solar_factor": 0.3,
        "actual_kwh": 2.0,
        "tdd": 0.5,
        "solar_normalization_delta": 0.1,
    }
    base.update(overrides)
    return base


class TestDailyHistory4DVectors:
    """#982 — 4D fields persisted in daily_history aggregates and vectors."""

    def test_full_4d_day_populates_vectors_and_aggregates(self):
        """All 24 hours carry 4D data → vectors fully populated, daily totals match.

        Sums hourly ``solar_normalization_delta_4d`` to daily total and
        averages ``dni``/``dhi`` per hour.  No phantom-zero handling
        triggered because every entry has all keys.
        """
        coord = _make_coord()
        proc = DailyProcessor(coord)
        day_logs = [
            _make_log(
                h,
                dni=500.0 + h,
                dhi=80.0 + h,
                solar_normalization_delta_4d=0.05,
                solar_impact_4d_kwh=0.2,
            )
            for h in range(24)
        ]
        result = proc.aggregate_logs(day_logs)

        # Daily aggregates: sums match the per-hour data.
        assert result["solar_normalization_delta_4d"] == pytest.approx(24 * 0.05, abs=1e-6)
        assert result["solar_impact_4d_kwh"] == pytest.approx(24 * 0.2, abs=0.01)

        hv = result["hourly_vectors"]
        assert "solar_norm_delta_4d" in hv and "dni" in hv and "dhi" in hv
        # Per-hour vector population.
        for h in range(24):
            assert hv["dni"][h] == pytest.approx(500.0 + h, abs=0.01)
            assert hv["dhi"][h] == pytest.approx(80.0 + h, abs=0.01)
            assert hv["solar_norm_delta_4d"][h] == pytest.approx(0.05, abs=1e-6)

    def test_missing_4d_keys_produce_none_not_zero(self):
        """The key-presence filter is the whole point of #982.

        An hour where 4D didn't fire has NO ``dni``/``dhi``/
        ``solar_normalization_delta_4d`` keys in its hourly_log entry
        (the existing ``**({"dni": ...} if count > 0 else {})`` pattern
        at ``hourly_processor.py:1003-1053``).  ``.get(..., 0.0)`` would
        treat those hours as DNI=0, dragging the daily average down
        toward zero.  We must produce ``None`` in the per-hour vector
        for those hours instead — distinct from "DNI was measured and
        happened to be 0".
        """
        coord = _make_coord()
        proc = DailyProcessor(coord)
        # 12 hours with 4D data, 12 without.  If we used .get(..., 0.0)
        # the daily DNI average would be ~250 (half of 500); with
        # key-presence filtering, populated hours show ~500 and missing
        # hours show None.
        day_logs = []
        for h in range(24):
            if h < 12:
                day_logs.append(_make_log(
                    h,
                    dni=500.0,
                    dhi=80.0,
                    solar_normalization_delta_4d=0.05,
                    solar_impact_4d_kwh=0.2,
                ))
            else:
                day_logs.append(_make_log(h))  # No 4D keys
        result = proc.aggregate_logs(day_logs)

        hv = result["hourly_vectors"]
        # Populated hours: real values, NOT dragged toward 0.
        for h in range(12):
            assert hv["dni"][h] == pytest.approx(500.0, abs=0.01)
            assert hv["dhi"][h] == pytest.approx(80.0, abs=0.01)
            assert hv["solar_norm_delta_4d"][h] == pytest.approx(0.05, abs=1e-6)
        # Missing hours: None, NOT 0.
        for h in range(12, 24):
            assert hv["dni"][h] is None, (
                f"hour {h} has no DNI key in log — vector must be None, "
                f"not 0 (that would drag the daily average)"
            )
            assert hv["dhi"][h] is None
            assert hv["solar_norm_delta_4d"][h] is None

        # Daily aggregates: sums only the present hours.
        assert result["solar_normalization_delta_4d"] == pytest.approx(12 * 0.05, abs=1e-6)
        assert result["solar_impact_4d_kwh"] == pytest.approx(12 * 0.2, abs=0.01)

    def test_no_4d_data_at_all_aggregates_are_none(self):
        """A day with zero 4D data → daily aggregates are None, not 0.

        Distinguishes "no 4D data" from "4D data summed to zero"; future
        Track B/C 4D-retrain code branches on that signal.
        """
        coord = _make_coord()
        proc = DailyProcessor(coord)
        day_logs = [_make_log(h) for h in range(24)]
        result = proc.aggregate_logs(day_logs)

        assert result["solar_normalization_delta_4d"] is None
        assert result["solar_impact_4d_kwh"] is None
        hv = result["hourly_vectors"]
        assert all(v is None for v in hv["dni"])
        assert all(v is None for v in hv["dhi"])
        assert all(v is None for v in hv["solar_norm_delta_4d"])

    def test_solar_disabled_omits_4d_vector_keys(self):
        """When solar isn't configured, the 4D vector keys are absent.

        Mirrors the existing 3D pattern: ``solar_rad`` and
        ``solar_norm_delta`` are only present under the ``solar_enabled``
        gate; the 4D additions live behind the same gate.
        """
        coord = _make_coord(solar_enabled=False)
        proc = DailyProcessor(coord)
        day_logs = [
            _make_log(
                h,
                dni=500.0,
                dhi=80.0,
                solar_normalization_delta_4d=0.05,
            )
            for h in range(24)
        ]
        result = proc.aggregate_logs(day_logs)

        hv = result["hourly_vectors"]
        assert "solar_rad" not in hv
        assert "solar_norm_delta" not in hv
        assert "dni" not in hv
        assert "dhi" not in hv
        assert "solar_norm_delta_4d" not in hv

    def test_phantom_zero_regression_dni_dhi(self):
        """Regression pin: explicit assertion that ``.get(k, 0.0)``-style
        averaging would be detectably wrong.

        A day where DNI is logged on every hour BUT one (e.g. weather
        entity briefly unavailable) must not bias the working hours
        toward 0.  Each populated hour must report its own value.
        """
        coord = _make_coord()
        proc = DailyProcessor(coord)
        day_logs = []
        for h in range(24):
            if h == 12:
                day_logs.append(_make_log(h))  # No DNI key this hour
            else:
                day_logs.append(_make_log(h, dni=700.0, dhi=100.0,
                                          solar_normalization_delta_4d=0.05))
        result = proc.aggregate_logs(day_logs)

        hv = result["hourly_vectors"]
        # 23 populated hours read 700, not some pulled-down average.
        for h in range(24):
            if h == 12:
                assert hv["dni"][h] is None
            else:
                assert hv["dni"][h] == pytest.approx(700.0, abs=0.01)


class TestDailyTotalDeltaPreference968:
    """#968 — daily ``solar_normalization_delta`` total prefers 4D per-entry.

    De-gated from ``experimental_4d_primary`` — aggregation paths consume
    whichever delta the hourly_log entry carries, with 4D winning when
    both are present.  Mixed-hour days produce a hybrid sum (4D where
    available, 3D otherwise).
    """

    def test_daily_total_prefers_4d_delta_when_present(self):
        """All entries carry both fields → daily total reflects 4D values."""
        coord = _make_coord()
        proc = DailyProcessor(coord)
        day_logs = [
            _make_log(
                h,
                solar_normalization_delta=0.10,       # 3D
                solar_normalization_delta_4d=0.07,    # 4D — should win
            )
            for h in range(24)
        ]
        result = proc.aggregate_logs(day_logs)
        # Daily total reflects 4D, not 3D.
        assert result["solar_normalization_delta"] == pytest.approx(24 * 0.07, abs=1e-6)

    def test_daily_total_falls_back_to_3d_when_4d_absent(self):
        """Entries carry only 3D → daily total reflects 3D values."""
        coord = _make_coord()
        proc = DailyProcessor(coord)
        day_logs = [_make_log(h, solar_normalization_delta=0.12) for h in range(24)]
        result = proc.aggregate_logs(day_logs)
        assert result["solar_normalization_delta"] == pytest.approx(24 * 0.12, abs=1e-6)

    def test_daily_total_mixed_4d_3d_per_entry(self):
        """Half-and-half hours → hybrid sum: 4D where present, 3D otherwise."""
        coord = _make_coord()
        proc = DailyProcessor(coord)
        day_logs = []
        for h in range(24):
            if h < 12:
                day_logs.append(_make_log(
                    h,
                    solar_normalization_delta=0.10,
                    solar_normalization_delta_4d=0.07,
                ))
            else:
                day_logs.append(_make_log(h, solar_normalization_delta=0.10))
        result = proc.aggregate_logs(day_logs)
        # 12 hours of 0.07 (4D) + 12 hours of 0.10 (3D fallback).
        expected = 12 * 0.07 + 12 * 0.10
        assert result["solar_normalization_delta"] == pytest.approx(expected, abs=1e-6)
