"""Tests for the direct-beam obstruction gate (#991).

Two surfaces:
  1. Gate application inside ``SolarCalculator.calculate_unit_potential_4d``:
     with ``coordinator._critical_elev_per_facade[f] = crit``, the
     corresponding ``pot_*_dir`` must be zeroed whenever
     ``sun_elev > crit``.  Diffuse is unaffected.  ``None`` = no gate
     (bit-identical to pre-#991 output).
  2. Fit routine ``LearningManager.fit_solar_obstruction``: synthetic
     step-residual data over the elevation range should produce the
     correct ``critical_elev``; flat residuals should leave
     ``learned=False``; sample-count and SSE-improvement gates must be
     enforced.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from custom_components.heating_analytics.learning import LearningManager
from custom_components.heating_analytics.solar import SolarCalculator
from custom_components.heating_analytics.const import (
    OBSTRUCTION_FIT_MIN_SAMPLES_PER_SIDE,
)


def _calc(critical_elev: dict | None = None) -> SolarCalculator:
    coord = MagicMock()
    coord._critical_elev_per_facade = critical_elev
    return SolarCalculator(coord)


# ---------------------------------------------------------------------------
# Gate application in calculate_unit_potential_4d
# ---------------------------------------------------------------------------


def test_gate_none_is_no_op():
    """``_critical_elev_per_facade = None`` → bit-identical to pre-gate."""
    calc = _calc(critical_elev=None)
    out = calc.calculate_unit_potential_4d(
        entity_id="x", dni=500.0, dhi=100.0,
        sun_elev_deg=45.0, sun_azimuth_deg=180.0,
        screen_config=(False, False, False), correction_percent=100.0,
    )
    assert abs(out[0] - 353.5534) < 0.5  # south direct
    assert abs(out[3] - 50.0) < 1e-9     # diffuse


def test_gate_all_none_is_no_op():
    """``{s, e, w} = None`` → identical to no-gate output."""
    calc = _calc(critical_elev={"s": None, "e": None, "w": None})
    out = calc.calculate_unit_potential_4d(
        entity_id="x", dni=500.0, dhi=100.0,
        sun_elev_deg=45.0, sun_azimuth_deg=180.0,
        screen_config=(False, False, False), correction_percent=100.0,
    )
    assert abs(out[0] - 353.5534) < 0.5


def test_south_gate_zeroes_above_cutoff():
    """``crit_s = 30°``, sun at elev=45° → ``pot_s = 0``; diffuse untouched."""
    calc = _calc(critical_elev={"s": 30.0, "e": None, "w": None})
    out = calc.calculate_unit_potential_4d(
        entity_id="x", dni=500.0, dhi=100.0,
        sun_elev_deg=45.0, sun_azimuth_deg=180.0,
        screen_config=(False, False, False), correction_percent=100.0,
    )
    assert out[0] == 0.0
    assert abs(out[3] - 50.0) < 1e-9


def test_south_gate_passes_below_cutoff():
    """``crit_s = 30°``, sun at elev=20° → south direct unchanged."""
    calc = _calc(critical_elev={"s": 30.0, "e": None, "w": None})
    out = calc.calculate_unit_potential_4d(
        entity_id="x", dni=500.0, dhi=100.0,
        sun_elev_deg=20.0, sun_azimuth_deg=180.0,
        screen_config=(False, False, False), correction_percent=100.0,
    )
    expected_s = 500.0 * math.cos(math.radians(20.0))
    assert abs(out[0] - expected_s) < 0.5


def test_gates_are_independent_per_facade():
    """``crit_s = 30°`` does NOT gate east or west."""
    calc = _calc(critical_elev={"s": 30.0, "e": None, "w": None})
    out = calc.calculate_unit_potential_4d(
        entity_id="x", dni=500.0, dhi=100.0,
        sun_elev_deg=45.0, sun_azimuth_deg=90.0,  # sun due east
        screen_config=(False, False, False), correction_percent=100.0,
    )
    # South direct is 0 (cos(90)=0 already), east must carry full signal.
    expected_e = 500.0 * math.cos(math.radians(45.0))
    assert out[0] == 0.0
    assert abs(out[1] - expected_e) < 0.5


# ---------------------------------------------------------------------------
# Fit routine: fit_solar_obstruction
# ---------------------------------------------------------------------------


def _make_fit_coord():
    """Build a coordinator with enough structure for the fit to run."""
    coord = MagicMock()
    coord.latitude = 52.0
    coord.longitude = 5.0
    coord.timezone = "UTC"
    coord.energy_sensors = ["sensor.heater1"]
    coord._solar_affected_set = None
    coord._critical_elev_per_facade = {"s": None, "e": None, "w": None}
    coord.screen_config = (False, False, False)
    coord.screen_config_for_entity = MagicMock(
        side_effect=lambda _eid: (False, False, False)
    )
    # Pre-learned 4D coefficients on the sole entity (heating regime).
    coord._solar_coefficients_4d_per_unit = {
        "sensor.heater1": {
            "heating": {
                "s": 0.0015, "e": 0.0010, "w": 0.0010,
                "diffuse": 0.0005, "learned": True,
            },
        },
    }
    solar = SolarCalculator(coord)
    coord.solar = solar
    return coord, solar


def _build_log_explicit(
    sun_positions: list[tuple[float, float]],
    *,
    true_crit_s: float | None = None,
    base: float = 3.0,
    sensor_id: str = "sensor.heater1",
) -> tuple[list[dict], dict[str, tuple[float, float]]]:
    """Synthesise hourly_log entries with explicit (elev, az) per row.

    ``sun_positions`` is a list of ``(elev_deg, az_deg)`` pairs; one
    log entry per pair, monotonically timestamped.  Returns ``(entries,
    sun_pos_by_ts)`` where the second dict maps each entry's timestamp
    string to its sun position — feed it to a stubbed
    ``get_approx_sun_pos`` so the fit reads back the planted values.
    """
    from custom_components.heating_analytics.solar import resolve_dni_dhi

    entries: list[dict] = []
    sun_pos_by_ts: dict[str, tuple[float, float]] = {}
    c_s, c_e, c_w, c_d = 0.0015, 0.0010, 0.0010, 0.0005
    start = datetime(2026, 4, 1, 0, 0, 0)

    for i, (elev, az) in enumerate(sun_positions):
        ts_dt = start + timedelta(hours=i)
        ts_str = ts_dt.strftime("%Y-%m-%dT%H:%M:%S")
        sun_pos_by_ts[ts_str] = (elev, az)

        dni, dhi, _src = resolve_dni_dhi(
            700.0, 150.0, None, None, elev,
            ts_dt.timetuple().tm_yday,
        )
        cos_elev = math.cos(math.radians(elev))
        az_rad = math.radians(az)
        t = 1.0
        pot_s = max(0.0, dni) * cos_elev * max(0.0, -math.cos(az_rad)) * t
        pot_e = max(0.0, dni) * cos_elev * max(0.0, math.sin(az_rad)) * t
        pot_w = max(0.0, dni) * cos_elev * max(0.0, -math.sin(az_rad)) * t
        pot_d = max(0.0, dhi) * 0.5 * (t + t + t) / 3.0

        if true_crit_s is not None and elev > true_crit_s:
            pot_s_actual = 0.0
        else:
            pot_s_actual = pot_s
        impact = (
            c_s * pot_s_actual + c_e * pot_e + c_w * pot_w + c_d * pot_d
        )
        actual = max(0.0, base - impact)
        entries.append({
            "timestamp": ts_str,
            "dni": 700.0,
            "dhi": 150.0,
            "ghi_wm2": None,
            "cloud_coverage": None,
            "correction_percent": 100.0,
            "auxiliary_active": False,
            "unit_modes": {sensor_id: "heating"},
            "unit_breakdown": {sensor_id: actual},
            "unit_expected_base": {sensor_id: base},
            "solar_dominant_entities": [],
        })
    return entries, sun_pos_by_ts


def _south_dominant_positions(elevs: list[float]) -> list[tuple[float, float]]:
    """List of (elev, az=180°) — sun straight south at given elevations."""
    return [(elev, 180.0) for elev in elevs]


def _stub_get_approx_sun_pos(sun_pos_by_ts: dict[str, tuple[float, float]]):
    """Build a get_approx_sun_pos replacement that reads from the planted
    timestamps.  The fit passes ``ts_dt + 30min`` to the lookup, so the
    stub matches on the ``ts_dt`` portion (the entry's timestamp string
    is ``ts_dt`` not ``mid_dt``; we reconstruct mid_dt → entry_dt).
    """
    def _lookup(mid_dt: datetime) -> tuple[float, float]:
        entry_dt = mid_dt - timedelta(minutes=30)
        key = entry_dt.strftime("%Y-%m-%dT%H:%M:%S")
        if key in sun_pos_by_ts:
            return sun_pos_by_ts[key]
        return (0.0, 180.0)  # below horizon — skipped
    return _lookup


def test_fit_detects_step_at_correct_elevation():
    """South sun planted at uniform elevations 10°→70°, overhang gate
    at 30°.  Fit should locate the step within ±5°.
    """
    coord, solar = _make_fit_coord()
    # Dense, uniform south-sun samples across full elevation range.
    elevs = list(range(10, 71)) * 4  # 4 samples per degree, 244 total
    positions = _south_dominant_positions([float(e) for e in elevs])
    entries, sun_pos_by_ts = _build_log_explicit(
        positions, true_crit_s=30.0,
    )
    solar.get_approx_sun_pos = _stub_get_approx_sun_pos(sun_pos_by_ts)
    lm = LearningManager()
    result = lm.fit_solar_obstruction(
        hourly_log=entries, coordinator=coord, dry_run=False,
    )
    s = result["s"]
    assert s["learned"] is True, s
    assert 25.0 <= s["best_critical_elev"] <= 35.0, s["best_critical_elev"]
    assert s["sse_improvement_ratio"] > 0.10
    assert s["n_below_best"] >= OBSTRUCTION_FIT_MIN_SAMPLES_PER_SIDE
    assert s["n_above_best"] >= OBSTRUCTION_FIT_MIN_SAMPLES_PER_SIDE
    # State written.
    assert coord._critical_elev_per_facade["s"] == s["best_critical_elev"]


def test_fit_flat_residual_leaves_unlearned():
    """No obstruction in synthetic data → ``learned=False``."""
    coord, solar = _make_fit_coord()
    elevs = list(range(10, 71)) * 4
    positions = _south_dominant_positions([float(e) for e in elevs])
    entries, sun_pos_by_ts = _build_log_explicit(
        positions, true_crit_s=None,
    )
    solar.get_approx_sun_pos = _stub_get_approx_sun_pos(sun_pos_by_ts)
    lm = LearningManager()
    result = lm.fit_solar_obstruction(
        hourly_log=entries, coordinator=coord, dry_run=False,
    )
    assert result["s"]["learned"] is False, result["s"]
    assert coord._critical_elev_per_facade["s"] is None


def test_fit_dry_run_does_not_write():
    """``dry_run=True`` → diagnostics returned, state untouched."""
    coord, solar = _make_fit_coord()
    elevs = list(range(10, 71)) * 4
    positions = _south_dominant_positions([float(e) for e in elevs])
    entries, sun_pos_by_ts = _build_log_explicit(
        positions, true_crit_s=30.0,
    )
    solar.get_approx_sun_pos = _stub_get_approx_sun_pos(sun_pos_by_ts)
    lm = LearningManager()
    result = lm.fit_solar_obstruction(
        hourly_log=entries, coordinator=coord, dry_run=True,
    )
    assert result["dry_run"] is True
    assert result["s"]["learned"] is True
    # State NOT written.
    assert coord._critical_elev_per_facade["s"] is None


def test_fit_insufficient_samples_skips():
    """Single sample → ``insufficient_samples`` skip per facade."""
    coord, solar = _make_fit_coord()
    # Only 5 south samples — well below 2 × MIN_SAMPLES_PER_SIDE = 20.
    positions = _south_dominant_positions([20.0, 30.0, 40.0, 50.0, 60.0])
    entries, sun_pos_by_ts = _build_log_explicit(
        positions, true_crit_s=30.0,
    )
    solar.get_approx_sun_pos = _stub_get_approx_sun_pos(sun_pos_by_ts)
    lm = LearningManager()
    result = lm.fit_solar_obstruction(
        hourly_log=entries, coordinator=coord, dry_run=False,
    )
    assert result["s"]["skip_reason"] == "insufficient_samples"
