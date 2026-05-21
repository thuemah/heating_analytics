"""Tests for ``_compute_total_power_4d_divergence_report`` (#962).

This block walks ``hourly_log`` and replays the 3D and 4D
``calculate_total_power`` variants side-by-side per hour, aggregating
per-cloud-regime divergence statistics.  Strict diagnostic — reads
only, writes nothing.

Tests use a ``MagicMock`` coordinator with a stubbed ``statistics``
manager so the per-hour replay is fully deterministic.  This mirrors
the pattern in ``tests/test_solar_diagnose.py`` rather than spinning
up a full ``HeatingDataCoordinator``: the divergence block's contract
is the *aggregation logic*, not the underlying ``calculate_total_power*``
implementation (those are covered by ``tests/test_total_power_4d_shadow.py``).
"""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from custom_components.heating_analytics.diagnostics import DiagnosticsEngine


def _make_log_entry(
    *,
    ts: str,
    dni: float | None = 500.0,
    dhi: float | None = 100.0,
    temp: float = 10.0,
    effective_wind: float = 2.0,
    correction_percent: float = 100.0,
    solar_factor: float = 0.5,
    solar_vector_s: float = 0.4,
    solar_vector_e: float = 0.1,
    solar_vector_w: float = 0.1,
    auxiliary_active: bool = False,
    unit_modes: dict | None = None,
) -> dict:
    entry = {
        "timestamp": ts,
        "temp": temp,
        "effective_wind": effective_wind,
        "correction_percent": correction_percent,
        "solar_factor": solar_factor,
        "solar_vector_s": solar_vector_s,
        "solar_vector_e": solar_vector_e,
        "solar_vector_w": solar_vector_w,
        "auxiliary_active": auxiliary_active,
        "unit_modes": unit_modes or {},
    }
    if dni is not None:
        entry["dni"] = dni
    if dhi is not None:
        entry["dhi"] = dhi
    return entry


def _make_coord(
    hourly_log: list[dict],
    *,
    r3_total: float = 2.0,
    r3_solar_applied: float = 0.3,
    r3_solar_heating_applied: float = 0.3,
    r3_global_base: float = 2.5,
    r4_total: float = 2.1,
    r4_solar_applied: float = 0.2,
    r4_solar_heating_applied: float = 0.2,
):
    """Build a minimal MagicMock coordinator wired for divergence-report tests.

    ``calculate_total_power`` and ``calculate_total_power_4d`` are
    stubbed to return constant payloads regardless of inputs.  Callers
    that need per-hour variation can replace ``coord.statistics.*``
    after construction.
    """
    coord = MagicMock()
    coord._hourly_log = hourly_log
    coord.solar = MagicMock()
    coord.solar.get_approx_sun_pos = MagicMock(return_value=(45.0, 180.0))

    coord.statistics = MagicMock()
    coord.statistics.calculate_total_power = MagicMock(return_value={
        "total_kwh": r3_total,
        "global_base_kwh": r3_global_base,
        "breakdown": {
            "solar_reduction_kwh": r3_solar_applied,
            "solar_heating_applied_kwh": r3_solar_heating_applied,
        },
    })
    coord.statistics.calculate_total_power_4d = MagicMock(return_value={
        "total_kwh": r4_total,
        "global_base_kwh": r3_global_base,
        "breakdown": {
            "solar_reduction_kwh": r4_solar_applied,
            "solar_heating_applied_kwh": r4_solar_heating_applied,
        },
    })
    return coord


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


def test_divergence_empty_log_returns_unavailable():
    """No hourly_log entries → ``available=False``."""
    coord = _make_coord([])
    result = DiagnosticsEngine(coord)._compute_total_power_4d_divergence_report(30)
    assert result["available"] is False
    assert "reason" in result
    assert result["n_eligible_hours"] == 0
    assert result["regime_counts"] == {"clear": 0, "broken": 0, "overcast": 0}


def test_divergence_skips_entries_without_dni_dhi():
    """Entries missing dni/dhi → counted in n_skipped_missing_fields."""
    today = datetime.now()
    log = [
        _make_log_entry(ts=(today - timedelta(hours=i + 1)).isoformat(),
                        dni=None, dhi=None)
        for i in range(5)
    ]
    coord = _make_coord(log)
    result = DiagnosticsEngine(coord)._compute_total_power_4d_divergence_report(30)
    assert result["available"] is False
    assert result["n_skipped_missing_fields"] == 5
    assert result["n_eligible_hours"] == 0


def test_divergence_regime_classification_clear_broken_overcast():
    """One synthetic hour per regime → regime_counts each show 1."""
    today = datetime.now()
    # clear: DNI > 400 AND DHI/DNI < 0.3 → (800, 100)
    # broken: not clear, not overcast → (200, 100)
    # overcast: DNI < 50 → (20, 80)
    log = [
        _make_log_entry(ts=(today - timedelta(hours=1)).isoformat(),
                        dni=800.0, dhi=100.0),
        _make_log_entry(ts=(today - timedelta(hours=2)).isoformat(),
                        dni=200.0, dhi=100.0),
        _make_log_entry(ts=(today - timedelta(hours=3)).isoformat(),
                        dni=20.0, dhi=80.0),
    ]
    coord = _make_coord(log)
    result = DiagnosticsEngine(coord)._compute_total_power_4d_divergence_report(30)
    assert result["available"] is True
    assert result["regime_counts"] == {"clear": 1, "broken": 1, "overcast": 1}
    assert result["n_eligible_hours"] == 3


def test_divergence_zero_when_4d_matches_3d():
    """When 4D returns identical solar to 3D, median_abs_delta is 0."""
    today = datetime.now()
    log = [
        _make_log_entry(ts=(today - timedelta(hours=i + 1)).isoformat(),
                        dni=200.0, dhi=100.0)  # all broken
        for i in range(6)
    ]
    coord = _make_coord(
        log,
        r3_total=2.0, r4_total=2.0,
        r3_solar_applied=0.3, r4_solar_applied=0.3,
        r3_solar_heating_applied=0.3, r4_solar_heating_applied=0.3,
    )
    result = DiagnosticsEngine(coord)._compute_total_power_4d_divergence_report(30)
    assert result["per_regime"]["broken"]["n_hours"] == 6
    assert result["per_regime"]["broken"]["median_abs_delta_solar_applied_kwh"] == 0.0
    assert result["per_regime"]["broken"]["median_abs_delta_total_kwh"] == 0.0


def test_divergence_verdict_thresholds():
    """Verdict transitions across the aligned / modest / divergent bands.

    All thresholds are evaluated on the broken regime.  Synthesise
    enough hours per regime so the ``insufficient_data`` short-circuit
    does not fire (≥ 5 hours each), and ≥ 20 broken hours so the
    threshold gate fires.
    """
    today = datetime.now()
    # 6 clear + 6 overcast + 25 broken — broken dominates the verdict.
    log: list[dict] = []
    h = 0
    for _ in range(6):
        h += 1
        log.append(_make_log_entry(ts=(today - timedelta(hours=h)).isoformat(),
                                    dni=800.0, dhi=100.0))
    for _ in range(6):
        h += 1
        log.append(_make_log_entry(ts=(today - timedelta(hours=h)).isoformat(),
                                    dni=20.0, dhi=80.0))
    for _ in range(25):
        h += 1
        log.append(_make_log_entry(ts=(today - timedelta(hours=h)).isoformat(),
                                    dni=200.0, dhi=100.0))

    # Aligned case: |delta_total| / global_base = 0.02 / 2.5 = 0.008 < 0.05.
    coord_aligned = _make_coord(
        log, r3_total=2.0, r4_total=2.02, r3_global_base=2.5,
    )
    r_aligned = DiagnosticsEngine(coord_aligned)._compute_total_power_4d_divergence_report(30)
    assert r_aligned["verdict"] == "4d_aligned_with_3d"

    # Diverges meaningfully: 0.5 / 2.5 = 0.20 ≥ 0.10.
    coord_div = _make_coord(
        log, r3_total=2.0, r4_total=2.5, r3_global_base=2.5,
    )
    r_div = DiagnosticsEngine(coord_div)._compute_total_power_4d_divergence_report(30)
    assert r_div["verdict"] == "4d_meaningfully_diverges_on_broken"

    # Modest: 0.2 / 2.5 = 0.08 — between 0.05 and 0.10.
    coord_mod = _make_coord(
        log, r3_total=2.0, r4_total=2.2, r3_global_base=2.5,
    )
    r_mod = DiagnosticsEngine(coord_mod)._compute_total_power_4d_divergence_report(30)
    assert r_mod["verdict"] == "4d_diverges_modestly_on_broken"


def test_divergence_defensive_against_per_hour_errors():
    """One log entry triggering an exception is counted, others continue."""
    today = datetime.now()
    log = [
        _make_log_entry(ts=(today - timedelta(hours=i + 1)).isoformat(),
                        dni=200.0, dhi=100.0)
        for i in range(5)
    ]
    coord = _make_coord(log)

    # Make the 4D call raise on every other invocation.  The block must
    # count those into n_skipped_errors but still process the rest.
    call_count = {"n": 0}
    real_payload = {
        "total_kwh": 2.1,
        "global_base_kwh": 2.5,
        "breakdown": {
            "solar_reduction_kwh": 0.2,
            "solar_heating_applied_kwh": 0.2,
        },
    }

    def maybe_raise(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] % 2 == 0:
            raise RuntimeError("synthetic per-hour error")
        return real_payload

    coord.statistics.calculate_total_power_4d = MagicMock(side_effect=maybe_raise)

    result = DiagnosticsEngine(coord)._compute_total_power_4d_divergence_report(30)
    # 5 calls; 2 raise (calls #2 and #4) → 3 successful.
    assert result["n_skipped_errors"] == 2
    assert result["n_eligible_hours"] == 3


def test_divergence_block_mounted_in_diagnose_solar_output():
    """The block must be a top-level key in diagnose_solar's return dict.

    We can verify the wiring on a populated coordinator without
    exercising the full diagnose_solar surface — call the method
    directly and confirm it returns a dict with the expected shape.
    Cross-check the mount line by direct lookup in the source module
    constants — the block must be listed as one of the result keys.
    """
    today = datetime.now()
    log = [
        _make_log_entry(ts=(today - timedelta(hours=1)).isoformat(),
                        dni=200.0, dhi=100.0),
    ]
    coord = _make_coord(log)
    engine = DiagnosticsEngine(coord)
    # Confirm the method exists and is wired into the diagnose_solar
    # return.  We cannot easily call the full diagnose_solar without
    # mocking dozens of attributes, so we both invoke the method
    # directly and grep the diagnose_solar source for the key name.
    block = engine._compute_total_power_4d_divergence_report(30)
    assert isinstance(block, dict)
    assert "available" in block

    import inspect
    src = inspect.getsource(engine.diagnose_solar)
    assert "total_power_4d_divergence" in src, (
        "Block must be wired into diagnose_solar's return dict"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
