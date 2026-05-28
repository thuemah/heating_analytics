"""Tests for the #1020 obstruction-fit hardening.

Acceptance-criteria coverage:

* Multi-window consensus — orchestrator-level (see
  ``test_fit_solar_obstruction_handler.py``).
* Plausibility rejection — boundary outside the side's range gets
  ``applicable=False`` with reason ``physically_implausible`` and is
  not surfaced as a suggestion.
* Cooling-floor skip — heating-only HIGH-shape data is honestly
  skipped with ``insufficient_cooling_for_high_regime``.
* Single-side trust — strong LOW signal with no HIGH data still
  produces a LOW suggestion (the #1016 opposite-side gate is gone).
* Auto-write removal — passing gate never mutates
  ``_critical_elev_per_facade_per_unit``.
* SSE threshold raised to 0.30 — borderline 0.15 improvement no
  longer passes.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from custom_components.heating_analytics.learning import (
    LearningManager,
    _boundary_applicable,
    _within_plausibility_range,
)
from custom_components.heating_analytics.solar import SolarCalculator
from custom_components.heating_analytics.const import (
    OBSTRUCTION_FIT_SSE_IMPROVEMENT_THRESHOLD,
    OBSTRUCTION_LOW_PLAUSIBLE_RANGE,
    OBSTRUCTION_HIGH_PLAUSIBLE_RANGE,
    OBSTRUCTION_FIT_MIN_COOLING_SAMPLES_FOR_HIGH,
)


# ---------------------------------------------------------------------------
# Constant-level sanity
# ---------------------------------------------------------------------------


def test_sse_threshold_raised_to_030():
    """Issue acceptance: threshold raised from 0.10 to 0.30."""
    assert OBSTRUCTION_FIT_SSE_IMPROVEMENT_THRESHOLD == 0.30


def test_plausibility_ranges_match_issue_spec():
    """Issue acceptance: LOW ∈ [2°, 20°], HIGH ∈ [20°, 60°]."""
    assert OBSTRUCTION_LOW_PLAUSIBLE_RANGE == (2.0, 20.0)
    assert OBSTRUCTION_HIGH_PLAUSIBLE_RANGE == (20.0, 60.0)


def test_cooling_floor_default_50():
    """Issue acceptance: HIGH-gate requires ≥ 50 cooling samples."""
    assert OBSTRUCTION_FIT_MIN_COOLING_SAMPLES_FOR_HIGH == 50


# ---------------------------------------------------------------------------
# _within_plausibility_range / _boundary_applicable unit tests
# ---------------------------------------------------------------------------


def test_within_plausibility_low_boundaries():
    assert _within_plausibility_range(2.0, "low") is True   # inclusive low
    assert _within_plausibility_range(20.0, "low") is True  # inclusive high
    assert _within_plausibility_range(1.9, "low") is False
    assert _within_plausibility_range(20.1, "low") is False
    assert _within_plausibility_range(None, "low") is False


def test_within_plausibility_high_boundaries():
    assert _within_plausibility_range(20.0, "high") is True
    assert _within_plausibility_range(60.0, "high") is True
    assert _within_plausibility_range(19.9, "high") is False
    assert _within_plausibility_range(60.1, "high") is False


def test_boundary_applicable_not_learned():
    out, reason = _boundary_applicable(
        {"learned": False, "sse_improvement_ratio": 0.9, "best_critical_elev": 10},
        "low",
    )
    assert out is False
    assert reason == "not_learned"


def test_boundary_applicable_below_sse_threshold():
    out, reason = _boundary_applicable(
        {"learned": True, "sse_improvement_ratio": 0.25, "best_critical_elev": 10},
        "low",
    )
    assert out is False
    assert reason == "below_sse_threshold"


def test_boundary_applicable_physically_implausible_low():
    """LOW best_critical_elev=25° → outside [2°, 20°] → implausible."""
    out, reason = _boundary_applicable(
        {"learned": True, "sse_improvement_ratio": 0.6, "best_critical_elev": 25.0},
        "low",
    )
    assert out is False
    assert reason == "physically_implausible"


def test_boundary_applicable_physically_implausible_high():
    """HIGH best_critical_elev=15° → outside [20°, 60°] → implausible."""
    out, reason = _boundary_applicable(
        {"learned": True, "sse_improvement_ratio": 0.6, "best_critical_elev": 15.0},
        "high",
    )
    assert out is False
    assert reason == "physically_implausible"


def test_boundary_applicable_single_window():
    """All local checks pass but stability=False → single_window_only."""
    out, reason = _boundary_applicable(
        {"learned": True, "sse_improvement_ratio": 0.6, "best_critical_elev": 10.0},
        "low",
        stable_across_windows=False,
    )
    assert out is False
    assert reason == "single_window_only"


def test_boundary_applicable_all_pass():
    """learned ∧ strong SSE ∧ plausible ∧ stable → applicable=True."""
    out, reason = _boundary_applicable(
        {"learned": True, "sse_improvement_ratio": 0.6, "best_critical_elev": 10.0},
        "low",
        stable_across_windows=True,
    )
    assert out is True
    assert reason is None


# ---------------------------------------------------------------------------
# End-to-end via fit_solar_obstruction
# ---------------------------------------------------------------------------


def _make_coord(*, with_cooling: bool = True):
    """Minimal coordinator stub for direct fit invocation."""
    coord = MagicMock()
    coord.latitude = 52.0
    coord.longitude = 5.0
    coord.timezone = "UTC"
    sensor = "sensor.heater1"
    coord.energy_sensors = [sensor]
    coord._solar_affected_set = None
    coord._critical_elev_per_facade_per_unit = {}
    coord.critical_elev_for_entity = MagicMock(
        side_effect=lambda eid: (
            coord._critical_elev_per_facade_per_unit.get(eid, {
                "s": {"low": None, "high": None},
                "e": {"low": None, "high": None},
                "w": {"low": None, "high": None},
            })
        )
    )
    coord.screen_config = (False, False, False)
    coord.screen_config_for_entity = MagicMock(
        side_effect=lambda _eid: (False, False, False)
    )
    coeffs: dict = {
        "heating": {
            "s": 0.0015, "e": 0.0010, "w": 0.0010,
            "diffuse": 0.0005, "learned": True,
        },
    }
    buckets = {"normal": 3.0}
    if with_cooling:
        coeffs["cooling"] = dict(coeffs["heating"])
        buckets["cooling"] = 3.0
    coord._solar_coefficients_4d_per_unit = {sensor: coeffs}
    coord._correlation_data_per_unit = {sensor: {"-2": dict(buckets)}}
    coord.balance_point = 17.0
    coord.statistics = MagicMock()
    coord.statistics._get_prediction_from_model = MagicMock(return_value=3.0)
    solar = SolarCalculator(coord)
    coord.solar = solar
    return coord, solar, sensor


def _build_entries(
    sun_positions, *, true_crit_s=None, regime="heating",
    sensor="sensor.heater1", start_offset_hours=0, base=3.0,
):
    from custom_components.heating_analytics.solar import resolve_dni_dhi
    wind_bucket = "normal" if regime == "heating" else "cooling"
    entries: list[dict] = []
    sun_pos: dict[str, tuple[float, float]] = {}
    c_s, c_e, c_w, c_d = 0.0015, 0.0010, 0.0010, 0.0005
    start = datetime(2026, 4, 1, 0, 0, 0) + timedelta(hours=start_offset_hours)
    for i, (elev, az) in enumerate(sun_positions):
        ts_dt = start + timedelta(hours=i)
        ts_str = ts_dt.strftime("%Y-%m-%dT%H:%M:%S")
        sun_pos[ts_str] = (elev, az)
        dni, dhi, _ = resolve_dni_dhi(
            700.0, 150.0, None, None, elev,
            ts_dt.timetuple().tm_yday,
        )
        cos_elev = math.cos(math.radians(elev))
        az_rad = math.radians(az)
        pot_s = max(0.0, dni) * cos_elev * max(0.0, -math.cos(az_rad))
        pot_e = max(0.0, dni) * cos_elev * max(0.0, math.sin(az_rad))
        pot_w = max(0.0, dni) * cos_elev * max(0.0, -math.sin(az_rad))
        pot_d = max(0.0, dhi) * 0.5
        if true_crit_s is not None and elev > true_crit_s:
            pot_s_actual = 0.0
        else:
            pot_s_actual = pot_s
        impact = c_s * pot_s_actual + c_e * pot_e + c_w * pot_w + c_d * pot_d
        if regime == "heating":
            actual = max(0.0, base - impact)
        else:
            actual = base + impact
        entries.append({
            "timestamp": ts_str,
            "dni": 700.0, "dhi": 150.0,
            "ghi_wm2": None, "cloud_coverage": None,
            "correction_percent": 100.0, "auxiliary_active": False,
            "unit_modes": {sensor: regime},
            "unit_breakdown": {sensor: actual},
            "temp": -2.0, "temp_key": "-2",
            "wind_bucket": wind_bucket,
            "unit_expected_base": {sensor: base},
            "solar_dominant_entities": [],
        })
    return entries, sun_pos


def _stub_sun_pos(sun_pos_by_ts):
    def _lookup(mid_dt):
        entry_dt = mid_dt - timedelta(minutes=30)
        return sun_pos_by_ts.get(
            entry_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            (0.0, 180.0),
        )
    return _lookup


def test_cooling_floor_skips_high_on_heating_only():
    """Heating-only data with HIGH-shape signal → cooling-floor fires,
    HIGH skip_reason = insufficient_cooling_for_high_regime, no
    suggestion surfaces."""
    coord, solar, sensor = _make_coord(with_cooling=True)
    elevs = list(range(10, 71)) * 4
    positions = [(float(e), 180.0) for e in elevs]
    entries, sun_pos = _build_entries(
        positions, true_crit_s=30.0, regime="heating", sensor=sensor,
    )
    solar.get_approx_sun_pos = _stub_sun_pos(sun_pos)
    lm = LearningManager()
    result = lm.fit_solar_obstruction(
        hourly_log=entries, coordinator=coord, dry_run=False,
    )
    s = result[sensor]["s"]
    assert s["high"]["learned"] is False
    assert s["high"]["skip_reason"] == "insufficient_cooling_for_high_regime"
    assert s["high"]["applicable"] is False
    assert s["high"]["applicable_reason"] == "not_learned"
    # No suggestion.
    high_suggestions = [
        sg for sg in result["suggested_gates"]
        if sg["entity_id"] == sensor and sg["facade"] == "s" and sg["side"] == "high"
    ]
    assert high_suggestions == []
    # State unchanged.
    assert sensor not in coord._critical_elev_per_facade_per_unit


def test_low_gate_works_without_cooling_data():
    """Single-side trust: heating-only data + LOW-shape signal at 6°
    produces an applicable LOW suggestion; cooling-floor does not
    block LOW.  Equivalent to the #1020 Toshiba W LOW=4° scenario
    on a synthetic install.
    """
    coord, solar, sensor = _make_coord(with_cooling=True)
    # Wide elevation range so adaptive low_min_eff = max(5°, data_min)
    # can land at the natural LOW gate.  Plant LOW gate at 6° (within
    # plausibility [2°, 20°]) — sun below 6° has pot=0, above has full pot.
    # Build the data: elevations 3° to 18°.
    elevs = list(range(3, 19)) * 8  # 128 heating samples
    positions = [(float(e), 180.0) for e in elevs]
    # Use a custom builder for LOW gate (true_crit_low) — current
    # _build_entries supports only HIGH-shape gating, so synthesise
    # the LOW signature directly.
    from custom_components.heating_analytics.solar import resolve_dni_dhi
    entries: list[dict] = []
    sun_pos: dict[str, tuple[float, float]] = {}
    c_s, c_e, c_w, c_d = 0.0015, 0.0010, 0.0010, 0.0005
    base = 3.0
    LOW_CRIT = 6.0
    start = datetime(2026, 4, 1, 0, 0, 0)
    for i, (elev, az) in enumerate(positions):
        ts_dt = start + timedelta(hours=i)
        ts_str = ts_dt.strftime("%Y-%m-%dT%H:%M:%S")
        sun_pos[ts_str] = (elev, az)
        dni, dhi, _ = resolve_dni_dhi(
            700.0, 150.0, None, None, elev,
            ts_dt.timetuple().tm_yday,
        )
        cos_elev = math.cos(math.radians(elev))
        az_rad = math.radians(az)
        pot_s = max(0.0, dni) * cos_elev * max(0.0, -math.cos(az_rad))
        pot_e = max(0.0, dni) * cos_elev * max(0.0, math.sin(az_rad))
        pot_w = max(0.0, dni) * cos_elev * max(0.0, -math.sin(az_rad))
        pot_d = max(0.0, dhi) * 0.5
        # LOW gate: zero out south potential below LOW_CRIT.
        pot_s_actual = pot_s if elev >= LOW_CRIT else 0.0
        impact = c_s * pot_s_actual + c_e * pot_e + c_w * pot_w + c_d * pot_d
        actual = max(0.0, base - impact)
        entries.append({
            "timestamp": ts_str,
            "dni": 700.0, "dhi": 150.0,
            "ghi_wm2": None, "cloud_coverage": None,
            "correction_percent": 100.0, "auxiliary_active": False,
            "unit_modes": {sensor: "heating"},
            "unit_breakdown": {sensor: actual},
            "temp": -2.0, "temp_key": "-2",
            "wind_bucket": "normal",
            "unit_expected_base": {sensor: base},
            "solar_dominant_entities": [],
        })
    solar.get_approx_sun_pos = _stub_sun_pos(sun_pos)
    lm = LearningManager()
    result = lm.fit_solar_obstruction(
        hourly_log=entries, coordinator=coord, dry_run=False,
    )
    s = result[sensor]["s"]
    assert s["low"]["learned"] is True, s["low"]
    assert 3.0 <= s["low"]["best_critical_elev"] <= 9.0, s["low"]
    assert s["low"]["applicable"] is True, s["low"]
    # HIGH: cooling-floor blocks (heating-only data) — sibling side
    # being unfittable does NOT cascade to LOW.  This is the central
    # single-side-trust property #1020 added.
    assert s["high"]["learned"] is False
    assert s["high"]["skip_reason"] == "insufficient_cooling_for_high_regime"
    # LOW suggestion surfaces despite HIGH being structurally blocked.
    low_suggestions = [
        sg for sg in result["suggested_gates"]
        if sg["entity_id"] == sensor and sg["facade"] == "s" and sg["side"] == "low"
    ]
    assert len(low_suggestions) == 1
    # State unchanged.
    assert sensor not in coord._critical_elev_per_facade_per_unit
