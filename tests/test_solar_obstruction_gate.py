"""Tests for the solar-window obstruction gate (v9).

Two surfaces:
  1. Gate application inside ``SolarCalculator.calculate_unit_potential_4d``:
     with ``coordinator.critical_elev_for_entity(eid)[f] = {"low": ..., "high": ...}``,
     the corresponding ``pot_*_dir`` must be zeroed whenever
     ``sun_elev < low`` OR ``sun_elev > high``.  Diffuse is unaffected.
     ``None`` per boundary = no gate on that side.
  2. Fit routine ``LearningManager.fit_solar_obstruction``: synthetic
     step-residual data over the elevation range should produce the
     correct ``critical_elev_low`` and ``critical_elev_high`` per entity;
     flat residuals should leave ``learned=False``; sample-count and
     SSE-improvement gates must be enforced.  Shutdown-flagged hours
     constrain the feasible range.
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


def _make_gate(low=None, high=None):
    return {"low": low, "high": high}

_NULL_GATE = {
    "s": _make_gate(), "e": _make_gate(), "w": _make_gate(),
}

def _calc(
    critical_elev: dict | None = None,
    *,
    entity_id: str = "x",
) -> SolarCalculator:
    """Build a SolarCalculator with a coordinator that returns ``critical_elev``
    for the named entity via ``critical_elev_for_entity``.  ``None`` →
    all-None (no gate).  Gate shape is v9 nested ``{"low": ..., "high": ...}``.
    """
    coord = MagicMock()
    gate = critical_elev or _NULL_GATE
    coord._critical_elev_per_facade_per_unit = {entity_id: gate}
    coord.critical_elev_for_entity = MagicMock(
        side_effect=lambda eid: (
            coord._critical_elev_per_facade_per_unit.get(eid, _NULL_GATE)
        )
    )
    return SolarCalculator(coord)


# ---------------------------------------------------------------------------
# Gate application in calculate_unit_potential_4d
# ---------------------------------------------------------------------------


def test_gate_none_is_no_op():
    """``critical_elev_for_entity`` returning all-None → bit-identical to pre-gate."""
    calc = _calc(critical_elev=None)
    out = calc.calculate_unit_potential_4d(
        entity_id="x", dni=500.0, dhi=100.0,
        sun_elev_deg=45.0, sun_azimuth_deg=180.0,
        screen_config=(False, False, False), correction_percent=100.0,
    )
    assert abs(out[0] - 353.5534) < 0.5  # south direct
    assert abs(out[3] - 50.0) < 1e-9     # diffuse


def test_gate_all_none_is_no_op():
    """``{s, e, w} = {low: None, high: None}`` → identical to no-gate output."""
    calc = _calc(critical_elev=_NULL_GATE)
    out = calc.calculate_unit_potential_4d(
        entity_id="x", dni=500.0, dhi=100.0,
        sun_elev_deg=45.0, sun_azimuth_deg=180.0,
        screen_config=(False, False, False), correction_percent=100.0,
    )
    assert abs(out[0] - 353.5534) < 0.5


def test_south_high_gate_zeroes_above_cutoff():
    """``crit_s.high = 30°``, sun at elev=45° → ``pot_s = 0``; diffuse untouched."""
    calc = _calc(critical_elev={
        "s": _make_gate(high=30.0), "e": _make_gate(), "w": _make_gate(),
    })
    out = calc.calculate_unit_potential_4d(
        entity_id="x", dni=500.0, dhi=100.0,
        sun_elev_deg=45.0, sun_azimuth_deg=180.0,
        screen_config=(False, False, False), correction_percent=100.0,
    )
    assert out[0] == 0.0
    assert abs(out[3] - 50.0) < 1e-9


def test_south_high_gate_passes_below_cutoff():
    """``crit_s.high = 30°``, sun at elev=20° → south direct unchanged."""
    calc = _calc(critical_elev={
        "s": _make_gate(high=30.0), "e": _make_gate(), "w": _make_gate(),
    })
    out = calc.calculate_unit_potential_4d(
        entity_id="x", dni=500.0, dhi=100.0,
        sun_elev_deg=20.0, sun_azimuth_deg=180.0,
        screen_config=(False, False, False), correction_percent=100.0,
    )
    expected_s = 500.0 * math.cos(math.radians(20.0))
    assert abs(out[0] - expected_s) < 0.5


def test_south_low_gate_zeroes_below_cutoff():
    """``crit_s.low = 15°``, sun at elev=10° → ``pot_s = 0``."""
    calc = _calc(critical_elev={
        "s": _make_gate(low=15.0), "e": _make_gate(), "w": _make_gate(),
    })
    out = calc.calculate_unit_potential_4d(
        entity_id="x", dni=500.0, dhi=100.0,
        sun_elev_deg=10.0, sun_azimuth_deg=180.0,
        screen_config=(False, False, False), correction_percent=100.0,
    )
    assert out[0] == 0.0


def test_south_low_gate_passes_above_cutoff():
    """``crit_s.low = 15°``, sun at elev=20° → south direct unchanged."""
    calc = _calc(critical_elev={
        "s": _make_gate(low=15.0), "e": _make_gate(), "w": _make_gate(),
    })
    out = calc.calculate_unit_potential_4d(
        entity_id="x", dni=500.0, dhi=100.0,
        sun_elev_deg=20.0, sun_azimuth_deg=180.0,
        screen_config=(False, False, False), correction_percent=100.0,
    )
    expected_s = 500.0 * math.cos(math.radians(20.0))
    assert abs(out[0] - expected_s) < 0.5


def test_south_window_gate_zeroes_outside_window():
    """``crit_s = {low: 15°, high: 30°}``, sun at elev=10° → zeroed;
    sun at elev=40° → zeroed; sun at elev=20° → passes."""
    calc = _calc(critical_elev={
        "s": _make_gate(low=15.0, high=30.0), "e": _make_gate(), "w": _make_gate(),
    })
    out1 = calc.calculate_unit_potential_4d(
        entity_id="x", dni=500.0, dhi=100.0,
        sun_elev_deg=10.0, sun_azimuth_deg=180.0,
        screen_config=(False, False, False), correction_percent=100.0,
    )
    assert out1[0] == 0.0  # below low → zeroed
    out2 = calc.calculate_unit_potential_4d(
        entity_id="x", dni=500.0, dhi=100.0,
        sun_elev_deg=40.0, sun_azimuth_deg=180.0,
        screen_config=(False, False, False), correction_percent=100.0,
    )
    assert out2[0] == 0.0  # above high → zeroed
    out3 = calc.calculate_unit_potential_4d(
        entity_id="x", dni=500.0, dhi=100.0,
        sun_elev_deg=20.0, sun_azimuth_deg=180.0,
        screen_config=(False, False, False), correction_percent=100.0,
    )
    expected_s = 500.0 * math.cos(math.radians(20.0))
    assert abs(out3[0] - expected_s) < 0.5  # inside window → passes


def test_gates_are_independent_per_facade():
    """``crit_s.high = 30°`` does NOT gate east or west."""
    calc = _calc(critical_elev={
        "s": _make_gate(high=30.0), "e": _make_gate(), "w": _make_gate(),
    })
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


def _make_fit_coord(
    *,
    with_cooling: bool = False,
    sensor_ids: list[str] | None = None,
):
    """Build a coordinator with enough structure for the fit to run.

    ``with_cooling=True`` seeds learned cooling-regime coefficients +
    cooling-wind-bucket base model on each sensor (#1006).
    ``sensor_ids`` defaults to ``["sensor.heater1"]``; multi-entity
    coords are used by the per-entity tests (#1009).
    """
    sensors = sensor_ids or ["sensor.heater1"]
    coord = MagicMock()
    coord.latitude = 52.0
    coord.longitude = 5.0
    coord.timezone = "UTC"
    coord.energy_sensors = list(sensors)
    coord._solar_affected_set = None
    coord._critical_elev_per_facade_per_unit = {}
    coord.critical_elev_for_entity = MagicMock(
        side_effect=lambda eid: (
            coord._critical_elev_per_facade_per_unit.get(
                eid, _NULL_GATE
            )
        )
    )
    coord.screen_config = (False, False, False)
    coord.screen_config_for_entity = MagicMock(
        side_effect=lambda _eid: (False, False, False)
    )
    # Pre-learned 4D coefficients (heating + optional cooling).  Same
    # magnitudes per entity for test simplicity.
    coeffs_per_unit: dict[str, dict] = {}
    base_data: dict[str, dict] = {}
    unit_buckets: dict[str, float] = {"normal": 3.0}
    if with_cooling:
        unit_buckets["cooling"] = 3.0
    for sid in sensors:
        coeffs: dict[str, dict] = {
            "heating": {
                "s": 0.0015, "e": 0.0010, "w": 0.0010,
                "diffuse": 0.0005, "learned": True,
            },
        }
        if with_cooling:
            coeffs["cooling"] = {
                "s": 0.0015, "e": 0.0010, "w": 0.0010,
                "diffuse": 0.0005, "learned": True,
            }
        coeffs_per_unit[sid] = coeffs
        base_data[sid] = {"-2": dict(unit_buckets)}
    coord._solar_coefficients_4d_per_unit = coeffs_per_unit
    coord._correlation_data_per_unit = base_data
    coord.balance_point = 17.0
    coord.statistics = MagicMock()
    coord.statistics._get_prediction_from_model = MagicMock(return_value=3.0)
    solar = SolarCalculator(coord)
    coord.solar = solar
    return coord, solar


def _build_log_explicit(
    sun_positions: list[tuple[float, float]],
    *,
    true_crit_s: float | None = None,
    base: float = 3.0,
    sensor_id: str = "sensor.heater1",
    regime: str = "heating",
    start_offset_hours: int = 0,
) -> tuple[list[dict], dict[str, tuple[float, float]]]:
    """Synthesise hourly_log entries with explicit (elev, az) per row.

    ``sun_positions`` is a list of ``(elev_deg, az_deg)`` pairs; one
    log entry per pair, monotonically timestamped.  Returns ``(entries,
    sun_pos_by_ts)`` where the second dict maps each entry's timestamp
    string to its sun position — feed it to a stubbed
    ``get_approx_sun_pos`` so the fit reads back the planted values.

    ``regime`` toggles the sign of solar impact on ``actual``:
      * ``"heating"`` — ``actual = base − impact`` (sun reduces load).
        ``unit_modes[sensor] = "heating"``, ``wind_bucket = "normal"``.
      * ``"cooling"`` — ``actual = base + impact`` (sun increases load).
        ``unit_modes[sensor] = "cooling"``, ``wind_bucket = "cooling"``.

    ``start_offset_hours`` shifts the timestamp range — use when
    composing heating + cooling samples in one log to keep timestamps
    monotonically distinct.
    """
    from custom_components.heating_analytics.solar import resolve_dni_dhi

    if regime not in ("heating", "cooling"):
        raise ValueError(f"regime must be 'heating' or 'cooling', got {regime!r}")
    mode_str = regime  # MODE_HEATING / MODE_COOLING are the strings "heating"/"cooling"
    wind_bucket = "normal" if regime == "heating" else "cooling"

    entries: list[dict] = []
    sun_pos_by_ts: dict[str, tuple[float, float]] = {}
    c_s, c_e, c_w, c_d = 0.0015, 0.0010, 0.0010, 0.0005
    start = datetime(2026, 4, 1, 0, 0, 0) + timedelta(hours=start_offset_hours)

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
        if regime == "heating":
            actual = max(0.0, base - impact)
        else:  # cooling — sun adds load
            actual = base + impact
        entries.append({
            "timestamp": ts_str,
            "dni": 700.0,
            "dhi": 150.0,
            "ghi_wm2": None,
            "cloud_coverage": None,
            "correction_percent": 100.0,
            "auxiliary_active": False,
            "unit_modes": {sensor_id: mode_str},
            "unit_breakdown": {sensor_id: actual},
            # Recalc-baseline inputs (#1005): the fit reads temp_key +
            # wind_bucket and looks up the per-unit base model via the
            # coordinator's statistics stub.
            "temp": -2.0,
            "temp_key": "-2",
            "wind_bucket": wind_bucket,
            "unit_expected_base": {sensor_id: base},
            "solar_dominant_entities": [],
        })
    return entries, sun_pos_by_ts


def _south_dominant_positions(elevs: list[float]) -> list[tuple[float, float]]:
    """List of (elev, az=180°) — sun straight south at given elevations."""
    return [(elev, 180.0) for elev in elevs]


def _build_log_with_cooling_supplement(
    sun_positions: list[tuple[float, float]],
    *,
    true_crit_s: float | None = None,
    sensor_id: str = "sensor.heater1",
    base: float = 3.0,
) -> tuple[list[dict], dict[str, tuple[float, float]]]:
    """Build heating + cooling log entries covering the same elevations.

    #1020 added a cooling-sample floor for HIGH-gate auto-suggestions:
    fits on heating-only data above the configured floor (default 50
    samples) are honestly skipped with
    ``insufficient_cooling_for_high_regime``.  Existing tests that
    exercised the HIGH path with heating-only data now need a
    cooling supplement to keep the apply-gate path reachable.  This
    helper concatenates a heating-regime block and a cooling-regime
    block over the same elevation positions, with disjoint timestamps.
    """
    heating_entries, heating_sun = _build_log_explicit(
        sun_positions, true_crit_s=true_crit_s,
        sensor_id=sensor_id, regime="heating", base=base,
    )
    cooling_entries, cooling_sun = _build_log_explicit(
        sun_positions, true_crit_s=true_crit_s,
        sensor_id=sensor_id, regime="cooling", base=base,
        start_offset_hours=len(sun_positions),
    )
    return heating_entries + cooling_entries, {**heating_sun, **cooling_sun}


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


SENSOR = "sensor.heater1"


def test_fit_detects_step_at_correct_elevation():
    """South sun planted at uniform elevations 10°→70°, overhang gate
    at 30° (high boundary).  Fit should locate the step within ±5°.

    #1020: needs cooling supplement for HIGH-gate suggestions (cooling-
    sample floor).  Auto-write removed — assertion now checks
    suggested_gates and ``applicable=True`` rather than state.
    """
    coord, solar = _make_fit_coord(with_cooling=True)
    elevs = list(range(10, 71)) * 4  # 4 samples per degree, 244 total
    positions = _south_dominant_positions([float(e) for e in elevs])
    entries, sun_pos_by_ts = _build_log_with_cooling_supplement(
        positions, true_crit_s=30.0,
    )
    solar.get_approx_sun_pos = _stub_get_approx_sun_pos(sun_pos_by_ts)
    lm = LearningManager()
    result = lm.fit_solar_obstruction(
        hourly_log=entries, coordinator=coord, dry_run=False,
    )
    s = result[SENSOR]["s"]
    high = s["high"]
    assert high["learned"] is True, high
    assert 25.0 <= high["best_critical_elev"] <= 35.0, high["best_critical_elev"]
    assert high["sse_improvement_ratio"] >= 0.30
    assert high["n_below_best"] >= OBSTRUCTION_FIT_MIN_SAMPLES_PER_SIDE
    assert high["n_above_best"] >= OBSTRUCTION_FIT_MIN_SAMPLES_PER_SIDE
    assert high["applicable"] is True, high
    # #1020: auto-write removed — gate state is unchanged, suggestion surfaces instead.
    assert SENSOR not in coord._critical_elev_per_facade_per_unit
    suggested = result.get("suggested_gates", [])
    assert any(
        sg["entity_id"] == SENSOR and sg["facade"] == "s" and sg["side"] == "high"
        for sg in suggested
    ), suggested


def test_fit_flat_residual_leaves_unlearned():
    """No obstruction in synthetic data → ``learned=False`` for both
    low and high, entity receives no entry in the per-unit dict.
    """
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
    assert result[SENSOR]["s"]["low"]["learned"] is False, result[SENSOR]["s"]["low"]
    assert result[SENSOR]["s"]["high"]["learned"] is False, result[SENSOR]["s"]["high"]
    # No write-through → entity remains absent from the per-unit dict.
    assert SENSOR not in coord._critical_elev_per_facade_per_unit


def test_fit_dry_run_field_passthrough():
    """``dry_run=True`` → cosmetic passthrough since #1020 (no writes
    happen regardless).  Diagnostics still returned, state untouched.
    """
    coord, solar = _make_fit_coord(with_cooling=True)
    elevs = list(range(10, 71)) * 4
    positions = _south_dominant_positions([float(e) for e in elevs])
    entries, sun_pos_by_ts = _build_log_with_cooling_supplement(
        positions, true_crit_s=30.0,
    )
    solar.get_approx_sun_pos = _stub_get_approx_sun_pos(sun_pos_by_ts)
    lm = LearningManager()
    result = lm.fit_solar_obstruction(
        hourly_log=entries, coordinator=coord, dry_run=True,
    )
    assert result["dry_run"] is True
    assert result[SENSOR]["s"]["high"]["learned"] is True
    # State NOT written (auto-write removed in #1020).
    assert SENSOR not in coord._critical_elev_per_facade_per_unit


def test_fit_recalc_baseline_is_canonical(monkeypatch):
    """#1005: baseline is ALWAYS recalc'd from the per-unit base model;
    the logged ``unit_expected_base`` field is ignored even when present.
    """
    coord, solar = _make_fit_coord(with_cooling=True)
    elevs = list(range(10, 71)) * 4
    positions = _south_dominant_positions([float(e) for e in elevs])
    entries, sun_pos_by_ts = _build_log_with_cooling_supplement(
        positions, true_crit_s=30.0,
    )
    for e in entries:
        e.pop("unit_expected_base", None)
    solar.get_approx_sun_pos = _stub_get_approx_sun_pos(sun_pos_by_ts)
    lm = LearningManager()
    result = lm.fit_solar_obstruction(
        hourly_log=entries, coordinator=coord, dry_run=False,
    )
    s = result[SENSOR]["s"]
    assert s["high"]["learned"] is True, s["high"]
    assert 25.0 <= s["high"]["best_critical_elev"] <= 35.0
    assert coord.statistics._get_prediction_from_model.called


def test_fit_recalc_skipped_without_temp_key():
    """Recalc requires ``temp_key`` + ``wind_bucket`` on the log entry;
    without them, the sample is dropped.
    """
    coord, solar = _make_fit_coord()
    positions = _south_dominant_positions([20.0, 30.0, 40.0])
    entries, sun_pos_by_ts = _build_log_explicit(
        positions, true_crit_s=30.0,
    )
    for e in entries:
        e.pop("temp_key", None)
        e.pop("wind_bucket", None)
    solar.get_approx_sun_pos = _stub_get_approx_sun_pos(sun_pos_by_ts)
    lm = LearningManager()
    result = lm.fit_solar_obstruction(
        hourly_log=entries, coordinator=coord, dry_run=False,
    )
    assert result[SENSOR]["s"]["n_samples"] == 0
    assert not coord.statistics._get_prediction_from_model.called


def test_fit_insufficient_samples_skips():
    """Few samples → both low and high get ``insufficient_samples`` skip."""
    coord, solar = _make_fit_coord()
    positions = _south_dominant_positions([20.0, 30.0, 40.0, 50.0, 60.0])
    entries, sun_pos_by_ts = _build_log_explicit(
        positions, true_crit_s=30.0,
    )
    solar.get_approx_sun_pos = _stub_get_approx_sun_pos(sun_pos_by_ts)
    lm = LearningManager()
    result = lm.fit_solar_obstruction(
        hourly_log=entries, coordinator=coord, dry_run=False,
    )
    assert result[SENSOR]["s"]["low"]["skip_reason"] == "insufficient_samples"
    assert result[SENSOR]["s"]["high"]["skip_reason"] == "insufficient_samples"


# ---------------------------------------------------------------------------
# Cooling-regime fit (#1006)
# ---------------------------------------------------------------------------


def test_fit_combined_modes_converges_on_same_elev():
    """Heating + cooling samples in one log, same planted critical_elev=30°.
    The combined per-facade SSE search should find the step within ±5°.
    """
    coord, solar = _make_fit_coord(with_cooling=True)
    elevs = list(range(10, 71)) * 2  # 122 samples per regime
    positions = _south_dominant_positions([float(e) for e in elevs])
    heating_entries, heating_sun = _build_log_explicit(
        positions, true_crit_s=30.0, regime="heating",
    )
    cooling_entries, cooling_sun = _build_log_explicit(
        positions, true_crit_s=30.0, regime="cooling",
        start_offset_hours=len(positions),  # disjoint timestamps
    )
    entries = heating_entries + cooling_entries
    sun_pos_by_ts = {**heating_sun, **cooling_sun}

    solar.get_approx_sun_pos = _stub_get_approx_sun_pos(sun_pos_by_ts)
    lm = LearningManager()
    result = lm.fit_solar_obstruction(
        hourly_log=entries, coordinator=coord, dry_run=False,
    )
    s = result[SENSOR]["s"]
    high = s["high"]
    assert high["learned"] is True, high
    assert 25.0 <= high["best_critical_elev"] <= 35.0, high["best_critical_elev"]
    # Both regimes contributed to the south facade samples.
    assert s["n_heating_samples"] > 0
    assert s["n_cooling_samples"] > 0
    assert s["n_samples"] == s["n_heating_samples"] + s["n_cooling_samples"]
    assert result["n_skipped_cooling_unlearned"] == 0


def test_fit_cooling_only_when_heating_unavailable():
    """Summer install: only cooling-mode samples in the log → fit still
    converges on the cooling-regime contributions alone.
    """
    coord, solar = _make_fit_coord(with_cooling=True)
    elevs = list(range(10, 71)) * 4  # 244 cooling samples
    positions = _south_dominant_positions([float(e) for e in elevs])
    entries, sun_pos_by_ts = _build_log_explicit(
        positions, true_crit_s=30.0, regime="cooling",
    )
    solar.get_approx_sun_pos = _stub_get_approx_sun_pos(sun_pos_by_ts)
    lm = LearningManager()
    result = lm.fit_solar_obstruction(
        hourly_log=entries, coordinator=coord, dry_run=False,
    )
    s = result[SENSOR]["s"]
    high = s["high"]
    assert high["learned"] is True, high
    assert 25.0 <= high["best_critical_elev"] <= 35.0, high["best_critical_elev"]
    assert s["n_heating_samples"] == 0
    assert s["n_cooling_samples"] > 0


def test_fit_cooling_skipped_when_coefficient_unlearned():
    """Cooling-regime entries on an entity whose cooling coefficient is
    not learned → samples dropped, counter incremented, no fit on cooling.
    """
    coord, solar = _make_fit_coord(with_cooling=False)  # no cooling coeff
    positions = _south_dominant_positions(
        [float(e) for e in range(10, 71)] * 2
    )
    entries, sun_pos_by_ts = _build_log_explicit(
        positions, true_crit_s=30.0, regime="cooling",
    )
    solar.get_approx_sun_pos = _stub_get_approx_sun_pos(sun_pos_by_ts)
    lm = LearningManager()
    result = lm.fit_solar_obstruction(
        hourly_log=entries, coordinator=coord, dry_run=False,
    )
    # No samples accumulated — fit skipped per facade.
    assert result[SENSOR]["s"]["n_samples"] == 0
    assert result[SENSOR]["s"]["high"]["learned"] is False
    # Counter reflects per-entity-per-entry skip.
    assert result["n_skipped_cooling_unlearned"] == len(entries)


def test_fit_cooling_uses_cooling_wind_bucket():
    """Cooling baseline lookup must route to COOLING_WIND_BUCKET regardless
    of the entry's stored wind_bucket field.  Verified by inspecting the
    args passed to the statistics stub.
    """
    coord, solar = _make_fit_coord(with_cooling=True)
    positions = _south_dominant_positions(
        [float(e) for e in range(10, 71)] * 2
    )
    entries, sun_pos_by_ts = _build_log_explicit(
        positions, true_crit_s=30.0, regime="cooling",
    )
    solar.get_approx_sun_pos = _stub_get_approx_sun_pos(sun_pos_by_ts)
    lm = LearningManager()
    lm.fit_solar_obstruction(
        hourly_log=entries, coordinator=coord, dry_run=False,
    )
    # The stub was called at least once; all calls must use bucket="cooling".
    assert coord.statistics._get_prediction_from_model.called
    for call in coord.statistics._get_prediction_from_model.call_args_list:
        # Signature: (unit_data, temp_key, wind_bucket, temp, balance_point)
        bucket_arg = call.args[2] if len(call.args) >= 3 else call.kwargs.get("wind_bucket")
        assert bucket_arg == "cooling", bucket_arg


def test_fit_heating_only_high_blocked_by_cooling_floor():
    """#1020: pure-heating log on a HIGH-shape signal is honestly
    skipped with ``insufficient_cooling_for_high_regime`` — the
    central case the cooling-floor was added to address.  LOW path
    on the same log is unaffected (LOW does not require cooling).
    """
    coord, solar = _make_fit_coord()
    elevs = list(range(10, 71)) * 4
    positions = _south_dominant_positions([float(e) for e in elevs])
    entries, sun_pos_by_ts = _build_log_explicit(
        positions, true_crit_s=30.0, regime="heating",
    )
    solar.get_approx_sun_pos = _stub_get_approx_sun_pos(sun_pos_by_ts)
    lm = LearningManager()
    result = lm.fit_solar_obstruction(
        hourly_log=entries, coordinator=coord, dry_run=False,
    )
    s = result[SENSOR]["s"]
    assert s["high"]["learned"] is False
    assert s["high"]["skip_reason"] == "insufficient_cooling_for_high_regime", s["high"]
    assert s["n_cooling_samples"] == 0
    assert s["n_heating_samples"] == s["n_samples"]
    assert result["n_skipped_cooling_unlearned"] == 0


# ---------------------------------------------------------------------------
# Per-entity gate (#1009)
# ---------------------------------------------------------------------------


def test_gate_is_per_entity_independent():
    """Two entities can carry independent gate state on the same facade.
    Entity A with crit_s=16°, Entity B with crit_s=None → identical sun
    elev produces different ``pot_s_dir`` for the two entities.
    """
    coord = MagicMock()
    coord._critical_elev_per_facade_per_unit = {
        "sensor.a": {"s": _make_gate(high=16.0), "e": _NULL_GATE["e"], "w": _NULL_GATE["w"]},
        "sensor.b": _NULL_GATE,
    }
    coord.critical_elev_for_entity = MagicMock(
        side_effect=lambda eid: (
            coord._critical_elev_per_facade_per_unit.get(
                eid, _NULL_GATE
            )
        )
    )
    calc = SolarCalculator(coord)
    out_a = calc.calculate_unit_potential_4d(
        entity_id="sensor.a", dni=500.0, dhi=100.0,
        sun_elev_deg=45.0, sun_azimuth_deg=180.0,
        screen_config=(False, False, False), correction_percent=100.0,
    )
    out_b = calc.calculate_unit_potential_4d(
        entity_id="sensor.b", dni=500.0, dhi=100.0,
        sun_elev_deg=45.0, sun_azimuth_deg=180.0,
        screen_config=(False, False, False), correction_percent=100.0,
    )
    assert out_a[0] == 0.0  # A gated at 16°, sun at 45° → zeroed
    assert out_b[0] > 0.0   # B has no gate → full south direct beam
    # Diffuse unaffected for both — same screen config.
    assert abs(out_a[3] - out_b[3]) < 1e-9


def test_gate_unknown_entity_defaults_to_none():
    """Entities without an entry in the per-unit dict default to no gate
    (graceful default matching pre-#991 behaviour).
    """
    coord = MagicMock()
    coord._critical_elev_per_facade_per_unit = {
        "sensor.known": {"s": _make_gate(high=20.0), "e": _NULL_GATE["e"], "w": _NULL_GATE["w"]},
    }
    coord.critical_elev_for_entity = MagicMock(
        side_effect=lambda eid: (
            coord._critical_elev_per_facade_per_unit.get(
                eid, _NULL_GATE
            )
        )
    )
    calc = SolarCalculator(coord)
    # Unknown entity at high sun elevation must still see direct beam.
    out = calc.calculate_unit_potential_4d(
        entity_id="sensor.unknown", dni=500.0, dhi=100.0,
        sun_elev_deg=45.0, sun_azimuth_deg=180.0,
        screen_config=(False, False, False), correction_percent=100.0,
    )
    assert out[0] > 0.0


def test_fit_per_entity_asymmetric_samples():
    """Two entities, only entity A receives obstruction-shaped samples;
    entity B receives flat-residual samples on the same elevation range.
    Fit must produce ``learned=True`` for A and ``learned=False`` for B.
    """
    sensor_a = "sensor.a"
    sensor_b = "sensor.b"
    coord, solar = _make_fit_coord(sensor_ids=[sensor_a, sensor_b], with_cooling=True)

    elevs = list(range(10, 71)) * 4
    positions = _south_dominant_positions([float(e) for e in elevs])

    a_entries, a_sun = _build_log_with_cooling_supplement(
        positions, true_crit_s=30.0, sensor_id=sensor_a,
    )
    b_entries, b_sun = _build_log_with_cooling_supplement(
        positions, true_crit_s=None, sensor_id=sensor_b,
    )
    # Offset B's timestamps so they don't collide with A's.
    from datetime import datetime as _dt2, timedelta as _td2
    base_shift = _td2(hours=2 * len(positions))
    new_b_sun: dict = {}
    for e in b_entries:
        ts_old = e["timestamp"]
        ts_new = (_dt2.fromisoformat(ts_old) + base_shift).strftime("%Y-%m-%dT%H:%M:%S")
        e["timestamp"] = ts_new
        new_b_sun[ts_new] = b_sun[ts_old]
    b_sun = new_b_sun
    # Merge entries by timestamp so each log row carries both entities'
    # data (mirrors production where unit_breakdown contains all units).
    merged: dict[str, dict] = {}
    for e in a_entries + b_entries:
        ts = e["timestamp"]
        if ts not in merged:
            merged[ts] = {
                "timestamp": ts,
                "dni": e["dni"], "dhi": e["dhi"],
                "ghi_wm2": None, "cloud_coverage": None,
                "correction_percent": 100.0, "auxiliary_active": False,
                "unit_modes": {}, "unit_breakdown": {},
                "temp": -2.0, "temp_key": "-2", "wind_bucket": "normal",
                "unit_expected_base": {},
                "solar_dominant_entities": [],
            }
        merged[ts]["unit_modes"].update(e["unit_modes"])
        merged[ts]["unit_breakdown"].update(e["unit_breakdown"])
        merged[ts]["unit_expected_base"].update(e["unit_expected_base"])

    entries = list(merged.values())
    sun_pos_by_ts = {**a_sun, **b_sun}
    solar.get_approx_sun_pos = _stub_get_approx_sun_pos(sun_pos_by_ts)

    lm = LearningManager()
    result = lm.fit_solar_obstruction(
        hourly_log=entries, coordinator=coord, dry_run=False,
    )
    # A learns its gate; B does not.
    assert result[sensor_a]["s"]["high"]["learned"] is True
    assert 25.0 <= result[sensor_a]["s"]["high"]["best_critical_elev"] <= 35.0
    assert result[sensor_b]["s"]["high"]["learned"] is False
    # #1020: no auto-write — both entities absent from state, suggestion
    # surfaces only for A.
    assert sensor_a not in coord._critical_elev_per_facade_per_unit
    assert sensor_b not in coord._critical_elev_per_facade_per_unit
    suggested = result.get("suggested_gates", [])
    a_suggestions = [
        sg for sg in suggested
        if sg["entity_id"] == sensor_a and sg["facade"] == "s" and sg["side"] == "high"
    ]
    b_suggestions = [
        sg for sg in suggested if sg["entity_id"] == sensor_b
    ]
    assert len(a_suggestions) == 1, suggested
    assert a_suggestions[0]["value"] == result[sensor_a]["s"]["high"]["best_critical_elev"]
    assert b_suggestions == [], b_suggestions


def test_fit_negative_cooling_impacts_are_skipped():
    """Cooling samples with ``actual < expected_base`` (negative impact)
    must be excluded from the obstruction fit — a binary gate search
    deterministically prefers gating these samples, biasing fits on
    weak-signal entities.  See docstring + #1006 follow-up.

    Construct a log where ALL cooling samples have negative impact;
    fit must produce zero cooling samples per facade.
    """
    coord, solar = _make_fit_coord(with_cooling=True)
    elevs = list(range(10, 71)) * 4
    positions = _south_dominant_positions([float(e) for e in elevs])
    entries, sun_pos_by_ts = _build_log_explicit(
        positions, true_crit_s=30.0, regime="cooling",
    )
    # Flip every cooling sample to actual < expected_base (negative
    # impact).  Subtract impact instead of adding it.
    for e in entries:
        sid = next(iter(e["unit_breakdown"]))
        actual = e["unit_breakdown"][sid]
        base = e["unit_expected_base"][sid]
        impact_magnitude = actual - base  # was positive
        e["unit_breakdown"][sid] = base - impact_magnitude  # now negative
    solar.get_approx_sun_pos = _stub_get_approx_sun_pos(sun_pos_by_ts)

    lm = LearningManager()
    result = lm.fit_solar_obstruction(
        hourly_log=entries, coordinator=coord, dry_run=True,
    )
    s = result[SENSOR]["s"]
    assert s["n_cooling_samples"] == 0, s
    assert s["n_samples"] == 0


def test_fit_below_rms_amplitude_threshold_not_learned():
    """Absolute-amplitude gate filters out fits whose per-sample RMS
    reduction is below the noise-floor threshold, even when the
    relative SSE improvement exceeds the relative threshold.

    Synthesise a fit with high *relative* improvement (small sse_flat
    that's mostly removed) but very small *absolute* magnitudes.
    """
    coord, solar = _make_fit_coord(with_cooling=True)
    # Tiny coefficients so cf_pot is in the noise floor (RMS reduction
    # ~ cf_pot magnitude ~ 0.0001 * 500 = 0.05 kWh per sample at peak,
    # but average over the elev range drops it below the 0.05 threshold).
    for regime in ("heating", "cooling"):
        coord._solar_coefficients_4d_per_unit[SENSOR][regime] = {
            "s": 0.00005, "e": 0.00003, "w": 0.00003,
            "diffuse": 0.00002, "learned": True,
        }

    elevs = list(range(10, 71)) * 4
    positions = _south_dominant_positions([float(e) for e in elevs])
    entries, sun_pos_by_ts = _build_log_with_cooling_supplement(
        positions, true_crit_s=30.0,
    )
    # Rebuild ``actual`` against the tiny coefficients so impact is small.
    # Heating: actual = base - impact (slight reduction); cooling: actual
    # = base + impact (slight increase) so the cooling-floor still gets
    # populated and the test actually exercises the noise-floor RMS gate
    # rather than being shadowed by the cooling-floor skip.
    for e in entries:
        sid = next(iter(e["unit_breakdown"]))
        base = e["unit_expected_base"][sid]
        is_cooling = (
            e["unit_modes"].get(sid) == "cooling"
        )
        if is_cooling:
            e["unit_breakdown"][sid] = base + 0.005  # tiny positive impact
        else:
            e["unit_breakdown"][sid] = base - 0.005  # tiny positive impact

    solar.get_approx_sun_pos = _stub_get_approx_sun_pos(sun_pos_by_ts)
    lm = LearningManager()
    result = lm.fit_solar_obstruction(
        hourly_log=entries, coordinator=coord, dry_run=True,
    )
    s = result[SENSOR]["s"]
    high = s["high"]
    # Result carries the absolute-amplitude field for inspection.
    assert "rms_reduction_kwh" in high
    if high["learned"] is False and high["skip_reason"] == "below_rms_amplitude_threshold":
        # The configured behaviour: fit was discarded on absolute amplitude.
        assert high["rms_reduction_kwh"] < 0.05
    else:
        # Either the relative threshold caught it first (also acceptable
        # — both gates filter noise), or the synthetic data happened to
        # land above the absolute floor.  In the latter case we'd want
        # to know — fail loudly so the test stays meaningful.
        assert high["skip_reason"] == "below_sse_threshold", (
            f"Expected below_rms_amplitude_threshold or below_sse_threshold, "
            f"got skip_reason={high['skip_reason']!r}, "
            f"rms_reduction_kwh={high['rms_reduction_kwh']!r}, "
            f"improvement_ratio={high['sse_improvement_ratio']!r}"
        )


def test_fit_include_cooling_false_skips_cooling_samples():
    """``include_cooling=False`` → all cooling-mode samples are skipped
    silently (heating-only fit).  Useful diagnostic for isolating
    cooling-driven gates from heating-driven ones.
    """
    coord, solar = _make_fit_coord(with_cooling=True)
    elevs = list(range(10, 71)) * 2
    positions = _south_dominant_positions([float(e) for e in elevs])
    heating_entries, heating_sun = _build_log_explicit(
        positions, true_crit_s=30.0, regime="heating",
    )
    cooling_entries, cooling_sun = _build_log_explicit(
        positions, true_crit_s=30.0, regime="cooling",
        start_offset_hours=len(positions),
    )
    entries = heating_entries + cooling_entries
    sun_pos_by_ts = {**heating_sun, **cooling_sun}
    solar.get_approx_sun_pos = _stub_get_approx_sun_pos(sun_pos_by_ts)

    lm = LearningManager()
    result = lm.fit_solar_obstruction(
        hourly_log=entries, coordinator=coord,
        include_cooling=False, dry_run=True,
    )
    s = result[SENSOR]["s"]
    assert s["n_cooling_samples"] == 0, s
    assert s["n_heating_samples"] > 0


def test_fit_days_back_restricts_log_window():
    """When ``days_back`` is provided, only entries within the last N
    days are consumed.  Older entries are dropped before the sample
    collector sees them.
    """
    coord, solar = _make_fit_coord()
    elevs = list(range(10, 71)) * 4
    positions = _south_dominant_positions([float(e) for e in elevs])
    entries, sun_pos_by_ts = _build_log_explicit(
        positions, true_crit_s=30.0,
    )
    # Backdate every entry to 200 days ago so a 30-day window drops them.
    from datetime import datetime, timezone, timedelta
    cutoff_long_ago = datetime.now(timezone.utc) - timedelta(days=200)
    for i, e in enumerate(entries):
        e["timestamp"] = (cutoff_long_ago + timedelta(hours=i)).isoformat()
    solar.get_approx_sun_pos = MagicMock(return_value=(45.0, 180.0))

    lm = LearningManager()
    # With days_back=30, all 200-days-ago entries are dropped → no samples.
    result_short = lm.fit_solar_obstruction(
        hourly_log=entries, coordinator=coord,
        days_back=30, dry_run=True,
    )
    assert result_short[SENSOR]["s"]["n_samples"] == 0

    # With days_back=None (default), all entries are kept.
    result_full = lm.fit_solar_obstruction(
        hourly_log=entries, coordinator=coord,
        days_back=None, dry_run=True,
    )
    assert result_full[SENSOR]["s"]["n_samples"] > 0


# ---------------------------------------------------------------------------
# #1016 apply-gate tests (opposite-side blind/explored) removed in #1020.
# The opposite-side dependency is gone — applicable is now four local
# checks per boundary.  New apply-gate behaviour is covered in
# ``test_obstruction_tightening_1020.py``.
# ---------------------------------------------------------------------------


def test_fit_data_does_not_reach_high_regime_honest_skip():
    """When data has enough cooling samples to clear the cooling-floor
    but still never reaches HIGH-regime elevations, the HIGH sweep
    reports ``data_does_not_reach_high_regime``.  Cooling supplement
    is needed under #1020 — without it cooling-floor would fire
    first.  Both gates measure data coverage, but the cooling-floor
    is about regime credibility (need cooling-mode evidence) while
    data_does_not_reach is about elevation reach.
    """
    coord, solar = _make_fit_coord(with_cooling=True)
    elevs = list(range(5, 15)) * 10  # 100 samples per regime, 5°-14°
    positions = _south_dominant_positions([float(e) for e in elevs])
    entries, sun_pos_by_ts = _build_log_with_cooling_supplement(
        positions, true_crit_s=None,
    )
    solar.get_approx_sun_pos = _stub_get_approx_sun_pos(sun_pos_by_ts)
    lm = LearningManager()
    result = lm.fit_solar_obstruction(
        hourly_log=entries, coordinator=coord, dry_run=False,
    )
    s = result[SENSOR]["s"]
    # HIGH skipped with the honest reason — cooling-floor passes
    # (n_cooling >= 50) so the data-reach check is reached.
    assert s["high"]["skip_reason"] == "data_does_not_reach_high_regime", s["high"]
    assert s["high"]["learned"] is False
    assert s["high"]["best_critical_elev"] is None
    # LOW sweep runs normally (its range still has data).
    assert s["low"]["skip_reason"] != "data_does_not_reach_low_regime"


def test_fit_coefficient_quality_monotonic_improvement(monkeypatch):
    """Numerical sanity that SSE-improvement is monotonic in coefficient
    quality.  Refitting coefficients on gated geometry sharpens them;
    the same fit run with sharpened coefficients must show ≥
    SSE-improvement.  This was the foundation of the #1016 convergence
    loop; the loop was removed in #1020 (auto-write gone, nothing to
    converge against), but the underlying monotonicity remains the
    correctness property batch_fit_solar_4d relies on when the user
    chooses to follow up an accepted gate with a refit.
    """
    coord, solar = _make_fit_coord(with_cooling=True)
    elevs = list(range(10, 71)) * 4
    positions = _south_dominant_positions([float(e) for e in elevs])
    entries, sun_pos_by_ts = _build_log_with_cooling_supplement(
        positions, true_crit_s=30.0,
    )
    solar.get_approx_sun_pos = _stub_get_approx_sun_pos(sun_pos_by_ts)
    lm = LearningManager()

    # PASS 1 — coefficients deliberately biased low (50 % of true).
    biased_regime = {
        "s": 0.0015 * 0.5,
        "e": 0.0010 * 0.5,
        "w": 0.0010 * 0.5,
        "diffuse": 0.0005 * 0.5,
        "learned": True,
    }
    coord._solar_coefficients_4d_per_unit[SENSOR] = {
        "heating": dict(biased_regime),
        "cooling": dict(biased_regime),
    }
    result1 = lm.fit_solar_obstruction(
        hourly_log=entries, coordinator=coord, dry_run=True,
    )
    imp1 = result1[SENSOR]["s"]["high"]["sse_improvement_ratio"]

    # PASS 2 — coefficients restored to true value (the refit's job).
    true_regime = {
        "s": 0.0015, "e": 0.0010, "w": 0.0010, "diffuse": 0.0005,
        "learned": True,
    }
    coord._solar_coefficients_4d_per_unit[SENSOR] = {
        "heating": dict(true_regime),
        "cooling": dict(true_regime),
    }
    result2 = lm.fit_solar_obstruction(
        hourly_log=entries, coordinator=coord, dry_run=True,
    )
    imp2 = result2[SENSOR]["s"]["high"]["sse_improvement_ratio"]

    # Monotonicity: refitted coefficients yield ≥ improvement.
    # Allow a 1e-6 noise floor on the inequality.
    assert imp2 + 1e-6 >= imp1, (
        f"convergence non-monotonic: pass1={imp1:.4f}, pass2={imp2:.4f}"
    )
    # And the refit moved the needle meaningfully on a biased start:
    # at 50 % bias the issue's analytical table predicts improvement
    # drops from 1.00 → ~0.69.  Demand at least a modest jump
    # (≥ 0.05) so a no-op convergence regression would fire.
    assert imp2 - imp1 >= 0.05, (
        f"convergence loop produced negligible improvement: "
        f"pass1={imp1:.4f}, pass2={imp2:.4f}"
    )


def test_fit_entity_id_arg_restricts_to_one_unit():
    """When ``entity_id=`` is passed, only that entity is fitted; other
    entities receive no entry in the result.
    """
    sensor_a = "sensor.a"
    sensor_b = "sensor.b"
    coord, solar = _make_fit_coord(sensor_ids=[sensor_a, sensor_b])

    elevs = list(range(10, 71)) * 4
    positions = _south_dominant_positions([float(e) for e in elevs])
    # Only entity A has samples in this synthetic log (we point both
    # entities at A's data, but request fit on B alone — B's coeffs
    # are learned, so it would otherwise have produced a fit too).
    entries, sun_pos_by_ts = _build_log_explicit(
        positions, true_crit_s=30.0, sensor_id=sensor_a,
    )
    solar.get_approx_sun_pos = _stub_get_approx_sun_pos(sun_pos_by_ts)

    lm = LearningManager()
    result = lm.fit_solar_obstruction(
        hourly_log=entries, coordinator=coord,
        entity_id=sensor_b, dry_run=False,
    )
    assert sensor_a not in result
    assert sensor_b in result
    # B had no samples (no entry in unit_breakdown) → all facades skipped.
    assert result[sensor_b]["s"]["n_samples"] == 0
