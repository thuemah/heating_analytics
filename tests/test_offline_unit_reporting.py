"""Offline fits, diagnostics and live shutdown detection read unit reporting
the way live per-unit learning does.

``unit_breakdown`` drops units at 0 kWh, so readers used to pick a reading
of an absent unit on their own: some as 0 kWh, some as "skip the hour".
Neither tells a reported zero hour (live learns 0) from a sensor that did
not report (live skips the unit).  Every reader here now goes through
``helpers.unit_hour_reported_kwh`` — except the per-unit min-base
calibration and ``diagnose_model``'s mode contamination, which want
"consumed energy", not "reported", on purpose.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

from custom_components.heating_analytics.diagnostics import DiagnosticsEngine
from custom_components.heating_analytics.learning import LearningManager
from custom_components.heating_analytics.observation import (
    build_strategies,
    detect_solar_shutdown_entities,
)
from custom_components.heating_analytics.solar import SolarCalculator

from tests.test_batch_fit_solar_4d import (
    _entry as _entry_4d,
    _make_coord as _make_coord_4d,
)
from tests.test_inequality_learning import _shutdown_entry, _shutdown_env
from tests.test_per_unit_min_base import (
    MockCoord as _MinBaseCoord,
    _calibrate,
    _fill_dark_samples,
)
from tests.test_solar_diagnose import _hour_entry as _diag_entry
from tests.test_solar_diagnose import TestElevationDiagnosticsLag as _Lag
from tests.test_solar_diagnose import _make_coord as _make_diag_coord
from tests.test_solar_obstruction_gate import (
    SENSOR as _OBS_SENSOR,
    _build_log_explicit,
    _make_fit_coord,
    _south_dominant_positions,
    _stub_get_approx_sun_pos,
)


def _not_reporting(entry: dict, entity_id: str) -> dict:
    entry["unit_breakdown"] = {k: v for k, v in entry["unit_breakdown"].items() if k != entity_id}
    entry["units_reporting"] = sorted(entry["unit_breakdown"])
    return entry


def _reported_zero(entry: dict, entity_id: str) -> dict:
    entry["unit_breakdown"] = {k: v for k, v in entry["unit_breakdown"].items() if k != entity_id}
    entry["units_reporting"] = sorted(set(entry["unit_breakdown"]) | {entity_id})
    return entry


# ---------------------------------------------------------------------------
# Absent read as 0 → an unreported hour is now skipped
# ---------------------------------------------------------------------------

_E3 = "sensor.heater3d"


def _collect_3d(entries, *, for_tobit=True, match_diagnose=False):
    coordinator = MagicMock()
    coordinator.model.correlation_data_per_unit = {_E3: {"10": {"normal": 2.0}}}
    return LearningManager()._collect_batch_fit_samples(
        entity_id=_E3,
        regime="heating",
        hourly_log=entries,
        entry_potentials=[(1.0, 0.0, 0.0, 1.0)] * len(entries),
        coordinator=coordinator,
        unit_threshold=0.1,
        screen_affected_entities=None,
        for_tobit=for_tobit,
        match_diagnose=match_diagnose,
        solar_coefficients_per_unit={},
    )


def _entry_3d(i: int, actual: float) -> dict:
    return {
        "timestamp": f"2026-04-01T{i:02d}:00:00",
        "unit_modes": {_E3: "heating"},
        "unit_breakdown": {_E3: actual},
        "unit_expected_breakdown": {_E3: 2.0},
        "temp_key": "10",
        "wind_bucket": "normal",
        "learning_status": "active",
    }


@pytest.mark.parametrize(
    ("for_tobit", "match_diagnose"),
    [(True, False), (False, True)],
    ids=["batch_fit_tobit", "apply_implied"],
)
def test_3d_collector_skips_hours_the_sensor_did_not_report(for_tobit, match_diagnose):
    entries = [_entry_3d(i, 1.5) for i in range(6)]
    entries += [_not_reporting(_entry_3d(6 + i, 1.5), _E3) for i in range(3)]
    samples, censored, drops = _collect_3d(
        entries, for_tobit=for_tobit, match_diagnose=match_diagnose,
    )
    assert drops["not_reporting"] == 3
    assert len(samples) == 6
    assert not any(censored)


def test_3d_collector_keeps_a_reported_zero_hour_as_censored():
    entries = [_entry_3d(i, 1.5) for i in range(6)]
    entries += [_reported_zero(_entry_3d(6 + i, 1.5), _E3) for i in range(3)]
    samples, censored, drops = _collect_3d(entries)
    assert drops["not_reporting"] == 0
    assert len(samples) == 9
    assert sum(censored) == 3


def test_3d_collector_legacy_absent_unit_reads_zero_after_its_first_report():
    """Entries without ``units_reporting``: absent before the unit's first
    report is "not reporting"; absent afterwards is a 0 kWh hour."""
    early = [_entry_3d(i, 1.5) for i in range(2)]
    for e in early:
        e["unit_breakdown"] = {}
    later = [_entry_3d(2 + i, 1.5) for i in range(4)]
    absent = [_entry_3d(6 + i, 1.5) for i in range(2)]
    for e in absent:
        e["unit_breakdown"] = {}
    samples, censored, drops = _collect_3d(early + later + absent)
    assert drops["not_reporting"] == 2
    assert len(samples) == 6
    assert sum(censored) == 2


def test_4d_collector_skips_hours_the_sensor_did_not_report():
    sid = "sensor.heater1"
    coord = _make_coord_4d(correlation_data_per_unit={sid: {"10": {"normal": 2.5}}})
    log = [
        _entry_4d(f"2026-04-{i + 1:02d}T11:00:00", sensor_id=sid, actual_kwh=1.5)
        for i in range(6)
    ]
    for e in log[4:]:
        _not_reporting(e, sid)
    samples, _, drops = LearningManager()._collect_batch_fit_samples_4d(
        entity_id=sid,
        regime="heating",
        hourly_log=log,
        coordinator=coord,
        screen_affected_entities=None,
    )
    assert drops["not_reporting"] == 2
    assert len(samples) == 4, drops


def test_diagnose_implied_skips_hours_the_sensor_did_not_report():
    base_dt = datetime(2026, 4, 1, 12, 0)
    entries = [
        _diag_entry(
            (base_dt + timedelta(hours=i)).isoformat(),
            solar_s=0.5, solar_e=0.15 if i % 2 else 0.05,
            actual=1.5, base=2.0, temp=5.0,
        )
        for i in range(12)
    ]
    clean = DiagnosticsEngine(_make_diag_coord(list(entries))).diagnose_solar(days_back=30)
    for i in range(3):
        e = _diag_entry(
            (base_dt + timedelta(hours=12 + i)).isoformat(),
            solar_s=0.5, solar_e=0.15, actual=1.5, base=2.0, temp=5.0,
        )
        entries.append(_not_reporting(e, "sensor.heater1"))
    offline = DiagnosticsEngine(_make_diag_coord(entries)).diagnose_solar(days_back=30)
    assert offline["global"]["excluded"]["not_reporting"] == 3
    # Read as 0 kWh they would have been counted as saturated hours.
    assert offline["global"]["excluded"]["saturated"] == clean["global"]["excluded"]["saturated"]
    assert (
        offline["per_unit"]["sensor.heater1"]["implied_coefficient_30d"]
        == clean["per_unit"]["sensor.heater1"]["implied_coefficient_30d"]
    )


# ---------------------------------------------------------------------------
# Absent read as "skip" → a reported zero hour now counts
# ---------------------------------------------------------------------------


def _shutdown_obstruction_run(mark):
    coord, solar = _make_fit_coord()
    positions = _south_dominant_positions([float(e) for e in range(10, 71)])
    entries, sun_pos_by_ts = _build_log_explicit(positions)
    for e in entries:
        e["solar_dominant_entities"] = [_OBS_SENSOR]
        mark(e, _OBS_SENSOR)
    solar.get_approx_sun_pos = _stub_get_approx_sun_pos(sun_pos_by_ts)
    result = LearningManager().fit_solar_obstruction(
        hourly_log=entries, coordinator=coord, dry_run=True,
    )
    return result, len(entries)


def test_obstruction_fit_takes_reported_zero_shutdown_hours_as_constraints():
    """A unit fully off in the sun (reported 0 kWh) is the cleanest proof of
    unobstructed sun; presence in ``unit_breakdown`` dropped exactly those."""
    result, n = _shutdown_obstruction_run(_reported_zero)
    assert result[_OBS_SENSOR]["s"]["n_shutdown_constraints"] == n
    assert result["n_skipped_learning_status"]["not_reporting"] == 0


def test_obstruction_fit_skips_hours_the_sensor_did_not_report():
    result, n = _shutdown_obstruction_run(_not_reporting)
    assert result[_OBS_SENSOR]["s"]["n_shutdown_constraints"] == 0
    assert result["n_skipped_learning_status"]["not_reporting"] == n


def test_lag_walk_counts_tails_the_unit_reported_at_zero():
    """A tail hour in which the unit is still off is the thermal-battery
    signal itself; it used to be dropped as missing."""
    entries, elevations = _Lag._make_train(
        originator_dt=datetime(2026, 4, 1, 12, 0), originator_elev=20.0,
        n_trains=12, tail_actual=0.7, base=2.0,
    )
    for e in entries:
        if e["solar_vector_s"] == 0.0:
            _reported_zero(e, "sensor.heater1")
    coord = _Lag._coord_with_elevations(entries, elevations)
    result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
    bucket = result["per_unit"]["sensor.heater1"]["elevation_diagnostics"]["lag"]["15-30"]
    for k in range(1, 7):
        assert bucket[f"lag_{k}"]["n"] == 12
        assert bucket[f"lag_{k}"]["mean_residual_kwh"] == pytest.approx(2.0)


def test_lag_walk_finds_tails_across_the_dst_fall_back():
    """The walk steps H + k hours on UTC instants.  Keyed on the timestamp
    string, every tail after the clocks went back carried the other offset
    and was missed."""
    oslo = ZoneInfo("Europe/Oslo")
    # 00:00+02:00 on the fall-back day; the next six hours cross 03:00→02:00.
    start_utc = datetime(2026, 10, 24, 22, 0, tzinfo=timezone.utc)
    entries = []
    elevations = []
    for k in range(7):
        ts = (start_utc + timedelta(hours=k)).astimezone(oslo).isoformat()
        if k == 0:
            entries.append(_diag_entry(ts, solar_s=0.5, actual=1.5, base=2.0))
            elevations.append(20.0)
        else:
            entries.append(_diag_entry(
                ts, solar_s=0.0, solar_factor=0.0, actual=1.7, base=2.0,
            ))
            elevations.append(-1.0)
    offsets = {e["timestamp"][-6:] for e in entries}
    assert offsets == {"+02:00", "+01:00"}
    coord = _Lag._coord_with_elevations(entries, elevations)
    result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
    bucket = result["per_unit"]["sensor.heater1"]["elevation_diagnostics"]["lag"]["15-30"]
    for k in range(7):
        assert bucket[f"lag_{k}"]["n"] == 1, k


def test_dni_dhi_shadow_per_entity_counts_reported_zero_hours():
    """The per-entity target ``unit_base − unit_actual`` takes a reported
    0 kWh hour as a sample, and leaves out hours the sensor did not report."""
    from tests.test_solar_diagnose import TestDniDhiShadowReport

    helper = TestDniDhiShadowReport()
    log = helper._make_log(n_clear=80, n_broken=80, n_overcast=80)
    for i, entry in enumerate(log):
        entry["unit_breakdown"] = {"sensor.heater1": 1.5, "sensor.heater2": 0.6}
        if i % 3 == 1:
            _reported_zero(entry, "sensor.heater2")
        elif i % 3 == 2:
            _not_reporting(entry, "sensor.heater2")
    coord = helper._coord_with_sun(log, correlation_data={"10": {"normal": 2.0}})
    coord.energy_sensors = ["sensor.heater1", "sensor.heater2"]
    coord._correlation_data_per_unit = {
        "sensor.heater1": {"10": {"normal": 1.6}},
        "sensor.heater2": {"10": {"normal": 0.8}},
    }
    result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
    per_e = result["dni_dhi_shadow"]["cross_check_actual"]["per_entity"]
    assert per_e["sensor.heater1"]["n_hours"] == 240
    # Two thirds reported (0.6 kWh or 0 kWh); one third did not report.
    assert per_e["sensor.heater2"]["n_hours"] == 160


# ---------------------------------------------------------------------------
# Presence means consumed energy — unchanged on purpose
# ---------------------------------------------------------------------------


def test_min_base_calibration_keeps_idle_hours_out():
    """The p10 is the noise floor of a running unit.  Reported zero hours
    stay out — counted in, p10 would collapse to the floor for any unit
    that cycles off in dark hours."""
    sid = "sensor.a"
    coord = _MinBaseCoord([sid])
    _fill_dark_samples(coord, {sid: [0.20 + 0.01 * i for i in range(60)]})
    clean = _calibrate(coord)

    ts = coord._hourly_log[-1]["timestamp"]
    for _ in range(40):
        coord._hourly_log.append({
            "timestamp": ts,
            "hour": int(ts[11:13]),
            "solar_factor": 0.0,
            "auxiliary_active": False,
            "unit_modes": {sid: "heating"},
            "unit_breakdown": {},
            "units_reporting": [sid],
            "learning_status": "active",
        })
    coord._per_unit_min_base_thresholds = {}
    result = _calibrate(coord)
    assert result["units"][sid]["dark_samples"] == clean["units"][sid]["dark_samples"]
    assert result["units"][sid]["p10_actual"] == clean["units"][sid]["p10_actual"]


# ---------------------------------------------------------------------------
# Live shutdown detection and the replay's inequality branch
# ---------------------------------------------------------------------------


def _detect(unit_actual_kwh):
    return detect_solar_shutdown_entities(
        solar_enabled=True,
        is_aux_dominant=False,
        potential_vector=(0.8, 0.0, 0.0),
        energy_sensors=["sensor.vp", "sensor.other"],
        unit_modes={},
        unit_actual_kwh=unit_actual_kwh,
        unit_expected_base_kwh={"sensor.vp": 1.0, "sensor.other": 1.0},
    )


def test_live_shutdown_detection_ignores_a_sensor_that_did_not_report():
    """Offline for the hour is no signal — not a shutdown."""
    assert _detect({"sensor.other": 1.0}) == ()


def test_live_shutdown_detection_still_flags_a_reported_zero():
    assert _detect({"sensor.vp": 0.0, "sensor.other": 1.0}) == ("sensor.vp",)


def _replay_shutdown(entries):
    coord = _shutdown_env()
    solar_coeffs: dict = {}
    diag = LearningManager().replay_solar_nlms(
        entries,
        solar_calculator=SolarCalculator(coord),
        screen_config=coord.screen_config,
        correlation_data_per_unit={"sensor.vp_stue": {"10": {"normal": 1.0}}},
        solar_coefficients_per_unit=solar_coeffs,
        learning_buffer_solar_per_unit={},
        energy_sensors=coord.energy_sensors,
        learning_rate=1.0,
        balance_point=15.0,
        aux_affected_entities=coord.aux_affected_entities,
        unit_strategies=coord._unit_strategies,
        daily_history={},
        return_diagnostics=True,
    )
    return diag, solar_coeffs


def test_replay_inequality_skips_a_unit_that_did_not_report():
    """Live skips an unreported unit before any solar step, the inequality
    lift included; the replay does the same."""
    entries = [
        _not_reporting(_shutdown_entry(f"2026-05-01T{h:02d}:00:00"), "sensor.vp_stue")
        for h in range(12, 20)
    ]
    diag, solar_coeffs = _replay_shutdown(entries)
    assert diag["inequality_updates"] == 0
    assert diag["unit_skipped_not_reporting"] == len(entries)
    assert "sensor.vp_stue" not in solar_coeffs


def test_replay_inequality_still_lifts_on_a_reported_zero():
    entries = [
        _reported_zero(_shutdown_entry(f"2026-05-01T{h:02d}:00:00"), "sensor.vp_stue")
        for h in range(12, 20)
    ]
    diag, _ = _replay_shutdown(entries)
    assert diag["inequality_updates"] > 0
    assert diag["unit_skipped_not_reporting"] == 0
