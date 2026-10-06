"""The per-unit retrain replay learns what live per-unit learning learned.

``LearningManager.replay_per_unit_models`` rebuilds the per-unit base
buckets for ``retrain_from_history`` and ``retrain_unit_from_history``.  It
used to apply only the cooling route: OFF / guest hours were learned,
DHW was learned as its metered kWh, zero hours were skipped, aux hours of
aux-affected units went into the base bucket, and the EMA ran at the raw
learning rate.  These tests run the same synthetic hours through live
``_process_per_unit_learning`` and through the replay (from the log
entries ``hourly_processor`` would have written) and require the two to
end in the same state — one test per decision.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from custom_components.heating_analytics.const import (
    COOLING_WIND_BUCKET,
    MODE_COOLING,
    MODE_DHW,
    MODE_GUEST_COOLING,
    MODE_GUEST_HEATING,
    MODE_HEATING,
    MODE_OFF,
    PER_UNIT_LEARNING_RATE_CAP,
)
from custom_components.heating_analytics.helpers import (
    first_reported_instants,
    hour_learning_skip_reason,
    log_entry_instant,
    unit_hour_reported_kwh,
)
from custom_components.heating_analytics.learning import LearningManager
from custom_components.heating_analytics.observation import (
    DirectMeter,
    ModelState,
    hourly_learning_sensors,
)

HP = "sensor.heat_pump"
CABLE = "sensor.heating_cable"
GUEST = "sensor.guest_room"
DHW = "sensor.hot_water"
SENSORS = [HP, CABLE, GUEST, DHW]
BP = 17.0
T0 = datetime(2026, 1, 10, 0, 0)


class _Solar:
    """Solar calculator stub: a fixed south coefficient, dot-product impact."""

    def calculate_unit_coefficient(self, entity_id, temp_key, mode):
        return {"s": 0.4, "e": 0.0, "w": 0.0}

    def calculate_unit_solar_impact(self, vector, coeff):
        return coeff["s"] * vector[0] + coeff["e"] * vector[1] + coeff["w"] * vector[2]


def _hour(
    i: int,
    delta: dict,
    *,
    temp: float = 5.0,
    modes: dict | None = None,
    aux: bool = False,
    status: str = "active",
    cooldown: bool = False,
    solar_factor: float = 0.0,
    solar_s: float = 0.0,
    shutdown: tuple = (),
) -> dict:
    """One synthetic hour: what the live collector saw."""
    return {
        "ts": (T0 + timedelta(hours=i)).isoformat(),
        "temp": temp,
        "temp_key": str(int(round(temp))),
        "wind_bucket": "normal",
        "delta": dict(delta),  # the meter map: present = reported
        "modes": dict(modes or {}),  # sparse, like coordinator._unit_modes
        "aux": aux,
        "status": status,
        "cooldown": cooldown,
        "solar_factor": solar_factor,
        "solar_s": solar_s,
        "shutdown": tuple(shutdown),
    }


def _fresh() -> dict:
    return {"corr": {}, "buf": {}, "counts": {}, "aux": {}, "aux_buf": {}}


def _lookup(corr: dict, entity_id: str, temp_key: str, bucket: str) -> float:
    return float(((corr.get(entity_id) or {}).get(temp_key) or {}).get(bucket, 0.0))


def _run_live(hours, sensors=SENSORS, *, learning_rate=0.10, aux_affected=None,
              solar=None, unit_min_base=None, all_sensors=None, state=None) -> dict:
    """Live per-unit learning, on the hours ``process_learning`` runs it for.

    ``sensors`` is the list ``hourly_processor`` hands live learning;
    ``all_sensors`` (default: the same) every energy sensor, whose expected
    base ``calculate_total_power`` accumulates.
    """
    lm = LearningManager()
    m = state if state is not None else _fresh()
    all_sensors = sensors if all_sensors is None else all_sensors
    for h in hours:
        if hour_learning_skip_reason({"learning_status": h["status"]}) is not None:
            continue

        def bucket(sid):
            mode = h["modes"].get(sid, MODE_HEATING)
            return COOLING_WIND_BUCKET if mode == MODE_COOLING else h["wind_bucket"]

        # What calculate_total_power accumulated during the hour: every
        # unit's expected base from the model as it stood.
        expected = {
            sid: _lookup(m["corr"], sid, h["temp_key"], bucket(sid)) for sid in all_sensors
        }
        lm._process_per_unit_learning(
            h["temp_key"], h["wind_bucket"], h["temp"], (h["solar_s"], 0.0, 0.0),
            sum(h["delta"].values()), 0.0,
            sensors, dict(h["delta"]),
            solar is not None, learning_rate, solar,
            lambda eid, tk, wb, t: _lookup(m["corr"], eid, tk, wb),
            m["buf"], m["corr"], m["counts"],
            h["aux"], m["aux"], m["aux_buf"],
            {}, {}, BP,
            h["modes"], {}, expected,
            aux_affected,
            is_cooldown_active=h["cooldown"],
            correction_percent=100.0,
            solar_dominant_entities=h["shutdown"],
            screen_config=None,
            screen_affected_entities=None,
            # Solar learning off; the solar stub still feeds headroom.
            solar_affected_entities=frozenset(),
            solar_factor=h["solar_factor"],
            unit_min_base=unit_min_base,
        )
    return m


def _entry(h: dict, sensors=SENSORS, *, aux_affected=None, legacy=False) -> dict:
    """The hourly_log entry hourly_processor writes for the hour."""
    entry = {
        "timestamp": h["ts"],
        "temp": h["temp"],
        "temp_key": h["temp_key"],
        "wind_bucket": h["wind_bucket"],
        "unit_breakdown": {
            sid: round(kwh, 3) for sid, kwh in h["delta"].items() if kwh > 0
        },
        "unit_modes": {
            sid: mode for sid, mode in h["modes"].items() if mode != MODE_HEATING
        },
        "auxiliary_active": h["aux"],
        "learning_status": "cooldown_post_aux" if h["cooldown"] else h["status"],
        "solar_factor": h["solar_factor"],
        "solar_vector_s": h["solar_s"],
        "solar_vector_e": 0.0,
        "solar_vector_w": 0.0,
        "correction_percent": 100.0,
        "solar_dominant_entities": list(h["shutdown"]),
    }
    if not legacy:
        entry["units_reporting"] = sorted(h["delta"])
    if h["cooldown"]:
        entry["aux_cooldown_entities"] = sorted(
            sid for sid in sensors if aux_affected is None or sid in aux_affected
        )
    return entry


def _run_replay(entries, sensors=SENSORS, *, learning_rate=0.10, aux_affected=None,
                solar=None, unit_min_base=None, replay_aux=True, target=None,
                state=None):
    m = state if state is not None else _fresh()
    model = ModelState(
        correlation_data={},
        correlation_data_per_unit=m["corr"],
        observation_counts=m["counts"],
        aux_coefficients={},
        aux_coefficients_per_unit=m["aux"],
        solar_coefficients_per_unit={},
        learned_u_coefficient=None,
        learning_buffer_per_unit=m["buf"],
        learning_buffer_aux_per_unit=m["aux_buf"],
    )
    report = LearningManager().replay_per_unit_models(
        entries,
        {sid: DirectMeter(sid) for sid in sensors},
        model,
        learning_rate,
        target_entity=target,
        balance_point=BP,
        solar_calculator=solar,
        solar_enabled=solar is not None,
        unit_min_base=unit_min_base,
        aux_affected_entities=set(aux_affected) if aux_affected is not None else None,
        replay_aux=replay_aux,
    )
    return m, report


def _assert_parity(hours, **kwargs):
    aux_affected = kwargs.get("aux_affected")
    live = _run_live(hours, **kwargs)
    replay_kwargs = {k: v for k, v in kwargs.items()}
    entries = [_entry(h, aux_affected=aux_affected) for h in hours]
    replayed, report = _run_replay(entries, **replay_kwargs)
    assert replayed["corr"] == live["corr"]
    assert replayed["buf"] == live["buf"]
    assert replayed["counts"] == live["counts"]
    assert replayed["aux"] == live["aux"]
    assert replayed["aux_buf"] == live["aux_buf"]
    return live, report


# ---------------------------------------------------------------------------
# One test per decision
# ---------------------------------------------------------------------------


def test_off_and_guest_hours_are_not_learned():
    hours = [
        _hour(i, {HP: 1.0, CABLE: 0.2, GUEST: 0.6},
              modes={GUEST: [MODE_GUEST_HEATING, MODE_OFF, MODE_GUEST_COOLING][i % 3]})
        for i in range(8)
    ]
    live, report = _assert_parity(hours, sensors=[HP, CABLE, GUEST])
    assert GUEST not in live["corr"] and GUEST not in live["buf"]
    assert report["units_skipped_mode"] == 8


def test_dhw_is_learned_as_zero():
    # DHW energy never enters the meter map, so the unit is absent.
    hours = [_hour(i, {HP: 1.0}, modes={DHW: MODE_DHW}) for i in range(5)]
    live, _ = _assert_parity(hours, sensors=[HP, DHW])
    assert live["corr"][DHW]["5"]["normal"] == 0.0


def test_dhw_hour_with_pre_switch_kwh_is_still_learned_as_zero():
    # A mid-hour switch to DHW leaves the heating part in the breakdown.
    hours = [_hour(i, {HP: 1.0, DHW: 0.7}, modes={DHW: MODE_DHW}) for i in range(4)]
    live, _ = _assert_parity(hours, sensors=[HP, DHW])
    assert live["corr"][DHW]["5"]["normal"] == 0.0


def test_reported_zero_hours_are_learned_and_offline_hours_skipped():
    # The cable reports 0 in most hours, 1.2 in some, and is offline twice.
    pattern = [0.0, 0.0, 1.2, 0.0, None, 0.0, 1.2, 0.0, None, 0.0]
    hours = []
    for i, kwh in enumerate(pattern):
        delta = {HP: 1.0}
        if kwh is not None:
            delta[CABLE] = kwh
        hours.append(_hour(i, delta))
    live, report = _assert_parity(hours, sensors=[HP, CABLE])
    # Jump-start from the first four reported hours: (0 + 0 + 1.2 + 0) / 4.
    assert report["units_skipped_not_reporting"] == 2
    assert report["units_learned_base_zero_kwh"] == 6
    assert live["counts"][CABLE]["5"]["normal"] == 1 + 4  # jump-start + 4 EMA steps


def test_aux_hours_of_aux_affected_units_go_to_the_aux_coefficient():
    dark = [_hour(i, {HP: 2.0, CABLE: 0.5}) for i in range(4)]
    aux = [_hour(4 + i, {HP: 0.8, CABLE: 0.5}, aux=True) for i in range(6)]
    live, report = _assert_parity(dark + aux, sensors=[HP, CABLE], aux_affected=[HP])
    # The heat pump's base bucket is untouched by aux hours...
    assert live["corr"][HP]["5"]["normal"] == 2.0
    # ...its aux coefficient learned the reduction (4 buffered, then EMA).
    assert live["aux"][HP]["5"]["normal"] > 0.0
    # The cable is outside the aux scope and learns its base.
    assert live["counts"][CABLE]["5"]["normal"] == 1 + 6
    assert report["units_learned_aux"] == 6


def test_aux_hours_are_not_learned_when_aux_is_not_replayed():
    """retrain_unit_from_history: aux-affected aux hours never reach the base."""
    dark = [_hour(i, {HP: 2.0}) for i in range(4)]
    aux = [_hour(4 + i, {HP: 0.1}, aux=True) for i in range(6)]
    entries = [_entry(h, aux_affected=[HP]) for h in dark + aux]
    m, report = _run_replay(entries, sensors=[HP], aux_affected=[HP], replay_aux=False)
    assert m["corr"][HP]["5"]["normal"] == 2.0
    assert m["aux"] == {} and m["aux_buf"] == {}
    assert report["units_skipped_aux_not_replayed"] == 6


def test_dhw_during_aux_learns_nothing():
    hours = [_hour(i, {HP: 1.0}, aux=True, modes={DHW: MODE_DHW}) for i in range(5)]
    _, report = _assert_parity(hours, sensors=[HP, DHW], aux_affected=[HP, DHW])
    assert report["units_skipped_dhw_during_aux"] == 5


def test_post_aux_cooldown_freezes_only_aux_affected_units():
    hours = [_hour(i, {HP: 1.5, CABLE: 0.4}, cooldown=True) for i in range(6)]
    live, report = _assert_parity(hours, sensors=[HP, CABLE], aux_affected=[HP])
    assert HP not in live["corr"] and HP not in live["buf"]
    assert live["counts"][CABLE]["5"]["normal"] == 1 + 2
    assert report["units_skipped_post_aux_cooldown"] == 6


@pytest.mark.parametrize(
    "status",
    ["skipped_mixed_mode", "skipped_dual_interference", "disabled", "skipped_no_data"],
)
def test_hours_live_per_unit_learning_skipped_are_not_learned(status):
    hours = [_hour(i, {HP: 1.0}) for i in range(4)]
    hours.append(_hour(4, {HP: 9.0}, status=status))
    live, report = _assert_parity(hours, sensors=[HP])
    assert live["corr"][HP]["5"]["normal"] == 1.0
    assert report["hours_skipped_learning_status"] == 1


@pytest.mark.parametrize("status", ["skipped_global_saturation", "disabled_global_only"])
def test_hours_with_only_the_global_write_skipped_are_learned(status):
    hours = [_hour(i, {HP: 1.0}, status=status) for i in range(5)]
    live, _ = _assert_parity(hours, sensors=[HP])
    assert live["counts"][HP]["5"]["normal"] == 2


def test_ema_rate_is_capped_and_weighted_by_headroom_and_snr():
    """3 % cap, headroom multiplier and SNR weight, not the raw rate."""
    dark = [_hour(i, {HP: 2.0}) for i in range(4)]
    sunny = [_hour(4, {HP: 1.0}, solar_factor=0.2, solar_s=1.0)]
    live, _ = _assert_parity(dark + sunny, sensors=[HP], solar=_Solar(), learning_rate=0.10)
    # headroom = (2.0 − 0.4) / 2.0 = 0.8; SNR = max(0.1, 1 − 3 × 0.2) = 0.4
    rate = PER_UNIT_LEARNING_RATE_CAP * 0.8 * 0.4
    assert live["corr"][HP]["5"]["normal"] == pytest.approx(2.0 + rate * (1.0 - 2.0), abs=1e-5)


def test_shutdown_scales_the_snr_weight():
    dark = [_hour(i, {HP: 2.0, CABLE: 2.0}) for i in range(4)]
    shut = [_hour(4, {HP: 0.0, CABLE: 1.0}, solar_factor=0.0, shutdown=(HP,))]
    _assert_parity(dark + shut, sensors=[HP, CABLE], unit_min_base={HP: 0.1, CABLE: 0.1})


def test_bucket_at_zero_keeps_learning_by_ema():
    """A thermostatic load's bucket can be exactly 0.0 once zero hours are
    learned; the next hour is an EMA step, not a new cold-start buffer."""
    hours = [_hour(i, {CABLE: 0.0}) for i in range(4)]
    hours.append(_hour(4, {CABLE: 1.0}))
    live, _ = _assert_parity(hours, sensors=[CABLE])
    assert live["corr"][CABLE]["5"]["normal"] == pytest.approx(0.03)
    assert live["buf"][CABLE]["5"]["normal"] == []


def test_cooling_hours_go_to_the_cooling_bucket():
    hours = [_hour(i, {HP: 0.9}, temp=26.0, modes={HP: MODE_COOLING}) for i in range(5)]
    live, _ = _assert_parity(hours, sensors=[HP])
    assert set(live["corr"][HP]["26"]) == {COOLING_WIND_BUCKET}


def test_sparse_modes_resolve_to_heating():
    """``unit_modes`` omits heating units; the replay resolves per sensor."""
    hours = [_hour(i, {HP: 1.0, CABLE: 0.3}, modes={CABLE: MODE_OFF}) for i in range(4)]
    live, _ = _assert_parity(hours, sensors=[HP, CABLE])
    assert HP in live["corr"] and CABLE not in live["corr"]


def test_mixed_script_matches_live():
    """Every decision interleaved over two temperatures, three units."""
    hours = []
    for i in range(40):
        temp = 3.0 if i % 2 else 6.0
        delta = {HP: 1.0 + 0.1 * (i % 5)}
        if i % 7 != 3:
            delta[CABLE] = 0.0 if i % 3 else 0.9
        modes = {GUEST: MODE_GUEST_HEATING} if i % 4 == 0 else {}
        delta[GUEST] = 0.4
        hours.append(_hour(
            i, delta, temp=temp, modes=modes,
            aux=(10 <= i < 16), cooldown=(16 <= i < 19),
            status="skipped_mixed_mode" if i == 25 else "active",
            solar_factor=0.3 if 28 <= i < 34 else 0.0,
            solar_s=0.8 if 28 <= i < 34 else 0.0,
        ))
    _assert_parity(hours, sensors=[HP, CABLE, GUEST], aux_affected=[HP], solar=_Solar())


# ---------------------------------------------------------------------------
# Reading reporting from the log (#1120)
# ---------------------------------------------------------------------------


def _log_entry(i, breakdown, *, reporting=None, has_breakdown=True):
    entry = {"timestamp": (T0 + timedelta(hours=i)).isoformat()}
    if has_breakdown:
        entry["unit_breakdown"] = dict(breakdown)
    if reporting is not None:
        entry["units_reporting"] = list(reporting)
    return entry


def test_units_reporting_is_authoritative():
    entry = _log_entry(0, {HP: 1.0}, reporting=[HP, CABLE])
    assert unit_hour_reported_kwh(entry, HP) == 1.0
    assert unit_hour_reported_kwh(entry, CABLE) == 0.0
    assert unit_hour_reported_kwh(entry, GUEST) is None


def test_legacy_absent_unit_reads_zero_once_it_has_reported():
    log = [
        _log_entry(0, {HP: 1.0}),
        _log_entry(1, {HP: 1.0, CABLE: 0.5}),
        _log_entry(2, {HP: 1.0}),
        _log_entry(3, {}),
    ]
    first = first_reported_instants(log)
    assert first[CABLE] == log_entry_instant(log[1])
    # Before its first report the cable was most likely not configured.
    assert unit_hour_reported_kwh(log[0], CABLE, first) is None
    assert unit_hour_reported_kwh(log[1], CABLE, first) == 0.5
    assert unit_hour_reported_kwh(log[2], CABLE, first) == 0.0
    # An empty breakdown is the unit's own idle hour (single-unit installs).
    assert unit_hour_reported_kwh(log[3], HP, first) == 0.0
    # Without an anchor an absent unit stays unreported.
    assert unit_hour_reported_kwh(log[2], CABLE) is None


def test_row_without_breakdown_carries_no_per_unit_evidence():
    """A CSV-imported row has no ``unit_breakdown`` at all."""
    first = first_reported_instants([_log_entry(0, {HP: 1.0})])
    assert unit_hour_reported_kwh(_log_entry(5, {}, has_breakdown=False), HP, first) is None


def test_first_report_counts_units_reporting_zero():
    log = [_log_entry(0, {}, reporting=[CABLE]), _log_entry(1, {CABLE: 0.4})]
    assert first_reported_instants(log)[CABLE] == log_entry_instant(log[0])


def test_legacy_replay_learns_zero_hours_after_first_report():
    """Entries logged before ``units_reporting``: the thermostatic load's
    zero hours are learned (it was reporting), its hours before it was
    added are not."""
    hours = [_hour(i, {HP: 1.0}) for i in range(3)]  # cable not configured yet
    hours += [_hour(3 + i, {HP: 1.0, CABLE: kwh}) for i, kwh in enumerate([0.8, 0, 0, 0, 0.8, 0])]
    live = _run_live(hours[3:], sensors=[HP, CABLE])
    entries = [_entry(h, legacy=True) for h in hours]
    replayed, report = _run_replay(entries, sensors=[HP, CABLE])
    assert replayed["corr"][CABLE] == live["corr"][CABLE]
    assert replayed["counts"][CABLE] == live["counts"][CABLE]
    assert report["legacy_absent_read_as_zero"] == 4
    # The pre-existence hours are not learned as zero.
    assert report["units_skipped_not_reporting"] == 3


# ---------------------------------------------------------------------------
# Targeted replay (retrain_unit_from_history)
# ---------------------------------------------------------------------------


def test_targeted_replay_learns_zero_hours_and_counts_other_units_for_snr():
    hours = [_hour(i, {HP: 2.0, CABLE: 0.0 if i % 2 else 0.6}) for i in range(8)]
    live = _run_live(hours, sensors=[HP, CABLE])
    entries = [_entry(h) for h in hours]
    replayed, report = _run_replay(entries, sensors=[HP, CABLE], target=CABLE)
    assert replayed["corr"][CABLE] == live["corr"][CABLE]
    assert HP not in replayed["corr"]
    assert report["units_learned_base_zero_kwh"] == 4


def test_targeted_reset_first_clears_observation_counts():
    state = _fresh()
    state["corr"][CABLE] = {"5": {"normal": 3.0}}
    state["counts"][CABLE] = {"5": {"normal": 500}}
    entries = [_entry(_hour(i, {CABLE: 0.0})) for i in range(4)]
    model = ModelState(
        correlation_data={},
        correlation_data_per_unit=state["corr"],
        observation_counts=state["counts"],
        aux_coefficients={},
        aux_coefficients_per_unit=state["aux"],
        solar_coefficients_per_unit={},
        learned_u_coefficient=None,
        learning_buffer_per_unit=state["buf"],
        learning_buffer_aux_per_unit=state["aux_buf"],
    )
    LearningManager().replay_per_unit_models(
        entries, {CABLE: DirectMeter(CABLE)}, model, 0.1,
        target_entity=CABLE, reset_first=True,
    )
    assert state["corr"][CABLE] == {"5": {"normal": 0.0}}
    assert state["counts"][CABLE] == {"5": {"normal": 1}}


def test_dry_run_leaves_counts_and_aux_untouched():
    dark = [_hour(i, {HP: 2.0}) for i in range(4)]
    aux = [_hour(4 + i, {HP: 0.8}, aux=True) for i in range(5)]
    state = _fresh()
    model = ModelState(
        correlation_data={},
        correlation_data_per_unit=state["corr"],
        observation_counts=state["counts"],
        aux_coefficients={},
        aux_coefficients_per_unit=state["aux"],
        solar_coefficients_per_unit={},
        learned_u_coefficient=None,
        learning_buffer_per_unit=state["buf"],
        learning_buffer_aux_per_unit=state["aux_buf"],
    )
    report = LearningManager().replay_per_unit_models(
        [_entry(h, aux_affected=[HP]) for h in dark + aux],
        {HP: DirectMeter(HP)}, model, 0.1,
        target_entity=HP, dry_run=True,
        aux_affected_entities={HP}, replay_aux=True,
    )
    assert state == _fresh()
    assert report["buckets_changed"] == 1
    assert report["units_learned_aux"] == 5


# ---------------------------------------------------------------------------
# Which units learn, and which the SNR weight counts
# ---------------------------------------------------------------------------

MPC = "sensor.mpc_heat_pump"


def _track_c_strategies():
    from custom_components.heating_analytics.observation import WeightedSmear

    return {
        HP: DirectMeter(HP),
        CABLE: DirectMeter(CABLE),
        MPC: WeightedSmear(MPC, use_synthetic=True),
    }


def _seeded_state():
    # The MPC unit carries per-unit buckets from before Track C was enabled.
    state = _fresh()
    state["corr"][MPC] = {"5": {"normal": 2.0}}
    return state


def _mpc_shutdown_hours():
    dark = [_hour(i, {HP: 2.0, CABLE: 2.0, MPC: 2.0}) for i in range(4)]
    shut = [_hour(4, {HP: 1.0, CABLE: 1.0, MPC: 0.0}, shutdown=(MPC,))]
    return dark + shut


def test_hourly_learning_sensors_exclude_weighted_smear_under_daily_learning():
    strategies = _track_c_strategies()
    assert hourly_learning_sensors([HP, CABLE, MPC], strategies, True) == [HP, CABLE]
    assert hourly_learning_sensors([HP, CABLE, MPC], strategies, False) == [HP, CABLE, MPC]


def test_track_c_snr_weight_counts_only_the_units_live_learns():
    """Track C forces daily learning, so live learns — and counts in the
    hour's shutdown fraction — the DirectMeter units only.  The sun shuts
    the MPC unit down: live's clean fraction is (2 − 1) / 2, not the
    (3 − 1) / 3 a count over every energy sensor gives."""
    strategies = _track_c_strategies()
    sensors = hourly_learning_sensors([HP, CABLE, MPC], strategies, True)
    hours = _mpc_shutdown_hours()
    live = _run_live(hours, sensors, all_sensors=[HP, CABLE, MPC], state=_seeded_state())
    entries = [_entry(h, sensors=[HP, CABLE, MPC]) for h in hours]
    replayed = _seeded_state()
    model = ModelState(
        correlation_data={},
        correlation_data_per_unit=replayed["corr"],
        observation_counts=replayed["counts"],
        aux_coefficients={},
        aux_coefficients_per_unit=replayed["aux"],
        solar_coefficients_per_unit={},
        learned_u_coefficient=None,
        learning_buffer_per_unit=replayed["buf"],
        learning_buffer_aux_per_unit=replayed["aux_buf"],
    )
    LearningManager().replay_per_unit_models(
        entries, strategies, model, 0.10,
        balance_point=BP, hourly_sensors=sensors,
    )
    assert replayed == live
    assert live["corr"][HP]["5"]["normal"] == pytest.approx(
        2.0 + PER_UNIT_LEARNING_RATE_CAP * 0.5 * (1.0 - 2.0), abs=1e-5
    )
    assert live["corr"][MPC] == {"5": {"normal": 2.0}}


def test_track_c_without_daily_learning_replays_every_sensor():
    """A config entry carrying Track C without daily learning (predating the
    config-flow rule that couples them): live learns every sensor hourly,
    the MPC unit included, and so does the replay."""
    strategies = _track_c_strategies()
    sensors = hourly_learning_sensors([HP, CABLE, MPC], strategies, False)
    hours = _mpc_shutdown_hours()
    live = _run_live(hours, sensors, state=_seeded_state())
    entries = [_entry(h, sensors=sensors) for h in hours]
    replayed = _seeded_state()
    model = ModelState(
        correlation_data={},
        correlation_data_per_unit=replayed["corr"],
        observation_counts=replayed["counts"],
        aux_coefficients={},
        aux_coefficients_per_unit=replayed["aux"],
        solar_coefficients_per_unit={},
        learned_u_coefficient=None,
        learning_buffer_per_unit=replayed["buf"],
        learning_buffer_aux_per_unit=replayed["aux_buf"],
    )
    LearningManager().replay_per_unit_models(
        entries, strategies, model, 0.10,
        balance_point=BP, hourly_sensors=sensors,
    )
    assert replayed == live
    assert MPC in live["counts"]
