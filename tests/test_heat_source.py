"""Heat-source type per unit, inferred from the unit's own history.

A synthetic ``daily_history`` with several units on the same weather, each
with a known heating curve::

    E_unit,hour = b + U · mean_h[max(0, cp − T_h) / (1 + κ·T_h)]

(κ = 0: direct electric; κ ≈ 0.035: air-source heat pump), plus a heat
pump capped on its coldest days, a unit that also cools, and one added
late.  The classifier must name the first two, refuse the capped one,
call the cooling one a reversible heat pump, and not treat the late unit's
earlier days as observations.  Then the state rules (hysteresis,
precedence, provenance) and the wiring (per-unit daily split, storage,
services, entity attributes, sensor rename).
"""
from __future__ import annotations

import math
import random
import sys
from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from homeassistant.util import dt as dt_util

from custom_components.heating_analytics import heat_source as hs
from custom_components.heating_analytics.const import (
    BP_CALIBRATION_MIN_DAYS,
    DOMAIN,
    HEAT_SOURCE_AIR_TO_WATER,
    HEAT_SOURCE_DIRECT_ELECTRIC,
    HEAT_SOURCE_FLAT_COP,
    HEAT_SOURCE_GROUND_SOURCE,
    HEAT_SOURCE_OUTDOOR_DEPENDENT_COP,
    HEAT_SOURCE_REVERSIBLE_HEAT_PUMP,
    HEAT_SOURCE_SWITCH_RUNS,
    HEAT_SOURCE_UNKNOWN,
    MODE_COOLING,
    MODE_DHW,
    MODE_GUEST_HEATING,
    MODE_OFF,
)
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.daily_processor import DailyProcessor
from custom_components.heating_analytics.helpers import unit_day_energy
from custom_components.heating_analytics.learning import (
    regime_energy_split,
    unit_regime_energy,
)
from custom_components.heating_analytics.storage import _heat_source_state_from

AIR_HP = "sensor.air_heat_pump"
PANEL = "sensor.panel_heaters"
CAPPED = "sensor.capped_heat_pump"
AC = "sensor.air_conditioner"
LATE = "sensor.late_cable"
BP = 17.0

# Per unit: heating curve (cp, U, b, κ, relative noise), an optional cap
# on the share of days above it, cooling-mode energy on warm days, and the
# day (counted back from today) the unit's meter first reported.
UNITS = {
    AIR_HP: dict(cp=17.0, u=0.08, b=0.05, kappa=0.035, rel=0.08),
    PANEL: dict(cp=17.0, u=0.05, b=0.03, kappa=0.0, rel=0.08, cool=0.004),
    CAPPED: dict(cp=17.0, u=0.08, b=0.05, kappa=0.035, rel=0.08, cap_share=0.25),
    AC: dict(cp=17.0, u=0.04, b=0.02, kappa=0.035, rel=0.10, cool=0.15),
    LATE: dict(cp=17.0, u=0.03, b=0.02, kappa=0.0, rel=0.08, first=45),
}


def _hourly_temps(mean: float, amp: float = 4.0) -> list[float]:
    return [round(mean + amp * math.sin(2 * math.pi * (h - 11) / 24), 3) for h in range(24)]


def _history(units=UNITS, days: int = 365, *, seed: int = 3, per_unit_split: bool = True) -> dict:
    """``daily_history`` over ``days`` ending yesterday, all units on the
    same weather; ``per_unit_split=False`` stores days the way they were
    aggregated before the per-unit split existed."""
    rng = random.Random(seed)
    end = dt_util.now().date()
    weather = []
    for i in range(days):
        d = end - timedelta(days=days - i)
        season = -math.cos(2 * math.pi * (d.timetuple().tm_yday - 15) / 365)
        weather.append((i, d, 7.0 + 11.0 * season + rng.gauss(0.0, 2.0), rng.uniform(0.0, 0.3)))

    heat: dict[str, list[float]] = {}
    for eid, spec in units.items():
        series = []
        for _, _, mean, _ in weather:
            temps = _hourly_temps(mean)
            demand = sum(
                max(0.0, spec["cp"] - t) / (1.0 + spec["kappa"] * t) for t in temps
            ) / 24
            series.append(max(0.0, (spec["b"] + spec["u"] * demand) * (1.0 + rng.gauss(0.0, spec["rel"]))))
        if spec.get("cap_share"):
            cap = sorted(series)[int((1.0 - spec["cap_share"]) * (len(series) - 1))]
            series = [min(v, cap) for v in series]
        heat[eid] = series

    history = {}
    for i, d, mean, solar in weather:
        temps = _hourly_temps(mean)
        unit_heat, unit_cool = {}, {}
        for eid, spec in units.items():
            if days - i > spec.get("first", days + 1):
                continue  # the meter did not exist yet
            h = 24 * heat[eid][i]
            c = 24 * spec.get("cool", 0.0) * sum(max(0.0, t - 20.0) for t in temps) / 24
            if h > 0:
                unit_heat[eid] = round(h, 3)
            if c > 0:
                unit_cool[eid] = round(c, 3)
        breakdown = {
            eid: round(unit_heat.get(eid, 0.0) + unit_cool.get(eid, 0.0), 3)
            for eid in set(unit_heat) | set(unit_cool)
        }
        entry = {
            "kwh": round(sum(breakdown.values()), 3),
            "temp": round(mean, 1),
            "wind": 2.0,
            "solar_factor": round(solar, 4),
            "aux_impact_kwh": 0.0,
            "guest_impact_kwh": 0.0,
            "regime_heating_kwh": round(sum(unit_heat.values()), 3),
            "regime_cooling_kwh": round(sum(unit_cool.values()), 3),
            "unit_breakdown": breakdown,
            "hourly_vectors": {"temp": temps},
        }
        if per_unit_split:
            entry["unit_heating_kwh"] = unit_heat
            entry["unit_cooling_kwh"] = unit_cool
        history[d.isoformat()] = entry
    return history


def _evidence_coord(units, *, mpc=None):
    coord = MagicMock()
    coord.energy_sensors = list(units)
    coord.mpc_managed_sensor = mpc
    coord.balance_point = BP
    coord.inertia_weights = None
    coord.wind_threshold = 8.0
    return coord


def _evidence(history, units, **kw):
    snapshot = {"daily_history": history, "unit_first_seen": hs.unit_first_seen(history)}
    return hs.unit_evidence(_evidence_coord(units, **kw), snapshot, dt_util.now().date())


@pytest.fixture(scope="module")
def evidence():
    return _evidence(_history(), UNITS)


# ---------------------------------------------------------------------------
# Classification from the evidence
# ---------------------------------------------------------------------------

def test_air_source_heat_pump_is_outdoor_dependent(evidence):
    ev = evidence[AIR_HP]
    assert ev["result"] == HEAT_SOURCE_OUTDOOR_DEPENDENT_COP
    assert min(ev["fit"]["supported_cop_slopes"]) >= 0.02
    assert ev["fit_without_coldest_days"]["curve"] == "outdoor"


def test_direct_electric_heating_is_flat(evidence):
    ev = evidence[PANEL]
    assert ev["result"] == HEAT_SOURCE_FLAT_COP
    assert max(ev["fit"]["supported_cop_slopes"]) <= 0.01


def test_heat_pump_capped_on_its_coldest_days_is_not_called_flat(evidence):
    """The cap flattens the curve where κ is measured; without the coldest
    days the curve bends again, so the unit stays unknown.  Capped on a
    quarter of all days here — the whole winter (30 %, an exact plateau)
    is beyond what dropping the coldest 20 % can catch (see const)."""
    ev = evidence[CAPPED]
    assert ev["result"] == HEAT_SOURCE_UNKNOWN
    assert ev["reason"] in ("curve_changes_without_coldest_days", "cop_slope_not_identified")


def test_unit_that_heats_and_cools_is_a_reversible_heat_pump(evidence):
    ev = evidence[AC]
    assert ev["result"] == HEAT_SOURCE_REVERSIBLE_HEAT_PUMP
    assert ev["reason"] == "heats_and_cools"
    assert ev["cooling_days"] >= 5 and ev["heating_days"] >= 5


def test_small_cooling_mode_energy_is_no_evidence_of_cooling(evidence):
    """The panel logs a little energy in cooling mode on warm days (a mode
    left set); well under the per-day floor, it does not make it cool."""
    assert evidence[PANEL]["cooling_days"] == 0


def test_days_before_a_units_first_report_are_not_observations(evidence):
    ev = evidence[LATE]
    assert ev["fit"]["days"] <= 45
    assert ev["result"] == HEAT_SOURCE_UNKNOWN
    assert ev["reason"] == "insufficient_days"


def test_days_aggregated_before_the_per_unit_split_are_used_when_all_heating():
    units = {AIR_HP: UNITS[AIR_HP]}
    evidence = _evidence(_history(units, per_unit_split=False), units)

    assert evidence[AIR_HP]["result"] == HEAT_SOURCE_OUTDOOR_DEPENDENT_COP
    assert evidence[AIR_HP]["fit"]["days"] > 300


def test_short_history_is_unknown():
    units = {PANEL: dict(UNITS[PANEL], cool=0.0)}
    evidence = _evidence(_history(units, days=BP_CALIBRATION_MIN_DAYS - 5), units)

    assert evidence[PANEL]["result"] == HEAT_SOURCE_UNKNOWN
    assert evidence[PANEL]["reason"] == "insufficient_days"


def test_mpc_managed_sensor_is_not_fitted():
    units = {AIR_HP: UNITS[AIR_HP]}
    evidence = _evidence(_history(units, days=30), units, mpc=AIR_HP)

    assert evidence[AIR_HP] == {"result": None, "reason": "mpc_managed"}


@pytest.mark.parametrize("fit, curve, reason", [
    ({"supported": [0.0, 0.01]}, "flat", "flat_curve"),
    ({"supported": [0.02, 0.03]}, "outdoor", "outdoor_dependent_curve"),
    ({"supported": [0.01, 0.02]}, None, "cop_slope_not_identified"),
    ({"supported": [0.03], "boundary": True}, None, "change_point_at_sweep_boundary"),
    ({"supported": [0.04, 0.05], "limits": ["cop_slope"]}, None, "cop_slope_at_limit"),
    ({"supported": [0.0], "limits": ["solar_shift"]}, "flat", "flat_curve"),
    ({"status": "poor_fit"}, None, "poor_fit"),
])
def test_curve_needs_the_whole_supported_set_on_one_side(monkeypatch, fit, curve, reason):
    monkeypatch.setattr(hs, "variance_prior", lambda days: None)
    monkeypatch.setattr(hs, "sweep", lambda days, **kw: {
        "status": fit.get("status", "fitted"),
        "optimum_at_sweep_boundary": fit.get("boundary", False),
        "limits_hit": fit.get("limits", []),
        "cop_slope_per_c": 0.0,
        "change_point": 17.0,
        "cop_sensitivity": {"supported_cop_slopes": fit.get("supported", [])},
    })

    result = hs._curve_fit([{}] * BP_CALIBRATION_MIN_DAYS)

    assert (result["curve"], result["reason"]) == (curve, reason)


def test_classify_unit_checks_in_order():
    outdoor = {"curve": "outdoor", "reason": "outdoor_dependent_curve"}
    flat = {"curve": "flat", "reason": "flat_curve"}
    undecided = {"curve": None, "reason": "cop_slope_not_identified"}

    assert hs.classify_unit(flat, flat, 5, 5)[0] == HEAT_SOURCE_REVERSIBLE_HEAT_PUMP
    assert hs.classify_unit(flat, flat, 5, 4)[0] == HEAT_SOURCE_FLAT_COP
    assert hs.classify_unit(outdoor, outdoor, 0, 0)[0] == HEAT_SOURCE_OUTDOOR_DEPENDENT_COP
    assert hs.classify_unit(flat, outdoor, 0, 0) == (
        HEAT_SOURCE_UNKNOWN, "curve_changes_without_coldest_days",
    )
    assert hs.classify_unit(undecided, undecided, 0, 0) == (
        HEAT_SOURCE_UNKNOWN, "cop_slope_not_identified",
    )


# ---------------------------------------------------------------------------
# State: hysteresis, precedence, provenance
# ---------------------------------------------------------------------------

def _run(result):
    return {AIR_HP: {"result": result, "reason": "r"}}


def test_first_classification_applies_at_once():
    state = {}
    changes = hs.apply_evidence(state, _run(HEAT_SOURCE_FLAT_COP), "t1")

    assert changes == [(AIR_HP, None, HEAT_SOURCE_FLAT_COP)]
    assert state["units"][AIR_HP]["inferred"] == HEAT_SOURCE_FLAT_COP
    assert state["last_run"] == "t1"


def test_unknown_run_keeps_the_class():
    state = {}
    hs.apply_evidence(state, _run(HEAT_SOURCE_FLAT_COP), "t1")
    changes = hs.apply_evidence(state, _run(HEAT_SOURCE_UNKNOWN), "t2")

    assert changes == []
    assert state["units"][AIR_HP]["inferred"] == HEAT_SOURCE_FLAT_COP
    assert state["units"][AIR_HP]["evidence"]["result"] == HEAT_SOURCE_UNKNOWN


def test_switching_class_needs_consecutive_runs():
    state = {}
    hs.apply_evidence(state, _run(HEAT_SOURCE_FLAT_COP), "t1")
    for i in range(HEAT_SOURCE_SWITCH_RUNS - 1):
        assert hs.apply_evidence(state, _run(HEAT_SOURCE_OUTDOOR_DEPENDENT_COP), f"s{i}") == []
    changes = hs.apply_evidence(state, _run(HEAT_SOURCE_OUTDOOR_DEPENDENT_COP), "t3")

    assert changes == [(AIR_HP, HEAT_SOURCE_FLAT_COP, HEAT_SOURCE_OUTDOOR_DEPENDENT_COP)]
    assert "pending" not in state["units"][AIR_HP]


def test_an_undecided_run_breaks_the_switch_streak():
    state = {}
    hs.apply_evidence(state, _run(HEAT_SOURCE_FLAT_COP), "t1")
    hs.apply_evidence(state, _run(HEAT_SOURCE_OUTDOOR_DEPENDENT_COP), "t2")
    hs.apply_evidence(state, _run(HEAT_SOURCE_UNKNOWN), "t3")
    hs.apply_evidence(state, _run(HEAT_SOURCE_OUTDOOR_DEPENDENT_COP), "t4")

    assert state["units"][AIR_HP]["inferred"] == HEAT_SOURCE_FLAT_COP


def test_mpc_evidence_leaves_the_state_alone():
    state = {}
    hs.apply_evidence(state, {AIR_HP: {"result": None, "reason": "mpc_managed"}}, "t1")

    assert state["units"] == {}


def test_resolution_precedence_user_then_mpc_then_inferred():
    state = {"units": {AIR_HP: {"inferred": HEAT_SOURCE_FLAT_COP}}}

    assert hs.resolve_heat_source(state, AIR_HP) == {
        "type": HEAT_SOURCE_FLAT_COP, "provenance": "inferred", "curve": "flat",
    }
    assert hs.resolve_heat_source(state, AIR_HP, AIR_HP) == {
        "type": HEAT_SOURCE_AIR_TO_WATER, "provenance": "mpc", "curve": "outdoor",
    }
    user = {AIR_HP: HEAT_SOURCE_DIRECT_ELECTRIC}
    assert hs.resolve_heat_source(state, AIR_HP, AIR_HP, user) == {
        "type": HEAT_SOURCE_DIRECT_ELECTRIC, "provenance": "user", "curve": "flat",
    }
    # An inferred class name is not a type a user can set.
    assert hs.resolve_heat_source(state, AIR_HP, None, {AIR_HP: HEAT_SOURCE_OUTDOOR_DEPENDENT_COP})[
        "provenance"
    ] == "inferred"
    assert hs.resolve_heat_source({}, PANEL) == {
        "type": HEAT_SOURCE_UNKNOWN, "provenance": None, "curve": None,
    }


# ---------------------------------------------------------------------------
# Per-unit daily energy
# ---------------------------------------------------------------------------

def test_unit_regime_energy_is_the_per_unit_regime_split():
    modes = {"ac": MODE_COOLING, "boiler": MODE_DHW, "spare": MODE_OFF, "guest": MODE_GUEST_HEATING}
    energy = {"rad": 2.0, "ac": 1.0, "boiler": 3.0, "spare": 0.5, "guest": 0.7, "idle": 0.0}

    per_unit = unit_regime_energy(modes, energy)

    assert per_unit == {"rad": (2.0, 0.0), "ac": (0.0, 1.0), "guest": (0.7, 0.0)}
    assert regime_energy_split(modes, energy) == pytest.approx((2.7, 1.0))


def _log(hour, breakdown, modes=None):
    return {
        "timestamp": f"2026-05-17T{hour:02d}:00:00",
        "hour": hour,
        "temp": 10.0,
        "effective_wind": 3.0,
        "solar_factor": 0.0,
        "actual_kwh": sum(breakdown.values()),
        "tdd": 0.5,
        "unit_breakdown": breakdown,
        "unit_modes": dict(modes or {}),
    }


def _processor():
    coord = MagicMock()
    coord.solar_enabled = False
    coord.balance_point = 15.0
    coord.hourly_solar_impact_kwh = MagicMock(return_value=0.0)
    return DailyProcessor(coord)


def test_aggregate_logs_splits_each_unit_per_hour():
    logs = [_log(h, {"hp": 2.0, "cable": 0.5}) for h in range(12)]
    logs += [_log(h, {"hp": 1.0, "boiler": 1.0}, {"hp": MODE_COOLING, "boiler": MODE_DHW}) for h in range(12, 24)]

    agg = _processor().aggregate_logs(logs)

    assert agg["unit_heating_kwh"] == {"hp": 24.0, "cable": 6.0}
    assert agg["unit_cooling_kwh"] == {"hp": 12.0}
    assert agg["regime_heating_kwh"] == pytest.approx(30.0)
    assert agg["regime_cooling_kwh"] == pytest.approx(12.0)


def test_aggregate_logs_without_unit_breakdown_writes_no_per_unit_split():
    logs = [_log(h, {"hp": 1.0}) for h in range(24)]
    for entry in logs:
        del entry["unit_breakdown"]

    agg = _processor().aggregate_logs(logs)

    for key in ("regime_heating_kwh", "unit_heating_kwh", "unit_cooling_kwh"):
        assert key not in agg


def test_backfill_drops_a_stored_per_unit_split_the_logs_cannot_support():
    proc = _processor()
    logs = [_log(h, {"hp": 1.0}) for h in range(24)]
    for entry in logs:
        del entry["unit_breakdown"]
    proc.coordinator._hourly_log = logs
    proc.coordinator._daily_history = {"2026-05-17": {
        "kwh": 24.0, "regime_heating_kwh": 24.0, "regime_cooling_kwh": 0.0,
        "unit_heating_kwh": {"hp": 24.0}, "unit_cooling_kwh": {},
    }}

    proc.backfill_from_hourly()

    day = proc.coordinator._daily_history["2026-05-17"]
    for key in ("regime_heating_kwh", "regime_cooling_kwh", "unit_heating_kwh", "unit_cooling_kwh"):
        assert key not in day


def test_unit_day_energy_reads_the_split_and_all_heating_legacy_days():
    split = {"kwh": 5.0, "unit_breakdown": {"hp": 5.0},
             "regime_heating_kwh": 3.0, "regime_cooling_kwh": 2.0,
             "unit_heating_kwh": {"hp": 3.0}, "unit_cooling_kwh": {"hp": 2.0}}
    legacy = {"kwh": 5.0, "unit_breakdown": {"hp": 4.0, "cable": 1.0},
              "regime_heating_kwh": 5.0, "regime_cooling_kwh": 0.0}
    legacy_dhw = dict(legacy, regime_heating_kwh=3.0)
    legacy_cooling = dict(legacy, regime_heating_kwh=4.0, regime_cooling_kwh=1.0)
    csv_import = {"kwh": 5.0, "unit_breakdown": {}, "regime_heating_kwh": 0.0,
                  "regime_cooling_kwh": 0.0}

    assert unit_day_energy(split, "hp") == (3.0, 2.0)
    assert unit_day_energy(split, "other") == (0.0, 0.0)
    assert unit_day_energy(legacy, "cable") == (1.0, 0.0)
    assert unit_day_energy(legacy_dhw, "hp") is None
    assert unit_day_energy(legacy_cooling, "hp") is None
    assert unit_day_energy(csv_import, "hp") is None
    assert unit_day_energy({"kwh": 1.0, "unit_breakdown": {"hp": 1.0}}, "hp") is None


def test_unit_first_seen_is_the_first_day_with_energy():
    history = {
        "2026-01-03": {"unit_breakdown": {"a": 1.0, "b": 0.0}},
        "2026-01-01": {"unit_breakdown": {"a": 0.0}},
        "2026-01-05": {"unit_breakdown": {"b": 2.0}},
    }
    assert hs.unit_first_seen(history) == {"a": "2026-01-03", "b": "2026-01-05"}


# ---------------------------------------------------------------------------
# Coordinator wiring
# ---------------------------------------------------------------------------

class _Hass:
    def __init__(self):
        self.states = MagicMock()
        self.states.get = MagicMock(return_value=None)
        self.data = {DOMAIN: {}}
        self.config_entries = MagicMock()
        self.bus = MagicMock()
        self.is_running = True

    async def async_add_executor_job(self, fn, *args):
        return fn(*args)


def _coordinator(mpc=None, heat_source_types=None) -> HeatingDataCoordinator:
    entry = MagicMock()
    entry.entry_id = "entry"
    entry.data = {
        "energy_sensors": [AIR_HP, PANEL],
        "outdoor_temp_sensor": "sensor.outdoor",
        "balance_point": BP,
    }
    if heat_source_types is not None:
        entry.data["heat_source_types"] = heat_source_types
    entry.options = {}
    coord = HeatingDataCoordinator(_Hass(), entry)
    coord.mpc_managed_sensor = mpc
    coord.storage = MagicMock()
    coord.storage.async_save_data = AsyncMock()
    coord.async_set_updated_data = MagicMock()
    return coord


@pytest.fixture
def notifications():
    notify = MagicMock()
    sys.modules["homeassistant.components"].persistent_notification.async_create = notify
    return notify


def _canned(results):
    def unit_evidence(coordinator, snapshot, today):
        return {eid: {"result": r, "reason": "canned", "fit": {"days": 300}} for eid, r in results.items()}
    return unit_evidence


async def test_classification_run_stores_classes_and_notifies_once(monkeypatch, notifications):
    coord = _coordinator()
    monkeypatch.setattr(hs, "unit_evidence", _canned({
        AIR_HP: HEAT_SOURCE_OUTDOOR_DEPENDENT_COP, PANEL: HEAT_SOURCE_UNKNOWN,
    }))

    report = await coord.async_classify_heat_sources()

    assert report["changes"] == [
        {"entity_id": AIR_HP, "from": None, "to": HEAT_SOURCE_OUTDOOR_DEPENDENT_COP},
    ]
    assert coord.heat_source_for(AIR_HP)["type"] == HEAT_SOURCE_OUTDOOR_DEPENDENT_COP
    assert coord.heat_source_for(AIR_HP)["provenance"] == "inferred"
    assert coord.heat_source_for(PANEL)["type"] == HEAT_SOURCE_UNKNOWN
    assert coord._heat_source_state["last_run"] is not None
    coord.storage.async_save_data.assert_awaited()
    assert notifications.call_count == 1
    assert AIR_HP in notifications.call_args[0][1]

    await coord.async_classify_heat_sources()
    assert notifications.call_count == 1


async def test_user_type_wins_and_auto_clears_it(monkeypatch, notifications):
    coord = _coordinator()
    monkeypatch.setattr(hs, "unit_evidence", _canned({AIR_HP: HEAT_SOURCE_FLAT_COP}))
    await coord.async_classify_heat_sources()

    result = await coord.async_set_heat_source_type(AIR_HP, HEAT_SOURCE_AIR_TO_WATER)
    assert (result["type"], result["provenance"]) == (HEAT_SOURCE_AIR_TO_WATER, "user")
    assert result["inferred"] == HEAT_SOURCE_FLAT_COP
    # Written to the config entry, where the config flow keeps it too.
    _, kwargs = coord.hass.config_entries.async_update_entry.call_args
    assert kwargs["data"]["heat_source_types"] == {AIR_HP: HEAT_SOURCE_AIR_TO_WATER}
    assert kwargs["data"]["energy_sensors"] == [AIR_HP, PANEL]

    result = await coord.async_set_heat_source_type(AIR_HP, "auto")
    assert (result["type"], result["provenance"]) == (HEAT_SOURCE_FLAT_COP, "inferred")
    _, kwargs = coord.hass.config_entries.async_update_entry.call_args
    assert kwargs["data"]["heat_source_types"] == {}

    with pytest.raises(ValueError):
        await coord.async_set_heat_source_type(AIR_HP, "coal")
    with pytest.raises(ValueError):
        # The inferred class names a curve, not a device.
        await coord.async_set_heat_source_type(AIR_HP, HEAT_SOURCE_FLAT_COP)
    with pytest.raises(ValueError):
        await coord.async_set_heat_source_type("sensor.unknown", HEAT_SOURCE_DIRECT_ELECTRIC)


def test_user_types_are_read_from_the_config_entry():
    coord = _coordinator(heat_source_types={
        AIR_HP: HEAT_SOURCE_GROUND_SOURCE,
        PANEL: HEAT_SOURCE_FLAT_COP,  # not a type a user sets: ignored
        "sensor.removed": HEAT_SOURCE_DIRECT_ELECTRIC,  # no longer configured
    })

    assert coord.heat_source_user_types == {AIR_HP: HEAT_SOURCE_GROUND_SOURCE}
    assert coord.heat_source_for(AIR_HP)["provenance"] == "user"
    assert coord.heat_source_for(PANEL)["provenance"] is None


async def test_units_with_a_user_type_are_not_notified(monkeypatch, notifications):
    coord = _coordinator()
    await coord.async_set_heat_source_type(AIR_HP, HEAT_SOURCE_DIRECT_ELECTRIC)
    monkeypatch.setattr(hs, "unit_evidence", _canned({AIR_HP: HEAT_SOURCE_FLAT_COP}))

    await coord.async_classify_heat_sources()

    notifications.assert_not_called()


async def test_mpc_managed_sensor_resolves_as_heat_pump():
    coord = _coordinator(mpc=AIR_HP)

    assert coord.heat_source_for(AIR_HP)["type"] == HEAT_SOURCE_AIR_TO_WATER
    assert coord.heat_source_for(AIR_HP)["provenance"] == "mpc"


async def test_background_run_is_weekly(monkeypatch):
    coord = _coordinator()
    run = AsyncMock()
    monkeypatch.setattr(coord, "async_classify_heat_sources", run)

    await coord.async_maybe_classify_heat_sources()
    assert run.await_count == 1

    coord._heat_source_state["last_run"] = (dt_util.now() - timedelta(days=3)).isoformat()
    await coord.async_maybe_classify_heat_sources()
    assert run.await_count == 1

    coord._heat_source_state["last_run"] = (dt_util.now() - timedelta(days=8)).isoformat()
    await coord.async_maybe_classify_heat_sources()
    assert run.await_count == 2


async def test_failed_background_run_does_not_raise(monkeypatch):
    coord = _coordinator()
    monkeypatch.setattr(coord, "async_classify_heat_sources", AsyncMock(side_effect=RuntimeError("x")))

    await coord.async_maybe_classify_heat_sources()


def test_background_run_is_not_scheduled_before_startup():
    coord = _coordinator()
    coord.hass.is_running = False
    coord.hass.async_create_background_task = MagicMock()

    coord._schedule_heat_source_classification()

    coord.hass.async_create_background_task.assert_not_called()


def test_background_run_is_scheduled_after_midnight(recwarn):
    coord = _coordinator()
    scheduled = []
    coord.hass.async_create_background_task = lambda coro, name: scheduled.append(coro)

    coord._schedule_heat_source_classification()

    assert len(scheduled) == 1
    scheduled[0].close()


def test_scheduling_without_background_tasks_closes_the_run(recwarn):
    coord = _coordinator()  # _Hass has no async_create_background_task

    coord._schedule_heat_source_classification()

    assert not [w for w in recwarn if "never awaited" in str(w.message)]


def test_mode_entity_shows_the_heat_source():
    from custom_components.heating_analytics.select import HeatingAnalyticsModeSelect

    coord = _coordinator()
    coord._heat_source_state = {"units": {AIR_HP: {
        "inferred": HEAT_SOURCE_FLAT_COP, "classified_at": "t", "evidence": {"days": 300},
    }}}
    entity = HeatingAnalyticsModeSelect(coord, AIR_HP)

    attrs = entity.extra_state_attributes
    assert attrs["heat_source"] == HEAT_SOURCE_FLAT_COP
    assert attrs["heat_source_provenance"] == "inferred"
    assert attrs["heat_source_evidence"] == {"days": 300}
    assert "heat_source_evidence" in entity._unrecorded_attributes


# ---------------------------------------------------------------------------
# Storage and rename
# ---------------------------------------------------------------------------

def test_stored_state_is_normalised():
    assert _heat_source_state_from(None) == {"last_run": None, "units": {}}
    assert _heat_source_state_from([1]) == {"last_run": None, "units": {}}
    assert _heat_source_state_from({"last_run": 5, "units": {"a": {"inferred": "flat_cop"}, "b": 3}}) == {
        "last_run": None, "units": {"a": {"inferred": "flat_cop"}},
    }


async def test_state_survives_save_and_load(mock_coordinator):
    from custom_components.heating_analytics.storage import StorageManager

    mock_coordinator.energy_sensors = [AIR_HP]
    state = {"last_run": "2026-09-01T00:00:00", "units": {
        AIR_HP: {"inferred": HEAT_SOURCE_FLAT_COP, "pending": HEAT_SOURCE_OUTDOOR_DEPENDENT_COP, "pending_runs": 1},
    }}
    mock_coordinator._heat_source_state = state
    storage = StorageManager(mock_coordinator)
    storage._store = AsyncMock()

    await storage.async_save_data(force=True)
    saved = storage._store.async_save.call_args[0][0]
    assert saved["heat_source"] == state

    mock_coordinator._heat_source_state = {}
    storage._store.async_load.return_value = {
        "heat_source": state,
        "daily_history": {},
        "correlation_data": {},
        "last_updated": dt_util.now().isoformat(),
    }
    await storage.async_load_data()
    assert mock_coordinator._heat_source_state == state


async def test_replacing_a_sensor_moves_its_heat_source_and_daily_history(hass):
    from tests.test_units_reporting_log import _coordinator as reporting_coordinator

    coordinator = reporting_coordinator(hass)
    old, new = "sensor.heater", "sensor.heater_v2"
    coordinator.entry.data["heat_source_types"] = {old: HEAT_SOURCE_DIRECT_ELECTRIC}
    coordinator.heat_source_user_types = {old: HEAT_SOURCE_DIRECT_ELECTRIC}
    coordinator._heat_source_state = {"last_run": None, "units": {old: {"inferred": HEAT_SOURCE_FLAT_COP}}}
    coordinator._daily_history = {"2023-10-26": {
        "unit_breakdown": {old: 5.0},
        "unit_expected_breakdown": {old: 4.0},
        "unit_heating_kwh": {old: 5.0},
        "unit_cooling_kwh": {},
    }}

    assert await coordinator.async_replace_sensor_source(old, new) is True

    assert coordinator._heat_source_state["units"] == {new: {"inferred": HEAT_SOURCE_FLAT_COP}}
    assert coordinator.heat_source_user_types == {new: HEAT_SOURCE_DIRECT_ELECTRIC}
    _, kwargs = hass.config_entries.async_update_entry.call_args_list[0]
    assert kwargs["data"]["heat_source_types"] == {new: HEAT_SOURCE_DIRECT_ELECTRIC}
    day = coordinator._daily_history["2023-10-26"]
    assert day["unit_breakdown"] == {new: 5.0}
    assert day["unit_expected_breakdown"] == {new: 4.0}
    assert day["unit_heating_kwh"] == {new: 5.0}


def test_calibrate_balance_point_reports_each_units_heat_source():
    from custom_components.heating_analytics.balance_point import calibrate_balance_point

    coord = _evidence_coord([AIR_HP])
    coord.model.daily_history = {}
    coord.model.hourly_log = []
    coord.model.correlation_data = {}
    coord.heat_source_for = lambda eid: {"type": HEAT_SOURCE_FLAT_COP, "provenance": "user"}

    report = calibrate_balance_point(coord)

    assert report["unit_heat_source"] == {AIR_HP: {"type": HEAT_SOURCE_FLAT_COP, "provenance": "user"}}
