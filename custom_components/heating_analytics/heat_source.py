"""Heat-source type per unit, inferred from the unit's own history (#1110).

The integration models electrical kWh, and whether a unit is a heat pump
(COP above 1, rising with the outdoor temperature) or a direct electric
load (COP 1) changes the shape of its heating curve.  This module infers
what the data can show and applies it without asking; the user can always
override, in the config flow or with ``set_heat_source_type`` — both write
the config entry's ``CONF_HEAT_SOURCE_TYPES``.

Classes are named after what is observable (``const.HEAT_SOURCE_*``):

* ``reversible_heat_pump`` — the unit used real energy in heating mode on
  cold days and in cooling mode on warm days.  A direct electric load
  cannot cool, so this is strong evidence; it wins over the curve.
* ``outdoor_dependent_cop`` / ``flat_cop`` — the relative COP slope κ of the
  unit's own heating energy against degree-hours, fitted with
  ``calibrate_balance_point``'s change-point model on the unit's days.
  ``flat_cop`` is direct electric *or* ground source: a constant COP only
  rescales U, so energy data cannot separate them.
* ``unknown`` — not enough evidence.  The default, and what every consumer
  treats as "no type", i.e. today's behaviour.

A class is inferred only when every check passes, so thin, noisy or
ambiguous data falls to ``unknown`` rather than to a wrong class:

1. the whole supported κ set (the profile-likelihood table of
   ``calibrate_balance_point``) lies on one side —
   ``≤ HEAT_SOURCE_FLAT_MAX_COP_SLOPE`` or ``≥ HEAT_SOURCE_OUTDOOR_MIN_COP_SLOPE``;
2. the change point is not on the sweep edge and κ is not on its limit;
3. the curve is the same without the coldest ``HEAT_SOURCE_COLD_TRIM_SHARE``
   of the unit's days — a heat pump at its capacity limit on the coldest
   days fits as flat otherwise.

The MPC-managed sensor is a heat pump by construction (the MPC integration
publishes its COP model) and is not fitted.  A meter shared by several
loads gets the class of the load that dominates it; that is accepted.

**Provenance decides what a consumer may use.**  The curve shape
(``heat_source_curve``) may come from an inferred type.  The COP *level*
(solar coefficients, Track B / C COP smearing) cannot be inferred from
energy data at all, so a consumer that needs it must use a type whose
provenance is ``user`` or ``mpc`` only.
"""
from __future__ import annotations

from datetime import date, timedelta

from .balance_point import collect_days, model_snapshot, sweep, variance_prior
from .const import (
    BP_CALIBRATION_LOOKBACK_DAYS,
    BP_CALIBRATION_MIN_DAYS,
    HEAT_SOURCE_AIR_TO_WATER,
    HEAT_SOURCE_COLD_TRIM_SHARE,
    HEAT_SOURCE_CURVE,
    HEAT_SOURCE_FLAT_COP,
    HEAT_SOURCE_FLAT_MAX_COP_SLOPE,
    HEAT_SOURCE_OUTDOOR_DEPENDENT_COP,
    HEAT_SOURCE_OUTDOOR_MIN_COP_SLOPE,
    HEAT_SOURCE_REVERSIBLE_HEAT_PUMP,
    HEAT_SOURCE_REVERSIBLE_MIN_DAYS,
    HEAT_SOURCE_REVERSIBLE_MIN_KWH,
    HEAT_SOURCE_SWITCH_RUNS,
    HEAT_SOURCE_TYPES,
    HEAT_SOURCE_UNKNOWN,
    HEAT_SOURCE_USER_TYPES,
)
from .helpers import finite_float, unit_day_energy

_MPC_MANAGED = "mpc_managed"


def unit_first_seen(daily_history: dict) -> dict[str, str]:
    """Each unit's first ``daily_history`` day with energy (``YYYY-MM-DD``).

    Days before it are not observations of the unit — a sensor added after
    setup would otherwise read as idle for every earlier day.
    """
    first: dict[str, str] = {}
    for key in sorted(k for k in (daily_history or {}) if isinstance(k, str)):
        entry = daily_history[key]
        if not isinstance(entry, dict):
            continue
        for field in ("unit_breakdown", "unit_heating_kwh", "unit_cooling_kwh"):
            values = entry.get(field)
            if not isinstance(values, dict):
                continue
            for eid, value in values.items():
                if eid not in first and (finite_float(value) or 0.0) > 0.0:
                    first[eid] = key[:10]
    return first


def heat_source_snapshot(coordinator) -> dict:
    """Copies of what the classification reads, taken on the event loop.

    ``calibrate_balance_point``'s snapshot (the window's ``daily_history``,
    deep-copied) plus each unit's first day with energy over the whole
    history, which the window alone cannot show.
    """
    snapshot = model_snapshot(coordinator)
    snapshot["unit_first_seen"] = unit_first_seen(coordinator.model.daily_history)
    return snapshot


def _curve_fit(days: list) -> dict:
    """The unit's κ fit and the curve it supports (``None``: undecided)."""
    n = len(days)
    if n < BP_CALIBRATION_MIN_DAYS:
        return {"status": "insufficient_days", "days": n, "curve": None,
                "reason": "insufficient_days"}
    prior = variance_prior(days)
    fit = sweep(days, heating=True, cop_sensitivity=True, prior=prior)
    supported = (fit.get("cop_sensitivity") or {}).get("supported_cop_slopes") or []
    curve = None
    if fit["status"] != "fitted":
        reason = fit["status"]
    elif fit.get("optimum_at_sweep_boundary"):
        reason = "change_point_at_sweep_boundary"
    elif "cop_slope" in (fit.get("limits_hit") or []):
        reason = "cop_slope_at_limit"
    elif supported and max(supported) <= HEAT_SOURCE_FLAT_MAX_COP_SLOPE + 1e-9:
        curve, reason = "flat", "flat_curve"
    elif supported and min(supported) >= HEAT_SOURCE_OUTDOOR_MIN_COP_SLOPE - 1e-9:
        curve, reason = "outdoor", "outdoor_dependent_curve"
    else:
        reason = "cop_slope_not_identified"
    return {
        "status": fit["status"],
        "days": n,
        "change_point": fit.get("change_point"),
        "cop_slope": fit.get("cop_slope_per_c"),
        "supported_cop_slopes": supported,
        "curve": curve,
        "reason": reason,
    }


def _regime_days(daily_history: dict, entity_id: str, balance_point: float,
                 first_day: str, today: date) -> tuple[int, int]:
    """Days with real heating-mode energy below the balance point, and with
    real cooling-mode energy above it."""
    end = today.isoformat()
    heating = cooling = 0
    for key, entry in (daily_history or {}).items():
        if not isinstance(key, str) or not (first_day <= key[:10] < end):
            continue
        temp = finite_float(entry.get("temp")) if isinstance(entry, dict) else None
        split = unit_day_energy(entry, entity_id) if temp is not None else None
        if split is None:
            continue
        if split[0] >= HEAT_SOURCE_REVERSIBLE_MIN_KWH and temp < balance_point:
            heating += 1
        if split[1] >= HEAT_SOURCE_REVERSIBLE_MIN_KWH and temp > balance_point:
            cooling += 1
    return heating, cooling


def classify_unit(full: dict, trimmed: dict, heating_days: int,
                  cooling_days: int) -> tuple[str, str]:
    """``(class, reason)`` from one unit's evidence."""
    if (
        heating_days >= HEAT_SOURCE_REVERSIBLE_MIN_DAYS
        and cooling_days >= HEAT_SOURCE_REVERSIBLE_MIN_DAYS
    ):
        return HEAT_SOURCE_REVERSIBLE_HEAT_PUMP, "heats_and_cools"
    if full["curve"] is None:
        return HEAT_SOURCE_UNKNOWN, full["reason"]
    if trimmed["curve"] != full["curve"]:
        return HEAT_SOURCE_UNKNOWN, "curve_changes_without_coldest_days"
    if full["curve"] == "outdoor":
        return HEAT_SOURCE_OUTDOOR_DEPENDENT_COP, full["reason"]
    return HEAT_SOURCE_FLAT_COP, full["reason"]


def unit_evidence(coordinator, snapshot: dict, today: date) -> dict:
    """Each energy sensor's evidence and the class it supports.

    Pure read of ``snapshot`` and of the coordinator's settings, so it can
    run in an executor.  Each fitted unit costs two change-point sweeps
    (all days, and without the coldest).
    """
    sensors = [s for s in (getattr(coordinator, "energy_sensors", None) or []) if isinstance(s, str)]
    mpc = getattr(coordinator, "mpc_managed_sensor", None)
    fitted = [s for s in sensors if s != mpc]
    first_seen = snapshot.get("unit_first_seen") or {}
    daily = snapshot["daily_history"]
    days = collect_days(
        coordinator, daily, today, units=fitted, unit_first_seen=first_seen,
    )["days"]
    balance_point = float(coordinator.balance_point)
    window_start = (today - timedelta(days=BP_CALIBRATION_LOOKBACK_DAYS)).isoformat()

    out: dict[str, dict] = {}
    for eid in sensors:
        if eid == mpc:
            out[eid] = {"result": None, "reason": _MPC_MANAGED}
            continue
        unit_days = [
            dict(d, y_heat=d["units_heat"][eid]) for d in days if eid in d["units_heat"]
        ]
        full = _curve_fit(unit_days)
        if full["curve"] is not None:
            cut = sorted(d["mean_t"] for d in unit_days)[
                int(HEAT_SOURCE_COLD_TRIM_SHARE * len(unit_days))
            ]
            trimmed = _curve_fit([d for d in unit_days if d["mean_t"] >= cut])
        else:
            # Nothing to confirm; skip the second sweep.
            trimmed = {"curve": None, "reason": "not_evaluated"}
        heating_days, cooling_days = _regime_days(
            daily, eid, balance_point,
            max(window_start, first_seen.get(eid, window_start)), today,
        )
        result, reason = classify_unit(full, trimmed, heating_days, cooling_days)
        out[eid] = {
            "result": result,
            "reason": reason,
            "fit": full,
            "fit_without_coldest_days": trimmed,
            "heating_days": heating_days,
            "cooling_days": cooling_days,
        }
    return out


def compact_evidence(evidence: dict) -> dict:
    """The part of a unit's evidence kept in storage and shown on entities."""
    full = evidence.get("fit") or {}
    trimmed = evidence.get("fit_without_coldest_days") or {}
    return {
        "result": evidence.get("result"),
        "reason": evidence.get("reason"),
        "days": full.get("days"),
        "cop_slope": full.get("cop_slope"),
        "supported_cop_slopes": full.get("supported_cop_slopes"),
        "supported_cop_slopes_without_coldest_days": trimmed.get("supported_cop_slopes"),
        "change_point": full.get("change_point"),
        "heating_days": evidence.get("heating_days"),
        "cooling_days": evidence.get("cooling_days"),
    }


def apply_evidence(state: dict, evidence: dict, now_iso: str) -> list[tuple[str, str | None, str]]:
    """Update the stored state from one classification run, in place.

    Hysteresis: a unit without a class takes the run's class at once (the
    checks above are the entry bar); a class is replaced only by another
    class found in ``HEAT_SOURCE_SWITCH_RUNS`` consecutive runs; a run that
    cannot classify keeps the class and breaks the streak.  Returns the
    ``(unit, old, new)`` class changes.
    """
    units = state.setdefault("units", {})
    changes: list[tuple[str, str | None, str]] = []
    for eid, ev in evidence.items():
        if ev.get("reason") == _MPC_MANAGED:
            continue
        unit = units.setdefault(eid, {})
        unit["evidence"] = compact_evidence(ev)
        unit["evaluated_at"] = now_iso
        result = ev.get("result")
        inferred = unit.get("inferred")
        if result in (None, HEAT_SOURCE_UNKNOWN) or result == inferred:
            unit.pop("pending", None)
            unit.pop("pending_runs", None)
            continue
        if inferred is None:
            unit["inferred"] = result
            unit["classified_at"] = now_iso
            changes.append((eid, None, result))
            continue
        runs = unit.get("pending_runs", 0) + 1 if unit.get("pending") == result else 1
        if runs >= HEAT_SOURCE_SWITCH_RUNS:
            unit["inferred"] = result
            unit["classified_at"] = now_iso
            unit.pop("pending", None)
            unit.pop("pending_runs", None)
            changes.append((eid, inferred, result))
        else:
            unit["pending"] = result
            unit["pending_runs"] = runs
    state["last_run"] = now_iso
    return changes


def user_heat_source_types(raw, energy_sensors) -> dict[str, str]:
    """The config entry's user types, kept only for configured sensors and
    settable types."""
    if not isinstance(raw, dict):
        return {}
    sensors = set(energy_sensors or [])
    return {
        eid: heat_source_type
        for eid, heat_source_type in raw.items()
        if eid in sensors and heat_source_type in HEAT_SOURCE_USER_TYPES
    }


def resolve_heat_source(state: dict, entity_id: str, mpc_sensor: str | None = None,
                        user_types: dict | None = None) -> dict:
    """The type a unit is treated as, and where it came from.

    The user's type (``user_types``, from the config entry) wins; then the
    MPC-managed sensor, a heat pump by construction; then the inferred
    class; else ``unknown`` (provenance ``None``).
    """
    unit = ((state or {}).get("units") or {}).get(entity_id) or {}
    user = (user_types or {}).get(entity_id)
    if user in HEAT_SOURCE_USER_TYPES:
        heat_source, provenance = user, "user"
    elif mpc_sensor is not None and entity_id == mpc_sensor:
        heat_source, provenance = HEAT_SOURCE_AIR_TO_WATER, "mpc"
    elif unit.get("inferred") in HEAT_SOURCE_TYPES:
        heat_source, provenance = unit["inferred"], "inferred"
    else:
        heat_source, provenance = HEAT_SOURCE_UNKNOWN, None
    return {
        "type": heat_source,
        "provenance": provenance,
        "curve": heat_source_curve(heat_source),
    }


def heat_source_curve(heat_source: str | None) -> str | None:
    """``"outdoor"``, ``"flat"`` or ``None`` (unknown) for a type."""
    return HEAT_SOURCE_CURVE.get(heat_source)
