"""Balance-point calibration from daily history (#1045).

Fits a change-point model of daily heating energy against degree-hours and
reports a suggested balance point with stability evidence.  **Suggestion
only** — nothing here writes configuration.  The balance point stays a
user-owned config-entry value, changed in the reconfigure flow; same
"informed decision, not automatic convergence" stance as the
obstruction-gate suggestions (#1020).

Model (ASHRAE RP-1050 three-parameter heating, "3PH", on degree-hours,
with a solar shift of the balance point and heat-pump efficiency)::

    E_d = b + U · mean_h[ max(0, base_d − T_h) / (1 + κ · T_h) ]
    base_d = cp − Δ · S_eff,d / S_ref
    S_eff,d = (1 − c) · S_d + c · S_(d−1)

per hour of day ``d``, where ``T_h`` are the day's hourly inertia
temperatures, ``S_d`` the day's mean ``solar_factor``, ``c`` the share of
yesterday's sun still acting today (heat stored in the building's mass),
``S_ref`` the window's clear-day level and ``1 + κ·T`` the heat pump's
relative COP.  ``(cp, Δ, c, κ)`` are swept; ``Δ = 0``, ``c = 0`` and
``κ = 0`` nest the simpler models (``κ = 0`` is direct electric heating).

``cp`` is the balance point of a day **without sun**.  That is the quantity
suggested, because it is what the rest of the system reads the balance
point *as*: the cold/mild regime boundary (``|BP − T| > 4``), the BP-2
cold-cooling shield and Track C's heating/cooling split — solar is modelled
separately there, never folded into the balance point.

Why each term exists (all quantified in simulation, see ``const.py``):

* **Daily, not hourly.**  An hourly fit had to filter its way to clean
  hours, and each filter carried its own bias — the solar filter read the
  saturated one-hour solar battery, so shoulder-season sun and heat
  released from thermal mass passed it.  A day integrates most of that,
  and ``daily_history`` is never trimmed, so a heating season is always in
  reach.
* **Solar shift and carry-over.**  A sunny day, and the day after, needs no
  heat well below the balance point.  A fit without these terms reads that
  as a lower balance point (≈ 1 °C at 30 % carry-over).
* **COP slope.**  Energy is heat demand / COP and COP rises with outdoor
  temperature, so the arm is convex; a straight arm crosses the floor
  several °C too early on an air-source heat pump.

Inputs, all from ``daily_history``:

* **y** = ``regime_heating_kwh`` — heating-mode energy accumulated per hour
  against that hour's own modes (#1051): DHW, OFF and cooling energy are
  already excluded.  Days without the split (recorded before it existed,
  or during downtime) and days whose per-unit breakdown does not account
  for their energy (imported from CSV) are skipped, not guessed.
* **x** = hourly inertia temperature reconstructed from
  ``hourly_vectors.temp`` exactly as live learning computes it: the
  coordinator's whole kernel, each reading weighted by its age, history
  cut at a gap longer than τ (see ``_inertia_kernel``), across midnight.
  Without a usable kernel the raw hourly temperature is used.  Only
  complete days are used: after downtime the missed hours' meter delta
  lands in the next logged hour.
* **S** = the day's mean *effective* ``solar_factor`` — after screens, so
  closed screens lower it, which is the sun that actually enters.  It does
  not depend on learned coefficients.
* Days with aux or guest energy, or more than
  ``BP_CALIBRATION_MAX_HIGH_WIND_HOURS`` high-wind hours, are skipped and
  counted by reason.  Aux is read from the aux model's logged reduction;
  aux running before that model has learned its temperature is invisible
  in daily records.

A unit sitting in cooling or DHW mode does not heat its room, and daily
records carry no modes to detect that (#1089).  Days no record can label —
a whole-home shutdown, a party, a metering glitch — are caught by one
robust pass that drops residual outliers, per arm, and refits.

Heating energy scatters more on cold days than on the flat arm, and the
COP-slope signal comes from those days, so the fit is feasible GLS: an
ordinary-LS pass models the variance (``σ² = s0² + s1²·ŷ²``) and finds the
outliers, and the reported fit is weighted by ``1/σ²`` — every threshold
below is on that variance-equalised scale.

Stability is checked on interleaved halves (even vs odd ISO weeks): both
cover the same seasons with little shared data, unlike nested windows.
It is also checked across the COP slope: cp and κ trade off (a steeper
curve moves the change point up), so the fit is repeated at fixed slopes
and cp must agree across every slope the data supports
(``cop_sensitivity``).  A shape term resting on its limit (κ, the solar
shift, or the carry-over when the shift is material) withholds the
suggestion as ``nuisance_at_limit``: cp is then conditional on a model
truncated at that limit.  More than ``BP_CALIBRATION_MAX_OUTLIER_SHARE``
of days dropped as outliers is ``poor_fit``.

``cp + b/U`` — where consumption proportional to ``|BP − T|`` would reach
zero — is reported beside ``cp``, never suggested.

The fit does not depend on the configured balance point, so history
spanning a BP change needs no special handling.
"""
from __future__ import annotations

import bisect
import copy
import math
from datetime import date, datetime, time, timedelta, timezone, tzinfo

from homeassistant.util import dt as dt_util

from .const import (
    BP_CALIBRATION_CARRYOVER_MIN_SHIFT_C,
    BP_CALIBRATION_COP_FLOOR,
    BP_CALIBRATION_COP_SLOPE_COARSE_STEP,
    BP_CALIBRATION_COP_SLOPE_MAX,
    BP_CALIBRATION_COP_SLOPE_STEP,
    BP_CALIBRATION_COP_TABLE_STEP,
    BP_CALIBRATION_LOOKBACK_DAYS,
    BP_CALIBRATION_MAX_HIGH_WIND_HOURS,
    BP_CALIBRATION_MAX_OUTLIER_SHARE,
    BP_CALIBRATION_MAX_WEIGHT_RATIO,
    BP_CALIBRATION_MIN_DAYS,
    BP_CALIBRATION_MIN_DAYS_PER_SIDE,
    BP_CALIBRATION_MIN_SUGGESTED_CHANGE_C,
    BP_CALIBRATION_OUTLIER_K,
    BP_CALIBRATION_PROPORTIONAL_GAP_WARN_C,
    BP_CALIBRATION_SOLAR_CARRYOVER_GRID,
    BP_CALIBRATION_SOLAR_REFERENCE_QUANTILE,
    BP_CALIBRATION_SOLAR_SHIFT_COARSE_STEP_C,
    BP_CALIBRATION_SOLAR_SHIFT_MAX_C,
    BP_CALIBRATION_SOLAR_SHIFT_STEP_C,
    BP_CALIBRATION_STABILITY_TOLERANCE_C,
    BP_CALIBRATION_SWEEP_MAX_C,
    BP_CALIBRATION_SWEEP_MIN_C,
    BP_CALIBRATION_SWEEP_STEP_C,
    BP_CALIBRATION_VARIANCE_FLOOR_SHARE,
)
from .helpers import (
    daily_energy_is_unit_attributed,
    local_day_hours,
    unit_day_energy,
    vector_slot_hours,
)

# Regime boundary in ``statistics._get_prediction_from_model`` and the BP-2
# shield offset in ``learning``.  Mirrored here only to count how many hours
# a suggested change would reclassify; they are not tuned by this module.
_COLD_REGIME_DELTA_T = 4.0
_BP2_SHIELD_OFFSET = 2.0

# 95 % two-sided chi-square(1) quantile, for the profile-likelihood interval.
_CHI2_1_95 = 3.841
# MAD → standard deviation under normality.
_MAD_SCALE = 1.4826
# Free parameters of the model: b, U, cp, Δ, c, κ.
_N_PARAMS = 6
# Autocorrelation lags in the interval's variance inflation.
_MAX_LAG_DAYS = 7


def _grid(lo: float, hi: float, step: float) -> list[float]:
    start = math.ceil(lo / step - 1e-9) * step
    out = []
    value = start
    while value <= hi + 1e-9:
        out.append(round(value, 4))
        value += step
    return out


def _key(kappa: float) -> float:
    return round(kappa, 4)


_COP_SLOPES = _grid(0.0, BP_CALIBRATION_COP_SLOPE_MAX, BP_CALIBRATION_COP_SLOPE_STEP)
# Every grid the search walks must be a subset of the precomputed tables
# (a slope outside them still works, via ``_table``, but costs a rebuild
# per day and evaluation).
for _grid_step in (BP_CALIBRATION_COP_SLOPE_COARSE_STEP, BP_CALIBRATION_COP_TABLE_STEP):
    assert {
        _key(k) for k in _grid(0.0, BP_CALIBRATION_COP_SLOPE_MAX, _grid_step)
    } <= {_key(k) for k in _COP_SLOPES}, "COP slope grids must nest in the table grid"


# ---------------------------------------------------------------------------
# Day assembly
# ---------------------------------------------------------------------------

def _parse_date(key) -> date | None:
    try:
        return date.fromisoformat(str(key)[:10])
    except ValueError:
        return None


def _float(value, default=None):
    # Real numbers (and numeric strings from hand-edited storage) only:
    # anything else — including objects that merely implement __float__ —
    # is not a stored value.
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _inertia_kernel(coordinator) -> tuple[list[float] | None, int | None, float | None]:
    """``(weights oldest → newest, window in hours, tau)`` or ``(None, None, None)``.

    Mirrors live learning: the whole kernel, each reading weighted by its
    age, history cut at a gap longer than tau
    (``coordinator._get_recent_log_temps`` + ``helpers.weighted_inertia``).
    Without a numeric ``inertia_tau`` there is no gap cut.
    """
    weights = getattr(coordinator, "inertia_weights", None)
    if not isinstance(weights, (list, tuple)) or not weights:
        return None, None, None
    out = []
    for w in weights:
        if isinstance(w, bool) or not isinstance(w, (int, float)):
            return None, None, None
        out.append(float(w))
    if sum(out) <= 0.0:
        return None, None, None
    tau = _float(getattr(coordinator, "inertia_tau", None))
    return out, len(out), (tau if tau is not None and tau > 0 else None)


def _weighted_inertia(recent: list[tuple[int, float]], weights: list[float]) -> float:
    """Age-weighted mean of ``(hour index, temp)`` readings, newest last.

    Same weighting as ``helpers.weighted_inertia``: the newest reading is
    age 0, each reading takes the kernel weight of its age, and the weights
    used are normalised.  Two readings of one age keep the newer.
    """
    newest = recent[-1][0]
    n = len(weights)
    by_age: dict[int, float] = {}
    for idx, t in recent:
        age = newest - idx
        if 0 <= age < n:
            by_age[age] = t
    total_w = sum(weights[n - 1 - a] for a in by_age)
    if total_w <= 0.0:
        return sum(by_age.values()) / len(by_age)
    return sum(weights[n - 1 - a] * t for a, t in by_age.items()) / total_w


def _prefix_tables(sorted_t: list[float]) -> dict:
    """Per COP slope, prefix sums of ``w`` and ``t·w`` over the sorted
    temperatures, ``w = 1 / max(floor, 1 + κ·t)`` — so a heating
    degree-hour sum at any base is one bisection."""
    tables = {}
    for kappa in _COP_SLOPES:
        pw, ptw = [0.0], [0.0]
        for t in sorted_t:
            w = 1.0 / max(BP_CALIBRATION_COP_FLOOR, 1.0 + kappa * t)
            pw.append(pw[-1] + w)
            ptw.append(ptw[-1] + t * w)
        tables[_key(kappa)] = (pw, ptw)
    return tables


def _table(day: dict, kappa: float):
    """``(prefix w, prefix t·w)`` for ``κ``; built and cached on demand for a
    slope outside the precomputed grid."""
    key = _key(kappa)
    table = day["tables"].get(key)
    if table is None:
        pw, ptw = [0.0], [0.0]
        for t in day["sorted_t"]:
            w = 1.0 / max(BP_CALIBRATION_COP_FLOOR, 1.0 + kappa * t)
            pw.append(pw[-1] + w)
            ptw.append(ptw[-1] + t * w)
        table = day["tables"][key] = (pw, ptw)
    return table


def collect_days(coordinator, daily_history: dict, today: date,
                 lookback_days: int = BP_CALIBRATION_LOOKBACK_DAYS, *,
                 units: list[str] | None = None,
                 unit_first_seen: dict[str, str] | None = None) -> dict:
    """Assemble usable days from ``daily_history``.

    Returns ``{"days": [...], "discarded": {...}, "days_in_window": int}``,
    days in chronological order (the serial-correlation estimate relies on
    it).  The inertia series runs over every day in date order — including
    days later discarded and a warm-up period before the window — and a
    missing day breaks it, as a gap longer than ``τ`` does live.

    With ``units``, each day also carries ``units_heat``: each unit's
    heating energy per hour (``helpers.unit_day_energy``), for the units
    whose split the day records, from the unit's first day with energy
    (``unit_first_seen``) on — before it the unit's zeros are not
    observations.
    """
    discarded = {
        "missing_regime_split": 0,
        "unattributed_energy": 0,
        "incomplete_day": 0,
        "auxiliary_heat": 0,
        "guest_mode": 0,
        "high_wind": 0,
        "invalid_energy": 0,
    }
    cutoff = today - timedelta(days=lookback_days)
    weights, window, tau = _inertia_kernel(coordinator)
    warmup = timedelta(days=(window // 24 + 1) if window else 0)
    wind_threshold = _float(getattr(coordinator, "wind_threshold", None))

    dated = []
    for key, entry in (daily_history or {}).items():
        d = _parse_date(key)
        if d is None or not isinstance(entry, dict):
            continue
        if cutoff - warmup <= d < today:
            dated.append((d, entry))
    dated.sort(key=lambda item: item[0])

    days = []
    days_in_window = 0
    recent: list[tuple[int, float]] = []  # (hour index, raw temperature)
    prev_date = None
    prev_solar = None
    for d, entry in dated:
        consecutive = prev_date is not None and d == prev_date + timedelta(days=1)
        prev_date = d
        solar = max(0.0, _float(entry.get("solar_factor"), 0.0) or 0.0)
        # Yesterday's sun is weather, used whether or not yesterday itself
        # was usable.  Without a record of yesterday, assume it matched today.
        solar_prev = prev_solar if consecutive and prev_solar is not None else solar
        prev_solar = solar

        vectors = entry.get("hourly_vectors") or {}
        raw = vectors.get("temp") or []
        slot_hours = vector_slot_hours(vectors)
        base_index = d.toordinal() * 24
        extra = 0  # clock hours so far today beyond one per slot
        temps: list[float] = []
        for slot, value in enumerate(raw[:24]):
            t = _float(value)
            if t is None:
                continue  # a missing hour drops out; only a gap > tau resets
            # A slot holding the two passes through the repeated DST
            # fall-back hour is two hours at their mean temperature.
            for repeat in range(slot_hours[slot]):
                extra += 1 if repeat else 0
                idx = base_index + slot + extra
                if weights:
                    # A gap longer than tau is a thermal discontinuity:
                    # history before it is dropped, as live does.
                    if recent and tau is not None and idx - recent[-1][0] > tau:
                        recent = []
                    recent.append((idx, t))
                    recent = [(i, v) for i, v in recent if i > idx - window]
                    temps.append(_weighted_inertia(recent, weights))
                else:
                    temps.append(t)

        if d < cutoff:
            continue
        days_in_window += 1

        if "regime_heating_kwh" not in entry or "regime_cooling_kwh" not in entry:
            discarded["missing_regime_split"] += 1
            continue
        if not daily_energy_is_unit_attributed(entry):
            discarded["unattributed_energy"] += 1
            continue
        n = len(temps)
        day_hours = local_day_hours(d)
        if n < min(day_hours, 24):
            # Complete days only (see ``const``): 23 hours on the
            # spring-forward day, 24 otherwise.  A fall-back day stored
            # before slots recorded their ``hours`` holds its 25 hours in
            # 24 slots, so 24 is all that can be asked of it.
            discarded["incomplete_day"] += 1
            continue
        if (_float(entry.get("aux_impact_kwh"), 0.0) or 0.0) > 0.0:
            discarded["auxiliary_heat"] += 1
            continue
        if (_float(entry.get("guest_impact_kwh"), 0.0) or 0.0) > 0.0:
            discarded["guest_mode"] += 1
            continue
        if wind_threshold is not None:
            hourly_wind = [
                (w, slot_hours[i])
                for i, w in enumerate(_float(v) for v in (vectors.get("wind") or [])[:24])
            ]
            hourly_wind = [(w, n) for w, n in hourly_wind if w is not None]
            if hourly_wind:
                windy = sum(n for w, n in hourly_wind if w >= wind_threshold)
                too_windy = windy > BP_CALIBRATION_MAX_HIGH_WIND_HOURS
            else:
                mean_wind = _float(entry.get("wind"))
                too_windy = mean_wind is not None and mean_wind >= wind_threshold
            if too_windy:
                discarded["high_wind"] += 1
                continue
        heat = _float(entry.get("regime_heating_kwh"))
        cool = _float(entry.get("regime_cooling_kwh"))
        if heat is None or cool is None or heat < 0.0 or cool < 0.0:
            discarded["invalid_energy"] += 1
            continue

        # Fall-back DST day: 25 hours of energy, two logs merged into one
        # temperature slot.
        energy_hours = day_hours if day_hours == 25 else n
        sorted_t = sorted(temps)
        extra_fields = {}
        if units is not None:
            units_heat = {}
            for eid in units:
                first = (unit_first_seen or {}).get(eid)
                if first is None or d.isoformat() < first:
                    continue
                split = unit_day_energy(entry, eid)
                if split is not None:
                    units_heat[eid] = split[0] / energy_hours
            extra_fields["units_heat"] = units_heat
        days.append({
            "date": d.isoformat(),
            "iso_week": d.isocalendar()[1],
            "sorted_t": sorted_t,
            "tables": _prefix_tables(sorted_t),
            "n": n,
            "mean_t": sum(sorted_t) / n,
            "y_heat": heat / energy_hours,
            "y_cool": cool / energy_hours,
            "solar": solar,
            "solar_prev": solar_prev,
            **extra_fields,
        })

    discarded["total_discarded"] = sum(discarded.values())
    return {"days": days, "discarded": discarded, "days_in_window": days_in_window}


def degree_hours(day: dict, base: float, *, heating: bool = True,
                 cop_slope: float = 0.0) -> float:
    """Mean over the day's hours of ``max(0, base − T) / (1 + κ·T)``
    (heating) or ``max(0, T − base)`` (cooling, no COP term)."""
    ts, n = day["sorted_t"], day["n"]
    pw, ptw = _table(day, cop_slope if heating else 0.0)
    if heating:
        k = bisect.bisect_left(ts, base)
        return (base * pw[k] - ptw[k]) / n
    k = bisect.bisect_right(ts, base)
    return ((ptw[n] - ptw[k]) - (pw[n] - pw[k]) * base) / n


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def _solar_reference(days: list) -> float:
    values = sorted(d["solar"] for d in days)
    if not values:
        return 0.0
    idx = int(BP_CALIBRATION_SOLAR_REFERENCE_QUANTILE * (len(values) - 1))
    return values[idx]


def fit_change_point(days: list, cp: float, shift: float = 0.0,
                     s_ref: float = 0.0, carryover: float = 0.0,
                     cop_slope: float = 0.0, *, heating: bool = True,
                     weights: list[float] | None = None) -> dict | None:
    """(Weighted) least-squares fit of ``y = b + U·DH`` at fixed ``(cp, Δ, c, κ)``.

    ``weights`` are inverse variances (see :func:`_variance_weights`);
    without them the fit is ordinary LS.  ``sse`` is the weighted sum of
    squares the search compares, ``sse_raw`` the unweighted one (kWh²/h²)
    the RMSE is reported from.

    Returns ``None`` when the candidate is not identified: fewer than
    ``BP_CALIBRATION_MIN_DAYS_PER_SIDE`` days on either side of the day's
    balance point, or no positive temperature response (``U ≤ 0``).  ``b``
    is constrained to ``≥ 0`` — a negative floor has no physical reading
    and would push the reported proportional zero below the change point.
    """
    n = len(days)
    if n < 3:
        return None
    ys, zs, active = [], [], []
    for d in days:
        s_eff = (1.0 - carryover) * d["solar"] + carryover * d["solar_prev"]
        base = cp - (shift * s_eff / s_ref if s_ref > 0.0 else 0.0)
        zs.append(degree_hours(d, base, heating=heating, cop_slope=cop_slope))
        ys.append(d["y_heat"] if heating else d["y_cool"])
        active.append((d["mean_t"] < base) if heating else (d["mean_t"] > base))
    n_active = sum(active)
    n_flat = n - n_active
    if n_active < BP_CALIBRATION_MIN_DAYS_PER_SIDE or n_flat < BP_CALIBRATION_MIN_DAYS_PER_SIDE:
        return None

    ws = weights if weights is not None else [1.0] * n
    sw = sum(ws)
    z_mean = sum(w * z for w, z in zip(ws, zs)) / sw
    y_mean = sum(w * y for w, y in zip(ws, ys)) / sw
    szz = sum(w * (z - z_mean) ** 2 for w, z in zip(ws, zs))
    if szz <= 0.0:
        return None
    szy = sum(w * (z - z_mean) * (y - y_mean) for w, z, y in zip(ws, zs, ys))
    slope = szy / szz
    floor = y_mean - slope * z_mean
    if floor < 0.0:
        # Refit through the origin (b = 0).
        s_zz0 = sum(w * z * z for w, z in zip(ws, zs))
        if s_zz0 <= 0.0:
            return None
        slope = sum(w * z * y for w, z, y in zip(ws, zs, ys)) / s_zz0
        floor = 0.0
    if slope <= 0.0:
        return None

    residuals = [y - floor - slope * z for z, y in zip(zs, ys)]
    return {
        "cp": cp,
        "shift": shift,
        "carryover": carryover,
        "cop_slope": cop_slope,
        "b": floor,
        "U": slope,
        "sse": sum(w * r * r for w, r in zip(ws, residuals)),
        "sse_raw": sum(r * r for r in residuals),
        "n": n,
        "n_active": n_active,
        "n_flat": n_flat,
        "residuals": residuals,
        "active": active,
    }


def _variance_inflation(residuals: list[float]) -> tuple[float, float]:
    """Newey-West (Bartlett) long-run variance factor over one week of lags.

    Returns ``(inflation, lag1)``.  Weather persists for days, so daily
    residuals are autocorrelated beyond lag 1; a lag-1-only correction
    leaves the interval too narrow when the model misses a multi-day
    effect.  Lags count usable days, not calendar days — discarded days
    shorten them, a documented approximation.  Negative autocorrelation
    sums are not allowed to *narrow* the interval.
    """
    n = len(residuals)
    if n < 3:
        return 1.0, 0.0
    mean = sum(residuals) / n
    dev = [r - mean for r in residuals]
    den = sum(x * x for x in dev)
    if den <= 0.0:
        return 1.0, 0.0
    max_lag = min(_MAX_LAG_DAYS, n - 2)
    rhos = [
        sum(dev[i] * dev[i - k] for i in range(k, n)) / den
        for k in range(1, max_lag + 1)
    ]
    long_run = 1.0 + 2.0 * sum(
        (1.0 - k / (max_lag + 1)) * rho for k, rho in enumerate(rhos, start=1)
    )
    return max(1.0, long_run), max(0.0, rhos[0])


def _near(values: list[float], centre: float, radius: float) -> list[float]:
    return [v for v in values if abs(v - centre) <= radius + 1e-9]


def _bases(days, cp, shift, s_ref, carryover):
    return [
        cp - (
            shift * ((1.0 - carryover) * d["solar"] + carryover * d["solar_prev"]) / s_ref
            if s_ref > 0.0 else 0.0
        )
        for d in days
    ]


def _fast_sse(days, ys, bases, cop_slope, heating, weights=None) -> float | None:
    """Weighted SSE of the constrained LS fit, from sums in one pass.

    The search evaluates thousands of candidates; this is
    ``fit_change_point`` without the per-day lists (same constraints, same
    result), which is rebuilt only for the winners.
    """
    ws = weights if weights is not None else [1.0] * len(days)
    kappa = cop_slope if heating else 0.0
    key = _key(kappa)
    tables = [d["tables"][key] if key in d["tables"] else _table(d, kappa) for d in days]
    bisect_left, bisect_right = bisect.bisect_left, bisect.bisect_right
    sw = sz = szz = szy = sy = syy = 0.0
    for d, (pw, ptw), y, base, w in zip(days, tables, ys, bases, ws):
        ts = d["sorted_t"]
        if heating:
            k = bisect_left(ts, base)
            z = (base * pw[k] - ptw[k]) / d["n"]
        else:
            k = bisect_right(ts, base)
            m = d["n"]
            z = ((ptw[m] - ptw[k]) - (pw[m] - pw[k]) * base) / m
        sw += w
        sz += w * z
        szz += w * z * z
        szy += w * z * y
        sy += w * y
        syy += w * y * y
    var_z = szz - sz * sz / sw
    if var_z <= 0.0:
        return None
    slope = (szy - sz * sy / sw) / var_z
    floor = (sy - slope * sz) / sw
    if floor < 0.0:
        if szz <= 0.0:
            return None
        slope = szy / szz
        floor = 0.0
    if slope <= 0.0:
        return None
    return (
        syy + sw * floor * floor + slope * slope * szz
        + 2.0 * floor * slope * sz - 2.0 * floor * sy - 2.0 * slope * szy
    )


def _profile(days, cp, space, s_ref, heating, weights=None, refine=True):
    """Best fit over the nuisance terms (Δ, c, κ) at a fixed change point.

    Two stages: the coarse grid, then the fine grid around the coarse
    optimum.  Δ = 0 is evaluated only with c = 0 (carry-over has nothing to
    act on without a shift).
    """
    ys = [d["y_heat"] if heating else d["y_cool"] for d in days]
    k_side = BP_CALIBRATION_MIN_DAYS_PER_SIDE

    def best_of(shifts, carries, slopes):
        best = None  # (sse, shift, carryover, cop_slope)
        for carryover in carries:
            for shift in shifts:
                if shift == 0.0 and carryover > 0.0:
                    continue
                bases = _bases(days, cp, shift, s_ref, carryover)
                n_active = sum(
                    1 for d, base in zip(days, bases)
                    if ((d["mean_t"] < base) if heating else (d["mean_t"] > base))
                )
                if n_active < k_side or len(days) - n_active < k_side:
                    continue
                for cop_slope in slopes:
                    sse = _fast_sse(days, ys, bases, cop_slope, heating, weights)
                    if sse is not None and (best is None or sse < best[0]):
                        best = (sse, shift, carryover, cop_slope)
        return best

    coarse = best_of(space["shifts_coarse"], space["carries"], space["slopes_coarse"])
    if coarse is None:
        return None
    if not refine:
        _, shift, carryover, cop_slope = coarse
        return fit_change_point(days, cp, shift, s_ref, carryover, cop_slope,
                                heating=heating, weights=weights)
    fine = best_of(
        _near(space["shifts"], coarse[1], BP_CALIBRATION_SOLAR_SHIFT_COARSE_STEP_C),
        [coarse[2]],
        _near(space["slopes"], coarse[3], BP_CALIBRATION_COP_SLOPE_COARSE_STEP),
    )
    _, shift, carryover, cop_slope = fine if fine is not None and fine[0] < coarse[0] else coarse
    return fit_change_point(days, cp, shift, s_ref, carryover, cop_slope,
                            heating=heating, weights=weights)


def _search_space(days: list, heating: bool) -> tuple[dict, float]:
    s_ref = _solar_reference(days)
    sunny = s_ref > 1e-6
    shifts = (
        _grid(0.0, BP_CALIBRATION_SOLAR_SHIFT_MAX_C, BP_CALIBRATION_SOLAR_SHIFT_STEP_C)
        if sunny else [0.0]
    )
    slopes = _COP_SLOPES if heating else [0.0]
    return {
        "shifts": shifts,
        "shifts_coarse": (
            _grid(0.0, BP_CALIBRATION_SOLAR_SHIFT_MAX_C, BP_CALIBRATION_SOLAR_SHIFT_COARSE_STEP_C)
            if sunny else [0.0]
        ),
        "carries": list(BP_CALIBRATION_SOLAR_CARRYOVER_GRID) if sunny else [0.0],
        "slopes": slopes,
        "slopes_coarse": (
            _grid(0.0, BP_CALIBRATION_COP_SLOPE_MAX, BP_CALIBRATION_COP_SLOPE_COARSE_STEP)
            if heating else [0.0]
        ),
    }, s_ref


def carryover_at_limit(carryover: float, shift: float, carries: list[float]) -> bool:
    """Is the carry-over resting on its limit *and* identified?

    It acts only through the solar shift; below
    ``BP_CALIBRATION_CARRYOVER_MIN_SHIFT_C`` it moves each day's balance
    point by a fraction of a degree, so its value — limit or not — carries
    no information.
    """
    return (
        len(carries) > 1
        and carryover == carries[-1]
        and shift >= BP_CALIBRATION_CARRYOVER_MIN_SHIFT_C
    )


def _sweep_once(days: list, *, heating: bool, current_bp: float | None,
                weights: list[float] | None = None, coarse: bool = False) -> dict:
    n = len(days)
    if n < BP_CALIBRATION_MIN_DAYS:
        return {"status": "insufficient_days", "days": n,
                "min_days": BP_CALIBRATION_MIN_DAYS}

    space, s_ref = _search_space(days, heating)
    means = sorted(d["mean_t"] for d in days)
    k = BP_CALIBRATION_MIN_DAYS_PER_SIDE
    observed = [round(means[0], 1), round(means[-1], 1)]
    # Sun lowers each day's balance point, so days with a daily mean below
    # cp can still sit on the flat side: the upper end is not capped at the
    # K-th warmest mean but extended by the largest shift, and the per-day
    # side count in ``fit_change_point`` rejects what cannot be identified.
    # Cooling: sun raises load, the lower end is extended symmetrically.
    reach = space["shifts"][-1]
    lo = max(BP_CALIBRATION_SWEEP_MIN_C, means[k - 1] - (0.0 if heating else reach))
    hi = min(BP_CALIBRATION_SWEEP_MAX_C, means[-k] + (reach if heating else 0.0))
    step = BP_CALIBRATION_SWEEP_STEP_C * (2 if coarse else 1)
    candidates = _grid(lo, hi, step) if lo <= hi else []
    if not candidates:
        return {"status": "data_does_not_bracket_sweep_range", "days": n,
                "observed_daily_mean_range": observed}

    fits = [
        f for f in (
            _profile(days, cp, space, s_ref, heating, weights, refine=not coarse)
            for cp in candidates
        ) if f
    ]
    if not fits:
        return {"status": "no_temperature_response", "days": n,
                "observed_daily_mean_range": observed}

    best = min(fits, key=lambda f: f["sse"])
    ws = weights if weights is not None else [1.0] * n
    standardised = [r * math.sqrt(w) for r, w in zip(best["residuals"], ws)]
    inflation, rho = _variance_inflation(standardised)
    sigma2 = best["sse"] / max(1, n - _N_PARAMS)
    delta = _CHI2_1_95 * sigma2 * inflation
    in_interval = [f["cp"] for f in fits if f["sse"] <= best["sse"] + delta]

    evaluated = [f["cp"] for f in fits]
    current_fit = None
    if current_bp is not None:
        current_fit = _profile(days, float(current_bp), space, s_ref, heating, weights)

    gap = best["b"] / best["U"]
    result = {
        "status": "fitted",
        "change_point": best["cp"],
        "solar_shift_c": best["shift"],
        "solar_carryover": best["carryover"],
        "solar_reference_factor": round(s_ref, 4),
        "floor_kwh_per_h": round(best["b"], 4),
        "slope_kwh_per_k_h": round(best["U"], 4),
        "proportional_zero": round(best["cp"] + (gap if heating else -gap), 2),
        "proportional_gap_c": round(gap, 2),
        "interval": [min(in_interval), max(in_interval)],
        "optimum_at_sweep_boundary": best["cp"] in (evaluated[0], evaluated[-1]),
        "solar_shift_at_sweep_boundary": (
            len(space["shifts"]) > 1 and best["shift"] == space["shifts"][-1]
        ),
        "solar_carryover_at_sweep_boundary": carryover_at_limit(
            best["carryover"], best["shift"], space["carries"]
        ),
        "swept_range": [evaluated[0], evaluated[-1]],
        "observed_daily_mean_range": observed,
        "days": n,
        "days_active_arm": best["n_active"],
        "days_flat": best["n_flat"],
        "weighted": weights is not None,
        "rmse_kwh_per_h": round(math.sqrt(best["sse_raw"] / n), 4),
        "current_rmse_kwh_per_h": (
            round(math.sqrt(current_fit["sse_raw"] / n), 4) if current_fit else None
        ),
        "current_within_interval": (
            current_fit is not None and current_fit["sse"] <= best["sse"] + delta
        ),
        "residual_lag1_autocorrelation": round(rho, 3),
        "variance_inflation": round(inflation, 2),
        "_best": best,
        "_delta": delta,
        "_candidates": candidates,
    }
    if heating:
        result["cop_slope_per_c"] = best["cop_slope"]
        result["cop_slope_at_sweep_boundary"] = (
            len(space["slopes"]) > 1 and best["cop_slope"] == space["slopes"][-1]
        )
    # Nuisance terms resting on their limit: the true optimum lies outside
    # what was searched, so cp is conditional on a truncated model.
    result["limits_hit"] = [
        name for name, hit in (
            ("cop_slope", result.get("cop_slope_at_sweep_boundary", False)),
            ("solar_shift", result["solar_shift_at_sweep_boundary"]),
            ("solar_carryover", result["solar_carryover_at_sweep_boundary"]),
        ) if hit
    ]
    return result


def _cop_sensitivity(days: list, fit: dict, weights: list[float] | None) -> dict:
    """The change point at each COP slope, and whether it holds.

    cp and κ trade off: a steeper COP curve moves the change point up.
    For each slope on the table grid the fit is repeated with κ fixed
    (solar terms still profiled), on the same days and weights as the
    fit.  A slope is *supported* when its SSE lies within the interval
    threshold of the overall best — or when it brackets the best slope
    (the best fit's slope sits on a finer grid), so the supported set is
    never empty and the check never compares cp with itself.  The change
    point must agree across all supported slopes, the best fit's included,
    within ``BP_CALIBRATION_STABILITY_TOLERANCE_C``.
    """
    space, s_ref = _search_space(days, True)
    best, delta = fit["_best"], fit["_delta"]
    best_kappa = best["cop_slope"]
    n = len(days)
    rows = []
    for kappa in _grid(0.0, BP_CALIBRATION_COP_SLOPE_MAX, BP_CALIBRATION_COP_TABLE_STEP):
        fixed = dict(space, slopes=[kappa], slopes_coarse=[kappa])
        fits = [
            f for f in (
                _profile(days, cp, fixed, s_ref, True, weights) for cp in fit["_candidates"]
            ) if f
        ]
        if not fits:
            continue
        local = min(fits, key=lambda f: f["sse"])
        within = [f["cp"] for f in fits if f["sse"] <= local["sse"] + delta]
        # Rows bracketing an off-grid best slope; on-grid, only its own row.
        neighbour = abs(kappa - best_kappa) < BP_CALIBRATION_COP_TABLE_STEP - 1e-9
        rows.append({
            "cop_slope": kappa,
            "change_point": local["cp"],
            "interval": [min(within), max(within)],
            "rmse_kwh_per_h": round(math.sqrt(local["sse_raw"] / n), 4),
            "supported": local["sse"] <= best["sse"] + delta or neighbour,
        })
    supported = [r for r in rows if r["supported"]]
    cps = [r["change_point"] for r in supported] + [best["cp"]]
    # With the best slope on the upper limit the data want a steeper curve
    # than was searched, so the slopes that would move cp further are
    # simply not in the table: stability cannot be assessed (``None``),
    # not "stable".  ``nuisance_at_limit`` is what withholds that case; this
    # keeps the answer honest without relying on the verdict order.
    at_limit = best_kappa >= BP_CALIBRATION_COP_SLOPE_MAX - 1e-9
    return {
        "change_point_by_cop_slope": rows,
        "best_cop_slope": best_kappa,
        "supported_cop_slopes": [r["cop_slope"] for r in supported],
        "change_point_range_over_supported": [min(cps), max(cps)],
        "stable": (
            None if at_limit
            else max(cps) - min(cps) <= BP_CALIBRATION_STABILITY_TOLERANCE_C
        ),
    }


def _variance_weights(residuals: list[float], fitted: list[float]) -> list[float]:
    """Inverse-variance weights from ``σ² = s0² + s1²·ŷ²`` (feasible GLS).

    ``s0², s1²`` come from regressing the squared residuals on ``(1, ŷ²)``.
    ``s0²`` is floored at ``BP_CALIBRATION_VARIANCE_FLOOR_SHARE`` of the mean
    squared residual, the weight ratio is capped at
    ``BP_CALIBRATION_MAX_WEIGHT_RATIO`` and weights are scaled to mean 1, so
    the weighted SSE stays on the scale of the unweighted one.  Degenerate
    input falls back to equal weights.
    """
    n = len(residuals)
    r2 = [r * r for r in residuals]
    f2 = [f * f for f in fitted]
    mean_r2 = sum(r2) / n if n else 0.0
    if n < 3 or mean_r2 <= 0.0:
        return [1.0] * n
    a11, a12, a22 = float(n), sum(f2), sum(x * x for x in f2)
    v1, v2 = sum(r2), sum(x * y for x, y in zip(f2, r2))
    det = a11 * a22 - a12 * a12
    if det <= 0.0:
        return [1.0] * n
    s0 = (v1 * a22 - v2 * a12) / det
    s1 = (a11 * v2 - a12 * v1) / det
    if s1 <= 0.0:
        return [1.0] * n
    s0 = max(s0, BP_CALIBRATION_VARIANCE_FLOOR_SHARE * mean_r2)
    raw = [1.0 / (s0 + s1 * x) for x in f2]
    cap = min(raw) * BP_CALIBRATION_MAX_WEIGHT_RATIO
    raw = [min(w, cap) for w in raw]
    scale = n / sum(raw)
    return [w * scale for w in raw]


def _outlier_mask(residuals: list[float], active: list[bool]) -> list[bool]:
    """``True`` for days to drop: beyond ``BP_CALIBRATION_OUTLIER_K`` robust
    standard deviations of their own arm.  Residuals arrive standardised
    by the variance model; judging each arm separately stays robust where
    that model is off.
    """
    drop = [False] * len(residuals)
    for arm in (True, False):
        idx = [i for i, a in enumerate(active) if a == arm]
        if len(idx) < BP_CALIBRATION_MIN_DAYS_PER_SIDE:
            continue
        values = sorted(residuals[i] for i in idx)
        median = values[len(values) // 2]
        mad = sorted(abs(residuals[i] - median) for i in idx)[len(idx) // 2]
        if mad <= 0.0:
            continue
        limit = BP_CALIBRATION_OUTLIER_K * _MAD_SCALE * mad
        for i in idx:
            if abs(residuals[i] - median) > limit:
                drop[i] = True
    return drop


def variance_prior(days: list, *, heating: bool = True) -> dict | None:
    """Pass 1: a coarse ordinary-LS fit, its variance model and outliers.

    Returns ``{"weights": {date: w}, "dropped": [date, ...]}`` or ``None``
    when the coarse fit fails.  The weights come from
    :func:`_variance_weights` on the kept days; outliers are judged on
    residuals standardised by a first variance model, per arm
    (``BP_CALIBRATION_OUTLIER_K``).  Coarse is enough: the pass only has to
    locate the curve well enough to model its scatter.  Computed once on
    all days and reused for the halves — the variance model is a property
    of the data, not of the half.
    """
    first = _sweep_once(days, heating=heating, current_bp=None, coarse=True)
    if first["status"] != "fitted":
        return None
    best = first["_best"]
    ys = [d["y_heat"] if heating else d["y_cool"] for d in days]
    fitted = [y - r for y, r in zip(ys, best["residuals"])]
    weights = _variance_weights(best["residuals"], fitted)
    standardised = [r * math.sqrt(w) for r, w in zip(best["residuals"], weights)]
    mask = _outlier_mask(standardised, best["active"])
    kept = [(d, r, f) for d, r, f, m in zip(days, best["residuals"], fitted, mask) if not m]
    kept_weights = _variance_weights([r for _, r, _ in kept], [f for _, _, f in kept])
    return {
        "weights": {d["date"]: w for (d, _, _), w in zip(kept, kept_weights)},
        "dropped": [d["date"] for d, m in zip(days, mask) if m],
    }


def sweep(days: list, *, heating: bool = True, current_bp: float | None = None,
          cop_sensitivity: bool = False, prior: dict | None = None) -> dict:
    """Search ``(cp, Δ, c, κ)`` by feasible GLS.

    ``prior`` is :func:`variance_prior`'s output (computed here when not
    given): the outliers to drop and inverse-variance weights for the
    rest.  The kept days are refitted with those weights, and that fit is
    what is reported: every SSE comparison and the interval (profile
    likelihood over ``cp``, ``ΔSSE ≤ χ²₁(0.95)·σ²``, σ² inflated for serial
    correlation over a week of lags) are on the weighted, i.e.
    variance-equalised, scale — heating energy scatters far more on cold
    days than on the flat arm.  More than
    ``BP_CALIBRATION_MAX_OUTLIER_SHARE`` of the days dropped is
    ``poor_fit``.

    The change-point range is the configured ceiling intersected with the
    data-feasible range.  ``cop_sensitivity`` (heating only) adds
    :func:`_cop_sensitivity` on the final days and weights.
    """
    if prior is None:
        prior = variance_prior(days, heating=heating) if len(days) >= BP_CALIBRATION_MIN_DAYS else None
    if prior is None:
        result = _sweep_once(days, heating=heating, current_bp=current_bp)
        dropped: list[str] = []
        weights = None
        kept = days
    else:
        dropped_set = set(prior["dropped"])
        kept = [d for d in days if d["date"] not in dropped_set]
        dropped = [d["date"] for d in days if d["date"] in dropped_set]
        weights = [prior["weights"].get(d["date"], 1.0) for d in kept]
        result = _sweep_once(kept, heating=heating, current_bp=current_bp, weights=weights)
    if cop_sensitivity and heating and result["status"] == "fitted":
        result["cop_sensitivity"] = _cop_sensitivity(kept, result, weights)
    for internal in ("_best", "_delta", "_candidates"):
        result.pop(internal, None)
    result["outlier_days_dropped"] = len(dropped)
    result["outlier_dates"] = dropped[:20]
    if days:
        result["outlier_share"] = round(len(dropped) / len(days), 3)
        if (
            result["status"] == "fitted"
            and len(dropped) / len(days) > BP_CALIBRATION_MAX_OUTLIER_SHARE
        ):
            result["status"] = "poor_fit"
    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _impact(hourly_log: list, cutoff_iso: str, current: float, suggested: float) -> dict:
    """How many logged hours a BP change would reclassify."""
    hours = regime = shield = side = 0
    for entry in hourly_log:
        if entry.get("timestamp", "") < cutoff_iso:
            continue
        try:
            t = float(entry["inertia_temp"])
        except (KeyError, TypeError, ValueError):
            continue
        hours += 1
        if (abs(current - t) > _COLD_REGIME_DELTA_T) != (abs(suggested - t) > _COLD_REGIME_DELTA_T):
            regime += 1
        if (t < current - _BP2_SHIELD_OFFSET) != (t < suggested - _BP2_SHIELD_OFFSET):
            shield += 1
        if (t > current) != (t > suggested):
            side += 1
    return {
        "hours_evaluated": hours,
        "cold_mild_regime_reclassified": regime,
        "bp2_shield_boundary_crossed": shield,
        "heating_cooling_side_changed": side,
    }


def _u_curve_minimum(correlation_data: dict) -> dict:
    """Cross-check: where the learned ``normal``-wind global curve bottoms out.

    Inherits bucket noise and needs a populated warm side, so it is
    reported for comparison only.  On a heating-only install the bottom is
    a flat floor rather than a point; the flat range is reported for that
    reason.
    """
    curve = {}
    for key, buckets in (correlation_data or {}).items():
        try:
            t = int(key)
            v = float((buckets or {}).get("normal"))
        except (TypeError, ValueError):
            continue
        curve[t] = v
    if len(curve) < 3:
        return {"status": "insufficient_buckets", "buckets": len(curve)}
    t_min = min(curve, key=curve.get)
    v_min = curve[t_min]
    tol = max(0.02, 0.10 * v_min)
    lo = hi = t_min
    while (lo - 1) in curve and curve[lo - 1] <= v_min + tol:
        lo -= 1
    while (hi + 1) in curve and curve[hi + 1] <= v_min + tol:
        hi += 1
    return {
        "status": "ok",
        "minimum_temp": t_min,
        "minimum_kwh": round(v_min, 4),
        "flat_bottom_range": [lo, hi],
        "has_warm_arm": any(t > hi and v > v_min + tol for t, v in curve.items()),
    }


def model_snapshot(coordinator) -> dict:
    """Copies of the model state the calibration reads.

    Taken in the event loop so the fit can run in an executor thread
    without racing the coordinator: ``daily_history`` entries are
    deep-copied (backfill and CSV import update them in place), limited to
    the window plus a warm-up week so the copy stays small.
    """
    model = coordinator.model
    first = (
        dt_util.now().date() - timedelta(days=BP_CALIBRATION_LOOKBACK_DAYS + 8)
    ).isoformat()
    heat_source_for = getattr(coordinator, "heat_source_for", None)
    sensors = getattr(coordinator, "energy_sensors", None)
    return {
        # Each unit's heat-source type (heat_source.py), reported beside
        # the fit; not yet used by it.
        "heat_source": (
            {eid: copy.deepcopy(heat_source_for(eid)) for eid in sensors}
            if callable(heat_source_for) and isinstance(sensors, list) else {}
        ),
        "daily_history": {
            k: copy.deepcopy(v)
            for k, v in (model.daily_history or {}).items()
            if str(k)[:10] >= first
        },
        "hourly_log": list(model.hourly_log or []),
        "correlation_data": {
            k: dict(v) if isinstance(v, dict) else v
            for k, v in (model.correlation_data or {}).items()
        },
    }


def calibrate_balance_point(coordinator, snapshot: dict | None = None) -> dict:
    """Service implementation for ``calibrate_balance_point``.

    ``snapshot`` is :func:`model_snapshot`'s output; without it the model
    is read directly (tests, callers already on the event loop).
    """
    snapshot = snapshot or model_snapshot(coordinator)
    current_bp = float(coordinator.balance_point)
    now = dt_util.now()
    collected = collect_days(coordinator, snapshot["daily_history"], now.date())
    days = collected["days"]

    prior = variance_prior(days) if len(days) >= BP_CALIBRATION_MIN_DAYS else None
    fit = sweep(days, heating=True, current_bp=current_bp, cop_sensitivity=True, prior=prior)
    halves = {
        name: sweep(
            [d for d in days if d["iso_week"] % 2 == parity],
            current_bp=current_bp,
            prior=prior,
        )
        for name, parity in (("even_weeks", 0), ("odd_weeks", 1))
    }
    report: dict = {
        "current_balance_point": current_bp,
        "method": "change_point_3ph_on_daily_degree_hours_with_solar_shift_and_cop",
        "lookback_days": BP_CALIBRATION_LOOKBACK_DAYS,
        "days_in_window": collected["days_in_window"],
        "days_used": len(days),
        "discarded_days": collected["discarded"],
        "fit": fit,
        "halves": halves,
        "stability": {"tolerance_c": BP_CALIBRATION_STABILITY_TOLERANCE_C},
        "suggested_balance_point": None,
        "apply_via": "reconfigure flow (Balance point field)",
        "unit_heat_source": snapshot.get("heat_source", {}),
    }

    if any(d["y_cool"] > 0.0 for d in days):
        report["cooling_change_point"] = sweep(days, heating=False)
    report["u_curve_cross_check"] = _u_curve_minimum(snapshot["correlation_data"])

    if fit["status"] != "fitted":
        report["verdict"] = "poor_fit" if fit["status"] == "poor_fit" else "insufficient_data"
        return report

    report["best_change_point"] = fit["change_point"]
    report["solar_shift_c"] = fit["solar_shift_c"]
    report["cop_slope_per_c"] = fit["cop_slope_per_c"]
    report["limits_hit"] = fit["limits_hit"]
    cop = fit.get("cop_sensitivity") or {}
    report["proportional_zero"] = fit["proportional_zero"]
    report["proportional_gap_c"] = fit["proportional_gap_c"]
    report["proportional_gap_large"] = (
        fit["proportional_gap_c"] > BP_CALIBRATION_PROPORTIONAL_GAP_WARN_C
    )

    half_statuses = {name: h["status"] for name, h in halves.items()}
    halves_fitted = all(status == "fitted" for status in half_statuses.values())
    half_cps = [h["change_point"] for h in halves.values() if h["status"] == "fitted"]
    stable = (
        halves_fitted
        and abs(half_cps[0] - half_cps[1]) <= BP_CALIBRATION_STABILITY_TOLERANCE_C
    )
    report["stability"].update({
        "half_statuses": half_statuses,
        "half_change_points": half_cps,
        "halves_fitted": halves_fitted,
        "stable": stable,
    })

    if fit["optimum_at_sweep_boundary"]:
        verdict = "optimum_at_sweep_boundary"
    elif fit["limits_hit"]:
        # cp sits inside the sweep but a shape term does not: the fit is
        # conditional on a model truncated at that limit (see limits_hit).
        verdict = "nuisance_at_limit"
    elif "poor_fit" in half_statuses.values():
        # A half the model does not describe — its change point is no
        # evidence either way.
        verdict = "poor_fit"
    elif not halves_fitted:
        # Not enough days to split in two — no evidence of disagreement.
        verdict = "insufficient_data_for_stability"
    elif not stable:
        verdict = "unstable_across_halves"
    elif cop.get("stable") is False:
        # The change point moves with the assumed COP curve by more than
        # the tolerance, across slopes the data cannot tell apart.
        verdict = "cp_depends_on_cop_slope"
    elif (
        fit["current_within_interval"]
        or abs(fit["change_point"] - current_bp) < BP_CALIBRATION_MIN_SUGGESTED_CHANGE_C
    ):
        verdict = "current_is_consistent"
    else:
        verdict = "suggest_change"
    report["verdict"] = verdict

    if verdict == "suggest_change":
        suggested = fit["change_point"]
        cutoff = (now - timedelta(days=BP_CALIBRATION_LOOKBACK_DAYS)).isoformat()
        report["suggested_balance_point"] = suggested
        report["impact"] = _impact(snapshot["hourly_log"], cutoff, current_bp, suggested)
        report["recommendation"] = (
            f"Set the balance point to {suggested} °C in the reconfigure flow, "
            "then run retrain_from_history so Track B's U-coefficient and the "
            "regime classification of borderline hours are re-derived under "
            "the new value."
        )
    return report
