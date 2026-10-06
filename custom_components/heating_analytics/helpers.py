"""Helper functions for Heating Analytics."""
from __future__ import annotations
import logging
import math
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from homeassistant.const import UnitOfSpeed

_LOGGER = logging.getLogger(__name__)


def convert_speed_to_ms(value: float, unit: str | None) -> float:
    """Convert speed to m/s."""
    if not unit:
        return value

    # Normalize unit string (though HA constants might be mixed case, usually lowercase or symbol)
    # We check against constants first, then string variants

    # Already in m/s - no conversion needed
    if unit in (UnitOfSpeed.METERS_PER_SECOND, "m/s", "ms"):
        return value

    # km/h
    if unit in (UnitOfSpeed.KILOMETERS_PER_HOUR, "km/h", "kmh", "km/t", "kph"):
        return value / 3.6

    # mph
    if unit in (UnitOfSpeed.MILES_PER_HOUR, "mph"):
        return value * 0.44704

    # knots
    if unit in (UnitOfSpeed.KNOTS, "kn", "kt", "knots"):
        return value * 0.514444

    # Unknown unit - log warning and return value as-is (assuming m/s)
    _LOGGER.warning(f"Unknown speed unit: {unit}, assuming value is in m/s")
    return value

def coerce_config_float(value, default: float, name: str) -> float:
    """Coerce a raw config scalar to a finite float, or fall back loudly.

    For values read at a persistence boundary (``entry.data``, the storage
    JSON) that feed bare arithmetic with no downstream guard.  Accepts
    ``int`` and ``float``; rejects ``bool`` (it subclasses ``int``, so
    ``True`` would silently become ``1.0``), strings, ``None``, non-finite
    values and integers too large to represent as a float.  Strings are rejected rather than parsed to match
    the ``solar_battery_decay`` precedent at the same boundary — one rule
    for every scalar there.

    A rejected value falls back to ``default`` and logs at ERROR, not
    silently: for load-bearing numbers like ``balance_point`` a wrong-but-
    running model is only acceptable if the substitution is visible.
    Failing hard instead would stop the integration from loading at all,
    leaving the user with no sensors and no UI path to repair ``entry.data``.
    ``None`` is treated as malformed too — callers pass ``.get(key, default)``
    so an absent key never reaches here as ``None``.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # ``float()`` inside the try: an int beyond float range (a hand-
        # edited ``10**400``) raises OverflowError here, and so would
        # ``math.isfinite`` on the raw int — either would escape the guard
        # and stop the integration loading on the very input it exists
        # to tolerate.
        try:
            result = float(value)
        except OverflowError:
            result = math.inf
        if math.isfinite(result):
            return result
    _LOGGER.error(
        "Config value %s=%r is not a finite number; using default %s. "
        "Reconfigure the integration to set a valid value.",
        name, value, default,
    )
    return float(default)


def hour_had_any_aux(entry: dict) -> bool:
    """Did auxiliary heat run at all during this ``hourly_log`` hour?

    For calibration and analysis code that wants "pure" hours with no aux
    influence.  ``auxiliary_active`` alone does not answer that: it is the
    learning-side **dominance** flag, set only when aux ran ≥ 80 % of the
    hour, so a 1–79 % aux hour carries ``auxiliary_active=False`` with a
    non-zero ``aux_impact_kwh`` and metered demand already reduced by aux.

    Do NOT use this where code deliberately mirrors live learning or
    prediction — those paths want the dominance flag exactly as logged
    (learning's own tiers: < 20 % learned as normal, 20–80 % skipped,
    ≥ 80 % aux-learned).
    """
    if entry.get("auxiliary_active"):
        return True
    try:
        return float(entry.get("aux_impact_kwh") or 0.0) > 0.0
    except (TypeError, ValueError):
        return False


# ``learning_status`` values under which live per-unit learning did not run
# for any entity, mapped to the discard-reason key offline fits report.
# Deliberately an explicit set, not a ``startswith("skipped_")`` prefix:
# ``skipped_global_saturation`` only skips the *global* base write — per-unit
# learning (solar NLMS included) still runs on those hours, and they are the
# right-censored rows Tobit exists to use.
_UNIT_LEARNING_SKIPPED_STATUSES = {
    "disabled": "learning_disabled",
    "skipped_no_data": "learning_no_data",
    "skipped_mixed_mode": "mixed_mode_aux",
    "skipped_dual_interference": "dual_interference",
}


# Every key :func:`unit_learning_skip_reason` can return, so callers can
# seed their discard counters and report a stable set of reasons.
UNIT_LEARNING_SKIP_REASONS = (
    *_UNIT_LEARNING_SKIPPED_STATUSES.values(),
    "post_aux_cooldown",
)


def hour_learning_skip_reason(entry: dict) -> str | None:
    """The part of :func:`unit_learning_skip_reason` that holds for every unit.

    Returns the discard-reason key when ``learning_status`` says live
    per-unit learning ran for no entity this hour, else ``None``.  Post-aux
    cooldown is not covered: it freezes only aux-affected units, so it
    needs the entity and is left to :func:`unit_learning_skip_reason`.
    """
    status = entry.get("learning_status")
    if not isinstance(status, str):
        return None
    return _UNIT_LEARNING_SKIPPED_STATUSES.get(status)


def unit_learning_skip_reason(
    entry: dict,
    entity_id: str,
    aux_affected_entities=None,
) -> str | None:
    """Why live per-unit learning did not learn ``entity_id`` from this hour.

    For offline fits that calibrate against what live learning uses
    (``batch_fit_solar`` / ``_4d``, ``fit_solar_obstruction``, the diagnose
    implied coefficient behind ``apply_implied_coefficient``, the per-unit
    min-base calibration) and for ``replay_solar_nlms``, which retrain uses
    to rebuild the live coefficients.  Returns a discard-reason key, or
    ``None`` when live learning did learn this entity from the hour.

    Complements the ``auxiliary_active`` (≥ 80 % aux) check those paths
    already make: that flag routes an hour to the aux path, while the
    statuses here cover the 20–80 % aux tier (``skipped_mixed_mode``) and
    the solar-and-aux guard (``skipped_dual_interference``), which the
    flag cannot see.  Aux under 20 % of the hour is learned as normal
    heating by design and is kept — which is why this is not
    :func:`hour_had_any_aux`.

    Post-aux cooldown freezes the global model, and per-unit learning skips
    only aux-affected entities; the rest keep learning.  Entries carrying
    ``aux_cooldown_entities`` (the affected units snapshotted at log time)
    are authoritative for which units the cooldown froze, whatever the
    status says — under daily learning a cooldown hour logs
    ``disabled_global_only``, and the aux scope may have been reconfigured
    since.  Older entries fall back to ``cooldown_post_aux`` plus the
    current ``aux_affected_entities`` (``None`` = every entity affected,
    matching ``_process_per_unit_learning``); a daily-learning cooldown
    hour logged before the snapshot existed is not recoverable and is kept.

    Entries without ``learning_status`` (logged before the field existed)
    are kept.
    """
    reason = hour_learning_skip_reason(entry)
    if reason is not None:
        return reason
    status = entry.get("learning_status")
    cooldown_entities = entry.get("aux_cooldown_entities")
    if isinstance(cooldown_entities, (list, tuple)):
        return "post_aux_cooldown" if entity_id in cooldown_entities else None
    if status == "cooldown_post_aux":
        if aux_affected_entities is None or entity_id in aux_affected_entities:
            return "post_aux_cooldown"
    return None


def _entry_reporting_units(entry: dict) -> set[str]:
    """Units an hourly-log entry shows reporting: its ``unit_breakdown`` keys
    plus ``units_reporting`` when the entry carries it."""
    units = set()
    breakdown = entry.get("unit_breakdown")
    if isinstance(breakdown, dict):
        units.update(breakdown)
    reporting = entry.get("units_reporting")
    if isinstance(reporting, (list, tuple)):
        units.update(reporting)
    return units


def first_reported_instants(entries, entity_ids=None) -> dict[str, datetime]:
    """The instant each unit first reported in the hourly log.

    The anchor for :func:`unit_hour_reported_kwh`'s rule on entries logged
    before ``units_reporting`` existed: a unit absent from such an entry is
    read as 0 kWh only once it has reported at least once.  Before that the
    sensor was most likely not configured yet (a sensor added after setup),
    and reading those hours as 0 would pull its buckets to zero for every
    temperature it had not yet seen.  Pass the whole log, not a ``days_back``
    window, so a unit that first reported before the window is anchored
    there.

    The first report in log order: the log is chronological (live appends
    are monotonic, CSV import re-sorts), so only an entry that introduces a
    unit not yet seen has its timestamp parsed, and the scan is cheap enough
    for a per-entity fit to call.  ``entity_ids`` restricts the result to
    those units and stops the scan once all of them are found.
    """
    wanted = None if entity_ids is None else set(entity_ids)
    first: dict[str, datetime] = {}
    for entry in entries or ():
        if wanted is not None and len(first) == len(wanted):
            break
        if not isinstance(entry, dict):
            continue
        units = _entry_reporting_units(entry)
        if wanted is not None:
            units &= wanted
        new_units = [eid for eid in units if eid not in first]
        if not new_units:
            continue
        instant = log_entry_instant(entry)
        if instant is None:
            continue
        for entity_id in new_units:
            first[entity_id] = instant
    return first


def log_first_reported(coordinator, entries, entity_ids=None) -> dict[str, datetime]:
    """:func:`first_reported_instants` over the coordinator's whole hourly log.

    For readers that get a ``days_back`` window: the anchor must come from the
    whole log.  Falls back to ``entries`` when the coordinator's log is not a
    real list (test stubs).
    """
    log = getattr(coordinator, "_hourly_log", None)
    return first_reported_instants(log if isinstance(log, list) else entries, entity_ids)


def unit_hour_reported_kwh(
    entry: dict,
    entity_id: str,
    first_reported: dict | None = None,
    *,
    instant: datetime | None = None,
) -> float | None:
    """The kWh ``entity_id``'s meter reported in a logged hour, or ``None``.

    ``None`` means the sensor did not report that hour: live per-unit
    learning skipped the unit, and so must anything that replays it.  A
    reported hour with no consumption is ``0.0`` and live learned it as 0 —
    for a thermostatic load (heating cable, thermostat) that is most hours.

    ``unit_breakdown`` drops units with 0 kWh, so it cannot tell a zero hour
    from an offline sensor.  Entries carrying ``units_reporting`` (the units
    that reported, written since that field was added) answer exactly: a
    unit in the list reported its ``unit_breakdown`` value, 0 when absent
    there; a unit not in the list did not report.

    Entries without it follow one written rule: a unit present in
    ``unit_breakdown`` reported that value; an absent unit reported 0 once
    it has reported at least once before (``first_reported``, from
    :func:`first_reported_instants` over the whole log), and did not report
    before that.  A configured sensor is absent from the meter map only
    when it is unavailable for the entire hour, which is rare, while a
    thermostatic load reads 0 in most hours — so "absent = 0" is the less
    wrong reading.  What it costs: an hour in which a sensor really was
    offline is learned as 0.  Anchoring on the unit's own first report,
    not on whether other units reported the same hour, keeps the rule
    working on single-unit installs, where an empty breakdown is the
    unit's own idle hour.  An entry with no ``unit_breakdown`` at all (a
    CSV-imported row) carries no per-unit evidence and reports nothing.

    ``instant`` is the entry's :func:`log_entry_instant`, for callers that
    already computed it.
    """
    breakdown = entry.get("unit_breakdown")
    if not isinstance(breakdown, dict):
        return None
    reporting = entry.get("units_reporting")
    if isinstance(reporting, (list, tuple)):
        if entity_id not in reporting:
            return None
        return float(breakdown.get(entity_id, 0.0) or 0.0)
    if entity_id in breakdown:
        return float(breakdown[entity_id] or 0.0)
    if not first_reported:
        return None
    first = first_reported.get(entity_id)
    if first is None:
        return None
    if instant is None:
        instant = log_entry_instant(entry)
    if instant is None or instant < first:
        return None
    return 0.0


def daily_energy_is_unit_attributed(entry: dict) -> bool:
    """Does a ``daily_history`` day's per-unit breakdown account for its energy?

    ``regime_heating_kwh`` / ``regime_cooling_kwh`` are summed from the
    hourly per-unit breakdown.  Hourly rows imported from CSV carry none,
    so a day built from them has a real ``kwh`` and a split of 0 / 0 —
    which reads as an idle day, not as a day without evidence.  Such a day
    fails this check; consumers of the split must treat it like a day
    that never recorded one.  Days with no meaningful energy pass.
    """
    from .const import DAILY_UNIT_ATTRIBUTION_MIN_SHARE

    try:
        kwh = float(entry.get("kwh") or 0.0)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(kwh) or kwh <= 0.05:
        return True
    breakdown = entry.get("unit_breakdown")
    if not isinstance(breakdown, dict):
        return False
    total = 0.0
    for value in breakdown.values():
        try:
            total += float(value or 0.0)
        except (TypeError, ValueError):
            continue
    return total >= DAILY_UNIT_ATTRIBUTION_MIN_SHARE * kwh


# The ``daily_history`` keys of the per-hour regime split (#1051) and its
# per-unit form: written together by ``aggregate_logs`` and dropped together
# when a day's logs cannot support them.
REGIME_SPLIT_KEYS = (
    "regime_heating_kwh",
    "regime_cooling_kwh",
    "unit_heating_kwh",
    "unit_cooling_kwh",
)


def unit_day_energy(entry: dict, entity_id: str) -> tuple[float, float] | None:
    """A unit's ``(heating_kwh, cooling_kwh)`` on a ``daily_history`` day.

    From the per-unit split (``unit_heating_kwh`` / ``unit_cooling_kwh``,
    summed per hour against that hour's modes; a unit absent from both
    used no energy in either regime).  A day aggregated before that split
    existed can still answer when all its energy was heating: its
    ``regime_heating_kwh`` equals the sum of its ``unit_breakdown`` and it
    has no cooling, so every unit's breakdown is heating energy.  ``None``
    when the day cannot say — no split, energy in OFF / DHW / cooling on a
    day without the per-unit split, or a breakdown that does not account
    for the day's energy (CSV import).
    """
    if not isinstance(entry, dict) or not daily_energy_is_unit_attributed(entry):
        return None
    heating = entry.get("unit_heating_kwh")
    cooling = entry.get("unit_cooling_kwh")
    if isinstance(heating, dict) and isinstance(cooling, dict):
        return (
            finite_float(heating.get(entity_id)) or 0.0,
            finite_float(cooling.get(entity_id)) or 0.0,
        )
    regime_heating = finite_float(entry.get("regime_heating_kwh"))
    regime_cooling = finite_float(entry.get("regime_cooling_kwh"))
    breakdown = entry.get("unit_breakdown")
    if regime_heating is None or regime_cooling is None or not isinstance(breakdown, dict):
        return None
    total = sum(finite_float(v) or 0.0 for v in breakdown.values())
    # Both sides are sums of values rounded to 3 decimals.
    if regime_cooling > 0.0 or abs(total - regime_heating) > max(0.01, 0.005 * total):
        return None
    return (finite_float(breakdown.get(entity_id)) or 0.0, 0.0)


def finite_float(value) -> float | None:
    """``value`` as a float when it is a finite real number, else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def recorded_balance_points(entries: list[dict]) -> set[float]:
    """The distinct ``bp_at_log_time`` values recorded on hourly entries.

    Hours logged before the field existed carry none and contribute
    nothing: the field arrived with a code update, not with a BP change,
    so the recorded hours of the same day speak for them.
    """
    bps = set()
    for e in entries:
        bp = finite_float(e.get("bp_at_log_time")) if isinstance(e, dict) else None
        if bp is not None:
            bps.add(bp)
    return bps


BP_CONFIG_RANGE = (10.0, 25.0)  # the reconfigure form's balance-point range


def infer_tdd_balance_point(entry: dict) -> tuple[bool, float | None]:
    """The BP a stored day's degree-days were computed at, read off its vectors.

    Each hourly slot holds ``T_h`` and ``tdd_h = |BP − T_h| / 24``, so the BP
    is one of ``T_h ± 24·tdd_h``, and the value most slots agree on is the
    BP of the day.  Returns ``(decided, bp)``:

    - ``(True, bp)`` when at least 90 % of the usable slots agree on one BP
      (rounded to 0.1 °C);
    - ``(True, None)`` when there are enough slots but no BP explains them —
      the day was logged under more than one BP, so its sum applies at none;
    - ``(False, None)`` when there are too few slots to say, or more than
      one BP in the configurable range explains them.

    A slot within 0.3 °C of its BP is skipped: both candidates sit on top of
    the temperature and cannot tell BPs apart.  The tolerance covers the
    0.1 °C rounding of logged temperatures and the 0.001 rounding of tdd.
    """
    vectors = entry.get("hourly_vectors")
    if not isinstance(vectors, dict):
        return False, None
    temps, tdds = vectors.get("temp"), vectors.get("tdd")
    if not isinstance(temps, list) or not isinstance(tdds, list):
        return False, None
    slot_hours = vector_slot_hours(vectors)
    pairs = []
    for i, (t, v) in enumerate(zip(temps[:24], tdds[:24])):
        t, v = finite_float(t), finite_float(v)
        if t is None or v is None:
            continue
        # A slot's tdd sums its hours (two in the repeated DST hour).
        v /= slot_hours[i]
        if v * 24.0 < 0.3:
            continue
        pairs.append((t - 24.0 * v, t + 24.0 * v))
    if len(pairs) < 6:
        return False, None
    tolerance = 0.15
    needed = 0.9 * len(pairs)
    winners: list[float] = []
    for pair in pairs:
        for candidate in pair:
            support = [
                lo if abs(lo - candidate) <= tolerance else hi
                for lo, hi in pairs
                if min(abs(lo - candidate), abs(hi - candidate)) <= tolerance
            ]
            if len(support) >= needed:
                bp = sum(support) / len(support)
                if all(abs(bp - w) > 2 * tolerance for w in winners):
                    winners.append(bp)
    if not winners:
        return True, None
    if len(winners) > 1:
        # A day that never crosses its BP fits both T + Δ and T − Δ; only
        # one of them is a balance point anyone could have configured.
        winners = [w for w in winners if BP_CONFIG_RANGE[0] <= w <= BP_CONFIG_RANGE[1]]
        if len(winners) != 1:
            return False, None
    return True, round(winners[0], 1)


def stored_tdd_matches(entry: dict, balance_point: float) -> bool:
    """May a ``daily_history`` day's stored ``tdd`` be read at ``balance_point``?

    ``balance_point`` on a day is the BP its ``tdd`` was summed at, or None
    when the hours were logged under different BPs (or under no recorded
    one) — such a tdd applies at no BP.  A day without the key at all
    predates the v10 migration that stamps every day; it keeps the
    reading it always had.
    """
    if "balance_point" not in entry:
        return True
    stored_bp = finite_float(entry.get("balance_point"))
    return stored_bp is not None and abs(stored_bp - balance_point) < 1e-6


def local_day_hours(d: date) -> int:
    """23, 24 or 25: the length of local day ``d``, for DST days."""
    from homeassistant.util import dt as dt_util

    tz = getattr(dt_util, "DEFAULT_TIME_ZONE", None)
    if not isinstance(tz, tzinfo):
        return 24
    start = datetime.combine(d, time(), tz).astimezone(timezone.utc)
    end = datetime.combine(d + timedelta(days=1), time(), tz).astimezone(timezone.utc)
    return int(round((end - start).total_seconds() / 3600.0))


def vector_slot_hours(vectors) -> list[int]:
    """Clock hours each of a day's 24 ``hourly_vectors`` slots covers.

    ``aggregate_logs`` folds every log entry of a local hour into one slot,
    and on the DST fall-back day slot 2 holds the two passes through the
    repeated hour: its temperature is their mean, its energy and tdd their
    sum.  ``hours`` records how many entries each slot holds.  Days stored
    before it existed, and any unreadable value, count one hour per slot.
    Whether a slot holds data at all is for the caller to decide from the
    field it reads.
    """
    raw = vectors.get("hours") if isinstance(vectors, dict) else None
    out = [1] * 24
    if isinstance(raw, list):
        for i, value in enumerate(raw[:24]):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if math.isfinite(value) and value >= 1:
                out[i] = int(round(value))
    return out


def daily_tdd_at_balance_point(entry: dict, balance_point: float) -> float:
    """Degree-days of one ``daily_history`` day, expressed at ``balance_point``.

    The stored ``tdd`` is fixed at the BP in force when the day was
    aggregated.  While ``stored_tdd_matches`` it is returned as stored — it
    is the exact sum.  Otherwise it is recomputed from the hourly
    temperature vector, ``Σ |BP − T_h| · hours_h / 24`` over the logged
    slots (``hours_h``: :func:`vector_slot_hours`).
    Without a vector, ``|BP − T_avg|``, which drops the extra degree-days
    of a day straddling the BP (small: those days sit near the BP, where
    degree-days are few).
    """
    stored = finite_float(entry.get("tdd"))
    if stored is not None and stored_tdd_matches(entry, balance_point):
        return stored
    vectors = entry.get("hourly_vectors")
    temps = vectors.get("temp") if isinstance(vectors, dict) else None
    if isinstance(temps, list):
        slot_hours = vector_slot_hours(vectors)
        valid = [
            (t, slot_hours[i])
            for i, t in enumerate(finite_float(v) for v in temps[:24])
            if t is not None
        ]
        if valid:
            return sum(abs(balance_point - t) * n for t, n in valid) / 24.0
    temp = finite_float(entry.get("temp"))
    if temp is not None:
        return abs(balance_point - temp)
    return stored if stored is not None else 0.0


def aux_affected_entities_of(coordinator):
    """The coordinator's ``aux_affected_entities`` as a set, or ``None``.

    ``None`` (every entity affected) when the attribute is missing or not a
    collection — e.g. a test stub — which is the conservative reading for
    :func:`unit_learning_skip_reason`.
    """
    value = getattr(coordinator, "aux_affected_entities", None)
    if isinstance(value, (list, tuple, set, frozenset)):
        return set(value)
    return None


def hour_start_utc(moment: datetime) -> datetime:
    """The start of the local clock hour holding ``moment``, as a UTC instant.

    The hour is floored in local time (so a half-hour-offset zone starts
    its hours at local ``:00``) and then converted, so the two passes
    through a repeated DST fall-back hour — ``02:00+02:00`` and
    ``02:00+01:00`` — are different values.  Comparing the local datetimes
    directly does not tell them apart: aware datetimes sharing a tzinfo
    compare naively, ignoring ``fold``.  The floor is taken in ``moment``'s
    own offset; a naive ``moment`` is then read as :func:`as_utc_instant`
    reads it.
    """
    return as_utc_instant(moment.replace(minute=0, second=0, microsecond=0))


def as_utc_instant(moment) -> datetime | None:
    """An aware UTC instant for ``moment``, or ``None`` if it is not a datetime.

    A naive datetime is read as local time (as UTC when no local zone is
    configured), matching how naive log timestamps were written.
    """
    from datetime import tzinfo as _tzinfo
    from homeassistant.util import dt as dt_util

    if not isinstance(moment, datetime):
        return None
    if moment.tzinfo is None:
        tz = getattr(dt_util, "DEFAULT_TIME_ZONE", None)
        moment = moment.replace(
            tzinfo=tz if isinstance(tz, _tzinfo) else timezone.utc
        )
    return moment.astimezone(timezone.utc)


def timestamp_instant(ts) -> datetime | None:
    """An ISO timestamp string as a UTC instant (see :func:`as_utc_instant`)."""
    from homeassistant.util import dt as dt_util

    if not isinstance(ts, str):
        return None
    try:
        return as_utc_instant(dt_util.parse_datetime(ts))
    except (TypeError, ValueError):
        return None


def log_entry_instant(entry: dict) -> datetime | None:
    """An hourly-log entry's ``timestamp`` as a UTC instant, or ``None``.

    For ordering and joining entries on time.  The timestamp string alone
    does neither: on the fall-back day ``02:00:00+01:00`` sorts before
    ``02:00:00+02:00`` although it is an hour later, and its local
    ``hour`` (or a ``YYYY-MM-DDTHH`` prefix) is shared by both entries.
    """
    if not isinstance(entry, dict):
        return None
    return timestamp_instant(entry.get("timestamp"))


def hour_slots(day_logs: list[dict]) -> list[tuple[object, dict]]:
    """A day's hourly-log entries in time order, each with its join key.

    The key is the entry's hour start as a UTC instant (floored in the
    timestamp's own offset, like :func:`hour_start_utc`), so the two passes through the
    repeated DST fall-back hour stay two hours — keying on the local
    ``hour`` number folded them into one and the second overwrote the
    first.  An entry whose timestamp cannot be read keys on its ``hour``
    and sorts first; production entries always carry a timestamp.
    Distributions joined against these entries must key the same way
    (:func:`hour_start_utc` on their parsed ``datetime``).
    """
    from homeassistant.util import dt as dt_util

    epoch = datetime.min.replace(tzinfo=timezone.utc)
    keyed = []
    for entry in day_logs:
        instant = None
        ts = entry.get("timestamp")
        if isinstance(ts, str):
            try:
                parsed = dt_util.parse_datetime(ts)
            except (TypeError, ValueError):
                parsed = None
            if isinstance(parsed, datetime):
                instant = hour_start_utc(parsed)
        key = instant if instant is not None else entry.get("hour", -1)
        keyed.append((instant or epoch, key, entry))
    keyed.sort(key=lambda item: item[0])
    return [(key, entry) for _, key, entry in keyed]


def get_last_year_iso_date(date_obj: date) -> date:
    """Get the corresponding date in the previous year based on ISO week and weekday.

    Handles the edge case where the current year has 53 weeks but the previous year only has 52.
    In that case, it falls back to Week 52.
    """
    year, week, weekday = date_obj.isocalendar()
    try:
        return date.fromisocalendar(year - 1, week, weekday)
    except ValueError:
        # Fallback to Week 52 if Week 53 doesn't exist in previous year
        return date.fromisocalendar(year - 1, 52, weekday)

def calculate_asymmetric_inertia(window: list[float]) -> tuple[float, str]:
    """Calculate effective temperature using asymmetric thermal inertia.

    Uses a slow profile (4h) when temperature is falling (heat shedding),
    a fast profile (2h) when temperature is rising (heat gaining),
    and a stable 3h profile otherwise.

    window: temperatures in chronological order (oldest first), same convention
            as the Gaussian kernel windows used in calibration.

    Returns a tuple of (effective_temperature, regime) where regime is one of
    'shedding', 'gaining', or 'stable'.
    """
    if not window:
        return 0.0, "stable"
    if len(window) == 1:
        return window[-1], "stable"

    current_temp = window[-1]
    trend_index = max(0, len(window) - 1 - 4)
    past_temp = window[trend_index]

    if current_temp < (past_temp - 0.5):
        weights = [0.20, 0.30, 0.30, 0.20]
        regime = "shedding"
    elif current_temp > (past_temp + 0.5):
        weights = [0.50, 0.50]
        regime = "gaining"
    else:
        weights = [0.34, 0.33, 0.33]
        regime = "stable"

    usable_window = window[-len(weights):]
    usable_weights = weights[-len(usable_window):]
    weight_sum = sum(usable_weights)
    eff_temp = sum(t * w for t, w in zip(usable_window, usable_weights)) / weight_sum
    return round(eff_temp, 2), regime


def weighted_inertia(
    temps: list,
    weights,
    tau: float | None = None,
) -> float | None:
    """Inertia temperature of an hourly-aligned temperature history.

    ``temps`` runs oldest → newest, one position per clock hour, ``None``
    for an hour with no reading; the newest position is the current hour
    and takes ``weights[-1]`` (``weights`` oldest → newest, as
    :func:`generate_exponential_kernel` returns them).  Each reading is
    weighted by its age, and the weights of the readings used are
    normalised, so a missing hour drops out instead of shifting the older
    ones onto the wrong weights.

    With ``tau``, history stops at the first break: two consecutive
    readings (or the current hour and the newest reading) more than ``tau``
    hours apart.  That is a thermal discontinuity — a long downtime — and
    what came before it says nothing about the building now.

    Returns ``None`` when no reading is usable.
    """
    used = inertia_samples(temps, weights, tau)
    total_w = sum(w for _, w in used)
    if total_w <= 0.0:
        return None
    return sum(t * w for t, w in used) / total_w


def inertia_samples(temps: list, weights, tau: float | None = None) -> list:
    """The ``(temp, weight)`` pairs :func:`weighted_inertia` averages.

    Oldest → newest, unnormalised kernel weights.  For display, so the
    shown weights reproduce the effective temperature exactly.
    """
    n_weights = len(weights)
    if n_weights == 0 or not temps:
        return []
    used = []
    previous_age = 0
    newest = len(temps) - 1
    for age in range(min(len(temps), n_weights)):
        t = temps[newest - age]
        if t is None:
            continue
        if tau is not None and age - previous_age > tau:
            break
        used.append((t, weights[n_weights - 1 - age]))
        previous_age = age
    used.reverse()
    return used


def inertia_temperatures(entries: list[dict], weights, tau: float | None) -> list:
    """Inertia temperature of every hourly-log entry, in input order.

    Each entry's own ``temp`` is its current hour; the entries before it
    (by time, not list position) are its history, aligned on their age in
    hours and cut at the first break as in :func:`weighted_inertia`.  An
    entry whose timestamp or ``temp`` cannot be read gets ``None`` and is
    left out of every other entry's history.
    """
    window = len(weights)
    points = []
    for index, entry in enumerate(entries):
        instant = log_entry_instant(entry)
        temp = finite_float(entry.get("temp")) if isinstance(entry, dict) else None
        if instant is not None and temp is not None:
            points.append((instant, temp, index))
    points.sort(key=lambda p: p[0])
    out: list = [None] * len(entries)
    for i, (instant, temp, index) in enumerate(points):
        aligned = [None] * window
        aligned[-1] = temp
        j = i - 1
        while j >= 0:
            age = int(round((instant - points[j][0]).total_seconds() / 3600.0))
            if age >= window:
                break
            if age >= 1 and aligned[window - 1 - age] is None:
                aligned[window - 1 - age] = points[j][1]
            j -= 1
        out[index] = weighted_inertia(aligned, weights, tau)
    return out


def generate_exponential_kernel(tau: float, window_hours: int = 168) -> tuple[float, ...]:
    """Generate a causal exponential decay kernel with time constant tau.

    Physically motivated by first-order thermal dynamics (RC-circuit analogy).
    Weights decay as e^(-t/tau) going back in time, giving a long tail with
    low but non-zero influence from days-old temperatures.

    tau: time constant in hours (higher = longer thermal memory)
    window_hours: how far back to look (default 7 days / 168 hours)
    Returns weights in oldest-to-newest order (same convention as Gaussian kernel).
    """
    # t=0 is most recent hour, t=window_hours-1 is oldest
    weights = [math.exp(-t / tau) for t in range(window_hours)]
    total = sum(weights)
    # Reverse to oldest-to-newest order
    return tuple(w / total for w in reversed(weights))


def solve_gauss_jordan(
    A: list[list[float]],
    b: list[float],
    *,
    ridge: float = 0.0,
    pivot_eps: float = 1e-12,
) -> list[float] | None:
    """Solve ``A · x = b`` via Gauss-Jordan elimination with partial pivoting.

    Dimension-agnostic — handles any square ``N×N`` system.  Returns the
    solution as a list of length ``N``, or ``None`` if the matrix is
    singular (any pivot magnitude < ``pivot_eps``).

    ``ridge`` adds a Tikhonov term to the diagonal before elimination
    (``A[i][i] += ridge``).  Used by the Tobit Newton step to guard
    rank-deficient Hessians at active-set boundaries.

    Pure Python — no numpy.  Inputs are not mutated; a working copy is
    built internally.
    """
    n = len(A)
    if n == 0 or len(b) != n:
        return None
    # Augmented [A | b] working matrix.
    M = [list(row) + [b[i]] for i, row in enumerate(A)]
    if ridge:
        for i in range(n):
            M[i][i] += ridge

    for col in range(n):
        pivot_row = col
        pivot_val = abs(M[col][col])
        for r in range(col + 1, n):
            if abs(M[r][col]) > pivot_val:
                pivot_val = abs(M[r][col])
                pivot_row = r
        if pivot_val < pivot_eps:
            return None
        if pivot_row != col:
            M[col], M[pivot_row] = M[pivot_row], M[col]

        pivot = M[col][col]
        for j in range(col, n + 1):
            M[col][j] /= pivot

        for r in range(n):
            if r == col:
                continue
            factor = M[r][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                M[r][j] -= factor * M[col][j]

    return [M[i][n] for i in range(n)]


def generate_gaussian_kernel(hours: int) -> tuple[float, ...]:
    """Generate a Gaussian/Bell-curve kernel for the given number of hours."""
    if hours == 1:
        return (1.0,)
    if hours == 2:
        return (0.5, 0.5)

    weights = []
    center = (hours - 1) / 2.0
    sigma = hours / 4.0

    for i in range(hours):
        x = i - center
        weights.append(math.exp(-(x**2) / (2 * sigma**2)))

    total = sum(weights)
    return tuple(w / total for w in weights)


def compute_base_ema_step(
    current_bucket: float,
    target: float,
    learning_rate: float,
    snr_weight: float,
) -> tuple[float, float]:
    """Pure-math kernel for the base-model EMA step (#967).

    The single arithmetic source of truth for the formula

        step       = learning_rate × snr_weight × (target − current_bucket)
        new_bucket = current_bucket + step

    Centralised here so diagnostic simulations and live learning provably
    use the same arithmetic — see #967 for the silent-drift hazard that
    motivated the extraction (if the live formula evolves without the
    diagnostic being patched in lockstep, ``base_model_4d_shadow`` and
    the promotion-gate metrics it feeds would silently characterise a
    model that no longer matches production).

    Caller owns:

    - Target construction (e.g. ``max(0, actual + delta)`` for the lift
      path, ``total_energy_kwh`` for the legacy path).
    - ``snr_weight`` computation via :func:`learning.compute_snr_weight`.
    - Post-step rounding or clamping (live learning rounds to 5 decimals
      before storing; diagnostic simulations leave the float unrounded).
    - Buffer-jumpstart vs EMA branching for cold-start (only applies to
      the live writer; diagnostic seeds from the current bucket).

    Returns ``(new_bucket_value, applied_step_size)``.  The step is
    returned separately so step-RMS jitter diagnostics can read it
    without re-deriving from the bucket delta.

    Consumers: the live writer (``learning.process_learning``), the
    retrain path (``learning.learn_from_historical_import``), and the
    diagnostic simulation (``diagnostics._compute_base_model_4d_shadow_report``).
    Call-form convention: callers that have already folded the SNR
    weight into an effective rate pass ``(effective_rate, 1.0)`` —
    multiplication by 1.0 is exact in IEEE-754, so the result is
    bit-identical to ``bucket + effective_rate × (target − bucket)``
    no matter how the effective rate was constructed.  Callers holding
    the factors separately (diagnostics) pass them as-is; Python's
    left-to-right evaluation makes ``lr × w × diff`` identical to
    ``(lr × w) × diff``.
    """
    step = learning_rate * snr_weight * (target - current_bucket)
    return current_bucket + step, step
