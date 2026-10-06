"""RetrainEngine — hosts retrain_from_history() extracted from coordinator.py.

Thin-delegate pattern: the engine holds a reference to the coordinator
and reaches back for state.  Public methods are called via delegates
on the coordinator so the external API is unchanged.
"""
from __future__ import annotations

import asyncio
import copy
import logging
import math
from datetime import date, datetime, timedelta

from homeassistant.util import dt as dt_util

from .const import (
    DEFAULT_DAILY_LEARNING_RATE,
    MODE_COOLING,
    MODE_DHW,
    MODE_GUEST_COOLING,
    MODE_GUEST_HEATING,
    MODE_HEATING,
    MODE_OFF,
    MODES_EXCLUDED_FROM_GLOBAL_LEARNING,
    RETRAIN_DRY_RUN_PREDICTION_DAYS,
    SNR_WEIGHT_FLOOR,
    SNR_WEIGHT_K,
)
from .daily_processor import (
    DailyProcessor,
    per_unit_replay_context,
    resmear_track_c_day,
)
from .helpers import finite_float, first_reported_instants, log_entry_instant
from .learning import compute_snr_weight, count_active_learnable_units
from .observation import DirectMeter, WeightedSmear

_LOGGER = logging.getLogger(__name__)


def _screen_affected_set_or_none(coordinator) -> frozenset[str] | None:
    """Return coordinator._screen_affected_set if it is a real set/frozenset.

    Guards against MagicMock-based test coordinators where ``getattr`` would
    return a MagicMock (truthy, `__contains__` returns False) and silently
    route every entity down the "unscreened" branch inside
    :meth:`learning.replay_solar_nlms` — masking inequality-path regressions
    in tests that don't explicitly set the attribute.
    """
    value = getattr(coordinator, "_screen_affected_set", None)
    if isinstance(value, (frozenset, set)):
        return value
    return None


def _solar_affected_set_or_none(coordinator) -> frozenset[str] | None:
    """Return coordinator._solar_affected_set if it is a real set/frozenset (#962).

    MagicMock-safe sibling of :func:`_screen_affected_set_or_none`.
    """
    value = getattr(coordinator, "_solar_affected_set", None)
    if isinstance(value, (frozenset, set)):
        return value
    return None


def _first_reported_or_none(coordinator) -> dict | None:
    """Each unit's first report over the whole hourly log, for replays.

    ``None`` (the replay anchors on the entries it gets) when the log is
    not a real list — MagicMock-safe like the helpers above.
    """
    log = getattr(coordinator, "_hourly_log", None)
    if isinstance(log, list):
        return first_reported_instants(log)
    return None


def _replay_report(result) -> dict:
    """The per-unit replay's counters for the service response."""
    return dict(result) if isinstance(result, dict) else {}


# The model state ``retrain_from_history`` rebuilds: the maps ``reset_first``
# clears and the U-coefficient.  A dry run replays into deep copies of
# exactly these (see ``RetrainEngine._dry_run``).
_RETRAIN_MODEL_MAPS = (
    "_correlation_data",
    "_correlation_data_per_unit",
    "_aux_coefficients",
    "_aux_coefficients_per_unit",
    "_learning_buffer_global",
    "_learning_buffer_per_unit",
    "_learning_buffer_aux_per_unit",
    "_solar_coefficients_per_unit",
    "_learning_buffer_solar_per_unit",
    "_observation_counts",
)
_RETRAIN_MODEL_STATE = _RETRAIN_MODEL_MAPS + ("_learned_u_coefficient",)

# A bucket counts as changed when it moved by more than this.
_DIFF_TOLERANCE = 1e-9


def _is_poisoned(entry: dict, daily_mode: bool = False) -> bool:
    """An hour the global retrain does not learn from."""
    s = entry.get("learning_status", "unknown")
    # In daily_learning_mode, "disabled" is the normal state for every hour
    # (Track A is blocked from writing to correlation_data). These hours
    # contain valid sensor data and must NOT be filtered out for Track B/C
    # daily aggregation.  Only genuine data-quality issues are poisoned.
    if daily_mode and s == "disabled":
        return False
    return s == "disabled" or s.startswith("skipped_") or s == "cooldown_post_aux"


def _install_model_copies(coordinator) -> dict:
    """Put deep copies of the retrained model state on ``coordinator``.

    Returns the live objects, for :func:`_reinstall_live_model`.  The live
    objects are set aside, not modified.  Besides the model state this
    covers the two pieces of shared state the replay writes elsewhere: the
    solar NLMS dead-zone counters on the LearningManager and each
    WeightedSmear strategy's per-day distribution.
    """
    live_state = {attr: getattr(coordinator, attr) for attr in _RETRAIN_MODEL_STATE}
    learning = getattr(coordinator, "learning", None)
    dead_zone = getattr(learning, "_dead_zone_counts", None)
    strategies = getattr(coordinator, "_unit_strategies", None)
    smear_state = [
        (strategy, strategy._distribution, strategy._daily_total)
        for strategy in (strategies.values() if isinstance(strategies, dict) else ())
        if isinstance(strategy, WeightedSmear)
    ]
    for attr, value in live_state.items():
        setattr(coordinator, attr, copy.deepcopy(value))
    if isinstance(dead_zone, dict):
        learning._dead_zone_counts = dict(dead_zone)
    return {
        "state": live_state,
        "dead_zone": dead_zone if isinstance(dead_zone, dict) else None,
        "smear": smear_state,
    }


def _reinstall_live_model(coordinator, live: dict) -> None:
    """Undo :func:`_install_model_copies`: the live objects go back."""
    for attr, value in live["state"].items():
        setattr(coordinator, attr, value)
    if live["dead_zone"] is not None:
        coordinator.learning._dead_zone_counts = live["dead_zone"]
    for strategy, distribution, daily_total in live["smear"]:
        strategy.set_distribution(distribution)
        strategy.set_daily_total(daily_total)


def _temp_band(temp) -> str:
    """The 5 °C band of a temperature (a bucket key or a float)."""
    try:
        t = float(temp)
    except (TypeError, ValueError):
        return "unknown"
    if not math.isfinite(t):
        return "unknown"
    low = int(math.floor(t / 5.0)) * 5
    return f"{low}..{low + 5}"


def _band_order(band: str) -> float:
    try:
        return float(band.split("..")[0])
    except ValueError:
        return math.inf


def _flatten_buckets(data) -> dict[tuple[str, str], float]:
    """``{temp_key: {wind_bucket: value}}`` as ``{(temp_key, bucket): value}``.

    A temperature holding a bare number (legacy global aux shape) keys on
    bucket ``"-"``.
    """
    flat: dict[tuple[str, str], float] = {}
    if not isinstance(data, dict):
        return flat
    for temp_key, cell in data.items():
        if isinstance(cell, dict):
            for bucket, value in cell.items():
                v = finite_float(value)
                if v is not None:
                    flat[(str(temp_key), str(bucket))] = v
        else:
            v = finite_float(cell)
            if v is not None:
                flat[(str(temp_key), "-")] = v
    return flat


def _delta_summary(deltas: list[float]) -> dict:
    if not deltas:
        return {"mean_delta": None, "mean_abs_delta": None, "max_abs_delta": None, "total_abs_delta": 0.0}
    abs_deltas = [abs(d) for d in deltas]
    return {
        "mean_delta": round(sum(deltas) / len(deltas), 5),
        "mean_abs_delta": round(sum(abs_deltas) / len(abs_deltas), 5),
        "max_abs_delta": round(max(abs_deltas), 5),
        "total_abs_delta": round(sum(abs_deltas), 5),
    }


def _bucket_map_diff(before, after) -> dict:
    """What a retrain changed in one ``{temp_key: {bucket: value}}`` map.

    Counts of buckets added, removed and changed, and the change of the
    changed ones (after − before, in the map's own unit), overall and per
    5 °C band of the bucket's temperature key and wind bucket.
    """
    b = _flatten_buckets(before)
    a = _flatten_buckets(after)
    added = [k for k in a if k not in b]
    removed = [k for k in b if k not in a]
    changed = {
        k: a[k] - b[k]
        for k in a
        if k in b and abs(a[k] - b[k]) > _DIFF_TOLERANCE
    }
    by_band: dict[str, dict[str, dict]] = {}
    groups: dict[tuple[str, str], dict] = {}
    for key in set(added) | set(removed) | set(changed):
        group = groups.setdefault(
            (_temp_band(key[0]), key[1]),
            {"added": 0, "removed": 0, "changed": 0, "deltas": []},
        )
        if key in changed:
            group["changed"] += 1
            group["deltas"].append(changed[key])
        elif key in a:
            group["added"] += 1
        else:
            group["removed"] += 1
    for (band, bucket), group in sorted(
        groups.items(), key=lambda item: (_band_order(item[0][0]), item[0][1])
    ):
        by_band.setdefault(band, {})[bucket] = {
            "added": group["added"],
            "removed": group["removed"],
            "changed": group["changed"],
            **_delta_summary(group["deltas"]),
        }
    return {
        "buckets_before": len(b),
        "buckets_after": len(a),
        "added": len(added),
        "removed": len(removed),
        "changed": len(changed),
        **_delta_summary(list(changed.values())),
        "by_band": by_band,
    }


def _per_unit_map_diff(before, after) -> dict:
    """:func:`_bucket_map_diff` per unit, and the units ranked by movement."""
    before = before if isinstance(before, dict) else {}
    after = after if isinstance(after, dict) else {}
    by_unit = {
        eid: _bucket_map_diff(before.get(eid, {}), after.get(eid, {}))
        for eid in sorted(set(before) | set(after))
    }
    ranked = sorted(
        (eid for eid, d in by_unit.items() if d["added"] or d["removed"] or d["changed"]),
        key=lambda eid: (-by_unit[eid]["total_abs_delta"], eid),
    )
    return {"by_unit": by_unit, "units_by_movement": ranked}


def _coefficient_view(value):
    if not isinstance(value, dict):
        return value
    return {
        k: (round(v, 5) if finite_float(v) is not None else v)
        for k, v in value.items()
    }


def _solar_coefficient_diff(before, after) -> dict:
    """Solar coefficients before / after, per (entity, regime) that changed."""
    before = before if isinstance(before, dict) else {}
    after = after if isinstance(after, dict) else {}
    out: dict[str, dict] = {}
    for eid in sorted(set(before) | set(after)):
        b_entity = before.get(eid) if isinstance(before.get(eid), dict) else {}
        a_entity = after.get(eid) if isinstance(after.get(eid), dict) else {}
        for regime in sorted(set(b_entity) | set(a_entity)):
            b = b_entity.get(regime)
            a = a_entity.get(regime)
            if b == a:
                continue
            out.setdefault(eid, {})[regime] = {
                "before": _coefficient_view(b),
                "after": _coefficient_view(a),
            }
    return out


def _round_or_none(value, digits: int = 4):
    v = finite_float(value)
    return round(v, digits) if v is not None else None


def _model_diff(live: dict, retrained: dict) -> dict:
    """What the retrain changed: live model state vs the retrained copies."""
    return {
        "global_base": _bucket_map_diff(
            live["_correlation_data"], retrained["_correlation_data"]
        ),
        "per_unit_base": _per_unit_map_diff(
            live["_correlation_data_per_unit"], retrained["_correlation_data_per_unit"]
        ),
        "global_aux": _bucket_map_diff(
            live["_aux_coefficients"], retrained["_aux_coefficients"]
        ),
        "per_unit_aux": _per_unit_map_diff(
            live["_aux_coefficients_per_unit"], retrained["_aux_coefficients_per_unit"]
        ),
        "solar_coefficients": _solar_coefficient_diff(
            live["_solar_coefficients_per_unit"], retrained["_solar_coefficients_per_unit"]
        ),
        "learned_u_coefficient": {
            "before": _round_or_none(live["_learned_u_coefficient"]),
            "after": _round_or_none(retrained["_learned_u_coefficient"]),
        },
    }


def _fit(pairs: list[tuple[float, float]]) -> dict:
    """Bias and RMSE of (predicted, actual) pairs."""
    if not pairs:
        return {"bias_kwh": None, "rmse_kwh": None}
    residuals = [p - a for p, a in pairs]
    return {
        "bias_kwh": round(sum(residuals) / len(residuals), 4),
        "rmse_kwh": round(math.sqrt(sum(r * r for r in residuals) / len(residuals)), 4),
    }


class RetrainEngine:
    """Hosts the retrain_from_history service implementation."""

    def __init__(self, coordinator) -> None:
        self.coordinator = coordinator

    async def retrain_from_history(
        self,
        days_back: int | None = None,
        reset_first: bool = False,
        experimental_cop_smear: bool = False,
        dry_run: bool = False,
    ) -> dict:
        """Retrain the learning model from existing hourly log data.

        Track A (daily_learning_mode=False): replays each logged hour through
        learn_from_historical_import(), honouring aux/base routing.

        Track B (daily_learning_mode=True): groups hours by day and applies the
        same midnight EMA logic as the live calibration. Thermal mass correction
        is applied when an indoor_temp_sensor is configured AND the hourly log
        contains 'indoor_temp' entries; otherwise it is skipped gracefully.

        ``dry_run=True`` runs the same replay into deep copies of the model
        and reports what it would change (``diff``, ``prediction_effect``)
        without writing or saving anything (see :meth:`_dry_run`).
        """
        coord = self.coordinator
        # The one await before the replay, which is synchronous
        # (:meth:`_replay`).  Returns without awaiting while Track B COP
        # smearing is disabled.
        cop_params = None
        if experimental_cop_smear and coord.daily_learning_mode:
            cop_params = await coord._fetch_track_b_cop_params(cache=not dry_run)

        entries = self._window_entries(days_back)
        if dry_run:
            return self._dry_run(entries, reset_first, experimental_cop_smear, cop_params)

        if reset_first:
            _LOGGER.info("retrain_from_history: Model reset before retraining.")
        result, cop_distributions = self._replay(
            entries, reset_first, experimental_cop_smear, cop_params,
        )
        result["dry_run"] = False
        if result["status"] != "completed":
            return result

        # Track B COP-smeared days keep their distribution for later replays.
        for date_str, distribution in cop_distributions.items():
            day_record = coord._daily_history.get(date_str)
            if isinstance(day_record, dict):
                day_record["track_b_cop_distribution"] = distribution
        if coord.daily_learning_mode:
            coord.data["learned_u_coefficient"] = coord._learned_u_coefficient
        await coord.storage.async_save_data(force=True)

        if result["mode"] == "strategy_dispatch":
            _LOGGER.info(
                f"retrain_from_history: Completed. "
                f"{result['days_processed']} days learned, "
                f"U-coefficient={coord._learned_u_coefficient}, "
                f"{result['solar_replay_updates']} NLMS updates."
            )
        else:
            _LOGGER.info(
                f"retrain_from_history Track A: Completed. "
                f"{result['learning_count']} entries learned, {result['skipped']} skipped, "
                f"{result['solar_replay_updates']} NLMS updates."
            )
        return result

    def _window_entries(self, days_back: int | None) -> list[dict]:
        """The logged hours a retrain replays, on the current inertia axis.

        Every path of the replay (Track A base, strategy dispatch, per-unit
        and solar replay) reads ``temp_key`` / ``inertia_temp`` from the
        entries, which were logged with the axis — and tau — in force at
        the time.
        """
        if days_back is not None:
            cutoff_str = (dt_util.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            entries = [e for e in self.coordinator._hourly_log if e.get("timestamp", "") >= cutoff_str]
        else:
            entries = list(self.coordinator._hourly_log)
        if entries:
            entries = self.coordinator._entries_on_current_axis(entries)
        return entries

    def _dry_run(
        self,
        entries: list[dict],
        reset_first: bool,
        experimental_cop_smear: bool,
        cop_params: dict | None,
    ) -> dict:
        """The retrain replayed into copies of the model; nothing written.

        Deep copies of the model state (``_RETRAIN_MODEL_STATE``) are put on
        the coordinator in place of the live objects, :meth:`_replay` runs
        on them — the same code, reading the same state through the same
        paths as a real retrain, ``reset_first`` included — and the live
        objects go back in ``finally``.  The live objects are set aside, not
        modified, and nothing can run in between: this method and
        :meth:`_replay` are synchronous, so no event-loop task (an hour
        boundary, a save) runs until the live model is back.  Executor
        threads can still read the coordinator meanwhile, as they can
        during a real retrain.

        Returns the replay's own result plus ``diff`` (live model vs the
        retrained copies) and ``prediction_effect`` (the most recent
        ``RETRAIN_DRY_RUN_PREDICTION_DAYS`` of the window predicted on both).
        """
        coord = self.coordinator
        prediction_entries = self._prediction_entries(entries)
        before = self._predict_hours(prediction_entries)

        live = _install_model_copies(coord)
        try:
            result, _cop_distributions = self._replay(
                entries, reset_first, experimental_cop_smear, cop_params,
            )
            if result["status"] == "completed":
                retrained = {attr: getattr(coord, attr) for attr in _RETRAIN_MODEL_STATE}
                result["diff"] = _model_diff(live["state"], retrained)
                result["prediction_effect"] = self._prediction_effect(
                    prediction_entries, before, self._predict_hours(prediction_entries),
                )
        finally:
            _reinstall_live_model(coord, live)

        result["dry_run"] = True
        _LOGGER.info(
            "retrain_from_history (dry run): %s entries replayed into a copy "
            "of the model (reset_first=%s); nothing written.",
            result.get("entries_processed", 0),
            reset_first,
        )
        return result

    @staticmethod
    def _prediction_entries(entries: list[dict]) -> list[dict]:
        """The window's most recent ``RETRAIN_DRY_RUN_PREDICTION_DAYS``."""
        cutoff = (
            dt_util.now() - timedelta(days=RETRAIN_DRY_RUN_PREDICTION_DAYS)
        ).strftime("%Y-%m-%d")
        return [e for e in entries if e.get("timestamp", "") >= cutoff]

    @staticmethod
    def _prediction_temp(entry: dict) -> float | None:
        """The hour's inertia temperature on the current axis, else its raw
        temperature."""
        temp = finite_float(entry.get("inertia_temp"))
        return temp if temp is not None else finite_float(entry.get("temp"))

    def _prediction_kwargs(self, entry: dict) -> dict | None:
        """``calculate_total_power`` inputs for one logged hour.

        The hour's inertia temperature on the current axis (as live
        predicts), logged wind, aux flag and sun: the 4D overrides under
        ``experimental_4d_primary`` when the hour carries DNI/DHI, else the
        logged effective 3D vector as the live boundary passes it (its
        potential is reconstructed at the current screen setting, as in
        every other log replay).  Every energy sensor gets a mode, the
        logged one or ``MODE_HEATING``: the logged map is sparse, and a
        unit missing from it would otherwise take its current mode.
        """
        coord = self.coordinator
        temp = self._prediction_temp(entry)
        if temp is None:
            return None
        logged_modes = entry.get("unit_modes") or {}
        kwargs: dict = {
            "temp": temp,
            "effective_wind": finite_float(entry.get("effective_wind")) or 0.0,
            "solar_impact": 0.0,
            "is_aux_active": bool(entry.get("auxiliary_active", False)),
            "unit_modes": {
                sid: logged_modes.get(sid, MODE_HEATING) for sid in coord.energy_sensors
            },
            "detailed": False,
        }
        dni = finite_float(entry.get("dni"))
        dhi = finite_float(entry.get("dhi"))
        instant = log_entry_instant(entry)
        if (
            getattr(coord, "experimental_4d_primary", False) is True
            and dni is not None and dhi is not None and instant is not None
        ):
            mid = instant + timedelta(minutes=30)
            elev, azim = coord.solar.get_approx_sun_pos(mid)
            correction = finite_float(entry.get("correction_percent"))
            kwargs.update(
                override_now=mid,
                override_dni_dhi=(dni, dhi) if elev > 0 else (0.0, 0.0),
                override_sun_pos=(elev, azim),
                override_correction_percent=correction if correction is not None else 100.0,
            )
        else:
            kwargs.update(
                override_solar_factor=finite_float(entry.get("solar_factor")) or 0.0,
                override_solar_vector=(
                    finite_float(entry.get("solar_vector_s")) or 0.0,
                    finite_float(entry.get("solar_vector_e")) or 0.0,
                    finite_float(entry.get("solar_vector_w")) or 0.0,
                ),
            )
        return kwargs

    def _predict_hours(self, entries: list[dict]) -> list[float | None]:
        """Each hour's modelled kWh on the model on the coordinator now."""
        predictions: list[float | None] = []
        for entry in entries:
            try:
                kwargs = self._prediction_kwargs(entry)
                result = (
                    self.coordinator.statistics.calculate_total_power(**kwargs)
                    if kwargs is not None else None
                )
                predictions.append(finite_float(result["total_kwh"]) if result else None)
            except Exception:  # noqa: BLE001 — one bad hour must not kill the report
                predictions.append(None)
        return predictions

    def _prediction_effect(
        self,
        entries: list[dict],
        before: list[float | None],
        after: list[float | None],
    ) -> dict:
        """Modelled kWh on the current and on the retrained model.

        Per day and per 5 °C band of the prediction temperature, set against
        what the model is trained to predict: each hour's metered energy
        minus the energy of units in OFF, DHW or guest mode, which live
        learning and both retrain tracks take off the same way (the model
        never predicts it; comparing with the raw meter reads that energy as
        a model shortfall).  That energy is reported beside it
        (``excluded_mode_kwh``).  An hour without a prediction or a metered
        kWh is left out (``hours_skipped``).  ``fit`` compares both models
        with it on the hours the global retrain learns from (not poisoned,
        no aux).  Both models have learned from these hours, so it says
        which one describes them better, not how either predicts unseen
        hours.
        """
        daily_mode = bool(self.coordinator.daily_learning_mode)

        def _group() -> dict:
            return {
                "hours": 0, "before": 0.0, "after": 0.0, "actual": 0.0,
                "excluded": 0.0, "fit_before": [], "fit_after": [],
            }

        total = _group()
        by_band: dict[str, dict] = {}
        by_day: dict[str, dict] = {}
        hours_skipped = 0
        for entry, b, a in zip(entries, before, after):
            metered = finite_float(entry.get("actual_kwh"))
            if b is None or a is None or metered is None:
                hours_skipped += 1
                continue
            excluded = DailyProcessor.compute_excluded_mode_energy([entry])
            actual = max(0.0, metered - excluded)
            fit = (
                not entry.get("auxiliary_active", False)
                and not _is_poisoned(entry, daily_mode=daily_mode)
            )
            groups = (
                total,
                by_band.setdefault(_temp_band(self._prediction_temp(entry)), _group()),
                by_day.setdefault(entry.get("timestamp", "")[:10], _group()),
            )
            for group in groups:
                group["hours"] += 1
                group["before"] += b
                group["after"] += a
                group["actual"] += actual
                group["excluded"] += excluded
                if fit:
                    group["fit_before"].append((b, actual))
                    group["fit_after"].append((a, actual))

        def _report(group: dict) -> dict:
            return {
                "hours": group["hours"],
                "modelled_kwh_before": round(group["before"], 3),
                "modelled_kwh_after": round(group["after"], 3),
                "actual_kwh": round(group["actual"], 3),
                "excluded_mode_kwh": round(group["excluded"], 3),
                "fit": {
                    "hours": len(group["fit_before"]),
                    "before": _fit(group["fit_before"]),
                    "after": _fit(group["fit_after"]),
                },
            }

        return {
            "days": RETRAIN_DRY_RUN_PREDICTION_DAYS,
            "hours_skipped": hours_skipped,
            **_report(total),
            "by_band": {
                band: _report(by_band[band])
                for band in sorted(by_band, key=_band_order)
            },
            "by_day": [
                {"date": day, **{k: v for k, v in _report(by_day[day]).items() if k != "fit"}}
                for day in sorted(by_day)
            ],
        }

    def _replay(
        self,
        entries: list[dict],
        reset_first: bool,
        experimental_cop_smear: bool,
        cop_params: dict | None,
    ) -> tuple[dict, dict]:
        """The retrain: reset, then replay ``entries`` into the model on the
        coordinator.  Writes the model state only — no save, no
        ``daily_history``, no ``data``.

        **Synchronous on purpose, and must stay so.**  A dry run swaps copies
        of the model onto the coordinator for the duration of this call
        (:meth:`_dry_run`); an ``await`` in here would let an hour boundary
        learn into the copies, and its hour would be lost when the live
        model goes back.  The only I/O the retrain needs (Track B COP
        params) is fetched before it.

        Returns the result and the Track B COP-smeared distributions by
        date, which a real retrain stores on ``daily_history``.
        """
        if reset_first:
            for attr in _RETRAIN_MODEL_MAPS:
                getattr(self.coordinator, attr).clear()
            self.coordinator._learned_u_coefficient = None

        cop_distributions: dict[str, list] = {}

        if not entries:
            return {"status": "no_data", "entries_processed": 0, "days_processed": 0, "learning_count": 0}, cop_distributions

        learning_count = 0

        if self.coordinator.daily_learning_mode:
            # Daily learning: batch aggregation per calendar day using strategy dispatch (#776).
            # Poisoned hours are excluded before grouping so the <22-hour guard
            # automatically rejects days with too many bad hours.
            daily_batches: dict[str, list] = {}
            # Every hour of each day, poisoned ones included: the Track C
            # re-smear builds its weights from the whole day, as the midnight
            # sync did (a missing hour would get zero weight).
            all_day_entries: dict[str, list] = {}
            for entry in entries:
                all_day_entries.setdefault(entry["timestamp"][:10], []).append(entry)
                if not _is_poisoned(entry, daily_mode=True):
                    daily_batches.setdefault(entry["timestamp"][:10], []).append(entry)
            # Track C days re-spread at the current settings (#1111).
            resmear_report = {
                "days_exact_cop": 0,
                "days_recovered_cop": 0,
                "days_stored": 0,
                "moved_share_mean": None,
                "moved_share_max": None,
            }
            moved_shares: list[float] = []

            # NLMS-replay 3-pass EM-lite (#847) for reset_first=True.
            # Only the Track B flat daily branch depends on the solar delta;
            # Track C and Track B COP-smear use pre-computed distributions
            # (synthetic_kwh_el) that are NOT biased by stored delta.  The
            # ``_b_flat_delta_source`` below controls how flat-daily
            # calculates retrain_solar_delta per pass:
            #   "stored"     — sum(e.solar_normalization_delta) from log
            #                  (current behaviour; use when reset_first=False)
            #   "none"       — 0 (priming pass; avoids reintroducing old
            #                  solar contamination before NLMS has re-run)
            #   "on_the_fly" — recompute from current solar coefficients
            #                  (refinement pass after NLMS replay)
            def _run_daily_pass(_b_flat_delta_source: str) -> tuple[int, int, int]:
                """One pass of the Track B/C daily-replay loop.

                ``_b_flat_delta_source`` in {"stored","none","on_the_fly"}
                selects which solar_normalization_delta is used for the
                Track B flat-daily path.  Track C and Track B COP-smear
                paths use pre-computed distributions and are unaffected by
                this parameter.
                Returns (days_processed, learning_count, days_skipped_mpc).
                """
                days_processed_local = 0
                learning_count_local = 0
                days_skipped_mpc_local = 0  # #855 Option B
                for date_str, day_entries in sorted(daily_batches.items()):
                    if len(day_entries) < 22:
                        _LOGGER.debug(f"retrain_from_history: Skipping {date_str} — {len(day_entries)}/24 hours")
                        continue

                    # #855 Option B: on Track C installs, a day without
                    # ``track_c_distribution`` is either an MPC outage OR
                    # pre-Track-C history.  Both cases are unsafe to process
                    # because the Track B fallback would write Track-B-semantic
                    # values (raw electrical ± thermal-mass correction) into
                    # the same correlation_data buckets that Track C fills
                    # with MPC-synthetic thermal-per-hour.  Skip entirely;
                    # users who want pre-Track-C days included can run
                    # retrain with a narrower ``days_back`` that excludes
                    # the pre-Track-C period, or temporarily disable Track C.
                    if self.coordinator.track_c_enabled:
                        if not self.coordinator._daily_history.get(date_str, {}).get("track_c_distribution"):
                            _LOGGER.warning(
                                "retrain: skipping %s — no track_c_distribution "
                                "(MPC outage or pre-Track-C history).",
                                date_str,
                            )
                            days_skipped_mpc_local += 1
                            continue

                    total_kwh = sum(e.get("actual_kwh", 0.0) for e in day_entries)
                    # Degree-days at the current BP, not the logged ``tdd``:
                    # that is fixed at the BP in force when each hour was
                    # logged, and retrain replays under today's settings.
                    bp = self.coordinator.balance_point
                    daily_tdd = 0.0
                    for e in day_entries:
                        t = finite_float(e.get("temp"))
                        if t is not None:
                            daily_tdd += abs(bp - t) / 24.0
                        else:
                            daily_tdd += finite_float(e.get("tdd")) or 0.0

                    if daily_tdd < 0.5 or total_kwh <= 0:
                        continue

                    # Mode filtering (#789): exclude OFF/DHW/Guest/Cooling from retrain.
                    excluded_kwh = self.coordinator._compute_excluded_mode_energy(day_entries)
                    total_kwh -= excluded_kwh

                    if total_kwh <= 0:
                        continue

                    # Determine q_adjusted for U-coefficient.
                    track_c_daily = self.coordinator._daily_history.get(date_str, {}).get("track_c_kwh")
                    if self.coordinator.track_c_enabled and track_c_daily is not None:
                        q_adjusted = track_c_daily
                        # Backward compat: days stored before non-MPC inclusion.
                        if self.coordinator.mpc_managed_sensor and "track_c_kwh_non_mpc" not in self.coordinator._daily_history.get(date_str, {}):
                            non_mpc_retrain = 0.0
                            for log_entry in day_entries:
                                breakdown = log_entry.get("unit_breakdown", {})
                                for sid in self.coordinator.energy_sensors:
                                    if sid != self.coordinator.mpc_managed_sensor:
                                        non_mpc_retrain += breakdown.get(sid, 0.0)
                            q_adjusted += non_mpc_retrain
                    else:
                        q_adjusted = total_kwh
                        if self.coordinator.thermal_mass_kwh_per_degree > 0.0:
                            from datetime import date as _date, timedelta as _td
                            prev_day_str = (
                                _date.fromisoformat(date_str) - _td(days=1)
                            ).isoformat()
                            end_temp = self.coordinator._daily_history.get(date_str, {}).get("midnight_indoor_temp")
                            start_temp = self.coordinator._daily_history.get(prev_day_str, {}).get("midnight_indoor_temp")
                            if end_temp is not None and start_temp is not None:
                                delta_t_indoor = end_temp - start_temp
                                q_adjusted = total_kwh - (self.coordinator.thermal_mass_kwh_per_degree * delta_t_indoor)

                    if q_adjusted <= 0:
                        continue

                    # U-coefficient update.
                    observed_u = q_adjusted / daily_tdd
                    if self.coordinator._learned_u_coefficient is None:
                        self.coordinator._learned_u_coefficient = observed_u
                    else:
                        self.coordinator._learned_u_coefficient += DEFAULT_DAILY_LEARNING_RATE * (
                            observed_u - self.coordinator._learned_u_coefficient
                        )

                    # Bucket learning: Track C uses strategy dispatch, Track B uses flat daily.
                    day_record = self.coordinator._daily_history.get(date_str, {})
                    track_c_dist = day_record.get("track_c_distribution")
                    if track_c_dist:
                        # The stored distribution is shaped by the balance
                        # point, inertia axis, wind thresholds and battery
                        # decay of its midnight; retrain applies it re-spread
                        # at the current ones (#1111).
                        track_c_dist, resmear_info = resmear_track_c_day(
                            self.coordinator,
                            all_day_entries.get(date_str, day_entries),
                            day_record,
                        )
                        status = resmear_info.get("status")
                        if status == "exact_cop":
                            resmear_report["days_exact_cop"] += 1
                        elif status == "recovered_cop":
                            resmear_report["days_recovered_cop"] += 1
                        else:
                            resmear_report["days_stored"] += 1
                        if "moved_share" in resmear_info:
                            moved_shares.append(resmear_info["moved_share"])
                        self.coordinator._apply_strategies_to_global_model(day_entries, track_c_dist)
                    else:
                        # Check for stored COP-smeared distribution, or generate
                        # on-the-fly if experimental_cop_smear is active (#793).
                        track_b_cop_dist = self.coordinator._daily_history.get(date_str, {}).get("track_b_cop_distribution")
                        if not track_b_cop_dist and experimental_cop_smear and cop_params is not None:
                            cop_dist = self.coordinator._track_b_cop_distribution(
                                day_entries, q_adjusted, date_str, cop_params,
                            )
                            if cop_dist is not None:
                                cop_smeared = self.coordinator._apply_strategies_to_global_model(
                                    day_entries, cop_dist,
                                )
                                cop_distributions[date_str] = cop_dist
                                _LOGGER.info(f"retrain COP-smear (#793): {date_str} -> {cop_smeared} bucket updates")
                                learning_count_local += 1
                                days_processed_local += 1
                                continue
                        if track_b_cop_dist:
                            self.coordinator._apply_strategies_to_global_model(day_entries, track_b_cop_dist)
                        else:
                            # Track B flattened daily bucket learning.
                            #
                            # Compute a DAY-LEVEL SNR weight: average the
                            # hour-level solar factors across the day and
                            # apply the same (FLOOR, K) mapping used on
                            # Track A.  Mostly-dark days retain full rate;
                            # sunny days down-weighted; all-shutdown zeroed.
                            # The target is raw ``q_adjusted / 24``.  The
                            # dark-equivalent bucket semantics are preserved
                            # (prediction consumers subtract solar impact
                            # from the base) because dark/overcast days
                            # dominate the weighted EMA.
                            avg_temp_retrain = sum(e.get("temp", 0.0) for e in day_entries) / len(day_entries)
                            daily_wind_retrain = sum(e.get("effective_wind", 0.0) for e in day_entries) / len(day_entries)
                            flat_temp_key = str(int(round(avg_temp_retrain)))
                            flat_wind_bucket = self.coordinator._get_wind_bucket(daily_wind_retrain)

                            avg_solar_factor = sum(
                                e.get("solar_factor", 0.0) for e in day_entries
                            ) / len(day_entries)
                            # Reuse the same shape as hour-level weight.
                            # Shutdown fraction at day level: count entries
                            # where ALL units shut down.  In practice this
                            # is rare at day granularity, but the check
                            # mirrors hour-level semantics.
                            n_all_shutdown = sum(
                                1 for e in day_entries
                                if len(e.get("solar_dominant_entities") or []) >= len(self.coordinator.energy_sensors)
                                and self.coordinator.energy_sensors
                            )
                            clean_fraction = (
                                (len(day_entries) - n_all_shutdown) / len(day_entries)
                                if day_entries else 1.0
                            )
                            day_weight = max(
                                SNR_WEIGHT_FLOOR,
                                1.0 - SNR_WEIGHT_K * max(0.0, avg_solar_factor),
                            ) * clean_fraction
                            q_hourly_avg = q_adjusted / 24.0
                            effective_rate = self.coordinator.learning_rate * day_weight

                            if flat_temp_key not in self.coordinator._correlation_data:
                                self.coordinator._correlation_data[flat_temp_key] = {}
                            current_pred = self.coordinator._correlation_data[flat_temp_key].get(flat_wind_bucket, 0.0)

                            if current_pred == 0.0:
                                # Seed from the raw hourly average.  A
                                # zero-weight day (all shutdown) is skipped
                                # to avoid seeding with actual ≈ 0.
                                if effective_rate > 0.0:
                                    self.coordinator._correlation_data[flat_temp_key][flat_wind_bucket] = round(q_hourly_avg, 5)
                            else:
                                new_pred = current_pred + effective_rate * (q_hourly_avg - current_pred)
                                self.coordinator._correlation_data[flat_temp_key][flat_wind_bucket] = round(new_pred, 5)

                    learning_count_local += 1
                    days_processed_local += 1
                return days_processed_local, learning_count_local, days_skipped_mpc_local

            # Single-pass: base learns with day-level SNR weighting, then
            # NLMS replay refreshes solar coefficients orthogonally.  The
            # ``_b_flat_delta_source`` argument no longer affects base
            # learning (SNR weighting ignores the delta) and is kept as
            # "none" for signature stability in the inner helper.
            days_processed, learning_count, days_skipped_mpc = _run_daily_pass("none")
            if moved_shares:
                resmear_report["moved_share_mean"] = round(
                    sum(moved_shares) / len(moved_shares), 4
                )
                resmear_report["moved_share_max"] = round(max(moved_shares), 4)
            # Per-unit models for DirectMeter sensors (needed for
            # isolate_sensor and attribution), in one hourly pass over every
            # entry: live per-unit learning runs hourly and does not depend
            # on the day filters above (a day short of hours, a Track C day
            # without a distribution), so neither does its replay.  The
            # replay picks the hours live learned from.  Before the solar
            # replay, which reads the per-unit base.
            per_unit_replay = self.coordinator._replay_per_unit_models(entries)
            solar_replay_diagnostics = self.coordinator.learning.replay_solar_nlms(
                entries,
                solar_calculator=self.coordinator.solar,
                screen_config=getattr(self.coordinator, "screen_config", None),
                correlation_data_per_unit=self.coordinator._correlation_data_per_unit,
                solar_coefficients_per_unit=self.coordinator._solar_coefficients_per_unit,
                learning_buffer_solar_per_unit=self.coordinator._learning_buffer_solar_per_unit,
                energy_sensors=self.coordinator.energy_sensors,
                learning_rate=self.coordinator.learning_rate,
                balance_point=self.coordinator.balance_point,
                aux_affected_entities=self.coordinator.aux_affected_entities,
                unit_strategies=self.coordinator._unit_strategies,
                daily_history=self.coordinator._daily_history,
                unit_min_base=self.coordinator._per_unit_min_base_thresholds or None,
                screen_affected_entities=_screen_affected_set_or_none(self.coordinator),
                solar_affected_entities=_solar_affected_set_or_none(self.coordinator),
                return_diagnostics=True,
                first_reported=_first_reported_or_none(self.coordinator),
            )
            solar_replay_updates = solar_replay_diagnostics.get("updates", 0)
            em_passes = 1

            return {
                "status": "completed",
                "mode": "strategy_dispatch",
                "entries_processed": len(entries),
                "days_processed": days_processed,
                "learning_count": learning_count,
                # #855 Option B: days skipped because Track C was enabled but
                # ``track_c_distribution`` was missing for the day (MPC outage
                # or pre-Track-C history).  Always 0 on non-Track-C installs.
                "days_skipped_mpc_unavailable": days_skipped_mpc,
                # Track C days re-spread at the current settings: how many
                # with the exact MPC COP, how many with the COP recovered
                # from the stored hours, how many kept as stored, and the
                # share of each day's energy that moved between hours.
                "track_c_resmear": resmear_report,
                "learned_u_coefficient": round(self.coordinator._learned_u_coefficient, 4) if self.coordinator._learned_u_coefficient is not None else None,
                "solar_replay_updates": solar_replay_updates,
                "solar_replay_diagnostics": solar_replay_diagnostics,
                "per_unit_replay": _replay_report(per_unit_replay),
                "em_passes": em_passes,
            }, cop_distributions

        else:
            # Track A: per-hour replay.
            #
            # When reset_first=True we run a three-pass EM-lite to break the
            # base ↔ solar circularity (#847 NLMS-replay fix):
            #   Pass 1  base priming        — replay with solar_norm = 0 to
            #                                 get shape without importing
            #                                 prior solar contamination via
            #                                 the stored delta
            #   Pass 1b per-unit replay     — populate correlation_data_per_unit
            #                                 so NLMS replay has unit-level
            #                                 base reference
            #   Pass 2  NLMS replay         — re-learn solar coefficients
            #                                 from raw historical vectors
            #                                 using the primed base
            #   Pass 3  base refinement     — clear base, replay with
            #                                 on-the-fly solar_norm computed
            #                                 from the re-learned coefficients
            #   Pass 3b per-unit replay     — final per-unit alignment
            #
            # When reset_first=False we keep the existing single-pass base
            # replay (uses stored delta) and append an NLMS replay at the
            # end so post-retrain solar coefficients are refined against
            # whatever base the user currently has.
            def _run_base_pass() -> tuple[int, int, list[dict]]:
                """Run the Track A base-replay pass over ``entries``.

                The base EMA is driven by the per-hour SNR weight; the
                aux path reads each entry's stored
                ``solar_normalization_delta`` to attribute aux reductions
                on hours where aux and sun overlapped.
                """
                skipped_local = 0
                processed_local: list[dict] = []
                learning_count_local = 0

                for entry in entries:
                    actual_kwh = entry.get("actual_kwh")
                    if actual_kwh is None:
                        skipped_local += 1
                        continue

                    # Mode filtering (#789 parity with live + Track B retrain):
                    # stored ``actual_kwh`` is total_energy_kwh (all units,
                    # all modes).  Live Track A learning uses
                    # ``learning_energy_kwh`` (OFF/DHW/Guest subtracted).
                    # Track B retrain has always subtracted via
                    # ``_compute_excluded_mode_energy``.  Track A retrain
                    # was the outlier — on hours where one unit was in
                    # DHW / OFF but others were heating, retrain inflated
                    # the base bucket by the excluded unit's kWh.  Fix:
                    # subtract excluded-mode energy per entry before
                    # feeding into learn_from_historical_import.
                    excluded_entry_kwh = 0.0
                    entry_unit_modes = entry.get("unit_modes", {}) or {}
                    entry_unit_breakdown = entry.get("unit_breakdown", {}) or {}
                    for _sid, _kwh in entry_unit_breakdown.items():
                        _mode = entry_unit_modes.get(_sid, MODE_HEATING)
                        if _mode in MODES_EXCLUDED_FROM_GLOBAL_LEARNING:
                            excluded_entry_kwh += _kwh
                    actual_kwh_filtered = max(0.0, actual_kwh - excluded_entry_kwh)

                    temp = entry.get("temp", 0.0)
                    # Re-bucketize from stored effective_wind against the
                    # current threshold rather than honoring the stored
                    # ``wind_bucket`` label.  The label is a cache from the
                    # threshold in effect at log-write time; honoring it
                    # makes retrain non-idempotent under threshold changes
                    # — e.g. raising ``extreme_wind_threshold`` would not
                    # empty extreme buckets on replay because labels stay
                    # stamped.  Matches the daily Track B path's live
                    # re-bucketization (retrain.py:260).
                    # Fall back to the stored label when ``effective_wind``
                    # is missing (very old logs / partial CSV imports) —
                    # otherwise legacy high/extreme samples would silently
                    # collapse to ``normal``.
                    eff_w = entry.get("effective_wind")
                    if isinstance(eff_w, (int, float)):
                        wind_bucket = self.coordinator._get_wind_bucket(eff_w)
                    else:
                        wind_bucket = entry.get("wind_bucket", "normal")
                    is_aux = entry.get("auxiliary_active", False)

                    if _is_poisoned(entry):
                        skipped_local += 1
                        continue

                    # The entry is already on the current inertia axis
                    # (``_window_entries``; poisoned hours still count as
                    # history there).  An entry it could not
                    # re-key keeps its logged key; without one, the raw temp.
                    temp_key_local = entry.get("temp_key")
                    if temp_key_local is None:
                        temp_key_local = str(int(round(temp)))

                    # The base EMA inside ``learn_from_historical_import``
                    # uses ``snr_weight`` to scale the step size.  The
                    # aux path still reads ``solar_normalization_delta``
                    # (pulled from the stored entry) so aux-active hours
                    # overlapping with sun are attributed correctly.
                    #
                    # #968: prefer the 4D delta whenever the entry carries
                    # it, falling back to 3D otherwise.  Independent of
                    # ``experimental_4d_primary`` — aggregation paths
                    # consume the best available signal regardless of the
                    # live read-path flag.
                    delta = entry.get(
                        "solar_normalization_delta_4d",
                        entry.get("solar_normalization_delta", 0.0),
                    )

                    # Active-units count proxy for retrain: use the
                    # entry's unit_breakdown (units with non-zero
                    # consumption this hour) instead of a per-unit
                    # base lookup, which would change pass-to-pass.
                    # This matches the spirit of count_active_learnable_units
                    # — a unit with no consumption isn't a signal-bearer
                    # this hour.  Mode filter still applies via entry's
                    # unit_modes.
                    breakdown_proxy = {
                        sid: float(kwh)
                        for sid, kwh in (entry.get("unit_breakdown") or {}).items()
                        if kwh and float(kwh) > 0.0
                    }
                    snr_w = compute_snr_weight(
                        entry.get("solar_factor", 0.0),
                        entry.get("solar_dominant_entities", []) or [],
                        total_units=count_active_learnable_units(
                            self.coordinator.energy_sensors,
                            entry.get("unit_modes", {}) or {},
                            breakdown_proxy,
                            min_base=0.0,  # presence > 0 in breakdown is enough
                        ),
                    )

                    status = self.coordinator.learning.learn_from_historical_import(
                        temp_key=temp_key_local,
                        wind_bucket=wind_bucket,
                        actual_kwh=actual_kwh_filtered,
                        is_aux_active=is_aux,
                        correlation_data=self.coordinator._correlation_data,
                        aux_coefficients=self.coordinator._aux_coefficients,
                        learning_rate=self.coordinator.learning_rate,
                        get_predicted_kwh_fn=self.coordinator._get_predicted_kwh,
                        actual_temp=temp,
                        solar_normalization_delta=delta,
                        snr_weight=snr_w,
                        solar_coefficients_per_unit=self.coordinator._solar_coefficients_per_unit,
                        energy_sensors=self.coordinator.energy_sensors,
                        unit_modes=entry.get("unit_modes"),
                    )
                    if "skipped" not in status:
                        learning_count_local += 1
                        processed_local.append(entry)
                    else:
                        skipped_local += 1
                return learning_count_local, skipped_local, processed_local

            solar_replay_updates = 0
            solar_replay_diagnostics: dict = {}
            skipped = 0
            learning_count = 0
            processed_entries: list[dict] = []

            # Single-pass Track A retrain + orthogonal NLMS replay.
            # Rationale: base learning uses the SNR-weighted per-hour rate;
            # dark hours dominate the weighted EMA so the resulting base
            # is solar-clean by construction.  NLMS replay then refreshes
            # solar coefficients against the final base.  The aux path
            # inside ``learn_from_historical_import`` still consumes the
            # stored ``solar_normalization_delta`` to attribute aux
            # reductions correctly on hours where aux and sun overlap.
            learning_count, skipped, processed_entries = _run_base_pass()
            # Per-unit models over every entry, not just the hours the
            # global pass learned: live per-unit learning also runs on
            # post-aux cooldown hours (units outside the aux scope), on
            # aux hours (aux coefficient or, outside the scope, base) and
            # on solar-saturated hours.  The replay picks the hours live
            # learned from.
            per_unit_replay = self.coordinator._replay_per_unit_models(entries)
            solar_replay_diagnostics = self.coordinator.learning.replay_solar_nlms(
                entries,
                solar_calculator=self.coordinator.solar,
                screen_config=getattr(self.coordinator, "screen_config", None),
                correlation_data_per_unit=self.coordinator._correlation_data_per_unit,
                solar_coefficients_per_unit=self.coordinator._solar_coefficients_per_unit,
                learning_buffer_solar_per_unit=self.coordinator._learning_buffer_solar_per_unit,
                energy_sensors=self.coordinator.energy_sensors,
                learning_rate=self.coordinator.learning_rate,
                balance_point=self.coordinator.balance_point,
                aux_affected_entities=self.coordinator.aux_affected_entities,
                unit_strategies=self.coordinator._unit_strategies,
                daily_history=self.coordinator._daily_history,
                unit_min_base=self.coordinator._per_unit_min_base_thresholds or None,
                screen_affected_entities=_screen_affected_set_or_none(self.coordinator),
                solar_affected_entities=_solar_affected_set_or_none(self.coordinator),
                return_diagnostics=True,
                first_reported=_first_reported_or_none(self.coordinator),
            )
            solar_replay_updates = solar_replay_diagnostics.get("updates", 0)
            em_passes = 1

            return {
                "status": "completed",
                "mode": "track_a_hourly",
                "entries_processed": len(entries),
                "days_processed": len({e["timestamp"][:10] for e in entries}),
                "learning_count": learning_count,
                "skipped": skipped,
                "solar_replay_updates": solar_replay_updates,
                "solar_replay_diagnostics": solar_replay_diagnostics,
                "per_unit_replay": _replay_report(per_unit_replay),
                "em_passes": em_passes,
            }, cop_distributions

    async def retrain_unit_from_history(
        self,
        entity_id: str,
        reset_first: bool = False,
        dry_run: bool = False,
        days_back: int | None = None,
    ) -> dict:
        """Targeted per-unit base-coefficient retrain from the hourly log.

        Scope: per-unit ``correlation_data_per_unit[entity_id]`` (and its
        observation counts) only.  Aux and solar per-unit state are
        untouched — those have dedicated services — so an aux-active hour
        of an aux-affected unit is not learned at all rather than going
        into the base bucket.  ``reset_first=False`` (default, nudge) leaves
        existing buckets in place and applies replay on top;
        ``reset_first=True`` wipes this entity's slice before replay.
        ``dry_run=True`` reports what would change without writing.

        Every logged hour in the window is handed to the replay, which
        learns the hours live per-unit learning learned: zero-kWh hours
        included, OFF / guest hours and hours the sensor did not report
        excluded (see ``LearningManager.replay_per_unit_models``).
        """
        coord = self.coordinator
        if entity_id not in coord.energy_sensors:
            return {
                "status": "unknown_entity",
                "entries_processed": 0,
                "buckets_modified": 0,
                "dry_run": bool(dry_run),
                "diff_summary": {},
            }

        if days_back is not None:
            cutoff_str = (dt_util.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            entries = [
                e for e in coord._hourly_log
                if e.get("timestamp", "") >= cutoff_str
            ]
        else:
            entries = list(coord._hourly_log)

        if not entries:
            return {
                "status": "no_data",
                "entries_processed": 0,
                "buckets_modified": 0,
                "dry_run": bool(dry_run),
                "diff_summary": {},
            }

        # Same inertia axis as live learning now (see retrain_from_history).
        entries = coord._entries_on_current_axis(entries)

        # ``reset_first`` is honoured inside ``replay_per_unit_models``
        # regardless of dry_run mode — the function clears the target
        # entity's slice on the deep-copies in dry-run and on live state
        # otherwise.  Previously the pop ran ONLY on live state, which
        # made dry-run + reset_first silently report nudge diffs.
        if reset_first and not dry_run:
            _LOGGER.info(
                "retrain_unit_from_history: clearing per-unit base state for %s",
                entity_id,
            )

        diagnostic = coord.learning.replay_per_unit_models(
            day_entries=entries,
            strategies=coord._unit_strategies,
            model=coord.get_model_state(),
            learning_rate=coord.learning_rate,
            target_entity=entity_id,
            dry_run=dry_run,
            reset_first=reset_first,
            replay_aux=False,
            **per_unit_replay_context(coord),
        ) or {}

        if not dry_run:
            await coord._async_save_data(force=True)

        _LOGGER.info(
            "retrain_unit_from_history: entity=%s reset_first=%s dry_run=%s "
            "entries=%d buckets_changed=%d days_back=%s",
            entity_id,
            reset_first,
            dry_run,
            diagnostic.get("entries_processed", 0),
            diagnostic.get("buckets_changed", 0),
            days_back,
        )

        return {
            "status": "ok",
            "entity_id": entity_id,
            "reset_first": bool(reset_first),
            "dry_run": bool(dry_run),
            "days_back": days_back,
            "entries_processed": diagnostic.get("entries_processed", 0),
            "buckets_modified": diagnostic.get("buckets_changed", 0),
            "diff_summary": diagnostic.get("diff_summary", {}),
            # Which hours the replay learned and why it skipped the rest.
            "replay": {
                k: v for k, v in diagnostic.items()
                if k not in ("diff_summary", "buckets_changed", "entries_processed")
            },
        }

