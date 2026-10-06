"""DailyProcessor — hosts the day-boundary processing extracted from coordinator.py.

Thin-delegate pattern: the processor holds a reference to the coordinator
and reaches back for state.  Public methods on the coordinator delegate
to this engine so the external API is unchanged.
"""
from __future__ import annotations

import logging
from datetime import datetime

from .const import (
    ATTR_TDD,
    DEFAULT_DAILY_LEARNING_RATE,
    MODES_EXCLUDED_FROM_GLOBAL_LEARNING,
    MODE_HEATING,
)
from .helpers import (
    REGIME_SPLIT_KEYS,
    aux_affected_entities_of,
    first_reported_instants,
    hour_slots,
    hour_start_utc,
    recorded_balance_points,
)
from .learning import unit_regime_energy
from .thermodynamics import ThermodynamicEngine

_LOGGER = logging.getLogger(__name__)


_COP_PARAM_KEYS = (
    "eta_carnot", "lwt", "f_defrost", "defrost_temp_threshold", "defrost_rh_threshold",
)


def _real_float(value) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def track_c_smear_record(coordinator, cop_params) -> dict:
    """What a Track C day was smeared at, stored as ``track_c_smear`` (#1111).

    The balance point, inertia tau, wind thresholds and solar battery decay
    behind the day's weights, and the MPC COP parameters behind its per-hour
    conversion.  With the parameters on the day, a retrain re-spreads it with
    the exact COP instead of the one recovered from the stored hours.
    Additive key on the ``daily_history`` day.
    """
    params = None
    if isinstance(cop_params, dict) and "eta_carnot" in cop_params:
        params = {
            key: float(cop_params[key]) for key in _COP_PARAM_KEYS
            if _real_float(cop_params.get(key)) is not None
        }
    return {
        "balance_point": _real_float(getattr(coordinator, "balance_point", None)),
        "inertia_tau": _real_float(getattr(coordinator, "inertia_tau", None)),
        "wind_threshold": _real_float(getattr(coordinator, "wind_threshold", None)),
        "extreme_wind_threshold": _real_float(
            getattr(coordinator, "extreme_wind_threshold", None)
        ),
        "solar_battery_decay": _real_float(getattr(coordinator, "solar_battery_decay", None)),
        "cop_params": params,
    }


def resmear_track_c_day(coordinator, day_logs: list[dict], day_record: dict) -> tuple[list | None, dict]:
    """The Track C distribution a retrain applies for one day (#1111).

    The stored ``track_c_distribution`` was smeared at the settings in force
    that midnight — balance point, inertia axis, wind thresholds, solar
    battery decay — and learning replayed under other settings must not
    reapply that shape.  This re-spreads the day's stored totals with weights
    from ``day_logs`` (the whole day, on the current inertia axis) at the
    coordinator's current settings, through the same smearing code as the
    midnight sync; see ``ThermodynamicEngine.resmear_distribution`` for the
    per-hour COP.  The stored distribution is not modified: it stays the
    record of what that midnight learned.

    Returns ``(distribution, info)``.  ``info["status"]`` is ``exact_cop`` /
    ``recovered_cop`` for a re-spread day, ``stored`` when the coordinator's
    settings are not readable (the stored distribution is returned
    unchanged), or ``no_distribution``.  ``info["moved_share"]`` is the share
    of the day's electrical energy that moved between hours.
    """
    from .thermodynamics import ThermodynamicEngine

    stored = day_record.get("track_c_distribution") if isinstance(day_record, dict) else None
    if not stored:
        return None, {"status": "no_distribution"}

    settings = (
        _real_float(getattr(coordinator, "balance_point", None)),
        _real_float(getattr(coordinator, "wind_threshold", None)),
        _real_float(getattr(coordinator, "extreme_wind_threshold", None)),
        _real_float(getattr(coordinator, "solar_battery_decay", None)),
    )
    if any(value is None for value in settings) or not isinstance(day_logs, list):
        return stored, {"status": "stored"}

    smear = day_record.get("track_c_smear")
    cop_params = smear.get("cop_params") if isinstance(smear, dict) else None
    weather = DailyProcessor(coordinator).track_c_weather(
        [d.get("datetime") for d in stored], day_logs,
    )
    engine = ThermodynamicEngine(balance_point=settings[0])
    distribution, method = engine.resmear_distribution(stored, weather, cop_params)

    old_total = sum(float(d.get("synthetic_kwh_el", 0.0) or 0.0) for d in stored)
    moved = sum(
        abs(float(new["synthetic_kwh_el"]) - float(old.get("synthetic_kwh_el", 0.0) or 0.0))
        for new, old in zip(distribution, stored)
    )
    moved_share = moved / (2.0 * old_total) if old_total > 0 else 0.0
    return distribution, {"status": method, "moved_share": moved_share}


def per_unit_replay_context(coordinator) -> dict:
    """Coordinator state ``LearningManager.replay_per_unit_models`` reads.

    What the replay needs to learn each logged hour the way live per-unit
    learning did: the sensors live learns hourly (and counts in the SNR
    weight), the wind thresholds, the balance point and the robust
    model lookup behind the expected unit base, the solar state behind the
    headroom multiplier, the per-unit min-base overrides behind the SNR
    unit count, the aux scope, and each unit's first report in the whole
    hourly log (the anchor of ``helpers.unit_hour_reported_kwh``'s legacy
    rule).

    Attributes that are not real values (test stubs, MagicMock
    coordinators) fall back to the conservative reading: no solar headroom,
    exact-bucket lookup, every entity aux-affected.
    """
    from .const import DEFAULT_BALANCE_POINT
    from .observation import hourly_learning_sensors
    from .solar import SolarCalculator
    from .statistics import StatisticsManager

    def _real_number(value, default):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return default

    statistics = getattr(coordinator, "statistics", None)
    solar = getattr(coordinator, "solar", None)
    screen_config = getattr(coordinator, "screen_config", None)
    screen_affected = getattr(coordinator, "_screen_affected_set", None)
    unit_min_base = getattr(coordinator, "_per_unit_min_base_thresholds", None)
    hourly_log = getattr(coordinator, "_hourly_log", None)
    solar_enabled = getattr(coordinator, "solar_enabled", False)
    energy_sensors = getattr(coordinator, "energy_sensors", None)
    strategies = getattr(coordinator, "_unit_strategies", None)
    daily_learning_mode = getattr(coordinator, "daily_learning_mode", None)
    hourly_sensors = (
        hourly_learning_sensors(energy_sensors, strategies, daily_learning_mode)
        if isinstance(energy_sensors, list)
        and isinstance(strategies, dict)
        and isinstance(daily_learning_mode, bool)
        else None
    )
    return {
        "hourly_sensors": hourly_sensors,
        "wind_threshold": _real_number(getattr(coordinator, "wind_threshold", None), None),
        "extreme_wind_threshold": _real_number(
            getattr(coordinator, "extreme_wind_threshold", None), None
        ),
        "balance_point": _real_number(
            getattr(coordinator, "balance_point", None), DEFAULT_BALANCE_POINT
        ),
        "get_prediction_from_model": (
            statistics._get_prediction_from_model
            if isinstance(statistics, StatisticsManager)
            else None
        ),
        "solar_calculator": solar if isinstance(solar, SolarCalculator) else None,
        "solar_enabled": isinstance(solar_enabled, (bool, int)) and bool(solar_enabled),
        "screen_config": screen_config if isinstance(screen_config, tuple) else None,
        "screen_affected_entities": (
            screen_affected if isinstance(screen_affected, (set, frozenset)) else None
        ),
        "unit_min_base": unit_min_base if isinstance(unit_min_base, dict) and unit_min_base else None,
        "aux_affected_entities": aux_affected_entities_of(coordinator),
        "first_reported": first_reported_instants(
            hourly_log if isinstance(hourly_log, list) else []
        ),
    }


class DailyProcessor:
    """Day-boundary processing engine.

    Hosts the midnight aggregation pipeline: log aggregation, Track B
    daily-flat learning, Track C thermodynamic midnight-sync, and the
    per-unit replay triggered at day close.  All state lives on the
    coordinator.
    """

    def __init__(self, coordinator) -> None:
        self.coordinator = coordinator

    def aggregate_logs(self, day_logs: list[dict]) -> dict:
        """Aggregate hourly logs into a daily summary."""
        if not day_logs:
            return {}

        total_kwh = sum(e.get("actual_kwh", 0.0) for e in day_logs)
        expected_kwh = sum(e.get("expected_kwh", 0.0) for e in day_logs)
        forecasted_kwh = sum(e.get("forecasted_kwh", 0.0) for e in day_logs)
        solar_impact = sum(
            self.coordinator.hourly_solar_impact_kwh(e) for e in day_logs
        )
        aux_impact = sum(e.get("aux_impact_kwh", 0.0) for e in day_logs)
        guest_impact = sum(e.get("guest_impact_kwh", 0.0) for e in day_logs)
        # #968: prefer 4D delta per-entry when present, falling back to 3D.
        # Independent of ``experimental_4d_primary`` — aggregation paths
        # consume the best available signal regardless of the live read-path
        # flag.  Mixed-hour days produce a hybrid sum by construction.
        solar_norm_delta_total = sum(
            e.get("solar_normalization_delta_4d", e.get("solar_normalization_delta", 0.0))
            for e in day_logs
        )

        # #982: 4D-pipeline daily aggregates.  These fields are only present
        # on hourly_log entries where the 4D shadow/live learner actually
        # fired (key-presence filter, NOT ``.get(..., 0.0)``).  Treating
        # missing entries as zero would conflate "no 4D signal this hour"
        # with "4D signal summed to zero" and bias daily totals downward.
        # When no day_logs carry 4D data: aggregate stays None — downstream
        # consumers branch on that (future Track B/C 4D-retrain).
        delta_4d_present = [
            e["solar_normalization_delta_4d"]
            for e in day_logs
            if "solar_normalization_delta_4d" in e
        ]
        solar_norm_delta_4d_total = (
            round(sum(delta_4d_present), 5) if delta_4d_present else None
        )
        impact_4d_present = [
            e["solar_impact_4d_kwh"]
            for e in day_logs
            if "solar_impact_4d_kwh" in e
        ]
        solar_impact_4d_kwh_total = (
            round(sum(impact_4d_present), 2) if impact_4d_present else None
        )

        # Sum thermodynamic gross values
        thermodynamic_gross_from_logs = sum(e.get("thermodynamic_gross_kwh", 0.0) for e in day_logs)

        # Check if ALL logs have the field
        has_complete_data = all("thermodynamic_gross_kwh" in e for e in day_logs)

        if has_complete_data:
            thermodynamic_gross_kwh = thermodynamic_gross_from_logs
        else:
            # Fallback for legacy/mixed data: Reconstruct hour-by-hour to handle straddling days
            # (e.g., heating at night, cooling during day)
            reconstructed_sum = 0.0
            for e in day_logs:
                # Per-hour reconstruction
                act = e.get("actual_kwh", 0.0)
                aux = e.get("aux_impact_kwh", 0.0)
                sol = self.coordinator.hourly_solar_impact_kwh(e)
                temp = e.get("temp", 0.0)

                # Mode-aware solar correction
                if temp >= self.coordinator.balance_point:
                    # Cooling: Solar ADDS load (Gross = Actual - Solar)
                    # (Wait, if solar adds load, actual is higher. So Base = Actual - Solar)
                    # Correct.
                    reconstructed_sum += (act + aux - sol)
                else:
                    # Heating: Solar REDUCES load (Gross = Actual + Solar)
                    reconstructed_sum += (act + aux + sol)

            thermodynamic_gross_kwh = reconstructed_sum

        # Breakdown sums
        unit_breakdown = {}
        unit_expected = {}

        # Thermal-regime energy split (#1051).  Accumulated per hour against
        # that hour's own modes, so a day that switches mode part-way through
        # attributes each hour correctly instead of stamping the whole day
        # with its end-of-day state.
        #
        # The split is persisted rather than the label: a future change to
        # THERMAL_REGIME_DOMINANCE_SHARE then reclassifies history instead of
        # leaving stale labels behind.  Absent keys mean "predates recording"
        # and must stay distinguishable from a recorded 0/0, which is a real
        # "idle" classification — see coordinator.thermal_regime_for_day.
        regime_heating_kwh = 0.0
        regime_cooling_kwh = 0.0
        # The same split per unit (the heat-source classifier's input).
        unit_heating_kwh: dict[str, float] = {}
        unit_cooling_kwh: dict[str, float] = {}
        # The split is only written when every hour carries a per-unit
        # breakdown.  Hourly rows imported from CSV have none, and summing
        # them would record 0 / 0 on a day with real consumption — an
        # "idle" day instead of a day without evidence.
        regime_split_known = all("unit_breakdown" in e for e in day_logs)

        for e in day_logs:
            for uid, val in e.get("unit_breakdown", {}).items():
                unit_breakdown[uid] = unit_breakdown.get(uid, 0.0) + val
            for uid, val in e.get("unit_expected_breakdown", {}).items():
                unit_expected[uid] = unit_expected.get(uid, 0.0) + val

            hour_units = unit_regime_energy(
                e.get("unit_modes", {}) or {},
                e.get("unit_breakdown", {}) or {},
            )
            for uid, (heating, cooling) in hour_units.items():
                regime_heating_kwh += heating
                regime_cooling_kwh += cooling
                if heating > 0.0:
                    unit_heating_kwh[uid] = unit_heating_kwh.get(uid, 0.0) + heating
                if cooling > 0.0:
                    unit_cooling_kwh[uid] = unit_cooling_kwh.get(uid, 0.0) + cooling

        # Averages
        avg_temp = sum(e["temp"] for e in day_logs) / len(day_logs)
        avg_wind = sum(e.get("effective_wind", 0.0) for e in day_logs) / len(day_logs)
        avg_solar = sum(e.get("solar_factor", 0.0) for e in day_logs) / len(day_logs)

        # TDD (Sum of hourly TDD)
        total_tdd = sum(e.get("tdd", 0.0) for e in day_logs)
        # The BP that sum is expressed at, so readers can tell whether the
        # stored tdd still applies: None when the hours disagree (the sum
        # applies at no single BP).  When no hour records one the key is
        # left out, so backfill keeps whatever the v10 migration stamped.
        tdd_bps = recorded_balance_points(day_logs)

        # Hourly Vectors (Kelvin Protocol: Data Aggregation)
        hourly_vectors = {
            "temp": [None] * 24,
            "wind": [None] * 24,
            "tdd": [None] * 24,
            "actual_kwh": [None] * 24,
            # Log entries (clock hours) folded into each local-hour slot:
            # 2 in the repeated DST fall-back hour, where temp/wind are the
            # mean and energy/tdd the sum of both.  Readers that rebuild the
            # day from the vectors weight each slot by it
            # (helpers.vector_slot_hours); days stored before it existed
            # count one hour per filled slot.
            "hours": [None] * 24,
        }
        if self.coordinator.solar_enabled:
            hourly_vectors["solar_rad"] = [None] * 24
            hourly_vectors["solar_norm_delta"] = [None] * 24
            # #982: 4D inputs + delta per hour for future Track B/C retraining
            # against 4D coefficients.  Persist in daily_history because
            # hourly_log trims after the retention window (90/180/365 days)
            # while daily_history is unbounded.
            hourly_vectors["solar_norm_delta_4d"] = [None] * 24
            hourly_vectors["dni"] = [None] * 24
            hourly_vectors["dhi"] = [None] * 24

        # Hour Collision Fix: Aggregate instead of overwrite
        # Iterate over hour slots (0-23) and aggregate all entries for that hour.
        # DST Handling:
        # - Spring Forward (23h): One hour slot will remain None (handled downstream).
        # - Fall Back (25h): the repeated hour is logged as an entry of its
        #   own, so slot 2 holds two entries.  They are aggregated here and
        #   the slot's ``hours`` records that it covers two clock hours.
        for hour in range(24):
            hour_entries = [e for e in day_logs if e.get("hour") == hour]
            if not hour_entries:
                continue

            count = len(hour_entries)

            # Average State Values
            hourly_avg_temp = sum(e["temp"] for e in hour_entries) / count
            hourly_avg_wind = sum(e.get("effective_wind", 0.0) for e in hour_entries) / count
            hourly_avg_solar = sum(e.get("solar_factor", 0.0) for e in hour_entries) / count

            # Sum Accumulated Values
            sum_load = sum(e.get("actual_kwh", 0.0) for e in hour_entries)
            sum_tdd = sum(e.get("tdd", 0.0) for e in hour_entries)

            hourly_vectors["hours"][hour] = count
            hourly_vectors["temp"][hour] = hourly_avg_temp
            hourly_vectors["wind"][hour] = hourly_avg_wind
            hourly_vectors["tdd"][hour] = sum_tdd # Sum of TDD contributions
            hourly_vectors["actual_kwh"][hour] = sum_load
            if self.coordinator.solar_enabled:
                hourly_vectors["solar_rad"][hour] = hourly_avg_solar
                # Sum (not average) — delta is an energy correction, not a rate
                hourly_vectors["solar_norm_delta"][hour] = sum(
                    e.get("solar_normalization_delta", 0.0) for e in hour_entries
                )
                # #982: 4D fields use key-presence filtering, NOT .get(..., 0.0)
                # — an hour where 4D didn't fire (key absent) must not drag the
                # average toward 0 (DNI/DHI) or contribute a phantom 0 to the
                # sum (delta).  None signals "no data" to downstream consumers.
                dni_present = [e["dni"] for e in hour_entries if "dni" in e]
                if dni_present:
                    hourly_vectors["dni"][hour] = round(
                        sum(dni_present) / len(dni_present), 2
                    )
                dhi_present = [e["dhi"] for e in hour_entries if "dhi" in e]
                if dhi_present:
                    hourly_vectors["dhi"][hour] = round(
                        sum(dhi_present) / len(dhi_present), 2
                    )
                delta_4d_present_hour = [
                    e["solar_normalization_delta_4d"]
                    for e in hour_entries
                    if "solar_normalization_delta_4d" in e
                ]
                if delta_4d_present_hour:
                    hourly_vectors["solar_norm_delta_4d"][hour] = round(
                        sum(delta_4d_present_hour), 5
                    )

        # Provenance (Last one wins)
        last_entry = day_logs[-1]
        primary = last_entry.get("primary_entity")
        secondary = last_entry.get("secondary_entity")
        crossover = last_entry.get("crossover_day")

        return {
            "kwh": round(total_kwh, 2),
            "expected_kwh": round(expected_kwh, 2),
            "forecasted_kwh": round(forecasted_kwh, 2),
            "aux_impact_kwh": round(aux_impact, 2),
            "solar_impact_kwh": round(solar_impact, 2),
            "guest_impact_kwh": round(guest_impact, 2),
            "solar_normalization_delta": round(solar_norm_delta_total, 5),
            "solar_normalization_delta_4d": solar_norm_delta_4d_total,
            "solar_impact_4d_kwh": solar_impact_4d_kwh_total,
            "thermodynamic_gross_kwh": round(thermodynamic_gross_kwh, 2),
            "tdd": round(total_tdd, 1),
            **(
                {"balance_point": next(iter(tdd_bps)) if len(tdd_bps) == 1 else None}
                if tdd_bps
                else {}
            ),
            "temp": round(avg_temp, 1),
            "wind": round(avg_wind, 1),
            "solar_factor": round(avg_solar, 3),
            "unit_breakdown": {k: round(v, 3) for k, v in unit_breakdown.items()},
            "unit_expected_breakdown": {k: round(v, 3) for k, v in unit_expected.items()},
            **(
                {
                    "regime_heating_kwh": round(regime_heating_kwh, 3),
                    "regime_cooling_kwh": round(regime_cooling_kwh, 3),
                    "unit_heating_kwh": {k: round(v, 3) for k, v in unit_heating_kwh.items()},
                    "unit_cooling_kwh": {k: round(v, 3) for k, v in unit_cooling_kwh.items()},
                }
                if regime_split_known
                else {}
            ),
            "primary_entity": primary,
            "secondary_entity": secondary,
            "crossover_day": crossover,
            "deviation": round(total_kwh - expected_kwh, 2),
            "hourly_vectors": hourly_vectors,
        }

    def backfill_from_hourly(self) -> int:
        """Backfill missing details in daily history from hourly logs."""
        if not self.coordinator._hourly_log:
            return 0

        # Group logs by date
        logs_by_date = {}
        for entry in self.coordinator._hourly_log:
            date_key = entry["timestamp"][:10]
            if date_key not in logs_by_date:
                logs_by_date[date_key] = []
            logs_by_date[date_key].append(entry)

        updated_count = 0

        for date_key, logs in logs_by_date.items():
            # Aggregate stats from logs
            agg = self.aggregate_logs(logs)

            if date_key not in self.coordinator._daily_history:
                # If we have enough logs (e.g. > 12h) we could create it,
                # but let's be safe and only enrich existing or create if > 20h
                if len(logs) >= 20:
                     self.coordinator._daily_history[date_key] = agg
                     updated_count += 1
            else:
                curr = self.coordinator._daily_history[date_key]
                hist_kwh = curr.get("kwh", 0.0)
                log_kwh = agg["kwh"]

                # Validity Check:
                # If aggregated log kWh is significantly less than history kWh,
                # the logs are likely partial (pruned). In this case, we DO NOT overwrite
                # the main stats (kwh, tdd, temp) but we CAN populate the breakdown fields
                # if they are missing, though they will be partial.
                # It's better to leave them missing than to store partial breakdowns that don't sum to Total.
                # However, if values match (within margin), we assume logs are complete and overwrite to enrich.

                # Margin: 5% or 1 kWh
                diff = abs(log_kwh - hist_kwh)
                threshold = max(1.0, hist_kwh * 0.05)

                if diff > threshold and hist_kwh > log_kwh:
                    # Logs are partial (pruned). Skip backfill for this day.
                    # We assume daily history is the source of truth for totals.
                    continue

                # Logs are complete (or match history). Enrich daily history.
                # We overwrite to ensure consistency (Sum of Parts == Whole)
                self.coordinator._daily_history[date_key].update(agg)
                if "regime_heating_kwh" not in agg:
                    # The logs cannot support a split (imported rows without
                    # a per-unit breakdown); drop one stored earlier rather
                    # than leave it describing different data.
                    for key in REGIME_SPLIT_KEYS:
                        curr.pop(key, None)
                updated_count += 1

        if updated_count > 0:
            _LOGGER.info(f"Backfilled/Enriched {updated_count} daily history entries from hourly logs.")

        return updated_count

    async def fetch_mpc_buffer_and_cop(self) -> tuple[list, dict | None] | None:
        """Fetch MPC hourly buffer and COP params.  Shared by live midnight
        sync and pre-midnight snapshot polling.

        Returns (mpc_records, cop_params) on success, or None on any failure
        (service missing, empty buffer, malformed response).  Never raises.
        """
        from homeassistant.exceptions import ServiceNotFound, HomeAssistantError

        service_data = {}
        if self.coordinator.mpc_entry_id:
            service_data["entry_id"] = self.coordinator.mpc_entry_id

        try:
            response = await self.coordinator.hass.services.async_call(
                "heatpump_mpc",
                "get_sh_hourly",
                service_data,
                blocking=True,
                return_response=True,
            )
        except ServiceNotFound:
            _LOGGER.debug("Track C fetch: heatpump_mpc.get_sh_hourly service not found.")
            return None
        except HomeAssistantError as err:
            _LOGGER.debug("Track C fetch: MPC service call failed (%s).", err)
            return None

        if isinstance(response, dict):
            mpc_records = response.get("buffer", response.get("data", response.get("hourly", [])))
        elif isinstance(response, list):
            mpc_records = response
        else:
            _LOGGER.debug("Track C fetch: Unexpected MPC response format (%s).", type(response))
            return None

        if not mpc_records:
            return None

        cop_params = None
        try:
            cop_response = await self.coordinator.hass.services.async_call(
                "heatpump_mpc",
                "get_cop_params",
                service_data,
                blocking=True,
                return_response=True,
            )
            if isinstance(cop_response, dict) and "eta_carnot" in cop_response:
                cop_params = cop_response
                self.coordinator._last_cop_params = cop_params  # Cache for Track B COP smearing (#793)
        except (ServiceNotFound, HomeAssistantError):
            pass

        return mpc_records, cop_params

    async def maybe_snapshot_track_c(self, current_time) -> None:
        """Take a Track C MPC snapshot if we've entered a trigger slot.

        Trigger slots: 22:00 hour, 23:00-23:54 hour, 23:55+ minute.  Each
        slot snapshots at most once per day — subsequent ticks in the same
        slot skip.  A snapshot overwrites any earlier one: the latest is
        always the freshest.  If the live call fails, the previous
        snapshot (if any) is preserved.
        """
        today_key = current_time.date().isoformat()
        hour = current_time.hour
        minute = current_time.minute

        slot_key: str | None = None
        if hour == 22:
            slot_key = f"{today_key}:2200"
        elif hour == 23 and minute < 55:
            slot_key = f"{today_key}:2300"
        elif hour == 23 and minute >= 55:
            slot_key = f"{today_key}:2355"

        if slot_key is None or self.coordinator._track_c_last_snapshot_slot == slot_key:
            return

        fetched = await self.fetch_mpc_buffer_and_cop()
        self.coordinator._track_c_last_snapshot_slot = slot_key
        if fetched is None:
            _LOGGER.debug(
                "Track C snapshot at %s failed — previous snapshot (if any) preserved.",
                current_time.strftime("%H:%M"),
            )
            return

        mpc_records, cop_params = fetched
        self.coordinator._track_c_snapshot = {
            "date": today_key,
            "captured_at": current_time.isoformat(),
            "slot": slot_key.split(":")[-1],
            "mpc_records": mpc_records,
            "cop_params": cop_params,
        }
        _LOGGER.info(
            "Track C snapshot captured at %s (%d records) — fallback ready for midnight sync.",
            current_time.strftime("%H:%M"), len(mpc_records),
        )

    async def run_track_c_midnight_sync(
        self, day_logs: list[dict], date_key: str
    ) -> tuple[float, list, str, dict] | None:
        """Fetch MPC thermal data and run the ThermodynamicEngine Midnight Sync.

        Returns (total_synthetic_el, distribution, source, smear) where:
          - total_synthetic_el: sum of synthetic_kwh_el across all 24 hours —
            the weather-smeared electrical equivalent used as q_adjusted in learning.
          - distribution: the full list of HourlyDistribution dicts for storage
            (enables future per-hour visualisation without recomputing).
          - source: "live" or "snapshot_<HHMM>" — identifies the data origin
            for daily_history tagging and diagnostics.
          - smear: the settings the day was smeared at and the MPC COP
            parameters (``track_c_smear_record``), stored beside the
            distribution so a retrain can re-spread it exactly.
        Returns None if the sync cannot proceed (live call failed AND no
        matching snapshot available).  Triggers Option B skip at the caller.
        """
        mpc_records = None
        cop_params = None
        source = "live"

        fetched = await self.fetch_mpc_buffer_and_cop()
        if fetched is not None:
            mpc_records, cop_params = fetched

        if mpc_records is None and self.coordinator._track_c_snapshot is not None:
            snap = self.coordinator._track_c_snapshot
            if snap.get("date") == date_key:
                mpc_records = snap["mpc_records"]
                cop_params = snap["cop_params"]
                source = f"snapshot_{snap['slot']}"
                _LOGGER.info(
                    "Track C: live MPC unavailable for %s — using snapshot captured at %s.",
                    date_key, snap["captured_at"],
                )

        if mpc_records is None:
            _LOGGER.warning(
                "Track C: no live MPC response and no usable snapshot for %s — "
                "skipping learning (Option B).",
                date_key,
            )
            return None

        if cop_params is not None:
            _LOGGER.debug(
                "Track C: COP params in use — η=%.3f, f_defrost=%.2f, LWT=%.1f (source=%s)",
                cop_params["eta_carnot"], cop_params.get("f_defrost", 0.85),
                cop_params.get("lwt", 35.0), source,
            )
        else:
            _LOGGER.info("Track C: no COP params — using daily avg COP fallback (source=%s).", source)

        # --- Filter MPC records to the target day ---
        # The MPC buffer holds up to 48 hours of rolling data.  We must select
        # only records whose date matches date_key to avoid inflating the
        # synthetic baseline with thermal production from adjacent days.
        from homeassistant.util import dt as _dt

        filtered_records = []
        for rec in mpc_records:
            try:
                rec_dt = _dt.parse_datetime(rec["datetime"])
                if rec_dt is not None and rec_dt.date().isoformat() == date_key:
                    filtered_records.append(rec)
            except (KeyError, TypeError, ValueError):
                continue

        if len(filtered_records) < 18:
            _LOGGER.warning(
                "Track C: Only %d/%d MPC records matched target day %s (need ≥18) — falling back to Track B.",
                len(filtered_records), len(mpc_records), date_key,
            )
            return None

        mpc_records = filtered_records

        weather_data = self.track_c_weather(
            [record["datetime"] for record in mpc_records], day_logs,
        )

        engine = ThermodynamicEngine(balance_point=self.coordinator.balance_point)
        try:
            distribution = engine.calculate_synthetic_baseline(mpc_records, weather_data, cop_params=cop_params)
        except (TypeError, KeyError, ValueError) as err:
            _LOGGER.error("Track C: ThermodynamicEngine failed (%s) — falling back to Track B.", err)
            return None

        total_synthetic_el = sum(h["synthetic_kwh_el"] for h in distribution)
        _LOGGER.info(
            "Track C Midnight Sync %s: total_synthetic_el=%.3f kWh from %d MPC records (source=%s).",
            date_key, total_synthetic_el, len(mpc_records), source,
        )

        # Clear the snapshot after successful consumption so it doesn't leak
        # into later days.  Any earlier snapshot from today is still eligible
        # — the day boundary itself is what invalidates yesterday's snapshot
        # via the ``date`` equality check above.
        if source.startswith("snapshot_"):
            self.coordinator._track_c_snapshot = None

        return (
            total_synthetic_el,
            distribution,
            source,
            track_c_smear_record(self.coordinator, cop_params),
        )

    def track_c_weather(self, datetimes: list[str], day_logs: list[dict]) -> list[dict]:
        """One WeatherData row per Track C hour, from the day's hourly log.

        The inputs of the smearing weights, at the coordinator's current
        balance point, wind thresholds and solar battery decay.  Used by the
        midnight sync and by the retrain re-smear (``resmear_track_c_day``),
        which passes entries on the current inertia axis.

        - delta_t = |balance_point − inertia_temp| (inertia-weighted temp
          mirrors Track A's model; falls back to raw temp if inertia_temp is
          not logged).
        - wind_factor = 3-bucket multiplier matching Track A wind buckets
          (1.0 / 1.3 / 1.6).
        - solar_factor = 1.0 − solar residual (inverted; 0 = no sun → full
          loss weight), with the solar battery decay applied so afternoon
          solar gain carries into evening hours.
        """
        from homeassistant.util import dt as _dt

        # Records join the log on the hour's UTC start instant, not the local
        # hour number: on the DST fall-back day both passes through hour 2
        # have a record and a log entry of their own.
        slots = hour_slots(day_logs)
        log_by_key = dict(slots)

        # Solar battery pre-pass: accumulate decay across hours so that afternoon
        # solar gain reduces evening loss weights, matching Track A's solar battery model.
        solar_residual_by_key = self._solar_residuals(slots)

        balance_point = self.coordinator.balance_point
        weather_data = []
        for dt_str in datetimes:
            try:
                record_dt = _dt.parse_datetime(dt_str)
                key = hour_start_utc(record_dt) if record_dt else None
            except (KeyError, TypeError, ValueError):
                key = None

            log_entry = log_by_key.get(key, {})
            # Fix 1: use inertia_temp (thermal-mass-weighted) rather than instantaneous
            # outdoor temp — consistent with how Track A models heat demand.
            # Use explicit None-check: dict.get(key, default) silently returns None
            # when the key exists with a None value (e.g. early startup entries).
            inertia_t = log_entry.get("inertia_temp")
            raw_t = log_entry.get("temp")
            outdoor_temp = (
                inertia_t if inertia_t is not None
                else raw_t if raw_t is not None
                else balance_point
            )
            eff_wind = log_entry.get("effective_wind")
            effective_wind: float = eff_wind if eff_wind is not None else 0.0

            # Fix 2: 3-bucket wind multiplier — mirrors Track A's discrete wind buckets
            # (normal / high / extreme) rather than an unbounded linear scale.
            if effective_wind >= self.coordinator.extreme_wind_threshold:
                wind_factor = 1.6
            elif effective_wind >= self.coordinator.wind_threshold:
                wind_factor = 1.3
            else:
                wind_factor = 1.0

            # Fix 3: solar factor with battery decay residual — evening hours after a
            # sunny afternoon still carry a non-zero solar offset, preventing the smearing
            # from over-weighting post-sunset hours (same as Track A's solar battery).
            solar_with_decay = solar_residual_by_key.get(key, 0.0)
            solar_factor = max(0.0, 1.0 - solar_with_decay)

            # Raw outdoor temp and humidity for per-hour COP calculation.
            # Use raw_t (not inertia) for COP — COP depends on instantaneous
            # air temperature at the evaporator, not thermally weighted.
            raw_outdoor = raw_t if raw_t is not None else balance_point
            rh = log_entry.get("humidity")
            rh = rh if rh is not None else 50.0

            weather_data.append({
                "datetime": dt_str,
                "delta_t": abs(balance_point - outdoor_temp),
                "is_cooling": outdoor_temp > balance_point,
                "wind_factor": wind_factor,
                "solar_factor": solar_factor,
                "outdoor_temp": raw_outdoor,
                "humidity": rh,
            })
        return weather_data

    def _solar_residuals(self, slots: list[tuple[object, dict]]) -> dict:
        """Solar battery residual per slot, keyed like ``helpers.hour_slots``.

        EMA of ``solar_factor`` across the day in time order.  An hour
        missing from the log is a step with no sun: the battery decays once
        for it, as it did when this walked a fixed 0–23 hour range.
        """
        decay = self.coordinator.solar_battery_decay
        battery = 0.0
        previous = None
        residuals: dict = {}
        for key, entry in slots:
            if isinstance(key, datetime) and isinstance(previous, datetime):
                missing = round((key - previous).total_seconds() / 3600.0) - 1
                if missing > 0:
                    battery *= decay ** missing
            raw_solar = entry.get("solar_factor")
            raw_solar = raw_solar if raw_solar is not None else 0.0
            battery = battery * decay + raw_solar * (1 - decay)
            residuals[key] = min(1.0, battery)
            previous = key
        return residuals

    def apply_strategies_to_global_model(
        self,
        day_logs: list[dict],
        track_c_distribution: list[dict] | None,
    ) -> int:
        """Delegate to LearningManager — see learning.py for implementation."""
        from homeassistant.util import dt as _dt
        return self.coordinator.learning.apply_strategies_to_global_model(
            day_logs=day_logs,
            track_c_distribution=track_c_distribution,
            strategies=self.coordinator._unit_strategies,
            model=self.coordinator.get_model_state(),
            learning_rate=self.coordinator.learning_rate,
            balance_point=self.coordinator.balance_point,
            wind_threshold=self.coordinator.wind_threshold,
            extreme_wind_threshold=self.coordinator.extreme_wind_threshold,
            parse_datetime_fn=_dt.parse_datetime,
        )

    def replay_per_unit_models(
        self, day_entries: list[dict], *, replay_aux: bool = True,
    ) -> dict | None:
        """Delegate to LearningManager — see learning.py for implementation.

        Used by ``retrain_from_history``, which rebuilds the per-unit aux
        coefficients too (``replay_aux``): its ``reset_first`` clears them
        and nothing else relearns them from history.
        """
        return self.coordinator.learning.replay_per_unit_models(
            day_entries=day_entries,
            strategies=self.coordinator._unit_strategies,
            model=self.coordinator.get_model_state(),
            learning_rate=self.coordinator.learning_rate,
            replay_aux=replay_aux,
            **per_unit_replay_context(self.coordinator),
        )

    async def try_track_b_cop_smearing(
        self,
        day_logs: list[dict],
        q_adjusted: float,
        date_key: str,
    ) -> int | None:
        """Attempt COP-weighted smearing for Track B (#793).

        When ENABLE_TRACK_B_COP_SMEARING is True and MPC COP params are
        available, distributes q_adjusted across 24 hours using per-hour
        COP weights instead of flat q/24.  Returns bucket update count,
        or None if smearing was not possible (flag off, no COP params).
        """
        cop_params = await self.fetch_track_b_cop_params()
        if cop_params is None:
            return None
        distribution = self.track_b_cop_distribution(
            day_logs, q_adjusted, date_key, cop_params,
        )
        if distribution is None:
            return None

        # Store distribution for strategy dispatch (same as Track C).
        bucket_updates = self.apply_strategies_to_global_model(
            day_logs, distribution,
        )

        # Persist distribution for retrain replay.
        self.coordinator._daily_history[date_key]["track_b_cop_distribution"] = distribution

        _LOGGER.info(
            f"Track B COP-smeared (#793): q_adjusted={q_adjusted:.2f} kWh "
            f"distributed across 24 hours using per-hour COP."
        )
        return bucket_updates

    async def fetch_track_b_cop_params(self, *, cache: bool = True) -> dict | None:
        """The MPC COP params Track B COP smearing uses, or ``None``.

        ``None`` while ENABLE_TRACK_B_COP_SMEARING is off (without awaiting
        anything) and when the params are neither cached nor fetchable.
        ``cache=False`` leaves ``_last_cop_params`` alone (a retrain dry
        run).  Split from :meth:`track_b_cop_distribution` so a retrain can
        fetch before its synchronous replay.
        """
        from .const import ENABLE_TRACK_B_COP_SMEARING
        if not ENABLE_TRACK_B_COP_SMEARING:
            return None

        # Try cached COP params (set by Track C midnight sync if it ran).
        cop_params = getattr(self.coordinator, '_last_cop_params', None)

        # If not cached, fetch directly from MPC.
        if cop_params is None and self.coordinator.mpc_entry_id:
            from homeassistant.exceptions import HomeAssistantError
            try:
                service_data = {"entry_id": self.coordinator.mpc_entry_id}
                cop_response = await self.coordinator.hass.services.async_call(
                    "heatpump_mpc", "get_cop_params",
                    service_data, blocking=True, return_response=True,
                )
                if isinstance(cop_response, dict) and "eta_carnot" in cop_response:
                    cop_params = cop_response
                    if cache:
                        self.coordinator._last_cop_params = cop_params
            except (TypeError, KeyError, AttributeError, HomeAssistantError) as err:
                # HomeAssistantError covers ServiceNotFound when the MPC
                # integration is uninstalled or not yet loaded (#878).
                _LOGGER.debug(f"Track B COP smearing: could not fetch COP params ({err})")
        return cop_params

    def track_b_cop_distribution(
        self,
        day_logs: list[dict],
        q_adjusted: float,
        date_key: str,
        cop_params: dict,
    ) -> list[dict] | None:
        """``q_adjusted`` spread over the day's hours with per-hour COP.

        Pure: writes nothing.  ``None`` when the smearing fails.
        """
        from homeassistant.util import dt as _dt
        from .thermodynamics import ThermodynamicEngine

        # Build weather data from hourly log (same logic as Track C): one row
        # per logged hour, in time order.  An hour missing from the log used
        # to get a row at the balance point, which carries no weight; the
        # repeated DST fall-back hour now gets a row of its own instead of
        # overwriting the first.
        slots = hour_slots(day_logs)
        solar_residual_by_key = self._solar_residuals(slots)
        # The placeholders below only fix the ratio between hours; spreading
        # q over the rows actually present keeps their sum at q_adjusted.
        per_row_kwh = q_adjusted / max(1, len(slots))

        weather_data = []
        synthetic_mpc_data = []
        for key, log_h in slots:
            inertia_t = log_h.get("inertia_temp")
            raw_t = log_h.get("temp")
            outdoor = inertia_t if inertia_t is not None else (raw_t if raw_t is not None else self.coordinator.balance_point)
            eff_wind = log_h.get("effective_wind") or 0.0

            if eff_wind >= self.coordinator.extreme_wind_threshold:
                wind_factor = 1.6
            elif eff_wind >= self.coordinator.wind_threshold:
                wind_factor = 1.3
            else:
                wind_factor = 1.0

            solar_with_decay = solar_residual_by_key.get(key, 0.0)
            solar_factor = max(0.0, 1.0 - solar_with_decay)

            raw_outdoor = raw_t if raw_t is not None else self.coordinator.balance_point
            rh = log_h.get("humidity")
            rh = rh if rh is not None else 50.0

            ts = log_h.get("timestamp", f"{date_key}T{log_h.get('hour', 0):02d}:00:00")
            weather_data.append({
                "datetime": ts,
                "delta_t": abs(self.coordinator.balance_point - outdoor),
                "is_cooling": outdoor > self.coordinator.balance_point,
                "wind_factor": wind_factor,
                "solar_factor": solar_factor,
                "outdoor_temp": raw_outdoor,
                "humidity": rh,
            })
            # Synthetic MPC record — we don't have thermal data, so use
            # placeholders.  With per-hour COP + renormalization, only
            # total_kwh_el matters (the thermal values cancel out).
            synthetic_mpc_data.append({
                "datetime": ts,
                "kwh_th_sh": per_row_kwh,  # Placeholder — ratio matters, not absolute
                "kwh_el_sh": per_row_kwh,  # COP=1 placeholder, overridden by per-hour COP
                "mode": "sh",
            })

        engine = ThermodynamicEngine(balance_point=self.coordinator.balance_point)
        try:
            distribution = engine.calculate_synthetic_baseline(
                synthetic_mpc_data, weather_data, cop_params=cop_params,
            )
        except (TypeError, KeyError, ValueError) as err:
            _LOGGER.warning(f"Track B COP smearing failed ({err}), falling back to flat.")
            return None
        return distribution

    @staticmethod
    def compute_excluded_mode_energy(day_logs: list[dict]) -> float:
        """Sum energy from units in modes excluded from global learning.

        Iterates hourly logs and totals kWh for any unit whose mode
        (per that hour's snapshot) is in MODES_EXCLUDED_FROM_GLOBAL_LEARNING.
        Units without a recorded mode default to MODE_HEATING (included).
        """
        excluded = 0.0
        for entry in day_logs:
            unit_modes = entry.get("unit_modes", {})
            breakdown = entry.get("unit_breakdown", {})
            for sid, kwh in breakdown.items():
                mode = unit_modes.get(sid, MODE_HEATING)
                if mode in MODES_EXCLUDED_FROM_GLOBAL_LEARNING:
                    excluded += kwh
        return excluded

    async def process(self, date_obj):
        """Process end of day."""
        key = date_obj.isoformat()
        day_logs = [e for e in self.coordinator._hourly_log if e["timestamp"].startswith(key)]

        # Validation: Ensure we have enough data (Kelvin Protocol)
        if len(day_logs) < 20:
            _LOGGER.warning(
                "Daily processing for %s: Incomplete data (%d/24 hours). Vectors may have gaps.",
                key,
                len(day_logs),
            )

        # Use Aggregation Helper to ensure full schema compliance
        if day_logs:
            daily_stats = self.aggregate_logs(day_logs)
        else:
            # Fallback if no logs (Downtime?)
            tdd = self.coordinator.data.get(ATTR_TDD, 0.0)
            kwh = self.coordinator._accumulated_energy_today
            avg_temp = self.coordinator.balance_point - tdd  # Approx

            # Fallback vectors
            empty_vector = [None] * 24
            hourly_vectors = {
                "temp": list(empty_vector),
                "wind": list(empty_vector),
                "tdd": list(empty_vector),
                "actual_kwh": list(empty_vector),
            }
            if self.coordinator.solar_enabled:
                hourly_vectors["solar_rad"] = list(empty_vector)

            daily_stats = {
                "kwh": round(kwh, 2),
                "tdd": round(tdd, 1),
                "balance_point": self.coordinator.balance_point,
                "temp": round(avg_temp, 1),
                "wind": 0.0,
                "solar_factor": 0.0,
                # Fill missing with safe defaults
                "expected_kwh": 0.0,
                "forecasted_kwh": 0.0,
                "aux_impact_kwh": 0.0,
                "solar_impact_kwh": 0.0,
                "guest_impact_kwh": 0.0,
                "unit_breakdown": {},
                "unit_expected_breakdown": {},
                "deviation": 0.0,
                "hourly_vectors": hourly_vectors,
            }

        self.coordinator._daily_history[key] = daily_stats

        # Daily Learning Mode — baseline selection and strategy dispatch (#776)
        current_indoor_temp = None
        if self.coordinator.indoor_temp_sensor:
            current_indoor_temp = self.coordinator._get_float_state(self.coordinator.indoor_temp_sensor)

        # Mode filtering (#789): exclude OFF/DHW/Guest energy from
        # daily learning so Track B/C match Track A's filtering semantics.
        # Cooling is included since #801 (saturation-aware solar normalization).
        excluded_mode_kwh = DailyProcessor.compute_excluded_mode_energy(day_logs) if day_logs else 0.0
        q_adjusted = daily_stats["kwh"] - excluded_mode_kwh
        track_c_distribution = None

        # Accumulate solar normalization delta over all hours (#792).
        # Used by Track B flat daily to normalize q_adjusted to dark-sky.
        daily_solar_delta = 0.0
        if day_logs:
            for entry in day_logs:
                # #968: prefer 4D delta per-entry when present; fall back to
                # 3D otherwise.  Flag-independent (aggregation path).
                daily_solar_delta += entry.get(
                    "solar_normalization_delta_4d",
                    entry.get("solar_normalization_delta", 0.0),
                )

        if excluded_mode_kwh > 0.0:
            _LOGGER.debug(
                "Daily mode filter (#789): excluded %.3f kWh "
                "(OFF/DHW/Guest) from %.2f kWh total.",
                excluded_mode_kwh, daily_stats["kwh"],
            )

        # #855 Option B: flag Track C installs whose MPC did not produce a
        # distribution this day.  When set, bucket learning AND U-coefficient
        # update are both skipped to avoid writing Track-B-semantic values
        # (raw electrical minus thermal-mass correction) into buckets that
        # normally hold MPC-synthetic thermal-per-hour values.  The U-coeff
        # is skipped for the same reason: mixing MPC-thermal q_adjusted with
        # electrical-only q_adjusted in the same EMA gives a meaningless value.
        track_c_outage_skip = False

        if self.coordinator.track_c_enabled and self.coordinator.daily_learning_mode and day_logs:
            # Track C: replace electrical baseline with thermodynamic synthetic baseline.
            track_c_result = await self.run_track_c_midnight_sync(day_logs, key)
            if track_c_result is not None:
                track_c_kwh, track_c_distribution, track_c_source, track_c_smear = track_c_result

                # Compute q_adjusted from strategy contributions for U-coefficient.
                # Only include non-MPC sensors that are in a learning-eligible mode (#789).
                non_mpc_daily_kwh = 0.0
                if self.coordinator.mpc_managed_sensor:
                    for log_entry in day_logs:
                        breakdown = log_entry.get("unit_breakdown", {})
                        unit_modes = log_entry.get("unit_modes", {})
                        for sid in self.coordinator.energy_sensors:
                            if sid != self.coordinator.mpc_managed_sensor:
                                mode = unit_modes.get(sid, MODE_HEATING)
                                if mode not in MODES_EXCLUDED_FROM_GLOBAL_LEARNING:
                                    non_mpc_daily_kwh += breakdown.get(sid, 0.0)

                q_adjusted = track_c_kwh + non_mpc_daily_kwh
                self.coordinator._daily_history[key]["track_c_kwh"] = round(q_adjusted, 3)
                self.coordinator._daily_history[key]["track_c_kwh_mpc_only"] = round(track_c_kwh, 3)
                self.coordinator._daily_history[key]["track_c_kwh_non_mpc"] = round(non_mpc_daily_kwh, 3)
                self.coordinator._daily_history[key]["track_c_distribution"] = track_c_distribution
                # What the day was smeared at, and the MPC COP parameters, so a
                # retrain under other settings can re-spread it (#1111).
                self.coordinator._daily_history[key]["track_c_smear"] = track_c_smear
                # S1 (#855 follow-up): explicit track identity per day.  "C_live"
                # or "C_<snapshot_HHMM>".  Lets diagnose_model, retrain and
                # future BP-aware consumers see the attribution source without
                # inferring it from which fields happen to be present.
                self.coordinator._daily_history[key]["track_used"] = (
                    "C_live" if track_c_source == "live" else f"C_{track_c_source}"
                )
            else:
                # #855 Option B: skip learning entirely on MPC outage.
                _LOGGER.warning(
                    "Track C unavailable for %s — skipping learning "
                    "(no bucket update, no U-coefficient update). "
                    "Mixing MPC-synthetic and raw-electrical q_adjusted "
                    "in the same model is a category error; waiting for "
                    "MPC recovery is the safer path.",
                    key,
                )
                self.coordinator._track_c_outage_count_session += 1
                track_c_outage_skip = True
                self.coordinator._daily_history[key]["track_used"] = "skipped_mpc_outage"
        elif self.coordinator.thermal_mass_kwh_per_degree > 0.0 and current_indoor_temp is not None and self.coordinator._last_midnight_indoor_temp is not None:
            delta_t_indoor = current_indoor_temp - self.coordinator._last_midnight_indoor_temp
            base_kwh = daily_stats["kwh"] - excluded_mode_kwh
            q_adjusted = base_kwh - (self.coordinator.thermal_mass_kwh_per_degree * delta_t_indoor)
            _LOGGER.debug(f"Daily Learning: Adjusted kWh from {base_kwh:.2f} to {q_adjusted:.2f} based on indoor delta T {delta_t_indoor:.2f}°C")

        if (
            self.coordinator.daily_learning_mode
            and self.coordinator.learning_enabled
            and daily_stats["tdd"] >= 0.5
            and q_adjusted > 0
            and not track_c_outage_skip  # #855 Option B
        ):
            if len(day_logs) >= 22:
                # U-coefficient: always updated daily regardless of track.
                observed_u = q_adjusted / daily_stats["tdd"]
                if self.coordinator._learned_u_coefficient is None:
                    self.coordinator._learned_u_coefficient = observed_u
                else:
                    self.coordinator._learned_u_coefficient = self.coordinator._learned_u_coefficient + DEFAULT_DAILY_LEARNING_RATE * (observed_u - self.coordinator._learned_u_coefficient)
                _LOGGER.info(f"Daily Learning: Updated U-coefficient to {self.coordinator._learned_u_coefficient:.4f} (Observed: {observed_u:.4f})")

                if track_c_distribution:
                    # --- Track C: per-hour bucket learning via strategies (#776) ---
                    bucket_updates = self.apply_strategies_to_global_model(
                        day_logs, track_c_distribution,
                    )
                    _LOGGER.info(f"Track C Strategy Learning: {bucket_updates} bucket updates from 24 hours.")
                else:
                    # --- Track B bucket learning ---
                    cop_smeared = await self.try_track_b_cop_smearing(
                        day_logs, q_adjusted, key,
                    )
                    if cop_smeared:
                        _LOGGER.info(f"Track B COP-smeared: {cop_smeared} bucket updates from 24 hours.")
                        self.coordinator._daily_history[key]["track_used"] = "B_cop"
                    else:
                        # Flat fallback: single q_adjusted/24 to one bucket.
                        # Apply accumulated solar normalization delta (#792).
                        q_solar_normalized = max(0.0, q_adjusted + daily_solar_delta)
                        q_hourly_avg = q_solar_normalized / 24.0
                        avg_temp = daily_stats["temp"]
                        daily_wind = daily_stats["wind"]
                        flat_temp_key = str(int(round(avg_temp)))
                        flat_wind_bucket = self.coordinator._get_wind_bucket(daily_wind)

                        if flat_temp_key not in self.coordinator._correlation_data:
                            self.coordinator._correlation_data[flat_temp_key] = {}
                        current_pred = self.coordinator._correlation_data[flat_temp_key].get(flat_wind_bucket, 0.0)

                        if current_pred == 0.0:
                            self.coordinator._correlation_data[flat_temp_key][flat_wind_bucket] = round(q_hourly_avg, 5)
                            _LOGGER.info(f"Track B Learning (Cold Start): T={flat_temp_key} W={flat_wind_bucket} -> {q_hourly_avg:.3f} kWh")
                        else:
                            new_pred = current_pred + self.coordinator.learning_rate * (q_hourly_avg - current_pred)
                            self.coordinator._correlation_data[flat_temp_key][flat_wind_bucket] = round(new_pred, 5)
                            _LOGGER.info(f"Track B Learning (EMA): T={flat_temp_key} W={flat_wind_bucket} -> {new_pred:.3f} kWh (was {current_pred:.3f}, actual avg {q_hourly_avg:.3f})")
                        # S1: tag day as Track B flat-daily attribution.
                        # Pure Track B installs (track_c_enabled=False) land
                        # here via the thermal-mass-correction elif branch;
                        # track_c_enabled installs never reach here because
                        # track_c_outage_skip guards the parent block.
                        self.coordinator._daily_history[key].setdefault("track_used", "B_flat")
            else:
                _LOGGER.info(f"Daily Learning skipped: Incomplete day ({len(day_logs)}/24 hours)")

        self.coordinator.data["learned_u_coefficient"] = self.coordinator._learned_u_coefficient

        # S1 (#855 follow-up): ensure every daily_history entry has an
        # explicit track_used tag so diagnose_model and future BP-aware
        # consumers can see composition without inferring from field
        # presence.  Non-daily-learning installs (plain Track A) land here
        # with no track_used set — mark them as "A".  All daily-mode
        # branches set their own tag above.
        if not self.coordinator.daily_learning_mode:
            self.coordinator._daily_history[key].setdefault("track_used", "A")

        if current_indoor_temp is not None:
            self.coordinator._last_midnight_indoor_temp = current_indoor_temp
            # Store midnight indoor temp per day so retrain_from_history can apply
            # thermal mass correction historically without a live sensor read.
            self.coordinator._daily_history[key]["midnight_indoor_temp"] = round(current_indoor_temp, 1)

        # Forecast Accuracy Tracking
        # Kelvin Protocol: Skip accuracy evaluation if learning is disabled (e.g. Vacation).
        if self.coordinator.learning_enabled:
            self.coordinator.forecast.log_accuracy(
                key,
                daily_stats["kwh"],
                daily_stats.get("aux_impact_kwh", 0.0),
                modeled_net_kwh=daily_stats.get("expected_kwh", 0.0),
                guest_impact_kwh=daily_stats.get("guest_impact_kwh", 0.0)
            )
        else:
            _LOGGER.info(f"Forecast accuracy update skipped for {key} (Learning Disabled).")

        _LOGGER.info(f"Daily Update for {key}: Energy={daily_stats['kwh']}, TDD={daily_stats['tdd']}")

        # CSV Auto-logging (if enabled)
        daily_log_entry = {
            "timestamp": key,
            "kwh": daily_stats["kwh"],
            "temp": daily_stats["temp"],
            "tdd": daily_stats["tdd"],
            # Include per-device breakdown (Actual)
            **{f"device_{i}": daily_stats.get("unit_breakdown", {}).get(entity_id, 0.0)
                for i, entity_id in enumerate(self.coordinator.energy_sensors)}
        }
        await self.coordinator.storage.append_daily_log_csv(daily_log_entry)

        self.coordinator._accumulated_energy_today = 0.0
        self.coordinator._daily_individual = {} # Reset daily individual trackers
        self.coordinator._daily_aux_breakdown = {} # Reset daily aux breakdown
        self.coordinator._daily_orphaned_aux = 0.0 # Reset daily orphaned accumulator

        # Cleanup last energy values for removed sensors at end of day
        # This ensures we don't carry dead references forever
        current_sensors = set(self.coordinator.energy_sensors)
        keys_to_remove = [k for k in self.coordinator._last_energy_values if k not in current_sensors]
        for k in keys_to_remove:
            del self.coordinator._last_energy_values[k]

        self.coordinator.data[ATTR_TDD] = 0.0

        await self.coordinator._async_save_data(force=True)
