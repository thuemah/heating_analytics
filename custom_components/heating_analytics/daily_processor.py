"""DailyProcessor — hosts the day-boundary processing extracted from coordinator.py.

Thin-delegate pattern: the processor holds a reference to the coordinator
and reaches back for state.  Public methods on the coordinator delegate
to this engine so the external API is unchanged.
"""
from __future__ import annotations

import logging

from .const import (
    ATTR_TDD,
    DEFAULT_DAILY_LEARNING_RATE,
    MODES_EXCLUDED_FROM_GLOBAL_LEARNING,
    MODE_HEATING,
)
from .thermodynamics import ThermodynamicEngine

_LOGGER = logging.getLogger(__name__)


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
        solar_impact = sum(e.get("solar_impact_kwh", 0.0) for e in day_logs)
        aux_impact = sum(e.get("aux_impact_kwh", 0.0) for e in day_logs)
        guest_impact = sum(e.get("guest_impact_kwh", 0.0) for e in day_logs)
        solar_norm_delta_total = sum(e.get("solar_normalization_delta", 0.0) for e in day_logs)

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
                sol = e.get("solar_impact_kwh", 0.0)
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

        for e in day_logs:
            for uid, val in e.get("unit_breakdown", {}).items():
                unit_breakdown[uid] = unit_breakdown.get(uid, 0.0) + val
            for uid, val in e.get("unit_expected_breakdown", {}).items():
                unit_expected[uid] = unit_expected.get(uid, 0.0) + val

        # Averages
        avg_temp = sum(e["temp"] for e in day_logs) / len(day_logs)
        avg_wind = sum(e.get("effective_wind", 0.0) for e in day_logs) / len(day_logs)
        avg_solar = sum(e.get("solar_factor", 0.0) for e in day_logs) / len(day_logs)

        # TDD (Sum of hourly TDD)
        total_tdd = sum(e.get("tdd", 0.0) for e in day_logs)

        # Hourly Vectors (Kelvin Protocol: Data Aggregation)
        hourly_vectors = {
            "temp": [None] * 24,
            "wind": [None] * 24,
            "tdd": [None] * 24,
            "actual_kwh": [None] * 24,
        }
        if self.coordinator.solar_enabled:
            hourly_vectors["solar_rad"] = [None] * 24
            hourly_vectors["solar_norm_delta"] = [None] * 24

        # Hour Collision Fix: Aggregate instead of overwrite
        # Iterate over hour slots (0-23) and aggregate all entries for that hour.
        # This handles cases where multiple logs exist for the same hour (e.g. restart).
        # DST Handling:
        # - Spring Forward (23h): One hour slot will remain None (handled downstream).
        # - Fall Back (25h): Two sets of logs map to same hour index. They are aggregated here.
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
            "thermodynamic_gross_kwh": round(thermodynamic_gross_kwh, 2),
            "tdd": round(total_tdd, 1),
            "temp": round(avg_temp, 1),
            "wind": round(avg_wind, 1),
            "solar_factor": round(avg_solar, 3),
            "unit_breakdown": {k: round(v, 3) for k, v in unit_breakdown.items()},
            "unit_expected_breakdown": {k: round(v, 3) for k, v in unit_expected.items()},
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
    ) -> tuple[float, list, str] | None:
        """Fetch MPC thermal data and run the ThermodynamicEngine Midnight Sync.

        Returns (total_synthetic_el, distribution, source) where:
          - total_synthetic_el: sum of synthetic_kwh_el across all 24 hours —
            the weather-smeared electrical equivalent used as q_adjusted in learning.
          - distribution: the full list of HourlyDistribution dicts for storage
            (enables future per-hour visualisation without recomputing).
          - source: "live" or "snapshot_<HHMM>" — identifies the data origin
            for daily_history tagging and diagnostics.
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

        # Build WeatherData from the day's hourly log entries (already available).
        # delta_t  = balance_point - inertia_temp  (inertia-weighted temp mirrors Track A model;
        #            falls back to raw temp if inertia_temp not logged)
        # wind_factor = 3-bucket multiplier matching Track A wind buckets (1.0/1.3/1.6)
        # solar_factor = 1.0 - solar (inverted; 0=no sun → full loss weight)
        #                — with solar thermal battery decay applied so afternoon solar
        #                  gain residual carries into evening hours (mirrors solar_battery_decay)
        weather_data = []
        log_by_hour = {e.get("hour", -1): e for e in day_logs}

        # Solar battery pre-pass: accumulate decay across hours so that afternoon
        # solar gain reduces evening loss weights, matching Track A's solar battery model.
        solar_battery = 0.0
        solar_residual_by_hour: dict[int, float] = {}
        for h in range(24):
            log_h = log_by_hour.get(h, {})
            raw_solar_h = log_h.get("solar_factor")
            raw_solar_h = raw_solar_h if raw_solar_h is not None else 0.0
            solar_battery = solar_battery * self.coordinator.solar_battery_decay + raw_solar_h * (1 - self.coordinator.solar_battery_decay)
            solar_residual_by_hour[h] = min(1.0, solar_battery)

        for record in mpc_records:
            try:
                record_dt = _dt.parse_datetime(record["datetime"])
                hour = record_dt.hour if record_dt else -1
            except (KeyError, TypeError, ValueError):
                hour = -1

            log_entry = log_by_hour.get(hour, {})
            # Fix 1: use inertia_temp (thermal-mass-weighted) rather than instantaneous
            # outdoor temp — consistent with how Track A models heat demand.
            # Use explicit None-check: dict.get(key, default) silently returns None
            # when the key exists with a None value (e.g. early startup entries).
            inertia_t = log_entry.get("inertia_temp")
            raw_t = log_entry.get("temp")
            outdoor_temp = (
                inertia_t if inertia_t is not None
                else raw_t if raw_t is not None
                else self.coordinator.balance_point
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
            solar_with_decay = solar_residual_by_hour.get(hour if hour >= 0 else 0, 0.0)
            solar_factor = max(0.0, 1.0 - solar_with_decay)

            # Raw outdoor temp and humidity for per-hour COP calculation.
            # Use raw_t (not inertia) for COP — COP depends on instantaneous
            # air temperature at the evaporator, not thermally weighted.
            raw_outdoor = raw_t if raw_t is not None else self.coordinator.balance_point
            rh = log_entry.get("humidity")
            rh = rh if rh is not None else 50.0

            weather_data.append({
                "datetime": record["datetime"],
                "delta_t": abs(self.coordinator.balance_point - outdoor_temp),
                "is_cooling": outdoor_temp > self.coordinator.balance_point,
                "wind_factor": wind_factor,
                "solar_factor": solar_factor,
                "outdoor_temp": raw_outdoor,
                "humidity": rh,
            })

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

        return total_synthetic_el, distribution, source

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

    def replay_per_unit_models(self, day_entries: list[dict]) -> None:
        """Delegate to LearningManager — see learning.py for implementation."""
        self.coordinator.learning.replay_per_unit_models(
            day_entries=day_entries,
            strategies=self.coordinator._unit_strategies,
            model=self.coordinator.get_model_state(),
            learning_rate=self.coordinator.learning_rate,
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
                    self.coordinator._last_cop_params = cop_params
            except (TypeError, KeyError, AttributeError, HomeAssistantError) as err:
                # HomeAssistantError covers ServiceNotFound when the MPC
                # integration is uninstalled or not yet loaded (#878).
                _LOGGER.debug(f"Track B COP smearing: could not fetch COP params ({err})")

        if cop_params is None:
            return None

        from homeassistant.util import dt as _dt
        from .thermodynamics import ThermodynamicEngine

        # Build weather data from hourly log (same logic as Track C).
        log_by_hour = {e.get("hour", -1): e for e in day_logs}
        solar_battery = 0.0
        solar_residual_by_hour: dict[int, float] = {}
        for h in range(24):
            log_h = log_by_hour.get(h, {})
            raw_solar_h = log_h.get("solar_factor") or 0.0
            solar_battery = solar_battery * self.coordinator.solar_battery_decay + raw_solar_h * (1 - self.coordinator.solar_battery_decay)
            solar_residual_by_hour[h] = min(1.0, solar_battery)

        weather_data = []
        synthetic_mpc_data = []
        for h in range(24):
            log_h = log_by_hour.get(h, {})
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

            solar_with_decay = solar_residual_by_hour.get(h, 0.0)
            solar_factor = max(0.0, 1.0 - solar_with_decay)

            raw_outdoor = raw_t if raw_t is not None else self.coordinator.balance_point
            rh = log_h.get("humidity")
            rh = rh if rh is not None else 50.0

            ts = log_h.get("timestamp", f"{date_key}T{h:02d}:00:00")
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
                "kwh_th_sh": q_adjusted / 24.0,  # Placeholder — ratio matters, not absolute
                "kwh_el_sh": q_adjusted / 24.0,   # COP=1 placeholder, overridden by per-hour COP
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
                daily_solar_delta += entry.get("solar_normalization_delta", 0.0)

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
                track_c_kwh, track_c_distribution, track_c_source = track_c_result

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
