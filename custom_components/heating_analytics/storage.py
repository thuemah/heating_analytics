"""Storage Manager Service."""
from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
from datetime import datetime, timedelta

from homeassistant.util import dt as dt_util
from homeassistant.helpers.storage import Store

from .const import (
    STORAGE_VERSION,
    STORAGE_KEY,
    ATTR_TDD,
    ATTR_FORECAST_TODAY,
    ATTR_PREDICTED,
    ATTR_TDD_DAILY_STABLE,
    ATTR_TEMP_FORECAST_TODAY,
    ATTR_LAST_HOUR_ACTUAL,
    ATTR_LAST_HOUR_EXPECTED,
    ATTR_LAST_HOUR_DEVIATION,
    ATTR_LAST_HOUR_DEVIATION_PCT,
    ATTR_MIDNIGHT_FORECAST,
    ATTR_MIDNIGHT_UNIT_ESTIMATES,
    ATTR_MIDNIGHT_UNIT_MODES,
)

_LOGGER = logging.getLogger(__name__)

class StorageManager:
    """Manages data persistence (JSON, CSV)."""

    def __init__(self, coordinator) -> None:
        """Initialize with reference to coordinator."""
        self.coordinator = coordinator
        self._store = Store(coordinator.hass, STORAGE_VERSION, STORAGE_KEY)
        self._last_save_time = None
        self._save_debounce_seconds = 60
        self._save_lock = asyncio.Lock()

    def _cleanup_removed_sensors(self, target_dict: dict, log_context: str | None = None) -> None:
        """Removes entries for sensors that are no longer tracked."""
        if self.coordinator.energy_sensors:
            removed_sensors = [s for s in target_dict if s not in self.coordinator.energy_sensors]
            if removed_sensors:
                if log_context:
                    _LOGGER.info(f"Removing deleted sensors from {log_context}: {removed_sensors}")
                for s in removed_sensors:
                    del target_dict[s]

    async def async_load_data(self):
        """Load data from storage."""
        try:
            data = await self._store.async_load()
            if not data:
                _LOGGER.warning("No data loaded from storage. Starting with fresh state.")
                # Notify user if storage was corrupt (HA moves to .corrupt file)
                storage_path = self._store.path
                import glob
                corrupt_files = glob.glob(f"{storage_path}.corrupt*")
                if corrupt_files:
                    _LOGGER.error(f"Corrupt storage file detected: {corrupt_files[0]}")
                    self.coordinator.hass.components.persistent_notification.create(
                        title="Heating Analytics: Storage Corruption Detected",
                        message=(
                            f"The storage file was corrupted and has been moved to:\n\n"
                            f"`{corrupt_files[0]}`\n\n"
                            f"**Heating Analytics has started with a fresh state.**\n\n"
                            f"**Possible causes:**\n"
                            f"- Manual editing of `.storage/` files (not recommended)\n"
                            f"- System crash during save operation\n\n"
                            f"**To restore data:**\n"
                            f"1. Fix the JSON syntax error in the corrupt file\n"
                            f"2. Use the `heating_analytics.restore_data` service to restore from backup\n\n"
                            f"**Prevention:** Use the provided services (import/export/backup) instead of manual editing."
                        ),
                        notification_id="heating_analytics_storage_corrupt"
                    )
                return

            # Load correlation data
            loaded_correlation = data.get("correlation_data", {})
            if isinstance(loaded_correlation, dict):
                self.coordinator._correlation_data.clear()
                self.coordinator._correlation_data.update(loaded_correlation)
            else:
                _LOGGER.warning("Loaded correlation_data is not a dictionary. Skipping update.")

            # Load correlation data per unit
            loaded_correlation_per_unit = data.get("correlation_data_per_unit", {})
            if isinstance(loaded_correlation_per_unit, dict):
                self.coordinator._correlation_data_per_unit = loaded_correlation_per_unit

                # Cleanup: Remove sensors that are no longer in configuration
                self._cleanup_removed_sensors(
                    self.coordinator._correlation_data_per_unit, "correlation data"
                )

                # MIGRATION: Convert old Aux Buckets (Unit Specific) to Coefficients
                # Iterate through units, temps, and buckets to find "with_auxiliary_heating"
                if self.coordinator._correlation_data_per_unit:
                    for entity_id, temps in self.coordinator._correlation_data_per_unit.items():
                        for temp_key, buckets in temps.items():
                            if "with_auxiliary_heating" in buckets:
                                # We remove it. We do NOT try to migrate it to a coefficient because
                                # we are switching to "Cold Start" learning for unit aux.
                                # Migrating an absolute value to a reduction coeff is risky without base context.
                                del buckets["with_auxiliary_heating"]
                                _LOGGER.info(f"Removed legacy 'with_auxiliary_heating' bucket for unit {entity_id} T={temp_key}. Unit will relearn aux impact.")

            else:
                _LOGGER.warning("Loaded correlation_data_per_unit is not a dictionary.")

            # Load unit aux coefficients
            loaded_aux_coefficients_per_unit = data.get("aux_coefficients_per_unit", {})
            if isinstance(loaded_aux_coefficients_per_unit, dict):
                self.coordinator._aux_coefficients_per_unit = loaded_aux_coefficients_per_unit
                self._cleanup_removed_sensors(self.coordinator._aux_coefficients_per_unit)

            # Load unit solar coefficients
            loaded_solar_coefficients_per_unit = data.get("solar_coefficients_per_unit", {})
            if isinstance(loaded_solar_coefficients_per_unit, dict):
                self.coordinator._solar_coefficients_per_unit = loaded_solar_coefficients_per_unit
                self._cleanup_removed_sensors(self.coordinator._solar_coefficients_per_unit)
            else:
                self.coordinator._solar_coefficients_per_unit = {}

            # Load unit modes
            loaded_unit_modes = data.get("unit_modes", {})
            if isinstance(loaded_unit_modes, dict):
                self.coordinator._unit_modes = loaded_unit_modes
                self._cleanup_removed_sensors(self.coordinator._unit_modes)
            else:
                self.coordinator._unit_modes = {}

            # Load learning buffer per unit
            loaded_buffer = data.get("learning_buffer_per_unit", {})
            if isinstance(loaded_buffer, dict):
                self.coordinator._learning_buffer_per_unit = loaded_buffer
                self._cleanup_removed_sensors(self.coordinator._learning_buffer_per_unit)
            else:
                self.coordinator._learning_buffer_per_unit = {}

            # Load learning buffer aux per unit
            loaded_buffer_aux = data.get("learning_buffer_aux_per_unit", {})
            if isinstance(loaded_buffer_aux, dict):
                self.coordinator._learning_buffer_aux_per_unit = loaded_buffer_aux
                self._cleanup_removed_sensors(self.coordinator._learning_buffer_aux_per_unit)
            else:
                self.coordinator._learning_buffer_aux_per_unit = {}

            # Load learning buffer solar per unit
            loaded_buffer_solar = data.get("learning_buffer_solar_per_unit", {})
            if isinstance(loaded_buffer_solar, dict):
                self.coordinator._learning_buffer_solar_per_unit = loaded_buffer_solar
                self._cleanup_removed_sensors(self.coordinator._learning_buffer_solar_per_unit)
            else:
                self.coordinator._learning_buffer_solar_per_unit = {}

            # Load global learning buffer
            loaded_buffer_global = data.get("learning_buffer_global", {})
            if isinstance(loaded_buffer_global, dict):
                self.coordinator._learning_buffer_global = loaded_buffer_global
            else:
                self.coordinator._learning_buffer_global = {}

            # Load observation counts
            loaded_observation_counts = data.get("observation_counts", {})
            if isinstance(loaded_observation_counts, dict):
                self.coordinator._observation_counts = loaded_observation_counts
                self._cleanup_removed_sensors(self.coordinator._observation_counts)

            # Load aux coefficients
            loaded_aux_coeff = data.get("aux_coefficients", {})
            if isinstance(loaded_aux_coeff, dict):
                self.coordinator._aux_coefficients = {}
                # MIGRATION: Convert legacy float coefficients to dict format (wind buckets)
                for temp_key, val in loaded_aux_coeff.items():
                    if isinstance(val, (int, float)):
                        # Legacy format: float value. Migrate to { "normal": value }
                        self.coordinator._aux_coefficients[temp_key] = {"normal": float(val)}
                        _LOGGER.info(f"Migrated legacy Aux coefficient for T={temp_key}: {val} -> normal={val}")
                    elif isinstance(val, dict):
                        # New format: Dict of buckets
                        self.coordinator._aux_coefficients[temp_key] = val
                    else:
                        _LOGGER.warning(f"Invalid format for aux coefficient T={temp_key}: {val}")

            # MIGRATION: Convert old Aux Buckets to Coefficients (Global)
            # We look for "with_auxiliary_heating" in correlation_data
            # and convert it to a subtractive coefficient (kW savings)
            if self.coordinator._correlation_data:
                for temp_key, buckets in self.coordinator._correlation_data.items():
                    if "with_auxiliary_heating" in buckets:
                        aux_val = buckets["with_auxiliary_heating"]

                        # Find a physical baseline
                        base_val = buckets.get("normal")
                        if base_val is None:
                            base_val = buckets.get("high_wind")

                        if base_val is not None and base_val > 0:
                            # Savings = Base - Aux (kW)
                            savings = base_val - aux_val
                            # Clamp to positive savings (Aux shouldn't increase consumption ideally, or we treat it as 0)
                            savings = max(0.0, savings)

                            # Store if significant
                            if savings > 0:
                                self.coordinator._aux_coefficients[temp_key] = round(savings, 3)
                                _LOGGER.info(f"Migrated Aux Bucket for T={temp_key}: Base={base_val}, Aux={aux_val} -> Coeff={savings:.3f} kW")

                        # Mark for cleanup - delete key so it doesn't persist
                        del buckets["with_auxiliary_heating"]

            # MIGRATION: Clean up "with_auxiliary_heating" from per-unit data structures
            if self.coordinator._correlation_data_per_unit:
                for entity_id, temp_data in self.coordinator._correlation_data_per_unit.items():
                    for temp_key, buckets in list(temp_data.items()):
                        if "with_auxiliary_heating" in buckets:
                            _LOGGER.info(f"Removing legacy 'with_auxiliary_heating' bucket from {entity_id} at T={temp_key}")
                            del buckets["with_auxiliary_heating"]

            if self.coordinator._learning_buffer_per_unit:
                for entity_id, temp_data in self.coordinator._learning_buffer_per_unit.items():
                    for temp_key, buckets in list(temp_data.items()):
                        if "with_auxiliary_heating" in buckets:
                            _LOGGER.info(f"Removing legacy 'with_auxiliary_heating' buffer from {entity_id} at T={temp_key}")
                            del buckets["with_auxiliary_heating"]

            if self.coordinator._observation_counts:
                for entity_id, temp_data in self.coordinator._observation_counts.items():
                    for temp_key, buckets in list(temp_data.items()):
                        if "with_auxiliary_heating" in buckets:
                            _LOGGER.info(f"Removing legacy 'with_auxiliary_heating' count from {entity_id} at T={temp_key}")
                            del buckets["with_auxiliary_heating"]

            # Load daily history
            loaded_daily_history = data.get("daily_history", {})
            if isinstance(loaded_daily_history, dict):
                # MIGRATION: Rename 'hdd' to 'tdd' in historical entries
                for entry in loaded_daily_history.values():
                    if not entry:
                        continue
                    if "hdd" in entry and "tdd" not in entry:
                        entry["tdd"] = entry.pop("hdd")

                    # VALIDATION: Ensure hourly_vectors is a dictionary if present
                    if "hourly_vectors" in entry:
                        if not isinstance(entry["hourly_vectors"], dict):
                            _LOGGER.warning("Invalid hourly_vectors format in daily_history entry. Resetting to None.")
                            entry["hourly_vectors"] = None
                        else:
                            # MIGRATION: Rename 'load' to 'actual_kwh' in vectors (Clarity Refactor)
                            vectors = entry["hourly_vectors"]
                            if "load" in vectors and "actual_kwh" not in vectors:
                                vectors["actual_kwh"] = vectors.pop("load")

                self.coordinator._daily_history = loaded_daily_history
            else:
                _LOGGER.warning("Loaded daily_history is not a dictionary.")

            self.coordinator._accumulated_energy_today = data.get("accumulated_energy_today", 0.0)
            self.coordinator._accumulated_expected_energy_hour = data.get("accumulated_expected_energy_hour", 0.0)

            # Load daily aux breakdown
            loaded_daily_aux_breakdown = data.get("daily_aux_breakdown", {})
            if isinstance(loaded_daily_aux_breakdown, dict):
                self.coordinator._daily_aux_breakdown = loaded_daily_aux_breakdown
            else:
                self.coordinator._daily_aux_breakdown = {}

            # Load daily individual
            loaded_daily_individual = data.get("daily_individual", {})
            if isinstance(loaded_daily_individual, dict):
                self.coordinator._daily_individual = loaded_daily_individual

                # Cleanup: Remove sensors that are no longer in configuration
                initial_count = len(self.coordinator._daily_individual)
                self._cleanup_removed_sensors(
                    self.coordinator._daily_individual, "daily tracking"
                )

                if len(self.coordinator._daily_individual) < initial_count:
                    # Recalculate accumulated energy today based on remaining valid sensors
                    self.coordinator._accumulated_energy_today = sum(self.coordinator._daily_individual.values())
                    _LOGGER.info(f"Recalculated accumulated_energy_today: {self.coordinator._accumulated_energy_today}")
            else:
                _LOGGER.warning("Loaded daily_individual is not a dictionary.")

            # Load lifetime individual
            loaded_lifetime_individual = data.get("lifetime_individual", {})
            if isinstance(loaded_lifetime_individual, dict):
                self.coordinator._lifetime_individual = loaded_lifetime_individual
                self._cleanup_removed_sensors(
                    self.coordinator._lifetime_individual, "lifetime tracking"
                )

            # Load hourly log
            loaded_hourly_log = data.get("hourly_log", [])
            if isinstance(loaded_hourly_log, list):
                # MIGRATION: Rename 'hdd' to 'tdd' in logs
                for entry in loaded_hourly_log:
                    if isinstance(entry, dict) and "hdd" in entry and "tdd" not in entry:
                        entry["tdd"] = entry.pop("hdd")
                self.coordinator._hourly_log = loaded_hourly_log
            else:
                _LOGGER.warning("Loaded hourly_log is not a list.")

            # Load hourly persistence (with hour validation)
            current_time = dt_util.now()
            current_hour = current_time.replace(minute=0, second=0, microsecond=0)
            stale_energy_baselines = False

            acc_start_iso = data.get("accumulation_start_time")
            if acc_start_iso:
                acc_start_time = dt_util.parse_datetime(acc_start_iso)
                # Only restore if still same hour
                if acc_start_time and acc_start_time.replace(minute=0, second=0, microsecond=0) == current_hour:
                    self.coordinator._accumulation_start_time = acc_start_time
                    self.coordinator._accumulated_energy_hour = data.get("accumulated_energy_hour", 0.0)
                    self.coordinator._accumulated_expected_energy_hour = data.get("accumulated_expected_energy_hour", 0.0)
                    self.coordinator._accumulated_aux_impact_hour = data.get("accumulated_aux_impact_hour", 0.0)
                    self.coordinator._accumulated_orphaned_aux = data.get("accumulated_orphaned_aux", 0.0)
                    self.coordinator._accumulated_aux_breakdown = data.get("accumulated_aux_breakdown", {})
                    self.coordinator._hourly_delta_per_unit = data.get("hourly_delta_per_unit", {})
                    self.coordinator._hourly_expected_per_unit = data.get("hourly_expected_per_unit", {})
                    self.coordinator._last_minute_processed = data.get("last_minute_processed")
                    # Restore gap filling state
                    self.coordinator.data["current_model_rate"] = data.get("current_model_rate", 0.0)
                    self.coordinator.data["current_unit_breakdown"] = data.get("current_unit_breakdown", {})
                    self.coordinator.data["current_aux_impact_rate"] = data.get("current_aux_impact_rate", 0.0)
                    self.coordinator.data["current_calc_temp"] = data.get("current_calc_temp")
                    _LOGGER.info(f"Restored hourly energy data from {acc_start_time.strftime('%H:%M')}")
                else:
                    # Different hour - start fresh
                    self.coordinator._accumulation_start_time = None
                    self.coordinator._accumulated_energy_hour = 0.0
                    self.coordinator._accumulated_expected_energy_hour = 0.0
                    self.coordinator._accumulated_aux_impact_hour = 0.0
                    self.coordinator._accumulated_orphaned_aux = 0.0
                    self.coordinator._accumulated_aux_breakdown = {}
                    self.coordinator._hourly_delta_per_unit = {}
                    self.coordinator._hourly_expected_per_unit = {}
                    self.coordinator._last_minute_processed = None
                    # Mark baselines as stale so they are discarded after the
                    # unconditional last_energy_values load below.
                    stale_energy_baselines = True

            # Load hourly aggregates (with hour validation)
            aggregates = data.get("hourly_aggregates", {})
            if aggregates and aggregates.get("hour_start"):
                saved_hour = dt_util.parse_datetime(aggregates["hour_start"])
                if saved_hour and saved_hour.replace(minute=0, second=0, microsecond=0) == current_hour:
                    # Same hour - restore aggregates
                    self.coordinator._hourly_start_time = saved_hour
                    self.coordinator._hourly_wind_sum = aggregates.get("wind_sum", 0.0)
                    self.coordinator._hourly_wind_values = aggregates.get("wind_values", [])
                    self.coordinator._hourly_temp_sum = aggregates.get("temp_sum", 0.0)
                    self.coordinator._hourly_solar_sum = aggregates.get("solar_sum", 0.0)
                    self.coordinator._hourly_bucket_counts = aggregates.get("bucket_counts", {"normal": 0, "high_wind": 0, "extreme_wind": 0, "with_auxiliary_heating": 0})
                    self.coordinator._hourly_aux_count = aggregates.get("aux_count", 0)
                    self.coordinator._hourly_sample_count = aggregates.get("sample_count", 0)
                    _LOGGER.info(f"Restored {self.coordinator._hourly_sample_count} hourly samples from {saved_hour.strftime('%H:%M')}")

            # Restore current effective wind for sensors (prevent 0.0 drop on restart)
            current_wind = 0.0
            wind_restored = False

            # Priority 1: Last sample from current partial hour (if available)
            if self.coordinator._hourly_wind_values:
                current_wind = self.coordinator._hourly_wind_values[-1]
                wind_restored = True

            # Priority 2: Last completed hour (if no current samples)
            elif self.coordinator._hourly_log:
                last_log = self.coordinator._hourly_log[-1]
                current_wind = last_log.get("effective_wind", 0.0)
                wind_restored = True

            if wind_restored:
                self.coordinator.data["effective_wind"] = current_wind
                _LOGGER.info(f"Restored effective wind state: {current_wind:.2f} m/s")

            # Load last energy values
            loaded_last_values = data.get("last_energy_values", {})
            if isinstance(loaded_last_values, dict):
                self.coordinator._last_energy_values = loaded_last_values
                self._cleanup_removed_sensors(self.coordinator._last_energy_values)
            # Discard stale baselines from a previous hour to prevent cross-hour
            # delta accumulation on the first post-restart reading.
            if stale_energy_baselines:
                self.coordinator._last_energy_values = {}

            # Check if "accumulated_energy_today" is from today, else reset
            last_save_date = data.get("last_save_date")
            today_str = dt_util.now().date().isoformat()

            # Load persisted daily state variables
            self.coordinator.data[ATTR_TDD] = data.get("tdd_accumulated", data.get("hdd_accumulated", 0.0))
            if "forecast_today" in data:
                self.coordinator.data[ATTR_FORECAST_TODAY] = data["forecast_today"]
            if "predicted_kwh" in data:
                self.coordinator.data[ATTR_PREDICTED] = data["predicted_kwh"]
            self.coordinator.data[ATTR_TDD_DAILY_STABLE] = data.get("tdd_daily_stable", data.get("hdd_daily_stable", 0.0))
            if "temp_forecast_today" in data:
                self.coordinator.data[ATTR_TEMP_FORECAST_TODAY] = data["temp_forecast_today"]

            # Load cached forecast (Restore to Long Term Cache immediately)
            cached_daily_raw = data.get("cached_daily_forecast")
            cached_long_term_daily = data.get("cached_long_term_daily")
            cached_date = data.get("cached_forecast_date")

            if cached_daily_raw:
                # Always populate long term cache with available data (even if stale, better than nothing)
                self.coordinator.forecast._cached_long_term_hourly = cached_daily_raw
                # Performance optimization: Rebuild cache index immediately
                self.coordinator.forecast._rebuild_optimized_cache(cached_daily_raw)

                if cached_date == today_str:
                    self.coordinator.forecast._cached_forecast_date = cached_date
                    _LOGGER.info(f"Restored cached hourly forecast for {cached_date}")
                else:
                    _LOGGER.info(f"Restored stale hourly forecast from {cached_date} into long-term cache.")

            if cached_long_term_daily:
                self.coordinator.forecast._cached_long_term_daily = cached_long_term_daily
                _LOGGER.info(f"Restored cached daily forecast ({len(cached_long_term_daily)} items).")

            if last_save_date != today_str:
                _LOGGER.info(f"Daily data reset detected: last_save_date={last_save_date}, today={today_str}")
                self.coordinator._accumulated_energy_today = 0.0
                self.coordinator._daily_individual = {}
                self.coordinator.data[ATTR_TDD] = 0.0
                self.coordinator.data[ATTR_FORECAST_TODAY] = 0.0
                self.coordinator.data[ATTR_PREDICTED] = 0.0
                self.coordinator.data[ATTR_TDD_DAILY_STABLE] = 0.0
                self.coordinator.data[ATTR_TEMP_FORECAST_TODAY] = None

                self.coordinator._last_energy_values = {}
                yesterday = dt_util.now().date() - timedelta(days=1)
                self.coordinator._last_day_processed = yesterday

            # Load last hour stats
            self.coordinator.data[ATTR_LAST_HOUR_ACTUAL] = data.get("last_hour_actual", 0.0)
            self.coordinator.data[ATTR_LAST_HOUR_EXPECTED] = data.get("last_hour_expected", 0.0)
            self.coordinator.data[ATTR_LAST_HOUR_DEVIATION] = data.get("last_hour_deviation", 0.0)
            self.coordinator.data[ATTR_LAST_HOUR_DEVIATION_PCT] = data.get("last_hour_deviation_pct", 0.0)

            # Load configuration values
            if "learning_enabled" in data:
                self.coordinator.learning_enabled = data["learning_enabled"]
            if "auxiliary_heating_active" in data:
                self.coordinator.auxiliary_heating_active = data["auxiliary_heating_active"]
            if "aux_cooldown_active" in data:
                self.coordinator._aux_cooldown_active = data["aux_cooldown_active"]
            if "aux_cooldown_start_time" in data:
                try:
                    self.coordinator._aux_cooldown_start_time = dt_util.parse_datetime(data["aux_cooldown_start_time"])
                except (ValueError, TypeError):
                    self.coordinator._aux_cooldown_start_time = None

            if "learning_rate" in data:
                self.coordinator.learning_rate = data["learning_rate"]

            forecast_history_loaded = False
            if "forecast_history" in data:
                self.coordinator.forecast._forecast_history = data["forecast_history"]
                forecast_history_loaded = True
                # Migration: Convert "unknown" sources to "primary" for historical continuity
                # AND NEW: Populate source_breakdown from hourly logs where available
                for entry in self.coordinator.forecast._forecast_history:
                    if entry.get("source") == "unknown":
                        entry["source"] = "primary"

                    if "source_breakdown" not in entry:
                        date_key = entry.get("date")
                        if date_key:
                            breakdown = {}
                            for log in self.coordinator._hourly_log:
                                if log["timestamp"].startswith(date_key):
                                    src = log.get("forecast_source", "unknown")
                                    if src == "unknown":
                                        src = entry.get("source", "primary") # Fallback

                                    if src not in breakdown:
                                        breakdown[src] = {
                                            "hours": 0,
                                            "forecast": 0.0,
                                            "actual": 0.0,
                                            "error": 0.0,
                                            "abs_error": 0.0
                                        }

                                    h_forecast = log.get("forecasted_kwh", 0.0)
                                    h_actual = log.get("actual_kwh", 0.0)
                                    h_error = h_actual - h_forecast

                                    breakdown[src]["hours"] += 1
                                    breakdown[src]["forecast"] += h_forecast
                                    breakdown[src]["actual"] += h_actual
                                    breakdown[src]["error"] += h_error
                                    breakdown[src]["abs_error"] += abs(h_error)

                            if breakdown:
                                # Round breakdown values
                                for src in breakdown:
                                    breakdown[src]["forecast"] = round(breakdown[src]["forecast"], 2)
                                    breakdown[src]["actual"] = round(breakdown[src]["actual"], 2)
                                    breakdown[src]["error"] = round(breakdown[src]["error"], 2)
                                    breakdown[src]["abs_error"] = round(breakdown[src]["abs_error"], 2)
                                entry["source_breakdown"] = breakdown
                            else:
                                # Fallback to legacy if no hourly logs found for that date
                                src = entry.get("source", "primary")
                                entry["source_breakdown"] = {
                                    src: {
                                        "hours": 24, # Assumption for historical entry
                                        "forecast": entry.get("forecast_kwh", 0.0),
                                        "actual": entry.get("actual_kwh", 0.0),
                                        "error": entry.get("error_kwh", 0.0),
                                        "abs_error": entry.get("abs_error_kwh", 0.0)
                                    }
                                }
            if "midnight_forecast_snapshot" in data:
                self.coordinator.forecast._midnight_forecast_snapshot = data["midnight_forecast_snapshot"]
                # Publish to coordinator.data immediately so sensors have values before first update_daily_forecast
                snap = data["midnight_forecast_snapshot"]
                today_str_snap = dt_util.now().date().isoformat()
                if snap.get("date") == today_str_snap:
                    self.coordinator.data[ATTR_MIDNIGHT_FORECAST] = snap["kwh"]
                    self.coordinator.data[ATTR_MIDNIGHT_UNIT_ESTIMATES] = snap.get("unit_estimates")
                    self.coordinator.data[ATTR_MIDNIGHT_UNIT_MODES] = snap.get("unit_modes")
                    _LOGGER.info(f"Restored midnight forecast snapshot: {snap['kwh']:.2f} kWh")

            # Restore Reference and Live Forecasts (fixes persistence bug)
            ref_forecast = data.get("reference_forecast")
            if ref_forecast:
                self.coordinator.forecast._reference_forecast = ref_forecast
                # Rebuild optimized reference map immediately
                self.coordinator.forecast._build_reference_map(ref_forecast)
                _LOGGER.info(f"Restored reference forecast ({len(ref_forecast)} items)")

            p_ref_forecast = data.get("primary_reference_forecast")
            if p_ref_forecast:
                self.coordinator.forecast._primary_reference_forecast = p_ref_forecast
                _LOGGER.info(f"Restored primary reference forecast ({len(p_ref_forecast)} items)")

            s_ref_forecast = data.get("secondary_reference_forecast")
            if s_ref_forecast:
                self.coordinator.forecast._secondary_reference_forecast = s_ref_forecast
                _LOGGER.info(f"Restored secondary reference forecast ({len(s_ref_forecast)} items)")

            live_forecast = data.get("live_forecast")
            if live_forecast:
                self.coordinator.forecast._live_forecast = live_forecast
                # Update long term cache if empty
                if not self.coordinator.forecast._cached_long_term_hourly:
                    self.coordinator.forecast._cached_long_term_hourly = live_forecast
                    self.coordinator.forecast._rebuild_optimized_cache(live_forecast)

            last_live_update = data.get("last_live_update")
            if last_live_update:
                try:
                    self.coordinator.forecast._last_live_update = dt_util.parse_datetime(last_live_update)
                except (ValueError, TypeError):
                    pass

            # Option B: Backfill Forecast History from Hourly Log if empty
            # Only backfill if we didn't explicitly load an empty list (reset state).
            # If key was missing (forecast_history_loaded=False), we treat it as recovery mode.
            if not forecast_history_loaded and self.coordinator._hourly_log:
                self.coordinator.forecast.backfill_history_from_logs()

            # Initialize stats once loaded
            self.coordinator.statistics.calculate_temp_stats()
            self.coordinator.statistics.calculate_potential_savings()

            _LOGGER.info(f"Loaded correlation data: {len(self.coordinator._correlation_data)} temp points, {len(self.coordinator._daily_history)} days history")

        except Exception as e:
            _LOGGER.error(f"Error loading data: {e}", exc_info=True)
            # Notify user of unexpected load errors
            self.coordinator.hass.components.persistent_notification.create(
                title="Heating Analytics: Data Load Error",
                message=(
                    f"Failed to load stored data due to an unexpected error:\n\n"
                    f"```\n{str(e)}\n```\n\n"
                    f"**Heating Analytics will start with a fresh state.**\n\n"
                    f"Check the Home Assistant logs for details. "
                    f"If you have a backup, use the `heating_analytics.restore_data` service to restore."
                ),
                notification_id="heating_analytics_load_error"
            )
        finally:
             self.coordinator._is_loaded = True
             self.coordinator.forecast._cached_forecast_uncertainty = None
             self.coordinator.data["daily_individual"] = self.coordinator._daily_individual
             self.coordinator.data["lifetime_individual"] = self.coordinator._lifetime_individual

    def _minify_forecast_data(self, data: list[dict] | None) -> list[dict] | None:
        """Minify forecast data to reduce storage size."""
        if not data:
            return None

        # Only keep fields essential for energy calculation and display
        keep_keys = {
            "datetime", "temperature", "templow", "wind_speed", "wind_gust_speed",
            "condition", "cloud_coverage", "precipitation", "_source"
        }

        minified = []
        for item in data:
            new_item = {k: item[k] for k in keep_keys if k in item}
            minified.append(new_item)
        return minified

    async def async_save_data(self, force: bool = False):
        """Save data to storage with rate limiting."""
        async with self._save_lock:
            try:
                now = dt_util.now()
                if not force and self._last_save_time is not None:
                    elapsed = (now - self._last_save_time).total_seconds()
                    if elapsed < self._save_debounce_seconds:
                        _LOGGER.debug(f"Skipping save (rate limit): {elapsed:.1f}s < {self._save_debounce_seconds}s")
                        return

                # Pruning: Retain last 730 days (2 years) of daily history to prevent bloat
                history_retention_days = 730
                cutoff_date = (now.date() - timedelta(days=history_retention_days)).isoformat()

                # Optimized pruning: Only prune if size suggests it's needed (e.g., > 800 items)
                # This avoids iterating dictionary keys on every save
                if len(self.coordinator._daily_history) > (history_retention_days + 100):
                     keys_to_remove = [
                         k for k in self.coordinator._daily_history
                         if k < cutoff_date
                     ]
                     if keys_to_remove:
                         for k in keys_to_remove:
                             del self.coordinator._daily_history[k]
                         _LOGGER.info(f"Pruned {len(keys_to_remove)} old daily history entries (older than {cutoff_date})")

                data = {
                    "correlation_data": self.coordinator._correlation_data,
                    "correlation_data_per_unit": self.coordinator._correlation_data_per_unit,
                    "learning_buffer_per_unit": self.coordinator._learning_buffer_per_unit,
                    "observation_counts": self.coordinator._observation_counts,
                    "daily_history": self.coordinator._daily_history,
                    "hourly_log": self.coordinator._hourly_log,
                    "accumulated_energy_today": self.coordinator._accumulated_energy_today,
                    "daily_aux_breakdown": self.coordinator._daily_aux_breakdown,
                    "daily_individual": self.coordinator._daily_individual,
                    "lifetime_individual": self.coordinator._lifetime_individual,
                    "accumulated_energy_hour": self.coordinator._accumulated_energy_hour,
                    "accumulated_expected_energy_hour": self.coordinator._accumulated_expected_energy_hour,
                    "accumulated_aux_impact_hour": self.coordinator._accumulated_aux_impact_hour,
                    "accumulated_orphaned_aux": self.coordinator._accumulated_orphaned_aux,
                    "accumulated_aux_breakdown": self.coordinator._accumulated_aux_breakdown,
                    "hourly_delta_per_unit": self.coordinator._hourly_delta_per_unit,
                    "hourly_expected_per_unit": self.coordinator._hourly_expected_per_unit,
                    "last_minute_processed": self.coordinator._last_minute_processed,
                    # Persist critical state for robust gap filling (resolves race condition)
                    "current_model_rate": self.coordinator.data.get("current_model_rate", 0.0),
                    "current_unit_breakdown": self.coordinator.data.get("current_unit_breakdown", {}),
                    "current_aux_impact_rate": self.coordinator.data.get("current_aux_impact_rate", 0.0),
                    "current_calc_temp": self.coordinator.data.get("current_calc_temp"),
                    "accumulation_start_time": self.coordinator._accumulation_start_time.isoformat() if self.coordinator._accumulation_start_time else None,
                    "hourly_aggregates": {
                        "hour_start": self.coordinator._hourly_start_time.isoformat() if self.coordinator._hourly_start_time else None,
                        "wind_sum": self.coordinator._hourly_wind_sum,
                        "wind_values": self.coordinator._hourly_wind_values,
                        "temp_sum": self.coordinator._hourly_temp_sum,
                        "solar_sum": self.coordinator._hourly_solar_sum,
                        "bucket_counts": self.coordinator._hourly_bucket_counts,
                        "aux_count": self.coordinator._hourly_aux_count,
                        "sample_count": self.coordinator._hourly_sample_count,
                    },
                    "last_energy_values": self.coordinator._last_energy_values,
                    "last_save_date": dt_util.now().date().isoformat(),
                    "last_updated": dt_util.now().isoformat(),
                    "tdd_accumulated": self.coordinator.data.get(ATTR_TDD, 0.0),
                    "forecast_today": self.coordinator.data.get(ATTR_FORECAST_TODAY, 0.0),
                    "predicted_kwh": self.coordinator.data.get(ATTR_PREDICTED, 0.0),
                    "tdd_daily_stable": self.coordinator.data.get(ATTR_TDD_DAILY_STABLE, 0.0),
                    "temp_forecast_today": self.coordinator.data.get(ATTR_TEMP_FORECAST_TODAY),
                    "cached_daily_forecast": self._minify_forecast_data(self.coordinator.forecast._cached_long_term_hourly),
                    "cached_long_term_daily": self._minify_forecast_data(self.coordinator.forecast._cached_long_term_daily),
                    "cached_forecast_date": self.coordinator.forecast._cached_forecast_date,
                    "forecast_history": self.coordinator.forecast._forecast_history,
                    "midnight_forecast_snapshot": self.coordinator.forecast._midnight_forecast_snapshot,
                    "reference_forecast": self._minify_forecast_data(self.coordinator.forecast._reference_forecast),
                    "primary_reference_forecast": self._minify_forecast_data(self.coordinator.forecast._primary_reference_forecast),
                    "secondary_reference_forecast": self._minify_forecast_data(self.coordinator.forecast._secondary_reference_forecast),
                    "live_forecast": self._minify_forecast_data(self.coordinator.forecast._live_forecast),
                    "last_live_update": self.coordinator.forecast._last_live_update.isoformat() if self.coordinator.forecast._last_live_update else None,
                    "last_hour_actual": self.coordinator.data.get(ATTR_LAST_HOUR_ACTUAL, 0.0),
                    "last_hour_expected": self.coordinator.data.get(ATTR_LAST_HOUR_EXPECTED, 0.0),
                    "last_hour_deviation": self.coordinator.data.get(ATTR_LAST_HOUR_DEVIATION, 0.0),
                    "last_hour_deviation_pct": self.coordinator.data.get(ATTR_LAST_HOUR_DEVIATION_PCT, 0.0),
                    "learning_enabled": self.coordinator.learning_enabled,
                    "auxiliary_heating_active": self.coordinator.auxiliary_heating_active,
                    "aux_cooldown_active": self.coordinator._aux_cooldown_active,
                    "aux_cooldown_start_time": self.coordinator._aux_cooldown_start_time.isoformat() if self.coordinator._aux_cooldown_start_time else None,
                    "learning_rate": self.coordinator.learning_rate,
                    "solar_correction_percent": self.coordinator.solar_correction_percent,
                    "aux_coefficients": self.coordinator._aux_coefficients,
                    "aux_coefficients_per_unit": self.coordinator._aux_coefficients_per_unit,
                    "learning_buffer_global": self.coordinator._learning_buffer_global,
                    "learning_buffer_per_unit": self.coordinator._learning_buffer_per_unit,
                    "learning_buffer_aux_per_unit": self.coordinator._learning_buffer_aux_per_unit,
                    "solar_coefficients_per_unit": self.coordinator._solar_coefficients_per_unit,
                    "learning_buffer_solar_per_unit": self.coordinator._learning_buffer_solar_per_unit,
                    "unit_modes": self.coordinator._unit_modes,
                    "solar_optimizer_data": self.coordinator.solar_optimizer.get_data(),
                }
                await self._store.async_save(data)
                self._last_save_time = now
                _LOGGER.debug("Data saved successfully")
            except Exception as e:
                _LOGGER.error(f"Error saving data: {e}")

    async def async_reset_learning_data(self):
        """Reset the learning data (correlation model)."""
        self.coordinator._correlation_data.clear()
        self.coordinator._correlation_data_per_unit.clear()
        self.coordinator._learning_buffer_per_unit.clear()
        self.coordinator._observation_counts.clear()
        # Also clear unit aux data
        self.coordinator._aux_coefficients_per_unit.clear()
        self.coordinator._learning_buffer_aux_per_unit.clear()
        # Clear unit solar data
        self.coordinator._solar_coefficients_per_unit.clear()
        self.coordinator._learning_buffer_solar_per_unit.clear()
        await self.async_save_data(force=True)
        _LOGGER.info("Learning data reset successfully.")

    async def async_backup_data(self, file_path: str):
        """Backup full system state to JSON file."""
        _LOGGER.info(f"Starting full system backup to {file_path}")

        data = {
            "correlation_data": self.coordinator._correlation_data,
            "correlation_data_per_unit": self.coordinator._correlation_data_per_unit,
            "learning_buffer_per_unit": self.coordinator._learning_buffer_per_unit,
            "observation_counts": self.coordinator._observation_counts,
            "daily_history": self.coordinator._daily_history,
            "hourly_log": self.coordinator._hourly_log,
            "accumulated_energy_today": self.coordinator._accumulated_energy_today,
            "daily_aux_breakdown": self.coordinator._daily_aux_breakdown,
            "daily_individual": self.coordinator._daily_individual,
            "lifetime_individual": self.coordinator._lifetime_individual,
            "accumulated_energy_hour": self.coordinator._accumulated_energy_hour,
            "accumulated_expected_energy_hour": self.coordinator._accumulated_expected_energy_hour,
            "accumulated_aux_impact_hour": self.coordinator._accumulated_aux_impact_hour,
            "accumulated_orphaned_aux": self.coordinator._accumulated_orphaned_aux,
            "accumulated_aux_breakdown": self.coordinator._accumulated_aux_breakdown,
            "hourly_delta_per_unit": self.coordinator._hourly_delta_per_unit,
            "accumulation_start_time": self.coordinator._accumulation_start_time.isoformat() if self.coordinator._accumulation_start_time else None,
            "last_energy_values": self.coordinator._last_energy_values,
            "last_save_date": dt_util.now().date().isoformat(),
            "last_updated": dt_util.now().isoformat(),
            "tdd_accumulated": self.coordinator.data.get(ATTR_TDD, 0.0),
            "forecast_today": self.coordinator.data.get(ATTR_FORECAST_TODAY, 0.0),
            "tdd_daily_stable": self.coordinator.data.get(ATTR_TDD_DAILY_STABLE, 0.0),
            "temp_forecast_today": self.coordinator.data.get(ATTR_TEMP_FORECAST_TODAY),
            "cached_long_term_hourly": self.coordinator.forecast._cached_long_term_hourly,
            "cached_long_term_daily": self.coordinator.forecast._cached_long_term_daily,
            "cached_forecast_date": self.coordinator.forecast._cached_forecast_date,
            "last_hour_actual": self.coordinator.data.get(ATTR_LAST_HOUR_ACTUAL, 0.0),
            "last_hour_expected": self.coordinator.data.get(ATTR_LAST_HOUR_EXPECTED, 0.0),
            "last_hour_deviation": self.coordinator.data.get(ATTR_LAST_HOUR_DEVIATION, 0.0),
            "last_hour_deviation_pct": self.coordinator.data.get(ATTR_LAST_HOUR_DEVIATION_PCT, 0.0),
            "learning_enabled": self.coordinator.learning_enabled,
            "auxiliary_heating_active": self.coordinator.auxiliary_heating_active,
            "aux_cooldown_active": self.coordinator._aux_cooldown_active,
            "aux_cooldown_start_time": self.coordinator._aux_cooldown_start_time.isoformat() if self.coordinator._aux_cooldown_start_time else None,
            "learning_rate": self.coordinator.learning_rate,
            "solar_correction_percent": self.coordinator.solar_correction_percent,
            "aux_coefficients": self.coordinator._aux_coefficients,
            "aux_coefficients_per_unit": self.coordinator._aux_coefficients_per_unit,
            "learning_buffer_per_unit": self.coordinator._learning_buffer_per_unit,
            "learning_buffer_aux_per_unit": self.coordinator._learning_buffer_aux_per_unit,
            "solar_coefficients_per_unit": self.coordinator._solar_coefficients_per_unit,
            "learning_buffer_solar_per_unit": self.coordinator._learning_buffer_solar_per_unit,
        }

        def _write_json():
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

            with open(file_path, mode='w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)

        try:
            await self.coordinator.hass.async_add_executor_job(_write_json)
            _LOGGER.info(f"Backup completed successfully to {file_path}")
        except Exception as e:
            _LOGGER.error(f"Backup failed: {e}")
            raise e

    async def async_restore_data(self, file_path: str):
        """Restore full system state from JSON file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        _LOGGER.info(f"Starting full system restore from {file_path}")

        def _read_json():
            with open(file_path, mode='r', encoding='utf-8') as f:
                return json.load(f)

        try:
            data = await self.coordinator.hass.async_add_executor_job(_read_json)

            if "correlation_data" not in data and "daily_history" not in data:
                 raise ValueError("Invalid backup file: Missing critical keys.")

            # Apply Data
            self.coordinator._correlation_data.clear()
            self.coordinator._correlation_data.update(data.get("correlation_data", {}))

            self.coordinator._correlation_data_per_unit = data.get("correlation_data_per_unit", {})
            self.coordinator._learning_buffer_per_unit = data.get("learning_buffer_per_unit", {})
            self.coordinator._observation_counts = data.get("observation_counts", {})

            self.coordinator._daily_history = data.get("daily_history", {})
            self.coordinator._hourly_log = data.get("hourly_log", [])

            self.coordinator._accumulated_energy_today = data.get("accumulated_energy_today", 0.0)
            self.coordinator._daily_aux_breakdown = data.get("daily_aux_breakdown", {})
            self.coordinator._daily_individual = data.get("daily_individual", {})
            self.coordinator._lifetime_individual = data.get("lifetime_individual", {})

            self.coordinator._accumulated_energy_hour = data.get("accumulated_energy_hour", 0.0)
            self.coordinator._accumulated_expected_energy_hour = data.get("accumulated_expected_energy_hour", 0.0)
            self.coordinator._accumulated_aux_impact_hour = data.get("accumulated_aux_impact_hour", 0.0)
            self.coordinator._accumulated_orphaned_aux = data.get("accumulated_orphaned_aux", 0.0)
            self.coordinator._accumulated_aux_breakdown = data.get("accumulated_aux_breakdown", {})
            self.coordinator._hourly_delta_per_unit = data.get("hourly_delta_per_unit", {})
            self.coordinator._hourly_expected_per_unit = data.get("hourly_expected_per_unit", {})
            self.coordinator._last_minute_processed = data.get("last_minute_processed")

            acc_start_str = data.get("accumulation_start_time")
            if acc_start_str:
                self.coordinator._accumulation_start_time = dt_util.parse_datetime(acc_start_str)
            else:
                self.coordinator._accumulation_start_time = None

            self.coordinator._last_energy_values = data.get("last_energy_values", {})

            self.coordinator.data[ATTR_TDD] = data.get("tdd_accumulated", data.get("hdd_accumulated", 0.0))
            self.coordinator.data[ATTR_FORECAST_TODAY] = data.get("forecast_today", 0.0)
            self.coordinator.data[ATTR_TDD_DAILY_STABLE] = data.get("tdd_daily_stable", data.get("hdd_daily_stable", 0.0))
            self.coordinator.data[ATTR_TEMP_FORECAST_TODAY] = data.get("temp_forecast_today")

            self.coordinator.forecast._cached_long_term_hourly = data.get("cached_long_term_hourly")
            self.coordinator.forecast._cached_long_term_daily = data.get("cached_long_term_daily")
            self.coordinator.forecast._cached_forecast_date = data.get("cached_forecast_date")

            self.coordinator.data[ATTR_LAST_HOUR_ACTUAL] = data.get("last_hour_actual", 0.0)
            self.coordinator.data[ATTR_LAST_HOUR_EXPECTED] = data.get("last_hour_expected", 0.0)
            self.coordinator.data[ATTR_LAST_HOUR_DEVIATION] = data.get("last_hour_deviation", 0.0)
            self.coordinator.data[ATTR_LAST_HOUR_DEVIATION_PCT] = data.get("last_hour_deviation_pct", 0.0)

            if "learning_enabled" in data:
                self.coordinator.learning_enabled = data["learning_enabled"]
            if "auxiliary_heating_active" in data:
                self.coordinator.auxiliary_heating_active = data["auxiliary_heating_active"]
            if "aux_cooldown_active" in data:
                self.coordinator._aux_cooldown_active = data["aux_cooldown_active"]
            if "aux_cooldown_start_time" in data:
                try:
                    self.coordinator._aux_cooldown_start_time = dt_util.parse_datetime(data["aux_cooldown_start_time"])
                except (ValueError, TypeError):
                    self.coordinator._aux_cooldown_start_time = None

            if "learning_rate" in data:
                self.coordinator.learning_rate = data["learning_rate"]
            if "solar_correction_percent" in data:
                self.coordinator.solar_correction_percent = data["solar_correction_percent"]

            # Restore Unit Aux Data
            self.coordinator._aux_coefficients_per_unit = data.get("aux_coefficients_per_unit", {})
            self.coordinator._learning_buffer_aux_per_unit = data.get("learning_buffer_aux_per_unit", {})

            # Restore Unit Solar Data
            self.coordinator._solar_coefficients_per_unit = data.get("solar_coefficients_per_unit", {})
            self.coordinator._learning_buffer_solar_per_unit = data.get("learning_buffer_solar_per_unit", {})

            # Restore Unit Modes
            self.coordinator._unit_modes = data.get("unit_modes", {})

            # Restore Solar Optimizer
            self.coordinator.solar_optimizer.set_data(data.get("solar_optimizer_data", {}))

            await self.async_save_data(force=True)

            self.coordinator.statistics.calculate_temp_stats()
            self.coordinator.statistics.calculate_potential_savings()

            _LOGGER.info("System restored successfully.")

        except Exception as e:
            _LOGGER.error(f"Restore failed: {e}")
            raise e

    def _append_to_csv_with_schema_evolution(self, file_path: str, row: dict):
        """Append a row to CSV with schema evolution and safe writing."""
        # Defensive check: filter out None keys to prevent CSV writer errors
        invalid_keys = [k for k in row.keys() if k is None or not isinstance(k, str)]
        if invalid_keys:
            _LOGGER.warning(f"CSV row contains invalid keys (None or non-string): {invalid_keys}. Filtering them out.")
            row = {k: v for k, v in row.items() if k is not None and isinstance(k, str)}

        if not row:
            _LOGGER.error(f"CSV row is empty after filtering invalid keys. Skipping write to {file_path}")
            return

        current_keys = list(row.keys())
        existing_header = None
        rows_to_rewrite = None
        file_exists = os.path.isfile(file_path)

        if file_exists:
            try:
                with open(file_path, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    try:
                        existing_header = next(reader)
                    except StopIteration:
                        pass # Empty file
            except Exception as e:
                _LOGGER.error(f"Error reading CSV header from {file_path}: {e}")
                return

        if existing_header:
            # Check for new columns
            new_columns = [k for k in current_keys if k not in existing_header]

            if new_columns:
                _LOGGER.info(f"CSV Schema change detected for {file_path}. Adding columns: {new_columns}")

                # Read full file to rewrite
                try:
                    with open(file_path, 'r', newline='', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        rows_to_rewrite = list(reader)
                except Exception as e:
                     _LOGGER.error(f"Error reading full CSV for migration {file_path}: {e}")
                     return

                target_header = existing_header + new_columns

                # Atomic Write: Write to temp file then rename
                temp_path = file_path + ".tmp"
                try:
                    with open(temp_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=target_header)
                        writer.writeheader()
                        if rows_to_rewrite:
                            writer.writerows(rows_to_rewrite)
                        writer.writerow(row)

                    # Atomic replacement
                    os.replace(temp_path, file_path)

                except Exception as e:
                    _LOGGER.error(f"Error rewriting CSV {file_path}: {e}")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

            else:
                # APPEND SAFE (Use existing header order)
                try:
                    with open(file_path, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=existing_header)
                        writer.writerow(row)
                except Exception as e:
                    _LOGGER.error(f"Error appending to CSV {file_path}: {e}")
        else:
            # NEW FILE
            try:
                directory = os.path.dirname(file_path)
                if directory and not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)

                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=current_keys)
                    writer.writeheader()
                    writer.writerow(row)
            except Exception as e:
                _LOGGER.error(f"Error creating CSV {file_path}: {e}")

    async def append_hourly_log_csv(self, log_entry: dict):
        """Append a log entry to the hourly CSV file if enabled."""
        if not self.coordinator.csv_auto_logging:
            return

        file_path = self.coordinator.csv_hourly_path
        if not file_path:
            return

        def _write_row():
            # Flatten log entry (expand unit breakdown if needed)
            # We need to flatten unit_breakdown and unit_expected_breakdown
            row = {
                "timestamp": log_entry["timestamp"],
                "hour": log_entry["hour"],
                "temp": log_entry["temp"],
                "inertia_temp": log_entry["inertia_temp"],
                "effective_wind": log_entry["effective_wind"],
                "wind_bucket": log_entry["wind_bucket"],
                "actual_kwh": log_entry["actual_kwh"],
                "expected_kwh": log_entry["expected_kwh"],
                "forecasted_kwh": log_entry.get("forecasted_kwh", 0.0),
                "deviation": log_entry["deviation"],
                "auxiliary_active": log_entry["auxiliary_active"],
                "solar_factor": log_entry.get("solar_factor", 0.0),
                "solar_impact_kwh": log_entry.get("solar_impact_kwh", 0.0),
                "model_updated": log_entry.get("model_updated", False),
            }

            # Add columns for each configured sensor.
            for i, entity_id in enumerate(self.coordinator.energy_sensors):
                 # Actual
                 act = log_entry.get("unit_breakdown", {}).get(entity_id, 0.0)
                 row[f"unit_{i}_actual"] = act
                 # Expected
                 exp = log_entry.get("unit_expected_breakdown", {}).get(entity_id, 0.0)
                 row[f"unit_{i}_expected"] = exp

            self._append_to_csv_with_schema_evolution(file_path, row)

        try:
             await self.coordinator.hass.async_add_executor_job(_write_row)
        except Exception as e:
             _LOGGER.warning(f"Failed to write to hourly CSV: {e}")

    async def append_daily_log_csv(self, log_entry: dict):
        """Append a log entry to the daily CSV file if enabled."""
        if not self.coordinator.csv_auto_logging:
            return

        file_path = self.coordinator.csv_daily_path
        if not file_path:
            return

        def _write_row():
            self._append_to_csv_with_schema_evolution(file_path, log_entry)

        try:
             await self.coordinator.hass.async_add_executor_job(_write_row)
        except Exception as e:
             _LOGGER.warning(f"Failed to write to daily CSV: {e}")

    async def export_csv_data(self, file_path: str, export_type: str):
        """Export data to CSV."""
        _LOGGER.info(f"Exporting {export_type} data to {file_path}")

        def _write_csv():
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

            if export_type == "daily":
                data = self.coordinator._daily_history
                if not data:
                    return
                # Get superset of all keys to handle mixed legacy/new entries safely
                all_keys = set()
                has_vectors = False
                for day_data in data.values():
                    all_keys.update(day_data.keys())
                    if "hourly_vectors" in day_data and day_data["hourly_vectors"]:
                        has_vectors = True

                # Filter out complex fields (arrays/dicts) like hourly_vectors for CSV
                valid_fieldnames = ["date"]
                for key in sorted(list(all_keys)):
                    # Check if this key corresponds to a complex type in any of the entries
                    is_complex = False
                    for day_data in data.values():
                        val = day_data.get(key)
                        if val is not None:
                            if isinstance(val, (list, dict)):
                                is_complex = True
                            break # Optimization: check the first non-null occurrence to decide type

                    if not is_complex:
                        valid_fieldnames.append(key)
                    else:
                        _LOGGER.debug(f"Excluding complex field '{key}' from CSV export.")

                # Flatten Vectors if available
                vector_fields = []
                if has_vectors:
                    for h in range(24):
                        vector_fields.append(f"vec_temp_{h:02d}")
                        vector_fields.append(f"vec_wind_{h:02d}")
                        vector_fields.append(f"vec_actual_{h:02d}")
                    valid_fieldnames.extend(vector_fields)

                with open(file_path, mode='w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=valid_fieldnames)
                    writer.writeheader()
                    for date_str, stats in sorted(data.items()):
                        # Construct row with only valid fields
                        row = {"date": date_str}
                        for k in valid_fieldnames:
                            if k == "date" or k in vector_fields:
                                continue
                            val = stats.get(k)
                            # Double check safety
                            if not isinstance(val, (list, dict)):
                                row[k] = val

                        # Populate flattened vectors
                        vectors = stats.get("hourly_vectors")
                        if vectors and isinstance(vectors, dict):
                            v_temp = vectors.get("temp", [])
                            v_wind = vectors.get("wind", [])
                            # Handle both legacy 'load' and new 'actual_kwh'
                            v_actual = vectors.get("actual_kwh", vectors.get("load", []))

                            for h in range(24):
                                if h < len(v_temp): row[f"vec_temp_{h:02d}"] = v_temp[h]
                                if h < len(v_wind): row[f"vec_wind_{h:02d}"] = v_wind[h]
                                if h < len(v_actual): row[f"vec_actual_{h:02d}"] = v_actual[h]

                        writer.writerow(row)

            elif export_type == "hourly":
                data = self.coordinator._hourly_log
                if not data:
                    return
                # Use keys from first item
                # Need to handle nested dicts (unit_breakdown) if we want them in CSV
                # For simple export, we might skip nested or JSON dump them.
                # Let's flatten unit breakdown similar to append_hourly_log_csv
                if not data:
                    return

                # Determine columns dynamically
                # Base columns
                base_sample = data[0]
                fieldnames = [k for k in base_sample.keys() if not isinstance(base_sample[k], dict)]

                # Add unit columns
                units = self.coordinator.energy_sensors
                for i, eid in enumerate(units):
                    fieldnames.append(f"unit_{i}_actual")
                    fieldnames.append(f"unit_{i}_expected")

                with open(file_path, mode='w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for entry in data:
                        row = {k: v for k, v in entry.items() if k in fieldnames}
                        # Populate unit data
                        ub = entry.get("unit_breakdown", {})
                        ueb = entry.get("unit_expected_breakdown", {})
                        for i, eid in enumerate(units):
                            row[f"unit_{i}_actual"] = ub.get(eid, 0.0)
                            row[f"unit_{i}_expected"] = ueb.get(eid, 0.0)
                        writer.writerow(row)

        try:
            await self.coordinator.hass.async_add_executor_job(_write_csv)
            _LOGGER.info("CSV Export completed.")
        except Exception as e:
             _LOGGER.error(f"CSV Export failed: {e}")
             raise e

    async def import_csv_data(self, file_path: str, mapping: dict, update_model: bool = True):
        """Import historical data from CSV."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        _LOGGER.info(f"Starting CSV import from {file_path}. Update model: {update_model}")

        def _process_csv():
            """Process the CSV file in a background thread."""
            import csv

            col_ts = mapping.get("timestamp")
            col_temp = mapping.get("temperature")
            col_energy = mapping.get("energy")
            col_wind_speed = mapping.get("wind_speed")
            col_wind_gust = mapping.get("wind_gust")
            col_is_aux = mapping.get("is_auxiliary")
            col_cloud = mapping.get("cloud_coverage")

            if not all([col_ts, col_temp]):
                _LOGGER.error("CSV import mapping is missing required keys: 'timestamp', 'temperature'")
                return None, 0

            # Weather-only mode if energy column is missing
            weather_only_mode = col_energy is None
            if weather_only_mode:
                _LOGGER.info("CSV import in weather-only mode (no energy column). Will enrich existing data with weather information.")

            all_rows = []
            with open(file_path, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    all_rows.append(row)

            try:
                all_rows.sort(key=lambda r: dt_util.parse_datetime(r[col_ts]))
            except (KeyError, TypeError, ValueError) as e:
                _LOGGER.error(f"Failed to sort CSV data by timestamp using column '{col_ts}'. Check mapping and data format. Error: {e}")
                return None, 0

            new_log_entries = []
            learning_count = 0
            temp_history = []

            for row in all_rows:
                try:
                    ts = dt_util.parse_datetime(row[col_ts])
                    if not ts:
                        continue

                    # Ensure timezone awareness (assume local if naive)
                    if ts.tzinfo is None:
                        ts = dt_util.as_local(ts)

                    temp = float(row[col_temp])
                    kwh = None
                    if col_energy and row.get(col_energy):
                        kwh = float(row[col_energy])

                    wind_speed = float(row[col_wind_speed]) if col_wind_speed and row.get(col_wind_speed) else 0.0
                    wind_gust = float(row[col_wind_gust]) if col_wind_gust and row.get(col_wind_gust) else None
                    is_aux = str(row.get(col_is_aux, '0')).lower() in ['1', 'true', 'yes', 'on'] if col_is_aux else False

                    effective_wind = self.coordinator._calculate_effective_wind(wind_speed, wind_gust)
                    wind_bucket = self.coordinator._get_wind_bucket(effective_wind)

                    # Solar Factor Calculation
                    solar_factor = 0.0
                    solar_impact_kwh = 0.0
                    if self.coordinator.solar_enabled:
                        # Parse cloud coverage from CSV (optional)
                        cloud_coverage = None
                        if col_cloud and row.get(col_cloud):
                            try:
                                cloud_coverage = float(row[col_cloud])
                            except (ValueError, TypeError):
                                pass

                        # Fallback to 50% if not provided
                        if cloud_coverage is None:
                            cloud_coverage = 50.0

                        # Calculate solar factor from sun position + cloud coverage
                        elev, azim = self.coordinator.solar.get_approx_sun_pos(ts)
                        solar_factor = self.coordinator.solar.calculate_solar_factor(elev, azim, cloud_coverage)

                    # Weather-only mode: Store weather enrichment data
                    if weather_only_mode:
                        # Store weather data that will be used to update existing entries
                        entry = {
                            "timestamp": ts.isoformat(),
                            "hour": ts.hour,
                            "temp": temp,
                            "tdd": round(abs(self.coordinator.balance_point - temp) / 24.0, 3),
                            "effective_wind": effective_wind,
                            "wind_bucket": wind_bucket,
                            "solar_factor": round(solar_factor, 3),
                        }
                        new_log_entries.append(entry)
                    else:
                        # Full mode: Create complete entry with energy data
                        if kwh is None:
                            continue

                        entry = {
                            "timestamp": ts.isoformat(),
                            "hour": ts.hour,
                            "temp": temp,
                            "tdd": round(abs(self.coordinator.balance_point - temp) / 24.0, 3),
                            "effective_wind": effective_wind,
                            "wind_bucket": wind_bucket,
                            "actual_kwh": kwh,
                            "expected_kwh": 0.0,
                            "auxiliary_active": is_aux,
                            "solar_factor": round(solar_factor, 3),
                            "solar_impact_kwh": round(solar_impact_kwh, 3),
                        }
                        new_log_entries.append(entry)

                        # Model learning only for full mode
                        if update_model and kwh is not None:
                            if len(temp_history) >= 4:
                                temp_history.pop(0)
                            temp_history.append(temp)
                            inertia_avg = sum(temp_history) / len(temp_history)
                            temp_key = str(int(round(inertia_avg)))

                            status = self.coordinator.learning.learn_from_historical_import(
                                temp_key=temp_key,
                                wind_bucket=wind_bucket,
                                actual_kwh=kwh,
                                is_aux_active=is_aux,
                                correlation_data=self.coordinator._correlation_data,
                                aux_coefficients=self.coordinator._aux_coefficients,
                                learning_rate=self.coordinator.learning_rate,
                                get_predicted_kwh_fn=self.coordinator._get_predicted_kwh,
                                actual_temp=temp,
                            )
                            if "skipped" not in status:
                                learning_count += 1

                except (ValueError, KeyError) as e:
                    _LOGGER.warning(f"Skipping row in CSV due to processing error: {e}. Row: {row}")
                    continue

            return new_log_entries, learning_count

        try:
            entries, learned_count = await self.coordinator.hass.async_add_executor_job(_process_csv)

            if entries is None:
                return

            if entries:
                # Check if we're in weather-only mode by looking at first entry
                is_weather_only = "actual_kwh" not in entries[0]

                if is_weather_only:
                    # Weather-only mode: Update existing entries with weather data
                    updated_count = 0
                    timestamp_to_entry = {e['timestamp']: e for e in entries}

                    # Identify all days currently present in hourly_log to avoid conflict with backfill
                    existing_log_days = set(e['timestamp'][:10] for e in self.coordinator._hourly_log)

                    for log_entry in self.coordinator._hourly_log:
                        if log_entry['timestamp'] in timestamp_to_entry:
                            weather_data = timestamp_to_entry[log_entry['timestamp']]
                            # Update weather fields
                            log_entry['temp'] = weather_data['temp']
                            log_entry['tdd'] = weather_data['tdd']
                            log_entry['effective_wind'] = weather_data['effective_wind']
                            log_entry['wind_bucket'] = weather_data['wind_bucket']
                            log_entry['solar_factor'] = weather_data['solar_factor']
                            updated_count += 1

                    # ROTATED DATA HANDLING: Update daily_history for days NOT in hourly_log
                    # Group orphan entries by date
                    rotated_updates_count = 0
                    orphan_updates_by_date = {}

                    for ts_iso, entry in timestamp_to_entry.items():
                        date_str = ts_iso[:10]
                        # Only process if this day is completely missing from hourly_log
                        # (If it's present, _backfill will handle/overwrite it, so we skip)
                        if date_str not in existing_log_days:
                            if date_str not in orphan_updates_by_date:
                                orphan_updates_by_date[date_str] = []
                            orphan_updates_by_date[date_str].append(entry)

                    # Apply updates to daily_history
                    for date_str, daily_entries in orphan_updates_by_date.items():
                        # GUARD: Ensure we have enough hourly data to form valid vectors
                        if len(daily_entries) < 20:
                            _LOGGER.warning(
                                f"Weather-only import for {date_str}: Incomplete data ({len(daily_entries)}/24 hours). Skipping vector update to prevent skew."
                            )
                            continue

                        if date_str in self.coordinator._daily_history:
                            history_entry = self.coordinator._daily_history[date_str]
                            vectors = history_entry.get("hourly_vectors")

                            # Initialize vectors if missing (Legacy data support)
                            if not vectors or not isinstance(vectors, dict):
                                vectors = {
                                    "temp": [None] * 24,
                                    "wind": [None] * 24,
                                    "tdd": [None] * 24,
                                    "actual_kwh": [None] * 24,
                                }
                                if self.coordinator.solar_enabled:
                                    vectors["solar_rad"] = [None] * 24
                                history_entry["hourly_vectors"] = vectors

                            updated_day = False

                            for entry in daily_entries:
                                hour = entry["hour"]
                                if 0 <= hour <= 23:
                                    # Update Vectors
                                    vectors["temp"][hour] = entry["temp"]
                                    vectors["wind"][hour] = entry["effective_wind"]
                                    vectors["tdd"][hour] = entry["tdd"]
                                    if self.coordinator.solar_enabled and "solar_rad" in vectors:
                                        vectors["solar_rad"][hour] = entry["solar_factor"]
                                    updated_day = True

                            if updated_day:
                                # Re-calculate daily aggregates from updated vectors
                                # We use the vectors as source of truth now
                                valid_temps = [v for v in vectors["temp"] if v is not None]
                                valid_winds = [v for v in vectors["wind"] if v is not None]
                                valid_tdds = [v for v in vectors["tdd"] if v is not None]
                                valid_solars = [v for v in vectors.get("solar_rad", []) if v is not None]

                                if valid_temps:
                                    history_entry["temp"] = round(sum(valid_temps) / len(valid_temps), 1)
                                if valid_winds:
                                    history_entry["wind"] = round(sum(valid_winds) / len(valid_winds), 1)
                                if valid_tdds:
                                    history_entry["tdd"] = round(sum(valid_tdds), 1)
                                if valid_solars:
                                    history_entry["solar_factor"] = round(sum(valid_solars) / len(valid_solars), 3)

                                rotated_updates_count += 1

                    unique_entries = []  # No new entries in weather-only mode
                    _LOGGER.info(f"Weather-Only: Updated {updated_count} hourly logs. Patched {rotated_updates_count} daily history entries (rotated data).")
                else:
                    # Full mode: Add new entries as before
                    existing_timestamps = {log['timestamp'] for log in self.coordinator._hourly_log}
                    unique_entries = [e for e in entries if e['timestamp'] not in existing_timestamps]

                    self.coordinator._hourly_log.extend(unique_entries)
                    self.coordinator._hourly_log.sort(key=lambda x: x["timestamp"])

                # Determine which days need daily history updates
                if is_weather_only:
                    # For weather-only, rebuild daily history for all affected days
                    days_to_update = set(e["timestamp"][:10] for e in entries)
                else:
                    # For full mode, only rebuild for days with new entries
                    days_to_update = set(e["timestamp"][:10] for e in unique_entries)

                if days_to_update:
                    _LOGGER.info(f"Updating daily history for {len(days_to_update)} days.")
                    for day_str in sorted(days_to_update):
                        day_entries = [e for e in self.coordinator._hourly_log if e["timestamp"].startswith(day_str)]
                        if day_entries:
                            self.coordinator._daily_history[day_str] = self.coordinator._aggregate_daily_logs(day_entries)

                    # Backfill to ensure consistency and enrich any other potential partial days
                    self.coordinator._backfill_daily_from_hourly()

                await self.async_save_data(force=True)

                if is_weather_only:
                    summary = f"Weather-only import completed. Updated {updated_count} hourly entries with weather data."
                else:
                    summary = f"Imported {len(unique_entries)} new hourly entries."
                    if update_model:
                        summary += f" Model update will be based on {learned_count} valid entries in the next step."
                _LOGGER.info(summary)
            else:
                _LOGGER.info("No new entries to import from CSV.")

        except Exception as e:
            _LOGGER.error(f"CSV Import failed: {e}", exc_info=True)
            raise e
