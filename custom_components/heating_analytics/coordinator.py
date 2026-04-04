"""Coordinator for Heating Analytics."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, date
import math

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import (
    DataUpdateCoordinator,
)
from homeassistant.util import dt as dt_util
from homeassistant.const import UnitOfSpeed

from .helpers import convert_speed_to_ms, generate_exponential_kernel
from .solar import SolarCalculator
from .thermodynamics import ThermodynamicEngine
from .forecast import ForecastManager
from .statistics import StatisticsManager
from .learning import LearningManager
from .storage import StorageManager
from .solar_optimizer import SolarOptimizer
from .const import (
    DOMAIN,
    ATTR_EFFICIENCY,
    ATTR_PREDICTED,
    ATTR_SOLAR_PREDICTED,
    ATTR_DEVIATION,
    ATTR_TDD,
    ATTR_FORECAST_TODAY,
    ATTR_CORRELATION_DATA,
    ATTR_LAST_HOUR_ACTUAL,
    ATTR_LAST_HOUR_EXPECTED,
    ATTR_LAST_HOUR_DEVIATION,
    ATTR_LAST_HOUR_DEVIATION_PCT,
    ATTR_POTENTIAL_SAVINGS,
    ATTR_ENERGY_TODAY,
    ATTR_EXPECTED_TODAY,
    ATTR_TDD_YESTERDAY,
    ATTR_TDD_LAST_7D,
    ATTR_TDD_LAST_30D,
    ATTR_EFFICIENCY_YESTERDAY,
    ATTR_EFFICIENCY_LAST_7D,
    ATTR_EFFICIENCY_LAST_30D,
    ATTR_EFFICIENCY_FORECAST_TODAY,
    ATTR_TDD_DAILY_STABLE,
    ATTR_TEMP_LAST_YEAR_DAY,
    ATTR_TEMP_LAST_YEAR_WEEK,
    ATTR_TEMP_LAST_YEAR_MONTH,
    ATTR_TEMP_FORECAST_TODAY,
    ATTR_TEMP_ACTUAL_TODAY,
    ATTR_TEMP_ACTUAL_WEEK,
    ATTR_TEMP_ACTUAL_MONTH,
    ATTR_WIND_LAST_YEAR_DAY,
    ATTR_WIND_LAST_YEAR_WEEK,
    ATTR_WIND_LAST_YEAR_MONTH,
    ATTR_WIND_ACTUAL_TODAY,
    ATTR_WIND_ACTUAL_WEEK,
    ATTR_WIND_ACTUAL_MONTH,
    ATTR_SOLAR_FACTOR,
    ATTR_SOLAR_IMPACT,
    ATTR_TDD_SO_FAR,
    ATTR_MIDNIGHT_FORECAST,
    ATTR_FORECAST_UNCERTAINTY,
    ATTR_DEVIATION_BREAKDOWN,
    ATTR_FORECAST_BLEND_CONFIG,
    ATTR_FORECAST_ACCURACY_BY_SOURCE,
    ATTR_FORECAST_DETAILS,
    ATTR_SOLAR_POTENTIAL,
    ATTR_SOLAR_GAIN_NOW,
    ATTR_HEATING_LOAD_OFFSET,
    ATTR_RECOMMENDATION_STATE,
    RECOMMENDATION_MAXIMIZE_SOLAR,
    RECOMMENDATION_INSULATE,
    RECOMMENDATION_MITIGATE_SOLAR,
    DEFAULT_WIND_THRESHOLD,
    DEFAULT_WIND_GUST_FACTOR,
    DEFAULT_EXTREME_WIND_THRESHOLD,
    DEFAULT_WIND_UNIT,
    DEFAULT_MAX_ENERGY_DELTA,
    CONF_WIND_UNIT,
    CONF_ENABLE_LIFETIME_TRACKING,
    CONF_SOLAR_ENABLED,
    CONF_SOLAR_AZIMUTH,
    DEFAULT_SOLAR_ENABLED,
    DEFAULT_SOLAR_AZIMUTH,
    DEFAULT_SOLAR_CORRECTION,
    CONF_AUX_AFFECTED_ENTITIES,
    DEFAULT_SOLAR_LEARNING_RATE,
    DEFAULT_SOLAR_COEFF_HEATING,
    DEFAULT_SOLAR_COEFF_COOLING,
    ENERGY_GUARD_THRESHOLD,
    CLOUD_COVERAGE_MAP,
    WIND_UNIT_MS,
    WIND_UNIT_KMH,
    WIND_UNIT_KNOTS,
    LEARNING_BUFFER_THRESHOLD,
    CONF_OUTDOOR_TEMP_SOURCE,
    CONF_WIND_SOURCE,
    CONF_WIND_GUST_SOURCE,
    CONF_SECONDARY_WEATHER_ENTITY,
    CONF_FORECAST_CROSSOVER_DAY,
    DEFAULT_FORECAST_CROSSOVER_DAY,
    CONF_THERMAL_INERTIA,
    DEFAULT_THERMAL_INERTIA_HOURS,
    CONF_INDOOR_TEMP_SENSOR,
    CONF_THERMAL_MASS,
    DEFAULT_THERMAL_MASS,
    DEFAULT_DAILY_LEARNING_RATE,
    CONF_DAILY_LEARNING_MODE,
    CONF_TRACK_C,
    CONF_MPC_ENTRY_ID,
    CONF_MPC_MANAGED_SENSOR,
    SOURCE_SENSOR,
    SOURCE_WEATHER,
    MODE_HEATING,
    MODE_COOLING,
    MODE_OFF,
    MODE_GUEST_HEATING,
    MODE_GUEST_COOLING,
    MODE_DHW,
    MODES_EXCLUDED_FROM_GLOBAL_LEARNING,
    DEFAULT_CSV_AUTO_LOGGING,
    DEFAULT_CSV_HOURLY_PATH,
    DEFAULT_CSV_DAILY_PATH,
    DEFAULT_CLOUD_COVERAGE,
    MIXED_MODE_LOW,
    MIXED_MODE_HIGH,
    DUAL_INTERFERENCE_THRESHOLD,
    COOLDOWN_MIN_HOURS,
    COOLDOWN_MAX_HOURS,
    COOLDOWN_CONVERGENCE_THRESHOLD,
    SOLAR_BATTERY_DECAY,
)

_LOGGER = logging.getLogger(__name__)

class HeatingDataCoordinator(DataUpdateCoordinator):
    """Class to manage fetching data from the API."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize."""
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=timedelta(minutes=1),
        )
        self.entry = entry

        # Internal state
        self._model_cache = None  # Lazy-created by self.model property (#775)
        self._correlation_data = {} # { "temp": { "wind_bucket": avg_kwh_per_hour } }
        self._correlation_data_per_unit = {} # { entity_id: { "temp": { "wind_bucket": avg_kwh_per_hour } } }
        # { entity_id: { "temp_key": { "wind_bucket": count } } }
        self._observation_counts = {}
        self._aux_coefficients = {} # { "temp": kw_reduction }
        self._aux_coefficients_per_unit = {} # { entity_id: { "temp": { "wind_bucket": kw_reduction } } }

        self._unit_modes = {} # { entity_id: MODE_HEATING/MODE_COOLING/MODE_OFF }

        # New Per-Unit Solar State
        self._solar_coefficients_per_unit = {} # { entity_id: { "temp": coeff } }

        # Solar thermal battery: accumulates solar impact across hours with exponential decay.
        # Carries residual solar heat (stored in building mass) into post-solar hours.
        # Reset to 0 on restart — recovers within a few hours.
        self._solar_battery_state: float = 0.0

        # Hour-scoped accumulators — delegated to ObservationCollector (#775).
        # The collector owns all hourly state; the coordinator keeps
        # backward-compatible aliases that point to the SAME dict objects.
        # ObservationCollector.reset() clears containers in-place, so
        # these aliases remain valid across hour boundaries.
        from .observation import ObservationCollector
        self._collector = ObservationCollector()

        self._hourly_delta_per_unit = self._collector.delta_per_unit
        self._hourly_expected_per_unit = self._collector.expected_per_unit
        self._hourly_expected_base_per_unit = self._collector.expected_base_per_unit

        self._learning_buffer_per_unit = {} # { entity_id: { "temp": { "wind_bucket": [normalized_values] } } }
        self._learning_buffer_aux_per_unit = {} # { entity_id: { "temp": { "wind_bucket": [reduction_values] } } }

        # New Per-Unit Solar Buffer
        self._learning_buffer_solar_per_unit = {} # { entity_id: { "temp": [coeff_values] } }

        self._learning_buffer_global = {} # { "temp": { "wind_bucket": [normalized_values] } }

        self._collector.last_minute_processed = None

        self.data = {
            ATTR_EFFICIENCY: 0.0,
            ATTR_PREDICTED: 0.0,
            ATTR_DEVIATION: 0.0,
            ATTR_TDD: 0.0,
            ATTR_FORECAST_TODAY: 0.0,
            ATTR_LAST_HOUR_ACTUAL: 0.0,
            ATTR_LAST_HOUR_EXPECTED: 0.0,
            ATTR_LAST_HOUR_DEVIATION: 0.0,
            ATTR_LAST_HOUR_DEVIATION_PCT: 0.0,
            ATTR_POTENTIAL_SAVINGS: 0.0,
            ATTR_ENERGY_TODAY: 0.0,
            ATTR_EXPECTED_TODAY: 0.0,
            ATTR_CORRELATION_DATA: self._correlation_data,  # Direct reference
            ATTR_TDD_YESTERDAY: None,
            ATTR_TDD_LAST_7D: None,
            ATTR_TDD_LAST_30D: None,
            ATTR_EFFICIENCY_YESTERDAY: None,
            ATTR_EFFICIENCY_LAST_7D: None,
            ATTR_EFFICIENCY_LAST_30D: None,
            ATTR_EFFICIENCY_FORECAST_TODAY: None,
            ATTR_TDD_DAILY_STABLE: 0.0,
            ATTR_TEMP_LAST_YEAR_DAY: None,
            ATTR_TEMP_LAST_YEAR_WEEK: None,
            ATTR_TEMP_LAST_YEAR_MONTH: None,
            ATTR_TEMP_FORECAST_TODAY: None,
            ATTR_TEMP_ACTUAL_TODAY: None,
            ATTR_TEMP_ACTUAL_WEEK: None,
            ATTR_TEMP_ACTUAL_MONTH: None,
            ATTR_WIND_LAST_YEAR_DAY: None,
            ATTR_WIND_LAST_YEAR_WEEK: None,
            ATTR_WIND_LAST_YEAR_MONTH: None,
            ATTR_WIND_ACTUAL_TODAY: None,
            ATTR_WIND_ACTUAL_WEEK: None,
            ATTR_WIND_ACTUAL_MONTH: None,
            ATTR_SOLAR_FACTOR: 0.0,
            ATTR_SOLAR_IMPACT: 0.0,
            "accumulated_solar_impact_kwh": 0.0,
            "accumulated_guest_impact_kwh": 0.0,
            "accumulated_aux_impact_kwh": 0.0,
            ATTR_TDD_SO_FAR: 0.0,
            ATTR_DEVIATION_BREAKDOWN: [],
            "hourly_log": [],
            "daily_individual": {},
            "savings_theory_normal": 0.0,
            "savings_theory_aux": 0.0,
            "missing_aux_data": False,
        }
        self._daily_history = {} # { "YYYY-MM-DD": { "kwh": float, "temp": float, "tdd": float } }
        self._daily_individual = {} # { entity_id: kwh_today }
        self._lifetime_individual = {} # { entity_id: kwh_lifetime }
        self._hourly_log = [] # List of dicts for hourly stats
        self._last_hour_processed = None
        self._last_day_processed = None
        self._last_energy_values = {} # { entity_id: value_at_last_update }
        self._learned_u_coefficient = None
        self._last_midnight_indoor_temp = None

        # Hourly aggregates are now owned by ObservationCollector (see above).
        # Only daily-scoped accumulators remain here.
        self._accumulated_energy_today = 0.0
        self._daily_aux_breakdown = {} # { entity_id: { "allocated": 0.0, "overflow": 0.0 } } - Daily accumulator
        self._daily_orphaned_aux = 0.0
        self._accumulation_start_time = None

        # Configuration parameters
        self.outdoor_temp_sensor = entry.data.get("outdoor_temp_sensor")
        self.indoor_temp_sensor = entry.data.get(CONF_INDOOR_TEMP_SENSOR)
        self.thermal_mass_kwh_per_degree = entry.data.get(CONF_THERMAL_MASS, DEFAULT_THERMAL_MASS)
        self.daily_learning_mode = entry.data.get(CONF_DAILY_LEARNING_MODE, False)
        self.track_c_enabled = entry.data.get(CONF_TRACK_C, False)
        self.mpc_entry_id = entry.data.get(CONF_MPC_ENTRY_ID)
        self.mpc_managed_sensor = entry.data.get(CONF_MPC_MANAGED_SENSOR)
        self.wind_speed_sensor = entry.data.get("wind_speed_sensor")
        self.wind_gust_sensor = entry.data.get("wind_gust_sensor")

        # Source Configuration (Default to SENSOR for backward compatibility if key missing,
        # though migration should handle it)
        self.outdoor_temp_source = entry.data.get(CONF_OUTDOOR_TEMP_SOURCE, SOURCE_SENSOR)
        self.wind_speed_source = entry.data.get(CONF_WIND_SOURCE, SOURCE_SENSOR)
        self.wind_gust_source = entry.data.get(CONF_WIND_GUST_SOURCE, SOURCE_SENSOR)

        self.weather_entity = entry.data.get("weather_entity")
        self.energy_sensors = entry.data.get("energy_sensors", [])

        # Per-unit learning strategies (#776) — must be after energy_sensors,
        # track_c_enabled, and mpc_managed_sensor are set.
        from .observation import build_strategies
        self._unit_strategies = build_strategies(
            energy_sensors=self.energy_sensors,
            track_c_enabled=self.track_c_enabled,
            mpc_managed_sensor=self.mpc_managed_sensor,
        )

        self.wind_gust_factor = entry.data.get("wind_gust_factor", DEFAULT_WIND_GUST_FACTOR)
        self.balance_point = entry.data.get("balance_point", 17.0)
        self.learning_rate = entry.data.get("learning_rate", 0.01)

        # New Config Params
        self.wind_threshold = entry.data.get("wind_threshold", DEFAULT_WIND_THRESHOLD)
        self.extreme_wind_threshold = entry.data.get("extreme_wind_threshold", DEFAULT_EXTREME_WIND_THRESHOLD)
        self.wind_unit = entry.data.get(CONF_WIND_UNIT, DEFAULT_WIND_UNIT)
        self.max_energy_delta = entry.data.get("max_energy_delta", DEFAULT_MAX_ENERGY_DELTA)
        self.enable_lifetime_tracking = entry.data.get(CONF_ENABLE_LIFETIME_TRACKING, False)

        # Solar Config
        self.solar_enabled = entry.data.get(CONF_SOLAR_ENABLED, DEFAULT_SOLAR_ENABLED)
        self.solar_azimuth = entry.data.get(CONF_SOLAR_AZIMUTH, DEFAULT_SOLAR_AZIMUTH)
        self.solar_correction_percent = DEFAULT_SOLAR_CORRECTION

        # Load Thermal Inertia Profile (Default to Normal/4h if missing)
        inertia_setting = self.entry.data.get(CONF_THERMAL_INERTIA, DEFAULT_THERMAL_INERTIA_HOURS)

        # Migration for old string values
        if isinstance(inertia_setting, str):
            if inertia_setting == "fast":
                inertia_setting = 2
            elif inertia_setting == "slow":
                inertia_setting = 12
            else:
                inertia_setting = 4

        # Store tau for use in gap detection (_get_inertia_parameters)
        self.inertia_tau = float(inertia_setting)
        # Generate weights dynamically; cap window at 5×tau (captures 99.3% of weight)
        # to avoid requesting hundreds of hours of history for large tau values.
        self.inertia_weights = generate_exponential_kernel(
            tau=self.inertia_tau,
            window_hours=min(int(inertia_setting * 5), 168),
        )

        # Load Aux Affected Entities (Default to all energy sensors if missing)
        self.aux_affected_entities = self.entry.data.get(CONF_AUX_AFFECTED_ENTITIES)
        if self.aux_affected_entities is None:
            # Backward compatibility: Assume all units are affected if not configured
            # Note: We respect explicit empty list [] as "No Units Affected"
            self.aux_affected_entities = self.energy_sensors

        # Optimization: Maintain a set for O(1) lookups in high-frequency loops
        self._aux_affected_set = set(self.aux_affected_entities) if self.aux_affected_entities else set()

        self.solar = SolarCalculator(self)
        self.solar_optimizer = SolarOptimizer(self)
        self.forecast = ForecastManager(self)
        self.statistics = StatisticsManager(self)
        self.learning = LearningManager()
        self.storage = StorageManager(self)

        # CSV Auto-logging configuration
        self.csv_auto_logging = entry.data.get("csv_auto_logging", DEFAULT_CSV_AUTO_LOGGING)
        self.csv_hourly_path = entry.data.get("csv_hourly_path", DEFAULT_CSV_HOURLY_PATH)
        self.csv_daily_path = entry.data.get("csv_daily_path", DEFAULT_CSV_DAILY_PATH)

        # Dynamic State Flags
        self.auxiliary_heating_active = False
        self.learning_enabled = True
        self._aux_cooldown_active = False
        self._aux_cooldown_start_time = None

        self._is_loaded = False

    async def _async_load_data(self):
        """Load data from storage."""
        await self.storage.async_load_data()
        updated_count = self._backfill_daily_from_hourly()
        if updated_count > 0:
            _LOGGER.info(f"Backfilled/Enriched {updated_count} daily history entries. Saving data.")
            await self._async_save_data(force=True)

    async def _async_save_data(self, force: bool = False):
        """Save data to storage."""
        await self.storage.async_save_data(force)

    async def async_reset_learning_data(self):
        """Reset the learning data (correlation model) and refresh all sensors."""
        await self.storage.async_reset_learning_data()
        await self.async_refresh()

    async def async_reset_forecast_accuracy(self):
        """Reset the forecast accuracy history."""
        self.forecast.reset_forecast_history()
        # Save explicitly to persist the empty state (prevents backfill on reload)
        await self._async_save_data(force=True)

    async def set_auxiliary_heating_active(self, active: bool):
        """Set auxiliary heating active state and manage cooldown transitions."""
        if self.auxiliary_heating_active == active:
            return

        # Detect Transition: True -> False (Aux Turned Off)
        if self.auxiliary_heating_active and not active:
            _LOGGER.info("Auxiliary Heating turned OFF. Initiating Model Cooldown/Decay Lock.")
            self._aux_cooldown_active = True
            self._aux_cooldown_start_time = dt_util.now()

        # Update State
        if active and self._aux_cooldown_active:
            _LOGGER.info("Aux re-activated during cooldown. Cancelling cooldown.")
            self._aux_cooldown_active = False
            self._aux_cooldown_start_time = None

        self.auxiliary_heating_active = active
        await self._async_save_data()

        # Trigger immediate UI update
        self.async_set_updated_data(self.data)

    async def async_exit_cooldown(self):
        """Manually exit auxiliary cooldown state."""
        if not self._aux_cooldown_active:
            _LOGGER.info("exit_cooldown called but no active cooldown found - ignoring")
            return
        _LOGGER.info("Manually exiting auxiliary cooldown")
        self._aux_cooldown_active = False
        self._aux_cooldown_start_time = None
        await self._async_save_data()
        self.async_set_updated_data(self.data)

    def get_unit_mode(self, entity_id: str) -> str:
        """Get current mode for a unit."""
        return self._unit_modes.get(entity_id, MODE_HEATING)

    async def async_set_unit_mode(self, entity_id: str, mode: str):
        """Set mode for a unit."""
        if entity_id not in self.energy_sensors:
            _LOGGER.warning(f"Attempted to set mode for unknown entity: {entity_id}")
            return

        self._unit_modes[entity_id] = mode
        await self._async_save_data()

        self.async_set_updated_data(self.data)

    async def async_reset_unit_learning_data(self, entity_id: str):
        """Reset learning data for a specific unit."""
        if not entity_id:
            _LOGGER.warning("Attempted to reset unit learning with empty entity_id")
            return

        _LOGGER.info(f"Resetting learning data for unit: {entity_id}")

        # Clear Model Data
        if entity_id in self._correlation_data_per_unit:
            del self._correlation_data_per_unit[entity_id]
            _LOGGER.debug(f"Cleared correlation data for {entity_id}")

        # Clear Unit Aux Data
        if entity_id in self._aux_coefficients_per_unit:
            del self._aux_coefficients_per_unit[entity_id]
            _LOGGER.debug(f"Cleared aux coefficients for {entity_id}")

        # Clear Unit Solar Data
        if entity_id in self._solar_coefficients_per_unit:
            del self._solar_coefficients_per_unit[entity_id]
            _LOGGER.debug(f"Cleared solar coefficients for {entity_id}")

        # Clear Observation Counts
        if entity_id in self._observation_counts:
            del self._observation_counts[entity_id]
            _LOGGER.debug(f"Cleared observation counts for {entity_id}")

        # Note: We do NOT clear _learning_buffer_per_unit, _learning_buffer_aux_per_unit, or _learning_buffer_solar_per_unit.
        # This allows any pre-existing buffer to be applied during the next "Cold Start" cycle.
        # If the user wanted to clear the buffer, they would need a separate action,
        # but the request is specifically "so that the buffered learning can be reapplied".

        await self._async_save_data(force=True)

    async def async_reset_solar_learning_data(self, entity_id: str | None = None):
        """Reset solar learning data for a specific unit or all units."""
        if entity_id:
            if entity_id in self._solar_coefficients_per_unit:
                del self._solar_coefficients_per_unit[entity_id]
                _LOGGER.debug(f"Cleared solar coefficients for {entity_id}")
            if entity_id in self._learning_buffer_solar_per_unit:
                del self._learning_buffer_solar_per_unit[entity_id]
                _LOGGER.debug(f"Cleared solar learning buffer for {entity_id}")
            _LOGGER.info(f"Solar learning reset for unit: {entity_id}")
        else:
            self._solar_coefficients_per_unit.clear()
            self._learning_buffer_solar_per_unit.clear()
            _LOGGER.info("Solar learning reset for all units")

        await self._async_save_data(force=True)

    async def async_migrate_aux_coefficients(self, new_aux_affected_entities: list[str]):
        """Migrate auxiliary coefficients when the affected entities list changes.

        Implements the 'Conservation Strategy':
        When a unit is removed from the aux list, its accumulated auxiliary reduction (coefficients)
        is redistributed to the remaining units to maintain the global Total_Aux_Reduction.

        Redistribution is proportional to the remaining units' existing coefficients (strength).
        Fallback: If no remaining units have data for a specific bucket, assign to the Primary Unit
        (highest total consumption).
        """
        old_list = self.aux_affected_entities or []
        # Identify removed units (In old but not in new)
        removed_units = [u for u in old_list if u not in new_aux_affected_entities]

        if not removed_units:
            _LOGGER.debug("Aux Migration: No units removed, skipping conservation logic.")
            # We still update the internal list to match the pending config change
            self.aux_affected_entities = new_aux_affected_entities
            # Also update the optimized set
            self._aux_affected_set = set(new_aux_affected_entities) if new_aux_affected_entities else set()
            return

        _LOGGER.info(f"Aux Migration: Removed units {removed_units}. Beginning conservation strategy.")

        # Helper to find Primary Unit among remaining
        def _get_primary_unit(candidates: list[str]) -> str | None:
            best_unit = None
            max_kwh = -1.0
            for entity_id in candidates:
                val = self._get_float_state(entity_id)
                if val is None: val = 0.0
                if val > max_kwh:
                    max_kwh = val
                    best_unit = entity_id

            if best_unit is None and candidates:
                return candidates[0]
            return best_unit

        primary_unit = _get_primary_unit(new_aux_affected_entities)

        changes_made = False

        for removed_unit in removed_units:
            if removed_unit not in self._aux_coefficients_per_unit:
                continue

            unit_data = self._aux_coefficients_per_unit[removed_unit]

            # Iterate through all temp keys and wind buckets for this unit
            for temp_key, wind_data in unit_data.items():
                for wind_bucket, coeff_value in wind_data.items():
                    if coeff_value <= 0:
                        continue

                    # We have a value to redistribute
                    # 1. Calculate total weight of remaining units for this specific bucket
                    total_weight = 0.0
                    eligible_weights = {}

                    for target_unit in new_aux_affected_entities:
                        # Get existing coeff
                        w = 0.0
                        if target_unit in self._aux_coefficients_per_unit:
                            if temp_key in self._aux_coefficients_per_unit[target_unit]:
                                w = self._aux_coefficients_per_unit[target_unit][temp_key].get(wind_bucket, 0.0)

                        eligible_weights[target_unit] = w
                        total_weight += w

                    # 2. Distribute
                    if total_weight > 0:
                        # Proportional Redistribution
                        for target_unit, weight in eligible_weights.items():
                            if weight > 0:
                                share = (weight / total_weight) * coeff_value
                                self._add_aux_coefficient(target_unit, temp_key, wind_bucket, share)
                                changes_made = True
                    else:
                        # Fallback: Assign to Primary Unit
                        if primary_unit:
                            _LOGGER.debug(f"Aux Migration: Fallback to primary {primary_unit} for {temp_key}/{wind_bucket} (Val: {coeff_value})")
                            self._add_aux_coefficient(primary_unit, temp_key, wind_bucket, coeff_value)
                            changes_made = True

            # Clean up the removed unit
            del self._aux_coefficients_per_unit[removed_unit]
            changes_made = True

        if changes_made:
            _LOGGER.info("Aux Migration: Completed successfully. Saving data.")
            await self._async_save_data(force=True)

        self.aux_affected_entities = new_aux_affected_entities
        self._aux_affected_set = set(new_aux_affected_entities) if new_aux_affected_entities else set()

    def _add_aux_coefficient(self, entity_id: str, temp_key: str, wind_bucket: str, value_to_add: float):
        """Helper to safely add value to a coefficient, clamped to base model."""
        if entity_id not in self._aux_coefficients_per_unit:
            self._aux_coefficients_per_unit[entity_id] = {}
        if temp_key not in self._aux_coefficients_per_unit[entity_id]:
            self._aux_coefficients_per_unit[entity_id][temp_key] = {}

        current = self._aux_coefficients_per_unit[entity_id][temp_key].get(wind_bucket, 0.0)
        new_val = current + value_to_add

        # Clamp: Aux coefficient cannot exceed base model for this unit/temp/wind
        # (aux reduction cannot be greater than what the unit would use without aux)
        unit_base_data = self._correlation_data_per_unit.get(entity_id, {})
        if temp_key in unit_base_data and wind_bucket in unit_base_data[temp_key]:
            base_limit = unit_base_data[temp_key][wind_bucket]
            new_val = min(new_val, base_limit)

        self._aux_coefficients_per_unit[entity_id][temp_key][wind_bucket] = round(new_val, 3)


    async def async_replace_sensor_source(self, old_entity_id: str, new_entity_id: str) -> bool:
        """Replace a heating unit's energy sensor with a new one, migrating all data."""
        if not old_entity_id or not new_entity_id:
             _LOGGER.error("Replace Sensor Source: Missing old or new entity ID.")
             return False

        if old_entity_id not in self.energy_sensors:
             _LOGGER.debug(f"Replace Sensor Source: Old entity '{old_entity_id}' not found in this coordinator. Skipping.")
             return False

        if new_entity_id in self.energy_sensors:
             _LOGGER.error(f"Replace Sensor Source: New entity '{new_entity_id}' is already configured in this coordinator. Aborting to prevent data corruption.")
             return False

        # Verify new entity exists in HA state machine
        if not self.hass.states.get(new_entity_id):
            _LOGGER.error(f"Replace Sensor Source: New entity '{new_entity_id}' does not exist in Home Assistant. Please check the entity ID.")
            return False

        _LOGGER.info(f"Replacing sensor source: '{old_entity_id}' -> '{new_entity_id}'")

        # 1. Update In-Memory Configuration
        try:
            self.energy_sensors.remove(old_entity_id)
        except ValueError:
            pass

        if new_entity_id not in self.energy_sensors:
            self.energy_sensors.append(new_entity_id)

        # Update Aux Affected Entities if applicable
        if self.aux_affected_entities is not None and old_entity_id in self.aux_affected_entities:
            try:
                self.aux_affected_entities.remove(old_entity_id)
                self._aux_affected_set.discard(old_entity_id)
            except ValueError:
                pass
            if new_entity_id not in self.aux_affected_entities:
                self.aux_affected_entities.append(new_entity_id)
                self._aux_affected_set.add(new_entity_id)

        # 2. Update Persistent Configuration (Config Entry)
        new_data = self.entry.data.copy()
        new_data["energy_sensors"] = self.energy_sensors
        if CONF_AUX_AFFECTED_ENTITIES in new_data:
            new_data[CONF_AUX_AFFECTED_ENTITIES] = self.aux_affected_entities
        self.hass.config_entries.async_update_entry(self.entry, data=new_data)
        _LOGGER.info(f"Config entry updated. New sensor list: {self.energy_sensors}")

        # 3. Migrate Dictionary Data
        def _migrate_key(d: dict, old_k: str, new_k: str):
            if old_k in d:
                # If new key exists, we overwrite it (assuming replacement intent)
                # or we could merge? For simplicity and safety of history, we overwrite/move.
                d[new_k] = d.pop(old_k)

        _migrate_key(self._correlation_data_per_unit, old_entity_id, new_entity_id)
        _migrate_key(self._aux_coefficients_per_unit, old_entity_id, new_entity_id) # New: Migrate unit aux coeffs
        _migrate_key(self._solar_coefficients_per_unit, old_entity_id, new_entity_id) # New: Migrate unit solar coeffs
        _migrate_key(self._unit_modes, old_entity_id, new_entity_id) # New: Migrate unit modes
        _migrate_key(self._observation_counts, old_entity_id, new_entity_id)
        _migrate_key(self._hourly_delta_per_unit, old_entity_id, new_entity_id)
        _migrate_key(self._hourly_expected_per_unit, old_entity_id, new_entity_id)
        _migrate_key(self._hourly_expected_base_per_unit, old_entity_id, new_entity_id)
        _migrate_key(self._learning_buffer_per_unit, old_entity_id, new_entity_id)
        _migrate_key(self._learning_buffer_aux_per_unit, old_entity_id, new_entity_id) # New: Migrate unit aux buffer
        _migrate_key(self._learning_buffer_solar_per_unit, old_entity_id, new_entity_id) # New: Migrate unit solar buffer
        _migrate_key(self._daily_individual, old_entity_id, new_entity_id)
        _migrate_key(self._lifetime_individual, old_entity_id, new_entity_id)

        # 4. Handle Last Energy Values (Baseline Reset)
        # We DELETE the old key to remove the old baseline.
        # We do NOT set the new key yet. This forces the system to treat the new sensor
        # as a "Fresh Start" on the next update loop, establishing a new baseline from its current value.
        # This prevents massive spikes from Old=5000kWh and New=0kWh.
        if old_entity_id in self._last_energy_values:
            del self._last_energy_values[old_entity_id]
            _LOGGER.info(f"Removed baseline for '{old_entity_id}'. New sensor '{new_entity_id}' will initialize baseline on next update.")

        # 5. Migrate Hourly Log History (Preserve Statistics)
        # We iterate through history and rename the keys in the breakdown dicts.
        count_migrated = 0
        for entry in self._hourly_log:
            # Unit Breakdown (Actual)
            if "unit_breakdown" in entry:
                _migrate_key(entry["unit_breakdown"], old_entity_id, new_entity_id)

            # Unit Expected Breakdown
            if "unit_expected_breakdown" in entry:
                _migrate_key(entry["unit_expected_breakdown"], old_entity_id, new_entity_id)

            count_migrated += 1

        _LOGGER.info(f"Migrated stats in {count_migrated} hourly log entries.")

        # 6. Save Everything
        await self._async_save_data(force=True)
        _LOGGER.info("Sensor replacement and data migration completed successfully.")

        # 7. Refresh to establish new baseline immediately
        # This prevents a "zero" period or drift until the next scheduled update.
        await self.async_refresh()

        # 8. Save again to persist the new baseline established by refresh
        # If we reload immediately after this without saving, the new baseline (in memory) is lost.
        await self._async_save_data(force=True)
        _LOGGER.info("New sensor baseline saved to disk.")

        return True

    async def async_backup_data(self, file_path: str):
        """Backup full system state to JSON file."""
        await self.storage.async_backup_data(file_path)

    async def async_restore_data(self, file_path: str):
        """Restore full system state from JSON file."""
        await self.storage.async_restore_data(file_path)

    async def export_csv_data(self, file_path: str, export_type: str):
        """Export data to CSV."""
        await self.storage.export_csv_data(file_path, export_type)

    async def import_csv_data(self, file_path: str, mapping: dict, update_model: bool = True):
        """Import historical data from CSV."""
        await self.storage.import_csv_data(file_path, mapping, update_model)

    async def retrain_from_history(self, days_back: int | None = None, reset_first: bool = False) -> dict:
        """Retrain the learning model from existing hourly log data.

        Track A (daily_learning_mode=False): replays each logged hour through
        learn_from_historical_import(), honouring aux/base routing.

        Track B (daily_learning_mode=True): groups hours by day and applies the
        same midnight EMA logic as the live calibration. Thermal mass correction
        is applied when an indoor_temp_sensor is configured AND the hourly log
        contains 'indoor_temp' entries; otherwise it is skipped gracefully.
        """
        if reset_first:
            self._correlation_data.clear()
            self._correlation_data_per_unit.clear()
            self._aux_coefficients.clear()
            self._aux_coefficients_per_unit.clear()
            self._learning_buffer_global.clear()
            self._learning_buffer_per_unit.clear()
            self._learning_buffer_aux_per_unit.clear()
            self._solar_coefficients_per_unit.clear()
            self._learning_buffer_solar_per_unit.clear()
            self._observation_counts.clear()
            self._learned_u_coefficient = None
            self._invalidate_model_cache()
            _LOGGER.info("retrain_from_history: Model reset before retraining.")

        if days_back is not None:
            from homeassistant.util import dt as dt_util
            from datetime import timedelta
            cutoff_str = (dt_util.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            entries = [e for e in self._hourly_log if e.get("timestamp", "") >= cutoff_str]
        else:
            entries = list(self._hourly_log)

        if not entries:
            return {"status": "no_data", "entries_processed": 0, "days_processed": 0, "learning_count": 0}

        learning_count = 0

        def _is_poisoned(entry: dict, daily_mode: bool = False) -> bool:
            s = entry.get("learning_status", "unknown")
            # In daily_learning_mode, "disabled" is the normal state for every hour
            # (Track A is blocked from writing to correlation_data). These hours
            # contain valid sensor data and must NOT be filtered out for Track B/C
            # daily aggregation.  Only genuine data-quality issues are poisoned.
            if daily_mode and s == "disabled":
                return False
            return s == "disabled" or s.startswith("skipped_") or s == "cooldown_post_aux"

        if self.daily_learning_mode:
            # Daily learning: batch aggregation per calendar day using strategy dispatch (#776).
            # Poisoned hours are excluded before grouping so the <22-hour guard
            # automatically rejects days with too many bad hours.
            daily_batches: dict[str, list] = {}
            for entry in entries:
                if not _is_poisoned(entry, daily_mode=True):
                    daily_batches.setdefault(entry["timestamp"][:10], []).append(entry)

            days_processed = 0
            for date_str, day_entries in sorted(daily_batches.items()):
                if len(day_entries) < 22:
                    _LOGGER.debug(f"retrain_from_history: Skipping {date_str} — {len(day_entries)}/24 hours")
                    continue

                total_kwh = sum(e.get("actual_kwh", 0.0) for e in day_entries)
                daily_tdd = sum(e.get("tdd", 0.0) for e in day_entries)

                if daily_tdd < 0.5 or total_kwh <= 0:
                    continue

                # Mode filtering (#789): exclude OFF/DHW/Guest/Cooling from retrain.
                excluded_kwh = self._compute_excluded_mode_energy(day_entries)
                total_kwh -= excluded_kwh

                if total_kwh <= 0:
                    continue

                # Determine q_adjusted for U-coefficient.
                track_c_daily = self._daily_history.get(date_str, {}).get("track_c_kwh")
                if self.track_c_enabled and track_c_daily is not None:
                    q_adjusted = track_c_daily
                    # Backward compat: days stored before non-MPC inclusion.
                    if self.mpc_managed_sensor and "track_c_kwh_non_mpc" not in self._daily_history.get(date_str, {}):
                        non_mpc_retrain = 0.0
                        for log_entry in day_entries:
                            breakdown = log_entry.get("unit_breakdown", {})
                            for sid in self.energy_sensors:
                                if sid != self.mpc_managed_sensor:
                                    non_mpc_retrain += breakdown.get(sid, 0.0)
                        q_adjusted += non_mpc_retrain
                else:
                    q_adjusted = total_kwh
                    if self.thermal_mass_kwh_per_degree > 0.0:
                        from datetime import date as _date, timedelta as _td
                        prev_day_str = (
                            _date.fromisoformat(date_str) - _td(days=1)
                        ).isoformat()
                        end_temp = self._daily_history.get(date_str, {}).get("midnight_indoor_temp")
                        start_temp = self._daily_history.get(prev_day_str, {}).get("midnight_indoor_temp")
                        if end_temp is not None and start_temp is not None:
                            delta_t_indoor = end_temp - start_temp
                            q_adjusted = total_kwh - (self.thermal_mass_kwh_per_degree * delta_t_indoor)

                if q_adjusted <= 0:
                    continue

                # U-coefficient update.
                observed_u = q_adjusted / daily_tdd
                if self._learned_u_coefficient is None:
                    self._learned_u_coefficient = observed_u
                else:
                    self._learned_u_coefficient += DEFAULT_DAILY_LEARNING_RATE * (
                        observed_u - self._learned_u_coefficient
                    )

                # Bucket learning: Track C uses strategy dispatch, Track B uses flat daily.
                track_c_dist = self._daily_history.get(date_str, {}).get("track_c_distribution")
                if track_c_dist:
                    self._apply_strategies_to_global_model(day_entries, track_c_dist)
                else:
                    # Track B flattened daily bucket learning.
                    q_hourly_avg = q_adjusted / 24.0
                    avg_temp_retrain = sum(e.get("temp", 0.0) for e in day_entries) / len(day_entries)
                    daily_wind_retrain = sum(e.get("effective_wind", 0.0) for e in day_entries) / len(day_entries)
                    flat_temp_key = str(int(round(avg_temp_retrain)))
                    flat_wind_bucket = self._get_wind_bucket(daily_wind_retrain)

                    if flat_temp_key not in self._correlation_data:
                        self._correlation_data[flat_temp_key] = {}
                    current_pred = self._correlation_data[flat_temp_key].get(flat_wind_bucket, 0.0)

                    if current_pred == 0.0:
                        self._correlation_data[flat_temp_key][flat_wind_bucket] = round(q_hourly_avg, 5)
                    else:
                        new_pred = current_pred + self.learning_rate * (q_hourly_avg - current_pred)
                        self._correlation_data[flat_temp_key][flat_wind_bucket] = round(new_pred, 5)

                # Per-unit model replay for DirectMeter sensors (needed for isolate_sensor).
                self._replay_per_unit_models(day_entries)

                learning_count += 1
                days_processed += 1

            self.data["learned_u_coefficient"] = self._learned_u_coefficient
            self._invalidate_model_cache()
            await self.storage.async_save_data(force=True)

            _LOGGER.info(
                f"retrain_from_history: Completed. "
                f"{days_processed} days learned, U-coefficient={self._learned_u_coefficient}"
            )
            return {
                "status": "completed",
                "mode": "strategy_dispatch",
                "entries_processed": len(entries),
                "days_processed": days_processed,
                "learning_count": learning_count,
                "learned_u_coefficient": round(self._learned_u_coefficient, 4) if self._learned_u_coefficient is not None else None,
            }

        else:
            # Track A: per-hour replay
            temp_history: list[float] = []
            skipped = 0

            for entry in entries:
                actual_kwh = entry.get("actual_kwh")
                if actual_kwh is None:
                    skipped += 1
                    continue

                temp = entry.get("temp", 0.0)
                wind_bucket = entry.get("wind_bucket", "normal")
                is_aux = entry.get("auxiliary_active", False)

                if len(temp_history) >= 4:
                    temp_history.pop(0)
                temp_history.append(temp)

                # Skip poisoned hours — but only after updating temp_history so the
                # inertia sliding window stays accurate for subsequent hours.
                if _is_poisoned(entry):
                    skipped += 1
                    continue

                inertia_avg = sum(temp_history) / len(temp_history)
                temp_key = str(int(round(inertia_avg)))

                status = self.learning.learn_from_historical_import(
                    temp_key=temp_key,
                    wind_bucket=wind_bucket,
                    actual_kwh=actual_kwh,
                    is_aux_active=is_aux,
                    correlation_data=self._correlation_data,
                    aux_coefficients=self._aux_coefficients,
                    learning_rate=self.learning_rate,
                    get_predicted_kwh_fn=self._get_predicted_kwh,
                    actual_temp=temp,
                )
                if "skipped" not in status:
                    learning_count += 1
                else:
                    skipped += 1

            self._invalidate_model_cache()
            await self.storage.async_save_data(force=True)

            _LOGGER.info(
                f"retrain_from_history Track A: Completed. "
                f"{learning_count} entries learned, {skipped} skipped."
            )
            return {
                "status": "completed",
                "mode": "track_a_hourly",
                "entries_processed": len(entries),
                "days_processed": len({e["timestamp"][:10] for e in entries}),
                "learning_count": learning_count,
                "skipped": skipped,
            }

    def _get_float_state(self, entity_id: str) -> float | None:
        """Helper to get float state from an entity."""
        if not entity_id:
            return None
        state = self.hass.states.get(entity_id)
        if state and state.state not in ("unknown", "unavailable"):
            try:
                return float(state.state)
            except ValueError:
                pass
        return None

    def _get_cloud_coverage(self) -> float:
        """Get cloud coverage in percent (0-100)."""
        if not self.weather_entity:
            return 50.0 # Conservative default

        state = self.hass.states.get(self.weather_entity)
        if not state:
            return 50.0

        # Try to find numeric attribute first
        cloud_attr = state.attributes.get("cloud_coverage")
        if cloud_attr is not None:
            try:
                val = float(cloud_attr)
                return max(0.0, min(100.0, val))
            except ValueError:
                pass

        # Fallback: Map condition state
        condition = state.state
        if condition in CLOUD_COVERAGE_MAP:
            return float(CLOUD_COVERAGE_MAP[condition])

        return DEFAULT_CLOUD_COVERAGE

    def _get_speed_in_ms(self, entity_id: str) -> float | None:
        """Get speed in m/s from an entity, converting from km/h, mph, or knots if needed."""
        if not entity_id:
            return None
        state = self.hass.states.get(entity_id)
        if state and state.state not in ("unknown", "unavailable"):
            try:
                value = float(state.state)
                unit = state.attributes.get("unit_of_measurement")
                return convert_speed_to_ms(value, unit)
            except ValueError:
                pass
        return None

    def _get_weather_attribute(self, attribute: str) -> float | None:
        """Get attribute from weather entity, handling units if necessary."""
        if not self.weather_entity:
            return None

        state = self.hass.states.get(self.weather_entity)
        if not state:
            return None

        val = state.attributes.get(attribute)
        if val is None:
            return None

        try:
            val = float(val)
        except (ValueError, TypeError):
            return None

        # Handle Wind Units
        if attribute in ("wind_speed", "wind_gust_speed"):
            unit = state.attributes.get("wind_speed_unit")
            return convert_speed_to_ms(val, unit)

        return val

    def _get_inertia_parameters(self) -> tuple[int, int]:
        """Helper to derive requirements from weights."""
        hours_back = len(self.inertia_weights) - 1
        # max_gap is the stale-data cutoff (gap detection), tied to tau – not the kernel window.
        # Logs older than tau hours are considered a thermal discontinuity and discarded.
        max_gap = int(self.inertia_tau)
        return hours_back, max_gap

    def _calculate_weighted_inertia(self, temps: list[float]) -> float:
        """Calculate weighted inertia temperature.

        Aligns available temps (Newest -> Oldest) to weights (Newest -> Oldest).
        Normalizes weights to 1.0 based on available samples.
        """
        if not temps:
            return 0.0

        # Weights are defined Oldest -> Newest (Time increasing)
        # e.g. [0.1, 0.2, 0.3, 0.4]
        # temps list is also Oldest -> Newest (History + Current)
        # e.g. [12, 11, 10] (H-2, H-1, Current)

        # We want to match Newest Temp to Newest Weight.
        # Temp[-1] matches Weight[-1]
        # Temp[-2] matches Weight[-2]

        num_temps = len(temps)
        num_weights = len(self.inertia_weights)

        # Take the last N weights where N = num_temps
        # But limited by num_weights (shouldn't happen if logic is correct, but safe)
        count = min(num_temps, num_weights)

        active_temps = temps[-count:]
        active_weights = self.inertia_weights[-count:]

        total_weight = sum(active_weights)
        if total_weight == 0:
            return sum(active_temps) / count # Fallback to simple average

        weighted_sum = sum(t * w for t, w in zip(active_temps, active_weights))
        return weighted_sum / total_weight

    def _get_recent_log_temps(self, reference_time: datetime, hours_back: int | None = None, max_gap_hours: int | None = None) -> list[float]:
        """Get recent temperatures from log, ensuring no thermodynamic discontinuity.

        Args:
            reference_time: The point in time to look backwards from.
            hours_back: Number of past hourly samples to retrieve (target). Defaults to derived from DEFAULT_INERTIA_WEIGHTS.
            max_gap_hours: Maximum allowable age of a log sample. If older, it's ignored.
                           This protects against using data from before a long downtime.
                           Defaults to derived from DEFAULT_INERTIA_WEIGHTS.
        """
        def_hours, def_gap = self._get_inertia_parameters()

        if hours_back is None:
            hours_back = def_hours
        if max_gap_hours is None:
            max_gap_hours = def_gap

        temps = []
        if not self._hourly_log:
            return temps

        # Calculate threshold: Logs older than max_gap_hours should be ignored
        cutoff_time = reference_time - timedelta(hours=max_gap_hours)

        # Filter FIRST: Select all logs that are recent (within tolerance)
        valid_logs = []
        # Optimization: Iterate backwards as logs are sorted by time (newest last)
        for log in reversed(self._hourly_log):
            try:
                timestamp_str = log.get("timestamp")
                if timestamp_str:
                    log_dt = dt_util.parse_datetime(timestamp_str)
                    if log_dt:
                        # Ensure timezone awareness for comparison
                        if log_dt.tzinfo is None and reference_time.tzinfo:
                            log_dt = log_dt.replace(tzinfo=reference_time.tzinfo)

                        if log_dt >= cutoff_time:
                            valid_logs.append(log)
                        else:
                            # Since we iterate backwards, once we hit an old log, we can stop
                            # assuming logs are sorted.
                            break
            except (ValueError, TypeError):
                pass

        # Restore chronological order (oldest first)
        valid_logs.reverse()

        # Take last N valid logs
        recent_logs = valid_logs[-hours_back:]
        for log in recent_logs:
            temps.append(log["temp"])

        return temps

    def _get_inertia_list(self, current_time: datetime) -> list[float]:
        """Get full list of temperatures for inertia calculation (History + Current).

        Centralizes the logic used by forecast seeding and inertia calculations.
        Returns the raw list of temperatures [H-3, H-2, H-1, Current].
        """
        # Get valid history using helper
        temps = self._get_recent_log_temps(current_time)

        # Add current temp (Strict Source Logic)
        current_temp = None

        if self.outdoor_temp_source == SOURCE_SENSOR and self.outdoor_temp_sensor:
            # Prefer hourly rolling average for consistency with historical log entries,
            # which are stored as hourly averages. Fall back to raw sensor reading only
            # if no samples have been collected yet this hour.
            if self._collector.sample_count > 0:
                current_temp = self._collector.temp_sum / self._collector.sample_count
            else:
                current_temp = self._get_float_state(self.outdoor_temp_sensor)

        elif self.outdoor_temp_source == SOURCE_WEATHER or (
            self.outdoor_temp_source == SOURCE_SENSOR and not self.outdoor_temp_sensor
        ):
             # Use Weather (or if Sensor source selected but no sensor configured - edge case)
             current_temp = self._get_weather_attribute("temperature")

        if current_temp is not None:
            temps.append(current_temp)

        # If list is empty but we have a running average (edge case?), return that seeded
        # Handled by caller if needed.

        return temps

    def _calculate_inertia_temp(self) -> float | None:
        """Calculate the inertia temperature (Weighted Average).

        Uses the last N-1 valid hours of history from hourly_log + current/latest known temp.
        N = len(self.inertia_weights).
        """
        current_time = dt_util.now()
        temps = self._get_inertia_list(current_time)

        if not temps:
            return None

        # Short-circuit: no history available, return current temp directly
        # (avoids floating-point rounding from kernel normalisation)
        if len(temps) == 1:
            return temps[0]

        return self._calculate_weighted_inertia(temps)

    def _calculate_effective_wind(self, speed: float, gust: float | None) -> float:
        """Calculate effective wind speed.

        Formula: wind_speed + (wind_gust - wind_speed) * gust_factor
        We treat wind_speed as the base load, and add a weighted component of the gust (turbulence)
        that exceeds the average speed.
        """
        # Ensure non-negative speed
        speed = max(0.0, speed)

        if gust is None:
            return speed

        gust = max(0.0, gust)

        # Calculate turbulence component (only if gust > speed)
        turbulence = max(0.0, gust - speed)

        return speed + (turbulence * self.wind_gust_factor)

    def _get_wind_bucket(self, effective_wind: float) -> str:
        """Determine wind bucket."""
        if effective_wind >= self.extreme_wind_threshold:
            return "extreme_wind"
        elif effective_wind >= self.wind_threshold:
            return "high_wind"
        return "normal"

    def _get_weather_wind_unit(self) -> str | None:
        """Get wind speed unit from weather entity."""
        if not self.weather_entity:
            return None
        weather_state = self.hass.states.get(self.weather_entity)
        if not weather_state:
            return None
        return weather_state.attributes.get("wind_speed_unit")

    def _get_sun_info_now(self):
        """Get current sun elevation and azimuth from HA."""
        sun_state = self.hass.states.get("sun.sun")
        if not sun_state:
            return 0.0, 0.0

        try:
            elev = float(sun_state.attributes.get("elevation", 0))
            azim = float(sun_state.attributes.get("azimuth", 0))
            return elev, azim
        except (ValueError, TypeError):
            return 0.0, 0.0

    def _update_live_predictions(self, calc_temp: float | None, temp_key: str, wind_bucket: str, current_time: datetime) -> float:
        """Update live prediction models and accumulate expected energy."""
        effective_wind = self.data.get("effective_wind", 0.0)

        # 1. Use Centralized Robust Calculation (StatisticsManager)
        # This replaces manual calls to _get_predicted_kwh, _get_aux_impact_kw, etc.
        # and automatically handles aggregation of units.

        solar_impact_global = self.data.get(ATTR_SOLAR_IMPACT, 0.0)

        current_prediction_rate = 0.0
        unit_breakdown = {}
        aux_impact_rate = 0.0
        result = {}

        if calc_temp is not None:
            result = self.statistics.calculate_total_power(
                calc_temp,
                effective_wind,
                solar_impact_global,
                is_aux_active=self.auxiliary_heating_active
            )

            current_prediction_rate = result["total_kwh"]
            unit_breakdown = result["unit_breakdown"]
            # Kelvin Protocol: Use Global Authority for Aux Impact
            # We must use the Global Model value, not the Sum of Units (which is subject to exclusions)
            # STRICT MODE: Default to 0.0 if global model is missing, do NOT fallback to unit sum.
            aux_impact_rate = result.get("global_aux_reduction_kwh", 0.0)

            # Sync solar impact to data for sensor exposure
            if self.solar_enabled:
                 self.data[ATTR_SOLAR_IMPACT] = result["breakdown"]["solar_reduction_kwh"]

        # Store Current Model Rate and Aux state for gap filling
        self.data["current_model_rate"] = current_prediction_rate
        self.data["current_calc_temp"] = calc_temp
        self.data["current_aux_impact_rate"] = aux_impact_rate
        self.data["current_unit_breakdown"] = unit_breakdown

        # Intra-hour accumulation (delegated to ObservationCollector)
        if self._collector.last_minute_processed != current_time.minute:
            # Calculate time step (minutes) to account for gaps (e.g., restart, downtime)
            minutes_step = 1  # Default to 1 minute
            if self._collector.last_minute_processed is not None:
                diff = current_time.minute - self._collector.last_minute_processed
                if diff > 0:
                    minutes_step = diff
                # If diff < 0 (hour rollover?), we rely on start_time reset logic
                # in _process_hourly_data to handle the fresh start, so minutes_step=1 is fine.
            else:
                # First run in hour (e.g., after restart)
                # Minute is 0-indexed, so :30 means 31 minutes have passed in this hour slot.
                minutes_step = current_time.minute + 1

            fraction = minutes_step / 60.0
            orphaned_part = result.get("breakdown", {}).get("orphaned_aux_savings", 0.0)

            self._collector.accumulate_expected(
                fraction=fraction,
                prediction_rate=current_prediction_rate,
                aux_impact_rate=aux_impact_rate,
                unit_breakdown=unit_breakdown,
                orphaned_part=orphaned_part,
            )

            # Mark minute as processed
            self._collector.last_minute_processed = current_time.minute

        return current_prediction_rate

    def _get_partial_log_for_current_hour(self) -> dict | None:
        """Retrieve partial log entry for the current hour if it exists (e.g. pre-restart data)."""
        if not self._hourly_log:
            return None

        current_time = dt_util.now()
        current_hour = current_time.hour
        today_iso = current_time.date().isoformat()

        # Check the most recent log entry
        last_entry = self._hourly_log[-1]
        if last_entry["hour"] == current_hour and last_entry["timestamp"].startswith(today_iso):
            return last_entry

        return None

    def _close_hour_gap(
        self,
        current_time: datetime,
        last_minute: int,
        avg_temp: float = 0.0,
        avg_wind: float = 0.0,
        avg_solar: float = 0.0,
        is_aux_active: bool = False,
        avg_solar_vector: tuple[float, float] | None = None
    ):
        """Close the gap at hour boundary by filling missing minutes.

        If the last processed minute was 58, and we are now in the next hour,
        we missed minute 59 (and potentially the 59->00 transition).
        This method calculates the missing expected energy for the tail of the previous hour
        to ensure the hourly log sums to exactly 60 minutes of expectation.

        Kelvin Protocol Improvement: Mean Imputation
        Uses hourly aggregates (Temp, Wind, Solar, Aux) to calculate the gap energy.
        This ensures consistency between the logged hour stats and the total accumulated energy,
        eliminating "Desync" caused by using point-in-time rates from the end of the hour.
        """
        # Calculate how many minutes were missed at the end of the previous hour
        # Example: last=58. Missed 59. (60 - (58+1)) = 1 minute.
        minutes_missing = 60 - (last_minute + 1)

        if minutes_missing <= 0:
            return

        fraction = minutes_missing / 60.0

        # Calculate Gap Power using Aggregates
        result = self.statistics.calculate_total_power(
            avg_temp,
            avg_wind,
            0.0, # solar_impact (ignored when override is used)
            is_aux_active=is_aux_active,
            override_solar_factor=avg_solar,
            override_solar_vector=avg_solar_vector,
            detailed=True
        )

        current_rate = result["total_kwh"]
        unit_breakdown = result["unit_breakdown"]
        aux_impact_rate = result.get("global_aux_reduction_kwh", 0.0)

        if current_rate > 0:
            self._collector.expected_energy_hour += (current_rate * fraction)
            _LOGGER.debug(
                f"Closed hour gap: Added {minutes_missing} min using aggregates "
                f"(Temp={avg_temp:.1f}, Wind={avg_wind:.1f}, Rate={current_rate:.2f} kW)"
            )

            # Accumulate gap aux impact
            if aux_impact_rate > 0:
                 self._collector.aux_impact_hour += (aux_impact_rate * fraction)

            # Distribute gap energy to units
            for entity_id, stats in unit_breakdown.items():
                unit_pred = stats["net_kwh"]
                if unit_pred > 0:
                    if entity_id not in self._hourly_expected_per_unit:
                        self._hourly_expected_per_unit[entity_id] = 0.0
                    self._hourly_expected_per_unit[entity_id] += (unit_pred * fraction)

                unit_base = stats.get("base_kwh", 0.0)
                if unit_base > 0:
                    if entity_id not in self._hourly_expected_base_per_unit:
                        self._hourly_expected_base_per_unit[entity_id] = 0.0
                    self._hourly_expected_base_per_unit[entity_id] += (unit_base * fraction)

                # Accumulate Gap Aux Breakdown
                if entity_id not in self._collector.aux_breakdown:
                    self._collector.aux_breakdown[entity_id] = {"allocated": 0.0, "overflow": 0.0}

                applied_aux = stats.get("aux_reduction_kwh", 0.0)
                overflow_aux = stats.get("overflow_kwh", 0.0)

                self._collector.aux_breakdown[entity_id]["allocated"] += (applied_aux * fraction)
                self._collector.aux_breakdown[entity_id]["overflow"] += (overflow_aux * fraction)

    def _calculate_daily_wind_penalty(self) -> float:
        """Calculate the total kWh penalty due to wind for the entire day (Past + Future)."""
        total_penalty = 0.0
        today_iso = dt_util.now().date().isoformat()

        # 1. Past Hours (Completed)
        for log in self._hourly_log:
            if not log["timestamp"].startswith(today_iso):
                continue

            # Recalculate model with 0 wind vs Actual Wind
            # Note: We use the stored temperature and auxiliary state for consistency.
            temp = log["temp"]
            aux_active = log.get("auxiliary_active", False)
            solar_impact = 0.0 # Solar cancels out in difference, keep 0 for simplicity

            # Use logged solar factor to ensure correct saturation logic
            s_factor = log.get("solar_factor", 0.0)

            # Use logged unit modes if available, else standard inference
            unit_modes = log.get("unit_modes")

            # Scenario 1: Actual Wind
            # Use 'effective_wind' from log
            eff_wind = log.get("effective_wind", 0.0)

            s_vector_s = log.get("solar_vector_s")
            s_vector_e = log.get("solar_vector_e")
            s_vector = (s_vector_s, s_vector_e) if s_vector_s is not None and s_vector_e is not None else None

            res_actual = self.statistics.calculate_total_power(
                temp=temp,
                effective_wind=eff_wind,
                solar_impact=solar_impact,
                is_aux_active=aux_active,
                unit_modes=unit_modes,
                override_solar_factor=s_factor,
                override_solar_vector=s_vector,
                detailed=False
            )

            # Scenario 2: No Wind
            res_no_wind = self.statistics.calculate_total_power(
                temp=temp,
                effective_wind=0.0,
                solar_impact=solar_impact,
                is_aux_active=aux_active,
                unit_modes=unit_modes,
                override_solar_factor=s_factor,
                override_solar_vector=s_vector,
                detailed=False
            )

            # Penalty = Energy(Wind) - Energy(NoWind)
            penalty = max(0.0, res_actual["total_kwh"] - res_no_wind["total_kwh"])
            total_penalty += penalty

        # 2. Current Partial Hour
        current_temp = self.data.get("current_calc_temp")
        if current_temp is not None:
            eff_wind = self.data.get("effective_wind", 0.0)
            minutes_passed = dt_util.now().minute
            fraction = minutes_passed / 60.0

            res_curr = self.statistics.calculate_total_power(
                current_temp, eff_wind, 0.0, is_aux_active=self.auxiliary_heating_active, detailed=False
            )
            res_curr_no_wind = self.statistics.calculate_total_power(
                current_temp, 0.0, 0.0, is_aux_active=self.auxiliary_heating_active, detailed=False
            )

            penalty = max(0.0, res_curr["total_kwh"] - res_curr_no_wind["total_kwh"])
            total_penalty += (penalty * fraction)

        # 3. Future Hours
        # Compare Future Normal vs Future No Wind
        future_normal, _, _ = self.forecast.calculate_future_energy(
            dt_util.now(), ignore_aux=(not self.auxiliary_heating_active)
        )

        future_no_wind, _, _ = self.forecast.calculate_future_energy(
            dt_util.now(), ignore_aux=(not self.auxiliary_heating_active), force_no_wind=True
        )

        total_penalty += max(0.0, future_normal - future_no_wind)

        return round(total_penalty, 2)

    def _update_daily_budgets(self, current_prediction_rate: float, current_time: datetime, minutes_passed: int):
        """Update daily energy budgets and forecasts."""
        expected_hour_so_far = self._collector.expected_energy_hour

        expected_today_sum = 0.0
        forecasted_past_sum = 0.0
        gross_past_sum = 0.0
        has_past_aux = False
        solar_today_sum = 0.0
        today_date_str = current_time.date().isoformat()

        for entry in reversed(self._hourly_log):
             if entry["timestamp"].startswith(today_date_str):
                 expected_today_sum += entry.get("expected_kwh") or 0.0
                 forecasted_past_sum += entry.get("forecasted_kwh") or entry.get("expected_kwh") or 0.0
                 gross_past_sum += entry.get("forecasted_kwh_gross") or entry.get("forecasted_kwh") or entry.get("expected_kwh") or 0.0
                 solar_today_sum += entry.get("solar_impact_kwh") or 0.0
                 if entry.get("auxiliary_active", False):
                     has_past_aux = True
             else:
                 break

        expected_today_total = expected_today_sum + expected_hour_so_far
        self.data[ATTR_EXPECTED_TODAY] = round(expected_today_total, 3)

        # Calculate Future Forecast
        # The running plan (Predicted) MUST account for current aux state for future hours.
        future_forecast_kwh, future_solar_kwh, future_unit_totals = self.forecast.calculate_future_energy(
            current_time, ignore_aux=(not self.auxiliary_heating_active)
        )

        # Calculate Current Hour Plan (Forecast based, not actual)
        # Use the Reference Forecast (locked at midnight) for stability.
        current_hour_plan_rate, current_hour_unit_breakdown = self.forecast.get_plan_for_hour(
            current_time,
            source='reference',
            ignore_aux=(not self.auxiliary_heating_active)
        )

        # Fallback: If no reference forecast, use live model (best guess)
        if current_hour_plan_rate == 0.0 and current_prediction_rate > 0.0:
            current_hour_plan_rate = current_prediction_rate

        # Calculate Daily Budget (The Plan)
        # = Locked Past + Stable Current Forecast (Reference) + Live Future Forecast
        budget_total = forecasted_past_sum + current_hour_plan_rate + future_forecast_kwh
        self.data[ATTR_PREDICTED] = round(budget_total, 2)

        # Calculate Gross Daily Budget (aux-unaware) for deviation denominator.
        if self.auxiliary_heating_active:
             future_gross_kwh_budget, _, _ = self.forecast.calculate_future_energy(
                 current_time, ignore_aux=True
             )
             current_hour_gross_rate_budget, _ = self.forecast.get_plan_for_hour(
                 current_time, source='reference', ignore_aux=True
             )
             if current_hour_gross_rate_budget == 0.0:
                  current_hour_gross_rate_budget = current_hour_plan_rate + self.data.get("current_aux_impact_rate", 0.0)
             budget_total_gross = gross_past_sum + current_hour_gross_rate_budget + future_gross_kwh_budget
        elif has_past_aux:
             budget_total_gross = gross_past_sum + current_hour_plan_rate + future_forecast_kwh
        else:
             budget_total_gross = budget_total

        self.data["predicted_gross"] = round(budget_total_gross, 2)

        # Calculate Thermodynamic Projection (The Reality Check)
        # = Actuals So Far + Live Forecast for Remaining Today
        # This answers: "Where will we end up if the house behaves as modeled for the rest of the day?"
        # Uses LIVE forecast for current hour (pro-rated) and future hours.

        # 1. Get Live Plan for Current Hour
        current_hour_live_rate, _ = self.forecast.get_plan_for_hour(
            current_time,
            source='live',
            ignore_aux=(not self.auxiliary_heating_active)
        )

        # 2. Calculate remaining portion of current hour (Forecast)
        remaining_fraction = max(0.0, (60 - minutes_passed) / 60.0)
        remaining_live_current_hour = current_hour_live_rate * remaining_fraction

        # 3. Sum parts
        # expected_today_total includes expected_hour_so_far (Actuals)
        thermodynamic_projection = expected_today_total + remaining_live_current_hour + future_forecast_kwh
        self.data["thermodynamic_projection_kwh"] = round(thermodynamic_projection, 2)

        # Calculate Thermodynamic Deviation
        # actual_today vs expected_today_total (Actual So Far vs Model So Far)
        # Note: Future components cancel out when projecting End of Day.
        actual_today = self._accumulated_energy_today
        thermo_deviation = actual_today - expected_today_total
        self.data["thermodynamic_deviation_kwh"] = round(thermo_deviation, 2)

        if thermodynamic_projection > ENERGY_GUARD_THRESHOLD:
            self.data["thermodynamic_deviation_pct"] = round((thermo_deviation / thermodynamic_projection) * 100, 1)
        else:
            self.data["thermodynamic_deviation_pct"] = 0.0

        # Calculate Solar Budget
        current_solar_kw = self.data.get(ATTR_SOLAR_IMPACT, 0.0)
        solar_budget_total = solar_today_sum + current_solar_kw + future_solar_kwh
        self.data[ATTR_SOLAR_PREDICTED] = round(solar_budget_total, 2)

        # Calculate Forecast Today (Projected Consumption)
        # This is Actual Consumption So Far + Live Forecast for the rest of the day.
        # This calculation is now mode-aware (respects aux heating) and replaces the
        # previous logic that was anchored to an aux-unaware midnight snapshot, which
        # caused incorrect deviation reporting.

        # FIX: Use `max(0, forecast - actual)` instead of `forecast * remaining_fraction`
        # to prevent "Double Accounting Risk" where actuals > planned rate for the elapsed time
        # causes a phantom spike in the daily total.
        # Note: self._accumulated_energy_today ALREADY includes the current hour's actuals.
        # So we add the "Remaining Budget" for the current hour.

        # We need actuals specifically for THIS hour to subtract from the hourly plan.
        actual_this_hour = self._collector.energy_hour

        # Smart Merge: Check for partial log data (pre-restart)
        partial_log = self._get_partial_log_for_current_hour()
        if partial_log:
            actual_this_hour += partial_log.get("actual_kwh", 0.0)

        remaining_current_hour = max(0.0, current_hour_plan_rate - actual_this_hour)

        forecast_today_val = (
            self._accumulated_energy_today  # Includes actual_this_hour
            + remaining_current_hour        # The part of the hourly plan we haven't "spent" yet
            + future_forecast_kwh
        )
        self.data[ATTR_FORECAST_TODAY] = round(forecast_today_val, 2)

        # Calculate Gross Forecast Today (Thermodynamic Demand - Aux Unaware)
        # Used for Thermal Stress Index to avoid "Mild Weather Fallacy" when Aux is active.

        # 1. Gross Future
        future_gross_kwh, _, _ = self.forecast.calculate_future_energy(
            current_time, ignore_aux=True
        )

        # 2. Gross Current Hour Plan
        # We need the full hour gross plan to calculate remaining

        # OPTIMIZATION: Only recalculate if aux is active. If inactive, Gross == Net,
        # and we can reuse current_hour_plan_rate (already calculated above).
        if self.auxiliary_heating_active:
            # Calculate strictly from Reference Forecast (Aux Ignored)
            current_hour_gross_rate, _ = self.forecast.get_plan_for_hour(
                current_time,
                source='reference',
                ignore_aux=True
            )

            # Fallback if forecast missing
            if current_hour_gross_rate == 0.0:
                 # Fallback: If no forecast for current hour, use the live model rate (Net).
                 # We must restore the Aux component to get the GROSS rate for the Stress Index.
                 current_hour_gross_rate = current_hour_plan_rate # Default to Net
                 current_aux_rate = self.data.get("current_aux_impact_rate", 0.0)
                 if current_aux_rate > 0:
                     current_hour_gross_rate += current_aux_rate
        else:
            # Optimization: When aux is inactive, Gross == Net (no need to recalculate)
            current_hour_gross_rate = current_hour_plan_rate

        # 3. Gross Actuals So Far (Net + Aux)
        # accumulated_aux_impact_kwh is updated just before this method call
        gross_actuals_so_far = self._accumulated_energy_today + self.data.get("accumulated_aux_impact_kwh", 0.0)

        # 4. Gross Remaining Current Hour
        # gross_actual_this_hour = net_actual_hour + aux_actual_hour
        # Note: We must include partial logs if any
        gross_actual_this_hour = self._collector.energy_hour + self._collector.aux_impact_hour

        if partial_log:
             gross_actual_this_hour += partial_log.get("actual_kwh", 0.0)
             gross_actual_this_hour += partial_log.get("aux_impact_kwh", 0.0)

        remaining_gross_current = max(0.0, current_hour_gross_rate - gross_actual_this_hour)

        forecast_gross_val = (
            gross_actuals_so_far
            + remaining_gross_current
            + future_gross_kwh
        )
        self.data["forecast_today_gross"] = round(forecast_gross_val, 2)

        # Calculate Per-Unit Forecast Today
        # Formula: Actual So Far + Remaining Current Hour (Forecast) + Future (Forecast)
        unit_forecast_today = {}
        for entity_id in self.energy_sensors:
            # 1. Actual So Far
            actual_total = self._daily_individual.get(entity_id, 0.0)

            # 2. Remaining Current Hour
            current_unit_data = current_hour_unit_breakdown.get(entity_id, {})
            current_unit_forecast_kwh = current_unit_data.get("net_kwh", 0.0) if current_unit_data else 0.0

            actual_unit_this_hour = self._hourly_delta_per_unit.get(entity_id, 0.0)
            remaining_unit = max(0.0, current_unit_forecast_kwh - actual_unit_this_hour)

            # 3. Future
            future_unit = future_unit_totals.get(entity_id, 0.0)

            # Note: actual_total includes actual_unit_this_hour.
            # So Total = (Actual_Prev + Actual_Now) + (Forecast_Hour - Actual_Now) + Future
            # If Actual_Now < Forecast_Hour => Total = Actual_Prev + Forecast_Hour + Future.
            # If Actual_Now > Forecast_Hour => Total = Actual_Prev + Actual_Now + Future.
            total_unit = actual_total + remaining_unit + future_unit
            unit_forecast_today[entity_id] = round(total_unit, 2)

        self.data["forecast_today_per_unit"] = unit_forecast_today

        # Calculate Efficiency Forecast Today
        tdd_forecast = self.data.get(ATTR_TDD_DAILY_STABLE, 0.0)
        if tdd_forecast > ENERGY_GUARD_THRESHOLD:
            self.data[ATTR_EFFICIENCY_FORECAST_TODAY] = round(forecast_today_val / tdd_forecast, 3)
        else:
            self.data[ATTR_EFFICIENCY_FORECAST_TODAY] = None

        # Calculate Forecast Uncertainty
        stats = self.forecast.calculate_uncertainty_stats()
        self.data[ATTR_FORECAST_UNCERTAINTY] = stats

        day_progress = (current_time.hour + (current_time.minute / 60.0)) / 24.0
        remaining_day_fraction = max(0.0, 1.0 - day_progress)
        max_dev = stats.get("p95_abs_error", 2.0)
        uncertainty_margin = max_dev * remaining_day_fraction

        self.data["confidence_interval_margin"] = round(uncertainty_margin, 2)
        self.data["confidence_interval_lower"] = round(forecast_today_val - uncertainty_margin, 2)
        self.data["confidence_interval_upper"] = round(forecast_today_val + uncertainty_margin, 2)

        # Calculate Daily Wind Chill Penalty (Full Day Impact)
        self.data["daily_wind_chill_penalty"] = self._calculate_daily_wind_penalty()

        # Per-Source Accuracy & Blend Config (New Unified Attribute)
        self.data[ATTR_FORECAST_DETAILS] = {
            "blend_config": {
                "primary_entity_id": self.weather_entity,
                "secondary_entity_id": self.entry.data.get(CONF_SECONDARY_WEATHER_ENTITY),
                "crossover_day": self.entry.data.get(CONF_FORECAST_CROSSOVER_DAY, DEFAULT_FORECAST_CROSSOVER_DAY)
            },
            "accuracy_by_source": {
                source: {"daily": stats["daily"]}
                for source, stats in self.forecast.calculate_per_source_uncertainty_stats().items()
            }
        }

    def _update_tdd_calculations(self, temp: float | None, minutes_passed: int):
        """Update TDD calculations for the current partial hour."""
        current_tdd_acc = self.data.get(ATTR_TDD, 0.0)

        if temp is not None:
             current_hour_tdd = (abs(self.balance_point - temp) / 24.0) * (minutes_passed / 60.0)
             total_tdd_so_far = current_tdd_acc + current_hour_tdd
        else:
             total_tdd_so_far = current_tdd_acc

        self.data[ATTR_TDD_SO_FAR] = round(total_tdd_so_far, 3)

    def _update_deviation_stats(self):
        """Update deviation statistics."""
        # Deviation (Today)
        # Gross-domain comparison cancels live forecast drift from both sides of diff.
        # Frozen midnight_forecast denominator prevents the percentage from scaling
        # with weather forecast revisions.
        predicted_gross = self.data.get("predicted_gross", 0.0)
        forecast_today_gross = self.data.get("forecast_today_gross", 0.0)
        midnight = self.data.get(ATTR_MIDNIGHT_FORECAST, 0.0)
        denominator = midnight if midnight > ENERGY_GUARD_THRESHOLD else predicted_gross

        if denominator > ENERGY_GUARD_THRESHOLD:
             diff = forecast_today_gross - predicted_gross
             deviation = (diff / denominator) * 100
             self.data[ATTR_DEVIATION] = round(deviation, 1)
        else:
             self.data[ATTR_DEVIATION] = 0.0

        # Deviation Breakdown
        self.data[ATTR_DEVIATION_BREAKDOWN] = self.statistics.calculate_deviation_breakdown()

        # Plan Revision Impact (formerly Weather Forecast Deviation)
        self.data["plan_revision_impact"] = self.forecast.calculate_plan_revision_impact()

        # Weather-Adjusted Deviation
        weather_dev_data = self.data.get("plan_revision_impact", {})
        weather_impact_kwh = weather_dev_data.get("estimated_impact_kwh", 0.0)

        actual_kwh = self.data.get(ATTR_ENERGY_TODAY, 0.0)
        expected_kwh = self.data.get(ATTR_EXPECTED_TODAY, 0.0)
        current_deviation_kwh = actual_kwh - expected_kwh
        weather_adjusted_kwh = current_deviation_kwh + weather_impact_kwh

        weather_adjusted_pct = 0.0
        if expected_kwh > ENERGY_GUARD_THRESHOLD:
            weather_adjusted_pct = (weather_adjusted_kwh / expected_kwh) * 100

        if abs(weather_adjusted_kwh) < 0.1:
            adjusted_explanation = "Forecast Accuracy Adjustment: Consumption matches the model perfectly ✨"
        elif weather_adjusted_kwh > 0:
            adjusted_explanation = (
                f"Forecast Accuracy Adjustment: You are using {weather_adjusted_kwh:+.1f} kWh "
                f"({weather_adjusted_pct:+.1f}%) more than the model expects 📈"
            )
        else:
            adjusted_explanation = (
                f"Forecast Accuracy Adjustment: You are using {abs(weather_adjusted_kwh):.1f} kWh "
                f"({abs(weather_adjusted_pct):.1f}%) less than the model expects 📉"
            )

        self.data["weather_adjusted_deviation"] = {
            "deviation_kwh": round(weather_adjusted_kwh, 2),
            "deviation_pct": round(weather_adjusted_pct, 1),
            "current_deviation_kwh": round(current_deviation_kwh, 2),
            "weather_impact_kwh": round(weather_impact_kwh, 2),
            "explanation": adjusted_explanation
        }

    async def _async_update_data(self):
        """Update data."""
        if not self._is_loaded:
             await self._async_load_data()

        current_time = dt_util.now()

        # Initialize trackers if this is the first run
        if self._accumulation_start_time is None:
            self._accumulation_start_time = current_time
        if self._last_day_processed is None:
            self._last_day_processed = current_time.date()
        if self._last_hour_processed is None:
            self._last_hour_processed = current_time.hour

        # --- Boundary Processing ---
        # First, handle hour/day rollovers to ensure all historical data is up-to-date
        # before any new calculations for the current period are made.
        # This resolves the midnight race condition.
        completed_hour = current_time.hour != self._last_hour_processed
        if completed_hour:
            await self._process_hourly_data(current_time)
            # Recalculate stats that depend on hourly_log (moved from per-minute execution)
            self.statistics.calculate_temp_stats()
            self.statistics.update_daily_savings_cache()
            _LOGGER.debug("Stats recalculated after hourly processing")

            self._last_hour_processed = current_time.hour
            self._accumulation_start_time = current_time

        completed_day = current_time.date() != self._last_day_processed
        if completed_day:
            if self._last_day_processed is not None:
                # Process the completed day
                await self._process_daily_data(self._last_day_processed)
            self._last_day_processed = current_time.date()

        # --- Forecast Update ---
        # Now that rollovers are handled, update the forecast.
        # This ensures the new day's snapshot is created with complete inertia data.
        # Update is forced on boundaries, otherwise periodic.
        if completed_hour or completed_day or current_time.minute % 15 == 0 or self.data.get(ATTR_TEMP_FORECAST_TODAY) is None:
            await self.forecast.update_daily_forecast()


        # --- Live Data Processing ---

        # 1. Fetch Sensor Data (Strict Source Logic)

        # Temp
        temp = None
        if self.outdoor_temp_source == SOURCE_SENSOR:
             if self.outdoor_temp_sensor:
                 temp = self._get_float_state(self.outdoor_temp_sensor)
                 if temp is None and self.hass.is_running:
                     _LOGGER.warning(f"Outdoor Temp Sensor '{self.outdoor_temp_sensor}' is unavailable.")
        else:
             temp = self._get_weather_attribute("temperature")

        # Wind Speed
        wind_speed = None
        if self.wind_speed_source == SOURCE_SENSOR:
             if self.wind_speed_sensor:
                 wind_speed = self._get_speed_in_ms(self.wind_speed_sensor)
                 if wind_speed is None and self.hass.is_running:
                      _LOGGER.warning(f"Wind Speed Sensor '{self.wind_speed_sensor}' is unavailable.")
        else:
             wind_speed = self._get_weather_attribute("wind_speed")

        # Wind Gust
        wind_gust = None
        if self.wind_gust_source == SOURCE_SENSOR:
             if self.wind_gust_sensor:
                 wind_gust = self._get_speed_in_ms(self.wind_gust_sensor)
                 if wind_gust is None:
                     # Gusts are often intermittent, maybe debug level?
                     _LOGGER.debug(f"Wind Gust Sensor '{self.wind_gust_sensor}' is unavailable.")
        else:
             wind_gust = self._get_weather_attribute("wind_gust_speed")

        # Fetch Humidity (for Track C per-hour COP / defrost penalty)
        humidity = self._get_weather_attribute("humidity")

        # Fetch Solar Data
        potential_solar_factor = 0.0
        solar_factor = 0.0
        solar_vector = (0.0, 0.0)
        if self.solar_enabled:
             elev, azim = self._get_sun_info_now()
             cloud = self._get_cloud_coverage()
             potential_solar_factor = self.solar.calculate_solar_factor(elev, azim, cloud)
             solar_factor = self.solar.calculate_effective_solar_factor(
                 potential_solar_factor, self.solar_correction_percent
             )
             self.data[ATTR_SOLAR_FACTOR] = round(solar_factor, 2)

             potential_solar_vector = self.solar.calculate_solar_vector(elev, azim, cloud)
             solar_vector = self.solar.calculate_effective_solar_vector(
                 potential_solar_vector, self.solar_correction_percent
             )

             if ATTR_SOLAR_IMPACT not in self.data:
                 self.data[ATTR_SOLAR_IMPACT] = 0.0

        if wind_speed is None:
            wind_speed = 0.0

        # Calculate Inertia & Temp Keys EARLY (Moved from Step 4)
        # We need these for Solar Impact calculation and other model lookups
        inertia_list = self._get_inertia_list(current_time)
        inertia_temp = None
        if inertia_list:
             inertia_temp = self._calculate_weighted_inertia(inertia_list)

        calc_temp = inertia_temp if inertia_temp is not None else temp
        temp_key = str(int(round(calc_temp))) if calc_temp is not None else "0"

        # Calculate Potential Solar Impact (Screens Up) for Sensors
        potential_impact_kw = 0.0
        if self.solar_enabled and temp_key:
             # Calculate what impact WOULD be if correction was 100%
             # We use the same unit coefficients as actual calculation
             total_potential = 0.0
             for entity_id in self.energy_sensors:
                 unit_coeff = self.solar.calculate_unit_coefficient(entity_id, temp_key)
                 total_potential += self.solar.calculate_unit_solar_impact(potential_solar_vector, unit_coeff)
             potential_impact_kw = total_potential

        self.data[ATTR_SOLAR_POTENTIAL] = round(potential_impact_kw, 3)

        # Calculate Recommendation State
        rec_state = "none"
        if self.solar_enabled and temp is not None:
             rec_state = self.solar_optimizer.get_recommendation_state(temp, potential_solar_factor)

        self.data[ATTR_RECOMMENDATION_STATE] = rec_state
        self.data["potential_solar_factor"] = round(potential_solar_factor, 3)

        # 2. Calculate Effective Wind & Conditions (if temp is available)
        if temp is not None:
            effective_wind = self._calculate_effective_wind(wind_speed, wind_gust)
            wind_bucket = self._get_wind_bucket(effective_wind)
            # Cache effective wind for sensors
            self.data["effective_wind"] = effective_wind

            # Update hourly aggregates (delegated to ObservationCollector)
            self._collector.accumulate_weather(
                temp=temp,
                effective_wind=effective_wind,
                wind_bucket=wind_bucket,
                solar_factor=solar_factor,
                solar_vector=solar_vector,
                is_aux_active=self.auxiliary_heating_active,
                current_time=current_time,
                humidity=humidity,
            )
        else:
            effective_wind = 0.0
            wind_bucket = "normal"
            if (self.outdoor_temp_sensor or self.weather_entity) and self.hass.is_running:
                _LOGGER.warning(
                    "Unable to retrieve outdoor temperature. Sensor '%s' and Weather Fallback '%s' both failed.",
                    self.outdoor_temp_sensor,
                    self.weather_entity
                )

        # 3. Energy Tracking (Robust Delta Calculation)
        hourly_delta = 0.0
        for entity_id in self.energy_sensors:
            val = self._get_float_state(entity_id)
            if val is not None:
                if entity_id in self._last_energy_values:
                    prev_val = self._last_energy_values[entity_id]
                    delta = val - prev_val

                    if delta < 0:
                        _LOGGER.warning(f"Energy meter reset detected for {entity_id}: {prev_val} -> {val}. Skipping this reading.")
                        self._last_energy_values[entity_id] = val
                        continue

                    if delta > self.max_energy_delta:
                        _LOGGER.warning(f"Energy spike detected for {entity_id}: {delta:.2f} kWh > {self.max_energy_delta} kWh. Skipping this reading and updating baseline.")
                        self._last_energy_values[entity_id] = val
                        continue

                    # DHW mode: energy goes to hot water tank, not space heating.
                    # Exclude from all heating totals. _last_energy_values is still
                    # updated below so the delta baseline is correct when mode switches back.
                    if self.get_unit_mode(entity_id) == MODE_DHW:
                        pass
                    else:
                        hourly_delta += delta

                        if entity_id not in self._daily_individual:
                            self._daily_individual[entity_id] = 0.0
                        self._daily_individual[entity_id] += delta

                        if entity_id not in self._hourly_delta_per_unit:
                            self._hourly_delta_per_unit[entity_id] = 0.0
                        self._hourly_delta_per_unit[entity_id] += delta

                        if self.enable_lifetime_tracking:
                            if entity_id not in self._lifetime_individual:
                                self._lifetime_individual[entity_id] = 0.0
                            self._lifetime_individual[entity_id] += delta

                self._last_energy_values[entity_id] = val

        self.data["daily_individual"] = self._daily_individual
        self.data["lifetime_individual"] = self._lifetime_individual

        self._collector.energy_hour += hourly_delta
        self._accumulated_energy_today += hourly_delta
        self.data[ATTR_ENERGY_TODAY] = round(self._accumulated_energy_today, 3)

        # 4. Live Prediction / Efficiency Display
        # (inertia_list and inertia_temp calculated earlier in Step 1)

        # Calculate Thermal State Attributes
        # Normalize weights to the active subset so they sum to 1.0,
        # allowing direct verification: sum(history[i] * weights[i]) == effective_temperature.
        _active_w = list(self.inertia_weights[-len(inertia_list):]) if inertia_list else []
        _w_sum = sum(_active_w)
        _display_weights = [round(w / _w_sum, 4) for w in _active_w] if _w_sum > 0 else _active_w
        thermal_state = {
             "raw_temperature": round(temp, 1) if temp is not None else None,
             "effective_temperature": round(inertia_temp, 1) if inertia_temp is not None else None,
             "inertia_history": [round(t, 1) for t in inertia_list],
             "weights": _display_weights,
             "samples_used": len(inertia_list),
             "last_updated": current_time.isoformat(),
             "balance_point": self.balance_point,
        }

        if temp is not None and inertia_temp is not None:
             lag = inertia_temp - temp
             thermal_state["thermal_lag"] = round(lag, 1)

             if lag > 0.5:
                 thermal_state["lag_status"] = "Shedding Heat" # House is warmer than outside
             elif lag < -0.5:
                 thermal_state["lag_status"] = "Gaining Heat" # House is colder than outside
             else:
                 thermal_state["lag_status"] = "Balanced"
        else:
             thermal_state["thermal_lag"] = None
             thermal_state["lag_status"] = "Unknown"

        # Extended State: Regime, Balance, and Instant TDD
        if inertia_temp is not None:
             # Regime (Cold vs Mild) - Matches statistics.py logic
             tdd_target = abs(self.balance_point - inertia_temp)
             if tdd_target > 4.0:
                 thermal_state["regime"] = "Cold"
                 thermal_state["regime_explanation"] = "Using Ratio Extrapolation (Physics-dominated)"
             else:
                 thermal_state["regime"] = "Mild"
                 thermal_state["regime_explanation"] = "Using Nearest Neighbor (Noise-dominated)"

             # Degrees to Balance
             diff = self.balance_point - inertia_temp
             thermal_state["degrees_to_balance"] = round(diff, 1)

             # Instant TDD
             thermal_state["instant_tdd"] = round(tdd_target / 24.0, 3)
        else:
             thermal_state["regime"] = None
             thermal_state["regime_explanation"] = None
             thermal_state["degrees_to_balance"] = None
             thermal_state["instant_tdd"] = None

        # Trend Calculation
        if len(inertia_list) >= 2:
             # Look back up to 2 hours if possible (3 points: Current, H-1, H-2)
             # List is [oldest, ..., newest]
             # If len=4: [H-3, H-2, H-1, Current]
             # Compare Current (index -1) with H-2 (index -3)
             hours_back = 0
             old_val = inertia_list[-1]

             if len(inertia_list) >= 3:
                  old_val = inertia_list[-3]
                  hours_back = 2
             elif len(inertia_list) == 2:
                  old_val = inertia_list[-2]
                  hours_back = 1

             diff = inertia_list[-1] - old_val
             rate = diff / hours_back if hours_back > 0 else 0.0
             thermal_state["trend_rate"] = round(rate, 2)

             if rate < -2.0: trend = "falling_fast"
             elif rate < -0.5: trend = "falling"
             elif rate > 2.0: trend = "rising_fast"
             elif rate > 0.5: trend = "rising"
             else: trend = "stable"
             thermal_state["temperature_trend"] = trend
        else:
             thermal_state["trend_rate"] = None
             thermal_state["temperature_trend"] = "unknown"

        self.data["thermal_state"] = thermal_state

        # (calc_temp and temp_key calculated earlier in Step 1)
        minutes_passed = current_time.minute

        # 4a. Update Live Predictions & Accumulation
        current_prediction_rate = self._update_live_predictions(
            calc_temp, temp_key, wind_bucket, current_time
        )

        # 4b. Update Accumulated Impacts (Moved up to support Daily Budgets)
        self._update_accumulated_impacts(current_time)

        # 4c. Update Daily Budgets & Forecasts
        self._update_daily_budgets(
            current_prediction_rate, current_time, minutes_passed
        )

        # 4d. Update TDD Calculations
        self._update_tdd_calculations(temp, minutes_passed)

        # 4e. Update Deviation Stats
        self._update_deviation_stats()
        self.statistics.calculate_potential_savings()

        return self.data

    def _update_accumulated_impacts(self, current_time: datetime):
        """Calculate and update the daily accumulated impacts for solar, guest, and aux modes."""
        today_date_str = current_time.date().isoformat()

        # 1. Sum impacts from today's completed hourly logs
        accumulated_solar = 0.0
        accumulated_guest = 0.0
        accumulated_aux = 0.0

        for entry in self._hourly_log:
            if entry["timestamp"].startswith(today_date_str):
                accumulated_solar += entry.get("solar_impact_kwh", 0.0)
                accumulated_guest += entry.get("guest_impact_kwh", 0.0)
                accumulated_aux += entry.get("aux_impact_kwh", 0.0)

        # 2. Add live, intra-hour impact
        #    - Solar: Use the current `ATTR_SOLAR_IMPACT` (in kW) and scale by minutes passed.
        #    - Aux: Use `last_hour_aux_impact_kwh` as a rate (kW) and scale.
        #    - Guest: Calculate live guest impact.
        minutes_fraction = current_time.minute / 60.0

        # Live Solar Impact
        # NOTE: self.data["solar_impact_kwh"] stores instantaneous power in kW due to a legacy naming convention.
        current_solar_kw = self.data.get(ATTR_SOLAR_IMPACT, 0.0)
        accumulated_solar += current_solar_kw * minutes_fraction

        # Live Aux Impact
        # Use the precise minute-by-minute accumulator
        accumulated_aux += self._collector.aux_impact_hour

        # Live Guest Impact
        # Guest units are not tracked in expected - their full consumption is the impact
        live_guest_impact = 0.0
        for entity_id, actual_kwh in self._hourly_delta_per_unit.items():
            unit_mode = self.get_unit_mode(entity_id)
            if unit_mode in (MODE_GUEST_HEATING, MODE_GUEST_COOLING):
                live_guest_impact += actual_kwh
        accumulated_guest += live_guest_impact

        # 3. Update the data dictionary
        self.data["accumulated_solar_impact_kwh"] = round(accumulated_solar, 3)
        self.data["accumulated_guest_impact_kwh"] = round(accumulated_guest, 3)
        self.data["accumulated_aux_impact_kwh"] = round(accumulated_aux, 3)

        # 4. Calculate Thermodynamic Gross Today (Net + Aux +/- Solar)
        accumulated_gross = 0.0
        for entry in self._hourly_log:
             if entry["timestamp"].startswith(today_date_str):
                 accumulated_gross += entry.get("thermodynamic_gross_kwh", 0.0)

        # Add live component
        # We need the current temp context to decide solar sign
        # We use the LAST KNOWN calc_temp stored in self.data (from _update_live_predictions)
        current_calc_temp = self.data.get("current_calc_temp")
        if current_calc_temp is None:
             # Fallback if unknown
             current_calc_temp = self.balance_point

        live_net = self._collector.energy_hour
        live_aux = self._collector.aux_impact_hour
        live_solar_impact = current_solar_kw * minutes_fraction

        live_gross = live_net + live_aux
        if current_calc_temp >= self.balance_point:
             live_gross -= live_solar_impact
        else:
             live_gross += live_solar_impact

        self.data["thermodynamic_gross_today_kwh"] = round(accumulated_gross + live_gross, 3)

    def _is_model_covered(self, temp_key: str, wind_bucket: str) -> bool:
        """Check if exact model data exists for this temp/wind combo."""
        if temp_key in self._correlation_data:
            # Check exact bucket first
            if wind_bucket in self._correlation_data[temp_key]:
                return self._correlation_data[temp_key][wind_bucket] > 0
            # "Covered" = has learned data (exact or fallback bucket), no TDD extrapolation
            if wind_bucket == "extreme_wind" and "high_wind" in self._correlation_data[temp_key]:
                return True
            if wind_bucket in ("extreme_wind", "high_wind") and "normal" in self._correlation_data[temp_key]:
                return True
        return False

    def _get_predicted_kwh(self, temp_key: str, wind_bucket: str, actual_temp: float) -> float:
        """Get predicted kWh for a given temp and wind using the centralized robust model."""
        return self.statistics._get_prediction_from_model(
            self._correlation_data, temp_key, wind_bucket, actual_temp, self.balance_point
        )

    def _get_predicted_kwh_per_unit(self, entity_id: str, temp_key: str, wind_bucket: str, actual_temp: float) -> float:
        """Get predicted kWh for a given unit, temp and wind using the centralized robust model."""
        unit_data = self._correlation_data_per_unit.get(entity_id, {})
        return self.statistics._get_prediction_from_model(
            unit_data, temp_key, wind_bucket, actual_temp, self.balance_point
        )

    def _get_aux_impact_kw(self, temp_key: str, wind_bucket: str = "normal", actual_temp: float = 0.0) -> float:
        """Get the Aux Impact (kW reduction) for a given temp key and wind bucket."""
        # Note: actual_temp is only needed if scaling were applied, which it is not for aux models.
        # We pass it to satisfy the function signature.
        if actual_temp == 0.0:
            try:
                actual_temp = float(temp_key)
            except ValueError:
                actual_temp = self.balance_point # A safe default if temp_key is invalid

        return self.statistics._get_prediction_from_model(
            self._aux_coefficients, temp_key, wind_bucket, actual_temp, self.balance_point, apply_scaling=False
        )

    def _get_aux_impact_kw_per_unit(self, entity_id: str, temp_key: str, wind_bucket: str = "normal", actual_temp: float = 0.0) -> float | None:
        """Get the Aux Impact (kW reduction) for a given unit, temp key, and wind bucket."""
        unit_data = self._aux_coefficients_per_unit.get(entity_id, {})
        if not unit_data:
            return None

        if actual_temp == 0.0:
            try:
                actual_temp = float(temp_key)
            except ValueError:
                actual_temp = self.balance_point # A safe default if temp_key is invalid

        return self.statistics._get_prediction_from_model(
            unit_data, temp_key, wind_bucket, actual_temp, self.balance_point, apply_scaling=False
        )

    def _get_unit_observation_count(self, entity_id: str, temp_key: str, wind_bucket: str) -> int:
        """Get observation count for a unit/temp/bucket.

        This count represents the number of learning cycles (hours) where this specific
        combination of conditions (Temperature + Wind) has been observed and used to
        train the model for this specific unit.
        """
        if entity_id in self._observation_counts:
            if temp_key in self._observation_counts[entity_id]:
                return self._observation_counts[entity_id][temp_key].get(wind_bucket, 0)
        return 0

    def calculate_unit_rolling_power_watts(self, entity_id: str) -> int:
        """Calculate a 60-minute rolling average power (Watts) for a specific unit.

        Uses the actual energy consumed in the current partial hour plus a
        proportional part of the previous hour's total consumption.
        This provides a continuous, non-zero power value that correctly reflects
        the last 60 minutes of activity.
        """
        now = dt_util.now()
        minutes_passed = now.minute
        fraction_passed = minutes_passed / 60.0

        # 1. Actual energy so far this hour
        kwh_now = self._hourly_delta_per_unit.get(entity_id, 0.0)

        # 2. Actual energy from the last completed hour
        kwh_last = 0.0
        if self._hourly_log:
            last_entry = self._hourly_log[-1]
            kwh_last = last_entry.get("unit_breakdown", {}).get(entity_id, 0.0)

        # 3. Combine to form a 60-minute window
        # (e.g. at :15, we take 15 mins from now + 45 mins from last hour)
        rolling_kwh = kwh_now + (kwh_last * (1.0 - fraction_passed))

        # 4. Convert kWh/h to Watts
        return int(round(rolling_kwh * 1000))

    def calculate_modeled_energy(self, start_date: date, end_date: date, pre_fetched_logs: dict | None = None) -> tuple[float, float, float | None, float | None, float]:
        """Calculate modeled energy for a date range."""
        return self.statistics.calculate_modeled_energy(start_date, end_date, pre_fetched_logs)

    async def async_compare_periods(self, p1_start: date, p1_end: date, p2_start: date, p2_end: date):
        """Handle the compare_periods service call."""
        _LOGGER.info(f"Comparing periods: {p1_start}-{p1_end} vs {p2_start}-{p2_end}")

        comparison_data = self.statistics.compare_periods(
            p1_start, p1_end, p2_start, p2_end
        )

        self.data["last_comparison"] = comparison_data

        # Trigger an update for the new sensor
        self.async_set_updated_data(self.data)

    # --- Data Contract Factory Methods (Issue #775) ---

    def _build_hourly_observation(
        self,
        current_time: datetime,
        *,
        avg_temp: float,
        inertia_temp: float,
        temp_key: str,
        effective_wind: float,
        wind_bucket: str,
        avg_solar_factor: float,
        avg_solar_vector: tuple[float, float],
        solar_impact_raw: float,
        effective_solar_impact: float,
        total_energy_kwh: float,
        learning_energy_kwh: float,
        guest_impact_kwh: float,
        expected_kwh: float,
        base_expected_kwh: float,
        aux_impact_kwh: float,
        aux_fraction: float,
        is_aux_dominant: bool,
        was_cooldown_active: bool,
        forecasted_kwh: float | None = None,
        forecasted_kwh_primary: float | None = None,
        forecasted_kwh_secondary: float | None = None,
        forecasted_kwh_gross: float | None = None,
        forecasted_kwh_gross_primary: float | None = None,
        forecasted_kwh_gross_secondary: float | None = None,
        forecast_source: str | None = None,
        recommendation_state: str = "none",
        correction_percent: float = 0.0,
        potential_solar_factor: float = 0.0,
    ) -> "HourlyObservation":
        """Build an immutable HourlyObservation from current coordinator state.

        Called at hour boundary after all aggregates are computed.
        Captures the frozen snapshot before accumulators are reset.
        """
        from .observation import HourlyObservation

        n = self._collector.sample_count
        hc = self._collector.humidity_count
        avg_humidity = round(self._collector.humidity_sum / hc, 1) if hc > 0 else None

        timestamp = self._collector.start_time if self._collector.start_time else current_time
        return HourlyObservation(
            timestamp=timestamp,
            hour=timestamp.hour,
            avg_temp=avg_temp,
            inertia_temp=inertia_temp,
            temp_key=temp_key,
            effective_wind=effective_wind,
            wind_bucket=wind_bucket,
            bucket_counts=dict(self._collector.bucket_counts),
            avg_humidity=avg_humidity,
            solar_factor=avg_solar_factor,
            solar_vector=avg_solar_vector,
            solar_impact_raw=solar_impact_raw,
            effective_solar_impact=effective_solar_impact,
            total_energy_kwh=total_energy_kwh,
            learning_energy_kwh=learning_energy_kwh,
            guest_impact_kwh=guest_impact_kwh,
            expected_kwh=expected_kwh,
            base_expected_kwh=base_expected_kwh,
            unit_breakdown={
                eid: round(kwh, 3)
                for eid, kwh in self._hourly_delta_per_unit.items()
                if kwh > 0
            },
            unit_expected={
                eid: round(kwh, 3)
                for eid, kwh in self._hourly_expected_per_unit.items()
                if kwh > 0
            },
            unit_expected_base={
                eid: round(kwh, 3)
                for eid, kwh in self._hourly_expected_base_per_unit.items()
                if kwh > 0
            },
            aux_impact_kwh=aux_impact_kwh,
            aux_fraction=aux_fraction,
            is_aux_dominant=is_aux_dominant,
            sample_count=self._collector.sample_count,
            unit_modes={
                entity_id: mode
                for entity_id, mode in self._unit_modes.items()
            },
            forecasted_kwh=forecasted_kwh,
            forecasted_kwh_primary=forecasted_kwh_primary,
            forecasted_kwh_secondary=forecasted_kwh_secondary,
            forecasted_kwh_gross=forecasted_kwh_gross,
            forecasted_kwh_gross_primary=forecasted_kwh_gross_primary,
            forecasted_kwh_gross_secondary=forecasted_kwh_gross_secondary,
            forecast_source=forecast_source,
            recommendation_state=recommendation_state,
            correction_percent=correction_percent,
            potential_solar_factor=potential_solar_factor,
            was_cooldown_active=was_cooldown_active,
        )

    @property
    def model(self) -> "ModelState":
        """Cached ModelState holding live references to learned model data.

        Since ModelState stores references (not copies), reading
        ``self.model.correlation_data`` is equivalent to reading
        ``self._correlation_data`` — but makes the dependency explicit
        and eliminates private-field access from external modules.

        The instance is created lazily and reused.  If the underlying
        dicts are replaced (e.g. by storage load), call
        ``_invalidate_model_cache()`` to force re-creation.
        """
        if self._model_cache is None:
            self._model_cache = self.get_model_state()
        return self._model_cache

    def _invalidate_model_cache(self) -> None:
        """Force re-creation of the cached ModelState on next access."""
        self._model_cache = None

    def get_model_state(self) -> "ModelState":
        """Return a snapshot of the current learned model state.

        Returns references (not deep copies) for performance.
        Consumers must treat the returned object as read-only.
        """
        from .observation import ModelState

        return ModelState(
            correlation_data=self._correlation_data,
            correlation_data_per_unit=self._correlation_data_per_unit,
            observation_counts=self._observation_counts,
            aux_coefficients=self._aux_coefficients,
            aux_coefficients_per_unit=self._aux_coefficients_per_unit,
            solar_coefficients_per_unit=self._solar_coefficients_per_unit,
            learned_u_coefficient=self._learned_u_coefficient,
            learning_buffer_global=self._learning_buffer_global,
            learning_buffer_per_unit=self._learning_buffer_per_unit,
            learning_buffer_aux_per_unit=self._learning_buffer_aux_per_unit,
            learning_buffer_solar_per_unit=self._learning_buffer_solar_per_unit,
            daily_history=self._daily_history,
            hourly_log=self._hourly_log,
        )

    async def _process_hourly_data(self, current_time: datetime):
        """Process the accumulated data for the past hour.

        Triggered at the start of a new hour (or restart).
        1. Calculates final stats for the completed hour (Temp, Effective Wind).
        2. Updates learning models (Correlation & Solar Coefficients).
        3. Persists hourly log entry.
        4. Saves data to storage.
        """
        # --- Aux Cooldown / Decay Management ---
        # Capture initial state for learning (so convergent hour is still protected)
        was_cooldown_active = self._aux_cooldown_active

        if self._aux_cooldown_active and self._aux_cooldown_start_time:
            # Calculate elapsed time in hours
            # Ensure timezone awareness compatibility
            start_ts = self._aux_cooldown_start_time
            if start_ts.tzinfo is None and current_time.tzinfo:
                start_ts = start_ts.replace(tzinfo=current_time.tzinfo)
            elif start_ts.tzinfo and current_time.tzinfo is None:
                start_ts = start_ts.replace(tzinfo=None)

            elapsed = (current_time - start_ts).total_seconds() / 3600.0

            # Condition 1: Max Timeout
            if elapsed >= COOLDOWN_MAX_HOURS:
                _LOGGER.info(f"Aux Cooldown: Max duration exceeded ({elapsed:.1f}h). Exiting lock.")
                self._aux_cooldown_active = False
                self._aux_cooldown_start_time = None

            # Condition 2: Min Duration + Convergence
            elif elapsed >= COOLDOWN_MIN_HOURS:
                # Calculate Convergence Ratio for Affected Units
                actual_sum = 0.0
                expected_sum = 0.0

                targets = self.aux_affected_entities or []

                for entity_id in targets:
                    actual_sum += self._hourly_delta_per_unit.get(entity_id, 0.0)
                    expected_sum += self._hourly_expected_base_per_unit.get(entity_id, 0.0)

                # Check Convergence
                if expected_sum > ENERGY_GUARD_THRESHOLD:
                    ratio = actual_sum / expected_sum
                    if ratio >= COOLDOWN_CONVERGENCE_THRESHOLD:
                        _LOGGER.info(f"Aux Cooldown: Convergence reached ({ratio:.1%}). Exiting lock.")
                        self._aux_cooldown_active = False
                        self._aux_cooldown_start_time = None
                    else:
                        _LOGGER.debug(f"Aux Cooldown: Active. Ratio {ratio:.1%} < {COOLDOWN_CONVERGENCE_THRESHOLD:.0%}")
                else:
                    _LOGGER.debug("Aux Cooldown: Active. No expected consumption to verify convergence.")

        # Calculate what was expected for the full hour (using aggregates)
        avg_temp = 0.0
        calculated_effective_wind = 0.0
        avg_solar_factor = 0.0
        avg_solar_vector = (0.0, 0.0)
        wind_bucket = "normal"
        aux_fraction = 0.0
        is_aux_dominant = False  # For learning purposes

        if self._collector.sample_count > 0:
             # Calculate averages from aggregates
             avg_temp = self._collector.temp_sum / self._collector.sample_count

             # Calculate 90th percentile for effective wind (Nearest Rank)
             eff_winds = sorted(self._collector.wind_values)
             idx = math.ceil(0.9 * len(eff_winds)) - 1
             calculated_effective_wind = eff_winds[idx]

             # Determine bucket for the passed hour
             if self.solar_enabled:
                 avg_solar_factor = self._collector.solar_sum / self._collector.sample_count
                 avg_solar_vector = (
                     self._collector.solar_vector_s_sum / self._collector.sample_count,
                     self._collector.solar_vector_e_sum / self._collector.sample_count
                 )

             # Determine base wind bucket (always physical now)
             wind_bucket = self._get_wind_bucket(calculated_effective_wind)

             # Calculate Aux Fraction
             aux_fraction = self._collector.aux_count / self._collector.sample_count

             # Check auxiliary heating dominant? (For Learning & Log)
             # Use 80% threshold for pure modes to ensure clean learning data (tolerant to restart)
             if aux_fraction >= 0.80:
                 is_aux_dominant = True

        # 0. Close any gap from the end of the previous hour
        # Use Mean Imputation (Aggregates) to ensure consistency with logged hour stats
        if self._collector.last_minute_processed is not None:
             self._close_hour_gap(
                 current_time,
                 self._collector.last_minute_processed,
                 avg_temp=avg_temp,
                 avg_wind=calculated_effective_wind,
                 avg_solar=avg_solar_factor,
                 is_aux_active=is_aux_dominant
             )

        # Save Last Hour Stats for Sensors

        # Solar Optimization Learning
        rec_state_avg = "none"
        actual_correction = self.solar_correction_percent
        potential_factor_avg = 0.0

        if self.solar_enabled and self._collector.sample_count > 0:
             # Calculate average potential factor for the hour
             target_dt = self._collector.start_time if self._collector.start_time else current_time
             mid_point = target_dt + timedelta(minutes=30)
             elev, azimuth = self.solar.get_approx_sun_pos(mid_point)

             # Fetch current cloud for learning constraint (and potential calc fallback)
             current_cloud = self._get_cloud_coverage()

             if actual_correction >= 5.0:
                 potential_factor_avg = (avg_solar_factor / (actual_correction / 100.0))
             else:
                 # Fallback if screens were closed (can't derive potential from effective)
                 potential_factor_avg = self.solar.calculate_solar_factor(elev, azimuth, current_cloud)

             potential_factor_avg = max(0.0, min(1.0, potential_factor_avg))

             rec_state_avg = self.solar_optimizer.get_recommendation_state(avg_temp, potential_factor_avg)
             self.solar_optimizer.learn_correction_percent(rec_state_avg, elev, azimuth, actual_correction, cloud_cover=current_cloud)

        # Calculate Inertia Temp for Learning (3 previous hours + this hour's average)
        # We look at hourly_log (previous hours) + avg_temp (this hour)
        # Use helper to ensure we don't pick up stale logs after a gap
        inertia_temps = self._get_recent_log_temps(current_time)
        inertia_temps.append(avg_temp)

        inertia_avg = self._calculate_weighted_inertia(inertia_temps)

        # Use inertia average for correlation key
        temp_key = str(int(round(inertia_avg)))

        # Total Energy (for logging & display)
        total_energy_kwh = self._collector.energy_hour

        # Energy for Global Model Learning (exclude guest and DHW modes)
        learning_energy_kwh = 0.0
        guest_impact_kwh = 0.0
        for entity_id, actual_kwh in self._hourly_delta_per_unit.items():
            unit_mode = self.get_unit_mode(entity_id)
            if unit_mode in (MODE_GUEST_HEATING, MODE_GUEST_COOLING):
                # Guest units are not tracked in expected - their full consumption is the impact
                guest_impact_kwh += actual_kwh
            elif unit_mode in (MODE_OFF, MODE_DHW):
                # Off and DHW energy do not inform the space-heating model
                pass
            else:
                learning_energy_kwh += actual_kwh


        # Expected (Physics / Reality)
        # Use the accumulated expected energy (Weighted sum of Normal/Aux minutes)
        # This ensures the log matches the "Expected So Far" calculation and respects mixed hours.
        expected_kwh = self._collector.expected_energy_hour

        # Forecasted (Plan) - Shadow Forecasting Logic
        # Calculate predictions for ALL sources using REFERENCE weather locked at midnight.
        forecasted_kwh = 0.0
        forecasted_kwh_primary = 0.0
        forecasted_kwh_secondary = 0.0

        target_dt = self._collector.start_time if self._collector.start_time else current_time

        # Get items for all sources
        f_item_blended = self.forecast.get_forecast_for_hour(target_dt, source='reference')
        f_item_primary = self.forecast.get_forecast_for_hour(target_dt, source='primary_reference')
        f_item_secondary = self.forecast.get_forecast_for_hour(target_dt, source='secondary_reference')

        # Setup context for calculation
        local_inertia_seed = list(inertia_temps[:-1])
        weather_wind_unit = self._get_weather_wind_unit()
        current_cloud = self._get_cloud_coverage()

        # Helper to process an item
        def _get_f_kwh(item, ignore_aux=False):
            if not item: return None
            val, _, _, _, _, _, _, _ = self.forecast._process_forecast_item(
                item, list(local_inertia_seed), weather_wind_unit, current_cloud, ignore_aux=ignore_aux
            )
            return val

        # Calculate values
        # Net Forecasts (Shadow) - Tracks expected consumption including Aux reduction
        forecasted_kwh = _get_f_kwh(f_item_blended, ignore_aux=False)
        forecasted_kwh_primary = _get_f_kwh(f_item_primary, ignore_aux=False)
        forecasted_kwh_secondary = _get_f_kwh(f_item_secondary, ignore_aux=False)

        # Gross Forecasts (Thermodynamic) - Tracks pure thermodynamic demand (Plan)
        # Used for accurate "Gross vs Gross" error calculation
        forecasted_kwh_gross = _get_f_kwh(f_item_blended, ignore_aux=True)
        forecasted_kwh_gross_primary = _get_f_kwh(f_item_primary, ignore_aux=True)
        forecasted_kwh_gross_secondary = _get_f_kwh(f_item_secondary, ignore_aux=True)

        # Fallback if no forecast available (Assume perfect forecast for baseline)
        if not f_item_blended: forecasted_kwh = expected_kwh
        if not f_item_primary: forecasted_kwh_primary = forecasted_kwh
        if not f_item_secondary: forecasted_kwh_secondary = forecasted_kwh

        # Calculate Base Expected (for Learning Reference)
        # This is the PHYSICAL BASE (without Aux or Solar)
        # NEW: Use robust calculation for base reference too
        # We need "Base" (Normal) prediction for the given conditions.

        # Calculate and store aux_impact in self.data
        # We want the *theoretical* impact of aux if it were active
        # calculate_total_power for aux active gives us: Base - Aux_Impact
        # So Aux_Impact = Base - (Net)
        # Or we can read it from the breakdown!

        # Proportional Aux Impact:
        # Use the precise accumulated value (No threshold) to ensure accurate
        # accounting of auxiliary impact even during mixed-mode hours.
        aux_impact_kwh = round(self._collector.aux_impact_hour, 3)

        # OPTIMIZATION: Call once with correct aux/solar context.
        # This returns BOTH the base values (unaffected by aux) AND the aux/solar reduction.
        res_analysis = self.statistics.calculate_total_power(
            inertia_avg, # Use inertia temp (float)
            calculated_effective_wind,
            0.0, # This arg is ignored by calculate_total_power
            is_aux_active=is_aux_dominant,
            detailed=False,
            override_solar_factor=avg_solar_factor,
            override_solar_vector=avg_solar_vector,
            known_aux_impact_kwh=aux_impact_kwh,
        )

        # Global Stabilizer: Use Global Base for learning Track A
        # Prevents feedback loop from Unit Sums
        base_expected_kwh = res_analysis.get("global_base_kwh", res_analysis["breakdown"]["base_kwh"])

        self.data["last_hour_aux_impact_kwh"] = aux_impact_kwh

        # Calculate Solar Impact (Saturated)
        # Use the authoritative calculation from statistics which correctly applies saturation
        # (Solar cannot reduce consumption below zero or below base - aux)
        solar_impact = 0.0
        if self.solar_enabled:
            solar_impact = res_analysis["breakdown"]["solar_reduction_kwh"]
            # Update global attr for consistency
            self.data[ATTR_SOLAR_IMPACT] = round(solar_impact, 3)

        # Solar Thermal Battery: accumulate impact with exponential decay.
        # Carries residual solar heat stored in building mass into post-solar hours,
        # preventing over-prediction of consumption in the hours after peak sun.
        # Only charge/decay when solar is enabled; on solar-disabled installs stays at 0.
        if self.solar_enabled:
            self._solar_battery_state = self._solar_battery_state * SOLAR_BATTERY_DECAY + solar_impact
        effective_solar_impact = self._solar_battery_state

        # Update Last Hour Data in Coordinator
        self.data[ATTR_LAST_HOUR_ACTUAL] = round(total_energy_kwh, 3)
        self.data[ATTR_LAST_HOUR_EXPECTED] = round(expected_kwh, 3)
        self.data[ATTR_LAST_HOUR_DEVIATION] = round(total_energy_kwh - expected_kwh, 3)
        self.data["last_hour_wind_bucket"] = wind_bucket
        self.data["last_hour_solar_impact_kwh"] = round(effective_solar_impact, 3)
        self.data["last_hour_guest_impact_kwh"] = round(guest_impact_kwh, 3)

        if expected_kwh > ENERGY_GUARD_THRESHOLD:
            self.data[ATTR_LAST_HOUR_DEVIATION_PCT] = round(((total_energy_kwh - expected_kwh) / expected_kwh) * 100, 1)
        else:
            self.data[ATTR_LAST_HOUR_DEVIATION_PCT] = 0.0

        if self._collector.sample_count > 0:
            # Determine Learning Eligibility
            # Skip learning if Mixed Mode to avoid model corruption
            # Skip entirely if Daily Learning Mode is active (strategies own midnight writes)
            is_mixed_mode = (MIXED_MODE_LOW < aux_fraction < MIXED_MODE_HIGH)
            should_learn = self.learning_enabled and not is_mixed_mode and not self.daily_learning_mode

            # Per-unit learning: DirectMeter sensors continue via Track A even when
            # daily_learning_mode blocks global learning.  WeightedSmear sensors are
            # excluded — their meter data is MPC-tainted. (#776)
            should_learn_per_unit = self.learning_enabled and not is_mixed_mode if self.daily_learning_mode else None

            # Dual Interference Guard:
            # If both Solar and Aux are significant, we cannot reliably attribute deviation.
            is_dual_interference = (solar_impact > DUAL_INTERFERENCE_THRESHOLD) and (aux_impact_kwh > DUAL_INTERFERENCE_THRESHOLD)
            if is_dual_interference:
                should_learn = False

            # Detect Guest Mode Activity
            # If any unit is in Guest Mode, aux learning must be disabled to prevent pollution
            # Base and solar learning can continue as they use learning_energy_kwh (guest-excluded)
            has_guest_activity = any(
                mode in (MODE_GUEST_HEATING, MODE_GUEST_COOLING)
                for mode in self._unit_modes.values()
            )

            # --- Build immutable observation snapshot (Issue #775) ---
            obs = self._build_hourly_observation(
                current_time,
                avg_temp=avg_temp,
                inertia_temp=inertia_avg,
                temp_key=temp_key,
                effective_wind=calculated_effective_wind,
                wind_bucket=wind_bucket,
                avg_solar_factor=avg_solar_factor,
                avg_solar_vector=avg_solar_vector,
                solar_impact_raw=solar_impact,
                effective_solar_impact=effective_solar_impact,
                total_energy_kwh=total_energy_kwh,
                learning_energy_kwh=learning_energy_kwh,
                guest_impact_kwh=guest_impact_kwh,
                expected_kwh=expected_kwh,
                base_expected_kwh=base_expected_kwh,
                aux_impact_kwh=aux_impact_kwh,
                aux_fraction=aux_fraction,
                is_aux_dominant=is_aux_dominant,
                was_cooldown_active=was_cooldown_active,
                forecasted_kwh=forecasted_kwh,
                forecasted_kwh_primary=forecasted_kwh_primary,
                forecasted_kwh_secondary=forecasted_kwh_secondary,
                forecasted_kwh_gross=forecasted_kwh_gross,
                forecasted_kwh_gross_primary=forecasted_kwh_gross_primary,
                forecasted_kwh_gross_secondary=forecasted_kwh_gross_secondary,
                forecast_source=f_item_blended.get("_source") if f_item_blended else None,
                recommendation_state=rec_state_avg,
                correction_percent=round(actual_correction, 1),
                potential_solar_factor=round(potential_factor_avg, 3),
            )

            # Delegate Learning to LearningManager (#775 Phase 3)
            from .observation import LearningConfig

            # Only DirectMeter sensors participate in hourly per-unit learning;
            # WeightedSmear sensors are excluded (MPC-tainted meter data). (#776)
            from .observation import DirectMeter
            hourly_sensors = [
                sid for sid, strat in self._unit_strategies.items()
                if isinstance(strat, DirectMeter)
            ] if self.daily_learning_mode else self.energy_sensors

            learning_config = LearningConfig(
                learning_enabled=should_learn,
                solar_enabled=self.solar_enabled,
                learning_rate=self.learning_rate,
                balance_point=self.balance_point,
                energy_sensors=hourly_sensors,
                aux_impact=self._get_aux_impact_kw(temp_key, wind_bucket),
                solar_calculator=self.solar,
                get_predicted_unit_base_fn=self._get_predicted_kwh_per_unit,
                aux_affected_entities=self.aux_affected_entities,
                has_guest_activity=has_guest_activity,
                per_unit_learning_enabled=should_learn_per_unit,
            )

            learning_result = self.learning.process_learning(
                obs=obs,
                model=self.get_model_state(),
                config=learning_config,
            )

            # Accumulate Real TDD (Thermal Degree Days)
            # Use ABSOLUTE difference to handle both Heating and Cooling
            tdd_contribution = abs(self.balance_point - avg_temp) / 24.0
            current_tdd_acc = self.data.get(ATTR_TDD, 0.0)
            self.data[ATTR_TDD] = round(current_tdd_acc + tdd_contribution, 3)

            # Prepare Unit Breakdown for Log (rounded)
            unit_breakdown = {
                eid: round(kwh, 3)
                for eid, kwh in self._hourly_delta_per_unit.items()
                if kwh > 0
            }

            # Prepare Unit Expected Breakdown for Log
            # This captures the true "mixed" expectation for the hour (solving "Majority Rule" contamination)
            unit_expected_breakdown = {
                eid: round(kwh, 3)
                for eid, kwh in self._hourly_expected_per_unit.items()
                if kwh > 0
            }

            # Hourly Log Entry
            # Use _hourly_start_time to represent the START of the hour period, not the END

            # If skipped due to mixed mode, clarify status
            final_learning_status = learning_result.get("learning_status", "unknown")
            if is_mixed_mode and self.learning_enabled:
                if final_learning_status != "cooldown_post_aux":
                    final_learning_status = "skipped_mixed_mode"
            elif is_dual_interference and self.learning_enabled:
                if final_learning_status != "cooldown_post_aux":
                    final_learning_status = "skipped_dual_interference"

            # Calculate Thermodynamic Gross (Actual + Aux + Solar Adjustment)
            # Make mode-aware: Add in Heating, Subtract in Cooling
            # Use effective_solar_impact (battery-smoothed) so the gross reflects
            # the full residual solar heat carried in building mass.
            solar_adjustment = effective_solar_impact
            if avg_temp >= self.balance_point:
                solar_adjustment = -effective_solar_impact

            thermodynamic_gross_kwh = total_energy_kwh + aux_impact_kwh + solar_adjustment

            log_entry = {
                "timestamp": self._collector.start_time.isoformat() if self._collector.start_time else current_time.isoformat(),
                "hour": self._collector.start_time.hour if self._collector.start_time else current_time.hour,
                "temp": round(avg_temp, 1),
                "tdd": round(tdd_contribution, 3),
                "unit_breakdown": unit_breakdown,
                "unit_expected_breakdown": unit_expected_breakdown,
                "temp_key": temp_key,
                "inertia_temp": round(inertia_avg, 2),
                "effective_wind": round(calculated_effective_wind, 2),
                "wind_bucket": wind_bucket,
                "humidity": obs.avg_humidity,
                "actual_kwh": round(total_energy_kwh, 3),
                "expected_kwh": round(expected_kwh, 3),
                "thermodynamic_gross_kwh": round(thermodynamic_gross_kwh, 3),
                "forecasted_kwh": round(forecasted_kwh, 3),
                "forecasted_kwh_primary": round(forecasted_kwh_primary, 3),
                "forecasted_kwh_secondary": round(forecasted_kwh_secondary, 3),
                "forecasted_kwh_gross": round(forecasted_kwh_gross, 3) if forecasted_kwh_gross is not None else None,
                "forecasted_kwh_gross_primary": round(forecasted_kwh_gross_primary, 3) if forecasted_kwh_gross_primary is not None else None,
                "forecasted_kwh_gross_secondary": round(forecasted_kwh_gross_secondary, 3) if forecasted_kwh_gross_secondary is not None else None,
                "forecast_source": f_item_blended.get("_source") if f_item_blended else None,
                "deviation": round(total_energy_kwh - expected_kwh, 3),
                "deviation_pct": self.data.get(ATTR_LAST_HOUR_DEVIATION_PCT, 0.0),
                "auxiliary_active": is_aux_dominant,
                "aux_impact_kwh": aux_impact_kwh,
                "guest_impact_kwh": round(guest_impact_kwh, 3),
                "solar_factor": round(avg_solar_factor, 3),
                "solar_vector_s": round(avg_solar_vector[0], 3),
                "solar_vector_e": round(avg_solar_vector[1], 3),
                "solar_impact_kwh": round(effective_solar_impact, 3),
                "solar_impact_raw_kwh": round(solar_impact, 3),
                "primary_entity": self.weather_entity,
                "secondary_entity": self.entry.data.get(CONF_SECONDARY_WEATHER_ENTITY),
                "crossover_day": self.entry.data.get(CONF_FORECAST_CROSSOVER_DAY, DEFAULT_FORECAST_CROSSOVER_DAY),
                # Model Update Info
                "model_temp_key": temp_key,
                "model_base_before": round(learning_result["model_base_before"], 5),
                "model_base_after": round(learning_result["model_base_after"], 5),
                "model_updated": learning_result["model_updated"],
                "aux_model_updated": learning_result.get("aux_model_updated", False),
                "aux_model_before": learning_result.get("aux_model_before"),
                "aux_model_after": learning_result.get("aux_model_after"),
                "learning_status": final_learning_status,
                "recommendation_state": rec_state_avg,
                "correction_percent": round(actual_correction, 1),
                "potential_solar_factor": round(potential_factor_avg, 3),
                # Only filter out MODE_HEATING (the true default) to reduce log clutter
                # Cooling, off, and guest modes MUST be logged for correct historical reconstruction
                "unit_modes": {
                    entity_id: mode
                    for entity_id, mode in self._unit_modes.items()
                    if mode != MODE_HEATING
                },
            }
            # Guard against duplicate entries (e.g., crash-at-boundary + restart scenario)
            entry_ts = log_entry["timestamp"]
            if any(e.get("timestamp") == entry_ts for e in self._hourly_log):
                _LOGGER.warning(f"Duplicate hourly entry detected for {entry_ts}, skipping append.")
            else:
                self._hourly_log.append(log_entry)

            # Retention Policy (keep last 2160 hours = 90 days)
            if len(self._hourly_log) > 2160:
                self._hourly_log = self._hourly_log[-2160:]

            _LOGGER.info(f"Hourly Update: Temp={avg_temp:.1f}, Wind={wind_bucket}, Energy={total_energy_kwh:.2f}, Solar={avg_solar_factor:.2f}")

            # CSV Auto-logging (if enabled)
            await self.storage.append_hourly_log_csv(log_entry)

        # Accumulate hourly aux breakdown into daily stats
        for entity_id, stats in self._collector.aux_breakdown.items():
            if entity_id not in self._daily_aux_breakdown:
                self._daily_aux_breakdown[entity_id] = {"allocated": 0.0, "overflow": 0.0}
            self._daily_aux_breakdown[entity_id]["allocated"] += stats.get("allocated", 0.0)
            self._daily_aux_breakdown[entity_id]["overflow"] += stats.get("overflow", 0.0)

        # Accumulate orphaned savings into daily total
        self._daily_orphaned_aux += self._collector.orphaned_aux

        # Save Logic (force save on hourly boundary)
        await self._async_save_data(force=True)

        # Reset all hour-scoped accumulators atomically (#775).
        # In-place clearing preserves coordinator aliases for dict fields.
        self._collector.reset()

    def _aggregate_daily_logs(self, day_logs: list[dict]) -> dict:
        """Aggregate hourly logs into a daily summary."""
        if not day_logs:
            return {}

        total_kwh = sum(e.get("actual_kwh", 0.0) for e in day_logs)
        expected_kwh = sum(e.get("expected_kwh", 0.0) for e in day_logs)
        forecasted_kwh = sum(e.get("forecasted_kwh", 0.0) for e in day_logs)
        solar_impact = sum(e.get("solar_impact_kwh", 0.0) for e in day_logs)
        aux_impact = sum(e.get("aux_impact_kwh", 0.0) for e in day_logs)
        guest_impact = sum(e.get("guest_impact_kwh", 0.0) for e in day_logs)

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
                if temp >= self.balance_point:
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
        if self.solar_enabled:
            hourly_vectors["solar_rad"] = [None] * 24

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
            if self.solar_enabled:
                hourly_vectors["solar_rad"][hour] = hourly_avg_solar

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

    def _backfill_daily_from_hourly(self) -> int:
        """Backfill missing details in daily history from hourly logs."""
        if not self._hourly_log:
            return 0

        # Group logs by date
        logs_by_date = {}
        for entry in self._hourly_log:
            date_key = entry["timestamp"][:10]
            if date_key not in logs_by_date:
                logs_by_date[date_key] = []
            logs_by_date[date_key].append(entry)

        updated_count = 0

        for date_key, logs in logs_by_date.items():
            # Aggregate stats from logs
            agg = self._aggregate_daily_logs(logs)

            if date_key not in self._daily_history:
                # If we have enough logs (e.g. > 12h) we could create it,
                # but let's be safe and only enrich existing or create if > 20h
                if len(logs) >= 20:
                     self._daily_history[date_key] = agg
                     updated_count += 1
            else:
                curr = self._daily_history[date_key]
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
                self._daily_history[date_key].update(agg)
                updated_count += 1

        if updated_count > 0:
            _LOGGER.info(f"Backfilled/Enriched {updated_count} daily history entries from hourly logs.")

        return updated_count

    async def _run_track_c_midnight_sync(
        self, day_logs: list[dict], date_key: str
    ) -> tuple[float, list] | None:
        """Fetch MPC thermal data and run the ThermodynamicEngine Midnight Sync.

        Returns (total_synthetic_el, distribution) where:
          - total_synthetic_el: sum of synthetic_kwh_el across all 24 hours —
            the weather-smeared electrical equivalent used as q_adjusted in learning.
          - distribution: the full list of HourlyDistribution dicts for storage
            (enables future per-hour visualisation without recomputing).
        Returns None if the sync cannot proceed (MPC unavailable, empty buffer, etc.).
        """
        from homeassistant.exceptions import ServiceNotFound, HomeAssistantError

        service_data = {}
        if self.mpc_entry_id:
            service_data["entry_id"] = self.mpc_entry_id

        try:
            response = await self.hass.services.async_call(
                "heatpump_mpc",
                "get_sh_hourly",
                service_data,
                blocking=True,
                return_response=True,
            )
        except ServiceNotFound:
            _LOGGER.warning("Track C: heatpump_mpc.get_sh_hourly service not found — falling back to Track B.")
            return None
        except HomeAssistantError as err:
            _LOGGER.warning("Track C: MPC service call failed (%s) — falling back to Track B.", err)
            return None

        # Service returns {"buffer": [...]} per heatpump_mpc/__init__.py.
        # Guard against alternative wrapping keys for forward-compatibility.
        if isinstance(response, dict):
            mpc_records = response.get("buffer", response.get("data", response.get("hourly", [])))
        elif isinstance(response, list):
            mpc_records = response
        else:
            _LOGGER.warning("Track C: Unexpected MPC response format (%s) — falling back to Track B.", type(response))
            return None

        if not mpc_records:
            _LOGGER.warning("Track C: MPC returned empty hourly buffer for %s — falling back to Track B.", date_key)
            return None

        # --- Fetch COP model parameters for per-hour conversion ---
        cop_params = None
        try:
            cop_response = await self.hass.services.async_call(
                "heatpump_mpc",
                "get_cop_params",
                service_data,
                blocking=True,
                return_response=True,
            )
            if isinstance(cop_response, dict) and "eta_carnot" in cop_response:
                cop_params = cop_response
                _LOGGER.debug(
                    "Track C: COP params received — η=%.3f, f_defrost=%.2f, LWT=%.1f",
                    cop_params["eta_carnot"], cop_params.get("f_defrost", 0.85), cop_params.get("lwt", 35.0),
                )
            else:
                _LOGGER.info("Track C: get_cop_params returned unexpected format — using daily avg COP fallback.")
        except (ServiceNotFound, HomeAssistantError) as err:
            _LOGGER.info("Track C: get_cop_params unavailable (%s) — using daily avg COP fallback.", err)

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
        #                  gain residual carries into evening hours (mirrors SOLAR_BATTERY_DECAY)
        weather_data = []
        log_by_hour = {e.get("hour", -1): e for e in day_logs}

        # Solar battery pre-pass: accumulate decay across hours so that afternoon
        # solar gain reduces evening loss weights, matching Track A's solar battery model.
        solar_battery = 0.0
        # Build ordered hour → raw_solar mapping for the 24-hour sequence.
        hours_ordered = sorted(log_by_hour.keys())
        solar_residual_by_hour: dict[int, float] = {}
        for h in range(24):
            log_h = log_by_hour.get(h, {})
            raw_solar_h = log_h.get("solar_factor")
            raw_solar_h = raw_solar_h if raw_solar_h is not None else 0.0
            solar_battery = solar_battery * SOLAR_BATTERY_DECAY + raw_solar_h
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
                else self.balance_point
            )
            eff_wind = log_entry.get("effective_wind")
            effective_wind: float = eff_wind if eff_wind is not None else 0.0

            # Fix 2: 3-bucket wind multiplier — mirrors Track A's discrete wind buckets
            # (normal / high / extreme) rather than an unbounded linear scale.
            if effective_wind >= self.extreme_wind_threshold:
                wind_factor = 1.6
            elif effective_wind >= self.wind_threshold:
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
            raw_outdoor = raw_t if raw_t is not None else self.balance_point
            rh = log_entry.get("humidity")
            rh = rh if rh is not None else 50.0

            weather_data.append({
                "datetime": record["datetime"],
                "delta_t": max(0.0, self.balance_point - outdoor_temp),
                "wind_factor": wind_factor,
                "solar_factor": solar_factor,
                "outdoor_temp": raw_outdoor,
                "humidity": rh,
            })

        engine = ThermodynamicEngine(balance_point=self.balance_point)
        try:
            distribution = engine.calculate_synthetic_baseline(mpc_records, weather_data, cop_params=cop_params)
        except Exception as err:
            _LOGGER.error("Track C: ThermodynamicEngine failed (%s) — falling back to Track B.", err)
            return None

        total_synthetic_el = sum(h["synthetic_kwh_el"] for h in distribution)
        _LOGGER.info(
            "Track C Midnight Sync %s: total_synthetic_el=%.3f kWh from %d MPC records.",
            date_key, total_synthetic_el, len(mpc_records),
        )
        return total_synthetic_el, distribution

    def _apply_strategies_to_global_model(
        self,
        day_logs: list[dict],
        track_c_distribution: list[dict] | None,
    ) -> int:
        """Delegate to LearningManager — see learning.py for implementation."""
        from homeassistant.util import dt as _dt
        return self.learning.apply_strategies_to_global_model(
            day_logs=day_logs,
            track_c_distribution=track_c_distribution,
            strategies=self._unit_strategies,
            model=self.get_model_state(),
            learning_rate=self.learning_rate,
            balance_point=self.balance_point,
            wind_threshold=self.wind_threshold,
            extreme_wind_threshold=self.extreme_wind_threshold,
            parse_datetime_fn=_dt.parse_datetime,
        )

    def _replay_per_unit_models(self, day_entries: list[dict]) -> None:
        """Delegate to LearningManager — see learning.py for implementation."""
        self.learning.replay_per_unit_models(
            day_entries=day_entries,
            strategies=self._unit_strategies,
            model=self.get_model_state(),
            learning_rate=self.learning_rate,
        )

    @staticmethod
    def _compute_excluded_mode_energy(day_logs: list[dict]) -> float:
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

    async def _process_daily_data(self, date_obj):
        """Process end of day."""
        key = date_obj.isoformat()
        day_logs = [e for e in self._hourly_log if e["timestamp"].startswith(key)]

        # Validation: Ensure we have enough data (Kelvin Protocol)
        if len(day_logs) < 20:
            _LOGGER.warning(
                "Daily processing for %s: Incomplete data (%d/24 hours). Vectors may have gaps.",
                key,
                len(day_logs),
            )

        # Use Aggregation Helper to ensure full schema compliance
        if day_logs:
            daily_stats = self._aggregate_daily_logs(day_logs)
        else:
            # Fallback if no logs (Downtime?)
            tdd = self.data.get(ATTR_TDD, 0.0)
            kwh = self._accumulated_energy_today
            avg_temp = self.balance_point - tdd  # Approx

            # Fallback vectors
            empty_vector = [None] * 24
            hourly_vectors = {
                "temp": list(empty_vector),
                "wind": list(empty_vector),
                "tdd": list(empty_vector),
                "actual_kwh": list(empty_vector),
            }
            if self.solar_enabled:
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

        self._daily_history[key] = daily_stats

        # Daily Learning Mode — baseline selection and strategy dispatch (#776)
        current_indoor_temp = None
        if self.indoor_temp_sensor:
            current_indoor_temp = self._get_float_state(self.indoor_temp_sensor)

        # Mode filtering (#789): exclude OFF/DHW/Guest/Cooling energy from
        # daily learning so Track B/C match Track A's filtering semantics.
        excluded_mode_kwh = self._compute_excluded_mode_energy(day_logs) if day_logs else 0.0
        q_adjusted = daily_stats["kwh"] - excluded_mode_kwh
        track_c_distribution = None

        if excluded_mode_kwh > 0.0:
            _LOGGER.debug(
                "Daily mode filter (#789): excluded %.3f kWh "
                "(OFF/DHW/Guest/Cooling) from %.2f kWh total.",
                excluded_mode_kwh, daily_stats["kwh"],
            )

        if self.track_c_enabled and self.daily_learning_mode and day_logs:
            # Track C: replace electrical baseline with thermodynamic synthetic baseline.
            track_c_result = await self._run_track_c_midnight_sync(day_logs, key)
            if track_c_result is not None:
                track_c_kwh, track_c_distribution = track_c_result

                # Compute q_adjusted from strategy contributions for U-coefficient.
                # Only include non-MPC sensors that are in a learning-eligible mode (#789).
                non_mpc_daily_kwh = 0.0
                if self.mpc_managed_sensor:
                    for log_entry in day_logs:
                        breakdown = log_entry.get("unit_breakdown", {})
                        unit_modes = log_entry.get("unit_modes", {})
                        for sid in self.energy_sensors:
                            if sid != self.mpc_managed_sensor:
                                mode = unit_modes.get(sid, MODE_HEATING)
                                if mode not in MODES_EXCLUDED_FROM_GLOBAL_LEARNING:
                                    non_mpc_daily_kwh += breakdown.get(sid, 0.0)

                q_adjusted = track_c_kwh + non_mpc_daily_kwh
                self._daily_history[key]["track_c_kwh"] = round(q_adjusted, 3)
                self._daily_history[key]["track_c_kwh_mpc_only"] = round(track_c_kwh, 3)
                self._daily_history[key]["track_c_kwh_non_mpc"] = round(non_mpc_daily_kwh, 3)
                self._daily_history[key]["track_c_distribution"] = track_c_distribution
            else:
                _LOGGER.warning("Track C unavailable for %s — applying Track B correction.", key)
                if self.thermal_mass_kwh_per_degree > 0.0 and current_indoor_temp is not None and self._last_midnight_indoor_temp is not None:
                    delta_t_indoor = current_indoor_temp - self._last_midnight_indoor_temp
                    base_kwh = daily_stats["kwh"] - excluded_mode_kwh
                    q_adjusted = base_kwh - (self.thermal_mass_kwh_per_degree * delta_t_indoor)
                    _LOGGER.debug(f"Track B fallback: Adjusted kWh from {base_kwh:.2f} to {q_adjusted:.2f} (ΔT={delta_t_indoor:.2f}°C)")
        elif self.thermal_mass_kwh_per_degree > 0.0 and current_indoor_temp is not None and self._last_midnight_indoor_temp is not None:
            delta_t_indoor = current_indoor_temp - self._last_midnight_indoor_temp
            base_kwh = daily_stats["kwh"] - excluded_mode_kwh
            q_adjusted = base_kwh - (self.thermal_mass_kwh_per_degree * delta_t_indoor)
            _LOGGER.debug(f"Daily Learning: Adjusted kWh from {base_kwh:.2f} to {q_adjusted:.2f} based on indoor delta T {delta_t_indoor:.2f}°C")

        if self.daily_learning_mode and self.learning_enabled and daily_stats["tdd"] >= 0.5 and q_adjusted > 0:
            if len(day_logs) >= 22:
                # U-coefficient: always updated daily regardless of track.
                observed_u = q_adjusted / daily_stats["tdd"]
                if self._learned_u_coefficient is None:
                    self._learned_u_coefficient = observed_u
                else:
                    self._learned_u_coefficient = self._learned_u_coefficient + DEFAULT_DAILY_LEARNING_RATE * (observed_u - self._learned_u_coefficient)
                _LOGGER.info(f"Daily Learning: Updated U-coefficient to {self._learned_u_coefficient:.4f} (Observed: {observed_u:.4f})")

                if track_c_distribution:
                    # --- Track C: per-hour bucket learning via strategies (#776) ---
                    bucket_updates = self._apply_strategies_to_global_model(
                        day_logs, track_c_distribution,
                    )
                    _LOGGER.info(f"Track C Strategy Learning: {bucket_updates} bucket updates from 24 hours.")
                else:
                    # --- Track B: flattened daily bucket learning ---
                    # Track B exists for buildings where per-hour data is unreliable
                    # (high thermal mass, modulation floor).  A single q_adjusted/24
                    # value is written to one (avg_temp, avg_wind) bucket per day.
                    q_hourly_avg = q_adjusted / 24.0
                    avg_temp = daily_stats["temp"]
                    daily_wind = daily_stats["wind"]
                    flat_temp_key = str(int(round(avg_temp)))
                    flat_wind_bucket = self._get_wind_bucket(daily_wind)

                    if flat_temp_key not in self._correlation_data:
                        self._correlation_data[flat_temp_key] = {}
                    current_pred = self._correlation_data[flat_temp_key].get(flat_wind_bucket, 0.0)

                    if current_pred == 0.0:
                        self._correlation_data[flat_temp_key][flat_wind_bucket] = round(q_hourly_avg, 5)
                        _LOGGER.info(f"Track B Learning (Cold Start): T={flat_temp_key} W={flat_wind_bucket} -> {q_hourly_avg:.3f} kWh")
                    else:
                        new_pred = current_pred + self.learning_rate * (q_hourly_avg - current_pred)
                        self._correlation_data[flat_temp_key][flat_wind_bucket] = round(new_pred, 5)
                        _LOGGER.info(f"Track B Learning (EMA): T={flat_temp_key} W={flat_wind_bucket} -> {new_pred:.3f} kWh (was {current_pred:.3f}, actual avg {q_hourly_avg:.3f})")
            else:
                _LOGGER.info(f"Daily Learning skipped: Incomplete day ({len(day_logs)}/24 hours)")

        self.data["learned_u_coefficient"] = self._learned_u_coefficient

        if current_indoor_temp is not None:
            self._last_midnight_indoor_temp = current_indoor_temp
            # Store midnight indoor temp per day so retrain_from_history can apply
            # thermal mass correction historically without a live sensor read.
            self._daily_history[key]["midnight_indoor_temp"] = round(current_indoor_temp, 1)

        # Forecast Accuracy Tracking
        # Kelvin Protocol: Skip accuracy evaluation if learning is disabled (e.g. Vacation).
        if self.learning_enabled:
            self.forecast.log_accuracy(
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
                for i, entity_id in enumerate(self.energy_sensors)}
        }
        await self.storage.append_daily_log_csv(daily_log_entry)

        self._accumulated_energy_today = 0.0
        self._daily_individual = {} # Reset daily individual trackers
        self._daily_aux_breakdown = {} # Reset daily aux breakdown
        self._daily_orphaned_aux = 0.0 # Reset daily orphaned accumulator

        # Cleanup last energy values for removed sensors at end of day
        # This ensures we don't carry dead references forever
        current_sensors = set(self.energy_sensors)
        keys_to_remove = [k for k in self._last_energy_values if k not in current_sensors]
        for k in keys_to_remove:
            del self._last_energy_values[k]

        self.data[ATTR_TDD] = 0.0

        await self._async_save_data(force=True)
