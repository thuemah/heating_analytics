"""Sensor platform for Heating Analytics."""
from __future__ import annotations

import json
import logging
from datetime import date, timedelta

from homeassistant.components.sensor import (
    SensorEntity,
    SensorDeviceClass,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import UnitOfEnergy, PERCENTAGE, UnitOfTemperature, EntityCategory, UnitOfSpeed
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import (
    DOMAIN,
    ATTR_EFFICIENCY,
    ATTR_PREDICTED,
    ATTR_SOLAR_PREDICTED,
    ATTR_DEVIATION,
    ATTR_TDD,
    ATTR_TDD_SO_FAR,
    ATTR_FORECAST_TODAY,
    ATTR_LAST_HOUR_ACTUAL,
    ATTR_LAST_HOUR_EXPECTED,
    ATTR_LAST_HOUR_DEVIATION,
    ATTR_LAST_HOUR_DEVIATION_PCT,
    ATTR_POTENTIAL_SAVINGS,
    ATTR_CORRELATION_DATA,
    ATTR_ENERGY_TODAY,
    ATTR_EXPECTED_TODAY,
    ATTR_TDD_DAILY_STABLE,
    ATTR_TEMP_LAST_YEAR_DAY,
    ATTR_TEMP_LAST_YEAR_WEEK,
    ATTR_TEMP_LAST_YEAR_MONTH,
    ATTR_TEMP_FORECAST_TODAY,
    ATTR_TEMP_ACTUAL_TODAY,
    ATTR_TEMP_ACTUAL_WEEK,
    ATTR_TEMP_ACTUAL_MONTH,
    ATTR_MIDNIGHT_FORECAST,
    ATTR_MIDNIGHT_UNIT_ESTIMATES,
    ATTR_MIDNIGHT_UNIT_MODES,
    ATTR_FORECAST_UNCERTAINTY,
    ATTR_DEVIATION_BREAKDOWN,
    ATTR_WIND_ACTUAL_TODAY,

    # TDD Stats
    ATTR_TDD_YESTERDAY,
    ATTR_TDD_LAST_7D,
    ATTR_TDD_LAST_30D,

    # Efficiency Stats
    ATTR_EFFICIENCY_YESTERDAY,
    ATTR_EFFICIENCY_LAST_7D,
    ATTR_EFFICIENCY_LAST_30D,
    ATTR_EFFICIENCY_FORECAST_TODAY,

    SENSOR_EFFICIENCY,
    SENSOR_WEATHER_PLAN_TODAY,
    SENSOR_DEVIATION,
    SENSOR_EFFECTIVE_WIND,
    SENSOR_CORRELATION_DATA,
    SENSOR_LAST_HOUR_ACTUAL,
    SENSOR_LAST_HOUR_EXPECTED,
    SENSOR_LAST_HOUR_DEVIATION,
    SENSOR_POTENTIAL_SAVINGS,
    SENSOR_ENERGY_TODAY,
    SENSOR_ENERGY_BASELINE_TODAY,
    SENSOR_ENERGY_ESTIMATE_TODAY,
    SENSOR_FORECAST_DETAILS,
    # New Consolidated Sensors
    SENSOR_MODEL_COMPARISON_DAY,
    SENSOR_MODEL_COMPARISON_WEEK,
    SENSOR_MODEL_COMPARISON_MONTH,
    SENSOR_WEEK_AHEAD_FORECAST,
    SENSOR_PERIOD_COMPARISON,
    SENSOR_THERMAL_STATE,

    # Forecast Attributes
    ATTR_FORECAST_DETAILS,
    ATTR_DAILY_FORECAST,
    ATTR_WEEKLY_SUMMARY,
    ATTR_FORECAST_RANGE_MIN,
    ATTR_FORECAST_RANGE_MAX,
    ATTR_AVG_TEMP_FORECAST,
    ATTR_AVG_WIND_FORECAST,
    ATTR_COLDEST_DAY,
    ATTR_WARMEST_DAY,
    ATTR_TYPICAL_WEEK_KWH,
    ATTR_VS_TYPICAL_KWH,
    ATTR_VS_TYPICAL_PCT,
    ATTR_PEAK_DAY,
    ATTR_LIGHTEST_DAY,
    ATTR_WEEK_START_DATE,
    ATTR_WEEK_END_DATE,
    ATTR_FORECAST_BLEND_CONFIG,
    ATTR_FORECAST_ACCURACY_BY_SOURCE,

    # Wind attributes
    ATTR_WIND_LAST_YEAR_DAY,
    ATTR_WIND_LAST_YEAR_WEEK,
    ATTR_WIND_LAST_YEAR_MONTH,
    ATTR_WIND_ACTUAL_WEEK,
    ATTR_WIND_ACTUAL_MONTH,
    ATTR_SOLAR_FACTOR,
    ATTR_SOLAR_IMPACT,
    ATTR_SOLAR_POTENTIAL,
    ATTR_SOLAR_GAIN_NOW,
    ATTR_HEATING_LOAD_OFFSET,
    ATTR_RECOMMENDATION_STATE,

    # Mode constants
    MODE_OFF,
    MODE_GUEST_HEATING,
    MODE_GUEST_COOLING,

    WIND_UNIT_KMH,
    WIND_UNIT_KNOTS,
    WIND_UNIT_MS,
    convert_from_ms,
    ENERGY_GUARD_THRESHOLD,
    STRESS_INDEX_LIGHT,
    STRESS_INDEX_MODERATE,
    STRESS_INDEX_HEAVY,
    CONFIDENCE_MIN_SAMPLES,
    CONFIDENCE_HIGH_SAMPLES,
    CONFIDENCE_HIGH_ERROR_MAX,
    CONFIDENCE_MEDIUM_ERROR_MAX,
    FORECAST_COMPARISON_FACTOR,
)
from .helpers import convert_speed_to_ms
from .coordinator import HeatingDataCoordinator
# Explanation module import
from .explanation import ExplanationFormatter
from homeassistant.util import dt as dt_util

from .sensors.base import HeatingAnalyticsBaseSensor
from .sensors.comparison import (
    HeatingModelComparisonDaySensor,
    HeatingModelComparisonWeekSensor,
    HeatingModelComparisonMonthSensor,
)

_LOGGER = logging.getLogger(__name__)

async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Heating Analytics sensors based on a config entry."""
    coordinator: HeatingDataCoordinator = hass.data[DOMAIN][entry.entry_id]

    entities = [
        HeatingEnergyTodaySensor(coordinator, entry),
        HeatingExpectedEnergyTodaySensor(coordinator, entry),
        HeatingEfficiencySensor(coordinator, entry),
        HeatingPredictedSensor(coordinator, entry),
        HeatingDeviationSensor(coordinator, entry),
        HeatingForecastTodaySensor(coordinator, entry),
        HeatingForecastDetailsSensor(coordinator, entry),
        # HeatingTDDSensor removed as requested
        HeatingEffectiveWindSensor(coordinator, entry),
        HeatingCorrelationDataSensor(coordinator, entry),
        HeatingLastHourActualSensor(coordinator, entry),
        HeatingLastHourExpectedSensor(coordinator, entry),
        HeatingLastHourDeviationSensor(coordinator, entry),
        HeatingPotentialSavingsSensor(coordinator, entry),
        # New Model Comparison Sensors
        HeatingModelComparisonDaySensor(coordinator, entry),
        HeatingModelComparisonWeekSensor(coordinator, entry),
        HeatingModelComparisonMonthSensor(coordinator, entry),
        HeatingWeekAheadForecastSensor(coordinator, entry),
        HeatingAnalyticsComparisonSensor(coordinator, entry),
        HeatingThermalStateSensor(coordinator, entry),
    ]

    # Add individual device sensors
    for entity_id in coordinator.energy_sensors:
        entities.append(HeatingDeviceDailySensor(coordinator, entry, entity_id))

        # Add lifetime sensors if enabled
        if coordinator.enable_lifetime_tracking:
            entities.append(HeatingDeviceLifetimeSensor(coordinator, entry, entity_id))

    async_add_entities(entities)

# HeatingAnalyticsBaseSensor moved to .sensors.base

class HeatingEnergyTodaySensor(HeatingAnalyticsBaseSensor):
    """Sensor for Actual Energy Consumption Today."""

    _attr_name = SENSOR_ENERGY_TODAY
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_state_class = SensorStateClass.TOTAL_INCREASING
    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_icon = "mdi:counter"

    @property
    def native_value(self) -> float:
        """Return the state of the sensor."""
        return self.coordinator.data.get(ATTR_ENERGY_TODAY, 0.0)

    @property
    def extra_state_attributes(self):
        """Return attributes with per-unit breakdown."""
        daily_individual = self.coordinator.data.get("daily_individual", {})
        total_energy = self.coordinator.data.get(ATTR_ENERGY_TODAY, 0.0)

        unit_breakdown = {}
        unit_percentages = {}
        active_count = 0
        total_configured = len(self.coordinator.energy_sensors)

        for entity_id, kwh in daily_individual.items():
            kwh_val = round(kwh, 3)
            if kwh_val <= 0:
                continue

            # Try to get friendly name
            state = self.coordinator.hass.states.get(entity_id)
            name = state.name if state else entity_id

            unit_breakdown[name] = kwh_val
            active_count += 1

            if total_energy > 0:
                unit_percentages[name] = round((kwh_val / total_energy) * 100, 1)
            else:
                unit_percentages[name] = 0.0

        return {
            "unit_breakdown_kwh": unit_breakdown,
            "unit_contribution_pct": unit_percentages,
            "active_units_count": active_count,
            "total_units_configured": total_configured,
            "last_update": dt_util.now().isoformat(),
        }

    @property
    def unique_id(self) -> str:
        """Return a unique ID."""
        return f"{self.entry.entry_id}_energy_today"


class HeatingExpectedEnergyTodaySensor(HeatingAnalyticsBaseSensor):
    """Sensor for Energy Baseline Today.

    Model expectation based on actual weather conditions.
    Shows what the model expects given the weather that actually occurred.
    """

    _attr_name = SENSOR_ENERGY_BASELINE_TODAY
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_state_class = SensorStateClass.TOTAL
    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_icon = "mdi:chart-bell-curve"

    def __init__(self, coordinator: HeatingDataCoordinator, entry: ConfigEntry) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator, entry)
        self._peak_load_cache = {}

    @property
    def native_value(self) -> float:
        """Return the state of the sensor."""
        return self.coordinator.data.get(ATTR_EXPECTED_TODAY, 0.0)

    @property
    def unique_id(self) -> str:
        """Return a unique ID."""
        return f"{self.entry.entry_id}_expected_today"

    @property
    def extra_state_attributes(self):
        """Return attributes with Actionable Insights."""
        coordinator = self.coordinator
        stats_mgr = coordinator.statistics
        forecast_mgr = coordinator.forecast

        attrs = {}

        # === GROUP 1: LOAD MONITOR (Current Status) ===
        # max_observed_load: Max daily kWh in history
        max_load = stats_mgr.get_max_historical_daily_kwh()
        forecast_today = coordinator.data.get(ATTR_FORECAST_TODAY, 0.0)

        # Aux-Unaware Load (Total Thermodynamic Demand)
        # Use Gross Forecast if available (Calculated in Coordinator)
        # Fallback to Net Forecast if legacy/unavailable
        forecast_gross = coordinator.data.get("forecast_today_gross", forecast_today)

        # Thermal Stress Index: (Gross Forecast / Max Historical) * 100
        # This tells "How hard is the system working relative to its peak capacity?"
        # Using Gross Forecast prevents the "Mild Weather Fallacy" when Aux is active.
        # Note: If historical max_load is suppressed by Aux, index may exceed 100%,
        # correctly indicating "Heavy" or "Extreme" stress relative to electrical capacity.
        if max_load > 0:
            stress_index = (forecast_gross / max_load) * 100
        else:
            stress_index = 0.0

        # Day Classification
        if stress_index < STRESS_INDEX_LIGHT:
            classification = "Light"
        elif stress_index < STRESS_INDEX_MODERATE:
            classification = "Moderate"
        elif stress_index < STRESS_INDEX_HEAVY:
            classification = "Heavy"
        else:
            classification = "Extreme"

        # Peak Load Hour (Simulation with State Tracking & Caching)
        peak_hour = None
        max_hour_kwh = -1.0
        now = dt_util.now()

        # Cache Strategy: Invalidate every 15 minutes or if forecast object changes
        cache_key = f"{now.hour}_{now.minute // 15}"

        if self._peak_load_cache.get("key") == cache_key:
            peak_hour = self._peak_load_cache.get("peak_hour")
            max_hour_kwh = self._peak_load_cache.get("max_kwh")
        else:
            forecast_source = forecast_mgr._get_live_forecast_or_ref()
            if forecast_source:
                # Initialize Inertia (Stateful Simulation)
                # We must start from CURRENT inertia and evolve it through the day
                inertia_now = coordinator._calculate_inertia_temp() or 0.0
                history_needed = len(coordinator.inertia_weights) - 1

                # Fetch recent history to seed the simulation correctly (instead of flat line)
                local_inertia = coordinator._get_inertia_list(now)
                if not local_inertia:
                    local_inertia = [inertia_now] * history_needed

                # Working copy for simulation
                sim_inertia = list(local_inertia)

                weather_wind_unit = coordinator._get_weather_wind_unit()
                current_cloud = coordinator._get_cloud_coverage()

                for item in forecast_source:
                    dt_str = item.get("datetime")
                    if dt_str:
                        f_dt = dt_util.parse_datetime(dt_str)
                        if f_dt and dt_util.as_local(f_dt).date() == now.date():
                            # Pass mutable sim_inertia: it will be updated by _process_forecast_item
                            # This fixes the "Borderline Technical Error" (Inertia Reset)
                            pred, _, _, _, _, _, _, _ = forecast_mgr._process_forecast_item(
                                item, sim_inertia, weather_wind_unit, current_cloud, ignore_aux=False
                            )
                            if pred > max_hour_kwh:
                                max_hour_kwh = pred
                                peak_hour = f_dt.strftime("%H:00")

            # Update Cache
            self._peak_load_cache = {
                "key": cache_key,
                "peak_hour": peak_hour,
                "max_kwh": max_hour_kwh
            }

        attrs["thermal_stress_index"] = round(stress_index, 1)
        attrs["day_classification"] = classification
        attrs["peak_load_hour"] = peak_hour

        # === GROUP 2: TACTICAL PLANNING (Future) ===
        # Remaining Energy Demand
        actual_so_far = coordinator.data.get(ATTR_ENERGY_TODAY, 0.0)
        remaining_demand = max(0.0, forecast_today - actual_so_far)
        attrs["remaining_energy_demand"] = round(remaining_demand, 1)

        # Load Trend
        # Compare avg(forecast[h+1:h+4]) vs avg(forecast[h-2:h+1])
        # We now calculate this directly using the forecast data (forecast.py)
        # This replaces the legacy approximation via Temperature/Inertia trend.
        attrs["load_trend"] = forecast_mgr.calculate_load_trend()

        # === GROUP 3: DIAGNOSTICS (Why?) ===
        # Recommendation
        attrs[ATTR_RECOMMENDATION_STATE] = coordinator.data.get(ATTR_RECOMMENDATION_STATE, "none")

        # Solar Details
        # Potential (Screens Up)
        attrs[ATTR_SOLAR_POTENTIAL] = coordinator.data.get(ATTR_SOLAR_POTENTIAL, 0.0)
        # Actual Gain (Screens Configured) - This is ATTR_SOLAR_IMPACT (kW)
        attrs[ATTR_SOLAR_GAIN_NOW] = coordinator.data.get(ATTR_SOLAR_IMPACT, 0.0)
        # Heating Load Offset (Effective Impact) - Same as Gain in heating mode
        attrs[ATTR_HEATING_LOAD_OFFSET] = coordinator.data.get(ATTR_SOLAR_IMPACT, 0.0)

        # Primary Driver
        # Calculate specific impacts
        current_temp = coordinator.data.get("current_calc_temp")
        if current_temp is None:
            current_temp = coordinator._calculate_inertia_temp()
        if current_temp is None:
            current_temp = coordinator.balance_point

        eff_wind = coordinator.data.get("effective_wind", 0.0)
        solar_impact = coordinator.data.get(ATTR_SOLAR_IMPACT, 0.0)

        # 1. Wind Penalty (Effective Wind vs 0 Wind)
        res_actual = stats_mgr.calculate_total_power(
            current_temp, eff_wind, solar_impact, is_aux_active=coordinator.auxiliary_heating_active
        )
        res_no_wind = stats_mgr.calculate_total_power(
            current_temp, 0.0, solar_impact, is_aux_active=coordinator.auxiliary_heating_active
        )
        wind_penalty_rate = max(0.0, res_actual["total_kwh"] - res_no_wind["total_kwh"])

        # 2. Temp Load (Thermodynamic Base Load)
        # res_no_wind represents the thermal load at 0 wind.
        temp_load_rate = res_no_wind["total_kwh"]

        # 3. Solar Deficit (Max Potential Solar - Actual Solar)
        # "Why is heating high? Because it's dark/cloudy."
        solar_deficit_rate = 0.0
        solar_gain_rate = res_actual["breakdown"]["solar_reduction_kwh"]

        if coordinator.solar_enabled:
            # Calculate Theoretical Max Solar (Cloud = 0)
            elev, azim = coordinator.solar.get_approx_sun_pos(now)
            max_solar_factor = coordinator.solar.calculate_solar_factor(elev, azim, cloud_coverage=0.0)

            res_max_solar = stats_mgr.calculate_total_power(
                current_temp, eff_wind, solar_impact,
                is_aux_active=coordinator.auxiliary_heating_active,
                override_solar_factor=max_solar_factor
            )
            max_potential_gain = res_max_solar["breakdown"]["solar_reduction_kwh"]
            solar_deficit_rate = max(0.0, max_potential_gain - solar_gain_rate)

        # Determine Primary Driver
        # Instruction: max(wind_penalty, solar_deficit, temp_deviation)
        # We compare the relative impact (kW) of each stressor.
        drivers = {
            "Temp": temp_load_rate,
            "Wind": wind_penalty_rate,
            "Solar_Deficit": solar_deficit_rate
        }
        primary_driver = max(drivers, key=drivers.get) if any(drivers.values()) else "None"

        attrs["primary_driver"] = primary_driver

        # Wind Chill Penalty kWh (Daily Impact)
        # Uses the precise calculation from the coordinator (Past Actuals + Future Forecast)
        # comparing Normal vs No-Wind scenarios.
        attrs["wind_chill_penalty_kwh"] = coordinator.data.get("daily_wind_chill_penalty", 0.0)
        attrs["solar_gain_kwh"] = round(coordinator.data.get("accumulated_solar_impact_kwh", 0.0), 1)

        # === GROUP 4: COMPARATIVE CONTEXT ===
        # Typical Day at this Temp
        # Use daily average temp for lookup
        avg_temp_today = coordinator.data.get(ATTR_TEMP_ACTUAL_TODAY)
        if avg_temp_today is None:
             avg_temp_today = current_temp

        typical_kwh, samples, confidence = stats_mgr.get_typical_day_consumption(avg_temp_today)

        attrs["typical_day_at_this_temp"] = typical_kwh
        attrs["confidence_level"] = confidence

        if typical_kwh:
            impact = forecast_today - typical_kwh
            attrs["weather_impact_vs_typical"] = f"{impact:+.1f}"
        else:
            attrs["weather_impact_vs_typical"] = None

        return attrs

class HeatingEfficiencySensor(HeatingAnalyticsBaseSensor):
    """Sensor for heating efficiency (kWh/TDD)."""

    _attr_name = SENSOR_EFFICIENCY
    _attr_native_unit_of_measurement = "kWh/TDD"
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:home-thermometer"

    @property
    def native_value(self) -> float:
        """Return the state of the sensor.

        Delegates to StatisticsManager for the Seamless Rolling Window calculation.
        """
        return self.coordinator.statistics.calculate_realtime_efficiency()

    @property
    def unique_id(self) -> str:
        """Return a unique ID."""
        return f"{self.entry.entry_id}_efficiency"

    @property
    def extra_state_attributes(self):
        """Return attributes for TDD statistics."""
        return {
            # Map tdd_today to the full-day stable calculation (Actual + Forecast)
            # Users expect "Today's TDD" to represent the daily load, not just the accumulated value.
            "tdd_today": self.coordinator.data.get(ATTR_TDD_DAILY_STABLE, 0.0),
            "tdd_accumulated": self.coordinator.data.get(ATTR_TDD, 0.0), # Original accumulation
            "tdd_so_far": self.coordinator.data.get(ATTR_TDD_SO_FAR, 0.0),
            "tdd_yesterday": self.coordinator.data.get(ATTR_TDD_YESTERDAY),
            "tdd_last_7d_avg": self.coordinator.data.get(ATTR_TDD_LAST_7D),
            "tdd_last_30d_avg": self.coordinator.data.get(ATTR_TDD_LAST_30D),
            "tdd_forecast_today": self.coordinator.data.get(ATTR_TDD_DAILY_STABLE),
            "efficiency_yesterday": self.coordinator.data.get(ATTR_EFFICIENCY_YESTERDAY),
            "efficiency_last_7d_avg": self.coordinator.data.get(ATTR_EFFICIENCY_LAST_7D),
            "efficiency_last_30d_avg": self.coordinator.data.get(ATTR_EFFICIENCY_LAST_30D),
            "efficiency_forecast_today": self.coordinator.data.get(ATTR_EFFICIENCY_FORECAST_TODAY),
        }

class HeatingPredictedSensor(HeatingAnalyticsBaseSensor):
    """Sensor for Weather Plan Today.

    Shows the full-day energy plan based on weather forecast.
    Past hours use forecast from when hour occurred (frozen plan).
    Current and future hours use live forecast.
    """

    _attr_name = SENSOR_WEATHER_PLAN_TODAY
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:chart-line"

    @property
    def native_value(self) -> float:
        """Return the state of the sensor."""
        return self.coordinator.data.get(ATTR_PREDICTED, 0.0)

    @property
    def extra_state_attributes(self):
        """Return attributes."""
        snapshot = self.coordinator.forecast._midnight_forecast_snapshot
        attrs = {
            "cached_total_24h": self.coordinator.data.get(ATTR_MIDNIGHT_FORECAST, 0.0),
            "snapshot_time": snapshot.get("timestamp") if snapshot else None,
        }
        forecast_details = self.coordinator.data.get(ATTR_FORECAST_DETAILS, {})
        attrs[ATTR_FORECAST_BLEND_CONFIG] = forecast_details.get("blend_config")
        attrs[ATTR_FORECAST_ACCURACY_BY_SOURCE] = forecast_details.get("accuracy_by_source")
        return attrs

    @property
    def unique_id(self) -> str:
        """Return a unique ID."""
        return f"{self.entry.entry_id}_predicted"

class HeatingForecastTodaySensor(HeatingAnalyticsBaseSensor):
    """Sensor for Energy Estimate Today.

    Best estimate of actual consumption today.
    Uses actual consumption so far + forecast for remaining hours.
    """

    _attr_name = SENSOR_ENERGY_ESTIMATE_TODAY
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:crystal-ball"

    @property
    def native_value(self) -> float:
        """Return the state of the sensor."""
        return self.coordinator.data.get(ATTR_FORECAST_TODAY, 0.0)

    @property
    def extra_state_attributes(self):
        """Return attributes with confidence assessment."""
        uncertainty = self.coordinator.data.get(ATTR_FORECAST_UNCERTAINTY, {})
        margin = self.coordinator.data.get("confidence_interval_margin", 0.0)
        forecast_val = self.coordinator.data.get(ATTR_FORECAST_TODAY, 0.0)
        samples = uncertainty.get("samples", 0)
        p50_error = uncertainty.get("p50_abs_error", 0.0)

        # Determine confidence level based on historical accuracy and sample size
        if samples < CONFIDENCE_MIN_SAMPLES:
            confidence_level = "low"
            confidence_reason = f"Only {samples} days of forecast history"
        elif p50_error < CONFIDENCE_HIGH_ERROR_MAX and samples >= CONFIDENCE_HIGH_SAMPLES:
            confidence_level = "high"
            confidence_reason = f"Excellent track record ({samples} days, median error {p50_error:.1f} kWh)"
        elif p50_error < CONFIDENCE_MEDIUM_ERROR_MAX:
            confidence_level = "medium"
            confidence_reason = f"Good track record ({samples} days, median error {p50_error:.1f} kWh)"
        else:
            confidence_level = "low"
            confidence_reason = f"Variable accuracy ({samples} days, median error {p50_error:.1f} kWh)"

        # Get weather context from explanation module
        temp_forecast = self.coordinator.data.get(ATTR_TEMP_FORECAST_TODAY)
        wind_forecast = self.coordinator.data.get(ATTR_AVG_WIND_FORECAST)

        formatter = ExplanationFormatter()
        weather_context = formatter.format_forecast_weather_context(
            temp=temp_forecast,
            wind=wind_forecast,
            wind_high_threshold=self.coordinator.wind_threshold,
            wind_extreme_threshold=self.coordinator.extreme_wind_threshold
        )

        # Create user-friendly explanation
        if forecast_val > 0:
            margin_pct = (margin / forecast_val) * 100 if forecast_val > 0 else 0
            if confidence_level == "high":
                summary = f"Today's forecast: {forecast_val:.1f} kWh (High confidence - typically within ±{p50_error:.1f} kWh). {weather_context}"
            elif confidence_level == "medium":
                summary = f"Today's forecast: {forecast_val:.1f} kWh (Medium confidence - typically within ±{p50_error:.1f} kWh). {weather_context}"
            else:
                summary = f"Today's forecast: {forecast_val:.1f} kWh (Low confidence - {confidence_reason}). {weather_context}"
        else:
            summary = "No forecast available"

        attrs = {
            "confidence_level": confidence_level,
            "confidence_reason": confidence_reason,
            "forecast_summary": summary,
            "confidence_interval_margin": margin,
            "confidence_interval_lower": self.coordinator.data.get("confidence_interval_lower"),
            "confidence_interval_upper": self.coordinator.data.get("confidence_interval_upper"),
            ATTR_MIDNIGHT_FORECAST: self.coordinator.data.get(ATTR_MIDNIGHT_FORECAST),
            ATTR_FORECAST_UNCERTAINTY: uncertainty,
        }

        # Process Midnight Unit Estimates
        raw_estimates = self.coordinator.data.get(ATTR_MIDNIGHT_UNIT_ESTIMATES)
        raw_modes = self.coordinator.data.get(ATTR_MIDNIGHT_UNIT_MODES)

        if raw_estimates and raw_modes:
            unit_estimates = {}
            total_active = 0.0

            for entity_id, kwh in raw_estimates.items():
                mode = raw_modes.get(entity_id)
                if mode == MODE_OFF:
                    continue

                state = self.coordinator.hass.states.get(entity_id)
                name = state.name if state else entity_id

                rounded_kwh = round(kwh, 2)
                unit_estimates[name] = rounded_kwh
                total_active += rounded_kwh

            if unit_estimates:
                unit_estimates["Total"] = round(total_active, 2)
                attrs[ATTR_MIDNIGHT_UNIT_ESTIMATES] = unit_estimates

        return attrs

    @property
    def unique_id(self) -> str:
        """Return a unique ID."""
        return f"{self.entry.entry_id}_forecast_today"


class HeatingForecastDetailsSensor(HeatingAnalyticsBaseSensor):
    """Sensor for Blended Forecast Details and Accuracy."""

    _attr_name = SENSOR_FORECAST_DETAILS
    _attr_icon = "mdi:text-box-check-outline"
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    @property
    def native_value(self) -> str:
        """Return a summary of which forecast source is performing better."""
        details = self.coordinator.data.get(ATTR_FORECAST_DETAILS, {})
        accuracy = details.get("accuracy_by_source", {})
        primary = accuracy.get("primary", {})
        secondary = accuracy.get("secondary", {})

        p_samples = primary.get("samples", 0)
        s_samples = secondary.get("samples", 0)

        if not secondary or not self.coordinator.data.get(ATTR_FORECAST_DETAILS, {}).get("blend_config", {}).get("secondary_entity_id"):
            return "Primary source only"

        if p_samples < CONFIDENCE_MIN_SAMPLES and s_samples < CONFIDENCE_MIN_SAMPLES:
            return "Gathering accuracy data"

        if s_samples < CONFIDENCE_MIN_SAMPLES:
            return f"Primary source is active ({p_samples} days logged)"

        if p_samples < CONFIDENCE_MIN_SAMPLES:
            return f"Secondary source is active ({s_samples} days logged)"

        p_error = primary.get("hourly", {}).get("p50_abs_error", 999)
        s_error = secondary.get("hourly", {}).get("p50_abs_error", 999)

        if p_error < s_error * FORECAST_COMPARISON_FACTOR:
            return f"Primary is performing better ({p_error:.1f} vs {s_error:.1f} kWh error)"
        elif s_error < p_error * FORECAST_COMPARISON_FACTOR:
            return f"Secondary is performing better ({s_error:.1f} vs {p_error:.1f} kWh error)"
        else:
            return f"Both sources have similar accuracy ({p_error:.1f} vs {s_error:.1f} kWh error)"

    @property
    def extra_state_attributes(self):
        """Return attributes with detailed forecast configuration and accuracy."""
        return self.coordinator.data.get(ATTR_FORECAST_DETAILS, {})

    @property
    def unique_id(self) -> str:
        """Return a unique ID."""
        return f"{self.entry.entry_id}_forecast_details"


class HeatingDeviationSensor(HeatingAnalyticsBaseSensor):
    """Sensor for deviation."""

    _attr_name = "Deviation Today"
    _attr_native_unit_of_measurement = PERCENTAGE
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:percent"

    @property
    def native_value(self) -> float:
        """Return the state of the sensor."""
        return self.coordinator.data.get(ATTR_DEVIATION, 0.0)

    @property
    def extra_state_attributes(self):
        """Return attributes with actionable insights."""
        # Raw coordinator data
        forecast = self.coordinator.data.get(ATTR_FORECAST_TODAY, 0.0)
        predicted = self.coordinator.data.get(ATTR_PREDICTED, 0.0)
        actual = self.coordinator.data.get(ATTR_ENERGY_TODAY, 0.0)
        expected = self.coordinator.data.get(ATTR_EXPECTED_TODAY, 0.0)

        # Calculate current deviation (actual vs expected so far)
        deviation_current_pct = 0.0
        if expected > ENERGY_GUARD_THRESHOLD:
            deviation_current_pct = ((actual - expected) / expected) * 100

        breakdown = self.coordinator.data.get(ATTR_DEVIATION_BREAKDOWN, [])
        plan_revision = self.coordinator.data.get("plan_revision_impact", {})
        weather_adjusted = self.coordinator.data.get("weather_adjusted_deviation", {})

        # Identify top contributor for explanation
        # breakdown is already sorted by abs(deviation), so the first one is the biggest contributor
        top_contributor = breakdown[0] if breakdown else None

        guest_impact = self.coordinator.data.get("accumulated_guest_impact_kwh", 0.0)

        # Format Summary using Explanation Module
        formatter = ExplanationFormatter()
        actionable_summary = formatter.format_behavioral_deviation(
            deviation_kwh=(actual - expected),
            deviation_pct=deviation_current_pct,
            top_contributor=top_contributor,
            weather_impact=plan_revision,
            guest_impact_kwh=guest_impact
        )

        # === REFACTORED ATTRIBUTES ===
        attributes = {
            # Summary
            "deviation_summary": actionable_summary,

            # Current state (as of now)
            "current_usage_kwh": round(actual, 1),
            "model_expected_sofar_kwh": round(expected, 1),
            "thermodynamic_gross_today_kwh": self.coordinator.data.get("thermodynamic_gross_today_kwh", 0.0),
            "deviation_current_kwh": round(actual - expected, 1),
            "deviation_current_pct": round(deviation_current_pct, 1),

            # End-of-day projections
            "end_of_day_forecast_kwh": round(forecast, 1),  # Coordinator's EOD forecast
            "model_prediction_kwh": round(predicted, 1),     # Model's EOD prediction
            "deviation_projected_kwh": round(forecast - predicted, 1),

            # Thermodynamic Projection (Model on Actuals So Far + Live Forecast)
            "thermodynamic_projection_kwh": self.coordinator.data.get("thermodynamic_projection_kwh", 0.0),
            "thermodynamic_deviation_kwh": self.coordinator.data.get("thermodynamic_deviation_kwh", 0.0),
            "thermodynamic_deviation_pct": self.coordinator.data.get("thermodynamic_deviation_pct", 0.0),

            # Global Factors
            "accumulated_solar_impact_kwh": self.coordinator.data.get(
                "accumulated_solar_impact_kwh", 0.0
            ),
            "accumulated_guest_impact_kwh": self.coordinator.data.get(
                "accumulated_guest_impact_kwh", 0.0
            ),
            "accumulated_aux_impact_kwh": self.coordinator.data.get(
                "accumulated_aux_impact_kwh", 0.0
            ),
            "aux_active": self.coordinator.auxiliary_heating_active,
            "aux_hours": self.coordinator.data.get("savings_aux_hours_today", 0.0),
        }

        # === CONTRIBUTORS (structured by type and deviation) ===
        # Separate tracked units from guest units
        tracked_units = []
        guest_units = []

        for item in breakdown:
            entity_id = item.get("entity_id")
            unit_mode = self.coordinator.get_unit_mode(entity_id)

            # Skip OFF units completely from the contributor list
            if unit_mode == MODE_OFF:
                continue

            if unit_mode in (MODE_GUEST_HEATING, MODE_GUEST_COOLING):
                guest_units.append(item)
            else:
                tracked_units.append(item)

        # Separate tracked units by positive/negative deviation
        above_expected = [x for x in tracked_units if x.get("deviation", 0.0) > 0.0]
        below_expected = [x for x in tracked_units if x.get("deviation", 0.0) < 0.0]

        def format_contributor(item):
            """Helper to format a contributor item."""
            contributor = {
                "name": item["name"],
                "deviation_kwh": round(item["deviation"], 2),
                "confidence": item["confidence"],
            }

            unusual = item.get("unusual", False)
            if unusual:
                contributor["unusual"] = True
                dev_score = item.get("deviation_score")
                dev_thresh = item.get("deviation_threshold")
                if dev_score is not None:
                    contributor["sensitivity_score"] = round(dev_score, 3)
                    contributor["sensitivity_threshold"] = round(dev_thresh, 3)
                    contributor["unusual_reason"] = f"Score {dev_score:.3f} > {dev_thresh:.3f}"

            return contributor

        contributors_dict = {}

        if above_expected:
            contributors_dict["above_expected"] = [
                format_contributor(item) for item in above_expected[:3]
            ]

        # Guest units in separate section (not part of tracked deviations)
        if guest_units:
            contributors_dict["guest_units"] = [
                format_contributor(item) for item in guest_units[:3]
            ]

        if below_expected:
            contributors_dict["below_expected"] = [
                format_contributor(item) for item in below_expected[:3]
            ]

        attributes["contributors"] = contributors_dict



        # === DIAGNOSTICS (useful for wood heating monitoring) ===
        diagnostics = {}

        # Current hour stability (critical for wood heating detection)
        sample_count = self.coordinator._hourly_sample_count

        if sample_count > 0:
            aux_count = self.coordinator._hourly_aux_count
            purity = aux_count / sample_count

            # Classify stability
            if purity < 0.05:
                status = "Stable (Normal)"
            elif purity > 0.95:
                status = "Stable (Auxiliary)"
            else:
                status = "Unstable (Mixed Mode)"

            diagnostics["current_hour_stability"] = status
            diagnostics["current_hour_purity_pct"] = round((1.0 - purity) * 100, 1)  # % Normal mode
        else:
            diagnostics["current_hour_stability"] = "Waiting for data"
            diagnostics["current_hour_purity_pct"] = None

        attributes["diagnostics"] = diagnostics

        return attributes

    @property
    def unique_id(self) -> str:
        """Return a unique ID."""
        return f"{self.entry.entry_id}_deviation"

class HeatingEffectiveWindSensor(HeatingAnalyticsBaseSensor):
    """Sensor for Effective Wind."""

    _attr_name = SENSOR_EFFECTIVE_WIND
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:windsock"

    @property
    def native_unit_of_measurement(self) -> str:
        """Return the unit of measurement."""
        return self.coordinator.wind_unit

    @property
    def native_value(self) -> float:
        """Return the state."""
        val = self.coordinator.data.get("effective_wind", 0.0)
        return round(convert_from_ms(val, self.coordinator.wind_unit), 1)

    @property
    def extra_state_attributes(self):
        """Return attributes."""
        # Calculate running average of effective wind for current hour using aggregates
        sample_count = self.coordinator._hourly_sample_count
        if sample_count > 0:
            avg_eff_wind = self.coordinator._hourly_wind_sum / sample_count
            projected_bucket = self.coordinator._get_wind_bucket(avg_eff_wind)
            data_quality = "complete" if sample_count > 30 else "partial"
        else:
            # If no samples (start of hour or restart without data), use current effective wind
            avg_eff_wind = self.coordinator.data.get("effective_wind", 0.0)
            projected_bucket = self.coordinator._get_wind_bucket(avg_eff_wind)
            data_quality = "insufficient"

        now = dt_util.now()
        weather_wind_unit = self.coordinator._get_weather_wind_unit()
        wind_unit = self.coordinator.wind_unit

        def _forecast_effective_wind(source):
            item = self.coordinator.forecast.get_forecast_for_hour(now, source=source)
            if not item:
                return None
            try:
                w_speed_ms = convert_speed_to_ms(float(item.get("wind_speed", 0.0)), weather_wind_unit)
                w_gust = item.get("wind_gust_speed")
                w_gust_ms = convert_speed_to_ms(float(w_gust), weather_wind_unit) if w_gust is not None else None
                eff = self.coordinator._calculate_effective_wind(w_speed_ms, w_gust_ms)
                return round(convert_from_ms(eff, wind_unit), 1)
            except (ValueError, TypeError):
                return None

        return {
            "running_average_this_hour": round(convert_from_ms(avg_eff_wind, wind_unit), 1),
            "running_average_this_hour_ms": round(avg_eff_wind, 1),
            "projected_wind_bucket": projected_bucket,
            "sample_count": sample_count,
            "data_quality": data_quality,
            "wind_threshold": f"{round(convert_from_ms(self.coordinator.wind_threshold, wind_unit), 1)} {wind_unit}",
            "extreme_wind_threshold": f"{round(convert_from_ms(self.coordinator.extreme_wind_threshold, wind_unit), 1)} {wind_unit}",
            "wind_gust_factor": self.coordinator.wind_gust_factor,
            "midnight_forecast_effective_wind_primary": _forecast_effective_wind('primary_reference'),
            "midnight_forecast_effective_wind_secondary": _forecast_effective_wind('secondary_reference'),
        }

    @property
    def unique_id(self) -> str:
        return f"{self.entry.entry_id}_effective_wind"

class HeatingCorrelationDataSensor(HeatingAnalyticsBaseSensor):
    """Sensor exposing correlation data as attributes for graphing."""

    _attr_name = SENSOR_CORRELATION_DATA
    _attr_icon = "mdi:chart-scatter-plot"
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    @property
    def native_value(self) -> str:
        return "Data"

    @property
    def extra_state_attributes(self):
        raw_data = self.coordinator.data.get(ATTR_CORRELATION_DATA, {})

        # Initialize lists
        categories = {
            "normal": {"x": [], "y": []},
            "high_wind": {"x": [], "y": []},
            "extreme_wind": {"x": [], "y": []},
        }

        # Sort keys
        try:
            sorted_temps = sorted(raw_data.keys(), key=lambda k: int(k))
        except ValueError:
            sorted_temps = sorted(raw_data.keys())

        for temp_str in sorted_temps:
            try:
                temp_val = int(temp_str)
            except ValueError:
                continue

            entry = raw_data[temp_str]
            for cat, lists in categories.items():
                if cat in entry:
                    val = entry[cat]
                    lists["x"].append(temp_val)
                    lists["y"].append(round(val * 24, 2))  # Convert to kWh/day

        attributes = {
            "normal_x": json.dumps(categories["normal"]["x"]),
            "normal_y": json.dumps(categories["normal"]["y"]),
            "high_wind_x": json.dumps(categories["high_wind"]["x"]),
            "high_wind_y": json.dumps(categories["high_wind"]["y"]),
            "extreme_wind_x": json.dumps(categories["extreme_wind"]["x"]),
            "extreme_wind_y": json.dumps(categories["extreme_wind"]["y"]),
        }

        # Add Aux Coefficients if available
        if self.coordinator._aux_coefficients:
            aux_x = []
            aux_y = []
            sorted_temps = sorted([int(k) for k in self.coordinator._aux_coefficients.keys()])
            for temp in sorted_temps:
                val = self.coordinator._aux_coefficients.get(str(temp))
                if val is not None:
                    aux_x.append(temp)
                    aux_y.append(val) # kW impact

            attributes["aux_impact_x"] = json.dumps(aux_x)
            attributes["aux_impact_y"] = json.dumps(aux_y)

        return {**raw_data, **attributes}

    @property
    def unique_id(self) -> str:
        return f"{self.entry.entry_id}_correlation_data"

class HeatingLastHourActualSensor(HeatingAnalyticsBaseSensor):
    """Sensor for Last Hour Actual Consumption."""

    _attr_name = SENSOR_LAST_HOUR_ACTUAL
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:flash"
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    @property
    def native_value(self) -> float:
        return self.coordinator.data.get(ATTR_LAST_HOUR_ACTUAL, 0.0)

    @property
    def extra_state_attributes(self):
        """Return attributes from the last hourly log."""
        if not self.coordinator._hourly_log:
            return {}

        last_entry = self.coordinator._hourly_log[-1]
        attributes = {
            "timestamp": last_entry.get("timestamp"),
            "avg_temperature": last_entry.get("temp"),
            "avg_effective_wind": last_entry.get("effective_wind"),
            "learning_status": last_entry.get("learning_status"),
        }

        # Calculate Top Consumers
        unit_breakdown = last_entry.get("unit_breakdown", {})
        total_kwh = last_entry.get("actual_kwh", 0.0)
        unit_expected = last_entry.get("unit_expected_breakdown", {})
        top_consumers = []

        if unit_breakdown:
            # Filter > 0 and Sort by consumption descending
            filtered_units = [
                (eid, kwh) for eid, kwh in unit_breakdown.items()
                if kwh > 0
            ]

            sorted_units = sorted(
                filtered_units,
                key=lambda item: item[1],
                reverse=True
            )

            # Take top 5
            for entity_id, kwh in sorted_units[:5]:
                state = self.coordinator.hass.states.get(entity_id)
                name = state.name if state else entity_id

                pct = (kwh / total_kwh * 100) if total_kwh > 0 else 0.0
                expected_kwh = unit_expected.get(entity_id, 0.0)

                top_consumers.append({
                    "name": name,
                    "kwh": kwh,
                    "expected_kwh": expected_kwh,
                    "pct_of_total": round(pct, 1)
                })

        if top_consumers:
            attributes["last_hour_top_consumers"] = top_consumers

        # Build last_hour_summary
        formatter = ExplanationFormatter()
        attributes["last_hour_summary"] = formatter.format_last_hour_summary(
            kwh=total_kwh,
            top_consumer_name=top_consumers[0]["name"] if top_consumers else None,
            top_consumer_pct=top_consumers[0]["pct_of_total"] if top_consumers else None
        )

        return attributes

    @property
    def unique_id(self) -> str:
        return f"{self.entry.entry_id}_last_hour_actual"

class HeatingLastHourExpectedSensor(HeatingAnalyticsBaseSensor):
    """Sensor for Last Hour Expected Consumption."""

    _attr_name = SENSOR_LAST_HOUR_EXPECTED
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:chart-bell-curve"
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    @property
    def native_value(self) -> float:
        return self.coordinator.data.get(ATTR_LAST_HOUR_EXPECTED, 0.0)

    @property
    def extra_state_attributes(self):
        """Return attributes exposing the model parameters used."""
        if not self.coordinator._hourly_log:
            return {}

        last_entry = self.coordinator._hourly_log[-1]
        expected = last_entry.get("expected_kwh", 0.0)
        solar_impact = last_entry.get("solar_impact_kwh", 0.0)

        # Base Model = Expected (Net) + Solar Deduction
        # Since solar adjustment is subtracted from base: Corrected = Base - Impact
        base_model = expected + solar_impact

        return {
            "temp_used": last_entry.get("inertia_temp"),
            "wind_bucket_used": last_entry.get("wind_bucket"),
            "solar_deduction_kwh": solar_impact,
            "solar_coefficient": last_entry.get("solar_factor"),
            "base_model_kwh": round(base_model, 3),
        }

    @property
    def unique_id(self) -> str:
        return f"{self.entry.entry_id}_last_hour_expected"

class HeatingLastHourDeviationSensor(HeatingAnalyticsBaseSensor):
    """Sensor for Last Hour Deviation."""

    _attr_name = SENSOR_LAST_HOUR_DEVIATION
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:delta"
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    @property
    def native_value(self) -> float:
        return self.coordinator.data.get(ATTR_LAST_HOUR_DEVIATION, 0.0)

    @property
    def extra_state_attributes(self):
        # Get last hour data from hourly log (survives restart)
        last_hour_wind_bucket = None
        last_hour_solar_impact = 0.0
        last_hour_aux_impact = 0.0
        last_hour_guest_impact = 0.0
        last_hour_timestamp = None

        if self.coordinator._hourly_log:
            last_entry = self.coordinator._hourly_log[-1]
            last_hour_wind_bucket = last_entry.get("wind_bucket")
            last_hour_solar_impact = last_entry.get("solar_impact_kwh", 0.0)
            last_hour_aux_impact = last_entry.get("aux_impact_kwh", 0.0)
            last_hour_guest_impact = last_entry.get("guest_impact_kwh", 0.0)
            last_hour_timestamp = last_entry.get("timestamp")

        attrs = {
            "percentage": self.coordinator.data.get(ATTR_LAST_HOUR_DEVIATION_PCT, 0.0),
            "last_hour_thermodynamic_gross_kwh": self.coordinator._hourly_log[-1].get("thermodynamic_gross_kwh") if self.coordinator._hourly_log else None,
            "last_hour_wind_bucket": last_hour_wind_bucket,
            "last_hour_solar_impact_kwh": round(last_hour_solar_impact, 3),
            "last_hour_aux_impact_kwh": round(last_hour_aux_impact, 3),
            "last_hour_guest_impact_kwh": round(last_hour_guest_impact, 3),
            "last_hour_timestamp": last_hour_timestamp,
            "last_hour_expected_kwh": self.coordinator.data.get(ATTR_LAST_HOUR_EXPECTED, 0.0),
            "last_hour_actual_kwh": self.coordinator.data.get(ATTR_LAST_HOUR_ACTUAL, 0.0),
        }

        if self.coordinator._hourly_log:
            last_entry = self.coordinator._hourly_log[-1]

            # Get raw values
            model_before = last_entry.get("model_base_before")
            model_after = last_entry.get("model_base_after")
            inertia_temp = last_entry.get("inertia_temp")

            attrs["model_updated_temp_category"] = last_entry.get("model_temp_key")

            if model_before is not None:
                attrs["model_value_before"] = f"{model_before:.5f}"
            else:
                attrs["model_value_before"] = None

            if model_after is not None:
                attrs["model_value_after"] = f"{model_after:.5f}"
            else:
                attrs["model_value_after"] = None

            attrs["model_updated"] = last_entry.get("model_updated", False)

            if inertia_temp is not None:
                attrs["inertia_temperature"] = f"{inertia_temp:.2f}"
            else:
                attrs["inertia_temperature"] = None

            # Calculate and format delta
            if model_after is not None and model_before is not None:
                delta = model_after - model_before
                attrs["model_delta"] = f"{delta:+.5f}"
            else:
                attrs["model_delta"] = None

        return attrs

    @property
    def unique_id(self) -> str:
        return f"{self.entry.entry_id}_last_hour_deviation"

class HeatingPotentialSavingsSensor(HeatingAnalyticsBaseSensor):
    """Sensor for Potential Savings (Auxiliary Heating)."""

    _attr_name = "AUX Savings Today"
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_state_class = SensorStateClass.TOTAL
    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_icon = "mdi:piggy-bank"

    @property
    def native_value(self) -> float:
        """Return Actual Savings (The Reward)."""
        return self.coordinator.data.get("savings_actual_kwh", 0.0)

    @property
    def extra_state_attributes(self):
        # Retrieve computed data
        savings_potential = self.coordinator.data.get(ATTR_POTENTIAL_SAVINGS, 0.0) # This is effectively Theoretical Max

        # Theoretical Max = Full Day Projection (Whole Day)
        theoretical_max = savings_potential

        # Current Savings Rate (kW)
        current_rate_kw = self.coordinator.data.get("current_savings_rate")
        if current_rate_kw is None:
             rate_display = "Unknown"
        else:
             rate_display = f"{current_rate_kw} kW"

        # Status
        active = self.coordinator.auxiliary_heating_active
        status = "Active" if active else "Passive"

        # === Detailed Breakdown Implementation ===
        # Retrieve live instantaneous breakdown (from current prediction loop)
        live_breakdown = self.coordinator.data.get("current_unit_breakdown", {})
        potential_breakdown = self.coordinator.data.get("potential_savings_breakdown", {})

        # Global Totals (Daily)
        model_total_aux_kwh = self.coordinator.data.get("accumulated_aux_impact_kwh", 0.0)

        # Retrieve Daily + Current Hour Accumulations
        daily_accum = self.coordinator._daily_aux_breakdown
        current_accum = self.coordinator._accumulated_aux_breakdown

        unit_breakdown_list = []
        global_allocated_sum = 0.0
        global_unassigned_sum = 0.0

        # Calculate time fraction for mean power calculation
        now = dt_util.now()
        # Use float minutes for higher precision (User request: fix inaccurate time division)
        minutes_passed = now.minute + (now.second / 60.0)
        # Avoid division by zero or extremely small numbers (use 6 seconds as minimum floor to stabilize start of hour)
        minutes_passed = max(0.1, minutes_passed)

        # Calculate sums for the breakdown list
        # We iterate over live_breakdown (all energy sensors) to ensure coverage
        for entity_id, stats in live_breakdown.items():
            # Filter: Only show if in Aux Affected Entities
            if self.coordinator.aux_affected_entities and entity_id not in self.coordinator.aux_affected_entities:
                continue

            state = self.coordinator.hass.states.get(entity_id)
            name = state.name if state else entity_id

            # Instantaneous values (Use Potential Breakdown if available to show "Current Savings Rate" breakdown)
            # This ensures the list aligns with 'current_savings_rate_kw' even when Passive.
            pot_stats = potential_breakdown.get(entity_id, {})

            raw_demand_kw = pot_stats.get("raw_aux_kwh", 0.0) # Rate in kW (Potential)
            applied_kw = pot_stats.get("aux_reduction_kwh", 0.0) # Rate in kW (Potential)
            overflow_kw = pot_stats.get("overflow_kwh", 0.0) # Rate in kW (Potential)
            clamped = pot_stats.get("clamped", False)

            # Combined Daily Values (History + Current Hour)
            hist_data = daily_accum.get(entity_id, {})
            curr_data = current_accum.get(entity_id, {})

            allocation_kwh = hist_data.get("allocated", 0.0) + curr_data.get("allocated", 0.0)
            overflow_kwh = hist_data.get("overflow", 0.0) + curr_data.get("overflow", 0.0)
            raw_demand_accum_kwh = allocation_kwh + overflow_kwh

            # Calculate Mean Power (kW) from accumulated Energy (kWh) - For Current Hour context
            # Note: mean_allocated_kw logic here is still based on current hour fraction for consistency with "live" feel
            curr_allocation = curr_data.get("allocated", 0.0)
            curr_raw_demand = curr_data.get("allocated", 0.0) + curr_data.get("overflow", 0.0)

            mean_allocated_kw = (curr_allocation / minutes_passed) * 60
            mean_demand_kw = (curr_raw_demand / minutes_passed) * 60

            unit_breakdown_list.append({
                "name": name,
                "mean_allocated_kw": round(mean_allocated_kw, 3), # Mean Power (Current Hour)
                "mean_demand_kw": round(mean_demand_kw, 3), # Mean Power (Current Hour)
                "clamped": clamped,
                "current_rate_w": int(applied_kw * 1000), # Shows Potential Rate
                "overflow_rate_w": int(overflow_kw * 1000) # Shows Potential Overflow Rate
            })

            global_allocated_sum += allocation_kwh
            global_unassigned_sum += overflow_kwh

        # Add Orphaned Global Savings (Not attached to any unit)
        # Sum of Daily (Past Hours) + Current Hour (Live)
        orphaned_daily = getattr(self.coordinator, "_daily_orphaned_aux", 0.0)
        orphaned_live = getattr(self.coordinator, "_accumulated_orphaned_aux", 0.0)

        global_unassigned_sum += (orphaned_daily + orphaned_live)

        # Determine Status
        status_flag = "balanced"
        if global_unassigned_sum > 0.01:
            status_flag = "partial_overflow"
        if global_unassigned_sum > 0.5: # Arbitrary threshold for critical
            status_flag = "critical_leak"

        # Detailed Aux Learning (Moved from Deviation sensors)
        learning_diagnostics = {}
        if self.coordinator._hourly_log:
            last_entry = self.coordinator._hourly_log[-1]
            learning_diagnostics["last_hour_learning_status"] = last_entry.get("learning_status", "unknown")
            learning_diagnostics["aux_model_updated"] = last_entry.get("aux_model_updated", False)

            aux_before = last_entry.get("aux_model_before")
            aux_after = last_entry.get("aux_model_after")

            if aux_before is not None:
                learning_diagnostics["aux_model_value_before"] = f"{aux_before:.5f}"
            else:
                learning_diagnostics["aux_model_value_before"] = None

            if aux_after is not None:
                learning_diagnostics["aux_model_value_after"] = f"{aux_after:.5f}"
            else:
                learning_diagnostics["aux_model_value_after"] = None

            if aux_after is not None and aux_before is not None:
                aux_delta = aux_after - aux_before
                learning_diagnostics["aux_model_delta"] = f"{aux_delta:+.5f}"
            else:
                learning_diagnostics["aux_model_delta"] = None

        return {
            # New Core Attributes
            "theoretical_max_savings": round(theoretical_max, 3),
            "current_savings_rate_kw": rate_display,
            "status": status,

            # Proposed Extension Attributes
            "model_total_aux_kwh": round(model_total_aux_kwh, 3),
            "allocated_total_kwh": round(global_allocated_sum, 3),
            "unassigned_kwh": round(global_unassigned_sum, 3),
            "leak_status": status_flag,
            "unit_breakdown": unit_breakdown_list,

            # Legacy / Debug / Detailed Attributes
            "auxiliary_heating_active": active,
            "aux_hours_today": self.coordinator.data.get("savings_aux_hours_today", 0.0),
            "aux_hours_list": self.coordinator.data.get("savings_aux_hours_list", []),

            # Learning Diagnostics
            **learning_diagnostics
        }

    @property
    def unique_id(self) -> str:
        return f"{self.entry.entry_id}_potential_savings"

class HeatingDeviceDailySensor(HeatingAnalyticsBaseSensor):
    """Sensor for individual device daily consumption."""

    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_state_class = SensorStateClass.TOTAL
    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_icon = "mdi:flash-outline"
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: HeatingDataCoordinator, entry: ConfigEntry, source_entity_id: str) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator, entry)
        self.source_entity_id = source_entity_id
        state = self.coordinator.hass.states.get(source_entity_id)
        friendly_name = state.name if state else source_entity_id
        self._attr_name = f"{friendly_name} Daily"
        self._attr_unique_id = f"{entry.entry_id}_daily_{source_entity_id}"

        # State for throttling
        self._last_reported_power = 0.0
        self._last_power_update = None

    @property
    def native_value(self) -> float:
        """Return the state."""
        return round(self.coordinator.data.get("daily_individual", {}).get(self.source_entity_id, 0.0), 3)

    @property
    def extra_state_attributes(self):
        """Return attributes for per-unit correlation and current status."""
        attrs = {}

        # 1. Current Conditions & Prediction (Moved to top as requested)
        # Re-calculate current conditions for this unit
        inertia_temp = self.coordinator._calculate_inertia_temp()
        eff_wind = self.coordinator.data.get("effective_wind", 0.0)

        if inertia_temp is not None:
            temp_current = round(inertia_temp, 1)
            temp_key_current = str(int(round(inertia_temp)))
        else:
            # Fallback
            temp_current = None
            temp_key_current = "0"

        # Wind Bucket
        wind_bucket_current = self.coordinator._get_wind_bucket(eff_wind)
        attrs["wind_bucket_current"] = wind_bucket_current

        # Prediction
        predicted_hourly = 0.0
        if temp_current is not None:
             predicted_hourly = self.coordinator._get_predicted_kwh_per_unit(
                 self.source_entity_id, temp_key_current, wind_bucket_current, temp_current
             )

        attrs["predicted_hourly_current"] = round(predicted_hourly, 3)

        # Throttled Power Update
        current_power = self.coordinator.calculate_unit_rolling_power_watts(self.source_entity_id)
        now = dt_util.now()
        should_update = False

        # 1. Update on hour change
        if self._last_power_update is None or now.hour != self._last_power_update.hour:
            should_update = True
        # 2. Update on significant change (> 5%) or non-zero transition
        elif self._last_reported_power == 0:
            if abs(current_power) > 0:
                should_update = True
        else:
            change_pct = abs(current_power - self._last_reported_power) / self._last_reported_power
            if change_pct > 0.05:
                should_update = True

        if should_update:
            self._last_reported_power = current_power
            self._last_power_update = now

        attrs["average_power_current"] = self._last_reported_power

        # Theoretical Daily Consumption (Option A: Forecast Today for this unit)
        # We replace the naive "Current Rate * 24" projection with the proper full-day forecast
        # which accounts for actuals so far + forecast for remainder of day.
        # As per PR feedback, we return None (Unknown) if the forecast is unavailable
        # instead of falling back to a misleading linear projection.
        unit_forecast = self.coordinator.data.get("forecast_today_per_unit", {}).get(self.source_entity_id)
        attrs["theoretical_daily_consumption"] = unit_forecast

        # Metadata
        if self.coordinator._hourly_log:
             attrs["last_learning_update"] = self.coordinator._hourly_log[-1]["timestamp"]
        else:
             attrs["last_learning_update"] = None

        # 2. Correlation Data (Moved to bottom)
        unit_data = self.coordinator._correlation_data_per_unit.get(self.source_entity_id, {})

        # Calculate Total Observations (Training Hours)
        unit_counts = self.coordinator._observation_counts.get(self.source_entity_id, {})
        total_observations = 0
        for temp_counts in unit_counts.values():
            for count in temp_counts.values():
                total_observations += count

        for temp_key, buckets in unit_data.items():
            try:
                temp_int = int(temp_key)
                if temp_int < 0:
                    prefix = f"correlation_minus_{abs(temp_int)}"
                else:
                    prefix = f"correlation_{temp_int}"
            except ValueError:
                continue

            for bucket, kwh in buckets.items():
                key_daily = f"{prefix}_{bucket}_daily"
                attrs[key_daily] = round(kwh * 24, 3)

        attrs["correlation_data_points"] = total_observations

        return attrs

class HeatingDeviceLifetimeSensor(HeatingAnalyticsBaseSensor):
    """Sensor for individual device lifetime consumption."""

    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_state_class = SensorStateClass.TOTAL_INCREASING
    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_icon = "mdi:flash"
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: HeatingDataCoordinator, entry: ConfigEntry, source_entity_id: str) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator, entry)
        self.source_entity_id = source_entity_id
        state = self.coordinator.hass.states.get(source_entity_id)
        friendly_name = state.name if state else source_entity_id
        self._attr_name = f"{friendly_name} Lifetime"
        self._attr_unique_id = f"{entry.entry_id}_lifetime_{source_entity_id}"
        self._last_reported_value = 0.0

    @property
    def native_value(self) -> float:
        """Return the state.

        Throttled to update only on significant changes (>= 0.1 kWh) to reduce DB bloat.
        Rounds to 1 decimal place.
        """
        current = self.coordinator.data.get("lifetime_individual", {}).get(self.source_entity_id, 0.0)

        if abs(current - self._last_reported_value) >= 0.1:
            self._last_reported_value = current

        return round(self._last_reported_value, 1)


# HeatingModelComparisonBaseSensor and derived classes moved to .sensors.comparison


class HeatingThermalStateSensor(HeatingAnalyticsBaseSensor):
    """Sensor for the building's Thermal State.

    Exposes the inertia-weighted effective temperature used for predictions,
    along with thermal lag and trend analysis.
    """

    _attr_name = SENSOR_THERMAL_STATE
    _attr_native_unit_of_measurement = UnitOfTemperature.CELSIUS
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_device_class = SensorDeviceClass.TEMPERATURE
    _attr_icon = "mdi:thermometer-chevron-up"

    @property
    def native_value(self) -> float | None:
        """Return the effective temperature."""
        data = self.coordinator.data.get("thermal_state", {})
        return data.get("effective_temperature")

    @property
    def extra_state_attributes(self):
        """Return rich thermal attributes."""
        attrs = self.coordinator.data.get("thermal_state", {}).copy()

        now = dt_util.now()
        primary_item = self.coordinator.forecast.get_forecast_for_hour(now, source='primary_reference')
        secondary_item = self.coordinator.forecast.get_forecast_for_hour(now, source='secondary_reference')
        attrs["midnight_forecast_temp_primary"] = float(primary_item["temperature"]) if primary_item and "temperature" in primary_item else None
        attrs["midnight_forecast_temp_secondary"] = float(secondary_item["temperature"]) if secondary_item and "temperature" in secondary_item else None

        return attrs

    @property
    def unique_id(self) -> str:
        """Return a unique ID."""
        return f"{self.entry.entry_id}_thermal_state"


class HeatingWeekAheadForecastSensor(HeatingAnalyticsBaseSensor):
    """Sensor for 7-Day Week Ahead Forecast."""

    _attr_name = SENSOR_WEEK_AHEAD_FORECAST
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:calendar-week"

    def _get_week_ahead_stats(self) -> dict:
        """Get week ahead stats with hour-based caching."""
        now = dt_util.now()

        # Check cache
        if (
            self._cached_stats
            and self._cached_time
            and self._cached_time.hour == now.hour
            and self._cached_time.date() == now.date()
        ):
            return self._cached_stats

        try:
            stats = self.coordinator.forecast.calculate_week_ahead_stats()
            self._cached_stats = stats
            self._cached_time = now
            return stats
        except Exception as e:
            _LOGGER.error("Error calculating week ahead forecast: %s", e)
            return {
                "total_kwh": 0.0,
                ATTR_WEEKLY_SUMMARY: "Error calculating forecast"
            }

    @property
    def native_value(self) -> float:
        """Return total predicted energy for next 7 days."""
        stats = self._get_week_ahead_stats()
        return stats.get("total_kwh", 0.0)

    @property
    def extra_state_attributes(self):
        """Return rich forecast attributes."""
        stats = self._get_week_ahead_stats()
        # Filter out the main state value to avoid duplication if desired,
        # but keep other relevant stats.
        return {k: v for k, v in stats.items() if k != "total_kwh"}

    @property
    def unique_id(self) -> str:
        return f"{self.entry.entry_id}_week_ahead_forecast"


class HeatingAnalyticsComparisonSensor(HeatingAnalyticsBaseSensor):
    """Sensor for displaying period comparison data."""

    _attr_name = SENSOR_PERIOD_COMPARISON
    _attr_icon = "mdi:compare"
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    @property
    def native_value(self) -> str:
        """Return the state of the sensor."""
        last_comparison = self.coordinator.data.get("last_comparison")
        if last_comparison:
            p1_start = last_comparison.get("period_1", {}).get("start_date")
            p2_start = last_comparison.get("period_2", {}).get("start_date")
            if p1_start and p2_start:
                return f"Comparing {p1_start} to {p2_start}"
        return "No comparison run"

    @property
    def extra_state_attributes(self):
        """Return the state attributes."""
        comparison = self.coordinator.data.get("last_comparison")
        if not comparison:
            return None

        attrs = dict(comparison)
        formatter = ExplanationFormatter()
        attrs["comparison_summary"] = formatter.format_comparison_summary(comparison)
        return attrs

    @property
    def unique_id(self) -> str:
        """Return a unique ID."""
        return f"{self.entry.entry_id}_period_comparison"
