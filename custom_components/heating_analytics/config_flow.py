"""Config flow for Heating Analytics integration."""
from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector

from .const import (
    DOMAIN,
    DEFAULT_NAME,
    DEFAULT_WIND_GUST_FACTOR,
    DEFAULT_LEARNING_RATE,
    DEFAULT_BALANCE_POINT,
    DEFAULT_WIND_THRESHOLD,
    DEFAULT_EXTREME_WIND_THRESHOLD,
    DEFAULT_CSV_AUTO_LOGGING,
    DEFAULT_CSV_HOURLY_PATH,
    DEFAULT_CSV_DAILY_PATH,
    CONF_WIND_UNIT,
    CONF_ENABLE_LIFETIME_TRACKING,
    CONF_SOLAR_ENABLED,
    CONF_SOLAR_AZIMUTH,
    CONF_HAS_AC_UNITS,
    DEFAULT_WIND_UNIT,
    DEFAULT_SOLAR_ENABLED,
    DEFAULT_SOLAR_AZIMUTH,
    WIND_UNIT_MS,
    WIND_UNIT_KMH,
    WIND_UNIT_KNOTS,
    DEFAULT_MAX_ENERGY_DELTA,
    convert_from_ms,
    convert_to_ms,
    CONF_OUTDOOR_TEMP_SOURCE,
    CONF_WIND_SOURCE,
    CONF_WIND_GUST_SOURCE,
    SOURCE_SENSOR,
    SOURCE_WEATHER,
    CONF_SECONDARY_WEATHER_ENTITY,
    CONF_FORECAST_CROSSOVER_DAY,
    DEFAULT_FORECAST_CROSSOVER_DAY,
    CONF_AUX_AFFECTED_ENTITIES,
    CONF_THERMAL_INERTIA,
    THERMAL_INERTIA_FAST,
    THERMAL_INERTIA_NORMAL,
    THERMAL_INERTIA_SLOW,
)

_LOGGER = logging.getLogger(__name__)

class HeatingAnalyticsConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Heating Analytics."""

    VERSION = 2

    def _validate_weather_fallback(self, user_input: dict[str, Any]) -> dict[str, str]:
        """Validate that weather entity has required attributes if sensors are missing."""
        errors = {}
        weather_entity_id = user_input.get("weather_entity")

        if not weather_entity_id:
            errors["base"] = "weather_entity_required"
            return errors

        weather_state = self.hass.states.get(weather_entity_id)
        if not weather_state:
            errors["base"] = "weather_entity_not_found"
            return errors

        # Validation relies on Source selection now
        # If Source is SENSOR, we expect a sensor.
        # If Source is WEATHER, we expect the weather entity to have the attribute.

        # Temp
        if user_input.get(CONF_OUTDOOR_TEMP_SOURCE) == SOURCE_WEATHER:
             if "temperature" not in weather_state.attributes:
                errors[CONF_OUTDOOR_TEMP_SOURCE] = "weather_missing_temperature"
        elif user_input.get(CONF_OUTDOOR_TEMP_SOURCE) == SOURCE_SENSOR:
             if not user_input.get("outdoor_temp_sensor"):
                  # This is handled by required field in schema, but good to double check
                  errors["outdoor_temp_sensor"] = "required"

        # Wind Speed
        if user_input.get(CONF_WIND_SOURCE) == SOURCE_WEATHER:
             if "wind_speed" not in weather_state.attributes:
                errors[CONF_WIND_SOURCE] = "weather_missing_wind_speed"

        # Wind Gust (Optional/Info)
        if user_input.get(CONF_WIND_GUST_SOURCE) == SOURCE_WEATHER:
            if "wind_gust_speed" not in weather_state.attributes:
                 _LOGGER.info(
                     f"Selected weather entity {weather_entity_id} does not have 'wind_gust_speed'. "
                     "Wind gusts will be assumed 0 unless provided by forecast."
                 )

        return errors

    def _get_schema(self, user_input: dict[str, Any], default_data: dict[str, Any], is_reconfigure: bool = False) -> vol.Schema:
        """Generate schema based on current selection."""

        # Helper to get current value (User Input > Default Data > Default Constant)
        def get_val(key, default=None):
            if user_input and key in user_input:
                return user_input[key]
            if default_data and key in default_data:
                return default_data[key]
            return default

        current_unit = get_val(CONF_WIND_UNIT, DEFAULT_WIND_UNIT)

        # Display Conversions
        if is_reconfigure:
            # Stored as m/s
            current_wind_threshold = get_val("wind_threshold", DEFAULT_WIND_THRESHOLD)
            current_extreme = get_val("extreme_wind_threshold", DEFAULT_EXTREME_WIND_THRESHOLD)
            display_wind_threshold = convert_from_ms(current_wind_threshold, current_unit)
            display_extreme = convert_from_ms(current_extreme, current_unit)
        else:
            # Default or input
            display_wind_threshold = get_val("wind_threshold", DEFAULT_WIND_THRESHOLD)
            display_extreme = get_val("extreme_wind_threshold", DEFAULT_EXTREME_WIND_THRESHOLD)

        # Max Values for Sliders
        max_val = 20.0
        max_extreme = 30.0
        if current_unit == WIND_UNIT_KMH:
            max_val = 80.0
            max_extreme = 120.0
        elif current_unit == WIND_UNIT_KNOTS:
            max_val = 40.0
            max_extreme = 60.0

        schema = {
            vol.Required(CONF_NAME, default=get_val(CONF_NAME, DEFAULT_NAME)): str,
            vol.Required(CONF_WIND_UNIT, default=current_unit): selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=[WIND_UNIT_MS, WIND_UNIT_KMH, WIND_UNIT_KNOTS],
                    mode=selector.SelectSelectorMode.DROPDOWN
                )
            ),
            vol.Required("weather_entity", default=get_val("weather_entity")): selector.EntitySelector(
                selector.EntitySelectorConfig(domain="weather")
            ),
        }

        # Dynamic Fields for Sources

        # Outdoor Temp
        temp_source = get_val(CONF_OUTDOOR_TEMP_SOURCE, SOURCE_SENSOR)
        schema[vol.Required(CONF_OUTDOOR_TEMP_SOURCE, default=temp_source)] = selector.SelectSelector(
            selector.SelectSelectorConfig(
                options=[
                    selector.SelectOptionDict(value=SOURCE_SENSOR, label="Dedicated Sensor"),
                    selector.SelectOptionDict(value=SOURCE_WEATHER, label="Weather Entity"),
                ],
                mode=selector.SelectSelectorMode.DROPDOWN
            )
        )
        if temp_source == SOURCE_SENSOR:
             schema[vol.Required("outdoor_temp_sensor", default=get_val("outdoor_temp_sensor"))] = selector.EntitySelector(
                selector.EntitySelectorConfig(domain="sensor", device_class="temperature")
            )

        # Wind Speed
        wind_source = get_val(CONF_WIND_SOURCE, SOURCE_SENSOR)
        schema[vol.Required(CONF_WIND_SOURCE, default=wind_source)] = selector.SelectSelector(
            selector.SelectSelectorConfig(
                options=[
                    selector.SelectOptionDict(value=SOURCE_SENSOR, label="Dedicated Sensor"),
                    selector.SelectOptionDict(value=SOURCE_WEATHER, label="Weather Entity"),
                ],
                mode=selector.SelectSelectorMode.DROPDOWN
            )
        )
        if wind_source == SOURCE_SENSOR:
             schema[vol.Required("wind_speed_sensor", default=get_val("wind_speed_sensor"))] = selector.EntitySelector(
                selector.EntitySelectorConfig(domain="sensor", device_class="wind_speed")
            )

        # Wind Gust
        gust_source = get_val(CONF_WIND_GUST_SOURCE, SOURCE_SENSOR)
        schema[vol.Required(CONF_WIND_GUST_SOURCE, default=gust_source)] = selector.SelectSelector(
            selector.SelectSelectorConfig(
                options=[
                    selector.SelectOptionDict(value=SOURCE_SENSOR, label="Dedicated Sensor"),
                    selector.SelectOptionDict(value=SOURCE_WEATHER, label="Weather Entity"),
                ],
                mode=selector.SelectSelectorMode.DROPDOWN
            )
        )
        if gust_source == SOURCE_SENSOR:
             schema[vol.Optional("wind_gust_sensor", default=get_val("wind_gust_sensor"))] = selector.EntitySelector(
                selector.EntitySelectorConfig(domain="sensor", device_class="wind_speed")
            )

        # Rest of Schema
        schema.update({
            vol.Required("energy_sensors", default=get_val("energy_sensors", [])): selector.EntitySelector(
                selector.EntitySelectorConfig(domain="sensor", device_class="energy", multiple=True)
            ),
            # Aux Affected Entities (Optional - Defaults to All)
            vol.Optional(CONF_AUX_AFFECTED_ENTITIES, default=get_val(CONF_AUX_AFFECTED_ENTITIES, get_val("energy_sensors", []))): selector.EntitySelector(
                selector.EntitySelectorConfig(domain="sensor", device_class="energy", multiple=True)
            ),
            # Parameters
            vol.Required("wind_gust_factor", default=get_val("wind_gust_factor", DEFAULT_WIND_GUST_FACTOR)): selector.NumberSelector(
                selector.NumberSelectorConfig(min=0.0, max=1.0, step=0.05, mode="slider")
            ),
            vol.Required("balance_point", default=get_val("balance_point", DEFAULT_BALANCE_POINT)): selector.NumberSelector(
                selector.NumberSelectorConfig(min=10, max=25, step=0.5, unit_of_measurement="°C")
            ),
            # Thermal Inertia Profile (User Selectable)
            vol.Required(CONF_THERMAL_INERTIA, default=get_val(CONF_THERMAL_INERTIA, THERMAL_INERTIA_NORMAL)): selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=[
                        THERMAL_INERTIA_FAST,
                        THERMAL_INERTIA_NORMAL,
                        THERMAL_INERTIA_SLOW
                    ],
                    mode=selector.SelectSelectorMode.DROPDOWN,
                    translation_key="thermal_inertia"
                )
            ),
            # Learning Rate (Convert float back to percentage for display)
            vol.Required("learning_rate", default=round(get_val("learning_rate", DEFAULT_LEARNING_RATE) * 100, 1)): selector.NumberSelector(
                selector.NumberSelectorConfig(min=0.1, max=10.0, step=0.1, unit_of_measurement="%")
            ),
            # Thresholds
            vol.Required("wind_threshold", default=round(display_wind_threshold, 1)): selector.NumberSelector(
                selector.NumberSelectorConfig(min=0.0, max=max_val, step=0.1, unit_of_measurement=current_unit)
            ),
            vol.Required("extreme_wind_threshold", default=round(display_extreme, 1)): selector.NumberSelector(
                selector.NumberSelectorConfig(min=0.0, max=max_extreme, step=0.1, unit_of_measurement=current_unit)
            ),
            # Spike Protection
            vol.Optional("max_energy_delta", default=get_val("max_energy_delta", DEFAULT_MAX_ENERGY_DELTA)): selector.NumberSelector(
                selector.NumberSelectorConfig(min=0.5, max=15.0, step=0.5, unit_of_measurement="kWh")
            ),
            # Solar Correction
            vol.Optional(CONF_SOLAR_ENABLED, default=get_val(CONF_SOLAR_ENABLED, DEFAULT_SOLAR_ENABLED)): selector.BooleanSelector(),
            vol.Optional(CONF_SOLAR_AZIMUTH, default=get_val(CONF_SOLAR_AZIMUTH, DEFAULT_SOLAR_AZIMUTH)): selector.NumberSelector(
                selector.NumberSelectorConfig(min=0, max=360, step=5, unit_of_measurement="°")
            ),
            # CSV Auto-logging
            vol.Optional("csv_auto_logging", default=get_val("csv_auto_logging", DEFAULT_CSV_AUTO_LOGGING)): selector.BooleanSelector(),
            vol.Optional("csv_hourly_path", default=get_val("csv_hourly_path", DEFAULT_CSV_HOURLY_PATH)): selector.TextSelector(),
            vol.Optional("csv_daily_path", default=get_val("csv_daily_path", DEFAULT_CSV_DAILY_PATH)): selector.TextSelector(),
            # Lifetime Energy Tracking
            vol.Optional(CONF_ENABLE_LIFETIME_TRACKING, default=get_val(CONF_ENABLE_LIFETIME_TRACKING, False)): selector.BooleanSelector(),
            # Global AC Capability Checkbox (To show/hide Mode Selects)
            vol.Optional(CONF_HAS_AC_UNITS, default=get_val(CONF_HAS_AC_UNITS, False)): selector.BooleanSelector(),

            # Advanced Forecast Settings
            vol.Optional(CONF_SECONDARY_WEATHER_ENTITY, default=get_val(CONF_SECONDARY_WEATHER_ENTITY)): selector.EntitySelector(
                selector.EntitySelectorConfig(domain="weather")
            ),
            vol.Optional(CONF_FORECAST_CROSSOVER_DAY, default=get_val(CONF_FORECAST_CROSSOVER_DAY, DEFAULT_FORECAST_CROSSOVER_DAY)): selector.NumberSelector(
                selector.NumberSelectorConfig(min=1, max=7, step=1, mode="slider")
            ),
        })

        return vol.Schema(schema)

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}

        # For initial step, we don't have existing config, so default_data is empty
        # However, we want to start with SENSOR sources by default
        default_data = {
            CONF_OUTDOOR_TEMP_SOURCE: SOURCE_SENSOR,
            CONF_WIND_SOURCE: SOURCE_SENSOR,
            CONF_WIND_GUST_SOURCE: SOURCE_SENSOR,
        }

        if user_input is not None:
            # Check for Mode Changes (Reload form if mode changed)
            # Logic: If user input has a source as 'sensor' but NO sensor key in input,
            # it means the field wasn't shown (or they cleared it? No, if hidden it's not in input).
            # If we simply reload, the _get_schema will use user_input to show/hide fields.

            # Detect if we need to reload due to visibility change
            # If Source is SENSOR but key missing -> Reload to show it
            reload_needed = False

            if user_input.get(CONF_OUTDOOR_TEMP_SOURCE) == SOURCE_SENSOR and "outdoor_temp_sensor" not in user_input:
                reload_needed = True
            if user_input.get(CONF_WIND_SOURCE) == SOURCE_SENSOR and "wind_speed_sensor" not in user_input:
                reload_needed = True
            # Gust is optional, so we do NOT force reload if it is missing,
            # allowing users to select Sensor Source but leave the sensor empty (effectively "No Gust Data").
            # if user_input.get(CONF_WIND_GUST_SOURCE) == SOURCE_SENSOR and "wind_gust_sensor" not in user_input:
            #     reload_needed = True

            # Also if Source is WEATHER but sensor key IS present (switched back?), we might want to reload to hide it?
            # Standard HA form usually submits all visible fields. If field was visible, it's in input.
            # If we switch to Weather, next reload hides it.
            # BUT: If we switch to Weather, we can also just Proceed if valid.

            if not reload_needed:
                # Validate input
                validation_errors = self._validate_weather_fallback(user_input)
                if validation_errors:
                    errors.update(validation_errors)
                else:
                    # Explicitly set optional sensors to None if not used (Source = Weather)
                    if user_input.get(CONF_OUTDOOR_TEMP_SOURCE) == SOURCE_WEATHER:
                        user_input["outdoor_temp_sensor"] = None
                    if user_input.get(CONF_WIND_SOURCE) == SOURCE_WEATHER:
                        user_input["wind_speed_sensor"] = None
                    if user_input.get(CONF_WIND_GUST_SOURCE) == SOURCE_WEATHER:
                        user_input["wind_gust_sensor"] = None

                    # Check unit selected
                    unit = user_input.get(CONF_WIND_UNIT, DEFAULT_WIND_UNIT)

                    # Convert thresholds to m/s for storage
                    if unit != WIND_UNIT_MS:
                        user_input["wind_threshold"] = convert_to_ms(user_input["wind_threshold"], unit)
                        user_input["extreme_wind_threshold"] = convert_to_ms(user_input["extreme_wind_threshold"], unit)

                    # Convert learning rate from % to float
                    if "learning_rate" in user_input:
                        user_input["learning_rate"] = user_input["learning_rate"] / 100.0

                    # Ensure aux_affected_entities defaults to ALL if empty/missing on fresh setup
                    # (This prevents accidental 0-reduction if user ignores the optional field)
                    aux_entities = user_input.get(CONF_AUX_AFFECTED_ENTITIES)
                    if aux_entities is None:
                         user_input[CONF_AUX_AFFECTED_ENTITIES] = user_input.get("energy_sensors", [])

                    return self.async_create_entry(title=user_input[CONF_NAME], data=user_input)

        # Generate Schema
        schema = self._get_schema(user_input, default_data, is_reconfigure=False)

        return self.async_show_form(
            step_id="user",
            data_schema=schema,
            errors=errors,
        )

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle reconfiguration."""
        errors: dict[str, str] = {}
        entry = self.hass.config_entries.async_get_entry(self.context["entry_id"])

        # Prepare existing data (handling migration implied if keys missing)
        default_data = {**entry.data}
        if CONF_OUTDOOR_TEMP_SOURCE not in default_data: default_data[CONF_OUTDOOR_TEMP_SOURCE] = SOURCE_SENSOR
        if CONF_WIND_SOURCE not in default_data: default_data[CONF_WIND_SOURCE] = SOURCE_SENSOR
        if CONF_WIND_GUST_SOURCE not in default_data: default_data[CONF_WIND_GUST_SOURCE] = SOURCE_SENSOR

        if user_input is not None:
             # Logic to detect "Source Change" vs "Submission"
             # If user switched Source to Sensor, but didn't provide sensor (because it was hidden), we must reload.

             reload_needed = False
             if user_input.get(CONF_OUTDOOR_TEMP_SOURCE) == SOURCE_SENSOR and "outdoor_temp_sensor" not in user_input:
                 reload_needed = True
             if user_input.get(CONF_WIND_SOURCE) == SOURCE_SENSOR and "wind_speed_sensor" not in user_input:
                 reload_needed = True
             # Gust is optional - no reload needed if missing
             # if user_input.get(CONF_WIND_GUST_SOURCE) == SOURCE_SENSOR and "wind_gust_sensor" not in user_input:
             #     reload_needed = True

             # Also reload if switching TO Weather, to hide the fields cleanly?
             # User expectation: "Disappearing would suffice".
             # If I select Weather and click submit, and it just saves -> That's fine.
             # If I select Weather and click submit, and it reloads -> That's weird.
             # So ONLY reload if we are MISSING required data for the chosen mode.

             if not reload_needed:
                # Validate input
                validation_errors = self._validate_weather_fallback(user_input)
                if validation_errors:
                    errors.update(validation_errors)
                else:
                    # Handle Data Migration (Conservation Strategy)
                    # If aux_affected_entities changed, we must migrate coefficients BEFORE reload
                    # because reload destroys the coordinator in memory.
                    new_aux_list = user_input.get(CONF_AUX_AFFECTED_ENTITIES)

                    # Ensure we have a valid list comparison
                    if new_aux_list is not None:
                        # Access the running coordinator
                        coordinator = self.hass.data.get(DOMAIN, {}).get(self.context["entry_id"])
                        if coordinator:
                            # Trigger migration
                            # This will redistribute coefficients from removed units to remaining ones
                            # and save the data to disk.
                            # The reload below will then load this fresh data.
                            await coordinator.async_migrate_aux_coefficients(new_aux_list)

                    # Clean up unused sensors
                    if user_input.get(CONF_OUTDOOR_TEMP_SOURCE) == SOURCE_WEATHER:
                        user_input["outdoor_temp_sensor"] = None
                    if user_input.get(CONF_WIND_SOURCE) == SOURCE_WEATHER:
                        user_input["wind_speed_sensor"] = None
                    if user_input.get(CONF_WIND_GUST_SOURCE) == SOURCE_WEATHER:
                        user_input["wind_gust_sensor"] = None

                    # Determine which unit was chosen in THIS form submission
                    # (user_input has priority)
                    new_unit = user_input.get(CONF_WIND_UNIT, default_data.get(CONF_WIND_UNIT))

                    # Convert inputs back to m/s for storage
                    if new_unit != WIND_UNIT_MS:
                        user_input["wind_threshold"] = convert_to_ms(user_input["wind_threshold"], new_unit)
                        user_input["extreme_wind_threshold"] = convert_to_ms(user_input["extreme_wind_threshold"], new_unit)

                    # Convert learning rate from % to float
                    if "learning_rate" in user_input:
                        user_input["learning_rate"] = user_input["learning_rate"] / 100.0

                    return self.async_update_reload_and_abort(
                        entry, data={**entry.data, **user_input}
                    )

        # Generate Schema
        # Pass user_input so the schema reflects the LATEST dropdown choice (e.g. if we are reloading)
        schema = self._get_schema(user_input, default_data, is_reconfigure=True)

        return self.async_show_form(
            step_id="reconfigure",
            data_schema=schema,
            errors=errors,
        )
