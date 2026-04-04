"""Config flow for Heating Analytics integration."""
from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.const import CONF_NAME
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector

from .const import (
    DOMAIN,
    DEFAULT_NAME,
    DEFAULT_WIND_GUST_FACTOR,
    DEFAULT_BALANCE_POINT,
    DEFAULT_WIND_THRESHOLD,
    DEFAULT_EXTREME_WIND_THRESHOLD,
    DEFAULT_CSV_AUTO_LOGGING,
    DEFAULT_CSV_HOURLY_PATH,
    DEFAULT_CSV_DAILY_PATH,
    CONF_WIND_UNIT,
    CONF_ENABLE_LIFETIME_TRACKING,
    CONF_SOLAR_ENABLED,
    CONF_HAS_AC_UNITS,
    DEFAULT_WIND_UNIT,
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
    DEFAULT_THERMAL_INERTIA_HOURS,
    CONF_INDOOR_TEMP_SENSOR,
    CONF_THERMAL_MASS,
    DEFAULT_THERMAL_MASS,
    CONF_DAILY_LEARNING_MODE,
    CONF_TRACK_C,
    CONF_MPC_ENTRY_ID,
    CONF_MPC_MANAGED_SENSOR,
)

_LOGGER = logging.getLogger(__name__)

# UI-only key — not stored in entry.data. Derived from indoor_temp_sensor presence on load.
_CONF_LOAD_SHIFT = "overnight_load_shift_correction"


class HeatingAnalyticsConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Heating Analytics."""

    VERSION = 2

    def __init__(self):
        self._flow_data: dict[str, Any] = {}
        self._entry = None  # populated during reconfigure

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _v(user_input, defaults, key, default=None):
        """Return value: user_input > defaults > default."""
        if user_input and key in user_input:
            return user_input[key]
        if defaults and key in defaults:
            return defaults[key]
        return default

    def _validate_basics(self, user_input: dict) -> dict[str, str]:
        """Validate step 1: weather entity exists and has temperature if no sensor."""
        errors: dict[str, str] = {}
        weather_id = user_input.get("weather_entity")
        if not weather_id:
            errors["base"] = "weather_entity_required"
            return errors
        state = self.hass.states.get(weather_id)
        if not state:
            errors["base"] = "weather_entity_not_found"
            return errors
        if not user_input.get("outdoor_temp_sensor"):
            if "temperature" not in state.attributes:
                errors["weather_entity"] = "weather_missing_temperature"
        return errors

    def _validate_physics(self, user_input: dict) -> dict[str, str]:
        """Validate step 2: weather entity has wind_speed if no sensor."""
        if not user_input.get("wind_speed_sensor"):
            weather_id = self._flow_data.get("weather_entity")
            if weather_id:
                state = self.hass.states.get(weather_id)
                if state and "wind_speed" not in state.attributes:
                    return {"base": "weather_missing_wind_speed"}
        return {}

    def _clear_absent_entity_keys(self, user_input: dict, keys: list) -> None:
        """Pop optional entity keys from flow_data when the user has cleared them.

        HA/voluptuous drops absent Optional keys from user_input entirely rather
        than sending None. Without this, a previously saved value in self._flow_data
        survives .update(user_input) unchanged even though the user cleared the field.
        """
        for key in keys:
            if not user_input.get(key):
                self._flow_data.pop(key, None)

    def _needs_feature_config_step(self) -> bool:
        """Return True when the feature_config page has at least one field to show."""
        daily = self._flow_data.get(CONF_DAILY_LEARNING_MODE, False)
        load_shift = self._flow_data.get(_CONF_LOAD_SHIFT, False)
        track_c = self._flow_data.get(CONF_TRACK_C, False)
        csv = self._flow_data.get("csv_auto_logging", False)
        return (daily and (load_shift or track_c)) or csv

    def _build_final_data(self, last_step_input: dict) -> dict:
        """Merge all steps and normalise values for storage."""
        data = {**self._flow_data, **last_step_input}

        # Derive source constants from sensor presence (no more source dropdowns)
        data[CONF_OUTDOOR_TEMP_SOURCE] = SOURCE_SENSOR if data.get("outdoor_temp_sensor") else SOURCE_WEATHER
        data[CONF_WIND_SOURCE] = SOURCE_SENSOR if data.get("wind_speed_sensor") else SOURCE_WEATHER
        data[CONF_WIND_GUST_SOURCE] = SOURCE_SENSOR if data.get("wind_gust_sensor") else SOURCE_WEATHER

        # Solar gain is always enabled
        data[CONF_SOLAR_ENABLED] = True

        # Strip absent/falsy optional entity keys so EntitySelector never shows "None".
        if not data.get(CONF_SECONDARY_WEATHER_ENTITY):
            data.pop(CONF_SECONDARY_WEATHER_ENTITY, None)
        # Clear indoor_temp_sensor when load-shift correction is not active so a
        # previously saved sensor does not leak in from _flow_data.
        if not (data.get(CONF_DAILY_LEARNING_MODE) and data.get(_CONF_LOAD_SHIFT)):
            data.pop(CONF_INDOOR_TEMP_SENSOR, None)
        # _CONF_LOAD_SHIFT is UI-only — never stored.
        data.pop(_CONF_LOAD_SHIFT, None)
        # Clear Track C state when daily_learning_mode is off (Track C requires it),
        # or when track_c_enabled is explicitly off. Prevents stale flags surviving
        # a reconfigure where the daily_learning_mode toggle was hidden and absent
        # from the input.
        if not data.get(CONF_DAILY_LEARNING_MODE):
            data.pop(CONF_TRACK_C, None)
            data.pop(CONF_MPC_ENTRY_ID, None)
            data.pop(CONF_MPC_MANAGED_SENSOR, None)
        elif not data.get(CONF_TRACK_C):
            data.pop(CONF_MPC_ENTRY_ID, None)
            data.pop(CONF_MPC_MANAGED_SENSOR, None)
        for key in [
            "outdoor_temp_sensor", "wind_speed_sensor", "wind_gust_sensor",
            CONF_INDOOR_TEMP_SENSOR,
        ]:
            if not data.get(key):
                data.pop(key, None)

        # Convert wind thresholds from display unit to m/s for storage
        unit = data.get(CONF_WIND_UNIT, DEFAULT_WIND_UNIT)
        if unit != WIND_UNIT_MS:
            data["wind_threshold"] = convert_to_ms(data["wind_threshold"], unit)
            data["extreme_wind_threshold"] = convert_to_ms(data["extreme_wind_threshold"], unit)

        # Ensure thermal mass is always present (Track B field may not have been shown)
        data.setdefault(CONF_THERMAL_MASS, DEFAULT_THERMAL_MASS)

        # Default aux_affected_entities to all energy sensors when left empty
        if not data.get(CONF_AUX_AFFECTED_ENTITIES):
            data[CONF_AUX_AFFECTED_ENTITIES] = data.get("energy_sensors", [])

        return data

    # ------------------------------------------------------------------ #
    # Schema builders                                                      #
    # ------------------------------------------------------------------ #

    def _schema_basics(self, user_input, defaults) -> vol.Schema:
        g = lambda k, d=None: self._v(user_input, defaults, k, d)
        schema: dict = {
            vol.Required(CONF_NAME, default=g(CONF_NAME, DEFAULT_NAME)): str,
            vol.Required("weather_entity", default=g("weather_entity")): selector.EntitySelector(
                selector.EntitySelectorConfig(domain="weather")
            ),
            vol.Required("energy_sensors", default=g("energy_sensors", [])): selector.EntitySelector(
                selector.EntitySelectorConfig(domain="sensor", device_class="energy", multiple=True)
            ),
        }
        schema[vol.Optional("outdoor_temp_sensor", description={"suggested_value": g("outdoor_temp_sensor")})] = (
            selector.EntitySelector(selector.EntitySelectorConfig(domain="sensor", device_class="temperature"))
        )
        return vol.Schema(schema)

    def _schema_physics(self, user_input, defaults) -> vol.Schema:
        g = lambda k, d=None: self._v(user_input, defaults, k, d)
        inertia = g(CONF_THERMAL_INERTIA, DEFAULT_THERMAL_INERTIA_HOURS)
        if isinstance(inertia, str):
            inertia = {"fast": 2, "slow": 12}.get(inertia, 4)
        schema: dict = {
            vol.Required(CONF_WIND_UNIT, default=g(CONF_WIND_UNIT, DEFAULT_WIND_UNIT)): selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=[WIND_UNIT_MS, WIND_UNIT_KMH, WIND_UNIT_KNOTS],
                    mode=selector.SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Optional(CONF_HAS_AC_UNITS, default=g(CONF_HAS_AC_UNITS, False)): selector.BooleanSelector(),
            vol.Required("balance_point", default=g("balance_point", DEFAULT_BALANCE_POINT)): selector.NumberSelector(
                selector.NumberSelectorConfig(min=10, max=25, step=0.5, unit_of_measurement="°C")
            ),
            vol.Required(CONF_THERMAL_INERTIA, default=int(inertia)): selector.NumberSelector(
                selector.NumberSelectorConfig(min=1, max=24, step=1, unit_of_measurement="h", mode="slider")
            ),
        }
        for field, device_class in [
            ("wind_speed_sensor", "wind_speed"),
            ("wind_gust_sensor", "wind_speed"),
        ]:
            schema[vol.Optional(field, description={"suggested_value": g(field)})] = selector.EntitySelector(
                selector.EntitySelectorConfig(domain="sensor", device_class=device_class)
            )
        return vol.Schema(schema)

    def _schema_advanced(self, user_input, defaults) -> vol.Schema:
        """Schema for toggles, wind tuning, and miscellaneous options.

        All boolean feature toggles live here. Their corresponding sub-fields
        (sensors, paths, entry selectors) are on the *next* page
        (feature_config) so no dynamic re-render is ever needed.
        """
        g = lambda k, d=None: self._v(user_input, defaults, k, d)
        current_unit = self._flow_data.get(CONF_WIND_UNIT, DEFAULT_WIND_UNIT)

        # Wind thresholds: stored in m/s, displayed in the unit chosen in step 2
        if user_input is not None and "wind_threshold" in user_input:
            display_threshold = user_input["wind_threshold"]
            display_extreme = user_input["extreme_wind_threshold"]
        else:
            stored_t = self._flow_data.get("wind_threshold", DEFAULT_WIND_THRESHOLD)
            stored_e = self._flow_data.get("extreme_wind_threshold", DEFAULT_EXTREME_WIND_THRESHOLD)
            if current_unit != WIND_UNIT_MS:
                display_threshold = convert_from_ms(stored_t, current_unit)
                display_extreme = convert_from_ms(stored_e, current_unit)
            else:
                display_threshold = stored_t
                display_extreme = stored_e

        max_wind = {WIND_UNIT_KMH: 80.0, WIND_UNIT_KNOTS: 40.0}.get(current_unit, 20.0)
        max_extreme = {WIND_UNIT_KMH: 120.0, WIND_UNIT_KNOTS: 60.0}.get(current_unit, 30.0)

        # Derive load-shift default from whether indoor_temp_sensor is already configured
        load_shift = g(_CONF_LOAD_SHIFT, bool(g(CONF_INDOOR_TEMP_SENSOR)))

        schema: dict = {
            vol.Optional(CONF_DAILY_LEARNING_MODE, default=g(CONF_DAILY_LEARNING_MODE, False)): selector.BooleanSelector(),
            vol.Optional(_CONF_LOAD_SHIFT, default=bool(load_shift)): selector.BooleanSelector(),
            vol.Optional(CONF_TRACK_C, default=g(CONF_TRACK_C, False)): selector.BooleanSelector(),
            vol.Required("wind_gust_factor", default=g("wind_gust_factor", DEFAULT_WIND_GUST_FACTOR)): selector.NumberSelector(
                selector.NumberSelectorConfig(min=0.0, max=1.0, step=0.05, mode="slider")
            ),
            vol.Required("wind_threshold", default=round(display_threshold, 1)): selector.NumberSelector(
                selector.NumberSelectorConfig(min=0.0, max=max_wind, step=0.1, unit_of_measurement=current_unit)
            ),
            vol.Required("extreme_wind_threshold", default=round(display_extreme, 1)): selector.NumberSelector(
                selector.NumberSelectorConfig(min=0.0, max=max_extreme, step=0.1, unit_of_measurement=current_unit)
            ),
            vol.Optional("max_energy_delta", default=g("max_energy_delta", DEFAULT_MAX_ENERGY_DELTA)): selector.NumberSelector(
                selector.NumberSelectorConfig(min=0.5, max=15.0, step=0.5, unit_of_measurement="kWh")
            ),
            vol.Optional(
                CONF_AUX_AFFECTED_ENTITIES,
                default=g(CONF_AUX_AFFECTED_ENTITIES, self._flow_data.get("energy_sensors", [])),
            ): selector.EntitySelector(
                selector.EntitySelectorConfig(domain="sensor", device_class="energy", multiple=True)
            ),
            vol.Optional("csv_auto_logging", default=g("csv_auto_logging", DEFAULT_CSV_AUTO_LOGGING)): selector.BooleanSelector(),
            vol.Optional(CONF_ENABLE_LIFETIME_TRACKING, default=g(CONF_ENABLE_LIFETIME_TRACKING, False)): selector.BooleanSelector(),
        }
        schema[vol.Optional(
            CONF_SECONDARY_WEATHER_ENTITY,
            description={"suggested_value": g(CONF_SECONDARY_WEATHER_ENTITY)},
        )] = selector.EntitySelector(selector.EntitySelectorConfig(domain="weather"))
        schema[vol.Optional(CONF_FORECAST_CROSSOVER_DAY, default=g(CONF_FORECAST_CROSSOVER_DAY, DEFAULT_FORECAST_CROSSOVER_DAY))] = (
            selector.NumberSelector(
                selector.NumberSelectorConfig(min=1, max=7, step=1, mode="slider")
            )
        )
        return vol.Schema(schema)

    def _schema_feature_config(self, user_input, defaults) -> vol.Schema:
        """Schema for sub-fields gated by the toggles set on the advanced page.

        Only shown when at least one toggle requires additional input.
        The schema is fully determined by _flow_data — no re-render needed.
        """
        g = lambda k, d=None: self._v(user_input, defaults, k, d)
        daily = self._flow_data.get(CONF_DAILY_LEARNING_MODE, False)
        load_shift = self._flow_data.get(_CONF_LOAD_SHIFT, False)
        track_c = self._flow_data.get(CONF_TRACK_C, False)
        csv_logging = self._flow_data.get("csv_auto_logging", False)

        schema: dict = {}

        if daily and load_shift:
            schema[vol.Optional(
                CONF_INDOOR_TEMP_SENSOR,
                description={"suggested_value": g(CONF_INDOOR_TEMP_SENSOR)},
            )] = selector.EntitySelector(
                selector.EntitySelectorConfig(domain="sensor", device_class="temperature")
            )
            schema[vol.Required(CONF_THERMAL_MASS, default=g(CONF_THERMAL_MASS, DEFAULT_THERMAL_MASS))] = (
                selector.NumberSelector(
                    selector.NumberSelectorConfig(min=0.0, max=50.0, step=0.1, unit_of_measurement="kWh/°C")
                )
            )

        if daily and track_c:
            schema[vol.Required(CONF_MPC_ENTRY_ID, default=g(CONF_MPC_ENTRY_ID, ""))] = (
                selector.ConfigEntrySelector(
                    selector.ConfigEntrySelectorConfig(integration="heatpump_mpc")
                )
            )
            # Which energy sensor is managed by the MPC?  Its meter data is
            # time-shifted by load-scheduling and must not be used for per-unit
            # Track A learning.  Track C replaces it with a synthetic baseline.
            configured_sensors = self._flow_data.get("energy_sensors", [])
            if configured_sensors:
                schema[vol.Required(CONF_MPC_MANAGED_SENSOR, default=g(CONF_MPC_MANAGED_SENSOR, configured_sensors[0]))] = (
                    selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=configured_sensors,
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    )
                )

        if csv_logging:
            schema[vol.Optional("csv_hourly_path", default=g("csv_hourly_path", DEFAULT_CSV_HOURLY_PATH))] = (
                selector.TextSelector()
            )
            schema[vol.Optional("csv_daily_path", default=g("csv_daily_path", DEFAULT_CSV_DAILY_PATH))] = (
                selector.TextSelector()
            )

        return vol.Schema(schema)

    # ------------------------------------------------------------------ #
    # Setup flow: user → physics → advanced → [feature_config] →          #
    #             create_entry                                             #
    # ------------------------------------------------------------------ #

    async def async_step_user(self, user_input=None) -> FlowResult:
        errors: dict[str, str] = {}
        if user_input is not None:
            errors = self._validate_basics(user_input)
            if not errors:
                self._flow_data.update(user_input)
                self._clear_absent_entity_keys(user_input, ["outdoor_temp_sensor"])
                return await self.async_step_physics()
        return self.async_show_form(
            step_id="user",
            data_schema=self._schema_basics(user_input, self._flow_data),
            errors=errors,
        )

    async def async_step_physics(self, user_input=None) -> FlowResult:
        errors: dict[str, str] = {}
        if user_input is not None:
            errors = self._validate_physics(user_input)
            if not errors:
                self._flow_data.update(user_input)
                self._clear_absent_entity_keys(user_input, ["wind_speed_sensor", "wind_gust_sensor"])
                self._flow_data[CONF_HAS_AC_UNITS] = bool(user_input.get(CONF_HAS_AC_UNITS, False))
                return await self.async_step_advanced()
        return self.async_show_form(
            step_id="physics",
            data_schema=self._schema_physics(user_input, self._flow_data),
            errors=errors,
        )

    async def async_step_advanced(self, user_input=None) -> FlowResult:
        if user_input is not None:
            self._flow_data.update(user_input)
            self._clear_absent_entity_keys(user_input, [CONF_SECONDARY_WEATHER_ENTITY])
            # Explicitly write all boolean keys so that a False value is always
            # present in _flow_data even if HA omits it from user_input.
            for _k in (CONF_DAILY_LEARNING_MODE, _CONF_LOAD_SHIFT, CONF_TRACK_C,
                       "csv_auto_logging", CONF_ENABLE_LIFETIME_TRACKING):
                self._flow_data[_k] = bool(user_input.get(_k, False))
            if self._needs_feature_config_step():
                return await self.async_step_feature_config()
            return self.async_create_entry(
                title=self._flow_data[CONF_NAME],
                data=self._build_final_data({}),
            )
        return self.async_show_form(
            step_id="advanced",
            data_schema=self._schema_advanced(None, self._flow_data),
        )

    async def async_step_feature_config(self, user_input=None) -> FlowResult:
        errors: dict[str, str] = {}
        if user_input is not None:
            # Clear optional entity fields that were visible but left empty.
            # vol.Optional with suggested_value omits the key from user_input
            # when blank; without this the stale value in _flow_data leaks in.
            self._clear_absent_entity_keys(user_input, [CONF_INDOOR_TEMP_SENSOR])
            if not errors:
                return self.async_create_entry(
                    title=self._flow_data[CONF_NAME],
                    data=self._build_final_data(user_input),
                )
        return self.async_show_form(
            step_id="feature_config",
            data_schema=self._schema_feature_config(user_input, self._flow_data),
            errors=errors,
        )

    # ------------------------------------------------------------------ #
    # Reconfigure flow: reconfigure → reconfigure_physics →              #
    #                   reconfigure_advanced → [reconfigure_feature_config]#
    #                   → update_reload_and_abort                         #
    # ------------------------------------------------------------------ #

    async def async_step_reconfigure(self, user_input=None) -> FlowResult:
        entry = self.hass.config_entries.async_get_entry(self.context["entry_id"])
        if user_input is None:
            self._entry = entry
            self._flow_data = {**entry.data}
        errors: dict[str, str] = {}
        if user_input is not None:
            errors = self._validate_basics(user_input)
            if not errors:
                # Sensor swap guard: simultaneous removal + addition almost always
                # means the user is trying to replace a broken/renamed sensor inline.
                # This destroys the learned history for the removed sensor.
                # The correct path is the 'replace_sensor' service call.
                old_sensors = set(self._flow_data.get("energy_sensors", []))
                new_sensors = set(user_input.get("energy_sensors", []))
                if (old_sensors - new_sensors) and (new_sensors - old_sensors):
                    errors["energy_sensors"] = "sensor_swap_detected"
            if not errors:
                self._flow_data.update(user_input)
                self._clear_absent_entity_keys(user_input, ["outdoor_temp_sensor"])
                return await self.async_step_reconfigure_physics()
        return self.async_show_form(
            step_id="reconfigure",
            data_schema=self._schema_basics(user_input, self._flow_data),
            errors=errors,
        )

    async def async_step_reconfigure_physics(self, user_input=None) -> FlowResult:
        errors: dict[str, str] = {}
        if user_input is not None:
            errors = self._validate_physics(user_input)
            if not errors:
                self._flow_data.update(user_input)
                self._clear_absent_entity_keys(user_input, ["wind_speed_sensor", "wind_gust_sensor"])
                self._flow_data[CONF_HAS_AC_UNITS] = bool(user_input.get(CONF_HAS_AC_UNITS, False))
                return await self.async_step_reconfigure_advanced()
        return self.async_show_form(
            step_id="reconfigure_physics",
            data_schema=self._schema_physics(user_input, self._flow_data),
            errors=errors,
        )

    async def async_step_reconfigure_advanced(self, user_input=None) -> FlowResult:
        if user_input is not None:
            self._flow_data.update(user_input)
            self._clear_absent_entity_keys(user_input, [CONF_SECONDARY_WEATHER_ENTITY])
            # Explicitly write all boolean keys so that a False value is always
            # present in _flow_data even if HA omits it from user_input.
            for _k in (CONF_DAILY_LEARNING_MODE, _CONF_LOAD_SHIFT, CONF_TRACK_C,
                       "csv_auto_logging", CONF_ENABLE_LIFETIME_TRACKING):
                self._flow_data[_k] = bool(user_input.get(_k, False))
            # Run aux migration here — CONF_AUX_AFFECTED_ENTITIES is on this page
            # and _flow_data now has the new value regardless of whether the
            # feature_config page is shown next or skipped entirely.
            new_aux = user_input.get(CONF_AUX_AFFECTED_ENTITIES)
            if new_aux is not None:
                coord = self.hass.data.get(DOMAIN, {}).get(self.context["entry_id"])
                if coord:
                    await coord.async_migrate_aux_coefficients(new_aux)
            if self._needs_feature_config_step():
                return await self.async_step_reconfigure_feature_config()
            return self.async_update_reload_and_abort(
                self._entry, data=self._build_final_data({})
            )
        return self.async_show_form(
            step_id="reconfigure_advanced",
            data_schema=self._schema_advanced(None, self._flow_data),
        )

    async def async_step_reconfigure_feature_config(self, user_input=None) -> FlowResult:
        errors: dict[str, str] = {}
        if user_input is not None:
            # Clear optional entity fields that were visible but left empty.
            self._clear_absent_entity_keys(user_input, [CONF_INDOOR_TEMP_SENSOR])
            if not errors:
                return self.async_update_reload_and_abort(
                    self._entry, data=self._build_final_data(user_input)
                )
        return self.async_show_form(
            step_id="reconfigure_feature_config",
            data_schema=self._schema_feature_config(user_input, self._flow_data),
            errors=errors,
        )
