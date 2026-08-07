"""Config flow for Heating Analytics integration.

Architecture notes for contributors
------------------------------------
- ``_schema_advanced()`` is shared by both ``async_step_advanced`` and
  ``async_step_reconfigure_advanced``.
- ``_build_final_data`` is called from 4 places (2 create, 2 reconfigure).
  It handles all normalisation: source derivation, boolean cleanup, unit
  conversion, key stripping.  New config fields should be normalised here.
- ``SelectSelector`` returns **string** values.  Convert to the correct type
  in ``_build_final_data`` before storage (see CONF_HOURLY_LOG_RETENTION_DAYS).
- ``_needs_feature_config_step()`` always returns True — wind tuning is always
  shown.  Do not "optimise" this to conditionally skip the step.
- Wind *sensor* fields are behind ``_CONF_DEDICATED_WIND`` (UI-only, not
  stored).  Wind *tuning* is outside the toggle — accessible to all users.
- Wind thresholds are only converted from display-unit to m/s when
  ``wind_from_user`` is True (form values).  ``setdefault`` values are already
  in m/s and must not be double-converted.

Field ordering on ``advanced`` / ``reconfigure_advanced``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Field order is dict insertion order in ``_schema_advanced``.  The axis that
governs it is **expected interaction frequency** — how likely this field is
the reason the user opened the page — **not** consequence.  A setting that is
dangerous but set once at setup belongs low; a setting a user returns to after
reading ``diagnose_solar`` belongs high.

The ``--- Mundane fields at the bottom ---`` group means exactly that
(``csv_auto_logging``, ``max_energy_delta``, ``hourly_log_retention_days``) —
routine, not **expert-only**.  Do not demote a high-consequence setting into
it: HA config flow has no collapsible sections, so "at the bottom" is the only
"tucked away" that exists, and parking an expert setting next to log retention
puts it directly in the path of a user scrolling down for something routine.
"Findable but not offered" needs a separate expert step, which does not exist.

Known deviation, not an example to copy: ``daily_learning_mode`` (Track B) and
``track_c_enabled`` (Track C) sit near the top and do not follow this axis —
both are high-consequence and set once at setup if ever.  Their placement
predates the rule.

``CONF_EXPERIMENTAL_4D_PRIMARY`` is first on a different basis entirely — it
must render under the readiness verdict in the step description.  See
``_schema_advanced``'s docstring; that position is pinned by a test.

Translation notes
~~~~~~~~~~~~~~~~~
Two files: ``translations/en.json`` and ``translations/nb.json``.
No ``strings.json`` — HA loads translations directly.  Config flow steps
``advanced`` and ``reconfigure_advanced`` have **identical data/data_description
keys** — both must be updated when adding a new field.  Use ``replace_all``
when editing both blocks simultaneously.

One deliberate exception: ``experimental_4d_primary`` exists only on
``reconfigure_advanced`` (the initial wizard never renders it).  Likewise the
``{four_d_readiness}`` placeholder belongs only to that step's description —
``advanced`` supplies no ``description_placeholders`` and would render it
literally.  Both are pinned by ``tests/test_translations_4d_readiness.py``.
"""
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
    CONF_SCREEN_SOUTH,
    CONF_SCREEN_EAST,
    CONF_SCREEN_WEST,
    CONF_SCREEN_AFFECTED_ENTITIES,
    CONF_SOLAR_AFFECTED_ENTITIES,
    DEFAULT_SCREEN_SOUTH,
    DEFAULT_SCREEN_EAST,
    DEFAULT_SCREEN_WEST,
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
    CONF_HOURLY_LOG_RETENTION_DAYS,
    DEFAULT_HOURLY_LOG_RETENTION_DAYS,
    HOURLY_LOG_RETENTION_OPTIONS,
    CONF_EXPERIMENTAL_4D_PRIMARY,
    DEFAULT_EXPERIMENTAL_4D_PRIMARY,
)

_LOGGER = logging.getLogger(__name__)

# UI-only key — not stored in entry.data. Derived from indoor_temp_sensor presence on load.
_CONF_LOAD_SHIFT = "overnight_load_shift_correction"
# UI-only key — not stored. Derived from wind_speed_sensor presence on load.
_CONF_DEDICATED_WIND = "use_dedicated_wind_sensor"

# Readiness lines for the 4D solar setting (#1062), surfaced in the
# reconfigure_advanced step description via ``description_placeholders``.
#
# Kept as a language-keyed dict rather than translation keys because HA
# substitutes placeholders as literal strings — a placeholder value cannot
# itself be a translation key.  The alternative (one English line inside
# otherwise-Norwegian prose) would be worst on exactly the field whose whole
# purpose is to remove confusion about what the setting asks.  Falls back to
# English for any language without an entry here.
#
# Warn, never refuse: a user who has just connected a GHI sensor is right to
# enable 4D before the 30-day window catches up, so no verdict produces a
# form error.  ``insufficient_data`` deliberately renders nothing — a fresh
# install has nothing useful to say and a "we don't know yet" line is noise.
_FOUR_D_READINESS_LINES: dict[str, dict[str, str]] = {
    "en": {
        "ready_to_enable": (
            "\n\n✅ 4D solar: your installation is ready. Your weather provider "
            "supplies direct/diffuse irradiance and the 4D model has learned all "
            "units currently in use — you can turn on the 4D solar model below."
        ),
        "enabled_and_ready": (
            "\n\n✅ 4D solar: enabled and working as intended — your weather data "
            "supports it and the model is trained."
        ),
        "enabled_but_not_ready": (
            "\n\n⚠️ 4D solar is enabled, but this installation does not meet the "
            "conditions for it ({reason}). Solar prediction may currently be "
            "degraded. Consider turning it off below, or run the 'diagnose_solar' "
            "service and check 'four_d_readiness' for details."
        ),
        "not_ready": (
            "\n\nℹ️ 4D solar: this installation is not ready for it ({reason}). "
            "Leaving the setting off is recommended. Details in the "
            "'diagnose_solar' service under 'four_d_readiness'."
        ),
    },
    "nb": {
        "ready_to_enable": (
            "\n\n✅ 4D-sol: installasjonen din er klar. Værleverandøren din gir "
            "direkte/diffus solstråling, og 4D-modellen har lært alle enheter som "
            "er i bruk nå — du kan slå på 4D-solmodellen nedenfor."
        ),
        "enabled_and_ready": (
            "\n\n✅ 4D-sol: slått på og fungerer som tiltenkt — værdataene dine "
            "støtter den, og modellen er opplært."
        ),
        "enabled_but_not_ready": (
            "\n\n⚠️ 4D-sol er slått på, men denne installasjonen oppfyller ikke "
            "vilkårene for den ({reason}). Solprediksjonen kan være svekket nå. "
            "Vurder å slå den av nedenfor, eller kjør tjenesten «diagnose_solar» "
            "og se på «four_d_readiness» for detaljer."
        ),
        "not_ready": (
            "\n\nℹ️ 4D-sol: denne installasjonen er ikke klar for den ({reason}). "
            "Det anbefales å la innstillingen stå av. Detaljer i tjenesten "
            "«diagnose_solar» under «four_d_readiness»."
        ),
    },
}

# Why a readiness verdict is negative, in user terms.  Both halves can fail
# at once, so the reasons compose.
_FOUR_D_REASON_TEXT: dict[str, dict[str, str]] = {
    "en": {
        "input": "your weather provider does not supply direct/diffuse irradiance",
        "learning": "the 4D model has not learned enough yet",
        "both": (
            "your weather provider does not supply direct/diffuse irradiance, and "
            "the 4D model has not learned enough yet"
        ),
    },
    "nb": {
        "input": "værleverandøren din gir ikke direkte/diffus solstråling",
        "learning": "4D-modellen har ikke lært nok ennå",
        "both": (
            "værleverandøren din gir ikke direkte/diffus solstråling, og "
            "4D-modellen har ikke lært nok ennå"
        ),
    },
}


class HeatingAnalyticsConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Heating Analytics."""

    VERSION = 3

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

    def _four_d_readiness_placeholder(self) -> str:
        """Readiness line for the 4D solar setting, or "" (#1062).

        Surfaces the verdict where the decision is actually made, so the
        user is not asked a question about their weather provider with
        nothing at the point of decision telling them the answer.

        Advisory only — never blocks the form.  Returns the empty string
        whenever readiness cannot be established or the coordinator is
        unreachable: a config flow that raises while rendering a hint is
        strictly worse than one that renders no hint.
        """
        try:
            coordinator = self.hass.data.get(DOMAIN, {}).get(self.context["entry_id"])
            if coordinator is None:
                return ""
            readiness = coordinator.evaluate_4d_readiness()
        except Exception:  # noqa: BLE001 — a hint must never break the form
            # Warning, not debug: swallowing the exception is required (the
            # form must open regardless), but a genuine bug in
            # ``evaluate_4d_readiness`` would otherwise be invisible at the
            # default log level and present only as a silently missing hint.
            # Reconfigure is a manual, one-shot flow, so this cannot spam.
            _LOGGER.warning(
                "4D readiness could not be evaluated; the reconfigure form "
                "will render without the readiness hint",
                exc_info=True,
            )
            return ""

        verdict = readiness.get("verdict")
        lang = getattr(self.hass.config, "language", None) or "en"
        # HA language codes can be regional ("nb-NO"); match on the base tag.
        lines = _FOUR_D_READINESS_LINES.get(
            lang.split("-")[0].lower(), _FOUR_D_READINESS_LINES["en"]
        )
        template = lines.get(verdict)
        if not template:
            return ""
        if "{reason}" not in template:
            return template

        input_ok = readiness.get("input", {}).get("supports_4d_primary") is True
        learning_ok = readiness.get("learning", {}).get("ready") is True
        reasons = _FOUR_D_REASON_TEXT.get(
            lang.split("-")[0].lower(), _FOUR_D_REASON_TEXT["en"]
        )
        if not input_ok and not learning_ok:
            reason = reasons["both"]
        elif not input_ok:
            reason = reasons["input"]
        else:
            reason = reasons["learning"]
        return template.format(reason=reason)

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
        """Validate step 2: weather entity has wind_speed attribute.

        Wind sensor override is configured on the feature_config page, not here.
        Always validate that the weather entity can supply wind data as a baseline.
        """
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
        """Always True — wind tuning fields are always shown on this page."""
        return True

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
        # _CONF_LOAD_SHIFT and _CONF_DEDICATED_WIND are UI-only — never stored.
        data.pop(_CONF_LOAD_SHIFT, None)
        # Clear wind sensor fields when dedicated wind toggle is off (#798).
        has_dedicated_wind = bool(data.get(_CONF_DEDICATED_WIND))
        if not has_dedicated_wind:
            data.pop("wind_speed_sensor", None)
            data.pop("wind_gust_sensor", None)
        data.pop(_CONF_DEDICATED_WIND, None)
        # Auto-enable Daily Learning when Track C is turned on — Track C is a
        # variant of daily learning that replaces flat q/24 with COP-weighted
        # smearing.  It makes no sense without daily mode.
        if data.get(CONF_TRACK_C) and not data.get(CONF_DAILY_LEARNING_MODE):
            data[CONF_DAILY_LEARNING_MODE] = True
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

        # Hourly log retention: SelectSelector returns a string, store as int (#820).
        if CONF_HOURLY_LOG_RETENTION_DAYS in data:
            data[CONF_HOURLY_LOG_RETENTION_DAYS] = int(data[CONF_HOURLY_LOG_RETENTION_DAYS])
        else:
            data[CONF_HOURLY_LOG_RETENTION_DAYS] = DEFAULT_HOURLY_LOG_RETENTION_DAYS

        # Ensure wind tuning fields have defaults.  Defaults are in m/s and
        # must NOT be converted — only user-submitted values (in display units)
        # need conversion.
        wind_from_user = "wind_threshold" in last_step_input
        data.setdefault("wind_gust_factor", DEFAULT_WIND_GUST_FACTOR)
        data.setdefault("wind_threshold", DEFAULT_WIND_THRESHOLD)
        data.setdefault("extreme_wind_threshold", DEFAULT_EXTREME_WIND_THRESHOLD)

        # Convert wind thresholds from display unit to m/s for storage.
        # Only convert when values came from user input (display units).
        # setdefault values are already in m/s.
        unit = data.get(CONF_WIND_UNIT, DEFAULT_WIND_UNIT)
        if unit != WIND_UNIT_MS and wind_from_user:
            data["wind_threshold"] = convert_to_ms(data["wind_threshold"], unit)
            data["extreme_wind_threshold"] = convert_to_ms(data["extreme_wind_threshold"], unit)

        # Ensure thermal mass is always present (Track B field may not have been shown)
        data.setdefault(CONF_THERMAL_MASS, DEFAULT_THERMAL_MASS)

        # Sentinel-based defaults: distinguish "user never saw the field"
        # (key absent → fill with all energy sensors) from "user deselected
        # all" (key present with empty list → preserve, means zero entities
        # are affected).  Truthiness-based `not data.get(...)` would
        # clobber a legitimate empty selection for e.g. a building with
        # zero external screens or no aux-affected units.
        if CONF_AUX_AFFECTED_ENTITIES not in data:
            data[CONF_AUX_AFFECTED_ENTITIES] = data.get("energy_sensors", [])
        if CONF_SCREEN_AFFECTED_ENTITIES not in data:
            data[CONF_SCREEN_AFFECTED_ENTITIES] = data.get("energy_sensors", [])
        if CONF_SOLAR_AFFECTED_ENTITIES not in data:
            data[CONF_SOLAR_AFFECTED_ENTITIES] = data.get("energy_sensors", [])

        return data

    # ------------------------------------------------------------------ #
    # Schema builders                                                      #
    # ------------------------------------------------------------------ #

    def _solar_affected_default_with_new(self, g) -> list[str]:
        """Default for CONF_SOLAR_AFFECTED_ENTITIES with new-sensor opt-in (#962).

        Returns ``saved_solar_list ∪ (current_energy_sensors − previous_energy_sensors)``
        so that energy sensors added since the last save default to checked.
        Initial-config / fresh-install (no previous list) returns the full
        current energy_sensors list.

        ``g`` is the bound ``(key, default)`` lookup that the schema builder
        already uses for resolving form defaults.
        """
        current_energy = list(self._flow_data.get("energy_sensors", []))
        if self._entry is None:
            # Fresh install: all current sensors default-in.
            return list(g(CONF_SOLAR_AFFECTED_ENTITIES, current_energy))
        previous_energy = list(self._entry.data.get("energy_sensors", []))
        saved_solar = list(g(CONF_SOLAR_AFFECTED_ENTITIES, current_energy))
        new_sensors = [s for s in current_energy if s not in previous_energy]
        # Preserve current-energy ordering for the union; deduplicate.
        merged = list(dict.fromkeys(saved_solar + new_sensors))
        # Filter to current_energy so removed sensors don't linger in default.
        current_set = set(current_energy)
        return [s for s in merged if s in current_set]

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
        # Optional local GHI sensor (W/m²).  No device_class filter —
        # user-scraped weather-station values often expose plain numeric
        # state without the irradiance device_class set, so the broader
        # `domain="sensor"` selector accommodates them.  Read by
        # ``_get_ghi`` with [0, 1500] W/m² clamping.  Drives
        # ``ghi_signal_agreement`` diagnostic; pipeline integration is
        # gated on that diagnostic's evidence.
        schema[vol.Optional("ghi_sensor", description={"suggested_value": g("ghi_sensor")})] = (
            selector.EntitySelector(selector.EntitySelectorConfig(domain="sensor"))
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
            vol.Required("balance_point", default=g("balance_point", DEFAULT_BALANCE_POINT)): selector.NumberSelector(
                selector.NumberSelectorConfig(min=10, max=25, step=0.5, unit_of_measurement="°C")
            ),
            vol.Required(CONF_THERMAL_INERTIA, default=int(inertia)): selector.NumberSelector(
                selector.NumberSelectorConfig(min=1, max=24, step=1, unit_of_measurement="h", mode="slider")
            ),
        }
        return vol.Schema(schema)

    def _schema_advanced(self, user_input, defaults, *, include_experimental_4d: bool = False) -> vol.Schema:
        """Schema for toggles and miscellaneous options.

        All boolean feature toggles live here. Their corresponding sub-fields
        (sensors, paths, entry selectors, wind tuning) are on the *next* page
        (feature_config) so no dynamic re-render is ever needed.

        ``include_experimental_4d`` adds the ``CONF_EXPERIMENTAL_4D_PRIMARY``
        toggle **as the first field**, so it renders directly beneath the
        readiness verdict that ``async_step_reconfigure_advanced`` injects
        into the step description.  Only the reconfigure path opts in — the
        initial setup wizard never shows it.

        Field order is dict insertion order and is load-bearing for that one
        field.  For the rest of the page the ordering axis is expected
        interaction frequency, not consequence — see "Field ordering" in the
        module docstring before placing a new field.
        """
        g = lambda k, d=None: self._v(user_input, defaults, k, d)

        # Derive load-shift default from whether indoor_temp_sensor is already configured
        load_shift = g(_CONF_LOAD_SHIFT, bool(g(CONF_INDOOR_TEMP_SENSOR)))

        schema: dict = {}
        # Solar model route (#962, #1062).  FIRST field on the page, because
        # the readiness verdict it answers to is rendered in the step
        # description immediately above it — the two must read as one unit.
        # Previously this sat 17 fields down, so the verdict at the top of
        # the page referred to a control the user had to scroll to find.
        # Dict insertion order is the rendered field order, hence the
        # separate insert ahead of the literal below.
        # Reconfigure-only — the initial setup wizard never exposes this.
        if include_experimental_4d:
            schema[vol.Optional(
                CONF_EXPERIMENTAL_4D_PRIMARY,
                default=bool(g(CONF_EXPERIMENTAL_4D_PRIMARY, DEFAULT_EXPERIMENTAL_4D_PRIMARY)),
            )] = selector.BooleanSelector()
        schema.update({
            vol.Optional(CONF_DAILY_LEARNING_MODE, default=g(CONF_DAILY_LEARNING_MODE, False)): selector.BooleanSelector(),
            vol.Optional(_CONF_LOAD_SHIFT, default=bool(load_shift)): selector.BooleanSelector(),
            vol.Optional(CONF_TRACK_C, default=g(CONF_TRACK_C, False)): selector.BooleanSelector(),
            vol.Optional(
                CONF_AUX_AFFECTED_ENTITIES,
                default=g(CONF_AUX_AFFECTED_ENTITIES, self._flow_data.get("energy_sensors", [])),
            ): selector.EntitySelector(
                selector.EntitySelectorConfig(domain="sensor", device_class="energy", multiple=True)
            ),
            vol.Optional(CONF_ENABLE_LIFETIME_TRACKING, default=g(CONF_ENABLE_LIFETIME_TRACKING, False)): selector.BooleanSelector(),
            # Per-direction screen presence (#826).  Default True for all three so
            # behaviour matches pre-1.3.3 (single composite floor) on upgrade.
            # Uncheck a direction if that facade has no external screens — its
            # transmittance then stays at 1.0 regardless of the slider, and the
            # solar coefficient for that direction encodes pure window physics.
            vol.Optional(CONF_SCREEN_SOUTH, default=g(CONF_SCREEN_SOUTH, DEFAULT_SCREEN_SOUTH)): selector.BooleanSelector(),
            vol.Optional(CONF_SCREEN_EAST, default=g(CONF_SCREEN_EAST, DEFAULT_SCREEN_EAST)): selector.BooleanSelector(),
            vol.Optional(CONF_SCREEN_WEST, default=g(CONF_SCREEN_WEST, DEFAULT_SCREEN_WEST)): selector.BooleanSelector(),
            # Which energy sensors' solar coefficients learn/predict against the
            # screen_config above.  Default: all sensors (preserves prior
            # behaviour).  Uncheck a sensor if its zone has no screens — its
            # coefficients then learn against pure transmittance=1.0 rather
            # than absorbing an avg_transmittance it never physically sees.
            vol.Optional(
                CONF_SCREEN_AFFECTED_ENTITIES,
                default=g(CONF_SCREEN_AFFECTED_ENTITIES, self._flow_data.get("energy_sensors", [])),
            ): selector.EntitySelector(
                selector.EntitySelectorConfig(domain="sensor", device_class="energy", multiple=True)
            ),
            # Per-entity scope for solar coefficient learning + prediction (#962).
            # Uncheck a sensor if its consumption does NOT respond to solar gain
            # in the room — typically (a) the sensor is in a room with no sun
            # exposure, or (b) the sensor is a floor-heating cable controlled
            # by a slab/floor-temperature thermostat (the slab's thermal mass
            # dominates over solar gain, so the cable runs to maintain slab
            # setpoint regardless of sun in the room above).  Unchecked
            # sensors get a zero solar coefficient and skip all five solar
            # learning paths — base/aux learning continues normally.
            # Removing a sensor here triggers an automatic solar reset for
            # that sensor.
            #
            # New-sensor default-in: if the user added energy sensors since
            # the last reconfigure, those new sensors default to *checked*
            # (included in the solar list).  Rationale: solar response is the
            # common case — a new energy sensor for a new HP, new room,
            # etc. is almost always solar-affected.  Slab-thermostat / interior-
            # load exclusions are the minority.  User can immediately uncheck
            # if the new sensor is in fact non-solar.  This diverges from
            # screen_/aux_affected_entities which both default new sensors to
            # *un*checked — but solar is the case where the wrong default
            # injects phantom coefficients (#962), so the trade-off is
            # asymmetric and the inverse default is justified.
            vol.Optional(
                CONF_SOLAR_AFFECTED_ENTITIES,
                default=self._solar_affected_default_with_new(g),
            ): selector.EntitySelector(
                selector.EntitySelectorConfig(domain="sensor", device_class="energy", multiple=True)
            ),
            # Derive dedicated-wind default from whether a wind sensor is already configured
            vol.Optional(_CONF_DEDICATED_WIND, default=bool(g("wind_speed_sensor"))): selector.BooleanSelector(),
        })
        schema[vol.Optional(
            CONF_SECONDARY_WEATHER_ENTITY,
            description={"suggested_value": g(CONF_SECONDARY_WEATHER_ENTITY)},
        )] = selector.EntitySelector(selector.EntitySelectorConfig(domain="weather"))
        schema[vol.Optional(CONF_FORECAST_CROSSOVER_DAY, default=g(CONF_FORECAST_CROSSOVER_DAY, DEFAULT_FORECAST_CROSSOVER_DAY))] = (
            selector.NumberSelector(
                selector.NumberSelectorConfig(min=1, max=7, step=1, mode="slider")
            )
        )
        # Experimental hotspot-loss attenuation γ (#950).  Multiplicative
        # scale (1 − γ) applied to per-unit solar prediction for screened
        # facades at sun elevation > 30°.  γ = 0.0 is a no-op default.
        schema[vol.Optional(
            "solar_hotspot_attenuation_gamma",
            default=g("solar_hotspot_attenuation_gamma", 0.0),
        )] = selector.NumberSelector(
            selector.NumberSelectorConfig(min=0.0, max=1.0, step=0.05, mode="slider")
        )
        # Experimental tail-aware redistribution α / τ (#948).  Temporal
        # spreading of per-unit solar prediction at low-elev hours.
        # α = 0.0 (default) is a no-op.
        schema[vol.Optional(
            "solar_redistribution_alpha",
            default=g("solar_redistribution_alpha", 0.0),
        )] = selector.NumberSelector(
            selector.NumberSelectorConfig(min=0.0, max=1.0, step=0.05, mode="slider")
        )
        schema[vol.Optional(
            "solar_redistribution_tau_hours",
            default=g("solar_redistribution_tau_hours", 2.0),
        )] = selector.NumberSelector(
            selector.NumberSelectorConfig(
                min=0.5, max=6.0, step=0.5, unit_of_measurement="h", mode="slider"
            )
        )
        # (CONF_EXPERIMENTAL_4D_PRIMARY was here; it is now inserted first so
        # it sits directly under the readiness verdict in the step
        # description — see the top of this method.)
        # --- Mundane fields at the bottom (routine, not expert-only) ---
        schema[vol.Optional("csv_auto_logging", default=g("csv_auto_logging", DEFAULT_CSV_AUTO_LOGGING))] = selector.BooleanSelector()
        schema[vol.Optional("max_energy_delta", default=g("max_energy_delta", DEFAULT_MAX_ENERGY_DELTA))] = (
            selector.NumberSelector(
                selector.NumberSelectorConfig(min=0.5, max=15.0, step=0.5, unit_of_measurement="kWh")
            )
        )
        # SelectSelector compares defaults against string option values; the
        # stored retention value is an int (converted in _build_final_data),
        # so cast to str here or the form falls back to the first option on
        # reconfigure and a no-op submit would overwrite the user's choice.
        schema[vol.Optional(
            CONF_HOURLY_LOG_RETENTION_DAYS,
            default=str(g(CONF_HOURLY_LOG_RETENTION_DAYS, DEFAULT_HOURLY_LOG_RETENTION_DAYS)),
        )] = selector.SelectSelector(
            selector.SelectSelectorConfig(
                options=[
                    selector.SelectOptionDict(value=str(v), label=f"{v} days")
                    for v in HOURLY_LOG_RETENTION_OPTIONS
                ],
                mode=selector.SelectSelectorMode.DROPDOWN,
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

        # Dedicated wind sensors — hidden behind toggle (#798).
        dedicated_wind = self._flow_data.get(_CONF_DEDICATED_WIND, False)
        if dedicated_wind:
            for field, device_class in [
                ("wind_speed_sensor", "wind_speed"),
                ("wind_gust_sensor", "wind_speed"),
            ]:
                schema[vol.Optional(field, description={"suggested_value": g(field)})] = (
                    selector.EntitySelector(
                        selector.EntitySelectorConfig(domain="sensor", device_class=device_class)
                    )
                )

        # Wind tuning — always visible (relevant regardless of wind source).
        current_unit = self._flow_data.get(CONF_WIND_UNIT, DEFAULT_WIND_UNIT)
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
        max_extreme = {WIND_UNIT_KMH: 180.0, WIND_UNIT_KNOTS: 90.0}.get(current_unit, 50.0)

        schema[vol.Required("wind_gust_factor", default=g("wind_gust_factor", DEFAULT_WIND_GUST_FACTOR))] = (
            selector.NumberSelector(
                selector.NumberSelectorConfig(min=0.0, max=1.0, step=0.05, mode="slider")
            )
        )
        schema[vol.Required("wind_threshold", default=round(display_threshold, 1))] = (
            selector.NumberSelector(
                selector.NumberSelectorConfig(min=0.0, max=max_wind, step=0.1, unit_of_measurement=current_unit)
            )
        )
        schema[vol.Required("extreme_wind_threshold", default=round(display_extreme, 1))] = (
            selector.NumberSelector(
                selector.NumberSelectorConfig(min=0.0, max=max_extreme, step=0.1, unit_of_measurement=current_unit)
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
                self._clear_absent_entity_keys(user_input, ["outdoor_temp_sensor", "ghi_sensor"])
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
                       "csv_auto_logging", CONF_ENABLE_LIFETIME_TRACKING, _CONF_DEDICATED_WIND):
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
                self._clear_absent_entity_keys(user_input, ["outdoor_temp_sensor", "ghi_sensor"])
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
                       "csv_auto_logging", CONF_ENABLE_LIFETIME_TRACKING, _CONF_DEDICATED_WIND,
                       CONF_EXPERIMENTAL_4D_PRIMARY):
                self._flow_data[_k] = bool(user_input.get(_k, False))
            # Run aux migration here — CONF_AUX_AFFECTED_ENTITIES is on this page
            # and _flow_data now has the new value regardless of whether the
            # feature_config page is shown next or skipped entirely.
            new_aux = user_input.get(CONF_AUX_AFFECTED_ENTITIES)
            if new_aux is not None:
                coord = self.hass.data.get(DOMAIN, {}).get(self.context["entry_id"])
                if coord:
                    await coord.async_migrate_aux_coefficients(new_aux)
            # Solar-affected migration (#962): reset solar learning for any
            # entity newly removed from the list.  No conservation strategy
            # — solar coefficients are not redistributable (each represents
            # one unit's window physics, not a divisible house-aggregate).
            new_solar = user_input.get(CONF_SOLAR_AFFECTED_ENTITIES)
            if new_solar is not None:
                coord = self.hass.data.get(DOMAIN, {}).get(self.context["entry_id"])
                if coord:
                    await coord.async_migrate_solar_affected(new_solar)
            if self._needs_feature_config_step():
                return await self.async_step_reconfigure_feature_config()
            return self.async_update_reload_and_abort(
                self._entry, data=self._build_final_data({})
            )
        return self.async_show_form(
            step_id="reconfigure_advanced",
            data_schema=self._schema_advanced(None, self._flow_data, include_experimental_4d=True),
            # 4D readiness at the point of decision (#1062).  Warn, don't
            # refuse — a user who just connected a GHI sensor is correct to
            # enable 4D before the 30-day source window catches up.
            description_placeholders={
                "four_d_readiness": self._four_d_readiness_placeholder(),
            },
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
