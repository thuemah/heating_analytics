"""The Heating Analytics integration."""
from __future__ import annotations

import logging
from datetime import timedelta
import voluptuous as vol

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform, EVENT_HOMEASSISTANT_STOP
from homeassistant.core import HomeAssistant, ServiceCall, Event, SupportsResponse
from homeassistant.exceptions import ConfigEntryNotReady, ServiceValidationError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers import entity_registry as er
from homeassistant.util import dt as dt_util

from .const import DOMAIN
from .coordinator import HeatingDataCoordinator

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = [
    Platform.SENSOR,
    Platform.NUMBER,
    Platform.SWITCH,
    Platform.SELECT
]

SERVICE_IMPORT_CSV = "import_from_csv"
SERVICE_EXPORT_CSV = "export_to_csv"
SERVICE_RESET_LEARNING = "reset_learning_data"
SERVICE_RESET_UNIT_LEARNING = "reset_unit_learning_data"
SERVICE_RESET_FORECAST_ACCURACY = "reset_forecast_accuracy"
SERVICE_BACKUP_DATA = "backup_data"
SERVICE_RESTORE_DATA = "restore_data"
SERVICE_REPLACE_SENSOR = "replace_sensor_source"
SERVICE_COMPARE_PERIODS = "compare_periods"
SERVICE_EXIT_COOLDOWN = "exit_cooldown"
SERVICE_GET_FORECAST = "get_forecast"
SERVICE_CALIBRATE_INERTIA = "calibrate_inertia"
SERVICE_CALIBRATE_WIND_THRESHOLDS = "calibrate_wind_thresholds"
SERVICE_CALIBRATE_UNIT_THRESHOLDS = "calibrate_unit_thresholds"
SERVICE_DIAGNOSE_MODEL = "diagnose_model"
SERVICE_DIAGNOSE_SOLAR = "diagnose_solar"
SERVICE_RESET_SOLAR_LEARNING = "reset_solar_learning"
SERVICE_RETRAIN_FROM_HISTORY = "retrain_from_history"
SERVICE_BATCH_FIT_SOLAR = "batch_fit_solar"
SERVICE_APPLY_IMPLIED_COEFFICIENT = "apply_implied_coefficient"
SERVICE_SET_EXPERIMENTAL_TOBIT_LIVE_LEARNER = "set_experimental_tobit_live_learner"
SERVICE_SET_TOBIT_LIVE_ENTITIES = "set_tobit_live_entities"
SERVICE_RESET_TOBIT_LIVE_STATE = "reset_tobit_live_state"
SERVICE_SCHEMA_CALIBRATE_INERTIA = vol.Schema({
    vol.Optional("entity_id"): cv.entity_id,
    vol.Optional("days", default=30): vol.All(vol.Coerce(int), vol.Range(min=1, max=90)),
    vol.Optional("centered_energy_average", default=False): cv.boolean,
    vol.Optional("test_asymmetric", default=False): cv.boolean,
    vol.Optional("test_delta_t_scaling", default=False): cv.boolean,
    vol.Optional("test_exponential_kernel", default=False): cv.boolean,
})

SERVICE_SCHEMA_CALIBRATE_WIND = vol.Schema({
    vol.Optional("entity_id"): cv.entity_id,
    vol.Optional("days", default=60): vol.All(vol.Coerce(int), vol.Range(min=1, max=180)),
})

SERVICE_SCHEMA_CALIBRATE_UNIT_THRESHOLDS = vol.Schema({
    vol.Optional("entity_id"): cv.entity_id,
    vol.Optional("days", default=30): vol.All(vol.Coerce(int), vol.Range(min=7, max=180)),
})

SERVICE_SCHEMA_IMPORT = vol.Schema({
    vol.Optional("entity_id"): cv.entity_id,
    vol.Required("file_path"): cv.string,
    vol.Optional("update_model", default=True): cv.boolean,
    vol.Required("column_mapping"): vol.Schema({
        vol.Required("timestamp"): cv.string,
        vol.Optional("energy"): cv.string,
        vol.Optional("temperature"): cv.string,
        vol.Optional("wind_speed"): cv.string,
        vol.Optional("wind_gust"): cv.string,
        vol.Optional("cloud_coverage"): cv.string,
        vol.Optional("is_auxiliary"): cv.string,
    })
})

SERVICE_SCHEMA_EXPORT = vol.Schema({
    vol.Optional("entity_id"): cv.entity_id,
    vol.Required("file_path"): cv.string,
    vol.Required("export_type", default="daily"): vol.In(["daily", "hourly"]),
})

SERVICE_SCHEMA_RESET = vol.Schema({
    vol.Optional("entity_id"): cv.entity_id,
})

SERVICE_SCHEMA_RESET_UNIT = vol.Schema({
    vol.Required("entity_id"): cv.entity_id,
})

SERVICE_SCHEMA_RESET_ACCURACY = vol.Schema({
    vol.Optional("entity_id"): cv.entity_id,
})

SERVICE_SCHEMA_RESET_SOLAR = vol.Schema({
    vol.Optional("entity_id"): cv.entity_id,  # Instance targeting
    vol.Optional("unit_entity_id"): cv.entity_id,  # Unit filter (which sensor to reset)
    vol.Optional("replay_from_history", default=False): cv.boolean,
    vol.Optional("days_back"): vol.All(vol.Coerce(int), vol.Range(min=1, max=730)),
})

SERVICE_SCHEMA_BACKUP = vol.Schema({
    vol.Optional("entity_id"): cv.entity_id,
    vol.Required("file_path"): cv.string,
})

SERVICE_SCHEMA_RESTORE = vol.Schema({
    vol.Optional("entity_id"): cv.entity_id,
    vol.Required("file_path"): cv.string,
})

SERVICE_SCHEMA_REPLACE_SENSOR = vol.Schema({
    vol.Optional("entity_id"): cv.entity_id,
    vol.Required("old_entity_id"): cv.entity_id,
    vol.Required("new_entity_id"): cv.entity_id,
})

SERVICE_SCHEMA_RETRAIN = vol.Schema({
    vol.Optional("entity_id"): cv.entity_id,
    vol.Optional("days_back"): vol.All(vol.Coerce(int), vol.Range(min=1, max=730)),
    vol.Optional("reset_first", default=False): cv.boolean,
    vol.Optional("experimental_cop_smear", default=False): cv.boolean,  # #793 hidden flag
})

SERVICE_SCHEMA_BATCH_FIT_SOLAR = vol.Schema({
    vol.Optional("entity_id"): cv.entity_id,
    vol.Optional("unit_entity_id"): cv.entity_id,
    vol.Optional("days_back", default=30): vol.All(
        vol.Coerce(int), vol.Range(min=1, max=730)
    ),
    vol.Optional("dry_run", default=False): cv.boolean,
    vol.Optional("seed_live_window", default=False): cv.boolean,
})

SERVICE_SCHEMA_APPLY_IMPLIED_COEFFICIENT = vol.Schema({
    vol.Optional("entity_id"): cv.entity_id,
    vol.Required("unit_entity_id"): cv.entity_id,
    # Guest modes are rejected at the schema layer: guest hours are
    # filtered out of ``_collect_batch_fit_samples`` (mode != target_mode),
    # so picking guest_heating / guest_cooling would always return
    # ``no_data``.  The coordinator method still accepts them as
    # synonyms for the underlying regime — they're routed via
    # _solar_coeff_regime — but the service-level dropdown should not
    # advertise an option that's functionally inert.
    vol.Optional("mode", default="heating"): vol.In(("heating", "cooling")),
    # ``min: 7`` is deliberate: stability windows split samples
    # chronologically into 3 chunks; with ``days_back < 7`` all three
    # chunks fall within the same solar phase (e.g. one sunny
    # afternoon), and the per-direction stability check would pass
    # within-afternoon noise instead of catching real temporal drift.
    # 7 gives ~3 chunks of 2-day spans — the minimum useful temporal
    # resolution for the guard.  Users wanting smaller windows must
    # set the coefficient manually via reset_solar_learning + custom
    # state.
    vol.Optional("days_back", default=30): vol.All(
        vol.Coerce(int), vol.Range(min=7, max=730)
    ),
    vol.Optional("dry_run", default=False): cv.boolean,
    vol.Optional("force", default=False): cv.boolean,
})

# Tobit live-learner experimental services (#904 stage 3, storage v5).
# All three persist immediately and trigger _async_save_data so settings
# survive restarts.  The ``confirm: true`` requirement on enable guards
# against accidental activation via copy-paste — the gate matches the
# storage-level "no UI" disposition (Alternative B in #912 design).
SERVICE_SCHEMA_SET_EXPERIMENTAL_TOBIT_LIVE_LEARNER = vol.Schema({
    vol.Optional("entity_id"): cv.entity_id,
    vol.Required("enabled"): cv.boolean,
    vol.Optional("confirm", default=False): cv.boolean,
})

SERVICE_SCHEMA_SET_TOBIT_LIVE_ENTITIES = vol.Schema({
    vol.Optional("entity_id"): cv.entity_id,
    vol.Required("entities"): vol.All(cv.ensure_list, [cv.entity_id]),
})

SERVICE_SCHEMA_RESET_TOBIT_LIVE_STATE = vol.Schema({
    vol.Optional("entity_id"): cv.entity_id,
    vol.Optional("unit_entity_id"): cv.entity_id,
})

SERVICE_SCHEMA_COMPARE_PERIODS = vol.Schema({
    vol.Optional("entity_id"): cv.entity_id,
    vol.Required("period_1_start"): cv.date,
    vol.Required("period_1_end"): cv.date,
    vol.Required("period_2_start"): cv.date,
    vol.Required("period_2_end"): cv.date,
})

SERVICE_SCHEMA_GET_FORECAST = vol.Schema({
    vol.Optional("entity_id"): cv.entity_id,
    vol.Optional("days", default=1): vol.All(vol.Coerce(int), vol.Range(min=1, max=14)),
    vol.Optional("isolate_sensor"): cv.entity_id,
})

SERVICE_SCHEMA_DIAGNOSE = vol.Schema({
    vol.Optional("entity_id"): cv.entity_id,
    vol.Optional("days", default=30): vol.All(vol.Coerce(int), vol.Range(min=1, max=365)),
})

def _get_coordinators(hass: HomeAssistant) -> list[HeatingDataCoordinator]:
    """Helper to get all active HeatingDataCoordinators."""
    return [
        coord
        for coord in hass.data.get(DOMAIN, {}).values()
        if isinstance(coord, HeatingDataCoordinator)
    ]

def _get_target_coordinator(
    hass: HomeAssistant, entity_id: str | None
) -> HeatingDataCoordinator:
    """Return the coordinator for entity_id, or the first available one.

    When entity_id is explicitly provided but lookup fails, raises
    ValueError instead of silently falling back to an arbitrary instance.
    When entity_id is None, returns the first available coordinator.
    """
    if entity_id:
        registry = er.async_get(hass)
        entry = registry.async_get(entity_id)
        if entry and entry.config_entry_id:
            coord = hass.data[DOMAIN].get(entry.config_entry_id)
            if coord:
                return coord
        raise ValueError(f"Could not find Heating Analytics instance for entity '{entity_id}'.")
    coordinators = _get_coordinators(hass)
    if coordinators:
        return coordinators[0]
    raise ValueError("No Heating Analytics instance found.")

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Heating Analytics from a config entry."""
    
    hass.data.setdefault(DOMAIN, {})
    
    coordinator = HeatingDataCoordinator(hass, entry)
    try:
        await coordinator.async_config_entry_first_refresh()
    except Exception as ex:  # noqa: BLE001 — HA config-entry setup boundary; any failure maps to ConfigEntryNotReady
        raise ConfigEntryNotReady(f"Timeout while waiting for initial data: {ex}") from ex

    hass.data[DOMAIN][entry.entry_id] = coordinator

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    # Register listener for Home Assistant Stop
    async def async_handle_stop(event: Event) -> None:
        """Handle Home Assistant stop event."""
        _LOGGER.info("Home Assistant stopping, saving Heating Analytics data.")
        await coordinator._async_save_data(force=True)

    entry.async_on_unload(
        hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STOP, async_handle_stop)
    )

    # Register Import Service
    async def handle_import_csv(call: ServiceCall):
        """Handle the CSV import service call."""
        entity_id = call.data.get("entity_id")
        file_path = call.data.get("file_path")
        mapping = call.data.get("column_mapping")
        update_model = call.data.get("update_model", True)

        _LOGGER.info(f"Service called to import CSV from {file_path} (Update Model: {update_model})")

        coord = _get_target_coordinator(hass, entity_id)
        await coord.import_csv_data(file_path, mapping, update_model)

    hass.services.async_register(
        DOMAIN,
        SERVICE_IMPORT_CSV,
        handle_import_csv,
        schema=SERVICE_SCHEMA_IMPORT
    )

    # Register Export Service
    async def handle_export_csv(call: ServiceCall):
        """Handle the CSV export service call."""
        entity_id = call.data.get("entity_id")
        file_path = call.data.get("file_path")
        export_type = call.data.get("export_type")

        _LOGGER.info(f"Service called to export CSV ({export_type}) to {file_path}")

        coord = _get_target_coordinator(hass, entity_id)
        await coord.export_csv_data(file_path, export_type)

    hass.services.async_register(
        DOMAIN,
        SERVICE_EXPORT_CSV,
        handle_export_csv,
        schema=SERVICE_SCHEMA_EXPORT
    )

    # Register Reset Service
    async def handle_reset_learning(call: ServiceCall):
        """Handle the reset learning data service call."""
        entity_id = call.data.get("entity_id")
        coord = _get_target_coordinator(hass, entity_id)
        _LOGGER.info("Service called to reset learning data (coordinator: %s).", coord.entry.entry_id)
        await coord.async_reset_learning_data()

    hass.services.async_register(
        DOMAIN,
        SERVICE_RESET_LEARNING,
        handle_reset_learning,
        schema=SERVICE_SCHEMA_RESET
    )

    # Register Reset Unit Learning Service
    async def handle_reset_unit_learning(call: ServiceCall):
        """Handle the reset unit learning data service call."""
        entity_id = call.data.get("entity_id")
        _LOGGER.info(f"Service called to reset learning data for {entity_id}")

        for coord in _get_coordinators(hass):
            await coord.async_reset_unit_learning_data(entity_id)

    hass.services.async_register(
        DOMAIN,
        SERVICE_RESET_UNIT_LEARNING,
        handle_reset_unit_learning,
        schema=SERVICE_SCHEMA_RESET_UNIT
    )

    # Register Reset Forecast Accuracy Service
    async def handle_reset_forecast_accuracy(call: ServiceCall):
        """Handle the reset forecast accuracy service call."""
        entity_id = call.data.get("entity_id")
        coord = _get_target_coordinator(hass, entity_id)
        _LOGGER.info("Service called to reset forecast accuracy (coordinator: %s).", coord.entry.entry_id)
        await coord.async_reset_forecast_accuracy()

    hass.services.async_register(
        DOMAIN,
        SERVICE_RESET_FORECAST_ACCURACY,
        handle_reset_forecast_accuracy,
        schema=SERVICE_SCHEMA_RESET_ACCURACY
    )

    # Register Reset Solar Learning Service
    async def handle_reset_solar_learning(call: ServiceCall) -> dict:
        """Handle the reset solar learning service call."""
        entity_id = call.data.get("entity_id")
        unit_entity_id = call.data.get("unit_entity_id")
        replay_from_history = call.data.get("replay_from_history", False)
        days_back = call.data.get("days_back")

        coord = _get_target_coordinator(hass, entity_id)
        scope = f"unit {unit_entity_id}" if unit_entity_id else "all units"
        replay_suffix = (
            f" + replay from history (days_back={days_back})"
            if replay_from_history else ""
        )
        _LOGGER.info(
            f"Service called: reset_solar_learning for {scope}{replay_suffix} "
            f"(coordinator={coord.entry.entry_id})"
        )
        return await coord.async_reset_solar_learning_data(
            unit_entity_id,
            replay_from_history=replay_from_history,
            days_back=days_back,
        )

    hass.services.async_register(
        DOMAIN,
        SERVICE_RESET_SOLAR_LEARNING,
        handle_reset_solar_learning,
        schema=SERVICE_SCHEMA_RESET_SOLAR,
        supports_response=SupportsResponse.ONLY,
    )

    # Register Retrain From History Service
    async def handle_retrain_from_history(call: ServiceCall) -> dict:
        """Handle the retrain from history service call."""
        entity_id = call.data.get("entity_id")
        days_back = call.data.get("days_back")
        reset_first = call.data.get("reset_first", False)
        experimental_cop_smear = call.data.get("experimental_cop_smear", False)

        coord = _get_target_coordinator(hass, entity_id)
        _LOGGER.info(
            f"Service called: retrain_from_history (days_back={days_back}, reset_first={reset_first}, "
            f"experimental_cop_smear={experimental_cop_smear}, coordinator={coord.entry.entry_id})"
        )
        return await coord.retrain_from_history(
            days_back=days_back, reset_first=reset_first,
            experimental_cop_smear=experimental_cop_smear,
        )

    hass.services.async_register(
        DOMAIN,
        SERVICE_RETRAIN_FROM_HISTORY,
        handle_retrain_from_history,
        schema=SERVICE_SCHEMA_RETRAIN,
        supports_response=SupportsResponse.ONLY,
    )

    # Register Batch-Fit Solar Service (#884)
    async def handle_batch_fit_solar(call: ServiceCall) -> dict:
        """Handle the batch-fit-solar service call.

        On-demand periodic batch least-squares fit per (entity, mode)
        regime over the modulating-regime hourly log.  Bridges the
        mild-weather catch-22 where NLMS / inequality both produce zero
        signal because expected base demand is near zero (e.g. west sun
        peaks at the warmest part of the day).

        ``days_back`` defaults to 30 at the service boundary (a fresh
        fit shortly after a release should not absorb pre-upgrade data).
        ``dry_run`` defaults to False; when True the service returns the
        would-be coefficients without writing them.
        """
        entity_id = call.data.get("entity_id")
        unit_entity_id = call.data.get("unit_entity_id")
        # voluptuous defaults guarantee these keys exist post-schema.
        days_back = call.data.get("days_back", 30)
        dry_run = call.data.get("dry_run", False)
        seed_live_window = call.data.get("seed_live_window", False)
        coord = _get_target_coordinator(hass, entity_id)
        scope = f"unit {unit_entity_id}" if unit_entity_id else "all units"
        suffix_parts = [f"last {days_back}d"]
        if dry_run:
            suffix_parts.append("dry-run")
        if seed_live_window:
            suffix_parts.append("seed-live-window")
        suffix = f" ({', '.join(suffix_parts)})"
        _LOGGER.info(
            f"Service called: batch_fit_solar for {scope}{suffix} "
            f"(coordinator={coord.entry.entry_id})"
        )
        return await coord.async_batch_fit_solar(
            entity_id=unit_entity_id,
            days_back=days_back,
            dry_run=dry_run,
            seed_live_window=seed_live_window,
        )

    hass.services.async_register(
        DOMAIN,
        SERVICE_BATCH_FIT_SOLAR,
        handle_batch_fit_solar,
        schema=SERVICE_SCHEMA_BATCH_FIT_SOLAR,
        supports_response=SupportsResponse.ONLY,
    )

    # Register Apply Implied Coefficient Service (#884 follow-up)
    async def handle_apply_implied_coefficient(call: ServiceCall) -> dict:
        """Handle the apply-implied-coefficient service call.

        Writes ``diagnose_solar``'s implied LS-fit into the live
        coefficient for one (unit, mode regime), with per-direction
        stability guards.  ``dry_run=true`` shows what would be
        applied without writing.  ``force=true`` overrides the
        per-direction stability guard (use after manually verifying
        the implied is trustworthy on a noisy install).  ``days_back``
        defaults to 30 — fitting only on recent data avoids
        contamination from pre-retrain or pre-upgrade entries.
        """
        entity_id = call.data.get("entity_id")
        unit_entity_id = call.data["unit_entity_id"]  # Required
        mode = call.data.get("mode", "heating")
        days_back = call.data.get("days_back", 30)
        dry_run = call.data.get("dry_run", False)
        force = call.data.get("force", False)
        coord = _get_target_coordinator(hass, entity_id)
        # Conditional ``last Nd`` mirrors the coordinator-side log line
        # — currently the schema default forces a non-None value here,
        # but the conditional keeps the log readable if the schema
        # default is ever changed to None.
        suffix = (
            f" [{mode}]"
            f"{f', last {days_back}d' if days_back is not None else ''}"
            f"{', dry-run' if dry_run else ''}"
            f"{', forced' if force else ''}"
        )
        _LOGGER.info(
            f"Service called: apply_implied_coefficient for {unit_entity_id}{suffix} "
            f"(coordinator={coord.entry.entry_id})"
        )
        return await coord.async_apply_implied_coefficient(
            entity_id=unit_entity_id,
            mode=mode,
            dry_run=dry_run,
            force=force,
            days_back=days_back,
        )

    hass.services.async_register(
        DOMAIN,
        SERVICE_APPLY_IMPLIED_COEFFICIENT,
        handle_apply_implied_coefficient,
        schema=SERVICE_SCHEMA_APPLY_IMPLIED_COEFFICIENT,
        supports_response=SupportsResponse.ONLY,
    )

    # ------------------------------------------------------------------
    # Tobit live-learner experimental services (#904 stage 3, storage v5)
    # ------------------------------------------------------------------
    async def handle_set_experimental_tobit_live_learner(call: ServiceCall):
        """Toggle the master flag for the Tobit live-learner.

        Default is True on 1.3.5+ via the load-time marker; this
        handler exists so users can explicitly disable.  Disable
        persists across restart because the marker is stamped both
        at load-time and here — no path leaves it unset post-1.3.5.
        ``confirm: true`` is required when enabling to guard against
        accidental activation via copy-paste from documentation
        snippets.
        """
        entity_id = call.data.get("entity_id")
        enabled = bool(call.data["enabled"])
        confirm = bool(call.data.get("confirm", False))
        if enabled and not confirm:
            # ServiceValidationError surfaces the message in HA's "Call
            # Service" UI; raw ValueError was reported as a generic
            # error and the rationale text never reached the user.
            raise ServiceValidationError(
                "set_experimental_tobit_live_learner: pass confirm=true "
                "alongside enabled=true to acknowledge that the Tobit "
                "learner replaces NLMS as primary writer for "
                "plausibility-passing entities."
            )
        coord = _get_target_coordinator(hass, entity_id)
        coord._experimental_tobit_live_learner = enabled
        # Stamp the marker — guarantees this explicit user action
        # commits regardless of whether the load-time flip happened
        # to land between save cycles.  Belt-and-braces against the
        # race where the user disables before async_load_data
        # completes (handler set False, load reads missing-marker,
        # load flips back to True): with the stamp here, a load
        # following a service-disable observes marker=True and
        # leaves the False intact.
        coord._tobit_default_applied = True
        scope_count = len(getattr(coord, "_tobit_live_entities", frozenset()))
        scope_desc = (
            "auto-mode (plausibility-gate decides per entity)"
            if scope_count == 0
            else f"scope-restricted to {scope_count} entities"
        )
        _LOGGER.info(
            "Tobit live-learner master flag set to %s — %s",
            enabled, scope_desc,
        )
        await coord._async_save_data(force=True)

    hass.services.async_register(
        DOMAIN,
        SERVICE_SET_EXPERIMENTAL_TOBIT_LIVE_LEARNER,
        handle_set_experimental_tobit_live_learner,
        schema=SERVICE_SCHEMA_SET_EXPERIMENTAL_TOBIT_LIVE_LEARNER,
    )

    async def handle_set_tobit_live_entities(call: ServiceCall):
        """Set the optional scope-override list for the Tobit live path.

        Replaces the full list (not additive).  Empty list = auto-mode
        (every eligible entity is candidate; plausibility-gate decides
        per entity whether Tobit writes — noise loads filtered, real
        VPs pass).  Non-empty list = scope-restrict to listed entities;
        others stay on NLMS.  The plausibility-gate still applies
        within the scope — explicit allow-listing does not bypass the
        noise-load filter.  Stamps the marker for parity with the flag
        handler.
        """
        entity_id = call.data.get("entity_id")
        entities = list(call.data["entities"])
        coord = _get_target_coordinator(hass, entity_id)
        coord._tobit_live_entities = frozenset(entities)
        coord._tobit_default_applied = True
        _LOGGER.info(
            "Tobit live-learner scope updated: %s",
            ", ".join(sorted(entities)) if entities
            else "auto-mode (empty list — plausibility-gate decides per entity)",
        )
        await coord._async_save_data(force=True)

    hass.services.async_register(
        DOMAIN,
        SERVICE_SET_TOBIT_LIVE_ENTITIES,
        handle_set_tobit_live_entities,
        schema=SERVICE_SCHEMA_SET_TOBIT_LIVE_ENTITIES,
    )

    async def handle_reset_tobit_live_state(call: ServiceCall):
        """Clear the running sufficient-statistic for one or all entities.

        Without ``unit_entity_id``: clears state for all entities.
        With ``unit_entity_id``: clears state for that single entity
        only (both regimes).  Coefficient values in
        ``solar_coefficients_per_unit`` are NOT touched — use
        ``reset_solar_learning`` for that.  After this call the live
        learner enters cold-start (NLMS-fallback fires until n_eff
        ≥ TOBIT_MIN_NEFF rebuilds).
        """
        entity_id = call.data.get("entity_id")
        unit_entity_id = call.data.get("unit_entity_id")
        coord = _get_target_coordinator(hass, entity_id)
        if unit_entity_id is None:
            cleared = len(coord._tobit_sufficient_stats)
            coord._tobit_sufficient_stats = {}
            coord._nlms_shadow_coefficients = {}
            _LOGGER.info(
                "Tobit live-learner state cleared for %d entities (cold-start)",
                cleared,
            )
        else:
            coord._tobit_sufficient_stats.pop(unit_entity_id, None)
            coord._nlms_shadow_coefficients.pop(unit_entity_id, None)
            _LOGGER.info(
                "Tobit live-learner state cleared for %s (cold-start)",
                unit_entity_id,
            )
        await coord._async_save_data(force=True)

    hass.services.async_register(
        DOMAIN,
        SERVICE_RESET_TOBIT_LIVE_STATE,
        handle_reset_tobit_live_state,
        schema=SERVICE_SCHEMA_RESET_TOBIT_LIVE_STATE,
    )

    # Register Backup Service
    async def handle_backup_data(call: ServiceCall):
        """Handle the backup data service call."""
        entity_id = call.data.get("entity_id")
        file_path = call.data.get("file_path")
        _LOGGER.info(f"Service called to backup data to {file_path}")

        coord = _get_target_coordinator(hass, entity_id)
        await coord.async_backup_data(file_path)

    hass.services.async_register(
        DOMAIN,
        SERVICE_BACKUP_DATA,
        handle_backup_data,
        schema=SERVICE_SCHEMA_BACKUP
    )

    # Register Restore Service
    async def handle_restore_data(call: ServiceCall):
        """Handle the restore data service call."""
        entity_id = call.data.get("entity_id")
        file_path = call.data.get("file_path")
        _LOGGER.info(f"Service called to restore data from {file_path}")

        coord = _get_target_coordinator(hass, entity_id)
        await coord.async_restore_data(file_path)

    hass.services.async_register(
        DOMAIN,
        SERVICE_RESTORE_DATA,
        handle_restore_data,
        schema=SERVICE_SCHEMA_RESTORE
    )

    # Register Replace Sensor Service
    async def handle_replace_sensor(call: ServiceCall):
        """Handle the replace sensor source service call."""
        entity_id = call.data.get("entity_id")
        old_id = call.data.get("old_entity_id")
        new_id = call.data.get("new_entity_id")

        if entity_id:
            # Target a specific instance.
            coord = _get_target_coordinator(hass, entity_id)
            _LOGGER.info(f"Service called to replace sensor: {old_id} -> {new_id} (coordinator={coord.entry.entry_id})")
            if await coord.async_replace_sensor_source(old_id, new_id):
                _LOGGER.info(f"Reloading entry {coord.entry.entry_id} to apply sensor replacement.")
                await hass.config_entries.async_reload(coord.entry.entry_id)
        else:
            # Legacy: try all instances.
            entries_to_reload = []
            _LOGGER.info(f"Service called to replace sensor: {old_id} -> {new_id} (all instances)")
            for coord in _get_coordinators(hass):
                if await coord.async_replace_sensor_source(old_id, new_id):
                    entries_to_reload.append(coord.entry.entry_id)
            for entry_id in entries_to_reload:
                _LOGGER.info(f"Reloading entry {entry_id} to apply sensor replacement.")
                await hass.config_entries.async_reload(entry_id)

    hass.services.async_register(
        DOMAIN,
        SERVICE_REPLACE_SENSOR,
        handle_replace_sensor,
        schema=SERVICE_SCHEMA_REPLACE_SENSOR
    )

    # Register Compare Periods Service
    async def handle_compare_periods(call: ServiceCall):
        """Handle the compare periods service call."""
        entity_id = call.data.get("entity_id")
        p1_start = call.data.get("period_1_start")
        p1_end = call.data.get("period_1_end")
        p2_start = call.data.get("period_2_start")
        p2_end = call.data.get("period_2_end")

        _LOGGER.info(f"Service called to compare periods: {p1_start}-{p1_end} vs {p2_start}-{p2_end}")

        coord = _get_target_coordinator(hass, entity_id)
        await coord.async_compare_periods(p1_start, p1_end, p2_start, p2_end)

    hass.services.async_register(
        DOMAIN,
        SERVICE_COMPARE_PERIODS,
        handle_compare_periods,
        schema=SERVICE_SCHEMA_COMPARE_PERIODS,
    )

    # Register Exit Cooldown Service
    async def handle_exit_cooldown(call: ServiceCall):
        """Handle the exit cooldown service call."""
        entity_id = call.data.get("entity_id")
        _LOGGER.info("Service called to exit auxiliary cooldown.")

        coord = _get_target_coordinator(hass, entity_id)
        await coord.async_exit_cooldown()

    hass.services.async_register(
        DOMAIN,
        SERVICE_EXIT_COOLDOWN,
        handle_exit_cooldown,
        schema=vol.Schema({vol.Optional("entity_id"): cv.entity_id}),
    )

    # Register Calibrate Wind Thresholds Service
    async def handle_calibrate_wind_thresholds(call: ServiceCall) -> dict:
        """Handle the calibrate wind thresholds service call."""
        entity_id = call.data.get("entity_id")
        days = call.data.get("days", 60)

        target_coordinator = _get_target_coordinator(hass, entity_id)
        _LOGGER.debug("Handling calibrate_wind_thresholds for %d days (coordinator: %s)", days, target_coordinator.entry.entry_id)
        result = target_coordinator.statistics.calibrate_wind_thresholds(days=days)
        return result

    hass.services.async_register(
        DOMAIN,
        SERVICE_CALIBRATE_WIND_THRESHOLDS,
        handle_calibrate_wind_thresholds,
        schema=SERVICE_SCHEMA_CALIBRATE_WIND,
        supports_response=SupportsResponse.ONLY,
    )

    # Register Calibrate Unit Thresholds Service
    async def handle_calibrate_unit_thresholds(call: ServiceCall) -> dict:
        """Handle the calibrate unit thresholds service call.

        Recomputes per-unit min-base thresholds from dark-hour actuals.
        Safe to call anytime; thresholds update in-place and are persisted
        on the next storage save.  Returns the same diagnostic dict that
        startup calibration logs.
        """
        entity_id = call.data.get("entity_id")
        days = call.data.get("days", 30)

        target_coordinator = _get_target_coordinator(hass, entity_id)
        _LOGGER.debug(
            "Handling calibrate_unit_thresholds for %d days (coordinator: %s)",
            days, target_coordinator.entry.entry_id,
        )
        result = target_coordinator._calibrate_per_unit_min_base_thresholds(
            sample_days=days,
        )
        if result.get("status") == "ok" and (
            result.get("updated") or result.get("rejected")
        ):
            await target_coordinator._async_save_data(force=True)
        return result

    hass.services.async_register(
        DOMAIN,
        SERVICE_CALIBRATE_UNIT_THRESHOLDS,
        handle_calibrate_unit_thresholds,
        schema=SERVICE_SCHEMA_CALIBRATE_UNIT_THRESHOLDS,
        supports_response=SupportsResponse.ONLY,
    )

    # Register Calibrate Inertia Service
    async def handle_calibrate_inertia(call: ServiceCall) -> dict:
        """Handle the calibrate inertia service call."""
        entity_id = call.data.get("entity_id")
        days = call.data.get("days", 30)
        centered = call.data.get("centered_energy_average", False)
        test_asymmetric = call.data.get("test_asymmetric", False)
        test_delta_t_scaling = call.data.get("test_delta_t_scaling", False)
        test_exponential_kernel = call.data.get("test_exponential_kernel", False)

        target_coordinator = _get_target_coordinator(hass, entity_id)
        _LOGGER.debug("Handling calibrate_inertia for %d days (coordinator: %s)", days, target_coordinator.entry.entry_id)
        result = target_coordinator.statistics.calibrate_inertia(days=days, centered_energy_average=centered, test_asymmetric=test_asymmetric, test_delta_t_scaling=test_delta_t_scaling, test_exponential_kernel=test_exponential_kernel)
        return result

    hass.services.async_register(
        DOMAIN,
        SERVICE_CALIBRATE_INERTIA,
        handle_calibrate_inertia,
        schema=SERVICE_SCHEMA_CALIBRATE_INERTIA,
        supports_response=SupportsResponse.ONLY,
    )

    # Register Get Forecast Service
    async def handle_get_forecast(call: ServiceCall) -> dict:
        """Handle the get forecast service call.

        When ``isolate_sensor`` is provided, the returned forecast represents
        only that sensor's share of the building total — computed as
        ``max(0, global − Σ per_unit for all *other* sensors)``.  This is the
        demand signal that an MPC solver needs: the portion of heat loss that
        the target unit must cover after all other units have contributed their
        predicted share.
        """
        entity_id = call.data.get("entity_id")
        days = call.data.get("days", 1)
        isolate_sensor = call.data.get("isolate_sensor")

        target_coordinator = _get_target_coordinator(hass, entity_id)
        _LOGGER.debug("Handling get_forecast for %d days (coordinator: %s)", days, target_coordinator.entry.entry_id)

        now = dt_util.now()
        start_time = now
        end_time = now + timedelta(days=days)

        result = target_coordinator.forecast.get_hourly_forecast(start_time, end_time)

        if isolate_sensor and isolate_sensor in target_coordinator.energy_sensors:
            # Subtraction-based forecast isolation: the global prediction
            # includes all units.  Subtract predicted contributions from every
            # unit *except* the target to isolate the target's demand.
            for hour_entry in result:
                breakdown = hour_entry.get("unit_breakdown", {})
                other_sum = sum(
                    stats.get("net_kwh", 0.0)
                    for sid, stats in breakdown.items()
                    if sid != isolate_sensor
                )
                isolated = max(0.0, hour_entry["kwh"] - other_sum)
                hour_entry["kwh"] = round(isolated, 2)
                hour_entry["isolated_for"] = isolate_sensor
                hour_entry["subtracted_kwh"] = round(other_sum, 2)

        return {"forecast": result}

    hass.services.async_register(
        DOMAIN,
        SERVICE_GET_FORECAST,
        handle_get_forecast,
        schema=SERVICE_SCHEMA_GET_FORECAST,
        supports_response=SupportsResponse.ONLY,
    )

    # Register Diagnose Model Service
    async def handle_diagnose_model(call: ServiceCall) -> dict:
        """Handle the diagnose model service call."""
        entity_id = call.data.get("entity_id")
        days = call.data.get("days", 30)
        coord = _get_target_coordinator(hass, entity_id)
        _LOGGER.info(f"Service called: diagnose_model (days={days}, coordinator={coord.entry.entry_id})")
        return coord.diagnose_model(days_back=days)

    hass.services.async_register(
        DOMAIN,
        SERVICE_DIAGNOSE_MODEL,
        handle_diagnose_model,
        schema=SERVICE_SCHEMA_DIAGNOSE,
        supports_response=SupportsResponse.ONLY,
    )

    # Register Diagnose Solar Service
    SERVICE_SCHEMA_DIAGNOSE_SOLAR = vol.Schema({
        vol.Optional("entity_id"): cv.entity_id,
        vol.Optional("days", default=30): vol.All(vol.Coerce(int), vol.Range(min=1, max=365)),
        vol.Optional("apply_battery_decay", default=False): bool,
    })

    async def handle_diagnose_solar(call: ServiceCall) -> dict:
        """Handle the diagnose solar service call."""
        entity_id = call.data.get("entity_id")
        days = call.data.get("days", 30)
        apply_decay = call.data.get("apply_battery_decay", False)
        coord = _get_target_coordinator(hass, entity_id)
        _LOGGER.info(f"Service called: diagnose_solar (days={days}, apply_battery_decay={apply_decay}, coordinator={coord.entry.entry_id})")
        return coord.diagnose_solar(days_back=days, apply_battery_decay=apply_decay)

    hass.services.async_register(
        DOMAIN,
        SERVICE_DIAGNOSE_SOLAR,
        handle_diagnose_solar,
        schema=SERVICE_SCHEMA_DIAGNOSE_SOLAR,
        supports_response=SupportsResponse.ONLY,
    )

    return True

async def async_migrate_entry(hass: HomeAssistant, config_entry: ConfigEntry) -> bool:
    """Migrate old entry."""
    _LOGGER.debug(f"Migrating from version {config_entry.version}")

    if config_entry.version == 1:
        new_data = {**config_entry.data}

        # Import constants locally to avoid circular imports if needed
        from .const import (
            CONF_OUTDOOR_TEMP_SOURCE,
            CONF_WIND_SOURCE,
            CONF_WIND_GUST_SOURCE,
            SOURCE_SENSOR,
            SOURCE_WEATHER
        )

        # Migrate Outdoor Temp Source
        if new_data.get("outdoor_temp_sensor"):
            new_data[CONF_OUTDOOR_TEMP_SOURCE] = SOURCE_SENSOR
        else:
            new_data[CONF_OUTDOOR_TEMP_SOURCE] = SOURCE_WEATHER

        # Migrate Wind Speed Source
        if new_data.get("wind_speed_sensor"):
            new_data[CONF_WIND_SOURCE] = SOURCE_SENSOR
        else:
            new_data[CONF_WIND_SOURCE] = SOURCE_WEATHER

        # Migrate Wind Gust Source
        if new_data.get("wind_gust_sensor"):
            new_data[CONF_WIND_GUST_SOURCE] = SOURCE_SENSOR
        else:
            new_data[CONF_WIND_GUST_SOURCE] = SOURCE_WEATHER

        hass.config_entries.async_update_entry(config_entry, version=2, data=new_data)
        _LOGGER.info("Migration to version 2 successful")

    return True

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    if unload_ok := await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        coordinator = hass.data[DOMAIN].pop(entry.entry_id)

        # Ensure final save before unload
        await coordinator._async_save_data()

        # Unregister services if this is the last entry
        if not hass.data[DOMAIN]:
            hass.services.async_remove(DOMAIN, SERVICE_IMPORT_CSV)
            hass.services.async_remove(DOMAIN, SERVICE_EXPORT_CSV)
            hass.services.async_remove(DOMAIN, SERVICE_RESET_LEARNING)
            hass.services.async_remove(DOMAIN, SERVICE_RESET_UNIT_LEARNING)
            hass.services.async_remove(DOMAIN, SERVICE_RESET_FORECAST_ACCURACY)
            hass.services.async_remove(DOMAIN, SERVICE_BACKUP_DATA)
            hass.services.async_remove(DOMAIN, SERVICE_RESTORE_DATA)
            hass.services.async_remove(DOMAIN, SERVICE_REPLACE_SENSOR)
            hass.services.async_remove(DOMAIN, SERVICE_COMPARE_PERIODS)
            hass.services.async_remove(DOMAIN, SERVICE_EXIT_COOLDOWN)
            hass.services.async_remove(DOMAIN, SERVICE_GET_FORECAST)
            hass.services.async_remove(DOMAIN, SERVICE_CALIBRATE_INERTIA)
            hass.services.async_remove(DOMAIN, SERVICE_CALIBRATE_WIND_THRESHOLDS)
            hass.services.async_remove(DOMAIN, SERVICE_RESET_SOLAR_LEARNING)
            hass.services.async_remove(DOMAIN, SERVICE_RETRAIN_FROM_HISTORY)
            # Diagnose surfaces
            hass.services.async_remove(DOMAIN, SERVICE_DIAGNOSE_MODEL)
            hass.services.async_remove(DOMAIN, SERVICE_DIAGNOSE_SOLAR)
            # Per-unit threshold calibration (#871)
            hass.services.async_remove(DOMAIN, SERVICE_CALIBRATE_UNIT_THRESHOLDS)
            # Solar coefficient on-demand fitters (#884, #904)
            hass.services.async_remove(DOMAIN, SERVICE_BATCH_FIT_SOLAR)
            hass.services.async_remove(DOMAIN, SERVICE_APPLY_IMPLIED_COEFFICIENT)
            # Tobit live-learner controls (#904 stage 3)
            hass.services.async_remove(
                DOMAIN, SERVICE_SET_EXPERIMENTAL_TOBIT_LIVE_LEARNER
            )
            hass.services.async_remove(DOMAIN, SERVICE_SET_TOBIT_LIVE_ENTITIES)
            hass.services.async_remove(DOMAIN, SERVICE_RESET_TOBIT_LIVE_STATE)

    return unload_ok
