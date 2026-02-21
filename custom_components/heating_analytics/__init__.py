"""The Heating Analytics integration."""
from __future__ import annotations

import logging
from datetime import timedelta
import voluptuous as vol

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform, EVENT_HOMEASSISTANT_STOP
from homeassistant.core import HomeAssistant, ServiceCall, Event, SupportsResponse
from homeassistant.exceptions import ConfigEntryNotReady
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

SERVICE_SCHEMA_IMPORT = vol.Schema({
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
    vol.Required("file_path"): cv.string,
    vol.Required("export_type", default="daily"): vol.In(["daily", "hourly"]),
})

SERVICE_SCHEMA_RESET = vol.Schema({})

SERVICE_SCHEMA_RESET_UNIT = vol.Schema({
    vol.Required("entity_id"): cv.entity_id,
})

SERVICE_SCHEMA_RESET_ACCURACY = vol.Schema({})

SERVICE_SCHEMA_BACKUP = vol.Schema({
    vol.Required("file_path"): cv.string,
})

SERVICE_SCHEMA_RESTORE = vol.Schema({
    vol.Required("file_path"): cv.string,
})

SERVICE_SCHEMA_REPLACE_SENSOR = vol.Schema({
    vol.Required("old_entity_id"): cv.entity_id,
    vol.Required("new_entity_id"): cv.entity_id,
})

SERVICE_SCHEMA_COMPARE_PERIODS = vol.Schema({
    vol.Required("period_1_start"): cv.date,
    vol.Required("period_1_end"): cv.date,
    vol.Required("period_2_start"): cv.date,
    vol.Required("period_2_end"): cv.date,
})

def _get_coordinators(hass: HomeAssistant) -> list[HeatingDataCoordinator]:
    """Helper to get all active HeatingDataCoordinators."""
    return [
        coord
        for coord in hass.data.get(DOMAIN, {}).values()
        if isinstance(coord, HeatingDataCoordinator)
    ]

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Heating Analytics from a config entry."""
    
    hass.data.setdefault(DOMAIN, {})
    
    coordinator = HeatingDataCoordinator(hass, entry)
    try:
        await coordinator.async_config_entry_first_refresh()
    except Exception as ex:
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
        file_path = call.data.get("file_path")
        mapping = call.data.get("column_mapping")
        update_model = call.data.get("update_model", True)

        _LOGGER.info(f"Service called to import CSV from {file_path} (Update Model: {update_model})")

        for coord in _get_coordinators(hass):
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
        file_path = call.data.get("file_path")
        export_type = call.data.get("export_type")

        _LOGGER.info(f"Service called to export CSV ({export_type}) to {file_path}")

        for coord in _get_coordinators(hass):
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
        _LOGGER.info("Service called to reset learning data.")

        for coord in _get_coordinators(hass):
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
        _LOGGER.info("Service called to reset forecast accuracy history.")

        for coord in _get_coordinators(hass):
            await coord.async_reset_forecast_accuracy()

    hass.services.async_register(
        DOMAIN,
        SERVICE_RESET_FORECAST_ACCURACY,
        handle_reset_forecast_accuracy,
        schema=SERVICE_SCHEMA_RESET_ACCURACY
    )

    # Register Backup Service
    async def handle_backup_data(call: ServiceCall):
        """Handle the backup data service call."""
        file_path = call.data.get("file_path")
        _LOGGER.info(f"Service called to backup data to {file_path}")

        for coord in _get_coordinators(hass):
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
        file_path = call.data.get("file_path")
        _LOGGER.info(f"Service called to restore data from {file_path}")

        for coord in _get_coordinators(hass):
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
        old_id = call.data.get("old_entity_id")
        new_id = call.data.get("new_entity_id")
        _LOGGER.info(f"Service called to replace sensor: {old_id} -> {new_id}")

        entries_to_reload = []

        for coord in _get_coordinators(hass):
            if await coord.async_replace_sensor_source(old_id, new_id):
                # Only reload if the replacement was actually performed
                entries_to_reload.append(coord.entry.entry_id)

        # Reload affected entries to update entity registry and baseline
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
        p1_start = call.data.get("period_1_start")
        p1_end = call.data.get("period_1_end")
        p2_start = call.data.get("period_2_start")
        p2_end = call.data.get("period_2_end")

        _LOGGER.info(f"Service called to compare periods: {p1_start}-{p1_end} vs {p2_start}-{p2_end}")

        for coord in _get_coordinators(hass):
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
        _LOGGER.info("Service called to exit auxiliary cooldown.")

        for coord in _get_coordinators(hass):
            await coord.async_exit_cooldown()

    hass.services.async_register(
        DOMAIN,
        SERVICE_EXIT_COOLDOWN,
        handle_exit_cooldown,
        schema=vol.Schema({}),
    )

    # Register Get Forecast Service
    async def handle_get_forecast(call: ServiceCall) -> dict:
        """Handle the get forecast service call."""
        entity_id = call.data.get("entity_id")
        days = call.data.get("days", 1)

        target_coordinator = None

        if entity_id:
            registry = er.async_get(hass)
            entry = registry.async_get(entity_id)
            if entry and entry.config_entry_id:
                target_coordinator = hass.data[DOMAIN].get(entry.config_entry_id)

        if not target_coordinator:
            # Default to first available
            coordinators = _get_coordinators(hass)
            if coordinators:
                target_coordinator = coordinators[0]

        if not target_coordinator:
             raise ValueError("No Heating Analytics instance found.")

        _LOGGER.debug(f"Handling get_forecast for {days} days (Coordinator: {target_coordinator.entry.entry_id})")

        now = dt_util.now()
        start_time = now
        end_time = now + timedelta(days=days)

        result = target_coordinator.forecast.get_hourly_forecast(start_time, end_time)
        return {"forecast": result}

    hass.services.async_register(
        DOMAIN,
        SERVICE_GET_FORECAST,
        handle_get_forecast,
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

    return unload_ok
