"""Number platform for Heating Analytics."""
from __future__ import annotations

from homeassistant.components.number import (
    NumberEntity,
    NumberMode,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import (
    DOMAIN,
)
from .coordinator import HeatingDataCoordinator


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Heating Analytics numbers based on a config entry."""
    coordinator: HeatingDataCoordinator = hass.data[DOMAIN][entry.entry_id]

    entities = [
        HeatingLearningRateNumber(coordinator, entry),
        HeatingSolarCorrectionNumber(coordinator, entry),
    ]

    async_add_entities(entities)


class HeatingNumberBase(CoordinatorEntity, NumberEntity):
    """Base class for Heating Analytics numbers."""

    _attr_entity_category = EntityCategory.CONFIG

    def __init__(self, coordinator: HeatingDataCoordinator, entry: ConfigEntry) -> None:
        """Initialize."""
        super().__init__(coordinator)
        self.entry = entry
        self._attr_has_entity_name = True

    @property
    def device_info(self):
        """Return device information."""
        return {
            "identifiers": {(DOMAIN, self.entry.entry_id)},
            "name": self.entry.title,
            "manufacturer": "Heating Analytics",
        }


class HeatingLearningRateNumber(HeatingNumberBase):
    """Number for Learning Rate."""

    _attr_name = "Learning Rate"
    _attr_mode = NumberMode.BOX
    _attr_native_min_value = 0.1
    _attr_native_max_value = 10.0
    _attr_native_step = 0.1
    _attr_icon = "mdi:percent"
    _attr_native_unit_of_measurement = "%"

    @property
    def native_value(self) -> float:
        """Return the value."""
        return round(self.coordinator.learning_rate * 100, 1)

    async def async_set_native_value(self, value: float) -> None:
        """Set the value."""
        self.coordinator.learning_rate = value / 100.0
        await self.coordinator._async_save_data()
        self.async_write_ha_state()

    @property
    def unique_id(self) -> str:
        return f"{self.entry.entry_id}_learning_rate"


class HeatingSolarCorrectionNumber(HeatingNumberBase):
    """Number for Solar Correction (Screens/Shading)."""

    _attr_name = "Solar Correction"
    _attr_mode = NumberMode.SLIDER
    _attr_native_min_value = 0.0
    _attr_native_max_value = 100.0
    _attr_native_step = 1.0
    _attr_icon = "mdi:window-shutter"
    _attr_native_unit_of_measurement = "%"

    @property
    def native_value(self) -> float:
        """Return the value."""
        return self.coordinator.solar_correction_percent

    async def async_set_native_value(self, value: float) -> None:
        """Set the value."""
        self.coordinator.solar_correction_percent = value
        await self.coordinator._async_save_data()
        self.async_write_ha_state()
        # Force update to refresh recommendation and potential stats immediately
        await self.coordinator.async_request_refresh()

    @property
    def unique_id(self) -> str:
        return f"{self.entry.entry_id}_solar_correction"
