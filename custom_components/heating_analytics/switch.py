"""Switch platform for Heating Analytics."""
from __future__ import annotations

from homeassistant.components.switch import SwitchEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.const import EntityCategory

from .const import DOMAIN
from .coordinator import HeatingDataCoordinator

async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Heating Analytics switches based on a config entry."""
    coordinator: HeatingDataCoordinator = hass.data[DOMAIN][entry.entry_id]

    entities = [
        HeatingLearningSwitch(coordinator, entry),
        HeatingAuxiliaryHeatingSwitch(coordinator, entry),
    ]

    async_add_entities(entities)


class HeatingSwitchBase(CoordinatorEntity, SwitchEntity):
    """Base class for Heating Analytics switches."""

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


class HeatingLearningSwitch(HeatingSwitchBase):
    """Switch to enable/disable learning."""

    _attr_name = "Learning Enabled"
    _attr_icon = "mdi:school"

    @property
    def is_on(self) -> bool:
        """Return true if switch is on."""
        return self.coordinator.learning_enabled

    async def async_turn_on(self, **kwargs) -> None:
        """Turn the switch on."""
        self.coordinator.learning_enabled = True
        await self.coordinator._async_save_data()
        self.async_write_ha_state()

    async def async_turn_off(self, **kwargs) -> None:
        """Turn the switch off."""
        self.coordinator.learning_enabled = False
        await self.coordinator._async_save_data()
        self.async_write_ha_state()

    @property
    def unique_id(self) -> str:
        return f"{self.entry.entry_id}_learning_enabled"


class HeatingAuxiliaryHeatingSwitch(HeatingSwitchBase):
    """Switch to manually toggle Auxiliary Untracked Heating Active state."""

    _attr_name = "Auxiliary Untracked Heating Active"
    _attr_icon = "mdi:fireplace"

    @property
    def is_on(self) -> bool:
        """Return true if switch is on."""
        return self.coordinator.auxiliary_heating_active

    async def async_turn_on(self, **kwargs) -> None:
        """Turn the switch on."""
        await self.coordinator.set_auxiliary_heating_active(True)
        self.async_write_ha_state()

    async def async_turn_off(self, **kwargs) -> None:
        """Turn the switch off."""
        await self.coordinator.set_auxiliary_heating_active(False)
        self.async_write_ha_state()

    @property
    def unique_id(self) -> str:
        return f"{self.entry.entry_id}_auxiliary_heating_active"
