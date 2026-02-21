"""Select platform for Heating Analytics."""
from __future__ import annotations

from homeassistant.components.select import SelectEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import (
    DOMAIN,
    MODE_HEATING,
    MODE_COOLING,
    MODE_OFF,
    MODE_GUEST_HEATING,
    MODE_GUEST_COOLING,
    CONF_HAS_AC_UNITS,
)
from .coordinator import HeatingDataCoordinator


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the select platform."""
    coordinator: HeatingDataCoordinator = hass.data[DOMAIN][entry.entry_id]

    # Only create mode selects if user has AC capability
    if not entry.data.get(CONF_HAS_AC_UNITS, False):
        return

    # Create a Select entity for each heating unit
    entities = []
    for entity_id in coordinator.energy_sensors:
        entities.append(
            HeatingAnalyticsModeSelect(coordinator, entity_id)
        )

    async_add_entities(entities)


class HeatingAnalyticsModeSelect(CoordinatorEntity, SelectEntity):
    """Select entity to control heating/cooling mode for a unit."""

    _attr_has_entity_name = True
    _attr_entity_category = EntityCategory.CONFIG
    _attr_icon = "mdi:hvac"

    def __init__(self, coordinator: HeatingDataCoordinator, source_entity_id: str) -> None:
        """Initialize the select entity."""
        super().__init__(coordinator)
        self._source_entity_id = source_entity_id

        # Unique ID based on the integration entry and the source entity
        self._attr_unique_id = f"{coordinator.entry.entry_id}_{source_entity_id}_mode"

        # Name derived from source entity
        # We try to get the friendly name of the source entity
        state = coordinator.hass.states.get(source_entity_id)
        source_name = state.name if state else source_entity_id
        self._attr_name = f"{source_name} Mode"

        self._attr_options = [
            MODE_HEATING,
            MODE_COOLING,
            MODE_OFF,
            MODE_GUEST_HEATING,
            MODE_GUEST_COOLING,
        ]

    @property
    def device_info(self):
        """Return device information."""
        return {
            "identifiers": {(DOMAIN, self.coordinator.entry.entry_id)},
            "name": self.coordinator.entry.title,
            "manufacturer": "Heating Analytics",
        }

    @property
    def current_option(self) -> str | None:
        """Return the selected entity option to represent the entity state."""
        return self.coordinator.get_unit_mode(self._source_entity_id)

    async def async_select_option(self, option: str) -> None:
        """Change the selected option."""
        await self.coordinator.async_set_unit_mode(self._source_entity_id, option)
