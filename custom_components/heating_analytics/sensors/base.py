"""Base sensor for Heating Analytics."""
from __future__ import annotations

from homeassistant.components.sensor import SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from ..const import DOMAIN
from ..coordinator import HeatingDataCoordinator


class HeatingAnalyticsBaseSensor(CoordinatorEntity, SensorEntity):
    """Base class for Heating Analytics sensors."""

    def __init__(self, coordinator: HeatingDataCoordinator, entry: ConfigEntry) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator)
        self.entry = entry
        self._attr_has_entity_name = True
        self._cached_stats = None
        self._cached_time = None
        self._cached_past_date = None
        self._cached_past_data = None  # Tuple: (model_past, solar_past, temp_past, wind_past, model_last_so_far, solar_last_so_far, temp_last_so_far, wind_last_so_far, model_last_remaining, solar_last_remaining, temp_last_remaining, wind_last_remaining, days_past, ly_total_days)

    @property
    def device_info(self):
        """Return device information."""
        return {
            "identifiers": {(DOMAIN, self.entry.entry_id)},
            "name": self.entry.title,
            "manufacturer": "Heating Analytics",
        }
