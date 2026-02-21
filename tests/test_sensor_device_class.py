"""Test sensor device classes."""
from unittest.mock import MagicMock
import pytest
from homeassistant.core import HomeAssistant
from homeassistant.components.sensor import SensorDeviceClass
from custom_components.heating_analytics.sensor import (
    HeatingPotentialSavingsSensor,
    HeatingDeviceDailySensor,
    HeatingDeviceLifetimeSensor,
)

@pytest.mark.asyncio
async def test_sensor_device_classes(hass: HomeAssistant):
    """Test that sensors have the correct device class."""
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.title = "Test Heating"

    coordinator = MagicMock()
    coordinator.energy_sensors = []
    coordinator.enable_lifetime_tracking = True
    # Fake hass state for device lookup
    coordinator.hass = hass
    coordinator.hass.states = MagicMock()
    coordinator.hass.states.get.return_value = MagicMock(name="Test Device")

    # Test HeatingPotentialSavingsSensor
    sensor_savings = HeatingPotentialSavingsSensor(coordinator, entry)
    assert sensor_savings._attr_device_class == SensorDeviceClass.ENERGY

    # Test HeatingDeviceDailySensor
    sensor_daily = HeatingDeviceDailySensor(coordinator, entry, "sensor.test_device")
    assert sensor_daily._attr_device_class == SensorDeviceClass.ENERGY

    # Test HeatingDeviceLifetimeSensor
    sensor_lifetime = HeatingDeviceLifetimeSensor(coordinator, entry, "sensor.test_device")
    assert sensor_lifetime._attr_device_class == SensorDeviceClass.ENERGY
