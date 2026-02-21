"""Test for missing device_info in HeatingAnalyticsModeSelect."""
import pytest
from unittest.mock import MagicMock
import sys

# We need to import the class to test.
# Since we mock homeassistant in conftest, we can import custom_components
from custom_components.heating_analytics.select import HeatingAnalyticsModeSelect
from custom_components.heating_analytics.const import DOMAIN

def test_select_missing_device_info(hass):
    """Test that HeatingAnalyticsModeSelect initially lacks device_info or is missing it."""

    # Mock Coordinator
    mock_coordinator = MagicMock()
    mock_coordinator.hass = hass
    mock_coordinator.entry = MagicMock()
    mock_coordinator.entry.entry_id = "test_entry_id"
    mock_coordinator.entry.title = "Test Device"

    # Mock source entity state
    mock_state = MagicMock()
    mock_state.name = "Test Unit"
    hass.states.get.return_value = mock_state

    # Initialize Entity
    entity = HeatingAnalyticsModeSelect(mock_coordinator, "sensor.test_unit")

    # Check for device_info
    # In the current (buggy) state, it might be missing or inherited from CoordinatorEntity which defaults to None?
    # Actually CoordinatorEntity doesn't implement device_info by default.

    device_info = entity.device_info

    # This assertion is expected to FAIL if device_info is missing (returns None)
    # or pass if we fix it.

    # If the class inherits from CoordinatorEntity and doesn't define device_info,
    # and CoordinatorEntity doesn't define it either, it should use the default which might be None
    # or raise AttributeError if accessed directly on some base classes, but typically it is a property.

    print(f"\nDevice Info: {device_info}")

    assert device_info is not None, "device_info should not be None"
    assert device_info["identifiers"] == {(DOMAIN, "test_entry_id")}
    assert device_info["name"] == "Test Device"
    assert device_info["manufacturer"] == "Heating Analytics"
