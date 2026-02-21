"""Test integration services."""
from unittest.mock import MagicMock, patch, AsyncMock
import pytest
import voluptuous as vol
from custom_components.heating_analytics import async_setup_entry, SERVICE_EXIT_COOLDOWN
from custom_components.heating_analytics.const import DOMAIN

@pytest.fixture
def mock_entry():
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.data = {
        "outdoor_temp_sensor": "sensor.temp",
        "energy_sensors": ["sensor.heater"]
    }
    return entry

@pytest.mark.asyncio
async def test_exit_cooldown_service_registration(hass, mock_entry):
    """Test that the exit_cooldown service is registered."""
    hass.data = {}

    # Mock services register
    hass.services.async_register = MagicMock()
    # Mock async_forward_entry_setups
    hass.config_entries.async_forward_entry_setups = AsyncMock()

    with patch("custom_components.heating_analytics.HeatingDataCoordinator") as mock_coord_cls:
        mock_coord = mock_coord_cls.return_value
        mock_coord.async_config_entry_first_refresh = AsyncMock()
        mock_coord.storage = MagicMock()
        mock_coord.storage.async_load_data = AsyncMock()

        await async_setup_entry(hass, mock_entry)

        # Check if async_register was called with SERVICE_EXIT_COOLDOWN
        # We look for a call where args[1] == SERVICE_EXIT_COOLDOWN
        found = False
        for call in hass.services.async_register.call_args_list:
            if call.args[0] == DOMAIN and call.args[1] == SERVICE_EXIT_COOLDOWN:
                found = True
                break

        assert found, "SERVICE_EXIT_COOLDOWN was not registered"

@pytest.mark.asyncio
async def test_exit_cooldown_service_call(hass, mock_entry):
    """Test that calling the service invokes the coordinator method."""
    hass.data = {}

    # Capture the handler function
    handler = None
    def mock_register(domain, service, callback, schema=None):
        nonlocal handler
        if domain == DOMAIN and service == SERVICE_EXIT_COOLDOWN:
            handler = callback

    hass.services.async_register = MagicMock(side_effect=mock_register)
    # Mock async_forward_entry_setups
    hass.config_entries.async_forward_entry_setups = AsyncMock()

    with patch("custom_components.heating_analytics.HeatingDataCoordinator") as mock_coord_cls:
        mock_coord = mock_coord_cls.return_value
        mock_coord.async_config_entry_first_refresh = AsyncMock()
        mock_coord.storage = MagicMock()
        mock_coord.storage.async_load_data = AsyncMock()
        mock_coord.async_exit_cooldown = AsyncMock()

        # Patch _get_coordinators to bypass isinstance check on Mock object
        with patch("custom_components.heating_analytics._get_coordinators", return_value=[mock_coord]):
            await async_setup_entry(hass, mock_entry)

            assert handler is not None, "Service handler was not captured"

            # Simulate Service Call
            call = MagicMock()
            await handler(call)

            # Verify coordinator method called
            mock_coord.async_exit_cooldown.assert_awaited_once()
