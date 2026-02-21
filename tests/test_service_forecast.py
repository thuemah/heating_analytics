"""Test the get_forecast service."""
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timedelta
import pytest

from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.util import dt as dt_util
from custom_components.heating_analytics import async_setup_entry, DOMAIN
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator

@pytest.mark.asyncio
async def test_get_forecast_service(hass: HomeAssistant):
    """Test the get_forecast service call."""

    # Configure hass mock for async methods
    hass.config_entries.async_forward_entry_setups = AsyncMock()

    # Mock Coordinator and its components
    # We use a loose mock instead of spec=HeatingDataCoordinator to avoid attribute issues
    coordinator = MagicMock()
    coordinator.entry = MagicMock()
    coordinator.entry.entry_id = "test_entry"
    coordinator.forecast = MagicMock()

    # Ensure async method is awaitable
    coordinator.async_config_entry_first_refresh = AsyncMock()
    coordinator._async_save_data = AsyncMock()

    # Expected return data
    mock_forecast_data = [
        {
            "datetime": "2023-10-27T10:00:00+01:00",
            "kwh": 1.5,
            "temp": 5.0,
            "wind_speed": 3.0,
            "solar_kwh": 0.0,
            "aux_impact_kwh": 0.0,
            "unit_breakdown": {}
        }
    ]
    coordinator.forecast.get_hourly_forecast.return_value = mock_forecast_data

    # Setup HASS data (simulating that the coordinator is already active)
    hass.data = {DOMAIN: {"test_entry": coordinator}}

    # Helper to capture the service handler
    captured_handler = None

    def mock_async_register(domain, service, handler, **kwargs):
        nonlocal captured_handler
        if service == "get_forecast":
            captured_handler = handler

    hass.services.async_register = mock_async_register

    # Mock config entry
    entry = MagicMock()
    entry.entry_id = "test_entry"

    # Mock the internal _get_coordinators to return our mocked coordinator
    # This is needed because handle_get_forecast calls it if no entity_id is provided
    with patch("custom_components.heating_analytics._get_coordinators", return_value=[coordinator]):
        # Mock the class instantiation in __init__.py
        with patch("custom_components.heating_analytics.HeatingDataCoordinator", return_value=coordinator):
             await async_setup_entry(hass, entry)

    assert captured_handler is not None, "Service was not registered"

    # Case 1: Call without entity_id (Default)
    # Use MagicMock for call to ensure .data is a dict
    call = MagicMock()
    call.data = {"days": 1}

    # We must mock _get_coordinators again because isinstance(Mock) check fails in real helper
    with patch("custom_components.heating_analytics._get_coordinators", return_value=[coordinator]):
        response = await captured_handler(call)

    assert "forecast" in response
    assert response["forecast"] == mock_forecast_data

    # Verify forecast method was called with correct duration
    coordinator.forecast.get_hourly_forecast.assert_called()
    args, _ = coordinator.forecast.get_hourly_forecast.call_args
    start, end = args
    assert isinstance(start, datetime)
    assert isinstance(end, datetime)
    # Allow small timing difference
    assert abs((end - start) - timedelta(days=1)) < timedelta(seconds=5)

    # Case 2: Call with Entity ID
    # We need to mock the entity registry
    coordinator.forecast.get_hourly_forecast.reset_mock()

    with patch("homeassistant.helpers.entity_registry.async_get") as mock_er_get:
        mock_registry = MagicMock()
        mock_er_get.return_value = mock_registry

        mock_entry = MagicMock()
        mock_entry.config_entry_id = "test_entry"
        mock_registry.async_get.return_value = mock_entry

        call_with_entity = MagicMock()
        call_with_entity.data = {"days": 2, "entity_id": "sensor.heating"}

        # Ensure hass.data is set correctly for lookup
        hass.data[DOMAIN]["test_entry"] = coordinator

        # No need to patch _get_coordinators here if entity lookup works
        response = await captured_handler(call_with_entity)

        assert response["forecast"] == mock_forecast_data

        # Verify forecast method was called with 2 days
        coordinator.forecast.get_hourly_forecast.assert_called()
        args, _ = coordinator.forecast.get_hourly_forecast.call_args
        start, end = args
        assert abs((end - start) - timedelta(days=2)) < timedelta(seconds=5)
