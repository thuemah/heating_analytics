"""Test to verify forecast persistence (Regression Check)."""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.forecast import ForecastManager
from custom_components.heating_analytics.storage import StorageManager

@pytest.fixture
def mock_hass():
    hass = MagicMock()
    hass.config.units.is_metric = True
    return hass

@pytest.fixture
def mock_entry():
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.data = {
        "energy_sensors": ["sensor.energy"],
        "weather_entity": "weather.home"
    }
    return entry

async def test_forecast_persistence(mock_hass, mock_entry):
    """Verify that reference and live forecasts are persisted and restored."""

    # 1. Setup Initial State
    with patch("custom_components.heating_analytics.storage.Store") as MockStore:
        # Mock Store for Save/Load
        # Use AsyncMock for async methods
        mock_store_instance = MockStore.return_value
        mock_store_instance.async_load = AsyncMock(return_value=None) # First load empty
        mock_store_instance.async_save = AsyncMock()

        coordinator = HeatingDataCoordinator(mock_hass, mock_entry)
        # Initialize storage (load empty)
        await coordinator.storage.async_load_data()

        # Populate Forecasts
        dummy_forecast = [
            {"datetime": "2023-10-27T12:00:00", "temperature": 10.0, "condition": "sunny"}
        ]
        coordinator.forecast._reference_forecast = list(dummy_forecast)
        coordinator.forecast._live_forecast = list(dummy_forecast)
        coordinator.forecast._primary_reference_forecast = list(dummy_forecast)

        # Trigger Save
        await coordinator.storage.async_save_data(force=True)

        # Verify Save called with correct data
        mock_store_instance.async_save.assert_called_once()
        saved_data = mock_store_instance.async_save.call_args[0][0]

        assert "reference_forecast" in saved_data
        assert "live_forecast" in saved_data
        assert len(saved_data["reference_forecast"]) == 1
        assert saved_data["reference_forecast"][0]["temperature"] == 10.0

        # 2. Simulate Restart (New Coordinator, Load from Saved Data)
        # We need the MockStore to return the saved_data on next load
        mock_store_instance.async_load.return_value = saved_data

        coordinator_2 = HeatingDataCoordinator(mock_hass, mock_entry)

        # Action: Load Data
        await coordinator_2.storage.async_load_data()

        # Verification
        assert coordinator_2.forecast._reference_forecast is not None
        assert len(coordinator_2.forecast._reference_forecast) == 1
        assert coordinator_2.forecast._reference_forecast[0]["temperature"] == 10.0

        assert coordinator_2.forecast._live_forecast is not None
        assert len(coordinator_2.forecast._live_forecast) == 1

        # Check Reference Map Rebuild
        assert coordinator_2.forecast._cached_reference_map is not None
        date_key = "2023-10-27"
        assert date_key in coordinator_2.forecast._cached_reference_map
        assert 12 in coordinator_2.forecast._cached_reference_map[date_key]
