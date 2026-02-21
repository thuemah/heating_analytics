"""Test daily processing logic."""
from unittest.mock import MagicMock, patch, AsyncMock
import pytest
from datetime import datetime, date
from homeassistant.core import HomeAssistant
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.const import ATTR_TDD

@pytest.mark.asyncio
async def test_daily_processing(hass: HomeAssistant):
    """Test daily processing logic."""
    entry = MagicMock()
    entry.data = {"balance_point": 17.0}

    with patch("custom_components.heating_analytics.storage.Store") as mock_store_cls:
        coordinator = HeatingDataCoordinator(hass, entry)
        coordinator._async_save_data = AsyncMock()
        coordinator.storage.append_daily_log_csv = AsyncMock()
        coordinator.forecast = MagicMock()
        coordinator.forecast.log_accuracy = MagicMock()


        # Setup Daily State
        coordinator._accumulated_energy_today = 10.0
        coordinator.data[ATTR_TDD] = 5.0

        # Setup Daily History
        coordinator._daily_history = {}

        # Setup Logs for the day (for Avg Temp calculation)
        # Note: _process_daily_data now sums kwh from logs, so we must populate logs with energy
        coordinator._hourly_log = [
            {"timestamp": "2023-10-27T10:00:00", "temp": 10.0, "effective_wind": 2.0, "solar_factor": 0.5, "actual_kwh": 4.0, "tdd": 2.0},
            {"timestamp": "2023-10-27T11:00:00", "temp": 12.0, "effective_wind": 4.0, "solar_factor": 0.7, "actual_kwh": 6.0, "tdd": 3.0}
        ]
        # Avg Temp = 11.0. Avg Wind = 3.0. Avg Solar = 0.6. Total kWh = 10.0. Total TDD = 5.0

        day_to_process = date(2023, 10, 27)

        await coordinator._process_daily_data(day_to_process)

        # 1. Verify History Updated
        key = "2023-10-27"
        assert key in coordinator._daily_history
        assert coordinator._daily_history[key]["kwh"] == 10.0
        assert coordinator._daily_history[key]["tdd"] == 5.0
        assert coordinator._daily_history[key]["temp"] == 11.0 # Calculated from logs
        assert coordinator._daily_history[key]["wind"] == 3.0
        assert coordinator._daily_history[key]["solar_factor"] == 0.6 # Verified new logic

        # 2. Verify Cleanup
        assert coordinator._accumulated_energy_today == 0.0
        assert coordinator.data[ATTR_TDD] == 0.0

        # 3. Verify CSV Log
        coordinator.storage.append_daily_log_csv.assert_awaited_once()
        log_entry = coordinator.storage.append_daily_log_csv.call_args[0][0]
        assert log_entry["kwh"] == 10.0
        assert log_entry["timestamp"] == key

        # 4. Verify Forecast Accuracy Logged
        coordinator.forecast.log_accuracy.assert_called_once_with(
            key, 10.0, 0.0, modeled_net_kwh=0.0, guest_impact_kwh=0.0
        )

@pytest.mark.asyncio
async def test_daily_processing_no_logs(hass: HomeAssistant):
    """Test daily processing fallback when no hourly logs exist."""
    entry = MagicMock()
    entry.data = {"balance_point": 17.0}

    with patch("custom_components.heating_analytics.storage.Store"):
        coordinator = HeatingDataCoordinator(hass, entry)
        coordinator._async_save_data = AsyncMock()
        coordinator.storage.append_daily_log_csv = AsyncMock()
        coordinator.forecast = MagicMock()
        coordinator.forecast.log_accuracy = MagicMock()

        coordinator._hourly_log = [] # Empty
        coordinator.data[ATTR_TDD] = 7.0 # Implies Avg Temp = 17 - 7 = 10.0
        coordinator._accumulated_energy_today = 5.0

        day_to_process = date(2023, 10, 27)

        await coordinator._process_daily_data(day_to_process)

        key = "2023-10-27"
        assert coordinator._daily_history[key]["temp"] == 10.0 # Fallback derived
        assert coordinator._daily_history[key]["wind"] == 0.0 # Default
        assert coordinator._daily_history[key]["solar_factor"] == 0.0 # Default Verified

        # Verify Forecast Accuracy Logged
        coordinator.forecast.log_accuracy.assert_called_once_with(
            key, 5.0, 0.0, modeled_net_kwh=0.0, guest_impact_kwh=0.0
        )

@pytest.mark.asyncio
async def test_daily_processing_learning_disabled(hass: HomeAssistant):
    """Test daily processing logic skips forecast accuracy when learning is disabled."""
    entry = MagicMock()
    entry.data = {"balance_point": 17.0}

    with patch("custom_components.heating_analytics.storage.Store"):
        coordinator = HeatingDataCoordinator(hass, entry)
        coordinator._async_save_data = AsyncMock()
        coordinator.storage.append_daily_log_csv = AsyncMock()
        coordinator.forecast = MagicMock()
        coordinator.forecast.log_accuracy = MagicMock()

        # Disable Learning
        coordinator.learning_enabled = False

        # Setup Daily State
        coordinator._accumulated_energy_today = 10.0
        coordinator.data[ATTR_TDD] = 5.0

        # Setup Logs for the day
        coordinator._hourly_log = [
            {"timestamp": "2023-10-27T10:00:00", "temp": 10.0, "effective_wind": 2.0, "solar_factor": 0.5, "actual_kwh": 4.0, "tdd": 2.0},
        ]

        day_to_process = date(2023, 10, 27)

        await coordinator._process_daily_data(day_to_process)

        # Verify Forecast Accuracy was NOT Logged
        coordinator.forecast.log_accuracy.assert_not_called()
