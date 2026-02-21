"""Test hourly processing logic."""
from unittest.mock import MagicMock, patch, AsyncMock
import pytest
from datetime import datetime
from homeassistant.core import HomeAssistant
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator

@pytest.mark.asyncio
async def test_hourly_processing_triggers(hass: HomeAssistant):
    """Test that hourly processing is triggered correctly."""
    entry = MagicMock()
    entry.data = {"balance_point": 17.0, "learning_rate": 0.1}

    with patch("custom_components.heating_analytics.storage.Store") as mock_store_cls:
        # Properly mock the store and its async method
        mock_store_instance = mock_store_cls.return_value
        mock_store_instance.async_load = AsyncMock(return_value={})
        mock_store_instance.async_save = AsyncMock()

        coordinator = HeatingDataCoordinator(hass, entry)
        # Mock storage.async_load_data to avoid the real call failing with MagicMock
        coordinator.storage.async_load_data = AsyncMock()
        coordinator._async_save_data = AsyncMock()
        coordinator._process_hourly_data = AsyncMock()
        coordinator.statistics.calculate_temp_stats = MagicMock()

        # 1. Initial Call (No last hour)
        current_time = datetime(2023, 10, 27, 12, 0, 0)
        with patch("custom_components.heating_analytics.coordinator.dt_util.now", return_value=current_time):
             await coordinator._async_update_data()

        # Should initialize last processed, but NOT trigger processing (first run)
        assert coordinator._last_hour_processed == 12
        coordinator._process_hourly_data.assert_not_called()

        # 2. Same Hour (12:30)
        current_time = datetime(2023, 10, 27, 12, 30, 0)
        with patch("custom_components.heating_analytics.coordinator.dt_util.now", return_value=current_time):
             await coordinator._async_update_data()

        coordinator._process_hourly_data.assert_not_called()

        # 3. Next Hour (13:00)
        current_time = datetime(2023, 10, 27, 13, 0, 0)
        with patch("custom_components.heating_analytics.coordinator.dt_util.now", return_value=current_time):
             await coordinator._async_update_data()

        coordinator._process_hourly_data.assert_awaited_once_with(current_time)
        assert coordinator._last_hour_processed == 13
        coordinator.statistics.calculate_temp_stats.assert_called_once()

@pytest.mark.asyncio
async def test_log_retention(hass: HomeAssistant):
    """Test that hourly logs are truncated (retention policy)."""
    entry = MagicMock()
    entry.data = {"balance_point": 17.0}

    with patch("custom_components.heating_analytics.storage.Store"):
        coordinator = HeatingDataCoordinator(hass, entry)
        coordinator._async_save_data = AsyncMock()

        # Fill logs with 2160 entries (90 days)
        coordinator._hourly_log = [{"id": i, "temp": 0.0} for i in range(2160)]

        coordinator._hourly_sample_count = 1
        coordinator._hourly_wind_values = [0.0]
        coordinator._hourly_temp_sum = 0.0

        current_time = datetime(2023, 10, 27, 13, 0, 0)
        await coordinator._process_hourly_data(current_time)

        # Should append 1, total 2161 -> Truncate to 2160
        assert len(coordinator._hourly_log) == 2160
        # The new one should be at the end
        assert coordinator._hourly_log[-1]["timestamp"] == current_time.isoformat()
        # The oldest one (id=0) should be gone. The one at index 0 should be id=1
        assert coordinator._hourly_log[0]["id"] == 1

@pytest.mark.asyncio
async def test_csv_append_trigger(hass: HomeAssistant):
    """Test that CSV appending is called during hourly processing."""
    entry = MagicMock()
    entry.data = {"csv_auto_logging": True}

    with patch("custom_components.heating_analytics.storage.Store") as mock_store_cls:
        coordinator = HeatingDataCoordinator(hass, entry)
        coordinator._async_save_data = AsyncMock()
        coordinator.storage.append_hourly_log_csv = AsyncMock()

        coordinator._hourly_sample_count = 1
        coordinator._hourly_wind_values = [0.0]
        coordinator._hourly_log = [{"temp": 0.0}] # For inertia

        current_time = datetime(2023, 10, 27, 13, 0, 0)
        await coordinator._process_hourly_data(current_time)

        coordinator.storage.append_hourly_log_csv.assert_awaited_once()
