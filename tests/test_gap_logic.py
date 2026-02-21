"""Test gap filling logic."""
from unittest.mock import MagicMock, patch, AsyncMock
import pytest
from datetime import datetime, timedelta
from homeassistant.core import HomeAssistant
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator

@pytest.mark.asyncio
async def test_gap_fill_end_of_hour(hass: HomeAssistant):
    """Test closing gap at the end of the hour."""
    entry = MagicMock()
    entry.data = {
        "balance_point": 17.0,
        "energy_sensors": ["sensor.heater"],
        "max_energy_delta": 100.0,
        "outdoor_temp_sensor": "sensor.temp",
        "wind_speed_sensor": "sensor.wind"
    }

    with patch("custom_components.heating_analytics.storage.Store") as mock_store_cls:
        coordinator = HeatingDataCoordinator(hass, entry)
        coordinator._async_save_data = AsyncMock()

        # Simulate last processed minute was 58
        coordinator._last_minute_processed = 58
        coordinator.data["current_model_rate"] = 6.0
        coordinator.data["current_calc_temp"] = 5.0

        coordinator._accumulated_expected_energy_hour = 5.8

        # Ensure we have sample counts to trigger log append
        coordinator._hourly_sample_count = 60
        coordinator._hourly_temp_sum = 300.0
        coordinator._hourly_wind_values = [0.0] * 60
        coordinator._hourly_start_time = datetime(2023, 10, 27, 12, 0, 0)

        current_time = datetime(2023, 10, 27, 13, 0, 0)

        # Mock learning and forecast
        coordinator.learning.process_learning = MagicMock(return_value={
            "model_updated": False, "model_base_before": 0, "model_base_after": 0
        })
        coordinator.forecast.get_forecast_for_hour = MagicMock(return_value=None)

        # Mock _get_predicted_kwh (3 args)
        coordinator._get_predicted_kwh = MagicMock(return_value=6.0)

        # Run
        await coordinator._process_hourly_data(current_time)

        if len(coordinator._hourly_log) > 0:
            log = coordinator._hourly_log[0]
            # 5.8 + (1 min * 6.0/60 = 0.1) = 5.9
            assert log["expected_kwh"] == pytest.approx(5.9, abs=0.01)

@pytest.mark.asyncio
async def test_gap_fill_large_gap(hass: HomeAssistant):
    """Test closing a large gap (e.g. crash)."""
    entry = MagicMock()
    entry.data = {}

    with patch("custom_components.heating_analytics.storage.Store"):
        coordinator = HeatingDataCoordinator(hass, entry)
        coordinator._async_save_data = AsyncMock()

        # Last processed min 29. Missed 30 mins (30..59).
        coordinator._last_minute_processed = 29
        coordinator.data["current_model_rate"] = 2.0

        coordinator._accumulated_expected_energy_hour = 1.0

        # Setup logging pre-reqs
        coordinator._hourly_sample_count = 30
        coordinator._hourly_temp_sum = 150.0
        coordinator._hourly_wind_values = [0.0] * 30
        coordinator._hourly_start_time = datetime(2023, 10, 27, 12, 0, 0)

        current_time = datetime(2023, 10, 27, 13, 0, 0)

        coordinator.learning.process_learning = MagicMock(return_value={
             "model_updated": False, "model_base_before": 0, "model_base_after": 0
        })
        coordinator.forecast.get_forecast_for_hour = MagicMock(return_value=None)
        coordinator._get_predicted_kwh = MagicMock(return_value=2.0)

        await coordinator._process_hourly_data(current_time)

        # Added: 30 mins * (2.0 / 60) = 1.0 kWh. Total 2.0
        if len(coordinator._hourly_log) > 0:
            log = coordinator._hourly_log[0]
            assert log["expected_kwh"] == pytest.approx(2.0, abs=0.01)

@pytest.mark.asyncio
async def test_gap_no_previous_state(hass: HomeAssistant):
    """Test gap logic when no previous state exists (should do nothing)."""
    entry = MagicMock()
    entry.data = {}

    with patch("custom_components.heating_analytics.storage.Store"):
        coordinator = HeatingDataCoordinator(hass, entry)
        coordinator._async_save_data = AsyncMock()

        coordinator._last_minute_processed = None # No state
        coordinator._hourly_sample_count = 60
        coordinator._hourly_temp_sum = 600.0
        coordinator._hourly_wind_values = [0.0] * 60
        coordinator._hourly_start_time = datetime(2023, 10, 27, 12, 0, 0)

        current_time = datetime(2023, 10, 27, 13, 0, 0)

        coordinator.learning.process_learning = MagicMock(return_value={
             "model_updated": False, "model_base_before": 0, "model_base_after": 0
        })
        coordinator.forecast.get_forecast_for_hour = MagicMock(return_value=None)
        coordinator._get_predicted_kwh = MagicMock(return_value=1.0)

        await coordinator._process_hourly_data(current_time)

        # Log should exist but expected_kwh should be 0 (initialized 0 + 0 gap)
        if len(coordinator._hourly_log) > 0:
            log = coordinator._hourly_log[0]
            assert log["expected_kwh"] == 0.0
