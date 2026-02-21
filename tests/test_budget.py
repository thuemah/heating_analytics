"""Test the Daily Budget calculation logic."""
from unittest.mock import MagicMock, patch, AsyncMock
import pytest
from homeassistant.core import HomeAssistant
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.const import ATTR_PREDICTED, ATTR_EXPECTED_TODAY, ATTR_FORECAST_TODAY

@pytest.mark.asyncio
async def test_daily_budget_calculation_integrated(hass: HomeAssistant):
    """Test the daily budget calculation within _async_update_data."""
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.data = {
        "balance_point": 17.0,
        "outdoor_temp_sensor": "sensor.temp",
        "wind_speed_sensor": "sensor.wind"
    }

    # Mock Store and external calls
    with patch("custom_components.heating_analytics.storage.Store") as mock_store_cls, \
         patch("custom_components.heating_analytics.coordinator.dt_util.now") as mock_now:

        # Configure Store mock to return an AsyncMock for async_load
        mock_store = mock_store_cls.return_value
        mock_store.async_load = AsyncMock(return_value={})
        mock_store.async_save = AsyncMock()

        coordinator = HeatingDataCoordinator(hass, entry)

        # Setup Time: 12:00 PM
        mock_now.return_value.hour = 12
        mock_now.return_value.minute = 0
        mock_now.return_value.date.return_value.isoformat.return_value = "2023-10-27"
        mock_now.return_value.isoformat.return_value = "2023-10-27T12:00:00"

        # 1. Setup Past Logs (00:00 - 11:00)
        # 12 hours passed. Let's say 2 logs exist for today.
        coordinator._hourly_log = [
            {"timestamp": "2023-10-27T10:00:00", "hour": 10, "expected_kwh": 1.0, "temp": 0.0, "wind_bucket": "normal", "effective_wind": 0.0},
            {"timestamp": "2023-10-27T11:00:00", "hour": 11, "expected_kwh": 1.0, "temp": 0.0, "wind_bucket": "normal", "effective_wind": 0.0}
        ]
        # expected_today_sum = 2.0

        # 2. Setup Current Hour (12:00)
        # Temp = 0.0. Model: Temp 0 -> 1.0 kWh/h
        coordinator._correlation_data = {
            "0": {"normal": 1.0}
        }

        # Mock States
        coordinator._get_float_state = MagicMock(side_effect=lambda entity: 0.0 if entity == "sensor.temp" else 0.0)
        coordinator._get_speed_in_ms = MagicMock(return_value=0.0)
        coordinator._get_cloud_coverage = MagicMock(return_value=50.0)
        coordinator._get_sun_info_now = MagicMock(return_value=(0,0))
        coordinator._calculate_inertia_temp = MagicMock(return_value=0.0)

        # Mock Future Forecast (13:00 onwards)
        # Return tuple (kwh, solar_kwh)
        coordinator.forecast.calculate_future_energy = MagicMock(return_value=(5.0, 0.0, {}))

        # Mock Async Save
        coordinator._async_save_data = AsyncMock()

        # Mock Breakdown (new feature)
        coordinator._calculate_deviation_breakdown = MagicMock(return_value=[])

        # Run Update
        await coordinator._async_update_data()

        # Verification
        # Budget = Past (2.0) + Current Full Hour (1.0) + Future (5.0) = 8.0
        assert coordinator.data[ATTR_PREDICTED] == 8.0

        # ATTR_EXPECTED_TODAY = Past (2.0) + Current So Far (1 min accumulated)
        # 1.0 kWh/h / 60 = 0.01666... -> 0.017 rounded
        # Total = 2.0 + 0.017 = 2.017
        assert coordinator.data[ATTR_EXPECTED_TODAY] == 2.017

        # ATTR_FORECAST_TODAY = Actual So Far (0.0) + Remaining Current (1.0) + Future (5.0) = 6.0
        # (Assuming accumulated energy starts at 0)
        assert coordinator.data[ATTR_FORECAST_TODAY] == 6.0
