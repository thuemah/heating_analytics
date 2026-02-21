"""Test HeatingModelComparisonWeekSensor for Hybrid correctness."""
from unittest.mock import MagicMock, patch
from datetime import date, datetime, timedelta
import pytest
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from custom_components.heating_analytics.sensors.comparison import HeatingModelComparisonWeekSensor
from custom_components.heating_analytics.const import ATTR_ENERGY_TODAY, ATTR_SOLAR_PREDICTED, ATTR_TEMP_ACTUAL_TODAY, ATTR_WIND_ACTUAL_TODAY

@pytest.fixture
def mock_coordinator():
    coordinator = MagicMock()
    coordinator.data = {}
    coordinator._daily_history = {}
    return coordinator

@pytest.fixture
def mock_entry():
    entry = MagicMock()
    entry.entry_id = "test_entry"
    return entry

@pytest.mark.asyncio
async def test_comparison_sensor_hybrid_calculation(hass: HomeAssistant, mock_coordinator, mock_entry):
    """Test that current_hybrid_kwh uses Actuals + Forecast, while current_model_kwh uses Model."""

    # Setup Dates
    # Assume today is Wednesday (weekday 2), so week started Monday (2 days ago).
    # Past days: Mon, Tue. Today: Wed. Future: Thu-Sun.
    today = date(2023, 10, 25) # Wednesday
    start_week = date(2023, 10, 23) # Monday

    with patch("homeassistant.util.dt.now", return_value=datetime(2023, 10, 25, 12, 0, 0)):
        # Setup Coordinator Data

        # 1. Past Data (Mon, Tue) in _daily_history
        # Actual kWh = 100.0, Modeled kWh = 80.0
        mock_coordinator._daily_history = {
            "2023-10-23": {"temp": 5.0, "wind": 3.0, "kwh": 100.0},
            "2023-10-24": {"temp": 5.0, "wind": 3.0, "kwh": 100.0}
        }

        # 2. Today Data (Wed)
        # Actual So Far = 50.0
        mock_coordinator.data = {
            ATTR_ENERGY_TODAY: 50.0,
            ATTR_TEMP_ACTUAL_TODAY: 5.0,
            ATTR_WIND_ACTUAL_TODAY: 3.0,
            ATTR_SOLAR_PREDICTED: 0.0
        }

        # 3. Forecast Remaining for Today
        # Assume 40.0 remaining
        mock_coordinator.forecast.calculate_future_energy.return_value = (40.0, 0.0, {})

        # 4. Forecast Future Days (Thu, Fri, Sat, Sun)
        # 4 days * 80.0 = 320.0
        # Mock get_future_day_prediction to return 80.0 per day
        mock_coordinator.forecast.get_future_day_prediction.side_effect = lambda d, i=None, ignore_aux=False: (80.0, 0.0, {"temp": 5.0, "wind": 3.0})

        # 5. Mock calculate_modeled_energy (used for Model Comparison / Fallback)
        # Should return 80.0 per day (Modeled)
        mock_coordinator.calculate_modeled_energy.return_value = (80.0, 0.0, 5.0, 3.0, 10.0)

        # 6. Mock calculate_hybrid_projection (used for current_model_kwh via _calculate_period_stats)
        # Since I didn't change statistics.py, this will return Pure Model.
        # Week Total: 7 days * 80.0 = 560.0
        mock_coordinator.statistics.calculate_hybrid_projection.return_value = (560.0, 0.0)

        # 7. Mock _get_wind_bucket
        mock_coordinator._get_wind_bucket.return_value = "normal"

        # 8. Mock calculate_historical_actual_sum (Last Year Actuals)
        mock_coordinator.statistics.calculate_historical_actual_sum.return_value = 500.0

        # 9. Mock WeatherImpactAnalyzer (to prevent error during analysis)
        # Since the sensor uses it internally, we need to ensure it doesn't crash.
        # It relies on coordinator methods, which are mostly mocked above.

        # Create Sensor
        sensor = HeatingModelComparisonWeekSensor(mock_coordinator, mock_entry)
        sensor.hass = hass
        sensor.async_write_ha_state = MagicMock()

        # Force update (trigger property access)
        attrs = sensor.extra_state_attributes

        # Assertions

        # current_model_kwh should come from calculate_hybrid_projection (Pure Model)
        assert attrs["current_model_kwh"] == 560.0

        # current_hybrid_kwh should be:
        # Past (Mon, Tue Actual): 100.0 + 100.0 = 200.0
        # Today (Wed Actual + Forecast): 50.0 + 40.0 = 90.0
        # Future (Thu-Sun Forecast): 80.0 * 4 = 320.0
        # Total Hybrid: 200 + 90 + 320 = 610.0

        # Note: My mocks for get_future_day_prediction are called for Thu-Sun (4 days).
        # _build_current_period_days iterates from start_week (Mon) to end_week (Sun).
        # Mon: _get_historical_day -> Uses actual kwh (100.0)
        # Tue: _get_historical_day -> Uses actual kwh (100.0)
        # Wed: _get_today_data -> Uses actual_so_far (50.0) + future (40.0) = 90.0
        # Thu: _get_forecast_day -> 80.0
        # Fri: _get_forecast_day -> 80.0
        # Sat: _get_forecast_day -> 80.0
        # Sun: _get_forecast_day -> 80.0

        # Total = 100+100+90+320 = 610.0

        assert attrs["current_hybrid_kwh"] == 610.0

        # Verify Hybrid is DIFFERENT from Model
        assert attrs["current_hybrid_kwh"] != attrs["current_model_kwh"]
