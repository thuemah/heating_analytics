"""Test that 'Today' in week ahead stats uses the midnight snapshot if available."""
from datetime import date, datetime, timedelta
import pytest
from unittest.mock import MagicMock, patch

from custom_components.heating_analytics.forecast import ForecastManager
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.const import (
    ATTR_DAILY_FORECAST,
)

@pytest.fixture
def mock_coordinator():
    """Mock the coordinator."""
    coordinator = MagicMock(spec=HeatingDataCoordinator)
    coordinator.hass = MagicMock()
    coordinator.hass.config.time_zone = "UTC"
    coordinator.weather_entity = "weather.test"
    coordinator.entry = MagicMock()
    coordinator.entry.data = {}
    coordinator.data = {}

    # Mock statistics manager
    coordinator.statistics = MagicMock()
    coordinator.statistics._get_daily_log_map = MagicMock(return_value={})

    # Mock calculate_modeled_energy
    coordinator.calculate_modeled_energy = MagicMock(return_value=(10.0, 0.0, 5.0, 5.0, 10.0))

    # Mock inertia helper
    coordinator._get_inertia_list = MagicMock(return_value=[])
    coordinator._calculate_inertia_temp = MagicMock(return_value=5.0)

    # Mock wind bucket
    coordinator._get_wind_bucket = MagicMock(return_value="normal")
    coordinator._is_model_covered = MagicMock(return_value=True)
    coordinator._get_weather_wind_unit = MagicMock(return_value="m/s")
    coordinator._calculate_effective_wind = MagicMock(return_value=0.0)
    coordinator.solar_enabled = False

    return coordinator

@patch("custom_components.heating_analytics.forecast.dt_util.now")
def test_week_ahead_uses_snapshot_for_today(mock_now, mock_coordinator):
    """Test that Today's kwh is overridden by midnight snapshot."""

    # Setup Today
    today = date(2023, 10, 27)
    now_time = datetime(2023, 10, 27, 12, 0, 0)
    mock_now.return_value = now_time

    forecast_manager = ForecastManager(mock_coordinator)

    # 1. Setup Midnight Snapshot
    snapshot_kwh = 50.0
    forecast_manager._midnight_forecast_snapshot = {
        "date": today.isoformat(),
        "kwh": snapshot_kwh,
        # other fields not strictly needed for this test logic
    }

    # 2. Mock get_future_day_prediction to return "live" partial value
    # Returns (kwh, solar, w_stats)
    live_kwh = 20.0
    live_temp = 5.0
    live_wind = 2.0
    w_stats = {
        "temp": live_temp,
        "wind": live_wind,
        "source": "hourly_forecast",
        "final_inertia": []
    }

    # We patch the method on the instance
    forecast_manager.get_future_day_prediction = MagicMock(return_value=(live_kwh, 0.0, w_stats))

    # 3. Run
    stats = forecast_manager.calculate_week_ahead_stats()

    # 4. Verify
    daily_forecast = stats[ATTR_DAILY_FORECAST]
    assert len(daily_forecast) == 7

    today_stats = daily_forecast[0]
    assert today_stats["date"] == today.isoformat()

    # Check that KWH matches snapshot, not live
    # This assertion is expected to FAIL before the fix
    assert today_stats["kwh"] == snapshot_kwh
    assert today_stats["kwh"] != live_kwh

    # Check that Temp/Wind still come from get_future_day_prediction
    assert today_stats["temp"] == live_temp
    assert today_stats["wind"] == live_wind

    # Verify get_future_day_prediction was called
    forecast_manager.get_future_day_prediction.assert_called()
