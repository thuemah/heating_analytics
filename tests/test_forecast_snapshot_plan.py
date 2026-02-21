"""Test that the midnight forecast snapshot includes the hourly plan."""
from datetime import datetime, timedelta
import pytest
from unittest.mock import MagicMock, patch

from custom_components.heating_analytics.forecast import ForecastManager
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator

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

    # Mock helpers needed for internal logic
    coordinator._get_weather_wind_unit = MagicMock(return_value="m/s")
    coordinator._get_cloud_coverage = MagicMock(return_value=50.0)
    coordinator.energy_sensors = [] # No units for simplicity
    coordinator.get_unit_mode = MagicMock(return_value="normal")
    coordinator._hourly_log = [] # No history
    coordinator.auxiliary_heating_active = False # Default state

    return coordinator

@patch("custom_components.heating_analytics.forecast.dt_util.start_of_local_day")
@patch("custom_components.heating_analytics.forecast.dt_util.now")
def test_snapshot_generation_includes_hourly_plan(mock_now, mock_start_of_day, mock_coordinator):
    """Test _capture_daily_forecast_snapshot generates hourly_plan."""

    # Setup time
    now_time = datetime(2023, 10, 27, 0, 5, 0)
    mock_now.return_value = now_time
    mock_start_of_day.return_value = datetime(2023, 10, 27, 0, 0, 0)

    fm = ForecastManager(mock_coordinator)

    # Setup Reference Forecast (2 items for simplicity)
    # Note: Logic filters by start_time (00:00) -> end_time (Next Day 00:00)
    fm._reference_forecast = [
        {"datetime": "2023-10-27T00:00:00", "temperature": 10.0},
        {"datetime": "2023-10-27T01:00:00", "temperature": 9.0}
    ]

    # Mock _process_forecast_item to return controllable values
    # Returns: (predicted_kwh, solar_kwh, inertia_val, raw_temp, w_speed, w_speed_ms, unit_breakdown)
    # We'll use side_effect to vary return values
    def process_side_effect(item, *args, **kwargs):
        temp = float(item["temperature"])
        if temp == 10.0:
            return (1.5, 0.0, 5.0, 10.0, 0.0, 0.0, {})
        else:
            return (2.0, 0.0, 4.0, 9.0, 0.0, 0.0, {})

    with patch.object(fm, '_process_forecast_item', side_effect=process_side_effect) as mock_process:
        snapshot = fm._capture_daily_forecast_snapshot()

        # Verify call args
        assert mock_process.call_count == 2

        # Verify Snapshot Structure
        assert "hourly_plan" in snapshot
        plan = snapshot["hourly_plan"]
        assert len(plan) == 2

        # Verify Item 0 (Hour 00)
        item0 = plan[0]
        assert item0["hour"] == 0
        assert item0["kwh"] == 1.5
        assert item0["inertia_temp"] == 5.0
        assert item0["aux_expected"] is False # Default ignore_aux=True

        # Verify Item 1 (Hour 01)
        item1 = plan[1]
        assert item1["hour"] == 1
        assert item1["kwh"] == 2.0
        assert item1["inertia_temp"] == 4.0
        assert item1["aux_expected"] is False
