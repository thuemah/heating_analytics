"""Test to reproduce the Week 53 boundary issue in Forecast Manager."""
from datetime import date, datetime
import pytest
from unittest.mock import MagicMock, patch

from custom_components.heating_analytics.forecast import ForecastManager

# Test fixture only — production runtime uses an exponential kernel from
# CONF_THERMAL_INERTIA via generate_exponential_kernel.  These 4 weights
# are an arbitrary mock value satisfying the weighted-average shape.
_INERTIA_WEIGHTS_FIXTURE = (0.20, 0.30, 0.30, 0.20)

@patch("custom_components.heating_analytics.forecast.dt_util.now")
def test_calculate_week_ahead_stats_boundary_failure(mock_now, mock_coordinator):
    """Test that calculate_week_ahead_stats fails to fetch correct range on Week 53 boundary."""
    # Specialized setup for this test
    mock_coordinator.hass.config.time_zone = "UTC"
    mock_coordinator.weather_entity = "weather.test"
    mock_coordinator.inertia_weights = _INERTIA_WEIGHTS_FIXTURE
    
    # We also need calculate_modeled_energy to return something so the method doesn't crash
    mock_coordinator.calculate_modeled_energy = MagicMock(return_value=(10.0, 0.0, 5.0, 5.0, 10.0))

    # Mock inertia helper
    mock_coordinator._get_inertia_list = MagicMock(return_value=[])
    mock_coordinator._calculate_inertia_temp = MagicMock(return_value=5.0)

    # Mock wind bucket
    mock_coordinator._get_wind_bucket = MagicMock(return_value="normal")

    # Setup: Sunday, Dec 27, 2020 (Week 52, Day 7)
    # Next day is Monday, Dec 28, 2020 (Week 53, Day 1)
    # Previous year (2019) has 52 weeks.
    # W52-D7 (2019) -> 2019-12-29
    # W53-D1 (2019) -> Fallback to W52-D1 -> 2019-12-23

    start_date = datetime(2020, 12, 27, 12, 0, 0)
    mock_now.return_value = start_date

    forecast_manager = ForecastManager(mock_coordinator)

    # Run the method
    forecast_manager.calculate_week_ahead_stats()

    # Verify the call to _get_daily_log_map
    # We expect the current buggy implementation to call it with start > end
    # Start: 2019-12-29
    # End (Start + 6 days = 2021-01-02 -> W53-D6 -> Fallback W52-D6): 2019-12-28

    mock_coordinator.statistics._get_daily_log_map.assert_called_once()
    args, _ = mock_coordinator.statistics._get_daily_log_map.call_args
    call_start, call_end = args

    print(f"Called with: start={call_start}, end={call_end}")

    # Assertion: The fixed code should call with min <= max covering the full range
    # Min date: 2019-12-23 (W53-D1 -> Fallback W52-D1)
    # Max date: 2019-12-29 (W52-D7)

    # We expect the range to be valid (start <= end)
    assert call_start <= call_end, f"Range invalid: start {call_start} > end {call_end}"

    assert call_start.isoformat() == "2019-12-23"
    assert call_end.isoformat() == "2019-12-29"
