"""Test for load trend calculation."""
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
from custom_components.heating_analytics.forecast import ForecastManager
from homeassistant.util import dt as dt_util

@pytest.fixture
def mock_coordinator():
    """Mock Coordinator."""
    coord = MagicMock()
    coord.data = {}
    coord._hourly_log = []
    # Mock methods required by ForecastManager
    coord._get_weather_wind_unit.return_value = "km/h"
    coord._get_cloud_coverage.return_value = 50.0
    coord._calculate_weighted_inertia.return_value = 10.0
    coord._calculate_effective_wind.return_value = 5.0
    coord._get_wind_bucket.return_value = "normal"
    # Mock statistics
    coord.statistics = MagicMock()
    coord.statistics.calculate_total_power.return_value = {
        "total_kwh": 5.0, # Default future pred
        "breakdown": {"solar_reduction_kwh": 0.0},
        "unit_breakdown": {}
    }

    # Mock inertia list
    coord._get_inertia_list.return_value = [10.0, 10.0, 10.0]

    return coord

def test_load_trend_increasing(mock_coordinator):
    """Test load trend calculation when increasing."""
    forecast = ForecastManager(mock_coordinator)

    # Setup Past Data (Low Load)
    # Current Hour (h)
    mock_coordinator.data["current_model_rate"] = 2.0

    # Past Hours (h-1, h-2)
    current_hour = 12

    mock_coordinator._hourly_log = [
        {"hour": 10, "expected_kwh": 1.0}, # h-2
        {"hour": 11, "expected_kwh": 1.5}, # h-1
    ]

    # Past Sum = 1.0 + 1.5 + 2.0 = 4.5
    # Avg Past = 1.5

    # Setup Future Data (High Load)
    # We mock _sum_forecast_energy_internal to return a plan
    # h+1, h+2, h+3
    hourly_plan = [
        {"hour": 13, "kwh": 3.0},
        {"hour": 14, "kwh": 4.0},
        {"hour": 15, "kwh": 5.0},
    ]

    forecast._sum_forecast_energy_internal = MagicMock(return_value={
        "hourly_plan": hourly_plan
    })

    # Mock dt_util.now()
    with patch("homeassistant.util.dt.now") as mock_now:
        mock_now.return_value = datetime(2023, 10, 27, 12, 30, 0)

        trend = forecast.calculate_load_trend()

        # Future Avg = (3+4+5)/3 = 4.0
        # Past Avg = 1.5
        # Delta = 2.5
        # Pct = (2.5/1.5)*100 = 166%

        assert trend == "Increasing (Fast)"

def test_load_trend_easing(mock_coordinator):
    """Test load trend calculation when decreasing."""
    forecast = ForecastManager(mock_coordinator)

    # Setup Past Data (High Load)
    mock_coordinator.data["current_model_rate"] = 5.0

    mock_coordinator._hourly_log = [
        {"hour": 10, "expected_kwh": 6.0},
        {"hour": 11, "expected_kwh": 5.5},
    ]

    # Past Sum = 6.0 + 5.5 + 5.0 = 16.5
    # Avg Past = 5.5

    # Setup Future Data (Low Load)
    hourly_plan = [
        {"hour": 13, "kwh": 4.0},
        {"hour": 14, "kwh": 3.0},
        {"hour": 15, "kwh": 2.0},
    ]

    forecast._sum_forecast_energy_internal = MagicMock(return_value={
        "hourly_plan": hourly_plan
    })

    with patch("homeassistant.util.dt.now") as mock_now:
        mock_now.return_value = datetime(2023, 10, 27, 12, 30, 0)

        trend = forecast.calculate_load_trend()

        # Future Avg = 3.0
        # Past Avg = 5.5
        # Delta = -2.5
        # Pct = -45%

        assert trend == "Easing (Fast)"

def test_load_trend_stable(mock_coordinator):
    """Test load trend calculation when stable."""
    forecast = ForecastManager(mock_coordinator)

    mock_coordinator.data["current_model_rate"] = 2.0
    mock_coordinator._hourly_log = [
        {"hour": 10, "expected_kwh": 2.0},
        {"hour": 11, "expected_kwh": 2.0},
    ]

    hourly_plan = [
        {"hour": 13, "kwh": 2.05},
        {"hour": 14, "kwh": 1.95},
        {"hour": 15, "kwh": 2.0},
    ]

    forecast._sum_forecast_energy_internal = MagicMock(return_value={
        "hourly_plan": hourly_plan
    })

    with patch("homeassistant.util.dt.now") as mock_now:
        mock_now.return_value = datetime(2023, 10, 27, 12, 30, 0)
        trend = forecast.calculate_load_trend()
        assert trend == "Stable"
