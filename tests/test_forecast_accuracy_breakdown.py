"""Test Forecast Accuracy Breakdown Logic."""
import pytest
from unittest.mock import MagicMock
from datetime import date
from custom_components.heating_analytics.forecast import ForecastManager

@pytest.fixture
def mock_coordinator():
    coord = MagicMock()
    coord._hourly_log = []
    # Mock snapshot
    coord.data = {}
    return coord

@pytest.fixture
def forecast_manager(mock_coordinator):
    fm = ForecastManager(mock_coordinator)
    fm._forecast_history = []
    fm._midnight_forecast_snapshot = {
        "date": "2023-01-01",
        "kwh": 100.0,
        "source": "primary"
    }
    return fm

def test_log_accuracy_day_night_split(forecast_manager, mock_coordinator):
    """Test that accuracy is split correctly between day and night."""

    date_key = "2023-01-01"

    # Create logs
    # Day: 06-22. Night: 00-06, 22-24.
    logs = []

    # Night: 00:00 -> Error = 10 (Actual 20, Forecast 10)
    logs.append({
        "timestamp": "2023-01-01T00:00:00",
        "hour": 0,
        "actual_kwh": 20.0,
        "forecasted_kwh": 10.0
    })

    # Day: 12:00 -> Error = 5 (Actual 15, Forecast 10)
    logs.append({
        "timestamp": "2023-01-01T12:00:00",
        "hour": 12,
        "actual_kwh": 15.0,
        "forecasted_kwh": 10.0
    })

    mock_coordinator._hourly_log = logs

    # Run
    forecast_manager.log_accuracy(date_key, 100.0)

    entry = forecast_manager._forecast_history[0]

    # Check breakdown
    # Day MAE: Hour 12 error is 5. Count 1. MAE = 5.0
    assert entry["day_mae"] == 5.0

    # Night MAE: Hour 0 error is 10. Count 1. MAE = 10.0
    assert entry["night_mae"] == 10.0

def test_log_accuracy_no_day_data(forecast_manager, mock_coordinator):
    """Test handling of missing day data (e.g. partial day logs)."""
    date_key = "2023-01-01"

    # Only Night log
    logs = [{
        "timestamp": "2023-01-01T00:00:00",
        "hour": 0,
        "actual_kwh": 20.0,
        "forecasted_kwh": 10.0
    }]
    mock_coordinator._hourly_log = logs

    forecast_manager.log_accuracy(date_key, 100.0)

    entry = forecast_manager._forecast_history[0]
    assert entry["night_mae"] == 10.0
    assert entry["day_mae"] is None
