"""Test Guest Mode impact on Forecast Accuracy."""
import pytest
from unittest.mock import MagicMock
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
    # Midnight snapshot predicts 50 kWh
    fm._midnight_forecast_snapshot = {
        "date": "2023-01-01",
        "kwh": 50.0,
        "source": "primary"
    }
    return fm

def test_log_accuracy_excludes_guest_energy(forecast_manager, mock_coordinator):
    """Test that accuracy calculation excludes guest energy."""

    date_key = "2023-01-01"

    # Scenario:
    # Forecast: 50 kWh (Base)
    # Actual Base: 50 kWh (Perfect match)
    # Guest Load: 10 kWh
    # Total Actual: 60 kWh

    # Hourly Log Entry (Simplified 1-hour day for clarity)
    logs = [{
        "timestamp": "2023-01-01T12:00:00",
        "hour": 12,
        "actual_kwh": 60.0, # Includes Guest
        "expected_kwh": 50.0, # Base
        "forecasted_kwh": 50.0, # Forecast for Base
        "guest_impact_kwh": 10.0,
        "aux_impact_kwh": 0.0,
        "thermodynamic_gross_kwh": 60.0 # Includes Guest
    }]

    mock_coordinator._hourly_log = logs

    # Call log_accuracy
    forecast_manager.log_accuracy(
        date_key,
        actual_kwh=60.0,
        aux_impact_kwh=0.0,
        modeled_net_kwh=50.0,
        guest_impact_kwh=10.0
    )

    entry = forecast_manager._forecast_history[0]

    # Verify guest impact is stored
    assert entry["guest_impact_kwh"] == 10.0

    # Without fix:
    # Gross Actual = Actual (60) + Aux (0) = 60
    # Forecast = 50
    # Error = 60 - 50 = 10

    # With fix:
    # Gross Actual = Actual (60) - Guest (10) + Aux (0) = 50
    # Error = 50 - 50 = 0

    assert entry["error_kwh"] == 0.0, f"Error should be 0.0 (Guest excluded), got {entry['error_kwh']}"
    assert entry["abs_error_kwh"] == 0.0

    # Verify breakdown (hourly check)
    # Day MAE should be 0.0
    assert entry["day_mae"] == 0.0, f"Day MAE should be 0.0, got {entry['day_mae']}"
