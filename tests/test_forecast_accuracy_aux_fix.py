"""Test Forecast Accuracy Logic (Aux Correction)."""
import pytest
from unittest.mock import MagicMock
from custom_components.heating_analytics.forecast import ForecastManager

@pytest.fixture
def mock_coordinator():
    coord = MagicMock()
    coord._hourly_log = []
    coord.data = {}
    return coord

@pytest.fixture
def forecast_manager(mock_coordinator):
    fm = ForecastManager(mock_coordinator)
    fm._forecast_history = []
    # Set up a "Gross" Midnight Forecast (Thermodynamic Demand)
    fm._midnight_forecast_snapshot = {
        "date": "2023-01-01",
        "kwh": 50.0, # 50 kWh Demand
        "source": "primary"
    }
    return fm

def test_log_accuracy_with_aux_correction(forecast_manager, mock_coordinator):
    """Test that accuracy calculation correctly accounts for Aux Impact."""
    date_key = "2023-01-01"

    # Scenario:
    # Thermodynamic Demand (Forecast): 50 kWh (Gross)
    # Actual Consumption (Net): 30 kWh
    # Aux Impact (Reduction): 20 kWh

    # We pass Net Actual AND Aux Impact to log_accuracy
    # Expected: Gross Actual = 30 + 20 = 50.
    # Error = 50 - 50 = 0.

    forecast_manager.log_accuracy(date_key, 30.0, aux_impact_kwh=20.0)

    entry = forecast_manager._forecast_history[0]

    # Verify apple-to-apple comparison
    assert entry["forecast_kwh"] == 50.0
    assert entry["actual_kwh"] == 50.0 # Gross Actual
    assert entry["net_actual_kwh"] == 30.0 # Net Actual (Original)
    assert entry["aux_impact_kwh"] == 20.0

    # Error should be 0.0
    assert entry["error_kwh"] == 0.0
    assert entry["abs_error_kwh"] == 0.0
