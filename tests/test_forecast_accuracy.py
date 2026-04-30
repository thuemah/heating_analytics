"""Forecast accuracy logging — day/night breakdown, gross-vs-net handling,
and aux correction. Consolidated from the previous accuracy_breakdown,
accuracy_gross_logic, and accuracy_aux_fix files."""
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
    """Forecast manager with a 50 kWh midnight snapshot — used by the
    gross-logic and aux-correction paths."""
    fm = ForecastManager(mock_coordinator)
    fm._forecast_history = []
    fm._midnight_forecast_snapshot = {
        "date": "2023-01-01",
        "kwh": 50.0,
        "source": "primary",
    }
    return fm


@pytest.fixture
def forecast_manager_100(mock_coordinator):
    """Variant with 100 kWh midnight snapshot — used by the day/night
    breakdown tests where snapshot magnitude is incidental."""
    fm = ForecastManager(mock_coordinator)
    fm._forecast_history = []
    fm._midnight_forecast_snapshot = {
        "date": "2023-01-01",
        "kwh": 100.0,
        "source": "primary",
    }
    return fm


# =============================================================================
# Day/night breakdown — accuracy is split by hour-of-day (06–22 day, else night)
# =============================================================================

def test_log_accuracy_day_night_split(forecast_manager_100, mock_coordinator):
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
    forecast_manager_100.log_accuracy(date_key, 100.0)

    entry = forecast_manager_100._forecast_history[0]

    # Check breakdown
    # Day MAE: Hour 12 error is 5. Count 1. MAE = 5.0
    assert entry["day_mae"] == 5.0

    # Night MAE: Hour 0 error is 10. Count 1. MAE = 10.0
    assert entry["night_mae"] == 10.0


def test_log_accuracy_no_day_data(forecast_manager_100, mock_coordinator):
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

    forecast_manager_100.log_accuracy(date_key, 100.0)

    entry = forecast_manager_100._forecast_history[0]
    assert entry["night_mae"] == 10.0
    assert entry["day_mae"] is None


# =============================================================================
# Gross vs net logic — legacy entries fall back to net comparison; new
# entries with forecasted_kwh_gross use gross-vs-gross + weather-error split
# =============================================================================

def test_log_accuracy_legacy_log_net_fallback(forecast_manager, mock_coordinator):
    """Test legacy behavior: Net vs Net error, but Gross Actual accumulation."""
    date_key = "2023-01-01"

    # Scenario:
    # Gross Demand: 50. Aux: 20.
    # Net Actual: 30. Net Forecast: 30.
    # Error (Net vs Net): 0.

    log = {
        "timestamp": "2023-01-01T12:00:00",
        "hour": 12,
        "actual_kwh": 30.0, # Net
        "forecasted_kwh": 30.0, # Net
        "aux_impact_kwh": 20.0,
        "expected_kwh": 30.0, # Modeled Net
        # Missing forecasted_kwh_gross
        "forecasted_kwh_primary": 30.0,
        "_source": "primary"
    }
    mock_coordinator._hourly_log = [log]

    # Global Call: Actual Net 30, Aux 20 -> Gross 50.
    forecast_manager.log_accuracy(date_key, 30.0, aux_impact_kwh=20.0)

    entry = forecast_manager._forecast_history[0]

    # Check Source Breakdown
    # Actual should be Gross (30 + 20 = 50)
    primary_stats = entry["source_breakdown"]["primary"]
    assert primary_stats["actual"] == 50.0

    # Error should be Net vs Net (30 - 30 = 0)
    # (Since we lack Gross Forecast)
    assert primary_stats["abs_error"] == 0.0

    # Weather Error should be 0 because gross forecast is missing
    assert primary_stats["weather_error"] == 0.0


def test_log_accuracy_new_log_gross_logic(forecast_manager, mock_coordinator):
    """Test new behavior: Gross vs Gross error and Weather Error."""
    date_key = "2023-01-01"

    # Scenario 1: Perfect Alignment
    # Gross Demand: 50. Aux: 20.
    # Net Actual: 25 (User saved 5kWh via efficiency?). Gross Actual: 45.
    # Net Forecast: 30 (Gross 50 - Model Aux 20).
    # Gross Forecast: 50.
    # Net Model (Expected): 30.

    log = {
        "timestamp": "2023-01-01T12:00:00",
        "hour": 12,
        "actual_kwh": 25.0, # Net
        "forecasted_kwh": 30.0, # Net
        "expected_kwh": 30.0, # Net Model
        "aux_impact_kwh": 20.0, # Actual Aux
        "forecasted_kwh_gross": 50.0, # New Field!
        "forecasted_kwh_gross_primary": 50.0, # New Field!
        "forecasted_kwh_primary": 30.0,
        "_source": "primary"
    }
    mock_coordinator._hourly_log = [log]

    # Global Call: Net 25, Aux 20 -> Gross 45.
    forecast_manager.log_accuracy(date_key, 25.0, aux_impact_kwh=20.0)

    entry = forecast_manager._forecast_history[0]

    # Check Source Breakdown
    primary_stats = entry["source_breakdown"]["primary"]

    # Actual should be Gross (25 + 20 = 45)
    assert primary_stats["actual"] == 45.0

    # Error should be Gross vs Gross (45 - 50 = -5)
    # Abs Error = 5.0
    assert primary_stats["abs_error"] == 5.0

    # Weather Error: Modeled Gross (30+20=50) - Forecast Gross (50) = 0
    assert primary_stats["weather_error"] == 0.0

    # Scenario 2: Weather Error Present
    # Let's try a case where Net vs Net differs from Gross vs Gross
    # Model Aux: 10. Actual Aux: 20.
    # Gross Forecast: 50. Net Forecast: 40.
    # Gross Actual: 50. Net Actual: 30.
    # Net Model (Expected): 40.

    # Wait, if Expected (Net) is 40 and Aux Impact is 20:
    # Modeled Gross = 40 + 20 = 60.
    # Forecast Gross = 50.
    # Weather Error = 60 - 50 = 10.

    log2 = {
        "timestamp": "2023-01-01T12:00:00",
        "hour": 12,
        "actual_kwh": 30.0, # Net
        "forecasted_kwh": 40.0, # Net (Model Aux 10)
        "expected_kwh": 40.0, # Net Model
        "aux_impact_kwh": 20.0, # Actual Aux (Modeled impact from logs)
        "forecasted_kwh_gross": 50.0, # Gross
        "forecasted_kwh_gross_primary": 50.0,
        "forecasted_kwh_primary": 40.0,
        "_source": "primary"
    }
    mock_coordinator._hourly_log = [log2]
    forecast_manager._forecast_history = []

    forecast_manager.log_accuracy(date_key, 30.0, aux_impact_kwh=20.0)
    entry2 = forecast_manager._forecast_history[0]
    stats2 = entry2["source_breakdown"]["primary"]

    assert stats2["actual"] == 50.0 # 30+20
    assert stats2["abs_error"] == 0.0 # 50-50 (Correct Thermodynamic Error)

    # Weather Error check
    # Modeled Gross (40+20=60) - Forecast Gross (50) = 10.0
    assert stats2["weather_error"] == 10.0
    assert stats2["abs_weather_error"] == 10.0


# =============================================================================
# Aux correction — log_accuracy reconstructs gross actual from net + aux impact
# so the comparison stays apples-to-apples against the gross thermodynamic
# midnight forecast.
# =============================================================================

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
