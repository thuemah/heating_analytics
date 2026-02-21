"""Test Weather Error Statistics Logic."""
import pytest
from unittest.mock import MagicMock
from custom_components.heating_analytics.forecast import ForecastManager
from datetime import timedelta
from homeassistant.util import dt as dt_util

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
    return fm

def test_weather_error_stats_calculation(forecast_manager):
    """Test calculation of MAE and Bias for Weather Error over a period."""
    today = dt_util.now().date()

    # Generate 5 days of history
    # Day 1: Weather Error +10
    # Day 2: Weather Error -10
    # Day 3: Weather Error +5
    # Day 4: Weather Error -5
    # Day 5: Weather Error 0

    history = []
    errors = [10.0, -10.0, 5.0, -5.0, 0.0]

    for i, err in enumerate(errors):
        date_val = (today - timedelta(days=i+1)).isoformat()

        breakdown = {
            "primary": {
                "weather_error": err,
                "abs_weather_error": abs(err), # Hourly absolute, but we use weather_error for daily calc
                "hours": 24,
                "actual": 100.0,
                "forecast": 100.0,
                "error": 0.0,
                "abs_error": 0.0
            }
        }

        entry = {
            "date": date_val,
            "primary_entity": "weather.primary",
            "source_breakdown": breakdown,
            "source": "primary" # dominant
        }
        history.append(entry)

    forecast_manager._forecast_history = history

    # Calculate stats for 7 days
    # Daily Weather Errors: 10, -10, 5, -5, 0
    # Abs Daily Errors: 10, 10, 5, 5, 0. Sum = 30. Avg = 30/5 = 6.0
    # Signed Daily Errors: 10 - 10 + 5 - 5 + 0 = 0. Avg = 0.

    stats = forecast_manager._calculate_period_stats("primary", 7, "weather.primary")

    daily_stats = stats["daily"]
    assert daily_stats["weather_mae_7d"] == 6.0
    assert daily_stats["weather_bias_7d"] == 0.0

def test_weather_error_stats_bias_check(forecast_manager):
    """Test Bias calculation specifically."""
    today = dt_util.now().date()

    # All positive errors (Systematic Bias)
    errors = [10.0, 10.0]

    history = []
    for i, err in enumerate(errors):
        date_val = (today - timedelta(days=i+1)).isoformat()
        breakdown = {
            "primary": {
                "weather_error": err,
                "abs_weather_error": abs(err),
                "hours": 24,
                "actual": 100.0,
                "forecast": 100.0,
                "error": 0.0,
                "abs_error": 0.0
            }
        }
        entry = {
            "date": date_val,
            "primary_entity": "weather.primary",
            "source_breakdown": breakdown,
            "source": "primary"
        }
        history.append(entry)

    forecast_manager._forecast_history = history
    stats = forecast_manager._calculate_period_stats("primary", 7, "weather.primary")

    # Avg Abs: 10. Avg Signed: 10.
    assert stats["daily"]["weather_mae_7d"] == 10.0
    assert stats["daily"]["weather_bias_7d"] == 10.0

def test_weather_error_legacy_missing_key(forecast_manager):
    """Test robustness when weather_error key is missing."""
    today = dt_util.now().date()

    # Legacy entry
    entry = {
        "date": (today - timedelta(days=1)).isoformat(),
        "primary_entity": "weather.primary",
        "source_breakdown": {
            "primary": {
                "hours": 24,
                "actual": 100.0,
                "forecast": 100.0,
                "error": 0.0,
                "abs_error": 0.0
                # Missing weather_error
            }
        },
        "source": "primary"
    }

    forecast_manager._forecast_history = [entry]
    stats = forecast_manager._calculate_period_stats("primary", 7, "weather.primary")

    # Should be 0.0 and safe
    assert stats["daily"]["weather_mae_7d"] == 0.0
    assert stats["daily"]["weather_bias_7d"] == 0.0
