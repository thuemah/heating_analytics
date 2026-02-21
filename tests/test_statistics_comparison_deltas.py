"""Test Statistics Manager Comparison Deltas."""
from unittest.mock import MagicMock
import pytest
from datetime import date
from custom_components.heating_analytics.statistics import StatisticsManager
from custom_components.heating_analytics.const import (
    ATTR_TEMP_ACTUAL_TODAY,
    ATTR_WIND_ACTUAL_TODAY,
    ATTR_ENERGY_TODAY,
    ATTR_SOLAR_PREDICTED,
)

@pytest.fixture
def mock_coordinator():
    coordinator = MagicMock()
    # Mock daily history with enough data for two periods
    coordinator._daily_history = {
        # Reference Period: Jan 1-2, 2023
        "2023-01-01": {"kwh": 10.0, "temp": 5.0, "wind": 2.0, "tdd": 15.0},
        "2023-01-02": {"kwh": 12.0, "temp": 6.0, "wind": 3.0, "tdd": 14.0},

        # Current Period: Jan 1-2, 2024
        "2024-01-01": {"kwh": 15.0, "temp": 4.0, "wind": 4.0, "tdd": 16.0},
        "2024-01-02": {"kwh": 18.0, "temp": 3.0, "wind": 5.0, "tdd": 17.0},
    }

    # Mock calculate_modeled_energy to return consistent values
    def side_effect_calc(start, end, logs=None):
        if start.year == 2023:
            return (20.0, 0.0, 5.5, 2.5, 29.0) # Model slightly lower than actual (22.0)
        else:
            return (30.0, 0.0, 3.5, 4.5, 33.0) # Model slightly lower than actual (33.0)

    coordinator.calculate_modeled_energy = MagicMock(side_effect=side_effect_calc)

    # Mock calculate_historical_actual_sum
    def side_effect_actual(start, end):
        if start.year == 2023:
            return 22.0 # 10 + 12
        else:
            return 33.0 # 15 + 18

    # Mock forecast for future dates (since conftest sets now to 2023)
    coordinator.forecast = MagicMock()
    def get_forecast_side_effect(date_obj, initial_inertia=None, ignore_aux=False):
        if date_obj.day == 1:
            return (15.0, 0.0, {"temp": 4.0, "wind": 4.0})
        else:
            return (18.0, 0.0, {"temp": 3.0, "wind": 5.0})
    coordinator.forecast.get_future_day_prediction.side_effect = get_forecast_side_effect

    # Mock calculate_future_energy for Today logic (if hit)
    coordinator.forecast.calculate_future_energy.return_value = (5.0, 0.0, {})

    # Mock coordinator data for Today logic
    coordinator.data = {
        ATTR_TEMP_ACTUAL_TODAY: 5.0,
        ATTR_WIND_ACTUAL_TODAY: 2.0,
        ATTR_ENERGY_TODAY: 0.0,
        ATTR_SOLAR_PREDICTED: 0.0,
    }

    # Mock helper method
    coordinator._get_wind_bucket.return_value = "normal"

    return coordinator

def test_compare_periods_deltas(mock_coordinator):
    """Test that compare_periods returns deltas and semantic analysis."""
    stats = StatisticsManager(mock_coordinator)

    stats.calculate_historical_actual_sum = MagicMock(side_effect=lambda s, e: 22.0 if s.year == 2023 else 33.0)
    stats.calculate_modeled_energy = MagicMock(side_effect=lambda s, e: (20.0, 0.0, 5.5, 2.5, 29.0) if s.year == 2023 else (30.0, 0.0, 3.5, 4.5, 33.0))

    # Reference Period (2023) - Past relative to test data, but conftest 'now' is 2023-01-01
    p_ref_start = date(2023, 1, 1)
    p_ref_end = date(2023, 1, 2)

    # Current Period (2024) - Future
    p_curr_start = date(2024, 1, 1)
    p_curr_end = date(2024, 1, 2)

    # Calling convention: compare_periods(Current, Reference)
    # This ensures Deltas = Current - Reference
    result = stats.compare_periods(p_curr_start, p_curr_end, p_ref_start, p_ref_end)

    # Check Period 1 (Current/2024) Data
    assert result["period_1"]["actual_kwh"] == 33.0
    assert result["period_1"]["modeled_kwh"] == 30.0

    # Check Period 2 (Reference/2023) Data
    assert result["period_2"]["actual_kwh"] == 22.0
    assert result["period_2"]["modeled_kwh"] == 20.0

    # Check Deltas (Expectation: Current - Reference)
    assert "delta_actual_kwh" in result
    assert result["delta_actual_kwh"] == 11.0

    assert "delta_modeled_kwh" in result
    assert result["delta_modeled_kwh"] == 10.0

    assert "delta_temp" in result
    assert result["delta_temp"] == -2.0

    assert "delta_wind" in result
    assert result["delta_wind"] == 2.0

    assert "delta_tdd" in result
    assert result["delta_tdd"] == 4.0

    # --- New Semantic Fields ---
    assert "summary" in result
    assert isinstance(result["summary"], str)
    # Check for keywords indicating analysis worked
    assert "Colder" in result["summary"] or "cold" in result["summary"]

    assert "drivers" in result
    assert isinstance(result["drivers"], list)

    assert "characterization" in result
    assert "Colder" in result["characterization"] or "Higher" in result["characterization"]

    # P1 Hybrid (2024): 15.0 + 18.0 = 33.0
    assert "hybrid_total_kwh" in result
    assert result["hybrid_total_kwh"] == 33.0

    # P2 Hybrid (2023): Today (5.0) + Future (18.0) = 23.0
    assert "hybrid_reference_kwh" in result
    assert result["hybrid_reference_kwh"] == 23.0
