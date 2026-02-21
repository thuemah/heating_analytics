"""Test Vector Reconstruction Logic in StatisticsManager."""
import pytest
from unittest.mock import MagicMock
from datetime import date
from custom_components.heating_analytics.statistics import StatisticsManager

@pytest.fixture
def mock_coordinator():
    coord = MagicMock()
    coord._daily_history = {}
    coord._hourly_log = []
    # Mock balance point for reconstruction logic
    coord.balance_point = 15.0
    return coord

@pytest.fixture
def stats_manager(mock_coordinator):
    return StatisticsManager(mock_coordinator)

def test_calculate_modeled_energy_uses_vectors(stats_manager, mock_coordinator):
    """Test that vectors are used when available in daily history."""

    # Setup Daily History with Vectors
    # Scenario: Cold Night (High Load), Warm Day (Low Load)
    # Vectors represent 24h
    vectors = {
        "temp": [0.0] * 12 + [20.0] * 12, # 0C night, 20C day
        "load": [10.0] * 12 + [0.0] * 12, # 10kWh night, 0kWh day
        "wind": [5.0] * 24,
        "tdd": [0.625] * 12 + [0.0] * 12, # (15-0)/24 = 0.625 per hour
        "solar_rad": [0.0] * 24
    }

    mock_coordinator._daily_history = {
        "2023-01-01": {
            "kwh": 120.0,
            "temp": 10.0, # Daily Avg
            "wind": 5.0,
            "hourly_vectors": vectors
        }
    }

    # Mock calculate_total_power to return simple linear model: Load = (15 - Temp)
    # Night: 15-0 = 15. Day: 15-20 = -5 -> 0.
    def side_effect(temp, effective_wind, solar_impact, is_aux_active, unit_modes=None, override_solar_factor=None, detailed=True):
        base = max(0.0, 15.0 - temp)
        return {
            "total_kwh": base,
            "breakdown": {"solar_reduction_kwh": 0.0}
        }

    stats_manager.calculate_total_power = MagicMock(side_effect=side_effect)

    # Run Calculation
    total_kwh, _, avg_temp, _, _ = stats_manager.calculate_modeled_energy(
        date(2023, 1, 1), date(2023, 1, 1)
    )

    # Verify:
    # If using Daily Avg (10C): Load = 15 - 10 = 5. Total = 5 * 24 = 120 kWh.
    # If using Vectors:
    # Night (12h): 15 - 0 = 15. Sum = 15 * 12 = 180.
    # Day (12h): 15 - 20 = 0. Sum = 0.
    # Total = 180 kWh.

    # The actual calculation in calculate_modeled_energy sums up the result of calculate_total_power
    # for each point.

    assert total_kwh == 180.0

    # Verify calculate_total_power was called 24 times (once per hour)
    assert stats_manager.calculate_total_power.call_count == 24

def test_calculate_modeled_energy_fallback_legacy(stats_manager, mock_coordinator):
    """Test fallback to daily average when vectors are missing."""

    mock_coordinator._daily_history = {
        "2023-01-01": {
            "kwh": 120.0,
            "temp": 10.0, # Daily Avg
            "wind": 5.0,
            # No hourly_vectors
        }
    }

    def side_effect(temp, effective_wind, solar_impact, is_aux_active, unit_modes=None, override_solar_factor=None, detailed=True):
        base = max(0.0, 15.0 - temp)
        return {
            "total_kwh": base,
            "breakdown": {"solar_reduction_kwh": 0.0}
        }

    stats_manager.calculate_total_power = MagicMock(side_effect=side_effect)

    total_kwh, _, _, _, _ = stats_manager.calculate_modeled_energy(
        date(2023, 1, 1), date(2023, 1, 1)
    )

    # Fallback uses Daily Avg (10C)
    # Load = (15 - 10) * 24 = 120.
    assert total_kwh == 120.0

    # Verify calculate_total_power was called 1 time (daily avg)
    assert stats_manager.calculate_total_power.call_count == 1

def test_calculate_modeled_energy_fallback_all_none_vectors(stats_manager, mock_coordinator):
    """Test fallback to daily average when vectors exist but are all None."""

    # Scenario: Coordinator writes 'None' vectors for downtime days
    vectors = {
        "temp": [None] * 24,
        "load": [None] * 24,
        "wind": [None] * 24,
        "tdd": [None] * 24,
        "solar_rad": [None] * 24
    }

    mock_coordinator._daily_history = {
        "2023-01-01": {
            "kwh": 120.0,
            "temp": 10.0, # Daily Avg
            "wind": 5.0,
            "hourly_vectors": vectors # Exists but useless
        }
    }

    # Mock calculate_total_power to return simple linear model: Load = (15 - Temp)
    def side_effect(temp, effective_wind, solar_impact, is_aux_active, unit_modes=None, override_solar_factor=None, detailed=True):
        base = max(0.0, 15.0 - temp)
        return {
            "total_kwh": base,
            "breakdown": {"solar_reduction_kwh": 0.0}
        }

    stats_manager.calculate_total_power = MagicMock(side_effect=side_effect)

    total_kwh, _, _, _, _ = stats_manager.calculate_modeled_energy(
        date(2023, 1, 1), date(2023, 1, 1)
    )

    # Fallback uses Daily Avg (10C) -> Load = (15 - 10) * 24 = 120.
    assert total_kwh == 120.0

    # Verify calculate_total_power was called 1 time (daily avg fallback)
    assert stats_manager.calculate_total_power.call_count == 1
