"""Test core logic gaps not covered by other tests."""
import pytest
from unittest.mock import MagicMock
from custom_components.heating_analytics.statistics import StatisticsManager, MIN_STABLE_TDD
from custom_components.heating_analytics.const import (
    ATTR_TDD_SO_FAR,
    ATTR_ENERGY_TODAY,
    ATTR_TDD_YESTERDAY,
    ATTR_EFFICIENCY_YESTERDAY,
    ENERGY_GUARD_THRESHOLD,
    TARGET_TDD_WINDOW
)

@pytest.fixture
def stats_manager():
    coordinator = MagicMock()
    coordinator.data = {}
    coordinator.balance_point = 20.0
    return StatisticsManager(coordinator)

def test_efficiency_seamless_rolling_blending(stats_manager):
    """
    Test seamless rolling efficiency blending.

    Logic:
    When total_tdd < MIN_STABLE_TDD (0.1), result should be a blend of:
    - Model (Instantaneous Efficiency)
    - Actual (Accumulated / TDD)

    Blend Factor (Confidence in Actual) = total_tdd / MIN_STABLE_TDD
    """
    # Setup for Blending Scenario
    # MIN_STABLE_TDD is 0.1
    # We want total_tdd = 0.02 (20% of stable threshold)
    # So confidence = 0.02 / 0.1 = 0.2

    tdd_so_far = 0.02
    stats_manager.coordinator.data[ATTR_TDD_SO_FAR] = tdd_so_far

    # Let's say Actual Energy = 0.1 kWh
    # Actual Efficiency = 0.1 / 0.02 = 5.0 kWh/TDD
    stats_manager.coordinator.data[ATTR_ENERGY_TODAY] = 0.1

    # Disable Yesterday Carryover for simplicity (simulate start of day or no history)
    stats_manager.coordinator.data[ATTR_TDD_YESTERDAY] = None

    # Mock Instantaneous Model
    # We need to mock calculate_instantaneous_efficiency to return a fixed value
    # Model Efficiency = 10.0 kWh/TDD
    stats_manager.calculate_instantaneous_efficiency = MagicMock(return_value=10.0)

    # Execution
    result = stats_manager.calculate_realtime_efficiency()

    # Verification
    # Expected Blended Efficiency:
    # Note: Code uses quadratic blending (confidence = ratio^2) to suppress noise.
    # Ratio = 0.2, Confidence = 0.2^2 = 0.04
    # Blend = (Model * (1 - confidence)) + (Actual * confidence)
    # Blend = (10.0 * (1 - 0.04)) + (5.0 * 0.04)
    # Blend = (10.0 * 0.96) + (0.2)
    # Blend = 9.6 + 0.2 = 9.8

    assert result == 9.8

def test_efficiency_tdd_threshold(stats_manager):
    """
    Test that efficiency calculation returns None when TDD rate is too low.

    Logic:
    If TDD/hr <= 0.05 (approx 1.2C delta), instantaneous efficiency is undefined
    because the denominator is too small/noisy.
    """
    # Setup Coordinator Data
    stats_manager.coordinator.data["current_model_rate"] = 1.0 # 1 kW

    # Case 1: TDD Rate too low (0.05)
    # TDD = abs(BP - Temp) / 24
    # 0.05 = Delta / 24 => Delta = 1.2
    # Temp = 20 - 1.2 = 18.8
    stats_manager.coordinator.data["current_calc_temp"] = 18.8

    result = stats_manager.calculate_instantaneous_efficiency()
    assert result is None, "Should return None for TDD rate <= 0.05"

    # Case 2: TDD Rate just above threshold (0.051)
    # 0.051 * 24 = 1.224
    # Temp = 20 - 1.224 = 18.776
    stats_manager.coordinator.data["current_calc_temp"] = 18.77

    result = stats_manager.calculate_instantaneous_efficiency()
    assert result is not None, "Should return value for TDD rate > 0.05"

    # Verify Calculation
    # TDD/hr = abs(20 - 18.77) / 24 = 1.23 / 24 = 0.05125
    # Eff = Rate / TDD_hr = 1.0 / 0.05125 approx 19.51
    assert result == pytest.approx(1.0 / ((20.0 - 18.77) / 24.0), 0.01)

def test_tdd_calculation_cooling_mode(stats_manager):
    """
    Test that TDD calculation handles Cooling Mode (Temp > Balance Point).

    Logic:
    TDD = abs(BP - Temp) / 24.
    Must handle Temp > BP correctly (positive result).
    """
    # Setup Coordinator Data
    stats_manager.coordinator.data["current_model_rate"] = 2.0 # 2 kW

    # Case: Cooling (Temp > BP)
    # Temp = 25.0 (BP = 20.0) -> Delta = 5.0
    stats_manager.coordinator.data["current_calc_temp"] = 25.0

    result = stats_manager.calculate_instantaneous_efficiency()

    # TDD/hr = 5.0 / 24.0 = 0.2083...
    # Eff = 2.0 / 0.2083... = 9.6

    assert result is not None
    assert result > 0
    assert result == pytest.approx(9.6, 0.1)
