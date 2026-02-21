"""Test efficiency calculation stability at low TDD (noise suppression)."""
import pytest
from unittest.mock import MagicMock
from custom_components.heating_analytics.statistics import StatisticsManager, MIN_STABLE_TDD, ATTR_TDD_SO_FAR, ATTR_ENERGY_TODAY

@pytest.fixture
def stats_manager():
    """Create a StatisticsManager instance with mocked coordinator."""
    coordinator = MagicMock()
    coordinator.data = {}
    return StatisticsManager(coordinator)

def test_efficiency_volatility_at_low_tdd(stats_manager):
    """Test that quadratic blending suppresses noise when TDD is very low."""

    # Scenario: Very low TDD (10% of stable threshold)
    low_tdd = MIN_STABLE_TDD * 0.1  # 0.01

    # Setup data
    stats_manager.coordinator.data[ATTR_TDD_SO_FAR] = low_tdd
    stats_manager.coordinator.data[ATTR_ENERGY_TODAY] = 0.0

    # Mock Instantaneous Efficiency (The Model)
    # Model says efficiency should be 10.0
    stats_manager.calculate_instantaneous_efficiency = MagicMock(return_value=10.0)

    # Force "Actual" calculation to be noisy
    # Actual = Energy / TDD
    # If we want Actual Efficiency = 50.0 (Huge Noise), then Energy = 50.0 * 0.01 = 0.5
    stats_manager.coordinator.data[ATTR_ENERGY_TODAY] = 50.0 * low_tdd

    # With Quadratic Blend:
    # confidence = (0.1)^2 = 0.01 (1%)
    # Blended = (10.0 * 0.99) + (50.0 * 0.01) = 9.9 + 0.5 = 10.4

    result = stats_manager.calculate_realtime_efficiency()

    print(f"\nTDD: {low_tdd}")
    print(f"Model Eff: 10.0")
    print(f"Actual Eff: 50.0 (Noise)")
    print(f"Result: {result}")

    # Assert that noise is suppressed (Result < 11.0)
    # Linear blend would have given 14.0
    assert result < 11.0, f"Result {result} is too volatile! Should be near 10.0"
    assert result >= 10.0, "Result should not be lower than model in this scenario"
