"""Test Safety Mechanisms (Division by Zero, etc)."""
from unittest.mock import MagicMock
import pytest
from custom_components.heating_analytics.statistics import StatisticsManager
from custom_components.heating_analytics.const import (
    ATTR_TDD_SO_FAR, ATTR_ENERGY_TODAY, ATTR_TDD_YESTERDAY, ATTR_EFFICIENCY_YESTERDAY
)

@pytest.fixture
def mock_coordinator():
    coord = MagicMock()
    coord.data = {}
    coord.balance_point = 17.0
    # Mock daily history
    coord._daily_history = {}
    return coord

def test_efficiency_zero_tdd_so_far(mock_coordinator):
    """Test calculate_realtime_efficiency with 0 TDD."""
    stats = StatisticsManager(mock_coordinator)

    mock_coordinator.data[ATTR_TDD_SO_FAR] = 0.0
    mock_coordinator.data[ATTR_ENERGY_TODAY] = 10.0

    # Yesterday is None
    mock_coordinator.data[ATTR_TDD_YESTERDAY] = None

    # Should fallback to instantaneous or None.
    # Mock fallback to fail (return None)
    mock_coordinator.data["current_model_rate"] = 0.0
    mock_coordinator.data["current_calc_temp"] = 17.0 # Balance point

    eff = stats.calculate_realtime_efficiency()
    assert eff is None

def test_efficiency_stats_zero_sum_tdd(mock_coordinator):
    """Test _calculate_efficiency_stats with 0 Sum TDD."""
    stats = StatisticsManager(mock_coordinator)

    # Mock history with 0 TDD but some kWh (e.g. summer water heating)
    keys = ["2023-01-01", "2023-01-02"]
    mock_coordinator._daily_history = {
        "2023-01-01": {"tdd": 0.0, "kwh": 5.0},
        "2023-01-02": {"tdd": 0.0, "kwh": 5.0},
    }

    avg_tdd, eff = stats._calculate_efficiency_stats(keys)

    assert avg_tdd == 0.0
    assert eff is None # Guarded against div by zero

def test_inst_efficiency_balance_point(mock_coordinator):
    """Test instantaneous efficiency at balance point."""
    stats = StatisticsManager(mock_coordinator)

    mock_coordinator.data["current_model_rate"] = 1.0
    mock_coordinator.data["current_calc_temp"] = 17.0 # Exactly balance point

    eff = stats.calculate_instantaneous_efficiency()

    assert eff is None
