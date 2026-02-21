import pytest
from unittest.mock import MagicMock
from custom_components.heating_analytics.statistics import StatisticsManager

@pytest.fixture
def stats_manager():
    coordinator = MagicMock()
    coordinator.balance_point = 20.0
    return StatisticsManager(coordinator)

def test_extrapolation_heating(stats_manager):
    """Test extrapolation below balance point (Heating)."""
    # BP = 20.0
    # Data at 10.0 (TDD = 10.0)
    data_map = {
        "10": {"normal": 2.0}
    }

    # Predict for 5.0 (TDD = 15.0)
    # Ratio = 15.0 / 10.0 = 1.5
    # Value = 2.0 * 1.5 = 3.0
    # actual_temp=5.0
    val = stats_manager._get_prediction_from_model(data_map, "5", "normal", 5.0, 20.0)
    assert val == 3.0

def test_extrapolation_cooling(stats_manager):
    """Test extrapolation above balance point (Cooling)."""
    # BP = 20.0
    # Data at 25.0 (TDD = 5.0)
    data_map = {
        "25": {"normal": 1.0}
    }

    # Predict for 30.0 (TDD = 10.0)
    # Ratio = 10.0 / 5.0 = 2.0
    # Value = 1.0 * 2.0 = 2.0
    # actual_temp=30.0
    val = stats_manager._get_prediction_from_model(data_map, "30", "normal", 30.0, 20.0)
    assert val == 2.0

def test_extrapolation_cross_bp(stats_manager):
    """Test extrapolation across balance point."""
    # BP = 20.0
    # Data at 15.0 (TDD = 5.0) - Heating data
    data_map = {
        "15": {"normal": 0.5}
    }

    # Predict for 25.0 (TDD = 5.0) - Cooling target
    # Ratio = 5.0 / 5.0 = 1.0
    # Value = 0.5 * 1.0 = 0.5
    # actual_temp=25.0
    val = stats_manager._get_prediction_from_model(data_map, "25", "normal", 25.0, 20.0)
    assert val == 0.5

def test_extrapolation_safety_guards_near_bp(stats_manager):
    """Test safety guards when source data is near balance point."""
    # BP = 20.0
    # Data at 20.0 (TDD = 0.0) - Exactly at BP
    data_map = {
        "20": {"normal": 0.1}
    }

    # Case 1: Target also near BP (e.g. 20.5, TDD=0.5)
    # tdd_source = abs(20.0 - 20.0) = 0.0
    # Since tdd_source < 0.1, we expect NO extrapolation, just return neighbor val.

    val = stats_manager._get_prediction_from_model(data_map, "21", "normal", 20.5, 20.0)
    assert val == 0.1

    # Case 2: Target far from BP (e.g. 15.0, TDD=5.0)
    # tdd_source = 0.0 < 0.1
    # We still expect NO extrapolation because source is noise.
    # Return neighbor val.

    val = stats_manager._get_prediction_from_model(data_map, "15", "normal", 15.0, 20.0)
    assert val == 0.1

def test_extrapolation_cooling_to_warmer(stats_manager):
    """Specifically test the case mentioned by user: extrapolation to warmer cooling temp."""
    # BP = 20.0
    # We have data for 25.0 (TDD = 5.0)
    data_map = {
        "25": {"normal": 1.2}
    }

    # Predict for 28.0 (TDD = 8.0)
    # Ratio = 8.0 / 5.0 = 1.6
    # expected = 1.2 * 1.6 = 1.92

    val = stats_manager._get_prediction_from_model(data_map, "28", "normal", 28.0, 20.0)
    assert val == 1.92
