
import pytest
from unittest.mock import MagicMock
from custom_components.heating_analytics.statistics import StatisticsManager

@pytest.fixture
def stats_manager():
    coordinator = MagicMock()
    coordinator.balance_point = 20.0
    return StatisticsManager(coordinator)

def test_get_prediction_regime_aware(stats_manager):
    # Setup data map
    # Balance point is 20.0
    # Mild Regime: TDD <= 4.0 => Temp >= 16.0
    # Cold Regime: TDD > 4.0 => Temp < 16.0

    data_map = {
        "18": {"normal": 2.0},
        "16": {"normal": 4.0},
        "10": {"normal": 10.0},
        "0": {"normal": 20.0}
    }

    # 1. Exact Match (Always prioritized)
    # Mild
    assert stats_manager._get_prediction_from_model(data_map, "18", "normal", 18.0, 20.0) == 2.0
    # Cold
    assert stats_manager._get_prediction_from_model(data_map, "10", "normal", 10.0, 20.0) == 10.0

    # 2. Mild Regime (TDD <= 4.0): Should use neighbor averaging
    # Target 17, neighbors 16 and 18 exist. TDD = |20-17| = 3.0 <= 4.0.
    # Neighbors: 18 (2.0) and 16 (4.0). Average = 3.0.
    # If it used TDD scaling from 16: 4.0 * (3.0 / 4.0) = 3.0 (Wait, TDD scaling might give same result here if linear)
    # Let's use non-linear data to distinguish.

    data_map_nonlinear = {
        "18": {"normal": 2.0}, # TDD=2
        "16": {"normal": 8.0}, # TDD=4
    }
    # Target 17: TDD=3.
    # Neighbor Average: (2.0 + 8.0) / 2 = 5.0
    # TDD Scaling from 16: 8.0 * (3/4) = 6.0
    # TDD Scaling from 18: 2.0 * (3/2) = 3.0

    # Mild Regime (17.0): Should be 5.0
    assert stats_manager._get_prediction_from_model(data_map_nonlinear, "17", "normal", 17.0, 20.0) == 5.0

    # 3. Cold Regime (TDD > 4.0): Should force TDD scaling and skip neighbor averaging
    data_map_cold = {
        "12": {"normal": 10.0}, # TDD=8
        "10": {"normal": 15.0}, # TDD=10
    }
    # Target 11: TDD=9.
    # Neighbor Average: (10.0 + 15.0) / 2 = 12.5
    # TDD Scaling from 12: 10.0 * (9/8) = 11.25
    # TDD Scaling from 10: 15.0 * (9/10) = 13.5
    # Extrapolation picks nearest: 12 is closer to 11 than 10? No, 10 and 12 are equidistant from 11.
    # The code picks first valid key that matches min_diff.

    # In Cold Regime (11.0), TDD=9 > 4.0.
    # It should SKIP neighbor averaging and go to extrapolation.
    # Extrapolation will find nearest neighbor (say 12) and scale.
    # result = 11.25 (if 12 is picked) or 13.5 (if 10 is picked). Both are NOT 12.5.

    res = stats_manager._get_prediction_from_model(data_map_cold, "11", "normal", 11.0, 20.0)
    assert res != 12.5
    assert res in [11.25, 13.5]

    # 4. Cold Regime Guard (TDD_source > 1.0)
    data_map_guard = {
        "19": {"normal": 1.0}, # TDD=1.0
        "10": {"normal": 20.0}   # TDD=10
    }
    # Target 5: TDD=15.
    # Nearest is 10 (TDD=10). TDD_source = 10.0 > 1.0. Scaling should happen.
    # 20.0 * (15/10) = 30.0
    assert stats_manager._get_prediction_from_model(data_map_guard, "5", "normal", 5.0, 20.0) == 30.0

    # Target -5: TDD=25.
    # Suppose we only have data at 19 (TDD=1.0).
    data_map_at_guard = {
        "19": {"normal": 1.0}
    }
    # Target 5: TDD=15.
    # Nearest is 19. TDD_source = 1.0.
    # In Cold Regime, guard is 1.0. tdd_source < 1.0 is FALSE (it's equal).
    # So it should scale: 1.0 * (15/1.0) = 15.0
    assert stats_manager._get_prediction_from_model(data_map_at_guard, "5", "normal", 5.0, 20.0) == 15.0

    # Below guard
    data_map_below_guard = {
        "20": {"normal": 1.0} # TDD=0.0
    }
    # Target 5: TDD=15.
    # Nearest is 20. TDD_source = 0.0.
    # 0.0 < 1.0 is TRUE. Returns neighbor_val (1.0).
    assert stats_manager._get_prediction_from_model(data_map_below_guard, "5", "normal", 5.0, 20.0) == 1.0

def test_cold_regime_wind_gap_regression(stats_manager):
    """
    Test that Cold Regime (Delta > 4.0) returns a value even if specific wind bucket is missing,
    instead of dropping to 0.0 (Regression Test).
    """
    # Setup Data Map: 12C has only Normal bucket.
    # Balance Point 20. Target 15. Delta = 5.0 > 4.0 (Cold Regime).
    # Map has 15C but only "normal". Requesting "high_wind".
    data_map = {
        "15": {
            "normal": 1.5
            # "high_wind" is MISSING
        },
        # Add a neighbor to allow extrapolation logic to trigger first
        "12": {
            "normal": 2.0,
            "high_wind": 2.2
        }
    }

    # Run Prediction for 15C, High Wind
    prediction = stats_manager._get_prediction_from_model(
        data_map, "15", "high_wind", 15.0, 20.0
    )

    # Assert
    assert prediction > 0.0, "Should handle wind gap in Cold Regime"

    # Logic Verification:
    # 1. Exact Match (15, high) -> Fail
    # 2. Cold Regime (Delta=5) -> Skip Wind Fallback
    # 3. Extrapolation -> Prefer Neighbor (12).
    # TDD Target (15C) = 5.0
    # TDD Source (12C) = 8.0
    # Ratio = 5/8 = 0.625
    # Value = 2.2 * 0.625 = 1.375

    # If it failed neighbor and used local fallback (15 Normal):
    # Value = 1.5

    assert 1.3 <= prediction <= 1.6

def test_cold_regime_ping_pong_recursion_prevention(stats_manager):
    """
    Test that ping-pong recursion between neighbors is prevented.

    Scenario: Request missing wind bucket, where both nearest neighbors
    also lack that bucket. Without proper handling, this causes infinite
    recursion: A → B → A → B → ...

    Regression Test for infinite recursion bug.
    """
    # Setup: Both 10C and 12C have only "normal" bucket
    # Request 11C with "high_wind" - missing in both neighbors
    data_map = {
        "10": {
            "normal": 1.0
            # "high_wind" MISSING
        },
        "12": {
            "normal": 2.0
            # "high_wind" MISSING
        }
    }

    # Run Prediction for 11C, High Wind (Cold Regime: TDD=9 > 4.0)
    # This should NOT cause infinite recursion
    prediction = stats_manager._get_prediction_from_model(
        data_map, "11", "high_wind", 11.0, 20.0
    )

    # Assert: Should return a valid value (not 0, not recursion error)
    assert prediction > 0.0, "Should handle ping-pong scenario without recursion"

    # Logic Verification:
    # 1. Exact Match (11, high_wind) -> FAIL (key doesn't exist)
    # 2. Cold Regime (TDD=9) -> Skip Wind Fallback
    # 3. Extrapolation -> Find nearest neighbor (10 or 12)
    # 4. Before recursing, check if neighbor has "high_wind" -> NO
    # 5. Use neighbor's available bucket ("normal") instead
    # 6. Recurse with fallback bucket -> SUCCESS
    #
    # If nearest is 10: 1.0 * (9/10) = 0.9
    # If nearest is 12: 2.0 * (9/8) = 2.25

    assert 0.8 <= prediction <= 2.5, f"Expected scaled value, got {prediction}"
