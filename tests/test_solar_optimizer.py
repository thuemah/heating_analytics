"""Tests for SolarOptimizer."""
import pytest
from unittest.mock import MagicMock

# Import the class under test
from custom_components.heating_analytics.solar_optimizer import SolarOptimizer
from custom_components.heating_analytics.const import (
    RECOMMENDATION_MAXIMIZE_SOLAR,
    RECOMMENDATION_INSULATE,
    RECOMMENDATION_MITIGATE_SOLAR,
)

@pytest.fixture
def mock_coordinator():
    """Mock coordinator with configuration."""
    coordinator = MagicMock()
    coordinator.balance_point = 17.0
    return coordinator

def test_recommendation_state(mock_coordinator):
    """Test recommendation state logic."""
    optimizer = SolarOptimizer(mock_coordinator)

    # 1. Cold and Sunny (Maximize)
    # Temp < 17, Potential > 0.1
    state = optimizer.get_recommendation_state(5.0, 0.5)
    assert state == RECOMMENDATION_MAXIMIZE_SOLAR

    # 2. Cold and Dark (Insulate)
    # Temp < 17, Potential <= 0.1
    state = optimizer.get_recommendation_state(5.0, 0.05)
    assert state == RECOMMENDATION_INSULATE

    # 3. Hot and Sunny (Mitigate)
    # Temp > 17, Potential > 0.1
    state = optimizer.get_recommendation_state(25.0, 0.5)
    assert state == RECOMMENDATION_MITIGATE_SOLAR

    # 4. Hot and Dark (None/Default)
    # Temp > 17, Potential <= 0.1
    state = optimizer.get_recommendation_state(25.0, 0.05)
    assert state == "none"

def test_prediction_defaults(mock_coordinator):
    """Test default predictions when no learning has occurred."""
    optimizer = SolarOptimizer(mock_coordinator)
    azimuth = 180.0

    # Maximize -> 100%
    pred = optimizer.predict_correction_percent(RECOMMENDATION_MAXIMIZE_SOLAR, 20.0, azimuth, 50.0)
    assert pred == 100.0

    # Insulate -> 0%
    pred = optimizer.predict_correction_percent(RECOMMENDATION_INSULATE, 20.0, azimuth, 50.0)
    assert pred == 0.0

    # Mitigate -> 0%
    pred = optimizer.predict_correction_percent(RECOMMENDATION_MITIGATE_SOLAR, 20.0, azimuth, 50.0)
    assert pred == 0.0

    # None -> Default
    pred = optimizer.predict_correction_percent("none", 20.0, azimuth, 50.0)
    assert pred == 50.0

def test_learning_and_persistence(mock_coordinator):
    """Test learning logic and data persistence."""
    optimizer = SolarOptimizer(mock_coordinator)

    state = RECOMMENDATION_MAXIMIZE_SOLAR
    elevation = 23.0 # Bucket "20"
    azimuth = 180.0 # Bucket "180"

    # Initial: Default is 100
    assert optimizer.predict_correction_percent(state, elevation, azimuth, 50.0) == 100.0

    # User sets to 80%
    optimizer.learn_correction_percent(state, elevation, azimuth, 80.0)

    # Should be 80.0 (First learn is jump start)
    assert optimizer.predict_correction_percent(state, elevation, azimuth, 50.0) == 80.0

    # User sets to 90%
    # EMA: current + 0.1 * (target - current)
    # 80 + 0.1 * (90 - 80) = 81.0
    optimizer.learn_correction_percent(state, elevation, azimuth, 90.0)
    assert optimizer.predict_correction_percent(state, elevation, azimuth, 50.0) == 81.0

    # Verify Persistence
    data = optimizer.get_data()
    assert "model" in data
    assert state in data["model"]
    assert "180" in data["model"][state]
    assert "20" in data["model"][state]["180"]
    assert data["model"][state]["180"]["20"] == 81.0

    # Verify Restore
    new_optimizer = SolarOptimizer(mock_coordinator)
    new_optimizer.set_data(data)
    assert new_optimizer.predict_correction_percent(state, elevation, azimuth, 50.0) == 81.0

def test_elevation_buckets(mock_coordinator):
    """Test elevation bucketing."""
    optimizer = SolarOptimizer(mock_coordinator)

    assert optimizer._get_elevation_bucket(5.0) == "0"
    assert optimizer._get_elevation_bucket(12.0) == "10"
    assert optimizer._get_elevation_bucket(19.9) == "10"
    assert optimizer._get_elevation_bucket(20.0) == "20"
    assert optimizer._get_elevation_bucket(-5.0) == "0"

def test_azimuth_buckets(mock_coordinator):
    """Test azimuth bucketing."""
    optimizer = SolarOptimizer(mock_coordinator)

    assert optimizer._get_azimuth_bucket(5.0) == "0"
    assert optimizer._get_azimuth_bucket(29.0) == "0"
    assert optimizer._get_azimuth_bucket(30.0) == "30"
    assert optimizer._get_azimuth_bucket(45.0) == "30"
    assert optimizer._get_azimuth_bucket(60.0) == "60"
    assert optimizer._get_azimuth_bucket(359.0) == "330"
    assert optimizer._get_azimuth_bucket(365.0) == "0" # Wrap around

def test_azimuth_learning_separation(mock_coordinator):
    """Test that learning in one azimuth bucket does not affect others."""
    optimizer = SolarOptimizer(mock_coordinator)
    state = RECOMMENDATION_MAXIMIZE_SOLAR
    elevation = 20.0

    az_south = 180.0
    az_west = 270.0

    # 1. Learn 50% for South
    optimizer.learn_correction_percent(state, elevation, az_south, 50.0)

    # 2. Verify South is 50%
    assert optimizer.predict_correction_percent(state, elevation, az_south, 100.0) == 50.0

    # 3. Verify West is still Default (100%)
    assert optimizer.predict_correction_percent(state, elevation, az_west, 100.0) == 100.0

    # 4. Learn 25% for West
    optimizer.learn_correction_percent(state, elevation, az_west, 25.0)

    # 5. Verify West is 25% and South remains 50%
    assert optimizer.predict_correction_percent(state, elevation, az_west, 100.0) == 25.0
    assert optimizer.predict_correction_percent(state, elevation, az_south, 100.0) == 50.0

def test_legacy_data_migration(mock_coordinator):
    """Test migration of legacy data format."""
    optimizer = SolarOptimizer(mock_coordinator)
    state = RECOMMENDATION_MAXIMIZE_SOLAR

    # Legacy Data: { state: { "20": 80.0, "30": 70.0 } }
    legacy_data = {
        "model": {
            state: {
                "20": 80.0,
                "30": 70.0
            }
        }
    }

    # Restore Legacy Data
    optimizer.set_data(legacy_data)

    # Verify Data Structure (Should be migrated)
    migrated_data = optimizer.get_data()
    model = migrated_data["model"][state]

    # Check that Azimuth 180 has the data
    assert "180" in model
    assert model["180"]["20"] == 80.0
    assert model["180"]["30"] == 70.0

    # Check that Azimuth 90 has the data
    assert "90" in model
    assert model["90"]["20"] == 80.0

    # Verify Predictions work for migrated data
    # Predict South (180) -> Should return learned 80%
    assert optimizer.predict_correction_percent(state, 23.0, 180.0, 100.0) == 80.0

    # Predict West (270) -> Should return learned 80% (copied)
    assert optimizer.predict_correction_percent(state, 23.0, 270.0, 100.0) == 80.0
