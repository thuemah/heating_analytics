"""Tests for SolarOptimizer Cloud Constraint."""
import pytest
from unittest.mock import MagicMock

from custom_components.heating_analytics.solar_optimizer import SolarOptimizer
from custom_components.heating_analytics.const import RECOMMENDATION_MAXIMIZE_SOLAR

@pytest.fixture
def mock_coordinator():
    """Mock coordinator with configuration."""
    coordinator = MagicMock()
    coordinator.balance_point = 17.0
    return coordinator

def test_learning_with_low_cloud_cover(mock_coordinator):
    """Test that learning occurs when cloud cover is low (< 20%)."""
    optimizer = SolarOptimizer(mock_coordinator)
    state = RECOMMENDATION_MAXIMIZE_SOLAR
    elevation = 23.0
    azimuth = 180.0

    # Cloud cover 10% -> Should learn
    optimizer.learn_correction_percent(state, elevation, azimuth, 80.0, cloud_cover=10.0)

    # Check that model was updated (default is 100%, we learned 80%)
    # Note: 1st learn is instant jump to value
    assert optimizer.predict_correction_percent(state, elevation, azimuth, 50.0) == 80.0

def test_learning_skipped_with_high_cloud_cover(mock_coordinator):
    """Test that learning is skipped when cloud cover is high (>= 20%)."""
    optimizer = SolarOptimizer(mock_coordinator)
    state = RECOMMENDATION_MAXIMIZE_SOLAR
    elevation = 23.0
    azimuth = 180.0

    # Cloud cover 25% -> Should NOT learn
    optimizer.learn_correction_percent(state, elevation, azimuth, 80.0, cloud_cover=25.0)

    # Check that model was NOT updated (should still return default/fallback)
    # Default for MAXIMIZE_SOLAR is 100.0
    assert optimizer.predict_correction_percent(state, elevation, azimuth, 50.0) == 100.0

def test_learning_boundary_condition(mock_coordinator):
    """Test boundary condition exactly at 20%."""
    optimizer = SolarOptimizer(mock_coordinator)
    state = RECOMMENDATION_MAXIMIZE_SOLAR
    elevation = 23.0
    azimuth = 180.0

    # Cloud cover 20% -> Should NOT learn ( < 20% required)
    optimizer.learn_correction_percent(state, elevation, azimuth, 80.0, cloud_cover=20.0)

    assert optimizer.predict_correction_percent(state, elevation, azimuth, 50.0) == 100.0

def test_learning_default_behavior(mock_coordinator):
    """Test that default behavior (0% cloud) still learns."""
    optimizer = SolarOptimizer(mock_coordinator)
    state = RECOMMENDATION_MAXIMIZE_SOLAR
    elevation = 23.0
    azimuth = 180.0

    # No cloud_cover passed (defaults to 0.0)
    optimizer.learn_correction_percent(state, elevation, azimuth, 80.0)

    assert optimizer.predict_correction_percent(state, elevation, azimuth, 50.0) == 80.0
