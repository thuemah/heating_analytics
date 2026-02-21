"""Test wind bucket classification logic."""
from unittest.mock import MagicMock, patch
import pytest
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.const import DEFAULT_WIND_THRESHOLD, DEFAULT_EXTREME_WIND_THRESHOLD

@pytest.fixture
def coordinator(hass):
    """Create a coordinator instance for testing."""
    entry = MagicMock()
    # Defaults
    entry.data = {
        "wind_threshold": DEFAULT_WIND_THRESHOLD, # 5.5
        "extreme_wind_threshold": DEFAULT_EXTREME_WIND_THRESHOLD, # 10.8
    }
    with patch("custom_components.heating_analytics.storage.Store"):
        return HeatingDataCoordinator(hass, entry)

def test_wind_bucket_normal(coordinator):
    """Test normal wind bucket."""
    # < 5.5
    assert coordinator._get_wind_bucket(0.0) == "normal"
    assert coordinator._get_wind_bucket(5.4) == "normal"

def test_wind_bucket_high(coordinator):
    """Test high wind bucket."""
    # >= 5.5 and < 10.8
    assert coordinator._get_wind_bucket(5.5) == "high_wind"
    assert coordinator._get_wind_bucket(10.7) == "high_wind"

def test_wind_bucket_extreme(coordinator):
    """Test extreme wind bucket."""
    # >= 10.8
    assert coordinator._get_wind_bucket(10.8) == "extreme_wind"
    assert coordinator._get_wind_bucket(20.0) == "extreme_wind"

def test_wind_bucket_auxiliary_active_returns_physical(coordinator):
    """Test that auxiliary heating returns physical bucket (Coefficient model)."""
    coordinator.auxiliary_heating_active = True

    # Should RETURN PHYSICAL BUCKET (Logic changed from "with_auxiliary_heating")
    assert coordinator._get_wind_bucket(0.0) == "normal"
    assert coordinator._get_wind_bucket(20.0) == "extreme_wind"

