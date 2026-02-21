"""Test effective wind calculation logic."""
from unittest.mock import MagicMock, patch
import pytest
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator

@pytest.fixture
def coordinator(hass):
    """Create a coordinator instance for testing."""
    entry = MagicMock()
    entry.data = {
        "wind_gust_factor": 0.6
    }
    with patch("custom_components.heating_analytics.storage.Store"):
        return HeatingDataCoordinator(hass, entry)

def test_effective_wind_no_gust(coordinator):
    """Test effective wind when gust is None or 0."""
    # Formula: speed + (max(0, gust-speed) * factor)

    # Gust None -> Speed
    assert coordinator._calculate_effective_wind(10.0, None) == 10.0

    # Gust 0 -> Speed (since gust < speed, max is 0)
    assert coordinator._calculate_effective_wind(10.0, 0.0) == 10.0

def test_effective_wind_gust_less_than_speed(coordinator):
    """Test effective wind when gust <= speed."""
    # Gust 5, Speed 10 -> Turbulence 0 -> 10
    assert coordinator._calculate_effective_wind(10.0, 5.0) == 10.0

    # Gust 10, Speed 10 -> Turbulence 0 -> 10
    assert coordinator._calculate_effective_wind(10.0, 10.0) == 10.0

def test_effective_wind_gust_greater_than_speed(coordinator):
    """Test effective wind when gust > speed."""
    # Gust 20, Speed 10 -> Turbulence 10. Factor 0.6.
    # 10 + 10 * 0.6 = 16.0
    assert coordinator._calculate_effective_wind(10.0, 20.0) == 16.0

    # Gust 15, Speed 5 -> Turbulence 10. Factor 0.6.
    # 5 + 10 * 0.6 = 11.0
    assert coordinator._calculate_effective_wind(5.0, 15.0) == 11.0

def test_effective_wind_negative_input_safety(coordinator):
    """Test that negative inputs are handled safely (clamped to 0)."""
    # Speed -5 -> 0. Gust 10. Turbulence 10. Result 6.0?
    # Logic: speed = max(0, speed) -> 0.
    # gust = max(0, gust) -> 10.
    # turbulence = max(0, 10 - 0) = 10.
    # 0 + 10 * 0.6 = 6.0
    assert coordinator._calculate_effective_wind(-5.0, 10.0) == 6.0

    # Speed 10. Gust -5 -> 0.
    # Turbulence max(0, 0 - 10) = 0.
    # Result 10.
    assert coordinator._calculate_effective_wind(10.0, -5.0) == 10.0
