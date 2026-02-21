"""Test Nearest Neighbor Logic for Learning System."""
from unittest.mock import MagicMock, patch
import pytest
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.statistics import StatisticsManager
from custom_components.heating_analytics.solar import SolarCalculator

@pytest.fixture
def mock_coordinator(hass):
    """Create a mock coordinator."""
    entry = MagicMock()
    with patch("custom_components.heating_analytics.coordinator.HeatingDataCoordinator", autospec=True) as MockCoord:
        coord = MockCoord(hass, entry)
        coord.hass = hass
        coord._correlation_data = {}
        coord._aux_coefficients = {}
        coord._solar_coefficients = {}
        coord._correlation_data_per_unit = {}
        coord._aux_coefficients_per_unit = {}
        coord._solar_coefficients_per_unit = {}
        coord.balance_point = 12.0 # Set balance point for logic

        # Create a real StatisticsManager and attach it to the mock coordinator
        stats_manager = StatisticsManager(coord)
        coord.statistics = stats_manager

        # Attach the real, refactored methods from the class to the instance.
        # This ensures that when the test calls the method on the mock instance,
        # it executes the real code, which in turn calls the real statistics manager.
        coord._get_predicted_kwh = HeatingDataCoordinator._get_predicted_kwh.__get__(coord)
        coord._get_aux_impact_kw = HeatingDataCoordinator._get_aux_impact_kw.__get__(coord)
        coord._get_predicted_kwh_per_unit = HeatingDataCoordinator._get_predicted_kwh_per_unit.__get__(coord)
        coord._get_aux_impact_kw_per_unit = HeatingDataCoordinator._get_aux_impact_kw_per_unit.__get__(coord)

        # Solar needs a real instance attached for its tests
        coord.solar = SolarCalculator(coord)

        yield coord

def test_predicted_kwh_exact_match(mock_coordinator):
    """Test that exact match is still preferred."""
    mock_coordinator._correlation_data = {
        "10": {"normal": 5.0}
    }
    # Exact match - add actual_temp
    assert mock_coordinator._get_predicted_kwh("10", "normal", actual_temp=10.0) == 5.0

def test_predicted_kwh_neighbor_match(mock_coordinator):
    """Test neighbor lookup (Temp +/- 1) for base prediction."""
    mock_coordinator._correlation_data = {
        "9": {"normal": 4.0},
        "11": {"normal": 6.0}
    }
    # Target "10" is missing, but 9 and 11 exist. Should average to 5.0.
    assert mock_coordinator._get_predicted_kwh("10", "normal", actual_temp=10.0) == 5.0

    # Target "12" is missing. "11" exists (6.0). "13" missing. Should be 6.0.
    assert mock_coordinator._get_predicted_kwh("12", "normal", actual_temp=12.0) == 6.0

def test_predicted_kwh_neighbor_priority_over_fallback(mock_coordinator):
    """Test that neighbor (same wind) is preferred over fallback wind (same temp)."""
    mock_coordinator._correlation_data = {
        "10": {"normal": 2.0},       # Fallback candidate for (10, high_wind)
        "9": {"high_wind": 8.0},     # Neighbor candidate for (10, high_wind)
        "11": {"high_wind": 10.0}    # Neighbor candidate for (10, high_wind)
    }

    # We want (10, high_wind).
    # Option A (Old): Fallback to (10, normal) = 2.0
    # Option B (New): Neighbor (9/11, high_wind) = average(8, 10) = 9.0
    # Requirement: "first-choice in logic-chain" -> Option B.

    assert mock_coordinator._get_predicted_kwh("10", "high_wind", actual_temp=10.0) == 9.0

def test_aux_impact_neighbor_logic(mock_coordinator):
    """Test neighbor lookup for aux coefficients."""
    mock_coordinator._aux_coefficients = {
        "9": {"normal": 1.0},
        "11": {"normal": 3.0}
    }
    # Target "10". Average of 1.0 and 3.0 is 2.0.
    assert mock_coordinator._get_aux_impact_kw("10", "normal", actual_temp=10.0) == 2.0

def test_solar_coefficient_neighbor_logic(mock_coordinator):
    """Test neighbor lookup for solar coefficients."""
    mock_coordinator._solar_coefficients_per_unit = {
        "unit_1": {
            "9": 0.5,
            "11": 0.7
        }
    }
    # Target "10". Average of 0.5 and 0.7 is 0.6.
    assert mock_coordinator.solar.calculate_unit_coefficient("unit_1", "10") == 0.6

    # Target "12". With balance_point=12.0, target_t=12 is MODE_COOLING.
    # Learned data (9 and 11) are MODE_HEATING.
    # Should fall back to DEFAULT_SOLAR_COEFF_COOLING = 0.17.
    assert mock_coordinator.solar.calculate_unit_coefficient("unit_1", "12") == 0.17

def test_per_unit_neighbor_logic(mock_coordinator):
    """Test neighbor lookup for per-unit predictions."""
    entity = "sensor.heater"
    mock_coordinator._correlation_data_per_unit = {
        entity: {
            "9": {"normal": 100.0},
            "11": {"normal": 200.0}
        }
    }
    # Target "10". Average 150.0.
    assert mock_coordinator._get_predicted_kwh_per_unit(entity, "10", "normal", actual_temp=10.0) == 150.0
