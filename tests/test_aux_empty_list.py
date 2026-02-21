"""Test suite for Empty Aux Affected Entities."""
from unittest.mock import MagicMock
import pytest
from custom_components.heating_analytics.statistics import StatisticsManager

@pytest.fixture
def mock_coordinator_empty_aux():
    coord = MagicMock()
    coord.energy_sensors = ["sensor.heater_1", "sensor.heater_2"]
    # User Explicitly selected NO units
    coord.aux_affected_entities = []
    coord._aux_affected_set = set() # Empty set
    coord.solar_enabled = False
    coord.balance_point = 18.0

    # Mock Models
    # Global Base: 10 kWh
    coord._get_predicted_kwh.return_value = 10.0

    # Mock Solar Saturation
    coord.solar.calculate_saturation.side_effect = lambda n, s, m: (0.0, 0.0, n)

    # Global Aux: 5 kWh reduction (Global Model still predicts savings!)
    coord._aux_coefficients = {"fake": "model"}

    # Unit modes
    coord.get_unit_mode.return_value = "heating"

    return coord

def test_empty_aux_list_unassigned_savings(mock_coordinator_empty_aux):
    """Verify that when no units are affected, global savings become unassigned."""
    stats = StatisticsManager(mock_coordinator_empty_aux)

    def side_effect(data_map, temp_key, wind_bucket, temp, bp, apply_scaling=True):
        if data_map == mock_coordinator_empty_aux._aux_coefficients:
            return 5.0 # Global Aux Reduction

        # Unit Data (Base)
        if data_map == {"unit": "data_1"}:
             return 6.0
        if data_map == {"unit": "data_2"}:
             return 4.0

        return 0.0

    stats._get_prediction_from_model = MagicMock(side_effect=side_effect)

    mock_coordinator_empty_aux._correlation_data_per_unit = {
        "sensor.heater_1": {"unit": "data_1"},
        "sensor.heater_2": {"unit": "data_2"}
    }
    mock_coordinator_empty_aux._aux_coefficients_per_unit = {
        "sensor.heater_1": {"unit": "aux_1"},
        "sensor.heater_2": {"unit": "aux_2"}
    }

    # Act
    result = stats.calculate_total_power(
        temp=0.0,
        effective_wind=0.0,
        solar_impact=0.0,
        is_aux_active=True
    )

    # Assert
    print("\nBreakdown:", result["breakdown"])

    assert result["global_aux_reduction_kwh"] == 5.0

    # Since no units affected, unit reduction should be 0
    assert result["breakdown"]["aux_reduction_kwh"] == 0.0

    # Ideally, the 5.0 global reduction should be reported as unassigned
    # Currently it is likely 0.0
    assert result["breakdown"]["unassigned_aux_savings"] == 5.0
