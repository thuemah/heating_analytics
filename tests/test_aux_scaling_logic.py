"""Test removal of scaling logic in StatisticsManager."""
from unittest.mock import MagicMock
import pytest
from custom_components.heating_analytics.statistics import StatisticsManager
from custom_components.heating_analytics.const import DOMAIN

class MockCoordinator:
    def __init__(self):
        self.hass = MagicMock()
        self.data = {}
        self.energy_sensors = ["sensor.heater"]
        self.aux_affected_entities = ["sensor.heater"]
        self.solar_enabled = False
        self.balance_point = 15.0
        self.solar = MagicMock()
        self.solar.calculate_saturation.side_effect = lambda net, pot, mode: (0.0, 0.0, net)
        self.solar.calculate_unit_coefficient.return_value = 0.0
        self.solar.calculate_unit_solar_impact.return_value = 0.0

        # Internal structures
        self._correlation_data_per_unit = {}
        self._aux_coefficients_per_unit = {}
        self._aux_coefficients = {}
        self._unit_modes = {}

    def _get_wind_bucket(self, wind):
        return "normal"

    def get_unit_mode(self, entity_id):
        return "heating"

    def _get_predicted_kwh(self, temp_key, wind_bucket, temp):
        # Global Base (Master) - Not critical for this test but needed for return
        return 10.0

@pytest.fixture
def stats_manager():
    coord = MockCoordinator()
    manager = StatisticsManager(coord)

    # Mock _get_prediction_from_model to return values from our setup
    def mock_get_pred(data_map, temp_key, wind_bucket, temp, bp, apply_scaling=True):
        if temp_key in data_map:
            return data_map[temp_key].get(wind_bucket, 0.0)
        return 0.0

    manager._get_prediction_from_model = MagicMock(side_effect=mock_get_pred)
    return manager

def test_scaling_removal_prevents_overflow(stats_manager):
    """Test that removing scaling prevents artificial overflow."""

    # Setup Scenario:
    # Unit Base = 3.0
    # Unit Raw Aux = 2.8 (Valid, < Base)
    # Global Aux = 3.22 (Implies 1.15 scaling factor if logic existed)

    coord = stats_manager.coordinator

    # Unit Data
    coord._correlation_data_per_unit = {
        "sensor.heater": {"5": {"normal": 3.0}}
    }
    coord._aux_coefficients_per_unit = {
        "sensor.heater": {"5": {"normal": 2.8}}
    }

    # Global Data (Aux)
    coord._aux_coefficients = {"5": {"normal": 3.22}}

    # Execute
    result = stats_manager.calculate_total_power(
        temp=5.0,
        effective_wind=0.0,
        solar_impact=0.0,
        is_aux_active=True,
        detailed=True
    )

    unit_stats = result["unit_breakdown"]["sensor.heater"]
    breakdown = result["breakdown"]

    # Assertions for "Honest" Reporting (No Scaling)

    # 1. Aux Reduction should be exactly what the unit learned (2.8)
    # With scaling, this would be 2.8 * (3.22/2.8) = 3.22, causing overflow
    assert unit_stats["aux_reduction_kwh"] == 2.8
    assert unit_stats["raw_aux_kwh"] == 2.8

    # 2. Net should be Base - Aux = 3.0 - 2.8 = 0.2
    # With scaling overflow, this would be 0.0
    assert unit_stats["net_kwh"] == 0.2

    # 3. Overflow/Clamped should be 0 (since 2.8 < 3.0)
    assert unit_stats["overflow_kwh"] == 0.0
    assert unit_stats["clamped"] is False

    # 4. Orphaned/Unassigned Savings should capture the difference
    # Global (3.22) - Unit (2.8) = 0.42
    # Floating point precision might apply
    assert breakdown["orphaned_aux_savings"] == pytest.approx(0.42)
    assert breakdown["unassigned_aux_savings"] == pytest.approx(0.42)

def test_cold_start_no_redistribution(stats_manager):
    """Test that cold start (unit=0, global>0) results in unassigned savings, not base-weighted distribution."""

    # Setup Scenario:
    # Unit Base = 5.0
    # Unit Raw Aux = 0.0 (Cold start / not learned yet)
    # Global Aux = 1.0 (Learned global savings)

    coord = stats_manager.coordinator

    # Unit Data
    coord._correlation_data_per_unit = {
        "sensor.heater": {"5": {"normal": 5.0}}
    }
    coord._aux_coefficients_per_unit = {
        "sensor.heater": {"5": {"normal": 0.0}}
    }

    # Global Data (Aux)
    coord._aux_coefficients = {"5": {"normal": 1.0}}

    # Execute
    result = stats_manager.calculate_total_power(
        temp=5.0,
        effective_wind=0.0,
        solar_impact=0.0,
        is_aux_active=True,
        detailed=True
    )

    unit_stats = result["unit_breakdown"]["sensor.heater"]
    breakdown = result["breakdown"]

    # Assertions

    # 1. Unit Aux should remain 0.0 (No artificial distribution)
    # With old logic (use_base_weighting), this would have received the 1.0 based on weight
    assert unit_stats["aux_reduction_kwh"] == 0.0

    # 2. Orphaned Savings should contain the full 1.0
    assert breakdown["orphaned_aux_savings"] == 1.0
