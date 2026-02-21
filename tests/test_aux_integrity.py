"""Integrity tests for Heating Analytics."""
import pytest
from unittest.mock import MagicMock
from custom_components.heating_analytics.statistics import StatisticsManager
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator

class MockSolarCalculator:
    def apply_correction(self, value, impact, temp):
        return value
    def calculate_unit_coefficient(self, entity_id, temp_key):
        return 0.1
    def calculate_unit_solar_impact(self, factor, coeff):
        return 0.0

    def calculate_saturation(self, net_demand, solar_potential, val):
        return 0.0, 0.0, net_demand

@pytest.fixture
def integrity_coordinator(hass):
    # Mock Coordinator
    coordinator = MagicMock()
    coordinator.hass = hass
    coordinator.data = {}
    coordinator.solar = MockSolarCalculator()
    coordinator.balance_point = 18.0
    coordinator.solar_enabled = False
    coordinator.auxiliary_heating_active = True
    coordinator.energy_sensors = []

    coordinator.wind_threshold = 5.5
    coordinator.extreme_wind_threshold = 10.8

    # Implement _get_wind_bucket from the actual code logic
    def _get_wind_bucket(effective_wind, ignore_aux=False):
        if effective_wind >= coordinator.extreme_wind_threshold:
            return "extreme_wind"
        elif effective_wind >= coordinator.wind_threshold:
            return "high_wind"
        return "normal"

    coordinator._get_wind_bucket.side_effect = _get_wind_bucket

    # Mock internal dicts
    coordinator._correlation_data_per_unit = {}
    coordinator._aux_coefficients_per_unit = {}
    coordinator.aux_affected_entities = []

    # Mock get_unit_mode
    coordinator.get_unit_mode.return_value = "heating"

    # Mock Statistics _get_prediction_from_model helper (since we mock statistics in some tests, but here we test StatisticsManager)
    # We are testing StatisticsManager, so we let it run. But it calls coordinator methods.

    return coordinator

def test_savings_calculation_high_wind_regression(integrity_coordinator):
    """
    Regression Test: Verify that high wind conditions correctly use the 'high_wind' bucket
    for auxiliary impact calculations, ensuring correct savings are reported.
    """
    stats = StatisticsManager(integrity_coordinator)

    # Scenario: High Wind (6.0 m/s > 5.5)
    temp = 0.0
    eff_wind = 6.0
    solar_impact = 0.0

    # Mock Global Base Prediction (Mocking _get_predicted_kwh on coordinator)
    def _get_predicted_kwh(temp_key, bucket, eff_wind=None):
        if temp_key == "0":
            if bucket == "high_wind":
                return 12.0
            return 10.0
        return 0.0
    integrity_coordinator._get_predicted_kwh.side_effect = _get_predicted_kwh

    # Mock Global Aux Coefficients directly in _aux_coefficients
    # Scenario:
    # Normal Wind Aux Impact: 2 kWh
    # High Wind Aux Impact: 4 kWh
    integrity_coordinator._aux_coefficients = {
        "0": {
            "normal": 2.0,
            "high_wind": 4.0
        }
    }

    # Run Calculation
    pred_norm, pred_aux, missing, _ = stats._calculate_savings_component(temp, eff_wind, solar_impact)

    # Expected behavior:
    # Bucket = "high_wind"
    # Normal Mode: Global Base = 12.0.
    # Aux Mode: Global Base = 12.0. Global Aux Reduction = 4.0. Total = 8.0.
    # Savings = 12.0 - 8.0 = 4.0.

    assert pred_norm == 12.0
    assert pred_aux == 8.0
    assert (pred_norm - pred_aux) == 4.0, "Should use high_wind bucket for aux impact"

def test_aux_integrity_no_scaling(integrity_coordinator):
    """
    Systems Integrity Check (Honest Reporting): Verify that Unit Models report learned values without
    artificial scaling, even if Global Model predicts higher savings. The gap should be visible as Orphaned Savings.
    """
    stats = StatisticsManager(integrity_coordinator)

    # Setup Scenario
    # 3 Units:
    # - Heater 1: Included in Aux (Affected)
    # - Heater 2: Included in Aux (Affected)
    # - Heater 3: Excluded from Aux (Not Affected) - Exclusion Case

    integrity_coordinator.energy_sensors = ["sensor.h1", "sensor.h2", "sensor.h3"]
    integrity_coordinator.aux_affected_entities = ["sensor.h1", "sensor.h2"]
    integrity_coordinator._aux_affected_set = set(integrity_coordinator.aux_affected_entities)

    temp = 5.0 # Temp Key "5"
    eff_wind = 2.0 # Normal Wind

    # Mock Global Model (The Authority)
    # Global Base = 30.0
    # Global Aux Reduction = 10.0 (The Truth)
    def _get_predicted_kwh(temp_key, bucket, eff_wind=None):
        return 30.0
    integrity_coordinator._get_predicted_kwh.side_effect = _get_predicted_kwh

    integrity_coordinator._aux_coefficients = {
        "5": {"normal": 10.0}
    }

    # Mock Unit Models
    # Unit 1: Base=10, Aux=2
    # Unit 2: Base=10, Aux=2
    # Unit 3: Base=10, Aux=0 (Not Affected)
    # Sum of Unit Aux = 2 + 2 = 4.0
    # Global Aux = 10.0
    # Gap = 6.0. Previously, this gap was filled by scaling. Now it should be Orphaned.

    integrity_coordinator._correlation_data_per_unit = {
        "sensor.h1": {"5": {"normal": 10.0}},
        "sensor.h2": {"5": {"normal": 10.0}},
        "sensor.h3": {"5": {"normal": 10.0}},
    }

    integrity_coordinator._aux_coefficients_per_unit = {
        "sensor.h1": {"5": {"normal": 2.0}},
        "sensor.h2": {"5": {"normal": 2.0}},
        "sensor.h3": {"5": {"normal": 0.0}}, # Should be ignored anyway
    }

    # Run Calculation
    result = stats.calculate_total_power(
        temp=temp,
        effective_wind=eff_wind,
        solar_impact=0.0,
        is_aux_active=True
    )

    # Verify Global Integrity
    assert result["total_kwh"] == 20.0 # 30 - 10
    assert result["global_aux_reduction_kwh"] == 10.0

    # Verify NO Scaling (Honest Reporting)
    bd = result["unit_breakdown"]

    # Check H1
    assert bd["sensor.h1"]["raw_aux_kwh"] == 2.0 # Raw
    assert bd["sensor.h1"]["aux_reduction_kwh"] == 2.0 # Not Scaled
    assert bd["sensor.h1"]["net_kwh"] == 8.0 # 10 - 2

    # Check H2
    assert bd["sensor.h2"]["raw_aux_kwh"] == 2.0
    assert bd["sensor.h2"]["aux_reduction_kwh"] == 2.0
    assert bd["sensor.h2"]["net_kwh"] == 8.0

    # Check H3
    assert bd["sensor.h3"]["aux_reduction_kwh"] == 0.0
    assert bd["sensor.h3"]["net_kwh"] == 10.0

    # Verify Orphaned Savings
    # Global (10) - Units (2+2=4) = 6.0 Orphaned
    assert result["breakdown"]["aux_reduction_kwh"] == 4.0
    assert result["breakdown"]["orphaned_aux_savings"] == pytest.approx(6.0)

    # Verify Unspecified kWh (Deviation between Track A and Track B)
    # Global Net (20.0) - Sum Unit Net (8+8+10 = 26.0) = -6.0
    assert result["breakdown"]["unspecified_kwh"] == pytest.approx(-6.0)
