import pytest
from unittest.mock import MagicMock
from custom_components.heating_analytics.statistics import StatisticsManager
from custom_components.heating_analytics.solar import SolarCalculator
from custom_components.heating_analytics.const import MODE_HEATING

class MockCoordinator:
    def __init__(self):
        self.energy_sensors = ["unit_1", "unit_2"]
        self.solar_azimuth = 180
        self.balance_point = 18.0
        self.solar_enabled = True
        self.solar_correction_percent = 100.0
        self.screen_config = (False, False, False)
        self.solar = SolarCalculator(self)
        self.data = {"solar_factor": 1.0}
        self.model = MagicMock()
        self.model.correlation_data_per_unit = {
            "unit_1": {"18": {"normal": 0.5}},
            "unit_2": {"18": {"normal": 2.0}},
        }
        self.model.aux_coefficients_per_unit = {}
        # Simple Mock for solar coefficients
        self.model.solar_coefficients_per_unit = {
            "unit_1": {"heating": {"s": 1.0, "e": 0.0, "w": 0.0}},
            "unit_2": {"heating": {"s": 0.5, "e": 0.0, "w": 0.0}},
        }
        
    def _get_predicted_kwh(self, temp_key, wind_bucket, temp):
        return 2.5 # Base sum

    def _get_wind_bucket(self, wind):
        return "normal"

    def get_unit_mode(self, entity_id):
        return MODE_HEATING

    def screen_config_for_entity(self, entity_id):
        return (False, False, False)

def test_solar_saturation_divergence():
    coordinator = MockCoordinator()
    stats = StatisticsManager(coordinator)
    
    # 1. Setup: 
    # Unit 1: Base 0.5, Solar Potential 1.0 -> Should saturate at 0.5
    # Unit 2: Base 2.0, Solar Potential 0.5 -> No saturation
    # Total Base: 2.5, Total Potential: 1.5
    
    # Run calculation
    # Temp 18 (Balance Point), Wind 0, Solar Impact 0 (overridden by factor 1.0)
    result = stats.calculate_total_power(
        temp=18.0,
        effective_wind=0.0,
        solar_impact=0.0,
        is_aux_active=False
    )
    
    # Current behavior analysis:
    # Unit 1 (Applied): min(1.0, 0.5) = 0.5
    # Unit 2 (Applied): min(0.5, 2.0) = 0.5
    # Sum Applied: 1.0
    # Global Net = 2.5 - 1.0 = 1.5
    
    # Expected behavior (if global saturation):
    # Total Potential: 1.5
    # Global Applied: min(1.5, 2.5) = 1.5
    # Global Net = 2.5 - 1.5 = 1.0
    
    unit_breakdown = result["unit_breakdown"]
    sum_unit_applied = sum(u["solar_reduction_kwh"] for u in unit_breakdown.values())
    global_total_kwh = result["total_kwh"]
    
    print(f"\nSum Unit Applied: {sum_unit_applied}")
    print(f"Global Net: {global_total_kwh}")
    
    # This assertion verifies the divergence currently exists
    assert sum_unit_applied == 1.0
    assert global_total_kwh == 1.5
