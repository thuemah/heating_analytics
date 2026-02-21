"""Unit tests for SolarCalculator with updated physics."""
import pytest
import math
from unittest.mock import MagicMock
from custom_components.heating_analytics.solar import SolarCalculator
from custom_components.heating_analytics.const import (
    DEFAULT_SOLAR_COEFF_HEATING,
    DEFAULT_SOLAR_COEFF_COOLING,
)

class MockCoordinator:
    def __init__(self):
        self.balance_point = 15.0
        self.solar_azimuth = 180  # Default South
        self._solar_coefficients_per_unit = {}

@pytest.fixture
def coordinator():
    return MockCoordinator()

@pytest.fixture
def calculator(coordinator):
    return SolarCalculator(coordinator)

def test_calculate_solar_factor_vertical_geometry(calculator):
    # Vertical Surface Logic:
    # Elevation 0 (Horizon) => cos(0) = 1.0 (Max Impact)
    # Elevation 90 (Zenith) => cos(90) = 0.0 (Min Impact)

    # Horizon (Sunset/Sunrise) - use 1.0 deg to avoid <= 0 guard
    # Elev 1, Azimuth 180 (Direct), Cloud 0
    # cos(1 deg) ~= 0.9998
    assert calculator.calculate_solar_factor(1, 180, 0) > 0.99

    # Zenith (High Noon in Tropics)
    # Elev 90, Azimuth 180
    # cos(90) = 0.0 => Total 0.0
    assert pytest.approx(calculator.calculate_solar_factor(90, 180, 0), 0.000000001) == 0.0

    # 45 degrees elevation
    # cos(45) approx 0.707
    expected = math.cos(math.radians(45))
    assert pytest.approx(calculator.calculate_solar_factor(45, 180, 0), 0.001) == expected

def test_calculate_solar_factor_azimuth_config(calculator, coordinator):
    # Case 1: Configured South (180), Sun at South (180)
    # Azimuth factor = 0.5 + 0.5 * cos(0) = 1.0
    coordinator.solar_azimuth = 180
    assert calculator.calculate_solar_factor(1, 180, 0) > 0.99

    # Case 2: Configured West (270), Sun at West (270)
    # Azimuth factor = 0.5 + 0.5 * cos(0) = 1.0
    coordinator.solar_azimuth = 270
    assert calculator.calculate_solar_factor(1, 270, 0) > 0.99

    # Case 3: Configured West (270), Sun at South (180)
    # Diff = 90 deg.
    # New Kelvin Twist: Zone 2 (75-90) -> Diffuse Floor = 0.1
    coordinator.solar_azimuth = 270
    # Elev 1 => Elev factor ~1.0. Az factor 0.1. Cloud 1.0 => Total 0.1
    assert pytest.approx(calculator.calculate_solar_factor(1, 180, 0), 0.01) == 0.1

    # Case 4: Configured West (270), Sun at East (90)
    # Diff = 180 deg.
    # New Kelvin Twist: Zone 3 (90-180) -> Backside Floor = 0.05
    coordinator.solar_azimuth = 270
    assert pytest.approx(calculator.calculate_solar_factor(1, 90, 0), 0.01) == 0.05

def test_calculate_unit_solar_impact(calculator, coordinator):
    # Setup
    unit_coeff = 2.5
    global_factor = 1.0

    # Impact = Factor * Coeff
    assert calculator.calculate_unit_solar_impact(global_factor, unit_coeff) == 2.5

    # Factor 0.5
    assert calculator.calculate_unit_solar_impact(0.5, unit_coeff) == 1.25

def test_calculate_unit_coefficient(calculator, coordinator):
    # 1. Global Default Fallback (Heating)
    # Target 10, BP 15 -> Heating
    assert calculator.calculate_unit_coefficient("unit_1", "10") == DEFAULT_SOLAR_COEFF_HEATING

    # 2. Global Default Fallback (Cooling)
    # Target 20, BP 15 -> Cooling
    assert calculator.calculate_unit_coefficient("unit_1", "20") == DEFAULT_SOLAR_COEFF_COOLING

    # 3. Exact Match
    coordinator._solar_coefficients_per_unit["unit_1"] = {"10": 0.08}
    assert calculator.calculate_unit_coefficient("unit_1", "10") == 0.08

    # 4. Closest Neighbor (Same Mode)
    # T=10 exists (0.08). T=12 should use T=10.
    assert calculator.calculate_unit_coefficient("unit_1", "12") == 0.08

    # 5. Broad Neighbor (Same Mode)
    coordinator._solar_coefficients_per_unit["unit_1"]["0"] = 0.12
    # Current state: {10: 0.08, 0: 0.12}. Target 2. Closest is 0.
    assert calculator.calculate_unit_coefficient("unit_1", "2") == 0.12
    # Target 8. Closest is 10.
    assert calculator.calculate_unit_coefficient("unit_1", "8") == 0.08

    # 6. Mode Separation
    # T=10 exists (Heating). T=20 (Cooling) should still use Cooling Default if no cooling learned.
    assert calculator.calculate_unit_coefficient("unit_1", "20") == DEFAULT_SOLAR_COEFF_COOLING

    # 7. Learned Cooling Neighbor
    coordinator._solar_coefficients_per_unit["unit_1"]["25"] = 0.30
    # Now T=20 should use T=25 (Closest cooling neighbor)
    assert calculator.calculate_unit_coefficient("unit_1", "20") == 0.30

def test_calculate_unit_coefficient_non_numeric(calculator, coordinator):
    # Mock coordinator get_unit_mode
    coordinator.get_unit_mode = MagicMock(return_value="cooling")
    assert calculator.calculate_unit_coefficient("unit_1", "invalid") == DEFAULT_SOLAR_COEFF_COOLING

def test_apply_correction(calculator, coordinator):
    # Heating (Temp 10 < 15): Subtract solar
    # Base 10, Impact 2 => 8
    assert calculator.apply_correction(10.0, 2.0, 10.0) == 8.0

    # Clamping
    assert calculator.apply_correction(1.0, 2.0, 10.0) == 0.0

    # Cooling (Temp 20 > 15): Add solar
    # Base 10, Impact 2 => 12
    assert calculator.apply_correction(10.0, 2.0, 20.0) == 12.0

def test_normalize_for_learning(calculator, coordinator):
    # Heating (Temp 10 < 15): Actual was reduced by solar. Add solar back.
    # Actual 8, Impact 2 => 10
    assert calculator.normalize_for_learning(8.0, 2.0, 10.0) == 10.0

    # Cooling (Temp 20 > 15): Actual was increased by solar. Subtract solar.
    # Actual 12, Impact 2 => 10
    assert calculator.normalize_for_learning(12.0, 2.0, 20.0) == 10.0

    # Clamping
    assert calculator.normalize_for_learning(1.0, 2.0, 20.0) == 0.0

