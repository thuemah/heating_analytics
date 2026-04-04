"""Unit tests for SolarCalculator with updated physics."""
import pytest
from tests.helpers import CoordinatorModelMixin
import math
from unittest.mock import MagicMock
from custom_components.heating_analytics.solar import SolarCalculator
from custom_components.heating_analytics.const import (
    DEFAULT_SOLAR_COEFF_HEATING,
    DEFAULT_SOLAR_COEFF_COOLING,
)

class MockCoordinator(CoordinatorModelMixin):
    def __init__(self):
        self.solar_azimuth = 180
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
    # Elevation 0 (Horizon) => 0.0
    # Elevation 90 (Zenith) => cos(90) = 0.0 (Min Impact)

    # 0 or negative elevation should yield 0.0
    assert calculator.calculate_solar_factor(0, 180, 0) == 0.0
    assert calculator.calculate_solar_factor(-5, 180, 0) == 0.0

    # Test standard behavior at 20 degrees
    # AM = 1 / sin(20) ~= 2.92
    # Intensity = 0.7 ^ AM ~= 0.352
    # raw_elev_factor = cos(20) ~= 0.939
    # expected_20 ~= 0.352 * 0.939 ~= 0.331
    am_20 = 1.0 / math.sin(math.radians(20))
    intensity_20 = 0.7 ** am_20
    expected_20 = math.cos(math.radians(20)) * intensity_20
    assert pytest.approx(calculator.calculate_solar_factor(20, 180, 0), 0.001) == expected_20

    # Zenith (High Noon in Tropics)
    # Elev 90, Azimuth 180
    # cos(90) = 0.0 => Total 0.0
    assert pytest.approx(calculator.calculate_solar_factor(90, 180, 0), 0.000000001) == 0.0

    # 45 degrees elevation
    am_45 = 1.0 / math.sin(math.radians(45))
    intensity_45 = 0.7 ** am_45
    expected_45 = math.cos(math.radians(45)) * intensity_45
    assert pytest.approx(calculator.calculate_solar_factor(45, 180, 0), 0.001) == expected_45

def test_calculate_solar_factor_atmospheric_attenuation(calculator):
    # New atmospheric model: AM = 1 / sin(elevation), Intensity = 0.7^AM

    # 1. Low angle (5 degrees)
    # AM = 1 / sin(5) = 11.47
    # Intensity = 0.7^11.47 ~= 0.016
    am_5 = 1.0 / math.sin(math.radians(5))
    intensity_5 = 0.7 ** am_5
    expected_5 = math.cos(math.radians(5)) * intensity_5
    assert pytest.approx(calculator.calculate_solar_factor(5.0, 180, 0), 0.001) == expected_5

    # 2. Extreme low angle (1 degree)
    # AM = 1 / sin(1) = 57.29
    # Intensity = 0.7^57.29 ~= 1.3e-9 (effectively 0)
    am_1 = 1.0 / math.sin(math.radians(1))
    intensity_1 = 0.7 ** am_1
    expected_1 = math.cos(math.radians(1)) * intensity_1
    assert pytest.approx(calculator.calculate_solar_factor(1.0, 180, 0), 0.001) == expected_1

    # 3. Mid angle (30 degrees)
    am_30 = 1.0 / math.sin(math.radians(30))
    intensity_30 = 0.7 ** am_30
    expected_30 = math.cos(math.radians(30)) * intensity_30
    assert pytest.approx(calculator.calculate_solar_factor(30.0, 180, 0), 0.001) == expected_30

def test_calculate_solar_factor_azimuth_config(calculator, coordinator):
    # Elev 20 => cos(20) ~0.9397. Intensity = 0.7^(1/sin(20)) ~0.352
    am_20 = 1.0 / math.sin(math.radians(20))
    intensity_20 = 0.7 ** am_20
    expected_elev = math.cos(math.radians(20)) * intensity_20

    # Case 1: Configured South (180), Sun at South (180)
    # Azimuth factor = 0.5 + 0.5 * cos(0) = 1.0
    coordinator.solar_azimuth = 180
    assert pytest.approx(calculator.calculate_solar_factor(20, 180, 0), 0.001) == expected_elev

    # Case 2: Configured West (270), Sun at West (270)
    # Azimuth factor = 0.5 + 0.5 * cos(0) = 1.0
    coordinator.solar_azimuth = 270
    assert pytest.approx(calculator.calculate_solar_factor(20, 270, 0), 0.001) == expected_elev

    # Case 3: Configured West (270), Sun at South (180)
    # Diff = 90 deg.
    # New Kelvin Twist: Zone 2 (75-90) -> Diffuse Floor = 0.1
    coordinator.solar_azimuth = 270
    # Elev 20 => Elev factor * 0.1. Cloud 1.0
    expected_diffuse = expected_elev * 0.1
    assert pytest.approx(calculator.calculate_solar_factor(20, 180, 0), 0.001) == expected_diffuse

    # Case 4: Configured West (270), Sun at East (90)
    # Diff = 180 deg.
    # New Kelvin Twist: Zone 3 (90-180) -> Backside Floor = 0.05
    coordinator.solar_azimuth = 270
    expected_backside = expected_elev * 0.05
    assert pytest.approx(calculator.calculate_solar_factor(20, 90, 0), 0.001) == expected_backside

def test_calculate_unit_solar_impact(calculator, coordinator):
    # Setup
    unit_coeff = {'s': 2.5, 'e': 0.0}
    global_factor = (1.0, 0.0)

    # Impact = Factor * Coeff
    assert calculator.calculate_unit_solar_impact(global_factor, unit_coeff) == 2.5

    # Factor 0.5
    assert calculator.calculate_unit_solar_impact((0.5, 0.0), unit_coeff) == 1.25

def test_calculate_unit_coefficient(calculator, coordinator):
    # 1. Global Default Fallback (Heating)
    # Target 10, BP 15 -> Heating
    coeff = calculator.calculate_unit_coefficient("unit_1", "10")
    assert coeff['s'] == DEFAULT_SOLAR_COEFF_HEATING

    # 2. Global Default Fallback (Cooling)
    # Target 20, BP 15 -> Cooling
    coeff_cooling = calculator.calculate_unit_coefficient("unit_1", "20")
    assert coeff_cooling['s'] == DEFAULT_SOLAR_COEFF_COOLING

    # 3. Learned Coefficient (Global per Unit, Not Temp-Stratified)
    # Setting a learned coefficient for the unit returns it for any temp key.
    coordinator._solar_coefficients_per_unit["unit_1"] = {"s": 0.08, "e": 0.0}
    assert calculator.calculate_unit_coefficient("unit_1", "10")["s"] == 0.08

    # 4. Same global coefficient returned for all temp keys regardless of mode.
    assert calculator.calculate_unit_coefficient("unit_1", "12")["s"] == 0.08
    assert calculator.calculate_unit_coefficient("unit_1", "20")["s"] == 0.08

    # 5. Remove learned coefficient -> falls back to mode-appropriate default.
    del coordinator._solar_coefficients_per_unit["unit_1"]
    coeff_cooling = calculator.calculate_unit_coefficient("unit_1", "20")
    assert coeff_cooling['s'] == DEFAULT_SOLAR_COEFF_COOLING

def test_calculate_unit_coefficient_non_numeric(calculator, coordinator):
    # Mock coordinator get_unit_mode
    coordinator.get_unit_mode = MagicMock(return_value="cooling")
    coeff = calculator.calculate_unit_coefficient("unit_1", "invalid")
    assert coeff['s'] == DEFAULT_SOLAR_COEFF_COOLING

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

