"""Test solar factor logic with extended types."""
import pytest
from unittest.mock import MagicMock
from custom_components.heating_analytics.solar import SolarCalculator
from custom_components.heating_analytics.const import MODE_HEATING, MODE_COOLING

class MockCoordinator:
    def __init__(self):
        self.balance_point = 15.0
        self.solar_window_area = 10.0
        self.solar_azimuth = 180
        self._solar_coefficients = {}

@pytest.fixture
def coordinator():
    return MockCoordinator()

@pytest.fixture
def calculator(coordinator):
    return SolarCalculator(coordinator)

def test_apply_correction_float_input(calculator):
    """Test apply_correction with float input (derived mode)."""
    # 1. Heating Mode (Temp 10 < BP 15)
    # Base 10, Impact 2. Should subtract. Result 8.
    assert calculator.apply_correction(10.0, 2.0, 10.0) == 8.0

    # 2. Cooling Mode (Temp 20 > BP 15)
    # Base 10, Impact 2. Should add. Result 12.
    assert calculator.apply_correction(10.0, 2.0, 20.0) == 12.0

    # 3. Clamping (Heating)
    # Base 1, Impact 2. Result -1 -> 0.
    assert calculator.apply_correction(1.0, 2.0, 10.0) == 0.0

def test_apply_correction_str_input(calculator):
    """Test apply_correction with string input (explicit mode)."""
    # 1. HEATING
    assert calculator.apply_correction(10.0, 2.0, MODE_HEATING) == 8.0

    # 2. COOLING
    assert calculator.apply_correction(10.0, 2.0, MODE_COOLING) == 12.0

    # 3. UNKNOWN (e.g. OFF)
    # Should return base unchanged
    assert calculator.apply_correction(10.0, 2.0, "unknown_mode") == 10.0

def test_normalize_for_learning_float_input(calculator):
    """Test normalize_for_learning with float input."""
    # 1. Heating (Temp 10 < 15)
    # Actual 8, Impact 2. Heating reduces actual. Normalize adds back. 8+2=10.
    assert calculator.normalize_for_learning(8.0, 2.0, 10.0) == 10.0

    # 2. Cooling (Temp 20 > 15)
    # Actual 12, Impact 2. Cooling increases actual. Normalize subtracts. 12-2=10.
    assert calculator.normalize_for_learning(12.0, 2.0, 20.0) == 10.0

def test_normalize_for_learning_str_input(calculator):
    """Test normalize_for_learning with string input."""
    # 1. HEATING
    assert calculator.normalize_for_learning(8.0, 2.0, MODE_HEATING) == 10.0

    # 2. COOLING
    assert calculator.normalize_for_learning(12.0, 2.0, MODE_COOLING) == 10.0
