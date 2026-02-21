"""Test reconstruction logic in StatisticsManager."""
from datetime import date
from unittest.mock import MagicMock
import pytest
from custom_components.heating_analytics.statistics import StatisticsManager

from custom_components.heating_analytics.const import ATTR_SOLAR_FACTOR

class MockCoordinator:
    def __init__(self):
        self.balance_point = 20.0
        self.energy_sensors = ["sensor.test_energy"]
        self._daily_history = {}
        self._hourly_log = []
        self.solar_enabled = True
        self.data = {}
        self._correlation_data = {}
        self._aux_coefficients = {}
        self.solar = MagicMock()
        self.solar.calculate_unit_coefficient.return_value = 0.0
        self.solar.calculate_unit_solar_impact.return_value = 0.0
        self.solar.apply_correction.side_effect = lambda n, s, m: n - s if m == "heating" else n + s
        self.solar.calculate_saturation.side_effect = lambda n, s, m: (0.0, 0.0, n)
        self._correlation_data_per_unit = {}
        self._aux_coefficients_per_unit = {}
        self._unit_modes = {}

    def _get_predicted_kwh(self, temp_key, wind_bucket, actual_temp):
        return actual_temp

    def get_unit_mode(self, entity_id):
        return "heating"

    def _get_wind_bucket(self, effective_wind):
        return "normal"


@pytest.fixture
def stats_manager():
    """Fixture for StatisticsManager."""
    coord = MockCoordinator()
    stats = StatisticsManager(coord)
    # Mock the robust prediction model to simplify testing reconstruction logic
    stats._get_prediction_from_model = MagicMock(side_effect=lambda data_map, temp_key, wind_bucket, actual_temp, bp, apply_scaling=True: float(actual_temp))
    return stats, coord


def test_reconstruction_off(stats_manager):
    """Test reconstruction when TDD is low."""
    stats, coord = stats_manager
    date_key = "2023-01-01"
    # Temp 20, BP 20 -> TDD 0. No reconstruction. Effective Temp should be 20.
    coord._daily_history = {
        date_key: {"temp": 20.0, "tdd": 0.0, "wind": 0.0, "kwh": 10.0}
    }

    kwh, _, _, _, _ = stats.calculate_modeled_energy(date.fromisoformat(date_key), date.fromisoformat(date_key))

    # Expect: Temp 20 * 24 = 480
    assert kwh == 480.0


def test_reconstruction_heating_normal(stats_manager):
    """Test heating reconstruction."""
    stats, coord = stats_manager
    date_key = "2023-01-01"
    # Temp 10, BP 20 -> TDD 10. Effective Temp should be 10 (20 - 10).
    coord._daily_history = {
        date_key: {"temp": 10.0, "tdd": 10.0, "wind": 0.0, "kwh": 10.0}
    }

    kwh, _, _, _, _ = stats.calculate_modeled_energy(date.fromisoformat(date_key), date.fromisoformat(date_key))

    # Expect: Temp 10 * 24 = 240
    assert kwh == 240.0

def test_reconstruction_cooling_mild(stats_manager):
    """Test mild cooling reconstruction."""
    stats, coord = stats_manager
    date_key = "2023-01-01"
    # Temp 22, BP 20 -> TDD 0.5 (Some heat, some cool? Or just cooling?)
    # Logic: If Temp >= BP, we are "Warm Side".
    # Calc Temp = BP + TDD.
    # 20 + 0.5 = 20.5.
    coord._daily_history = {
        date_key: {"temp": 22.0, "tdd": 0.5, "wind": 0.0, "kwh": 10.0}
    }

    kwh, _, _, _, _ = stats.calculate_modeled_energy(date.fromisoformat(date_key), date.fromisoformat(date_key))

    # 20.5 * 24 = 492.0
    assert kwh == 492.0

def test_reconstruction_cooling_strong(stats_manager):
    """Test strong cooling reconstruction."""
    stats, coord = stats_manager
    date_key = "2023-01-01"
    # Temp 25, BP 20 -> TDD 0 (if only heating tracked) or CDD?
    # Integration assumes TDD tracks absolute difference?
    # "tdd": 5.0.
    # Temp >= BP -> Warm Side.
    # Calc = 20 + 5 = 25.
    coord._daily_history = {
        date_key: {"temp": 25.0, "tdd": 5.0, "wind": 0.0, "kwh": 10.0}
    }

    kwh, _, _, _, _ = stats.calculate_modeled_energy(date.fromisoformat(date_key), date.fromisoformat(date_key))

    # 25 * 24 = 600
    assert kwh == 600.0

def test_reconstruction_heating_near_bp(stats_manager):
    """Test heating near balance point."""
    stats, coord = stats_manager
    date_key = "2023-01-01"
    # Temp 19, BP 20 -> TDD 1. Effective Temp should be 19 (20 - 1).
    coord._daily_history = {
        date_key: {"temp": 19.0, "tdd": 1.0, "wind": 0.0, "kwh": 10.0}
    }

    kwh, _, _, _, _ = stats.calculate_modeled_energy(date.fromisoformat(date_key), date.fromisoformat(date_key))

    # 19 * 24 = 456.0
    assert kwh == 456.0

def test_reconstruction_exact_bp(stats_manager):
    """Test exact balance point."""
    stats, coord = stats_manager
    date_key = "2023-01-01"
    # Temp 20, BP 20 -> TDD 0. Effective Temp should be 20.
    coord._daily_history = {
        date_key: {"temp": 20.0, "tdd": 0.0, "wind": 0.0, "kwh": 10.0}
    }

    kwh, _, _, _, _ = stats.calculate_modeled_energy(date.fromisoformat(date_key), date.fromisoformat(date_key))

    # 20 * 24 = 480
    assert kwh == 480.0
