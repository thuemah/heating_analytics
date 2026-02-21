"""Test the _update_accumulated_impacts method in the coordinator."""
from unittest.mock import MagicMock
import pytest
from homeassistant.util import dt as dt_util

from custom_components.heating_analytics.coordinator import HeatingDataCoordinator


@pytest.fixture
def mock_coordinator():
    """Fixture for a mock coordinator."""
    coordinator = MagicMock(spec=HeatingDataCoordinator)
    coordinator.data = {}
    coordinator._hourly_log = []
    coordinator._hourly_delta_per_unit = {}
    coordinator._hourly_expected_base_per_unit = {}
    coordinator.auxiliary_heating_active = False
    coordinator.get_unit_mode = MagicMock(return_value="heating")
    coordinator._accumulated_aux_impact_hour = 0.0  # Added for new minute-by-minute tracking
    coordinator._accumulated_energy_hour = 0.0
    coordinator.balance_point = 17.0
    return coordinator


def test_accumulated_impacts_start_of_day(mock_coordinator):
    """Test accumulated impacts at the beginning of the day."""
    current_time = dt_util.parse_datetime("2023-01-01T00:15:00+00:00")
    mock_coordinator.data = {
        "accumulated_solar_impact_kwh": 0.0,
        "accumulated_guest_impact_kwh": 0.0,
        "accumulated_aux_impact_kwh": 0.0,
        "solar_impact_kwh": 0.2,
        "last_hour_aux_impact_kwh": 0.5,
    }
    mock_coordinator.auxiliary_heating_active = True

    # Simulate accumulation (since we replaced the naive calculation)
    # The new logic uses _accumulated_aux_impact_hour directly
    # Previously: 0.5 * 0.25 = 0.125
    mock_coordinator._accumulated_aux_impact_hour = 0.125

    HeatingDataCoordinator._update_accumulated_impacts(mock_coordinator, current_time)

    assert mock_coordinator.data["accumulated_solar_impact_kwh"] == pytest.approx(0.2 * 0.25)
    assert mock_coordinator.data["accumulated_guest_impact_kwh"] == 0.0
    # Now validates against the precise accumulator
    assert mock_coordinator.data["accumulated_aux_impact_kwh"] == pytest.approx(0.125)


def test_accumulated_impacts_with_hourly_logs(mock_coordinator):
    """Test accumulated impacts with existing hourly logs."""
    current_time = dt_util.parse_datetime("2023-01-01T02:30:00+00:00")
    mock_coordinator._hourly_log = [
        {"timestamp": "2023-01-01T00:00:00", "solar_impact_kwh": 0.1, "guest_impact_kwh": 0.0, "aux_impact_kwh": 0.3},
        {"timestamp": "2023-01-01T01:00:00", "solar_impact_kwh": 0.15, "guest_impact_kwh": 0.5, "aux_impact_kwh": 0.35},
    ]
    mock_coordinator.data = {
        "accumulated_solar_impact_kwh": 0.0,
        "accumulated_guest_impact_kwh": 0.0,
        "accumulated_aux_impact_kwh": 0.0,
        "solar_impact_kwh": 0.3,
        "last_hour_aux_impact_kwh": 0.6,
    }
    mock_coordinator.auxiliary_heating_active = True

    # Simulate accumulation for the current hour
    # Previously: 0.6 * 0.5 = 0.3
    mock_coordinator._accumulated_aux_impact_hour = 0.3

    HeatingDataCoordinator._update_accumulated_impacts(mock_coordinator, current_time)

    assert mock_coordinator.data["accumulated_solar_impact_kwh"] == pytest.approx(0.1 + 0.15 + (0.3 * 0.5))
    assert mock_coordinator.data["accumulated_guest_impact_kwh"] == pytest.approx(0.5)
    # 0.3 (Log1) + 0.35 (Log2) + 0.3 (Live) = 0.95
    assert mock_coordinator.data["accumulated_aux_impact_kwh"] == pytest.approx(0.3 + 0.35 + 0.3)


def test_live_guest_impact(mock_coordinator):
    """Test the live guest impact calculation."""
    current_time = dt_util.parse_datetime("2023-01-01T01:30:00+00:00")
    mock_coordinator.get_unit_mode.return_value = "guest_heating"
    mock_coordinator._hourly_delta_per_unit = {"sensor.test": 1.0}
    # This value is already prorated for the 30 minutes that have passed
    mock_coordinator._hourly_expected_base_per_unit = {"sensor.test": 0.4}
    mock_coordinator.data = {
        "accumulated_solar_impact_kwh": 0.0,
        "accumulated_guest_impact_kwh": 0.0,
        "accumulated_aux_impact_kwh": 0.0,
        "solar_impact_kwh": 0.0,
        "last_hour_aux_impact_kwh": 0.0,
    }
    HeatingDataCoordinator._update_accumulated_impacts(mock_coordinator, current_time)

    # Guest units contribute their full consumption as impact (Baseline = 0)
    assert mock_coordinator.data["accumulated_guest_impact_kwh"] == pytest.approx(1.0)
