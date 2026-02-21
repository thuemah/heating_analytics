"""Test Thermodynamic Integrity and Jensen's Inequality handling."""
from unittest.mock import MagicMock, patch
from datetime import date, datetime, timedelta
import pytest

from custom_components.heating_analytics.statistics import StatisticsManager

# Balance Point = 15.0
# Model: 1.0 kWh per degree below 15.
# Predicted(T) = max(0, 15 - T)

def mock_get_predicted_kwh(temp_key, bucket, actual_temp):
    """Mock that uses the precise temperature for calculation."""
    # This mock now correctly uses the non-rounded temp, fixing the core issue.
    t = actual_temp
    # Simple linear model: 1 kWh per degree diff
    if t < 15.0:
        return (15.0 - t) * 1.0
    return 0.0

def mock_get_predicted_kwh_per_unit(entity_id, temp_key, bucket, actual_temp):
    """Mock for per-unit prediction that also uses precise temperature."""
    # Split across 2 units
    return mock_get_predicted_kwh(temp_key, bucket, actual_temp) / 2.0

@pytest.fixture
def mock_coordinator():
    coord = MagicMock()
    coord.balance_point = 15.0
    coord.solar_enabled = False
    # Use a lambda to pass the arguments correctly from the MagicMock call to the standalone mock function
    coord._get_predicted_kwh.side_effect = lambda tk, b, at: mock_get_predicted_kwh(tk, b, at)
    coord.energy_sensors = ["sensor.heater_1", "sensor.heater_2"]
    coord._get_predicted_kwh_per_unit.side_effect = lambda eid, tk, b, at: mock_get_predicted_kwh_per_unit(eid, tk, b, at)

    # Fix: update _get_wind_bucket signature to accept ignore_aux
    coord._get_wind_bucket.side_effect = lambda w, ignore_aux=False: "normal"

    coord._daily_history = {}
    coord._hourly_log = []

    # Mock solar calculator
    coord.solar = MagicMock()
    coord.solar.apply_correction = MagicMock(side_effect=lambda v, i, t: v)
    coord.solar.calculate_saturation.side_effect = lambda net, pot, val: (0.0, 0.0, net)
    coord.solar.calculate_unit_coefficient.return_value = 0.0
    coord.solar.calculate_unit_solar_impact.return_value = 0.0

    return coord

def test_jensens_inequality_daily_fallback(mock_coordinator):
    """Test that daily fallback uses TDD to reconstruct effective temperature."""
    stats = StatisticsManager(mock_coordinator)

    target_date = date(2023, 1, 1)
    date_iso = target_date.isoformat()

    mock_coordinator._daily_history[date_iso] = {
        "kwh": 60.0,
        "tdd": 2.5,
        "temp": 14.9,
        "wind": 0.0,
        "solar_factor": 0.0
    }

    predicted_kwh, _, _, _, _ = stats.calculate_modeled_energy(target_date, target_date)

    # Reconstructed Temp = 15 - 2.5 = 12.5C
    # Model(12.5) = (15 - 12.5) * 1.0 = 2.5 kW
    # Since we split across 2 units: (2.5/2 + 2.5/2) = 2.5 kW Total
    # Daily = 2.5 * 24 = 60.0 kWh

    # The previous test asserted 72.0 (3.0 * 24) implying a different TDD or temp.
    # 2.5 TDD = 2.5 * 24 = 60 DegreeHours.
    # If Temp was 14.9, diff is 0.1. 0.1 * 24 = 2.4 DegreeHours.
    # The reconstruction logic uses TDD (2.5) if provided > 0.1.
    # So it uses TDD=2.5. Reconstructed temp = 15 - 2.5 = 12.5.
    # Model at 12.5 = 2.5 kW.
    # Daily = 60 kWh.

    # Why did it assert 72.0 before? The old mock implementation used the rounded
    # temperature (12C from temp_key) instead of the reconstructed 12.5C.
    # Model(12) = 3.0 kW * 24h = 72 kWh.
    # With the corrected mock using actual_temp, it should be:
    # Model(12.5) = 2.5 kW * 24h = 60 kWh.

    assert predicted_kwh == pytest.approx(60.0, abs=1.0)
