"""Test savings calculation logic for mixed hours."""
import pytest
from unittest.mock import MagicMock
from datetime import datetime
from custom_components.heating_analytics.statistics import StatisticsManager

@pytest.fixture
def mock_coordinator():
    """Create a mock coordinator."""
    coord = MagicMock()
    coord.data = {}
    coord._hourly_log = []
    coord.auxiliary_heating_active = False # Currently OFF
    coord._accumulated_aux_impact_hour = 2.5 # But was ON earlier (mixed hour)
    coord._accumulated_energy_hour = 5.0

    # Mock calculate_total_power dependencies
    coord.energy_sensors = ["sensor.heat_pump"]
    coord.get_unit_mode.return_value = "heating"
    return coord

def test_mixed_hour_savings_captured(mock_coordinator):
    """Test that savings are captured even if auxiliary_active is False, provided aux_impact_kwh > 0."""
    stats = StatisticsManager(mock_coordinator)

    # Get the "Today" from dt_util (which is mocked via conftest implicitly for imports in statistics.py)
    # But here in test, we need to ensure we use the same date.
    # Since we can't easily access the mocked dt_util return value without importing homeassistant.util.dt
    # We will assume 2023-01-01 as per conftest, OR use the same mock.

    # Let's inspect the mock value used in statistics.py by checking today_iso logic?
    # No, easier to just use the fixed date known from conftest: 2023-01-01.
    today_iso = "2023-01-01"

    # Setup a mixed hour log:
    mixed_log = {
        "timestamp": today_iso + "T10:00:00",
        "hour": 10,
        "temp": 0.0,
        "effective_wind": 0.0,
        "solar_impact_kwh": 0.0,
        "auxiliary_active": False,
        "aux_impact_kwh": 5.0,
        "actual_kwh": 10.0,
        "unit_modes": {"sensor.heat_pump": "heating"}
    }
    mock_coordinator._hourly_log = [mixed_log]

    def side_effect_calc(*args, **kwargs):
        is_aux = kwargs.get("is_aux_active", False)
        if not is_aux:
            return {"total_kwh": 15.0}
        else:
            return {"total_kwh": 10.0, "unit_breakdown": {}}

    stats.calculate_total_power = MagicMock(side_effect=side_effect_calc)
    stats._calculate_savings_component = MagicMock(return_value=(15.0, 10.0, False, {}))

    stats.update_daily_savings_cache()

    cache = stats._daily_savings_cache
    # Should now be 5.0 (sum of aux_impact_kwh)
    assert cache["actual_savings"] == 5.0

def test_live_mixed_hour_savings_captured(mock_coordinator):
    """Test that LIVE savings are captured from coordinator accumulator."""
    stats = StatisticsManager(mock_coordinator)

    today_iso = "2023-01-01"

    # Mock cache to be empty
    stats._daily_savings_cache = {
        "date": today_iso,
        "theory_normal": 0.0, "theory_aux": 0.0, "aux_hours": 0.0, "aux_minutes": 0,
        "actual_savings": 0.0, "aux_list": [], "missing_data": False
    }

    # Mock Live Data (Coordinator tracks accumulated impact)
    mock_coordinator.data = {
        "current_calc_temp": 0.0,
        "effective_wind": 0.0,
        # This is the key value the sensor uses now
        "accumulated_aux_impact_kwh": 2.5
    }
    mock_coordinator.auxiliary_heating_active = False # Currently OFF
    mock_coordinator._accumulated_aux_impact_hour = 2.5

    stats._calculate_savings_component = MagicMock(return_value=(10.0, 5.0, False, {}))
    mock_coordinator.forecast.calculate_future_energy.return_value = (0.0, 0.0, {})

    stats.calculate_potential_savings()

    # Result stored in coordinator.data
    assert mock_coordinator.data["savings_actual_kwh"] == 2.5
