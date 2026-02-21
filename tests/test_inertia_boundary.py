"""Test inertia boundary and gap logic."""
from unittest.mock import MagicMock, patch
import pytest
from datetime import datetime, timedelta, timezone
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator

# Mock config entry
class MockConfigEntry:
    def __init__(self):
        self.entry_id = "test_entry"
        self.data = {
            "outdoor_temp_sensor": "sensor.outdoor_temp",
            "energy_sensors": ["sensor.heater_1"],
        }

@pytest.fixture
def mock_hass():
    hass = MagicMock()
    hass.data = {}
    return hass

@pytest.fixture
def coordinator(mock_hass):
    entry = MockConfigEntry()
    mock_hass.data["heating_analytics"] = {}
    mock_hass.data["heating_analytics"][entry.entry_id] = MagicMock()
    mock_hass.states.get = MagicMock()

    with patch("custom_components.heating_analytics.storage.Store"):
        # Patch statistics manager to avoid complex init
        with patch("custom_components.heating_analytics.statistics.StatisticsManager"):
            coord = HeatingDataCoordinator(mock_hass, entry)
            # Re-attach real StatisticsManager if needed, or mock its methods
            # For now we mock it
            coord.statistics = MagicMock()

    coord._is_loaded = True
    return coord

# Fixed time: 15:00 UTC
FIXED_NOW = datetime(2023, 1, 1, 15, 0, 0, tzinfo=timezone.utc)

@pytest.fixture
def mock_time():
    with patch("custom_components.heating_analytics.coordinator.dt_util") as mock_dt:
        mock_dt.now.return_value = FIXED_NOW
        mock_dt.UTC = timezone.utc

        def parse_dt(dt_str):
            try:
                d = datetime.fromisoformat(dt_str)
                if d.tzinfo is None:
                    d = d.replace(tzinfo=timezone.utc)
                return d
            except (ValueError, TypeError):
                return None

        mock_dt.parse_datetime.side_effect = parse_dt
        yield mock_dt

def test_get_recent_log_temps_boundary(coordinator, mock_time):
    """Test off-by-one error in _get_recent_log_temps."""
    # We are at 15:00. Processing hour 14:00-15:00.
    # We want history logs for 11:00, 12:00, 13:00 (3 logs).
    # Combined with current 14:00, this gives 4h inertia.

    # Logs in history (timestamps are start of hour)
    coordinator._hourly_log = [
        {"timestamp": "2023-01-01T11:00:00+00:00", "temp": 11.0}, # T_4 (should be included)
        {"timestamp": "2023-01-01T12:00:00+00:00", "temp": 12.0}, # T_3
        {"timestamp": "2023-01-01T13:00:00+00:00", "temp": 13.0}, # T_2
    ]

    # Current time 15:00
    current_time = FIXED_NOW # 15:00

    # Max gap 4 hours. Cutoff = 15:00 - 4h = 11:00.
    # Logic should be: log_time >= cutoff.
    # 11:00 >= 11:00 -> True.
    # Current code: log_time > cutoff. -> False.

    temps = coordinator._get_recent_log_temps(current_time, hours_back=3, max_gap_hours=4)

    # If bug exists, 11:00 is excluded, so we get [12.0, 13.0] (2 items)
    # If fixed, we get [11.0, 12.0, 13.0] (3 items)

    assert len(temps) == 3, f"Expected 3 logs, got {len(temps)}: {temps}"
    assert temps == [11.0, 12.0, 13.0]

def test_close_hour_gap_per_unit(coordinator, mock_time):
    """Test that _close_hour_gap updates per-unit expectations."""
    # Setup state
    coordinator.data["current_model_rate"] = 10.0 # Global rate
    coordinator.data["current_calc_temp"] = 5.0
    coordinator.data["effective_wind"] = 10.0
    coordinator.data[None] = 0.0 # Solar impact attr placeholder (mocked key access?)
    # Fix attribute access for ATTR_SOLAR_IMPACT which is constant
    # We just mock statistics.calculate_total_power

    # Mock statistics response
    coordinator.statistics.calculate_total_power.return_value = {
        "total_kwh": 10.0,
        "global_base_kwh": 10.0,
        "breakdown": {"base_kwh": 10.0, "aux_reduction_kwh": 0.0, "solar_reduction_kwh": 0.0},
        "unit_breakdown": {
            "sensor.heater_1": {"net_kwh": 6.0, "base_kwh": 6.0},
            "sensor.heater_2": {"net_kwh": 4.0, "base_kwh": 4.0} # Assume another heater exists in breakdown even if not in energy_sensors for this test
        }
    }

    coordinator._hourly_expected_per_unit = {"sensor.heater_1": 100.0}

    # Simulate gap
    # Last processed: minute 29. Missed 30 mins (30..59).
    # Fraction = 30/60 = 0.5.
    # Refactor: Pass aggregates. We use avg_temp=5.0 to match setup.
    coordinator._close_hour_gap(FIXED_NOW, last_minute=29, avg_temp=5.0, avg_wind=10.0)

    # Global accumulated expected should increase by 10.0 * 0.5 = 5.0
    # But we mock _accumulated_expected_energy_hour which is 0.0 init
    assert coordinator._accumulated_expected_energy_hour == 5.0

    # Per unit expected should increase
    # Heater 1: 6.0 * 0.5 = 3.0. Total 100 + 3 = 103.
    # Heater 2: 4.0 * 0.5 = 2.0. Total 0 + 2 = 2.

    assert "sensor.heater_1" in coordinator._hourly_expected_per_unit
    assert coordinator._hourly_expected_per_unit["sensor.heater_1"] == 103.0

    assert "sensor.heater_2" in coordinator._hourly_expected_per_unit
    assert coordinator._hourly_expected_per_unit["sensor.heater_2"] == 2.0
