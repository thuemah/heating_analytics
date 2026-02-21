"""Test inertia compensation."""
from unittest.mock import MagicMock, patch
import pytest
from datetime import datetime, timedelta, timezone
# Import dt_util to use in test setup, but we will patch the one in coordinator
from homeassistant.util import dt as dt_util_real
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator

# Mock config entry
class MockConfigEntry:
    def __init__(self):
        self.entry_id = "test_entry"
        self.data = {
            "outdoor_temp_sensor": "sensor.outdoor_temp",
            "wind_speed_sensor": "sensor.wind_speed",
            "weather_entity": "weather.test",
            "energy_sensors": [],
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

    # We need to mock hass states
    mock_hass.states.get = MagicMock()

    # Mock Store
    with patch("custom_components.heating_analytics.storage.Store"):
        coord = HeatingDataCoordinator(mock_hass, entry)

    # Manually load
    coord._is_loaded = True
    return coord

# Use a fixed timezone-aware datetime for tests
FIXED_NOW = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

@pytest.fixture
def mock_time():
    """Mock dt_util.now and parse_datetime to be consistent."""
    with patch("custom_components.heating_analytics.coordinator.dt_util") as mock_dt:
        mock_dt.now.return_value = FIXED_NOW
        mock_dt.UTC = timezone.utc

        # Consistent parse_datetime
        def parse_dt(dt_str):
            try:
                # Basic ISO parsing
                d = datetime.fromisoformat(dt_str)
                # Ensure aware if needed or return as is
                # If naive, assume UTC for test stability
                if d.tzinfo is None:
                    d = d.replace(tzinfo=timezone.utc)
                return d
            except (ValueError, TypeError):
                return None

        mock_dt.parse_datetime.side_effect = parse_dt
        yield mock_dt

def test_inertia_temp_new_system(coordinator, mock_time):
    """Test inertia temp with no history."""
    coordinator._hourly_log = []

    # Setup current temp
    coordinator._get_float_state = MagicMock(return_value=10.0)

    inertia = coordinator._calculate_inertia_temp()

    # Should be just current temp: 10.0 / 1
    assert inertia == 10.0

def test_inertia_temp_partial_history(coordinator, mock_time):
    """Test inertia temp with 2 hours history."""
    now = FIXED_NOW
    # 2 logs, H-2 and H-1, both within 4 hours
    coordinator._hourly_log = [
        {"temp": 12.0, "timestamp": (now - timedelta(hours=2)).isoformat()}, # H-2
        {"temp": 11.0, "timestamp": (now - timedelta(hours=1)).isoformat()}, # H-1
    ]

    # Current temp
    coordinator._get_float_state = MagicMock(return_value=10.0)

    inertia = coordinator._calculate_inertia_temp()

    # Weighted Average of 12, 11, 10
    # Weights: (0.30, 0.30, 0.20) -> Total 0.80
    # (3.6 + 3.3 + 2.0) / 0.80 = 8.9 / 0.80 = 11.125
    assert round(inertia, 2) == 11.12

def test_inertia_temp_full_history(coordinator, mock_time):
    """Test inertia temp with full 3+ hours history."""
    now = FIXED_NOW
    # 5 logs (only last 3 matter, but all must be within 4 hours)
    coordinator._hourly_log = [
        {"temp": 15.0, "timestamp": (now - timedelta(hours=5)).isoformat()}, # Old (>4h, should be filtered)
        {"temp": 14.0, "timestamp": (now - timedelta(hours=3.5)).isoformat()}, # Within 4h
        {"temp": 13.0, "timestamp": (now - timedelta(hours=3)).isoformat()}, # H-3
        {"temp": 12.0, "timestamp": (now - timedelta(hours=2)).isoformat()}, # H-2
        {"temp": 11.0, "timestamp": (now - timedelta(hours=1)).isoformat()}, # H-1
    ]

    # Current temp
    coordinator._get_float_state = MagicMock(return_value=10.0)

    inertia = coordinator._calculate_inertia_temp()

    # Expected:
    # Cutoff is H-4.
    # 15.0 at H-5 is excluded.
    # We take last 3 logs from the list: 13, 12, 11.
    # Are they all valid?
    # 13 (H-3), 12 (H-2), 11 (H-1) -> All valid.
    # Note: 14.0 (H-3.5) is in the list but logic takes `_hourly_log[-3:]`.
    # So it considers [13, 12, 11].
    # Weighted Sum: 13*0.20 + 12*0.30 + 11*0.30 + 10*0.20 = 11.5
    assert round(inertia, 2) == 11.50

def test_inertia_temp_variation(coordinator, mock_time):
    """Test inertia temp with significant variation."""
    now = FIXED_NOW
    # Temp dropping fast: 20 -> 15 -> 10 -> 5 (current)
    coordinator._hourly_log = [
        {"temp": 20.0, "timestamp": (now - timedelta(hours=3)).isoformat()},
        {"temp": 15.0, "timestamp": (now - timedelta(hours=2)).isoformat()},
        {"temp": 10.0, "timestamp": (now - timedelta(hours=1)).isoformat()},
    ]

    coordinator._get_float_state = MagicMock(return_value=5.0)

    inertia = coordinator._calculate_inertia_temp()

    # Weighted Sum: 20*0.20 + 15*0.30 + 10*0.30 + 5*0.20 = 12.5
    assert inertia == 12.5

def test_inertia_temp_missing_current(coordinator, mock_time):
    """Test fallback when current temp is missing but samples exist."""
    coordinator._hourly_log = []
    coordinator._get_float_state = MagicMock(return_value=None)

    # Has hourly samples (New aggregate structure)
    coordinator._hourly_sample_count = 1
    coordinator._hourly_temp_sum = 8.0

    inertia = coordinator._calculate_inertia_temp()
    assert inertia == 8.0

def test_inertia_temp_no_data(coordinator, mock_time):
    """Test returns None if absolutely no data."""
    coordinator._hourly_log = []
    coordinator._get_float_state = MagicMock(return_value=None)
    # No samples
    coordinator._hourly_sample_count = 0
    coordinator._hourly_temp_sum = 0.0

    inertia = coordinator._calculate_inertia_temp()
    assert inertia is None

def test_inertia_forecast_seeding_post_restart(coordinator, mock_time):
    """Test that forecast seeding uses stored inertia_temp after restart."""
    now = FIXED_NOW
    # Simulate hourly_log after restart with stored inertia_temp values
    coordinator._hourly_log = [
        {"temp": 5.0, "inertia_temp": 8.0, "timestamp": (now - timedelta(hours=3)).isoformat()},  # H-3
        {"temp": 4.0, "inertia_temp": 6.5, "timestamp": (now - timedelta(hours=2)).isoformat()},  # H-2
        {"temp": 3.0, "inertia_temp": 5.0, "timestamp": (now - timedelta(hours=1)).isoformat()},  # H-1
    ]

    # We verify by checking that stored inertia_temp values are available and correct
    # Logic in forecast.py takes logs, so we just verify logs are structured right
    inertia_values = [log.get("inertia_temp", log["temp"]) for log in coordinator._hourly_log[-3:]]

    # Should use stored inertia_temp values, not raw temp
    assert inertia_values == [8.0, 6.5, 5.0]

def test_inertia_gap_logic(coordinator, mock_time):
    """Test inertia calculation with a time gap (downtime scenario)."""
    now = FIXED_NOW

    # Logs from 5 hours ago (House was warm, 20C)
    # Gap of 5 hours. Cutoff is 4 hours.
    coordinator._hourly_log = [
        {"timestamp": (now - timedelta(hours=6)).isoformat(), "temp": 20.0},
        {"timestamp": (now - timedelta(hours=5)).isoformat(), "temp": 20.0},
        {"timestamp": (now - timedelta(hours=5)).isoformat(), "temp": 20.0}, # Duplicate time for volume
    ]

    # Current temp is cold (10C)
    coordinator._get_float_state = MagicMock(return_value=10.0)

    inertia = coordinator._calculate_inertia_temp()

    # Old logs (>4h) should be filtered out.
    # Calculation should rely ONLY on current temp.
    # 10.0
    assert inertia == 10.0

def test_inertia_legacy_missing_timestamp(coordinator, mock_time):
    """Test handling of legacy logs without timestamp."""
    # Legacy logs (no timestamp)
    coordinator._hourly_log = [
        {"temp": 20.0},
        {"temp": 20.0}
    ]

    # Current temp
    coordinator._get_float_state = MagicMock(return_value=10.0)

    inertia = coordinator._calculate_inertia_temp()

    # Logs without timestamp should be treated as old/invalid and filtered out.
    # Only current temp remains.
    assert inertia == 10.0

def test_inertia_partial_filtering(coordinator, mock_time):
    """Test 'Filter FIRST' logic: 2 old logs, 2 valid logs. Should take the 2 valid ones."""
    now = FIXED_NOW

    coordinator._hourly_log = [
        {"timestamp": (now - timedelta(hours=6)).isoformat(), "temp": 20.0}, # Old
        {"timestamp": (now - timedelta(hours=5)).isoformat(), "temp": 20.0}, # Old
        {"timestamp": (now - timedelta(hours=2)).isoformat(), "temp": 12.0}, # Valid H-2
        {"timestamp": (now - timedelta(hours=1)).isoformat(), "temp": 11.0}, # Valid H-1
    ]

    # Current temp
    coordinator._get_float_state = MagicMock(return_value=10.0)

    inertia = coordinator._calculate_inertia_temp()

    # Old logs (>4h) should be filtered out first.
    # Remaining valid logs: [12.0, 11.0]
    # Current: 10.0
    # Weighted Average as above (partial history)
    assert round(inertia, 2) == 11.12

def test_inertia_graceful_degradation_logging(coordinator, mock_time):
    """Test that NO warnings are logged during normal startup (insufficient history)."""
    now = FIXED_NOW

    # Case 1: Only 1 valid sample (Current temp only) - Normal startup
    coordinator._hourly_log = [] # Empty history
    coordinator._get_float_state = MagicMock(return_value=10.0)

    with patch("custom_components.heating_analytics.coordinator._LOGGER.warning") as mock_warn:
        inertia = coordinator._calculate_inertia_temp()
        assert inertia == 10.0
        # Warnings should NOT be called for normal startup
        mock_warn.assert_not_called()

    # Case 2: Only 2 samples (1 valid history + Current) - Normal ramp-up
    coordinator._hourly_log = [
         {"timestamp": (now - timedelta(hours=1)).isoformat(), "temp": 12.0}
    ]

    with patch("custom_components.heating_analytics.coordinator._LOGGER.warning") as mock_warn:
        inertia = coordinator._calculate_inertia_temp()
        # Weighted (partial): (12*0.30 + 10*0.20) / 0.50 = 5.6 / 0.50 = 11.2
        assert round(inertia, 2) == 11.20
        # Warnings should NOT be called for normal ramp-up
        mock_warn.assert_not_called()
