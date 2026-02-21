"""Test temperature key calculation logic."""
from unittest.mock import MagicMock, patch, AsyncMock
import pytest
from datetime import datetime, timedelta, timezone
from homeassistant.util import dt as dt_util
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator

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

@pytest.mark.asyncio
async def test_inertia_temp_calculation(hass, mock_time):
    """Test the inertia temperature calculation (rolling average)."""
    entry = MagicMock()
    entry.data = {"outdoor_temp_sensor": "sensor.outdoor_temp"}

    with patch("custom_components.heating_analytics.storage.Store"):
        coordinator = HeatingDataCoordinator(hass, entry)

        # Mock get_float_state for current temp
        coordinator._get_float_state = MagicMock(return_value=10.0)

        # 1. No History
        coordinator._hourly_log = []
        # Expect: current temp
        assert coordinator._calculate_inertia_temp() == 10.0

        now = FIXED_NOW

        # 2. History available (within 4 hours)
        coordinator._hourly_log = [
            {"temp": 0.0, "timestamp": (now - timedelta(hours=3)).isoformat()},
            {"temp": 5.0, "timestamp": (now - timedelta(hours=2)).isoformat()},
            {"temp": 5.0, "timestamp": (now - timedelta(hours=1)).isoformat()} # Last 3
        ]
        # Expect: Weighted Avg. Weights=(0.20, 0.30, 0.30, 0.20)
        # 0.20*0 + 0.30*5 + 0.30*5 + 0.20*10 = 0 + 1.5 + 1.5 + 2 = 5.0
        assert coordinator._calculate_inertia_temp() == 5.0

        # 3. Partial History (e.g., just started)
        coordinator._hourly_log = [{"temp": 6.0, "timestamp": (now - timedelta(hours=1)).isoformat()}]
        # Expect: Weighted Avg of (6.0, 10.0). Weights aligned right: (0.30, 0.20). Sum=0.50
        # (6*0.30 + 10*0.20)/0.50 = (1.8+2.0)/0.50 = 3.8/0.50 = 7.6
        assert coordinator._calculate_inertia_temp() == pytest.approx(7.6, 0.01)

@pytest.mark.asyncio
async def test_temp_key_rounding(hass, mock_time):
    """Test temperature key rounding logic."""
    entry = MagicMock()
    entry.data = {"balance_point": 17.0, "learning_rate": 0.1}

    with patch("custom_components.heating_analytics.storage.Store"):
        coordinator = HeatingDataCoordinator(hass, entry)
        coordinator._async_save_data = AsyncMock()

        current_time = FIXED_NOW

        # Case A: 5.4
        coordinator._hourly_sample_count = 1
        coordinator._hourly_wind_values = [0.0]
        coordinator._hourly_bucket_counts = {"normal": 1}
        coordinator._hourly_temp_sum = 5.4
        coordinator._hourly_log = []

        await coordinator._process_hourly_data(current_time)
        assert coordinator._hourly_log[0]["temp_key"] == "5"

        # Case B: 5.6
        coordinator._hourly_sample_count = 1
        coordinator._hourly_wind_values = [0.0] # Reset needed!
        coordinator._hourly_bucket_counts = {"normal": 1}
        coordinator._hourly_temp_sum = 5.6
        coordinator._hourly_log = []

        await coordinator._process_hourly_data(current_time)
        assert coordinator._hourly_log[0]["temp_key"] == "6"

        # Case C: -1.5
        coordinator._hourly_sample_count = 1
        coordinator._hourly_wind_values = [0.0] # Reset needed!
        coordinator._hourly_bucket_counts = {"normal": 1}
        coordinator._hourly_temp_sum = -1.5
        coordinator._hourly_log = []

        await coordinator._process_hourly_data(current_time)
        assert coordinator._hourly_log[0]["temp_key"] == "-2"

@pytest.mark.asyncio
async def test_inertia_fallback(hass, mock_time):
    """Test fallback when current temp is unavailable."""
    entry = MagicMock()
    entry.data = {"outdoor_temp_sensor": "sensor.outdoor_temp"}

    with patch("custom_components.heating_analytics.storage.Store"):
        coordinator = HeatingDataCoordinator(hass, entry)

        # Current temp unavailable
        coordinator._get_float_state = MagicMock(return_value=None)

        now = FIXED_NOW

        # History available
        coordinator._hourly_log = [
            {"temp": 5.0, "timestamp": (now - timedelta(hours=2)).isoformat()},
            {"temp": 5.0, "timestamp": (now - timedelta(hours=1)).isoformat()}
        ]

        # Should return average of history only?
        # Code:
        # if current_temp is None:
        #    if samples > 0: use sample avg
        #    else: skip current

        # Scenario 1: No samples, sensor dead.
        assert coordinator._calculate_inertia_temp() == 5.0

        # Scenario 2: Samples available (e.g. from earlier in hour), sensor dead now.
        coordinator._hourly_sample_count = 10
        coordinator._hourly_temp_sum = 100.0 # Avg 10.0

        # Expect: Weighted Avg(5, 5, 10). Weights aligned right: (0.30, 0.30, 0.20). Sum=0.80
        # (5*0.30 + 5*0.30 + 10*0.20)/0.80 = (1.5 + 1.5 + 2.0)/0.80 = 5.0/0.80 = 6.25
        assert coordinator._calculate_inertia_temp() == pytest.approx(6.25, 0.01)
