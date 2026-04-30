
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator

@pytest.fixture
def mock_coordinator_backfill(hass):
    entry = MagicMock()
    entry.data = {
        "balance_point": 15.0,
        "energy_sensors": ["sensor.heater_1", "sensor.heater_2"],
        "aux_affected_entities": ["sensor.heater_1"]
    }

    with patch("custom_components.heating_analytics.storage.Store"):
        coordinator = HeatingDataCoordinator(hass, entry)
        # Mock storage manager to avoid actual file I/O
        coordinator.storage.async_save_data = MagicMock()
        return coordinator

def test_backfill_creates_vectors_and_breakdowns(mock_coordinator_backfill):
    coordinator = mock_coordinator_backfill

    # Setup: Create 24 hours of logs for a specific day
    day_str = "2023-10-27"
    logs = []

    for hour in range(24):
        entry = {
            "timestamp": f"{day_str}T{hour:02d}:00:00",
            "hour": hour,
            "temp": 10.0,
            "tdd": 0.5, # (15 - 10) / 24 = 5/24 ~= 0.208. Let's just use dummy values.
            "effective_wind": 5.0,
            "wind_bucket": "normal",
            "actual_kwh": 1.0,
            "expected_kwh": 0.9,
            "forecasted_kwh": 0.9,
            "solar_factor": 0.0,
            "solar_impact_kwh": 0.0,
            "aux_impact_kwh": 0.1,
            "guest_impact_kwh": 0.0,
            "unit_breakdown": {
                "sensor.heater_1": 0.6,
                "sensor.heater_2": 0.4
            },
            "unit_expected_breakdown": {
                "sensor.heater_1": 0.5,
                "sensor.heater_2": 0.4
            },
            "primary_entity": "weather.home",
            "secondary_entity": "weather.backup",
            "crossover_day": 0
        }
        logs.append(entry)

    coordinator._hourly_log = logs
    coordinator._daily_history = {} # Start empty

    # Action: Run Backfill
    updated_count = coordinator._backfill_daily_from_hourly()

    # Verification
    assert updated_count == 1
    assert day_str in coordinator._daily_history

    daily = coordinator._daily_history[day_str]

    # Check Aggregates
    assert daily["kwh"] == 24.0
    assert daily["expected_kwh"] == 21.6 # 0.9 * 24
    assert daily["aux_impact_kwh"] == 2.4 # 0.1 * 24

    # Check Unit Breakdown (Summed)
    assert daily["unit_breakdown"]["sensor.heater_1"] == pytest.approx(14.4) # 0.6 * 24
    assert daily["unit_breakdown"]["sensor.heater_2"] == pytest.approx(9.6) # 0.4 * 24

    # Check Vectors
    vectors = daily["hourly_vectors"]
    assert vectors is not None
    assert len(vectors["actual_kwh"]) == 24
    assert vectors["actual_kwh"][0] == 1.0
    assert vectors["temp"][0] == 10.0

    # Check Provenance
    assert daily["primary_entity"] == "weather.home"
    assert daily["secondary_entity"] == "weather.backup"

def test_backfill_handles_partial_data_correctly(mock_coordinator_backfill):
    coordinator = mock_coordinator_backfill
    day_str = "2023-10-28"

    # Scenario 1: Only 10 hours of logs (incomplete)
    # Logic says: if len < 20, create it if missing, but maybe warn?
    # actually code says:
    # if date_key not in self._daily_history:
    #     if len(logs) >= 20: ...
    # So if < 20 and not in history, it does NOT create it.

    logs = []
    for hour in range(10):
        entry = {
            "timestamp": f"{day_str}T{hour:02d}:00:00",
            "hour": hour,
            "actual_kwh": 1.0,
            "temp": 10.0,
            "tdd": 0.1,
            "unit_breakdown": {}
        }
        logs.append(entry)

    coordinator._hourly_log = logs
    coordinator._daily_history = {}

    updated_count = coordinator._backfill_daily_from_hourly()

    # Assert NOT created because < 20 logs
    assert updated_count == 0
    assert day_str not in coordinator._daily_history

    # Scenario 2: Existing history exists, but is "full" (24h worth of energy).
    # Logs are partial (e.g. rotated out).
    # Logic: diff = abs(log_kwh - hist_kwh). If diff > threshold and hist > log, SKIP.

    # Create fake history entry that looks complete
    coordinator._daily_history[day_str] = {
        "kwh": 24.0, # 24 hours * 1.0
        "temp": 10.0,
        "hourly_vectors": None # Missing vectors, trying to backfill
    }

    # Logs sum to 10.0 (10 hours)
    # History is 24.0
    # Diff = 14.0. Threshold = max(1, 24*0.05) = 1.2.
    # Diff > Threshold and Hist > Log.
    # Should SKIP backfill to avoid overwriting good total with partial sum.

    updated_count = coordinator._backfill_daily_from_hourly()

    assert updated_count == 0
    # Ensure vectors are STILL None (didn't overwrite)
    assert coordinator._daily_history[day_str]["hourly_vectors"] is None

    # Scenario 3: Logs are complete (24h), History exists (but maybe partial or same).
    # Create 24 logs
    logs = []
    for hour in range(24):
        entry = {
            "timestamp": f"{day_str}T{hour:02d}:00:00",
            "hour": hour,
            "actual_kwh": 1.0,
            "temp": 10.0,
            "tdd": 0.1,
            "unit_breakdown": {}
        }
        logs.append(entry)
    coordinator._hourly_log = logs

    # History exists (24.0). Logs sum to 24.0. Match.
    # Should update/enrich.

    updated_count = coordinator._backfill_daily_from_hourly()

    assert updated_count == 1
    # Verify vectors now populated
    assert coordinator._daily_history[day_str]["hourly_vectors"] is not None
    assert len(coordinator._daily_history[day_str]["hourly_vectors"]["actual_kwh"]) == 24

def test_backfill_aggregates_duplicate_hours(mock_coordinator_backfill):
    """Test DST Fallback scenario (25h day)."""
    coordinator = mock_coordinator_backfill
    day_str = "2023-10-29"
    logs = []

    # Normal hours 0-23. But let's duplicate hour 2.
    # Hour 2A: 1.0 kWh
    # Hour 2B: 2.0 kWh
    # Total for Hour 2 should be 3.0 kWh.

    entry_a = {
        "timestamp": f"{day_str}T02:00:00+02:00",
        "hour": 2,
        "actual_kwh": 1.0,
        "temp": 10.0,
        "tdd": 0.1,
        "unit_breakdown": {}
    }
    entry_b = {
        "timestamp": f"{day_str}T02:00:00+01:00", # Clock went back
        "hour": 2,
        "actual_kwh": 2.0,
        "temp": 12.0, # Diff temp
        "tdd": 0.2,
        "unit_breakdown": {}
    }

    logs.append(entry_a)
    logs.append(entry_b)

    # Fill rest with 0 to make it "valid" (>20)
    for h in range(24):
        if h == 2: continue
        logs.append({
            "timestamp": f"{day_str}T{h:02d}:00:00",
            "hour": h,
            "actual_kwh": 0.0,
            "temp": 10.0,
            "tdd": 0.0,
            "unit_breakdown": {}
        })

    coordinator._hourly_log = logs
    coordinator._daily_history = {}

    coordinator._backfill_daily_from_hourly()

    daily = coordinator._daily_history[day_str]
    vectors = daily["hourly_vectors"]

    # Hour 2 should be sum of A+B
    assert vectors["actual_kwh"][2] == 3.0

    # Temp should be average of A+B (10+12)/2 = 11
    assert vectors["temp"][2] == 11.0
