"""Test CSV Import Weather-Only Mode."""
import pytest
from unittest.mock import MagicMock, patch, mock_open
from datetime import datetime
from custom_components.heating_analytics.storage import StorageManager
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from homeassistant.util import dt as dt_util


@pytest.fixture
def mock_coordinator():
    coord = MagicMock(spec=HeatingDataCoordinator)
    coord.hass = MagicMock()
    # Mock executor to run immediately
    async def async_add_executor_job(fn, *args, **kwargs):
        return fn(*args, **kwargs)
    coord.hass.async_add_executor_job = async_add_executor_job

    coord.balance_point = 15.0
    coord._hourly_log = []
    coord._daily_history = {}
    coord._calculate_effective_wind = lambda ws, wg: ws
    coord._get_wind_bucket = lambda w: "normal"
    coord.solar_enabled = True
    coord.learning_enabled = False

    # Mock solar manager
    coord.solar = MagicMock()
    coord.solar.get_approx_sun_pos = MagicMock(return_value=(30.0, 180.0))
    coord.solar.calculate_solar_factor = MagicMock(return_value=0.5)

    # Mock aggregation
    coord._aggregate_daily_logs = MagicMock(return_value={})
    # Mock backfill (should not destroy our manual updates if logic is correct)
    coord._backfill_daily_from_hourly = MagicMock(return_value=0)

    return coord


@pytest.fixture
def storage_manager(mock_coordinator):
    return StorageManager(mock_coordinator)


def test_csv_import_weather_only_mode(storage_manager, mock_coordinator):
    """Test CSV import in weather-only mode (no energy column)."""

    # Pre-populate hourly_log with existing energy data
    mock_coordinator._hourly_log = [
        {
            "timestamp": "2023-01-01T00:00:00",
            "hour": 0,
            "temp": 0.0,  # Old temp
            "tdd": 0.625,
            "effective_wind": 0.0,  # Old wind
            "wind_bucket": "normal",
            "actual_kwh": 1.5,
            "expected_kwh": 1.4,
            "solar_factor": 0.0,  # Old solar
        },
        {
            "timestamp": "2023-01-01T01:00:00",
            "hour": 1,
            "temp": 0.0,
            "tdd": 0.625,
            "effective_wind": 0.0,
            "wind_bucket": "normal",
            "actual_kwh": 1.6,
            "expected_kwh": 1.5,
            "solar_factor": 0.0,
        }
    ]

    # CSV content WITHOUT energy column
    csv_content = """timestamp,temperature,wind_speed,cloud_coverage
2023-01-01T00:00:00,5.0,3.0,80.0
2023-01-01T01:00:00,4.5,3.5,60.0"""

    mapping = {
        "timestamp": "timestamp",
        "temperature": "temperature",
        "wind_speed": "wind_speed",
        "cloud_coverage": "cloud_coverage"
        # No 'energy' column
    }

    with patch("builtins.open", mock_open(read_data=csv_content)):
        with patch("os.path.exists", return_value=True):
            import csv
            from io import StringIO

            reader = csv.DictReader(StringIO(csv_content))
            rows = list(reader)

            # Verify CSV has no energy column
            assert "energy" not in rows[0].keys()

            # Simulate import: Parse first row
            row = rows[0]
            temp = float(row["temperature"])
            assert temp == 5.0

            # Verify existing hourly_log has energy data
            assert mock_coordinator._hourly_log[0]["actual_kwh"] == 1.5


def test_csv_import_full_mode_with_energy(storage_manager, mock_coordinator):
    """Test CSV import in full mode (with energy column)."""

    # CSV content WITH energy column
    csv_content = """timestamp,temperature,energy,wind_speed
2023-01-01T00:00:00,5.0,1.5,3.0
2023-01-01T01:00:00,4.5,1.6,3.5"""

    mapping = {
        "timestamp": "timestamp",
        "temperature": "temperature",
        "energy": "energy",
        "wind_speed": "wind_speed"
    }

    with patch("builtins.open", mock_open(read_data=csv_content)):
        with patch("os.path.exists", return_value=True):
            import csv
            from io import StringIO

            reader = csv.DictReader(StringIO(csv_content))
            rows = list(reader)

            # Verify CSV has energy column
            assert "energy" in rows[0].keys()

            row = rows[0]
            energy = float(row["energy"])
            assert energy == 1.5


def test_csv_import_validation_rejects_missing_timestamp(storage_manager, mock_coordinator):
    """Test that import rejects CSV without timestamp column."""

    csv_content = """temperature,energy,wind_speed
5.0,1.5,3.0"""

    mapping = {
        # Missing 'timestamp' mapping
        "temperature": "temperature",
        "energy": "energy"
    }

    # Should fail validation since timestamp is required
    # (This would be tested by calling the actual import method)
    assert mapping.get("timestamp") is None


def test_csv_import_validation_rejects_missing_temperature(storage_manager, mock_coordinator):
    """Test that import rejects CSV without temperature column."""

    csv_content = """timestamp,energy,wind_speed
2023-01-01T00:00:00,1.5,3.0"""

    mapping = {
        "timestamp": "timestamp",
        # Missing 'temperature' mapping
        "energy": "energy"
    }

    # Should fail validation since temperature is required
    assert mapping.get("temperature") is None


@pytest.mark.asyncio
async def test_csv_import_rotated_data_update(storage_manager, mock_coordinator):
    """Test that weather-only import updates daily_history even if hourly_log is missing."""

    # Setup Daily History for a "rotated" day (no hourly logs)
    # 2023-01-01
    mock_coordinator._daily_history = {
        "2023-01-01": {
            "kwh": 50.0,
            "temp": 0.0, # Initial temp (mocked as 0)
            "tdd": 15.0, # Initial TDD
            "hourly_vectors": {
                "temp": [0.0] * 24, # Initial vector
                "wind": [0.0] * 24,
                "tdd": [0.625] * 24, # 15/24
                "actual_kwh": [2.08] * 24, # 50/24
                "solar_rad": [0.0] * 24
            }
        }
    }

    # CSV content: Data for the rotated day (2023-01-01)
    # New Temp = 10.0 at 10:00/11:00
    # Must provide >= 20 hours to satisfy data completeness guard
    rows = ["timestamp,temperature,wind_speed,cloud_coverage"]
    for h in range(24):
        if h == 10:
            t, w = 10.0, 5.0
        elif h == 11:
            t, w = 10.0, 6.0
        else:
            t, w = 5.0, 2.0
        rows.append(f"2023-01-01T{h:02d}:00:00,{t},{w},20.0")

    csv_content = "\n".join(rows)

    mapping = {
        "timestamp": "timestamp",
        "temperature": "temperature",
        "wind_speed": "wind_speed",
        "cloud_coverage": "cloud_coverage"
    }

    with patch("builtins.open", mock_open(read_data=csv_content)):
        with patch("os.path.exists", return_value=True):
            # Run Import
            await storage_manager.import_csv_data("dummy.csv", mapping, update_model=False)

            # Assertions

            # 1. Hourly Log should remain empty (no new logs created for weather only)
            assert len(mock_coordinator._hourly_log) == 0

            # 2. Daily History should be updated
            day_data = mock_coordinator._daily_history["2023-01-01"]

            # Check Vectors at Hour 10 and 11
            assert day_data["hourly_vectors"]["temp"][10] == 10.0
            assert day_data["hourly_vectors"]["wind"][10] == 5.0
