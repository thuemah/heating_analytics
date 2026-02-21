"""Test Reproduction of Rotated Data Issue (Missing Vectors)."""
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

@pytest.mark.asyncio
async def test_csv_import_rotated_data_missing_vectors(storage_manager, mock_coordinator):
    """Test that weather-only import creates vectors for legacy rotated data."""

    # Setup Legacy Daily History (Rotated Day, Missing Vectors)
    mock_coordinator._daily_history = {
        "2023-01-01": {
            "kwh": 50.0,
            "temp": 0.0,
            "tdd": 15.0,
            # hourly_vectors IS MISSING (simulating pre-Kelvin data)
        }
    }

    # CSV content: Data for the rotated day (2023-01-01)
    # New Temp = 10.0 at 10:00
    # Must provide >= 20 hours to satisfy data completeness guard
    rows = ["timestamp,temperature,wind_speed,cloud_coverage"]
    for h in range(24):
        t = 10.0 if h == 10 else 5.0
        w = 5.0 if h == 10 else 2.0
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
            day_data = mock_coordinator._daily_history["2023-01-01"]

            # 1. Vectors should have been created
            assert "hourly_vectors" in day_data, "hourly_vectors should be created for legacy data"
            vectors = day_data["hourly_vectors"]
            assert isinstance(vectors, dict)

            # 2. Vector content should be updated from CSV
            assert vectors["temp"][10] == 10.0
            assert vectors["wind"][10] == 5.0

            # 3. Energy data in vectors should be None/Unknown (as per requirement)
            # or defaulted to something safe.
            assert vectors["actual_kwh"][10] is None
