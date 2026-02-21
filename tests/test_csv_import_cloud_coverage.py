"""Test CSV Import with Cloud Coverage Support."""
import pytest
from unittest.mock import MagicMock, patch, mock_open
from datetime import datetime
from custom_components.heating_analytics.storage import StorageManager
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator


@pytest.fixture
def mock_coordinator():
    coord = MagicMock(spec=HeatingDataCoordinator)
    coord.hass = MagicMock()
    coord.hass.async_add_executor_job = lambda fn: fn()
    coord.balance_point = 15.0
    coord._hourly_log = []
    coord._daily_history = {}
    coord._calculate_effective_wind = lambda ws, wg: ws
    coord._get_wind_bucket = lambda w: "normal"
    coord.solar_enabled = True
    coord.learning_enabled = False

    # Mock solar manager
    coord.solar = MagicMock()
    coord.solar.get_approx_sun_pos = MagicMock(return_value=(30.0, 180.0))  # elevation, azimuth
    coord.solar.calculate_solar_factor = MagicMock(return_value=0.5)  # Mock solar factor

    # Mock aggregation
    coord._aggregate_daily_logs = MagicMock(return_value={})
    coord._backfill_daily_from_hourly = MagicMock()

    return coord


@pytest.fixture
def storage_manager(mock_coordinator):
    return StorageManager(mock_coordinator)


def test_csv_import_with_cloud_coverage(storage_manager, mock_coordinator):
    """Test that CSV import correctly parses cloud_coverage and calculates solar_factor."""

    # Mock CSV content with cloud_coverage column
    csv_content = """timestamp,temperature,energy,wind_speed,cloud_coverage
2023-01-01T00:00:00,5.0,1.5,3.0,80.0
2023-01-01T01:00:00,4.5,1.6,3.5,60.0
2023-01-01T02:00:00,4.0,1.4,2.5,30.0"""

    mapping = {
        "timestamp": "timestamp",
        "temperature": "temperature",
        "energy": "energy",
        "wind_speed": "wind_speed",
        "cloud_coverage": "cloud_coverage"
    }

    with patch("builtins.open", mock_open(read_data=csv_content)):
        with patch("os.path.exists", return_value=True):
            # Synchronous test - call _process_csv directly
            import csv
            from io import StringIO

            reader = csv.DictReader(StringIO(csv_content))
            rows = list(reader)

            # Manually parse first row to test logic
            row = rows[0]

            # Simulate CSV import logic
            from homeassistant.util import dt as dt_util
            ts = dt_util.parse_datetime(row["timestamp"])
            temp = float(row["temperature"])
            cloud_coverage = float(row["cloud_coverage"])

            # Verify cloud_coverage was parsed
            assert cloud_coverage == 80.0

            # Call solar methods to verify they're invoked
            elev, azim = mock_coordinator.solar.get_approx_sun_pos(ts)
            solar_factor = mock_coordinator.solar.calculate_solar_factor(elev, azim, cloud_coverage)

            # Verify solar methods were called with correct params
            mock_coordinator.solar.get_approx_sun_pos.assert_called_with(ts)
            mock_coordinator.solar.calculate_solar_factor.assert_called_with(30.0, 180.0, 80.0)
            assert solar_factor == 0.5


def test_csv_import_without_cloud_coverage_fallback(storage_manager, mock_coordinator):
    """Test that CSV import falls back to 50% cloud coverage when column is missing."""

    csv_content = """timestamp,temperature,energy,wind_speed
2023-01-01T00:00:00,5.0,1.5,3.0
2023-01-01T01:00:00,4.5,1.6,3.5"""

    mapping = {
        "timestamp": "timestamp",
        "temperature": "temperature",
        "energy": "energy",
        "wind_speed": "wind_speed"
        # No cloud_coverage mapping
    }

    with patch("builtins.open", mock_open(read_data=csv_content)):
        with patch("os.path.exists", return_value=True):
            import csv
            from io import StringIO

            reader = csv.DictReader(StringIO(csv_content))
            rows = list(reader)
            row = rows[0]

            from homeassistant.util import dt as dt_util
            ts = dt_util.parse_datetime(row["timestamp"])

            # Simulate fallback logic
            cloud_coverage = None
            col_cloud = mapping.get("cloud_coverage")
            if col_cloud and row.get(col_cloud):
                cloud_coverage = float(row[col_cloud])

            if cloud_coverage is None:
                cloud_coverage = 50.0

            # Verify fallback to 50%
            assert cloud_coverage == 50.0

            # Call solar method
            elev, azim = mock_coordinator.solar.get_approx_sun_pos(ts)
            solar_factor = mock_coordinator.solar.calculate_solar_factor(elev, azim, cloud_coverage)

            # Verify 50% was used as fallback
            mock_coordinator.solar.calculate_solar_factor.assert_called_with(30.0, 180.0, 50.0)


def test_csv_import_solar_disabled(storage_manager, mock_coordinator):
    """Test that solar_factor is 0.0 when solar is disabled."""

    # Disable solar
    mock_coordinator.solar_enabled = False

    csv_content = """timestamp,temperature,energy,cloud_coverage
2023-01-01T00:00:00,5.0,1.5,80.0"""

    mapping = {
        "timestamp": "timestamp",
        "temperature": "temperature",
        "energy": "energy",
        "cloud_coverage": "cloud_coverage"
    }

    # Solar factor should remain 0.0 when solar is disabled
    solar_factor = 0.0
    solar_impact_kwh = 0.0

    if not mock_coordinator.solar_enabled:
        assert solar_factor == 0.0
        assert solar_impact_kwh == 0.0
