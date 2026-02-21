
import pytest
import os
import csv
from unittest.mock import MagicMock
from custom_components.heating_analytics.storage import StorageManager
from custom_components.heating_analytics.const import ATTR_TDD

@pytest.fixture
def mock_coordinator(hass, tmp_path):
    coordinator = MagicMock()
    coordinator.hass = hass

    # Configure async_add_executor_job to execute the function immediately
    async def mock_executor(func, *args, **kwargs):
        return func(*args, **kwargs)

    coordinator.hass.async_add_executor_job.side_effect = mock_executor

    coordinator.csv_auto_logging = True
    # Use real temp paths
    coordinator.csv_hourly_path = str(tmp_path / "test_hourly_log.csv")
    coordinator.csv_daily_path = str(tmp_path / "test_daily_log.csv")
    coordinator.energy_sensors = ["sensor.heat_pump_energy"]
    coordinator.data = {}
    return coordinator

@pytest.mark.asyncio
async def test_csv_hourly_logging_schema_evolution(hass, mock_coordinator):
    """Test that CSV logging handles schema evolution correctly."""

    # Setup
    storage = StorageManager(mock_coordinator)
    file_path = mock_coordinator.csv_hourly_path

    # 1. First write (Initial Schema)
    log_entry_1 = {
        "timestamp": "2023-10-27T10:00:00",
        "hour": 10,
        "temp": 15.0,
        "inertia_temp": 14.8,
        "effective_wind": 5.0,
        "wind_bucket": "normal",
        "actual_kwh": 1.5,
        "expected_kwh": 1.4,
        "deviation": 0.1,
        "auxiliary_active": False,
        "unit_breakdown": {"sensor.heat_pump_energy": 1.5},
        "unit_expected_breakdown": {"sensor.heat_pump_energy": 1.4}
    }

    await storage.append_hourly_log_csv(log_entry_1)

    assert os.path.exists(file_path)
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["temp"] == "15.0"
        assert "unit_0_actual" in rows[0]

    # 2. Second write (Same Schema)
    log_entry_2 = {
        "timestamp": "2023-10-27T11:00:00",
        "hour": 11,
        "temp": 16.0,
        "inertia_temp": 15.5,
        "effective_wind": 4.0,
        "wind_bucket": "normal",
        "actual_kwh": 1.2,
        "expected_kwh": 1.3,
        "deviation": -0.1,
        "auxiliary_active": False,
        "unit_breakdown": {"sensor.heat_pump_energy": 1.2},
        "unit_expected_breakdown": {"sensor.heat_pump_energy": 1.3}
    }

    await storage.append_hourly_log_csv(log_entry_2)

    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 2
        assert rows[1]["temp"] == "16.0"

    # 3. Third write (Schema Evolution: New Column via new sensor)
    # Simulate adding a new sensor
    mock_coordinator.energy_sensors.append("sensor.aux_heater")

    log_entry_3 = {
        "timestamp": "2023-10-27T12:00:00",
        "hour": 12,
        "temp": 14.0,
        "inertia_temp": 14.5,
        "effective_wind": 6.0,
        "wind_bucket": "high_wind",
        "actual_kwh": 2.5,
        "expected_kwh": 2.0,
        "deviation": 0.5,
        "auxiliary_active": True,
        "unit_breakdown": {"sensor.heat_pump_energy": 1.5, "sensor.aux_heater": 1.0},
        "unit_expected_breakdown": {"sensor.heat_pump_energy": 1.4, "sensor.aux_heater": 0.6}
    }

    await storage.append_hourly_log_csv(log_entry_3)

    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

        assert len(rows) == 3
        # Check if new columns exist
        assert "unit_1_actual" in fieldnames
        assert "unit_1_expected" in fieldnames

        # Check Row 1 (Old) - should have empty values for new cols
        # DictReader returns None or empty string depending on handling, but usually if restkey/restval not set and row is short,
        # missing keys in the row are missing from the dict OR values are None.
        # But here we REWROTE the file, so the CSV on disk HAS commas for the empty fields.
        # So they should be empty strings.
        assert rows[0]["unit_1_actual"] == ""

        # Check Row 3 (New) - should have values
        assert rows[2]["unit_1_actual"] == "1.0"
        assert rows[2]["unit_0_actual"] == "1.5"


@pytest.mark.asyncio
async def test_csv_daily_logging_schema_evolution(hass, mock_coordinator):
    """Test that CSV daily logging handles schema evolution correctly."""

    storage = StorageManager(mock_coordinator)
    file_path = mock_coordinator.csv_daily_path

    # 1. Initial write
    log_1 = {"date": "2023-10-27", "tdd": 10.0, "kwh": 20.0}
    await storage.append_daily_log_csv(log_1)

    # 2. Schema Change (New metric)
    log_2 = {"date": "2023-10-28", "tdd": 12.0, "kwh": 24.0, "solar_kwh": 5.0}
    await storage.append_daily_log_csv(log_2)

    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

        assert "solar_kwh" in fieldnames
        assert len(rows) == 2
        # Use None check if DictReader returns None for missing fields, or empty string if it's quoted empty
        val_0 = rows[0].get("solar_kwh")
        assert val_0 == "" or val_0 is None

        assert rows[1]["solar_kwh"] == "5.0"
        assert rows[1]["date"] == "2023-10-28"
