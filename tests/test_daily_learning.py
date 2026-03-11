import pytest
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime, date, timedelta
from homeassistant.util import dt as dt_util
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.storage import StorageManager
from custom_components.heating_analytics.const import DEFAULT_THERMAL_MASS

@pytest.fixture
def mock_coordinator():
    coord = MagicMock()
    coord.hass = MagicMock()
    coord.entry = MagicMock()
    coord.entry.entry_id = "test_entry"
    coord.data = {}
    coord._learned_u_coefficient = None
    coord._last_midnight_indoor_temp = None
    coord.thermal_mass_kwh_per_degree = 0.0
    coord.learning_enabled = True
    coord.learning_rate = 0.01
    coord._correlation_data = {}
    coord._hourly_wind_values = []
    coord._hourly_log = []
    return coord

def _get_real_coordinator(hass):
    entry = MagicMock()
    entry.data = {
        "energy_sensors": ["sensor.heater1"],
        "outdoor_temp_sensor": "sensor.outdoor_temp",
        "thermal_mass_kwh_per_degree": 2.9,
        "indoor_temp_sensor": "sensor.indoor_temp"
    }
    coord = HeatingDataCoordinator(hass, entry)
    coord.storage = MagicMock()
    coord.forecast = MagicMock()
    coord.statistics = MagicMock()
    return coord

@pytest.mark.asyncio
async def test_daily_learning_storage_save_load(mock_coordinator):
    storage = StorageManager(mock_coordinator)
    storage._store = AsyncMock()

    mock_coordinator._learned_u_coefficient = 1.234
    mock_coordinator._last_midnight_indoor_temp = 22.5

    await storage.async_save_data(force=True)
    saved_data = storage._store.async_save.call_args[0][0]
    assert saved_data["learned_u_coefficient"] == 1.234
    assert saved_data["last_midnight_indoor_temp"] == 22.5

    storage._store.async_load.return_value = {
        "learned_u_coefficient": 5.678,
        "last_midnight_indoor_temp": 19.0,
        "accumulation_start_time": None,
        "last_updated": dt_util.now().isoformat(),
        "daily_history": {},
        "correlation_data": {}
    }
    await storage.async_load_data()
    assert mock_coordinator._learned_u_coefficient == 5.678
    assert mock_coordinator.data["learned_u_coefficient"] == 5.678
    assert mock_coordinator._last_midnight_indoor_temp == 19.0

@pytest.mark.asyncio
async def test_daily_learning_thermal_mass_correction(hass):
    coord = _get_real_coordinator(hass)
    coord.learning_enabled = True
    coord._last_midnight_indoor_temp = 20.0

    def _mock_float_state(entity_id):
        if entity_id == "sensor.indoor_temp":
            return 22.0
        return None

    coord._get_float_state = MagicMock(side_effect=_mock_float_state)
    coord._get_wind_bucket = MagicMock(return_value="normal")
    coord.storage.append_daily_log_csv = AsyncMock()
    coord._async_save_data = AsyncMock()

    base_time = dt_util.now().replace(hour=0, minute=0, second=0, microsecond=0)
    for i in range(24):
        coord._hourly_log.append({
            "timestamp": (base_time + timedelta(hours=i)).isoformat(),
            "hour": i,
            "actual_kwh": 2.0,
            "expected_kwh": 2.0,
            "temp": 0.0,
            "tdd": 1.0,
            "effective_wind": 0.0,
            "wind_bucket": "normal",
        })

    date_obj = base_time.date()
    await coord._process_daily_data(date_obj)

    assert coord._learned_u_coefficient is not None
    assert round(coord._learned_u_coefficient, 4) == 1.7583
    assert coord._last_midnight_indoor_temp == 22.0

    assert "0" in coord._correlation_data
    assert "normal" in coord._correlation_data["0"]
    assert round(coord._correlation_data["0"]["normal"], 4) == 1.7583

@pytest.mark.asyncio
async def test_daily_learning_rejects_partial_day(hass):
    coord = _get_real_coordinator(hass)
    coord.learning_enabled = True
    coord._last_midnight_indoor_temp = 20.0

    def _mock_float_state(entity_id):
        if entity_id == "sensor.indoor_temp":
            return 20.0
        return None

    coord._get_float_state = MagicMock(side_effect=_mock_float_state)
    coord._get_wind_bucket = MagicMock(return_value="normal")
    coord.storage.append_daily_log_csv = AsyncMock()
    coord._async_save_data = AsyncMock()

    base_time = dt_util.now().replace(hour=0, minute=0, second=0, microsecond=0)
    for i in range(19): # Only 19 hours
        coord._hourly_log.append({
            "timestamp": (base_time + timedelta(hours=i)).isoformat(),
            "hour": i,
            "actual_kwh": 2.0,
            "expected_kwh": 2.0,
            "temp": 0.0,
            "tdd": 1.0,
            "effective_wind": 0.0,
            "wind_bucket": "normal",
        })

    date_obj = base_time.date()
    await coord._process_daily_data(date_obj)

    assert coord._learned_u_coefficient is None, "Should reject partial day learning"
