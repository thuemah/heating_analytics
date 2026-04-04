import pytest
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime, date, timedelta
from homeassistant.util import dt as dt_util
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.storage import StorageManager
from custom_components.heating_analytics.const import DEFAULT_THERMAL_MASS, MODE_OFF, MODE_DHW, MODE_COOLING

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
    coord._collector.wind_values = []
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
    coord.daily_learning_mode = True
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
            "temp_key": "0",
            "tdd": 1.0,
            "effective_wind": 0.0,
            "wind_bucket": "normal",
            "unit_breakdown": {"sensor.heater1": 2.0},
        })

    date_obj = base_time.date()
    await coord._process_daily_data(date_obj)

    assert coord._learned_u_coefficient is not None
    # U-coefficient uses thermal-mass-corrected total: (48 - 2.9*2) / 24 = 1.7583
    assert round(coord._learned_u_coefficient, 4) == 1.7583
    assert coord._last_midnight_indoor_temp == 22.0

    # Track B flattened daily learning writes q_adjusted/24 to one bucket.
    # q_adjusted = 48 - 2.9*2 = 42.2, q_hourly_avg = 42.2/24 = 1.7583
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
            "temp_key": "0",
            "tdd": 1.0,
            "effective_wind": 0.0,
            "wind_bucket": "normal",
            "unit_breakdown": {"sensor.heater1": 2.0},
        })

    date_obj = base_time.date()
    await coord._process_daily_data(date_obj)

    assert coord._learned_u_coefficient is None, "Should reject partial day learning"


# --- Mode filtering (#789) ---


def test_compute_excluded_mode_energy():
    """_compute_excluded_mode_energy sums kWh for excluded modes."""
    day_logs = [
        {
            "unit_breakdown": {"sensor.h1": 2.0, "sensor.h2": 1.0, "sensor.ac": 0.5},
            "unit_modes": {"sensor.h2": "off", "sensor.ac": "cooling"},
            # sensor.h1 has no mode entry → defaults to heating → included
        },
        {
            "unit_breakdown": {"sensor.h1": 2.0, "sensor.h2": 0.5, "sensor.ac": 0.8},
            "unit_modes": {"sensor.h2": "dhw", "sensor.ac": "cooling"},
        },
    ]
    excluded = HeatingDataCoordinator._compute_excluded_mode_energy(day_logs)
    # h2: 1.0 (off) + 0.5 (dhw) = 1.5
    # ac: 0.5 (cooling) + 0.8 (cooling) = 1.3
    assert round(excluded, 2) == 2.8


def test_compute_excluded_mode_energy_no_modes():
    """All units default to heating when unit_modes is absent."""
    day_logs = [{"unit_breakdown": {"sensor.h1": 3.0}}]
    assert HeatingDataCoordinator._compute_excluded_mode_energy(day_logs) == 0.0


@pytest.mark.asyncio
async def test_daily_learning_excludes_off_dhw_energy(hass):
    """Track B q_adjusted must exclude OFF/DHW energy (#789)."""
    coord = _get_real_coordinator(hass)
    coord.learning_enabled = True
    coord.daily_learning_mode = True
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
        # sensor.heater1 alternates: 20 hours heating (2 kWh), 4 hours DHW (1 kWh)
        mode = "dhw" if i < 4 else "heating"
        kwh = 1.0 if i < 4 else 2.0
        unit_modes = {"sensor.heater1": mode} if mode != "heating" else {}
        coord._hourly_log.append({
            "timestamp": (base_time + timedelta(hours=i)).isoformat(),
            "hour": i,
            "actual_kwh": kwh,
            "expected_kwh": 2.0,
            "temp": 0.0,
            "temp_key": "0",
            "tdd": 1.0,
            "effective_wind": 0.0,
            "wind_bucket": "normal",
            "unit_breakdown": {"sensor.heater1": kwh},
            "unit_modes": unit_modes,
        })

    date_obj = base_time.date()
    await coord._process_daily_data(date_obj)

    # Total raw = 4*1 + 20*2 = 44 kWh.  Excluded (DHW) = 4*1 = 4 kWh.
    # Filtered base = 44 - 4 = 40.  thermal_mass = 2.9 * (22-20) = 5.8
    # q_adjusted = 40 - 5.8 = 34.2.  U = 34.2 / 24 = 1.425
    assert coord._learned_u_coefficient is not None
    assert round(coord._learned_u_coefficient, 3) == 1.425
