"""Test Wind Unit Conversion consistency."""
from unittest.mock import MagicMock, patch
import pytest
from datetime import datetime
import sys
import logging

from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.forecast import ForecastManager
from custom_components.heating_analytics.helpers import convert_speed_to_ms
from custom_components.heating_analytics.const import DOMAIN

@pytest.fixture
def mock_hass():
    """Mock Home Assistant."""
    hass = MagicMock()
    hass.config.units.is_metric = True
    return hass

@pytest.fixture
def mock_entry():
    """Mock ConfigEntry."""
    entry = MagicMock()
    entry.data = {
        "outdoor_temp_sensor": "sensor.outdoor_temp",
        "weather_entity": "weather.home",
        "wind_speed_sensor": "sensor.wind_speed",
    }
    return entry

@pytest.fixture
def coordinator(mock_hass, mock_entry):
    """Mock Coordinator."""
    coord = HeatingDataCoordinator(mock_hass, mock_entry)
    # Mock internal components to avoid complex setup
    coord.solar = MagicMock()
    coord.statistics = MagicMock()
    coord.learning = MagicMock()
    coord.storage = MagicMock()
    return coord

def test_coordinator_wind_conversion_kph(coordinator):
    """Test coordinator handles 'kph' correctly."""
    # Setup mock state for wind sensor
    state = MagicMock()
    state.state = "36.0" # 36 km/h = 10 m/s
    state.attributes = {"unit_of_measurement": "kph"}
    coordinator.hass.states.get.return_value = state

    # Test
    speed = coordinator._get_speed_in_ms("sensor.wind_speed")
    assert speed == 10.0

    # Test km/t
    state.attributes = {"unit_of_measurement": "km/t"}
    speed = coordinator._get_speed_in_ms("sensor.wind_speed")
    assert speed == 10.0

def test_forecast_wind_conversion_kph_success(coordinator):
    """Test forecast manager handles 'kph' correctly (Bug Fix Verification)."""
    forecast_manager = ForecastManager(coordinator)

    # Mock cached forecast (Used as Live Forecast)
    forecast_manager._live_forecast = [
        {
            "datetime": "2023-01-01T12:00:00",
            "temperature": 10.0,
            "wind_speed": 36.0, # 36 kph should be 10 m/s
            "condition": "cloudy"
        }
    ]

    # Mock weather entity state for unit
    weather_state = MagicMock()
    weather_state.attributes = {"wind_speed_unit": "kph"}
    coordinator.hass.states.get.return_value = weather_state

    # Mock coordinator methods needed
    coordinator._calculate_effective_wind = MagicMock(return_value=10.0)
    coordinator._get_wind_bucket = lambda w, ignore_aux=False: "normal"

    # FIX: _get_predicted_kwh takes 3 args: temp_key, wind_bucket, effective_wind
    # We define a proper lambda accepting 3 arguments
    coordinator._get_predicted_kwh = MagicMock(side_effect=lambda t, w, e=None: 1.0)

    # Mock solar
    coordinator.solar_enabled = False

    start_time = datetime(2023, 1, 1, 0, 0, 0)
    end_time = datetime(2023, 1, 1, 23, 59, 59)
    inertia = [10.0]

    # Create a local mock for dt_util used in forecast.py
    # Ensure it returns comparable datetime objects
    local_mock_dt = MagicMock()
    local_mock_dt.parse_datetime.side_effect = lambda x: datetime.fromisoformat(x)
    local_mock_dt.as_local.side_effect = lambda x: x

    # Patch dt_util inside forecast.py
    with patch("custom_components.heating_analytics.forecast.dt_util", local_mock_dt):
        forecast_manager.sum_forecast_energy(start_time, end_time, inertia)

    # Check what wind speed was passed
    args, _ = coordinator._calculate_effective_wind.call_args
    passed_speed = args[0]

    # Assert that it IS 10.0 (Correctly converted)
    assert passed_speed == 10.0

def test_ms_no_conversion():
    """Test that m/s values are returned unchanged without warning."""
    # Test standard m/s
    val = convert_speed_to_ms(10.0, "m/s")
    assert val == 10.0

    # Test variant 'ms'
    val = convert_speed_to_ms(5.5, "ms")
    assert val == 5.5

def test_unknown_unit_logging(caplog):
    """Test that unknown units trigger a warning."""
    with caplog.at_level(logging.WARNING):
        val = convert_speed_to_ms(10.0, "parsecs/hour")
        assert val == 10.0
        assert "Unknown speed unit: parsecs/hour" in caplog.text

def test_ms_no_warning_logging(caplog):
    """Test that m/s does NOT trigger warning (regression test for bug)."""
    with caplog.at_level(logging.WARNING):
        val = convert_speed_to_ms(10.0, "m/s")
        assert val == 10.0
        assert "Unknown speed unit" not in caplog.text
