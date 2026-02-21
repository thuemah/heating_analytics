"""Test weather fallback logic (Strict Source Selection)."""
from unittest.mock import MagicMock, patch, AsyncMock
import pytest
from datetime import datetime
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.const import (
    CONF_OUTDOOR_TEMP_SOURCE,
    CONF_WIND_SOURCE,
    CONF_WIND_GUST_SOURCE,
    SOURCE_SENSOR,
    SOURCE_WEATHER,
)

@pytest.fixture
def mock_entry():
    entry = MagicMock()
    entry.data = {
        "outdoor_temp_sensor": "sensor.temp",
        "weather_entity": "weather.home",
        "wind_speed_sensor": "sensor.wind",
        "wind_gust_sensor": "sensor.gust",
        # Default to SENSOR if not specified, but we specify explicitly in tests
        CONF_OUTDOOR_TEMP_SOURCE: SOURCE_SENSOR,
        CONF_WIND_SOURCE: SOURCE_SENSOR,
        CONF_WIND_GUST_SOURCE: SOURCE_SENSOR,
        "wind_threshold": 5.5,
        "extreme_wind_threshold": 10.8,
    }
    return entry

@pytest.mark.asyncio
async def test_weather_source_selection(hass, mock_entry):
    """Test that coordinator respects source selection (Weather vs Sensor)."""
    # Config: Temp from Weather, Wind from Sensor
    mock_entry.data[CONF_OUTDOOR_TEMP_SOURCE] = SOURCE_WEATHER
    mock_entry.data[CONF_WIND_SOURCE] = SOURCE_SENSOR

    coordinator = HeatingDataCoordinator(hass, mock_entry)
    coordinator.solar = MagicMock()
    coordinator.statistics = MagicMock()
    # Fix return value for calculate_total_power
    coordinator.statistics.calculate_total_power.return_value = {
        'total_kwh': 1.0,
        'breakdown': {
            'base_kwh': 1.0,
            'aux_reduction_kwh': 0.0,
            'solar_reduction_kwh': 0.0
        },
        'unit_breakdown': {}
    }

    coordinator.learning = MagicMock()
    coordinator.storage = MagicMock()
    coordinator.storage.async_load_data = AsyncMock()
    coordinator.storage.async_save_data = AsyncMock()

    # Mock Forecast
    coordinator.forecast = MagicMock()
    coordinator.forecast.update_daily_forecast = AsyncMock()
    coordinator.forecast.calculate_future_energy.return_value = (0.0, 0.0, {})
    coordinator.forecast.calculate_weather_deviation.return_value = {}
    coordinator.forecast.calculate_plan_revision_impact.return_value = {}

    # FIX: Return tuple of 7 elements for unpacking
    coordinator.forecast._process_forecast_item.return_value = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {}, 0.0)

    coordinator.forecast.get_forecast_for_hour.return_value = None
    # FIX: Mock get_plan_for_hour to return a 2-tuple
    coordinator.forecast.get_plan_for_hour.return_value = (0.0, {})

    # Mock states
    weather_state = MagicMock()
    weather_state.attributes = {"temperature": 15.0, "wind_speed": 5.0} # Weather has temp

    sensor_temp = MagicMock()
    sensor_temp.state = "10.0" # Sensor has different temp

    sensor_wind = MagicMock()
    sensor_wind.state = "20.0" # Sensor has wind
    sensor_wind.attributes = {"unit_of_measurement": "m/s"}

    def get_state_side_effect(entity_id):
        if entity_id == "weather.home":
            return weather_state
        if entity_id == "sensor.temp":
            return sensor_temp
        if entity_id == "sensor.wind":
            return sensor_wind
        return None
    hass.states.get.side_effect = get_state_side_effect

    # Mock time
    with patch("custom_components.heating_analytics.coordinator.dt_util.now") as mock_now:
        mock_now.return_value = datetime(2023, 1, 1, 12, 0, 0)

        # Run update
        # We need to mock _process_hourly_data if it runs
        coordinator._process_hourly_data = AsyncMock()

        await coordinator._async_update_data()

        # Verify Temp comes from Weather (15.0)
        assert coordinator._hourly_temp_sum == 15.0

        # Verify Wind comes from Sensor (20.0)
        assert coordinator._hourly_wind_sum == 20.0

@pytest.mark.asyncio
async def test_mixed_source_selection(hass, mock_entry):
    """Test mixed sources (Temp=Sensor, Wind=Weather)."""
    mock_entry.data[CONF_OUTDOOR_TEMP_SOURCE] = SOURCE_SENSOR
    mock_entry.data[CONF_WIND_SOURCE] = SOURCE_WEATHER

    coordinator = HeatingDataCoordinator(hass, mock_entry)
    coordinator.solar = MagicMock()
    coordinator.statistics = MagicMock()
    # Fix return value for calculate_total_power
    coordinator.statistics.calculate_total_power.return_value = {
        'total_kwh': 1.0,
        'breakdown': {
            'base_kwh': 1.0,
            'aux_reduction_kwh': 0.0,
            'solar_reduction_kwh': 0.0
        },
        'unit_breakdown': {}
    }

    coordinator.learning = MagicMock()
    coordinator.storage = MagicMock()
    coordinator.storage.async_load_data = AsyncMock()
    coordinator.storage.async_save_data = AsyncMock()

    coordinator.forecast = MagicMock()
    coordinator.forecast.update_daily_forecast = AsyncMock()
    coordinator.forecast.calculate_future_energy.return_value = (0.0, 0.0, {})
    coordinator.forecast.calculate_weather_deviation.return_value = {}
    coordinator.forecast.calculate_plan_revision_impact.return_value = {}
    coordinator.forecast._process_forecast_item.return_value = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {}, 0.0)
    coordinator.forecast._process_forecast_item.return_value = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {}, 0.0)
    coordinator.forecast._process_forecast_item.return_value = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {}, 0.0)
    coordinator.forecast.get_forecast_for_hour.return_value = None
    # FIX: Mock get_plan_for_hour to return a 2-tuple
    coordinator.forecast.get_plan_for_hour.return_value = (0.0, {})

    # Mock states
    weather_state = MagicMock()
    weather_state.attributes = {"temperature": 15.0, "wind_speed": 5.0} # Weather wind=5

    sensor_temp = MagicMock()
    sensor_temp.state = "10.0" # Sensor temp=10

    sensor_wind = MagicMock()
    sensor_wind.state = "20.0"
    sensor_wind.attributes = {"unit_of_measurement": "m/s"}

    def get_state_side_effect(entity_id):
        if entity_id == "weather.home": return weather_state
        if entity_id == "sensor.temp": return sensor_temp
        if entity_id == "sensor.wind": return sensor_wind
        return None
    hass.states.get.side_effect = get_state_side_effect

    with patch("custom_components.heating_analytics.coordinator.dt_util.now") as mock_now:
        mock_now.return_value = datetime(2023, 1, 1, 12, 0, 0)
        coordinator._process_hourly_data = AsyncMock()
        await coordinator._async_update_data()

        # Verify Temp comes from Sensor (10.0)
        assert coordinator._hourly_temp_sum == 10.0

        # Verify Wind comes from Weather (5.0)
        assert coordinator._hourly_wind_sum == 5.0

@pytest.mark.asyncio
async def test_strict_mode_no_fallback(hass, mock_entry):
    """Test that if Source=Sensor and Sensor is None, it does NOT fallback to Weather."""
    mock_entry.data[CONF_OUTDOOR_TEMP_SOURCE] = SOURCE_SENSOR

    coordinator = HeatingDataCoordinator(hass, mock_entry)
    coordinator.solar = MagicMock()
    coordinator.statistics = MagicMock()
    # Fix return value for calculate_total_power
    coordinator.statistics.calculate_total_power.return_value = {
        'total_kwh': 1.0,
        'breakdown': {
            'base_kwh': 1.0,
            'aux_reduction_kwh': 0.0,
            'solar_reduction_kwh': 0.0
        },
        'unit_breakdown': {}
    }

    coordinator.learning = MagicMock()
    coordinator.storage = MagicMock()
    coordinator.storage.async_load_data = AsyncMock()
    coordinator.storage.async_save_data = AsyncMock()

    coordinator.forecast = MagicMock()
    coordinator.forecast.update_daily_forecast = AsyncMock()
    coordinator.forecast.calculate_future_energy.return_value = (0.0, 0.0, {})
    coordinator.forecast.calculate_weather_deviation.return_value = {}
    coordinator.forecast.calculate_plan_revision_impact.return_value = {}
    coordinator.forecast._process_forecast_item.return_value = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {}, 0.0)
    coordinator.forecast._process_forecast_item.return_value = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {}, 0.0)
    coordinator.forecast.get_forecast_for_hour.return_value = None
    # FIX: Mock get_plan_for_hour to return a 2-tuple
    coordinator.forecast.get_plan_for_hour.return_value = (0.0, {})

    # Weather exists, but Sensor is unavailable/missing
    weather_state = MagicMock()
    weather_state.attributes = {"temperature": 15.0}

    def get_state_side_effect(entity_id):
        if entity_id == "weather.home": return weather_state
        if entity_id == "sensor.temp": return None # Unavailable
        return None
    hass.states.get.side_effect = get_state_side_effect

    with patch("custom_components.heating_analytics.coordinator.dt_util.now") as mock_now:
        mock_now.return_value = datetime(2023, 1, 1, 12, 0, 0)
        coordinator._process_hourly_data = AsyncMock()
        await coordinator._async_update_data()

        # Should be 0.0 (initialized) because update failed to get temp
        # And strict mode means no fallback.
        assert coordinator._hourly_temp_sum == 0.0
        assert coordinator._hourly_sample_count == 0 # Or 1 with 0?
        # If temp is None, it skips the "Calculate Effective Wind & Conditions" block
        # So sample count remains 0.

@pytest.mark.asyncio
async def test_optional_gust_sensor(hass, mock_entry):
    """Test that GUST can be optional even if Source is SENSOR."""
    # Config has:
    # Wind Source = Sensor
    # Gust Source = Sensor
    # Wind Speed Sensor = Set
    # Gust Sensor = None (Empty)

    mock_entry.data["wind_speed_sensor"] = "sensor.wind_speed"
    mock_entry.data[CONF_WIND_SOURCE] = SOURCE_SENSOR

    mock_entry.data["wind_gust_sensor"] = None
    mock_entry.data[CONF_WIND_GUST_SOURCE] = SOURCE_SENSOR # Selected Sensor, but didn't provide one

    # Must provide a temperature for effective_wind to be calculated
    mock_entry.data["outdoor_temp_sensor"] = "sensor.temp"
    mock_entry.data[CONF_OUTDOOR_TEMP_SOURCE] = SOURCE_SENSOR

    coordinator = HeatingDataCoordinator(hass, mock_entry)
    coordinator.solar = MagicMock()
    coordinator.statistics = MagicMock()
    # Fix return value for calculate_total_power
    coordinator.statistics.calculate_total_power.return_value = {
        'total_kwh': 1.0,
        'breakdown': {
            'base_kwh': 1.0,
            'aux_reduction_kwh': 0.0,
            'solar_reduction_kwh': 0.0
        },
        'unit_breakdown': {}
    }

    coordinator.learning = MagicMock()
    coordinator.storage = MagicMock()
    coordinator.storage.async_load_data = AsyncMock()
    coordinator.storage.async_save_data = AsyncMock()
    coordinator.forecast = MagicMock()
    coordinator.forecast.update_daily_forecast = AsyncMock()
    coordinator.forecast.calculate_future_energy.return_value = (0.0, 0.0, {})
    coordinator.forecast.calculate_weather_deviation.return_value = {}
    coordinator.forecast.calculate_plan_revision_impact.return_value = {}
    coordinator.forecast._process_forecast_item.return_value = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {}, 0.0)
    coordinator.forecast.get_forecast_for_hour.return_value = None
    # FIX: Mock get_plan_for_hour to return a 2-tuple
    coordinator.forecast.get_plan_for_hour.return_value = (0.0, {})

    # Mock states
    wind_state = MagicMock()
    wind_state.state = "10.0" # m/s
    wind_state.attributes = {"unit_of_measurement": "m/s"}

    temp_state = MagicMock()
    temp_state.state = "10.0"

    def get_state_side_effect(entity_id):
        if entity_id == "sensor.wind_speed":
            return wind_state
        if entity_id == "sensor.temp":
            return temp_state
        return None
    hass.states.get.side_effect = get_state_side_effect

    with patch("custom_components.heating_analytics.coordinator.dt_util.now") as mock_now:
        mock_now.return_value = datetime(2023, 1, 1, 12, 0, 0)
        coordinator._process_hourly_data = AsyncMock()
        await coordinator._async_update_data()

        # Should work fine, treating gust as 0.0
        # Wind = 10.0, Gust = None -> Effective = 10.0
        assert coordinator.data["effective_wind"] == 10.0
