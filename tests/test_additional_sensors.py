"""Test additional Sensor entities."""
from unittest.mock import MagicMock, patch
import pytest
from homeassistant.core import HomeAssistant
from homeassistant.const import UnitOfEnergy, PERCENTAGE, UnitOfTemperature
from custom_components.heating_analytics.const import (
    DOMAIN,
    ATTR_ENERGY_TODAY,
    ATTR_EXPECTED_TODAY,
    ATTR_PREDICTED,
    ATTR_MIDNIGHT_FORECAST,
    ATTR_CORRELATION_DATA,
    ATTR_LAST_HOUR_EXPECTED,
    ATTR_LAST_HOUR_ACTUAL,
    ATTR_LAST_HOUR_DEVIATION,
    ATTR_LAST_HOUR_DEVIATION_PCT,
    ATTR_TEMP_ACTUAL_TODAY,
    ATTR_WIND_ACTUAL_TODAY,
    ATTR_SOLAR_PREDICTED,
    ATTR_WEEKLY_SUMMARY,
    SENSOR_MODEL_COMPARISON_DAY,
    SENSOR_MODEL_COMPARISON_WEEK,
    SENSOR_MODEL_COMPARISON_MONTH,
    SENSOR_WEEK_AHEAD_FORECAST
)
from custom_components.heating_analytics.sensor import (
    HeatingEnergyTodaySensor,
    HeatingExpectedEnergyTodaySensor,
    HeatingPredictedSensor,
    HeatingEffectiveWindSensor,
    HeatingCorrelationDataSensor,
    HeatingLastHourExpectedSensor,
    HeatingLastHourDeviationSensor,
    HeatingModelComparisonDaySensor,
    HeatingModelComparisonWeekSensor,
    HeatingModelComparisonMonthSensor,
    HeatingWeekAheadForecastSensor
)
from datetime import date, datetime
from homeassistant.util import dt as dt_util

@pytest.fixture
def mock_coordinator():
    coordinator = MagicMock()
    coordinator.data = {}
    # Initialize internal stats
    coordinator._hourly_log = []
    coordinator._hourly_bucket_counts = {"with_auxiliary_heating": 0}
    coordinator._hourly_sample_count = 0
    coordinator._hourly_wind_sum = 0.0
    coordinator.wind_unit = "m/s"
    coordinator.wind_threshold = 5.5
    coordinator.extreme_wind_threshold = 10.8
    coordinator.wind_gust_factor = 0.6

    # Mock helpers
    coordinator._get_wind_bucket.return_value = "normal"
    coordinator.forecast.get_future_day_prediction.return_value = (10.0, 0.0, {"temp": 5.0, "wind": 3.0})
    coordinator.calculate_modeled_energy.return_value = (100.0, 0.0, 5.0, 3.0, 10.0)
    coordinator.statistics.calculate_historical_actual_sum.return_value = 90.0
    coordinator.statistics.calculate_hybrid_projection.return_value = (120.0, 5.0)
    coordinator.forecast.calculate_future_energy.return_value = (10.0, 2.0, None)

    return coordinator

@pytest.fixture
def mock_entry():
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.title = "Test Heating"
    return entry

@pytest.mark.asyncio
async def test_energy_today_sensor(hass: HomeAssistant, mock_coordinator, mock_entry):
    """Test HeatingEnergyTodaySensor."""
    mock_coordinator.data = {
        ATTR_ENERGY_TODAY: 25.5,
        "daily_individual": {"sensor.heater_1": 15.5, "sensor.heater_2": 10.0}
    }
    mock_coordinator.energy_sensors = ["sensor.heater_1", "sensor.heater_2"]

    # Mock state retrieval
    mock_state1 = MagicMock()
    mock_state1.name = "Heater 1"
    mock_state2 = MagicMock()
    mock_state2.name = "Heater 2"
    hass.states.get = lambda x: mock_state1 if x == "sensor.heater_1" else mock_state2
    mock_coordinator.hass = hass

    sensor = HeatingEnergyTodaySensor(mock_coordinator, mock_entry)
    sensor.hass = hass

    assert sensor.native_value == 25.5

    attrs = sensor.extra_state_attributes
    assert attrs["unit_breakdown_kwh"] == {"Heater 1": 15.5, "Heater 2": 10.0}
    assert attrs["unit_contribution_pct"] == {"Heater 1": 60.8, "Heater 2": 39.2}
    assert attrs["active_units_count"] == 2

@pytest.mark.asyncio
async def test_expected_today_sensor(hass: HomeAssistant, mock_coordinator, mock_entry):
    """Test HeatingExpectedEnergyTodaySensor."""
    mock_coordinator.data = {ATTR_EXPECTED_TODAY: 24.0}
    sensor = HeatingExpectedEnergyTodaySensor(mock_coordinator, mock_entry)
    assert sensor.native_value == 24.0

@pytest.mark.asyncio
async def test_predicted_sensor(hass: HomeAssistant, mock_coordinator, mock_entry):
    """Test HeatingPredictedSensor."""
    mock_coordinator.data = {
        ATTR_PREDICTED: 30.0,
        ATTR_MIDNIGHT_FORECAST: 28.0
    }
    mock_coordinator.forecast._midnight_forecast_snapshot = {"timestamp": "2023-01-01T00:00:00"}

    sensor = HeatingPredictedSensor(mock_coordinator, mock_entry)
    assert sensor.native_value == 30.0
    assert sensor.extra_state_attributes["cached_total_24h"] == 28.0

@pytest.mark.asyncio
async def test_effective_wind_sensor(hass: HomeAssistant, mock_coordinator, mock_entry):
    """Test HeatingEffectiveWindSensor."""
    mock_coordinator.data = {"effective_wind": 5.0}
    mock_coordinator._hourly_sample_count = 10
    mock_coordinator._hourly_wind_sum = 48.0 # avg 4.8

    sensor = HeatingEffectiveWindSensor(mock_coordinator, mock_entry)

    assert sensor.native_value == 5.0
    attrs = sensor.extra_state_attributes
    assert attrs["running_average_this_hour"] == 4.8
    assert attrs["data_quality"] == "partial" # < 30 samples

@pytest.mark.asyncio
async def test_correlation_data_sensor(hass: HomeAssistant, mock_coordinator, mock_entry):
    """Test HeatingCorrelationDataSensor."""
    mock_coordinator.data = {
        ATTR_CORRELATION_DATA: {
            "0": {"normal": 1.0, "high_wind": 1.2},
            "-5": {"normal": 2.0}
        }
    }

    sensor = HeatingCorrelationDataSensor(mock_coordinator, mock_entry)

    assert sensor.native_value == "Data"
    attrs = sensor.extra_state_attributes

    # Check if x/y lists are generated and are JSON arrays of integers
    assert "normal_x" in attrs
    # JSON array [ -5, 0 ]
    assert '[-5, 0]' in attrs["normal_x"] or '[-5,0]' in attrs["normal_x"]

@pytest.mark.asyncio
async def test_last_hour_sensors(hass: HomeAssistant, mock_coordinator, mock_entry):
    """Test Last Hour Expected and Deviation Sensors."""
    mock_coordinator._hourly_log = [{
        "expected_kwh": 5.0,
        "solar_impact_kwh": 1.0,
        "inertia_temp": 2.5,
        "wind_bucket": "normal",
        "solar_factor": 0.2,
        "model_base_before": 4.5,
        "model_base_after": 4.6,
        "model_temp_key": "2",
        "model_updated": True
    }]

    mock_coordinator.data = {
        ATTR_LAST_HOUR_EXPECTED: 5.0,
        ATTR_LAST_HOUR_DEVIATION: -0.5,
        ATTR_LAST_HOUR_DEVIATION_PCT: -10.0,
        ATTR_LAST_HOUR_ACTUAL: 4.5
    }

    # Expected Sensor
    sensor_exp = HeatingLastHourExpectedSensor(mock_coordinator, mock_entry)
    assert sensor_exp.native_value == 5.0
    attrs_exp = sensor_exp.extra_state_attributes
    assert attrs_exp["base_model_kwh"] == 6.0 # 5.0 + 1.0

    # Deviation Sensor
    sensor_dev = HeatingLastHourDeviationSensor(mock_coordinator, mock_entry)
    assert sensor_dev.native_value == -0.5
    attrs_dev = sensor_dev.extra_state_attributes
    assert attrs_dev["percentage"] == -10.0
    assert attrs_dev["model_delta"] == "+0.10000"

@pytest.mark.asyncio
async def test_model_comparison_sensors(hass: HomeAssistant, mock_coordinator, mock_entry):
    """Test Model Comparison Sensors."""

    # Mock dt_util.now to avoid MagicMock comparison errors in _calculate_period_stats
    with patch("custom_components.heating_analytics.sensor.dt_util.now") as mock_now:
        mock_now.return_value = datetime(2025, 1, 15, 12, 0, 0)

        # Day Sensor
        sensor_day = HeatingModelComparisonDaySensor(mock_coordinator, mock_entry)

        mock_coordinator.data[ATTR_PREDICTED] = 10.0
        mock_coordinator.data[ATTR_SOLAR_PREDICTED] = 2.0
        mock_coordinator.data[ATTR_TEMP_ACTUAL_TODAY] = 5.0
        mock_coordinator.data[ATTR_WIND_ACTUAL_TODAY] = 3.0

        # When _get_or_calculate_stats runs, it will call _calculate_period_stats.
        # It checks if today > start_date. With 2025-01-15, today is start_date.
        # It calls coordinator.calculate_modeled_energy for past days.
        # Our fixture mocked it to return (100.0, ...).
        # It calls statistics.calculate_hybrid_projection -> mocked to (120.0, 5.0).

        # IMPORTANT: Fix the mocked return of calculate_modeled_energy to prevent unpacking errors if logic changed.
        # The sensor expects 4 values: model_past, solar_past, temp_past, wind_past

        val = sensor_day.native_value
        assert val == 20.0 # 120 (curr) - 100 (last - from cache or calc)

        attrs = sensor_day.extra_state_attributes
        assert attrs["current_model_kwh"] == 120.0

        # Week Sensor
        # Last Year = 100 (So Far) + 100 (Remaining) = 200.
        # Current = 120.
        # Diff = -80.0
        sensor_week = HeatingModelComparisonWeekSensor(mock_coordinator, mock_entry)
        assert sensor_week.native_value == -80.0

        # Month Sensor
        # Last Year = 100 (So Far) + 100 (Remaining) = 200.
        # Diff = -80.0
        sensor_month = HeatingModelComparisonMonthSensor(mock_coordinator, mock_entry)
        assert sensor_month.native_value == -80.0

@pytest.mark.asyncio
async def test_week_ahead_forecast_sensor(hass: HomeAssistant, mock_coordinator, mock_entry):
    """Test HeatingWeekAheadForecastSensor."""
    mock_coordinator.forecast.calculate_week_ahead_stats.return_value = {
        "total_kwh": 200.0,
        ATTR_WEEKLY_SUMMARY: "Looks cold"
    }

    with patch("custom_components.heating_analytics.sensor.dt_util.now") as mock_now:
        mock_now.return_value = datetime(2025, 1, 15, 12, 0, 0)
        sensor = HeatingWeekAheadForecastSensor(mock_coordinator, mock_entry)

        assert sensor.native_value == 200.0
        attrs = sensor.extra_state_attributes
        assert attrs[ATTR_WEEKLY_SUMMARY] == "Looks cold"
