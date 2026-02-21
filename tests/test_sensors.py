"""Test Sensor entities."""
from unittest.mock import MagicMock, patch
import pytest
from datetime import datetime, timedelta, timezone
from homeassistant.core import HomeAssistant
from homeassistant.components.sensor import SensorDeviceClass, SensorStateClass
from custom_components.heating_analytics.const import (
    DOMAIN, ATTR_FORECAST_TODAY, ATTR_MIDNIGHT_FORECAST, ATTR_FORECAST_UNCERTAINTY,
    ATTR_DEVIATION, ATTR_PREDICTED, ATTR_ENERGY_TODAY, ATTR_EXPECTED_TODAY,
    ATTR_DEVIATION_BREAKDOWN, ATTR_POTENTIAL_SAVINGS,
    ATTR_TEMP_FORECAST_TODAY, ATTR_AVG_WIND_FORECAST,
    ATTR_LAST_HOUR_DEVIATION, ATTR_LAST_HOUR_DEVIATION_PCT,
    ATTR_LAST_HOUR_EXPECTED, ATTR_LAST_HOUR_ACTUAL
)
from custom_components.heating_analytics.sensor import (
    HeatingForecastTodaySensor,
    HeatingDeviationSensor,
    HeatingPotentialSavingsSensor,
    HeatingDeviceDailySensor,
    HeatingDeviceLifetimeSensor,
    HeatingLastHourActualSensor,
    HeatingLastHourDeviationSensor
)

@pytest.fixture
def mock_coordinator():
    coordinator = MagicMock()
    coordinator.data = {}
    # Initialize internal stats to prevent comparison errors with MagicMock
    coordinator._hourly_log = []
    coordinator._hourly_bucket_counts = {"with_auxiliary_heating": 0}
    coordinator._hourly_sample_count = 0
    coordinator._hourly_wind_sum = 0.0
    # Initialize orphaned accumulators to avoid MagicMock recursion
    coordinator._daily_orphaned_aux = 0.0
    coordinator._accumulated_orphaned_aux = 0.0
    coordinator.wind_unit = "m/s"
    coordinator.wind_threshold = 5.5
    coordinator.extreme_wind_threshold = 10.8
    coordinator.wind_gust_factor = 0.6
    return coordinator

@pytest.fixture
def mock_entry():
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.title = "Test Heating"
    return entry

@pytest.mark.asyncio
async def test_forecast_today_sensor(hass: HomeAssistant, mock_coordinator, mock_entry):
    """Test HeatingForecastTodaySensor."""
    # Setup data
    mock_coordinator.data = {
        ATTR_FORECAST_TODAY: 50.0,
        "confidence_interval_margin": 5.0,
        "confidence_interval_lower": 45.0,
        "confidence_interval_upper": 55.0,
        ATTR_MIDNIGHT_FORECAST: 48.0,
        ATTR_FORECAST_UNCERTAINTY: {"samples": 10, "p50_abs_error": 0.1},
        ATTR_TEMP_FORECAST_TODAY: -5.0,
        ATTR_AVG_WIND_FORECAST: 7.0
    }

    sensor = HeatingForecastTodaySensor(mock_coordinator, mock_entry)
    sensor.hass = hass
    sensor.async_write_ha_state = MagicMock()

    assert sensor.native_value == 50.0

    attrs = sensor.extra_state_attributes
    assert attrs["confidence_interval_margin"] == 5.0
    assert attrs["confidence_interval_lower"] == 45.0
    assert attrs["confidence_interval_upper"] == 55.0
    assert attrs[ATTR_MIDNIGHT_FORECAST] == 48.0
    assert attrs[ATTR_FORECAST_UNCERTAINTY] == {"samples": 10, "p50_abs_error": 0.1}
    assert sensor.unique_id == "test_entry_forecast_today"

    # Verify weather context is included in forecast_summary
    forecast_summary = attrs["forecast_summary"]
    assert "50.0 kWh" in forecast_summary
    assert "very cold (-5.0°C)" in forecast_summary
    assert "strong wind" in forecast_summary


@pytest.mark.asyncio
async def test_deviation_sensor(hass: HomeAssistant, mock_coordinator, mock_entry):
    """Test HeatingDeviationSensor."""
    mock_coordinator.data = {
        ATTR_DEVIATION: 10.0,
        ATTR_FORECAST_TODAY: 110.0,
        ATTR_PREDICTED: 100.0,
        ATTR_ENERGY_TODAY: 55.0,
        ATTR_EXPECTED_TODAY: 50.0,
        ATTR_DEVIATION_BREAKDOWN: [
            {"name": "Heater 1", "deviation": 5.0, "confidence": "high"},
            {"name": "Heater 2", "deviation": -2.0, "confidence": "medium"},
            {"name": "Heater 3", "deviation": 1.0, "confidence": "low"},
            {"name": "Heater 4", "deviation": 0.5, "confidence": "low"},
        ],
        "weather_forecast_deviation": {"estimated_impact_kwh": 2.0},
        "weather_adjusted_deviation": {"deviation_kwh": 8.0}
    }

    sensor = HeatingDeviationSensor(mock_coordinator, mock_entry)
    sensor.hass = hass
    sensor.async_write_ha_state = MagicMock()

    assert sensor.native_value == 10.0

    attrs = sensor.extra_state_attributes

    # Check breakdown attributes (structured as above/below expected)
    contributors = attrs["contributors"]
    assert "above_expected" in contributors
    assert "below_expected" in contributors

    # Above expected: Heater 1, 3, 4 (sorted by abs deviation)
    above = contributors["above_expected"]
    assert len(above) == 3
    assert above[0]["name"] == "Heater 1"
    assert above[0]["deviation_kwh"] == 5.0
    assert above[0]["confidence"] == "high"

    assert above[1]["name"] == "Heater 3"
    assert above[1]["deviation_kwh"] == 1.0

    assert above[2]["name"] == "Heater 4"
    assert above[2]["deviation_kwh"] == 0.5

    # Below expected: Heater 2
    below = contributors["below_expected"]
    assert len(below) == 1
    assert below[0]["name"] == "Heater 2"
    assert below[0]["deviation_kwh"] == -2.0
    assert below[0]["confidence"] == "medium"

    # Check calculated attributes
    assert attrs["end_of_day_forecast_kwh"] == 110.0
    assert attrs["model_prediction_kwh"] == 100.0
    assert attrs["deviation_projected_kwh"] == 10.0 # 110 - 100

    # Check new attribute
    # In this mock, coordinator.data doesn't have "thermodynamic_gross_today_kwh", so it should default to 0.0 or be None if using .get()
    assert attrs["thermodynamic_gross_today_kwh"] == 0.0

    # Check aux active (mocked coordinator doesn't set it explicitly in data dict, but attr access works)
    # coordinator.auxiliary_heating_active is likely MagicMock unless set

    # Check historical
    # Deviation so far = ((55 - 50) / 50) * 100 = 10.0%
    assert attrs["deviation_current_pct"] == 10.0

    # Weather context structure changed slightly
    # attribute "weather" removed/refactored into summary


@pytest.mark.asyncio
async def test_potential_savings_sensor(hass: HomeAssistant, mock_coordinator, mock_entry):
    """Test HeatingPotentialSavingsSensor."""
    mock_coordinator.data = {
        ATTR_POTENTIAL_SAVINGS: 10.0, # Theoretical Potential
        "savings_theory_normal": 20.0,
        "savings_theory_aux": 10.0,
        "savings_actual_kwh": 8.0, # Realized Savings
        "missing_aux_data": True,
        "current_savings_rate": 0.6 # kW
    }
    mock_coordinator.auxiliary_heating_active = True

    sensor = HeatingPotentialSavingsSensor(mock_coordinator, mock_entry)
    sensor.hass = hass
    sensor.async_write_ha_state = MagicMock()

    # Native value is now Actual Savings
    assert sensor.native_value == 8.0
    assert sensor.name == "AUX Savings Today"

    attrs = sensor.extra_state_attributes

    # Check new Core Attributes
    assert attrs["theoretical_max_savings"] == 10.0 # 20 - 10
    assert "missed_savings_kwh" not in attrs
    assert attrs["current_savings_rate_kw"] == "0.6 kW"
    assert attrs["status"] == "Active"

    # Check Legacy/Debug attributes
    assert "consumption_normal" not in attrs
    assert "consumption_with_auxiliary_heating" not in attrs
    assert "savings_percentage" not in attrs
    assert "explanation" not in attrs
    assert attrs["auxiliary_heating_active"] is True

    # Check detailed aux learning attributes (moved here)
    # Since hourly_log is mocked as empty list in fixture, these should not be present
    assert "last_hour_learning_status" not in attrs

    # Let's mock a log entry to test
    mock_coordinator._hourly_log = [{
        "learning_status": "active",
        "aux_model_updated": True,
        "aux_model_before": 0.5,
        "aux_model_after": 0.6
    }]

    # Re-instantiate sensor to get fresh attrs
    sensor_with_log = HeatingPotentialSavingsSensor(mock_coordinator, mock_entry)
    attrs_log = sensor_with_log.extra_state_attributes

    assert attrs_log["last_hour_learning_status"] == "active"
    assert attrs_log["aux_model_value_before"] == "0.50000"
    assert attrs_log["aux_model_value_after"] == "0.60000"
    assert attrs_log["aux_model_delta"] == "+0.10000"

    # Test "Unknown" rate
    mock_coordinator.data["current_savings_rate"] = None
    attrs_unknown = sensor.extra_state_attributes
    assert attrs_unknown["current_savings_rate_kw"] == "Unknown"

    # Verify Device Class and State Class
    assert sensor._attr_device_class == SensorDeviceClass.ENERGY
    assert sensor._attr_state_class == SensorStateClass.TOTAL


@pytest.mark.asyncio
async def test_device_daily_sensor(mock_coordinator, mock_entry):
    """Test HeatingDeviceDailySensor."""
    hass = MagicMock()
    entity_id = "sensor.heater_1"

    # Mock hass states
    mock_state = MagicMock()
    mock_state.name = "Heater One"
    hass.states.get.return_value = mock_state
    mock_coordinator.hass = hass

    # Setup data
    mock_coordinator.data = {
        "daily_individual": {entity_id: 12.345},
        "effective_wind": 5.0,
        "forecast_today_per_unit": {entity_id: 12.0}
    }

    # Mock methods used in attributes
    mock_coordinator._calculate_inertia_temp.return_value = 10.2
    mock_coordinator._get_wind_bucket.return_value = "normal"
    mock_coordinator._get_predicted_kwh_per_unit.return_value = 0.5

    # Mock correlation data
    mock_coordinator._correlation_data_per_unit = {
        entity_id: {
            "10": {"normal": 0.5},
            "-5": {"normal": 0.8}
        }
    }

    mock_coordinator._hourly_log = [{"timestamp": "2023-10-27T12:00:00"}]

    sensor = HeatingDeviceDailySensor(mock_coordinator, mock_entry, entity_id)
    sensor.hass = hass
    sensor.async_write_ha_state = MagicMock()

    # Check value
    assert sensor.native_value == 12.345
    assert sensor.name == "Heater One Daily"

    # Check attributes
    attrs = sensor.extra_state_attributes

    # 1. Current Prediction
    assert "temp_current" not in attrs
    assert attrs["wind_bucket_current"] == "normal"
    assert attrs["predicted_hourly_current"] == 0.5
    assert attrs["theoretical_daily_consumption"] == 12.0 # 0.5 * 24

    # 2. Correlation Data
    # Hourly attributes removed
    assert "correlation_10_normal" not in attrs
    assert attrs["correlation_10_normal_daily"] == 12.0

    # Check negative temp key formatting
    assert "correlation_minus_5_normal" not in attrs
    assert attrs["correlation_minus_5_normal_daily"] == 19.2 # 0.8 * 24


@pytest.mark.asyncio
async def test_device_lifetime_sensor(mock_coordinator, mock_entry):
    """Test HeatingDeviceLifetimeSensor."""
    hass = MagicMock()
    entity_id = "sensor.heater_1"

    mock_state = MagicMock()
    mock_state.name = "Heater One"
    hass.states.get.return_value = mock_state
    mock_coordinator.hass = hass

    mock_coordinator.data = {
        "lifetime_individual": {entity_id: 1234.567}
    }

    sensor = HeatingDeviceLifetimeSensor(mock_coordinator, mock_entry, entity_id)
    sensor.hass = hass

    assert sensor.native_value == 1234.6


@pytest.mark.asyncio
async def test_last_hour_actual_sensor_summary(mock_coordinator, mock_entry):
    """Test HeatingLastHourActualSensor with last_hour_summary."""
    hass = MagicMock()

    # Mock entity states for top consumers
    mock_state_1 = MagicMock()
    mock_state_1.name = "Varmekabel"
    mock_state_2 = MagicMock()
    mock_state_2.name = "Varmeovn"

    def get_state(entity_id):
        if entity_id == "sensor.heater_1":
            return mock_state_1
        elif entity_id == "sensor.heater_2":
            return mock_state_2
        return None

    hass.states.get = get_state
    mock_coordinator.hass = hass

    # Setup hourly log with unit_breakdown
    mock_coordinator._hourly_log = [
        {
            "timestamp": "2025-12-30T10:00:00",
            "temp": -5.0,
            "effective_wind": 3.5,
            "learning_status": "learning",
            "actual_kwh": 7.5,
            "unit_breakdown": {
                "sensor.heater_1": 1.2,  # 16% of 7.5
                "sensor.heater_2": 0.8,
                "sensor.heater_3": 0.5,
            }
        }
    ]

    sensor = HeatingLastHourActualSensor(mock_coordinator, mock_entry)
    sensor.hass = hass

    assert sensor.native_value == 0.0  # coordinator.data doesn't have it, uses default

    attrs = sensor.extra_state_attributes
    assert attrs["timestamp"] == "2025-12-30T10:00:00"
    assert attrs["avg_temperature"] == -5.0
    assert attrs["avg_effective_wind"] == 3.5
    assert attrs["learning_status"] == "learning"

    # Check top consumers
    assert "last_hour_top_consumers" in attrs
    assert len(attrs["last_hour_top_consumers"]) == 3
    assert attrs["last_hour_top_consumers"][0]["name"] == "Varmekabel"
    assert attrs["last_hour_top_consumers"][0]["kwh"] == 1.2

    # Check summary
    assert "last_hour_summary" in attrs
    summary = attrs["last_hour_summary"]
    assert "7.5 kWh consumed" in summary
    assert "led by Varmekabel" in summary
    assert "16%" in summary


@pytest.mark.asyncio
async def test_last_hour_actual_sensor_no_consumers(mock_coordinator, mock_entry):
    """Test HeatingLastHourActualSensor summary when no consumers."""
    hass = MagicMock()
    mock_coordinator.hass = hass

    # Setup hourly log without unit_breakdown
    mock_coordinator._hourly_log = [
        {
            "timestamp": "2025-12-30T10:00:00",
            "temp": -5.0,
            "effective_wind": 3.5,
            "learning_status": "learning",
            "actual_kwh": 3.2,
            "unit_breakdown": {}
        }
    ]

    sensor = HeatingLastHourActualSensor(mock_coordinator, mock_entry)
    sensor.hass = hass

    attrs = sensor.extra_state_attributes

    # Check summary without consumers
    assert "last_hour_summary" in attrs
    assert attrs["last_hour_summary"] == "3.2 kWh consumed"
    assert "led by" not in attrs["last_hour_summary"]


@pytest.mark.asyncio
async def test_last_hour_actual_sensor_no_consumption(mock_coordinator, mock_entry):
    """Test HeatingLastHourActualSensor summary when no consumption."""
    hass = MagicMock()
    mock_coordinator.hass = hass

    # Setup hourly log with zero consumption
    mock_coordinator._hourly_log = [
        {
            "timestamp": "2025-12-30T10:00:00",
            "temp": 15.0,
            "effective_wind": 1.0,
            "learning_status": "learning",
            "actual_kwh": 0.0,
            "unit_breakdown": {}
        }
    ]

    sensor = HeatingLastHourActualSensor(mock_coordinator, mock_entry)
    sensor.hass = hass

    attrs = sensor.extra_state_attributes

    # Check summary for no consumption
    assert "last_hour_summary" in attrs
    assert attrs["last_hour_summary"] == "No consumption recorded"


@pytest.mark.asyncio
async def test_last_hour_deviation_sensor(hass: HomeAssistant, mock_coordinator, mock_entry):
    """Test HeatingLastHourDeviationSensor attributes."""
    # Setup hourly log
    mock_coordinator._hourly_log = [
        {
            "timestamp": "2023-10-27T10:00:00",
            "model_base_before": 0.55555,
            "model_base_after": 0.66666,
            "model_temp_key": "10",
            "model_updated": True
        }
    ]
    mock_coordinator.data = {
        ATTR_LAST_HOUR_DEVIATION: 1.5,
        ATTR_LAST_HOUR_DEVIATION_PCT: 10.0,
        ATTR_LAST_HOUR_EXPECTED: 15.0,
        ATTR_LAST_HOUR_ACTUAL: 16.5,
    }

    sensor = HeatingLastHourDeviationSensor(mock_coordinator, mock_entry)
    sensor.hass = hass
    sensor.async_write_ha_state = MagicMock()

    assert sensor.native_value == 1.5

    attrs = sensor.extra_state_attributes

    # Check restored attributes
    assert attrs["model_value_before"] == "0.55555"
    assert attrs["model_value_after"] == "0.66666"
    assert attrs["model_updated"] is True
    assert attrs["model_updated_temp_category"] == "10"

@pytest.mark.asyncio
async def test_device_daily_throttling(mock_coordinator, mock_entry):
    """Test that redundant attributes are removed and power is throttled."""
    entity_id = "sensor.heater_1"

    # Setup mocks needed for calculation
    mock_coordinator._calculate_inertia_temp.return_value = 10.0
    mock_coordinator._get_wind_bucket.return_value = "normal"
    mock_coordinator._get_predicted_kwh_per_unit.return_value = 0.5
    mock_coordinator.calculate_unit_rolling_power_watts.return_value = 1000.0

    # 1. Initialize sensor
    sensor = HeatingDeviceDailySensor(mock_coordinator, mock_entry, entity_id)

    # Define initial time
    initial_time = datetime(2023, 10, 27, 10, 0, 0, tzinfo=timezone.utc)

    # First call: Establish baseline
    with patch("homeassistant.util.dt.now", return_value=initial_time):
        attrs = sensor.extra_state_attributes

    # VERIFY REMOVAL
    assert "temp_current" not in attrs, "temp_current should be removed"
    assert "effective_wind" not in attrs, "effective_wind should be removed"

    # VERIFY INITIAL POWER
    assert attrs["average_power_current"] == 1000.0

    # --- TEST THROTTLING ---

    # Case 2: Small change (< 5%) within same hour -> THROTTLED
    mock_coordinator.calculate_unit_rolling_power_watts.return_value = 1040.0 # +4%
    next_time = initial_time + timedelta(minutes=10)

    with patch("homeassistant.util.dt.now", return_value=next_time):
        attrs_2 = sensor.extra_state_attributes

    # Should still report 1000.0 (old value)
    assert attrs_2["average_power_current"] == 1000.0, "Power should be throttled for small changes"

    # Case 3: Significant change (> 5%) -> UPDATED
    mock_coordinator.calculate_unit_rolling_power_watts.return_value = 1060.0 # +6% from baseline 1000

    with patch("homeassistant.util.dt.now", return_value=next_time):
        attrs_3 = sensor.extra_state_attributes

    # Should update to 1060.0
    assert attrs_3["average_power_current"] == 1060.0, "Power should update on >5% change"

    # Case 4: Hourly Update -> UPDATED regardless of change size
    # Reset baseline to 1060.0 (which is now the last reported)
    # We change it slightly to 1062.0 (< 0.2% change)
    mock_coordinator.calculate_unit_rolling_power_watts.return_value = 1062.0

    # Time advances to next hour
    hour_later = initial_time + timedelta(hours=1, minutes=5) # 11:05

    with patch("homeassistant.util.dt.now", return_value=hour_later):
        attrs_4 = sensor.extra_state_attributes

    # Should update because hour changed
    assert attrs_4["average_power_current"] == 1062.0, "Power should update on hour change"
