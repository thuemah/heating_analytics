"""Test Rolling Power calculation."""
from unittest.mock import MagicMock, patch
import pytest
from datetime import datetime, timezone
from homeassistant.util import dt as dt_util
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.sensor import HeatingDeviceDailySensor

@pytest.fixture
def coordinator(hass):
    entry = MagicMock()
    entry.data = {
        "energy_sensors": ["sensor.heater_1"],
        "balance_point": 17.0,
    }
    # We use a real instance but mock out dependencies
    with patch("custom_components.heating_analytics.storage.StorageManager.async_load_data"):
        coord = HeatingDataCoordinator(hass, entry)
        coord._is_loaded = True
        return coord

@pytest.mark.asyncio
async def test_calculate_unit_rolling_power_watts(coordinator):
    """Test the rolling power calculation logic."""
    entity_id = "sensor.heater_1"

    # 1. Test at minute 0 (should use 100% of last hour)
    # We must patch where it is USED, which is inside coordinator module
    with patch("custom_components.heating_analytics.coordinator.dt_util.now", return_value=datetime(2023, 10, 27, 12, 0, tzinfo=timezone.utc)):
        coordinator._hourly_log = [
            {"unit_breakdown": {entity_id: 0.6}} # 0.6 kWh last hour
        ]
        coordinator._hourly_delta_per_unit = {entity_id: 0.05} # 0.05 so far this hour

        # Formula: 0.05 + 0.6 * (1.0 - 0/60) = 0.65 kWh
        power = coordinator.calculate_unit_rolling_power_watts(entity_id)
        assert power == 650

    # 2. Test at minute 30 (should use 50% now, 50% last)
    with patch("custom_components.heating_analytics.coordinator.dt_util.now", return_value=datetime(2023, 10, 27, 12, 30, tzinfo=timezone.utc)):
        coordinator._hourly_log = [
            {"unit_breakdown": {entity_id: 0.8}} # 0.8 kWh last hour
        ]
        coordinator._hourly_delta_per_unit = {entity_id: 0.2} # 0.2 kWh so far (30 mins)

        # Formula: 0.2 + 0.8 * (1.0 - 0.5) = 0.2 + 0.4 = 0.6 kWh
        power = coordinator.calculate_unit_rolling_power_watts(entity_id)
        assert power == 600

    # 3. Test at minute 45 (should use 75% now, 25% last)
    with patch("custom_components.heating_analytics.coordinator.dt_util.now", return_value=datetime(2023, 10, 27, 12, 45, tzinfo=timezone.utc)):
        coordinator._hourly_log = [
            {"unit_breakdown": {entity_id: 1.0}} # 1.0 kWh last hour
        ]
        coordinator._hourly_delta_per_unit = {entity_id: 0.9} # 0.9 kWh so far (45 mins)

        # Formula: 0.9 + 1.0 * (1.0 - 0.75) = 0.9 + 0.25 = 1.15 kWh
        power = coordinator.calculate_unit_rolling_power_watts(entity_id)
        assert power == 1150

@pytest.mark.asyncio
async def test_sensor_attribute_uses_rolling_power(coordinator):
    """Test that the sensor attribute calls the coordinator method."""
    entity_id = "sensor.heater_1"
    mock_entry = MagicMock()
    mock_entry.entry_id = "test_entry"

    sensor = HeatingDeviceDailySensor(coordinator, mock_entry, entity_id)
    sensor.hass = MagicMock()

    # Mock the coordinator methods
    with patch.object(coordinator, 'calculate_unit_rolling_power_watts', return_value=456) as mock_rolling, \
         patch.object(coordinator, '_calculate_inertia_temp', return_value=10.0), \
         patch.object(coordinator, '_get_wind_bucket', return_value="normal"), \
         patch.object(coordinator, '_get_predicted_kwh_per_unit', return_value=0.3):

        attrs = sensor.extra_state_attributes
        assert attrs["average_power_current"] == 456
        mock_rolling.assert_called_once_with(entity_id)
