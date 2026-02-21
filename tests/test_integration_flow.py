"""Test integration flow."""
from unittest.mock import MagicMock, patch, AsyncMock
import pytest
from custom_components.heating_analytics import async_setup_entry
from custom_components.heating_analytics.const import DOMAIN
from homeassistant.exceptions import ConfigEntryNotReady
# Platform is usually in homeassistant.const, not custom_components const
from homeassistant.const import Platform

@pytest.fixture
def mock_entry():
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.data = {
        "outdoor_temp_sensor": "sensor.temp",
        "energy_sensors": ["sensor.heater"]
    }
    return entry

@pytest.mark.asyncio
async def test_async_setup_entry_success(hass, mock_entry):
    """Test successful setup."""
    # Ensure hass.config_entries.async_forward_entry_setups is awaitable
    hass.config_entries.async_forward_entry_setups = AsyncMock()

    # Configure hass.data as a real dict to support 'in' operator
    hass.data = {}

    with patch("custom_components.heating_analytics.HeatingDataCoordinator") as mock_coord_cls:
        mock_coord = mock_coord_cls.return_value
        mock_coord.async_config_entry_first_refresh = AsyncMock()
        mock_coord.storage = MagicMock()
        mock_coord.storage.async_load_data = AsyncMock()

        assert await async_setup_entry(hass, mock_entry)

        assert DOMAIN in hass.data
        assert hass.data[DOMAIN][mock_entry.entry_id] == mock_coord

        mock_coord.async_config_entry_first_refresh.assert_awaited_once()
        hass.config_entries.async_forward_entry_setups.assert_awaited_once()

@pytest.mark.asyncio
async def test_setup_failure_handling(hass, mock_entry):
    """Test setup failure handling."""
    # Define a real exception for ConfigEntryNotReady since pytest.raises requires BaseException subclass
    class RealConfigEntryNotReady(Exception):
        pass

    hass.data = {} # Ensure dict behavior

    with patch("custom_components.heating_analytics.HeatingDataCoordinator") as mock_coord_cls:
        # Patch the exception class where it is IMPORTED in __init__.py
        with patch("custom_components.heating_analytics.ConfigEntryNotReady", RealConfigEntryNotReady):
            mock_coord = mock_coord_cls.return_value
            # Define async_config_entry_first_refresh on mock
            mock_coord.async_config_entry_first_refresh = AsyncMock(side_effect=Exception("API Error"))

            # Should raise ConfigEntryNotReady
            with pytest.raises(RealConfigEntryNotReady):
                await async_setup_entry(hass, mock_entry)

def test_coordinator_data_flow_to_sensor(hass, mock_entry):
    """Test that coordinator data flows to sensor entity."""
    # We define a MockCoordinator that behaves enough like real one
    class MockCoordinator:
        def __init__(self, hass, entry):
            self.hass = hass
            self.data = {"test_key": "test_val"}
            self.energy_sensors = []
            self.enable_lifetime_tracking = False
            self.statistics = MagicMock()
            self.forecast = MagicMock()
            # FIX: 3 args
            self._get_predicted_kwh = lambda t, w, e=None: 1.0
            self._get_wind_bucket = lambda w: "normal"
            self._aux_coefficients = {}

    coord = MockCoordinator(hass, mock_entry)

    # Instantiate a sensor
    from custom_components.heating_analytics.sensor import HeatingAnalyticsBaseSensor

    class TestSensor(HeatingAnalyticsBaseSensor):
        def __init__(self, coordinator, entry):
            super().__init__(coordinator, entry)
            self._attr_unique_id = "test_id"

        @property
        def native_value(self):
            return self.coordinator.data.get("test_key")

    sensor = TestSensor(coord, mock_entry)
    assert sensor.native_value == "test_val"
