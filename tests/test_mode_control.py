"""Tests for Mode Control (Phase 3)."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util

from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.const import (
    DOMAIN,
    MODE_HEATING,
    MODE_COOLING,
    MODE_OFF,
    STORAGE_KEY,
    STORAGE_VERSION,
)

# Mock Coordinator fixture
@pytest.fixture
def mock_coordinator(hass):
    """Create a mock coordinator."""
    entry = MagicMock()
    entry.data = {
        "outdoor_temp_sensor": "sensor.outdoor_temp",
        "energy_sensors": ["sensor.heater_1", "sensor.heater_2"],
        "solar_enabled": True
    }

    # Patch dependencies
    with patch("custom_components.heating_analytics.coordinator.StorageManager") as mock_storage_cls, \
         patch("custom_components.heating_analytics.coordinator.ForecastManager"), \
         patch("custom_components.heating_analytics.coordinator.StatisticsManager"), \
         patch("custom_components.heating_analytics.coordinator.LearningManager"), \
         patch("custom_components.heating_analytics.coordinator.SolarCalculator") as mock_solar_cls:

        # Setup Coordinator
        coord = HeatingDataCoordinator(hass, entry)

        # Setup Solar Mock
        mock_solar_instance = mock_solar_cls.return_value
        coord.solar = mock_solar_instance

        # Setup Storage Mock (Real-ish behavior needed for persistence tests)
        # But for unit tests, we usually mock the underlying store or the manager methods.
        # Here we want to test that Coordinator calls Storage correctly.
        mock_storage_instance = mock_storage_cls.return_value
        # Configure AsyncMock for async methods
        mock_storage_instance.async_save_data = AsyncMock()
        mock_storage_instance.async_load_data = AsyncMock()
        coord.storage = mock_storage_instance

        # Initialize internal structures
        coord._unit_modes = {}

        # Mock async_set_updated_data (since it's a DataUpdateCoordinator method we didn't mock out, wait...
        # We are using the REAL HeatingDataCoordinator, so it inherits from DataUpdateCoordinator.
        # But we are mocking many internals. DataUpdateCoordinator needs hass.
        # It seems we are hitting an issue where async_set_updated_data is not available or failing?
        # Actually DataUpdateCoordinator has async_set_updated_data.
        # Ah, we mocked `super().__init__`? No, we didn't.
        # But `HeatingDataCoordinator` calls `super().__init__`.
        # The error says: 'HeatingDataCoordinator' object has no attribute 'async_set_updated_data'.
        # This implies that `super().__init__` might not have been called correctly or something in the environment is off.
        # Let's just mock it for the test since we don't care about the listener update here.
        coord.async_set_updated_data = MagicMock()

        return coord

@pytest.mark.asyncio
async def test_mode_selection(mock_coordinator):
    """Test setting and getting unit modes."""
    entity_id = "sensor.heater_1"

    # Default should be HEATING (if not set)
    # The getter currently defaults to HEATING in coordinator.py: return self._unit_modes.get(entity_id, MODE_HEATING)
    assert mock_coordinator.get_unit_mode(entity_id) == MODE_HEATING

    # Set to COOLING
    await mock_coordinator.async_set_unit_mode(entity_id, MODE_COOLING)
    assert mock_coordinator.get_unit_mode(entity_id) == MODE_COOLING
    assert mock_coordinator._unit_modes[entity_id] == MODE_COOLING

    # Verify Save was triggered
    mock_coordinator.storage.async_save_data.assert_called()

    # Set to OFF
    await mock_coordinator.async_set_unit_mode(entity_id, MODE_OFF)
    assert mock_coordinator.get_unit_mode(entity_id) == MODE_OFF

    # Set to HEATING
    await mock_coordinator.async_set_unit_mode(entity_id, MODE_HEATING)
    assert mock_coordinator.get_unit_mode(entity_id) == MODE_HEATING

@pytest.mark.asyncio
async def test_solar_interaction_with_modes(mock_coordinator):
    """Test that solar correction behaves differently based on mode."""
    # We are testing the integration of coordinator -> solar.apply_correction
    # We need to use the REAL SolarCalculator logic for this test, or mock it carefully.
    # Let's use the REAL SolarCalculator logic by importing it,
    # but still use the mock coordinator structure.

    from custom_components.heating_analytics.solar import SolarCalculator
    real_solar = SolarCalculator(mock_coordinator)

    base_kwh = 10.0
    solar_impact = 2.0

    # Test HEATING: Impact reduces consumption
    # 10 - 2 = 8
    result_heating = real_solar.apply_correction(base_kwh, solar_impact, MODE_HEATING)
    assert result_heating == 8.0

    # Test COOLING: Impact increases consumption
    # 10 + 2 = 12
    result_cooling = real_solar.apply_correction(base_kwh, solar_impact, MODE_COOLING)
    assert result_cooling == 12.0

    # Test OFF: Zero Baseline
    # 0
    result_off = real_solar.apply_correction(base_kwh, solar_impact, MODE_OFF)
    assert result_off == 0.0

    # Test Clamping (Heating)
    # 1.0 - 2.0 = -1.0 -> 0.0
    result_clamped = real_solar.apply_correction(1.0, 2.0, MODE_HEATING)
    assert result_clamped == 0.0

@pytest.mark.asyncio
async def test_learning_normalization_with_modes(mock_coordinator):
    """Test that learning normalization behaves differently based on mode."""
    from custom_components.heating_analytics.solar import SolarCalculator
    real_solar = SolarCalculator(mock_coordinator)

    actual_kwh = 8.0
    solar_impact = 2.0

    # HEATING: Actual (8) was reduced by solar (2). Base (Dark) = 8 + 2 = 10
    norm_heating = real_solar.normalize_for_learning(actual_kwh, solar_impact, MODE_HEATING)
    assert norm_heating == 10.0

    # COOLING: Actual (8) was increased by solar (2). Base (Dark) = 8 - 2 = 6
    norm_cooling = real_solar.normalize_for_learning(actual_kwh, solar_impact, MODE_COOLING)
    assert norm_cooling == 6.0

    # OFF: No correction
    norm_off = real_solar.normalize_for_learning(actual_kwh, solar_impact, MODE_OFF)
    assert norm_off == 8.0

@pytest.mark.asyncio
async def test_persistence_integration(hass):
    """Test that unit modes are saved and loaded correctly via StorageManager."""
    # This requires a more integrated test with the real StorageManager
    # and mocking the underlying Store.

    from custom_components.heating_analytics.storage import StorageManager

    # Setup Coordinator
    entry = MagicMock()
    entry.data = {
        "energy_sensors": ["sensor.h1", "sensor.h2"],
        "outdoor_temp_source": "sensor",
        "wind_source": "sensor",
        "wind_gust_source": "sensor"
    }
    coord = HeatingDataCoordinator(hass, entry)
    # Mock dependencies to avoid side effects
    coord.forecast = MagicMock()
    coord.statistics = MagicMock()
    coord.solar = MagicMock()

    # Setup StorageManager
    storage = StorageManager(coord)
    coord.storage = storage

    # Mock the internal Store
    mock_store = AsyncMock()
    storage._store = mock_store

    # 1. Test Saving
    coord._unit_modes = {
        "sensor.h1": MODE_COOLING,
        "sensor.h2": MODE_OFF
    }

    await storage.async_save_data(force=True)

    # Verify save called with correct data structure
    args, _ = mock_store.async_save.call_args
    saved_data = args[0]

    assert "unit_modes" in saved_data
    assert saved_data["unit_modes"]["sensor.h1"] == MODE_COOLING
    assert saved_data["unit_modes"]["sensor.h2"] == MODE_OFF

    # 2. Test Loading
    # Clear memory
    coord._unit_modes = {}

    # Setup mock load return
    mock_store.async_load.return_value = {
        "unit_modes": {
            "sensor.h1": MODE_HEATING, # Changed for verification
            "sensor.h2": MODE_COOLING
        },
        "energy_sensors": ["sensor.h1", "sensor.h2"] # Context usually not in saved data like this but good for test
    }

    await storage.async_load_data()

    assert coord._unit_modes["sensor.h1"] == MODE_HEATING
    assert coord._unit_modes["sensor.h2"] == MODE_COOLING

@pytest.mark.asyncio
async def test_persistence_cleanup(hass):
    """Test that removed sensors are cleaned up from unit_modes on load."""
    from custom_components.heating_analytics.storage import StorageManager

    entry = MagicMock()
    # Only sensor.h1 is configured
    entry.data = {"energy_sensors": ["sensor.h1"]}
    coord = HeatingDataCoordinator(hass, entry)
    coord.forecast = MagicMock()
    coord.statistics = MagicMock()
    coord.solar = MagicMock()

    storage = StorageManager(coord)
    coord.storage = storage
    mock_store = AsyncMock()
    storage._store = mock_store

    # Saved data contains h1 and h2
    mock_store.async_load.return_value = {
        "unit_modes": {
            "sensor.h1": MODE_COOLING,
            "sensor.h2": MODE_OFF # Should be removed
        }
    }

    await storage.async_load_data()

    assert "sensor.h1" in coord._unit_modes
    assert "sensor.h2" not in coord._unit_modes

@pytest.mark.asyncio
async def test_calculate_total_power_respects_unit_modes(mock_coordinator):
    """Test that StatisticsManager.calculate_total_power passes the correct mode to solar."""
    from custom_components.heating_analytics.statistics import StatisticsManager

    # Instantiate a real StatisticsManager and attach it to the coordinator,
    # overriding the mock from the fixture.
    stats = StatisticsManager(mock_coordinator)
    mock_coordinator.statistics = stats

    # Set modes
    mock_coordinator._unit_modes = {
        "sensor.heater_1": MODE_COOLING,
        "sensor.heater_2": MODE_HEATING,
    }

    # Mock Solar methods
    mock_coordinator.solar.calculate_unit_coefficient.return_value = 1.0
    mock_coordinator.solar.calculate_unit_solar_impact.return_value = 0.5
    # calculate_saturation returns (applied, wasted, final_net)
    mock_coordinator.solar.calculate_saturation = MagicMock(return_value=(0.5, 0.0, 10.0))

    # Mock both per-unit and global prediction data
    mock_coordinator._correlation_data_per_unit = {
        "sensor.heater_1": {"10": {"normal": 5.0}},
        "sensor.heater_2": {"10": {"normal": 5.0}},
    }
    mock_coordinator._correlation_data = {"10": {"normal": 10.0}}
    mock_coordinator.balance_point = 15.0
    mock_coordinator._aux_coefficients = {}

    # Call method
    stats.calculate_total_power(
        temp=10.0, effective_wind=1.0, solar_impact=0.0, is_aux_active=False
    )

    # Verify calls to calculate_saturation
    assert mock_coordinator.solar.calculate_saturation.call_count == 2

    calls = mock_coordinator.solar.calculate_saturation.call_args_list
    found_cooling = False
    found_heating = False

    for call in calls:
        args = call.args  # (net_demand, solar_potential, mode)
        mode = args[2]
        if mode == MODE_COOLING:
            found_cooling = True
        elif mode == MODE_HEATING:
            found_heating = True

    assert found_cooling, "Did not find call with MODE_COOLING"
    assert found_heating, "Did not find call with MODE_HEATING"
