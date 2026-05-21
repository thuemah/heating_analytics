"""Tests for Mode Control (Phase 3)."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from homeassistant.core import HomeAssistant
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.const import (
    DOMAIN,
    MODE_HEATING,
    MODE_COOLING,
    MODE_OFF,
    STORAGE_KEY,
    STORAGE_VERSION,
    MODE_GUEST_COOLING
)

@pytest.mark.asyncio
async def test_mode_selection(hass, mock_entry):
    """Test setting and getting unit modes."""
    # Specialized setup: real coordinator with mocks for dependencies
    with patch("custom_components.heating_analytics.coordinator.StorageManager"), \
         patch("custom_components.heating_analytics.coordinator.ForecastManager"), \
         patch("custom_components.heating_analytics.coordinator.StatisticsManager"), \
         patch("custom_components.heating_analytics.coordinator.LearningManager"), \
         patch("custom_components.heating_analytics.coordinator.SolarCalculator"):

        coord = HeatingDataCoordinator(hass, mock_entry)
        coord.async_set_updated_data = MagicMock()
        coord.storage.async_save_data = AsyncMock()
        coord.energy_sensors = ["sensor.heater_1"]

        entity_id = "sensor.heater_1"

        # Default should be HEATING (if not set)
        assert coord.get_unit_mode(entity_id) == MODE_HEATING

        # Set to COOLING
        await coord.async_set_unit_mode(entity_id, MODE_COOLING)
        assert coord.get_unit_mode(entity_id) == MODE_COOLING
        assert coord._unit_modes[entity_id] == MODE_COOLING

        # Verify Save was triggered
        coord.storage.async_save_data.assert_called()

        # Set to OFF
        await coord.async_set_unit_mode(entity_id, MODE_OFF)
        assert coord.get_unit_mode(entity_id) == MODE_OFF

        # Set to HEATING
        await coord.async_set_unit_mode(entity_id, MODE_HEATING)
        assert coord.get_unit_mode(entity_id) == MODE_HEATING

@pytest.mark.asyncio
async def test_solar_interaction_with_modes(mock_coordinator):
    """Test that solar correction behaves differently based on mode."""
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
async def test_persistence_integration(hass, mock_entry):
    """Test that unit modes are saved and loaded correctly via StorageManager."""
    from custom_components.heating_analytics.storage import StorageManager

    # Setup Coordinator
    coord = HeatingDataCoordinator(hass, mock_entry)
    coord.energy_sensors = ["sensor.h1", "sensor.h2"]
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
        "energy_sensors": ["sensor.h1", "sensor.h2"]
    }

    await storage.async_load_data()

    assert coord._unit_modes["sensor.h1"] == MODE_HEATING
    assert coord._unit_modes["sensor.h2"] == MODE_COOLING

@pytest.mark.asyncio
async def test_persistence_cleanup(hass, mock_entry):
    """Test that removed sensors are cleaned up from unit_modes on load."""
    from custom_components.heating_analytics.storage import StorageManager

    # Only sensor.h1 is configured
    mock_entry.data = {"energy_sensors": ["sensor.h1"]}
    coord = HeatingDataCoordinator(hass, mock_entry)
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

    # Setup mock_coordinator to use its _unit_modes in get_unit_mode
    mock_coordinator._unit_modes = {
        "sensor.heater_1": MODE_COOLING,
        "sensor.heater_2": MODE_HEATING,
    }
    mock_coordinator.get_unit_mode.side_effect = lambda eid: mock_coordinator._unit_modes.get(eid, MODE_HEATING)

    # Instantiate a real StatisticsManager and attach it to the coordinator
    stats = StatisticsManager(mock_coordinator)
    mock_coordinator.statistics = stats

    mock_coordinator.energy_sensors = ["sensor.heater_1", "sensor.heater_2"]

    # Mock Solar methods
    mock_coordinator.solar.calculate_unit_coefficient.return_value = {"s": 1.0, "e": 0.0, "w": 0.0}
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
    mock_coordinator._get_predicted_kwh.return_value = 10.0
    mock_coordinator.model.aux_coefficients = {}

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
