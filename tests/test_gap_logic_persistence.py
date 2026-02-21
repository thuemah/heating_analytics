"""Test gap logic persistence."""
import pytest
from unittest.mock import MagicMock, AsyncMock
from custom_components.heating_analytics.storage import StorageManager
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from homeassistant.util import dt as dt_util

@pytest.mark.asyncio
async def test_gap_state_persistence_and_restoration(hass):
    """Test that gap filling state is persisted and restored correctly."""

    # Setup Coordinator
    entry = MagicMock()
    entry.data = {
        "energy_sensors": ["sensor.heater"]
    }
    coordinator = HeatingDataCoordinator(hass, entry)

    # Initialize Storage
    storage = StorageManager(coordinator)

    # Set up state that MUST be persisted for robust gap filling
    test_unit_breakdown = {"sensor.heater": {"net_kwh": 5.0, "base_kwh": 5.0, "aux_reduction_kwh": 0.0, "overflow_kwh": 0.0, "solar_reduction_kwh": 0.0}}

    coordinator.data["current_model_rate"] = 5.0
    coordinator.data["current_aux_impact_rate"] = 1.0
    coordinator.data["current_unit_breakdown"] = test_unit_breakdown
    coordinator.data["current_calc_temp"] = 10.5
    coordinator._last_minute_processed = 58
    # CRITICAL: Set start time to now so restoration succeeds (same hour check)
    coordinator._accumulation_start_time = dt_util.now()

    # Mock store
    mock_store = AsyncMock()
    storage._store = mock_store

    # --- SAVE ---
    await storage.async_save_data(force=True)

    # Inspect what was saved
    saved_data = mock_store.async_save.call_args[0][0]

    # These assertions verify the fix. Before fix, they should FAIL.
    assert "current_model_rate" in saved_data, "current_model_rate not saved"
    assert "current_aux_impact_rate" in saved_data, "current_aux_impact_rate not saved"
    assert "current_unit_breakdown" in saved_data, "current_unit_breakdown not saved"
    assert "current_calc_temp" in saved_data, "current_calc_temp not saved"

    # --- LOAD ---
    # Create new coordinator to simulate restart
    new_coordinator = HeatingDataCoordinator(hass, entry)
    new_storage = StorageManager(new_coordinator)
    new_storage._store = AsyncMock()

    # Mock return data
    new_storage._store.async_load.return_value = saved_data

    await new_storage.async_load_data()

    # Verify restoration
    assert new_coordinator.data.get("current_model_rate") == 5.0
    assert new_coordinator.data.get("current_aux_impact_rate") == 1.0
    assert new_coordinator.data.get("current_unit_breakdown") == test_unit_breakdown
    assert new_coordinator.data.get("current_calc_temp") == 10.5
