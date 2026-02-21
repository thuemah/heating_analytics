import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime
from homeassistant.core import HomeAssistant

from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.const import DOMAIN

async def test_aux_migration_conservation_logic(hass):
    """Test the conservation strategy for aux migration."""
    # 1. Setup Mock Coordinator
    entry = MagicMock()
    entry.data = {
        "energy_sensors": ["sensor.heater_a", "sensor.heater_b", "sensor.heater_c"],
        "aux_affected_entities": ["sensor.heater_a", "sensor.heater_b", "sensor.heater_c"]
    }

    coordinator = HeatingDataCoordinator(hass, entry)

    # Mock States for Total Consumption (Primary Unit Logic)
    # A=1000, B=2000, C=500 (B is Primary)
    hass.states.get = MagicMock()
    def get_state(entity_id):
        state = MagicMock()
        if entity_id == "sensor.heater_a": state.state = "1000"
        elif entity_id == "sensor.heater_b": state.state = "2000"
        elif entity_id == "sensor.heater_c": state.state = "500"
        else: return None
        return state
    hass.states.get.side_effect = get_state

    # 2. Setup Initial Data (Aux Coefficients)
    # 3 Units, Temp Key "0", Wind "normal"
    # A: 0.5 kW reduction
    # B: 1.0 kW reduction
    # C: 0.3 kW reduction
    # Total Reduction = 1.8 kW

    coordinator._aux_coefficients_per_unit = {
        "sensor.heater_a": {"0": {"normal": 0.5}},
        "sensor.heater_b": {"0": {"normal": 1.0}},
        "sensor.heater_c": {"0": {"normal": 0.3}},
    }

    coordinator._aux_coefficients = {"0": {"normal": 1.8}} # Global check

    # 3. Execute Migration: Remove C
    new_list = ["sensor.heater_a", "sensor.heater_b"]

    # We need to implement this method on the coordinator first,
    # but for TDD we will define the logic here or call the method assuming it exists.
    # Since we can't edit the class in the test file, we will patch the instance or mock the method
    # BUT the goal is to test the actual implementation.
    # So I will fail the test if the method doesn't exist, which is good TDD.

    # Mock save to avoid disk errors
    coordinator._async_save_data = AsyncMock()

    await coordinator.async_migrate_aux_coefficients(new_list)

    # 4. Verification

    # C should be gone
    assert "sensor.heater_c" not in coordinator._aux_coefficients_per_unit

    # Total Reduction for A+B should be roughly 1.8 (Conservation)
    # A was 0.5, B was 1.0. Total remaining weight = 1.5.
    # C was 0.3.
    # A gets: 0.5 + (0.5/1.5 * 0.3) = 0.5 + 0.1 = 0.6
    # B gets: 1.0 + (1.0/1.5 * 0.3) = 1.0 + 0.2 = 1.2

    coeffs_a = coordinator._aux_coefficients_per_unit["sensor.heater_a"]["0"]["normal"]
    coeffs_b = coordinator._aux_coefficients_per_unit["sensor.heater_b"]["0"]["normal"]

    assert coeffs_a == pytest.approx(0.6, 0.01)
    assert coeffs_b == pytest.approx(1.2, 0.01)
    assert (coeffs_a + coeffs_b) == pytest.approx(1.8, 0.01)

async def test_aux_migration_fallback_logic(hass):
    """Test fallback to Primary Unit when remaining units have no data."""
    entry = MagicMock()
    entry.data = {
        "energy_sensors": ["sensor.heater_a", "sensor.heater_b"],
        "aux_affected_entities": ["sensor.heater_a", "sensor.heater_b"]
    }
    coordinator = HeatingDataCoordinator(hass, entry)

    # Mock States: A is Primary (highest consumption)
    hass.states.get = MagicMock()
    def get_state(entity_id):
        state = MagicMock()
        if entity_id == "sensor.heater_a": state.state = "5000"
        elif entity_id == "sensor.heater_b": state.state = "100"
        return state
    hass.states.get.side_effect = get_state

    # Setup Data: B has data, A has NO data for this bucket
    # Actually, if we remove B, we move data to A.
    # Bucket "0/normal"
    coordinator._aux_coefficients_per_unit = {
        "sensor.heater_b": {"0": {"normal": 0.5}},
        # A has no entry for "0"
    }

    new_list = ["sensor.heater_a"]

    # Mock save
    coordinator._async_save_data = AsyncMock()

    await coordinator.async_migrate_aux_coefficients(new_list)

    # Verification
    assert "sensor.heater_b" not in coordinator._aux_coefficients_per_unit

    # A should inherit B's value because A is the only remaining (and Primary)
    # Even though A had weight 0 (missing), fallback logic assigns to Primary.

    assert "sensor.heater_a" in coordinator._aux_coefficients_per_unit
    assert "0" in coordinator._aux_coefficients_per_unit["sensor.heater_a"]
    assert coordinator._aux_coefficients_per_unit["sensor.heater_a"]["0"]["normal"] == 0.5
