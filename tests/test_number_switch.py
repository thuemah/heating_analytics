"""Test Number and Switch entities."""
from unittest.mock import MagicMock, patch, AsyncMock
import pytest
from homeassistant.core import HomeAssistant
from homeassistant.const import CONF_NAME
from custom_components.heating_analytics.const import DOMAIN
from custom_components.heating_analytics.number import HeatingLearningRateNumber
from custom_components.heating_analytics.switch import HeatingLearningSwitch, HeatingAuxiliaryHeatingSwitch

@pytest.mark.asyncio
async def test_learning_rate_number(hass: HomeAssistant):
    """Test HeatingLearningRateNumber entity."""
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.title = "Test Heating"

    coordinator = MagicMock()
    coordinator.learning_rate = 0.01
    coordinator._async_save_data = AsyncMock()

    # Instantiate Number
    entity = HeatingLearningRateNumber(coordinator, entry)
    entity.hass = hass
    entity.async_write_ha_state = MagicMock()

    # Check attributes
    assert entity.unique_id == "test_entry_learning_rate"
    assert entity.name == "Learning Rate"
    assert entity.native_value == 1.0  # 0.01 * 100

    # Test Set Value
    await entity.async_set_native_value(2.5)

    # Verify coordinator update
    assert coordinator.learning_rate == 0.025 # 2.5 / 100
    coordinator._async_save_data.assert_called_once()


@pytest.mark.asyncio
async def test_learning_switch(hass: HomeAssistant):
    """Test HeatingLearningSwitch entity."""
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.title = "Test Heating"

    coordinator = MagicMock()
    coordinator.learning_enabled = False
    coordinator._async_save_data = AsyncMock()

    # Instantiate Switch
    entity = HeatingLearningSwitch(coordinator, entry)
    entity.hass = hass
    entity.async_write_ha_state = MagicMock()

    assert entity.is_on is False

    # Turn On
    await entity.async_turn_on()

    assert coordinator.learning_enabled is True
    coordinator._async_save_data.assert_called()

    # Turn Off
    await entity.async_turn_off()

    assert coordinator.learning_enabled is False


@pytest.mark.asyncio
async def test_auxiliary_switch(hass: HomeAssistant):
    """Test HeatingAuxiliaryHeatingSwitch entity."""
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.title = "Test Heating"

    coordinator = MagicMock()
    coordinator.auxiliary_heating_active = False
    coordinator._async_save_data = AsyncMock()

    async def set_aux_active(active):
        coordinator.auxiliary_heating_active = active
        await coordinator._async_save_data()

    coordinator.set_auxiliary_heating_active = AsyncMock(side_effect=set_aux_active)

    # Instantiate Switch
    entity = HeatingAuxiliaryHeatingSwitch(coordinator, entry)
    entity.hass = hass
    entity.async_write_ha_state = MagicMock()

    assert entity.is_on is False

    # Turn On
    await entity.async_turn_on()

    assert coordinator.auxiliary_heating_active is True
    coordinator._async_save_data.assert_called()
