"""Test Lifetime Sensor Optimization."""
from unittest.mock import MagicMock
import pytest
from homeassistant.core import HomeAssistant
from custom_components.heating_analytics.sensor import HeatingDeviceLifetimeSensor

@pytest.fixture
def mock_coordinator():
    coordinator = MagicMock()
    coordinator.data = {}
    return coordinator

@pytest.fixture
def mock_entry():
    entry = MagicMock()
    entry.entry_id = "test_entry"
    return entry

@pytest.mark.asyncio
async def test_lifetime_sensor_throttling(mock_coordinator, mock_entry):
    """Test that lifetime sensor updates are throttled and rounded."""
    hass = MagicMock()
    entity_id = "sensor.heater_1"

    mock_state = MagicMock()
    mock_state.name = "Heater One"
    hass.states.get.return_value = mock_state
    mock_coordinator.hass = hass

    # Initialize with 0
    mock_coordinator.data = {
        "lifetime_individual": {entity_id: 1000.0}
    }

    sensor = HeatingDeviceLifetimeSensor(mock_coordinator, mock_entry, entity_id)
    sensor.hass = hass

    # 1. Initial Call -> Should update from 0.0 to 1000.0
    # Current code rounds to 3 decimals. Proposed code rounds to 1.
    # We will assert the new behavior we WANT.
    assert sensor.native_value == 1000.0

    # 2. Small Change (< 0.1) -> Should NOT update (remain 1000.0)
    # 1000.0 -> 1000.05
    mock_coordinator.data["lifetime_individual"][entity_id] = 1000.05
    assert sensor.native_value == 1000.0

    # 3. Another Small Change (cumulative < 0.1 from last reported) -> Should NOT update
    # 1000.0 -> 1000.09
    mock_coordinator.data["lifetime_individual"][entity_id] = 1000.09
    assert sensor.native_value == 1000.0

    # 4. Large Change (>= 0.1) -> Should Update
    # 1000.0 -> 1000.12
    mock_coordinator.data["lifetime_individual"][entity_id] = 1000.12
    assert sensor.native_value == 1000.1

    # 5. Verify rounding (one decimal)
    mock_coordinator.data["lifetime_individual"][entity_id] = 1000.26
    # 0.26 - 0.12 = 0.14 >= 0.1, so it updates
    # Should round to 1 decimal: 1000.3
    assert sensor.native_value == 1000.3
