"""Comprehensive test for sensor attributes."""
from unittest.mock import MagicMock
import pytest
import json
from homeassistant.core import HomeAssistant
from custom_components.heating_analytics.sensor import HeatingCorrelationDataSensor
from custom_components.heating_analytics.const import ATTR_CORRELATION_DATA

@pytest.fixture
def mock_hass():
    """Mock Home Assistant."""
    return MagicMock()

@pytest.fixture
def mock_entry():
    """Mock Config Entry."""
    entry = MagicMock()
    entry.entry_id = "test_entry"
    return entry

@pytest.fixture
def mock_coordinator():
    """Mock Coordinator."""
    coord = MagicMock()
    coord.data = {}
    coord._aux_coefficients = {}
    return coord

@pytest.mark.asyncio
async def test_correlation_data_sensor_attributes(hass: HomeAssistant, mock_coordinator, mock_entry):
    """Test HeatingCorrelationDataSensor attributes and JSON generation."""
    mock_coordinator.data = {
        ATTR_CORRELATION_DATA: {
            "0": {"normal": 1.0, "high_wind": 1.2},
            "-5": {"normal": 2.0, "extreme_wind": 2.5},
            "10": {"normal": 0.5}
        }
    }

    # Add Aux Coefficients (New Style)
    mock_coordinator._aux_coefficients = {
        "10": 0.4
    }

    sensor = HeatingCorrelationDataSensor(mock_coordinator, mock_entry)
    assert sensor.native_value == "Data"

    attrs = sensor.extra_state_attributes

    # Verify sorting and JSON structure
    # Expected order: -5, 0, 10

    # Normal: -5 -> 2.0, 0 -> 1.0, 10 -> 0.5. (kWh/day = val * 24)
    # -5 -> 48.0, 0 -> 24.0, 10 -> 12.0
    import json
    normal_x = json.loads(attrs["normal_x"])
    normal_y = json.loads(attrs["normal_y"])

    assert normal_x == [-5, 0, 10]
    assert normal_y == [48.0, 24.0, 12.0]

    # Extreme Wind: only at -5 -> 2.5 * 24 = 60.0
    extreme_x = json.loads(attrs["extreme_wind_x"])
    extreme_y = json.loads(attrs["extreme_wind_y"])
    assert extreme_x == [-5]
    assert extreme_y == [60.0]

    # Auxiliary Impact (New Attribute)
    # 10 -> 0.4 (kW)
    aux_x = json.loads(attrs["aux_impact_x"])
    aux_y = json.loads(attrs["aux_impact_y"])
    assert aux_x == [10]
    assert aux_y == [0.4]
