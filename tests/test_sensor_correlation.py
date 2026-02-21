"""Test sensors module."""
from unittest.mock import MagicMock
import pytest
import json
from homeassistant.const import EntityCategory
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
    # FIX: Add _aux_coefficients
    coord._aux_coefficients = {}
    return coord

def test_correlation_sensor_attributes(mock_hass, mock_entry, mock_coordinator):
    """Test that the correlation sensor returns correctly formatted attributes as JSON strings."""

    # Mock data in coordinator
    # Structure: { "temp": { "wind_bucket": avg_kwh_per_hour } }
    # Temperatures: 5, 0, -5
    mock_data = {
        "5": {
            "normal": 1.0, # 24 kWh/day
            "high_wind": 1.5, # 36 kWh/day
            "extreme_wind": 2.0, # 48 kWh/day
        },
        "0": {
            "normal": 2.0, # 48
            "high_wind": 2.5, # 60
            # Missing extreme
        },
        "-5": {
            "normal": 3.0, # 72
            "high_wind": 3.5, # 84
            "extreme_wind": 4.0, # 96
        },
        # Unsorted input to test sorting
        "10": {
            "normal": 0.5 # 12
        }
    }

    mock_coordinator.data = {
        ATTR_CORRELATION_DATA: mock_data
    }

    # Mock Aux Impact
    mock_coordinator._aux_coefficients = {
        "0": 0.2
    }

    sensor = HeatingCorrelationDataSensor(mock_coordinator, mock_entry)

    # Check attributes
    attrs = sensor.extra_state_attributes

    assert "normal_x" in attrs
    assert "normal_y" in attrs
    assert "high_wind_x" in attrs
    assert "extreme_wind_x" in attrs

    # Check Aux
    assert "aux_impact_x" in attrs
    aux_x = json.loads(attrs["aux_impact_x"])
    assert aux_x == [0]

    # Verify Sorting (Lowest temp first? Or numerical order?)
    # Implementation sorts by int(k)
    # So: -5, 0, 5, 10

    normal_x = json.loads(attrs["normal_x"])
    assert normal_x == [-5, 0, 5, 10]

    # Verify Values (kWh/day)
    # -5 normal: 3.0 * 24 = 72.0
    normal_y = json.loads(attrs["normal_y"])
    assert normal_y[0] == 72.0
