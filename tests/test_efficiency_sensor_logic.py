"""Tests for the Efficiency Sensor Logic."""
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
import sys
from custom_components.heating_analytics.const import (
    ATTR_TDD,
    ATTR_TDD_SO_FAR,
    ATTR_TDD_YESTERDAY,
    ATTR_TDD_LAST_7D,
    ATTR_TDD_LAST_30D,
    ATTR_TDD_DAILY_STABLE,
    ATTR_EFFICIENCY_YESTERDAY,
    ATTR_EFFICIENCY_LAST_7D,
    ATTR_EFFICIENCY_LAST_30D,
    ATTR_EFFICIENCY_FORECAST_TODAY
)

# Move imports inside tests/fixtures to avoid metaclass conflict during collection
# from custom_components.heating_analytics.sensor import HeatingEfficiencySensor

@pytest.fixture
def mock_coordinator():
    """Mock the HeatingDataCoordinator."""
    coordinator = MagicMock()
    coordinator.data = {}
    coordinator.statistics = MagicMock()
    return coordinator

@pytest.fixture
def mock_entry():
    """Mock ConfigEntry."""
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.title = "Test Heating"
    return entry

def test_efficiency_calculation_delegation(mock_coordinator, mock_entry):
    """Test that native_value delegates to StatisticsManager."""
    # Import inside test to avoid metaclass conflict with pytest-homeassistant-custom-component
    from custom_components.heating_analytics.sensor import HeatingEfficiencySensor

    sensor = HeatingEfficiencySensor(mock_coordinator, mock_entry)

    # Setup return value
    mock_coordinator.statistics.calculate_realtime_efficiency.return_value = 12.5

    # Check value
    assert sensor.native_value == 12.5
    mock_coordinator.statistics.calculate_realtime_efficiency.assert_called_once()

def test_efficiency_attributes(mock_coordinator, mock_entry):
    """Test that attributes are correctly retrieved from coordinator data."""
    from custom_components.heating_analytics.sensor import HeatingEfficiencySensor

    sensor = HeatingEfficiencySensor(mock_coordinator, mock_entry)

    # Setup data using correct constants
    mock_coordinator.data = {
        ATTR_TDD: 10.0,
        ATTR_TDD_SO_FAR: 5.0,
        ATTR_TDD_YESTERDAY: 15.0,
        ATTR_TDD_LAST_7D: 12.0,
        ATTR_TDD_LAST_30D: 11.0,
        ATTR_TDD_DAILY_STABLE: 20.0, # Forecast today
        ATTR_EFFICIENCY_YESTERDAY: 3.5,
        ATTR_EFFICIENCY_LAST_7D: 3.6,
        ATTR_EFFICIENCY_LAST_30D: 3.7,
        ATTR_EFFICIENCY_FORECAST_TODAY: 3.8
    }

    # Get attributes
    attrs = sensor.extra_state_attributes

    # tdd_today should now map to ATTR_TDD_DAILY_STABLE (20.0) not ATTR_TDD (10.0)
    assert attrs["tdd_today"] == 20.0
    assert attrs["tdd_accumulated"] == 10.0
    assert attrs["tdd_so_far"] == 5.0
    assert attrs["tdd_yesterday"] == 15.0
    assert attrs["tdd_last_7d_avg"] == 12.0
    assert attrs["efficiency_yesterday"] == 3.5
    assert attrs["efficiency_last_7d_avg"] == 3.6

def test_efficiency_sensor_unique_id(mock_coordinator, mock_entry):
    """Test unique ID generation."""
    from custom_components.heating_analytics.sensor import HeatingEfficiencySensor

    sensor = HeatingEfficiencySensor(mock_coordinator, mock_entry)
    assert sensor.unique_id == "test_entry_efficiency"
