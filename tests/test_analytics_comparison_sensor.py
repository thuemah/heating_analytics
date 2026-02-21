"""Test HeatingAnalyticsComparisonSensor attributes."""
from unittest.mock import MagicMock
import pytest
from custom_components.heating_analytics.sensor import HeatingAnalyticsComparisonSensor

@pytest.fixture
def mock_coordinator():
    coordinator = MagicMock()
    # Mock data with the structure returned by statistics.compare_periods
    coordinator.data = {
        "last_comparison": {
            "period_1": {"start_date": "2023-01-01"},
            "period_2": {"start_date": "2024-01-01"},
            "delta_actual_kwh": 10.0,
            "delta_temp": -2.0
        }
    }
    return coordinator

@pytest.fixture
def mock_entry():
    entry = MagicMock()
    entry.entry_id = "test_entry"
    return entry

def test_comparison_sensor_attributes(mock_coordinator, mock_entry):
    """Test that sensor attributes expose the comparison data including deltas."""
    sensor = HeatingAnalyticsComparisonSensor(mock_coordinator, mock_entry)

    # Check state
    assert sensor.native_value == "Comparing 2023-01-01 to 2024-01-01"

    # Check attributes
    attrs = sensor.extra_state_attributes
    assert attrs["delta_actual_kwh"] == 10.0
    assert attrs["delta_temp"] == -2.0
    assert "period_1" in attrs
