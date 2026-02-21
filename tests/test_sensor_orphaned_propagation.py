"""Test suite for Orphaned Savings Propagation to Sensor."""
from unittest.mock import MagicMock
import pytest
from custom_components.heating_analytics.sensor import HeatingPotentialSavingsSensor
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator

@pytest.fixture
def mock_coordinator_with_orphaned():
    coord = MagicMock(spec=HeatingDataCoordinator)
    coord.energy_sensors = ["sensor.heater_1"]
    coord.aux_affected_entities = ["sensor.heater_1"]
    coord.auxiliary_heating_active = True

    # Mock hass object
    coord.hass = MagicMock()
    coord.hass.states.get.return_value = MagicMock()
    coord.hass.states.get.return_value.name = "Heater 1"

    # Mock Statistics Data structure
    coord.data = {
        "current_unit_breakdown": {
            "sensor.heater_1": {
                "raw_aux_kwh": 2.0,
                "aux_reduction_kwh": 2.0, # 2.0 Allocated
                "overflow_kwh": 1.0,      # 1.0 Overflow (Unit)
                "clamped": True
            }
        },
        "potential_savings_breakdown": {},
        "accumulated_aux_impact_kwh": 10.0,
        "savings_aux_hours_today": 5.0,
        "savings_aux_hours_list": [],
    }

    # Mock Internal Accumulators
    coord._daily_aux_breakdown = {
        "sensor.heater_1": {"allocated": 5.0, "overflow": 2.0}
    }
    coord._accumulated_aux_breakdown = {
        "sensor.heater_1": {"allocated": 1.0, "overflow": 0.5}
    }

    # Mock Orphaned Accumulators (The Fix)
    coord._daily_orphaned_aux = 3.0
    coord._accumulated_orphaned_aux = 0.5

    # Mock hourly log to avoid error accessing learning status
    coord._hourly_log = []

    return coord

def test_sensor_includes_orphaned_savings(mock_coordinator_with_orphaned):
    """Verify that HeatingPotentialSavingsSensor includes orphaned savings in unassigned total."""
    entry = MagicMock()
    entry.entry_id = "test_entry"

    sensor = HeatingPotentialSavingsSensor(mock_coordinator_with_orphaned, entry)

    # Calculate expected values
    # Allocated: Daily (5.0) + Live (1.0) = 6.0
    # Unit Overflow: Daily (2.0) + Live (0.5) = 2.5
    # Orphaned: Daily (3.0) + Live (0.5) = 3.5
    # Total Unassigned = Unit Overflow + Orphaned = 2.5 + 3.5 = 6.0

    attrs = sensor.extra_state_attributes

    print(f"\nUnassigned: {attrs['unassigned_kwh']}")
    print(f"Allocated: {attrs['allocated_total_kwh']}")

    assert attrs["allocated_total_kwh"] == 6.0
    assert attrs["unassigned_kwh"] == 6.0
    assert attrs["leak_status"] == "critical_leak" # > 0.5
