"""Test regarding Potential Savings Sensor breakdown logic."""
from unittest.mock import MagicMock
import pytest
from custom_components.heating_analytics.sensor import HeatingPotentialSavingsSensor
from custom_components.heating_analytics.const import ATTR_POTENTIAL_SAVINGS

async def test_potential_savings_breakdown_mismatch(hass):
    """Test that breakdown shows POTENTIAL rate when passive, matching global rate."""
    coordinator = MagicMock()
    coordinator.hass = hass
    coordinator.aux_affected_entities = ["sensor.heater"]
    coordinator.auxiliary_heating_active = False # Passive

    # Setup Data
    coordinator.data = {
        ATTR_POTENTIAL_SAVINGS: 22.355,
        "savings_actual_kwh": 0.679,
        "current_savings_rate": 0.94, # kW
        "accumulated_aux_impact_kwh": 11.12,

        # Current Unit Breakdown (Actual = Passive = 0)
        "current_unit_breakdown": {
            "sensor.heater": {
                "raw_aux_kwh": 0.0,
                "aux_reduction_kwh": 0.0,
                "overflow_kwh": 0.0,
                "clamped": False
            }
        },

        # New: Potential Breakdown (Calculated by statistics.py)
        "potential_savings_breakdown": {
            "sensor.heater": {
                "raw_aux_kwh": 0.94,
                "aux_reduction_kwh": 0.94,
                "overflow_kwh": 0.0,
                "clamped": False
            }
        },

        # Savings stats
        "savings_aux_hours_today": 13,
        "savings_aux_hours_list": []
    }

    # Mock internal accumulators
    coordinator._daily_aux_breakdown = {
        "sensor.heater": {"allocated": 10.894, "overflow": 0.228}
    }
    coordinator._accumulated_aux_breakdown = {
        "sensor.heater": {"allocated": 0.0, "overflow": 0.0}
    }
    # Initialize orphaned accumulators to avoid MagicMock recursion
    coordinator._daily_orphaned_aux = 0.0
    coordinator._accumulated_orphaned_aux = 0.0
    # Initialize _hourly_log to avoid MagicMock recursion in extra_state_attributes
    coordinator._hourly_log = []

    # Mock State for name lookup
    mock_state = MagicMock()
    mock_state.name = "Heater"
    hass.states.get = MagicMock(return_value=mock_state)

    # Instantiate Sensor
    entry = MagicMock()
    entry.entry_id = "test_entry"
    sensor = HeatingPotentialSavingsSensor(coordinator, entry)

    # Execute
    attrs = sensor.extra_state_attributes
    breakdown = attrs["unit_breakdown"]

    # Verify Fix
    assert attrs["current_savings_rate_kw"] == "0.94 kW"
    assert len(breakdown) == 1
    item = breakdown[0]

    assert item["name"] == "Heater"
    # Current rate should be 940 (0.94 kW) because we now use potential breakdown
    assert item["current_rate_w"] == 940

    print("\nVerification Successful: current_rate_w matches global potential rate (0.94 kW)")
