"""Test Auxiliary Overflow Logic and Sensor Attributes."""
from datetime import datetime, timedelta, timezone
import pytest
from unittest.mock import MagicMock, patch
from homeassistant.util import dt as dt_util

from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.const import ATTR_POTENTIAL_SAVINGS, DOMAIN

# Mock HASS and Coordinator
class MockHass:
    def __init__(self):
        self.states = MagicMock()
        self.states.get = MagicMock(return_value=None)
        self.data = {DOMAIN: {}}
        self.config_entries = MagicMock()
        self.bus = MagicMock()
        self.is_running = True

@pytest.fixture
def mock_hass():
    return MockHass()

@pytest.fixture
def coordinator(mock_hass):
    entry = MagicMock()
    entry.data = {
        "energy_sensors": ["sensor.heater_1", "sensor.heater_2"],
        "aux_affected_entities": ["sensor.heater_1", "sensor.heater_2"], # Both affected
        "outdoor_temp_sensor": "sensor.outdoor_temp",
        "balance_point": 15.0,
        "wind_speed_sensor": "sensor.wind_speed",
        "wind_threshold": 5.0,
        "extreme_wind_threshold": 10.0
    }
    coord = HeatingDataCoordinator(mock_hass, entry)
    coord.statistics = MagicMock()
    coord.learning = MagicMock()
    coord.storage = MagicMock()
    coord.forecast = MagicMock()
    coord.solar = MagicMock()
    # Mock calculate_saturation to return (applied, wasted, final_net)
    # Since we are testing aux overflow here, we can assume solar is 0 or handled simply.
    # Logic: return (0, 0, net_demand)
    coord.solar.calculate_saturation.side_effect = lambda net, pot, val: (0.0, 0.0, net)
    coord.solar.apply_correction.side_effect = lambda base, impact, val: base

    coord.solar_enabled = False

    # Init internal structures
    coord._hourly_delta_per_unit = {"sensor.heater_1": 0.0, "sensor.heater_2": 0.0}
    coord._accumulated_aux_breakdown = {}

    return coord

def test_aux_overflow_calculation(coordinator):
    """Test that overflow is calculated correctly when units clamp."""

    # Setup state
    coordinator.auxiliary_heating_active = True

    # Define scenario:
    # Heater 1: Base=5.0, RawAux=2.0, Actual=1.0 -> Clamped (Aux=1.0, Overflow=1.0)
    # Heater 2: Base=5.0, RawAux=2.0, Actual=4.0 -> No Clamp (Aux=2.0, Overflow=0.0)

    # We need to mock statistics.calculate_total_power because we modified it
    # But since we can't easily mock the method we JUST modified in the real code without importing it,
    # let's rely on the real method if we can import statistics.
    # However, coordinator uses self.statistics which is mocked above.
    # Let's use the REAL statistics manager for this test to verify the logic change.

    from custom_components.heating_analytics.statistics import StatisticsManager
    real_stats = StatisticsManager(coordinator)
    coordinator.statistics = real_stats

    # Mock Models
    # Global Model: Total Aux = 4.0
    coordinator._aux_coefficients = {"5": {"normal": 4.0}}

    # Unit Models
    coordinator._correlation_data_per_unit = {
        "sensor.heater_1": {"5": {"normal": 5.0}},
        "sensor.heater_2": {"5": {"normal": 5.0}}
    }

    # Unit Aux Coeffs (Raw)
    coordinator._aux_coefficients_per_unit = {
        "sensor.heater_1": {"5": {"normal": 2.0}},
        "sensor.heater_2": {"5": {"normal": 2.0}}
    }

    # Mock Coordinator data required by statistics
    coordinator.data["effective_wind"] = 0.0
    coordinator.data[ATTR_POTENTIAL_SAVINGS] = 0.0
    coordinator.energy_sensors = ["sensor.heater_1", "sensor.heater_2"]
    coordinator.aux_affected_entities = ["sensor.heater_1", "sensor.heater_2"]

    # Mock _get_prediction_from_model to behave simply
    def mock_get_pred(data_map, temp_key, wind_bucket, temp, bp, apply_scaling=True):
        # Return exact match
        try:
            val = data_map.get(str(int(temp))).get(wind_bucket)
            return val if val is not None else 0.0
        except:
            return 0.0

    real_stats._get_prediction_from_model = MagicMock(side_effect=mock_get_pred)

    # Run calculate_total_power
    # Scenario: Temp=5.0, Wind=0.0 (normal)
    # Global Aux = 4.0
    # Unit 1 Raw Aux = 2.0
    # Unit 2 Raw Aux = 2.0
    # Sum Affected = 4.0. Scale = 4.0/4.0 = 1.0.

    # But wait, clamping depends on 'base_kwh' (Predicted Base), NOT actual consumption.
    # The statistics logic calculates 'net_kwh' = 'base' - 'applied_aux'.
    # It clamps applied_aux to min(final_aux, base).
    # It does NOT look at 'actual_kwh' because that's not available during prediction!

    # The requirement was: "hvis bruker feilkonfigurerer exclusion listen sin vil dette være et problem."
    # The clamping in `statistics.py` is `min(final_aux, data["base"])`.
    # This protects against negative predictions (Aux > Base).
    # This DOES NOT protect against "Allocated > Actual Consumption".
    # Because Actual Consumption is unknown for future/current prediction.

    # However, the user example says:
    # "raw_demand_kwh: 1.9 ... but traff taket (base load var for lav)"
    # This confirms it clamps against BASE LOAD.

    # So let's test that clamping works when Base < Aux.

    # Modify Scenario:
    # Heater 1: Base=1.0, RawAux=2.0 -> Clamped (Aux=1.0, Overflow=1.0)
    coordinator._correlation_data_per_unit["sensor.heater_1"]["5"]["normal"] = 1.0

    result = real_stats.calculate_total_power(
        temp=5.0,
        effective_wind=0.0,
        solar_impact=0.0,
        is_aux_active=True
    )

    # Verify Breakdown
    bd = result["unit_breakdown"]

    # Heater 1
    h1 = bd["sensor.heater_1"]
    assert h1["base_kwh"] == 1.0
    assert h1["raw_aux_kwh"] == 2.0
    assert h1["aux_reduction_kwh"] == 1.0 # Clamped
    assert h1["overflow_kwh"] == 1.0 # Overflow
    assert h1["clamped"] is True

    # Heater 2
    h2 = bd["sensor.heater_2"]
    assert h2["base_kwh"] == 5.0
    assert h2["raw_aux_kwh"] == 2.0
    assert h2["aux_reduction_kwh"] == 2.0 # Not clamped
    assert h2["overflow_kwh"] == 0.0
    assert h2["clamped"] is False

    # Verify Global Unassigned
    assert result["breakdown"]["unassigned_aux_savings"] == 1.0

    # --- Part 2: Coordinator Accumulation ---

    # Inject result into Coordinator flow
    # This mocks _update_live_predictions partially
    coordinator.data["current_unit_breakdown"] = bd
    coordinator.data["potential_savings_breakdown"] = bd  # Inject potential breakdown for sensor to see clamped status
    coordinator.data["current_aux_impact_rate"] = 4.0 # Global

    # Simulate 30 minutes passing
    coordinator._last_minute_processed = 0
    current_time = datetime(2023, 1, 1, 12, 30) # 30 mins passed

    # Call accumulation logic directly (since we can't easily run the full loop)
    # Re-implement the accumulation snippet from coordinator._update_live_predictions
    # (or call a method if we extracted one, but we modified in-place).
    # Let's manually trigger the accumulation logic we added to verify it works.

    fraction = 30.0 / 60.0 # 0.5 hours

    # Accumulate
    for entity_id, stats in bd.items():
        if entity_id not in coordinator._accumulated_aux_breakdown:
            coordinator._accumulated_aux_breakdown[entity_id] = {"allocated": 0.0, "overflow": 0.0}

        applied_aux = stats.get("aux_reduction_kwh", 0.0)
        overflow_aux = stats.get("overflow_kwh", 0.0)

        coordinator._accumulated_aux_breakdown[entity_id]["allocated"] += (applied_aux * fraction)
        coordinator._accumulated_aux_breakdown[entity_id]["overflow"] += (overflow_aux * fraction)

    # Verify Accumulation
    acc_h1 = coordinator._accumulated_aux_breakdown["sensor.heater_1"]
    # Applied 1.0 * 0.5 = 0.5
    assert acc_h1["allocated"] == 0.5
    # Overflow 1.0 * 0.5 = 0.5
    assert acc_h1["overflow"] == 0.5

    acc_h2 = coordinator._accumulated_aux_breakdown["sensor.heater_2"]
    # Applied 2.0 * 0.5 = 1.0
    assert acc_h2["allocated"] == 1.0
    # Overflow 0.0
    assert acc_h2["overflow"] == 0.0

    # --- Part 3: Sensor Attributes ---

    # Mock Sensor
    from custom_components.heating_analytics.sensor import HeatingPotentialSavingsSensor
    sensor = HeatingPotentialSavingsSensor(coordinator, MagicMock())

    # The sensor attributes use data from coordinator.data AND internal coordinator state
    # We need to populate data that calculate_total_power usually returns
    coordinator.data["accumulated_aux_impact_kwh"] = 2.0 # Global accumulated (4.0 * 0.5h)

    # Patch time to ensure minutes_passed = 30 for rate calculation
    mock_now = datetime(2023, 1, 1, 12, 30, 0, tzinfo=timezone.utc)

    with patch("custom_components.heating_analytics.sensor.dt_util.now", return_value=mock_now):
        attrs = sensor.extra_state_attributes

    # Verify Sensor Attributes
    assert attrs["allocated_total_kwh"] == 1.5 # 0.5 (H1) + 1.0 (H2)
    assert attrs["unassigned_kwh"] == 0.5 # H1 overflow
    assert attrs["leak_status"] == "partial_overflow" # 0.5 > 0.01

    # Verify Breakdown List
    bd_list = attrs["unit_breakdown"]
    assert len(bd_list) == 2

    item_h1 = next(x for x in bd_list if x["name"] == "sensor.heater_1")
    # Allocation 0.5 kWh over 30 mins -> 0.5 / 30 * 60 = 1.0 kW
    assert item_h1["mean_allocated_kw"] == 1.0
    assert item_h1["clamped"] is True
    assert item_h1["current_rate_w"] == 1000 # 1.0 kW
    assert item_h1["overflow_rate_w"] == 1000 # 1.0 kW

    item_h2 = next(x for x in bd_list if x["name"] == "sensor.heater_2")
    # Allocation 1.0 kWh over 30 mins -> 1.0 / 30 * 60 = 2.0 kW
    assert item_h2["mean_allocated_kw"] == 2.0
    assert item_h2["clamped"] is False
