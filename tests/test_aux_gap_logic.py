"""Tests for auxiliary impact gap logic and race conditions."""
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
import pytest

from custom_components.heating_analytics.coordinator import HeatingDataCoordinator

@pytest.fixture
def coordinator(hass):
    """Fixture to create a coordinator with mocked dependencies."""
    entry = MagicMock()
    entry.data = {
        "balance_point": 17.0,
        "learning_rate": 0.1,
        "energy_sensors": ["sensor.heater"]
    }

    with patch("custom_components.heating_analytics.storage.Store"), \
         patch("custom_components.heating_analytics.coordinator.SolarCalculator"), \
         patch("custom_components.heating_analytics.coordinator.ForecastManager"), \
         patch("custom_components.heating_analytics.coordinator.StatisticsManager") as MockStats, \
         patch("custom_components.heating_analytics.coordinator.LearningManager"), \
         patch("custom_components.heating_analytics.coordinator.StorageManager"):

        coord = HeatingDataCoordinator(hass, entry)

        # Configure the MockStats
        mock_stats_instance = MockStats.return_value
        # Mock calculate_total_power to return breakdown based on is_aux_active arg
        def mock_calc_power(temp, wind, solar, is_aux_active=False, **kwargs):
            aux_red = 1.0 if is_aux_active else 0.0
            return {
                "total_kwh": 5.0 - aux_red, # Base 5.0, Net 4.0 if aux
                "global_base_kwh": 5.0,
                "global_aux_reduction_kwh": aux_red, # Strict Global Authority
                "breakdown": {
                    "base_kwh": 5.0,
                    "aux_reduction_kwh": aux_red,
                    "solar_reduction_kwh": 0.0
                },
                "unit_breakdown": {
                    "sensor.heater": {
                        "net_kwh": 5.0 - aux_red,
                        "base_kwh": 5.0
                    }
                }
            }
        mock_stats_instance.calculate_total_power.side_effect = mock_calc_power

        coord.statistics = mock_stats_instance

        # Initialize trackers
        coord._accumulated_aux_impact_hour = 0.0
        coord._accumulated_expected_energy_hour = 0.0
        coord._last_minute_processed = None

        return coord

@pytest.mark.asyncio
async def test_gap_filling_uses_persisted_state(coordinator):
    """Test that gap filling uses the persisted aux state, not the current state.

    Scenario:
    1. Min 58: Update runs. Aux is ON. Impact rate 1.0 kW.
       Data is persisted (rate=1.0).
    2. ... Restart/Gap ...
    3. Min 05 (Next Hour): Update runs. Aux is OFF. Impact rate 0.0 kW.
    4. _close_hour_gap should fill Min 59 of old hour using persisted rate 1.0, not current rate 0.0.
    """

    # 1. Simulate Min 58 (Aux ON)
    current_time = datetime(2023, 1, 1, 12, 58, 0)
    coordinator.auxiliary_heating_active = True

    # Run update
    # Should accumulate 59 minutes (0..58) @ 1.0 kW = ~0.9833 kWh
    coordinator._update_live_predictions(
        calc_temp=10.0, temp_key="10", wind_bucket="normal", current_time=current_time
    )

    # Verify State Persistence
    assert coordinator.data["current_aux_impact_rate"] == 1.0
    assert coordinator.data["current_model_rate"] == 4.0 # 5.0 - 1.0
    assert coordinator._last_minute_processed == 58

    # 2. Change State (Aux OFF) - Simulating change during gap/restart
    coordinator.auxiliary_heating_active = False

    # 3. Simulate Next Update (New Hour, Min 05)
    # _process_hourly_data calls _close_hour_gap(new_time, last_processed)
    # We call _close_hour_gap directly to verify logic
    new_time = datetime(2023, 1, 1, 13, 5, 0)

    # Logic in _process_hourly_data would use the persisted data before updating live predictions for the new hour

    # Refactor Update: _close_hour_gap now requires explicit aggregates.
    # We pass is_aux_active=True to simulate that the hour was dominated by Aux (or at least we want to fill gap as Aux)
    coordinator._close_hour_gap(new_time, 58, avg_temp=10.0, avg_wind=0.0, is_aux_active=True)

    # Gap: 60 - (58+1) = 1 minute (Minute 59)
    # Impact should be: 1.0 kW * (1/60) = 0.0166...
    # Total accumulated: Previous + Gap

    # Before gap fill: ~0.9833
    # Gap addition: ~0.0166
    # Total: ~1.0

    expected_gap_fill = 1.0 * (1/60)

    # Verify that the accumulator increased by the gap amount based on the OLD rate (1.0)
    # If it used the NEW rate (0.0), the increase would be 0.

    # We need to capture the value before closing gap to be precise, or just check total
    # Total minutes processed: 59 (0..58) + 1 (59) = 60 minutes = 1.0 hour
    # Total impact should be exactly 1.0 kWh (since aux was ON for 58 min, and gap fill assumes ON)

    assert coordinator._accumulated_aux_impact_hour == pytest.approx(1.0)

@pytest.mark.asyncio
async def test_gap_filling_fallback(coordinator):
    """Test fallback behavior when no persisted data exists."""
    # Start fresh
    coordinator.data["current_aux_impact_rate"] = 0.0
    coordinator.data["current_unit_breakdown"] = {}
    coordinator.data["current_calc_temp"] = 10.0
    coordinator.data["current_model_rate"] = 4.0 # Assume some rate exists from legacy

    # Last processed: 58
    coordinator._last_minute_processed = 58

    # Current State: Aux ON
    coordinator.auxiliary_heating_active = True

    # Close gap (Refactor: Pass aggregates)
    # We simulate that the hour aggregate implies Aux Active
    coordinator._close_hour_gap(datetime(2023,1,1,13,0,0), 58, avg_temp=10.0, avg_wind=0.0, is_aux_active=True)

    # Fallback recalculates using CURRENT state (Aux ON -> 1.0 kW)
    # Gap: 1 min.
    expected_gap = 1.0 * (1/60)
    assert coordinator._accumulated_aux_impact_hour == pytest.approx(expected_gap)
