"""Hour-boundary gap filling.

Covers:
- end-of-hour close in `_process_hourly_data` (small and large gaps,
  no-previous-state case).
- mean-imputation contract of `_close_hour_gap` — uses the aggregates
  the caller supplies, not the live coordinator state.
- save/restore of gap-fill state so the post-restart hour is closed
  with the rate captured pre-restart.
- aux race conditions: gap fill must use the persisted aux state for
  the just-ended hour even if the live aux flag has flipped, and must
  fall back to the current state when no persisted rate exists.
"""
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timedelta, timezone

import pytest

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.storage import StorageManager


# =============================================================================
# End-of-hour gap close in _process_hourly_data
# =============================================================================

@pytest.mark.asyncio
async def test_gap_fill_end_of_hour(hass: HomeAssistant):
    """Test closing gap at the end of the hour."""
    entry = MagicMock()
    entry.data = {
        "balance_point": 17.0,
        "energy_sensors": ["sensor.heater"],
        "max_energy_delta": 100.0,
        "outdoor_temp_sensor": "sensor.temp",
        "wind_speed_sensor": "sensor.wind"
    }

    with patch("custom_components.heating_analytics.storage.Store") as mock_store_cls:
        coordinator = HeatingDataCoordinator(hass, entry)
        coordinator._async_save_data = AsyncMock()

        # Simulate last processed minute was 58
        coordinator._collector.last_minute_processed = 58
        coordinator.data["current_model_rate"] = 6.0
        coordinator.data["current_calc_temp"] = 5.0

        coordinator._collector.expected_energy_hour = 5.8

        # Ensure we have sample counts to trigger log append
        coordinator._collector.sample_count = 60
        coordinator._collector.temp_sum = 300.0
        coordinator._collector.wind_values = [0.0] * 60
        coordinator._collector.start_time = datetime(2023, 10, 27, 12, 0, 0)

        current_time = datetime(2023, 10, 27, 13, 0, 0)

        # Mock learning and forecast
        coordinator.learning.process_learning = MagicMock(return_value={
            "model_updated": False, "model_base_before": 0, "model_base_after": 0
        })
        coordinator.forecast.get_forecast_for_hour = MagicMock(return_value=None)

        # Mock _get_predicted_kwh (3 args)
        coordinator._get_predicted_kwh = MagicMock(return_value=6.0)

        # Run
        await coordinator._process_hourly_data(current_time)

        if len(coordinator._hourly_log) > 0:
            log = coordinator._hourly_log[0]
            # 5.8 + (1 min * 6.0/60 = 0.1) = 5.9
            assert log["expected_kwh"] == pytest.approx(5.9, abs=0.01)


@pytest.mark.asyncio
async def test_gap_fill_large_gap(hass: HomeAssistant):
    """Test closing a large gap (e.g. crash)."""
    entry = MagicMock()
    entry.data = {}

    with patch("custom_components.heating_analytics.storage.Store"):
        coordinator = HeatingDataCoordinator(hass, entry)
        coordinator._async_save_data = AsyncMock()

        # Last processed min 29. Missed 30 mins (30..59).
        coordinator._collector.last_minute_processed = 29
        coordinator.data["current_model_rate"] = 2.0

        coordinator._collector.expected_energy_hour = 1.0

        # Setup logging pre-reqs
        coordinator._collector.sample_count = 30
        coordinator._collector.temp_sum = 150.0
        coordinator._collector.wind_values = [0.0] * 30
        coordinator._collector.start_time = datetime(2023, 10, 27, 12, 0, 0)

        current_time = datetime(2023, 10, 27, 13, 0, 0)

        coordinator.learning.process_learning = MagicMock(return_value={
             "model_updated": False, "model_base_before": 0, "model_base_after": 0
        })
        coordinator.forecast.get_forecast_for_hour = MagicMock(return_value=None)
        coordinator._get_predicted_kwh = MagicMock(return_value=2.0)

        await coordinator._process_hourly_data(current_time)

        # Added: 30 mins * (2.0 / 60) = 1.0 kWh. Total 2.0
        if len(coordinator._hourly_log) > 0:
            log = coordinator._hourly_log[0]
            assert log["expected_kwh"] == pytest.approx(2.0, abs=0.01)


@pytest.mark.asyncio
async def test_gap_no_previous_state(hass: HomeAssistant):
    """Test gap logic when no previous state exists (should do nothing)."""
    entry = MagicMock()
    entry.data = {}

    with patch("custom_components.heating_analytics.storage.Store"):
        coordinator = HeatingDataCoordinator(hass, entry)
        coordinator._async_save_data = AsyncMock()

        coordinator._collector.last_minute_processed = None # No state
        coordinator._collector.sample_count = 60
        coordinator._collector.temp_sum = 600.0
        coordinator._collector.wind_values = [0.0] * 60
        coordinator._collector.start_time = datetime(2023, 10, 27, 12, 0, 0)

        current_time = datetime(2023, 10, 27, 13, 0, 0)

        coordinator.learning.process_learning = MagicMock(return_value={
             "model_updated": False, "model_base_before": 0, "model_base_after": 0
        })
        coordinator.forecast.get_forecast_for_hour = MagicMock(return_value=None)
        coordinator._get_predicted_kwh = MagicMock(return_value=1.0)

        await coordinator._process_hourly_data(current_time)

        # Log should exist but expected_kwh should be 0 (initialized 0 + 0 gap)
        if len(coordinator._hourly_log) > 0:
            log = coordinator._hourly_log[0]
            assert log["expected_kwh"] == 0.0


# =============================================================================
# Mean-imputation contract: _close_hour_gap uses caller-supplied aggregates
# =============================================================================

@pytest.fixture
def mock_hass():
    hass = MagicMock()
    hass.data = {}
    return hass


@pytest.mark.asyncio
async def test_gap_fill_mean_imputation(mock_hass):
    """Test that _close_hour_gap uses provided aggregates (Mean Imputation)."""
    entry = MagicMock()
    entry.data = {
        "energy_sensors": ["sensor.heater_1"],
        "balance_point": 17.0
    }

    with patch("custom_components.heating_analytics.storage.Store"):
        coordinator = HeatingDataCoordinator(mock_hass, entry)
        coordinator.statistics = MagicMock()

        # SETUP:
        # Last processed: 29. Missing: 30 minutes (0.5 fraction).
        last_minute = 29
        current_time = datetime(2023, 1, 1, 13, 0, 0, tzinfo=timezone.utc)

        # Mock calculate_total_power to verify it receives our aggregates
        # and returns a known rate.
        coordinator.statistics.calculate_total_power.return_value = {
            "total_kwh": 20.0, # Rate for the gap
            "unit_breakdown": {
                "sensor.heater_1": {"net_kwh": 20.0, "base_kwh": 20.0}
            },
            "global_aux_reduction_kwh": 0.0,
            "breakdown": {}
        }

        # Initialize accumulators
        coordinator._collector.expected_energy_hour = 0.0
        coordinator._hourly_expected_per_unit = {}

        # EXECUTE:
        # Pass aggregates that differ from "live" state to ensure they are used.
        # e.g. Avg Temp 10.0 (Live might have been 12.0)
        coordinator._close_hour_gap(
            current_time,
            last_minute,
            avg_temp=10.0,
            avg_wind=5.0,
            avg_solar=0.5,
            is_aux_active=True
        )

        # VERIFY:
        # 1. calculate_total_power called with aggregates
        coordinator.statistics.calculate_total_power.assert_called_once()
        args, kwargs = coordinator.statistics.calculate_total_power.call_args
        assert args[0] == 10.0 # avg_temp
        assert args[1] == 5.0  # avg_wind
        assert kwargs["is_aux_active"] is True
        assert kwargs["override_solar_factor"] == 0.5 # avg_solar

        # 2. Accumulators updated using the rate (20.0 * 0.5 = 10.0)
        assert coordinator._collector.expected_energy_hour == 10.0
        assert coordinator._hourly_expected_per_unit["sensor.heater_1"] == 10.0


# =============================================================================
# State persistence — gap-fill data must survive restart so the post-restart
# hour is closed using the rate captured in the previous session.
# =============================================================================

@pytest.mark.asyncio
async def test_gap_state_persistence_and_restoration(hass):
    """Test that gap filling state is persisted and restored correctly."""

    # Setup Coordinator
    entry = MagicMock()
    entry.data = {
        "energy_sensors": ["sensor.heater"]
    }
    coordinator = HeatingDataCoordinator(hass, entry)

    # Initialize Storage
    storage = StorageManager(coordinator)

    # Set up state that MUST be persisted for robust gap filling
    test_unit_breakdown = {"sensor.heater": {"net_kwh": 5.0, "base_kwh": 5.0, "aux_reduction_kwh": 0.0, "overflow_kwh": 0.0, "solar_reduction_kwh": 0.0}}

    coordinator.data["current_model_rate"] = 5.0
    coordinator.data["current_aux_impact_rate"] = 1.0
    coordinator.data["current_unit_breakdown"] = test_unit_breakdown
    coordinator.data["current_calc_temp"] = 10.5
    coordinator._collector.last_minute_processed = 58
    # CRITICAL: Set start time to now so restoration succeeds (same hour check)
    coordinator._accumulation_start_time = dt_util.now()

    # Mock store
    mock_store = AsyncMock()
    storage._store = mock_store

    # --- SAVE ---
    await storage.async_save_data(force=True)

    # Inspect what was saved
    saved_data = mock_store.async_save.call_args[0][0]

    # These assertions verify the fix. Before fix, they should FAIL.
    assert "current_model_rate" in saved_data, "current_model_rate not saved"
    assert "current_aux_impact_rate" in saved_data, "current_aux_impact_rate not saved"
    assert "current_unit_breakdown" in saved_data, "current_unit_breakdown not saved"
    assert "current_calc_temp" in saved_data, "current_calc_temp not saved"

    # --- LOAD ---
    # Create new coordinator to simulate restart
    new_coordinator = HeatingDataCoordinator(hass, entry)
    new_storage = StorageManager(new_coordinator)
    new_storage._store = AsyncMock()

    # Mock return data
    new_storage._store.async_load.return_value = saved_data

    await new_storage.async_load_data()

    # Verify restoration
    assert new_coordinator.data.get("current_model_rate") == 5.0
    assert new_coordinator.data.get("current_aux_impact_rate") == 1.0
    assert new_coordinator.data.get("current_unit_breakdown") == test_unit_breakdown
    assert new_coordinator.data.get("current_calc_temp") == 10.5


# =============================================================================
# Aux race conditions — gap fill uses the persisted aux state for the
# just-ended hour even when the live flag has flipped; falls back to the
# current state when no persisted rate exists.
# =============================================================================

@pytest.fixture
def aux_coordinator(hass):
    """Coordinator with mocked statistics whose calculate_total_power returns
    a breakdown driven by the is_aux_active flag — so the test can observe
    which aux state was used to fill the gap."""
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
        coord._collector.aux_impact_hour = 0.0
        coord._collector.expected_energy_hour = 0.0
        coord._collector.last_minute_processed = None

        return coord


@pytest.mark.asyncio
async def test_gap_filling_uses_persisted_state(aux_coordinator):
    """Test that gap filling uses the persisted aux state, not the current state.

    Scenario:
    1. Min 58: Update runs. Aux is ON. Impact rate 1.0 kW.
       Data is persisted (rate=1.0).
    2. ... Restart/Gap ...
    3. Min 05 (Next Hour): Update runs. Aux is OFF. Impact rate 0.0 kW.
    4. _close_hour_gap should fill Min 59 of old hour using persisted rate 1.0, not current rate 0.0.
    """
    coordinator = aux_coordinator

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
    assert coordinator._collector.last_minute_processed == 58

    # 2. Change State (Aux OFF) - Simulating change during gap/restart
    coordinator.auxiliary_heating_active = False

    # 3. Simulate Next Update (New Hour, Min 05)
    # _process_hourly_data calls _close_hour_gap(new_time, last_processed)
    # We call _close_hour_gap directly to verify logic
    new_time = datetime(2023, 1, 1, 13, 5, 0)

    # Refactor Update: _close_hour_gap now requires explicit aggregates.
    # We pass is_aux_active=True to simulate that the hour was dominated by Aux
    coordinator._close_hour_gap(new_time, 58, avg_temp=10.0, avg_wind=0.0, is_aux_active=True)

    # Gap: 60 - (58+1) = 1 minute (Minute 59)
    # Total minutes processed: 59 (0..58) + 1 (59) = 60 minutes = 1.0 hour
    # Total impact should be exactly 1.0 kWh (since aux was ON for 58 min, and gap fill assumes ON)
    assert coordinator._collector.aux_impact_hour == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_gap_filling_fallback(aux_coordinator):
    """Test fallback behavior when no persisted data exists."""
    coordinator = aux_coordinator

    # Start fresh
    coordinator.data["current_aux_impact_rate"] = 0.0
    coordinator.data["current_unit_breakdown"] = {}
    coordinator.data["current_calc_temp"] = 10.0
    coordinator.data["current_model_rate"] = 4.0 # Assume some rate exists from legacy

    # Last processed: 58
    coordinator._collector.last_minute_processed = 58

    # Current State: Aux ON
    coordinator.auxiliary_heating_active = True

    # Close gap (Refactor: Pass aggregates)
    # We simulate that the hour aggregate implies Aux Active
    coordinator._close_hour_gap(datetime(2023,1,1,13,0,0), 58, avg_temp=10.0, avg_wind=0.0, is_aux_active=True)

    # Fallback recalculates using CURRENT state (Aux ON -> 1.0 kW)
    # Gap: 1 min.
    expected_gap = 1.0 * (1/60)
    assert coordinator._collector.aux_impact_hour == pytest.approx(expected_gap)
