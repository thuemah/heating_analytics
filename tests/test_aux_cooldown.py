"""Tests for auxiliary cooldown logic."""
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timedelta
import pytest
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.const import (
    MODE_HEATING,
    COOLDOWN_MIN_HOURS,
    COOLDOWN_MAX_HOURS,
    COOLDOWN_CONVERGENCE_THRESHOLD
)

@pytest.fixture
def coordinator(hass):
    """Fixture to create a coordinator with mocked dependencies."""
    entry = MagicMock()
    entry.data = {
        "balance_point": 17.0,
        "learning_rate": 0.1,
        "energy_sensors": ["sensor.heater", "sensor.heater_2"],
        "aux_affected_entities": ["sensor.heater"] # Only heater 1 is affected
    }

    with patch("custom_components.heating_analytics.storage.Store"), \
         patch("custom_components.heating_analytics.coordinator.SolarCalculator"), \
         patch("custom_components.heating_analytics.coordinator.ForecastManager") as MockForecast, \
         patch("custom_components.heating_analytics.coordinator.StatisticsManager"), \
         patch("custom_components.heating_analytics.coordinator.LearningManager") as MockLearning, \
         patch("custom_components.heating_analytics.coordinator.StorageManager") as MockStorage:

        coord = HeatingDataCoordinator(hass, entry)

        # Mock Learning Manager behavior
        coord.learning.process_learning.return_value = {
            "learning_status": "ok",
            "model_updated": False,
            "model_base_before": 5.0,
            "model_base_after": 5.0
        }

        # Mock Forecast Manager
        coord.forecast.get_forecast_for_hour.return_value = None

        # Mock Storage Manager (Async)
        coord.storage.async_save_data = AsyncMock()
        coord.storage.append_hourly_log_csv = AsyncMock()

        # Mock DataUpdateCoordinator method to avoid base class errors
        coord.async_set_updated_data = MagicMock()

        # Initialize trackers
        coord._hourly_delta_per_unit = {}
        coord._hourly_expected_base_per_unit = {}

        # Ensure sample count > 0 so learning triggers
        coord._hourly_sample_count = 60
        coord._hourly_wind_values = [0.0] * 60

        return coord

@pytest.mark.asyncio
async def test_aux_cooldown_transition(coordinator):
    """Test that cooldown is triggered when Aux turns off."""

    # 1. Start with Aux ON
    coordinator.auxiliary_heating_active = True
    assert coordinator._aux_cooldown_active is False

    # 2. Turn Aux OFF
    await coordinator.set_auxiliary_heating_active(False)

    # 3. Verify Cooldown Started
    assert coordinator.auxiliary_heating_active is False
    assert coordinator._aux_cooldown_active is True
    assert coordinator._aux_cooldown_start_time is not None

@pytest.mark.asyncio
async def test_aux_cooldown_lock_duration(coordinator):
    """Test that cooldown remains active within min duration."""

    # Setup Cooldown
    start_time = datetime(2023, 1, 1, 12, 0, 0)
    coordinator._aux_cooldown_active = True
    coordinator._aux_cooldown_start_time = start_time
    coordinator.auxiliary_heating_active = False

    # Simulate Hour 1 (Elapsed = 1h < MIN)
    current_time = start_time + timedelta(hours=1)

    # Run processing
    await coordinator._process_hourly_data(current_time)

    # Verify Cooldown STILL Active
    assert coordinator._aux_cooldown_active is True

    # Verify Learning Called with is_cooldown_active=True
    call_args = coordinator.learning.process_learning.call_args
    assert call_args.kwargs["is_cooldown_active"] is True

@pytest.mark.asyncio
async def test_aux_cooldown_convergence_failure(coordinator):
    """Test that cooldown remains active if convergence not reached (after min duration)."""

    # Setup Cooldown
    start_time = datetime(2023, 1, 1, 12, 0, 0)
    coordinator._aux_cooldown_active = True
    coordinator._aux_cooldown_start_time = start_time

    # Simulate Hour 3 (Elapsed = 3h > MIN)
    current_time = start_time + timedelta(hours=3)

    # Setup Divergence (Actual < Expected)
    # Target Threshold is 0.95
    # Actual=8, Expected=10 -> Ratio=0.8
    coordinator._hourly_delta_per_unit = {"sensor.heater": 8.0}
    coordinator._hourly_expected_base_per_unit = {"sensor.heater": 10.0}

    await coordinator._process_hourly_data(current_time)

    # Verify Cooldown STILL Active
    assert coordinator._aux_cooldown_active is True

@pytest.mark.asyncio
async def test_aux_cooldown_convergence_success(coordinator):
    """Test that cooldown exits if convergence reached."""

    # Setup Cooldown
    start_time = datetime(2023, 1, 1, 12, 0, 0)
    coordinator._aux_cooldown_active = True
    coordinator._aux_cooldown_start_time = start_time

    # Simulate Hour 3
    current_time = start_time + timedelta(hours=3)

    # Setup Convergence (Actual >= 0.95 * Expected)
    # Actual=9.6, Expected=10.0 -> Ratio=0.96
    coordinator._hourly_delta_per_unit = {"sensor.heater": 9.6}
    coordinator._hourly_expected_base_per_unit = {"sensor.heater": 10.0}

    await coordinator._process_hourly_data(current_time)

    # Verify Cooldown EXITED
    assert coordinator._aux_cooldown_active is False
    assert coordinator._aux_cooldown_start_time is None

    # Verify Learning Called with is_cooldown_active=True
    # Because the convergent hour itself should still be protected (cooldown active during the hour)
    call_args = coordinator.learning.process_learning.call_args
    assert call_args.kwargs["is_cooldown_active"] is True

@pytest.mark.asyncio
async def test_aux_cooldown_max_timeout(coordinator):
    """Test that cooldown exits after max duration."""

    # Setup Cooldown
    start_time = datetime(2023, 1, 1, 12, 0, 0)
    coordinator._aux_cooldown_active = True
    coordinator._aux_cooldown_start_time = start_time

    # Simulate Hour 7 (Elapsed = 7h > MAX 6h)
    current_time = start_time + timedelta(hours=7)

    # Setup Divergence (Should be ignored due to timeout)
    coordinator._hourly_delta_per_unit = {"sensor.heater": 5.0}
    coordinator._hourly_expected_base_per_unit = {"sensor.heater": 10.0}

    await coordinator._process_hourly_data(current_time)

    # Verify Cooldown EXITED
    assert coordinator._aux_cooldown_active is False

@pytest.mark.asyncio
async def test_aux_reactivation_during_cooldown(coordinator):
    """Test that aux reactivation cancels cooldown state."""

    # 1. Start with Cooldown Active
    coordinator._aux_cooldown_active = True
    coordinator._aux_cooldown_start_time = datetime.now()
    coordinator.auxiliary_heating_active = False

    # 2. Turn Aux ON (Re-activation)
    await coordinator.set_auxiliary_heating_active(True)

    # 3. Verify Cooldown Cancelled
    assert coordinator.auxiliary_heating_active is True
    assert coordinator._aux_cooldown_active is False
    assert coordinator._aux_cooldown_start_time is None

@pytest.mark.asyncio
async def test_cooldown_default_all_affected(hass):
    """Test that null aux_affected_entities defaults to all sensors during cooldown."""

    # Custom coordinator setup for this test
    entry = MagicMock()
    # Explicitly set aux_affected_entities to None (the key missing implies default)
    # But here we set it to None explicitly to verify logic
    entry.data = {
        "balance_point": 17.0,
        "learning_rate": 0.1,
        "energy_sensors": ["sensor.heater", "sensor.heater_2"],
        "aux_affected_entities": None
    }

    with patch("custom_components.heating_analytics.storage.Store"), \
         patch("custom_components.heating_analytics.coordinator.SolarCalculator"), \
         patch("custom_components.heating_analytics.coordinator.ForecastManager"), \
         patch("custom_components.heating_analytics.coordinator.StatisticsManager"), \
         patch("custom_components.heating_analytics.coordinator.LearningManager"), \
         patch("custom_components.heating_analytics.coordinator.StorageManager"):

        coord = HeatingDataCoordinator(hass, entry)

        # Configure Mocks
        coord.learning.process_learning.return_value = {
            "learning_status": "ok",
            "model_updated": False,
            "model_base_before": 5.0,
            "model_base_after": 5.0
        }
        coord.forecast.get_forecast_for_hour.return_value = None
        coord.storage.async_save_data = AsyncMock()
        coord.storage.append_hourly_log_csv = AsyncMock()
        coord.async_set_updated_data = MagicMock()

        # Mock forecast item processing to return a valid tuple
        coord.forecast._process_forecast_item.return_value = (0.0, 0.0, {}, 0.0, {}, 0.0, 0.0, 0.0)

        # Verify default behavior in init
        # Note: In init it does: self.aux_affected_entities = self.energy_sensors if None
        assert coord.aux_affected_entities == ["sensor.heater", "sensor.heater_2"]

        # Setup Cooldown
        start_time = datetime(2023, 1, 1, 12, 0, 0)
        coord._aux_cooldown_active = True
        coord._aux_cooldown_start_time = start_time

        # Simulate Hour 3 (Check Convergence)
        current_time = start_time + timedelta(hours=3)

        # Setup stats so that ONLY the sum of BOTH units converges
        # Heater 1: 5.0 / 10.0 = 0.5 (Not convergent)
        # Heater 2: 4.6 / 0.0 (Wait, expected sum needs to be tracked)

        # Scenario:
        # Heater 1: Expected 10, Actual 5 (50%)
        # Heater 2: Expected 10, Actual 14 (140%)
        # Total: Expected 20, Actual 19 (95%) -> Convergent!

        coord._hourly_expected_base_per_unit = {
            "sensor.heater": 10.0,
            "sensor.heater_2": 10.0
        }
        coord._hourly_delta_per_unit = {
            "sensor.heater": 5.0,
            "sensor.heater_2": 14.0
        }

        # Ensure sample count > 0 to run logic
        coord._hourly_sample_count = 60
        coord._hourly_wind_values = [0.0] * 60

        # Mock forecast item processing to return a valid tuple
        # val, solar, breakdown, gross, gross_breakdown, _, _
        coord.forecast._process_forecast_item.return_value = (0.0, 0.0, {}, 0.0, {}, 0.0, 0.0, 0.0)

        await coord._process_hourly_data(current_time)

        # If BOTH units were considered, it should have converged
        assert coord._aux_cooldown_active is False


@pytest.mark.asyncio
async def test_cooldown_non_affected_learning(coordinator):
    """Test that process_learning is called with is_cooldown_active=True."""
    # This verifies the coordinator passes the flag correctly.
    # The actual selective learning logic is unit tested in test_learning_direct.py

    # Setup Cooldown
    coordinator._aux_cooldown_active = True
    coordinator._aux_cooldown_start_time = datetime(2023, 1, 1, 12, 0, 0)

    current_time = datetime(2023, 1, 1, 13, 0, 0)

    await coordinator._process_hourly_data(current_time)

    # Verify Learning Manager called correctly
    call_args = coordinator.learning.process_learning.call_args
    assert call_args is not None
    assert call_args.kwargs["is_cooldown_active"] is True
    # Verify aux_affected_entities is passed
    assert call_args.kwargs["aux_affected_entities"] == ["sensor.heater"]

@pytest.mark.asyncio
async def test_manual_exit_cooldown(coordinator):
    """Test manual exit of cooldown via service call logic."""

    # 1. Start with Cooldown Active
    coordinator._aux_cooldown_active = True
    coordinator._aux_cooldown_start_time = datetime(2023, 1, 1, 12, 0, 0)

    # 2. Call async_exit_cooldown
    await coordinator.async_exit_cooldown()

    # 3. Verify Cooldown Exited
    assert coordinator._aux_cooldown_active is False
    assert coordinator._aux_cooldown_start_time is None

    # Verify persistence was saved
    coordinator.storage.async_save_data.assert_awaited()

    # Verify UI update triggered
    coordinator.async_set_updated_data.assert_called()
