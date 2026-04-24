"""Test the hourly learning logic."""
from unittest.mock import MagicMock, patch, AsyncMock
import pytest
from datetime import datetime
from homeassistant.core import HomeAssistant
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator

@pytest.mark.asyncio
async def test_hourly_learning_basic(hass: HomeAssistant):
    """Test basic hourly learning without solar."""
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.data = {
        "balance_point": 17.0,
        "outdoor_temp_sensor": "sensor.temp",
        "learning_rate": 0.1,
        "energy_sensors": ["sensor.heater"]
    }

    with patch("custom_components.heating_analytics.storage.Store") as mock_store_cls:
        mock_store = mock_store_cls.return_value
        mock_store.async_load = AsyncMock(return_value={})
        mock_store.async_save = AsyncMock()

        coordinator = HeatingDataCoordinator(hass, entry)
        coordinator._async_save_data = AsyncMock()

        # Initial State: Temp 0 -> Expected 1.0
        coordinator._correlation_data = {"0": {"normal": 1.0}}
        coordinator._correlation_data_per_unit = {"sensor.heater": {"0": {"normal": 1.0}}}

        # Simulate accumulated data for the hour
        coordinator._collector.sample_count = 60
        coordinator._collector.temp_sum = 0.0 # Avg 0.0
        coordinator._collector.wind_values = [0.0] * 60
        coordinator._collector.bucket_counts = {"normal": 60, "high_wind": 0, "extreme_wind": 0}
        coordinator._collector.energy_hour = 2.0 # Actual was 2.0 (High!)

        # FIX: Set expected energy explicitly to match the scenario
        # In real operation, this accumulates minute-by-minute.
        coordinator._collector.expected_energy_hour = 1.0

        coordinator._hourly_delta_per_unit = {"sensor.heater": 2.0}

        current_time = datetime(2023, 10, 27, 13, 0, 0) # End of hour (processing 12:00-13:00)
        coordinator._collector.start_time = datetime(2023, 10, 27, 12, 0, 0)

        # Run
        await coordinator._process_hourly_data(current_time)

        # Verify Learning
        # Old Prediction: 1.0
        # Actual: 2.0
        # Diff: 1.0
        # Global rate: 0.1 -> New global = 1.0 + 0.1 * 1.0 = 1.1
        # Per-unit rate: min(0.1, 0.03) = 0.03 -> New per-unit = 1.0 + 0.03 * 1.0 = 1.03

        assert coordinator._correlation_data["0"]["normal"] == 1.1
        assert coordinator._correlation_data_per_unit["sensor.heater"]["0"]["normal"] == 1.03

        # Verify Log
        assert len(coordinator._hourly_log) == 1
        log = coordinator._hourly_log[0]
        assert log["hour"] == 12
        assert log["actual_kwh"] == 2.0
        assert log["expected_kwh"] == 1.0 # The expectation BEFORE update

async def test_hourly_bucket_auxiliary(hass: HomeAssistant):
    """Test that auxiliary heating updates Aux Coefficient (Refactor Check)."""
    entry = MagicMock()
    entry.data = {"balance_point": 17.0, "learning_rate": 0.1, "energy_sensors": ["sensor.heater"]}

    with patch("custom_components.heating_analytics.storage.Store"):
        coordinator = HeatingDataCoordinator(hass, entry)
        coordinator._async_save_data = AsyncMock()

        # Setup: Normal bucket exists with value 1.0
        coordinator._correlation_data = {"0": {"normal": 1.0}}
        # FIX: Set per-unit data
        coordinator._correlation_data_per_unit = {"sensor.heater": {"0": {"normal": 1.0}}}

        # Initial Aux Coefficient is 0.0 (New Format: Nested dict)
        coordinator._aux_coefficients = {"0": {"normal": 0.0}}
        # FIX: Set per-unit aux
        coordinator._aux_coefficients_per_unit = {"sensor.heater": {"0": {"normal": 0.0}}}

        coordinator._collector.sample_count = 60
        coordinator._collector.temp_sum = 0.0
        coordinator._collector.wind_values = [0.0] * 60

        # Dominant Aux active
        # Refactor: _hourly_bucket_counts only tracks physical buckets
        coordinator._collector.bucket_counts = {"normal": 60}
        coordinator.auxiliary_heating_active = True
        coordinator._collector.aux_count = 60 # Dominant

        # Scenario:
        # Base Model (Normal) = 1.0 kWh
        # Actual Consumption (Aux) = 0.5 kWh
        # Implied Aux Impact = 1.0 - 0.5 = 0.5 kWh
        # Learning Rate = 0.1 (simulated)

        coordinator._collector.energy_hour = 0.5
        coordinator._collector.expected_energy_hour = 1.0

        # FIX: Populate unit deltas so per-unit breakdown is calculated
        coordinator._hourly_delta_per_unit = {"sensor.heater": 0.5}

        current_time = datetime(2023, 10, 27, 13, 0, 0)

        # Verify that we are using the entry-provided learning rate (0.1)
        # The constants DEFAULT_AUX_LEARNING_RATE should no longer be used.
        await coordinator._process_hourly_data(current_time)

        # Should have updated aux coefficient
        # new = 0.0 + 0.1 * (0.5 - 0.0) = 0.05
        assert coordinator._aux_coefficients["0"]["normal"] == 0.05

        # Base model should NOT change
        assert coordinator._correlation_data["0"]["normal"] == 1.0
