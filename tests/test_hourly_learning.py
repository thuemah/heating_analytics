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
        coordinator._hourly_sample_count = 60
        coordinator._hourly_temp_sum = 0.0 # Avg 0.0
        coordinator._hourly_wind_values = [0.0] * 60
        coordinator._hourly_bucket_counts = {"normal": 60, "high_wind": 0, "extreme_wind": 0}
        coordinator._accumulated_energy_hour = 2.0 # Actual was 2.0 (High!)

        # FIX: Set expected energy explicitly to match the scenario
        # In real operation, this accumulates minute-by-minute.
        coordinator._accumulated_expected_energy_hour = 1.0

        coordinator._hourly_delta_per_unit = {"sensor.heater": 2.0}

        current_time = datetime(2023, 10, 27, 13, 0, 0) # End of hour (processing 12:00-13:00)
        coordinator._hourly_start_time = datetime(2023, 10, 27, 12, 0, 0)

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

@pytest.mark.asyncio
async def test_hourly_learning_with_solar(hass: HomeAssistant):
    """Test hourly learning with solar normalization."""
    entry = MagicMock()
    entry.data = {
        "balance_point": 17.0,
        "learning_rate": 0.1,
        "energy_sensors": ["sensor.heater"],
        "solar_enabled": True,
        "solar_window_area": 10.0
    }

    with patch("custom_components.heating_analytics.storage.Store") as mock_store_cls:
        coordinator = HeatingDataCoordinator(hass, entry)
        coordinator._async_save_data = AsyncMock()
        coordinator.storage.async_save_data = AsyncMock()

        # Initial State: Temp 0 -> Expected Base 2.0
        coordinator._correlation_data = {"0": {"normal": 2.0}}
        # FIX: Set per-unit data
        coordinator._correlation_data_per_unit = {"sensor.heater": {"0": {"normal": 2.0}}}

        # Setup Solar Mock
        # Solar Factor 1.0 -> Impact 0.5 kWh
        # We need to mock both Global (for deprecated check) and Unit methods
        coordinator.solar.calculate_solar_impact_kw = MagicMock(return_value=0.5)

        # New logic calculates unit impact during processing.
        # It calls calculate_unit_coefficient and calculate_unit_solar_impact.
        # We need to ensure calculate_unit_solar_impact returns something consistent
        # with the test assumption (Impact 0.5).
        # We can just mock normalize_for_learning to do the math we want.

        # Apply correction: Base - Impact. 2.0 - 0.5 = 1.5 Expected.
        coordinator.solar.apply_correction = MagicMock(side_effect=lambda base, impact, temp: base - impact)

        # Normalize: Actual + Impact.
        # The test expects normalized to be 2.0 (Actual 1.5 + Impact 0.5).
        # In process_learning, solar_impact is passed.
        # In the new logic, unit_solar_impact is calculated.
        # We need to mock normalize_for_learning to handle whatever impact is passed.

        coordinator.solar.normalize_for_learning = MagicMock(side_effect=lambda actual, impact, temp: actual + impact)

        # Mock unit calls to ensure `unit_solar_impact` is also 0.5 if used
        coordinator.solar.calculate_unit_coefficient = MagicMock(return_value=1.0)
        coordinator.solar.calculate_unit_solar_impact = MagicMock(return_value=0.5)

        # Data
        coordinator._hourly_sample_count = 60
        coordinator._hourly_temp_sum = 0.0
        coordinator._hourly_wind_values = [0.0] * 60
        coordinator._hourly_bucket_counts = {"normal": 60}
        coordinator._hourly_solar_sum = 60.0 # Avg 1.0

        # Actual Consumption was 1.5 (Matched expected with solar)
        coordinator._accumulated_energy_hour = 1.5
        coordinator._accumulated_expected_energy_hour = 1.5 # FIX: Set expected
        coordinator._hourly_delta_per_unit = {"sensor.heater": 1.5}

        current_time = datetime(2023, 10, 27, 13, 0, 0)

        await coordinator._process_hourly_data(current_time)

        # Verification
        # Expected (Base): 2.0
        # Expected (Solar): 1.5
        # Actual: 1.5

        # Learning:
        # Normalized Actual = Actual (1.5) + Impact (0.5) = 2.0
        # Base Prediction = 2.0
        # New Base = 2.0 + 0.1 * (2.0 - 2.0) = 2.0 (No Change)

        assert coordinator._correlation_data["0"]["normal"] == 2.0

        # Scenario 2: Actual was higher (1.8) -> Normalized 2.3
        # New Base = 2.0 + 0.1 * (2.3 - 2.0) = 2.03

        # Reset for next run (simulation of next hour data accumulation)
        coordinator._hourly_sample_count = 60
        coordinator._hourly_temp_sum = 0.0
        coordinator._hourly_wind_values = [0.0] * 60
        coordinator._hourly_bucket_counts = {"normal": 60}
        coordinator._hourly_solar_sum = 60.0 # Avg 1.0
        coordinator._accumulated_energy_hour = 1.8
        coordinator._accumulated_expected_energy_hour = 1.5 # Still expected 1.5
        coordinator._hourly_delta_per_unit = {"sensor.heater": 1.8}

        await coordinator._process_hourly_data(current_time)
        assert coordinator._correlation_data["0"]["normal"] == pytest.approx(2.03, abs=0.001)

@pytest.mark.asyncio
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

        coordinator._hourly_sample_count = 60
        coordinator._hourly_temp_sum = 0.0
        coordinator._hourly_wind_values = [0.0] * 60

        # Dominant Aux active
        # Refactor: _hourly_bucket_counts only tracks physical buckets
        coordinator._hourly_bucket_counts = {"normal": 60}
        coordinator.auxiliary_heating_active = True
        coordinator._hourly_aux_count = 60 # Dominant

        # Scenario:
        # Base Model (Normal) = 1.0 kWh
        # Actual Consumption (Aux) = 0.5 kWh
        # Implied Aux Impact = 1.0 - 0.5 = 0.5 kWh
        # Learning Rate = 0.1 (simulated)

        coordinator._accumulated_energy_hour = 0.5
        coordinator._accumulated_expected_energy_hour = 1.0

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
