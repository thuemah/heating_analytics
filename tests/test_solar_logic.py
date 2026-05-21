"""Solar logic tests — deviation breakdown plus type-tolerant correction
and learning-normalisation paths through SolarCalculator."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime
from custom_components.heating_analytics.solar import SolarCalculator
from custom_components.heating_analytics.const import (
    ATTR_SOLAR_IMPACT,
    MODE_HEATING,
    MODE_COOLING,
)

@pytest.fixture
def calculator(mock_coordinator):
    return SolarCalculator(mock_coordinator)

def test_deviation_breakdown_solar_logic(mock_coordinator):
    """Test that deviation breakdown applies solar correction correctly."""
    coord = mock_coordinator
    coord.energy_sensors = ["sensor.heater1"]
    coord.balance_point = 15.0
    coord.solar_enabled = True
    coord.data[ATTR_SOLAR_IMPACT] = 0.4 # Global Impact
    coord.solar_correction_percent = 100.0

    # Initialize internal structures
    coord._correlation_data = {"10": {"normal": 1.0}} # 1 kWh predicted at 10C
    coord._correlation_data_per_unit = {"sensor.heater1": {"10": {"normal": 1.0}}}
    coord._observation_counts = {"sensor.heater1": {"10": {"normal": 100}}}
    coord._solar_coefficients_per_unit = {
        "sensor.heater1": {
            "heating": {"s": 1.0, "e": 0.0, "w": 0.0},
            "cooling": {"s": 0.0, "e": 0.0, "w": 0.0},
        }
    }

    # Setup context
    now = datetime(2023, 10, 10, 10, 30, 0) # 10:30 AM
    # Patch local dt_util in statistics, as StatisticsManager imports it locally
    with patch("custom_components.heating_analytics.statistics.dt_util.now", return_value=now):
        # 1. Setup Hourly Log (Past hour 09:00-10:00)
        # Temp 10C (Heating), Solar Impact 0.2 kWh
        # Base Prediction 1.0 kWh. Solar reduces it to 0.8 kWh.
        log_entry = {
            "timestamp": "2023-10-10T09:00:00",
            "hour": 9,
            "temp": 10.0,
            "temp_key": "10",
            "wind_bucket": "normal",
            "solar_factor": 0.2,  # Added factor so new logic calculates impact
            "solar_impact_kwh": 0.2, # Global impact (ignored by new logic in favor of recalc)
            "expected_kwh": 0.8,
            "actual_kwh": 0.8,
            "unit_expected_breakdown": {"sensor.heater1": 0.8}
        }
        coord._hourly_log.append(log_entry)

        # 2. Setup Current State (Partial hour 10:00-10:30)
        # Temp 10C, Solar Impact 0.4 kW (instant)
        coord.data["effective_wind"] = 0.0
        coord.data["solar_factor"] = 0.4 # Need factor for new calc
        coord.data["solar_vector_s"] = 0.4
        coord.data["solar_vector_e"] = 0.0
        coord.data["solar_vector_w"] = 0.0

        # New logic uses solar_factor * unit_coeff to get impact.
        # We set coeff=1.0 in setup above.
        # So past hour: factor 0.2 * coeff 1.0 = 0.2 impact. Correct.
        # Current hour: factor 0.4 * coeff 1.0 = 0.4 impact. Correct.

        # Mock inertia temp
        coord._calculate_inertia_temp = MagicMock(return_value=10.0)

        # 3. Execute Breakdown
        from custom_components.heating_analytics.statistics import StatisticsManager
        real_stats = StatisticsManager(coord)
        breakdown = real_stats.calculate_deviation_breakdown()

        unit_stat = breakdown[0]
        assert unit_stat["entity_id"] == "sensor.heater1"
        assert unit_stat["expected"] == 1.1

@pytest.mark.asyncio
async def test_process_hourly_data_solar_logic(mock_coordinator):
    """Test that hourly processing applies solar logic correctly."""
    # Use real coordinator but with mock dependencies
    from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
    
    mock_coordinator.entry.data["solar_enabled"] = True
    mock_coordinator.entry.data["energy_sensors"] = ["sensor.heater1"]
    
    # Patch Store to prevent file I/O
    with patch("custom_components.heating_analytics.storage.Store"):
        coord = HeatingDataCoordinator(mock_coordinator.hass, mock_coordinator.entry)
        
    # Setup context
    # Hour just finished (10:00-11:00)
    current_time = datetime(2023, 10, 10, 11, 0, 0)

    # Aggregates
    coord._collector.sample_count = 60
    coord._collector.temp_sum = 10.0 * 60 # Avg 10C
    coord._collector.wind_values = [0.0] * 60
    coord._collector.solar_sum = 0.5 * 60 # Avg Factor 0.5
    coord._collector.solar_vector_s_sum = 0.5 * 60
    coord._collector.solar_vector_e_sum = 0.0
    coord._collector.solar_vector_w_sum = 0.0
    coord._collector.energy_hour = 0.8 # Actual
    coord._collector.expected_energy_hour = 0.8 # Expected (to match logic)

    # Initialize structures
    coord._correlation_data = {"10": {"normal": 1.0}}
    coord._correlation_data_per_unit = {"sensor.heater1": {"10": {"normal": 1.0}}}
    coord._solar_coefficients_per_unit = {
        "sensor.heater1": {
            "heating": {"s": 0.4, "e": 0.0, "w": 0.0},
            "cooling": {"s": 0.0, "e": 0.0, "w": 0.0},
        }
    }
    coord.solar_battery_decay = 0.50

    # Execute
    await coord._process_hourly_data(current_time)

    # Verify Log Entry
    log_entry = coord._hourly_log[-1]

    # Logic:
    # Temp 10C. Base Expected 1.0.
    # Unit Solar Impact = 0.2.
    # Expected = 1.0 - 0.2 = 0.8.

    assert log_entry["expected_kwh"] == 0.8

    # Solar Impact is battery-smoothed (EMA): first hour with raw=0.2 gives
    # state = 0 * decay + 0.2 * (1 - decay) = 0.2 * 0.50 = 0.1
    assert log_entry["solar_impact_kwh"] == pytest.approx(0.1, abs=0.01)



def test_apply_correction_float_input(calculator):
    """Test apply_correction with float input (derived mode)."""
    # 1. Heating Mode (Temp 10 < BP 15)
    # Base 10, Impact 2. Should subtract. Result 8.
    assert calculator.apply_correction(10.0, 2.0, 10.0) == 8.0

    # 2. Cooling Mode (Temp 20 > BP 15)
    # Base 10, Impact 2. Should add. Result 12.
    assert calculator.apply_correction(10.0, 2.0, 20.0) == 12.0

    # 3. Clamping (Heating)
    # Base 1, Impact 2. Result -1 -> 0.
    assert calculator.apply_correction(1.0, 2.0, 10.0) == 0.0

def test_apply_correction_str_input(calculator):
    """Test apply_correction with string input (explicit mode)."""
    # 1. HEATING
    assert calculator.apply_correction(10.0, 2.0, MODE_HEATING) == 8.0

    # 2. COOLING
    assert calculator.apply_correction(10.0, 2.0, MODE_COOLING) == 12.0

    # 3. UNKNOWN (e.g. OFF)
    # Should return base unchanged
    assert calculator.apply_correction(10.0, 2.0, "unknown_mode") == 10.0

