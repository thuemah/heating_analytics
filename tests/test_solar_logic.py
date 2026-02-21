
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timedelta
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.const import ATTR_SOLAR_IMPACT

# Mock HomeAssistant
class MockHass:
    def __init__(self):
        self.config = MagicMock()
        self.config.latitude = 50.0
        self.config.longitude = 10.0
        self.states = MagicMock()
        self.data = {"heating_analytics": {}}
        self.services = MagicMock()
        self.async_add_executor_job = AsyncMock()
        self.is_running = True

@pytest.fixture
def mock_coordinator():
    hass = MockHass()
    entry = MagicMock()
    entry.data = {
        "energy_sensors": ["sensor.heater1"],
        "balance_point": 15.0,
        "solar_enabled": True,
        "solar_window_area": 10.0,
        "csv_auto_logging": False
    }

    # Patch Store to prevent file I/O
    with patch("custom_components.heating_analytics.storage.Store"):
        coord = HeatingDataCoordinator(hass, entry)
        # Initialize internal structures
        coord._correlation_data = {"10": {"normal": 1.0}} # 1 kWh predicted at 10C
        coord._correlation_data_per_unit = {"sensor.heater1": {"10": {"normal": 1.0}}}
        coord._observation_counts = {"sensor.heater1": {"10": {"normal": 100}}}
        coord._hourly_log = []
        coord.solar_enabled = True
        coord._async_save_data = AsyncMock()

        # Populate new structures
        # Use simple coefficient of 1.0 for unit such that impact = factor * 1.0
        # If we set factor=0.4 (from current state test below), and we want result 0.4, coeff should be 1.0?
        # Or if we want specific result.
        coord._solar_coefficients_per_unit = {"sensor.heater1": {"10": 1.0}}

        return coord

def test_deviation_breakdown_solar_logic(mock_coordinator):
    """Test that deviation breakdown applies solar correction correctly."""
    coord = mock_coordinator

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
            "actual_kwh": 0.8
        }
        coord._hourly_log.append(log_entry)

        # 2. Setup Current State (Partial hour 10:00-10:30)
        # Temp 10C, Solar Impact 0.4 kW (instant)
        coord.data["effective_wind"] = 0.0
        coord.data[ATTR_SOLAR_IMPACT] = 0.4 # Global Impact
        coord.data["solar_factor"] = 0.4 # Need factor for new calc

        # New logic uses solar_factor * unit_coeff to get impact.
        # We set coeff=1.0 in fixture.
        # So past hour: factor 0.2 * coeff 1.0 = 0.2 impact. Correct.
        # Current hour: factor 0.4 * coeff 1.0 = 0.4 impact. Correct.

        # Mock inertia temp
        coord._calculate_inertia_temp = MagicMock(return_value=10.0)

        # 3. Execute Breakdown
        breakdown = coord.statistics.calculate_deviation_breakdown()

        # 4. Verify Calculation
        # Past Hour (09:00-10:00):
        # Base = 1.0. Solar = 0.2. Expected = 0.8.
        # Unit Base = 1.0. Share = 1.0/1.0 = 1.0.
        # Unit Solar = 0.2 * 1.0 = 0.2.
        # Unit Expected = 1.0 - 0.2 = 0.8.

        # Current Partial (10:00-10:30) - 30 mins = 0.5h
        # Base Rate = 1.0 kWh/h.
        # Solar Rate = 0.4 kW.
        # Unit Share = 1.0.
        # Unit Solar Rate = 0.4 kW.
        # Unit Rate = 1.0 - 0.4 = 0.6 kW.
        # Unit Expected = 0.6 * 0.5h = 0.3 kWh.

        # Total Expected = 0.8 + 0.3 = 1.1 kWh.

        unit_stat = breakdown[0]
        assert unit_stat["entity_id"] == "sensor.heater1"
        assert unit_stat["expected"] == 1.1

@pytest.mark.asyncio
async def test_process_hourly_data_solar_logic(mock_coordinator):
    """Test that hourly processing applies solar logic correctly."""
    coord = mock_coordinator

    # Setup context
    # Hour just finished (10:00-11:00)
    current_time = datetime(2023, 10, 10, 11, 0, 0)

    # Aggregates
    coord._hourly_sample_count = 60
    coord._hourly_temp_sum = 10.0 * 60 # Avg 10C
    coord._hourly_wind_values = [0.0] * 60
    coord._hourly_solar_sum = 0.5 * 60 # Avg Factor 0.5
    coord._accumulated_energy_hour = 0.8 # Actual
    coord._accumulated_expected_energy_hour = 0.8 # Expected (to match logic)

    # NEW LOGIC: We no longer mock calculate_solar_impact_kw (Global) for the final result.
    # We must mock unit coefficient and calculation.

    # Mock unit coefficient retrieval to return a known value
    # coord.solar is a SolarCalculator instance.
    coord.solar.calculate_unit_coefficient = MagicMock(return_value=0.4) # Coeff 0.4

    # Mock calculate_unit_solar_impact.
    # Factor=0.5, Coeff=0.4 -> Impact = 0.2
    coord.solar.calculate_unit_solar_impact = MagicMock(return_value=0.2)

    # Mock global impact for reference (though not used for total sum in process_hourly)
    coord.solar.calculate_solar_impact_kw = MagicMock(return_value=99.9) # Should be ignored for total

    # Execute
    await coord._process_hourly_data(current_time)

    # Verify Log Entry
    log_entry = coord._hourly_log[-1]

    # Logic:
    # Temp 10C. Base Expected 1.0.
    # Unit Solar Impact = 0.2.
    # Expected = 1.0 - 0.2 = 0.8.

    # Note: accumulated_expected_energy_hour is pre-calculated during the hour.
    # The log just records it.

    assert log_entry["expected_kwh"] == 0.8

    # Solar Impact should be the SUM of units (0.2), not the old global calc
    assert log_entry["solar_impact_kwh"] == 0.2

    # Verify we called the new unit method
    coord.solar.calculate_unit_solar_impact.assert_called()
