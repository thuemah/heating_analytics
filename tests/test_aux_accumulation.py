"""Tests for auxiliary impact accumulation logic."""
from unittest.mock import MagicMock, patch
from datetime import datetime
import pytest

from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.const import MODE_HEATING

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

        # Configure the MockStats to behave predictably
        mock_stats_instance = MockStats.return_value
        # Mock calculate_total_power to return 1.0 kW aux reduction when active
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

        # Manually initialize trackers
        coord._accumulated_aux_impact_hour = 0.0
        coord._last_minute_processed = None

        return coord

@pytest.mark.asyncio
async def test_aux_impact_accumulation_logic(coordinator):
    """Test that aux impact accumulates correctly minute-by-minute."""

    # 1. Start of hour (Minute 0)
    current_time = datetime(2023, 1, 1, 12, 0, 0)

    # Aux Active
    coordinator.auxiliary_heating_active = True

    # Run update for first minute
    # _update_live_predictions handles fraction calculation
    # Since _last_minute_processed is None, it assumes start of hour -> Minute 0 (1/60th hour passed?)
    # Logic: minutes_step = current_time.minute + 1 = 1. Fraction = 1/60.

    coordinator._update_live_predictions(
        calc_temp=10.0, temp_key="10", wind_bucket="normal", current_time=current_time
    )

    # Expected: 1.0 kW * (1/60) h = 0.0166... kWh
    expected_impact = 1.0 * (1/60)
    assert abs(coordinator._accumulated_aux_impact_hour - expected_impact) < 0.0001
    assert coordinator._last_minute_processed == 0

    # 2. Advance time 29 minutes (Minute 29) - Aux still ON
    current_time = datetime(2023, 1, 1, 12, 29, 0)
    # Diff = 29 - 0 = 29 minutes.
    # Total accumulated should be 30 minutes worth.

    coordinator._update_live_predictions(
        calc_temp=10.0, temp_key="10", wind_bucket="normal", current_time=current_time
    )

    expected_impact = 1.0 * (30/60) # 0.5 kWh
    assert abs(coordinator._accumulated_aux_impact_hour - expected_impact) < 0.0001
    assert coordinator._last_minute_processed == 29

    # 3. Turn Aux OFF (Minute 30)
    coordinator.auxiliary_heating_active = False
    current_time = datetime(2023, 1, 1, 12, 30, 0)
    # Diff = 30 - 29 = 1 minute.
    # Impact rate should be 0.0 for this minute.

    coordinator._update_live_predictions(
        calc_temp=10.0, temp_key="10", wind_bucket="normal", current_time=current_time
    )

    # Accumulator should NOT change (or increase by 0)
    # Still 0.5 kWh
    assert abs(coordinator._accumulated_aux_impact_hour - 0.5) < 0.0001
    assert coordinator._last_minute_processed == 30

    # 4. Verify _update_accumulated_impacts uses the accumulator
    coordinator._update_accumulated_impacts(current_time)

    # Data should reflect the accumulator
    assert coordinator.data["accumulated_aux_impact_kwh"] == 0.5

@pytest.mark.asyncio
async def test_aux_impact_log_persistence(coordinator):
    """Test that the accumulated value is used in hourly log without threshold."""

    # Pre-load accumulator with a small value
    # E.g., active for only 3 minutes (5% of hour) -> Should strictly be logged
    # 1.0 kW * 3/60 = 0.05 kWh
    coordinator._accumulated_aux_impact_hour = 0.05
    coordinator._hourly_aux_count = 3
    coordinator._hourly_sample_count = 60 # 5% fraction
    # Initialize list for percentile calculation to avoid IndexError
    coordinator._hourly_wind_values = [0.0] * 60

    # Mock dependencies for _process_hourly_data
    current_time = datetime(2023, 1, 1, 13, 0, 0)
    coordinator.learning.process_learning.return_value = {
        "learning_status": "ok",
        "model_base_before": 5.0,
        "model_base_after": 5.0,
        "model_updated": False
    }

    # Set start time for the hour
    coordinator._hourly_start_time = datetime(2023, 1, 1, 12, 0, 0)

    # Mock forecast stuff called in _process_hourly_data
    coordinator.forecast.get_forecast_for_hour.return_value = None

    # Mock storage async methods
    coordinator.storage.append_hourly_log_csv = MagicMock()
    coordinator.storage.append_hourly_log_csv.return_value = None
    async def async_mock(*args, **kwargs): return None
    coordinator.storage.append_hourly_log_csv = async_mock

    # Also mock async_save_data called at the end
    coordinator.storage.async_save_data = async_mock

    await coordinator._process_hourly_data(current_time)

    # Check log entry
    assert len(coordinator._hourly_log) == 1
    log = coordinator._hourly_log[0]

    # aux_impact_kwh should be exactly 0.05 (rounded to 3 decimals)
    assert log["aux_impact_kwh"] == 0.05

    # Check data update
    assert coordinator.data["last_hour_aux_impact_kwh"] == 0.05

    # Check accumulator reset
    assert coordinator._accumulated_aux_impact_hour == 0.0
