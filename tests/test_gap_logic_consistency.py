"""Test gap logic consistency using Mean Imputation."""
from unittest.mock import MagicMock, patch
import pytest
from datetime import datetime, timezone
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator

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
        coordinator._accumulated_expected_energy_hour = 0.0
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
        assert coordinator._accumulated_expected_energy_hour == 10.0
        assert coordinator._hourly_expected_per_unit["sensor.heater_1"] == 10.0
