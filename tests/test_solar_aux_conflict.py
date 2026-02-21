"""Test preventing model pollution when Solar and Aux are both active."""
from unittest.mock import MagicMock, patch, AsyncMock
import pytest
from datetime import datetime
from homeassistant.core import HomeAssistant
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.const import MODE_HEATING

@pytest.mark.asyncio
async def test_solar_aux_dual_interference(hass: HomeAssistant):
    """Test that learning IS disabled when both Solar and Aux impacts are significant."""
    entry = MagicMock()
    entry.data = {
        "balance_point": 17.0,
        "learning_rate": 0.1,
        "energy_sensors": ["sensor.heater"],
        "solar_enabled": True,
        "aux_affected_entities": ["sensor.heater"]
    }

    with patch("custom_components.heating_analytics.storage.Store") as mock_store_cls:
        mock_store = mock_store_cls.return_value
        mock_store.async_load = AsyncMock(return_value={})
        mock_store.async_save = AsyncMock()

        coordinator = HeatingDataCoordinator(hass, entry)
        coordinator._async_save_data = AsyncMock()

        # 1. Setup Base Model
        # Temp "0" -> Normal Bucket -> Base Consumption 2.0 kWh
        coordinator._correlation_data = {"0": {"normal": 2.0}}
        coordinator._correlation_data_per_unit = {"sensor.heater": {"0": {"normal": 2.0}}}

        # 2. Setup Aux Model
        # Temp "0" -> Normal Bucket -> Aux Savings 0.5 kWh
        coordinator._aux_coefficients = {"0": {"normal": 0.5}}
        coordinator._aux_coefficients_per_unit = {"sensor.heater": {"0": {"normal": 0.5}}}

        # 3. Setup Solar Mocks
        # We need significant solar impact. Let's say 0.5 kWh.
        coordinator.solar.calculate_unit_coefficient = MagicMock(return_value=1.0)
        coordinator.solar.calculate_unit_solar_impact = MagicMock(return_value=0.5)

        # We also need to mock normalize_for_learning because process_learning uses it
        coordinator.solar.normalize_for_learning = MagicMock(side_effect=lambda actual, impact, temp: actual + impact)

        # 4. Simulate Hourly Data
        coordinator._hourly_sample_count = 60
        coordinator._hourly_temp_sum = 0.0 # Temp 0
        coordinator._hourly_wind_values = [0.0] * 60
        coordinator._hourly_bucket_counts = {"normal": 60, "high_wind": 0, "extreme_wind": 0}
        coordinator._hourly_solar_sum = 1.0 # Avg Solar Factor

        # Aux Dominant
        coordinator._hourly_aux_count = 60
        coordinator.auxiliary_heating_active = True

        # 5. Set Impacts
        # Solar Impact will be calculated as 0.5 inside (mocked above)
        # Aux Impact (Accumulated) - Significant!
        coordinator._accumulated_aux_impact_hour = 1.0

        # 6. Set Actuals
        # Expected Net = Base(2.0) - Solar(0.5) - Aux(1.0) = 0.5
        # Let's say Actual is 0.8 (Deviation of +0.3)
        coordinator._accumulated_energy_hour = 0.8
        coordinator._hourly_delta_per_unit = {"sensor.heater": 0.8}

        current_time = datetime(2023, 10, 27, 13, 0, 0)
        coordinator._hourly_start_time = datetime(2023, 10, 27, 12, 0, 0)

        # 7. Run
        await coordinator._process_hourly_data(current_time)

        # 8. Assertions
        assert len(coordinator._hourly_log) == 1
        last_log = coordinator._hourly_log[-1]

        # Check Impacts were recorded correctly
        assert last_log["solar_impact_kwh"] == 0.5
        assert last_log["aux_impact_kwh"] == 1.0

        # Verify Guard Logic
        print(f"Log Learning Status: {last_log['learning_status']}")

        # Should be skipped due to dual interference
        assert last_log["learning_status"] == "skipped_dual_interference"
        assert last_log["model_updated"] is False
        assert last_log["aux_model_updated"] is False

        # Check that coefficients did NOT change
        assert coordinator._correlation_data["0"]["normal"] == 2.0
        assert coordinator._aux_coefficients["0"]["normal"] == 0.5
