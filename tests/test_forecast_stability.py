"""Test to verify Forecast Stability Fix."""
from unittest.mock import MagicMock, patch, AsyncMock
import pytest
from datetime import datetime
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.const import ATTR_FORECAST_TODAY

@pytest.mark.asyncio
async def test_forecast_today_uses_plan_rate(hass):
    """Verify that Forecast Today uses the Planned Rate (Forecast) for the remaining hour, not the volatile Live Rate."""

    # 1. Setup Coordinator with Mock Entry
    entry = MagicMock()
    entry.data = {
        "balance_point": 20.0,
        "outdoor_temp_sensor": "sensor.temp",
        "wind_speed_sensor": "sensor.wind"
    }

    with patch("custom_components.heating_analytics.storage.Store") as mock_store_cls, \
         patch("custom_components.heating_analytics.coordinator.dt_util.now") as mock_now:

        mock_store = mock_store_cls.return_value
        mock_store.async_load = AsyncMock(return_value={})
        mock_store.async_save = AsyncMock()

        coordinator = HeatingDataCoordinator(hass, entry)

        # 2. Mock Internal Methods
        # Mock predicted kWh to return 1.0 for "Live" conditions (Temp 10)
        # and 2.0 for "Forecast" conditions (Temp 0)
        def mock_get_predicted(temp_key, wind, eff_wind=None):
            if temp_key == "10": return 1.0 # Live (Warm)
            if temp_key == "0": return 2.0  # Forecast (Cold)
            return 0.0

        coordinator._get_predicted_kwh = MagicMock(side_effect=mock_get_predicted)

        # Mock Forecast Manager
        # forecast_item_now returning a "Cold" forecast
        coordinator.forecast.get_forecast_for_hour = MagicMock(return_value={
            "temperature": 0.0, "wind_speed": 0.0, "datetime": "2023-01-01T12:00:00"
        })

        # Mock _process_forecast_item to return 2.0 (Plan Rate)
        # Returns: (predicted, solar, inertia, raw_temp, wind, wind_ms, unit_breakdown, aux_impact_kwh)
        coordinator.forecast._process_forecast_item = MagicMock(return_value=(2.0, 0.0, 0.0, 0.0, 0.0, 0.0, {}, 0.0))

        # Mock Future Forecast
        coordinator.forecast.calculate_future_energy = MagicMock(return_value=(10.0, 0.0, {}))

        # 3. Simulate Update
        current_time = datetime(2023, 1, 1, 12, 0, 0)
        mock_now.return_value = current_time

        # Live Condition: Temp 10 (Warm), Rate 1.0
        current_prediction_rate = 1.0
        minutes_passed = 0 # Start of hour

        # Call _update_daily_budgets directly
        coordinator._update_daily_budgets(current_prediction_rate, current_time, minutes_passed)

        # 4. Assertions
        # Forecast Today = Actual(0) + Remaining(Current Hour) + Future(10.0)
        # Remaining Fraction at min 0 = 1.0

        # IF using Live Rate (Old Behavior): Remaining = 1.0 * 1.0 = 1.0 -> Total 11.0
        # IF using Plan Rate (New Behavior): Remaining = 2.0 * 1.0 = 2.0 -> Total 12.0

        forecast_today = coordinator.data[ATTR_FORECAST_TODAY]

        # We expect the STABLE value (Plan Rate), so 12.0
        assert forecast_today == 12.0, f"Expected 12.0 (Plan Rate), got {forecast_today} (Likely Live Rate)"
