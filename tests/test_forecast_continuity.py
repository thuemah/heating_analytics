"""Test forecast continuity logic."""
from unittest.mock import MagicMock, patch
import pytest
from datetime import datetime
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.const import ATTR_FORECAST_TODAY

@pytest.mark.asyncio
async def test_forecast_continuity_high_consumption(hass):
    """Test that Forecast Today does not spike when actual consumption exceeds linear projection.

    Scenario:
    - Hourly Forecast: 1.0 kWh
    - Time: 12:15 (25% passed)
    - Actual: 0.9 kWh (90% of budget consumed in 25% of time)

    Naive Logic:
    - Remaining Fraction: 0.75
    - Remaining Forecast: 1.0 * 0.75 = 0.75
    - Total: 0.9 + 0.75 = 1.65 (Spike > 1.0)

    Correct Logic:
    - Remaining Budget: max(0, 1.0 - 0.9) = 0.1
    - Total: 0.9 + 0.1 = 1.0 (Converges to Forecast)
    """

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
        mock_store.async_load = MagicMock(return_value={}) # Use MagicMock for async return if not awaited directly in test setup
        mock_store.async_save = MagicMock()

        coordinator = HeatingDataCoordinator(hass, entry)

        # 2. Mock Forecast Manager
        # forecast_item_now returning 1.0 kWh rate for this hour
        # (Using MagicMock to simulate the return from _process_forecast_item)
        coordinator.forecast.get_forecast_for_hour = MagicMock(return_value={"some": "item"})

        # Mock _process_forecast_item to return 1.0 kWh/h
        # Returns: (predicted, solar, inertia, raw_temp, wind, wind_ms, unit_breakdown)
        coordinator.forecast._process_forecast_item = MagicMock(return_value=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, {}))

        # Mock Future Forecast (Next hours) = 10.0
        coordinator.forecast.calculate_future_energy = MagicMock(return_value=(10.0, 0.0, {}))

        # 3. Setup State
        # Time: 12:15
        current_time = datetime(2023, 1, 1, 12, 15, 0)
        minutes_passed = 15

        # Actual Consumption So Far (Total Today)
        # Assume 5.0 from previous hours + 0.9 from this hour
        previous_hours_total = 5.0
        current_hour_actual = 0.9

        coordinator._accumulated_energy_today = previous_hours_total + current_hour_actual
        coordinator._accumulated_energy_hour = current_hour_actual

        # 4. Execute Update
        # current_prediction_rate doesn't matter for the forecast logic as it uses forecast_item_now
        # but we pass 1.0 for consistency
        coordinator._update_daily_budgets(1.0, current_time, minutes_passed)

        # 5. Verify
        forecast_val = coordinator.data[ATTR_FORECAST_TODAY]

        # Calculation:
        # Actual Total = 5.9
        # Remaining Current Hour = max(0, 1.0 - 0.9) = 0.1
        # Future = 10.0
        # Expected Total = 5.9 + 0.1 + 10.0 = 16.0

        # If Naive Logic was used:
        # Remaining = 1.0 * (45/60) = 0.75
        # Total = 5.9 + 0.75 + 10.0 = 16.65 (Spike)

        assert forecast_val == 16.0, f"Expected 16.0, got {forecast_val}. Phantom spike detected!"
