"""Test the Thermodynamic Projection logic in Coordinator."""
from unittest.mock import MagicMock, patch
import pytest
from datetime import datetime, timedelta
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator

@pytest.mark.asyncio
async def test_thermodynamic_projection_calculation():
    """Test the calculation of thermodynamic projection."""
    hass = MagicMock()
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.data = {}

    with patch("custom_components.heating_analytics.storage.Store") as mock_store_cls, \
         patch("custom_components.heating_analytics.coordinator.dt_util.now") as mock_now:

        mock_store = mock_store_cls.return_value
        mock_store.async_load = MagicMock()
        mock_store.async_save = MagicMock()

        coordinator = HeatingDataCoordinator(hass, entry)

        # Setup Time
        now = datetime(2023, 10, 27, 12, 30, 0)
        mock_now.return_value = now

        # Setup Data State
        coordinator._accumulated_expected_energy_hour = 2.0 # expected_hour_so_far
        coordinator._accumulated_energy_today = 15.0 # Actual so far

        # Setup Logs (expected_today_sum)
        # 10 logs of 1.0 each = 10.0
        coordinator._hourly_log = [
            {"timestamp": "2023-10-27T10:00:00", "hour": 10, "expected_kwh": 5.0, "forecasted_kwh": 5.0, "solar_impact_kwh": 0.0},
            {"timestamp": "2023-10-27T11:00:00", "hour": 11, "expected_kwh": 5.0, "forecasted_kwh": 5.0, "solar_impact_kwh": 0.0}
        ]
        # expected_today_sum = 10.0

        # Mock Forecast Manager
        coordinator.forecast = MagicMock()

        # Mock calculate_future_energy (future_forecast_kwh)
        coordinator.forecast.calculate_future_energy.return_value = (20.0, 0.0, {})

        # Mock get_plan_for_hour
        # Call 1: source='reference' (for current_hour_plan_rate) -> 5.0
        # Call 2: source='live' (for current_hour_live_rate) -> 6.0
        def mock_get_plan(dt, source='reference', ignore_aux=False):
            if source == 'reference':
                return 5.0, {}
            elif source == 'live':
                return 6.0, {}
            return 0.0, {}

        coordinator.forecast.get_plan_for_hour.side_effect = mock_get_plan

        # Mock dependencies called by _update_daily_budgets to avoid side effects
        coordinator._calculate_daily_wind_penalty = MagicMock(return_value=0.0)
        coordinator.statistics.calculate_potential_savings = MagicMock()

        # Run _update_daily_budgets
        # current_prediction_rate = 0.0 (irrelevant if get_plan works)
        coordinator._update_daily_budgets(0.0, now, 30)

        # Verify
        # Expected So Far = 10.0 (Logs) + 2.0 (Accum) = 12.0
        # Remaining Live = 6.0 * (30/60) = 3.0
        # Future = 20.0
        # Total Projection = 12.0 + 3.0 + 20.0 = 35.0

        assert coordinator.data["thermodynamic_projection_kwh"] == 35.0

        # Verify Deviation
        # Actual = 15.0
        # Expected So Far = 12.0
        # Deviation = 15.0 - 12.0 = 3.0 (Positive means using MORE than expected, which is bad? Or house is warmer?)
        # Wait, if Actual > Expected, deviation is positive.
        # "Huset presterer 4.6% bedre enn forventet" usually means Actual < Expected (Less energy used).
        # Deviation calculation is typically (Actual - Expected).
        # So -1.7 kWh means Actual was 1.7 kWh LESS than Expected.
        # Here, Actual (15) > Expected (12). Deviation = 3.0.
        assert coordinator.data["thermodynamic_deviation_kwh"] == 3.0

        # Verify Pct
        # 3.0 / 35.0 * 100 = 8.57... -> 8.6
        assert coordinator.data["thermodynamic_deviation_pct"] == 8.6
