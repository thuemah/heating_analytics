"""Test to verify that the midnight forecast snapshot is aux-neutral."""
from unittest.mock import MagicMock, patch, AsyncMock
import pytest
from datetime import datetime, timedelta
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator

@pytest.fixture
def mock_entry():
    """Mock ConfigEntry."""
    entry = MagicMock()
    entry.data = {
        "energy_sensors": ["sensor.energy_meter"],
        "weather_entity": "weather.test",
        "outdoor_temp_sensor": "sensor.test_temp",
        "balance_point": 18.0,
        "solar_enabled": False,
    }
    return entry


@pytest.mark.asyncio
async def test_midnight_forecast_is_aux_neutral(hass, mock_entry):
    """Verify that _capture_daily_forecast_snapshot ignores the current aux state."""

    # 1. Setup Coordinator
    with patch("custom_components.heating_analytics.storage.Store") as mock_store_cls, \
         patch("custom_components.heating_analytics.forecast.dt_util") as mock_dt_util:

        mock_store = mock_store_cls.return_value
        mock_store.async_load = AsyncMock(return_value={})
        mock_store.async_save = AsyncMock()

        # Mock time to be at the start of a day
        start_of_day = datetime(2023, 1, 1, 0, 0, 0)
        mock_dt_util.now.return_value = start_of_day
        mock_dt_util.start_of_local_day.return_value = start_of_day
        # Mock other dt functions used in the forecast logic
        mock_dt_util.as_local.side_effect = lambda d: d
        mock_dt_util.parse_datetime.side_effect = lambda s: datetime.fromisoformat(s)

        coordinator = HeatingDataCoordinator(hass, mock_entry)
        coordinator.solar.get_approx_sun_pos = MagicMock(return_value=(0, 0))

        # 2. Mock Models to return different values for Normal vs Aux
        def mock_prediction(data_map, temp_key, wind_bucket, actual_temp, balance_point, apply_scaling=True):
            if data_map == coordinator._correlation_data:
                return 1.0  # Normal model predicts 1.0 kWh/h
            if data_map == coordinator._aux_coefficients:
                return 0.5  # Aux model REDUCES by 0.5 kWh/h
            return 0.0

        coordinator.statistics._get_prediction_from_model = MagicMock(side_effect=mock_prediction)

        # 3. Mock Weather Forecast
        # Create a simple 24-hour forecast
        forecast_data = []
        for i in range(24):
            forecast_data.append({
                "datetime": (start_of_day + timedelta(hours=i)).isoformat(),
                "temperature": 10.0,
                "wind_speed": 0.0,
                "wind_gust_speed": 0.0,
                "condition": "sunny"
            })
        coordinator.forecast._reference_forecast = forecast_data

        # 4. Set Aux State to ACTIVE
        coordinator.auxiliary_heating_active = True

        # 5. Trigger Snapshot Calculation
        result_snapshot = coordinator.forecast._capture_daily_forecast_snapshot()
        result_kwh = result_snapshot['kwh']

        # 6. Assertions
        # If ignore_aux=False (old behavior), the prediction would be (1.0 - 0.5) * 24 = 12.0 kWh
        # If ignore_aux=True (new behavior), the prediction should be 1.0 * 24 = 24.0 kWh

        expected_kwh = 24.0
        assert result_kwh == pytest.approx(expected_kwh), \
            f"Midnight forecast should have ignored aux state and been {expected_kwh}, but was {result_kwh}"
