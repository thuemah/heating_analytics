"""Test Forecast Auxiliary Handling Logic."""
from datetime import date, datetime
from unittest.mock import MagicMock, patch
import pytest

from custom_components.heating_analytics.forecast import ForecastManager

# Mock dt_util
@pytest.fixture
def mock_dt_util():
    with patch("custom_components.heating_analytics.forecast.dt_util") as mock:
        mock.now.return_value = datetime(2023, 10, 10, 12, 0, 0) # Today
        mock.as_local.side_effect = lambda x: x
        mock.parse_datetime.side_effect = lambda x: datetime.fromisoformat(x)
        mock.get_time_zone.return_value = None
        yield mock

# Mock Coordinator
@pytest.fixture
def mock_coordinator():
    coord = MagicMock()
    coord.hass = MagicMock()
    coord.hass.config.time_zone = "UTC"
    coord.weather_entity = "weather.home"
    coord._get_cloud_coverage.return_value = 50.0
    coord._get_float_state.return_value = 10.0
    coord._calculate_effective_wind.return_value = 5.0

    # Real logic for _get_wind_bucket (NO LONGER RETURNS AUX BUCKET)
    coord.auxiliary_heating_active = True
    coord.extreme_wind_threshold = 15.0
    coord.wind_threshold = 10.0

    def real_get_wind_bucket(effective_wind, ignore_aux=False):
        # Always return physical bucket
        if effective_wind >= 15.0: return "extreme_wind"
        if effective_wind >= 10.0: return "high_wind"
        return "normal"

    coord._get_wind_bucket.side_effect = real_get_wind_bucket

    coord._get_predicted_kwh.return_value = 1.0
    coord.solar_enabled = False
    return coord

def test_future_forecast_ignores_aux(mock_coordinator, mock_dt_util):
    """Test that future forecast prediction calls _process_forecast_item with ignore_aux=True."""
    fm = ForecastManager(mock_coordinator)

    # Mock _process_forecast_item to verify it receives ignore_aux=True
    with patch.object(fm, '_process_forecast_item', return_value=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, {})) as mock_process:

        # Setup 24h data for Tomorrow
        target_date = date(2023, 10, 11) # Tomorrow
        hourly_data = []
        for h in range(24):
            dt = datetime(2023, 10, 11, h, 0, 0)
            hourly_data.append({
                "datetime": dt.isoformat(),
                "temperature": 10.0,
                "wind_speed": 5.0
            })
        fm._cached_long_term_hourly = hourly_data

        # Ensure Aux is active on Coordinator
        mock_coordinator.auxiliary_heating_active = True

        # Call logic
        fm.get_future_day_prediction(target_date, ignore_aux=True)

        # Verification: Check that _process_forecast_item was called with ignore_aux=True
        # This confirms that the ForecastManager is correctly propagating the flag,
        # ensuring that even if Aux is active globally, future days are predicted as "Normal".
        assert mock_process.call_count > 0
        _, kwargs = mock_process.call_args
        assert kwargs.get("ignore_aux") is True

def test_today_forecast_respects_aux(mock_coordinator, mock_dt_util):
    """Test that today's forecast calls _process_forecast_item with ignore_aux=False."""
    fm = ForecastManager(mock_coordinator)

    with patch.object(fm, '_process_forecast_item', return_value=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, {})) as mock_process:

        # Setup 24h data for Today
        target_date = date(2023, 10, 10) # Today
        hourly_data = []
        for h in range(24):
            dt = datetime(2023, 10, 10, h, 0, 0)
            hourly_data.append({
                "datetime": dt.isoformat(),
                "temperature": 10.0,
                "wind_speed": 5.0
            })
        fm._cached_long_term_hourly = hourly_data

        # Ensure Aux is active on Coordinator
        mock_coordinator.auxiliary_heating_active = True

        # Call with ignore_aux=False (Today behavior)
        fm.get_future_day_prediction(target_date, ignore_aux=False)

        # Verification
        assert mock_process.call_count > 0
        _, kwargs = mock_process.call_args
        assert kwargs.get("ignore_aux") is False
