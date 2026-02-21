"""Test Smart Fill logic for forecasts."""
from datetime import datetime, date, timedelta, timezone
from unittest.mock import MagicMock, patch, PropertyMock
import pytest
from custom_components.heating_analytics.forecast import ForecastManager

@pytest.fixture
def mock_fm(hass):
    coord = MagicMock()
    # FIX: 3 args
    coord._get_predicted_kwh = MagicMock(side_effect=lambda t, w, e=None: 1.0)
    coord._calculate_effective_wind = MagicMock(return_value=5.0)
    coord._get_wind_bucket = MagicMock(return_value="normal")
    coord._get_cloud_coverage = MagicMock(return_value=50.0)

    # FIX: Mock return value for aux impact (must be float) to prevent TypeError
    # Use configure_mock to be safer
    coord.configure_mock(**{
        "_get_aux_impact_kw.return_value": 0.0,
        "auxiliary_heating_active": False
    })

    coord.extreme_wind_threshold = 15.0
    coord.wind_threshold = 10.0
    coord.hass = hass

    fm = ForecastManager(coord)
    fm._cached_hourly_by_date = {}
    return fm

def test_smart_fill_logic_future_dense(mock_fm):
    """Test smart fill when future data is dense enough (>=12h)."""
    # Explicitly disable aux to avoid TypeError path
    mock_fm.coordinator.auxiliary_heating_active = False
    mock_fm.coordinator._get_aux_impact_kw.return_value = 0.0

    # Setup 12 hours of data for tomorrow
    target_date = date(2023, 10, 27)
    hourly_data = []
    for h in range(12): # 0..11
        hourly_data.append({
            "datetime": datetime(2023, 10, 27, h, 0, 0).isoformat(),
            "temperature": 10.0,
            "wind_speed": 5.0
        })
    mock_fm._cached_hourly_by_date[target_date.isoformat()] = hourly_data

    # Mock time (Yesterday)
    now = datetime(2023, 10, 26, 12, 0, 0)

    # Patch dt_util in the module to handle timezone and now()
    with patch("custom_components.heating_analytics.forecast.dt_util") as mock_dt:
        mock_dt.now.return_value = now
        mock_dt.parse_datetime.side_effect = lambda x: datetime.fromisoformat(x)
        mock_dt.as_local.side_effect = lambda x: x
        mock_dt.get_time_zone.return_value = timezone.utc

        # Patch CLASS method to avoid instance binding issues
        with patch("custom_components.heating_analytics.forecast.ForecastManager._process_forecast_item",
                      return_value=(1.0, 0.0, 10.0, 10.0, 5.0, 5.0, {}, 0.0)) as mock_proc:

            # Should use Smart Fill
            result = mock_fm.get_future_day_prediction(target_date)

            assert result is not None
            # 12 hours * 1.0 = 12.0. Scale to 24?
            assert result[0] == 24.0

            # Verify call count (should be called for each hour + fill)
            assert mock_proc.call_count >= 12
