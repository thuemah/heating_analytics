"""Test gap filling logic in ForecastManager."""
from datetime import datetime, date
from unittest.mock import MagicMock, patch
import pytest
import sys

# Import the module to ensure it's loaded
import custom_components.heating_analytics.forecast

from custom_components.heating_analytics.forecast import ForecastManager

@pytest.fixture
def mock_fm(hass):
    coord = MagicMock()
    # Mock _get_predicted_kwh with 3 args
    coord._get_predicted_kwh = MagicMock(return_value=1.0) # 1 kWh/h
    coord._calculate_effective_wind = MagicMock(return_value=5.0)
    coord._get_wind_bucket = MagicMock(return_value="normal")
    coord._get_cloud_coverage = MagicMock(return_value=50.0)

    # FIX: Mock return value for aux impact (must be float)
    coord._get_aux_impact_kw = MagicMock(return_value=0.0)

    # New: Add extreme/high thresholds if accessed
    coord.extreme_wind_threshold = 15.0
    coord.wind_threshold = 10.0
    coord.hass = hass # Link hass

    fm = ForecastManager(coord)
    # Mock cache
    fm._cached_hourly_by_date = {}
    return fm

def test_calculate_future_energy_gap_filling(mock_fm):
    """Test 'Smart Merge' of Live and Reference forecasts."""
    # Setup Live: 12:00 -> 23:00 (Missing 00:00 -> 11:00)
    live = []
    for h in range(12, 24):
        live.append({
            "datetime": datetime(2023, 10, 27, h, 0, 0).isoformat(),
            "temperature": 10.0,
            "wind_speed": 5.0,
            "condition": "cloudy"
        })
    mock_fm._live_forecast = live

    # Setup Reference: Full day (00:00 -> 23:00)
    ref = []
    for h in range(24):
        ref.append({
            "datetime": datetime(2023, 10, 27, h, 0, 0).isoformat(),
            "temperature": 5.0, # Different temp to distinguish
            "wind_speed": 5.0,
            "condition": "sunny"
        })
    mock_fm._reference_forecast = ref

    # Mock time
    now = datetime(2023, 10, 27, 0, 30, 0)

    # We patch dt_util ON THE MODULE to ensure _merge_and_fill_forecast works
    with patch.object(custom_components.heating_analytics.forecast, "dt_util") as mock_dt:
        # FIX: Mock now() to return a real datetime for comparison in _merge_and_fill_forecast
        mock_dt.now.return_value = now
        mock_dt.parse_datetime.side_effect = lambda x: datetime.fromisoformat(x)
        mock_dt.as_local.side_effect = lambda x: x

        # Mock sum_forecast_energy to bypass datetime/mock issues in the loop
        # We verify that the MERGED data is passed to it.
        with patch.object(mock_fm, 'sum_forecast_energy', return_value=(175.0, 0, 0.0, {})) as mock_sum:
            total_kwh, total_solar, _ = mock_fm.calculate_future_energy(now)

            assert total_kwh == 175.0

            # Verify the data passed to sum_forecast_energy
            args, kwargs = mock_sum.call_args
            source_data = kwargs.get('source_data')

            # Find 10:00 entry -> Should be Ref (5.0C)
            entry_10 = next((x for x in source_data if "10:00" in x["datetime"]), None)
            assert entry_10 is not None
            assert entry_10["temperature"] == 5.0

            # Find 13:00 entry -> Should be Live (10.0C)
            entry_13 = next((x for x in source_data if "13:00" in x["datetime"]), None)
            assert entry_13 is not None
            assert entry_13["temperature"] == 10.0

def test_calculate_future_energy_reference_only(mock_fm):
    """Test fallback to reference when live is empty."""
    mock_fm._live_forecast = []
    ref = []
    for h in range(24):
        ref.append({
            "datetime": datetime(2023, 10, 27, h, 0, 0).isoformat(),
            "temperature": 5.0,
            "wind_speed": 5.0
        })
    mock_fm._reference_forecast = ref

    now = datetime(2023, 10, 27, 0, 30, 0)

    with patch.object(custom_components.heating_analytics.forecast, "dt_util") as mock_dt:
        mock_dt.now.return_value = now
        mock_dt.parse_datetime.side_effect = lambda x: datetime.fromisoformat(x)
        mock_dt.as_local.side_effect = lambda x: x

        with patch.object(mock_fm, 'sum_forecast_energy', return_value=(23.0, 0, 0.0, {})) as mock_sum:
            total, _, _ = mock_fm.calculate_future_energy(now)
            assert total == 23.0

            # Check source data
            _, kwargs = mock_sum.call_args
            source_data = kwargs.get('source_data')
            assert len(source_data) == 24
            assert source_data[0]["temperature"] == 5.0

def test_calculate_future_energy_live_only(mock_fm):
    """Test live only (reference missing/empty)."""
    live = []
    for h in range(24):
        live.append({
            "datetime": datetime(2023, 10, 27, h, 0, 0).isoformat(),
            "temperature": 10.0,
            "wind_speed": 5.0
        })
    mock_fm._live_forecast = live
    mock_fm._reference_forecast = []

    now = datetime(2023, 10, 27, 0, 30, 0)

    with patch.object(custom_components.heating_analytics.forecast, "dt_util") as mock_dt:
        mock_dt.now.return_value = now
        mock_dt.parse_datetime.side_effect = lambda x: datetime.fromisoformat(x)
        mock_dt.as_local.side_effect = lambda x: x

        with patch.object(mock_fm, 'sum_forecast_energy', return_value=(46.0, 0, 0.0, {})) as mock_sum:
            total, _, _ = mock_fm.calculate_future_energy(now)
            assert total == 46.0

            # Check source data
            _, kwargs = mock_sum.call_args
            source_data = kwargs.get('source_data')
            assert len(source_data) == 24
            assert source_data[0]["temperature"] == 10.0
