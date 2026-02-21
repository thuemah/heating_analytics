"""Test the get_plan_for_hour method."""
from unittest.mock import MagicMock, patch
import pytest
from datetime import datetime
from custom_components.heating_analytics.forecast import ForecastManager

@pytest.mark.asyncio
async def test_get_plan_for_hour_sources():
    """Test get_plan_for_hour with different sources."""
    coordinator = MagicMock()
    # Mock dependencies
    coordinator._get_inertia_list.return_value = [10.0, 10.0]
    coordinator._get_weather_wind_unit.return_value = "km/h"
    coordinator._get_cloud_coverage.return_value = 50.0
    coordinator.solar_enabled = False # Simplify

    manager = ForecastManager(coordinator)

    # Mock get_forecast_for_hour to return different items based on source
    def mock_get_forecast(target_dt, source='live'):
        if source == 'reference':
            return {"temperature": 10.0, "wind_speed": 10.0, "datetime": target_dt.isoformat()}
        elif source == 'live':
            return {"temperature": 5.0, "wind_speed": 20.0, "datetime": target_dt.isoformat()}
        return None

    manager.get_forecast_for_hour = MagicMock(side_effect=mock_get_forecast)

    # Mock _process_forecast_item to return dummy values
    # Returns (predicted, solar, inertia, raw_temp, wind, wind_ms, unit_breakdown)
    def mock_process(item, *args, **kwargs):
        temp = item["temperature"]
        # Dummy prediction logic: kwh = temp * 2
        return (temp * 2.0, 0.0, temp, temp, 0.0, 0.0, {})

    manager._process_forecast_item = MagicMock(side_effect=mock_process)

    target_dt = datetime(2023, 10, 27, 12, 0, 0)

    # Test Reference
    kwh_ref, _ = manager.get_plan_for_hour(target_dt, source='reference')
    assert kwh_ref == 20.0 # 10.0 * 2
    manager.get_forecast_for_hour.assert_called_with(target_dt, source='reference')

    # Test Live
    kwh_live, _ = manager.get_plan_for_hour(target_dt, source='live')
    assert kwh_live == 10.0 # 5.0 * 2
    manager.get_forecast_for_hour.assert_called_with(target_dt, source='live')
