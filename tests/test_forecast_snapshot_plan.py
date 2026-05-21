"""Test that the midnight forecast snapshot includes the hourly plan."""
from datetime import datetime
import pytest
from unittest.mock import MagicMock, patch

from custom_components.heating_analytics.forecast import ForecastManager

# Test fixture only — production runtime uses an exponential kernel from
# CONF_THERMAL_INERTIA via generate_exponential_kernel.  These 4 weights
# are an arbitrary mock value satisfying the weighted-average shape.
_INERTIA_WEIGHTS_FIXTURE = (0.20, 0.30, 0.30, 0.20)

@patch("custom_components.heating_analytics.forecast.dt_util.start_of_local_day")
@patch("custom_components.heating_analytics.forecast.dt_util.now")
def test_snapshot_generation_includes_hourly_plan(mock_now, mock_start_of_day, mock_coordinator):
    """Test _capture_daily_forecast_snapshot generates hourly_plan."""
    # Setup specialized coordinator state
    mock_coordinator.hass.config.time_zone = "UTC"
    mock_coordinator.weather_entity = "weather.test"
    mock_coordinator._get_weather_wind_unit.return_value = "m/s"
    mock_coordinator._get_cloud_coverage.return_value = 50.0
    mock_coordinator.inertia_weights = _INERTIA_WEIGHTS_FIXTURE

    # Setup time
    now_time = datetime(2023, 10, 27, 0, 5, 0)
    mock_now.return_value = now_time
    mock_start_of_day.return_value = datetime(2023, 10, 27, 0, 0, 0)

    fm = ForecastManager(mock_coordinator)

    # Setup Reference Forecast (2 items for simplicity)
    # Note: Logic filters by start_time (00:00) -> end_time (Next Day 00:00)
    fm._reference_forecast = [
        {"datetime": "2023-10-27T00:00:00", "temperature": 10.0},
        {"datetime": "2023-10-27T01:00:00", "temperature": 9.0}
    ]

    # Mock _process_forecast_item to return controllable values
    # Returns: (predicted_kwh, solar_kwh, inertia_val, raw_temp, w_speed, w_speed_ms, unit_breakdown, aux_impact_kwh)
    # We'll use side_effect to vary return values
    def process_side_effect(item, *args, **kwargs):
        temp = float(item["temperature"])
        if temp == 10.0:
            return (1.5, 0.0, 5.0, 10.0, 0.0, 0.0, {}, 0.0, 0.0, None)
        else:
            return (2.0, 0.0, 4.0, 9.0, 0.0, 0.0, {}, 0.0, 0.0, None)

    with patch.object(fm, '_process_forecast_item', side_effect=process_side_effect) as mock_process:
        snapshot = fm._capture_daily_forecast_snapshot()

        # Verify call args
        assert mock_process.call_count == 2

        # Verify Snapshot Structure
        assert "hourly_plan" in snapshot
        plan = snapshot["hourly_plan"]
        assert len(plan) == 2

        # Verify Item 0 (Hour 00)
        item0 = plan[0]
        assert item0["hour"] == 0
        assert item0["kwh"] == 1.5
        assert item0["inertia_temp"] == 5.0
        assert item0["aux_expected"] is False # Default ignore_aux=True

        # Verify Item 1 (Hour 01)
        item1 = plan[1]
        assert item1["hour"] == 1
        assert item1["kwh"] == 2.0
        assert item1["inertia_temp"] == 4.0
        assert item1["aux_expected"] is False

        # #980: when no 4D dispatch occurred (spy returned dni_dhi_meta=None
        # for all items), hourly_inputs_4d is an empty list in the internal
        # snapshot result.
        assert snapshot.get("hourly_inputs_4d") == []


@patch("custom_components.heating_analytics.forecast.dt_util.start_of_local_day")
@patch("custom_components.heating_analytics.forecast.dt_util.now")
def test_snapshot_captures_4d_inputs_when_flag_on(mock_now, mock_start_of_day, mock_coordinator):
    """#980: when 4D dispatch runs, snapshot includes per-hour DNI/DHI inputs.

    Spy returns a populated ``dni_dhi_meta`` dict for one item and None
    for the other (simulating one sun-up hour and one sun-down hour).
    The snapshot's ``hourly_inputs_4d`` should contain only the populated
    entry — one entry per hour the 4D leg actually fired (AC #1).
    """
    mock_coordinator.hass.config.time_zone = "UTC"
    mock_coordinator.weather_entity = "weather.test"
    mock_coordinator._get_weather_wind_unit.return_value = "m/s"
    mock_coordinator._get_cloud_coverage.return_value = 50.0
    mock_coordinator.inertia_weights = _INERTIA_WEIGHTS_FIXTURE

    now_time = datetime(2023, 10, 27, 0, 5, 0)
    mock_now.return_value = now_time
    mock_start_of_day.return_value = datetime(2023, 10, 27, 0, 0, 0)

    fm = ForecastManager(mock_coordinator)
    fm._reference_forecast = [
        {"datetime": "2023-10-27T00:00:00", "temperature": 10.0},
        {"datetime": "2023-10-27T12:00:00", "temperature": 9.0},
    ]

    populated_meta = {
        "hour": "2023-10-27T12:00:00",
        "dni": 612.4,
        "dhi": 87.1,
        "dni_dhi_source": "native",
        "cloud_coverage": 22.0,
        "correction_percent": 100.0,
        "sun_elev_deg": 54.2,
    }

    def process_side_effect(item, *args, **kwargs):
        temp = float(item["temperature"])
        if temp == 10.0:
            # Night-time hour: no 4D dispatch, meta=None
            return (1.5, 0.0, 5.0, 10.0, 0.0, 0.0, {}, 0.0, 0.0, None)
        else:
            # Mid-day hour: 4D dispatched, meta populated
            return (2.0, 0.0, 4.0, 9.0, 0.0, 0.0, {}, 0.0, 0.0, populated_meta)

    with patch.object(fm, '_process_forecast_item', side_effect=process_side_effect):
        snapshot = fm._capture_daily_forecast_snapshot()

    inputs_4d = snapshot.get("hourly_inputs_4d")
    assert inputs_4d == [populated_meta], (
        f"Expected one entry mirroring the populated meta; got {inputs_4d}"
    )
