
"""Tests for the forecast blending logic in the ForecastManager."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta, timezone

from custom_components.heating_analytics.const import (
    CONF_SECONDARY_WEATHER_ENTITY,
    CONF_FORECAST_CROSSOVER_DAY,
)
from custom_components.heating_analytics.forecast import ForecastManager

# No autouse fixture - we'll patch individually in each test

# Helper function to get the fixed timestamp (for use in tests)
def get_fixed_now():
    """Get a fixed timestamp for deterministic tests."""
    from datetime import datetime, timezone
    return datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)

def get_start_of_day():
    """Get start of day for the fixed timestamp."""
    return get_fixed_now().replace(hour=0, minute=0, second=0, microsecond=0)

def create_forecast(start_dt, num_days, source_id):
    """Generate a list of mock forecast items."""
    forecast = []
    for i in range(num_days * 24):
        f_dt = start_dt + timedelta(hours=i)
        forecast.append({
            "datetime": f_dt.isoformat(),
            "temperature": 10,
            "_test_source": source_id
        })
    return forecast

@pytest.fixture
def mock_coordinator():
    """Fixture for a mock coordinator."""
    coordinator = MagicMock()
    coordinator.hass = MagicMock()
    coordinator.weather_entity = "weather.primary"
    coordinator.config_entry = MagicMock()
    # Use a real dictionary for data so .get() works naturally
    coordinator.config_entry.data = {
        CONF_SECONDARY_WEATHER_ENTITY: "weather.secondary",
        CONF_FORECAST_CROSSOVER_DAY: 4,
    }

    # Crucial: ForecastManager uses self.coordinator.entry, not self.coordinator.config_entry directly sometimes?
    # No, it calls self.coordinator.entry.data.
    # In the coordinator implementation, self.entry = config_entry.
    # So we must mock this alias.
    coordinator.entry = coordinator.config_entry

    coordinator.hass.services.async_call = AsyncMock()
    return coordinator

@pytest.mark.asyncio
@patch('custom_components.heating_analytics.forecast.dt_util.now')
async def test_forecast_blending_with_crossover(mock_dt_now, mock_coordinator):
    """Test that forecasts are blended correctly with a hard crossover day."""
    FIXED_NOW = get_fixed_now()
    START_OF_DAY = get_start_of_day()

    # Mock dt_util.now() to return our fixed timestamp
    mock_dt_now.return_value = FIXED_NOW

    fm = ForecastManager(mock_coordinator)

    # Ensure correct data
    mock_coordinator.entry.data = {
         CONF_SECONDARY_WEATHER_ENTITY: "weather.secondary",
         CONF_FORECAST_CROSSOVER_DAY: 4,
    }

    primary_forecast = create_forecast(START_OF_DAY, 5, "primary")
    secondary_forecast = create_forecast(START_OF_DAY, 7, "secondary")

    async def side_effect(*args, **kwargs):
        service_data = args[2]
        entity_id = service_data.get("entity_id")
        forecast_type = service_data.get("type")

        if forecast_type == "daily":
            return {entity_id: {"forecast": []}}

        if entity_id == "weather.primary":
            return {"weather.primary": {"forecast": primary_forecast}}
        if entity_id == "weather.secondary":
            return {"weather.secondary": {"forecast": secondary_forecast}}
        return {}

    mock_coordinator.hass.services.async_call.side_effect = side_effect

    _, _, blended_hourly, _, _, _ = await fm._fetch_and_blend_forecasts()

    assert len(blended_hourly) == 7 * 24

    crossover_date = FIXED_NOW.date() + timedelta(days=4)

    # Use datetime.fromisoformat instead of dt_util.parse_datetime to avoid Mock issues
    from datetime import datetime
    for item in blended_hourly:
        item_date = datetime.fromisoformat(item["datetime"]).date()
        if item_date < crossover_date:
            assert item["_source"] == "primary"
        else:
            assert item["_source"] == "secondary"

@pytest.mark.asyncio
@patch('custom_components.heating_analytics.forecast.dt_util.now')
async def test_primary_only_mode(mock_dt_now, mock_coordinator):
    """Test correct operation with only a primary entity."""
    FIXED_NOW = get_fixed_now()
    mock_dt_now.return_value = FIXED_NOW
    START_OF_DAY = FIXED_NOW.replace(hour=0, minute=0, second=0, microsecond=0)

    # Empty dict implies no secondary weather entity and no crossover day
    mock_coordinator.entry.data = {}

    fm = ForecastManager(mock_coordinator)

    START_OF_DAY = get_start_of_day()
    primary_forecast = create_forecast(START_OF_DAY, 5, "primary")

    async def side_effect(*args, **kwargs):
        service_data = args[2]
        entity_id = service_data.get("entity_id")
        forecast_type = service_data.get("type")
        if forecast_type == "daily":
            return {entity_id: {"forecast": []}}
        if entity_id == "weather.primary":
            return {"weather.primary": {"forecast": primary_forecast}}
        return {}

    mock_coordinator.hass.services.async_call.side_effect = side_effect

    _, _, blended_hourly, _, _, _ = await fm._fetch_and_blend_forecasts()

    assert len(blended_hourly) == 5 * 24
    assert all(item["_source"] == "primary" for item in blended_hourly)

@pytest.mark.asyncio
@patch('custom_components.heating_analytics.forecast.dt_util.now')
async def test_blending_with_gaps_before_crossover(mock_dt_now, mock_coordinator):
    """Test that the secondary forecast fills gaps in the primary before the crossover."""
    FIXED_NOW = get_fixed_now()
    START_OF_DAY = get_start_of_day()

    # Mock dt_util.now() to return our fixed timestamp
    mock_dt_now.return_value = FIXED_NOW

    # Ensure crossover config is present
    mock_coordinator.entry.data = {
         CONF_SECONDARY_WEATHER_ENTITY: "weather.secondary",
         CONF_FORECAST_CROSSOVER_DAY: 4,
    }

    fm = ForecastManager(mock_coordinator)
    primary_forecast = create_forecast(START_OF_DAY, 2, "primary") + create_forecast(START_OF_DAY + timedelta(days=3), 1, "primary")
    secondary_forecast = create_forecast(START_OF_DAY, 7, "secondary")

    async def side_effect(*args, **kwargs):
        service_data = args[2]
        entity_id = service_data.get("entity_id")
        forecast_type = service_data.get("type")

        if forecast_type == "daily":
            return {entity_id: {"forecast": []}}

        if entity_id == "weather.primary": return {"weather.primary": {"forecast": primary_forecast}}
        if entity_id == "weather.secondary": return {"weather.secondary": {"forecast": secondary_forecast}}
        return {}

    mock_coordinator.hass.services.async_call.side_effect = side_effect

    _, _, blended_hourly, _, _, _ = await fm._fetch_and_blend_forecasts()

    assert len(blended_hourly) == 7 * 24

    day2_date = FIXED_NOW.date() + timedelta(days=2)
    # Use datetime.fromisoformat instead of dt_util.parse_datetime to avoid Mock issues
    from datetime import datetime
    day2_items = [item for item in blended_hourly if datetime.fromisoformat(item["datetime"]).date() == day2_date]

    assert len(day2_items) == 24
    assert all(item["_source"] == "secondary" for item in day2_items)
    assert blended_hourly[0]["_source"] == "primary"


def test_log_accuracy_source_tracking(mock_coordinator):
    """Test that log_accuracy correctly identifies the dominant source from hourly logs."""
    fm = ForecastManager(mock_coordinator)

    START_OF_DAY = get_start_of_day()
    FIXED_NOW = get_fixed_now()
    today_str = FIXED_NOW.date().isoformat()
    fm._midnight_forecast_snapshot = {"date": today_str, "kwh": 50.0, "source": "primary"}

    # Scenario 1: Primary is dominant in hourly logs
    mock_coordinator._hourly_log = []
    for h in range(18):
        mock_coordinator._hourly_log.append({
            "timestamp": f"{today_str}T{h:02d}:00:00",
            "hour": h,
            "forecast_source": "primary",
            "forecasted_kwh": 2.0,
            "forecasted_kwh_primary": 2.0,
            "actual_kwh": 2.1
        })
    for h in range(18, 24):
        mock_coordinator._hourly_log.append({
            "timestamp": f"{today_str}T{h:02d}:00:00",
            "hour": h,
            "forecast_source": "secondary",
            "forecasted_kwh": 2.0,
            "forecasted_kwh_secondary": 2.0,
            "actual_kwh": 2.5
        })

    fm.log_accuracy(today_str, 55.0)

    assert len(fm._forecast_history) == 1
    assert fm._forecast_history[0]["source"] == "primary"
    assert "primary" in fm._forecast_history[0]["source_breakdown"]
    assert "secondary" in fm._forecast_history[0]["source_breakdown"]
    assert fm._forecast_history[0]["source_breakdown"]["primary"]["hours"] == 18
    assert fm._forecast_history[0]["source_breakdown"]["secondary"]["hours"] == 6

    # Scenario 2: Secondary is dominant
    fm._midnight_forecast_snapshot["source"] = "secondary"
    fm._forecast_history = []
    mock_coordinator._hourly_log = []
    for h in range(10):
        mock_coordinator._hourly_log.append({
            "timestamp": f"{today_str}T{h:02d}:00:00",
            "hour": h,
            "forecast_source": "primary",
            "forecasted_kwh": 2.0,
            "forecasted_kwh_primary": 2.0,
            "actual_kwh": 2.1
        })
    for h in range(10, 24):
        mock_coordinator._hourly_log.append({
            "timestamp": f"{today_str}T{h:02d}:00:00",
            "hour": h,
            "forecast_source": "secondary",
            "forecasted_kwh": 2.0,
            "forecasted_kwh_secondary": 2.0,
            "actual_kwh": 2.5
        })

    fm.log_accuracy(today_str, 55.0)

    assert len(fm._forecast_history) == 1
    assert fm._forecast_history[0]["source"] == "secondary"
    # Both sources track for all hours now in shadow mode if logs available
    assert fm._forecast_history[0]["source_breakdown"]["secondary"]["hours"] == 14
