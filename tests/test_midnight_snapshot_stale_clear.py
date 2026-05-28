"""Regression test for stale midnight forecast in coordinator.data.

Codex P1 review caught this: when the 00:00-00:20 capture window is missed
(or any condition where today's snapshot.date doesn't match today_str),
the pre-fix write-only path at ``forecast.py`` left ``coordinator.data``
carrying yesterday's published ``ATTR_MIDNIGHT_FORECAST`` value forever.
Downstream sensors consume ``coordinator.data.get(ATTR, 0.0)`` directly,
so the missed-day "will not be published today" semantic was silently
violated — yesterday's budget kept showing as today's forecast.

The fix pops the three ATTR keys on every tick where the snapshot is
stale or sentinel-flagged, so ``.get(..., 0.0)`` resolves to the default
on missed days.  These tests pin that behaviour.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from custom_components.heating_analytics.const import (
    ATTR_MIDNIGHT_FORECAST,
    ATTR_MIDNIGHT_UNIT_ESTIMATES,
    ATTR_MIDNIGHT_UNIT_MODES,
)
from custom_components.heating_analytics.forecast import ForecastManager


@pytest.mark.asyncio
@patch("custom_components.heating_analytics.forecast.dt_util.now")
async def test_missed_day_clears_stale_attributes(mock_now, mock_coordinator):
    """Yesterday's snapshot → today is missed → ATTR keys must be cleared.

    Direct reproduction of the Codex P1 finding: simulate that the
    process restarted with yesterday's snapshot still in memory AND
    yesterday's values still published to ``coordinator.data``.  The
    first ``update_daily_forecast`` tick past the 00:20 window on the
    new day must clear the three ATTR keys so downstream sensors fall
    back to their ``.get(..., 0.0)`` defaults.
    """
    mock_coordinator.weather_entity = "weather.test"
    mock_coordinator.hass.config.time_zone = "UTC"
    mock_coordinator.hass.is_running = True
    mock_coordinator.data = {
        ATTR_MIDNIGHT_FORECAST: 42.5,
        ATTR_MIDNIGHT_UNIT_ESTIMATES: {"sensor.foo": 10.0},
        ATTR_MIDNIGHT_UNIT_MODES: {"sensor.foo": "heating"},
    }

    fm = ForecastManager(mock_coordinator)
    # Snapshot from yesterday still in memory.
    fm._midnight_forecast_snapshot = {
        "date": "2026-05-16",
        "kwh": 42.5,
        "unit_estimates": {"sensor.foo": 10.0},
        "unit_modes": {"sensor.foo": "heating"},
    }
    fm._cached_forecast_date = "2026-05-16"
    fm._reference_forecast = []  # Bypass refresh path
    fm._live_forecast = []
    fm._last_live_update = datetime(2026, 5, 17, 1, 30, 0)

    # Today: 01:30 — past the 00:00-00:20 capture window, no fresh capture.
    mock_now.return_value = datetime(2026, 5, 17, 1, 30, 0)

    # Stub fetch so no real API calls happen.
    fm._fetch_and_blend_forecasts = AsyncMock(
        return_value=([], [], [], [], [], [])
    )

    await fm.update_daily_forecast()

    # The three ATTR keys must be cleared — sensors will now resolve to
    # the .get(..., 0.0) default rather than yesterday's 42.5 kWh.
    assert ATTR_MIDNIGHT_FORECAST not in mock_coordinator.data, (
        f"Stale midnight forecast left in coordinator.data: "
        f"{mock_coordinator.data.get(ATTR_MIDNIGHT_FORECAST)}"
    )
    assert ATTR_MIDNIGHT_UNIT_ESTIMATES not in mock_coordinator.data
    assert ATTR_MIDNIGHT_UNIT_MODES not in mock_coordinator.data
    # Sentinel recorded so we don't keep retrying.
    assert fm._midnight_forecast_snapshot.get("date") == "2026-05-17:missed"


@pytest.mark.asyncio
@patch("custom_components.heating_analytics.forecast.dt_util.now")
async def test_valid_today_snapshot_remains_published(mock_now, mock_coordinator):
    """Today's snapshot present → ATTR keys populated as usual.

    Negative control: the clear-path must NOT fire when today's snapshot
    is valid.  Pins that the fix doesn't over-zealously wipe live data.
    """
    mock_coordinator.weather_entity = "weather.test"
    mock_coordinator.hass.config.time_zone = "UTC"
    mock_coordinator.hass.is_running = True
    mock_coordinator.data = {}

    fm = ForecastManager(mock_coordinator)
    fm._midnight_forecast_snapshot = {
        "date": "2026-05-17",
        "kwh": 38.0,
        "unit_estimates": {"sensor.bar": 9.0},
        "unit_modes": {"sensor.bar": "heating"},
    }
    fm._cached_forecast_date = "2026-05-17"
    fm._reference_forecast = []
    fm._live_forecast = []
    fm._last_live_update = datetime(2026, 5, 17, 12, 0, 0)

    mock_now.return_value = datetime(2026, 5, 17, 12, 0, 0)

    fm._fetch_and_blend_forecasts = AsyncMock(
        return_value=([], [], [], [], [], [])
    )

    await fm.update_daily_forecast()

    assert mock_coordinator.data.get(ATTR_MIDNIGHT_FORECAST) == 38.0
    assert mock_coordinator.data.get(ATTR_MIDNIGHT_UNIT_ESTIMATES) == {"sensor.bar": 9.0}
