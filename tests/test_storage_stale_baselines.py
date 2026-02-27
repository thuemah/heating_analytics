"""Test that stale energy baselines are discarded on restore after an hour boundary."""
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, AsyncMock

from custom_components.heating_analytics.storage import StorageManager

TZ = timezone(timedelta(hours=1))  # Europe/Oslo (+01:00)

SAVED_BASELINES = {"sensor.heat_pump": 1523.4, "sensor.floor_heat": 342.1}


def _make_coordinator():
    coordinator = MagicMock()
    coordinator.hass = MagicMock()
    coordinator._correlation_data = {}
    coordinator.forecast = MagicMock()
    coordinator.statistics = MagicMock()
    coordinator.energy_sensors = []  # Empty → _cleanup_removed_sensors is a no-op
    coordinator._hourly_wind_values = []  # Prevent MagicMock truthy → wind format crash
    coordinator._hourly_log = []
    coordinator.data = {}
    return coordinator


def _storage_data(accumulation_start_iso: str) -> dict:
    return {
        "accumulation_start_time": accumulation_start_iso,
        "last_energy_values": dict(SAVED_BASELINES),
        "hourly_log": [],
        # Must match today's date (mocked to 2026-02-25) or the daily-reset branch
        # will overwrite _last_energy_values with {} before we can test it.
        "last_save_date": "2026-02-25",
    }


def test_stale_baselines_cleared_on_hour_mismatch():
    """Baselines saved in hour 7 must not produce a cross-hour delta in hour 9.

    Scenario: HA was down from 07:50 to 09:05. On restore the stored
    accumulation_start_time (07:30) does not match the current hour (09:xx),
    so _last_energy_values must be cleared to avoid absorbing the gap energy
    into hour 9's actual_kwh on the first post-restart reading.
    """
    async def _run():
        coordinator = _make_coordinator()
        now = datetime(2026, 2, 25, 9, 5, 0, tzinfo=TZ)

        with patch("custom_components.heating_analytics.storage.dt_util.now", return_value=now), \
             patch("custom_components.heating_analytics.storage.Store") as mock_store_cls:
            mock_store_cls.return_value.async_load = AsyncMock(
                return_value=_storage_data("2026-02-25T07:30:00+01:00")
            )
            storage = StorageManager(coordinator)
            await storage.async_load_data()

        assert coordinator._last_energy_values == {}, (
            "Baselines from hour 7 should be discarded when restoring into hour 9"
        )

    asyncio.run(_run())


def test_baselines_preserved_same_hour():
    """Baselines saved during the same hour are kept so mid-hour deltas remain accurate.

    Scenario: HA restarted at 09:05 while hour 9 was in progress. The stored
    accumulation_start_time (09:00) matches the current hour, so the baselines
    should be restored as-is to continue the partial hour correctly.
    """
    async def _run():
        coordinator = _make_coordinator()
        now = datetime(2026, 2, 25, 9, 5, 0, tzinfo=TZ)

        with patch("custom_components.heating_analytics.storage.dt_util.now", return_value=now), \
             patch("custom_components.heating_analytics.storage.Store") as mock_store_cls:
            mock_store_cls.return_value.async_load = AsyncMock(
                return_value=_storage_data("2026-02-25T09:00:00+01:00")
            )
            storage = StorageManager(coordinator)
            await storage.async_load_data()

        assert coordinator._last_energy_values == SAVED_BASELINES, (
            "Baselines from the same hour should be preserved"
        )

    asyncio.run(_run())
