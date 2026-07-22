"""Snapshot lifecycle for the model-comparison sensors.

The comparison sensors must never compute in their property getters — the
cold path (past-day model evaluations, hybrid projection, explanation
formatting) takes 0.5-1.6 s and stalls the event loop when run inline.
These tests pin the inverted flow: getters read a precomputed snapshot,
the heavy build runs as a background executor job scheduled from
coordinator updates, and the first build waits for
EVENT_HOMEASSISTANT_STARTED so it never competes with startup.
"""
from datetime import date, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.heating_analytics.sensors.comparison import (
    HeatingModelComparisonDaySensor,
    HeatingModelComparisonWeekSensor,
)
from custom_components.heating_analytics.const import (
    ATTR_ENERGY_TODAY,
    ATTR_PREDICTED,
    ATTR_SOLAR_PREDICTED,
    ATTR_TEMP_ACTUAL_TODAY,
    ATTR_WIND_ACTUAL_TODAY,
)

DT_NOW_PATCH = "custom_components.heating_analytics.sensors.comparison.dt_util.now"


def _make_hass(is_running=True):
    hass = MagicMock()
    hass.is_running = is_running
    # Close scheduled coroutines so unawaited-coroutine warnings don't leak.
    hass.async_create_background_task = MagicMock(
        side_effect=lambda coro, name=None: coro.close()
    )
    return hass


def _setup_week_mocks(mock_coordinator):
    """Minimal coherent data set for a full week-sensor snapshot build."""
    mock_coordinator._daily_history = {
        "2023-10-23": {"temp": 5.0, "wind": 3.0, "kwh": 100.0},
        "2023-10-24": {"temp": 5.0, "wind": 3.0, "kwh": 100.0},
    }
    mock_coordinator.data = {
        ATTR_ENERGY_TODAY: 50.0,
        ATTR_PREDICTED: 90.0,
        ATTR_TEMP_ACTUAL_TODAY: 5.0,
        ATTR_WIND_ACTUAL_TODAY: 3.0,
        ATTR_SOLAR_PREDICTED: 0.0,
    }
    mock_coordinator.forecast.calculate_future_energy.return_value = (40.0, 0.0, {})
    mock_coordinator.forecast.get_future_day_prediction.side_effect = (
        lambda d, i=None, ignore_aux=False: (80.0, 0.0, {"temp": 5.0, "wind": 3.0})
    )
    mock_coordinator.calculate_modeled_energy.return_value = (80.0, 0.0, 5.0, 3.0, 10.0)
    mock_coordinator.statistics.calculate_hybrid_projection.return_value = (560.0, 0.0)
    mock_coordinator.statistics.calculate_historical_actual_sum.return_value = 500.0
    mock_coordinator.statistics._get_daily_log_map.return_value = {}
    mock_coordinator._get_wind_bucket.return_value = "normal"


class TestSnapshotGetters:
    def test_getters_return_none_before_first_snapshot(self, mock_coordinator, mock_entry):
        sensor = HeatingModelComparisonWeekSensor(mock_coordinator, mock_entry)
        assert sensor.native_value is None
        assert sensor.extra_state_attributes is None

    def test_getters_read_stored_snapshot_without_computing(self, mock_coordinator, mock_entry):
        sensor = HeatingModelComparisonWeekSensor(mock_coordinator, mock_entry)
        sensor._compute_native_value = MagicMock(
            side_effect=AssertionError("getter must not compute")
        )
        sensor._compute_extra_attributes = MagicMock(
            side_effect=AssertionError("getter must not compute")
        )
        sensor._snapshot = {"native_value": 1.5, "attributes": {"week_number": 43}}
        assert sensor.native_value == 1.5
        assert sensor.extra_state_attributes == {"week_number": 43}


class TestSnapshotScheduling:
    async def test_added_to_hass_schedules_immediately_when_running(
        self, mock_coordinator, mock_entry
    ):
        sensor = HeatingModelComparisonWeekSensor(mock_coordinator, mock_entry)
        sensor.hass = _make_hass(is_running=True)

        await sensor.async_added_to_hass()

        assert sensor.hass.async_create_background_task.call_count == 1
        sensor.hass.bus.async_listen_once.assert_not_called()

    async def test_added_to_hass_defers_first_build_to_started_event(
        self, mock_coordinator, mock_entry
    ):
        import asyncio

        from homeassistant.const import EVENT_HOMEASSISTANT_STARTED

        sensor = HeatingModelComparisonWeekSensor(mock_coordinator, mock_entry)
        sensor.hass = _make_hass(is_running=False)

        await sensor.async_added_to_hass()

        assert sensor.hass.async_create_background_task.call_count == 0
        sensor.hass.bus.async_listen_once.assert_called_once_with(
            EVENT_HOMEASSISTANT_STARTED, sensor._on_hass_started
        )

        # The listener must be a coroutine function: the event bus runs plain
        # sync listeners as executor jobs, off the loop, where the loop-only
        # scheduling APIs are unavailable.
        assert asyncio.iscoroutinefunction(sensor._on_hass_started)

        # Firing the startup event schedules the first build.
        await sensor._on_hass_started(None)
        assert sensor.hass.async_create_background_task.call_count == 1

    def test_failed_task_creation_does_not_wedge_refresh_flag(
        self, mock_coordinator, mock_entry
    ):
        """If task creation raises, the in-flight flag must reset — a stuck
        flag would coalesce-skip every future refresh permanently.
        """
        sensor = HeatingModelComparisonWeekSensor(mock_coordinator, mock_entry)
        sensor.hass = MagicMock()
        sensor.hass.is_running = True
        def _close_and_raise(coro, name=None):
            coro.close()
            raise RuntimeError("loop-only API called off-loop")

        sensor.hass.async_create_background_task = MagicMock(
            side_effect=_close_and_raise
        )

        with pytest.raises(RuntimeError):
            sensor._handle_coordinator_update()
        assert sensor._refresh_in_flight is False

        # The next tick schedules normally again.
        sensor.hass.async_create_background_task = MagicMock(
            side_effect=lambda coro, name=None: coro.close()
        )
        sensor._handle_coordinator_update()
        assert sensor.hass.async_create_background_task.call_count == 1

    def test_coordinator_update_schedules_refresh_when_running(
        self, mock_coordinator, mock_entry
    ):
        sensor = HeatingModelComparisonWeekSensor(mock_coordinator, mock_entry)
        sensor.hass = _make_hass(is_running=True)

        sensor._handle_coordinator_update()

        assert sensor.hass.async_create_background_task.call_count == 1

    def test_coordinator_update_skipped_before_start(self, mock_coordinator, mock_entry):
        sensor = HeatingModelComparisonWeekSensor(mock_coordinator, mock_entry)
        sensor.hass = _make_hass(is_running=False)

        sensor._handle_coordinator_update()

        assert sensor.hass.async_create_background_task.call_count == 0

    def test_refresh_coalesces_while_in_flight(self, mock_coordinator, mock_entry):
        sensor = HeatingModelComparisonWeekSensor(mock_coordinator, mock_entry)
        sensor.hass = _make_hass(is_running=True)

        sensor._handle_coordinator_update()
        sensor._handle_coordinator_update()
        sensor._handle_coordinator_update()

        assert sensor.hass.async_create_background_task.call_count == 1


class TestSnapshotRefresh:
    async def test_refresh_stores_snapshot_and_writes_state(
        self, mock_coordinator, mock_entry
    ):
        sensor = HeatingModelComparisonWeekSensor(mock_coordinator, mock_entry)
        sensor.hass = MagicMock()
        sensor.hass.async_add_executor_job = AsyncMock(side_effect=lambda f, *a: f(*a))
        sensor.async_write_ha_state = MagicMock()
        sensor._build_snapshot = MagicMock(
            return_value={"native_value": 2.5, "attributes": {"week_number": 43}}
        )
        sensor._refresh_in_flight = True

        await sensor._async_refresh_snapshot()

        assert sensor.native_value == 2.5
        assert sensor.extra_state_attributes == {"week_number": 43}
        assert sensor._refresh_in_flight is False
        sensor.async_write_ha_state.assert_called_once()

    async def test_refresh_failure_keeps_last_snapshot_and_clears_flag(
        self, mock_coordinator, mock_entry
    ):
        sensor = HeatingModelComparisonWeekSensor(mock_coordinator, mock_entry)
        sensor.hass = MagicMock()
        sensor.hass.async_add_executor_job = AsyncMock(side_effect=RuntimeError("boom"))
        sensor.async_write_ha_state = MagicMock()
        sensor._snapshot = {"native_value": 9.0, "attributes": {"week_number": 42}}
        sensor._refresh_in_flight = True

        await sensor._async_refresh_snapshot()

        # Last good snapshot survives; flag cleared so the next tick retries.
        assert sensor.native_value == 9.0
        assert sensor._refresh_in_flight is False
        sensor.async_write_ha_state.assert_not_called()


class TestSnapshotColdPathReductions:
    def test_week_snapshot_builds_current_period_days_once(
        self, mock_coordinator, mock_entry
    ):
        """The day list built for the attribute path must be shared with the
        stats aggregation — the pre-snapshot code built it twice per cold pass.
        """
        _setup_week_mocks(mock_coordinator)

        with patch(DT_NOW_PATCH, return_value=datetime(2023, 10, 25, 12, 0, 0)):
            sensor = HeatingModelComparisonWeekSensor(mock_coordinator, mock_entry)
            sensor._build_current_period_days = MagicMock(
                wraps=sensor._build_current_period_days
            )

            sensor._snapshot = sensor._build_snapshot()

            assert sensor._build_current_period_days.call_count == 1
            # And the snapshot is fully populated from that single build.
            assert sensor.extra_state_attributes["current_model_kwh"] == 560.0

    def test_last_year_period_days_cached_per_calendar_day(
        self, mock_coordinator, mock_entry
    ):
        sensor = HeatingModelComparisonWeekSensor(mock_coordinator, mock_entry)
        built = [{"date": "2022-10-24", "kwh": 1.0}]
        sensor._build_last_year_period_days = MagicMock(return_value=built)
        ly_start, ly_end = date(2022, 10, 24), date(2022, 10, 30)

        with patch(DT_NOW_PATCH, return_value=datetime(2023, 10, 25, 12, 0, 0)):
            r1 = sensor._get_last_year_period_days(ly_start, ly_end)
            r2 = sensor._get_last_year_period_days(ly_start, ly_end)
        assert sensor._build_last_year_period_days.call_count == 1
        assert r1 is r2

        # A different range on the same day rebuilds.
        with patch(DT_NOW_PATCH, return_value=datetime(2023, 10, 25, 13, 0, 0)):
            sensor._get_last_year_period_days(ly_start, date(2022, 10, 31))
        assert sensor._build_last_year_period_days.call_count == 2

        # A calendar-day rollover invalidates the cache.
        with patch(DT_NOW_PATCH, return_value=datetime(2023, 10, 26, 0, 5, 0)):
            sensor._get_last_year_period_days(ly_start, ly_end)
        assert sensor._build_last_year_period_days.call_count == 3

    def test_day_sensor_last_year_lookup_uses_prefetched_log_map(
        self, mock_coordinator, mock_entry
    ):
        """Regression: the day sensor's last-year lookup re-scanned the hourly
        log per tick because it bypassed the prefetched map.
        """
        mock_coordinator._daily_history = {
            "2022-10-25": {"temp": 6.0, "wind": 2.0, "kwh": 40.0},
        }
        mock_coordinator.data = {
            ATTR_ENERGY_TODAY: 10.0,
            ATTR_PREDICTED: 20.0,
            ATTR_TEMP_ACTUAL_TODAY: 5.0,
            ATTR_WIND_ACTUAL_TODAY: 3.0,
            ATTR_SOLAR_PREDICTED: 0.0,
        }
        mock_coordinator.forecast.calculate_future_energy.return_value = (5.0, 0.0, {})
        mock_coordinator.forecast.get_future_day_prediction.return_value = None
        mock_coordinator.calculate_modeled_energy.return_value = (30.0, 0.0, 5.0, 3.0, 10.0)
        mock_coordinator.statistics.calculate_hybrid_projection.return_value = (25.0, 0.0)
        mock_coordinator.statistics.calculate_historical_actual_sum.return_value = 40.0
        mock_coordinator.statistics._get_daily_log_map.return_value = {}
        mock_coordinator._get_wind_bucket.return_value = "normal"

        with patch(DT_NOW_PATCH, return_value=datetime(2023, 10, 25, 12, 0, 0)):
            sensor = HeatingModelComparisonDaySensor(mock_coordinator, mock_entry)
            sensor._snapshot = sensor._build_snapshot()

        ly = date(2022, 10, 25)
        ly_calls = [
            c
            for c in mock_coordinator.calculate_modeled_energy.call_args_list
            if c.args and c.args[0] == ly
        ]
        # The stats past-data section also evaluates (ly, ly) without a map —
        # that path is day-cached and not the regression target.  The
        # attribute path's day_last lookup must carry the prefetched map.
        assert ly_calls, "last-year day was never evaluated"
        assert any(
            len(c.args) >= 3 and c.args[2] is not None for c in ly_calls
        ), "last-year lookup bypassed the prefetched hourly-log map"
