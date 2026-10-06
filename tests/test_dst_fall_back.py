"""DST fall-back: the repeated hour is an hour of its own.

When the clocks go back (02:59+02:00 → 02:00+01:00) the local hour number
stays at 2.  The hour boundary used to fire on a change in that number, so
the repeated hour and the one before it were logged as one two-hour entry.
The boundary now compares the hour's start instant, the repeated hour gets
its own entry, and every consumer that folded entries by local hour keeps
the two apart (or, for the 24-slot daily vectors, records that slot 2 holds
two hours).
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest
from homeassistant.util import dt as dt_util

from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.daily_processor import DailyProcessor
from custom_components.heating_analytics.helpers import (
    daily_tdd_at_balance_point,
    hour_slots,
    hour_start_utc,
    infer_tdd_balance_point,
    local_day_hours,
    log_entry_instant,
    vector_slot_hours,
)
from custom_components.heating_analytics.learning import LearningManager
from custom_components.heating_analytics.observation import (
    DirectMeter,
    ModelState,
    ObservationCollector,
    WeightedSmear,
)
from custom_components.heating_analytics.statistics import StatisticsManager
from custom_components.heating_analytics.storage import _parse_hour_start

OSLO = ZoneInfo("Europe/Oslo")
FALL_BACK = date(2026, 10, 25)
SPRING_FORWARD = date(2026, 3, 29)


def _local_hours(day: date, tz=OSLO) -> list[datetime]:
    """Every clock-hour start of local ``day``, in order, as local datetimes."""
    start = datetime.combine(day, datetime.min.time(), tz).astimezone(timezone.utc)
    end = datetime.combine(day + timedelta(days=1), datetime.min.time(), tz).astimezone(timezone.utc)
    out = []
    t = start
    while t < end:
        out.append(t.astimezone(tz))
        t += timedelta(hours=1)
    return out


def _entry(local: datetime, **extra) -> dict:
    return {
        "timestamp": local.isoformat(),
        "hour": local.hour,
        "temp": 5.0,
        "inertia_temp": 5.0,
        "temp_key": "5",
        "effective_wind": 1.0,
        "wind_bucket": "normal",
        "solar_factor": 0.0,
        "actual_kwh": 1.0,
        "tdd": 10.0 / 24.0,
        "unit_breakdown": {"sensor.a": 1.0},
        "unit_modes": {},
        **extra,
    }


# ---------------------------------------------------------------------------
# The instant helpers
# ---------------------------------------------------------------------------


class TestHourStartInstant:
    def test_repeated_hour_is_a_different_instant(self):
        first = datetime(2026, 10, 25, 2, 30, tzinfo=OSLO, fold=0)
        second = first.replace(fold=1)
        # The trap: aware datetimes sharing a tzinfo compare naively.
        assert first == second
        assert hour_start_utc(first) != hour_start_utc(second)
        assert hour_start_utc(second) - hour_start_utc(first) == timedelta(hours=1)

    def test_floor_is_taken_in_local_time(self):
        """A half-hour-offset zone starts its hours at local :00."""
        kolkata = ZoneInfo("Asia/Kolkata")
        moment = datetime(2026, 1, 1, 10, 45, tzinfo=kolkata)
        assert hour_start_utc(moment) == datetime(2026, 1, 1, 4, 30, tzinfo=timezone.utc)

    def test_log_entry_instant_orders_where_the_string_does_not(self):
        hours = _local_hours(FALL_BACK)
        first, second = _entry(hours[2]), _entry(hours[3])
        assert first["timestamp"] == "2026-10-25T02:00:00+02:00"
        assert second["timestamp"] == "2026-10-25T02:00:00+01:00"
        assert second["timestamp"] < first["timestamp"]  # text order is wrong
        assert log_entry_instant(second) > log_entry_instant(first)

    def test_hour_slots_keeps_both_passes_in_time_order(self):
        entries = [_entry(h) for h in _local_hours(FALL_BACK)]
        shuffled = entries[::-1]
        slots = hour_slots(shuffled)
        assert len(slots) == 25
        assert len({key for key, _ in slots}) == 25
        assert [e["timestamp"] for _, e in slots] == [e["timestamp"] for e in entries]

    def test_local_day_hours(self):
        with patch.object(dt_util, "DEFAULT_TIME_ZONE", OSLO, create=True):
            assert local_day_hours(FALL_BACK) == 25
            assert local_day_hours(SPRING_FORWARD) == 23
            assert local_day_hours(date(2026, 6, 1)) == 24


# ---------------------------------------------------------------------------
# The source: hour boundary and the collector's timestamp
# ---------------------------------------------------------------------------


async def _run_minute_loop(hass, day: date) -> list[datetime]:
    """Drive the real ``_async_update_data`` minute by minute across ``day``.

    Returns the times at which an hour boundary fired.
    """
    entry = MagicMock()
    entry.data = {"balance_point": 17.0, "learning_rate": 0.1}
    with patch("custom_components.heating_analytics.storage.Store") as mock_store_cls:
        mock_store_cls.return_value.async_load = AsyncMock(return_value={})
        mock_store_cls.return_value.async_save = AsyncMock()
        coordinator = HeatingDataCoordinator(hass, entry)
    coordinator.storage.async_load_data = AsyncMock()
    coordinator._async_save_data = AsyncMock()
    coordinator._process_hourly_data = AsyncMock()
    coordinator._process_daily_data = AsyncMock()
    coordinator.statistics.calculate_temp_stats = MagicMock()
    coordinator.statistics.update_daily_savings_cache = MagicMock()
    coordinator.forecast.update_daily_forecast = AsyncMock()

    start = datetime.combine(day, datetime.min.time(), OSLO).astimezone(timezone.utc)
    end = datetime.combine(day + timedelta(days=1), datetime.min.time(), OSLO).astimezone(timezone.utc)
    t = start
    while t <= end:
        now = t.astimezone(OSLO)
        with patch("custom_components.heating_analytics.coordinator.dt_util.now", return_value=now):
            try:
                await coordinator._async_update_data()
            except Exception:  # noqa: BLE001 — only the boundary matters here
                pass
        t += timedelta(minutes=1)
    return [call.args[0] for call in coordinator._process_hourly_data.await_args_list]


@pytest.mark.asyncio
async def test_fall_back_day_fires_25_boundaries(hass):
    fired = await _run_minute_loop(hass, FALL_BACK)
    # 01:00 … 23:00 and the next midnight, with 02:00 twice.
    assert len(fired) == 25
    twos = [t for t in fired if t.hour == 2]
    assert [t.isoformat() for t in twos] == [
        "2026-10-25T02:00:00+02:00",
        "2026-10-25T02:00:00+01:00",
    ]


@pytest.mark.asyncio
async def test_spring_forward_day_fires_23_boundaries(hass):
    fired = await _run_minute_loop(hass, SPRING_FORWARD)
    assert len(fired) == 23
    assert not any(t.hour == 2 for t in fired)


def test_boundary_without_recorded_instant_uses_the_hour_number():
    """State saved by an older version carries only the number: an hour
    missed across the restart must still be finalized."""
    coord = MagicMock()
    coord._last_hour_start = None
    coord._last_hour_processed = 11
    now = datetime(2026, 5, 1, 12, 5, tzinfo=OSLO)
    assert HeatingDataCoordinator._hour_boundary_crossed(coord, now) is True
    coord._last_hour_processed = 12
    assert HeatingDataCoordinator._hour_boundary_crossed(coord, now) is False
    coord._last_hour_processed = None
    assert HeatingDataCoordinator._hour_boundary_crossed(coord, now) is False


def test_recorded_instant_decides_over_the_number():
    coord = MagicMock()
    first = datetime(2026, 10, 25, 2, 59, tzinfo=OSLO, fold=0)
    coord._last_hour_start = hour_start_utc(first)
    coord._last_hour_processed = 2
    repeated = datetime(2026, 10, 25, 2, 0, tzinfo=OSLO, fold=1)
    assert HeatingDataCoordinator._hour_boundary_crossed(coord, first) is False
    assert HeatingDataCoordinator._hour_boundary_crossed(coord, repeated) is True


def test_collector_stamps_the_repeated_hour_with_its_own_offset():
    collector = ObservationCollector()
    now = datetime(2026, 10, 25, 2, 7, tzinfo=OSLO, fold=1)
    collector.accumulate_weather(
        temp=5.0, effective_wind=1.0, wind_bucket="normal", solar_factor=0.0,
        solar_vector=(0.0, 0.0, 0.0), is_aux_active=False, current_time=now,
    )
    assert collector.start_time.isoformat() == "2026-10-25T02:00:00+01:00"


class TestLastHourStartPersistence:
    def test_round_trip(self):
        instant = hour_start_utc(datetime(2026, 10, 25, 2, 0, tzinfo=OSLO, fold=1))
        assert _parse_hour_start(instant.isoformat()) == instant

    @pytest.mark.parametrize("value", [None, 3, "", "not a time", "2026-10-25T02:00:00"])
    def test_absent_or_unreadable_is_none(self, value):
        assert _parse_hour_start(value) is None


def test_partial_log_does_not_match_the_hour_before_the_repeat():
    hours = _local_hours(FALL_BACK)
    coord = MagicMock()
    coord._hourly_log = [_entry(hours[2])]  # 02:00+02:00, already closed
    # Built through UTC: ``aware + timedelta`` resets ``fold`` to 0, so
    # ``hours[3] + 20 min`` would land back in the first pass.
    repeated = (hours[3].astimezone(timezone.utc) + timedelta(minutes=20)).astimezone(OSLO)
    assert repeated.isoformat() == "2026-10-25T02:20:00+01:00"
    with patch("custom_components.heating_analytics.coordinator.dt_util.now", return_value=repeated):
        assert HeatingDataCoordinator._get_partial_log_for_current_hour(coord) is None
    same = (hours[2].astimezone(timezone.utc) + timedelta(minutes=20)).astimezone(OSLO)
    with patch("custom_components.heating_analytics.coordinator.dt_util.now", return_value=same):
        assert HeatingDataCoordinator._get_partial_log_for_current_hour(coord) is coord._hourly_log[-1]


# ---------------------------------------------------------------------------
# Consumers that folded entries by local hour
# ---------------------------------------------------------------------------


def _model() -> ModelState:
    return ModelState(
        correlation_data={},
        correlation_data_per_unit={},
        observation_counts={},
        aux_coefficients={},
        aux_coefficients_per_unit={},
        solar_coefficients_per_unit={},
        learned_u_coefficient=None,
    )


def _apply(day_logs, strategies, distribution=None):
    return LearningManager().apply_strategies_to_global_model(
        day_logs=day_logs,
        track_c_distribution=distribution,
        strategies=strategies,
        model=_model(),
        learning_rate=0.1,
        balance_point=15.0,
        wind_threshold=8.0,
        extreme_wind_threshold=14.0,
        parse_datetime_fn=datetime.fromisoformat,
    )


class TestStrategies:
    def test_smeared_daily_total_keeps_both_repeated_hours(self):
        smear = WeightedSmear("sensor.a", use_synthetic=False)
        _apply([_entry(h) for h in _local_hours(FALL_BACK)], {"sensor.a": smear})
        assert smear._daily_total == pytest.approx(25.0)

    def test_every_logged_hour_contributes_once(self):
        seen = []
        meter = DirectMeter("sensor.a")
        original = meter.get_hourly_contribution

        def spy(key, weight, log_entry):
            seen.append(key)
            return original(key, weight, log_entry)

        meter.get_hourly_contribution = spy
        _apply([_entry(h) for h in _local_hours(FALL_BACK)], {"sensor.a": meter})
        assert len(seen) == 25
        assert len(set(seen)) == 25

    def test_track_c_distribution_joins_each_pass_to_its_own_record(self):
        hours = _local_hours(FALL_BACK)
        distribution = [
            {"datetime": h.isoformat(), "synthetic_kwh_el": float(i)}
            for i, h in enumerate(hours)
        ]
        smear = WeightedSmear("sensor.mpc", use_synthetic=True)
        got = []
        original = smear.get_hourly_contribution

        def spy(key, weight, log_entry):
            value = original(key, weight, log_entry)
            got.append(value or 0.0)
            return value

        smear.get_hourly_contribution = spy
        _apply([_entry(h) for h in hours], {"sensor.mpc": smear}, distribution)
        assert got == [float(i) for i in range(25)]


def _processor(solar_enabled=False, decay=0.5):
    coord = MagicMock()
    coord.solar_enabled = solar_enabled
    coord.balance_point = 15.0
    coord.solar_battery_decay = decay
    coord.hourly_solar_impact_kwh = MagicMock(return_value=0.0)
    return DailyProcessor(coord)


class TestSolarResiduals:
    def test_missing_hour_decays_the_battery_like_a_dark_hour(self):
        """Matches the fixed 0–23 walk it replaces on an ordinary day."""
        proc = _processor(decay=0.5)
        day = [
            datetime(2026, 5, 1, h, tzinfo=timezone.utc) for h in (10, 11, 13)
        ]
        logs = [_entry(d, solar_factor=0.8) for d in day]
        residuals = proc._solar_residuals(hour_slots(logs))
        # Old walk: hour 12 missing → one step with no sun.
        b = 0.0
        expected = {}
        for h in range(24):
            s = 0.8 if h in (10, 11, 13) else 0.0
            b = b * 0.5 + s * 0.5
            expected[h] = min(1.0, b)
        assert [residuals[hour_start_utc(d)] for d in day] == pytest.approx(
            [expected[10], expected[11], expected[13]]
        )


class TestDailyVectors:
    def test_fall_back_slot_records_two_hours(self):
        proc = _processor()
        logs = [_entry(h) for h in _local_hours(FALL_BACK)]
        agg = proc.aggregate_logs(logs)
        hours = agg["hourly_vectors"]["hours"]
        assert hours[2] == 2
        assert sum(hours) == 25
        assert agg["hourly_vectors"]["actual_kwh"][2] == pytest.approx(2.0)
        assert agg["kwh"] == pytest.approx(25.0)
        assert agg["tdd"] == pytest.approx(round(25 * 10.0 / 24.0, 1))

    def test_slot_hours_default_to_one(self):
        assert vector_slot_hours({}) == [1] * 24
        assert vector_slot_hours({"hours": [None, 2, "x", 0, True] + [1] * 19}) == [1, 2] + [1] * 22

    def test_tdd_recomputed_from_vectors_counts_the_repeated_hour(self):
        temps = [5.0] * 24
        hours = [1] * 24
        hours[2] = 2
        entry = {
            "tdd": 99.0,
            "balance_point": 18.0,  # stored at another BP → recompute
            "hourly_vectors": {"temp": temps, "tdd": [0.0] * 24, "hours": hours},
        }
        assert daily_tdd_at_balance_point(entry, 15.0) == pytest.approx(25 * 10.0 / 24.0)

    def test_bp_inference_reads_a_two_hour_slot_per_hour(self):
        temps = [float(t) for t in range(24)]
        tdds = [abs(15.0 - t) / 24.0 for t in temps]
        hours = [1] * 24
        hours[2] = 2
        tdds[2] *= 2
        entry = {"hourly_vectors": {"temp": temps, "tdd": tdds, "hours": hours}}
        assert infer_tdd_balance_point(entry) == (True, 15.0)


def test_modeled_energy_from_vectors_covers_25_hours(mock_coordinator):
    """The vector path models the fall-back day as the 25 hours it had, as
    the hourly-log path does from its 25 entries."""
    mock_coordinator.balance_point = 15.0
    stats = StatisticsManager(mock_coordinator)
    hours = [1] * 24
    hours[2] = 2
    mock_coordinator._daily_history = {
        FALL_BACK.isoformat(): {
            "kwh": 250.0,
            "temp": 5.0,
            "wind": 0.0,
            "hourly_vectors": {
                "temp": [5.0] * 24,
                "actual_kwh": [10.0] * 24,
                "wind": [0.0] * 24,
                "tdd": [10.0 / 24.0] * 24,
                "hours": hours,
            },
        }
    }
    stats.calculate_total_power = MagicMock(
        return_value={"total_kwh": 10.0, "breakdown": {"solar_reduction_kwh": 0.0}}
    )
    total_kwh, _, avg_temp, _, total_tdd = stats.calculate_modeled_energy(FALL_BACK, FALL_BACK)
    assert total_kwh == pytest.approx(250.0)
    assert total_tdd == pytest.approx(round(25 * 10.0 / 24.0, 1))
    assert avg_temp == pytest.approx(5.0)
