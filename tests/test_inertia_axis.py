"""One inertia axis for live learning, prediction, forecast, retrain and calibration.

Live learning used to take only the newest ``tau - 1`` hours of the
``min(5·tau, 168)``-hour kernel: ``max_gap_hours = tau`` was applied as an
age cutoff on every hour, measured from the boundary time.  The forecast and
``calibrate_inertia`` used the whole kernel, and Track A retrain a flat
4-hour mean.  All of them now share one computation
(``helpers.weighted_inertia``): the whole kernel, every reading weighted by
its age in hours, history cut only at a gap longer than tau.
"""
from __future__ import annotations

import math
import types
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.helpers import (
    generate_exponential_kernel,
    inertia_samples,
    inertia_temperatures,
    weighted_inertia,
)
from custom_components.heating_analytics.statistics import StatisticsManager
from tests.helpers import bind_inertia_axis

TAU = 4.0
KERNEL = generate_exponential_kernel(tau=TAU, window_hours=20)
N = len(KERNEL)
T0 = datetime(2026, 1, 10, 0, 0, tzinfo=timezone.utc)


def _by_age(ages: dict[int, float]) -> float:
    return sum(KERNEL[N - 1 - a] * t for a, t in ages.items()) / sum(KERNEL[N - 1 - a] for a in ages)


def _live_stub(log):
    live = types.SimpleNamespace(inertia_weights=KERNEL, inertia_tau=TAU, _hourly_log=log)
    for name in ("_get_recent_log_temps", "_calculate_weighted_inertia"):
        setattr(live, name, types.MethodType(getattr(HeatingDataCoordinator, name), live))
    return live


# ---------------------------------------------------------------------------
# helpers.weighted_inertia
# ---------------------------------------------------------------------------


class TestWeightedInertia:
    def test_uses_the_whole_kernel(self):
        temps = [float(i) for i in range(N)]
        assert len(inertia_samples(temps, KERNEL, TAU)) == N
        assert weighted_inertia(temps, KERNEL, TAU) == pytest.approx(
            sum(t * w for t, w in zip(temps, KERNEL)) / sum(KERNEL)
        )

    def test_missing_hour_keeps_older_readings_on_their_age(self):
        temps = [7.0, 6.0, None, 4.0, 3.0]  # ages 4, 3, (2 missing), 1, 0
        assert weighted_inertia(temps, KERNEL, TAU) == pytest.approx(
            _by_age({0: 3.0, 1: 4.0, 3: 6.0, 4: 7.0})
        )

    def test_gap_of_tau_keeps_history_longer_breaks_it(self):
        four_hole = [9.0] + [None] * 3 + [1.0]  # readings 4 h apart
        assert weighted_inertia(four_hole, KERNEL, TAU) == pytest.approx(_by_age({0: 1.0, 4: 9.0}))
        five_hole = [9.0] + [None] * 4 + [1.0]  # 5 h apart > tau
        assert weighted_inertia(five_hole, KERNEL, TAU) == pytest.approx(1.0)

    def test_newest_reading_older_than_tau_leaves_only_the_current_hour(self):
        temps = [9.0, 9.0] + [None] * 5 + [1.0]
        assert weighted_inertia(temps, KERNEL, TAU) == pytest.approx(1.0)

    def test_nothing_usable(self):
        assert weighted_inertia([], KERNEL, TAU) is None
        assert weighted_inertia([None, None], KERNEL, TAU) is None


# ---------------------------------------------------------------------------
# Live history (coordinator) and the batch series (retrain) agree
# ---------------------------------------------------------------------------


def _hourly_log(hours: list[int], temp_of=lambda h: 5.0 + 6.0 * math.sin(h / 3.0)) -> list[dict]:
    return [
        {
            "timestamp": (T0 + timedelta(hours=h)).isoformat(),
            "temp": round(temp_of(h), 3),
            "temp_key": "99",  # a stale key from another axis
        }
        for h in hours
    ]


def test_batch_series_matches_the_live_computation():
    """Retrain's re-keying reproduces, hour by hour, what live learning
    computes at each boundary — including missing hours and a break."""
    hours = [h for h in range(60) if h not in (7, 8, 30, 31, 32, 33, 34)]
    log = _hourly_log(hours)
    series = inertia_temperatures(log, KERNEL, TAU)

    live = _live_stub([])
    for entry, expected in zip(log, series):
        start = datetime.fromisoformat(entry["timestamp"])
        window = live._get_recent_log_temps(start) + [entry["temp"]]
        assert live._calculate_weighted_inertia(window) == pytest.approx(expected)
        live._hourly_log.append(entry)


def test_live_history_covers_the_kernel_not_tau_minus_one_hours():
    log = _hourly_log(range(30))
    live = _live_stub(log)
    closed_hour = T0 + timedelta(hours=30)
    history = live._get_recent_log_temps(closed_hour)
    assert len(history) == N - 1
    assert history[-1] == log[-1]["temp"]


def test_live_history_counts_the_repeated_dst_hour_as_its_own_age():
    oslo = ZoneInfo("Europe/Oslo")
    start = datetime(2026, 10, 25, 0, 0, tzinfo=oslo).astimezone(timezone.utc)
    stamps = [(start + timedelta(hours=i)).astimezone(oslo) for i in range(4)]
    assert [s.isoformat()[11:] for s in stamps] == [
        "00:00:00+02:00", "01:00:00+02:00", "02:00:00+02:00", "02:00:00+01:00",
    ]
    log = [{"timestamp": s.isoformat(), "temp": float(i)} for i, s in enumerate(stamps)]
    live = _live_stub(log)
    now_hour = (start + timedelta(hours=4)).astimezone(oslo)  # 03:00+01:00
    assert live._get_recent_log_temps(now_hour) == [0.0, 1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# The production call site: the logged key of a closed hour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_closed_hour_is_keyed_on_the_whole_kernel(hass):
    entry = MagicMock()
    entry.data = {"balance_point": 17.0, "learning_rate": 0.1, "thermal_inertia": 4}
    with patch("custom_components.heating_analytics.storage.Store"):
        coordinator = HeatingDataCoordinator(hass, entry)
    coordinator._async_save_data = AsyncMock()

    # 10 cold hours, then 3 mild ones: the old tau − 1 window saw only the
    # mild end, the whole kernel still sees the cold.
    history = {h: (-10.0 if h < 10 else 8.0) for h in range(13)}
    coordinator._hourly_log = [
        {"timestamp": (T0 + timedelta(hours=h)).isoformat(), "temp": t}
        for h, t in history.items()
    ]
    closed = T0 + timedelta(hours=13)
    coordinator._collector.start_time = closed
    coordinator._collector.sample_count = 1
    coordinator._collector.wind_values = [0.0]
    coordinator._collector.bucket_counts = {"normal": 1}
    coordinator._collector.temp_sum = 8.0

    await coordinator._process_hourly_data(closed + timedelta(hours=1, seconds=30))

    logged = coordinator._hourly_log[-1]
    ages = {0: 8.0, **{13 - h: t for h, t in history.items()}}
    expected = _by_age({a: t for a, t in ages.items() if a < N})
    assert logged["inertia_temp"] == pytest.approx(round(expected, 2))
    assert logged["temp_key"] == str(int(round(expected)))
    # The previous rule (current + 2 newest hours) would have logged 8.
    assert logged["temp_key"] != "8"


# ---------------------------------------------------------------------------
# Replays re-key entries on the current axis
# ---------------------------------------------------------------------------


class TestEntriesOnCurrentAxis:
    def _coord(self, log):
        coord = MagicMock()
        coord._hourly_log = log
        return bind_inertia_axis(coord, TAU)

    def test_window_keeps_its_warm_up_from_the_whole_log(self):
        log = _hourly_log(range(40))
        coord = self._coord(log)
        window = log[30:]
        rekeyed = coord._entries_on_current_axis(window)
        full = inertia_temperatures(log, KERNEL, TAU)[30:]
        assert [e["inertia_temp"] for e in rekeyed] == pytest.approx([round(v, 2) for v in full])
        assert all(e["temp_key"] == str(int(round(v))) for e, v in zip(rekeyed, full))

    def test_log_is_not_modified(self):
        log = _hourly_log(range(5))
        coord = self._coord(log)
        rekeyed = coord._entries_on_current_axis(list(log))
        assert all(e["temp_key"] == "99" for e in log)
        assert all(e["temp_key"] != "99" for e in rekeyed)

    def test_unreadable_entry_keeps_its_stored_key(self):
        log = _hourly_log(range(3)) + [{"timestamp": "garbage", "temp": 1.0, "temp_key": "42"}]
        coord = self._coord(log)
        assert coord._entries_on_current_axis(log)[-1]["temp_key"] == "42"

    def test_entries_outside_the_log_are_keyed_among_themselves(self):
        coord = self._coord([])
        entries = _hourly_log(range(3))
        rekeyed = coord._entries_on_current_axis(entries)
        expected = inertia_temperatures(entries, KERNEL, TAU)
        assert [e["inertia_temp"] for e in rekeyed] == pytest.approx([round(v, 2) for v in expected])


# ---------------------------------------------------------------------------
# calibrate_inertia: previous vs current axis (diagnostic)
# ---------------------------------------------------------------------------


def test_calibrate_inertia_compares_the_previous_and_current_axis(mock_coordinator):
    """Consumption generated on the whole kernel: the current axis must fit
    it better than the tau − 1 window, and a real share of hours re-bins."""
    from homeassistant.util import dt as dt_util

    mock_coordinator.balance_point = 17.0
    now = dt_util.now()
    start = now - timedelta(days=20)
    temps = [5.0 + 6.0 * math.sin(i / 3.7) + 3.0 * math.sin(i / 17.0) for i in range(20 * 24)]
    logs = []
    for i, t in enumerate(temps):
        eff = weighted_inertia(temps[max(0, i - N + 1): i + 1], KERNEL, TAU)
        logs.append({
            "timestamp": (start + timedelta(hours=i)).isoformat(),
            "temp": t,
            "actual_kwh": max(0.0, 17.0 - eff) / 24.0 * 3.0,
            "wind_bucket": "normal",
            "learning_status": "active",
            "solar_impact_kwh": 0.0,
        })
    mock_coordinator._hourly_log = logs
    mock_coordinator.hourly_solar_impact_kwh = MagicMock(return_value=0.0)

    result = StatisticsManager(mock_coordinator).calibrate_inertia(days=30)
    cmp = result["live_axis_comparison"]

    assert cmp["tau"] == TAU
    assert cmp["kernel_hours"] == N
    assert cmp["previous_axis_hours"] == 3
    assert cmp["current_axis"]["r2"] == pytest.approx(1.0, abs=1e-6)
    assert cmp["previous_axis"]["r2"] < cmp["current_axis"]["r2"]
    assert cmp["r2_change"] > 0
    assert 0.0 < cmp["temp_key_changed_share"] <= 1.0
    assert cmp["by_outdoor_temp"]
    assert result["recommended_tau"] == 4
