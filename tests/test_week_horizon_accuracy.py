"""Week-horizon forecast accuracy: 7-day midnight plans scored against gross actuals.

The Week Ahead range band is the p95 of MEASURED week-sum errors
(|sum(planned) - sum(gross actual)| per stored plan), replacing the former
p95_daily x 7 construction that triple-counted uncertainty.  Actuals come
from the day-ahead accuracy log (_forecast_history[i]["actual_kwh"], the
gross value log_accuracy stores), so week scoring is definitionally
identical to day-ahead scoring.  These tests cover the scoring math, the
window-validity filters, the day-keyed cache, the sample gate on the band
attributes, the midnight capture (executor wrapper + store), and storage
round-trip.
"""
from datetime import date, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.heating_analytics.forecast import ForecastManager
from custom_components.heating_analytics.const import (
    ATTR_DAILY_FORECAST,
    ATTR_FORECAST_RANGE_MIN,
    ATTR_FORECAST_RANGE_MAX,
    WEEK_PLAN_RETENTION_DAYS,
    WEEK_HORIZON_MIN_WINDOWS,
)

# Test fixture only — production runtime uses an exponential kernel from
# CONF_THERMAL_INERTIA via generate_exponential_kernel.
_INERTIA_WEIGHTS_FIXTURE = (0.20, 0.30, 0.30, 0.20)

_TODAY = date(2023, 10, 27)
_NOW = datetime(2023, 10, 27, 12, 0, 0)


def _prime_actuals(fm, days: list[date], gross_kwh: float = 9.0):
    """Fill the day-ahead accuracy log with gross actuals for the given days.

    Mirrors what log_accuracy writes: one entry per scored day carrying
    "actual_kwh" (gross = kwh + aux - guest).  Days where learning was
    disabled (Kelvin protocol) simply have no entry.
    """
    for d in days:
        fm._forecast_history.append(
            {"date": d.isoformat(), "actual_kwh": gross_kwh})


def _plan(made_on: date, planned_daily: float = 10.0,
          primary: str = "weather.test") -> dict:
    return {
        "made_on": made_on.isoformat(),
        "planned_kwh": [planned_daily] * 7,
        "primary_entity": primary,
    }


@patch("custom_components.heating_analytics.forecast.dt_util.now")
def test_week_horizon_scoring_uses_logged_gross_actuals(mock_now, mock_coordinator):
    """Week error = |sum(planned) - sum(actual_kwh from the day-ahead log)|."""
    mock_now.return_value = _NOW
    mock_coordinator.weather_entity = "weather.test"

    fm = ForecastManager(mock_coordinator)

    made_on = _TODAY - timedelta(days=14)
    fm._week_plan_history.append(_plan(made_on, planned_daily=10.0))
    # gross actual 9.0/day -> week sum 63; planned 70 -> error 7
    _prime_actuals(fm, [made_on + timedelta(days=k) for k in range(7)])

    stats = fm._calculate_week_horizon_stats()
    assert stats["samples"] == 1
    assert stats["p50_abs_error"] == pytest.approx(7.0)
    assert stats["p95_abs_error"] == pytest.approx(7.0)


@patch("custom_components.heating_analytics.forecast.dt_util.now")
def test_week_horizon_window_validity_filters(mock_now, mock_coordinator):
    """Windows drop on: too recent, outside 90d, entity mismatch, missing
    scored day (covers both no-actuals and Kelvin learning-disabled days),
    and malformed actual_kwh."""
    mock_now.return_value = _NOW
    mock_coordinator.weather_entity = "weather.test"

    fm = ForecastManager(mock_coordinator)

    # Too recent: last day of the window has not completed yet.
    recent = _TODAY - timedelta(days=3)
    fm._week_plan_history.append(_plan(recent))
    _prime_actuals(fm, [recent + timedelta(days=k) for k in range(7)])

    # Outside the trailing 90-day percentile window.
    stale = _TODAY - timedelta(days=100)
    fm._week_plan_history.append(_plan(stale))
    _prime_actuals(fm, [stale + timedelta(days=k) for k in range(7)])

    # Weather-provider mismatch: plan made under a different primary entity.
    mismatch = _TODAY - timedelta(days=30)
    fm._week_plan_history.append(_plan(mismatch, primary="weather.other"))
    _prime_actuals(fm, [mismatch + timedelta(days=k) for k in range(7)])

    # Missing scored day: day 3 was never logged (no actuals, or learning
    # was disabled that day — same absence either way).
    gap = _TODAY - timedelta(days=45)
    fm._week_plan_history.append(_plan(gap))
    _prime_actuals(fm, [gap + timedelta(days=k) for k in range(7) if k != 3])

    # Malformed actual: entry exists but actual_kwh is None (legacy shape).
    broken = _TODAY - timedelta(days=60)
    fm._week_plan_history.append(_plan(broken))
    _prime_actuals(fm, [broken + timedelta(days=k) for k in range(7) if k != 2])
    fm._forecast_history.append(
        {"date": (broken + timedelta(days=2)).isoformat(), "actual_kwh": None})

    stats = fm._calculate_week_horizon_stats()
    assert stats["samples"] == 0

    # Control: one fully valid plan among the rejects IS scored.
    valid = _TODAY - timedelta(days=20)
    fm._week_plan_history.append(_plan(valid))
    _prime_actuals(fm, [valid + timedelta(days=k) for k in range(7)])
    # Mirror the production invalidation (log_accuracy / backfill / reset).
    fm._cached_week_horizon_stats = None
    stats = fm._calculate_week_horizon_stats()
    assert stats["samples"] == 1


@patch("custom_components.heating_analytics.forecast.dt_util.now")
def test_week_horizon_stats_cached_per_day(mock_now, mock_coordinator):
    """The scoring result is cached per calendar day and re-derived after
    explicit invalidation or a date change."""
    mock_now.return_value = _NOW
    mock_coordinator.weather_entity = "weather.test"

    fm = ForecastManager(mock_coordinator)
    made_on = _TODAY - timedelta(days=14)
    fm._week_plan_history.append(_plan(made_on))
    _prime_actuals(fm, [made_on + timedelta(days=k) for k in range(7)])

    first = fm._calculate_week_horizon_stats()
    assert first["samples"] == 1

    # Mutating inputs without invalidation returns the cached result...
    second_plan = _TODAY - timedelta(days=20)
    fm._week_plan_history.append(_plan(second_plan))
    _prime_actuals(fm, [second_plan + timedelta(days=k) for k in range(7)])
    assert fm._calculate_week_horizon_stats() is first

    # ...explicit invalidation (log_accuracy / backfill / reset) recomputes...
    fm._cached_week_horizon_stats = None
    assert fm._calculate_week_horizon_stats()["samples"] == 2

    # ...and a new calendar day recomputes without explicit invalidation.
    mock_now.return_value = _NOW + timedelta(days=1)
    assert fm._calculate_week_horizon_stats() is not first


def _setup_week_ahead(mock_coordinator):
    """Minimal mock wiring for _calculate_week_ahead_stats_internal."""
    mock_coordinator.hass.config.time_zone = "UTC"
    mock_coordinator.weather_entity = "weather.test"
    mock_coordinator.inertia_weights = _INERTIA_WEIGHTS_FIXTURE
    mock_coordinator.statistics._get_daily_log_map = MagicMock(return_value={})
    mock_coordinator.calculate_modeled_energy = MagicMock(
        return_value=(10.0, 0.0, 5.0, 5.0, 10.0))
    mock_coordinator._get_inertia_list = MagicMock(return_value=[])
    mock_coordinator._calculate_inertia_temp = MagicMock(return_value=5.0)
    mock_coordinator._get_wind_bucket = MagicMock(return_value="normal")
    mock_coordinator._is_model_covered = MagicMock(return_value=True)
    mock_coordinator.solar_enabled = False

    fm = ForecastManager(mock_coordinator)
    w_stats = {"temp": 5.0, "wind": 2.0, "source": "hourly_forecast",
               "final_inertia": []}
    fm.get_future_day_prediction = MagicMock(return_value=(20.0, 0.0, w_stats))
    return fm


@patch("custom_components.heating_analytics.forecast.dt_util.now")
def test_band_omitted_below_sample_gate(mock_now, mock_coordinator):
    """No range band (and no sample attr) until enough scorable windows."""
    mock_now.return_value = _NOW
    fm = _setup_week_ahead(mock_coordinator)

    stats = fm._calculate_week_ahead_stats_internal()

    assert ATTR_FORECAST_RANGE_MIN not in stats
    assert ATTR_FORECAST_RANGE_MAX not in stats
    assert "week_error_samples" not in stats


@patch("custom_components.heating_analytics.forecast.dt_util.now")
def test_band_from_measured_week_error_clamped_at_zero(mock_now, mock_coordinator):
    """At >= WEEK_HORIZON_MIN_WINDOWS the band is total -/+ p95(week error),
    with the lower bound clamped at zero."""
    mock_now.return_value = _NOW
    fm = _setup_week_ahead(mock_coordinator)

    # Gross actuals 0.0 for every day any plan touches: today-20 .. today-1.
    _prime_actuals(fm, [_TODAY - timedelta(days=i) for i in range(1, 21)],
                   gross_kwh=0.0)
    # 14 scorable plans, each 300 kWh/week vs 0 actual -> |error| = 300.
    for i in range(WEEK_HORIZON_MIN_WINDOWS):
        fm._week_plan_history.append(
            _plan(_TODAY - timedelta(days=7 + i), planned_daily=300.0 / 7))

    stats = fm._calculate_week_ahead_stats_internal()

    # 7 forecast days x 20 kWh = 140 total; p95 = 300 > 140 -> min clamps to 0.
    assert stats["total_kwh"] == pytest.approx(140.0)
    assert stats[ATTR_FORECAST_RANGE_MIN] == 0.0
    assert stats[ATTR_FORECAST_RANGE_MAX] == pytest.approx(140.0 + 300.0, abs=0.5)
    assert stats["week_error_samples"] == WEEK_HORIZON_MIN_WINDOWS


def test_store_week_plan_appends_dedupes_and_trims(mock_coordinator):
    """The store step keeps the 7 displayed kwh values once per day, capped."""
    fm = ForecastManager(mock_coordinator)
    fm._midnight_forecast_snapshot = {"primary_entity": "weather.test"}

    daily = [{"kwh": float(i)} for i in range(7)]
    stats = {ATTR_DAILY_FORECAST: daily}

    fm._store_week_plan("2023-10-27", stats)
    assert len(fm._week_plan_history) == 1
    entry = fm._week_plan_history[0]
    assert entry["made_on"] == "2023-10-27"
    assert entry["planned_kwh"] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert entry["primary_entity"] == "weather.test"

    # Same day twice: no dupe.
    fm._store_week_plan("2023-10-27", stats)
    assert len(fm._week_plan_history) == 1

    # A short plan (weather outage) is not stored.
    fm._store_week_plan("2023-10-28", {ATTR_DAILY_FORECAST: daily[:3]})
    assert len(fm._week_plan_history) == 1

    # Retention trim: oldest entries fall off at the cap.
    fm._week_plan_history = [
        {"made_on": f"old-{i}", "planned_kwh": [1.0] * 7,
         "primary_entity": "weather.test"}
        for i in range(WEEK_PLAN_RETENTION_DAYS)
    ]
    fm._store_week_plan("2023-10-29", stats)
    assert len(fm._week_plan_history) == WEEK_PLAN_RETENTION_DAYS
    assert fm._week_plan_history[-1]["made_on"] == "2023-10-29"
    assert fm._week_plan_history[0]["made_on"] == "old-1"


async def test_async_capture_runs_in_executor_and_seeds_cache(mock_coordinator):
    """The capture wrapper computes off the event loop, seeds the week-ahead
    stats cache, and stores the plan."""
    fm = ForecastManager(mock_coordinator)
    fm._midnight_forecast_snapshot = {"primary_entity": "weather.test"}

    stats = {ATTR_DAILY_FORECAST: [{"kwh": 1.0}] * 7}
    mock_coordinator.hass.async_add_executor_job = AsyncMock(return_value=stats)

    await fm._async_capture_week_plan("2023-10-27")

    mock_coordinator.hass.async_add_executor_job.assert_awaited_once_with(
        fm._calculate_week_ahead_stats_internal)
    assert fm._cached_week_ahead_stats is stats
    assert fm._cached_week_ahead_timestamp is not None
    assert len(fm._week_plan_history) == 1


async def test_async_capture_failure_never_raises(mock_coordinator):
    """A failing computation is swallowed (best-effort capture): no plan,
    no cache seed, no exception into the update loop."""
    fm = ForecastManager(mock_coordinator)
    mock_coordinator.hass.async_add_executor_job = AsyncMock(
        side_effect=RuntimeError("boom"))
    mock_coordinator.hass.is_running = False  # startup: warning suppressed

    await fm._async_capture_week_plan("2023-10-27")

    assert fm._week_plan_history == []
    assert fm._cached_week_ahead_stats is None


def test_reset_forecast_history_clears_week_plans(mock_coordinator):
    fm = ForecastManager(mock_coordinator)
    fm._week_plan_history = [_plan(_TODAY)]
    fm._cached_week_horizon_stats = ("2023-10-27", {"samples": 1})
    fm.reset_forecast_history()
    assert fm._week_plan_history == []
    assert fm._cached_week_horizon_stats is None


@pytest.fixture
def _persist_hass():
    hass = MagicMock()
    hass.config.units.is_metric = True
    return hass


@pytest.fixture
def _persist_entry():
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.data = {
        "energy_sensors": ["sensor.energy"],
        "weather_entity": "weather.home",
    }
    return entry


async def test_week_plan_history_persistence_roundtrip(_persist_hass, _persist_entry):
    """week_plan_history survives save/load; absent key defaults to []."""
    from custom_components.heating_analytics.coordinator import HeatingDataCoordinator

    with patch("custom_components.heating_analytics.storage.Store") as MockStore:
        mock_store_instance = MockStore.return_value
        mock_store_instance.async_load = AsyncMock(return_value=None)
        mock_store_instance.async_save = AsyncMock()

        coordinator = HeatingDataCoordinator(_persist_hass, _persist_entry)
        await coordinator.storage.async_load_data()
        # Pre-existing store without the key (no migration): defaults to [].
        assert coordinator.forecast._week_plan_history == []

        plan = {"made_on": "2023-10-20", "planned_kwh": [1.0] * 7,
                "primary_entity": "weather.home"}
        coordinator.forecast._week_plan_history = [plan]

        await coordinator.storage.async_save_data(force=True)
        saved_data = mock_store_instance.async_save.call_args[0][0]
        assert saved_data["week_plan_history"] == [plan]

        # Simulate restart: load from the saved payload.
        mock_store_instance.async_load.return_value = saved_data
        coordinator_2 = HeatingDataCoordinator(_persist_hass, _persist_entry)
        await coordinator_2.storage.async_load_data()
        assert coordinator_2.forecast._week_plan_history == [plan]

        # Corrupt (non-list) value is discarded rather than crashing.
        saved_data["week_plan_history"] = {"bad": "shape"}
        coordinator_3 = HeatingDataCoordinator(_persist_hass, _persist_entry)
        await coordinator_3.storage.async_load_data()
        assert coordinator_3.forecast._week_plan_history == []
