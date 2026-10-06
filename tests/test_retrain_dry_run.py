"""``retrain_from_history(dry_run=True)`` previews a retrain and writes nothing.

The dry run swaps deep copies of the model state onto the coordinator, runs
the same synchronous replay a real retrain runs, reports the difference and
puts the live objects back.  These tests pin the four properties that make
that safe and useful:

- nothing is written (model state, history, ``data``, shared learner state,
  and no save), with and without ``reset_first``;
- the dry run's copies end exactly where a real retrain from the same state
  ends, on both tracks;
- the replay cannot yield to the event loop, so a live write scheduled while
  the copies are installed lands in live state;
- the diff and the prediction comparison report what changed.
"""
from __future__ import annotations

import asyncio
import copy
import inspect
import math
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from homeassistant.util import dt as dt_util

from custom_components.heating_analytics import const
from custom_components.heating_analytics.const import (
    DOMAIN,
    MODE_COOLING,
    MODE_DHW,
    MODE_GUEST_HEATING,
    MODE_HEATING,
)
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.observation import WeightedSmear
from custom_components.heating_analytics.retrain import (
    _RETRAIN_MODEL_STATE,
    RetrainEngine,
    _bucket_map_diff,
    _install_model_copies,
    _per_unit_map_diff,
    _reinstall_live_model,
    _solar_coefficient_diff,
)

HP = "sensor.heat_pump"
CABLE = "sensor.cable"


class _Hass:
    def __init__(self):
        self.states = MagicMock()
        self.states.get = MagicMock(return_value=None)
        self.data = {DOMAIN: {}}
        self.config_entries = MagicMock()
        self.bus = MagicMock()
        self.services = MagicMock()
        self.is_running = True


def _coordinator(*, daily: bool) -> HeatingDataCoordinator:
    """A real coordinator: real learning, statistics and solar managers."""
    entry = MagicMock()
    entry.data = {
        "energy_sensors": [HP, CABLE],
        "outdoor_temp_sensor": "sensor.outdoor",
        "balance_point": 15.0,
        "wind_speed_sensor": "sensor.wind",
        "wind_threshold": 5.0,
        "extreme_wind_threshold": 10.0,
        "daily_learning_mode": daily,
    }
    entry.options = {}
    coord = HeatingDataCoordinator(_Hass(), entry)
    coord.storage = MagicMock()
    coord.storage.async_save_data = AsyncMock()
    return coord


def _log(days: int = 10, *, offset: float = 0.0) -> list[dict]:
    """Synthetic hours: a heat pump following temperature and sun, and a
    thermostatic cable that runs one hour in three (reported zero hours
    otherwise)."""
    now = dt_util.now().replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(days=days)
    out = []
    for i in range(days * 24):
        ts = start + timedelta(hours=i)
        t = 3.0 + offset + 6.0 * math.sin(i / 24 * 2 * math.pi) + (i % 7) * 0.3
        sun = 0.6 * max(0.0, math.sin((ts.hour - 6) / 12 * math.pi)) if 7 <= ts.hour <= 17 else 0.0
        hp = max(0.0, (15.0 - t) * 0.12 - sun * 0.4)
        cable = 0.3 if i % 3 == 0 else 0.0
        breakdown = {HP: round(hp, 3)}
        if cable:
            breakdown[CABLE] = cable
        out.append({
            "timestamp": ts.isoformat(),
            "hour": ts.hour,
            "temp": t,
            "temp_key": str(int(round(t))),
            "inertia_temp": t,
            "effective_wind": 3.0 + (i % 5),
            "wind_bucket": "normal",
            "actual_kwh": round(hp + cable, 3),
            "unit_breakdown": breakdown,
            "units_reporting": sorted([HP, CABLE]),
            "unit_modes": {},
            "learning_status": "active",
            "auxiliary_active": False,
            "solar_factor": sun,
            "solar_vector_s": sun * 0.8,
            "solar_vector_e": sun * 0.1,
            "solar_vector_w": sun * 0.1,
            "correction_percent": 100.0,
            "solar_dominant_entities": [],
            "solar_normalization_delta": 0.0,
            "tdd": abs(15.0 - t) / 24,
        })
    return out


async def _trained(*, daily: bool) -> HeatingDataCoordinator:
    """A coordinator whose model already holds a retrain of an older log."""
    coord = _coordinator(daily=daily)
    coord._hourly_log = _log(offset=-2.0)
    await coord.retrain_from_history()
    coord._hourly_log = _log()
    coord._daily_history = {e["timestamp"][:10]: {} for e in coord._hourly_log}
    coord.storage.async_save_data.reset_mock()
    return coord


def _state(coord) -> dict:
    return {attr: copy.deepcopy(getattr(coord, attr)) for attr in _RETRAIN_MODEL_STATE}


def _stored(coord) -> dict:
    return {
        "daily_history": copy.deepcopy(coord._daily_history),
        "hourly_log": copy.deepcopy(coord._hourly_log),
        "data": copy.deepcopy(dict(coord.data)),
        "dead_zone": copy.deepcopy(coord.learning._dead_zone_counts),
    }


# ---------------------------------------------------------------------------
# Nothing written
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("daily", [False, True])
@pytest.mark.parametrize("reset_first", [False, True])
async def test_dry_run_writes_nothing(daily, reset_first):
    coord = await _trained(daily=daily)
    coord.learning._dead_zone_counts[(HP, "heating")] = 3
    live_objects = {attr: getattr(coord, attr) for attr in _RETRAIN_MODEL_STATE}
    live_dead_zone = coord.learning._dead_zone_counts
    state_before = _state(coord)
    stored_before = _stored(coord)
    assert state_before["_correlation_data"], "the model must hold something to protect"

    result = await coord.retrain_from_history(dry_run=True, reset_first=reset_first)

    assert result["status"] == "completed"
    assert result["dry_run"] is True
    assert _state(coord) == state_before
    assert _stored(coord) == stored_before
    # The live objects themselves are back, not equal copies of them.
    for attr, obj in live_objects.items():
        if isinstance(obj, dict):
            assert getattr(coord, attr) is obj, attr
    assert coord.learning._dead_zone_counts is live_dead_zone
    coord.storage.async_save_data.assert_not_awaited()


async def test_dry_run_on_empty_window_writes_nothing():
    coord = await _trained(daily=False)
    coord._hourly_log = []
    state_before = _state(coord)

    result = await coord.retrain_from_history(dry_run=True, reset_first=True)

    assert result["status"] == "no_data"
    assert result["dry_run"] is True
    assert _state(coord) == state_before
    coord.storage.async_save_data.assert_not_awaited()


def test_weighted_smear_distribution_goes_back():
    """The Track C dispatch leaves the last replayed day's distribution on
    each WeightedSmear strategy; the dry run puts the live one back."""
    coord = _coordinator(daily=True)
    smear = WeightedSmear(CABLE, use_synthetic=True)
    smear.set_distribution({"live": 1})
    smear.set_daily_total(4.2)
    coord._unit_strategies[CABLE] = smear

    live = _install_model_copies(coord)
    smear.set_distribution({"replayed": 2})
    smear.set_daily_total(0.0)
    _reinstall_live_model(coord, live)

    assert smear._distribution == {"live": 1}
    assert smear._daily_total == 4.2


# ---------------------------------------------------------------------------
# Same replay
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("daily", [False, True])
@pytest.mark.parametrize("reset_first", [False, True])
async def test_dry_run_copies_match_real_run(daily, reset_first):
    dry = await _trained(daily=daily)
    real = await _trained(daily=daily)
    assert _state(dry) == _state(real)

    engine = dry._retrain
    replay = engine._replay
    live_dead_zone = dry.learning._dead_zone_counts
    captured: dict = {}

    def _capturing_replay(*args, **kwargs):
        out = replay(*args, **kwargs)
        assert dry.learning._dead_zone_counts is not live_dead_zone
        captured["state"] = _state(dry)
        captured["dead_zone"] = copy.deepcopy(dry.learning._dead_zone_counts)
        return out

    engine._replay = _capturing_replay
    dry_result = await dry.retrain_from_history(dry_run=True, reset_first=reset_first)
    real_result = await real.retrain_from_history(reset_first=reset_first)

    assert captured["state"] == _state(real)
    assert captured["dead_zone"] == real.learning._dead_zone_counts
    reported = {k: v for k, v in dry_result.items() if k not in ("dry_run", "diff", "prediction_effect")}
    assert reported == {k: v for k, v in real_result.items() if k != "dry_run"}
    # The diff describes the real run's end state.
    assert dry_result["diff"]["learned_u_coefficient"]["after"] == (
        round(real._learned_u_coefficient, 4) if real._learned_u_coefficient is not None else None
    )
    assert dry_result["diff"]["global_base"]["buckets_after"] == sum(
        len(cell) for cell in real._correlation_data.values()
    )


async def test_prediction_effect_matches_the_real_models():
    dry = await _trained(daily=False)
    real = await _trained(daily=False)
    entries = real._retrain._prediction_entries(real._retrain._window_entries(None))
    before = real._retrain._predict_hours(entries)

    result = await dry.retrain_from_history(dry_run=True)
    await real.retrain_from_history()
    after = real._retrain._predict_hours(entries)

    effect = result["prediction_effect"]
    assert effect["hours"] == len(entries)
    assert effect["hours_skipped"] == 0
    assert effect["modelled_kwh_before"] == pytest.approx(sum(before), abs=1e-3)
    assert effect["modelled_kwh_after"] == pytest.approx(sum(after), abs=1e-3)
    assert effect["actual_kwh"] == pytest.approx(sum(e["actual_kwh"] for e in entries), abs=1e-3)
    residuals = [a - e["actual_kwh"] for a, e in zip(after, entries)]
    assert effect["fit"]["after"]["bias_kwh"] == pytest.approx(
        sum(residuals) / len(residuals), abs=1e-4
    )
    assert effect["fit"]["after"]["rmse_kwh"] == pytest.approx(
        math.sqrt(sum(r * r for r in residuals) / len(residuals)), abs=1e-4
    )
    assert sum(band["hours"] for band in effect["by_band"].values()) == len(entries)
    assert sum(day["hours"] for day in effect["by_day"]) == len(entries)


async def test_fit_uses_only_the_hours_the_retrain_learns_from():
    """Aux and poisoned hours are modelled but not fitted; an hour with
    energy in an excluded mode is fitted against the rest of its energy;
    an hour without a metered kWh is not compared at all."""
    coord = await _trained(daily=False)
    log = coord._hourly_log
    log[-1]["auxiliary_active"] = True
    log[-2]["learning_status"] = "skipped_no_data"
    log[-3]["unit_modes"] = {CABLE: MODE_DHW}
    log[-3]["unit_breakdown"] = {**log[-3]["unit_breakdown"], CABLE: 0.3}
    log[-4]["actual_kwh"] = None  # no actual: left out of the comparison

    effect = (await coord.retrain_from_history(dry_run=True))["prediction_effect"]

    assert effect["hours"] == len(log) - 1
    assert effect["hours_skipped"] == 1
    assert effect["fit"]["hours"] == len(log) - 3


async def test_energy_the_model_does_not_predict_is_not_compared_with_it():
    """Guest, off and hot-water energy is metered but never predicted (live
    learning and retrain take it off); comparing with the raw meter read it
    as a model shortfall.  It is taken off the actual and shown beside it."""
    coord = await _trained(daily=False)
    log = coord._hourly_log
    metered = sum(e["actual_kwh"] for e in log)
    guest_kwh = 0.0
    for entry in log[-48:]:
        entry["unit_modes"] = {HP: MODE_GUEST_HEATING}
        guest_kwh += entry["unit_breakdown"].get(HP, 0.0)
    assert guest_kwh > 1.0

    effect = (await coord.retrain_from_history(dry_run=True))["prediction_effect"]

    assert effect["excluded_mode_kwh"] == pytest.approx(guest_kwh, abs=1e-3)
    assert effect["actual_kwh"] == pytest.approx(metered - guest_kwh, abs=1e-3)
    assert sum(day["excluded_mode_kwh"] for day in effect["by_day"]) == pytest.approx(guest_kwh, abs=1e-2)


def test_prediction_modes_default_to_heating_not_the_current_mode():
    """The logged mode map is sparse; a unit missing from it was heating,
    whatever it is doing now."""
    coord = _coordinator(daily=False)
    coord._unit_modes = {HP: MODE_COOLING}
    entry = _log(days=1)[0]
    entry["unit_modes"] = {CABLE: MODE_COOLING}

    kwargs = coord._retrain._prediction_kwargs(entry)

    assert kwargs["unit_modes"] == {HP: MODE_HEATING, CABLE: MODE_COOLING}
    assert kwargs["temp"] == entry["inertia_temp"]


# ---------------------------------------------------------------------------
# Isolation from live learning
# ---------------------------------------------------------------------------

def test_replay_cannot_yield_to_the_event_loop():
    """The dry run's window is the synchronous replay: an ``await`` in it
    would let an hour boundary learn into the copies, and its hour would be
    lost when the live model goes back."""
    for name in ("_replay", "_dry_run", "_predict_hours", "_prediction_effect"):
        assert not inspect.iscoroutinefunction(getattr(RetrainEngine, name)), name


async def test_live_write_during_dry_run_lands_in_live_state():
    coord = await _trained(daily=False)
    live_corr = coord._correlation_data
    replay_solar = coord.learning.replay_solar_nlms

    def _replay_solar_scheduling_a_live_hour(*args, **kwargs):
        # A live hour boundary queued while the copies are installed.
        def _live_hour():
            coord._correlation_data.setdefault("42", {})["normal"] = 9.9

        asyncio.get_running_loop().call_soon(_live_hour)
        return replay_solar(*args, **kwargs)

    coord.learning.replay_solar_nlms = _replay_solar_scheduling_a_live_hour
    await coord.retrain_from_history(dry_run=True, reset_first=True)
    await asyncio.sleep(0)

    assert coord._correlation_data is live_corr
    assert live_corr["42"]["normal"] == 9.9


async def test_exception_in_the_replay_reinstalls_the_live_model():
    coord = await _trained(daily=False)
    live_objects = {attr: getattr(coord, attr) for attr in _RETRAIN_MODEL_STATE}
    state_before = _state(coord)
    coord.learning.replay_solar_nlms = MagicMock(side_effect=RuntimeError("boom"))

    with pytest.raises(RuntimeError):
        await coord.retrain_from_history(dry_run=True, reset_first=True)

    assert _state(coord) == state_before
    for attr, obj in live_objects.items():
        if isinstance(obj, dict):
            assert getattr(coord, attr) is obj, attr


# ---------------------------------------------------------------------------
# Track B COP smearing: fetched before the replay, stored only for real
# ---------------------------------------------------------------------------

async def test_cop_params_fetch_does_not_await_while_disabled():
    coord = _coordinator(daily=True)
    coord.mpc_entry_id = "mpc"
    coord.hass.services.async_call = AsyncMock()

    assert await coord._fetch_track_b_cop_params() is None
    coord.hass.services.async_call.assert_not_awaited()


@pytest.mark.parametrize("dry_run", [False, True])
async def test_cop_smeared_distribution_is_stored_only_by_a_real_run(monkeypatch, dry_run):
    monkeypatch.setattr(const, "ENABLE_TRACK_B_COP_SMEARING", True)
    coord = await _trained(daily=True)
    coord.mpc_entry_id = "mpc"
    coord._last_cop_params = None
    cop_params = {"eta_carnot": 0.42, "lwt": 35.0, "f_defrost": 0.85}
    coord.hass.services.async_call = AsyncMock(return_value=cop_params)
    sentinel = [{"hour": 0, "synthetic_kwh_el": 1.0}]
    coord._daily_processor.track_b_cop_distribution = MagicMock(return_value=sentinel)
    coord._daily_processor.apply_strategies_to_global_model = MagicMock(return_value=24)

    result = await coord.retrain_from_history(experimental_cop_smear=True, dry_run=dry_run)

    assert result["status"] == "completed"
    assert result["days_processed"] > 0
    smeared_days = [
        day for day, record in coord._daily_history.items()
        if "track_b_cop_distribution" in record
    ]
    if dry_run:
        assert smeared_days == []
        assert coord._last_cop_params is None
    else:
        assert len(smeared_days) == result["days_processed"]
        assert coord._last_cop_params == cop_params


# ---------------------------------------------------------------------------
# Diff contents
# ---------------------------------------------------------------------------

def test_bucket_map_diff_counts_and_bands():
    before = {
        "-3": {"normal": 1.0, "high_wind": 1.2},
        "2": {"normal": 0.8},
        "7": {"normal": 0.5},
    }
    after = {
        "-3": {"normal": 1.1, "high_wind": 1.2},  # changed +0.1, unchanged
        "2": {"normal": 0.6, "high_wind": 0.9},  # changed −0.2, added
        "12": {"normal": 0.2},  # added; "7" removed
    }

    diff = _bucket_map_diff(before, after)

    assert (diff["buckets_before"], diff["buckets_after"]) == (4, 5)
    assert (diff["added"], diff["removed"], diff["changed"]) == (2, 1, 2)
    assert diff["mean_delta"] == pytest.approx(-0.05)
    assert diff["mean_abs_delta"] == pytest.approx(0.15)
    assert diff["max_abs_delta"] == pytest.approx(0.2)
    assert list(diff["by_band"]) == ["-5..0", "0..5", "5..10", "10..15"]
    assert diff["by_band"]["-5..0"]["normal"]["mean_delta"] == pytest.approx(0.1)
    assert "high_wind" not in diff["by_band"]["-5..0"]
    assert diff["by_band"]["0..5"]["high_wind"]["added"] == 1
    assert diff["by_band"]["5..10"]["normal"]["removed"] == 1
    assert diff["by_band"]["10..15"]["normal"]["added"] == 1


def test_per_unit_diff_ranks_units_by_movement():
    before = {HP: {"0": {"normal": 1.0}}, CABLE: {"0": {"normal": 0.2}}}
    after = {HP: {"0": {"normal": 1.05}}, CABLE: {"0": {"normal": 0.5}}}

    diff = _per_unit_map_diff(before, after)

    assert diff["units_by_movement"] == [CABLE, HP]
    assert diff["by_unit"][CABLE]["mean_delta"] == pytest.approx(0.3)


def test_solar_diff_reports_changed_regimes_only():
    before = {HP: {"heating": {"s": 0.3, "e": 0.1, "w": 0.0}, "cooling": {"s": 0.4, "e": 0.0, "w": 0.0}}}
    after = {HP: {"heating": {"s": 0.35, "e": 0.1, "w": 0.0}, "cooling": {"s": 0.4, "e": 0.0, "w": 0.0}}}

    diff = _solar_coefficient_diff(before, after)

    assert diff == {HP: {"heating": {
        "before": {"s": 0.3, "e": 0.1, "w": 0.0},
        "after": {"s": 0.35, "e": 0.1, "w": 0.0},
    }}}
