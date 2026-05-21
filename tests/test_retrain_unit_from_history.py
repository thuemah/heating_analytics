"""Tests for the ``retrain_unit_from_history`` service.

Targeted per-unit base coefficient retrain from the hourly log, with
nudge / reset / dry_run semantics.  Implementation lives in
``RetrainEngine.retrain_unit_from_history`` and
``LearningManager.replay_per_unit_models(target_entity=..., dry_run=...)``.
"""
from __future__ import annotations

import copy

import pytest

from custom_components.heating_analytics.learning import LearningManager
from custom_components.heating_analytics.observation import (
    DirectMeter,
    ModelState,
)
from custom_components.heating_analytics.retrain import RetrainEngine


class _StubCoordinator:
    """Minimal coordinator stub for RetrainEngine.retrain_unit_from_history.

    Real-coordinator surface we exercise: ``energy_sensors``,
    ``_hourly_log``, ``_correlation_data_per_unit``,
    ``_learning_buffer_per_unit``, ``learning``, ``_unit_strategies``,
    ``learning_rate``, ``get_model_state()``, ``_async_save_data()``.
    """

    def __init__(self, energy_sensors: list[str], hourly_log: list[dict]) -> None:
        self.energy_sensors = list(energy_sensors)
        self._hourly_log = list(hourly_log)
        self._correlation_data_per_unit: dict = {}
        self._learning_buffer_per_unit: dict = {}
        self.learning = LearningManager()
        self._unit_strategies = {sid: DirectMeter(sid) for sid in energy_sensors}
        self.learning_rate = 0.10
        self.wind_threshold = 8.0
        self.extreme_wind_threshold = 10.8
        self.save_calls = 0

    def _get_wind_bucket(self, effective_wind: float) -> str:
        if effective_wind >= self.extreme_wind_threshold:
            return "extreme_wind"
        if effective_wind >= self.wind_threshold:
            return "high_wind"
        return "normal"

    def get_model_state(self) -> ModelState:
        return ModelState(
            correlation_data={},
            correlation_data_per_unit=self._correlation_data_per_unit,
            observation_counts={},
            aux_coefficients={},
            aux_coefficients_per_unit={},
            solar_coefficients_per_unit={},
            learned_u_coefficient=None,
            learning_buffer_per_unit=self._learning_buffer_per_unit,
        )

    async def _async_save_data(self, force: bool = False) -> None:
        self.save_calls += 1


def _entry(
    *,
    timestamp: str,
    entity_id: str,
    actual_kwh: float,
    temp_key: str = "8",
    wind_bucket: str = "normal",
    extra_breakdown: dict | None = None,
) -> dict:
    breakdown = {entity_id: actual_kwh}
    if extra_breakdown:
        breakdown.update(extra_breakdown)
    return {
        "timestamp": timestamp,
        "temp_key": temp_key,
        "wind_bucket": wind_bucket,
        "unit_breakdown": breakdown,
        "unit_modes": {sid: "heating" for sid in breakdown},
        "auxiliary_active": False,
        "learning_status": "ok",
    }


@pytest.mark.asyncio
async def test_dry_run_does_not_mutate_state():
    entity = "sensor.unit_a"
    coord = _StubCoordinator(
        energy_sensors=[entity],
        hourly_log=[
            _entry(timestamp="2026-04-01T00:00:00", entity_id=entity, actual_kwh=2.0)
            for _ in range(10)
        ],
    )
    coord._correlation_data_per_unit[entity] = {"8": {"normal": 0.30}}
    before_corr = copy.deepcopy(coord._correlation_data_per_unit)
    before_buf = copy.deepcopy(coord._learning_buffer_per_unit)

    engine = RetrainEngine(coord)
    result = await engine.retrain_unit_from_history(
        entity_id=entity, reset_first=False, dry_run=True,
    )

    assert result["status"] == "ok"
    assert result["dry_run"] is True
    assert result["entries_processed"] == 10
    assert result["buckets_modified"] >= 1
    assert "8/normal" in result["diff_summary"]
    # State unchanged.
    assert coord._correlation_data_per_unit == before_corr
    assert coord._learning_buffer_per_unit == before_buf
    # Dry-run does not persist.
    assert coord.save_calls == 0


@pytest.mark.asyncio
async def test_nudge_shifts_buckets_toward_historical_mean():
    entity = "sensor.unit_a"
    # Bucket starts too low; observations are consistently 2.0 — EMA
    # should pull bucket upward but not all the way (partial convergence).
    coord = _StubCoordinator(
        energy_sensors=[entity],
        hourly_log=[
            _entry(timestamp=f"2026-04-{d:02d}T00:00:00", entity_id=entity, actual_kwh=2.0)
            for d in range(1, 11)
        ],
    )
    coord._correlation_data_per_unit[entity] = {"8": {"normal": 0.30}}

    engine = RetrainEngine(coord)
    result = await engine.retrain_unit_from_history(
        entity_id=entity, reset_first=False, dry_run=False,
    )

    assert result["status"] == "ok"
    after = coord._correlation_data_per_unit[entity]["8"]["normal"]
    # Pulled upward from 0.30, but EMA at lr=0.10 over 10 steps doesn't
    # reach 2.0.
    assert after > 0.30
    assert after < 2.0
    assert coord.save_calls == 1


@pytest.mark.asyncio
async def test_reset_first_clears_only_target_entity_state():
    entity_a = "sensor.unit_a"
    entity_b = "sensor.unit_b"
    coord = _StubCoordinator(
        energy_sensors=[entity_a, entity_b],
        hourly_log=[
            _entry(
                timestamp=f"2026-04-{d:02d}T00:00:00",
                entity_id=entity_a,
                actual_kwh=1.5,
                extra_breakdown={entity_b: 1.0},
            )
            for d in range(1, 8)
        ],
    )
    coord._correlation_data_per_unit[entity_a] = {"8": {"normal": 0.50}}
    coord._correlation_data_per_unit[entity_b] = {"8": {"normal": 0.75}}
    coord._learning_buffer_per_unit[entity_a] = {"8": {"normal": [0.4, 0.5]}}
    coord._learning_buffer_per_unit[entity_b] = {"8": {"normal": [0.7]}}

    engine = RetrainEngine(coord)
    result = await engine.retrain_unit_from_history(
        entity_id=entity_a, reset_first=True, dry_run=False,
    )

    assert result["status"] == "ok"
    # entity_b state is fully untouched.
    assert coord._correlation_data_per_unit[entity_b] == {"8": {"normal": 0.75}}
    assert coord._learning_buffer_per_unit[entity_b] == {"8": {"normal": [0.7]}}


@pytest.mark.asyncio
async def test_unknown_entity_returns_error_status():
    coord = _StubCoordinator(energy_sensors=["sensor.real"], hourly_log=[])
    engine = RetrainEngine(coord)
    result = await engine.retrain_unit_from_history(
        entity_id="sensor.not_in_config", reset_first=False, dry_run=False,
    )
    assert result["status"] == "unknown_entity"
    assert result["entries_processed"] == 0
    assert coord.save_calls == 0


@pytest.mark.asyncio
async def test_entity_filter_isolation():
    """Retraining one entity must not touch the other entity's buckets."""
    entity_a = "sensor.unit_a"
    entity_b = "sensor.unit_b"
    coord = _StubCoordinator(
        energy_sensors=[entity_a, entity_b],
        hourly_log=[
            _entry(
                timestamp=f"2026-04-{d:02d}T00:00:00",
                entity_id=entity_a,
                actual_kwh=2.0,
                extra_breakdown={entity_b: 0.9},
            )
            for d in range(1, 8)
        ],
    )
    coord._correlation_data_per_unit[entity_b] = {"8": {"normal": 0.42}}

    engine = RetrainEngine(coord)
    await engine.retrain_unit_from_history(
        entity_id=entity_a, reset_first=False, dry_run=False,
    )

    # entity_a populated by replay; entity_b unchanged.
    assert entity_a in coord._correlation_data_per_unit
    assert coord._correlation_data_per_unit[entity_b] == {"8": {"normal": 0.42}}


@pytest.mark.asyncio
async def test_days_back_window_filters_log_by_timestamp():
    """``days_back`` must restrict the replay window to the most recent N days."""
    entity = "sensor.unit_a"
    from datetime import timedelta, datetime, timezone
    from unittest.mock import patch

    fixed_now = datetime(2026, 5, 17, 12, 0, tzinfo=timezone.utc)

    # Old entries (200 days ago) at one value; recent entries (last 15 days)
    # at a clearly distinct value.  Pre-seed the bucket at 0.50.
    old_entries = [
        _entry(
            timestamp=(fixed_now - timedelta(days=200 - i)).strftime("%Y-%m-%dT%H:%M:%S"),
            entity_id=entity,
            actual_kwh=5.0,
        )
        for i in range(5)
    ]
    recent_entries = [
        _entry(
            timestamp=(fixed_now - timedelta(days=15 - i)).strftime("%Y-%m-%dT%H:%M:%S"),
            entity_id=entity,
            actual_kwh=1.0,
        )
        for i in range(10)
    ]
    coord = _StubCoordinator(
        energy_sensors=[entity],
        hourly_log=old_entries + recent_entries,
    )
    coord._correlation_data_per_unit[entity] = {"8": {"normal": 0.50}}

    engine = RetrainEngine(coord)
    # Scoped patch so dt_util.now is restored after this test — naive
    # ``dt_util.now.return_value = ...`` leaks the mock into subsequent
    # tests (caught the 65-failure regression in solar_diagnose tests).
    with patch("custom_components.heating_analytics.retrain.dt_util") as mock_dt:
        mock_dt.now.return_value = fixed_now
        result = await engine.retrain_unit_from_history(
            entity_id=entity, reset_first=False, dry_run=False, days_back=30,
        )

    # Only 10 entries (last 15 days) should have been replayed; the 200-day-old
    # entries fall outside the window.
    assert result["entries_processed"] == 10
    after = coord._correlation_data_per_unit[entity]["8"]["normal"]
    # Recent entries pull bucket toward 1.0 (above 0.50 seed).  If the
    # old 5.0 entries had leaked in, the bucket would have shot well past 1.0.
    assert 0.50 < after < 1.0


@pytest.mark.asyncio
async def test_no_data_returns_status_when_log_empty_for_entity():
    """Log has entries, but none have positive breakdown for the target entity."""
    entity_a = "sensor.unit_a"
    entity_b = "sensor.unit_b"
    coord = _StubCoordinator(
        energy_sensors=[entity_a, entity_b],
        hourly_log=[
            _entry(timestamp="2026-04-01T00:00:00", entity_id=entity_b, actual_kwh=1.0),
        ],
    )
    engine = RetrainEngine(coord)
    result = await engine.retrain_unit_from_history(
        entity_id=entity_a, reset_first=False, dry_run=False,
    )
    assert result["status"] == "no_data"
    assert result["entries_processed"] == 0
    assert coord.save_calls == 0


def test_replay_per_unit_models_target_entity_isolation():
    """Unit test of the LearningManager extension: ``target_entity`` filter."""
    manager = LearningManager()
    strategies = {
        "sensor.a": DirectMeter("sensor.a"),
        "sensor.b": DirectMeter("sensor.b"),
    }
    model = ModelState(
        correlation_data={},
        correlation_data_per_unit={},
        observation_counts={},
        aux_coefficients={},
        aux_coefficients_per_unit={},
        solar_coefficients_per_unit={},
        learned_u_coefficient=None,
    )
    entries = [
        {
            "temp_key": "8",
            "wind_bucket": "normal",
            "unit_modes": {"sensor.a": "heating", "sensor.b": "heating"},
            "unit_breakdown": {"sensor.a": 1.0, "sensor.b": 2.0},
        }
        for _ in range(5)
    ]
    result = manager.replay_per_unit_models(
        entries, strategies, model, learning_rate=1.0, target_entity="sensor.a",
    )
    # Only sensor.a got writes.
    assert "sensor.a" in model.correlation_data_per_unit
    assert "sensor.b" not in model.correlation_data_per_unit
    # Diagnostic returned because target_entity was set.
    assert result is not None
    assert result["entries_processed"] == 5
    assert result["buckets_changed"] >= 1


def test_replay_per_unit_models_dry_run_isolated_from_state():
    manager = LearningManager()
    strategies = {"sensor.a": DirectMeter("sensor.a")}
    model = ModelState(
        correlation_data={},
        correlation_data_per_unit={"sensor.a": {"8": {"normal": 0.50}}},
        observation_counts={},
        aux_coefficients={},
        aux_coefficients_per_unit={},
        solar_coefficients_per_unit={},
        learned_u_coefficient=None,
        learning_buffer_per_unit={},
    )
    before = copy.deepcopy(model.correlation_data_per_unit)
    entries = [
        {
            "temp_key": "8",
            "wind_bucket": "normal",
            "unit_modes": {"sensor.a": "heating"},
            "unit_breakdown": {"sensor.a": 2.0},
        }
        for _ in range(5)
    ]
    result = manager.replay_per_unit_models(
        entries, strategies, model, learning_rate=1.0,
        target_entity="sensor.a", dry_run=True,
    )
    assert model.correlation_data_per_unit == before
    assert result["buckets_changed"] >= 1
    summary = result["diff_summary"]["8/normal"]
    assert summary["before"] == 0.50
    assert summary["after"] > 0.50
    assert summary["delta"] == summary["after"] - summary["before"]


def test_replay_per_unit_models_reset_first_clears_before_replay():
    """``reset_first=True`` clears the target entity's slice before the
    replay loop runs, so a buffer-jumpstart fills from scratch instead
    of EMA-blending with stale prior.  Verified by feeding observations
    where the seed value is far from the observation mean: with reset,
    the post-replay bucket should land near the observation mean (LEARNING_BUFFER_THRESHOLD
    samples averaged); without reset, EMA at lr=1.0 lands on the last
    observation (which equals observation mean only by construction here)
    — so we use lr=0.1 to distinguish.
    """
    manager = LearningManager()
    strategies = {"sensor.a": DirectMeter("sensor.a")}
    model = ModelState(
        correlation_data={},
        correlation_data_per_unit={"sensor.a": {"8": {"normal": 5.0}}},  # far-off seed
        observation_counts={},
        aux_coefficients={},
        aux_coefficients_per_unit={},
        solar_coefficients_per_unit={},
        learned_u_coefficient=None,
        learning_buffer_per_unit={},
    )
    entries = [
        {
            "temp_key": "8",
            "wind_bucket": "normal",
            "unit_modes": {"sensor.a": "heating"},
            "unit_breakdown": {"sensor.a": 1.0},
        }
        for _ in range(10)
    ]
    result = manager.replay_per_unit_models(
        entries, strategies, model, learning_rate=0.1,
        target_entity="sensor.a", reset_first=True,
    )
    # Reset wiped seed=5.0; the first LEARNING_BUFFER_THRESHOLD samples
    # buffer-fill and average to ~1.0; remaining samples EMA toward 1.0
    # from 1.0 (no-op).  Bucket lands near 1.0.
    after = model.correlation_data_per_unit["sensor.a"]["8"]["normal"]
    assert abs(after - 1.0) < 0.01, (
        f"reset_first should have wiped the 5.0 seed; bucket={after}"
    )
    assert result["buckets_changed"] >= 1


def test_replay_per_unit_models_reset_first_in_dry_run_reports_reset_diff():
    """P2 regression pin: ``reset_first=True, dry_run=True`` must report
    the diff of a reset-then-replay, NOT a nudge-on-top-of-prior diff.

    Pre-fix: ``reset_first`` was honoured only on live state, never on
    dry-run copies, so dry-run + reset_first produced the misleading
    nudge diff.  This test would have failed before the fix.
    """
    manager = LearningManager()
    strategies = {"sensor.a": DirectMeter("sensor.a")}
    model = ModelState(
        correlation_data={},
        correlation_data_per_unit={"sensor.a": {"8": {"normal": 5.0}}},
        observation_counts={},
        aux_coefficients={},
        aux_coefficients_per_unit={},
        solar_coefficients_per_unit={},
        learned_u_coefficient=None,
        learning_buffer_per_unit={},
    )
    before = copy.deepcopy(model.correlation_data_per_unit)
    entries = [
        {
            "temp_key": "8",
            "wind_bucket": "normal",
            "unit_modes": {"sensor.a": "heating"},
            "unit_breakdown": {"sensor.a": 1.0},
        }
        for _ in range(10)
    ]
    result = manager.replay_per_unit_models(
        entries, strategies, model, learning_rate=0.1,
        target_entity="sensor.a", reset_first=True, dry_run=True,
    )
    # Live state untouched (dry_run guarantee).
    assert model.correlation_data_per_unit == before
    # Diff reports reset-then-replay: ``before`` in the summary is the
    # pre-reset seed (5.0), ``after`` is the post-reset-replay value
    # (~1.0).  If reset_first had been ignored in dry_run, ``after`` would
    # have been the nudge result: 5.0 × 0.9^10 + 1.0 × (1−0.9^10) ≈ 2.39.
    summary = result["diff_summary"]["8/normal"]
    assert summary["before"] == 5.0
    assert abs(summary["after"] - 1.0) < 0.01, (
        f"dry_run + reset_first should report reset-then-replay diff "
        f"(after ≈ 1.0); got after={summary['after']}.  If after is "
        f"around 2.39, reset_first was ignored in dry_run mode (P2 bug)."
    )
