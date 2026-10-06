"""The hourly log records which units reported, and keeps per-unit fields
consistent — the evidence the per-unit retrain replay reads.

- ``units_reporting`` tells a reported zero hour (live learns 0) from an
  offline sensor (live skips the unit); ``unit_breakdown`` drops both.
- Under daily learning, a dual-interference hour blocks per-unit learning as
  it does under hourly learning, so its ``skipped_dual_interference`` label
  is true for replays.
- Replacing a sensor renames the unit in every per-unit field of the log.
- ``retrain_from_history`` replays the per-unit models once over every
  entry, on both learning tracks.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.daily_processor import (
    DailyProcessor,
    per_unit_replay_context,
)
from custom_components.heating_analytics.observation import DirectMeter
from custom_components.heating_analytics.retrain import RetrainEngine
from tests.test_track_c_outage_skip import _day_entries, _retrain_coord


def _coordinator(hass, *, daily_learning_mode=False):
    entry = MagicMock()
    entry.data = {
        "balance_point": 17.0,
        "learning_rate": 0.1,
        "energy_sensors": ["sensor.heater", "sensor.cable"],
        "solar_enabled": True,
        "aux_affected_entities": ["sensor.heater"],
    }
    with patch("custom_components.heating_analytics.storage.Store") as mock_store_cls:
        mock_store = mock_store_cls.return_value
        mock_store.async_load = AsyncMock(return_value={})
        mock_store.async_save = AsyncMock()
        coordinator = HeatingDataCoordinator(hass, entry)
    coordinator._async_save_data = AsyncMock()
    coordinator.daily_learning_mode = daily_learning_mode
    return coordinator


def _prime_hour(coordinator, *, aux: bool, delta: dict):
    coordinator._collector.sample_count = 60
    coordinator._collector.temp_sum = 0.0
    coordinator._collector.wind_values = [0.0] * 60
    coordinator._collector.bucket_counts = {"normal": 60, "high_wind": 0, "extreme_wind": 0}
    coordinator._collector.solar_sum = 1.0
    coordinator._collector.aux_count = 60 if aux else 0
    coordinator.auxiliary_heating_active = aux
    coordinator._collector.aux_impact_hour = 1.0 if aux else 0.0
    coordinator._collector.energy_hour = sum(delta.values())
    coordinator._hourly_delta_per_unit = dict(delta)
    coordinator._collector.start_time = datetime(2023, 10, 27, 12, 0, 0)


@pytest.mark.asyncio
async def test_log_records_reporting_units_including_zero_hours(hass):
    coordinator = _coordinator(hass)
    coordinator._correlation_data = {"0": {"normal": 2.0}}
    coordinator._correlation_data_per_unit = {"sensor.heater": {"0": {"normal": 2.0}}}
    _prime_hour(coordinator, aux=False, delta={"sensor.heater": 1.5, "sensor.cable": 0.0})

    await coordinator._process_hourly_data(datetime(2023, 10, 27, 13, 0, 0))

    entry = coordinator._hourly_log[-1]
    assert entry["unit_breakdown"] == {"sensor.heater": 1.5}
    assert entry["units_reporting"] == ["sensor.cable", "sensor.heater"]


@pytest.mark.asyncio
async def test_offline_sensor_is_absent_from_units_reporting(hass):
    coordinator = _coordinator(hass)
    _prime_hour(coordinator, aux=False, delta={"sensor.heater": 1.5})

    await coordinator._process_hourly_data(datetime(2023, 10, 27, 13, 0, 0))

    assert coordinator._hourly_log[-1]["units_reporting"] == ["sensor.heater"]


@pytest.mark.asyncio
async def test_dual_interference_blocks_per_unit_learning_under_daily_learning(hass):
    """Sun and aux both significant: per-unit learning is blocked on the
    daily track as on the hourly one, matching the logged label."""
    coordinator = _coordinator(hass, daily_learning_mode=True)
    coordinator._correlation_data = {"0": {"normal": 2.0}}
    coordinator._correlation_data_per_unit = {
        "sensor.heater": {"0": {"normal": 2.0}},
        "sensor.cable": {"0": {"normal": 0.5}},
    }
    coordinator._aux_coefficients_per_unit = {"sensor.heater": {"0": {"normal": 0.5}}}
    coordinator.solar.calculate_unit_coefficient = MagicMock(return_value={"s": 1.0, "e": 0.0, "w": 0.0})
    coordinator.solar.calculate_unit_solar_impact = MagicMock(return_value=0.5)
    _prime_hour(coordinator, aux=True, delta={"sensor.heater": 0.8, "sensor.cable": 0.1})

    await coordinator._process_hourly_data(datetime(2023, 10, 27, 13, 0, 0))

    entry = coordinator._hourly_log[-1]
    assert entry["learning_status"] == "skipped_dual_interference"
    assert coordinator._aux_coefficients_per_unit["sensor.heater"]["0"]["normal"] == 0.5
    assert coordinator._correlation_data_per_unit["sensor.cable"]["0"]["normal"] == 0.5
    assert coordinator._learning_buffer_per_unit == {}


@pytest.mark.asyncio
async def test_replacing_a_sensor_renames_it_in_every_per_unit_log_field(hass):
    coordinator = _coordinator(hass)
    old, new = "sensor.heater", "sensor.heater_v2"
    coordinator._hourly_log = [{
        "timestamp": "2023-10-27T12:00:00",
        "unit_breakdown": {old: 1.0, "sensor.cable": 0.2},
        "unit_expected_breakdown": {old: 0.9},
        "unit_expected_base": {old: 1.1},
        "unit_modes": {old: "guest_heating"},
        "units_reporting": ["sensor.cable", old],
        "solar_dominant_entities": [old],
        "aux_cooldown_entities": [old],
    }]

    assert await coordinator.async_replace_sensor_source(old, new) is True

    entry = coordinator._hourly_log[0]
    assert entry["unit_breakdown"] == {"sensor.cable": 0.2, new: 1.0}
    assert entry["unit_expected_breakdown"] == {new: 0.9}
    assert entry["unit_expected_base"] == {new: 1.1}
    assert entry["unit_modes"] == {new: "guest_heating"}
    assert entry["units_reporting"] == ["sensor.cable", new]
    assert entry["solar_dominant_entities"] == [new]
    assert entry["aux_cooldown_entities"] == [new]


# ---------------------------------------------------------------------------
# retrain_from_history wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_daily_retrain_replays_per_unit_once_over_every_entry():
    """The per-unit replay no longer rides on the day loop: a day the global
    pass skips (short of hours here) is still replayed per unit."""
    full_day = _day_entries("2026-04-10")
    short_day = _day_entries("2026-04-11")[:10]
    coord = _retrain_coord(full_day + short_day, track_c_enabled=False)

    result = await RetrainEngine(coord).retrain_from_history(days_back=None)

    assert coord._replay_per_unit_models.call_count == 1
    (replayed,), _ = coord._replay_per_unit_models.call_args
    assert len(replayed) == 34
    assert "per_unit_replay" in result


@pytest.mark.asyncio
async def test_track_c_day_without_distribution_is_still_replayed_per_unit():
    entries = _day_entries("2026-04-10")
    coord = _retrain_coord(entries, track_c_enabled=True, daily_history={"2026-04-10": {"kwh": 12.0}})

    result = await RetrainEngine(coord).retrain_from_history(days_back=None)

    assert result["days_skipped_mpc_unavailable"] == 1
    (replayed,), _ = coord._replay_per_unit_models.call_args
    assert len(replayed) == 24


def test_retrain_wrapper_replays_aux_and_reads_coordinator_state():
    """``retrain_from_history``'s wrapper rebuilds per-unit aux too (its
    ``reset_first`` clears them) and hands the replay the coordinator's
    state — conservatively where the coordinator is a mock."""
    coord = MagicMock()
    coord._unit_strategies = {"sensor.heater": DirectMeter("sensor.heater")}
    coord.learning_rate = 0.05
    coord.balance_point = 16.0
    coord.aux_affected_entities = ["sensor.heater"]
    coord._hourly_log = [{"timestamp": "2026-04-10T00:00:00", "unit_breakdown": {"sensor.heater": 1.0}}]
    coord.learning.replay_per_unit_models = MagicMock(return_value={"entries_processed": 1})

    DailyProcessor(coord).replay_per_unit_models(coord._hourly_log)

    kwargs = coord.learning.replay_per_unit_models.call_args.kwargs
    assert kwargs["replay_aux"] is True
    assert kwargs["balance_point"] == 16.0
    assert kwargs["aux_affected_entities"] == {"sensor.heater"}
    assert set(kwargs["first_reported"]) == {"sensor.heater"}
    # Mock attributes fall back to the conservative reading.
    assert kwargs["solar_enabled"] is False
    assert kwargs["solar_calculator"] is None
    assert kwargs["get_prediction_from_model"] is None
    assert kwargs["unit_min_base"] is None
    # Learning mode unknown → the replay takes the DirectMeter strategies.
    assert kwargs["hourly_sensors"] is None


def test_replay_context_uses_the_real_model_lookup(hass):
    coordinator = _coordinator(hass)
    context = per_unit_replay_context(coordinator)
    assert context["get_prediction_from_model"] == coordinator.statistics._get_prediction_from_model
    assert context["solar_calculator"] is coordinator.solar
    assert context["balance_point"] == 17.0


def test_replay_context_takes_the_sensors_live_learns_hourly(hass):
    """Under daily learning the MPC (WeightedSmear) unit is neither learned
    nor counted in the SNR weight by live learning; the replay gets the same
    list ``hourly_processor`` hands live learning."""
    from custom_components.heating_analytics.observation import WeightedSmear

    coordinator = _coordinator(hass, daily_learning_mode=True)
    coordinator._unit_strategies = {
        "sensor.heater": DirectMeter("sensor.heater"),
        "sensor.cable": WeightedSmear("sensor.cable", use_synthetic=True),
    }
    assert per_unit_replay_context(coordinator)["hourly_sensors"] == ["sensor.heater"]

    coordinator.daily_learning_mode = False
    assert per_unit_replay_context(coordinator)["hourly_sensors"] == [
        "sensor.heater", "sensor.cable",
    ]
