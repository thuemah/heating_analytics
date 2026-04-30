"""Tests for the coordinator-side wiring of batch_fit_solar (#884).

The unit tests in ``test_batch_fit_solar.py`` cover
``LearningManager.batch_fit_solar_coefficients`` directly.  This file
covers the coordinator wrapper ``async_batch_fit_solar`` and its
side-effects:

- ``_last_batch_fit_per_unit`` is populated keyed by entity_id with a
  timestamp + per-(regime) diagnostic block.
- ``_async_save_data(force=True)`` is awaited only when at least one
  regime applied (the ``applied_any`` gate).
- The return contract carries ``status``, ``unit_entity_id``,
  ``timestamp``, ``applied_count``, ``skipped_count``, and ``per_unit``.
- The diagnose_solar ``last_batch_fit`` field surfaces the same data.
- Top-level skips (e.g. WeightedSmear, unknown_entity) bypass the
  ``_last_batch_fit_per_unit`` write.
"""
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

import pytest

from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.const import MODE_HEATING
from tests.helpers import stratified_coeff


def _entry(ts, *, sensor_id="sensor.heater1", solar_s=0.5, actual_kwh=1.5):
    return {
        "timestamp": ts,
        "hour": int(ts[11:13]),
        "temp": 10.0,
        "temp_key": "10",
        "wind_bucket": "normal",
        "solar_factor": solar_s,
        "solar_vector_s": solar_s,
        "solar_vector_e": 0.0,
        "solar_vector_w": 0.0,
        "correction_percent": 100.0,
        "auxiliary_active": False,
        "actual_kwh": actual_kwh,
        "unit_modes": {sensor_id: MODE_HEATING},
        "unit_breakdown": {sensor_id: actual_kwh},
        "solar_dominant_entities": [],
        "learning_status": "logged",
    }


def _make_coord(*, with_log: bool = True):
    """Real HeatingDataCoordinator with mocked HA + a synthetic log."""
    hass = MagicMock()
    hass.config = MagicMock()
    hass.config.latitude = 60.0
    hass.config.longitude = 10.0
    hass.is_running = True
    hass.states = MagicMock()
    hass.data = {"heating_analytics": {}}
    hass.services = MagicMock()
    hass.async_add_executor_job = AsyncMock()

    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.data = {
        "energy_sensors": ["sensor.heater1"],
        "balance_point": 15.0,
        "solar_enabled": True,
        "csv_auto_logging": False,
    }

    with patch("custom_components.heating_analytics.storage.Store"):
        coord = HeatingDataCoordinator(hass, entry)
        # Inject synthetic log + base for the fit to find samples.
        if with_log:
            true_coeff = {"s": 1.0, "e": 0.0, "w": 0.0}
            entries = []
            for i in range(40):
                impact = true_coeff["s"] * 0.5
                actual = max(0.0, 2.0 - impact)
                entries.append(_entry(
                    f"2026-04-{10 + i // 24:02d}T{i % 24:02d}:00:00",
                    actual_kwh=actual,
                ))
            coord._hourly_log = entries
        else:
            coord._hourly_log = []
        coord._correlation_data_per_unit = {
            "sensor.heater1": {"10": {"normal": 2.0}}
        }
        # Async save is mocked so we can spy on it.
        coord._async_save_data = AsyncMock()
        return coord


@pytest.mark.asyncio
async def test_async_batch_fit_solar_populates_last_batch_fit_dict():
    """``_last_batch_fit_per_unit`` carries timestamp + regimes after a fit."""
    coord = _make_coord()
    result = await coord.async_batch_fit_solar()
    assert "sensor.heater1" in coord._last_batch_fit_per_unit
    entry = coord._last_batch_fit_per_unit["sensor.heater1"]
    # Timestamp is dt_util.utcnow().isoformat() — under the conftest
    # MagicMock for homeassistant.util.dt, the value is a MagicMock too.
    # We only assert presence here; format validation happens implicitly
    # via the real HA runtime.
    assert "timestamp" in entry
    assert "regimes" in entry
    assert "heating" in entry["regimes"]
    assert "cooling" in entry["regimes"]
    # Heating regime fit succeeded; cooling skipped (no cooling samples).
    heating = entry["regimes"]["heating"]
    assert heating["sample_count"] == 40
    assert heating.get("applied") is True


@pytest.mark.asyncio
async def test_async_batch_fit_solar_persists_when_any_regime_applies():
    """``_async_save_data(force=True)`` awaited only when something applied."""
    coord = _make_coord()
    await coord.async_batch_fit_solar()
    coord._async_save_data.assert_awaited_once()
    # The save call uses force=True so the debounce window is bypassed.
    args, kwargs = coord._async_save_data.await_args
    assert kwargs.get("force") is True or (args and args[0] is True)


@pytest.mark.asyncio
async def test_async_batch_fit_solar_persists_skip_records():
    """Below-threshold log → skip-records written → save IS triggered.

    Pre-#902 the save gate was ``applied_any`` only, leaving skip-records
    in memory only and losing them on HA restart.  With persistent
    last_batch_fit_per_unit (#902), any state change to that dict —
    including per-regime skip-records on an empty-log run — must trigger
    a save so the records survive a restart.
    """
    coord = _make_coord(with_log=False)  # empty log
    await coord.async_batch_fit_solar()
    # Save IS expected: per-regime skip-records are durable state.
    coord._async_save_data.assert_awaited_once()
    # last_batch_fit recorded the per-unit attempt with insufficient_samples.
    assert "sensor.heater1" in coord._last_batch_fit_per_unit
    heating = coord._last_batch_fit_per_unit["sensor.heater1"]["regimes"]["heating"]
    assert heating["skip_reason"] == "insufficient_samples"


@pytest.mark.asyncio
async def test_async_batch_fit_solar_persists_top_level_skip():
    """Top-level skip (unknown entity) also triggers a save so the skip-
    record survives restart.
    """
    coord = _make_coord()
    await coord.async_batch_fit_solar(entity_id="sensor.does_not_exist")
    coord._async_save_data.assert_awaited_once()
    record = coord._last_batch_fit_per_unit["sensor.does_not_exist"]
    assert record["skip_reason"] == "unknown_entity"


@pytest.mark.asyncio
async def test_async_batch_fit_solar_return_contract():
    """Returned dict matches the documented service contract."""
    coord = _make_coord()
    result = await coord.async_batch_fit_solar()
    assert result["status"] == "ok"
    assert result["unit_entity_id"] is None
    assert "timestamp" in result
    assert "applied_count" in result and result["applied_count"] >= 1
    assert "skipped_count" in result
    assert "per_unit" in result
    assert "sensor.heater1" in result["per_unit"]


@pytest.mark.asyncio
async def test_async_batch_fit_solar_filter_to_single_entity():
    """``entity_id`` filter scopes the fit to one sensor."""
    coord = _make_coord()
    result = await coord.async_batch_fit_solar(entity_id="sensor.heater1")
    assert result["unit_entity_id"] == "sensor.heater1"
    assert "sensor.heater1" in result["per_unit"]


@pytest.mark.asyncio
async def test_async_batch_fit_solar_top_level_skip_records_reason():
    """Top-level skips (unknown entity, weighted_smear) record a skip-reason
    entry in ``_last_batch_fit_per_unit`` so diagnose_solar can show
    "skipped + reason" instead of null (which is indistinguishable from
    "never run").  The recorded entry has empty ``regimes`` and a
    ``skip_reason`` field.
    """
    coord = _make_coord()
    # Filter to an entity that isn't in energy_sensors → ``unknown_entity``
    # top-level skip.
    result = await coord.async_batch_fit_solar(entity_id="sensor.does_not_exist")
    assert result["per_unit"]["sensor.does_not_exist"] == {
        "skip_reason": "unknown_entity"
    }
    # Skip-record present with reason and empty regimes.
    assert "sensor.does_not_exist" in coord._last_batch_fit_per_unit
    record = coord._last_batch_fit_per_unit["sensor.does_not_exist"]
    assert record["skip_reason"] == "unknown_entity"
    assert record["regimes"] == {}
    assert "timestamp" in record


@pytest.mark.asyncio
async def test_diagnose_solar_surfaces_skip_reason():
    """``_format_last_batch_fit`` exposes ``skip_reason`` for top-level skips."""
    coord = _make_coord()
    await coord.async_batch_fit_solar(entity_id="sensor.does_not_exist")
    formatted = coord._diagnostics._format_last_batch_fit("sensor.does_not_exist")
    assert formatted is not None
    assert formatted["skip_reason"] == "unknown_entity"
    assert formatted["regimes"] == {}


@pytest.mark.asyncio
async def test_diagnose_solar_no_skip_reason_for_successful_fit():
    """Successful fits do NOT include a ``skip_reason`` key — only the
    timestamp + regimes."""
    coord = _make_coord()
    await coord.async_batch_fit_solar()
    formatted = coord._diagnostics._format_last_batch_fit("sensor.heater1")
    assert formatted is not None
    assert "skip_reason" not in formatted
    assert "regimes" in formatted


@pytest.mark.asyncio
async def test_last_batch_fit_save_payload_includes_field():
    """The save payload includes ``last_batch_fit_per_unit`` so the field
    survives HA restart."""
    coord = _make_coord()
    # Restore a real save path so we can capture the payload.
    coord._async_save_data = HeatingDataCoordinator._async_save_data.__get__(coord)
    coord._last_batch_fit_per_unit = {
        "sensor.heater1": {
            "timestamp": "2026-04-27T10:00:00",
            "skip_reason": "weighted_smear_excluded",
            "regimes": {},
        },
    }
    saved_payload = {}

    async def _capture(data):
        saved_payload.update(data)

    coord.storage._store.async_save = _capture
    await coord._async_save_data(force=True)
    assert "last_batch_fit_per_unit" in saved_payload
    assert saved_payload["last_batch_fit_per_unit"]["sensor.heater1"][
        "skip_reason"
    ] == "weighted_smear_excluded"


@pytest.mark.asyncio
async def test_last_batch_fit_load_restores_field():
    """``async_load_data`` restores ``_last_batch_fit_per_unit`` from the
    persisted payload, including skip-records for currently-registered
    sensors.  Records for unregistered sensors are dropped by
    ``_cleanup_removed_sensors`` (consistent with how every other
    per-unit dict is loaded — prevents orphaned records accumulating
    across config changes)."""
    coord = _make_coord(with_log=False)
    coord._last_batch_fit_per_unit = {}  # start empty

    persisted = {
        "correlation_data": {},
        "last_batch_fit_per_unit": {
            # Successful fit: registered sensor → survives.
            "sensor.heater1": {
                "timestamp": "2026-04-27T10:00:00",
                "regimes": {"heating": {"applied": True, "sample_count": 40}},
            },
        },
    }

    async def _replay():
        return persisted

    coord.storage._store.async_load = _replay
    await coord.storage.async_load_data()
    assert "sensor.heater1" in coord._last_batch_fit_per_unit
    assert coord._last_batch_fit_per_unit["sensor.heater1"]["regimes"][
        "heating"
    ]["applied"] is True


@pytest.mark.asyncio
async def test_last_batch_fit_load_handles_missing_field():
    """Pre-existing installs (saved before this field was added) load
    cleanly with an empty dict — no migration needed."""
    coord = _make_coord(with_log=False)
    coord._last_batch_fit_per_unit = {"stale": {"foo": "bar"}}  # should be replaced

    persisted = {"correlation_data": {}}  # no last_batch_fit_per_unit key

    async def _replay():
        return persisted

    coord.storage._store.async_load = _replay
    await coord.storage.async_load_data()
    assert coord._last_batch_fit_per_unit == {}


@pytest.mark.asyncio
async def test_diagnose_solar_surfaces_last_batch_fit():
    """After a batch fit, ``diagnose_solar`` per-unit output exposes
    the in-memory ``last_batch_fit`` block.  This is the user-facing
    integration of ``_last_batch_fit_per_unit``.
    """
    coord = _make_coord()
    await coord.async_batch_fit_solar()
    # The DiagnosticsEngine's _format_last_batch_fit reads
    # coord._last_batch_fit_per_unit; verify the format directly via
    # the helper rather than running the full diagnose_solar path
    # (which requires a more complete mock setup).
    formatted = coord._diagnostics._format_last_batch_fit("sensor.heater1")
    assert formatted is not None
    assert "timestamp" in formatted
    assert "regimes" in formatted
    assert "heating" in formatted["regimes"]
    # Unknown entity → None (graceful fallback).
    assert coord._diagnostics._format_last_batch_fit("sensor.unknown") is None
