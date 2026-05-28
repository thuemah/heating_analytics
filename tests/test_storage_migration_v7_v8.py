"""Storage v7 → v8 migration tests (#1009).

Lifts the obstruction gate from building-level to per-entity:

- Drops the v7 ``_critical_elev_per_facade`` (single shared dict).
- Creates ``_critical_elev_per_facade_per_unit`` as an empty dict.

**Discard, do not seed.**  Existing fitted values are dropped on
upgrade.  Users must rerun ``fit_solar_obstruction`` to restore
per-entity gates.  Rationale documented in
``storage._migrate_v7_to_v8`` and CLAUDE.md invariant #7.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.heating_analytics.storage import (
    StorageManager,
    _migrate_v7_to_v8,
)


def _make_coord():
    coord = MagicMock()
    coord.entry = MagicMock()
    coord.entry.entry_id = "test_entry"
    coord.solar_azimuth = 180.0
    coord.solar_battery_decay = 0.80
    coord.energy_sensors = ["sensor.heater1"]
    coord._correlation_data = {}
    coord._correlation_data_per_unit = {}
    coord._aux_coefficients_per_unit = {}
    coord._solar_coefficients_per_unit = {}
    coord._solar_coefficients_4d_per_unit = {}
    coord._learning_buffer_solar_4d_per_unit = {}
    coord._critical_elev_per_facade_per_unit = {}
    coord._unit_modes = {}
    coord._observation_counts = {}
    coord._learning_buffer_per_unit = {}
    coord._learning_buffer_aux_per_unit = {}
    coord._learning_buffer_solar_per_unit = {}
    coord._aux_coefficients = {}
    coord._daily_history = {}
    coord._hourly_log = []
    coord._daily_individual = {}
    coord._lifetime_individual = {}
    coord._daily_aux_breakdown = {}
    coord._aux_history = {}
    coord._forecast_history = []
    coord._forecast_breakdown = {}
    coord._unit_forecast_breakdowns = {}
    coord._daily_unit_forecasts = {}
    coord._learned_u_coefficient = None
    coord._aux_cooldown_active = False
    coord._aux_cooldown_start_time = None
    coord._learned_inertia = None
    coord._inertia_calibration_history = []
    coord._inertia_history = []
    coord._wind_calibration_history = []
    coord._per_unit_min_base_thresholds = {}
    coord._daily_persisted_attrs = {}
    coord.solar_optimizer = MagicMock()
    coord.solar_optimizer.set_data = MagicMock()
    coord.hass = MagicMock()
    coord.hass.async_add_executor_job = AsyncMock()
    return coord


def test_migration_drops_v7_field_and_adds_empty_v8():
    """v7 field is removed; v8 field initialised to an empty dict."""
    data = {"_critical_elev_per_facade": {"s": 28.5, "e": None, "w": None}}
    out = _migrate_v7_to_v8(data)
    assert "_critical_elev_per_facade" not in out
    assert out["_critical_elev_per_facade_per_unit"] == {}


def test_migration_discards_fitted_values():
    """**Discard, do not seed.**  Pre-existing v7 gate values are
    explicitly NOT carried into the per-entity dict — refit-required.
    """
    data = {
        "_critical_elev_per_facade": {"s": 30.0, "e": 45.0, "w": 22.0},
    }
    out = _migrate_v7_to_v8(data)
    assert "_critical_elev_per_facade" not in out
    assert out["_critical_elev_per_facade_per_unit"] == {}
    # No entity inherits the building-level value.
    assert not any(
        v.get("s") == 30.0 for v in out["_critical_elev_per_facade_per_unit"].values()
    )


def test_migration_idempotent():
    """Applying twice yields the same result.  Second application is a
    no-op because the v7 field is already gone and the v8 field already
    exists (setdefault preserves it).
    """
    data = {"_critical_elev_per_facade": {"s": 28.5, "e": None, "w": None}}
    out1 = _migrate_v7_to_v8(data)
    out2 = _migrate_v7_to_v8(out1)
    assert out1 == out2
    assert "_critical_elev_per_facade" not in out2
    assert out2["_critical_elev_per_facade_per_unit"] == {}


def test_migration_preserves_existing_v8_dict():
    """If a v8-shape dict already exists (idempotency / partial replay),
    ``setdefault`` semantics leave it untouched.
    """
    data = {
        "_critical_elev_per_facade_per_unit": {
            "sensor.heater1": {"s": 25.0, "e": None, "w": None},
        },
    }
    out = _migrate_v7_to_v8(data)
    assert out["_critical_elev_per_facade_per_unit"] == {
        "sensor.heater1": {"s": 25.0, "e": None, "w": None},
    }


def test_migration_handles_missing_v7_field():
    """If somehow we reach this migration without a v7 field (defensive),
    the v8 field is still initialised to an empty dict.
    """
    data = {"correlation_data": {}}
    out = _migrate_v7_to_v8(data)
    assert out["_critical_elev_per_facade_per_unit"] == {}


@pytest.mark.asyncio
async def test_full_chain_through_load_clears_gates():
    """v7-shape input with fitted gates → run through full migration chain
    → coordinator state populated with empty per-entity dict (gates
    discarded; refit-required).
    """
    coord = _make_coord()
    pre = {
        "solar_coefficients_per_unit": {},
        "_critical_elev_per_facade": {"s": 30.0, "e": 45.0, "w": None},
    }
    with patch("custom_components.heating_analytics.storage.Store") as mock_store_cls:
        mock_store = mock_store_cls.return_value
        sm = StorageManager(coord)
        migrated = await sm._async_migrate(
            old_major_version=7, old_minor_version=0, old_data=pre
        )
        assert "_critical_elev_per_facade" not in migrated
        assert migrated["_critical_elev_per_facade_per_unit"] == {}

        mock_store.async_load = AsyncMock(return_value=migrated)
        await sm.async_load_data()

    assert coord._critical_elev_per_facade_per_unit == {}


@pytest.mark.asyncio
async def test_full_chain_loads_existing_v8_state():
    """v8-shape input with fitted per-entity gates → v8→v9 migration
    wraps single-float values into ``{"low": None, "high": value}``.
    """
    coord = _make_coord()
    pre = {
        "solar_coefficients_per_unit": {},
        "_critical_elev_per_facade_per_unit": {
            "sensor.heater1": {"s": 22.0, "e": None, "w": 55.5},
            "sensor.heater2": {"s": None, "e": None, "w": None},
        },
    }
    with patch("custom_components.heating_analytics.storage.Store") as mock_store_cls:
        mock_store = mock_store_cls.return_value
        sm = StorageManager(coord)
        # v8 data goes through v8→v9 migration.
        migrated = await sm._async_migrate(
            old_major_version=8, old_minor_version=0, old_data=pre
        )
        mock_store.async_load = AsyncMock(return_value=migrated)
        await sm.async_load_data()

    assert coord._critical_elev_per_facade_per_unit == {
        "sensor.heater1": {
            "s": {"low": None, "high": 22.0},
            "e": {"low": None, "high": None},
            "w": {"low": None, "high": 55.5},
        },
        "sensor.heater2": {
            "s": {"low": None, "high": None},
            "e": {"low": None, "high": None},
            "w": {"low": None, "high": None},
        },
    }
