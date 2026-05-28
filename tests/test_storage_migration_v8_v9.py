"""Storage v8 → v9 migration tests (solar-window low+high gate).

Wraps each entity's v8 single-float ``critical_elev`` into the v9
nested ``{"low": None, "high": old_value}`` shape.  The v8 value was
a pure upper-horizon gate; the migration maps it to ``high`` and
leaves ``low`` at ``None``.

Idempotent — running twice produces the same result.  Already-nested
values pass through unchanged.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.heating_analytics.storage import (
    StorageManager,
    _migrate_v8_to_v9,
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


def test_migration_wraps_single_float_to_high():
    """v8 single-float → v9 ``{"low": None, "high": value}``."""
    data = {
        "_critical_elev_per_facade_per_unit": {
            "sensor.a": {"s": 22.0, "e": None, "w": 55.5},
        },
    }
    out = _migrate_v8_to_v9(data)
    gates = out["_critical_elev_per_facade_per_unit"]
    assert gates["sensor.a"]["s"] == {"low": None, "high": 22.0}
    assert gates["sensor.a"]["e"] == {"low": None, "high": None}
    assert gates["sensor.a"]["w"] == {"low": None, "high": 55.5}


def test_migration_wraps_all_none_flat():
    """v8 all-None → v9 all-``{"low": None, "high": None}``."""
    data = {
        "_critical_elev_per_facade_per_unit": {
            "sensor.a": {"s": None, "e": None, "w": None},
        },
    }
    out = _migrate_v8_to_v9(data)
    gates = out["_critical_elev_per_facade_per_unit"]
    assert gates["sensor.a"] == {
        "s": {"low": None, "high": None},
        "e": {"low": None, "high": None},
        "w": {"low": None, "high": None},
    }


def test_migration_preserves_existing_nested_shape():
    """Already v9 nested data passes through unchanged."""
    data = {
        "_critical_elev_per_facade_per_unit": {
            "sensor.a": {
                "s": {"low": 10.0, "high": 30.0},
                "e": {"low": None, "high": 45.0},
                "w": {"low": None, "high": None},
            },
        },
    }
    out = _migrate_v8_to_v9(data)
    assert out["_critical_elev_per_facade_per_unit"] == data["_critical_elev_per_facade_per_unit"]


def test_migration_idempotent():
    """Applying twice yields the same result."""
    data = {
        "_critical_elev_per_facade_per_unit": {
            "sensor.a": {"s": 22.0, "e": None, "w": 55.5},
        },
    }
    out1 = _migrate_v8_to_v9(data)
    out2 = _migrate_v8_to_v9(out1)
    assert out1 == out2
    assert out2["_critical_elev_per_facade_per_unit"]["sensor.a"]["s"] == {
        "low": None, "high": 22.0,
    }


def test_migration_handles_missing_field():
    """If ``_critical_elev_per_facade_per_unit`` is absent, initialises to {}."""
    data = {"correlation_data": {}}
    out = _migrate_v8_to_v9(data)
    assert out["_critical_elev_per_facade_per_unit"] == {}


def test_migration_handles_non_dict_entity():
    """Non-dict entity values → empty nested gate."""
    data = {
        "_critical_elev_per_facade_per_unit": {
            "sensor.a": "bad_value",
        },
    }
    out = _migrate_v8_to_v9(data)
    gates = out["_critical_elev_per_facade_per_unit"]
    assert gates["sensor.a"] == {
        "s": {"low": None, "high": None},
        "e": {"low": None, "high": None},
        "w": {"low": None, "high": None},
    }


@pytest.mark.asyncio
async def test_full_chain_v8_data_loaded_as_v9():
    """v8-shape input with fitted gates → v9 coordinator state after full
    migration chain + load.
    """
    coord = _make_coord()
    pre = {
        "solar_coefficients_per_unit": {},
        "_critical_elev_per_facade_per_unit": {
            "sensor.heater1": {"s": 22.0, "e": None, "w": 55.5},
        },
    }
    with patch("custom_components.heating_analytics.storage.Store") as mock_store_cls:
        mock_store = mock_store_cls.return_value
        sm = StorageManager(coord)
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
    }


@pytest.mark.asyncio
async def test_full_chain_v9_data_loaded_directly():
    """v9-shape input loaded directly → coordinator state populated correctly."""
    coord = _make_coord()
    pre = {
        "solar_coefficients_per_unit": {},
        "_critical_elev_per_facade_per_unit": {
            "sensor.heater1": {
                "s": {"low": 10.0, "high": 30.0},
                "e": {"low": None, "high": 45.0},
                "w": {"low": None, "high": None},
            },
        },
    }
    with patch("custom_components.heating_analytics.storage.Store") as mock_store_cls:
        mock_store = mock_store_cls.return_value
        sm = StorageManager(coord)
        migrated = await sm._async_migrate(
            old_major_version=9, old_minor_version=0, old_data=pre
        )
        mock_store.async_load = AsyncMock(return_value=migrated)
        await sm.async_load_data()

    assert coord._critical_elev_per_facade_per_unit == {
        "sensor.heater1": {
            "s": {"low": 10.0, "high": 30.0},
            "e": {"low": None, "high": 45.0},
            "w": {"low": None, "high": None},
        },
    }
