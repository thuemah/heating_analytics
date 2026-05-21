"""Storage v5 → v6 migration tests (#954).

Scaffolds parallel 4D solar-coefficient state alongside the existing
3D table.  Migration must:

- Add ``_solar_coefficients_4d_per_unit`` and
  ``_learning_buffer_solar_4d_per_unit`` top-level keys as empty dicts.
- Be idempotent (setdefault semantics).
- Preserve any pre-existing 4D dict if one is somehow present.
- Round-trip cleanly through ``_async_migrate`` so the post-migration
  shape is what ``async_load_data`` expects (canonical-shape rule).
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.heating_analytics.storage import (
    StorageManager,
    _migrate_v5_to_v6,
)


def _make_coord():
    """Coordinator stub mirroring the surface StorageManager touches.

    Mirrors ``test_storage_migration._make_coord`` but trimmed to the
    attrs the v5→v6 load path needs.
    """
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


# ---------------------------------------------------------------------------
# Direct unit tests on _migrate_v5_to_v6
# ---------------------------------------------------------------------------


def test_migration_adds_empty_4d_dicts():
    """Minimal v5 dict → migration adds both 4D keys as empty dicts."""
    data = {
        "solar_coefficients_per_unit": {},
        "learning_buffer_solar_per_unit": {},
    }
    out = _migrate_v5_to_v6(data)
    assert out["_solar_coefficients_4d_per_unit"] == {}
    assert out["_learning_buffer_solar_4d_per_unit"] == {}


def test_migration_idempotent():
    """Applying twice produces the same result.  Pre-existing populated
    4D dict survives a second migration call unchanged (setdefault)."""
    data = {
        "_solar_coefficients_4d_per_unit": {
            "sensor.a": {
                "heating": {
                    "s": 0.3, "e": 0.1, "w": 0.0,
                    "diffuse": 0.05, "learned": True,
                },
            },
        },
    }
    out1 = _migrate_v5_to_v6(data)
    out2 = _migrate_v5_to_v6(out1)
    assert out1 == out2
    assert out2["_solar_coefficients_4d_per_unit"]["sensor.a"]["heating"]["diffuse"] == 0.05
    assert out2["_solar_coefficients_4d_per_unit"]["sensor.a"]["heating"]["learned"] is True
    assert out2["_learning_buffer_solar_4d_per_unit"] == {}


def test_migration_preserves_existing_4d_data():
    """A v5 dict that already has 4D data (defensive case) preserves it."""
    pre = {
        "_solar_coefficients_4d_per_unit": {
            "sensor.a": {
                "heating": {
                    "s": 0.3, "e": 0.1, "w": 0.0,
                    "diffuse": 0.05, "learned": True,
                },
                "cooling": {},
            },
        },
    }
    out = _migrate_v5_to_v6(pre)
    assert out["_solar_coefficients_4d_per_unit"] == pre["_solar_coefficients_4d_per_unit"]
    # New key still added.
    assert out["_learning_buffer_solar_4d_per_unit"] == {}


# ---------------------------------------------------------------------------
# Full chain through _async_migrate and async_load_data
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_chain_through_load():
    """v3-shape minimal input → run through full migration chain
    (_async_migrate from v3 → v6) → coordinator state has both 4D dicts
    populated as empty.

    This catches the migration class of bug (#874) where the migration
    emits a non-canonical shape that the load path silently drops.
    """
    coord = _make_coord()
    pre_v6 = {
        "solar_coefficients_per_unit": {
            "sensor.heater1": {"s": 0.40, "e": 0.20, "w": 0.10},
        },
        "learning_buffer_solar_per_unit": {
            "sensor.heater1": [(0.5, 0.0, 0.0, 0.3)],
        },
    }

    with patch("custom_components.heating_analytics.storage.Store") as mock_store_cls:
        mock_store = mock_store_cls.return_value
        sm = StorageManager(coord)
        migrated = await sm._async_migrate(
            old_major_version=3, old_minor_version=0, old_data=pre_v6
        )
        # The migration chain must include v5 → v6 step.
        assert "_solar_coefficients_4d_per_unit" in migrated
        assert "_learning_buffer_solar_4d_per_unit" in migrated
        assert migrated["_solar_coefficients_4d_per_unit"] == {}
        assert migrated["_learning_buffer_solar_4d_per_unit"] == {}

        mock_store.async_load = AsyncMock(return_value=migrated)
        await sm.async_load_data()

    # Coordinator state populated as canonical empty dicts.
    assert coord._solar_coefficients_4d_per_unit == {}
    assert coord._learning_buffer_solar_4d_per_unit == {}
