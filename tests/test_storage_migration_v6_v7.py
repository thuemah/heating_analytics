"""Storage v6 → v7 migration tests (#991).

Adds ``_critical_elev_per_facade`` for the per-facade direct-beam
obstruction gate.  Migration must:

- Add ``_critical_elev_per_facade`` as ``{"s": None, "e": None, "w": None}``.
- Be idempotent (setdefault semantics).
- Preserve any pre-existing fitted values (defensive case).
- Round-trip cleanly through ``_async_migrate`` so the post-migration
  shape is what ``async_load_data`` expects.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.heating_analytics.storage import (
    StorageManager,
    _migrate_v6_to_v7,
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
    coord._critical_elev_per_facade = {"s": None, "e": None, "w": None}
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


def test_migration_adds_empty_critical_elev_dict():
    """Minimal v6 dict → migration adds the per-facade key with all-None."""
    data = {
        "_solar_coefficients_4d_per_unit": {},
        "_learning_buffer_solar_4d_per_unit": {},
    }
    out = _migrate_v6_to_v7(data)
    assert out["_critical_elev_per_facade"] == {"s": None, "e": None, "w": None}


def test_migration_idempotent():
    """Applying twice yields the same result; populated values preserved."""
    data = {
        "_critical_elev_per_facade": {"s": 28.5, "e": None, "w": None},
    }
    out1 = _migrate_v6_to_v7(data)
    out2 = _migrate_v6_to_v7(out1)
    assert out1 == out2
    assert out2["_critical_elev_per_facade"]["s"] == 28.5


def test_migration_preserves_existing_values():
    """Pre-existing fitted values survive a defensive re-migration."""
    pre = {
        "_critical_elev_per_facade": {"s": 32.0, "e": 45.0, "w": None},
    }
    out = _migrate_v6_to_v7(pre)
    assert out["_critical_elev_per_facade"] == pre["_critical_elev_per_facade"]


@pytest.mark.asyncio
async def test_full_chain_through_load():
    """v3-shape minimal input → run through full migration chain (v3 → v7)
    → coordinator state populated with canonical-empty per-facade dict."""
    coord = _make_coord()
    pre_v7 = {
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
            old_major_version=3, old_minor_version=0, old_data=pre_v7
        )
        assert "_critical_elev_per_facade" in migrated
        assert migrated["_critical_elev_per_facade"] == {
            "s": None, "e": None, "w": None,
        }

        mock_store.async_load = AsyncMock(return_value=migrated)
        await sm.async_load_data()

    # Coordinator state populated as canonical-empty dict.
    assert coord._critical_elev_per_facade == {"s": None, "e": None, "w": None}


@pytest.mark.asyncio
async def test_full_chain_preserves_fitted_values():
    """Pre-v7 data with fitted critical_elev values is preserved through
    the load path."""
    coord = _make_coord()
    pre = {
        "solar_coefficients_per_unit": {},
        "_critical_elev_per_facade": {"s": 30.0, "e": None, "w": 55.5},
    }
    with patch("custom_components.heating_analytics.storage.Store") as mock_store_cls:
        mock_store = mock_store_cls.return_value
        sm = StorageManager(coord)
        migrated = await sm._async_migrate(
            old_major_version=6, old_minor_version=0, old_data=pre
        )
        assert migrated["_critical_elev_per_facade"]["s"] == 30.0
        assert migrated["_critical_elev_per_facade"]["w"] == 55.5

        mock_store.async_load = AsyncMock(return_value=migrated)
        await sm.async_load_data()

    assert coord._critical_elev_per_facade == {
        "s": 30.0, "e": None, "w": 55.5,
    }
