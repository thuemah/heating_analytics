"""Storage v6 → v7 migration tests (#991).

Adds ``_critical_elev_per_facade`` for the per-facade direct-beam
obstruction gate.  Migration must:

- Add ``_critical_elev_per_facade`` as ``{"s": None, "e": None, "w": None}``.
- Be idempotent (setdefault semantics).
- Preserve any pre-existing fitted values (defensive case).

Full-chain load tests for the obstruction state live in
``test_storage_migration_v7_v8.py`` after the v8 lift to per-entity
(#1009) — the v7 single-dict field no longer exists on the loaded
coordinator state.  The pure-function tests here verify ``_migrate_v6_to_v7``
in isolation so the migration chain remains traceable.
"""
from custom_components.heating_analytics.storage import _migrate_v6_to_v7


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
