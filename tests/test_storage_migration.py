"""Storage-migration tests across the v1→v2→v3→v4 chain.

Sections:
- Pre-v3 migration helpers (`_migrate_pre_v3`) and round-trip through
  `async_load_data` for the v2 → pre-v3 cleanup.
- v3 → v4 migration helpers (`_migrate_v3_to_v4`, `_sanitize_stratified`)
  for mode-stratified solar coefficients.
- End-to-end chain through `StorageManager._async_migrate` →
  `async_load_data` / `async_restore_data`, guarding against the class
  of bug where a migration emits a non-canonical shape that the load
  filter silently drops.
"""
import json
import math
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.heating_analytics.storage import (
    StorageManager,
    _migrate_pre_v3,
    _migrate_v3_to_v4,
    _sanitize_stratified,
)


# =============================================================================
# Pre-v3 migration chain — `_migrate_pre_v3`
# =============================================================================


# ---- solar_coefficients_per_unit ----

def test_solar_coeffs_flat_dict_is_sanitized():
    data = {"solar_coefficients_per_unit": {"s1": {"s": 1.0, "e": -0.5, "w": 0.2}}}
    out = _migrate_pre_v3(data)
    assert out["solar_coefficients_per_unit"]["s1"] == {"s": 1.0, "e": 0.0, "w": 0.2}


def test_solar_coeffs_temp_stratified_picks_first_valid_dict():
    data = {
        "solar_coefficients_per_unit": {
            "s1": {
                "-5": {"s": 0.3, "e": 0.1, "w": 0.0},
                "0": {"s": 0.4, "e": 0.2, "w": 0.0},
            }
        }
    }
    out = _migrate_pre_v3(data)
    coeff = out["solar_coefficients_per_unit"]["s1"]
    assert "s" in coeff and "e" in coeff and "w" in coeff
    assert coeff["s"] in (0.3, 0.4)


def test_solar_coeffs_scalar_uses_azimuth_decomposition():
    # Pure south azimuth = 180° → -cos(180°)=1 → all weight to south.
    data = {"solar_coefficients_per_unit": {"s1": {"-5": 0.5}}}
    out = _migrate_pre_v3(data, solar_azimuth=180.0)
    coeff = out["solar_coefficients_per_unit"]["s1"]
    assert coeff["s"] == pytest.approx(0.5, abs=1e-4)
    assert coeff["e"] == 0.0
    assert coeff["w"] == 0.0


def test_solar_coeffs_scalar_southeast_splits_between_s_and_e():
    # Azimuth 135° = SE → -cos(135°)=+0.707 (south component),
    # sin(135°)=+0.707 (east component), -sin(135°)=-0.707 (west clamped to 0).
    data = {"solar_coefficients_per_unit": {"s1": {"-5": 1.0}}}
    out = _migrate_pre_v3(data, solar_azimuth=135.0)
    coeff = out["solar_coefficients_per_unit"]["s1"]
    assert coeff["s"] == pytest.approx(math.sqrt(0.5), abs=1e-4)
    assert coeff["e"] == pytest.approx(math.sqrt(0.5), abs=1e-4)
    assert coeff["w"] == 0.0


def test_solar_coeffs_scalar_west_fills_only_w():
    # Azimuth 270° = W → -cos=0 (s), sin=-1 clamped → e=0, -sin=+1 → w=value.
    data = {"solar_coefficients_per_unit": {"s1": {"-5": 0.4}}}
    out = _migrate_pre_v3(data, solar_azimuth=270.0)
    coeff = out["solar_coefficients_per_unit"]["s1"]
    assert coeff["s"] == 0.0
    assert coeff["e"] == 0.0
    assert coeff["w"] == pytest.approx(0.4, abs=1e-4)


def test_solar_coeffs_missing_key_noop():
    out = _migrate_pre_v3({})
    assert "solar_coefficients_per_unit" not in out or out["solar_coefficients_per_unit"] == {}


# ---- learning_buffer_solar_per_unit ----

def test_solar_buffer_4tuple_preserved():
    data = {"learning_buffer_solar_per_unit": {"s1": [(0.1, 0.2, 0.3, 0.4)]}}
    out = _migrate_pre_v3(data)
    assert out["learning_buffer_solar_per_unit"]["s1"] == [(0.1, 0.2, 0.3, 0.4)]


def test_solar_buffer_3tuple_migrated_with_zero_w():
    data = {"learning_buffer_solar_per_unit": {"s1": [(0.1, 0.2, 0.9), [0.3, 0.4, 1.2]]}}
    out = _migrate_pre_v3(data)
    assert out["learning_buffer_solar_per_unit"]["s1"] == [
        (0.1, 0.2, 0.0, 0.9),
        (0.3, 0.4, 0.0, 1.2),
    ]


def test_solar_buffer_temp_stratified_dict_flattened():
    data = {
        "learning_buffer_solar_per_unit": {
            "s1": {
                "-5": [(0.1, 0.2, 0.3, 0.9)],
                "0": [(0.4, 0.5, 0.6)],  # 3-tuple gets padded
            }
        }
    }
    out = _migrate_pre_v3(data)
    samples = out["learning_buffer_solar_per_unit"]["s1"]
    assert len(samples) == 2
    assert (0.1, 0.2, 0.3, 0.9) in samples
    assert (0.4, 0.5, 0.0, 0.6) in samples


# ---- aux_coefficients ----

def test_aux_coeff_legacy_float_becomes_normal_dict():
    data = {"aux_coefficients": {"-5": 0.8, "0": {"normal": 0.5, "high_wind": 0.7}}}
    out = _migrate_pre_v3(data)
    assert out["aux_coefficients"]["-5"] == {"normal": 0.8}
    assert out["aux_coefficients"]["0"] == {"normal": 0.5, "high_wind": 0.7}


# ---- with_auxiliary_heating bucket removal ----

def test_with_aux_heating_removed_from_per_unit_structures():
    data = {
        "correlation_data_per_unit": {
            "s1": {"-5": {"normal": 0.5, "with_auxiliary_heating": 0.1}}
        },
        "learning_buffer_per_unit": {
            "s1": {"-5": {"normal": [], "with_auxiliary_heating": []}}
        },
        "observation_counts": {
            "s1": {"-5": {"normal": 3, "with_auxiliary_heating": 1}}
        },
    }
    out = _migrate_pre_v3(data)
    assert "with_auxiliary_heating" not in out["correlation_data_per_unit"]["s1"]["-5"]
    assert "with_auxiliary_heating" not in out["learning_buffer_per_unit"]["s1"]["-5"]
    assert "with_auxiliary_heating" not in out["observation_counts"]["s1"]["-5"]


def test_with_aux_heating_translated_to_aux_coefficient_savings():
    # Base (normal) = 1.0 kW, aux = 0.4 kW → savings = 0.6 kW.
    # Canonical v3 shape is {bucket: value}; translation must emit
    # {"normal": 0.6} so async_load_data's dict-filter does not drop it.
    data = {
        "correlation_data": {"-5": {"normal": 1.0, "with_auxiliary_heating": 0.4}},
    }
    out = _migrate_pre_v3(data)
    assert out["aux_coefficients"]["-5"] == {"normal": 0.6}
    assert "with_auxiliary_heating" not in out["correlation_data"]["-5"]


def test_with_aux_heating_drops_nonpositive_savings():
    # Aux >= base: savings would be <= 0 → clamped to 0, no coefficient written.
    data = {"correlation_data": {"-5": {"normal": 0.3, "with_auxiliary_heating": 0.5}}}
    out = _migrate_pre_v3(data)
    assert "with_auxiliary_heating" not in out["correlation_data"]["-5"]
    # No aux_coefficients["-5"] should be set (or aux_coefficients stays absent/empty)
    assert "-5" not in out.get("aux_coefficients", {})


def test_with_aux_heating_zero_normal_does_not_fall_through_to_high_wind():
    """Zero ``normal`` baseline must NOT fall through to ``high_wind``.

    Pre-v3 semantics used ``base_val = buckets.get("normal")`` followed
    by ``if base_val is None: base_val = buckets.get("high_wind")``.
    A legitimate ``normal = 0.0`` (mild bucket, no calm-weather heating
    demand) skipped the aux translation entirely.  Using truthy-``or``
    here would fall through to ``high_wind`` and overstate aux savings.
    """
    data = {
        "correlation_data": {
            "-5": {
                "normal": 0.0,
                "high_wind": 1.0,
                "with_auxiliary_heating": 0.3,
            }
        }
    }
    out = _migrate_pre_v3(data)
    # normal=0.0 → base_val=0.0 → ``base_val > 0`` False → no coefficient written.
    assert "-5" not in out.get("aux_coefficients", {})
    assert "with_auxiliary_heating" not in out["correlation_data"]["-5"]


def test_with_aux_heating_missing_normal_falls_through_to_high_wind():
    """When ``normal`` is absent, fall through to ``high_wind`` (pre-v3 semantics)."""
    data = {
        "correlation_data": {
            "-5": {
                "high_wind": 1.0,
                "with_auxiliary_heating": 0.4,
            }
        }
    }
    out = _migrate_pre_v3(data)
    assert out["aux_coefficients"]["-5"] == {"normal": 0.6}


# ---- hdd -> tdd / load -> actual_kwh ----

def test_hdd_renamed_to_tdd_in_daily_history():
    data = {"daily_history": {"2026-01-01": {"hdd": 12.5, "kwh": 100.0}}}
    out = _migrate_pre_v3(data)
    assert out["daily_history"]["2026-01-01"]["tdd"] == 12.5
    assert "hdd" not in out["daily_history"]["2026-01-01"]


def test_load_renamed_to_actual_kwh_in_hourly_vectors():
    data = {
        "daily_history": {
            "2026-01-01": {"hourly_vectors": {"load": [1, 2, 3]}}
        }
    }
    out = _migrate_pre_v3(data)
    vectors = out["daily_history"]["2026-01-01"]["hourly_vectors"]
    assert vectors["actual_kwh"] == [1, 2, 3]
    assert "load" not in vectors


def test_hdd_renamed_to_tdd_in_hourly_log():
    data = {"hourly_log": [{"timestamp": "2026-01-01T00:00", "hdd": 2.0}]}
    out = _migrate_pre_v3(data)
    assert out["hourly_log"][0]["tdd"] == 2.0
    assert "hdd" not in out["hourly_log"][0]


# ---- battery_model ----

def test_battery_model_legacy_is_scaled_down():
    data = {"solar_battery_state": 10.0}
    out = _migrate_pre_v3(data, solar_battery_decay=0.80)
    assert out["solar_battery_state"] == pytest.approx(10.0 * 0.20)
    assert out["battery_model"] == "ema"


def test_battery_model_already_ema_is_noop():
    data = {"solar_battery_state": 10.0, "battery_model": "ema"}
    out = _migrate_pre_v3(data, solar_battery_decay=0.80)
    assert out["solar_battery_state"] == 10.0
    assert out["battery_model"] == "ema"


# ---- forecast_history unknown -> primary ----

def test_forecast_history_unknown_becomes_primary():
    data = {
        "forecast_history": [
            {"date": "2026-01-01", "source": "unknown"},
            {"date": "2026-01-02", "source": "primary"},
        ]
    }
    out = _migrate_pre_v3(data)
    assert out["forecast_history"][0]["source"] == "primary"
    assert out["forecast_history"][1]["source"] == "primary"


# ---- idempotency ----

def test_idempotent_on_canonical_v3_shape():
    """Running the migration twice yields the same result as running once."""
    data = {
        "solar_coefficients_per_unit": {"s1": {"s": 0.1, "e": 0.2, "w": 0.3}},
        "learning_buffer_solar_per_unit": {"s1": [(0.1, 0.2, 0.3, 0.4)]},
        "aux_coefficients": {"-5": {"normal": 0.5}},
        "correlation_data": {"-5": {"normal": 1.0}},
        "correlation_data_per_unit": {"s1": {"-5": {"normal": 0.5}}},
        "daily_history": {"2026-01-01": {"tdd": 12.5}},
        "hourly_log": [{"timestamp": "2026-01-01T00:00", "tdd": 2.0}],
        "solar_battery_state": 5.0,
        "battery_model": "ema",
        "forecast_history": [{"date": "2026-01-01", "source": "primary"}],
    }
    once = _migrate_pre_v3(dict(data))
    twice = _migrate_pre_v3(dict(once))
    assert once == twice


def test_empty_dict_gets_battery_model_tag():
    # The battery-model migration is the one unconditional tag applied
    # to any pre-v3 blob — even an empty dict gets marked post-EMA with
    # a zero-valued state.  Subsequent migrations are no-ops.
    out = _migrate_pre_v3({})
    assert out["battery_model"] == "ema"
    assert out["solar_battery_state"] == 0.0


def test_idempotent_on_legacy_with_auxiliary_heating_shape():
    """Previously non-idempotent: float emitted on run 1, dict on run 2.

    Pins the blocker fix — translation must emit canonical dict shape so
    a second migrate pass produces identical output.
    """
    legacy = {
        "correlation_data": {"-5": {"normal": 1.0, "with_auxiliary_heating": 0.4}},
    }
    once = _migrate_pre_v3({k: dict(v) if isinstance(v, dict) else v for k, v in legacy.items()})
    twice = _migrate_pre_v3({k: dict(v) if isinstance(v, dict) else v for k, v in once.items()})
    assert once == twice
    assert once["aux_coefficients"]["-5"] == {"normal": 0.6}


# ---- end-to-end: a fully pre-v3 blob ----

# ---- Round-trip through async_load_data (blocker regression test) ----
# The with_auxiliary_heating -> aux_coefficients translation must emit the
# canonical v3 shape so the load filter does not drop it.  Previously the
# translation wrote a raw float, which async_load_data then filtered out
# because it expects `{bucket: value}` dicts.


def _make_coordinator_for_load(**extra):
    coord = MagicMock()
    coord.hass = MagicMock()
    coord.energy_sensors = []
    coord._correlation_data = {}
    coord.forecast = MagicMock()
    coord.forecast._cached_long_term_hourly = None
    coord.forecast._forecast_history = []
    coord.forecast._cached_forecast_uncertainty = None
    coord.statistics = MagicMock()
    coord.solar_azimuth = 180.0
    coord.solar_battery_decay = 0.80
    coord.data = {}
    for attr, val in extra.items():
        setattr(coord, attr, val)
    return coord


@pytest.mark.asyncio
async def test_roundtrip_with_auxiliary_heating_translation_survives_load_filter():
    """Legacy aux-bucket input must produce a canonical dict in coordinator state."""
    coord = _make_coordinator_for_load()

    legacy_data = {
        "correlation_data": {"-5": {"normal": 1.0, "with_auxiliary_heating": 0.4}},
    }
    migrated = _migrate_pre_v3(dict(legacy_data))

    with patch("custom_components.heating_analytics.storage.Store") as mock_store_cls:
        mock_store = mock_store_cls.return_value
        mock_store.async_load = AsyncMock(return_value=migrated)

        storage = StorageManager(coord)
        await storage.async_load_data()

    # The migrated aux coefficient must be a dict {"normal": 0.6} and survive
    # the isinstance-dict filter in async_load_data.
    assert coord._aux_coefficients == {"-5": {"normal": 0.6}}


@pytest.mark.asyncio
async def test_async_migrate_hook_applies_pre_v3_transforms():
    """The HA-registered migrate hook must route old data through _migrate_pre_v3."""
    coord = _make_coordinator_for_load()

    with patch("custom_components.heating_analytics.storage.Store"):
        storage = StorageManager(coord)
        legacy = {
            "solar_battery_state": 10.0,
            "hourly_log": [{"timestamp": "2026-01-01T00:00", "hdd": 2.0}],
        }
        migrated = await storage._async_migrate(2, 0, legacy)

    assert migrated["battery_model"] == "ema"
    assert migrated["solar_battery_state"] == pytest.approx(2.0)
    assert migrated["hourly_log"][0]["tdd"] == 2.0


def test_end_to_end_legacy_blob_normalizes_all_fields():
    data = {
        "correlation_data": {"-5": {"normal": 1.0, "with_auxiliary_heating": 0.4}},
        "correlation_data_per_unit": {
            "s1": {"-5": {"normal": 0.5, "with_auxiliary_heating": 0.1}}
        },
        "aux_coefficients": {"-10": 0.9},
        "solar_coefficients_per_unit": {
            "s1": {"-5": {"s": 0.3, "e": 0.1, "w": 0.0}}
        },
        "learning_buffer_solar_per_unit": {
            "s1": {"-5": [(0.1, 0.2, 0.9)]}
        },
        "daily_history": {
            "2026-01-01": {"hdd": 10.0, "hourly_vectors": {"load": [1, 2, 3]}}
        },
        "hourly_log": [{"timestamp": "2026-01-01T00:00", "hdd": 2.0}],
        "solar_battery_state": 4.0,
        "forecast_history": [{"date": "2026-01-01", "source": "unknown"}],
    }
    out = _migrate_pre_v3(data, solar_battery_decay=0.80)

    assert out["aux_coefficients"]["-10"] == {"normal": 0.9}
    assert out["aux_coefficients"]["-5"] == {"normal": 0.6}  # from correlation translation
    assert "with_auxiliary_heating" not in out["correlation_data"]["-5"]
    assert "with_auxiliary_heating" not in out["correlation_data_per_unit"]["s1"]["-5"]
    assert out["solar_coefficients_per_unit"]["s1"] == {"s": 0.3, "e": 0.1, "w": 0.0}
    assert out["learning_buffer_solar_per_unit"]["s1"] == [(0.1, 0.2, 0.0, 0.9)]
    assert out["daily_history"]["2026-01-01"]["tdd"] == 10.0
    assert out["daily_history"]["2026-01-01"]["hourly_vectors"]["actual_kwh"] == [1, 2, 3]
    assert out["hourly_log"][0]["tdd"] == 2.0
    assert out["solar_battery_state"] == pytest.approx(0.8)
    assert out["battery_model"] == "ema"
    assert out["forecast_history"][0]["source"] == "primary"


# =============================================================================
# v3 → v4 migration helpers — mode-stratified solar coefficients
# =============================================================================

# ---- solar_coefficients_per_unit ----

def test_v4_flat_seeds_both_regimes():
    """Pre-v4 flat {s,e,w} seeds heating AND cooling with same values."""
    data = {"solar_coefficients_per_unit": {"s1": {"s": 0.40, "e": 0.30, "w": 0.20}}}
    out = _migrate_v3_to_v4(data)
    coeff = out["solar_coefficients_per_unit"]["s1"]
    assert coeff["heating"] == {"s": 0.40, "e": 0.30, "w": 0.20}
    assert coeff["cooling"] == {"s": 0.40, "e": 0.30, "w": 0.20}


def test_v4_clamps_negative_components_during_seed():
    data = {"solar_coefficients_per_unit": {"s1": {"s": 1.0, "e": -0.5, "w": 0.2}}}
    out = _migrate_v3_to_v4(data)
    coeff = out["solar_coefficients_per_unit"]["s1"]
    assert coeff["heating"] == {"s": 1.0, "e": 0.0, "w": 0.2}
    assert coeff["cooling"] == {"s": 1.0, "e": 0.0, "w": 0.2}


def test_v4_already_stratified_passes_through():
    """Idempotent: re-running on v4 data is a no-op (sanitized)."""
    data = {
        "solar_coefficients_per_unit": {
            "s1": {
                "heating": {"s": 0.40, "e": 0.30, "w": 0.20},
                "cooling": {"s": 0.10, "e": 0.05, "w": 0.02},
            }
        }
    }
    out = _migrate_v3_to_v4(data)
    coeff = out["solar_coefficients_per_unit"]["s1"]
    assert coeff["heating"] == {"s": 0.40, "e": 0.30, "w": 0.20}
    assert coeff["cooling"] == {"s": 0.10, "e": 0.05, "w": 0.02}


def test_v4_idempotent_double_migration():
    """Running migration twice produces same result as once."""
    data = {"solar_coefficients_per_unit": {"s1": {"s": 0.40, "e": 0.30, "w": 0.20}}}
    out1 = _migrate_v3_to_v4(data)
    # deep-copy via dict() since migration mutates in-place
    coeff_after_first = {
        eid: {regime: dict(values) for regime, values in entry.items()}
        for eid, entry in out1["solar_coefficients_per_unit"].items()
    }
    out2 = _migrate_v3_to_v4(out1)
    assert out2["solar_coefficients_per_unit"]["s1"] == coeff_after_first["s1"]


def test_v4_partial_stratified_fills_missing_regime():
    """Entry with only 'heating' present gets cooling filled with zeros."""
    data = {
        "solar_coefficients_per_unit": {
            "s1": {"heating": {"s": 0.40, "e": 0.30, "w": 0.20}}
        }
    }
    out = _migrate_v3_to_v4(data)
    coeff = out["solar_coefficients_per_unit"]["s1"]
    assert coeff["heating"] == {"s": 0.40, "e": 0.30, "w": 0.20}
    assert coeff["cooling"] == {"s": 0.0, "e": 0.0, "w": 0.0}


def test_v4_skips_non_dict_entries():
    """Non-dict entries are dropped (defensive — should not happen in practice)."""
    data = {
        "solar_coefficients_per_unit": {
            "s1": {"s": 0.40, "e": 0.30, "w": 0.20},
            "s2": "garbage",
            "s3": [1, 2, 3],
        }
    }
    out = _migrate_v3_to_v4(data)
    assert "s1" in out["solar_coefficients_per_unit"]
    assert "s2" not in out["solar_coefficients_per_unit"]
    assert "s3" not in out["solar_coefficients_per_unit"]


def test_v4_empty_solar_coeffs_dict_is_safe():
    data = {"solar_coefficients_per_unit": {}}
    out = _migrate_v3_to_v4(data)
    assert out["solar_coefficients_per_unit"] == {}


def test_v4_missing_solar_coeffs_key_is_safe():
    data = {"correlation_data": {"some": "data"}}
    out = _migrate_v3_to_v4(data)
    assert "solar_coefficients_per_unit" not in out
    assert out["correlation_data"] == {"some": "data"}


# ---- learning_buffer_solar_per_unit ----

def test_v4_buffer_flat_list_routed_to_heating():
    """Pre-v4 flat list is routed to the heating regime."""
    samples = [(0.5, 0.0, 0.0, 0.3), (0.4, 0.1, 0.0, 0.25)]
    data = {"learning_buffer_solar_per_unit": {"s1": samples}}
    out = _migrate_v3_to_v4(data)
    buf = out["learning_buffer_solar_per_unit"]["s1"]
    assert buf["heating"] == samples
    assert buf["cooling"] == []


def test_v4_buffer_already_stratified_passes_through():
    data = {
        "learning_buffer_solar_per_unit": {
            "s1": {
                "heating": [(0.5, 0.0, 0.0, 0.3)],
                "cooling": [(0.0, 0.4, 0.0, 0.1)],
            }
        }
    }
    out = _migrate_v3_to_v4(data)
    buf = out["learning_buffer_solar_per_unit"]["s1"]
    assert buf["heating"] == [(0.5, 0.0, 0.0, 0.3)]
    assert buf["cooling"] == [(0.0, 0.4, 0.0, 0.1)]


def test_v4_buffer_partial_stratified_fills_missing():
    data = {
        "learning_buffer_solar_per_unit": {
            "s1": {"heating": [(0.5, 0.0, 0.0, 0.3)]}
        }
    }
    out = _migrate_v3_to_v4(data)
    buf = out["learning_buffer_solar_per_unit"]["s1"]
    assert buf["heating"] == [(0.5, 0.0, 0.0, 0.3)]
    assert buf["cooling"] == []


def test_v4_buffer_idempotent():
    data = {"learning_buffer_solar_per_unit": {"s1": [(0.5, 0.0, 0.0, 0.3)]}}
    out1 = _migrate_v3_to_v4(data)
    out2 = _migrate_v3_to_v4(out1)
    assert out2["learning_buffer_solar_per_unit"]["s1"]["heating"] == [
        (0.5, 0.0, 0.0, 0.3)
    ]
    assert out2["learning_buffer_solar_per_unit"]["s1"]["cooling"] == []


def test_v4_buffer_missing_key_is_safe():
    data = {"correlation_data": {}}
    out = _migrate_v3_to_v4(data)
    assert "learning_buffer_solar_per_unit" not in out


# ---- _sanitize_stratified ----

def test_sanitize_stratified_fills_missing_regimes():
    out = _sanitize_stratified({})
    assert out == {
        "heating": {"s": 0.0, "e": 0.0, "w": 0.0},
        "cooling": {"s": 0.0, "e": 0.0, "w": 0.0},
    }


def test_sanitize_stratified_clamps_negative_per_regime():
    out = _sanitize_stratified({
        "heating": {"s": 1.0, "e": -0.5, "w": 0.2},
        "cooling": {"s": -0.1, "e": 0.3, "w": -0.2},
    })
    assert out["heating"] == {"s": 1.0, "e": 0.0, "w": 0.2}
    assert out["cooling"] == {"s": 0.0, "e": 0.3, "w": 0.0}


def test_sanitize_stratified_handles_non_dict_input():
    """Defensive: non-dict input returns canonical empty stratified dict."""
    assert _sanitize_stratified(None) == {
        "heating": {"s": 0.0, "e": 0.0, "w": 0.0},
        "cooling": {"s": 0.0, "e": 0.0, "w": 0.0},
    }
    assert _sanitize_stratified("garbage") == {
        "heating": {"s": 0.0, "e": 0.0, "w": 0.0},
        "cooling": {"s": 0.0, "e": 0.0, "w": 0.0},
    }


def test_sanitize_stratified_handles_non_dict_inner_regime():
    """Non-dict regime value falls back to zeros for that regime."""
    out = _sanitize_stratified({
        "heating": {"s": 0.4, "e": 0.3, "w": 0.2},
        "cooling": "garbage",
    })
    assert out["heating"] == {"s": 0.4, "e": 0.3, "w": 0.2}
    assert out["cooling"] == {"s": 0.0, "e": 0.0, "w": 0.0}


# =============================================================================
# End-to-end chain — StorageManager._async_migrate → async_load_data /
# async_restore_data, with the v3 → v4 transform applied through HA Store hook.
# =============================================================================

def _make_coord(**overrides):
    """Coordinator stub with the surface area StorageManager touches."""
    coord = MagicMock()
    coord.entry = MagicMock()
    coord.entry.entry_id = "test_entry"
    coord.solar_azimuth = 180.0
    coord.solar_battery_decay = 0.80
    coord.energy_sensors = ["sensor.heater1"]
    # Coordinator attrs the load path writes into — initialise as the
    # canonical empty shapes so post-load assertions can introspect.
    coord._correlation_data = {}
    coord._correlation_data_per_unit = {}
    coord._aux_coefficients_per_unit = {}
    coord._solar_coefficients_per_unit = {}
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
    for k, v in overrides.items():
        setattr(coord, k, v)
    return coord


@pytest.mark.asyncio
async def test_async_migrate_chain_emits_canonical_v4_shape():
    """``_async_migrate`` from v3 produces shape ``async_load_data`` accepts.

    Specifically tests the migration class of bug from #874: a migration
    that returns a non-canonical inner shape which the load path silently
    drops because of ``isinstance(..., dict)`` filters.  v4 nests
    ``solar_coefficients_per_unit[entity]`` to ``{"heating": {...},
    "cooling": {...}}``; load uses ``_sanitize_stratified`` which expects
    that exact shape.
    """
    coord = _make_coord()
    sm = StorageManager(coord)
    pre_v4 = {
        "solar_coefficients_per_unit": {
            "sensor.heater1": {"s": 0.40, "e": 0.20, "w": 0.10},
        },
        "learning_buffer_solar_per_unit": {
            "sensor.heater1": [(0.5, 0.0, 0.0, 0.3)],
        },
    }
    migrated = await sm._async_migrate(
        old_major_version=3, old_minor_version=0, old_data=pre_v4
    )
    # Coefficient shape: {"heating": {...}, "cooling": {...}}.
    coeff = migrated["solar_coefficients_per_unit"]["sensor.heater1"]
    assert "heating" in coeff and "cooling" in coeff
    assert coeff["heating"] == {"s": 0.40, "e": 0.20, "w": 0.10}
    assert coeff["cooling"] == {"s": 0.40, "e": 0.20, "w": 0.10}
    # Buffer: {"heating": [...], "cooling": []}.
    buf = migrated["learning_buffer_solar_per_unit"]["sensor.heater1"]
    assert buf == {"heating": [(0.5, 0.0, 0.0, 0.3)], "cooling": []}


@pytest.mark.asyncio
async def test_full_chain_load_after_v3_to_v4_populates_coordinator():
    """Complete round-trip: v3 data → migrate → load → coordinator state.

    The Store hook sequence is: ``Store.async_load`` reads the disk file,
    sees ``version=3`` < current ``version=4``, calls ``_async_migrate``,
    returns the migrated data; ``async_load_data`` then deserialises it
    into the coordinator's underscored attrs via the canonical-v4 shape.
    """
    coord = _make_coord()
    pre_v4 = {
        "solar_coefficients_per_unit": {
            "sensor.heater1": {"s": 0.40, "e": 0.20, "w": 0.10},
        },
        "learning_buffer_solar_per_unit": {
            "sensor.heater1": [(0.5, 0.0, 0.0, 0.3)],
        },
    }

    # Patch Store at the import path so the StorageManager's internal
    # reference is the same MagicMock we configure here.  Pattern matches
    # ``test_storage_migration_v3.py``.
    with patch("custom_components.heating_analytics.storage.Store") as mock_store_cls:
        mock_store = mock_store_cls.return_value
        sm = StorageManager(coord)
        # Run the migration first so the mock returns the post-migration shape.
        migrated = await sm._async_migrate(
            old_major_version=3, old_minor_version=0, old_data=pre_v4
        )
        mock_store.async_load = AsyncMock(return_value=migrated)
        await sm.async_load_data()

    # Coefficients populated in canonical v4 shape — both regimes
    # present, with cooling seeded from heating.
    coeff = coord._solar_coefficients_per_unit["sensor.heater1"]
    assert coeff["heating"]["s"] == 0.40
    assert coeff["cooling"]["s"] == 0.40

    # Buffer populated as nested dict, not flat list.
    buf = coord._learning_buffer_solar_per_unit["sensor.heater1"]
    assert "heating" in buf and "cooling" in buf
    assert buf["heating"] == [(0.5, 0.0, 0.0, 0.3)]
    assert buf["cooling"] == []


@pytest.mark.asyncio
async def test_restore_data_runs_migration_chain_for_v3_backup():
    """Ad-hoc JSON backups skip the Store hook — ``async_restore_data``
    must run the migration chain explicitly, otherwise a v3 backup
    would land in coordinator state with the legacy flat shape and
    crash on the next regime read.
    """
    coord = _make_coord()
    sm = StorageManager(coord)
    pre_v4 = {
        "correlation_data": {"10": {"normal": 1.5}},  # required key
        "daily_history": {},  # required key
        "solar_coefficients_per_unit": {
            "sensor.heater1": {"s": 0.40, "e": 0.20, "w": 0.10},
        },
        "learning_buffer_solar_per_unit": {
            "sensor.heater1": [(0.5, 0.0, 0.0, 0.3)],
        },
    }
    # Write to a tempfile so async_restore_data can read it via
    # async_add_executor_job.  Mock the executor to call the function
    # directly (synchronously, since the test runs in event loop).
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(pre_v4, f)
        backup_path = f.name

    try:
        async def _run_executor(fn, *args, **kwargs):
            return fn(*args, **kwargs)
        coord.hass.async_add_executor_job = _run_executor

        await sm.async_restore_data(backup_path)

        # Coefficients restored in canonical v4 shape — both regimes
        # present, cooling seeded from heating.
        coeff = coord._solar_coefficients_per_unit["sensor.heater1"]
        assert coeff["heating"] == {"s": 0.40, "e": 0.20, "w": 0.10}
        assert coeff["cooling"] == {"s": 0.40, "e": 0.20, "w": 0.10}

        # Buffer restored as nested dict.
        buf = coord._learning_buffer_solar_per_unit["sensor.heater1"]
        assert buf["heating"] == [(0.5, 0.0, 0.0, 0.3)]
        assert buf["cooling"] == []
    finally:
        os.unlink(backup_path)


@pytest.mark.asyncio
async def test_async_migrate_idempotent_on_v4_data():
    """Already-migrated data passes through unchanged (sanitised).

    The Store may invoke the migration hook in pathological cases (e.g.
    user manually downgrades version field but the data is already
    canonical).  ``_migrate_v3_to_v4`` must be idempotent.
    """
    coord = _make_coord()
    sm = StorageManager(coord)
    canonical_v4 = {
        "solar_coefficients_per_unit": {
            "sensor.heater1": {
                "heating": {"s": 0.40, "e": 0.20, "w": 0.10},
                "cooling": {"s": 0.10, "e": 0.05, "w": 0.02},
            },
        },
        "learning_buffer_solar_per_unit": {
            "sensor.heater1": {
                "heating": [(0.5, 0.0, 0.0, 0.3)],
                "cooling": [(0.0, 0.0, 0.4, 0.1)],
            },
        },
    }
    out = await sm._async_migrate(
        old_major_version=3, old_minor_version=0, old_data=canonical_v4
    )
    # Heating regime preserved bit-exact; cooling regime preserved bit-exact.
    coeff = out["solar_coefficients_per_unit"]["sensor.heater1"]
    assert coeff["heating"] == {"s": 0.40, "e": 0.20, "w": 0.10}
    assert coeff["cooling"] == {"s": 0.10, "e": 0.05, "w": 0.02}
    buf = out["learning_buffer_solar_per_unit"]["sensor.heater1"]
    assert buf["heating"] == [(0.5, 0.0, 0.0, 0.3)]
    assert buf["cooling"] == [(0.0, 0.0, 0.4, 0.1)]
