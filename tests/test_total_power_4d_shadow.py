"""Tests for ``StatisticsManager.calculate_total_power_4d`` (#962).

The 4D variant is the read-path counterpart to the experimental 4D
shadow solar pipeline.  It is **strict shadow** — no live consumer
calls it.  These tests pin the contract and document the deliberate
omissions vs. the 3D path.

Test fixture pattern follows ``tests/test_solar_saturation.py``: a real
``HeatingDataCoordinator`` + real ``StatisticsManager`` + real
``SolarCalculator`` with the few external boundaries (sun-position,
weather attributes, GHI sensor) mocked.  Pure-compute paths are
exercised end-to-end so any silent contract drift surfaces.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from custom_components.heating_analytics.const import (
    DOMAIN,
    MODE_HEATING,
    MODE_COOLING,
)
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.solar import SolarCalculator
from custom_components.heating_analytics.statistics import StatisticsManager


ENTITY_A = "sensor.heater_1"
ENTITY_B = "sensor.heater_2"


class _MockHass:
    def __init__(self):
        self.states = MagicMock()
        self.states.get = MagicMock(return_value=None)
        self.data = {DOMAIN: {}}
        self.config_entries = MagicMock()
        self.bus = MagicMock()
        self.is_running = True
        # Lat/lon used only when get_approx_sun_pos goes to astral.  We
        # mock that method directly so the real values do not matter.
        self.config = MagicMock()
        self.config.latitude = 60.0
        self.config.longitude = 10.0


def _build_coordinator(
    *,
    entities=(ENTITY_A,),
    with_screens=False,
) -> HeatingDataCoordinator:
    """Build a real coordinator with a real StatisticsManager + SolarCalculator."""
    hass = _MockHass()
    entry = MagicMock()
    entry.data = {
        "energy_sensors": list(entities),
        "aux_affected_entities": list(entities),
        "outdoor_temp_sensor": "sensor.outdoor_temp",
        "balance_point": 15.0,
        "wind_speed_sensor": "sensor.wind_speed",
        "solar_enabled": True,
        # Screen affected list — wire entity_a in when with_screens=True
        # so screen_config_for_entity returns the install screen_config
        # for it.  entity_b stays unaffected to test the per-entity
        # screen-routing case (test #5).
        "screen_affected_entities": [ENTITY_A] if with_screens else [],
        "solar_affected_entities": list(entities),
    }
    coord = HeatingDataCoordinator(hass, entry)
    coord.statistics = StatisticsManager(coord)
    coord.solar = SolarCalculator(coord)

    # Configure screen_config: south-screened install when with_screens.
    if with_screens:
        coord.screen_config = (True, False, False)
    else:
        coord.screen_config = (False, False, False)
    coord._screen_affected_set = set([ENTITY_A]) if with_screens else set()

    # Seed the base model + aux dicts so _get_prediction_from_model
    # returns predictable values.
    coord._correlation_data = {"_id": "global_base"}
    coord._aux_coefficients = {"_id": "global_aux"}
    coord._correlation_data_per_unit = {
        e: {"_id": f"unit_base_{e}"} for e in entities
    }
    coord._aux_coefficients_per_unit = {
        e: {"_id": f"unit_aux_{e}"} for e in entities
    }

    coord._hourly_delta_per_unit = {e: 0.0 for e in entities}
    coord._collector.aux_breakdown = {}

    # Default mode = heating.
    coord.get_unit_mode = MagicMock(return_value=MODE_HEATING)

    # Force a predictable installation screen slider.
    coord.solar_correction_percent = 100.0
    # Default carryover state (test #8 will override).
    coord._solar_carryover_state = 0.0

    return coord


def _bind_prediction(coord, *, base_kwh: float, aux_kwh: float):
    """Make _get_prediction_from_model return fixed values."""
    def mock_get_pred(data_map, temp_key, wind_bucket, temp, bp, apply_scaling=True):
        doc_id = data_map.get("_id", "") if isinstance(data_map, dict) else ""
        if doc_id.startswith("unit_base") or doc_id == "global_base":
            return base_kwh
        if doc_id.startswith("unit_aux") or doc_id == "global_aux":
            return aux_kwh
        return 0.0
    coord.statistics._get_prediction_from_model = MagicMock(side_effect=mock_get_pred)
    coord._get_predicted_kwh = MagicMock(return_value=base_kwh)


def _bind_sun(coord, *, elev: float, az: float = 180.0):
    coord.solar.get_approx_sun_pos = MagicMock(return_value=(elev, az))


def _bind_weather(coord, *, dni=None, dhi=None, ghi=None, cloud=None):
    coord._get_weather_attribute_with_fallback = MagicMock(
        side_effect=lambda attr: dni if attr == "direct_normal_irradiance"
        else (dhi if attr == "diffuse_radiation" else None)
    )
    coord._get_ghi = MagicMock(return_value=ghi)
    coord._get_cloud_coverage = MagicMock(return_value=cloud)


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_4d_shadow_returns_same_shape_as_3d():
    """Both functions return matching top-level + breakdown keys.

    4D adds two marker fields (``pipeline`` and ``dni_dhi_source``)
    and fixes ``carryover_release_kwh`` at 0 by design — every other
    key must match exactly.
    """
    coord = _build_coordinator()
    _bind_prediction(coord, base_kwh=1.0, aux_kwh=0.0)
    _bind_sun(coord, elev=45.0, az=180.0)
    _bind_weather(coord, dni=500.0, dhi=100.0)
    coord.balance_point = 15.0
    coord.auxiliary_heating_active = False
    coord.aux_affected_entities = []

    now = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
    result_3d = coord.statistics.calculate_total_power(
        temp=10.0, effective_wind=0.0, solar_impact=0.0,
        is_aux_active=False, override_now=now,
    )
    result_4d = coord.statistics.calculate_total_power_4d(
        temp=10.0, effective_wind=0.0, solar_impact=0.0,
        is_aux_active=False, override_now=now,
    )

    expected_extra_top = {"pipeline", "dni_dhi_source"}
    extra = set(result_4d.keys()) - set(result_3d.keys())
    assert extra == expected_extra_top
    assert result_3d.keys() <= result_4d.keys()
    # Breakdown keys identical.
    assert set(result_4d["breakdown"].keys()) == set(result_3d["breakdown"].keys())
    assert result_4d["pipeline"] == "4d_shadow"
    # carryover_release_kwh is fixed at 0 in 4D regardless of 3D's value.
    assert result_4d["breakdown"]["carryover_release_kwh"] == 0.0


def test_4d_shadow_zero_coefficients_zero_solar():
    """Empty ``_solar_coefficients_4d_per_unit`` -> 0 solar reduction.

    Even with full sun, an unlearned 4D regime returns a zero coefficient
    vector — no default-azimuth-decomposition fallback exists in 4D
    (deliberate; see calculate_unit_coefficient_4d docstring).
    """
    coord = _build_coordinator()
    _bind_prediction(coord, base_kwh=2.0, aux_kwh=0.0)
    _bind_sun(coord, elev=45.0, az=180.0)
    _bind_weather(coord, dni=800.0, dhi=150.0)
    coord.balance_point = 15.0
    coord.auxiliary_heating_active = False
    coord.aux_affected_entities = []
    coord._solar_coefficients_4d_per_unit = {}  # empty: no learning yet

    result = coord.statistics.calculate_total_power_4d(
        temp=10.0, effective_wind=0.0, solar_impact=0.0,
        is_aux_active=False,
        override_now=datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
    )
    assert result["breakdown"]["solar_reduction_kwh"] == 0.0
    assert result["dni_dhi_source"] == "native"  # sun > 0 + dni/dhi available


def test_4d_shadow_dark_hour_matches_3d_solar_zero():
    """Below-horizon hour: 3D and 4D both yield 0 solar; base/aux match."""
    coord = _build_coordinator()
    _bind_prediction(coord, base_kwh=1.5, aux_kwh=0.3)
    _bind_sun(coord, elev=-5.0, az=180.0)  # below horizon
    _bind_weather(coord, dni=0.0, dhi=0.0)
    coord.balance_point = 15.0
    coord.auxiliary_heating_active = True
    coord.aux_affected_entities = [ENTITY_A]

    now = datetime(2025, 12, 1, 3, 0, 0, tzinfo=timezone.utc)
    r3 = coord.statistics.calculate_total_power(
        temp=5.0, effective_wind=0.0, solar_impact=0.0,
        is_aux_active=True, override_now=now,
    )
    r4 = coord.statistics.calculate_total_power_4d(
        temp=5.0, effective_wind=0.0, solar_impact=0.0,
        is_aux_active=True, override_now=now,
    )
    assert r3["breakdown"]["solar_reduction_kwh"] == 0.0
    assert r4["breakdown"]["solar_reduction_kwh"] == 0.0
    assert r3["global_base_kwh"] == r4["global_base_kwh"]
    assert r3["global_aux_reduction_kwh"] == r4["global_aux_reduction_kwh"]
    assert r4["dni_dhi_source"] == "no_sun"


def test_4d_shadow_saturation_clipped():
    """Coefficient large enough to exceed base -> wasted > 0, applied capped."""
    coord = _build_coordinator()
    _bind_prediction(coord, base_kwh=0.5, aux_kwh=0.0)  # low base
    _bind_sun(coord, elev=45.0, az=180.0)
    _bind_weather(coord, dni=800.0, dhi=200.0)
    coord.balance_point = 15.0
    coord.auxiliary_heating_active = False
    coord.aux_affected_entities = []
    # Massive south coefficient -> solar potential will exceed base 0.5.
    coord._solar_coefficients_4d_per_unit = {
        ENTITY_A: {
            "heating": {
                "s": 0.05, "e": 0.0, "w": 0.0, "diffuse": 0.005, "learned": True,
            }
        }
    }

    result = coord.statistics.calculate_total_power_4d(
        temp=10.0, effective_wind=0.0, solar_impact=0.0,
        is_aux_active=False,
        override_now=datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
    )
    bd_entity = result["unit_breakdown"][ENTITY_A]
    assert bd_entity["raw_solar_kwh"] > bd_entity["solar_reduction_kwh"]
    assert bd_entity["solar_wasted_kwh"] > 0.0
    assert bd_entity["solar_reduction_kwh"] <= bd_entity["base_kwh"]
    # unspecified_kwh tracks global_net - sum(unit nets); must stay clean
    # under saturation (no surprise residual).  Tolerate 1e-3 rounding.
    assert abs(result["breakdown"]["unspecified_kwh"]) < 0.01


def test_4d_shadow_unscreened_entity_uses_t1():
    """Per-entity screen routing: unscreened entity sees t=1.0 on all dirs.

    With install screen at 50% slider and south-only screened config:
    the screened ENTITY_A's south potential takes a hit; ENTITY_B
    (NOT in screen_affected_entities) sees full transmittance across
    all directions.  We assert ENTITY_B's solar = unscreened-equivalent.
    """
    coord = _build_coordinator(entities=(ENTITY_A, ENTITY_B), with_screens=True)
    # Install screens half-deployed.
    coord.solar_correction_percent = 50.0
    _bind_prediction(coord, base_kwh=5.0, aux_kwh=0.0)
    _bind_sun(coord, elev=45.0, az=180.0)
    _bind_weather(coord, dni=600.0, dhi=100.0)
    coord.balance_point = 15.0
    coord.auxiliary_heating_active = False
    coord.aux_affected_entities = []

    # Both entities have identical 4D coefficients.
    coeffs = {
        "heating": {
            "s": 0.001, "e": 0.0, "w": 0.0, "diffuse": 0.001, "learned": True,
        }
    }
    coord._solar_coefficients_4d_per_unit = {
        ENTITY_A: coeffs,
        ENTITY_B: coeffs,
    }

    result = coord.statistics.calculate_total_power_4d(
        temp=10.0, effective_wind=0.0, solar_impact=0.0,
        is_aux_active=False,
        override_now=datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
    )
    raw_a = result["unit_breakdown"][ENTITY_A]["raw_solar_kwh"]
    raw_b = result["unit_breakdown"][ENTITY_B]["raw_solar_kwh"]
    # ENTITY_B (unscreened) must see >= ENTITY_A (south-screened at 50 %)
    # because its south transmittance is 1.0 vs ENTITY_A's ~0.54.
    assert raw_b > raw_a, (
        f"Unscreened entity should receive more solar than south-screened: "
        f"a={raw_a}, b={raw_b}"
    )

    # Sanity: compute the unscreened-equivalent on a different
    # coordinator with screen_pct=100 and confirm ENTITY_B matches it
    # (modulo rounding) — proves the unscreened entity ignored the
    # slider entirely.
    coord_ref = _build_coordinator(entities=(ENTITY_B,), with_screens=False)
    coord_ref.solar_correction_percent = 100.0
    _bind_prediction(coord_ref, base_kwh=5.0, aux_kwh=0.0)
    _bind_sun(coord_ref, elev=45.0, az=180.0)
    _bind_weather(coord_ref, dni=600.0, dhi=100.0)
    coord_ref.balance_point = 15.0
    coord_ref.auxiliary_heating_active = False
    coord_ref.aux_affected_entities = []
    coord_ref._solar_coefficients_4d_per_unit = {ENTITY_B: coeffs}
    ref = coord_ref.statistics.calculate_total_power_4d(
        temp=10.0, effective_wind=0.0, solar_impact=0.0,
        is_aux_active=False,
        override_now=datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
    )
    assert abs(ref["unit_breakdown"][ENTITY_B]["raw_solar_kwh"] - raw_b) < 1e-3


def test_4d_shadow_mode_stratified_coeffs():
    """Heating vs cooling regime picks the right per-regime 4D coefficient."""
    coord = _build_coordinator()
    _bind_prediction(coord, base_kwh=2.0, aux_kwh=0.0)
    _bind_sun(coord, elev=45.0, az=180.0)
    _bind_weather(coord, dni=600.0, dhi=100.0)
    coord.balance_point = 15.0
    coord.auxiliary_heating_active = False
    coord.aux_affected_entities = []
    coord._solar_coefficients_4d_per_unit = {
        ENTITY_A: {
            "heating": {
                "s": 0.001, "e": 0.0, "w": 0.0, "diffuse": 0.0, "learned": True,
            },
            "cooling": {
                "s": 0.005, "e": 0.0, "w": 0.0, "diffuse": 0.0, "learned": True,
            },
        }
    }

    coord.get_unit_mode = MagicMock(return_value=MODE_HEATING)
    r_heat = coord.statistics.calculate_total_power_4d(
        temp=10.0, effective_wind=0.0, solar_impact=0.0,
        is_aux_active=False,
        override_now=datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
    )
    # Force the cooling cold-start gate off — fresh coordinator has no
    # cooling base bucket so _is_cooling_solar_cold_start returns True.
    coord.statistics._is_cooling_solar_cold_start = MagicMock(return_value=False)
    coord.get_unit_mode = MagicMock(return_value=MODE_COOLING)
    r_cool = coord.statistics.calculate_total_power_4d(
        temp=10.0, effective_wind=0.0, solar_impact=0.0,
        is_aux_active=False,
        unit_modes={ENTITY_A: MODE_COOLING},
        override_now=datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
    )
    raw_heat = r_heat["unit_breakdown"][ENTITY_A]["raw_solar_kwh"]
    raw_cool = r_cool["unit_breakdown"][ENTITY_A]["raw_solar_kwh"]
    # Cooling coefficient is 5x heating; raw solar should be ~5x.
    assert raw_cool > raw_heat * 4.0, (
        f"Mode-stratified coefficients should differ ~5x: "
        f"heat={raw_heat}, cool={raw_cool}"
    )


def test_4d_shadow_dni_dhi_unavailable_zero_solar():
    """No GHI, no native DNI/DHI, no cloud_coverage -> 0 solar, source 'unavailable'."""
    coord = _build_coordinator()
    _bind_prediction(coord, base_kwh=2.0, aux_kwh=0.0)
    _bind_sun(coord, elev=45.0, az=180.0)
    _bind_weather(coord, dni=None, dhi=None, ghi=None, cloud=None)
    coord.balance_point = 15.0
    coord.auxiliary_heating_active = False
    coord.aux_affected_entities = []
    # Even with a learned 4D coefficient, no irradiance signal -> 0.
    coord._solar_coefficients_4d_per_unit = {
        ENTITY_A: {
            "heating": {
                "s": 0.01, "e": 0.0, "w": 0.0, "diffuse": 0.01, "learned": True,
            }
        }
    }

    result = coord.statistics.calculate_total_power_4d(
        temp=10.0, effective_wind=0.0, solar_impact=0.0,
        is_aux_active=False,
        override_now=datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
    )
    assert result["breakdown"]["solar_reduction_kwh"] == 0.0
    assert result["dni_dhi_source"] == "unavailable"


def test_4d_shadow_no_carryover_release():
    """Even with a live ``_solar_carryover_state``, 4D output sets it to 0.

    Documents the deliberate omission of #896 from the 4D shadow path.
    """
    coord = _build_coordinator()
    _bind_prediction(coord, base_kwh=2.0, aux_kwh=0.0)
    _bind_sun(coord, elev=45.0, az=180.0)
    _bind_weather(coord, dni=500.0, dhi=100.0)
    coord.balance_point = 15.0
    coord.auxiliary_heating_active = False
    coord.aux_affected_entities = []
    coord._solar_carryover_state = 5.0  # arbitrary non-zero
    coord.solar_battery_decay = 0.80

    result = coord.statistics.calculate_total_power_4d(
        temp=10.0, effective_wind=0.0, solar_impact=0.0,
        is_aux_active=False,
        override_now=datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
    )
    assert result["breakdown"]["carryover_release_kwh"] == 0.0


def test_4d_shadow_override_dni_dhi_injects_values_and_labels_source():
    """``override_dni_dhi`` skips ``resolve_dni_dhi`` and tags source.

    The replay path used by ``_compute_total_power_4d_divergence_report``
    passes historical (dni, dhi) directly.  When the override is set
    the resolution ladder is skipped entirely and the output
    ``dni_dhi_source`` is ``"replay_override"``.  The override also
    bypasses the ``solar_enabled`` gate so historical replay survives
    a user toggling solar off after the data was collected.
    """
    coord = _build_coordinator()
    _bind_prediction(coord, base_kwh=2.0, aux_kwh=0.0)
    _bind_sun(coord, elev=45.0, az=180.0)
    # Bind the weather signal ladder to None so that ANY use of
    # resolve_dni_dhi would return ("none", 0, 0) — proves the override
    # is actually skipping the call.
    _bind_weather(coord, dni=None, dhi=None, ghi=None, cloud=None)
    coord.balance_point = 15.0
    coord.auxiliary_heating_active = False
    coord.aux_affected_entities = []
    coord._solar_coefficients_4d_per_unit = {
        ENTITY_A: {
            "heating": {
                "s": 0.001, "e": 0.0, "w": 0.0, "diffuse": 0.001, "learned": True,
            }
        }
    }

    result = coord.statistics.calculate_total_power_4d(
        temp=10.0, effective_wind=0.0, solar_impact=0.0,
        is_aux_active=False,
        override_now=datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
        override_dni_dhi=(700.0, 120.0),
    )
    assert result["dni_dhi_source"] == "replay_override"
    # Solar reduction must be > 0 — proves the injected DNI/DHI actually
    # fed the per-entity potential pipeline despite the dead live ladder.
    assert result["breakdown"]["solar_reduction_kwh"] > 0.0

    # Bypass-solar-enabled regression: even with solar_enabled=False,
    # override_dni_dhi still drives the 4D leg.
    coord2 = _build_coordinator()
    _bind_prediction(coord2, base_kwh=2.0, aux_kwh=0.0)
    _bind_sun(coord2, elev=45.0, az=180.0)
    _bind_weather(coord2, dni=None, dhi=None, ghi=None, cloud=None)
    coord2.balance_point = 15.0
    coord2.auxiliary_heating_active = False
    coord2.aux_affected_entities = []
    coord2._solar_coefficients_4d_per_unit = {
        ENTITY_A: {
            "heating": {
                "s": 0.001, "e": 0.0, "w": 0.0, "diffuse": 0.001, "learned": True,
            }
        }
    }
    coord2.solar_enabled = False
    result2 = coord2.statistics.calculate_total_power_4d(
        temp=10.0, effective_wind=0.0, solar_impact=0.0,
        is_aux_active=False,
        override_now=datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
        override_dni_dhi=(700.0, 120.0),
    )
    assert result2["dni_dhi_source"] == "replay_override"
    assert result2["breakdown"]["solar_reduction_kwh"] > 0.0


def test_4d_shadow_override_sun_pos_bypasses_astral():
    """``override_sun_pos`` skips ``get_approx_sun_pos``.

    The diagnose replay path computes sun position once per hour and
    injects it to avoid paying the astral cost per call.  We mock
    ``get_approx_sun_pos`` to raise — proves the call is skipped when
    the override is set.
    """
    coord = _build_coordinator()
    _bind_prediction(coord, base_kwh=2.0, aux_kwh=0.0)
    # Make get_approx_sun_pos raise — if the override path is wired
    # correctly this exception is never hit.
    coord.solar.get_approx_sun_pos = MagicMock(
        side_effect=AssertionError("astral should not be called")
    )
    _bind_weather(coord, dni=None, dhi=None, ghi=None, cloud=None)
    coord.balance_point = 15.0
    coord.auxiliary_heating_active = False
    coord.aux_affected_entities = []
    coord._solar_coefficients_4d_per_unit = {
        ENTITY_A: {
            "heating": {
                "s": 0.001, "e": 0.0, "w": 0.0, "diffuse": 0.001, "learned": True,
            }
        }
    }

    # Both overrides must be set together to fully bypass the live
    # ladder.  ``override_sun_pos`` on its own only skips astral —
    # ``resolve_dni_dhi`` still runs.
    result = coord.statistics.calculate_total_power_4d(
        temp=10.0, effective_wind=0.0, solar_impact=0.0,
        is_aux_active=False,
        override_now=datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
        override_dni_dhi=(700.0, 120.0),
        override_sun_pos=(45.0, 180.0),
    )
    assert result["dni_dhi_source"] == "replay_override"
    assert result["breakdown"]["solar_reduction_kwh"] > 0.0


def test_4d_shadow_override_correction_percent_overrides_live_slider():
    """``override_correction_percent`` wins over the live coordinator state.

    Regression test for the replay-path bug where the divergence block
    fed historical DNI/DHI but the 4D leg used the present-day
    ``solar_correction_percent`` for transmittance — producing
    artificial 3D-vs-4D divergence whenever the slider had moved
    during the replay window.  Strategy: set the live slider to the
    fully-closed position (0 %, maximum attenuation on screened
    facades) and replay with ``override_correction_percent=100`` (open,
    full transmittance); the solar reduction must match a reference
    coordinator whose live slider is 100, proving the override
    actually replaces the live value at the transmittance call.
    """
    # Coordinator with a south-screened install + live slider at 0 %
    # (closed).  Without the override, a south-facing entity would see
    # near-zero solar.
    coord = _build_coordinator(with_screens=True)
    coord.solar_correction_percent = 0.0
    _bind_prediction(coord, base_kwh=5.0, aux_kwh=0.0)
    _bind_sun(coord, elev=45.0, az=180.0)
    _bind_weather(coord, dni=600.0, dhi=100.0)
    coord.balance_point = 15.0
    coord.auxiliary_heating_active = False
    coord.aux_affected_entities = []
    coord._solar_coefficients_4d_per_unit = {
        ENTITY_A: {
            "heating": {
                "s": 0.001, "e": 0.0, "w": 0.0, "diffuse": 0.001, "learned": True,
            }
        }
    }
    result_override = coord.statistics.calculate_total_power_4d(
        temp=10.0, effective_wind=0.0, solar_impact=0.0,
        is_aux_active=False,
        override_now=datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
        override_dni_dhi=(600.0, 100.0),
        override_sun_pos=(45.0, 180.0),
        override_correction_percent=100.0,
    )

    # Reference coordinator: same setup but live slider already at 100 %,
    # no override.  The two solar reductions must match (modulo rounding).
    coord_ref = _build_coordinator(with_screens=True)
    coord_ref.solar_correction_percent = 100.0
    _bind_prediction(coord_ref, base_kwh=5.0, aux_kwh=0.0)
    _bind_sun(coord_ref, elev=45.0, az=180.0)
    _bind_weather(coord_ref, dni=600.0, dhi=100.0)
    coord_ref.balance_point = 15.0
    coord_ref.auxiliary_heating_active = False
    coord_ref.aux_affected_entities = []
    coord_ref._solar_coefficients_4d_per_unit = coord._solar_coefficients_4d_per_unit
    result_ref = coord_ref.statistics.calculate_total_power_4d(
        temp=10.0, effective_wind=0.0, solar_impact=0.0,
        is_aux_active=False,
        override_now=datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
        override_dni_dhi=(600.0, 100.0),
        override_sun_pos=(45.0, 180.0),
    )

    sr_override = result_override["breakdown"]["solar_reduction_kwh"]
    sr_ref = result_ref["breakdown"]["solar_reduction_kwh"]
    assert sr_override > 0.0, "override path produced zero solar"
    assert abs(sr_override - sr_ref) < 1e-3, (
        f"override_correction_percent should match live slider at same value: "
        f"override={sr_override}, ref={sr_ref}"
    )

    # Sanity: without the override, the closed-slider coordinator
    # should produce strictly LESS solar than the open one.
    result_no_override = coord.statistics.calculate_total_power_4d(
        temp=10.0, effective_wind=0.0, solar_impact=0.0,
        is_aux_active=False,
        override_now=datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
        override_dni_dhi=(600.0, 100.0),
        override_sun_pos=(45.0, 180.0),
    )
    assert result_no_override["breakdown"]["solar_reduction_kwh"] < sr_override, (
        "without override, closed-slider coordinator should produce less solar"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
