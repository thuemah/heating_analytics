"""Tests for the 4D shadow inequality learner (#954 commit 7).

4D shadow path uses the unified ``_update_unit_solar_inequality`` with components=("s","e","w","diffuse"):
- One-sided projected-gradient (invariant #5) — only LIFTS coefficients.
- Heating regime only (cooling shutdown inverted — out of scope).
- Strict shadow — writes only to ``solar_coefficients_4d_per_unit``.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from custom_components.heating_analytics.learning import LearningManager


def _make_solar_calc_stub():
    """Mock SolarCalculator that returns benign defaults for all methods."""
    sc = MagicMock()
    sc.coordinator = None
    sc.calculate_unit_coefficient.return_value = {"s": 0.0, "e": 0.0, "w": 0.0}
    sc.calculate_unit_solar_impact.return_value = 0.0
    sc.get_approx_sun_pos.return_value = (-1.0, 180.0)  # below horizon -> 4D skips
    sc.reconstruct_potential_vector.return_value = (0.0, 0.0, 0.0)
    return sc


def test_inequality_4d_lifts_coefficients_on_shutdown():
    """Constraint violated -> all four components lift, none decrease."""
    manager = LearningManager()
    coeffs_4d = {
        "u1": {
            "heating": {"s": 0.05, "e": 0.05, "w": 0.05, "diffuse": 0.05, "learned": True},
            "cooling": {"s": 0.0, "e": 0.0, "w": 0.0, "diffuse": 0.0},
        }
    }
    # Battery values non-zero across all four components.
    battery_4d = (0.6, 0.4, 0.3, 0.2)
    expected_base = 2.0
    # Constraint LHS pre-update = 0.05*(0.6+0.4+0.3+0.2) = 0.075.
    # margin*base = 0.9*2.0 = 1.8 -> deficit = 1.725 -> definitely violated.

    before = dict(coeffs_4d["u1"]["heating"])
    status = manager._update_unit_solar_inequality(
        entity_id="u1",
        expected_unit_base=expected_base,
        battery_filtered_potential=battery_4d,
        solar_coefficients_per_unit=coeffs_4d,
        components=("s", "e", "w", "diffuse"),
    )
    assert status == "updated"
    after = coeffs_4d["u1"]["heating"]
    for k in ("s", "e", "w", "diffuse"):
        assert after[k] >= before[k], f"{k} decreased: {before[k]} -> {after[k]}"
        assert after[k] > before[k], f"{k} did not lift: {before[k]} -> {after[k]}"


def test_inequality_4d_one_sided():
    """Constraint already satisfied -> no update."""
    manager = LearningManager()
    coeffs_4d = {
        "u_ok": {
            "heating": {"s": 2.0, "e": 2.0, "w": 2.0, "diffuse": 2.0, "learned": True},
            "cooling": {"s": 0.0, "e": 0.0, "w": 0.0, "diffuse": 0.0},
        }
    }
    battery_4d = (0.6, 0.4, 0.3, 0.2)
    expected_base = 1.0
    # LHS = 2.0*(0.6+0.4+0.3+0.2) = 3.0 >> margin*base = 0.9.

    before = dict(coeffs_4d["u_ok"]["heating"])
    status = manager._update_unit_solar_inequality(
        entity_id="u_ok",
        expected_unit_base=expected_base,
        battery_filtered_potential=battery_4d,
        solar_coefficients_per_unit=coeffs_4d,
        components=("s", "e", "w", "diffuse"),
    )
    assert status == "non_binding"
    assert coeffs_4d["u_ok"]["heating"] == before


def test_inequality_4d_skips_unscreened_entity():
    """Entity NOT in screen_affected_entities -> 4D dict unchanged.

    Drives the full _process_per_unit_learning path (the gate that
    decides whether to call _update_unit_solar_inequality at all
    lives there, NOT in the 4D method itself).
    """
    from custom_components.heating_analytics.observation import (
        HourlyObservation,
        LearningConfig,
        ModelState,
    )
    from custom_components.heating_analytics.const import (
        MODE_HEATING,
        SOLAR_SHUTDOWN_MIN_BASE,
    )
    from datetime import datetime, timezone

    manager = LearningManager()

    # Entity is NOT in screen_affected_entities -> inequality (3D + 4D)
    # must be skipped entirely.  Verify the 4D dict stays empty.
    coeffs_3d: dict = {}
    coeffs_4d: dict = {}
    buffer_4d: dict = {}
    correlation_data: dict = {}
    correlation_data_per_unit: dict = {}

    # Pre-existing 4D entry so we can spot any unwanted mutation.
    coeffs_4d["u_unscreened"] = {
        "heating": {"s": 0.1, "e": 0.1, "w": 0.1, "diffuse": 0.1, "learned": True},
        "cooling": {"s": 0.0, "e": 0.0, "w": 0.0, "diffuse": 0.0},
    }
    before_state = {
        k: v for k, v in coeffs_4d["u_unscreened"]["heating"].items()
    }

    # Run _process_per_unit_learning directly with a shutdown-flagged
    # entity NOT in screen_affected_entities.
    manager._process_per_unit_learning(
        temp_key="0",
        wind_bucket="normal",
        avg_temp=0.0,
        avg_solar_vector=(0.6, 0.0, 0.0),
        total_energy_kwh=0.0,
        base_expected_kwh=2.0,
        energy_sensors=["u_unscreened"],
        hourly_delta_per_unit={"u_unscreened": 0.0},
        solar_enabled=True,
        learning_rate=0.01,
        solar_calculator=_make_solar_calc_stub(),
        get_predicted_unit_base_fn=lambda eid, t, w, at: 2.0,
        learning_buffer_per_unit={},
        correlation_data_per_unit=correlation_data_per_unit,
        observation_counts={},
        is_aux_active=False,
        aux_coefficients_per_unit={},
        learning_buffer_aux_per_unit={},
        solar_coefficients_per_unit=coeffs_3d,
        learning_buffer_solar_per_unit={},
        balance_point=15.0,
        unit_modes={"u_unscreened": MODE_HEATING},
        hourly_expected_per_unit={},
        hourly_expected_base_per_unit={"u_unscreened": 2.0},
        aux_affected_entities=None,
        is_cooldown_active=False,
        correction_percent=100.0,
        solar_dominant_entities=("u_unscreened",),  # shutdown-flagged
        screen_config=(True, True, True),
        screen_affected_entities=frozenset(),  # u_unscreened NOT in set
        solar_affected_entities=None,
        battery_filtered_potential=(0.6, 0.0, 0.0),
        solar_coefficients_4d_per_unit=coeffs_4d,
        learning_buffer_solar_4d_per_unit=buffer_4d,
        obs=HourlyObservation(
            timestamp=datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc),
            hour=12,
            avg_temp=0.0,
            inertia_temp=0.0,
            temp_key="0",
            effective_wind=0.0,
            wind_bucket="normal",
            bucket_counts={"normal": 30},
            avg_humidity=None,
            solar_factor=0.6,
            solar_vector=(0.6, 0.0, 0.0),
            solar_impact_raw=0.0,
            effective_solar_impact=0.0,
            total_energy_kwh=0.0,
            learning_energy_kwh=0.0,
            guest_impact_kwh=0.0,
            expected_kwh=2.0,
            base_expected_kwh=2.0,
            unit_breakdown={},
            unit_expected={},
            unit_expected_base={"u_unscreened": 2.0},
            aux_impact_kwh=0.0,
            aux_fraction=0.0,
            is_aux_dominant=False,
            sample_count=30,
            unit_modes={"u_unscreened": MODE_HEATING},
            battery_filtered_potential=(0.6, 0.0, 0.0),
            battery_filtered_potential_4d=(0.6, 0.0, 0.0, 0.2),
        ),
    )

    # Unscreened entity gated out of inequality (both 3D and 4D).
    assert coeffs_4d["u_unscreened"]["heating"] == before_state
