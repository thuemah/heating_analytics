"""Tests for #992 commit 3: ``_compute_4d_normalization_delta``
per-entity saturation + ``calculate_total_power_4d`` per-(scope, mode)
partition clamp.

Pre-c3 ``_compute_4d_normalization_delta`` returned the raw signed sum
``Σ_(heating) c_4d · pot_4d − Σ_(cooling) c_4d · pot_4d`` without any
per-entity saturation.  Under ``experimental_4d_primary = True`` this
delta replaces the 3D-saturated ``obs.solar_normalization_delta`` at
``learning.py:701``, inflating ``normalized_actual = max(0,
total_energy_kwh + solar_normalization_delta)`` whenever 4D-solar
overshoot exceeded what the entity's own base could absorb — the base
EMA then drifted upward on every sunny hour.

Post-c3 the helper applies per-entity saturation matching
``solar.calculate_saturation`` semantics: heating is clamped to the
entity's ``unit_expected_base``; cooling is additive (no clamp).

``calculate_total_power_4d`` also receives the per-(scope, mode)
partition refactor that ``calculate_total_power`` got in commits 1+2.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from custom_components.heating_analytics.const import (
    MODE_COOLING,
    MODE_HEATING,
)
from custom_components.heating_analytics.learning import LearningManager
from custom_components.heating_analytics.observation import LearningConfig


def _make_obs(*, unit_modes, unit_breakdown, unit_expected_base):
    """Minimal observation-like object for the helper under test."""
    obs = MagicMock()
    obs.timestamp = MagicMock()
    # Provide arithmetic on timestamp + timedelta — return a real datetime.
    from datetime import datetime, timedelta as _td
    real_ts = datetime(2026, 5, 20, 12, 0, 0)
    obs.timestamp.__add__ = lambda self, other: real_ts + other
    obs.unit_modes = unit_modes
    obs.unit_breakdown = unit_breakdown
    obs.unit_expected_base = unit_expected_base
    obs.correction_percent = 100.0
    obs.dni_avg = 800.0
    obs.dhi_avg = 100.0
    obs.ghi_avg = None
    obs.cloud_avg = None
    return obs


def _make_config(*, energy_sensors, solar_calculator):
    return LearningConfig(
        learning_enabled=True,
        solar_enabled=True,
        learning_rate=0.1,
        balance_point=12.0,
        energy_sensors=energy_sensors,
        aux_impact=0.0,
        solar_calculator=solar_calculator,
        screen_config=(False, False, False),
        experimental_4d_primary=True,
    )


def _make_solar_calc(*, per_entity_pot_4d, sun_elev=45.0, sun_az=180.0):
    """Solar calculator stub that returns scripted potential per entity.

    ``per_entity_pot_4d`` is ``{entity_id: (s, e, w, diffuse)}`` and the
    mock dispatches on the first positional arg (entity_id).
    """
    sc = MagicMock()
    sc.get_approx_sun_pos = MagicMock(return_value=(sun_elev, sun_az))

    def _potential_4d(entity_id, *_args, **_kwargs):
        return per_entity_pot_4d.get(entity_id, (0.0, 0.0, 0.0, 0.0))

    sc.calculate_unit_potential_4d = MagicMock(side_effect=_potential_4d)
    sc.coordinator = MagicMock()
    return sc


class TestNormalizationDeltaSaturation:
    """Per-entity saturation gate in ``_compute_4d_normalization_delta``."""

    def test_heating_overshoot_clamped_to_entity_base(self):
        """Single heating entity with predicted_4d > base.

        Pre-c3: delta = predicted_4d (no clamp) = 10.0
        Post-c3: delta = min(predicted_4d, base) = base = 3.0
        """
        # Coefficients (s, e, w, diffuse). Potential set so dot product
        # = 10.0.  s·c=10 dominates; pot=(10, 0, 0, 0) and c=(1,0,0,0).
        per_entity_pot = {"hp.vp": (10.0, 0.0, 0.0, 0.0)}
        sc = _make_solar_calc(per_entity_pot_4d=per_entity_pot)
        cfg = _make_config(energy_sensors=["hp.vp"], solar_calculator=sc)

        obs = _make_obs(
            unit_modes={"hp.vp": MODE_HEATING},
            unit_breakdown={"hp.vp": 5.0},     # actual consumption this hour
            unit_expected_base={"hp.vp": 3.0}, # dark-sky base = clamp ceiling
        )

        coefficients = {
            "hp.vp": {"heating": {"s": 1.0, "e": 0.0, "w": 0.0, "diffuse": 0.0}},
        }

        mgr = LearningManager()
        delta = mgr._compute_4d_normalization_delta(obs, cfg, coefficients)

        # Saturated to 3.0 (entity base), NOT 10.0 (raw c·p).
        assert delta == pytest.approx(3.0), (
            f"4D delta must saturate to per-entity base; got {delta}"
        )

    def test_heating_under_base_returns_raw_predicted(self):
        """When predicted_4d < base, saturation is a no-op and delta = predicted_4d.

        Sanity check: the clamp is one-sided (caps overshoot only).
        """
        per_entity_pot = {"hp.vp": (1.5, 0.0, 0.0, 0.0)}
        sc = _make_solar_calc(per_entity_pot_4d=per_entity_pot)
        cfg = _make_config(energy_sensors=["hp.vp"], solar_calculator=sc)
        obs = _make_obs(
            unit_modes={"hp.vp": MODE_HEATING},
            unit_breakdown={"hp.vp": 5.0},
            unit_expected_base={"hp.vp": 5.0},  # large base, no clamping
        )
        coefficients = {
            "hp.vp": {"heating": {"s": 1.0, "e": 0.0, "w": 0.0, "diffuse": 0.0}},
        }
        mgr = LearningManager()
        delta = mgr._compute_4d_normalization_delta(obs, cfg, coefficients)
        assert delta == pytest.approx(1.5)

    def test_cooling_remains_additive_no_clamp(self):
        """Cooling-mode entities are subtracted from delta with no saturation
        (matches MODE_COOLING branch of ``solar.calculate_saturation``).
        """
        per_entity_pot = {"hp.ac": (8.0, 0.0, 0.0, 0.0)}
        sc = _make_solar_calc(per_entity_pot_4d=per_entity_pot)
        cfg = _make_config(energy_sensors=["hp.ac"], solar_calculator=sc)
        obs = _make_obs(
            unit_modes={"hp.ac": MODE_COOLING},
            unit_breakdown={"hp.ac": 2.0},
            unit_expected_base={"hp.ac": 1.0},  # would clamp if heating; not for cooling
        )
        coefficients = {
            "hp.ac": {"cooling": {"s": 1.0, "e": 0.0, "w": 0.0, "diffuse": 0.0}},
        }
        mgr = LearningManager()
        delta = mgr._compute_4d_normalization_delta(obs, cfg, coefficients)
        # Cooling: -predicted_4d, regardless of base.
        assert delta == pytest.approx(-8.0)

    def test_mixed_heating_cooling_individual_treatment(self):
        """One heating (clamped) + one cooling (additive).

        delta = min(heating_predicted, heating_base) − cooling_predicted
              = min(10, 2) − 5 = 2 − 5 = −3
        """
        per_entity_pot = {
            "hp.heat": (10.0, 0.0, 0.0, 0.0),
            "hp.cool": (5.0, 0.0, 0.0, 0.0),
        }
        sc = _make_solar_calc(per_entity_pot_4d=per_entity_pot)
        cfg = _make_config(
            energy_sensors=["hp.heat", "hp.cool"], solar_calculator=sc
        )
        obs = _make_obs(
            unit_modes={"hp.heat": MODE_HEATING, "hp.cool": MODE_COOLING},
            unit_breakdown={"hp.heat": 1.5, "hp.cool": 2.0},
            unit_expected_base={"hp.heat": 2.0, "hp.cool": 1.0},
        )
        coefficients = {
            "hp.heat": {"heating": {"s": 1.0, "e": 0.0, "w": 0.0, "diffuse": 0.0}},
            "hp.cool": {"cooling": {"s": 1.0, "e": 0.0, "w": 0.0, "diffuse": 0.0}},
        }
        mgr = LearningManager()
        delta = mgr._compute_4d_normalization_delta(obs, cfg, coefficients)
        assert delta == pytest.approx(-3.0)

    def test_missing_expected_base_treats_entity_as_zero_base(self):
        """When ``unit_expected_base`` is missing for an entity, the
        heating clamp defaults to 0 — the entity contributes nothing
        rather than producing inflated 4D-only normalization.
        """
        per_entity_pot = {"hp.vp": (10.0, 0.0, 0.0, 0.0)}
        sc = _make_solar_calc(per_entity_pot_4d=per_entity_pot)
        cfg = _make_config(energy_sensors=["hp.vp"], solar_calculator=sc)
        obs = _make_obs(
            unit_modes={"hp.vp": MODE_HEATING},
            unit_breakdown={"hp.vp": 5.0},
            unit_expected_base={},  # missing!
        )
        coefficients = {
            "hp.vp": {"heating": {"s": 1.0, "e": 0.0, "w": 0.0, "diffuse": 0.0}},
        }
        mgr = LearningManager()
        delta = mgr._compute_4d_normalization_delta(obs, cfg, coefficients)
        # min(10, 0) = 0.  Safe default.
        assert delta == pytest.approx(0.0)
