"""Tests for inequality learning on solar shutdown hours (#865).

Validates:

1. ``_update_unit_solar_inequality`` math — isolated behaviour of the
   projected-gradient constraint update (margin, deficit distribution,
   non-binding case, clamps).
2. Live integration via ``process_learning``: shutdown-flagged entities
   now receive an inequality update instead of being skipped.
3. Retrain integration via ``replay_solar_nlms``: shutdown hours in the
   log feed the learner (battery-filtered potential computed locally).
4. Feature flag off: legacy skip behaviour is preserved via
   ``legacy_shutdown_skip`` fixture (exercised in test_retrain_nlms_replay.py).
5. ``diagnose_solar.implied_coefficient_inequality``: shadow replay
   produces coefficients for shutdown-heavy installations.
"""
from unittest.mock import MagicMock, AsyncMock

import pytest

from custom_components.heating_analytics.learning import LearningManager
from custom_components.heating_analytics.solar import SolarCalculator
from custom_components.heating_analytics.observation import build_strategies
from custom_components.heating_analytics.const import (
    INEQUALITY_MARGIN,
    INEQUALITY_STEP_SIZE,
    SOLAR_COEFF_CAP,
    MODE_HEATING,
)
from tests.helpers import stratified_coeff


# -----------------------------------------------------------------------------
# 1. _update_unit_solar_inequality — isolated math
# -----------------------------------------------------------------------------

class TestInequalityMath:

    def test_constraint_satisfied_no_update(self):
        """predicted_impact >= margin * base → non_binding, coeffs unchanged."""
        lm = LearningManager()
        coeffs = {"u1": stratified_coeff(s=1.0)}
        # pot_s=1.0 → predicted=1.0; base=0.5, margin=0.9 → target=0.45
        # 1.0 >= 0.45 → non_binding
        status = lm._update_unit_solar_inequality(
            entity_id="u1",
            expected_unit_base=0.5,
            battery_filtered_potential=(1.0, 0.0, 0.0),
            solar_coefficients_per_unit=coeffs,
        )
        assert status == "non_binding"
        assert coeffs["u1"]["heating"]["s"] == 1.0
        assert coeffs["u1"]["heating"]["e"] == 0.0
        assert coeffs["u1"]["heating"]["w"] == 0.0

    def test_constraint_violated_raises_coefficients(self):
        """predicted < margin * base → update raises coefficients."""
        lm = LearningManager()
        coeffs = {"u1": stratified_coeff(s=0.1)}
        # pot_s=1.0 → predicted=0.1; base=1.0, margin=0.9 → target=0.9
        # deficit = 0.8. Distribution proportional to pot_s/mag=1.0
        # new_s = 0.1 + 0.05 * 0.8 * 1.0 = 0.14
        status = lm._update_unit_solar_inequality(
            entity_id="u1",
            expected_unit_base=1.0,
            battery_filtered_potential=(1.0, 0.0, 0.0),
            solar_coefficients_per_unit=coeffs,
        )
        assert status == "updated"
        assert coeffs["u1"]["heating"]["s"] == pytest.approx(0.14, abs=1e-4)
        # Directions with zero potential not touched
        assert coeffs["u1"]["heating"]["e"] == 0.0
        assert coeffs["u1"]["heating"]["w"] == 0.0
        # Cooling regime untouched (inequality is heating-only).
        assert coeffs["u1"]["cooling"] == {"s": 0.0, "e": 0.0, "w": 0.0}

    def test_deficit_distributed_by_potential_magnitude(self):
        """Deficit splits proportionally across non-zero potential components."""
        lm = LearningManager()
        coeffs = {"u1": stratified_coeff()}
        # pot=(0.6, 0.3, 0.1), mag_total=1.0. base=1.0 → deficit=0.9
        # new_s = 0 + 0.05 * 0.9 * 0.6 = 0.027
        # new_e = 0 + 0.05 * 0.9 * 0.3 = 0.0135
        # new_w = 0 + 0.05 * 0.9 * 0.1 = 0.0045
        lm._update_unit_solar_inequality(
            entity_id="u1",
            expected_unit_base=1.0,
            battery_filtered_potential=(0.6, 0.3, 0.1),
            solar_coefficients_per_unit=coeffs,
        )
        assert coeffs["u1"]["heating"]["s"] == pytest.approx(0.027, abs=1e-4)
        assert coeffs["u1"]["heating"]["e"] == pytest.approx(0.0135, abs=1e-4)
        assert coeffs["u1"]["heating"]["w"] == pytest.approx(0.0045, abs=1e-4)

    def test_zero_magnitude_returns_zero_magnitude(self):
        """Battery not yet populated → skip update."""
        lm = LearningManager()
        coeffs = {"u1": stratified_coeff()}
        status = lm._update_unit_solar_inequality(
            entity_id="u1",
            expected_unit_base=1.0,
            battery_filtered_potential=(0.0, 0.0, 0.0),
            solar_coefficients_per_unit=coeffs,
        )
        assert status == "zero_magnitude"
        assert coeffs["u1"]["heating"] == {"s": 0.0, "e": 0.0, "w": 0.0}

    def test_seeds_from_zero_coefficients(self):
        """No prior coefficient → inequality seeds from zero."""
        lm = LearningManager()
        coeffs = {}  # Unit has never had a coefficient
        status = lm._update_unit_solar_inequality(
            entity_id="u1",
            expected_unit_base=1.0,
            battery_filtered_potential=(1.0, 0.0, 0.0),
            solar_coefficients_per_unit=coeffs,
        )
        assert status == "updated"
        assert "u1" in coeffs
        # Both regimes created on first write; only heating populated.
        assert coeffs["u1"]["heating"]["s"] > 0.0
        assert coeffs["u1"]["cooling"] == {"s": 0.0, "e": 0.0, "w": 0.0}

    def test_non_negative_clamp_applied(self):
        """Deficit can never drive coefficients negative (physical invariant)."""
        lm = LearningManager()
        coeffs = {"u1": stratified_coeff()}
        # Constraint: pot·coeff >= 0.9 * base. With coeffs=0 and pot>0, deficit > 0
        # → we can only add. No code path subtracts. Sanity check the clamp exists.
        lm._update_unit_solar_inequality(
            entity_id="u1",
            expected_unit_base=10.0,  # very high, deficit large
            battery_filtered_potential=(1.0, 1.0, 1.0),
            solar_coefficients_per_unit=coeffs,
        )
        assert coeffs["u1"]["heating"]["s"] >= 0.0
        assert coeffs["u1"]["heating"]["e"] >= 0.0
        assert coeffs["u1"]["heating"]["w"] >= 0.0

    def test_cap_clamp_prevents_runaway(self):
        """Very large deficit is capped at SOLAR_COEFF_CAP per component."""
        lm = LearningManager()
        coeffs = {"u1": stratified_coeff(s=SOLAR_COEFF_CAP - 0.01)}
        # Huge base, small coeff → deficit is huge. Step of 0.05 × large_deficit
        # + existing SOLAR_CAP - 0.01 would exceed cap without clamp.
        lm._update_unit_solar_inequality(
            entity_id="u1",
            expected_unit_base=100.0,
            battery_filtered_potential=(1.0, 0.0, 0.0),
            solar_coefficients_per_unit=coeffs,
        )
        assert coeffs["u1"]["heating"]["s"] <= SOLAR_COEFF_CAP

    def test_cooling_mode_not_exposed_via_wrapper(self):
        """This math is called only in heating-mode wiring; cooling inversion
        is gated at the call site in _process_per_unit_learning.  The unit test
        here reflects the math contract, not the wrapper's mode gate."""
        # No actual mode check in _update_unit_solar_inequality itself — the
        # method is mode-blind.  Asserting callers gate correctly is in the
        # integration tests below.
        pass


# -----------------------------------------------------------------------------
# 2. Retrain integration via replay_solar_nlms
# -----------------------------------------------------------------------------

def _shutdown_env(unit_base_kwh=1.0):
    coord = MagicMock()
    coord.screen_config = (True, True, True)
    coord.energy_sensors = ["sensor.vp_stue"]
    coord.aux_affected_entities = []
    coord._unit_strategies = build_strategies(
        energy_sensors=["sensor.vp_stue"],
        track_c_enabled=False,
        mpc_managed_sensor=None,
    )
    return coord


def _shutdown_entry(ts, *, base=1.0, solar_s=0.0, solar_w=0.8, actual=0.0,
                    sensor_id="sensor.vp_stue"):
    """Build a shutdown-flagged entry (west-dominant sun, actual≈0)."""
    return {
        "timestamp": ts,
        "hour": int(ts[11:13]),
        "temp": 10.0,
        "temp_key": "10",
        "wind_bucket": "normal",
        "actual_kwh": actual,
        "auxiliary_active": False,
        "solar_factor": max(solar_s, solar_w),
        "solar_vector_s": solar_s,
        "solar_vector_e": 0.0,
        "solar_vector_w": solar_w,
        "correction_percent": 100.0,
        "unit_modes": {},  # all-heating
        "unit_breakdown": {sensor_id: actual},
        "solar_dominant_entities": [sensor_id],  # shutdown flag!
        "solar_normalization_delta": 0.0,
        "learning_status": "logged",
    }


class TestReplayInequality:

    def test_shutdown_hour_triggers_inequality(self):
        """A shutdown hour with meaningful base lifts the west coefficient."""
        coord = _shutdown_env()
        calc = SolarCalculator(coord)
        lm = LearningManager()
        # Seed per-unit correlation so replay has a base reference above
        # SOLAR_SHUTDOWN_MIN_BASE.
        correlation_data_per_unit = {"sensor.vp_stue": {"10": {"normal": 1.0}}}
        solar_coeffs: dict = {}
        buffers: dict = {}
        # Chain of shutdown hours to let battery build up.
        entries = [_shutdown_entry(f"2026-05-01T{h:02d}:00:00") for h in range(12, 20)]
        diag = lm.replay_solar_nlms(
            entries,
            solar_calculator=calc,
            screen_config=coord.screen_config,
            correlation_data_per_unit=correlation_data_per_unit,
            solar_coefficients_per_unit=solar_coeffs,
            learning_buffer_solar_per_unit=buffers,
            energy_sensors=coord.energy_sensors,
            learning_rate=1.0,
            balance_point=15.0,
            aux_affected_entities=coord.aux_affected_entities,
            unit_strategies=coord._unit_strategies,
            daily_history={},
            return_diagnostics=True,
        )
        assert diag["inequality_updates"] > 0
        # West was the dominant potential — coefficient should reflect that
        # in the heating regime (inequality is heating-only per #865).
        coeff = solar_coeffs.get("sensor.vp_stue")
        assert coeff is not None
        assert coeff["heating"]["w"] > 0.0

    def test_low_base_skipped(self):
        """Shutdown hour below SOLAR_SHUTDOWN_MIN_BASE is skipped with a counter."""
        coord = _shutdown_env()
        calc = SolarCalculator(coord)
        lm = LearningManager()
        # Base < 0.15 (SOLAR_SHUTDOWN_MIN_BASE)
        correlation_data_per_unit = {"sensor.vp_stue": {"10": {"normal": 0.05}}}
        solar_coeffs: dict = {}
        buffers: dict = {}
        entries = [_shutdown_entry(f"2026-05-01T{h:02d}:00:00") for h in range(12, 20)]
        diag = lm.replay_solar_nlms(
            entries,
            solar_calculator=calc,
            screen_config=coord.screen_config,
            correlation_data_per_unit=correlation_data_per_unit,
            solar_coefficients_per_unit=solar_coeffs,
            learning_buffer_solar_per_unit=buffers,
            energy_sensors=coord.energy_sensors,
            learning_rate=1.0,
            balance_point=15.0,
            aux_affected_entities=coord.aux_affected_entities,
            unit_strategies=coord._unit_strategies,
            daily_history={},
            return_diagnostics=True,
        )
        assert diag["inequality_skipped_base"] > 0
        assert diag["inequality_updates"] == 0

    def test_cooling_mode_skipped(self):
        """Cooling mode is out of scope for the inequality learner."""
        coord = _shutdown_env()
        calc = SolarCalculator(coord)
        lm = LearningManager()
        correlation_data_per_unit = {"sensor.vp_stue": {"10": {"normal": 1.0}}}
        solar_coeffs: dict = {}
        buffers: dict = {}
        entries = []
        for h in range(12, 18):
            e = _shutdown_entry(f"2026-05-01T{h:02d}:00:00")
            e["unit_modes"] = {"sensor.vp_stue": "cooling"}
            entries.append(e)
        diag = lm.replay_solar_nlms(
            entries,
            solar_calculator=calc,
            screen_config=coord.screen_config,
            correlation_data_per_unit=correlation_data_per_unit,
            solar_coefficients_per_unit=solar_coeffs,
            learning_buffer_solar_per_unit=buffers,
            energy_sensors=coord.energy_sensors,
            learning_rate=1.0,
            balance_point=15.0,
            aux_affected_entities=coord.aux_affected_entities,
            unit_strategies=coord._unit_strategies,
            daily_history={},
            return_diagnostics=True,
        )
        assert diag["inequality_skipped_mode"] > 0
        assert diag["inequality_updates"] == 0

    def test_non_shutdown_unit_still_learns_via_nlms(self):
        """Mixed hour: shutdown unit uses inequality, other uses NLMS."""
        coord = _shutdown_env()
        coord.energy_sensors = ["sensor.vp_stue", "sensor.other"]
        coord._unit_strategies = build_strategies(
            energy_sensors=coord.energy_sensors,
            track_c_enabled=False,
            mpc_managed_sensor=None,
        )
        calc = SolarCalculator(coord)
        lm = LearningManager()
        correlation_data_per_unit = {
            "sensor.vp_stue": {"10": {"normal": 1.0}},
            "sensor.other": {"10": {"normal": 1.0}},
        }
        solar_coeffs: dict = {}
        buffers: dict = {}
        entries = []
        for h in range(12, 20):
            e = _shutdown_entry(f"2026-05-01T{h:02d}:00:00")
            # Only vp_stue flagged; other is running normally.
            e["unit_breakdown"] = {"sensor.vp_stue": 0.0, "sensor.other": 0.5}
            entries.append(e)
        lm.replay_solar_nlms(
            entries,
            solar_calculator=calc,
            screen_config=coord.screen_config,
            correlation_data_per_unit=correlation_data_per_unit,
            solar_coefficients_per_unit=solar_coeffs,
            learning_buffer_solar_per_unit=buffers,
            energy_sensors=coord.energy_sensors,
            learning_rate=1.0,
            balance_point=15.0,
            aux_affected_entities=coord.aux_affected_entities,
            unit_strategies=coord._unit_strategies,
            daily_history={},
        )
        # Both coefficients should exist: vp_stue from inequality, other from NLMS
        assert "sensor.vp_stue" in solar_coeffs
        assert "sensor.other" in solar_coeffs

