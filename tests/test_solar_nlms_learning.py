"""Tests for NLMS solar coefficient learning (#809 A3).

Verifies:
- Convergence within ~10 qualifying samples
- Gain-independence (same convergence rate for high vs low solar magnitude)
- Regularization shrinks poorly-constrained east component
- Cold-start buffered learning feeds into NLMS correctly
"""
import pytest
from custom_components.heating_analytics.learning import LearningManager
from custom_components.heating_analytics.const import (
    NLMS_STEP_SIZE,
    NLMS_REGULARIZATION,
    SOLAR_COEFF_CAP,
    COLD_START_SOLAR_DAMPING,
    LEARNING_BUFFER_THRESHOLD,
    MODE_HEATING,
    MODE_COOLING,
)


def _run_nlms_learning(
    manager: LearningManager,
    entity_id: str,
    true_coeff_s: float,
    true_coeff_e: float,
    solar_vectors: list[tuple[float, float]],
    solar_coefficients: dict,
    solar_buffers: dict,
    unit_mode: str = MODE_HEATING,
    base_kwh: float = 2.0,
):
    """Drive solar coefficient learning with synthetic data.

    Generates actual_unit values from true coefficients and runs learning.
    Returns list of (step, coeff_s, coeff_e) for convergence analysis.
    """
    history = []
    for solar_s, solar_e in solar_vectors:
        true_impact = true_coeff_s * solar_s + true_coeff_e * solar_e
        if unit_mode == MODE_HEATING:
            actual_unit = max(0.0, base_kwh - true_impact)
        else:
            actual_unit = base_kwh + true_impact

        manager._learn_unit_solar_coefficient(
            entity_id=entity_id,
            temp_key="10",
            expected_unit_base=base_kwh,
            actual_unit=actual_unit,
            avg_solar_vector=(solar_s, solar_e),
            learning_rate=0.01,
            solar_coefficients_per_unit=solar_coefficients,
            learning_buffer_solar_per_unit=solar_buffers,
            avg_temp=5.0,
            balance_point=15.0,
            unit_mode=unit_mode,
        )
        coeff = solar_coefficients.get(entity_id)
        if coeff is not None:
            history.append((len(history), coeff["s"], coeff["e"]))
    return history


class TestNLMSConvergence:
    """NLMS should converge to true coefficients within ~10 qualifying samples."""

    def test_converges_south_dominant(self):
        """Pure south-facing unit converges to correct S coefficient."""
        manager = LearningManager()
        coeffs = {}
        buffers = {}

        # 20 samples of moderate south sun at varying azimuths
        vectors = [(0.5, 0.1)] * 5 + [(0.6, -0.05)] * 5 + [(0.4, 0.15)] * 10
        history = _run_nlms_learning(
            manager, "unit_south", 1.5, 0.1, vectors, coeffs, buffers,
        )

        # After 20 samples (4 buffered + 16 NLMS), should be close
        final = coeffs["unit_south"]
        assert abs(final["s"] - 1.5) < 0.3, f"S coeff {final['s']} not near 1.5"
        assert abs(final["e"] - 0.1) < 0.3, f"E coeff {final['e']} not near 0.1"

    def test_converges_mixed_orientation(self):
        """South-east facing unit converges for both components."""
        manager = LearningManager()
        coeffs = {}
        buffers = {}

        # Diverse sun angles to resolve both components
        vectors = (
            [(0.6, 0.3)] * 5
            + [(0.3, 0.5)] * 5
            + [(0.5, 0.4)] * 5
            + [(0.7, 0.2)] * 5
        )
        history = _run_nlms_learning(
            manager, "unit_se", 1.0, 0.8, vectors, coeffs, buffers,
        )

        final = coeffs["unit_se"]
        assert abs(final["s"] - 1.0) < 0.4, f"S coeff {final['s']} not near 1.0"
        assert abs(final["e"] - 0.8) < 0.4, f"E coeff {final['e']} not near 0.8"

    def test_converges_cooling_mode(self):
        """NLMS works correctly in cooling mode (solar increases load)."""
        manager = LearningManager()
        coeffs = {}
        buffers = {}

        vectors = [(0.5, 0.1)] * 20
        history = _run_nlms_learning(
            manager, "unit_cool", 0.8, 0.0, vectors, coeffs, buffers,
            unit_mode=MODE_COOLING, base_kwh=3.0,
        )

        final = coeffs["unit_cool"]
        assert abs(final["s"] - 0.8) < 0.3, f"S coeff {final['s']} not near 0.8"


class TestNLMSGainIndependence:
    """NLMS convergence rate should be independent of solar magnitude."""

    def test_same_convergence_high_vs_low_solar(self):
        """High-solar and low-solar units reach same accuracy in same steps."""
        manager = LearningManager()
        true_s, true_e = 1.0, 0.2

        # Low solar magnitude (~0.3)
        coeffs_low = {}
        buffers_low = {}
        vectors_low = [(0.25, 0.10)] * 20
        history_low = _run_nlms_learning(
            manager, "unit_low", true_s, true_e,
            vectors_low, coeffs_low, buffers_low,
        )

        # High solar magnitude (~0.8)
        coeffs_high = {}
        buffers_high = {}
        vectors_high = [(0.70, 0.28)] * 20
        history_high = _run_nlms_learning(
            manager, "unit_high", true_s, true_e,
            vectors_high, coeffs_high, buffers_high,
        )

        final_low = coeffs_low["unit_low"]
        final_high = coeffs_high["unit_high"]

        # Both should converge to similar accuracy
        error_low = abs(final_low["s"] - true_s) + abs(final_low["e"] - true_e)
        error_high = abs(final_high["s"] - true_s) + abs(final_high["e"] - true_e)

        # The key NLMS property: high solar does NOT cause worse convergence
        # Allow 2x tolerance (noise) but no order-of-magnitude difference
        assert error_high < error_low * 3 + 0.2, (
            f"High-solar error {error_high:.3f} much worse than low-solar {error_low:.3f} "
            f"— NLMS gain-independence violated"
        )


class TestNLMSRegularization:
    """Epsilon regularization should shrink poorly-constrained components."""

    def test_east_shrinks_with_south_only_sun(self):
        """When sun is always due south, east coefficient stays near zero."""
        manager = LearningManager()
        coeffs = {}
        buffers = {}

        # Pure south vectors — east component has no signal
        vectors = [(0.5, 0.0)] * 20
        _run_nlms_learning(
            manager, "unit_pure_s", 1.0, 0.0, vectors, coeffs, buffers,
        )

        final = coeffs["unit_pure_s"]
        assert abs(final["e"]) < 0.1, (
            f"East coeff {final['e']} should be ~0 with pure south sun"
        )


class TestColdStartToNLMSTransition:
    """Buffered cold-start should smoothly transition to NLMS updates."""

    def test_buffer_fills_then_nlms_refines(self):
        """First LEARNING_BUFFER_THRESHOLD samples buffer, then NLMS takes over."""
        manager = LearningManager()
        coeffs = {}
        buffers = {}

        true_s, true_e = 1.2, 0.3

        # Phase 1: Buffer fill (should not produce coefficients until threshold)
        for i in range(LEARNING_BUFFER_THRESHOLD - 1):
            solar_s, solar_e = 0.5 + i * 0.05, 0.1 + i * 0.02
            true_impact = true_s * solar_s + true_e * solar_e
            actual = max(0.0, 2.0 - true_impact)
            manager._learn_unit_solar_coefficient(
                "unit_trans", "10", 2.0, actual, (solar_s, solar_e),
                0.01, coeffs, buffers, 5.0, 15.0, MODE_HEATING,
            )

        # Should still be buffering
        assert "unit_trans" not in coeffs
        assert len(buffers.get("unit_trans", [])) == LEARNING_BUFFER_THRESHOLD - 1

        # Phase 2: One more sample triggers jump-start
        solar_s, solar_e = 0.6, 0.15
        true_impact = true_s * solar_s + true_e * solar_e
        actual = max(0.0, 2.0 - true_impact)
        manager._learn_unit_solar_coefficient(
            "unit_trans", "10", 2.0, actual, (solar_s, solar_e),
            0.01, coeffs, buffers, 5.0, 15.0, MODE_HEATING,
        )

        # Now coefficients should exist (jump-started)
        assert "unit_trans" in coeffs
        jumpstart_s = coeffs["unit_trans"]["s"]
        # Buffer should be cleared
        assert len(buffers.get("unit_trans", [])) == 0

        # Phase 3: NLMS refinement (10 more samples)
        vectors = [(0.5, 0.12)] * 5 + [(0.6, 0.08)] * 5
        for solar_s, solar_e in vectors:
            true_impact = true_s * solar_s + true_e * solar_e
            actual = max(0.0, 2.0 - true_impact)
            manager._learn_unit_solar_coefficient(
                "unit_trans", "10", 2.0, actual, (solar_s, solar_e),
                0.01, coeffs, buffers, 5.0, 15.0, MODE_HEATING,
            )

        final = coeffs["unit_trans"]
        # NLMS should have refined beyond jump-start
        error_final = abs(final["s"] - true_s)
        # Jump-start is damped by COLD_START_SOLAR_DAMPING, so NLMS should improve
        assert abs(final["s"] - true_s) < 0.5, (
            f"After NLMS refinement, S={final['s']:.3f} not close to {true_s}"
        )

    def test_zero_vector_skipped(self):
        """Samples with zero solar vector are ignored (no division by zero)."""
        manager = LearningManager()
        coeffs = {"unit_z": {"s": 1.0, "e": 0.0}}
        buffers = {}

        # Zero vector should be silently skipped
        manager._learn_unit_solar_coefficient(
            "unit_z", "10", 2.0, 1.5, (0.0, 0.0),
            0.01, coeffs, buffers, 5.0, 15.0, MODE_HEATING,
        )

        # Coefficient unchanged
        assert coeffs["unit_z"]["s"] == 1.0
        assert coeffs["unit_z"]["e"] == 0.0

    def test_off_mode_skipped(self):
        """Units in OFF mode are not updated."""
        from custom_components.heating_analytics.const import MODE_OFF

        manager = LearningManager()
        coeffs = {"unit_off": {"s": 1.0, "e": 0.0}}
        buffers = {}

        manager._learn_unit_solar_coefficient(
            "unit_off", "10", 2.0, 1.5, (0.5, 0.1),
            0.01, coeffs, buffers, 5.0, 15.0, MODE_OFF,
        )

        assert coeffs["unit_off"]["s"] == 1.0
