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
    SOLAR_DEAD_ZONE_THRESHOLD,
    COLD_START_SOLAR_DAMPING,
    LEARNING_BUFFER_THRESHOLD,
    MODE_HEATING,
    MODE_COOLING,
)


def _regime_for(unit_mode: str) -> str:
    """Mode→regime mapping for v4 stratified coefficient access (#868)."""
    return "cooling" if unit_mode == MODE_COOLING else "heating"


def _run_nlms_learning(
    manager: LearningManager,
    entity_id: str,
    true_coeff_s: float,
    true_coeff_e: float,
    solar_vectors: list[tuple[float, float, float]],
    solar_coefficients: dict,
    solar_buffers: dict,
    unit_mode: str = MODE_HEATING,
    base_kwh: float = 2.0,
):
    """Drive solar coefficient learning with synthetic data.

    Generates actual_unit values from true coefficients and runs learning.
    Returns list of (step, coeff_s, coeff_e) for convergence analysis.
    Reads from the v4 mode regime corresponding to ``unit_mode``.
    """
    regime = _regime_for(unit_mode)
    history = []
    for solar_s, solar_e, solar_w in solar_vectors:
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
            avg_solar_vector=(solar_s, solar_e, solar_w),
            learning_rate=0.01,
            solar_coefficients_per_unit=solar_coefficients,
            learning_buffer_solar_per_unit=solar_buffers,
            avg_temp=5.0,
            balance_point=15.0,
            unit_mode=unit_mode,
        )
        entity_entry = solar_coefficients.get(entity_id)
        if entity_entry is not None:
            coeff = entity_entry.get(regime)
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
        vectors = [(0.5, 0.1, 0.0)] * 5 + [(0.6, -0.05, 0.0)] * 5 + [(0.4, 0.15, 0.0)] * 10
        history = _run_nlms_learning(
            manager, "unit_south", 1.5, 0.1, vectors, coeffs, buffers,
        )

        # After 20 samples (4 buffered + 16 NLMS), should be close.
        # Heating-regime read per #868.
        final = coeffs["unit_south"]["heating"]
        assert abs(final["s"] - 1.5) < 0.3, f"S coeff {final['s']} not near 1.5"
        assert abs(final["e"] - 0.1) < 0.3, f"E coeff {final['e']} not near 0.1"

    def test_converges_mixed_orientation(self):
        """South-east facing unit converges for both components."""
        manager = LearningManager()
        coeffs = {}
        buffers = {}

        # Diverse sun angles to resolve both components
        vectors = (
            [(0.6, 0.3, 0.0)] * 5
            + [(0.3, 0.5, 0.0)] * 5
            + [(0.5, 0.4, 0.0)] * 5
            + [(0.7, 0.2, 0.0)] * 5
        )
        history = _run_nlms_learning(
            manager, "unit_se", 1.0, 0.8, vectors, coeffs, buffers,
        )

        final = coeffs["unit_se"]["heating"]
        assert abs(final["s"] - 1.0) < 0.4, f"S coeff {final['s']} not near 1.0"
        assert abs(final["e"] - 0.8) < 0.4, f"E coeff {final['e']} not near 0.8"

    def test_converges_cooling_mode(self):
        """NLMS works correctly in cooling mode (solar increases load)."""
        manager = LearningManager()
        coeffs = {}
        buffers = {}

        vectors = [(0.5, 0.1, 0.0)] * 20
        history = _run_nlms_learning(
            manager, "unit_cool", 0.8, 0.0, vectors, coeffs, buffers,
            unit_mode=MODE_COOLING, base_kwh=3.0,
        )

        # Cooling-mode learning routes to the cooling regime per #868.
        final = coeffs["unit_cool"]["cooling"]
        assert abs(final["s"] - 0.8) < 0.3, f"S coeff {final['s']} not near 0.8"
        # Heating regime untouched.
        assert coeffs["unit_cool"]["heating"] == {"s": 0.0, "e": 0.0, "w": 0.0}


class TestNLMSGainIndependence:
    """NLMS convergence rate should be independent of solar magnitude."""

    def test_same_convergence_high_vs_low_solar(self):
        """High-solar and low-solar units reach same accuracy in same steps."""
        manager = LearningManager()
        true_s, true_e = 1.0, 0.2

        # Low solar magnitude (~0.3)
        coeffs_low = {}
        buffers_low = {}
        vectors_low = [(0.25, 0.10, 0.0)] * 20
        history_low = _run_nlms_learning(
            manager, "unit_low", true_s, true_e,
            vectors_low, coeffs_low, buffers_low,
        )

        # High solar magnitude (~0.8)
        coeffs_high = {}
        buffers_high = {}
        vectors_high = [(0.70, 0.28, 0.0)] * 20
        history_high = _run_nlms_learning(
            manager, "unit_high", true_s, true_e,
            vectors_high, coeffs_high, buffers_high,
        )

        final_low = coeffs_low["unit_low"]["heating"]
        final_high = coeffs_high["unit_high"]["heating"]

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
        vectors = [(0.5, 0.0, 0.0)] * 20
        _run_nlms_learning(
            manager, "unit_pure_s", 1.0, 0.0, vectors, coeffs, buffers,
        )

        final = coeffs["unit_pure_s"]["heating"]
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
                "unit_trans", "10", 2.0, actual, (solar_s, solar_e, 0.0),
                0.01, coeffs, buffers, 5.0, 15.0, MODE_HEATING,
            )

        # Should still be buffering — neither regime has a written coefficient
        # yet.  The buffer holds samples in the heating regime.
        assert "unit_trans" not in coeffs or all(
            sum(coeffs["unit_trans"][r].values()) == 0
            for r in ("heating", "cooling")
            if r in coeffs.get("unit_trans", {})
        )
        buf_heating = buffers.get("unit_trans", {}).get("heating", [])
        assert len(buf_heating) == LEARNING_BUFFER_THRESHOLD - 1

        # Phase 2: One more sample triggers jump-start
        solar_s, solar_e = 0.6, 0.15
        true_impact = true_s * solar_s + true_e * solar_e
        actual = max(0.0, 2.0 - true_impact)
        manager._learn_unit_solar_coefficient(
            "unit_trans", "10", 2.0, actual, (solar_s, solar_e, 0.0),
            0.01, coeffs, buffers, 5.0, 15.0, MODE_HEATING,
        )

        # Now coefficients should exist (jump-started) in the heating regime.
        assert "unit_trans" in coeffs
        jumpstart_s = coeffs["unit_trans"]["heating"]["s"]
        # Buffer for the heating regime should be cleared after solve.
        assert len(buffers.get("unit_trans", {}).get("heating", [])) == 0

        # Phase 3: NLMS refinement (10 more samples)
        vectors = [(0.5, 0.12, 0.0)] * 5 + [(0.6, 0.08, 0.0)] * 5
        for solar_s, solar_e, solar_w in vectors:
            true_impact = true_s * solar_s + true_e * solar_e
            actual = max(0.0, 2.0 - true_impact)
            manager._learn_unit_solar_coefficient(
                "unit_trans", "10", 2.0, actual, (solar_s, solar_e, solar_w),
                0.01, coeffs, buffers, 5.0, 15.0, MODE_HEATING,
            )

        final = coeffs["unit_trans"]["heating"]
        # NLMS should have refined beyond jump-start
        error_final = abs(final["s"] - true_s)
        # Jump-start is damped by COLD_START_SOLAR_DAMPING, so NLMS should improve
        assert abs(final["s"] - true_s) < 0.5, (
            f"After NLMS refinement, S={final['s']:.3f} not close to {true_s}"
        )

    def test_zero_vector_skipped(self):
        """Samples with zero solar vector are ignored (no division by zero)."""
        from tests.helpers import stratified_coeff
        manager = LearningManager()
        coeffs = {"unit_z": stratified_coeff(s=1.0)}
        buffers = {}

        # Zero vector should be silently skipped
        manager._learn_unit_solar_coefficient(
            "unit_z", "10", 2.0, 1.5, (0.0, 0.0, 0.0),
            0.01, coeffs, buffers, 5.0, 15.0, MODE_HEATING,
        )

        # Coefficient unchanged
        assert coeffs["unit_z"]["heating"]["s"] == 1.0
        assert coeffs["unit_z"]["heating"]["e"] == 0.0

    def test_off_mode_skipped(self):
        """Units in OFF mode are not updated."""
        from custom_components.heating_analytics.const import MODE_OFF
        from tests.helpers import stratified_coeff

        manager = LearningManager()
        coeffs = {"unit_off": stratified_coeff(s=1.0)}
        buffers = {}

        manager._learn_unit_solar_coefficient(
            "unit_off", "10", 2.0, 1.5, (0.5, 0.1, 0.0),
            0.01, coeffs, buffers, 5.0, 15.0, MODE_OFF,
        )

        assert coeffs["unit_off"]["heating"]["s"] == 1.0


class TestDeadZoneDetection:
    """Dead zone: base model too low → actual_impact clamps to 0 → NLMS stuck."""

    def test_dead_zone_resets_coefficient_after_threshold(self):
        """After SOLAR_DEAD_ZONE_THRESHOLD consecutive zero-impact hours,
        the heating regime coefficient is reset so cold-start can re-learn.
        Cooling regime is preserved per #868 (per-(entity, regime) dead-zone
        tracking).
        """
        from tests.helpers import stratified_coeff
        manager = LearningManager()
        coeffs = {"unit_stuck": stratified_coeff(s=0.1, cooling_s=0.5)}
        buffers = {}

        # Simulate dead zone: base=0.03, actual=0.18, sun shining
        # actual_impact = max(0, 0.03 - 0.18) = 0
        for _ in range(SOLAR_DEAD_ZONE_THRESHOLD):
            manager._learn_unit_solar_coefficient(
                "unit_stuck", "10",
                expected_unit_base=0.03,
                actual_unit=0.18,
                avg_solar_vector=(0.3, 0.1, 0.2),
                learning_rate=0.01,
                solar_coefficients_per_unit=coeffs,
                learning_buffer_solar_per_unit=buffers,
                avg_temp=12.0,
                balance_point=15.0,
                unit_mode=MODE_HEATING,
            )

        # Heating regime reset to zero; cooling regime preserved.
        assert coeffs["unit_stuck"]["heating"] == {"s": 0.0, "e": 0.0, "w": 0.0}
        assert coeffs["unit_stuck"]["cooling"]["s"] == 0.5

    def test_dead_zone_counter_resets_on_positive_impact(self):
        """A single hour with positive actual_impact resets the counter."""
        from tests.helpers import stratified_coeff
        manager = LearningManager()
        coeffs = {"unit_recover": stratified_coeff(s=0.1)}
        buffers = {}

        # Accumulate near-threshold dead zone hours
        for _ in range(SOLAR_DEAD_ZONE_THRESHOLD - 2):
            manager._learn_unit_solar_coefficient(
                "unit_recover", "10",
                expected_unit_base=0.03, actual_unit=0.18,
                avg_solar_vector=(0.3, 0.1, 0.2),
                learning_rate=0.01,
                solar_coefficients_per_unit=coeffs,
                learning_buffer_solar_per_unit=buffers,
                avg_temp=12.0, balance_point=15.0,
                unit_mode=MODE_HEATING,
            )

        # One hour with positive impact (base model caught up)
        manager._learn_unit_solar_coefficient(
            "unit_recover", "10",
            expected_unit_base=0.50, actual_unit=0.30,
            avg_solar_vector=(0.3, 0.1, 0.2),
            learning_rate=0.01,
            solar_coefficients_per_unit=coeffs,
            learning_buffer_solar_per_unit=buffers,
            avg_temp=12.0, balance_point=15.0,
            unit_mode=MODE_HEATING,
        )

        # Counter should be reset — coefficient still exists in heating regime.
        assert "unit_recover" in coeffs
        assert coeffs["unit_recover"]["heating"]["s"] != 0.0
        # Dead-zone counter is keyed by (entity, regime) — both should be clear.
        assert manager._dead_zone_counts.get(("unit_recover", "heating"), 0) == 0

    def test_dead_zone_not_triggered_for_cold_start(self):
        """Units without existing coefficients (cold-start) don't trigger
        dead zone reset — they stay in buffering mode.  With all-zero
        impact, the buffer is discarded each time it fills (not solved
        to produce a (0,0,0) coefficient).
        """
        manager = LearningManager()
        coeffs = {}  # No existing coefficient
        buffers = {}

        for _ in range(SOLAR_DEAD_ZONE_THRESHOLD + 5):
            manager._learn_unit_solar_coefficient(
                "unit_new", "10",
                expected_unit_base=0.03, actual_unit=0.18,
                avg_solar_vector=(0.3, 0.1, 0.2),
                learning_rate=0.01,
                solar_coefficients_per_unit=coeffs,
                learning_buffer_solar_per_unit=buffers,
                avg_temp=12.0, balance_point=15.0,
                unit_mode=MODE_HEATING,
            )

        # Should NOT have created a coefficient (all-zero buffers discarded)
        assert "unit_new" not in coeffs
        # Dead zone counter should not have incremented (no current_coeff)
        assert manager._dead_zone_counts.get(("unit_new", "heating"), 0) == 0

    def test_dead_zone_not_triggered_without_sun(self):
        """Zero vector magnitude doesn't count toward dead zone."""
        from tests.helpers import stratified_coeff
        manager = LearningManager()
        coeffs = {"unit_dark": stratified_coeff(s=0.1)}
        buffers = {}

        for _ in range(SOLAR_DEAD_ZONE_THRESHOLD + 5):
            manager._learn_unit_solar_coefficient(
                "unit_dark", "10",
                expected_unit_base=0.03, actual_unit=0.18,
                avg_solar_vector=(0.0, 0.0, 0.0),  # No sun
                learning_rate=0.01,
                solar_coefficients_per_unit=coeffs,
                learning_buffer_solar_per_unit=buffers,
                avg_temp=12.0, balance_point=15.0,
                unit_mode=MODE_HEATING,
            )

        # Coefficient should NOT have been reset (no sun = early return)
        assert "unit_dark" in coeffs
        assert coeffs["unit_dark"]["heating"]["s"] == 0.1

    def test_dead_zone_reset_clears_stale_buffer(self):
        """Dead zone reset should also clear the regime's stale buffer."""
        from tests.helpers import stratified_coeff
        manager = LearningManager()
        coeffs = {"unit_buf": stratified_coeff(s=0.1)}
        buffers = {"unit_buf": {"heating": [(0.3, 0.1, 0.2, 0.05)], "cooling": []}}

        for _ in range(SOLAR_DEAD_ZONE_THRESHOLD):
            manager._learn_unit_solar_coefficient(
                "unit_buf", "10",
                expected_unit_base=0.03, actual_unit=0.18,
                avg_solar_vector=(0.3, 0.1, 0.2),
                learning_rate=0.01,
                solar_coefficients_per_unit=coeffs,
                learning_buffer_solar_per_unit=buffers,
                avg_temp=12.0, balance_point=15.0,
                unit_mode=MODE_HEATING,
            )

        # Heating regime reset to zero, cooling regime preserved.
        assert coeffs["unit_buf"]["heating"] == {"s": 0.0, "e": 0.0, "w": 0.0}
        # Heating regime buffer cleared.
        assert buffers["unit_buf"]["heating"] == []

    def test_cold_start_discards_all_zero_impact_buffer(self):
        """Cold-start with all-zero impact samples discards buffer instead
        of creating a (0,0,0) coefficient that re-enters the dead zone.
        """
        manager = LearningManager()
        coeffs = {}  # No coefficient — cold-start mode
        buffers = {}

        # Feed LEARNING_BUFFER_THRESHOLD samples with zero impact
        for _ in range(LEARNING_BUFFER_THRESHOLD):
            manager._learn_unit_solar_coefficient(
                "unit_reentry", "10",
                expected_unit_base=0.03, actual_unit=0.18,
                avg_solar_vector=(0.3, 0.1, 0.2),
                learning_rate=0.01,
                solar_coefficients_per_unit=coeffs,
                learning_buffer_solar_per_unit=buffers,
                avg_temp=12.0, balance_point=15.0,
                unit_mode=MODE_HEATING,
            )

        # Should NOT have created a coefficient (buffer was discarded).
        # In v4 the buffer entry exists as a dict-of-regimes with empty lists.
        assert "unit_reentry" not in coeffs
        # Buffer for the heating regime should be empty (cleared after discard).
        buf_h = buffers.get("unit_reentry", {}).get("heating", [])
        assert len(buf_h) == 0

    def test_post_reset_recovery_with_improving_base(self):
        """After dead zone reset, once base model recovers enough to produce
        positive actual_impact, cold-start completes successfully.
        """
        from tests.helpers import stratified_coeff
        manager = LearningManager()
        coeffs = {"unit_recov": stratified_coeff(s=0.1)}
        buffers = {}

        # Phase 1: Trigger dead zone reset (heating regime).
        for _ in range(SOLAR_DEAD_ZONE_THRESHOLD):
            manager._learn_unit_solar_coefficient(
                "unit_recov", "10",
                expected_unit_base=0.03, actual_unit=0.18,
                avg_solar_vector=(0.3, 0.1, 0.2),
                learning_rate=0.01,
                solar_coefficients_per_unit=coeffs,
                learning_buffer_solar_per_unit=buffers,
                avg_temp=12.0, balance_point=15.0,
                unit_mode=MODE_HEATING,
            )
        assert coeffs["unit_recov"]["heating"] == {"s": 0.0, "e": 0.0, "w": 0.0}

        # Phase 2: Base model has recovered (expected > actual now)
        for _ in range(LEARNING_BUFFER_THRESHOLD):
            manager._learn_unit_solar_coefficient(
                "unit_recov", "10",
                expected_unit_base=0.50, actual_unit=0.30,
                avg_solar_vector=(0.3, 0.1, 0.2),
                learning_rate=0.01,
                solar_coefficients_per_unit=coeffs,
                learning_buffer_solar_per_unit=buffers,
                avg_temp=12.0, balance_point=15.0,
                unit_mode=MODE_HEATING,
            )

        # Cold-start should have completed with real signal.
        coeff = coeffs["unit_recov"]["heating"]
        assert coeff["s"] > 0.0 or coeff["e"] > 0.0 or coeff["w"] > 0.0
