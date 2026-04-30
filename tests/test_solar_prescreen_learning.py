"""Tests for pre-screen solar coefficient learning (#809 A2).

Verifies:
- Coefficients are invariant to screen position changes
- Transmittance reconstruction correctly recovers potential vector
- DRY: SolarCalculator._screen_transmittance is the single source of truth
- Screen transmittance floor prevents vector collapse
"""
import pytest
from custom_components.heating_analytics.learning import LearningManager
from custom_components.heating_analytics.solar import SolarCalculator
from custom_components.heating_analytics.const import (
    DEFAULT_SOLAR_MIN_TRANSMITTANCE,
    SOLAR_COEFF_CAP,
    MODE_HEATING,
)
from tests.helpers import CoordinatorModelMixin


class TestScreenTransmittanceDRY:
    """The _screen_transmittance static method is the canonical implementation."""

    def test_fully_open(self):
        """100% correction = fully open = transmittance 1.0."""
        assert SolarCalculator._screen_transmittance(100.0) == 1.0

    def test_fully_closed(self):
        """0% correction = fully closed = transmittance equals floor."""
        result = SolarCalculator._screen_transmittance(0.0)
        assert result == pytest.approx(DEFAULT_SOLAR_MIN_TRANSMITTANCE)

    def test_midpoint(self):
        """50% correction gives expected linear interpolation."""
        mn = DEFAULT_SOLAR_MIN_TRANSMITTANCE
        expected = mn + (1.0 - mn) * 0.5
        assert SolarCalculator._screen_transmittance(50.0) == pytest.approx(expected)

    def test_floor_prevents_zero(self):
        """Even at 0%, transmittance never reaches zero."""
        assert SolarCalculator._screen_transmittance(0.0) > 0.0

    def test_monotonic(self):
        """Transmittance increases monotonically with correction_percent."""
        values = [SolarCalculator._screen_transmittance(p) for p in range(0, 101, 10)]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1]


class TestPreScreenLearningRobustness:
    """Pre-screen learning should work reliably even with screens closed.

    The key benefit of #809 A2 is that learning uses the potential (pre-screen)
    vector, which has good signal-to-noise even at low screen_percent. The
    coefficient absorbs the transmittance factor (coeff ≈ phys × trans), which
    is correct: the prediction path applies screen_transmittance separately.
    """

    def _learn_with_screen(
        self,
        correction_percent: float,
        true_coeff_s: float,
        true_coeff_e: float,
        base_kwh: float = 2.0,
    ) -> dict[str, float]:
        """Run learning with a specific screen position and return final coefficient."""
        manager = LearningManager()
        coeffs = {}
        buffers = {}

        transmittance = SolarCalculator._screen_transmittance(correction_percent)

        # Simulate 20 hours of diverse sun
        potential_vectors = [
            (0.6, 0.2, 0.0), (0.5, 0.3, 0.0), (0.7, 0.1, 0.0), (0.4, 0.25, 0.0),
            (0.55, 0.15, 0.0), (0.65, 0.08, 0.0), (0.45, 0.22, 0.0), (0.5, 0.18, 0.0),
            (0.6, 0.12, 0.0), (0.58, 0.2, 0.0), (0.5, 0.1, 0.0), (0.6, 0.15, 0.0),
            (0.55, 0.25, 0.0), (0.7, 0.05, 0.0), (0.45, 0.3, 0.0), (0.5, 0.2, 0.0),
            (0.6, 0.1, 0.0), (0.55, 0.15, 0.0), (0.65, 0.2, 0.0), (0.5, 0.25, 0.0),
        ]

        for pot_s, pot_e, pot_w in potential_vectors:
            # Physical heat gain = phys_coeff × potential × transmittance
            true_impact = (true_coeff_s * pot_s + true_coeff_e * pot_e) * transmittance
            actual_unit = max(0.0, base_kwh - true_impact)

            # Learning receives potential vector (after reconstruction in
            # _process_per_unit_learning), so we pass potential directly.
            manager._learn_unit_solar_coefficient(
                entity_id="unit_screen",
                temp_key="10",
                expected_unit_base=base_kwh,
                actual_unit=actual_unit,
                avg_solar_vector=(pot_s, pot_e, pot_w),
                learning_rate=0.01,
                solar_coefficients_per_unit=coeffs,
                learning_buffer_solar_per_unit=buffers,
                avg_temp=5.0,
                balance_point=15.0,
                unit_mode=MODE_HEATING,
            )

        # Mode-stratified per #868 — return the heating regime view.
        entity = coeffs.get("unit_screen")
        if isinstance(entity, dict) and "heating" in entity:
            return entity["heating"]
        return {"s": 0.0, "e": 0.0, "w": 0.0}

    def test_learning_converges_with_screens_closed(self):
        """Learning still converges when screens are nearly closed (10%).

        Before #809 A2, the effective vector would be tiny at low correction,
        causing learning to stall (magnitude guard). With potential vectors,
        learning proceeds normally.
        """
        true_s = 1.2
        coeff = self._learn_with_screen(10.0, true_s, 0.3)
        trans = SolarCalculator._screen_transmittance(10.0)

        # Coefficient converges to phys_coeff × transmittance
        expected_s = true_s * trans
        assert abs(coeff["s"] - expected_s) < 0.3, (
            f"With screens at 10%, S={coeff['s']:.3f} should be near {expected_s:.3f}"
        )

    def test_prediction_exact_at_full_open(self):
        """At 100% screen (trans=1.0), prediction matches true impact exactly.

        This is the ideal case: no transmittance factor in the coefficient,
        so coeff × potential × 1.0 = phys_coeff × potential.
        """
        true_s, true_e = 1.2, 0.3
        test_potential = (0.5, 0.15, 0.0)

        coeff = self._learn_with_screen(100.0, true_s, true_e)

        class MockCoord(CoordinatorModelMixin):
            solar_azimuth = 180
            balance_point = 15.0
            _solar_coefficients_per_unit = {}

        calc = SolarCalculator(MockCoord())
        predicted = calc.calculate_unit_solar_impact(test_potential, coeff)
        true_impact = true_s * test_potential[0] + true_e * test_potential[1]

        assert abs(predicted - true_impact) < 0.1, (
            f"At 100% screen: predicted={predicted:.3f}, true={true_impact:.3f}"
        )

    def test_varying_screen_produces_stable_coefficient(self):
        """When screen position varies during training, coefficient still converges.

        This is the realistic scenario: screens open in morning, close at noon.
        The coefficient should converge to an average-weighted value.
        """
        manager = LearningManager()
        coeffs = {}
        buffers = {}
        true_s, true_e = 1.0, 0.2

        # Mixed screen positions during training
        screen_schedule = [100, 100, 80, 50, 30, 30, 50, 80, 100, 100] * 2

        for i, pct in enumerate(screen_schedule):
            trans = SolarCalculator._screen_transmittance(pct)
            pot_s = 0.5 + (i % 5) * 0.05
            pot_e = 0.1 + (i % 3) * 0.05

            true_impact = (true_s * pot_s + true_e * pot_e) * trans
            actual = max(0.0, 2.0 - true_impact)

            manager._learn_unit_solar_coefficient(
                "unit_vary", "10", 2.0, actual, (pot_s, pot_e, 0.0),
                0.01, coeffs, buffers, 5.0, 15.0, MODE_HEATING,
            )

        # Should have converged (not diverged or oscillated).
        # Heating-regime read per #868.
        assert "unit_vary" in coeffs
        final = coeffs["unit_vary"]["heating"]
        assert 0.0 < final["s"] < SOLAR_COEFF_CAP, f"S={final['s']} out of range"
        assert abs(final["e"]) < SOLAR_COEFF_CAP, f"E={final['e']} out of range"


class TestTransmittanceReconstruction:
    """Verify that dividing effective vector by transmittance recovers potential."""

    def test_roundtrip_potential_effective_potential(self):
        """potential → effective → reconstruct potential matches original."""
        potential = (0.6, 0.25, 0.0)
        for pct in [0, 25, 50, 75, 100]:
            t = SolarCalculator._screen_transmittance(pct)
            effective = (potential[0] * t, potential[1] * t, potential[2] * t)
            recovered = (effective[0] / t, effective[1] / t, effective[2] / t)
            assert recovered[0] == pytest.approx(potential[0], abs=1e-10)
            assert recovered[1] == pytest.approx(potential[1], abs=1e-10)
            assert recovered[2] == pytest.approx(potential[2], abs=1e-10)

    def test_floor_prevents_division_amplification(self):
        """At 0% correction, floor prevents extreme amplification."""
        t = SolarCalculator._screen_transmittance(0.0)
        # With floor ~0.20, dividing by t amplifies by ~5x at most
        assert 1.0 / t < 6.0, f"Amplification factor {1/t:.1f} too large"


class TestPredictionPathIsTransmittanceFree:
    """The prediction path must NOT apply screen_transmittance separately.

    CLAUDE.md invariant #1: ``coeff × potential`` with no extra transmittance
    factor — the coefficient absorbs ``avg_transmittance`` via the NLMS
    learning target.  A separate transmittance factor at prediction would
    yield ``phys × trans² × potential`` (the trans² bug that motivated #809
    and the current design).
    """

    def test_calculate_unit_solar_impact_is_dot_product_only(self):
        """Impact == coeff . potential_vector, regardless of any external
        screen state — the function signature accepts only (vector, coeff).
        """
        from tests.helpers import CoordinatorModelMixin

        class MockCoord(CoordinatorModelMixin):
            solar_azimuth = 180
            balance_point = 15.0
            _solar_coefficients_per_unit = {}

        calc = SolarCalculator(MockCoord())

        coeff = {"s": 1.5, "e": 0.3, "w": 0.0}
        potential_vector = (0.6, 0.2, 0.0)

        impact = calc.calculate_unit_solar_impact(potential_vector, coeff)
        expected = 1.5 * 0.6 + 0.3 * 0.2
        assert impact == pytest.approx(expected)
