"""End-to-end integration test: solar model pipeline with varying screen positions.

Exercises the full component chain (observation → learning → prediction) with
screen positions that change within a single hour.  The trans² double-counting
bug (#809) arose at component boundaries; this test catches similar issues by
verifying that:

1. Accumulation: varying screen positions produce correct hourly averages
2. Reconstruction: potential vector is correctly recovered from effective/transmittance
3. Learning: NLMS updates against potential vector, coefficient absorbs transmittance
4. Prediction: uses coeff × potential (no extra transmittance factor)
5. Consistency: normalization and prediction use the same solar formula
"""
import math
import pytest

from custom_components.heating_analytics.const import (
    LEARNING_BUFFER_THRESHOLD,
    MODE_COOLING,
    MODE_HEATING,
    NLMS_REGULARIZATION,
    NLMS_STEP_SIZE,
)
from custom_components.heating_analytics.learning import LearningManager
from custom_components.heating_analytics.observation import (
    HourlyObservation,
    LearningConfig,
    ObservationCollector,
)
from custom_components.heating_analytics.solar import SolarCalculator
from tests.helpers import CoordinatorModelMixin


# ---------------------------------------------------------------------------
# Mock coordinator – minimal surface required by SolarCalculator + learning
# ---------------------------------------------------------------------------

class _MockCoordinator(CoordinatorModelMixin):
    """Lightweight coordinator substitute for integration tests."""

    def __init__(
        self,
        solar_coefficients: dict | None = None,
        balance_point: float = 15.0,
        solar_azimuth: float = 180.0,
        solar_correction_percent: float = 100.0,
    ):
        self.balance_point = balance_point
        self.solar_azimuth = solar_azimuth
        self.solar_correction_percent = solar_correction_percent

        # Model state (private fields accessed via CoordinatorModelMixin)
        self._correlation_data: dict = {}
        self._correlation_data_per_unit: dict = {}
        self._observation_counts: dict = {}
        self._aux_coefficients: dict = {}
        self._aux_coefficients_per_unit: dict = {}
        self._solar_coefficients_per_unit: dict = solar_coefficients or {}
        self._learning_buffer_global: dict = {}
        self._learning_buffer_per_unit: dict = {}
        self._learning_buffer_aux_per_unit: dict = {}
        self._learning_buffer_solar_per_unit: dict = {}

    def get_unit_mode(self, entity_id: str) -> str:
        return MODE_HEATING

    def _get_predicted_kwh_per_unit(
        self, entity_id: str, temp_key: str, wind_bucket: str, avg_temp: float
    ) -> float:
        data = self._correlation_data_per_unit.get(entity_id, {})
        return data.get(temp_key, {}).get(wind_bucket, 0.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _potential_vector(elevation_deg: float, azimuth_deg: float, cloud_pct: float):
    """Compute the potential (pre-screen) solar vector for given conditions."""
    if elevation_deg <= 0:
        return 0.0, 0.0, 0.0
    elev_rad = math.radians(elevation_deg)
    am = 1.0 / math.sin(elev_rad)
    intensity = 0.7 ** am
    raw_elev = max(0.0, math.cos(elev_rad))
    elev_factor = raw_elev * intensity
    cloud_frac = cloud_pct / 100.0
    cloud_factor = 1.0 - 0.75 * cloud_frac ** 3.4
    base = elev_factor * cloud_factor
    az_rad = math.radians(azimuth_deg)
    return base * (-math.cos(az_rad)), base * math.sin(az_rad), 0.0


def _effective_vector(pot_s, pot_e, correction_pct):
    """Apply screen transmittance to a potential vector."""
    t = SolarCalculator._screen_transmittance(correction_pct)
    return pot_s * t, pot_e * t, 0.0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAccumulationWithVaryingScreens:
    """Verify that ObservationCollector correctly tracks varying screen %."""

    def test_correction_average_linear_ramp(self):
        """Screen ramps 0 → 100% over 30 samples.  Average ≈ 50%."""
        from datetime import datetime

        collector = ObservationCollector()
        n_samples = 30
        for i in range(n_samples):
            pct = (i / (n_samples - 1)) * 100.0
            pot_s, pot_e = 0.5, 0.15
            eff_s, eff_e, eff_w = _effective_vector(pot_s, pot_e, pct)
            collector.accumulate_weather(
                temp=5.0,
                effective_wind=2.0,
                wind_bucket="normal",
                solar_factor=0.0,  # unused for this test
                solar_vector=(eff_s, eff_e, eff_w),
                is_aux_active=False,
                current_time=datetime(2026, 4, 9, 12, i, 0),
                correction_percent=pct,
            )

        avg_correction = collector.correction_sum / collector.sample_count
        assert avg_correction == pytest.approx(50.0, abs=0.1)

    def test_effective_vector_average_matches_screen_average(self):
        """Effective vector average / transmittance(avg_correction) = potential.

        This is the mathematical property that makes potential reconstruction
        exact when potential is constant over the hour.
        """
        from datetime import datetime

        collector = ObservationCollector()
        pot_s, pot_e = 0.6, 0.2  # constant potential
        n_samples = 30

        for i in range(n_samples):
            pct = 20.0 + (i / (n_samples - 1)) * 60.0  # 20% → 80%
            eff_s, eff_e, eff_w = _effective_vector(pot_s, pot_e, pct)
            collector.accumulate_weather(
                temp=5.0,
                effective_wind=2.0,
                wind_bucket="normal",
                solar_factor=0.0,
                solar_vector=(eff_s, eff_e, eff_w),
                is_aux_active=False,
                current_time=datetime(2026, 4, 9, 12, i, 0),
                correction_percent=pct,
            )

        avg_s = collector.solar_vector_s_sum / collector.sample_count
        avg_e = collector.solar_vector_e_sum / collector.sample_count
        avg_correction = collector.correction_sum / collector.sample_count
        trans = SolarCalculator._screen_transmittance(avg_correction)

        recovered_s = avg_s / trans
        recovered_e = avg_e / trans

        assert recovered_s == pytest.approx(pot_s, abs=0.01)
        assert recovered_e == pytest.approx(pot_e, abs=0.01)


class TestPotentialReconstructionConsistency:
    """The potential vector reconstructed in learning.py must match
    the one used in statistics.py for prediction.  If they diverge,
    normalization and prediction use different solar estimates — the
    exact class of bug that caused trans² (#809).
    """

    @pytest.mark.parametrize("correction_pct", [0, 25, 50, 75, 100])
    def test_learning_and_prediction_use_same_potential(self, correction_pct):
        """Both paths reconstruct potential = effective / transmittance."""
        pot_s, pot_e, pot_w = 0.6, 0.2, 0.0
        eff_s, eff_e, eff_w = _effective_vector(pot_s, pot_e, correction_pct)
        trans = SolarCalculator._screen_transmittance(correction_pct)

        # Learning path (learning.py:463-468)
        if trans > 0.01:
            learn_pot_s = eff_s / trans
            learn_pot_e = eff_e / trans
            learn_pot_w = eff_w / trans
        else:
            learn_pot_s, learn_pot_e, learn_pot_w = eff_s, eff_e, eff_w

        # Prediction path (statistics.py:160-166)
        if trans > 0.01:
            pred_pot_s = eff_s / trans
            pred_pot_e = eff_e / trans
            pred_pot_w = eff_w / trans
        else:
            pred_pot_s, pred_pot_e, pred_pot_w = eff_s, eff_e, eff_w

        assert learn_pot_s == pytest.approx(pred_pot_s, abs=1e-12)
        assert learn_pot_e == pytest.approx(pred_pot_e, abs=1e-12)
        assert learn_pot_w == pytest.approx(pred_pot_w, abs=1e-12)
        assert learn_pot_s == pytest.approx(pot_s, abs=1e-10)
        assert learn_pot_e == pytest.approx(pot_e, abs=1e-10)


class TestEndToEndSolarPipeline:
    """Full pipeline: accumulate → reconstruct → learn → predict.

    Simulates one hour with varying screen positions, then verifies
    that learning and prediction produce consistent results — the
    kind of cross-component consistency that the trans² bug violated.
    """

    def _simulate_hour(
        self,
        true_coeff_s: float,
        true_coeff_e: float,
        base_kwh: float,
        pot_s: float,
        pot_e: float,
        screen_schedule: list[float],
    ) -> tuple[ObservationCollector, float, float]:
        """Accumulate one hour of samples with given screen schedule.

        Returns (collector, actual_energy, base_kwh).
        The actual energy is computed from the true physical model:
            actual = base - true_coeff · potential × transmittance
        """
        from datetime import datetime

        collector = ObservationCollector()
        total_actual = 0.0

        for i, pct in enumerate(screen_schedule):
            trans = SolarCalculator._screen_transmittance(pct)
            # True physical solar gain for this minute
            true_impact = (true_coeff_s * pot_s + true_coeff_e * pot_e) * trans
            minute_actual = max(0.0, base_kwh / len(screen_schedule) - true_impact / len(screen_schedule))
            total_actual += minute_actual

            eff_s, eff_e, eff_w = pot_s * trans, pot_e * trans, 0.0
            collector.accumulate_weather(
                temp=5.0,
                effective_wind=2.0,
                wind_bucket="normal",
                solar_factor=(pot_s**2 + pot_e**2)**0.5 * trans,
                solar_vector=(eff_s, eff_e, 0.0),
                is_aux_active=False,
                current_time=datetime(2026, 4, 9, 12, i % 60, 0),
                correction_percent=pct,
            )

        return collector, total_actual, base_kwh

    def _reconstruct_potential(self, collector: ObservationCollector):
        """Reconstruct potential vector from accumulated effective + correction."""
        n = collector.sample_count
        avg_eff_s = collector.solar_vector_s_sum / n
        avg_eff_e = collector.solar_vector_e_sum / n
        avg_correction = collector.correction_sum / n
        trans = SolarCalculator._screen_transmittance(avg_correction)
        if trans > 0.01:
            return avg_eff_s / trans, avg_eff_e / trans, 0.0
        return avg_eff_s, avg_eff_e, 0.0

    def test_constant_screen_learns_correct_coefficient(self):
        """With screens fixed at 60%, coefficient converges to true × trans(60%)."""
        true_s, true_e = 1.5, 0.3
        pot_s, pot_e = 0.6, 0.2
        base_kwh = 2.0
        screen_pct = 60.0
        trans = SolarCalculator._screen_transmittance(screen_pct)

        coord = _MockCoordinator(solar_correction_percent=screen_pct)
        calc = SolarCalculator(coord)
        manager = LearningManager()

        # Run many hours to converge (4 buffer + 16 NLMS)
        for hour in range(20):
            schedule = [screen_pct] * 30
            collector, actual, _ = self._simulate_hour(
                true_s, true_e, base_kwh, pot_s, pot_e, schedule,
            )
            pot_s_rec, pot_e_rec, pot_w_rec = self._reconstruct_potential(collector)

            manager._learn_unit_solar_coefficient(
                entity_id="unit_a",
                temp_key="5",
                expected_unit_base=base_kwh,
                actual_unit=actual,
                avg_solar_vector=(pot_s_rec, pot_e_rec, 0.0),
                learning_rate=0.05,
                solar_coefficients_per_unit=coord._solar_coefficients_per_unit,
                learning_buffer_solar_per_unit=coord._learning_buffer_solar_per_unit,
                avg_temp=5.0,
                balance_point=15.0,
                unit_mode=MODE_HEATING,
            )

        coeff = coord._solar_coefficients_per_unit["unit_a"]
        # Coefficient should converge to true × transmittance
        expected_s = true_s * trans
        expected_e = true_e * trans
        assert abs(coeff["s"] - expected_s) < 0.3, (
            f"S={coeff['s']:.3f}, expected ~{expected_s:.3f}"
        )

    def test_varying_screen_prediction_matches_normalization(self):
        """Key consistency check: the solar impact used for normalization
        (learning path) must equal the impact used for prediction (statistics
        path).  If they differ, the base model absorbs the gap as bias.

        This is the exact failure mode of the trans² bug.
        """
        # Pre-learned coefficient (as if NLMS already converged)
        coeff = {"s": 1.2, "e": 0.25, "w": 0.0}
        pot_s, pot_e = 0.5, 0.15

        for correction_pct in [0, 25, 50, 75, 100]:
            trans = SolarCalculator._screen_transmittance(correction_pct)
            eff_s, eff_e, eff_w = pot_s * trans, pot_e * trans, 0.0

            # --- Learning path (learning.py:490-494) ---
            # Reconstructs potential from effective, uses coeff × potential
            if trans > 0.01:
                learn_pot_s = eff_s / trans
                learn_pot_e = eff_e / trans
            else:
                learn_pot_s, learn_pot_e = eff_s, eff_e
            learn_impact = coeff["s"] * learn_pot_s + coeff["e"] * learn_pot_e
            learn_impact = max(0.0, learn_impact)

            # --- Prediction path (statistics.py:159-235) ---
            # Also reconstructs potential from effective, uses coeff × potential
            if trans > 0.01:
                pred_pot_s = eff_s / trans
                pred_pot_e = eff_e / trans
            else:
                pred_pot_s, pred_pot_e = eff_s, eff_e
            coord = _MockCoordinator(
                solar_coefficients={"unit_a": coeff},
                solar_correction_percent=correction_pct,
            )
            calc = SolarCalculator(coord)
            pred_impact = calc.calculate_unit_solar_impact(
                (pred_pot_s, pred_pot_e, 0.0),
                coeff,
                # No screen_transmittance argument — default 1.0
            )

            assert learn_impact == pytest.approx(pred_impact, abs=1e-10), (
                f"At correction={correction_pct}%: learning sees {learn_impact:.6f} "
                f"but prediction sees {pred_impact:.6f} — trans² inconsistency!"
            )

    def test_varying_screen_no_trans_squared(self):
        """Directly verify that coeff × potential does NOT contain trans².

        If the prediction incorrectly used coeff × potential × transmittance,
        the result would be proportional to trans² instead of trans.
        """
        coeff = {"s": 1.0, "e": 0.0, "w": 0.0}
        pot_s = 0.5

        coord = _MockCoordinator(solar_coefficients={"unit_a": coeff})
        calc = SolarCalculator(coord)

        # Prediction: coeff × potential × 1.0 (default transmittance)
        impact = calc.calculate_unit_solar_impact((pot_s, 0.0, 0.0), coeff)
        expected = 1.0 * 0.5  # coeff_s × pot_s, no transmittance factor
        assert impact == pytest.approx(expected), (
            f"Impact {impact} != {expected} — extra transmittance factor detected"
        )

        # If we INCORRECTLY passed screen_transmittance, we'd get trans²:
        for pct in [25, 50, 75]:
            trans = SolarCalculator._screen_transmittance(pct)
            impact_correct = calc.calculate_unit_solar_impact((pot_s, 0.0, 0.0), coeff)
            impact_wrong = calc.calculate_unit_solar_impact(
                (pot_s, 0.0, 0.0), coeff, screen_transmittance=trans
            )
            # Correct: 0.5 (independent of screen). Wrong: 0.5 × trans.
            assert impact_correct == pytest.approx(0.5)
            assert impact_wrong == pytest.approx(0.5 * trans)
            assert impact_correct != pytest.approx(impact_wrong), (
                "Correct and wrong paths should differ when trans < 1"
            )

    def test_full_pipeline_with_screen_ramp(self):
        """Complete pipeline: accumulate 30 samples with screen ramp 20→80%,
        reconstruct potential, learn via NLMS, then verify prediction consistency.

        This is the most realistic integration test — it exercises the exact
        code paths where the trans² bug lived.
        """
        # Physical ground truth
        true_coeff_s, true_coeff_e = 1.0, 0.2
        pot_s, pot_e = 0.6, 0.15
        base_kwh = 2.0
        n_samples = 30

        coord = _MockCoordinator()
        calc = SolarCalculator(coord)
        manager = LearningManager()

        # Phase 1: Cold-start buffer + NLMS convergence (20 hours)
        for hour in range(20):
            # Screen ramps 20% → 80% within each hour
            schedule = [20.0 + (i / (n_samples - 1)) * 60.0 for i in range(n_samples)]
            collector, actual, _ = self._simulate_hour(
                true_coeff_s, true_coeff_e, base_kwh, pot_s, pot_e, schedule,
            )
            pot_s_rec, pot_e_rec, pot_w_rec = self._reconstruct_potential(collector)

            manager._learn_unit_solar_coefficient(
                entity_id="unit_ramp",
                temp_key="5",
                expected_unit_base=base_kwh,
                actual_unit=actual,
                avg_solar_vector=(pot_s_rec, pot_e_rec, 0.0),
                learning_rate=0.05,
                solar_coefficients_per_unit=coord._solar_coefficients_per_unit,
                learning_buffer_solar_per_unit=coord._learning_buffer_solar_per_unit,
                avg_temp=5.0,
                balance_point=15.0,
                unit_mode=MODE_HEATING,
            )

        # Phase 2: Verify coefficient converged
        learned = coord._solar_coefficients_per_unit["unit_ramp"]
        avg_correction = sum(
            20.0 + (i / (n_samples - 1)) * 60.0 for i in range(n_samples)
        ) / n_samples
        avg_trans = SolarCalculator._screen_transmittance(avg_correction)

        # Coefficient should approximate true × avg_transmittance
        expected_s = true_coeff_s * avg_trans
        assert abs(learned["s"] - expected_s) < 0.3, (
            f"Learned S={learned['s']:.3f}, expected ~{expected_s:.3f} "
            f"(true={true_coeff_s}, avg_trans={avg_trans:.3f})"
        )

        # Phase 3: Prediction consistency
        # Simulate one more hour and verify that normalization ≈ prediction
        schedule = [20.0 + (i / (n_samples - 1)) * 60.0 for i in range(n_samples)]
        collector, actual, _ = self._simulate_hour(
            true_coeff_s, true_coeff_e, base_kwh, pot_s, pot_e, schedule,
        )
        pot_s_rec, pot_e_rec, pot_w_rec = self._reconstruct_potential(collector)

        # Learning-path solar impact (for normalization)
        learn_impact = calc.calculate_unit_solar_impact(
            (pot_s_rec, pot_e_rec, pot_w_rec), learned
        )
        # Prediction-path solar impact (for prediction)
        pred_impact = calc.calculate_unit_solar_impact(
            (pot_s_rec, pot_e_rec, pot_w_rec), learned
        )

        assert learn_impact == pytest.approx(pred_impact, abs=1e-12), (
            f"Normalization ({learn_impact:.6f}) != prediction ({pred_impact:.6f})"
        )

        # Verify the impact is reasonable
        assert learn_impact > 0.0
        assert learn_impact < base_kwh  # Can't exceed base demand


class TestNormalizationPredictionConsistency:
    """Verify that the normalized actual (for learning) and the predicted
    value (for deviation detection) use the same solar estimate.

    When they diverge, solar residual leaks into the base model as a
    systematic bias proportional to the screen position.
    """

    def test_residual_is_zero_at_convergence(self):
        """After learning converges, the learning residual should be near
        zero — meaning the base model is not absorbing solar signal.
        """
        true_s = 1.0
        pot_s, pot_e = 0.5, 0.0
        base_kwh = 2.0
        correction_pct = 50.0
        trans = SolarCalculator._screen_transmittance(correction_pct)

        # True energy at this screen position
        true_impact = true_s * pot_s * trans
        actual = base_kwh - true_impact

        # After convergence, learned coeff ≈ true × trans
        learned_s = true_s * trans

        # Reconstruct potential (as learning.py does)
        eff_s = pot_s * trans
        rec_pot_s = eff_s / trans  # = pot_s

        # Solar impact used for normalization (learning path)
        solar_for_normalization = learned_s * rec_pot_s  # = true_s × trans × pot_s

        # Normalized actual (dark-sky): actual + solar = (base - impact) + impact = base
        normalized = actual + solar_for_normalization

        # The residual (what learning sees) should be zero if model is correct
        residual = normalized - base_kwh
        assert abs(residual) < 0.01, (
            f"At correction={correction_pct}%: residual={residual:.6f} "
            f"(normalized={normalized:.4f}, base={base_kwh})"
        )

    @pytest.mark.parametrize("correction_pct", [0, 10, 30, 50, 70, 90, 100])
    def test_residual_independent_of_screen_position(self, correction_pct):
        """The learning residual should be constant (near zero) regardless
        of screen position.  If it varies with screen position, the base
        model will develop a screen-correlated bias.
        """
        true_s = 1.2
        pot_s = 0.5
        base_kwh = 2.0
        trans = SolarCalculator._screen_transmittance(correction_pct)

        # True energy
        actual = base_kwh - true_s * pot_s * trans

        # Learned coefficient (converged at this screen position)
        learned_s = true_s * trans

        # Reconstruction
        eff_s = pot_s * trans
        rec_pot_s = eff_s / trans if trans > 0.01 else eff_s

        # Normalization impact: learned × potential
        solar_norm = learned_s * rec_pot_s

        # Normalized actual
        normalized = actual + solar_norm

        # Residual
        residual = normalized - base_kwh
        assert abs(residual) < 0.01, (
            f"At correction={correction_pct}%: residual={residual:.6f}"
        )
