"""Tests for 3-component (S, E, W) solar vector decomposition (#832).

Verifies:
- Non-negative basis functions at all azimuths
- Correct cardinal direction isolation (morning→E, noon→S, afternoon→W)
- Northern sun correctly clamped to zero south component
- Backward-compatible coefficient and log loading
- West coefficient convergence via NLMS
"""
import pytest
import math
from custom_components.heating_analytics.solar import SolarCalculator
from custom_components.heating_analytics.learning import LearningManager
from custom_components.heating_analytics.storage import _sanitize_solar_coeff
from custom_components.heating_analytics.const import MODE_HEATING
from tests.helpers import CoordinatorModelMixin


class _MockCoord(CoordinatorModelMixin):
    solar_azimuth = 180
    balance_point = 15.0
    _solar_coefficients_per_unit = {}


class TestNonNegativeBasis:
    """All three basis functions must be >= 0 at every azimuth."""

    @pytest.mark.parametrize("azimuth", list(range(0, 361, 15)))
    def test_all_components_non_negative(self, azimuth):
        calc = SolarCalculator(_MockCoord())
        s, e, w = calc.calculate_solar_vector(30.0, float(azimuth), 0.0)
        assert s >= 0.0, f"South negative at az={azimuth}: {s}"
        assert e >= 0.0, f"East negative at az={azimuth}: {e}"
        assert w >= 0.0, f"West negative at az={azimuth}: {w}"

    def test_zero_below_horizon(self):
        calc = SolarCalculator(_MockCoord())
        s, e, w = calc.calculate_solar_vector(-5.0, 180.0, 0.0)
        assert (s, e, w) == (0.0, 0.0, 0.0)


class TestCardinalDirections:
    """Verify correct component isolation at cardinal azimuths."""

    def test_solar_noon_pure_south(self):
        """Az=180: all energy in south, zero east and west."""
        calc = SolarCalculator(_MockCoord())
        s, e, w = calc.calculate_solar_vector(30.0, 180.0, 0.0)
        assert s > 0.0
        assert e == pytest.approx(0.0, abs=1e-10)
        assert w == pytest.approx(0.0, abs=1e-10)

    def test_due_east_pure_east(self):
        """Az=90: all energy in east, zero south and west."""
        calc = SolarCalculator(_MockCoord())
        s, e, w = calc.calculate_solar_vector(30.0, 90.0, 0.0)
        assert s == pytest.approx(0.0, abs=1e-10)
        assert e > 0.0
        assert w == pytest.approx(0.0, abs=1e-10)

    def test_due_west_pure_west(self):
        """Az=270: all energy in west, zero south and east."""
        calc = SolarCalculator(_MockCoord())
        s, e, w = calc.calculate_solar_vector(30.0, 270.0, 0.0)
        assert s == pytest.approx(0.0, abs=1e-10)
        assert e == pytest.approx(0.0, abs=1e-10)
        assert w > 0.0

    def test_due_north_all_zero(self):
        """Az=0 (or 360): sun due north, all components zero."""
        calc = SolarCalculator(_MockCoord())
        s, e, w = calc.calculate_solar_vector(30.0, 0.0, 0.0)
        assert s == pytest.approx(0.0, abs=1e-10)
        assert e == pytest.approx(0.0, abs=1e-10)
        assert w == pytest.approx(0.0, abs=1e-10)

    def test_morning_has_east_no_west(self):
        """Az=120 (SE morning): positive S and E, zero W."""
        calc = SolarCalculator(_MockCoord())
        s, e, w = calc.calculate_solar_vector(30.0, 120.0, 0.0)
        assert s > 0.0
        assert e > 0.0
        assert w == pytest.approx(0.0, abs=1e-10)

    def test_afternoon_has_west_no_east(self):
        """Az=240 (SW afternoon): positive S and W, zero E."""
        calc = SolarCalculator(_MockCoord())
        s, e, w = calc.calculate_solar_vector(30.0, 240.0, 0.0)
        assert s > 0.0
        assert e == pytest.approx(0.0, abs=1e-10)
        assert w > 0.0


class TestNorthernSunHighLatitude:
    """At 60°N in summer, the sun can be north — south component must be 0."""

    def test_ne_sun_south_clamped(self):
        """Az=30 (NNE): south component is 0, east captures the gain."""
        calc = SolarCalculator(_MockCoord())
        s, e, w = calc.calculate_solar_vector(5.0, 30.0, 0.0)
        assert s == pytest.approx(0.0, abs=1e-10)
        assert e > 0.0
        assert w == pytest.approx(0.0, abs=1e-10)

    def test_nw_sun_south_clamped(self):
        """Az=330 (NNW): south component is 0, west captures the gain."""
        calc = SolarCalculator(_MockCoord())
        s, e, w = calc.calculate_solar_vector(5.0, 330.0, 0.0)
        assert s == pytest.approx(0.0, abs=1e-10)
        assert e == pytest.approx(0.0, abs=1e-10)
        assert w > 0.0


class TestEastWestOrthogonality:
    """East and West components never overlap — disjoint temporal support."""

    @pytest.mark.parametrize("azimuth", list(range(0, 361, 5)))
    def test_east_west_never_both_positive(self, azimuth):
        """At any azimuth, at most one of E and W is non-zero."""
        calc = SolarCalculator(_MockCoord())
        _, e, w = calc.calculate_solar_vector(30.0, float(azimuth), 0.0)
        assert e * w == pytest.approx(0.0, abs=1e-15), (
            f"E={e} and W={w} both non-zero at az={azimuth}"
        )


class TestBackwardCompatibility:
    """Old data without 'w' key loads correctly."""

    def test_sanitize_old_format(self):
        """Old {"s": x, "e": y} format gets "w": 0.0 via default."""
        old = {"s": 1.5, "e": 0.3}
        result = _sanitize_solar_coeff(old)
        assert result == {"s": 1.5, "e": 0.3, "w": 0.0}

    def test_sanitize_preserves_w(self):
        """New {"s": x, "e": y, "w": z} format preserved."""
        new = {"s": 1.5, "e": 0.3, "w": 0.2}
        result = _sanitize_solar_coeff(new)
        assert result == {"s": 1.5, "e": 0.3, "w": 0.2}

    def test_sanitize_clamps_negative(self):
        """Negative coefficients are clamped to 0."""
        bad = {"s": -0.5, "e": 0.3, "w": -0.1}
        result = _sanitize_solar_coeff(bad)
        assert result["s"] == 0.0
        assert result["e"] == 0.3
        assert result["w"] == 0.0


class TestWestCoefficientConvergence:
    """NLMS should converge a west coefficient for a west-facing unit."""

    def test_nlms_learns_west_coefficient(self):
        """With afternoon-only sun (W>0, E=0), coeff_w converges."""
        manager = LearningManager()
        coeffs = {}
        buffers = {}
        true_w = 0.8

        # 20 samples of afternoon sun (west-only)
        for _ in range(20):
            solar_s, solar_e, solar_w = 0.3, 0.0, 0.4
            true_impact = true_w * solar_w  # only west contribution
            actual = max(0.0, 2.0 - true_impact)
            manager._learn_unit_solar_coefficient(
                "unit_west", "10", 2.0, actual,
                (solar_s, solar_e, solar_w),
                0.01, coeffs, buffers, 5.0, 15.0, MODE_HEATING,
            )

        assert "unit_west" in coeffs
        # Heating-regime read per #868.
        final = coeffs["unit_west"]["heating"]
        assert final["w"] > 0.3, f"West coeff {final['w']} should be significant"
        assert abs(final["w"] - true_w) < 0.4, (
            f"West coeff {final['w']} not converging toward {true_w}"
        )


class TestColdStart3x3Solver:
    """Cold-start 3×3 least squares correctly initialises all three coefficients."""

    def test_cold_start_with_mixed_morning_afternoon(self):
        """With diverse sun angles spanning morning and afternoon, all 3 coefficients are initialised.

        This test would have caught the Cramer's rule det_w transposition bug:
        with incorrect det_w, the W coefficient was negative and clamped to 0.
        """
        manager = LearningManager()
        coeffs = {}
        buffers = {}
        true_s, true_e, true_w = 0.5, 0.3, 0.4

        # 4 samples: 2 morning (E>0, W=0), 2 afternoon (E=0, W>0)
        samples = [
            # (solar_s, solar_e, solar_w) — morning SE
            (0.3, 0.35, 0.0),
            # morning E-dominant
            (0.1, 0.45, 0.0),
            # afternoon SW
            (0.3, 0.0, 0.35),
            # afternoon W-dominant
            (0.1, 0.0, 0.45),
        ]
        for solar_s, solar_e, solar_w in samples:
            true_impact = true_s * solar_s + true_e * solar_e + true_w * solar_w
            actual = max(0.0, 2.0 - true_impact)
            manager._learn_unit_solar_coefficient(
                "unit_3x3", "10", 2.0, actual,
                (solar_s, solar_e, solar_w),
                0.01, coeffs, buffers, 5.0, 15.0, MODE_HEATING,
            )

        assert "unit_3x3" in coeffs, "Cold start should have triggered with 4 samples"
        final = coeffs["unit_3x3"]["heating"]
        # All three coefficients should be positive (not clamped to 0)
        assert final["s"] > 0.05, f"S coeff {final['s']} should be positive"
        assert final["e"] > 0.05, f"E coeff {final['e']} should be positive"
        assert final["w"] > 0.05, f"W coeff {final['w']} should be positive (was 0 with old det_w bug)"

    def test_cold_start_collinear_fallback(self):
        """When all data is from one half of the day, collinear fallback triggers."""
        manager = LearningManager()
        coeffs = {}
        buffers = {}

        # 4 samples, all afternoon (W>0, E=0) — the 3×3 determinant will be
        # near-singular because sum_ee ≈ 0
        for _ in range(4):
            solar_s, solar_e, solar_w = 0.3, 0.0, 0.4
            true_impact = 0.5 * solar_s + 0.0 * solar_e + 0.8 * solar_w
            actual = max(0.0, 2.0 - true_impact)
            manager._learn_unit_solar_coefficient(
                "unit_collinear", "10", 2.0, actual,
                (solar_s, solar_e, solar_w),
                0.01, coeffs, buffers, 5.0, 15.0, MODE_HEATING,
            )

        assert "unit_collinear" in coeffs, "Collinear fallback should still produce coefficients"
        final = coeffs["unit_collinear"]["heating"]
        # S and W should be positive (projected along dominant direction)
        assert final["s"] >= 0.0
        assert final["w"] >= 0.0
        # E should be 0 (no east signal)
        assert final["e"] == 0.0


class TestEffectiveVectorScreenTransmittance:
    """Screen transmittance applies equally to all 3 components."""

    def test_all_three_scaled(self):
        calc = SolarCalculator(_MockCoord())
        potential = (0.5, 0.3, 0.2)
        result = calc.calculate_effective_solar_vector(potential, 50.0)
        t = SolarCalculator._screen_transmittance(50.0)
        assert result[0] == pytest.approx(0.5 * t)
        assert result[1] == pytest.approx(0.3 * t)
        assert result[2] == pytest.approx(0.2 * t)

    def test_full_open_identity(self):
        calc = SolarCalculator(_MockCoord())
        potential = (0.5, 0.3, 0.2)
        result = calc.calculate_effective_solar_vector(potential, 100.0)
        assert result == pytest.approx(potential)
