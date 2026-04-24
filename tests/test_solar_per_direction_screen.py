"""Tests for per-direction screen transmittance (#826).

Verifies:
- _screen_transmittance_vector returns 3-tuple aligned with (S, E, W)
- Screened directions follow SCREEN_DIRECT_TRANSMITTANCE floor + slider ramp
- Unscreened directions stay at 1.0 regardless of slider
- Legacy (screen_config=None) falls back to DEFAULT_SOLAR_MIN_TRANSMITTANCE
- Effective vector applies per-direction factor independently
- Coefficient learning normalisation ≡ prediction (CLAUDE.md invariant #1)
"""
from unittest.mock import MagicMock

import math
import pytest

from custom_components.heating_analytics.solar import SolarCalculator
from custom_components.heating_analytics.const import (
    DEFAULT_SOLAR_MIN_TRANSMITTANCE,
    SCREEN_DIRECT_TRANSMITTANCE,
)


class TestScreenTransmittanceVector:
    """Per-direction transmittance vector (S, E, W)."""

    def test_legacy_none_returns_uniform_composite_floor(self):
        """screen_config=None → uniform composite floor at 0%."""
        s, e, w = SolarCalculator._screen_transmittance_vector(0.0, None)
        assert s == e == w == pytest.approx(DEFAULT_SOLAR_MIN_TRANSMITTANCE)

    def test_legacy_none_fully_open(self):
        """screen_config=None at 100% → all 1.0."""
        s, e, w = SolarCalculator._screen_transmittance_vector(100.0, None)
        assert s == e == w == pytest.approx(1.0)

    def test_all_screened_closed(self):
        """All three screened, slider 0% → all = SCREEN_DIRECT_TRANSMITTANCE."""
        s, e, w = SolarCalculator._screen_transmittance_vector(0.0, (True, True, True))
        assert s == e == w == pytest.approx(SCREEN_DIRECT_TRANSMITTANCE)

    def test_all_screened_open(self):
        """All three screened, slider 100% → all = 1.0."""
        s, e, w = SolarCalculator._screen_transmittance_vector(100.0, (True, True, True))
        assert s == e == w == pytest.approx(1.0)

    def test_south_only_closed(self):
        """South screened only, slider 0% → S=floor, E=W=1.0."""
        s, e, w = SolarCalculator._screen_transmittance_vector(0.0, (True, False, False))
        assert s == pytest.approx(SCREEN_DIRECT_TRANSMITTANCE)
        assert e == pytest.approx(1.0)
        assert w == pytest.approx(1.0)

    def test_east_west_screened_south_open(self):
        """E/W screened, S unscreened → S=1.0, E/W=floor at 0%."""
        s, e, w = SolarCalculator._screen_transmittance_vector(0.0, (False, True, True))
        assert s == pytest.approx(1.0)
        assert e == pytest.approx(SCREEN_DIRECT_TRANSMITTANCE)
        assert w == pytest.approx(SCREEN_DIRECT_TRANSMITTANCE)

    def test_no_screens_anywhere_slider_ignored(self):
        """All unchecked → transmittance always 1.0 across slider range."""
        for pct in [0, 25, 50, 75, 100]:
            s, e, w = SolarCalculator._screen_transmittance_vector(
                pct, (False, False, False)
            )
            assert s == e == w == 1.0, f"slider={pct}"

    def test_screened_ramp_monotonic(self):
        """Screened direction is monotone increasing in slider."""
        prev = -1.0
        for pct in range(0, 101, 10):
            s, _, _ = SolarCalculator._screen_transmittance_vector(
                pct, (True, False, False)
            )
            assert s >= prev
            prev = s
        assert s == pytest.approx(1.0)

    def test_clamps_out_of_range_slider(self):
        """Slider outside [0, 100] is clamped before mapping."""
        s_lo, _, _ = SolarCalculator._screen_transmittance_vector(-50.0, (True, True, True))
        s_hi, _, _ = SolarCalculator._screen_transmittance_vector(150.0, (True, True, True))
        assert s_lo == pytest.approx(SCREEN_DIRECT_TRANSMITTANCE)
        assert s_hi == pytest.approx(1.0)

    def test_malformed_screen_config_falls_back_to_legacy(self):
        """Defensive: empty tuple / wrong shape → legacy composite floor."""
        s, e, w = SolarCalculator._screen_transmittance_vector(0.0, ())
        assert s == e == w == pytest.approx(DEFAULT_SOLAR_MIN_TRANSMITTANCE)

    def test_scalar_wrapper_averages_three_directions(self):
        """_screen_transmittance scalar = arithmetic mean of vector form."""
        cfg = (True, False, False)
        scalar = SolarCalculator._screen_transmittance(0.0, cfg)
        s, e, w = SolarCalculator._screen_transmittance_vector(0.0, cfg)
        assert scalar == pytest.approx((s + e + w) / 3.0)


class TestEffectiveVectorPerDirection:
    """calculate_effective_solar_vector applies per-direction transmittance."""

    def _calc(self, screen_config):
        coord = MagicMock()
        coord.screen_config = screen_config
        return SolarCalculator(coord)

    def test_unscreened_direction_passes_through(self):
        calc = self._calc((True, False, True))  # only east unscreened
        eff = calc.calculate_effective_solar_vector((1.0, 1.0, 1.0), 0.0)
        # South screened → SCREEN_DIRECT_TRANSMITTANCE; East unscreened → 1.0; West screened → SCREEN_DIRECT_TRANSMITTANCE
        assert eff[0] == pytest.approx(SCREEN_DIRECT_TRANSMITTANCE)
        assert eff[1] == pytest.approx(1.0)
        assert eff[2] == pytest.approx(SCREEN_DIRECT_TRANSMITTANCE)

    def test_override_arg_overrides_coordinator(self):
        calc = self._calc((True, True, True))
        eff = calc.calculate_effective_solar_vector(
            (1.0, 1.0, 1.0), 0.0, screen_config=(False, False, False)
        )
        assert eff == (1.0, 1.0, 1.0)

    def test_legacy_coordinator_no_attr(self):
        """Coordinator without screen_config attribute → legacy path."""
        coord = MagicMock(spec=[])  # no attributes
        calc = SolarCalculator(coord)
        eff = calc.calculate_effective_solar_vector((1.0, 1.0, 1.0), 0.0)
        # All three components scaled by composite legacy floor (0.30)
        assert eff[0] == pytest.approx(DEFAULT_SOLAR_MIN_TRANSMITTANCE)
        assert eff[1] == pytest.approx(DEFAULT_SOLAR_MIN_TRANSMITTANCE)
        assert eff[2] == pytest.approx(DEFAULT_SOLAR_MIN_TRANSMITTANCE)


class TestPredictionEqualsNormalization:
    """CLAUDE.md invariant #1: prediction = normalization (no trans² bug).

    Per direction since #826: the coefficient absorbs avg per-direction
    transmittance via NLMS learning target (base − actual).  Both
    prediction and normalisation must therefore use ``coeff × potential``
    with no extra transmittance factor — verified per direction.
    """

    @pytest.mark.parametrize(
        "screen_config,correction_pct",
        [
            ((True, True, True), 50.0),
            ((True, False, False), 30.0),
            ((False, True, True), 70.0),
            ((False, False, False), 0.0),
            (None, 50.0),  # legacy
        ],
    )
    def test_potential_round_trip_per_direction(self, screen_config, correction_pct):
        """Potential → effective → potential reconstruction is exact per direction."""
        potential = (0.6, 0.4, 0.3)
        t_s, t_e, t_w = SolarCalculator._screen_transmittance_vector(
            correction_pct, screen_config
        )
        eff = (potential[0] * t_s, potential[1] * t_e, potential[2] * t_w)
        recovered = (
            eff[0] / t_s if t_s > 0.01 else eff[0],
            eff[1] / t_e if t_e > 0.01 else eff[1],
            eff[2] / t_w if t_w > 0.01 else eff[2],
        )
        for a, b in zip(potential, recovered):
            assert a == pytest.approx(b)


class TestUnscreenedDirectionDecoupledFromSlider:
    """Closing south screens does not change the model's east-window estimate."""

    def test_east_unscreened_predicted_impact_constant_across_slider(self):
        """Unscreened east → impact attributed to east is independent of slider."""
        coord = MagicMock()
        coord.screen_config = (True, False, True)  # east unscreened
        coord.solar_azimuth = 180
        calc = SolarCalculator(coord)

        coeff = {"s": 0.5, "e": 0.7, "w": 0.4}
        # Pure east potential vector
        potential = (0.0, 0.5, 0.0)

        impacts = []
        for pct in [0, 25, 50, 75, 100]:
            eff = calc.calculate_effective_solar_vector(potential, pct)
            t_s, t_e, t_w = SolarCalculator._screen_transmittance_vector(
                pct, coord.screen_config
            )
            # Reconstruct potential per direction (matches statistics.py path)
            pot_recon = (
                eff[0] / t_s if t_s > 0.01 else eff[0],
                eff[1] / t_e if t_e > 0.01 else eff[1],
                eff[2] / t_w if t_w > 0.01 else eff[2],
            )
            impact = calc.calculate_unit_solar_impact(pot_recon, coeff)
            impacts.append(impact)

        # All impacts identical because the east channel is unattenuated
        # AND the input is pure east — slider should have zero effect.
        for v in impacts[1:]:
            assert v == pytest.approx(impacts[0]), (
                "East-window impact should be slider-invariant when east is unscreened"
            )
