"""Tests for SolarCalculator.reconstruct_potential_vector (#876)."""
import pytest

from custom_components.heating_analytics.solar import SolarCalculator


# ---- equivalence with the pre-#876 inline pattern ----
# The helper must produce identical output to:
#   t_s, t_e, t_w = _screen_transmittance_vector(correction, screen_config)
#   pot_x = eff_x / t_x if t_x > 0.01 else eff_x
# across the slider range and every screen-config combination.


def _legacy_pattern(effective, correction_pct, screen_config):
    """Pre-#876 inline reconstruction, kept here as the regression oracle."""
    t_s, t_e, t_w = SolarCalculator._screen_transmittance_vector(
        correction_pct, screen_config
    )
    return (
        effective[0] / t_s if t_s > 0.01 else effective[0],
        effective[1] / t_e if t_e > 0.01 else effective[1],
        effective[2] / t_w if t_w > 0.01 else effective[2],
    )


@pytest.mark.parametrize("correction_pct", [0, 25, 50, 75, 100])
@pytest.mark.parametrize(
    "screen_config",
    [
        None,
        (True, True, True),
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (False, False, False),
        (True, True, False),
    ],
)
@pytest.mark.parametrize(
    "effective",
    [
        (0.4, 0.3, 0.2),      # all-nonzero baseline
        (0.0, 0.3, 0.2),      # no south (e.g. pure-east morning)
        (0.4, 0.0, 0.0),      # south-only (midday equator-facing sun)
        (0.0, 0.0, 0.0),      # dark hour
        (1e-9, 1e-9, 1e-9),   # near-floor magnitudes
    ],
)
def test_helper_matches_legacy_pattern(correction_pct, screen_config, effective):
    assert SolarCalculator.reconstruct_potential_vector(
        effective, correction_pct, screen_config
    ) == _legacy_pattern(effective, correction_pct, screen_config)


def test_screens_fully_open_returns_effective_unchanged():
    # 100 % open: every direction has t=1.0 regardless of screen config,
    # so reconstruction is the identity.
    effective = (0.5, 0.4, 0.3)
    out = SolarCalculator.reconstruct_potential_vector(
        effective, 100.0, (True, True, True)
    )
    assert out == pytest.approx(effective)


def test_unscreened_directions_pass_through_at_any_correction():
    # An unscreened facade has t=1.0 always; the slider only affects
    # screened facades.  Verify the south component (unscreened) survives
    # unchanged at a fully-closed slider while east/west reconstruct.
    effective = (0.4, 0.3, 0.2)
    out = SolarCalculator.reconstruct_potential_vector(
        effective, 0.0, (False, True, True)
    )
    assert out[0] == pytest.approx(0.4)  # unscreened south
    assert out[1] > effective[1]  # screened east got divided up
    assert out[2] > effective[2]  # screened west got divided up


def test_min_transmittance_guard_against_division_by_zero():
    # Below the floor (0.01 default), reconstruction returns the effective
    # value unchanged — matches every legacy call site's guard.
    out = SolarCalculator.reconstruct_potential_vector(
        (0.5, 0.5, 0.5), 0.0, (True, True, True),
        min_transmittance=0.99,  # forces every direction below floor
    )
    assert out == (0.5, 0.5, 0.5)


def test_legacy_screen_config_uses_composite_floor():
    # screen_config=None invokes the legacy composite floor
    # (DEFAULT_SOLAR_MIN_TRANSMITTANCE = 0.30).  Verify the divide
    # uses that floor at slider 0 %.
    effective = (0.3, 0.3, 0.3)
    out = SolarCalculator.reconstruct_potential_vector(effective, 0.0, None)
    assert out[0] == pytest.approx(0.3 / 0.30, rel=1e-4)
    assert out[1] == pytest.approx(0.3 / 0.30, rel=1e-4)
    assert out[2] == pytest.approx(0.3 / 0.30, rel=1e-4)
