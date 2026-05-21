"""Tests for SolarCalculator.calculate_unit_potential_4d (#954).

Pure-compute method, no coordinator state used in the body — the
coordinator argument is accepted for API symmetry only.  A bare
``MagicMock`` for the coordinator is sufficient.

Diffuse term uses a fixed internal SVF = 0.5 (vertical window sees half
the hemisphere); per-facade asymmetry is absorbed by ``c_diff``.  Per-
facade SVF config was removed in #991 — obstruction-induced elevation
structure is handled by the direct-beam gate, not by tuning this term.
"""
from __future__ import annotations

import math
from unittest.mock import MagicMock

from custom_components.heating_analytics.solar import SolarCalculator
from custom_components.heating_analytics.const import SCREEN_DIRECT_TRANSMITTANCE


def _calc() -> SolarCalculator:
    return SolarCalculator(MagicMock())


def test_south_at_noon():
    """Sun in the south at 45° elevation lights only the south facade
    directly; diffuse uses fixed SVF=0.5 * mean transmittance (1.0
    unscreened)."""
    calc = _calc()
    pot_s, pot_e, pot_w, pot_diff = calc.calculate_unit_potential_4d(
        entity_id="sensor.x",
        dni=500.0,
        dhi=100.0,
        sun_elev_deg=45.0,
        sun_azimuth_deg=180.0,
        screen_config=(False, False, False),
        correction_percent=100.0,
    )
    # 500 * cos(45) * (-cos(180)) * 1.0 = 500 * 0.7071 * 1
    assert abs(pot_s - 353.5534) < 0.5
    assert abs(pot_e) < 1e-9
    assert abs(pot_w) < 1e-9
    # 100 * 0.5 * mean(1, 1, 1) = 50.0
    assert abs(pot_diff - 50.0) < 1e-9


def test_east_at_sunrise():
    """Sun due east at low elevation lights only the east facade."""
    calc = _calc()
    pot_s, pot_e, pot_w, pot_diff = calc.calculate_unit_potential_4d(
        entity_id="sensor.x",
        dni=300.0,
        dhi=80.0,
        sun_elev_deg=10.0,
        sun_azimuth_deg=90.0,
        screen_config=(False, False, False),
        correction_percent=100.0,
    )
    expected_e = 300.0 * math.cos(math.radians(10.0)) * 1.0 * 1.0
    assert abs(pot_e - expected_e) < 0.5
    assert pot_s == 0.0
    assert pot_w == 0.0
    assert abs(pot_diff - 40.0) < 1e-9


def test_west_at_sunset():
    """Sun due west lights only the west facade."""
    calc = _calc()
    pot_s, pot_e, pot_w, _ = calc.calculate_unit_potential_4d(
        entity_id="sensor.x",
        dni=400.0,
        dhi=50.0,
        sun_elev_deg=15.0,
        sun_azimuth_deg=270.0,
        screen_config=(False, False, False),
        correction_percent=100.0,
    )
    assert pot_w > 0.0
    assert abs(pot_s) < 1e-9
    assert abs(pot_e) < 1e-9


def test_below_horizon():
    """Negative elevation yields a zero vector regardless of irradiance."""
    calc = _calc()
    out = calc.calculate_unit_potential_4d(
        entity_id="sensor.x",
        dni=1000.0,
        dhi=200.0,
        sun_elev_deg=-5.0,
        sun_azimuth_deg=180.0,
        screen_config=(False, False, False),
        correction_percent=100.0,
    )
    assert out == (0.0, 0.0, 0.0, 0.0)


def test_screen_attenuation_per_direction():
    """Only the south facade screened; correction_percent=0 (closed).

    South direct should scale by SCREEN_DIRECT_TRANSMITTANCE (~0.08);
    east/west stay at 1.0.  Diffuse uses 0.5 * mean(t_s, t_e, t_w).
    """
    calc = _calc()
    pot_s, pot_e, pot_w, pot_diff = calc.calculate_unit_potential_4d(
        entity_id="sensor.x",
        dni=500.0,
        dhi=100.0,
        sun_elev_deg=45.0,
        sun_azimuth_deg=180.0,
        screen_config=(True, False, False),
        correction_percent=0.0,
    )
    cos45 = math.cos(math.radians(45.0))
    expected_s_unscreened = 500.0 * cos45  # before screen factor
    assert abs(pot_s - expected_s_unscreened * SCREEN_DIRECT_TRANSMITTANCE) < 0.5
    assert abs(pot_e) < 1e-9
    assert abs(pot_w) < 1e-9
    avg_t = (SCREEN_DIRECT_TRANSMITTANCE + 1.0 + 1.0) / 3.0
    expected_diff = 100.0 * 0.5 * avg_t
    assert abs(pot_diff - expected_diff) < 1e-9
