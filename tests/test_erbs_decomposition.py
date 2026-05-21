"""Tests for Erbs (1982) GHI -> (DNI, DHI) decomposition and the
resolve_dni_dhi input ladder.  Pure math; no HA fixtures needed.
"""
from __future__ import annotations

import math

import pytest

from custom_components.heating_analytics.solar import (
    erbs_decomposition,
    resolve_dni_dhi,
)


def test_erbs_kt_low_cloudy():
    """Heavy overcast: kT ~ 0.147, kd ~ 0.987 (linear branch)."""
    ghi = 100.0
    elev = 30.0
    doy = 180
    dni, dhi = erbs_decomposition(ghi, elev, doy)

    # Expected values
    sin_elev = math.sin(math.radians(elev))
    e_0 = 1.0 + 0.033 * math.cos(2.0 * math.pi * doy / 365.0)
    kT = ghi / (1367.0 * e_0 * sin_elev)
    assert kT < 0.22
    kd = 1.0 - 0.09 * kT
    expected_dhi = kd * ghi
    expected_dni = (ghi - expected_dhi) / sin_elev

    assert dhi == pytest.approx(expected_dhi, abs=0.5)
    assert dni == pytest.approx(expected_dni, abs=0.5)
    # Sanity: heavy overcast, DHI ~ 98.7, DNI ~ 2.6
    assert 95.0 < dhi < 100.0
    assert 0.0 <= dni < 10.0


def test_erbs_kt_mid_broken_cloud():
    """Broken cloud: kT in polynomial branch."""
    ghi = 400.0
    elev = 45.0
    doy = 180
    dni, dhi = erbs_decomposition(ghi, elev, doy)

    sin_elev = math.sin(math.radians(elev))
    e_0 = 1.0 + 0.033 * math.cos(2.0 * math.pi * doy / 365.0)
    kT = ghi / (1367.0 * e_0 * sin_elev)
    assert 0.22 < kT <= 0.80  # In polynomial branch

    # kd should be sensible
    kd = dhi / ghi
    assert 0.165 < kd < 1.0
    assert dni > 0.0
    assert dhi > 0.0


def test_erbs_kt_high_clear():
    """Clear sky: kT ~ 0.78, kd ~ 0.165 (high-kT branch or near-boundary)."""
    ghi = 900.0
    elev = 60.0
    doy = 172  # summer solstice
    dni, dhi = erbs_decomposition(ghi, elev, doy)

    sin_elev = math.sin(math.radians(elev))
    e_0 = 1.0 + 0.033 * math.cos(2.0 * math.pi * doy / 172)
    # Just check the result shape; allow either branch since kT is near 0.80.
    kd = dhi / ghi
    # Clear-sky kd should be small (near 0.165 in the constant branch
    # or close to it via polynomial tail).
    assert kd < 0.30
    assert dni > 700.0
    # Expected DNI ~ (900 - 148.5) / sin(60) ~ 868 when in constant branch.
    assert dni == pytest.approx(868.0, abs=80.0)


def test_erbs_below_horizon():
    """Sun below horizon: returns (0, 0) regardless of GHI input."""
    assert erbs_decomposition(500.0, -5.0, 180) == (0.0, 0.0)
    assert erbs_decomposition(0.0, -5.0, 180) == (0.0, 0.0)
    assert erbs_decomposition(100.0, 0.0, 180) == (0.0, 0.0)


def test_resolve_ladder_priority():
    """Verify the 4-step ladder ordering and edge cases."""
    # 1. GHI present -> erbs_from_ghi (even with native DNI/DHI also present)
    dni, dhi, src = resolve_dni_dhi(
        dni_in=500.0, dhi_in=100.0, ghi_in=300.0,
        cloud_coverage_pct=50.0, sun_elev_deg=30.0, day_of_year=180,
    )
    assert src == "erbs_from_ghi"
    assert dni >= 0.0 and dhi >= 0.0

    # 2. Native DNI+DHI when no GHI
    dni, dhi, src = resolve_dni_dhi(
        dni_in=500.0, dhi_in=100.0, ghi_in=None,
        cloud_coverage_pct=50.0, sun_elev_deg=30.0, day_of_year=180,
    )
    assert src == "native"
    assert dni == pytest.approx(500.0, abs=0.5)
    assert dhi == pytest.approx(100.0, abs=0.5)

    # 3. Cloud-coverage synthetic fallback
    dni, dhi, src = resolve_dni_dhi(
        dni_in=None, dhi_in=None, ghi_in=None,
        cloud_coverage_pct=50.0, sun_elev_deg=30.0, day_of_year=180,
    )
    assert src == "kasten_synthetic"
    assert dni > 0.0
    assert dhi > 0.0

    # 4. Nothing -> none
    dni, dhi, src = resolve_dni_dhi(
        dni_in=None, dhi_in=None, ghi_in=None,
        cloud_coverage_pct=None, sun_elev_deg=30.0, day_of_year=180,
    )
    assert src == "none"
    assert dni == 0.0 and dhi == 0.0

    # 5. Sun below horizon -> no_sun regardless of inputs
    dni, dhi, src = resolve_dni_dhi(
        dni_in=500.0, dhi_in=100.0, ghi_in=300.0,
        cloud_coverage_pct=50.0, sun_elev_deg=-5.0, day_of_year=180,
    )
    assert src == "no_sun"
    assert dni == 0.0 and dhi == 0.0
