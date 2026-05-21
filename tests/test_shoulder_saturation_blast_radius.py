"""Tests for the ``shoulder_saturation_blast_radius`` block of ``diagnose_solar`` (#928).

Quantifies whether the hard ``max(0, base − solar)`` saturation clamp is the
dominant driver of shoulder-bucket deviation by re-computing four "expected"
variants per hour and comparing median absolute residuals against actual.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock

from custom_components.heating_analytics.diagnostics import DiagnosticsEngine


def _coord(hourly_log, *, balance_point=17.0, solar=None,
           solar_coefficients_4d_per_unit=None, energy_sensors=None):
    coord = MagicMock()
    coord._hourly_log = hourly_log
    coord.balance_point = balance_point
    # Explicitly None (rather than MagicMock auto-attr) so the replay path
    # short-circuits cleanly in tests that don't exercise it.
    coord.solar = solar
    coord._solar_coefficients_4d_per_unit = solar_coefficients_4d_per_unit or {}
    coord.energy_sensors = energy_sensors or []
    coord.is_solar_affected = lambda eid: True
    coord.screen_config_for_entity = lambda eid: (False, False, False)
    return coord


def _entry(
    ts,
    *,
    temp,
    actual,
    expected_3d,
    solar_eff,
    solar_wasted=0.0,
    solar_4d=None,
):
    e = {
        "timestamp": ts,
        "temp": temp,
        "actual_kwh": actual,
        "expected_kwh": expected_3d,
        "solar_impact_kwh": solar_eff,
        "solar_wasted_kwh": solar_wasted,
    }
    if solar_4d is not None:
        e["solar_impact_4d_kwh"] = solar_4d
    return e


def _ts(i):
    return (datetime.now() - timedelta(days=1, hours=-i)).isoformat()


def test_returns_no_shoulder_when_log_outside_window():
    log = [_entry(_ts(0), temp=5.0, actual=2.0, expected_3d=2.1, solar_eff=0.0)]
    out = DiagnosticsEngine(_coord(log))._compute_shoulder_saturation_blast_radius(30)
    assert out["available"] is False
    assert out["reason"] == "no_shoulder_hours"


def test_field_example_from_issue_surfaces_as_saturation_event():
    """Reproduce the 928 hour-17 evidence: base=0.043, raw_solar=0.234,
    actual=0.187, expected_kwh=0.0 (clamped), wasted=0.234-0.043=0.191.

    Saturation event must be counted; 3D-clamped residual = 0.187,
    3D-unclamped residual = |0.187 - (0.0 - 0.191)| = |0.187 + 0.191| = 0.378.
    Clamp is correctly the better answer here (actual > 0, unclamped goes
    negative).  The diagnostic surfaces both so the user can see that
    soft-saturation isn't a free win when raw solar genuinely exceeds base.
    """
    log = []
    # Need >=5 shoulder hours to clear the no_shoulder gate; replicate
    # the issue example five times.
    for i in range(5):
        log.append(_entry(
            _ts(i), temp=15.0, actual=0.187, expected_3d=0.0,
            solar_eff=0.043, solar_wasted=0.191,
        ))
    out = DiagnosticsEngine(_coord(log, balance_point=17.0))\
        ._compute_shoulder_saturation_blast_radius(30)

    assert out["available"] is True
    assert out["n_shoulder_hours"] == 5
    assert out["n_saturation_events"] == 5
    assert out["saturation_event_share"] == 1.0

    sat = out["median_abs_residual"]["saturation_events_only"]
    assert sat["expected_3d_clamped"] == 0.187
    # Unclamped: actual - (expected - wasted) = 0.187 - (-0.191) = 0.378
    assert abs(sat["expected_3d_unclamped"] - 0.378) < 1e-3
    # No 4D data in this fixture
    assert sat["expected_4d_clamped"] is None
    assert sat["expected_4d_unclamped"] is None


def test_4d_unclamped_better_when_4d_estimate_matches_actual():
    """When 4D solar accurately predicts the smaller real impact, the
    unclamped 4D residual beats the 3D clamp by more than 20% → verdict
    flips to ``4d_unclamped_meaningfully_better``.
    """
    log = []
    # 5 saturation events: 3D over-estimates solar (0.30 raw, 0.05 base
    # → wasted 0.25, expected clamps to 0).  4D estimates correctly
    # (solar=0.04 ≈ matches actual demand structure).
    # sum_base = expected + solar_eff = 0 + 0.05 = 0.05
    # expected_4d_unclamped = 0.05 - 0.04 = 0.01
    # actual = 0.10 → residual_4d_unclamped = |0.10 - 0.01| = 0.09
    # residual_3d_clamped = 0.10
    # ratio improvement: (0.10 - 0.09) / 0.10 = 0.10 → not enough
    # Adjust to make 4D clearly better:
    # expected_4d_unclamped = 0.05 - 0.04 = 0.01, actual = 0.02
    # residual_4d_unclamped = 0.01, residual_3d_clamped = 0.02 → 50% better
    for i in range(5):
        log.append(_entry(
            _ts(i), temp=15.0, actual=0.02, expected_3d=0.0,
            solar_eff=0.05, solar_wasted=0.25,
            solar_4d=0.04,
        ))
    out = DiagnosticsEngine(_coord(log, balance_point=17.0))\
        ._compute_shoulder_saturation_blast_radius(30)

    assert out["available"] is True
    assert out["n_4d_tagged_hours"] == 5
    assert out["verdict"] == "4d_unclamped_meaningfully_better"


def test_replay_populates_4d_when_logged_field_missing():
    """When `solar_impact_4d_kwh` is not logged but DNI/DHI are present,
    the post-mortem replay synthesises a 4D estimate by calling the live
    pipeline (`get_approx_sun_pos` + `calculate_unit_potential_4d`).
    This keeps the 4D variants populated during the shadow learner's
    rollout window instead of waiting weeks for native tagging.
    """
    # Mock SolarCalculator: sun_pos returns (elev=30, az=200) for any
    # timestamp; potential_4d returns a fixed vector.  Coefficients are
    # picked so that 0.10 + 0.20 + 0.05 + 0.10 = 0.45 (per-unit), times
    # one entity = 0.45 whole-house.
    from custom_components.heating_analytics.solar import SolarCalculator
    solar = MagicMock(spec=SolarCalculator)
    solar.get_approx_sun_pos = MagicMock(return_value=(30.0, 200.0))
    solar.calculate_unit_potential_4d = MagicMock(
        return_value=(1.0, 2.0, 0.5, 1.0)
    )
    coeffs_4d = {
        "sensor.heater1": {
            "heating": {"s": 0.10, "e": 0.10, "w": 0.10, "diffuse": 0.10},
        },
    }
    log = []
    for i in range(5):
        e = _entry(
            _ts(i), temp=15.0, actual=0.20, expected_3d=0.0,
            solar_eff=0.05, solar_wasted=0.20,
        )
        # Add DNI/DHI so resolve_dni_dhi takes the "native" branch.
        e["dni"] = 600.0
        e["dhi"] = 200.0
        e["unit_modes"] = {"sensor.heater1": "heating"}
        e["correction_percent"] = 0.0
        log.append(e)

    coord = _coord(
        log, balance_point=17.0, solar=solar,
        solar_coefficients_4d_per_unit=coeffs_4d,
        energy_sensors=["sensor.heater1"],
    )
    out = DiagnosticsEngine(coord)\
        ._compute_shoulder_saturation_blast_radius(30)

    assert out["available"] is True
    assert out["n_4d_tagged_hours"] == 0
    assert out["n_4d_replayed_hours"] == 5
    assert out["n_4d_unavailable_hours"] == 0

    # Per-unit replay total: (1*0.1 + 2*0.1 + 0.5*0.1 + 1*0.1) = 0.45
    # sum_base = expected + solar_eff = 0 + 0.05 = 0.05
    # expected_4d_unclamped = 0.05 - 0.45 = -0.40
    # expected_4d_clamped = max(0, -0.40) = 0
    # actual = 0.20 → resid_4d_clamped = 0.20, resid_4d_unclamped = 0.60
    sat = out["median_abs_residual"]["saturation_events_only"]
    assert sat["expected_4d_clamped"] == 0.20
    assert abs(sat["expected_4d_unclamped"] - 0.60) < 1e-3


def test_replay_skips_hour_when_no_dni_no_cloud():
    """No DNI/DHI and no cloud_coverage → replay returns None and the
    hour is counted as 4D-unavailable (does not contribute to 4D sums).
    """
    from custom_components.heating_analytics.solar import SolarCalculator
    solar = MagicMock(spec=SolarCalculator)
    solar.get_approx_sun_pos = MagicMock(return_value=(30.0, 200.0))
    log = []
    for i in range(5):
        e = _entry(
            _ts(i), temp=15.0, actual=0.20, expected_3d=0.0,
            solar_eff=0.05, solar_wasted=0.20,
        )
        # No dni, no dhi, no cloud_coverage → resolve_dni_dhi returns "none"
        log.append(e)
    coord = _coord(
        log, balance_point=17.0, solar=solar,
        energy_sensors=["sensor.heater1"],
    )
    out = DiagnosticsEngine(coord)\
        ._compute_shoulder_saturation_blast_radius(30)

    assert out["n_4d_tagged_hours"] == 0
    assert out["n_4d_replayed_hours"] == 0
    assert out["n_4d_unavailable_hours"] == 5
    sat = out["median_abs_residual"]["saturation_events_only"]
    assert sat["expected_4d_clamped"] is None
    assert sat["expected_4d_unclamped"] is None


def test_filters_to_shoulder_window():
    """Hours outside [BP-3, BP+1] are excluded."""
    log = [
        _entry(_ts(0), temp=2.0, actual=5.0, expected_3d=4.5, solar_eff=0.0),  # cold
        _entry(_ts(1), temp=14.0, actual=0.5, expected_3d=0.4, solar_eff=0.0),  # in window
        _entry(_ts(2), temp=15.0, actual=0.5, expected_3d=0.4, solar_eff=0.0),  # in window
        _entry(_ts(3), temp=16.0, actual=0.5, expected_3d=0.4, solar_eff=0.0),  # in window
        _entry(_ts(4), temp=17.0, actual=0.5, expected_3d=0.4, solar_eff=0.0),  # in window
        _entry(_ts(5), temp=18.0, actual=0.5, expected_3d=0.4, solar_eff=0.0),  # in window
        _entry(_ts(6), temp=22.0, actual=0.0, expected_3d=0.0, solar_eff=0.0),  # warm
    ]
    out = DiagnosticsEngine(_coord(log, balance_point=17.0))\
        ._compute_shoulder_saturation_blast_radius(30)
    assert out["available"] is True
    assert out["n_shoulder_hours"] == 5  # 14, 15, 16, 17, 18
    assert out["n_saturation_events"] == 0  # no wasted in any
