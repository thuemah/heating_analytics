"""Tests for 4D shadow NLMS solar coefficient learner (#954 commit 6).

The 4D learner mirrors the 3D NLMS path with an added diffuse
component.  It is strictly shadow — writes only to
``solar_coefficients_4d_per_unit``; no read-path consumer yet.
"""
from __future__ import annotations

import pytest

from custom_components.heating_analytics.learning import LearningManager
from custom_components.heating_analytics.const import (
    COLD_START_SOLAR_DAMPING,
    LEARNING_BUFFER_THRESHOLD,
    MODE_HEATING,
    NLMS_STEP_SIZE,
)


# Cold-start buffer threshold used by the 4D learner.
BUFFER_THRESHOLD_4D = max(LEARNING_BUFFER_THRESHOLD, 5)


def _feed(
    manager: LearningManager,
    entity_id: str,
    true_coeffs: tuple[float, float, float, float],
    samples: list[tuple[float, float, float, float]],
    coeffs_4d: dict,
    buffers_4d: dict,
    base_kwh: float = 2.0,
) -> None:
    """Drive the 4D learner with samples whose actual_impact follows truth."""
    ts, te, tw, td = true_coeffs
    for s, e, w, d in samples:
        true_impact = ts * s + te * e + tw * w + td * d
        actual = max(0.0, base_kwh - true_impact)
        manager._learn_unit_solar_coefficient(
            entity_id=entity_id,
            temp_key="0",
            expected_unit_base=base_kwh,
            actual_unit=actual,
            avg_solar_vector=(s, e, w, d),
            learning_rate=0.0,
            solar_coefficients_per_unit=coeffs_4d,
            learning_buffer_solar_per_unit=buffers_4d,
            avg_temp=0.0,
            balance_point=0.0,
            unit_mode=MODE_HEATING,
            components=("s", "e", "w", "diffuse"),
        )


def test_4d_cold_start_writes_after_buffer():
    """Diverse 4D samples populate the 4D dict via 4x4 LS jump-start."""
    manager = LearningManager()
    coeffs_4d: dict = {}
    buffers_4d: dict = {}

    # Five linearly-independent (s, e, w, d) samples covering each axis.
    # True coefficients: (0.4, 0.2, 0.1, 0.05).
    samples = [
        (1.0, 0.0, 0.0, 0.5),
        (0.0, 1.0, 0.0, 0.4),
        (0.0, 0.0, 1.0, 0.3),
        (0.5, 0.5, 0.0, 0.4),
        (0.2, 0.2, 0.2, 0.6),
    ]
    _feed(manager, "u1", (0.4, 0.2, 0.1, 0.05), samples, coeffs_4d, buffers_4d)

    assert "u1" in coeffs_4d
    h = coeffs_4d["u1"]["heating"]
    assert h.get("learned") is True
    # Damped by COLD_START_SOLAR_DAMPING; verify within tolerance.
    expected = {
        "s": 0.4 * COLD_START_SOLAR_DAMPING,
        "e": 0.2 * COLD_START_SOLAR_DAMPING,
        "w": 0.1 * COLD_START_SOLAR_DAMPING,
        "diffuse": 0.05 * COLD_START_SOLAR_DAMPING,
    }
    for k, v in expected.items():
        assert abs(h[k] - v) < 0.1, f"{k}: {h[k]} not within 0.1 of {v}"


def test_4d_nlms_step_after_seed():
    """Pre-seeded 4D coeff updates via NLMS in proportion to step size."""
    manager = LearningManager()
    # Seed the heating regime directly to bypass cold-start.
    coeffs_4d = {
        "u_seed": {
            "heating": {"s": 0.5, "e": 0.0, "w": 0.0, "diffuse": 0.0, "learned": True},
            "cooling": {"s": 0.0, "e": 0.0, "w": 0.0, "diffuse": 0.0},
        }
    }
    buffers_4d: dict = {}

    # Single sample where true impact uses c_s = 1.0 (so error = +0.5*s).
    s, e, w, d = (0.8, 0.0, 0.0, 0.0)
    base = 3.0
    # true coefficient is 1.0 -> true_impact = 0.8 -> actual = 2.2
    true_impact = 1.0 * s
    actual = base - true_impact
    manager._learn_unit_solar_coefficient(
        entity_id="u_seed",
        temp_key="0",
        expected_unit_base=base,
        actual_unit=actual,
        avg_solar_vector=(s, e, w, d),
        learning_rate=0.0,
        solar_coefficients_per_unit=coeffs_4d,
        learning_buffer_solar_per_unit=buffers_4d,
        avg_temp=0.0,
        balance_point=0.0,
        unit_mode=MODE_HEATING,
        components=("s", "e", "w", "diffuse"),
    )

    # Pre-update s = 0.5, predicted = 0.5*0.8 = 0.4, error = 0.4.
    # step = NLMS_STEP_SIZE * 0.4 / (0.64 + NLMS_REGULARIZATION)
    # delta_s = step * 0.8
    from custom_components.heating_analytics.const import NLMS_REGULARIZATION
    expected_step = NLMS_STEP_SIZE * 0.4 / (0.64 + NLMS_REGULARIZATION)
    expected_new_s = 0.5 + expected_step * 0.8
    assert abs(coeffs_4d["u_seed"]["heating"]["s"] - expected_new_s) < 1e-4
    # Other components untouched (their potential was zero).
    assert coeffs_4d["u_seed"]["heating"]["e"] == 0.0
    assert coeffs_4d["u_seed"]["heating"]["w"] == 0.0
    assert coeffs_4d["u_seed"]["heating"]["diffuse"] == 0.0


def test_4d_singular_buffer_returns_silently():
    """Diffuse always zero -> 4x4 LS singular -> learner returns w/o write."""
    manager = LearningManager()
    coeffs_4d: dict = {}
    buffers_4d: dict = {}

    samples = [
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.5, 0.5, 0.0, 0.0),
        (0.2, 0.2, 0.2, 0.0),
    ]
    _feed(manager, "u_sing", (0.4, 0.2, 0.1, 0.0), samples, coeffs_4d, buffers_4d)

    # Singular matrix -> learner skipped the write but preserved the buffer
    # so it can keep collecting once sun-diversity unlocks the system.
    assert "u_sing" not in coeffs_4d
    assert "u_sing" in buffers_4d
    assert len(buffers_4d["u_sing"]["heating"]) >= BUFFER_THRESHOLD_4D


def test_4d_shadow_no_3d_side_effect():
    """3D coefficient state must be identical with or without the 4D shadow."""
    # Drive the SAME hour sequence through process_learning twice;
    # once with the 4D dicts plumbed, once without.  Compare 3D state.
    # Here we simulate by calling _learn_unit_solar_coefficient (3D)
    # in two parallel runs — the 4D learner only writes to the 4D dict,
    # so the 3D coeff must match bit-for-bit between runs.
    from custom_components.heating_analytics.const import LEARNING_BUFFER_THRESHOLD

    samples = [(0.6, 0.2, 0.05), (0.4, 0.3, 0.0), (0.5, 0.25, 0.05), (0.7, 0.15, 0.0), (0.55, 0.2, 0.05)]
    base = 2.0
    true_s, true_e = 1.0, 0.3

    def run_3d_only():
        m = LearningManager()
        c3d: dict = {}
        b3d: dict = {}
        for (s, e, w) in samples:
            ti = true_s * s + true_e * e
            actual = max(0.0, base - ti)
            m._learn_unit_solar_coefficient(
                entity_id="u_par",
                temp_key="5",
                expected_unit_base=base,
                actual_unit=actual,
                avg_solar_vector=(s, e, w),
                learning_rate=0.01,
                solar_coefficients_per_unit=c3d,
                learning_buffer_solar_per_unit=b3d,
                avg_temp=5.0,
                balance_point=15.0,
                unit_mode=MODE_HEATING,
            )
        return c3d

    def run_3d_and_4d():
        m = LearningManager()
        c3d: dict = {}
        b3d: dict = {}
        c4d: dict = {}
        b4d: dict = {}
        for (s, e, w) in samples:
            ti = true_s * s + true_e * e
            actual = max(0.0, base - ti)
            m._learn_unit_solar_coefficient(
                entity_id="u_par",
                temp_key="5",
                expected_unit_base=base,
                actual_unit=actual,
                avg_solar_vector=(s, e, w),
                learning_rate=0.01,
                solar_coefficients_per_unit=c3d,
                learning_buffer_solar_per_unit=b3d,
                avg_temp=5.0,
                balance_point=15.0,
                unit_mode=MODE_HEATING,
            )
            m._learn_unit_solar_coefficient(
                entity_id="u_par",
                temp_key="5",
                expected_unit_base=base,
                actual_unit=actual,
                avg_solar_vector=(s, e, w, 0.1),
                learning_rate=0.01,
                solar_coefficients_per_unit=c4d,
                learning_buffer_solar_per_unit=b4d,
                avg_temp=5.0,
                balance_point=15.0,
                unit_mode=MODE_HEATING,
                components=("s", "e", "w", "diffuse"),
            )
        return c3d

    only_3d = run_3d_only()
    with_4d = run_3d_and_4d()
    assert only_3d == with_4d, (
        f"3D state diverged when 4D shadow ran in parallel:\n"
        f"  3d-only: {only_3d}\n"
        f"  with-4d: {with_4d}"
    )
