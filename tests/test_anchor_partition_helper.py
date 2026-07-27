"""Property tests for the shared anchor-allocation helper.

``allocate_anchor_partitions`` is the arithmetic both
``calculate_total_power`` (3D) and ``calculate_total_power_4d`` share.
It is pure — no coordinator, no model state — so the Kelvin Protocol
anchor invariants (CLAUDE.md Component Boundary Invariant #8) can be
asserted directly here rather than inferred through a full prediction
call with mocks.
"""

import pytest

from custom_components.heating_analytics.statistics import (
    allocate_anchor_partitions,
)


def _call(**overrides):
    """Invoke the helper with a sane baseline, overriding named fields."""
    kwargs = {
        "global_base": 10.0,
        "global_aux_reduction": 0.0,
        "global_solar_effect": 0.0,
        "unit_sum_base": 8.0,
        "sum_base_aux_affected": 0.0,
        "base_heating_in_scope": 5.0,
        "base_cooling_in_scope": 2.0,
        "base_not_in_scope": 1.0,
        "sum_applied_heating": 0.0,
        "sum_applied_cooling": 0.0,
    }
    kwargs.update(overrides)
    return allocate_anchor_partitions(**kwargs)


def test_pre_solar_partition_sum_equals_anchor_exactly():
    """Shares sum to 1, so with no solar the result IS the anchor.

    This is the property that makes the per-partition split safe: the
    allocation redistributes the global anchor, it never changes its
    magnitude.
    """
    global_net, applied_eff, clip = _call(global_base=10.0)

    assert global_net == pytest.approx(10.0)
    assert applied_eff == 0.0
    assert clip == 0.0


def test_anchor_is_global_base_minus_scoped_aux():
    """Aux reduces the anchor before allocation."""
    global_net, _, _ = _call(
        global_base=10.0,
        global_aux_reduction=3.0,
        unit_sum_base=8.0,
        sum_base_aux_affected=8.0,  # every unit aux-affected -> frac = 1.0
    )

    assert global_net == pytest.approx(7.0)


def test_aux_clamp_ceiling_is_expressed_in_anchor_scale():
    """#1035 follow-up bug 1: the ceiling scales with the anchor.

    ``global_base`` (10) sits above ``unit_sum_base`` (5) — the
    aux-active / partially-learned case.  Half the per-unit base is
    aux-affected, so the ceiling must be ``global_base x 0.5 = 5.0``,
    not the raw per-unit sum ``2.5``.  A 4.0 kWh global aux belief is
    below the anchor-scale ceiling and must therefore pass through
    whole; clamping it against the raw per-unit sum would discard
    1.5 kWh of valid aux savings and over-predict by that much.
    """
    global_net, _, _ = _call(
        global_base=10.0,
        global_aux_reduction=4.0,
        unit_sum_base=5.0,
        sum_base_aux_affected=2.5,
    )

    # Anchor-scale ceiling: min(4.0, 10.0 * 0.5) = 4.0 -> anchor 6.0.
    assert global_net == pytest.approx(6.0)
    # The per-unit-scale ceiling would have been min(4.0, 2.5) = 2.5,
    # leaving an anchor of 7.5 — the over-clamp this fix removes.
    assert global_net != pytest.approx(7.5)


def test_aux_clamp_still_protects_out_of_scope_base():
    """The clamp is not a no-op: an oversized aux belief is capped.

    Only a quarter of base is aux-affected, so a 9.0 kWh global aux
    reduction cannot delete the other three quarters.
    """
    global_net, _, _ = _call(
        global_base=10.0,
        global_aux_reduction=9.0,
        unit_sum_base=8.0,
        sum_base_aux_affected=2.0,  # frac = 0.25 -> ceiling 2.5
    )

    assert global_net == pytest.approx(7.5)


def test_unclamped_heating_reduces_to_anchor_plus_solar_effect():
    """With no clipping the partitioned result equals the legacy form.

    ``anchor + global_solar_effect`` is what the cold-start fallback
    computes; the partition split must agree with it whenever solar
    does not exceed its allocation.
    """
    global_net, applied_eff, clip = _call(
        global_base=10.0,
        sum_applied_heating=1.0,
        sum_applied_cooling=0.5,
        global_solar_effect=0.5 - 1.0,
    )

    assert clip == 0.0
    assert applied_eff == pytest.approx(1.0)
    assert global_net == pytest.approx(10.0 - 0.5)


def test_clip_conserves_solar_energy():
    """#1035 follow-up bug 2: excess applied is reclassified, not lost.

    Heating solar is saturated against the raw per-unit base upstream;
    when the anchor sits below that sum the excess must move from
    applied to wasted so the learning / battery aggregates stay
    energy-consistent.
    """
    sum_applied_heating = 20.0
    global_net, applied_eff, clip = _call(
        global_base=4.0,  # anchor well below the per-unit sum
        sum_applied_heating=sum_applied_heating,
    )

    assert applied_eff + clip == pytest.approx(sum_applied_heating)
    assert clip > 0.0
    # Applied can never exceed the heating partition's allocation.
    alloc_heating = 4.0 * (5.0 / 8.0)
    assert applied_eff == pytest.approx(alloc_heating)


def test_cold_start_fallback_matches_legacy_track_a_formula():
    """Empty partitions fall back to pre-#992 behaviour bit-identically."""
    global_net, applied_eff, clip = _call(
        global_base=10.0,
        global_aux_reduction=2.0,
        global_solar_effect=-1.0,
        base_heating_in_scope=0.0,
        base_cooling_in_scope=0.0,
        base_not_in_scope=0.0,
        sum_applied_heating=1.0,
    )

    assert global_net == pytest.approx(max(0.0, (10.0 - 2.0) + -1.0))
    # No anchor scaling on this path, so nothing is reclassified.
    assert applied_eff == pytest.approx(1.0)
    assert clip == 0.0


def test_cold_start_fallback_clamps_at_zero():
    """Solar larger than the aux-reduced base cannot go negative."""
    global_net, _, _ = _call(
        global_base=1.0,
        global_solar_effect=-5.0,
        base_heating_in_scope=0.0,
        base_cooling_in_scope=0.0,
        base_not_in_scope=0.0,
    )

    assert global_net == 0.0


def test_zero_unit_sum_base_does_not_divide_by_zero():
    """``unit_sum_base = 0`` with populated partitions degrades safely."""
    global_net, _, _ = _call(
        global_base=10.0,
        global_aux_reduction=3.0,
        unit_sum_base=0.0,
        sum_base_aux_affected=1.0,
    )

    # frac_aux falls back to 0.0, so no aux is subtracted.
    assert global_net == pytest.approx(10.0)


def test_cooling_solar_is_additive():
    """Cooling solar raises demand; heating solar lowers it."""
    heating_net, _, _ = _call(global_base=10.0, sum_applied_heating=2.0)
    cooling_net, _, _ = _call(global_base=10.0, sum_applied_cooling=2.0)

    assert heating_net == pytest.approx(8.0)
    assert cooling_net == pytest.approx(12.0)
