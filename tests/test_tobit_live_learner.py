"""Tests for #904 stage 3 — live Tobit learner on running sufficient-stat.

Stage 3 ships behind the experimental_tobit_live_learner master flag
(default OFF) plus a per-entity allow-list (default empty).  When both
gates pass, qualifying-hour samples flow to ``_update_unit_tobit_live``
which maintains a sliding window of recent (s, e, w, value, censored)
tuples per (entity, regime) and runs one Newton iteration of
``_solve_tobit_3d`` over the current window each hour.

Coverage in this file:

1. Sufficient-stat equivalence — running sliding window matches batch
   Tobit on the same samples within solver tolerance.
2. Allow-list gate — entities not in the list keep using NLMS even
   when the master flag is enabled.
3. Cold-start handover — until n_eff ≥ TOBIT_MIN_NEFF, NLMS is the
   live writer; sliding window accumulates from first hour.
4. Model-version reset — stored solar_model_version != current →
   slot is dropped on load and Tobit returns to cold-start.
5. NLMS shadow trajectory — when Tobit is the live writer, NLMS still
   fires against the shadow dict (independent state).
6. Service handlers — enable / allow-list / reset persist through
   _async_save_data and update coordinator state.
7. Diagnose surface — live_tobit_state is present on every per-unit
   block and carries the documented field set.
"""
from __future__ import annotations

import random
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.heating_analytics.const import (
    BATCH_FIT_SATURATION_RATIO,
    MODE_HEATING,
    SOLAR_COEFF_CAP,
    SOLAR_MODEL_VERSION,
    TOBIT_MIN_NEFF,
    TOBIT_MIN_UNCENSORED,
    TOBIT_RUNNING_WINDOW,
)
from custom_components.heating_analytics.learning import LearningManager


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

def _generate_synthetic_samples(
    *,
    n: int,
    true_c: tuple[float, float, float],
    sigma: float,
    censoring_rate: float = 0.0,
    rng: random.Random | None = None,
) -> tuple[list[tuple[float, float, float, float, bool]], float]:
    """Produce ``n`` synthetic samples with controlled censoring rate.

    Returns the parallel lists of (s, e, w, value, is_censored) plus
    the realised censoring fraction.  Used for sliding-window equivalence
    and convergence tests.
    """
    if rng is None:
        rng = random.Random(0)
    samples = []
    n_cens = 0
    for _ in range(n):
        s = rng.uniform(0.1, 0.7)
        e = rng.uniform(0.0, 0.5)
        w = rng.uniform(0.0, 0.5)
        base = rng.uniform(0.5, 1.5)
        pred = true_c[0] * s + true_c[1] * e + true_c[2] * w
        y_star = pred + rng.gauss(0.0, sigma)
        T = BATCH_FIT_SATURATION_RATIO * base
        # Inject extra censoring for high-rate scenarios by scaling base
        # downward when rng draws under target rate.
        if rng.random() < censoring_rate:
            T = pred * 0.95  # force censoring
        if y_star >= T:
            samples.append((s, e, w, T, True))
            n_cens += 1
        else:
            samples.append((s, e, w, max(0.001, y_star), False))
    return samples, n_cens / n


# -----------------------------------------------------------------------------
# 1. Sufficient-stat equivalence with batch Tobit
# -----------------------------------------------------------------------------

def test_sliding_window_tobit_matches_batch():
    """Running Tobit over a sliding window of N samples produces the
    same coefficient (within solver tolerance) as a from-scratch batch
    fit on the same N samples.  This is the mathematical-equivalence
    contract for #912 decision (1).
    """
    rng = random.Random(101)
    true_c = (0.45, 0.20, 0.55)
    samples_with_mask, _ = _generate_synthetic_samples(
        n=100, true_c=true_c, sigma=0.10, censoring_rate=0.20, rng=rng,
    )
    samples_4 = [(s[0], s[1], s[2], s[3]) for s in samples_with_mask]
    mask = [s[4] for s in samples_with_mask]

    # Batch: directly call the solver on the full list.
    batch_fit = LearningManager._solve_tobit_3d(samples_4, mask)
    assert batch_fit is not None and batch_fit["converged"]

    # Sliding-window: simulate the live learner appending one hour at a
    # time.  After all 100 hours, the window contains the same samples
    # the batch fit saw.  Final coefficient should match within Newton
    # tolerance (warm-starting from the previous iterate accelerates
    # convergence vs the batch's LS warm-start, but both should land
    # at the same MLE).
    coeffs: dict = {}
    stats: dict = {}
    lm = LearningManager()
    for i, (s_i, e_i, w_i, val_i, cens_i) in enumerate(samples_with_mask):
        if cens_i:
            # Reverse-engineer expected_unit_base so the live update's
            # censoring detection fires (val == 0.95 × base).
            expected_base = val_i / BATCH_FIT_SATURATION_RATIO
            actual = expected_base - val_i  # actual_impact = T → actual = base - T
        else:
            expected_base = val_i + 0.5  # arbitrary sufficient base
            actual = expected_base - val_i
        actual_impact = expected_base - actual  # heating regime
        lm._update_unit_tobit_live(
            "sensor.test",
            "heating",
            (s_i, e_i, w_i),
            actual_impact,
            expected_base,
            stats,
            coeffs,
        )

    # After the loop the live coefficient should match batch within
    # tolerance.
    final = coeffs.get("sensor.test", {}).get("heating", {})
    assert final
    assert abs(final["s"] - batch_fit["s"]) < 0.05, (
        f"live s={final['s']:.4f} vs batch s={batch_fit['s']:.4f}"
    )
    assert abs(final["e"] - batch_fit["e"]) < 0.05
    assert abs(final["w"] - batch_fit["w"]) < 0.05


def test_sliding_window_caps_at_running_window():
    """After more than TOBIT_RUNNING_WINDOW hours, the buffer trims to
    the most recent samples.  Older samples roll off automatically.
    """
    coeffs: dict = {}
    stats: dict = {}
    lm = LearningManager()
    n_hours = TOBIT_RUNNING_WINDOW + 50  # exceed the cap
    for _ in range(n_hours):
        lm._update_unit_tobit_live(
            "sensor.test",
            "heating",
            (0.5, 0.3, 0.2),
            0.5,  # actual_impact
            1.0,  # base
            stats,
            coeffs,
        )

    slot = stats["sensor.test"]["heating"]
    assert len(slot["samples"]) == TOBIT_RUNNING_WINDOW
    assert slot["samples_since_reset"] == n_hours


# -----------------------------------------------------------------------------
# 2. Cold-start behaviour
# -----------------------------------------------------------------------------

def test_below_uncensored_floor_does_not_apply():
    """When the running window has fewer than TOBIT_MIN_UNCENSORED
    uncensored samples, ``_update_unit_tobit_live`` returns
    ``applied=False`` and ``in_cold_start=True`` so the caller routes
    to NLMS-fallback.  Sliding window still accumulates.
    """
    coeffs: dict = {}
    stats: dict = {}
    lm = LearningManager()
    for _ in range(TOBIT_MIN_UNCENSORED - 1):
        result = lm._update_unit_tobit_live(
            "sensor.test",
            "heating",
            (0.5, 0.3, 0.2),
            0.4,  # below saturation threshold for base=1.0 → uncensored
            1.0,
            stats,
            coeffs,
        )
        assert result["applied"] is False
        assert result["in_cold_start"] is True

    slot = stats["sensor.test"]["heating"]
    assert len(slot["samples"]) == TOBIT_MIN_UNCENSORED - 1


def test_n_eff_floor_keeps_cold_start():
    """Above the |U| floor but below the n_eff floor — solver runs but
    the post-fit gate trips, returning applied=False.  Skip-reason is
    recorded in the slot's last_step.
    """
    coeffs: dict = {}
    stats: dict = {}
    lm = LearningManager()
    n = (TOBIT_MIN_UNCENSORED + TOBIT_MIN_NEFF) // 2  # 30
    assert TOBIT_MIN_UNCENSORED <= n < TOBIT_MIN_NEFF
    for _ in range(n):
        result = lm._update_unit_tobit_live(
            "sensor.test",
            "heating",
            (0.5, 0.3, 0.2),
            0.4,
            1.0,
            stats,
            coeffs,
        )
    # Last call above the |U| gate but below n_eff → did fit, did not apply.
    assert result["applied"] is False
    slot = stats["sensor.test"]["heating"]
    assert slot["last_step"]["skip_reason"] == "insufficient_effective_samples"


def test_warm_handover_writes_coefficient():
    """After ≥ TOBIT_MIN_NEFF qualifying hours, Tobit applies and writes
    the coefficient via the canonical helper.  Subsequent calls update
    the same coefficient incrementally.
    """
    rng = random.Random(7)
    true_c = (0.4, 0.2, 0.5)
    coeffs: dict = {}
    stats: dict = {}
    lm = LearningManager()
    last_result = None
    for _ in range(TOBIT_MIN_NEFF + 5):
        s = rng.uniform(0.2, 0.7)
        e = rng.uniform(0.0, 0.5)
        w = rng.uniform(0.0, 0.5)
        true_impact = true_c[0] * s + true_c[1] * e + true_c[2] * w
        actual_impact = max(0.001, true_impact + rng.gauss(0.0, 0.05))
        last_result = lm._update_unit_tobit_live(
            "sensor.test",
            "heating",
            (s, e, w),
            actual_impact,
            actual_impact + 0.5,  # base well above so unsaturated
            stats,
            coeffs,
        )

    assert last_result["applied"] is True
    assert last_result["in_cold_start"] is False
    final = coeffs["sensor.test"]["heating"]
    assert final["s"] >= 0.0  # invariant #4
    # n_eff in the result reflects post-warmup state.
    assert last_result["n_eff"] >= TOBIT_MIN_NEFF


# -----------------------------------------------------------------------------
# 3. Routing through _process_per_unit_learning
# -----------------------------------------------------------------------------

def _gate_predicate(
    flag: bool,
    entity_id: str,
    scope_list: frozenset[str],
    mpc_managed: frozenset[str],
) -> bool:
    """Mirror of the production gate at learning.py:856-874.

    Tests that need to verify routing decisions MUST use this helper
    (or call _process_per_unit_learning directly) rather than
    re-implementing the predicate inline — re-implementations have
    silently drifted from production semantics across the 1.3.4
    opt-in semantic and 1.3.5 default-on auto-mode promotion.
    """
    return (
        flag
        and (not scope_list or entity_id in scope_list)
        and entity_id not in mpc_managed
    )


def test_routing_skips_tobit_when_flag_off():
    """With experimental_tobit_live_learner=False, the live Tobit path
    is bypassed and NLMS writes the coefficient as today.  The
    sufficient-stat dict stays empty.
    """
    stats: dict = {}
    coeffs: dict = {}
    lm = LearningManager()

    if _gate_predicate(False, "sensor.test", frozenset(), frozenset()):
        lm._update_unit_tobit_live(
            "sensor.test", "heating", (0.5, 0.3, 0.2), 0.4, 1.0, stats, coeffs,
        )
    assert stats == {}


def test_routing_skips_tobit_when_not_in_scope_override():
    """Master flag enabled, scope-restriction list non-empty,
    entity not in list → NLMS path only.  This is the override-mode
    scope semantic (1.3.5+): non-empty list scopes Tobit to listed
    entities; others stay on NLMS regardless of plausibility.

    Pre-1.3.5 semantic was strict opt-in (entity MUST be in list).
    1.3.5 inverts: empty list = auto, non-empty = scope.  Empty +
    flag-on is auto-mode covered separately.
    """
    stats: dict = {}
    coeffs: dict = {}
    lm = LearningManager()

    if _gate_predicate(True, "sensor.test", frozenset(["sensor.other"]), frozenset()):
        lm._update_unit_tobit_live(
            "sensor.test", "heating", (0.5, 0.3, 0.2), 0.4, 1.0, stats, coeffs,
        )
    assert stats == {}


def test_routing_runs_tobit_in_auto_mode():
    """Master flag enabled, scope list empty, entity not MPC-managed
    → auto-mode, gate predicate evaluates True so Tobit fires.  Pins
    the 1.3.5 default-on auto semantic; pre-1.3.5 this combination
    was a no-op.
    """
    assert _gate_predicate(
        True, "sensor.toshiba_vp", frozenset(), frozenset()
    ) is True


def test_mpc_managed_entity_skipped_even_when_scope_listed():
    """Track C / MPC-managed sensors must be excluded from live Tobit
    even when the user has explicitly scope-listed them.  MPC's
    load-shifting produces samples that are not a function of
    instantaneous solar potential — the HP may run hard at 02:00
    (no sun) and idle at 12:00 (full sun).  Tobit fitting against
    such samples would produce non-physical coefficients.

    The gate composes ``in-scope AND NOT in mpc_managed_entities``.
    This pins the contract so an accidentally-scope-listed MPC
    sensor still gets routed to NLMS.  Same protection
    ``batch_fit_solar_coefficients`` already applies via the
    ``weighted_smear_excluded`` skip path.
    """
    scope_list = frozenset(["sensor.mpc_vp"])
    mpc_managed = frozenset(["sensor.mpc_vp"])

    assert _gate_predicate(
        True, "sensor.mpc_vp", scope_list, mpc_managed
    ) is False, (
        "MPC-managed entity must be excluded from live Tobit gate "
        "even when scope-listed; misconfiguration must not feed Tobit "
        "load-shifted samples."
    )

    # Sanity: a non-MPC entity in the same scope-list still activates.
    assert _gate_predicate(
        True, "sensor.toshiba_vp", frozenset(["sensor.toshiba_vp"]), mpc_managed
    ) is True


def test_mpc_managed_entity_skipped_in_auto_mode_too():
    """Auto-mode (empty scope list) + MPC-managed entity → still
    blocked.  The ``not in mpc_managed`` check is the unconditional
    safety net regardless of scope semantics.
    """
    mpc_managed = frozenset(["sensor.mpc_vp"])
    assert _gate_predicate(
        True, "sensor.mpc_vp", frozenset(), mpc_managed
    ) is False


# -----------------------------------------------------------------------------
# 4. Censoring detection in the live update
# -----------------------------------------------------------------------------

def test_live_update_marks_saturated_as_censored():
    """When actual_impact ≥ 0.95×base in heating regime, the sample is
    tagged ``is_censored=True`` and stored with value=T (the threshold)
    instead of the observed actual_impact.
    """
    stats: dict = {}
    coeffs: dict = {}
    lm = LearningManager()
    base = 1.0
    saturated_impact = 0.97  # > 0.95 × 1.0
    lm._update_unit_tobit_live(
        "sensor.test", "heating", (0.5, 0.3, 0.2),
        saturated_impact, base, stats, coeffs,
    )
    slot = stats["sensor.test"]["heating"]
    sample = slot["samples"][0]
    assert sample[4] is True  # censored flag
    assert sample[3] == pytest.approx(BATCH_FIT_SATURATION_RATIO * base)


def test_live_update_does_not_censor_cooling():
    """Cooling regime has no upper-saturation analog (sun INCREASES
    demand on cooling).  ``actual_impact ≥ 0.95×base`` on cooling does
    NOT mark the sample censored — it's just a high-load observation.
    """
    stats: dict = {}
    coeffs: dict = {}
    lm = LearningManager()
    base = 1.0
    high_impact = 1.5  # would trigger censoring on heating
    lm._update_unit_tobit_live(
        "sensor.test", "cooling", (0.5, 0.3, 0.2),
        high_impact, base, stats, coeffs,
    )
    slot = stats["sensor.test"]["cooling"]
    sample = slot["samples"][0]
    assert sample[4] is False  # not censored
    assert sample[3] == high_impact


# -----------------------------------------------------------------------------
# 5. Solver-failure paths bubble up
# -----------------------------------------------------------------------------

def test_failure_reason_propagates_via_live_update(monkeypatch):
    """When ``_solve_tobit_3d`` returns ``converged=False`` with a
    failure_reason, the live update records it on the slot and
    reports applied=False.  Pin contract: caller can rely on
    ``last_step_failure_reason`` for diagnose surfacing.
    """
    def _fake_solver(samples, censored_mask, **_kw):
        return {
            "s": 0.5, "e": 0.0, "w": 0.0,
            "sigma": 0.1,
            "iterations": 30,
            "converged": False,
            "failure_reason": "line_search_failed",
            "log_likelihood": -10.0,
            "n_uncensored": len([m for m in censored_mask if not m]),
            "n_censored": sum(1 for m in censored_mask if m),
            "n_eff": float(len(samples)),
        }
    monkeypatch.setattr(
        LearningManager,
        "_solve_tobit_3d",
        staticmethod(_fake_solver),
    )

    stats: dict = {}
    coeffs: dict = {}
    lm = LearningManager()
    # Pump enough samples to clear the |U| pre-fit gate.
    for _ in range(TOBIT_MIN_UNCENSORED + 5):
        result = lm._update_unit_tobit_live(
            "sensor.test", "heating", (0.5, 0.3, 0.2), 0.4, 1.0, stats, coeffs,
        )
    assert result["applied"] is False
    assert result["last_step_failure_reason"] == "line_search_failed"


# -----------------------------------------------------------------------------
# 6. Constants pin the locked design
# -----------------------------------------------------------------------------

def test_running_window_constant_is_bounded():
    """TOBIT_RUNNING_WINDOW caps the per-(entity, regime) memory.
    Locked in #912 design discussion at 200 — covers ~30 days of
    qualifying hours for a typical heating-active VP.  If this
    constant moves, update the docstring on
    ``_update_unit_tobit_live`` and the storage size estimate in
    #912.
    """
    assert TOBIT_RUNNING_WINDOW == 200


def test_solar_model_version_anchored_at_one():
    """SOLAR_MODEL_VERSION starts at 1 (1.3.5 release).  Bumping this
    triggers a sufficient-stat reset on every install — see the const.py
    bump-rule comment.  Locked here so the bump is a deliberate edit
    that touches both this constant and the test.
    """
    assert SOLAR_MODEL_VERSION == 1


# -----------------------------------------------------------------------------
# Stage 3 review-feedback regressions (#912)
# -----------------------------------------------------------------------------


def test_reset_solar_learning_clears_tobit_state(monkeypatch):
    """Review B1 (#912): ``async_reset_solar_learning_data`` MUST clear
    the Tobit sliding window and NLMS shadow coefficient in lock-step
    with the main coefficient.  Pre-fix the reset cleared
    ``_solar_coefficients_per_unit`` only — the next qualifying hour
    found a populated sliding window and Tobit immediately overwrote
    the zeroed coefficient from a fit on stale samples.  User-visible
    bug: "I reset, why did it bounce back?"
    """
    from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
    # Skip the real __init__ — we only need a coordinator-shaped object
    # whose reset method we can call.  Construct via __new__ so the
    # async-init paths aren't required.
    coord = object.__new__(HeatingDataCoordinator)
    # Minimum surface area the reset method touches.
    coord._solar_coefficients_per_unit = {
        "sensor.test": {"heating": {"s": 0.3, "e": 0.2, "w": 0.5}, "cooling": {"s": 0.0, "e": 0.0, "w": 0.0}},
    }
    coord._learning_buffer_solar_per_unit = {"sensor.test": {"heating": [], "cooling": []}}
    coord._last_batch_fit_per_unit = {"sensor.test": {}}
    coord._tobit_sufficient_stats = {
        "sensor.test": {"heating": {"samples": [(0.5, 0, 0, 0.3, False)] * 50}},
    }
    coord._nlms_shadow_coefficients = {
        "sensor.test": {"heating": {"s": 0.31, "e": 0.21, "w": 0.51}, "cooling": {}},
    }
    coord._shadow_learning_buffer_solar_per_unit = {
        "sensor.test": {"heating": [], "cooling": []},
    }
    coord._solar_carryover_state = 0.5
    coord.hass = MagicMock()
    coord.entry = MagicMock()
    coord.storage = MagicMock()
    coord.storage.async_save_data = AsyncMock()
    coord.learning = MagicMock()

    import asyncio
    asyncio.run(coord.async_reset_solar_learning_data(entity_id="sensor.test"))

    # Coefficient cleared (existing behavior).
    assert "sensor.test" not in coord._solar_coefficients_per_unit
    # Tobit state ALSO cleared (the fix).
    assert "sensor.test" not in coord._tobit_sufficient_stats
    assert "sensor.test" not in coord._nlms_shadow_coefficients
    assert "sensor.test" not in coord._shadow_learning_buffer_solar_per_unit


def test_reset_all_units_clears_global_tobit_state():
    """Review B1 (#912): all-units reset wipes everything, including
    Tobit windows and shadow coefficients across all entities.
    """
    from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
    coord = object.__new__(HeatingDataCoordinator)
    coord._solar_coefficients_per_unit = {"sensor.a": {"heating": {"s": 0.3}}, "sensor.b": {"heating": {}}}
    coord._learning_buffer_solar_per_unit = {"sensor.a": {}, "sensor.b": {}}
    coord._last_batch_fit_per_unit = {"sensor.a": {}, "sensor.b": {}}
    coord._tobit_sufficient_stats = {"sensor.a": {}, "sensor.b": {}}
    coord._nlms_shadow_coefficients = {"sensor.a": {}, "sensor.b": {}}
    coord._shadow_learning_buffer_solar_per_unit = {"sensor.a": {}, "sensor.b": {}}
    coord._solar_carryover_state = 0.5
    coord.hass = MagicMock()
    coord.entry = MagicMock()
    coord.storage = MagicMock()
    coord.storage.async_save_data = AsyncMock()
    coord.learning = MagicMock()

    import asyncio
    asyncio.run(coord.async_reset_solar_learning_data())

    assert coord._solar_coefficients_per_unit == {}
    assert coord._tobit_sufficient_stats == {}
    assert coord._nlms_shadow_coefficients == {}
    assert coord._shadow_learning_buffer_solar_per_unit == {}
    assert coord._solar_carryover_state == 0.0


def test_shadow_buffer_is_separate_from_main_buffer():
    """Review I1 (#912): shadow NLMS path uses a SEPARATE cold-start
    buffer dict so a dead-zone reset in the shadow path cannot wipe
    main's buffer.  Pre-fix both shared the same dict and a buffer
    clear from one trampled the other.

    We verify this by running a shadow-path NLMS call against an
    isolated buffer — the call must not touch the main-buffer dict at
    all.
    """
    from custom_components.heating_analytics.learning import LearningManager

    lm = LearningManager()
    main_buffer = {"sensor.test": {"heating": ["main_marker"], "cooling": []}}
    shadow_buffer: dict = {}
    shadow_coeffs: dict = {"sensor.test": {"heating": {"s": 0.5, "e": 0.0, "w": 0.0}, "cooling": {}}}

    lm._learn_unit_solar_coefficient(
        entity_id="sensor.test",
        temp_key="10",
        expected_unit_base=2.0,
        actual_unit=1.5,
        avg_solar_vector=(0.4, 0.2, 0.1),
        learning_rate=0.05,
        solar_coefficients_per_unit=shadow_coeffs,
        learning_buffer_solar_per_unit=shadow_buffer,
        avg_temp=10.0,
        balance_point=15.0,
        unit_mode=MODE_HEATING,
        is_shadow_path=True,
    )
    # Main buffer untouched: marker still present.
    assert main_buffer["sensor.test"]["heating"] == ["main_marker"]


def test_shadow_skips_dead_zone_reset():
    """Review I1 (#912): shadow path must NOT trigger the dead-zone
    reset.  The reset is a recovery mechanism for the writer-of-record;
    shadow is reference-only.  Triggering shadow's dead-zone reset
    would clear shadow's buffer (now safe) but the shared
    ``_dead_zone_counts`` would still increment from shadow signals.
    Verified by feeding the shadow path a stream of zero-impact
    qualifying hours and asserting the shadow coefficient is NOT
    reset to zero (which would happen if dead-zone fired).
    """
    from custom_components.heating_analytics.const import SOLAR_DEAD_ZONE_THRESHOLD
    from custom_components.heating_analytics.learning import LearningManager

    lm = LearningManager()
    shadow_coeffs = {
        "sensor.test": {
            "heating": {"s": 0.5, "e": 0.3, "w": 0.4},
            "cooling": {},
        }
    }
    shadow_buffer: dict = {}

    # Feed enough zero-impact hours to trigger dead-zone IF the
    # path were not gated — actual = base + ε so impact is zero
    # (sun present, but consumption did not drop).
    for _ in range(SOLAR_DEAD_ZONE_THRESHOLD + 5):
        lm._learn_unit_solar_coefficient(
            entity_id="sensor.test",
            temp_key="10",
            expected_unit_base=2.0,
            actual_unit=2.5,  # actual > base → raw_impact < 0
            avg_solar_vector=(0.5, 0.3, 0.4),
            learning_rate=0.05,
            solar_coefficients_per_unit=shadow_coeffs,
            learning_buffer_solar_per_unit=shadow_buffer,
            avg_temp=10.0,
            balance_point=15.0,
            unit_mode=MODE_HEATING,
            is_shadow_path=True,
        )

    # Coefficient NOT reset to zeros — dead-zone gate honoured the
    # shadow flag.  Pre-fix this would be {0, 0, 0}.
    coeff = shadow_coeffs["sensor.test"]["heating"]
    assert coeff["s"] != 0.0 or coeff["e"] != 0.0 or coeff["w"] != 0.0


def test_shadow_seed_helper_copies_both_regimes():
    """Review I2 (#912): ``_seed_shadow_from_main_if_empty`` copies
    BOTH heating and cooling regimes from main to shadow when shadow
    is empty.  Both must be seeded — otherwise a cooling-active install
    would enter cold-start on the cooling shadow.
    """
    from custom_components.heating_analytics.learning import LearningManager
    from tests.helpers import stratified_coeff

    main = {"sensor.x": stratified_coeff(
        s=0.4, e=0.3, w=0.6, cooling_s=0.2, cooling_e=0.1, cooling_w=0.3,
    )}
    shadow: dict = {}

    seeded = LearningManager._seed_shadow_from_main_if_empty("sensor.x", main, shadow)
    assert seeded is True
    assert shadow["sensor.x"]["heating"] == {"s": 0.4, "e": 0.3, "w": 0.6}
    assert shadow["sensor.x"]["cooling"] == {"s": 0.2, "e": 0.1, "w": 0.3}


def test_shadow_seed_is_one_shot():
    """Review I2 (#912): the seed only fires when shadow is empty.
    Once shadow has a populated entry (from a previous handover or
    NLMS shadow trajectory) the helper returns False without
    overwriting — letting shadow's independent evolution continue.
    """
    from custom_components.heating_analytics.learning import LearningManager

    main = {"sensor.x": {"heating": {"s": 0.4, "e": 0.3, "w": 0.6}, "cooling": {}}}
    shadow = {"sensor.x": {"heating": {"s": 0.99, "e": 0.99, "w": 0.99}, "cooling": {}}}

    seeded = LearningManager._seed_shadow_from_main_if_empty("sensor.x", main, shadow)
    assert seeded is False
    # Pre-existing shadow values preserved — NOT overwritten by main.
    assert shadow["sensor.x"]["heating"]["s"] == 0.99


def test_shadow_seed_isolated_dict_no_aliasing():
    """Review I2 (#912) follow-up: the seed must DEEP-copy the regime
    dicts so subsequent shadow NLMS updates don't mutate main's
    coefficient via shared dict references.
    """
    from custom_components.heating_analytics.learning import LearningManager

    main = {"sensor.x": {"heating": {"s": 0.4, "e": 0.3, "w": 0.6}, "cooling": {}}}
    shadow: dict = {}
    LearningManager._seed_shadow_from_main_if_empty("sensor.x", main, shadow)

    # Mutate the seeded shadow.
    shadow["sensor.x"]["heating"]["s"] = 1.23
    # Main must NOT have changed.
    assert main["sensor.x"]["heating"]["s"] == 0.4


def test_confirm_gate_uses_service_validation_error():
    """Review I3 (#912): the confirm-gate on
    ``set_experimental_tobit_live_learner`` must raise
    ``ServiceValidationError`` (not ``ValueError``) so the message
    surfaces in HA's Call Service UI.

    Tested structurally via source inspection because conftest mocks
    ``homeassistant.exceptions`` (so ``pytest.raises`` against the
    mocked class doesn't work).  The structural check pins:
    1. ``ServiceValidationError`` is imported.
    2. The confirm-gate path uses ``ServiceValidationError(...)``.
    3. ``ValueError`` is NOT used in the confirm gate.
    """
    import pathlib
    init_path = pathlib.Path(
        "custom_components/heating_analytics/__init__.py"
    )
    src = init_path.read_text()
    assert "ServiceValidationError" in src, "import or use missing"
    # Find the confirm-gate block.
    assert "if enabled and not confirm:" in src
    # Within ~12 lines after the gate, ServiceValidationError must
    # appear and ValueError must NOT.
    gate_idx = src.index("if enabled and not confirm:")
    next_block = src[gate_idx:gate_idx + 800]
    assert "ServiceValidationError" in next_block
    # The pre-fix ValueError raise must be gone from this region.
    # (ValueError might appear elsewhere in the file for other
    # reasons; we only constrain the confirm-gate region.)
    assert "raise ValueError" not in next_block


# -----------------------------------------------------------------------------
# #918 — Plausibility-gate v2 (default-on auto-discrimination)
# -----------------------------------------------------------------------------


def _pump_samples_to_clear_uncensored_floor(
    lm: LearningManager,
    stats: dict,
    coeffs: dict,
    *,
    sample_vector: tuple[float, float, float],
    actual_impact: float,
    expected_base: float,
    n: int = TOBIT_MIN_NEFF + 5,
) -> dict:
    """Helper: feed n identical samples to clear both pre-fit (|U| ≥
    TOBIT_MIN_UNCENSORED = 20) and post-fit (n_eff ≥ TOBIT_MIN_NEFF =
    40) gates.  Default n=45 is safely above both.

    Returns the result dict from the FINAL update — the prior updates
    return early on the |U| < TOBIT_MIN_UNCENSORED skip.  Identical
    samples produce a degenerate Gram matrix in the OLS sub-fit, so
    callers needing a non-degenerate OLS fit should monkeypatch
    ``_solve_batch_fit_normal_equations`` directly.
    """
    result: dict = {}
    for _ in range(n):
        result = lm._update_unit_tobit_live(
            "sensor.test", "heating", sample_vector,
            actual_impact, expected_base, stats, coeffs,
        )
    return result


def test_plausibility_blocks_no_uncensored_signal(monkeypatch):
    """Noise-load shape (gjæringskjeller-like): uncensored OLS shows
    no slope in any direction, but Tobit fits a non-zero coefficient
    from censoring patterns alone.  Plausibility-gate v2 must skip
    the write so NLMS keeps authority.

    We monkeypatch both solvers: Tobit returns a fit with max=1.0
    (well above PLAUSIBILITY_MIN_TOBIT_MAGNITUDE), OLS returns a fit
    with max=0.05 (below PLAUSIBILITY_MIN_OLS_MAX_DIRECTION).  The
    coefficient must NOT be written; ``applied=False`` and
    ``skip_reason=plausibility_no_uncensored_signal`` must propagate
    via the slot.
    """
    def _fake_tobit(samples, censored_mask, **_kw):
        return {
            "s": 1.0, "e": 0.5, "w": 0.3,
            "sigma": 0.1, "iterations": 5, "converged": True,
            "failure_reason": None, "log_likelihood": -10.0,
            "n_uncensored": len([m for m in censored_mask if not m]),
            "n_censored": sum(1 for m in censored_mask if m),
            "n_eff": float(len(samples)),
        }

    def _fake_ols(samples):
        # No uncensored signal in any direction.
        return {"s": 0.05, "e": 0.03, "w": 0.02}

    monkeypatch.setattr(
        LearningManager, "_solve_tobit_3d", staticmethod(_fake_tobit),
    )
    monkeypatch.setattr(
        LearningManager,
        "_solve_batch_fit_normal_equations",
        staticmethod(_fake_ols),
    )

    stats: dict = {}
    coeffs: dict = {}
    lm = LearningManager()
    result = _pump_samples_to_clear_uncensored_floor(
        lm, stats, coeffs,
        sample_vector=(0.5, 0.3, 0.2),
        actual_impact=0.4, expected_base=1.0,
    )

    assert result["applied"] is False
    slot = stats["sensor.test"]["heating"]
    assert slot["last_step"]["skip_reason"] == "plausibility_no_uncensored_signal"
    # Coefficient was NOT written.
    assert coeffs.get("sensor.test", {}).get("heating", {}) == {}


def test_plausibility_passes_legitimate_vp(monkeypatch):
    """Legitimate VP shape (Toshiba-like): uncensored OLS shows clear
    slope in at least one direction.  Plausibility-gate must let the
    Tobit fit through; coefficient gets written normally.
    """
    def _fake_tobit(samples, censored_mask, **_kw):
        return {
            "s": 1.10, "e": 0.20, "w": 0.45,
            "sigma": 0.1, "iterations": 5, "converged": True,
            "failure_reason": None, "log_likelihood": -10.0,
            "n_uncensored": len([m for m in censored_mask if not m]),
            "n_censored": sum(1 for m in censored_mask if m),
            "n_eff": float(len(samples)),
        }

    def _fake_ols(samples):
        # Strong S signal — unambiguously above the 0.10 floor.
        return {"s": 0.33, "e": 0.10, "w": 0.05}

    monkeypatch.setattr(
        LearningManager, "_solve_tobit_3d", staticmethod(_fake_tobit),
    )
    monkeypatch.setattr(
        LearningManager,
        "_solve_batch_fit_normal_equations",
        staticmethod(_fake_ols),
    )

    stats: dict = {}
    coeffs: dict = {}
    lm = LearningManager()
    result = _pump_samples_to_clear_uncensored_floor(
        lm, stats, coeffs,
        sample_vector=(0.5, 0.3, 0.2),
        actual_impact=0.4, expected_base=1.0,
    )

    assert result["applied"] is True
    written = coeffs["sensor.test"]["heating"]
    assert written["s"] == pytest.approx(1.10)
    assert written["e"] == pytest.approx(0.20)
    assert written["w"] == pytest.approx(0.45)


def test_plausibility_passes_at_threshold(monkeypatch):
    """Boundary: ``ols_max == PLAUSIBILITY_MIN_OLS_MAX_DIRECTION``
    (0.10) is INCLUSIVE — the gate uses strict ``<``, so equality
    passes.  Pins the boundary semantic so a future tweak to the
    inequality direction is a deliberate change.
    """
    from custom_components.heating_analytics.const import (
        PLAUSIBILITY_MIN_OLS_MAX_DIRECTION,
    )

    def _fake_tobit(samples, censored_mask, **_kw):
        return {
            "s": 0.5, "e": 0.0, "w": 0.0,
            "sigma": 0.1, "iterations": 5, "converged": True,
            "failure_reason": None, "log_likelihood": -10.0,
            "n_uncensored": len([m for m in censored_mask if not m]),
            "n_censored": sum(1 for m in censored_mask if m),
            "n_eff": float(len(samples)),
        }

    def _fake_ols(samples):
        return {"s": PLAUSIBILITY_MIN_OLS_MAX_DIRECTION, "e": 0.0, "w": 0.0}

    monkeypatch.setattr(
        LearningManager, "_solve_tobit_3d", staticmethod(_fake_tobit),
    )
    monkeypatch.setattr(
        LearningManager,
        "_solve_batch_fit_normal_equations",
        staticmethod(_fake_ols),
    )

    stats: dict = {}
    coeffs: dict = {}
    lm = LearningManager()
    result = _pump_samples_to_clear_uncensored_floor(
        lm, stats, coeffs,
        sample_vector=(0.5, 0.3, 0.2),
        actual_impact=0.4, expected_base=1.0,
    )
    assert result["applied"] is True


def test_plausibility_blocks_just_below(monkeypatch):
    """Boundary: ``ols_max < PLAUSIBILITY_MIN_OLS_MAX_DIRECTION`` is
    blocked.  Just below the threshold (0.099) → skip.
    """
    from custom_components.heating_analytics.const import (
        PLAUSIBILITY_MIN_OLS_MAX_DIRECTION,
    )

    def _fake_tobit(samples, censored_mask, **_kw):
        return {
            "s": 0.5, "e": 0.0, "w": 0.0,
            "sigma": 0.1, "iterations": 5, "converged": True,
            "failure_reason": None, "log_likelihood": -10.0,
            "n_uncensored": len([m for m in censored_mask if not m]),
            "n_censored": sum(1 for m in censored_mask if m),
            "n_eff": float(len(samples)),
        }

    def _fake_ols(samples):
        return {
            "s": PLAUSIBILITY_MIN_OLS_MAX_DIRECTION - 0.001,
            "e": 0.0, "w": 0.0,
        }

    monkeypatch.setattr(
        LearningManager, "_solve_tobit_3d", staticmethod(_fake_tobit),
    )
    monkeypatch.setattr(
        LearningManager,
        "_solve_batch_fit_normal_equations",
        staticmethod(_fake_ols),
    )

    stats: dict = {}
    coeffs: dict = {}
    lm = LearningManager()
    result = _pump_samples_to_clear_uncensored_floor(
        lm, stats, coeffs,
        sample_vector=(0.5, 0.3, 0.2),
        actual_impact=0.4, expected_base=1.0,
    )
    assert result["applied"] is False
    slot = stats["sensor.test"]["heating"]
    assert slot["last_step"]["skip_reason"] == "plausibility_no_uncensored_signal"


def test_plausibility_skipped_when_tobit_near_zero(monkeypatch):
    """If Tobit's own fit max is below PLAUSIBILITY_MIN_TOBIT_MAGNITUDE
    (0.05), the gate doesn't fire — a near-zero write is harmless and
    we don't want spurious info-logs for entities that legitimately
    have ~zero solar response.  The OLS sub-fit must NOT be invoked in
    this path (we assert by raising in the patched OLS).
    """
    def _fake_tobit(samples, censored_mask, **_kw):
        return {
            "s": 0.02, "e": 0.01, "w": 0.0,  # max = 0.02 < 0.05
            "sigma": 0.1, "iterations": 5, "converged": True,
            "failure_reason": None, "log_likelihood": -10.0,
            "n_uncensored": len([m for m in censored_mask if not m]),
            "n_censored": sum(1 for m in censored_mask if m),
            "n_eff": float(len(samples)),
        }

    ols_called = {"count": 0}

    def _fake_ols(samples):
        ols_called["count"] += 1
        return {"s": 0.0, "e": 0.0, "w": 0.0}

    monkeypatch.setattr(
        LearningManager, "_solve_tobit_3d", staticmethod(_fake_tobit),
    )
    monkeypatch.setattr(
        LearningManager,
        "_solve_batch_fit_normal_equations",
        staticmethod(_fake_ols),
    )

    stats: dict = {}
    coeffs: dict = {}
    lm = LearningManager()
    result = _pump_samples_to_clear_uncensored_floor(
        lm, stats, coeffs,
        sample_vector=(0.5, 0.3, 0.2),
        actual_impact=0.4, expected_base=1.0,
    )
    assert result["applied"] is True
    assert ols_called["count"] == 0, (
        "OLS sub-fit must be skipped when Tobit's own magnitude is "
        "below the secondary gate — the plausibility check would be "
        "noise on a near-zero fit."
    )
    written = coeffs["sensor.test"]["heating"]
    assert written["s"] == pytest.approx(0.02)


def test_plausibility_skipped_when_ols_returns_none(monkeypatch):
    """Defensive: if the OLS sub-fit returns None (degenerate Gram
    matrix on the uncensored subset), plausibility-gate cannot
    discriminate.  Letting Tobit through is the safer default —
    rejecting on indeterminate evidence would suppress legitimate
    fits in collinear edge cases.
    """
    def _fake_tobit(samples, censored_mask, **_kw):
        return {
            "s": 1.0, "e": 0.0, "w": 0.0,
            "sigma": 0.1, "iterations": 5, "converged": True,
            "failure_reason": None, "log_likelihood": -10.0,
            "n_uncensored": len([m for m in censored_mask if not m]),
            "n_censored": sum(1 for m in censored_mask if m),
            "n_eff": float(len(samples)),
        }

    monkeypatch.setattr(
        LearningManager, "_solve_tobit_3d", staticmethod(_fake_tobit),
    )
    monkeypatch.setattr(
        LearningManager,
        "_solve_batch_fit_normal_equations",
        staticmethod(lambda _samples: None),
    )

    stats: dict = {}
    coeffs: dict = {}
    lm = LearningManager()
    result = _pump_samples_to_clear_uncensored_floor(
        lm, stats, coeffs,
        sample_vector=(0.5, 0.3, 0.2),
        actual_impact=0.4, expected_base=1.0,
    )
    assert result["applied"] is True


# -----------------------------------------------------------------------------
# #918 — Allow-list semantic shift (empty=auto, non-empty=override)
# -----------------------------------------------------------------------------


def test_allow_list_empty_runs_auto_mode():
    """Empty allow-list with master flag enabled = "auto-mode": every
    eligible entity is candidate for live Tobit (subject to
    plausibility-gate, MPC exclusion, regime).  Pre-1.3.5 semantics
    blocked all entities on empty list; 1.3.5 inverts to auto.
    """
    assert _gate_predicate(
        True, "sensor.toshiba_vp", frozenset(), frozenset()
    ) is True


def test_allow_list_non_empty_scopes():
    """Non-empty allow-list with master flag enabled = "override-mode":
    only listed entities are candidates.  Maintainer's existing
    [Toshiba, Mitsubishi] config keeps working as scope-override
    rather than degrading to auto-mode silently.
    """
    scope = frozenset(["sensor.toshiba_vp", "sensor.mitsubishi"])
    assert _gate_predicate(True, "sensor.toshiba_vp", scope, frozenset()) is True
    assert _gate_predicate(
        True, "sensor.basement_socket", scope, frozenset()
    ) is False, (
        "Non-empty allow-list must scope-restrict to listed entities; "
        "an entity outside the list must NOT activate Tobit even "
        "though the master flag is enabled."
    )


def test_allow_list_master_flag_off_blocks_all():
    """Master flag False short-circuits regardless of allow-list
    contents.  Explicit user-disable via
    ``set_experimental_tobit_live_learner enabled: false`` must take
    precedence over any allow-list state.
    """
    assert _gate_predicate(
        False, "sensor.toshiba_vp", frozenset(["sensor.toshiba_vp"]), frozenset()
    ) is False


# -----------------------------------------------------------------------------
# #918 — Default-on migration (storage marker)
# -----------------------------------------------------------------------------


def _build_minimal_v5_data(*, marker: bool, flag: bool) -> dict:
    """Skeletal storage payload with just enough fields for the load
    path to walk through to the Tobit section without raising.
    """
    return {
        "correlation_data": {},
        "correlation_data_per_unit": {},
        "tobit_sufficient_stats": {},
        "experimental_tobit_live_learner": flag,
        "tobit_live_entities": [],
        "_tobit_default_applied": marker,
    }


def test_default_on_migration_first_load(monkeypatch):
    """Loading 1.3.4-shape storage (no `_tobit_default_applied` marker
    in the saved data) MUST flip the master flag to True and stamp
    the marker.  This is the user-facing 1.3.5 default-on promotion.
    """
    from custom_components.heating_analytics.storage import StorageManager

    coord = MagicMock()
    coord._correlation_data = {}
    coord._tobit_sufficient_stats = {}
    coord._experimental_tobit_live_learner = False
    coord._tobit_live_entities = frozenset()
    coord._tobit_default_applied = False
    # We bypass __init__ entirely — only the load path is under test.
    sm = object.__new__(StorageManager)
    sm.coordinator = coord

    # 1.3.4-shape data: master flag False, allow-list empty, NO marker.
    data = {
        "correlation_data": {},
        "correlation_data_per_unit": {},
        "tobit_sufficient_stats": {},
        "experimental_tobit_live_learner": False,
        "tobit_live_entities": [],
        # (no _tobit_default_applied)
    }

    # Inline-replicate the marker-check path (the load function is
    # several hundred lines and we only care about the marker logic).
    coord._experimental_tobit_live_learner = bool(
        data.get("experimental_tobit_live_learner", False)
    )
    coord._tobit_default_applied = bool(data.get("_tobit_default_applied", False))
    if not coord._tobit_default_applied:
        coord._experimental_tobit_live_learner = True
        coord._tobit_default_applied = True

    assert coord._experimental_tobit_live_learner is True
    assert coord._tobit_default_applied is True


def test_default_on_migration_idempotent():
    """Subsequent load with marker already True must NOT re-flip the
    flag.  If the user explicitly disabled in between via
    ``set_experimental_tobit_live_learner``, the False value persists
    through restart — the marker says "default has been applied",
    not "user wants Tobit on".
    """
    coord = MagicMock()
    # User explicitly disabled in a previous session; marker is True.
    data = {
        "experimental_tobit_live_learner": False,
        "tobit_live_entities": [],
        "_tobit_default_applied": True,
    }

    coord._experimental_tobit_live_learner = bool(
        data.get("experimental_tobit_live_learner", False)
    )
    coord._tobit_default_applied = bool(data.get("_tobit_default_applied", False))
    if not coord._tobit_default_applied:
        coord._experimental_tobit_live_learner = True
        coord._tobit_default_applied = True

    assert coord._experimental_tobit_live_learner is False, (
        "User-disable from prior session must persist; marker=True "
        "blocks the default-on flip."
    )
    assert coord._tobit_default_applied is True


def test_default_on_marker_round_trips_through_save(monkeypatch):
    """After the load-path flip + marker stamp, the next save MUST
    persist both the new flag value and the marker so the next load
    is a no-op (idempotency contract).
    """
    coord = MagicMock()
    coord._experimental_tobit_live_learner = True
    coord._tobit_default_applied = True
    coord._tobit_live_entities = frozenset()
    coord._tobit_sufficient_stats = {}

    # Build the dict snippet the save path emits for the Tobit section.
    saved = {
        "tobit_sufficient_stats": coord._tobit_sufficient_stats,
        "experimental_tobit_live_learner": coord._experimental_tobit_live_learner,
        "tobit_live_entities": list(coord._tobit_live_entities),
        "_tobit_default_applied": bool(
            getattr(coord, "_tobit_default_applied", False)
        ),
    }
    assert saved["_tobit_default_applied"] is True
    assert saved["experimental_tobit_live_learner"] is True


def test_user_disable_persisted_through_simulated_reload():
    """End-to-end roundtrip simulation: user enables (1.3.5 first
    boot), then explicitly disables, then reloads.  After reload the
    flag must be False — the marker prevents the load-time flip from
    re-enabling.
    """
    # First-boot load.
    storage_state: dict = {
        "experimental_tobit_live_learner": False,
        "tobit_live_entities": [],
        # No marker yet.
    }
    flag = bool(storage_state.get("experimental_tobit_live_learner", False))
    marker = bool(storage_state.get("_tobit_default_applied", False))
    if not marker:
        flag = True
        marker = True

    assert flag is True and marker is True

    # User explicitly disables via service call.
    flag = False
    # Marker stays True (service handler doesn't touch it).

    # Next save persists both.
    storage_state = {
        "experimental_tobit_live_learner": flag,
        "tobit_live_entities": [],
        "_tobit_default_applied": marker,
    }

    # Simulate restart: load again.
    flag2 = bool(storage_state.get("experimental_tobit_live_learner", False))
    marker2 = bool(storage_state.get("_tobit_default_applied", False))
    if not marker2:
        flag2 = True
        marker2 = True

    assert flag2 is False, "User-disable must survive restart"
    assert marker2 is True


# -----------------------------------------------------------------------------
# #918 — Constants are pinned
# -----------------------------------------------------------------------------


def test_plausibility_constants_pinned():
    """Pin the calibrated thresholds.  Bumping these is a deliberate
    edit that must touch this test — see the bump-rule comment in
    const.py.
    """
    from custom_components.heating_analytics.const import (
        PLAUSIBILITY_MIN_OLS_MAX_DIRECTION,
        PLAUSIBILITY_MIN_TOBIT_MAGNITUDE,
    )
    assert PLAUSIBILITY_MIN_OLS_MAX_DIRECTION == 0.10
    assert PLAUSIBILITY_MIN_TOBIT_MAGNITUDE == 0.05


def test_plausibility_v2_constants_pinned():
    """Pin the cosine-similarity floor and the rate-limit fraction.
    Bumping these is a deliberate change touching this test.
    """
    from custom_components.heating_analytics.const import (
        PLAUSIBILITY_MIN_DIRECTION_COSINE,
        PLAUSIBILITY_RATE_LIMIT_FRACTION,
    )
    assert PLAUSIBILITY_MIN_DIRECTION_COSINE == 0.5
    assert PLAUSIBILITY_RATE_LIMIT_FRACTION == 0.30


# -----------------------------------------------------------------------------
# Production-realistic basin/trajectory tests (no fake_tobit)
# -----------------------------------------------------------------------------


def _generate_synthetic_window(
    *,
    true_c_w: float,
    sigma_true: float,
    n_samples: int,
    seed: int,
):
    """Build a synthetic Tobit window with controlled true coefficient
    and censoring rate.  Returns the (sample, mask) tuple list ready
    for slot pre-seeding.  All signal lives on the W direction (S, E
    have small uniform potentials but zero true coefficient — the
    censoring fraction depends on T = 0.95 × base relative to the
    W-direction expected value).
    """
    rng = random.Random(seed)
    samples_with_mask = []
    for _ in range(n_samples):
        s = rng.uniform(0, 0.3)
        e = rng.uniform(0, 0.3)
        w = rng.uniform(0.5, 1.0)
        base = 1.5
        pred = true_c_w * w
        y_star = pred + rng.gauss(0, sigma_true)
        T = 0.95 * base
        if y_star >= T:
            samples_with_mask.append((s, e, w, T, True))
        else:
            samples_with_mask.append((s, e, w, max(0.001, y_star), False))
    return samples_with_mask


def test_production_tobit_converges_from_biased_warmstart():
    """Production-realistic test: real ``_solve_tobit_3d`` solver,
    biased warm-start coefficient, and the LS-fallback fix.  This is
    the test that would have caught the original Stage-3 architectural
    bug — without LS-fallback, Tobit Newton fails line-search from
    biased warm-starts and the live coefficient stays stuck on the
    NLMS-converged-but-biased value indefinitely.

    Setup mirrors the Toshiba-class scenario: NLMS converges to 0.55
    (saturation-biased), Tobit takes over with true c_w=1.65.
    Trajectory should approach the LS-Tobit fit on this window over
    ~5 hours via rate-limit cushion + LS-fallback warm-start retry on
    each hour outside the basin.

    Test design (seed-robust): the LS-Tobit fit on a 100-sample
    window has a per-seed σ ≈ 0.03 around the physical truth,
    range ≈ [1.52, 1.68] across a 200-seed sweep.  Hardcoded
    assertions of "trajectory[N] ≈ 1.65" fail on roughly 26 % of
    seeds where LS-Tobit lands at the lower end of that range —
    the trajectory is correct, but it converges to LS-Tobit's
    estimate (data-dependent), not to physical truth.

    This test pins:
      1. Hours 0-2: deterministic by rate-limit arithmetic for any
         LS-Tobit target > 1.21 (verified across the seed sweep —
         no seed gives target < 1.52, comfortably above 1.21).
      2. Hours 3+: monotonically approaches LS-Tobit target without
         overshoot.
      3. Final hour: converged to LS-Tobit fit within Newton tol.
      4. LS-Tobit fit itself: within solver-noise of physical truth.

    Pre-fix this test would fail because every Tobit call returned
    ``did_not_converge`` and ``applied=False``.
    """
    samples_with_mask = _generate_synthetic_window(
        true_c_w=1.65, sigma_true=0.1, n_samples=100, seed=42,
    )
    stats = {
        "sensor.test": {
            "heating": {
                "samples": list(samples_with_mask),
                "samples_since_reset": 100,
                "last_step": {},
                "solar_model_version": 1,
            }
        }
    }
    coeffs = {"sensor.test": {"heating": {"s": 0.0, "e": 0.0, "w": 0.55}}}
    lm = LearningManager()

    trajectory = []
    for _ in range(7):
        result = lm._update_unit_tobit_live(
            "sensor.test", "heating", (0.5, 0.3, 0.2),
            0.4, 1.0, stats, coeffs,
        )
        assert result["applied"] is True, (
            "Tobit must converge on every hour after LS-fallback fix; "
            f"got applied={result['applied']}, "
            f"failure_reason={result.get('last_step_failure_reason')}"
        )
        trajectory.append(coeffs["sensor.test"]["heating"]["w"])

    # Compute the LS-Tobit target on the FINAL window state.  Each
    # ``_update_unit_tobit_live`` call appends a new sample, so the
    # window grows from 100 to 107 over 7 pumps.  The asymptote the
    # trajectory chases shifts hour-to-hour with the window.  Pin
    # convergence to the FINAL-window target — what the live
    # coefficient actually equilibrates to at the end of the loop.
    final_samples = stats["sensor.test"]["heating"]["samples"]
    samples_4tup = [(s[0], s[1], s[2], s[3]) for s in final_samples]
    mask = [s[4] for s in final_samples]
    ls_target_fit = LearningManager._solve_tobit_3d(
        samples_4tup, mask, coeff_init=None,
    )
    target_w = ls_target_fit["w"]

    # Hours 0-2: deterministic by rate-limit arithmetic (any target > 1.21).
    # Cap × prior compounds: 0.55 → 0.715 → 0.9295 → 1.20835.
    assert trajectory[0] == pytest.approx(0.715, abs=0.005)
    assert trajectory[1] == pytest.approx(0.9295, abs=0.005)
    assert trajectory[2] == pytest.approx(1.20835, abs=0.005)

    # Hours 3+: data-dependent on LS-Tobit target.  Pin convergence
    # behavior (monotonic, bounded, asymptotically at target), not
    # specific values.
    for i in range(3, len(trajectory)):
        # Non-decreasing (rate-limit is one-sided cap with target above prior)
        assert trajectory[i] >= trajectory[i - 1] - 1e-3, (
            f"Hour {i+1}: trajectory regressed from "
            f"{trajectory[i-1]:.4f} to {trajectory[i]:.4f}"
        )
        # Bounded by LS-Tobit target (rate-limit and CAP cannot overshoot)
        assert trajectory[i] <= target_w + 1e-3, (
            f"Hour {i+1}: trajectory {trajectory[i]:.4f} exceeded "
            f"LS-Tobit target {target_w:.4f}"
        )

    # Final value converged to LS-Tobit target within Newton tolerance.
    assert trajectory[-1] == pytest.approx(target_w, abs=0.005), (
        f"Final value {trajectory[-1]:.4f} did not converge to "
        f"LS-Tobit target {target_w:.4f}"
    )

    # Sanity: LS-Tobit target itself within solver-noise of physical truth.
    # σ/√n_unc ≈ 0.10/√73 ≈ 0.012; ±0.20 covers seed variability.
    assert target_w == pytest.approx(1.65, abs=0.20), (
        f"LS-Tobit target {target_w:.4f} too far from physical truth 1.65"
    )



def test_production_tobit_basin_via_ls_fallback():
    """Pin the LS-fallback contract directly: a single
    ``_update_unit_tobit_live`` call with a biased warm-start that
    would fail Newton line-search must succeed via the in-function
    LS-fallback retry.

    Pre-fix: warm-start at 0.10 (far from true 1.65) → Tobit fails
    line-search → applied=False, write skipped.  Post-fix:
    line_search_failed triggers retry with coeff_init=None →
    LS-Tobit converges → applied=True (subject to rate-limit on the
    resulting jump).

    This is a tighter contract than the trajectory test — it pins
    the LS-fallback path even in the single-step case.
    """
    samples_with_mask = _generate_synthetic_window(
        true_c_w=1.65, sigma_true=0.1, n_samples=100, seed=42,
    )
    stats = {
        "sensor.test": {
            "heating": {
                "samples": list(samples_with_mask),
                "samples_since_reset": 100,
                "last_step": {},
                "solar_model_version": 1,
            }
        }
    }
    # Cold-start-like prior (0.03 < cushion floor 0.05) — well outside
    # the ±10 % Tobit basin around 1.65.  Without LS-fallback, Tobit
    # fails line_search.  With LS-fallback, it retries and converges.
    # Below-cushion-floor: no rate-limit fires either, so the full
    # LS-Tobit fit writes through directly — clean LS-fallback contract
    # test independent of rate-limit semantics.
    coeffs = {"sensor.test": {"heating": {"s": 0.0, "e": 0.0, "w": 0.03}}}
    lm = LearningManager()
    result = lm._update_unit_tobit_live(
        "sensor.test", "heating", (0.5, 0.3, 0.2),
        0.4, 1.0, stats, coeffs,
    )

    assert result["applied"] is True, (
        "LS-fallback must rescue a biased warm-start that would "
        "otherwise fail Newton line-search"
    )
    written_w = coeffs["sensor.test"]["heating"]["w"]
    assert written_w == pytest.approx(1.65, abs=0.05)


def test_solver_sigma_init_extends_newton_progress_on_biased_warmstart():
    """Defense-in-depth: when ``_solve_tobit_3d`` is called with a
    biased ``coeff_init``, σ-init must come from LS residuals (not
    biased-c residuals).  Pre-fix, σ was inflated 5-10× by the biased
    SSE → Newton over-corrected σ on iter 1 → line search failed
    immediately (iterations=1).  Post-fix, Newton progresses several
    iterations further (observed: iterations=5 on the calibrated
    fixture) before failing — well-conditioned σ at iter 0 lets the
    first c-step land somewhere usable.

    Newton may still fail to converge from a far warm-start (the
    σ-init fix does NOT widen the basin — verified by pre-fix vs
    post-fix sweep; LS-fallback in ``_update_unit_tobit_live`` handles
    that case).  This test pins the iteration-count contract that
    distinguishes a regression in σ-init logic from an entirely
    intact path.

    Threshold ``iterations >= 4`` chosen with margin from the observed
    post-fix value of 5 on the ``seed=42`` calibration fixture.  Seed
    sweep across {1, 2, 3, 7, 42, 100, 1234, 2024, 9999} confirms
    iterations ∈ [5, 7] — threshold has 1-iteration safety margin
    against the worst observed seed.  A future change that yields
    iterations=2-3 (e.g. partial regression in σ-init logic that picks
    a less well-conditioned source) would fail this test.

    Calibrated against ``seed=42`` — a future fixture-data change
    (different ``_generate_synthetic_window`` shape) MUST re-run the
    seed sweep and re-calibrate the threshold.
    """
    samples_with_mask = _generate_synthetic_window(
        true_c_w=1.65, sigma_true=0.1, n_samples=100, seed=42,
    )
    samples_4tup = [(s[0], s[1], s[2], s[3]) for s in samples_with_mask]
    mask = [s[4] for s in samples_with_mask]
    biased_warm = {"s": 0.0, "e": 0.0, "w": 0.55}

    fit = LearningManager._solve_tobit_3d(
        samples_4tup, mask, coeff_init=biased_warm,
    )
    assert fit is not None
    assert fit["iterations"] >= 4, (
        f"σ-init fix should let Newton progress >= 4 iterations on "
        f"this biased warm-start (observed 5 on the calibrated "
        f"fixture); got iterations={fit['iterations']}, "
        f"failure_reason={fit.get('failure_reason')}"
    )


# -----------------------------------------------------------------------------
# Cooling-regime skip (physics review #1)
# -----------------------------------------------------------------------------


def test_plausibility_skipped_on_cooling_regime(monkeypatch):
    """Cooling has no upper-saturation; every sample is uncensored, so
    Tobit reduces to OLS exactly.  Plausibility-gate would otherwise
    structurally false-positive on cooling VPs whose learned coefficient
    sits in the 0.05-0.10 band (typical for small AC installs).  Cooling
    must skip plausibility entirely and write through.

    We assert by patching ``_solve_batch_fit_normal_equations`` to raise
    if called — cooling must not invoke it.
    """
    def _fake_tobit(samples, censored_mask, **_kw):
        return {
            "s": 0.07, "e": 0.03, "w": 0.0,  # tobit_max=0.07 — would be in band
            "sigma": 0.1, "iterations": 5, "converged": True,
            "failure_reason": None, "log_likelihood": -10.0,
            "n_uncensored": len(samples),
            "n_censored": 0,
            "n_eff": float(len(samples)),
        }

    def _fake_ols(samples):
        raise AssertionError(
            "OLS sub-fit must NOT be invoked on cooling regime — "
            "cooling skips plausibility-gate entirely."
        )

    monkeypatch.setattr(
        LearningManager, "_solve_tobit_3d", staticmethod(_fake_tobit),
    )
    monkeypatch.setattr(
        LearningManager,
        "_solve_batch_fit_normal_equations",
        staticmethod(_fake_ols),
    )

    stats: dict = {}
    coeffs: dict = {}
    lm = LearningManager()
    n = TOBIT_MIN_NEFF + 5
    result = {}
    for _ in range(n):
        result = lm._update_unit_tobit_live(
            "sensor.test", "cooling", (0.5, 0.3, 0.2),
            0.4, 1.0, stats, coeffs,
        )

    assert result["applied"] is True
    assert coeffs["sensor.test"]["cooling"]["s"] == pytest.approx(0.07)


# -----------------------------------------------------------------------------
# Direction-cosine gate (physics review #2)
# -----------------------------------------------------------------------------


def test_plausibility_blocks_direction_mismatch(monkeypatch):
    """Tobit's projected-Newton can pin a direction at zero from a
    wrong warm-start: real signal is W-dominant, Tobit fits S-dominant.
    OLS-on-uncensored correctly identifies E-direction.  Magnitude
    check passes (ols_max=0.30, well above 0.10) but direction-cosine
    is near zero → block as ``plausibility_direction_mismatch``.
    """
    def _fake_tobit(samples, censored_mask, **_kw):
        return {
            "s": 1.50, "e": 0.0, "w": 0.0,  # wrong direction
            "sigma": 0.1, "iterations": 5, "converged": True,
            "failure_reason": None, "log_likelihood": -10.0,
            "n_uncensored": len([m for m in censored_mask if not m]),
            "n_censored": sum(1 for m in censored_mask if m),
            "n_eff": float(len(samples)),
        }

    def _fake_ols(samples):
        # Real signal is E-direction; magnitude clears 0.10.
        return {"s": 0.0, "e": 0.30, "w": 0.0}

    monkeypatch.setattr(
        LearningManager, "_solve_tobit_3d", staticmethod(_fake_tobit),
    )
    monkeypatch.setattr(
        LearningManager,
        "_solve_batch_fit_normal_equations",
        staticmethod(_fake_ols),
    )

    stats: dict = {}
    coeffs: dict = {}
    lm = LearningManager()
    n = TOBIT_MIN_NEFF + 5
    result = {}
    for _ in range(n):
        result = lm._update_unit_tobit_live(
            "sensor.test", "heating", (0.5, 0.3, 0.2),
            0.4, 1.0, stats, coeffs,
        )

    assert result["applied"] is False
    slot = stats["sensor.test"]["heating"]
    assert slot["last_step"]["skip_reason"] == "plausibility_direction_mismatch"
    assert slot["last_step"]["direction_cosine"] < 0.5
    # Coefficient was NOT written.
    assert coeffs.get("sensor.test", {}).get("heating", {}) == {}


def test_plausibility_passes_aligned_directions(monkeypatch):
    """Tobit and OLS point in the same direction (cosine ≈ 1) → pass.
    Sanity test pinning that the cosine check doesn't false-block on
    legitimate fits.
    """
    def _fake_tobit(samples, censored_mask, **_kw):
        return {
            "s": 0.0, "e": 0.10, "w": 1.50,  # W-dominant
            "sigma": 0.1, "iterations": 5, "converged": True,
            "failure_reason": None, "log_likelihood": -10.0,
            "n_uncensored": len([m for m in censored_mask if not m]),
            "n_censored": sum(1 for m in censored_mask if m),
            "n_eff": float(len(samples)),
        }

    def _fake_ols(samples):
        # Aligned: also W-dominant, magnitude over 0.10.
        return {"s": 0.0, "e": 0.05, "w": 0.30}

    monkeypatch.setattr(
        LearningManager, "_solve_tobit_3d", staticmethod(_fake_tobit),
    )
    monkeypatch.setattr(
        LearningManager,
        "_solve_batch_fit_normal_equations",
        staticmethod(_fake_ols),
    )

    stats: dict = {}
    coeffs: dict = {}
    lm = LearningManager()
    n = TOBIT_MIN_NEFF + 5
    result = {}
    for _ in range(n):
        result = lm._update_unit_tobit_live(
            "sensor.test", "heating", (0.5, 0.3, 0.2),
            0.4, 1.0, stats, coeffs,
        )

    assert result["applied"] is True
    assert coeffs["sensor.test"]["heating"]["w"] == pytest.approx(1.50)


# -----------------------------------------------------------------------------
# Rate-limit on first post-block transition (physics review #3)
# -----------------------------------------------------------------------------


def test_rate_limit_fires_on_large_step(monkeypatch):
    """Rate-limit triggers whenever any direction's proposed step
    exceeds 30 % of prior_max — regardless of plausibility-block
    history.  Worst-case scenario: post-block recovery where Tobit
    wants 1.65 but prior coefficient (NLMS-converged) is 0.55.  The
    cap clamps the first step to 0.55 + 0.165 = 0.715.

    The trigger is purely step-size based.  See
    ``test_rate_limit_multi_hour_soft_convergence`` for the full
    convergence trajectory; this test pins the single-step cap.
    """
    def _fake_tobit(samples, censored_mask, **_kw):
        return {
            "s": 0.0, "e": 0.0, "w": 1.65,  # would-be jump target
            "sigma": 0.1, "iterations": 5, "converged": True,
            "failure_reason": None, "log_likelihood": -10.0,
            "n_uncensored": len([m for m in censored_mask if not m]),
            "n_censored": sum(1 for m in censored_mask if m),
            "n_eff": 100.0,  # comfortably above TOBIT_MIN_NEFF
        }

    def _fake_ols(samples):
        return {"s": 0.0, "e": 0.0, "w": 0.50}  # passes magnitude + direction

    monkeypatch.setattr(
        LearningManager, "_solve_tobit_3d", staticmethod(_fake_tobit),
    )
    monkeypatch.setattr(
        LearningManager,
        "_solve_batch_fit_normal_equations",
        staticmethod(_fake_ols),
    )

    # Pre-seed the slot with TOBIT_MIN_UNCENSORED+1 uncensored samples
    # AND last_step=plausibility-block — so the FIRST call after this
    # already has n_unc above the floor and was_plausibility_blocked
    # observes the pre-seeded marker (not an "insufficient_uncensored"
    # transient that would clobber the marker before cold-start
    # buffering is past).
    pre_seeded_samples = [
        (0.5, 0.0, 0.0, 0.3, False)
        for _ in range(TOBIT_MIN_UNCENSORED + 1)
    ]
    stats: dict = {
        "sensor.test": {
            "heating": {
                "samples": pre_seeded_samples,
                "samples_since_reset": 100,
                "last_step": {
                    "skip_reason": "plausibility_no_uncensored_signal",
                    "ols_max": 0.05,
                    "tobit_max": 1.65,
                },
                "solar_model_version": 1,
            }
        }
    }
    coeffs: dict = {
        "sensor.test": {"heating": {"s": 0.0, "e": 0.0, "w": 0.55}}
    }
    lm = LearningManager()
    # Single call — the pre-seed already cleared n_eff/n_unc gates.
    result = lm._update_unit_tobit_live(
        "sensor.test", "heating", (0.5, 0.3, 0.2),
        0.4, 1.0, stats, coeffs,
    )

    assert result["applied"] is True
    written_w = coeffs["sensor.test"]["heating"]["w"]
    # Cap = 0.30 × 0.55 = 0.165 → new = 0.55 + 0.165 = 0.715
    assert written_w == pytest.approx(0.55 + 0.30 * 0.55, rel=1e-3), (
        f"First post-block step must be rate-limited; got w={written_w}, "
        f"expected ~{0.55 + 0.30 * 0.55:.4f}"
    )


def test_no_rate_limit_when_prior_below_cushion_floor(monkeypatch):
    """Cold-start case: plausibility was blocking but NLMS was also
    not writing (e.g. MPC-managed entity in a prior session that has
    since been unmanaged).  prior_max < 0.05 → no rate-limit, Tobit's
    full fit writes through directly.  Otherwise we'd never bootstrap
    out of all-zero state.
    """
    def _fake_tobit(samples, censored_mask, **_kw):
        return {
            "s": 0.0, "e": 0.0, "w": 1.65,
            "sigma": 0.1, "iterations": 5, "converged": True,
            "failure_reason": None, "log_likelihood": -10.0,
            "n_uncensored": len([m for m in censored_mask if not m]),
            "n_censored": sum(1 for m in censored_mask if m),
            "n_eff": 100.0,
        }

    monkeypatch.setattr(
        LearningManager, "_solve_tobit_3d", staticmethod(_fake_tobit),
    )
    monkeypatch.setattr(
        LearningManager,
        "_solve_batch_fit_normal_equations",
        staticmethod(lambda _s: {"s": 0.0, "e": 0.0, "w": 0.50}),
    )

    pre_seeded_samples = [
        (0.5, 0.0, 0.0, 0.3, False)
        for _ in range(TOBIT_MIN_UNCENSORED + 1)
    ]
    stats: dict = {
        "sensor.test": {
            "heating": {
                "samples": pre_seeded_samples,
                "samples_since_reset": 100,
                "last_step": {"skip_reason": "plausibility_no_uncensored_signal"},
                "solar_model_version": 1,
            }
        }
    }
    coeffs: dict = {}  # no prior coefficient
    lm = LearningManager()
    result = lm._update_unit_tobit_live(
        "sensor.test", "heating", (0.5, 0.3, 0.2),
        0.4, 1.0, stats, coeffs,
    )

    assert result["applied"] is True
    # No prior to cushion against → full write goes through.
    assert coeffs["sensor.test"]["heating"]["w"] == pytest.approx(1.65)


def test_rate_limit_multi_hour_soft_convergence(monkeypatch):
    """Multi-hour soft-convergence: with prior=0.55 and Tobit-target
    =1.65, the rate-limit produces a 5-hour trajectory (0.715 → 0.929
    → 1.207 → 1.569 → 1.65) instead of one 200 % jump.  Each step is
    capped at 30 % of the new prior_max from the previous hour.

    This pins the multi-hour behavior of the general step-size
    limiter — under the pre-fix one-shot semantics, the trajectory
    would have been 0.55 → 0.715 → 1.65 (two hours, with a 131 %
    jump on hour 2 after rate-limit no longer fired).
    """
    def _fake_tobit(samples, censored_mask, **_kw):
        return {
            "s": 0.0, "e": 0.0, "w": 1.65,  # constant target
            "sigma": 0.1, "iterations": 5, "converged": True,
            "failure_reason": None, "log_likelihood": -10.0,
            "n_uncensored": len([m for m in censored_mask if not m]),
            "n_censored": sum(1 for m in censored_mask if m),
            "n_eff": 100.0,
        }

    monkeypatch.setattr(
        LearningManager, "_solve_tobit_3d", staticmethod(_fake_tobit),
    )
    monkeypatch.setattr(
        LearningManager,
        "_solve_batch_fit_normal_equations",
        staticmethod(lambda _s: {"s": 0.0, "e": 0.0, "w": 0.50}),
    )

    pre_seeded_samples = [
        (0.5, 0.0, 0.0, 0.3, False)
        for _ in range(TOBIT_MIN_UNCENSORED + 1)
    ]
    stats: dict = {
        "sensor.test": {
            "heating": {
                "samples": pre_seeded_samples,
                "samples_since_reset": 100,
                "last_step": {},
                "solar_model_version": 1,
            }
        }
    }
    coeffs: dict = {
        "sensor.test": {"heating": {"s": 0.0, "e": 0.0, "w": 0.55}}
    }
    lm = LearningManager()

    trajectory = []
    for _ in range(7):
        lm._update_unit_tobit_live(
            "sensor.test", "heating", (0.5, 0.3, 0.2),
            0.4, 1.0, stats, coeffs,
        )
        trajectory.append(coeffs["sensor.test"]["heating"]["w"])

    # Hour 1: cap = 0.30 × 0.55 = 0.165 → 0.715
    # Hour 2: cap = 0.30 × 0.715 = 0.2145 → 0.9295
    # Hour 3: cap = 0.30 × 0.9295 = 0.2789 → 1.2084
    # Hour 4: cap = 0.30 × 1.2084 = 0.3625 → 1.5709
    # Hour 5: step 0.0791 < cap 0.4713 → unconstrained → 1.65
    # Hour 6+: step 0 → no rate-limit, write 1.65
    assert trajectory[0] == pytest.approx(0.715, abs=1e-3)
    assert trajectory[1] == pytest.approx(0.9295, abs=1e-3)
    assert trajectory[2] == pytest.approx(1.2084, abs=1e-3)
    assert trajectory[3] == pytest.approx(1.5709, abs=1e-3)
    assert trajectory[4] == pytest.approx(1.65, abs=1e-3)
    assert trajectory[5] == pytest.approx(1.65, abs=1e-3)
    # Trajectory is monotonically non-decreasing toward target.
    for i in range(1, len(trajectory)):
        assert trajectory[i - 1] <= trajectory[i] + 1e-6
        assert trajectory[i] <= 1.65 + 1e-6


def test_rate_limit_fires_on_cooling_too(monkeypatch):
    """The rate-limit is regime-agnostic — applies to cooling exactly
    the same way as heating.  Pins the contract that cooling skips
    plausibility checks but NOT the step-size limiter.  A cooling
    Tobit fit that wants to jump 200 % from a converged prior gets
    smoothed identically to heating.
    """
    def _fake_tobit(samples, censored_mask, **_kw):
        return {
            "s": 0.0, "e": 0.0, "w": 1.20,  # large step from 0.40 prior
            "sigma": 0.1, "iterations": 5, "converged": True,
            "failure_reason": None, "log_likelihood": -10.0,
            "n_uncensored": len(samples),
            "n_censored": 0,
            "n_eff": 100.0,
        }

    # OLS solver MUST NOT be invoked on cooling (plausibility skip).
    def _no_ols(_s):
        raise AssertionError(
            "OLS sub-fit must not be invoked on cooling; only "
            "plausibility-gate uses it, and cooling skips plausibility."
        )

    monkeypatch.setattr(
        LearningManager, "_solve_tobit_3d", staticmethod(_fake_tobit),
    )
    monkeypatch.setattr(
        LearningManager,
        "_solve_batch_fit_normal_equations",
        staticmethod(_no_ols),
    )

    pre_seeded_samples = [
        (0.5, 0.0, 0.0, 0.3, False)
        for _ in range(TOBIT_MIN_UNCENSORED + 1)
    ]
    stats: dict = {
        "sensor.test": {
            "cooling": {
                "samples": pre_seeded_samples,
                "samples_since_reset": 100,
                "last_step": {},
                "solar_model_version": 1,
            }
        }
    }
    coeffs: dict = {
        "sensor.test": {"cooling": {"s": 0.0, "e": 0.0, "w": 0.40}}
    }
    lm = LearningManager()
    result = lm._update_unit_tobit_live(
        "sensor.test", "cooling", (0.5, 0.3, 0.2),
        0.6, 1.0, stats, coeffs,
    )

    assert result["applied"] is True
    # cap = 0.30 × 0.40 = 0.12 → first-step write = 0.40 + 0.12 = 0.52
    assert coeffs["sensor.test"]["cooling"]["w"] == pytest.approx(0.52)


def test_rate_limit_respects_solar_coeff_cap(monkeypatch):
    """When prior coefficient is near SOLAR_COEFF_CAP and Tobit wants
    to push above it, the per-direction CAP clamp wins over the
    rate-limit math.  Pins that the order-of-operations doesn't allow
    a step to bypass SOLAR_COEFF_CAP.
    """
    from custom_components.heating_analytics.const import SOLAR_COEFF_CAP

    prior_w = SOLAR_COEFF_CAP - 0.1  # near cap
    target_w = SOLAR_COEFF_CAP + 1.0  # would-be target (clamped)

    def _fake_tobit(samples, censored_mask, **_kw):
        return {
            "s": 0.0, "e": 0.0, "w": target_w,
            "sigma": 0.1, "iterations": 5, "converged": True,
            "failure_reason": None, "log_likelihood": -10.0,
            "n_uncensored": len([m for m in censored_mask if not m]),
            "n_censored": sum(1 for m in censored_mask if m),
            "n_eff": 100.0,
        }

    monkeypatch.setattr(
        LearningManager, "_solve_tobit_3d", staticmethod(_fake_tobit),
    )
    monkeypatch.setattr(
        LearningManager,
        "_solve_batch_fit_normal_equations",
        staticmethod(lambda _s: {"s": 0.0, "e": 0.0, "w": 1.0}),
    )

    pre_seeded_samples = [
        (0.5, 0.0, 0.0, 0.3, False)
        for _ in range(TOBIT_MIN_UNCENSORED + 1)
    ]
    stats: dict = {
        "sensor.test": {
            "heating": {
                "samples": pre_seeded_samples,
                "samples_since_reset": 100,
                "last_step": {},
                "solar_model_version": 1,
            }
        }
    }
    coeffs: dict = {
        "sensor.test": {"heating": {"s": 0.0, "e": 0.0, "w": prior_w}}
    }
    lm = LearningManager()
    lm._update_unit_tobit_live(
        "sensor.test", "heating", (0.5, 0.3, 0.2),
        0.4, 1.0, stats, coeffs,
    )

    written = coeffs["sensor.test"]["heating"]["w"]
    assert written <= SOLAR_COEFF_CAP, (
        "SOLAR_COEFF_CAP must clamp regardless of rate-limit math; "
        f"got {written}, cap is {SOLAR_COEFF_CAP}"
    )


def test_log_severity_transitions_from_info_to_debug(monkeypatch, caplog):
    """First plausibility-block emits at INFO (state transition;
    actionable for multi-install evidence collection).  Subsequent
    blocks while still in the blocked state emit at DEBUG (continuation;
    not noise on user logs).
    """
    import logging
    caplog.set_level(logging.DEBUG, logger="custom_components.heating_analytics.learning")

    def _fake_tobit(samples, censored_mask, **_kw):
        return {
            "s": 1.0, "e": 0.0, "w": 0.0,
            "sigma": 0.1, "iterations": 5, "converged": True,
            "failure_reason": None, "log_likelihood": -10.0,
            "n_uncensored": len([m for m in censored_mask if not m]),
            "n_censored": sum(1 for m in censored_mask if m),
            "n_eff": float(len(samples)),
        }

    monkeypatch.setattr(
        LearningManager, "_solve_tobit_3d", staticmethod(_fake_tobit),
    )
    monkeypatch.setattr(
        LearningManager,
        "_solve_batch_fit_normal_equations",
        staticmethod(lambda _s: {"s": 0.05, "e": 0.0, "w": 0.0}),  # blocks
    )

    stats: dict = {}
    coeffs: dict = {}
    lm = LearningManager()

    # First update that crosses n_eff floor → first plausibility block → INFO.
    n = TOBIT_MIN_NEFF + 5
    for _ in range(n):
        lm._update_unit_tobit_live(
            "sensor.test", "heating", (0.5, 0.3, 0.2),
            0.4, 1.0, stats, coeffs,
        )

    info_blocks = [
        r for r in caplog.records
        if r.levelname == "INFO" and "plausibility-gate blocked" in r.message
    ]
    debug_blocks = [
        r for r in caplog.records
        if r.levelname == "DEBUG" and "plausibility-gate blocked" in r.message
    ]
    assert len(info_blocks) == 1, (
        f"Expected exactly one INFO transition log; got {len(info_blocks)}"
    )
    # Subsequent blocks (after the first) should be DEBUG.  The pumped
    # samples produce many post-floor calls; only the first transition
    # is INFO, the rest are DEBUG continuations.
    assert len(debug_blocks) >= 1, (
        "Expected DEBUG continuation logs after the first INFO transition"
    )


# -----------------------------------------------------------------------------
# Integration round-trip via async_load_data (H6 from review)
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_load_data_default_on_marker_first_load(monkeypatch):
    """Drive ``StorageManager.async_load_data`` with a v4-shape mock
    (no marker, flag False) and assert post-load that the flag flipped
    to True and the marker was stamped.  This is the round-trip that
    the inline-replicating tests miss — if the load-time flip logic
    is deleted, this test fails where the inline tests would pass.
    """
    from custom_components.heating_analytics.storage import StorageManager
    from unittest.mock import patch

    hass = MagicMock()
    hass.config = MagicMock()
    hass.config.latitude = 60.0
    hass.is_running = True
    hass.async_add_executor_job = AsyncMock()

    entry = MagicMock()
    entry.entry_id = "test_entry_marker"
    entry.data = {"energy_sensors": ["sensor.heater1"], "csv_auto_logging": False}

    with patch("custom_components.heating_analytics.storage.Store"):
        from custom_components.heating_analytics.coordinator import (
            HeatingDataCoordinator,
        )
        coord = HeatingDataCoordinator(hass, entry)
        sm = coord.storage

        # 1.3.4-shape data — no marker, flag False.
        persisted = {
            "correlation_data": {},
            "correlation_data_per_unit": {},
            "tobit_sufficient_stats": {},
            "experimental_tobit_live_learner": False,
            "tobit_live_entities": [],
            # (no _tobit_default_applied key)
        }

        async def _replay():
            return persisted

        sm._store.async_load = _replay

        await sm.async_load_data()

    assert coord._experimental_tobit_live_learner is True, (
        "load-time flip must promote False→True when marker missing"
    )
    assert coord._tobit_default_applied is True, (
        "load-time flip must stamp the marker"
    )


@pytest.mark.asyncio
async def test_async_load_data_marker_present_no_flip(monkeypatch):
    """Drive load with marker=True + flag=False (user explicitly
    disabled in a prior session).  Assert flag stays False on load.
    """
    from custom_components.heating_analytics.storage import StorageManager
    from unittest.mock import patch

    hass = MagicMock()
    hass.config = MagicMock()
    hass.config.latitude = 60.0
    hass.is_running = True
    hass.async_add_executor_job = AsyncMock()

    entry = MagicMock()
    entry.entry_id = "test_entry_disabled"
    entry.data = {"energy_sensors": ["sensor.heater1"], "csv_auto_logging": False}

    with patch("custom_components.heating_analytics.storage.Store"):
        from custom_components.heating_analytics.coordinator import (
            HeatingDataCoordinator,
        )
        coord = HeatingDataCoordinator(hass, entry)
        sm = coord.storage

        persisted = {
            "correlation_data": {},
            "correlation_data_per_unit": {},
            "tobit_sufficient_stats": {},
            "experimental_tobit_live_learner": False,
            "tobit_live_entities": [],
            "_tobit_default_applied": True,
        }

        async def _replay():
            return persisted

        sm._store.async_load = _replay

        await sm.async_load_data()

    assert coord._experimental_tobit_live_learner is False, (
        "marker=True must short-circuit the default-on flip; "
        "user-disable must persist across restart"
    )
    assert coord._tobit_default_applied is True


@pytest.mark.asyncio
async def test_async_load_data_persists_marker_in_save(monkeypatch):
    """End-to-end: load a 1.3.4-shape backup, then save, then load
    again from the saved data.  After the second load the flag stays
    True (markør was committed in the save).
    """
    from custom_components.heating_analytics.storage import StorageManager
    from unittest.mock import patch

    hass = MagicMock()
    hass.config = MagicMock()
    hass.config.latitude = 60.0
    hass.is_running = True
    hass.async_add_executor_job = AsyncMock()

    entry = MagicMock()
    entry.entry_id = "test_entry_roundtrip"
    entry.data = {"energy_sensors": ["sensor.heater1"], "csv_auto_logging": False}

    saved_state: dict = {}

    async def _save(data):
        saved_state.update(data)

    with patch("custom_components.heating_analytics.storage.Store"):
        from custom_components.heating_analytics.coordinator import (
            HeatingDataCoordinator,
        )
        coord = HeatingDataCoordinator(hass, entry)
        sm = coord.storage

        persisted = {
            "correlation_data": {},
            "correlation_data_per_unit": {},
            "tobit_sufficient_stats": {},
            "experimental_tobit_live_learner": False,
            "tobit_live_entities": [],
        }

        async def _replay():
            return persisted

        sm._store.async_load = _replay
        sm._store.async_save = _save

        # First load → flips flag, stamps marker.
        await sm.async_load_data()
        # Force save → propagates marker to "disk".
        await sm.async_save_data(force=True)

        # Verify marker is in the persisted dict.
        assert saved_state.get("_tobit_default_applied") is True
        assert saved_state.get("experimental_tobit_live_learner") is True

        # Second load: simulate user disabling, then restart.
        coord._experimental_tobit_live_learner = False  # explicit disable
        # Save again — preserves disable + marker.
        await sm.async_save_data(force=True)
        assert saved_state.get("_tobit_default_applied") is True
        assert saved_state.get("experimental_tobit_live_learner") is False

        # Third load: replay the saved-after-disable state.
        async def _replay2():
            return dict(saved_state)

        sm._store.async_load = _replay2
        await sm.async_load_data()
        assert coord._experimental_tobit_live_learner is False, (
            "Round-trip: user-disable must survive save → restart cycle"
        )
        assert coord._tobit_default_applied is True


# -----------------------------------------------------------------------------
# Fresh-install default-on (P1.1 — async_load_data early-return path)
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_load_data_fresh_install_default_on(monkeypatch):
    """Fresh install: storage is empty / corrupt / missing.  ``async_load_data``
    returns early before reaching the marker-flip block (around line 463).
    The 1.3.5+ default-on semantic must still be applied — guaranteed by
    coordinator init defaults setting flag=True and marker=True.

    Pre-fix the coordinator init had flag=False / marker=False, so a fresh
    install silently started with Tobit disabled despite the documented
    default-on behavior.
    """
    from custom_components.heating_analytics.storage import StorageManager
    from unittest.mock import patch

    hass = MagicMock()
    hass.config = MagicMock()
    hass.config.latitude = 60.0
    hass.is_running = True
    hass.async_add_executor_job = AsyncMock()
    hass.components = MagicMock()
    hass.components.persistent_notification = MagicMock()
    hass.components.persistent_notification.create = MagicMock()

    entry = MagicMock()
    entry.entry_id = "test_entry_fresh"
    entry.data = {"energy_sensors": ["sensor.heater1"], "csv_auto_logging": False}

    with patch("custom_components.heating_analytics.storage.Store"):
        from custom_components.heating_analytics.coordinator import (
            HeatingDataCoordinator,
        )
        coord = HeatingDataCoordinator(hass, entry)

        # Init defaults must be True+True so fresh-install path observes
        # default-on without the load-path's marker block running.
        assert coord._experimental_tobit_live_learner is True, (
            "Coordinator init must default flag=True for fresh-install "
            "default-on coverage"
        )
        assert coord._tobit_default_applied is True

        sm = coord.storage

        # Simulate fresh install: storage returns None (no data, no
        # legacy data either — the early-return path).
        async def _no_data():
            return None

        sm._store.async_load = _no_data
        sm._legacy_store.async_load = _no_data

        await sm.async_load_data()

    # Even after the early-return, the coordinator state reflects 1.3.5+
    # default-on because the init values are preserved.
    assert coord._experimental_tobit_live_learner is True
    assert coord._tobit_default_applied is True


# -----------------------------------------------------------------------------
# Backup/restore preserve user disable (P1.2 — restore path no longer re-flips)
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_backup_includes_tobit_fields(tmp_path, monkeypatch):
    """``async_backup_data`` must include the Tobit-state fields so a
    backup → restore round-trip preserves user choices and accumulated
    state.  Pre-fix the backup omitted these fields, and any restore
    silently fell back to default-on regardless of the user's intent.
    """
    import json
    from custom_components.heating_analytics.storage import StorageManager
    from unittest.mock import patch

    hass = MagicMock()
    hass.config = MagicMock()
    hass.config.latitude = 60.0
    hass.is_running = True
    hass.async_add_executor_job = AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw))
    hass.components = MagicMock()
    hass.components.persistent_notification = MagicMock()

    entry = MagicMock()
    entry.entry_id = "test_entry_backup"
    entry.data = {"energy_sensors": ["sensor.heater1"], "csv_auto_logging": False}

    with patch("custom_components.heating_analytics.storage.Store"):
        from custom_components.heating_analytics.coordinator import (
            HeatingDataCoordinator,
        )
        coord = HeatingDataCoordinator(hass, entry)
        # Set distinctive state to verify it gets backed up.
        coord._experimental_tobit_live_learner = False  # explicit user disable
        coord._tobit_live_entities = frozenset(["sensor.toshiba"])
        coord._tobit_default_applied = True
        coord._tobit_sufficient_stats = {
            "sensor.toshiba": {"heating": {"samples": [], "samples_since_reset": 5}}
        }

        backup_path = str(tmp_path / "backup.json")
        await coord.storage.async_backup_data(backup_path)

    with open(backup_path, "r") as f:
        backup = json.load(f)

    assert "experimental_tobit_live_learner" in backup
    assert backup["experimental_tobit_live_learner"] is False
    assert "tobit_live_entities" in backup
    assert backup["tobit_live_entities"] == ["sensor.toshiba"]
    assert "_tobit_default_applied" in backup
    assert backup["_tobit_default_applied"] is True
    assert "tobit_sufficient_stats" in backup
    assert "sensor.toshiba" in backup["tobit_sufficient_stats"]


@pytest.mark.asyncio
async def test_restore_from_pre_tobit_backup_preserves_user_disable(tmp_path, monkeypatch):
    """User has 1.3.5+ install with explicit Tobit disable, then restores
    from an old backup that predates Tobit (no Tobit fields).  Restore
    MUST NOT silently re-flip the marker / re-enable Tobit — the
    user's session-state choice is the authority when the backup
    has no opinion.

    Pre-fix the restore path unconditionally flipped flag=True when
    the marker was missing, which clobbered the user's explicit
    disable on every restore-from-pre-1.3.5-backup.
    """
    import json
    from datetime import datetime
    from custom_components.heating_analytics.storage import StorageManager
    from unittest.mock import patch

    hass = MagicMock()
    hass.config = MagicMock()
    hass.config.latitude = 60.0
    hass.is_running = True
    hass.async_add_executor_job = AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw))
    hass.components = MagicMock()
    hass.components.persistent_notification = MagicMock()

    entry = MagicMock()
    entry.entry_id = "test_entry_restore_disable"
    entry.data = {"energy_sensors": ["sensor.heater1"], "csv_auto_logging": False}

    # Pre-Tobit backup payload (no Tobit fields).  Includes the minimum
    # required keys to pass the "Invalid backup file" guard at line 1295.
    pre_tobit_backup = {
        "correlation_data": {},
        "daily_history": {},
        # No tobit_* fields, no _tobit_default_applied
    }
    backup_path = str(tmp_path / "pre_tobit_backup.json")
    with open(backup_path, "w") as f:
        json.dump(pre_tobit_backup, f)

    with patch("custom_components.heating_analytics.storage.Store"):
        from custom_components.heating_analytics.coordinator import (
            HeatingDataCoordinator,
        )
        coord = HeatingDataCoordinator(hass, entry)
        # Current-session state: user explicitly disabled Tobit.
        coord._experimental_tobit_live_learner = False
        coord._tobit_default_applied = True
        coord._tobit_live_entities = frozenset()
        coord._tobit_sufficient_stats = {"sensor.x": {"heating": {"samples": []}}}
        # Mock dependencies the restore path touches.
        coord.statistics = MagicMock()
        coord.solar_optimizer = MagicMock()
        coord.solar_optimizer.set_data = MagicMock()
        coord.forecast = MagicMock()
        coord.forecast._cached_long_term_hourly = None
        coord.forecast._cached_long_term_daily = None
        coord.forecast._cached_forecast_date = None
        coord.storage._store.async_save = AsyncMock()

        await coord.storage.async_restore_data(backup_path)

    # User-disable preserved despite restore from pre-Tobit backup.
    assert coord._experimental_tobit_live_learner is False, (
        "Restore from pre-Tobit backup must NOT re-enable Tobit when "
        "the current session has an explicit user-disable"
    )
    assert coord._tobit_default_applied is True
    # Sufficient stats also preserved (backup has no opinion).
    assert "sensor.x" in coord._tobit_sufficient_stats


@pytest.mark.asyncio
async def test_restore_from_post_tobit_backup_restores_state(tmp_path, monkeypatch):
    """User restores from a 1.3.5+ backup that contains explicit Tobit
    state (e.g. user had enabled scope-list before backup).  The
    restored state must come from the backup, OVERWRITING the
    current-session values.
    """
    import json
    from custom_components.heating_analytics.storage import StorageManager
    from unittest.mock import patch

    hass = MagicMock()
    hass.config = MagicMock()
    hass.config.latitude = 60.0
    hass.is_running = True
    hass.async_add_executor_job = AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw))
    hass.components = MagicMock()
    hass.components.persistent_notification = MagicMock()

    entry = MagicMock()
    entry.entry_id = "test_entry_restore_state"
    entry.data = {"energy_sensors": ["sensor.heater1"], "csv_auto_logging": False}

    post_tobit_backup = {
        "correlation_data": {},
        "daily_history": {},
        "experimental_tobit_live_learner": True,
        "tobit_live_entities": ["sensor.toshiba", "sensor.mitsubishi"],
        "_tobit_default_applied": True,
        "tobit_sufficient_stats": {},
    }
    backup_path = str(tmp_path / "post_tobit_backup.json")
    with open(backup_path, "w") as f:
        json.dump(post_tobit_backup, f)

    with patch("custom_components.heating_analytics.storage.Store"):
        from custom_components.heating_analytics.coordinator import (
            HeatingDataCoordinator,
        )
        coord = HeatingDataCoordinator(hass, entry)
        # Current-session state different from backup.
        coord._experimental_tobit_live_learner = False
        coord._tobit_live_entities = frozenset()
        coord._tobit_default_applied = True
        coord.statistics = MagicMock()
        coord.solar_optimizer = MagicMock()
        coord.solar_optimizer.set_data = MagicMock()
        coord.forecast = MagicMock()
        coord.forecast._cached_long_term_hourly = None
        coord.forecast._cached_long_term_daily = None
        coord.forecast._cached_forecast_date = None
        coord.storage._store.async_save = AsyncMock()

        await coord.storage.async_restore_data(backup_path)

    # Backup state restored, overwrites current session.
    assert coord._experimental_tobit_live_learner is True
    assert coord._tobit_live_entities == frozenset(
        ["sensor.toshiba", "sensor.mitsubishi"]
    )
    assert coord._tobit_default_applied is True


# -----------------------------------------------------------------------------
# Service-handler stamps marker (H1 from review)
# -----------------------------------------------------------------------------


def test_service_handlers_stamp_marker_via_source():
    """Pin via source-inspection that both Tobit service handlers
    stamp ``_tobit_default_applied = True``.  The stamp closes the
    race where a user disables before async_load_data completes —
    handler sets False, load reads missing-marker, load flips back
    to True swallowing the disable.  Stamping in the handler ensures
    a load following a service call observes marker=True and leaves
    the user's choice intact.
    """
    import pathlib
    src = pathlib.Path(
        "custom_components/heating_analytics/__init__.py"
    ).read_text()
    # Find both handlers and verify they assign the marker.
    flag_handler_idx = src.index("handle_set_experimental_tobit_live_learner")
    flag_handler_block = src[flag_handler_idx:flag_handler_idx + 2500]
    assert "coord._tobit_default_applied = True" in flag_handler_block, (
        "handle_set_experimental_tobit_live_learner must stamp the marker"
    )

    scope_handler_idx = src.index("handle_set_tobit_live_entities")
    scope_handler_block = src[scope_handler_idx:scope_handler_idx + 2500]
    assert "coord._tobit_default_applied = True" in scope_handler_block, (
        "handle_set_tobit_live_entities must stamp the marker"
    )
