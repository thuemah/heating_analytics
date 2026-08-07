"""Tests for the battery-recommendation noise floor (#1066).

Before this, every battery sweep picked its winner with ``< best - 1e-6``
and reported it as a recommendation regardless of margin, so a converged
install sat at ``review_recommended`` permanently over differences in the
milliwatt-hour range.  None of the three battery verdicts had any
assertion on it anywhere in the suite, which is why the behaviour could
drift this far unnoticed.

Four layers, tested separately:

* ``paired_loss_improvement`` — the pure statistic.
* ``_assess_battery_bias`` — the bias flag's sample-size gate, dispersion,
  and the relative-deviation filter against current consumption.
* ``battery_decay_verdict`` — which of the two readings owns the answer.
* ``summary`` — that the bias is routed into ``any_action`` exactly once.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from custom_components.heating_analytics.const import (
    BATTERY_BIAS_MIN_HOURS,
    BATTERY_RECOMMENDATION_MIN_T,
    BATTERY_RESIDUAL_BIAS_KWH,
    BATTERY_RESIDUAL_BIAS_RELATIVE,
    SOLAR_BATTERY_DECAY,
)
from custom_components.heating_analytics.diagnostics import (
    DiagnosticsEngine,
    _assess_battery_bias,
    battery_decay_verdict,
    battery_feedback_verdict,
    paired_loss_improvement,
)


# ---------------------------------------------------------------------
# The paired statistic
# ---------------------------------------------------------------------

class TestPairedLossImprovement:
    """The honest test on a paired replay is the per-hour difference.

    Both replays cover the same hours under different parameters, so the
    hour-to-hour variance cancels and what remains is the parameter's
    own effect.  Comparing two aggregate RMSEs throws that away.
    """

    def test_consistent_improvement_is_significant(self):
        """Every hour improved by a similar amount -> real effect."""
        baseline = [1.0 + 0.01 * (i % 7) for i in range(200)]
        candidate = [b * 0.5 for b in baseline]
        result = paired_loss_improvement(baseline, candidate)
        assert result["significant"] is True
        assert result["t_statistic"] >= BATTERY_RECOMMENDATION_MIN_T
        assert result["mean_improvement"] > 0

    def test_tiny_improvement_swamped_by_noise_is_not_significant(self):
        """The case that motivated the issue.

        A minute average gain buried in large hour-to-hour swings: the
        aggregate RMSE moves, but the per-hour evidence does not support
        a claim.  Alternating signs give a large dispersion around a
        near-zero mean.
        """
        baseline = [1.0 if i % 2 == 0 else 0.1 for i in range(200)]
        candidate = [
            (0.1 if i % 2 == 0 else 1.0) - 0.0001 for i in range(200)
        ]
        result = paired_loss_improvement(baseline, candidate)
        assert result["significant"] is False

    def test_worse_candidate_is_never_significant(self):
        baseline = [0.5] * 100
        candidate = [0.9 + 0.01 * (i % 5) for i in range(100)]
        result = paired_loss_improvement(baseline, candidate)
        assert result["mean_improvement"] < 0
        assert result["significant"] is False

    def test_identical_candidate_declines_rather_than_dividing_by_zero(self):
        """An exact tie has zero dispersion; it is not evidence."""
        residuals = [0.3, -0.2, 0.5, 0.1] * 25
        result = paired_loss_improvement(residuals, list(residuals))
        assert result["significant"] is False
        assert result["t_statistic"] is None
        assert result["std_error"] == 0.0

    def test_too_few_paired_hours_declines(self):
        assert paired_loss_improvement([0.5], [0.1])["significant"] is False
        assert paired_loss_improvement([], [])["significant"] is False

    def test_pairs_are_truncated_to_the_shorter_list(self):
        """Defensive: misaligned lengths must not raise."""
        result = paired_loss_improvement([1.0] * 10, [0.5] * 4)
        assert result["n_paired_hours"] == 4

    def test_selection_penalty_suppresses_argmin_false_positives(self):
        """The screen must survive being handed the argmin of a noise grid.

        The candidate reaching this function is the winner of a sweep over
        many candidates, tested on the residuals that selected it. Without
        a selection penalty the bar is met by chance a large fraction of
        the time — the review measured 56 % against a pure-noise null, and
        the milder construction below reproduces ~19 %.

        Constructed with ZERO true effect: every candidate is the baseline
        plus independent noise, so any "improvement" is selection artefact.
        """
        import random

        rng = random.Random(7)

        def trial(*, penalty: bool) -> bool:
            base = [rng.gauss(0, 0.15) for _ in range(180)]
            cands = [
                [b + rng.gauss(0, 0.02) for b in base] for _ in range(110)
            ]
            best = min(cands, key=lambda c: sum(r * r for r in c))
            return paired_loss_improvement(
                base, best,
                n_candidates_considered=110 if penalty else 1,
            )["significant"]

        unpenalised = sum(trial(penalty=False) for _ in range(120))
        penalised = sum(trial(penalty=True) for _ in range(120))

        assert unpenalised > 10, (
            "the null construction must actually produce false positives, "
            f"or this test proves nothing (got {unpenalised}/120)"
        )
        assert penalised <= 2, (
            "selection penalty must suppress argmin-driven false positives "
            f"(got {penalised}/120)"
        )

    def test_penalty_scales_with_candidate_count(self):
        """More candidates searched -> higher bar, monotonically."""
        base = [0.5] * 60
        cand = [0.4] * 60
        t1 = paired_loss_improvement(base, cand, n_candidates_considered=1)
        t110 = paired_loss_improvement(base, cand, n_candidates_considered=110)
        assert t110["threshold_applied"] > t1["threshold_applied"]
        assert t1["threshold_applied"] == BATTERY_RECOMMENDATION_MIN_T

    def test_serial_correlation_inflates_the_standard_error(self):
        """Consecutive EMA-driven hours are not independent draws.

        A strongly autocorrelated difference series must be judged more
        harshly than an independent one with the same mean and variance.
        """
        n = 120
        # Independent: alternating; autocorrelated: long runs.
        indep = [0.10 if i % 2 == 0 else -0.02 for i in range(n)]
        runs = [0.10 if (i // 20) % 2 == 0 else -0.02 for i in range(n)]

        def se(diffs):
            base = [1.0] * len(diffs)
            cand = [(1.0 - d) ** 0.5 for d in diffs]
            return paired_loss_improvement(base, cand)

        r_indep, r_runs = se(indep), se(runs)
        assert r_runs["lag1_autocorrelation"] > r_indep["lag1_autocorrelation"]
        assert r_runs["se_inflation_factor"] > r_indep["se_inflation_factor"]

    def test_below_minimum_paired_hours_declines_with_a_reason(self):
        """A fixed t bar is meaningless at n=2 (critical value is 12.7)."""
        result = paired_loss_improvement([1.0] * 5, [0.1] * 5)
        assert result["significant"] is False
        assert result["declined_reason"] == "too_few_paired_hours"

    def test_operates_on_squared_residuals_not_signed_ones(self):
        """Sign of the residual must not matter — only its magnitude.

        A candidate that flips every residual's sign while preserving
        magnitude fits exactly as well, and must not read as an
        improvement.
        """
        baseline = [0.4, -0.3, 0.6, -0.2] * 30
        flipped = [-r for r in baseline]
        result = paired_loss_improvement(baseline, flipped)
        assert result["mean_improvement"] == 0.0
        assert result["significant"] is False


# ---------------------------------------------------------------------
# Verdict mapping
# ---------------------------------------------------------------------

class TestFeedbackVerdict:
    """The k sweep's own improvement was computed and never read.

    Since k was retired the sweep is observability only, so even a clean
    winner is reported as research rather than advice — there is no
    supported way for the user to act on it.
    """

    def test_significant_interior_optimum_is_reported_as_research(self):
        """Named, not recommended: k cannot be set, so nothing to act on."""
        verdict = battery_feedback_verdict(0.4, False, True)
        assert verdict == "research_optimum_k_0.4"
        assert not verdict.startswith("consider_"), (
            "the consider_ prefix is what raises any_action"
        )

    def test_boundary_optimum_names_the_condition_not_the_edge_value(self):
        """The real case: 11 candidates monotonically decreasing to k=1.0.

        A minimum on the last candidate means the sweep ran out of grid,
        not that it located an optimum.  Reporting ``consider_k_1.0``
        presents "the sweep found nothing" as "the sweep found 1.0".
        """
        verdict = battery_feedback_verdict(1.0, True, True)
        assert verdict == "optimum_at_sweep_boundary"
        assert "1.0" not in verdict

    def test_boundary_outranks_significance(self):
        """Even a statistically clean margin at the edge is not a finding."""
        assert battery_feedback_verdict(1.0, True, True) == (
            "optimum_at_sweep_boundary"
        )

    def test_interior_optimum_below_the_noise_floor_is_withheld(self):
        assert battery_feedback_verdict(0.4, False, False) == (
            "improvement_below_noise_floor"
        )

    def test_zero_optimum_is_unchanged(self):
        assert battery_feedback_verdict(0.0, False, False) == (
            "no_improvement_available"
        )

    @pytest.mark.parametrize(
        "opt_k,boundary,significant",
        [
            (1.0, True, True),
            (0.4, False, False),
            (0.0, False, False),
            # The winning case too: no verdict from this sweep may ask
            # for action, because the setting it names cannot be set.
            (0.4, False, True),
            (0.7, False, True),
        ],
    )
    def test_no_verdict_from_this_sweep_ever_asks_for_action(
        self, opt_k, boundary, significant
    ):
        """``any_action`` keys on the ``consider_`` prefix."""
        assert not battery_feedback_verdict(
            opt_k, boundary, significant
        ).startswith("consider_")


class TestDecayVerdict:
    def test_unchanged_recommendation_is_ok(self):
        assert battery_decay_verdict(0.5, 0.5, None) == "ok"

    def test_clean_change_is_recommended(self):
        assert battery_decay_verdict(0.5, 0.7, None) == "consider_decay_0.7"

    @pytest.mark.parametrize(
        "reason",
        ["windows_disagree", "optimum_at_sweep_boundary", "below_noise_floor"],
    )
    def test_any_withheld_reason_replaces_the_recommendation(self, reason):
        """The reason is the verdict — the user learns why, not a value.

        ``windows_disagree`` is the real case on a converged install:
        post-sunset wants 0.7, morning wants 0.5, and surfacing only the
        post-sunset answer presents window-specific overfitting as a
        property of the building.
        """
        verdict = battery_decay_verdict(0.5, 0.7, reason)
        assert verdict == reason
        assert not verdict.startswith("consider_")

    def test_no_evidence_is_not_a_clean_bill_of_health(self):
        """``sweep_produced_evidence=False`` must never read as ``ok``.

        ``best`` is seeded with the live ``(decay, k)``, so a sweep that
        qualified no candidate still yields ``recommended == current``.
        """
        assert battery_decay_verdict(
            0.5, 0.5, None, sweep_produced_evidence=False
        ) == "insufficient_data"

    def test_no_evidence_lets_the_bias_reading_answer(self):
        assert battery_decay_verdict(
            0.5, 0.5, None,
            sweep_produced_evidence=False, bias_assessment="too_fast",
        ) == "too_fast"

    def test_a_sweep_that_ran_keeps_its_own_answer(self):
        """The bias accompanies the sweep's answer; it does not replace it.

        "No decay on this grid fits better" is an answer about decay, and
        it stays the verdict.  The bias reading is carried beside it in
        ``summary.battery_decay["bias_assessment"]`` and raises
        ``any_action`` through its own operand.
        """
        assert battery_decay_verdict(
            0.5, 0.5, None,
            sweep_produced_evidence=True, bias_assessment="too_slow",
        ) == "ok"

    def test_a_withheld_reason_is_never_replaced_by_the_bias(self):
        """The regression this ordering exists to prevent.

        ``withheld_reason`` sits beside a populated ``recommended_decay``
        that is the raw argmin.  Returning ``too_fast`` here leaves 0.7
        in the payload with nothing saying it was withheld — a user sets
        it by hand and ``apply_battery_decay: true`` then refuses to
        write the value the summary appeared to endorse.
        """
        assert battery_decay_verdict(
            0.5, 0.7, "windows_disagree",
            sweep_produced_evidence=True, bias_assessment="too_fast",
        ) == "windows_disagree"

    def test_an_actionable_recommendation_outranks_the_bias(self):
        """When the sweep names a value, that value is the useful answer.

        Both readings describe the same residual; the one that tells the
        user what to change wins.  This is the only case where the bias
        is legitimately not surfaced.
        """
        assert battery_decay_verdict(
            0.5, 0.7, None,
            sweep_produced_evidence=True, bias_assessment="too_fast",
        ) == "consider_decay_0.7"

    @pytest.mark.parametrize("bias", [None, "ok", "insufficient_data"])
    def test_a_quiet_bias_reading_does_not_displace_the_sweep(self, bias):
        assert battery_decay_verdict(
            0.5, 0.5, None, sweep_produced_evidence=True, bias_assessment=bias
        ) == "ok"


# ---------------------------------------------------------------------
# The relative-deviation filter
# ---------------------------------------------------------------------

class TestRelativeBiasFilter:
    """0.05 kWh means different things on a 0.3 kWh and a 3 kWh hour.

    Both gates must pass, so this can only ever narrow what fires
    relative to the absolute-only behaviour it replaces.
    """

    def test_same_absolute_miss_fires_on_a_small_hour(self):
        n = BATTERY_BIAS_MIN_HOURS
        out = _assess_battery_bias([-0.10] * n, [0.30] * n)
        assert out["assessment"] == "too_fast"
        assert out["relative_deviation"] == pytest.approx(0.3333, abs=1e-4)

    def test_same_absolute_miss_abstains_on_a_large_hour(self):
        """0.10 kWh against a 3 kWh hour is 3 % — inside the base model.

        This is the case that made the flag permanent on a converged
        install: it clears the absolute floor comfortably and carries no
        actionable information.
        """
        n = BATTERY_BIAS_MIN_HOURS
        out = _assess_battery_bias([-0.10] * n, [3.0] * n)
        assert out["assessment"] == "ok"
        assert out["relative_deviation"] == pytest.approx(0.0333, abs=1e-4)

    def test_the_absolute_floor_is_still_enforced(self):
        """A large *relative* miss on a tiny hour is still noise.

        0.01 kWh on a 0.06 kWh hour is 17 %, well over the relative gate,
        and 10 Wh of absolute energy.  Retaining the absolute floor is
        what keeps the relative gate from firing on rounding.
        """
        n = BATTERY_BIAS_MIN_HOURS
        out = _assess_battery_bias([-0.01] * n, [0.06] * n)
        assert out["relative_deviation"] > BATTERY_RESIDUAL_BIAS_RELATIVE
        assert abs(out["mean_residual_kwh"]) < BATTERY_RESIDUAL_BIAS_KWH
        assert out["assessment"] == "ok"

    def test_sign_is_preserved_through_the_relative_gate(self):
        n = BATTERY_BIAS_MIN_HOURS
        assert _assess_battery_bias([0.30] * n, [1.0] * n)["assessment"] == "too_slow"
        assert _assess_battery_bias([-0.30] * n, [1.0] * n)["assessment"] == "too_fast"

    def test_thin_sample_abstains_regardless_of_magnitude(self):
        n = BATTERY_BIAS_MIN_HOURS - 1
        out = _assess_battery_bias([-0.90] * n, [1.0] * n)
        assert out["assessment"] == "insufficient_data"

    def test_no_usable_expectation_abstains_rather_than_falling_back(self):
        """Falling back to the absolute reading is the behaviour being fixed."""
        n = BATTERY_BIAS_MIN_HOURS
        out = _assess_battery_bias([-0.30] * n, [0.0] * n)
        assert out["relative_deviation"] is None
        assert out["assessment"] == "insufficient_data"

    def test_ratio_of_means_not_mean_of_ratios(self):
        """One tiny-denominator hour must not dominate the reading.

        Mean of per-hour ratios here is ~1.7 (fires); ratio of means is
        0.06 (abstains).  The latter weights each hour by the energy it
        actually carries.
        """
        residuals = [-0.10] * 10 + [-0.05] * 20
        expected = [0.06] * 10 + [3.0] * 20
        out = _assess_battery_bias(residuals, expected)
        assert out["relative_deviation"] == pytest.approx(0.0331, abs=1e-3)
        assert out["assessment"] == "ok"

    def test_empty_input_abstains(self):
        out = _assess_battery_bias([], [])
        assert out["assessment"] == "insufficient_data"
        assert out["n_hours"] == 0


# ---------------------------------------------------------------------
# The bias flag
# ---------------------------------------------------------------------

# Day counts chosen so qualifying-hour totals straddle
# BATTERY_BIAS_MIN_HOURS (30): 9 x 6 = 54 vs 4 x 6 = 24.
WELL_SAMPLED_DAYS = 9
THIN_DAYS = 4

SUNNY_SOLAR_VECTOR = 0.6   # solar_vector_s on a sunny hour
SUNNY_HOURS = range(8, 16)          # raw solar > 0 -> last_sunny_h = 15
POST_SUNSET_HOURS = range(16, 22)   # the 6-hour replay window per day


def _entry(
    day: int, hour: int, *, raw_solar: float, actual: float, expected: float = 1.0
) -> dict:
    """One hourly-log entry with the fields both battery paths read.

    ``solar_impact_raw_kwh`` drives the sweep's sunny/post-sunset split
    (``raw > 0.01``); ``solar_factor`` + ``solar_impact_kwh`` drive the
    ``battery_residuals`` gate (``solar_factor < 0.01`` and
    ``solar_impact_kwh > 0.05``).  Both must be set coherently or the two
    paths disagree about which hours exist — the trap the first version of
    this fixture fell into, where every hour looked sunny and the sweep
    silently evaluated nothing.
    """
    sunny = raw_solar > 0.01
    return {
        "timestamp": f"2026-07-{day:02d}T{hour:02d}:00:00",
        "hour": hour,
        "temp": 5.0,
        "inertia_temp": 5.0,
        "temp_key": "5",
        "wind_bucket": "normal",
        "solar_factor": SUNNY_SOLAR_VECTOR if sunny else 0.0,
        "solar_vector_s": SUNNY_SOLAR_VECTOR if sunny else 0.0,
        "solar_vector_e": 0.0,
        "solar_vector_w": 0.0,
        # Post-sunset hours carry residual battery charge (that is what
        # makes them qualifying battery hours) but no raw solar input.
        "solar_impact_kwh": raw_solar if sunny else 0.5,
        "solar_impact_raw_kwh": raw_solar,
        "solar_heating_wasted_kwh": 0.0,
        "actual_kwh": actual,
        "expected_kwh": expected,
        "correction_percent": 100.0,
        "auxiliary_active": False,
        "guest_impact_kwh": 0.0,
        "unit_modes": {"sensor.heater1": "heating"},
        "unit_breakdown": {"sensor.heater1": actual},
        "unit_expected_breakdown": {"sensor.heater1": expected},
        "solar_dominant_entities": [],
    }


def _build_log(residual: float, n_days: int) -> list[dict]:
    """``n_days`` full days, each with 8 sunny + 6 post-sunset hours.

    Every post-sunset hour carries the given residual, so the sweep sees
    ``n_days × 6`` evaluable hours and ``battery_residuals`` collects the
    same count.
    """
    log = []
    for d in range(1, n_days + 1):
        for h in SUNNY_HOURS:
            # actual = expected - modelled solar, so implied solar matches
            # what the mocked coefficient predicts and the per-unit
            # ``over_predicting_solar`` / ``under_predicting_solar`` flags
            # stay quiet.  Without this the fixture raises the summary via
            # a per-unit flag and cannot isolate the battery contribution
            # it exists to test.
            log.append(_entry(d, h, raw_solar=0.8, actual=1.0 - SUNNY_SOLAR_VECTOR))
        for h in POST_SUNSET_HOURS:
            log.append(_entry(d, h, raw_solar=0.0, actual=1.0 + residual))
    return log


def _coord_with_days(residual: float, n_days: int):
    """Yields exactly ``n_days × len(POST_SUNSET_HOURS)`` qualifying hours.

    Parameterised by whole days rather than an hour target: the sweep's
    post-sunset window is defined per day, so a partial day would give the
    two paths different hour counts and make the sample-size assertions
    ambiguous.
    """
    coord = MagicMock()
    coord._hourly_log = _build_log(residual, n_days)
    coord.solar_battery_decay = SOLAR_BATTERY_DECAY
    coord.battery_thermal_feedback_k = 0.0
    coord.energy_sensors = ["sensor.heater1"]
    coord.solar_correction_percent = 100.0
    coord.solar_azimuth = 180
    coord.balance_point = 15.0
    coord.screen_config = (True, True, True)
    coord.screen_config_for_entity = MagicMock(return_value=(True, True, True))
    coord.hass.config.latitude = 60.0
    coord.hass.config.longitude = 10.0
    coord.experimental_4d_primary = False
    coord.solar_enabled = True
    coord.solar = MagicMock()
    coord.solar.calculate_unit_coefficient = MagicMock(
        return_value={"s": 1.0, "e": 0.0, "w": 0.0}
    )
    # Must return a real float: the sunny hours in this fixture now reach
    # the per-unit coefficient accumulation, and a MagicMock return value
    # propagates into `mean_delta < -0.1` as a TypeError.  The dot product
    # is what the real method computes.
    coord.solar.calculate_unit_solar_impact = MagicMock(
        side_effect=lambda pot, coeff: (
            pot[0] * coeff.get("s", 0.0)
            + pot[1] * coeff.get("e", 0.0)
            + pot[2] * coeff.get("w", 0.0)
        )
    )
    return coord


def _battery_health(residual: float, n_days: int) -> dict:
    result = DiagnosticsEngine(
        _coord_with_days(residual, n_days)
    ).diagnose_solar(days_back=90)
    return result["global"]["battery_decay_health"]


class TestBiasFlagSampleGate:
    """``too_fast`` / ``too_slow`` used to fire on any sample size."""

    def test_clear_bias_over_enough_hours_still_fires(self):
        """The gate must not silence a genuine, well-sampled bias."""
        health = _battery_health(-0.30, WELL_SAMPLED_DAYS)
        assert health["assessment"] == "too_fast"

    def test_same_bias_on_a_thin_sample_abstains(self):
        """Identical mean, too few hours -> no claim.

        ``insufficient_data`` deliberately, not ``ok``: no evidence is
        not evidence of health.
        """
        health = _battery_health(-0.30, THIN_DAYS)
        assert health["assessment"] == "insufficient_data"

    def test_positive_bias_reads_as_too_slow(self):
        health = _battery_health(0.30, WELL_SAMPLED_DAYS)
        assert health["assessment"] == "too_slow"

    def test_residual_inside_the_threshold_is_ok(self):
        health = _battery_health(BATTERY_RESIDUAL_BIAS_KWH / 2, WELL_SAMPLED_DAYS)
        assert health["assessment"] == "ok"

    def test_dispersion_is_reported_alongside_the_mean(self):
        """A reader must be able to tell a real bias from an artefact."""
        health = _battery_health(-0.30, WELL_SAMPLED_DAYS)
        assert "std_residual_kwh" in health
        assert "std_error_kwh" in health
        assert health["qualifying_post_sunset_hours"] >= BATTERY_BIAS_MIN_HOURS

    def test_thresholds_are_echoed_not_hidden(self):
        """The bare ±0.05 literal is now named and surfaced."""
        health = _battery_health(-0.30, WELL_SAMPLED_DAYS)
        assert health["bias_threshold_kwh"] == BATTERY_RESIDUAL_BIAS_KWH
        assert health["min_hours_for_assessment"] == BATTERY_BIAS_MIN_HOURS


class TestBiasIsRoutedNotSuppressed:
    """The bias reading reaches ``any_action`` through the decay verdict.

    ``battery_decay_*`` is filtered out of ``bool(global_flags)`` so there
    is one path into ``any_action`` rather than two.  That is routing, not
    suppression: an earlier revision justified the filter as stopping one
    signal from raising the verdict "twice", which a boolean OR cannot do,
    and the effect was that a well-sampled bias vanished from the summary
    whenever the sweep merely ran.
    """

    def test_well_sampled_bias_reaches_the_summary_beside_the_verdict(self):
        """A real bias raises the summary even when the sweep ran.

        The sweep genuinely runs on this fixture (``WELL_SAMPLED_DAYS × 6``
        evaluable hours) and reaches its own conclusion about *decay*.
        That conclusion is not a finding that the residual is unbiased, so
        the bias reading must still reach ``any_action`` — through its own
        operand, without displacing what the sweep said.
        """
        coord = _coord_with_days(-0.30, WELL_SAMPLED_DAYS)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=90)

        # Precondition: the sweep must actually have run, or this is
        # testing the zero-evidence path below by accident.
        cal = result["global"]["battery_calibration"]
        assert cal["post_sunset_hours_evaluated"] > 0
        assert cal["sweep_produced_evidence"] is True

        flags = result["summary"]["global_flags"]
        assert any(f.startswith("battery_decay_") for f in flags), (
            "the flag itself must still be reported"
        )
        decay = result["summary"]["battery_decay"]
        assert decay["bias_assessment"] == "too_fast"
        assert result["summary"]["verdict"] == "review_recommended", (
            "a well-sampled bias must reach the summary even when the "
            "calibration sweep itself recommends no change"
        )

    def test_the_summary_never_shows_a_withheld_value_as_endorsed(self):
        """The regression a review caught in the first version of this fix.

        On this fixture the sweep recommends a decay and then withholds
        it.  ``recommended_decay`` is populated either way, so if nothing
        in the summary carries the withholding reason a reader takes the
        number as advice — and ``apply_battery_decay: true`` refuses to
        write it.
        """
        coord = _coord_with_days(-0.30, WELL_SAMPLED_DAYS)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=90)

        cal = result["global"]["battery_calibration"]
        withheld = cal.get("recommendation_withheld_reason")
        if not withheld:
            pytest.skip("fixture produced a clean recommendation")

        decay = result["summary"]["battery_decay"]
        assert decay["recommended_decay"] is not None
        assert decay["recommendation_withheld_reason"] == withheld
        assert decay["verdict"] == withheld, (
            "the verdict must name the withholding reason, not be "
            "overwritten by the bias reading"
        )

    def test_the_flags_list_alone_does_not_raise_the_summary(self):
        """Routing, not double-counting: one path, and it is the verdict.

        With the bias below the relative gate the flag is absent and the
        sweep has nothing to say, so nothing raises the summary — the
        state #1066 exists to produce on a converged install.
        """
        coord = _coord_with_days(-0.02, WELL_SAMPLED_DAYS)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=90)

        flags = result["summary"]["global_flags"]
        assert not any(f.startswith("battery_decay_") for f in flags)
        # Whatever the sweep concluded about decay, the bias reading is
        # quiet, so nothing here reaches ``any_action``.
        assert result["summary"]["battery_decay"]["verdict"] not in (
            "too_fast", "too_slow",
        )
        assert result["summary"]["verdict"] == "no_action_needed"

    def test_zero_evidence_sweep_does_not_read_as_a_clean_bill_of_health(self):
        """The regression the review caught (#1066 follow-up).

        ``best`` is seeded with the live ``(decay, k)``, so a sweep that
        qualified no candidate still yields a truthy ``calibration`` whose
        ``recommended_decay == current_decay``.  Reading that as ``"ok"``
        turned "the sweep saw nothing" into "the current value is
        optimal" — and since the decay verdict is the single decay
        contributor to ``any_action``, it silently swallowed a
        well-sampled bias the merge base reported.

        Fixture: every hour carries raw solar, so ``last_sunny_h`` is the
        final hour of each day and the post-sunset window lands on hours
        that do not exist.  ``battery_residuals`` still fills, because its
        gate is ``solar_factor``, not raw solar.
        """
        coord = _coord_with_days(-0.30, WELL_SAMPLED_DAYS)
        for e in coord._hourly_log:
            e["solar_impact_raw_kwh"] = 0.8   # every hour looks sunny
            e["solar_factor"] = 0.0           # ...but none is, to the gate
            e["solar_impact_kwh"] = 0.5
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=90)

        cal = result["global"]["battery_calibration"]
        assert cal["post_sunset_hours_evaluated"] == 0, "fixture precondition"

        health = result["global"]["battery_decay_health"]
        assert health["assessment"] == "too_fast"
        # The bias flag is the only evidence there is, so it must reach
        # the verdict rather than being filtered out on both routes.
        assert result["summary"]["battery_decay"]["verdict"] == "too_fast"
        assert result["summary"]["verdict"] == "review_recommended"

    def test_zero_evidence_and_no_bias_reports_insufficient_data(self):
        """Absent both, the verdict abstains rather than claiming health."""
        coord = _coord_with_days(0.0, WELL_SAMPLED_DAYS)
        for e in coord._hourly_log:
            e["solar_impact_raw_kwh"] = 0.8
            e["solar_factor"] = 0.0
            e["solar_impact_kwh"] = 0.5
            # Forcing solar_factor to 0 makes the sunny hours qualify as
            # battery hours too, and those carry a -0.6 residual by
            # construction.  Zero every residual so this fixture isolates
            # "no evidence from either source".
            e["actual_kwh"] = e["expected_kwh"]
            e["unit_breakdown"] = {"sensor.heater1": e["actual_kwh"]}
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=90)

        assert result["global"]["battery_calibration"][
            "post_sunset_hours_evaluated"
        ] == 0
        assert result["summary"]["battery_decay"]["verdict"] == "insufficient_data"

    def test_the_paired_hours_band_reports_insufficient_not_noise_floor(self):
        """5-9 evaluable hours: the surface populates, the screen declines.

        ``BATTERY_RECOMMENDATION_MIN_PAIRED_HOURS`` is 10, so below it the
        paired screen returns ``too_few_paired_hours`` without measuring
        anything.  ``below_noise_floor`` would claim a measurement was
        made and came up short.  This band was untested — the
        ``declined_reason`` term could be dropped with the suite green.

        Two days x 6 post-sunset hours is 12; trimming to a single day
        plus part of another lands inside the band.
        """
        coord = _coord_with_days(-0.30, 2)
        # Keep 8 post-sunset hours: above the 5-candidate floor, below
        # the 10-hour paired floor.
        post_sunset = [
            e for e in coord._hourly_log if e["solar_impact_raw_kwh"] <= 0.01
        ]
        for e in post_sunset[8:]:
            coord._hourly_log.remove(e)

        result = DiagnosticsEngine(coord).diagnose_solar(days_back=90)
        cal = result["global"]["battery_calibration"]
        assert 5 <= cal["post_sunset_hours_evaluated"] < 10, "fixture band"
        assert cal["qualifying_candidates"] > 0, "the surface did populate"
        assert cal["sweep_produced_evidence"] is False
        assert result["summary"]["battery_decay"]["verdict"] != "below_noise_floor"

    def test_thin_sample_no_longer_reaches_the_summary_at_all(self):
        """A residual too thin to assess must not raise the verdict."""
        coord = _coord_with_days(-0.30, THIN_DAYS)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=90)

        flags = result["summary"]["global_flags"]
        assert not any(f.startswith("battery_decay_") for f in flags)
        assert result["summary"]["battery_decay"]["verdict"] not in (
            "too_fast", "too_slow"
        )


# ---------------------------------------------------------------------
# The sweeps' live-k anchor and off-grid handling
# ---------------------------------------------------------------------

class TestSweepAnchoredOnLiveK:
    """The counterfactual anchor is the live k, not the k=0 grid corner.

    ``residual_live = actual - expected`` where ``expected`` is already
    net of whatever release the live k produced, so anchoring the delta
    at k=0 offsets every candidate by ``release_live - release_0``.  On an
    install running k > 0 that let the sweep "discover" the value already
    in use.  None of this was covered — the whole change could be
    reverted with the suite still green.
    """

    def _sweep(self, k_live, *, wasted=0.0):
        coord = _coord_with_days(-0.02, WELL_SAMPLED_DAYS)
        coord.battery_thermal_feedback_k = k_live
        if wasted:
            # Without saturation-wasted solar every k trajectory is
            # identical and the anchor cannot be observed at all — the
            # trap the first version of this test fell into, where the
            # whole anchoring change could be reverted with it green.
            for e in coord._hourly_log:
                if e["solar_impact_raw_kwh"] > 0.01:
                    e["solar_heating_wasted_kwh"] = wasted
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=90)
        return result["global"]["battery_feedback_sweep"]

    def test_the_live_candidate_scores_exactly_the_logged_residual(self):
        """The defining property of the anchor, on a sweep where k bites.

        ``residual_alt(k_live)`` must reduce to ``actual - expected``,
        because the release delta against a baseline replayed at the same
        k is identically zero.  Anchored at the k=0 corner instead, every
        candidate is offset by ``release_live - release_0`` and the row
        for the live setting disagrees with the baseline it is quoted
        against — inside one payload.
        """
        k_live = 0.5
        sweep = self._sweep(k_live, wasted=0.4)
        # Precondition: k must actually change the trajectories here, or
        # the assertion below is vacuous.
        rmses = {
            k: row["global"]["rmse_kwh"] for k, row in sweep["sweep"].items()
        }
        assert len(set(rmses.values())) > 1, "fixture: k has no effect"

        assert sweep["sweep"][str(k_live)]["global"]["rmse_kwh"] == (
            sweep["global_rmse_at_baseline_kwh"]
        )

    def test_improvement_is_measured_against_the_live_setting(self):
        """``rmse_improvement_kwh`` is baseline minus optimum, both live-anchored."""
        sweep = self._sweep(0.5, wasted=0.4)
        assert sweep["rmse_improvement_kwh"] == pytest.approx(
            sweep["global_rmse_at_baseline_kwh"]
            - sweep["global_rmse_at_optimum_kwh"],
            abs=1e-4,
        )

    def test_baseline_is_the_live_k(self):
        assert self._sweep(0.4)["baseline_k"] == 0.4

    def test_live_k_is_never_reported_as_a_finding(self):
        """The property the anchor exists to guarantee.

        At the live setting the release delta is identically zero, so the
        candidate equal to it scores exactly the logged residual and
        cannot beat itself by the 1e-6 argmin margin.
        """
        for k_live in (0.0, 0.3, 0.7, 1.0):
            sweep = self._sweep(k_live)
            if sweep["empirical_optimum_k"] == k_live:
                assert sweep["rmse_improvement_kwh"] <= 0.0001

    def test_no_improvement_available_is_expressible_at_k_above_zero(self):
        """The old k=0 seed made every sweep report a change.

        With ``best_k`` seeded from the live value, "nothing beats what
        you are running" has a representation.
        """
        sweep = self._sweep(0.5)
        if sweep["empirical_optimum_k"] == 0.5:
            assert battery_feedback_verdict(
                sweep["empirical_optimum_k"],
                sweep["optimum_at_sweep_boundary"],
                sweep["recommendation_significance"].get("significant", False),
                current_k=sweep["baseline_k"],
            ) == "no_improvement_available"

    def test_all_candidates_tying_reports_no_change(self):
        """The argmin seed must be rounded like the values it is compared to.

        Candidate RMSEs come out of ``per_k_results`` already rounded to
        4 dp.  Seeded with the raw baseline, a candidate that ties exactly
        can still round DOWN by up to 5e-5 and clear the 1e-6 argmin
        margin — 50x the epsilon — so a sweep where k changes nothing
        reported a spurious move away from the live setting.

        This fixture has no saturation-wasted solar, so every k
        trajectory is identical by construction and every candidate ties.
        """
        sweep = self._sweep(0.3)
        rmses = {k: r["global"]["rmse_kwh"] for k, r in sweep["sweep"].items()}
        assert len(set(rmses.values())) == 1, "fixture: candidates must tie"
        assert sweep["empirical_optimum_k"] == 0.3
        assert sweep["rmse_improvement_kwh"] == 0.0

    def test_improvement_is_never_reported_as_negative_zero(self):
        """``-0.0`` in the payload reads as a regression to a user."""
        import math
        sweep = self._sweep(0.3)
        assert not math.copysign(1.0, sweep["rmse_improvement_kwh"]) < 0

    def test_the_per_cell_baseline_row_survives_the_cells_cleanup(self):
        """The row the headline numbers are quoted from must keep its cells.

        ``per_cell_at_optimum``'s baseline column is read from the live-k
        row, so stripping that row's ``cells`` while retaining an
        unrelated k=0 row leaves two disagreeing anchors in one payload.
        """
        k_live = 0.5
        sweep = self._sweep(k_live, wasted=0.4)
        if sweep["empirical_optimum_k"] == k_live:
            pytest.skip("fixture produced no change; collapse branch taken")

        rows_with_cells = {
            k for k, row in sweep["sweep"].items() if "cells" in row
        }
        assert rows_with_cells == {
            str(k_live), str(sweep["empirical_optimum_k"]),
        }

    def test_no_change_collapses_on_the_live_row_not_the_zero_row(self):
        """Collapse is keyed on the live k, not the k=0 grid corner."""
        sweep = self._sweep(0.3)
        assert sweep["empirical_optimum_k"] == 0.3, "fixture: no change"
        assert sweep["per_cell_at_optimum"] is None
        rows_with_cells = {
            k for k, row in sweep["sweep"].items() if "cells" in row
        }
        assert rows_with_cells == {"0.3"}

    def test_current_k_is_reported_coerced_everywhere_in_the_payload(self):
        """One attribute, one reported answer.

        Four reads of ``battery_thermal_feedback_k`` reach the response;
        hardening some of them produced a payload carrying three
        different answers for one setting.
        """
        coord = _coord_with_days(-0.02, WELL_SAMPLED_DAYS)
        coord.battery_thermal_feedback_k = "0.7"
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=90)
        sweep = result["global"]["battery_feedback_sweep"]
        assert sweep["current_k"] == 0.0
        assert sweep["baseline_k"] == 0.0
        assert result["global"]["battery_calibration"]["current_k"] == 0.0
        assert result["summary"]["battery_feedback"]["current_k"] == 0.0

    def test_off_grid_live_k_does_not_crash_the_service(self):
        """Regression: unguarded ``per_k_results[str(best_k)]``.

        ``k_live`` is a free config value and ``best_k`` is seeded from
        it, so an off-grid live k that no candidate beats used to reach a
        dict lookup for a row that was never built — KeyError, taking the
        whole ``diagnose_solar`` service with it.
        """
        sweep = self._sweep(0.35)
        assert sweep["baseline_k"] == 0.35
        assert "empirical_optimum_k" in sweep

    @pytest.mark.parametrize("bad", ["0.0", None, "not-a-number", object()])
    def test_non_numeric_live_k_falls_back_to_the_default(self, bad):
        """Both readings of the attribute guard, not just one.

        ``k_live`` feeds ``k_live > 0.0`` inside ``_replay_score`` and the
        apply gate; ``k_live_sweep`` feeds the feedback sweep.  Hardening
        one and leaving the other raised TypeError mid-sweep.
        """
        coord = _coord_with_days(-0.02, WELL_SAMPLED_DAYS)
        coord.battery_thermal_feedback_k = bad
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=90)
        assert result["global"]["battery_feedback_sweep"]["baseline_k"] == 0.0


class TestApplyGateRespectsTheWithholdingChain:
    """``apply_battery_decay: true`` applies the recommendation made.

    A single call used to emit ``recommendation_withheld_reason:
    "windows_disagree"`` and persist that same value to ``entry.data`` in
    the same pass — the payload saying "do not act" while acting.  The
    flag stays explicit and user-driven; what it writes is now the
    recommendation the diagnostic actually made.
    """

    def _run(self, residual, *, apply_it):
        coord = _coord_with_days(residual, WELL_SAMPLED_DAYS)
        coord.entry.data = {}
        result = DiagnosticsEngine(coord).diagnose_solar(
            days_back=90, apply_battery_decay=apply_it
        )
        return coord, result["global"]["battery_calibration"]

    def test_a_withheld_recommendation_is_not_written(self):
        coord, cal = self._run(-0.02, apply_it=True)
        if not cal.get("recommendation_withheld_reason"):
            pytest.skip("fixture produced a clean recommendation")
        assert cal["apply_skipped_reason"] == cal["recommendation_withheld_reason"]
        assert cal.get("applied") is not True
        coord.hass.config_entries.async_update_entry.assert_not_called()

    def test_the_reason_is_surfaced_so_the_user_learns_why(self):
        _, cal = self._run(-0.02, apply_it=True)
        assert cal["apply_requested"] is True
        assert "apply_skipped_reason" in cal

    def test_not_requesting_apply_never_writes(self):
        coord, cal = self._run(-0.02, apply_it=False)
        assert cal["apply_requested"] is False
        assert cal["apply_skipped_reason"] is None
        coord.hass.config_entries.async_update_entry.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
