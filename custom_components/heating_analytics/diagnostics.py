"""DiagnosticsEngine — hosts diagnose_model() and diagnose_solar() extracted from coordinator.py.

Thin-delegate pattern: the engine holds a reference to the coordinator
and reaches back for state.  Public methods are called via delegates
on the coordinator so the external API is unchanged.
"""
from __future__ import annotations

import logging
import math
from datetime import date as _date, datetime, timedelta

from homeassistant.util import dt as dt_util

from .const import (
    ENERGY_GUARD_THRESHOLD,
    BATTERY_BIAS_MIN_HOURS,
    BATTERY_RECOMMENDATION_MIN_PAIRED_HOURS,
    BATTERY_RECOMMENDATION_MIN_T,
    BATTERY_RESIDUAL_BIAS_KWH,
    BATTERY_RESIDUAL_BIAS_RELATIVE,
    DNI_DHI_REAL_SOURCE_DOMINANCE_MIN,
    DNI_DHI_SOURCE_MIX_MIN_HOURS,
    REPAIR_DNI_DHI_OUTAGE_CLEAR_AT,
    REPAIR_DNI_DHI_OUTAGE_MIN_HOURS,
    REPAIR_DNI_DHI_OUTAGE_RAISE_BELOW,
    REPAIR_DNI_DHI_OUTAGE_WINDOW_HOURS,
    MODE_COOLING,
    MODE_DHW,
    MODE_GUEST_COOLING,
    MODE_GUEST_HEATING,
    MODE_HEATING,
    MODE_OFF,
    SOLAR_BATTERY_DECAY,
)
from .helpers import compute_base_ema_step
from .learning import (
    compute_snr_weight,
    evaluate_4d_learning_readiness,
    _solar_coeff_regime,
)
from .solar import (
    SolarCalculator,
    resolve_dni_dhi,
    reconstruct_hour_inputs,
    HOUR_INPUT_FAIL_SUN_BELOW_HORIZON,
)

_LOGGER = logging.getLogger(__name__)

# Elevation buckets for `elevation_diagnostics` (#927).  Half-open
# intervals [lo, hi) covering 0–90 °; the 60–90° bucket is closed at
# 90 to admit zenith samples on equatorial installs.  Norway maxes
# out at ~54 ° solar elevation in late June so the top bucket is
# rarely populated here, but the schema is location-agnostic.
ELEVATION_BUCKETS: tuple[tuple[int, int], ...] = (
    (0, 15), (15, 30), (30, 45), (45, 60), (60, 90),
)


def paired_loss_improvement(
    baseline_residuals: list[float],
    candidate_residuals: list[float],
    *,
    n_candidates_considered: int = 1,
) -> dict:
    """Paired improvement screen for one battery sweep candidate (#1066).

    A **screen, not a calibrated significance test.**  See
    ``const.BATTERY_RECOMMENDATION_MIN_T`` for the full reasoning; the
    short version is that the candidate handed to this function is the
    argmin over the sweep grid and is then tested on the same residuals
    that selected it, so the raw t statistic is not a p-value and must
    not be presented as one.  Two adjustments below make the screen
    harsh enough to be useful anyway.

    The replays are *paired*: the same hours are replayed under two
    parameter settings, so ``baseline_residuals[h]`` and
    ``candidate_residuals[h]`` describe the same hour and differ only by
    the parameter.  That makes the per-hour difference of squared
    residuals the right statistic — it removes the hour-to-hour variance
    that dominates the raw RMSE and would otherwise swamp the effect.

        d_h = residual_baseline[h]^2 - residual_candidate[h]^2

    Positive ``d_h`` means the candidate fit that hour better.

    Deliberately on the **squared-residual** scale rather than the RMSE
    scale: that is where the replay's loss is defined and what the sweep
    minimises.  RMSE improvement remains reported alongside; it simply no
    longer decides whether a recommendation is worth showing.

    Args:
        baseline_residuals: per-hour residuals under the reference
            parameter — the LIVE setting, not an arbitrary grid corner.
        candidate_residuals: per-hour residuals under the candidate.
            Index-aligned with the baseline; see ``_replay_score``.
        n_candidates_considered: how many candidates the argmin chose
            from.  Drives the selection penalty.  The default of 1 means
            "no selection took place" and applies no penalty — only pass
            that for a genuinely pre-specified comparison.

    Returns a dict carrying ``mean_improvement``, ``std_error`` (after
    serial-correlation inflation), ``t_statistic``, the
    ``threshold_applied`` it was judged against, and ``significant``.
    ``significant`` is False below
    ``BATTERY_RECOMMENDATION_MIN_PAIRED_HOURS`` and for a degenerate
    zero-variance difference, which is what an identical candidate
    produces — an exact tie is not evidence of improvement.
    """
    n = min(len(baseline_residuals), len(candidate_residuals))
    # Selection penalty: sqrt(2 ln m) is the leading term in the expected
    # maximum of m standard normals, which is what argmin-then-test faces.
    threshold = BATTERY_RECOMMENDATION_MIN_T
    if n_candidates_considered > 1:
        threshold += (2.0 * math.log(n_candidates_considered)) ** 0.5

    if n < BATTERY_RECOMMENDATION_MIN_PAIRED_HOURS:
        return {
            "n_paired_hours": n,
            "mean_improvement": None,
            "std_error": None,
            "t_statistic": None,
            "threshold_applied": round(threshold, 3),
            "n_candidates_considered": n_candidates_considered,
            "significant": False,
            "declined_reason": "too_few_paired_hours",
        }

    diffs = [
        baseline_residuals[i] * baseline_residuals[i]
        - candidate_residuals[i] * candidate_residuals[i]
        for i in range(n)
    ]
    mean_d = sum(diffs) / n
    # Sample variance (n-1): these hours are a sample of the install's
    # behaviour, not the population of all hours it will ever see.
    var = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)

    if var <= 0.0:
        # Zero dispersion.  Either every hour improved by exactly the
        # same amount (not physically plausible) or the candidate is the
        # baseline.  Neither is evidence, so decline rather than divide.
        return {
            "n_paired_hours": n,
            "mean_improvement": round(mean_d, 8),
            "std_error": 0.0,
            "t_statistic": None,
            "threshold_applied": round(threshold, 3),
            "n_candidates_considered": n_candidates_considered,
            "significant": False,
            "declined_reason": "zero_dispersion",
        }

    # Serial-correlation inflation.  The window is consecutive
    # post-sunset hours driven by a recursive EMA, so d_h carries the
    # autocorrelation of the underlying model residual while var/n
    # assumes independence.  Standard first-order correction on the lag-1
    # autocorrelation; only applied when positive, since negative r1
    # would *deflate* the SE and this is a screen, not an estimator.
    lag1_cov = sum(
        (diffs[i] - mean_d) * (diffs[i + 1] - mean_d) for i in range(n - 1)
    ) / (n - 1)
    r1 = lag1_cov / var if var > 0 else 0.0
    r1 = max(0.0, min(0.95, r1))
    inflation = ((1.0 + r1) / (1.0 - r1)) ** 0.5

    std_err = ((var / n) ** 0.5) * inflation
    t_stat = mean_d / std_err
    return {
        "n_paired_hours": n,
        "mean_improvement": round(mean_d, 8),
        "std_error": round(std_err, 8),
        "t_statistic": round(t_stat, 3),
        "lag1_autocorrelation": round(r1, 3),
        "se_inflation_factor": round(inflation, 3),
        "threshold_applied": round(threshold, 3),
        "n_candidates_considered": n_candidates_considered,
        "significant": t_stat >= threshold,
    }


def _coerce_scalar(value, default: float) -> float:
    """Config-sourced scalar as a float, or ``default`` if it is not one.

    One helper rather than an inline guard per read site, because the
    defect this exists for is *disagreement between read sites*, not any
    single read: ``battery_thermal_feedback_k`` was read four times in
    ``diagnose_solar`` and hardening two of them produced a payload
    reporting three different answers for one setting.

    ``isinstance`` rather than ``float()``/``except``: a numeric string
    means something upstream is wrong, and the configured default is the
    safe reading of "wrong".  ``bool`` is excluded because it is a
    subclass of ``int`` and ``True`` would otherwise silently become
    ``1.0`` — a deliberate tightening of the ``forecast.py`` guard this
    follows, not a copy of it.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def _assess_battery_bias(
    residuals: list[float],
    expected: list[float],
) -> dict:
    """Post-sunset residual bias, filtered as a deviation against current.

    Answers one question: is the post-sunset mean residual far enough
    from zero, *relative to what the model is currently predicting for
    those hours*, to be worth telling the user about?

    **Why relative and not a bare kWh floor.**  The pre-existing gate was
    ``|mean_residual| > BATTERY_RESIDUAL_BIAS_KWH`` (0.05 kWh), an
    absolute number that means very different things on different
    installs and in different seasons: 0.05 kWh against a 0.3 kWh
    post-sunset hour is a 17 % miss and worth acting on; the same 0.05
    kWh against a 3 kWh midwinter hour is under 2 % and is inside the
    noise of the base model itself.  Reporting both as the same finding
    is what made the flag fire on a converged install.  Direct
    application of the Analysis Standards rule that errors are measured
    in their own regime — here the regime is the size of the hour.

    Both gates must pass.  The absolute floor is retained unchanged as a
    documented default, so this can only ever *narrow* what fires
    relative to the previous behaviour, never widen it:

    * ``|mean_residual| > BATTERY_RESIDUAL_BIAS_KWH`` (0.05 kWh)
    * ``|mean_residual| / mean(expected) > BATTERY_RESIDUAL_BIAS_RELATIVE``

    ``mean(expected)`` is the mean predicted consumption over the same
    hours the residuals came from, so the ratio is a like-for-like
    fractional miss on the post-sunset tail.  It is a ratio of means
    rather than a mean of ratios deliberately: the latter is dominated by
    the smallest-denominator hours, which on the post-sunset tail are
    exactly the hours whose absolute error carries least energy.

    Returns the assessment plus every input that produced it, so a reader
    can see *why* it abstained rather than having to infer it.
    ``insufficient_data`` below ``BATTERY_BIAS_MIN_HOURS`` — no evidence
    is not evidence of health.
    """
    n = len(residuals)
    if n == 0:
        return {
            "assessment": "insufficient_data",
            "n_hours": 0,
            "mean_residual_kwh": None,
            "mean_expected_kwh": None,
            "relative_deviation": None,
            "std_residual_kwh": None,
            "std_error_kwh": None,
        }

    mean_residual = sum(residuals) / n
    if n > 1:
        _var = sum((r - mean_residual) ** 2 for r in residuals) / (n - 1)
        std_residual = _var ** 0.5
        std_error_residual = (_var / n) ** 0.5
    else:
        std_residual = None
        std_error_residual = None

    paired = min(n, len(expected))
    mean_expected = (
        sum(expected[:paired]) / paired if paired > 0 else 0.0
    )
    relative = (
        abs(mean_residual) / mean_expected if mean_expected > 0.0 else None
    )

    result = {
        "n_hours": n,
        "mean_residual_kwh": round(mean_residual, 4),
        "mean_expected_kwh": round(mean_expected, 4) if paired else None,
        "relative_deviation": round(relative, 4) if relative is not None else None,
        "std_residual_kwh": (
            round(std_residual, 4) if std_residual is not None else None
        ),
        "std_error_kwh": (
            round(std_error_residual, 4) if std_error_residual is not None else None
        ),
    }

    if n < BATTERY_BIAS_MIN_HOURS:
        result["assessment"] = "insufficient_data"
        return result
    # ``relative is None`` means no usable expectation to divide by.  The
    # absolute reading alone cannot be put in context, so abstain rather
    # than fall back to it — falling back is the behaviour being fixed.
    if relative is None:
        result["assessment"] = "insufficient_data"
        return result
    if (
        abs(mean_residual) > BATTERY_RESIDUAL_BIAS_KWH
        and relative > BATTERY_RESIDUAL_BIAS_RELATIVE
    ):
        # Negative residual = actual < expected = battery under-credits
        # post-sunset = decays too fast.  Positive = opposite.
        result["assessment"] = "too_slow" if mean_residual > 0 else "too_fast"
    else:
        result["assessment"] = "ok"
    return result


def battery_feedback_verdict(
    optimum_k: float,
    optimum_at_sweep_boundary: bool,
    significant: bool,
    current_k: float = 0.0,
) -> str:
    """Verdict for the thermal-feedback (k) sweep (#1066).

    Extracted from the summary block so the suppressing conditions are a
    named, testable mapping rather than a decision procedure buried
    mid-method.  The bug this replaces was precisely that the sweep's own
    ``rmse_improvement_kwh`` was computed, reported, and never read by
    the verdict — an easy thing to miss inline and hard to miss here.

    **This verdict never asks the user to act, and that is deliberate.**
    ``battery_thermal_feedback_k`` was retired in 1.3.5 — there is no UI
    for it, ``coordinator.__init__`` strips the key from ``entry.data`` on
    every init, and the apply path no longer writes it.  A verdict of
    ``consider_k_1.0`` would therefore name a value the user has no
    supported way to set, and it used to raise the summary to
    ``review_recommended`` over exactly that.  The winning candidate is
    reported as ``research_optimum_k_*`` instead: same information, no
    implied instruction, and it does not feed ``any_action``.

    The sweep is kept rather than deleted because it is the evidence base
    for the retirement decision itself.  Deleting it would make "does k
    earn its place" unanswerable with data, and a retirement that cannot
    be re-examined is a worse outcome than one that costs a diagnostic
    block.

    ``current_k`` is what "no change" means.  The first check used to be
    ``optimum_k == 0.0``, which is only the same question on an install
    running the default; on an install that has adopted k > 0, comparing
    against 0.0 makes the sweep either recommend its own live value as a
    finding or report ``no_improvement_available`` for what is in fact a
    proposed *reduction* to zero.

    Order matters: a boundary optimum is reported as such even when the
    margin clears the screen, because "the sweep did not bracket the
    optimum" describes the result more accurately than any statement
    about the edge value.
    """
    if optimum_k == current_k:
        return "no_improvement_available"
    if optimum_at_sweep_boundary:
        return "optimum_at_sweep_boundary"
    if not significant:
        return "improvement_below_noise_floor"
    # ``research_`` prefix, not ``consider_``: the ``any_action`` test is
    # ``verdict.startswith("consider_")``, so the prefix is what decides
    # whether this raises the summary.  Renaming it is the mechanism, not
    # cosmetics — do not "restore" the old string.
    return f"research_optimum_k_{optimum_k}"


def battery_decay_verdict(
    current_decay: float | None,
    recommended_decay: float | None,
    withheld_reason: str | None,
    sweep_produced_evidence: bool = True,
    bias_assessment: str | None = None,
) -> str:
    """Verdict for the (decay, k) calibration sweep (#1066).

    ``withheld_reason`` is computed where the sweep ran — it needs the
    residuals and both window surfaces — and carries the three
    suppressing conditions in priority order.  This maps it onto the
    user-facing verdict so both battery verdicts have the same shape.

    ``sweep_produced_evidence`` is load-bearing and not cosmetic.
    ``best`` is initialised to the *live* ``(decay, k)`` and only replaced
    by a candidate clearing ``MIN_POST_SUNSET_HOURS_FOR_RECOMMENDATION``,
    so a sweep that qualified **no** candidate still produces a truthy
    ``calibration`` block whose ``recommended_decay == current_decay``.
    Reading that as ``"ok"`` turns "the sweep saw nothing" into "the
    sweep says the current value is optimal".

    **Be precise about what this changed: the threshold, not the
    instrument.**  An earlier revision gated on ``hours_evaluated <= 0``.
    The effective condition here is
    ``n_post_sunset >= BATTERY_RECOMMENDATION_MIN_PAIRED_HOURS`` (10) —
    still an hour count, ten instead of zero.  The ``bool(surface)`` term
    at the call site cannot change the answer (``best_rmse`` starts at
    ``inf``, so any qualifying candidate fills ``best_residuals``, and an
    empty surface therefore always declines with
    ``too_few_paired_hours``); it is kept as a guard so the two cannot
    decouple silently, not because it discriminates.  What the change buys
    is real but narrower than "a boolean, not a count": ``<= 0`` missed
    the 1–4 hour window that populated no surface at all, and it missed
    the 5–9 band where the surface populates but the paired screen
    declines before measuring anything — where ``below_noise_floor``
    implies a measurement that was never made and ``insufficient_data``
    is the honest label.

    ``bias_assessment`` is the post-sunset mean-residual reading from
    ``battery_decay_health``, already filtered to a *relative* deviation
    against current consumption (see ``_assess_battery_bias``).  It is
    consulted **only** when the sweep produced no evidence, and it is the
    only evidence there is in that case.

    **It must not displace a sweep answer, and the reason is concrete.**
    An earlier revision returned the bias from the ``withheld_reason``
    branch too.  That branch is reached with a non-null
    ``recommended_decay`` sitting in the summary beside the verdict, so a
    user saw ``recommended_decay: 0.85`` next to ``verdict: too_fast``
    and no trace of the fact that 0.85 had been *withheld* for
    ``windows_disagree`` — the one thing that says "do not act on this
    number".  ``apply_battery_decay: true`` would then refuse to write
    the value the payload appeared to endorse.  The bias still reaches
    ``any_action``, but as its own operand there rather than by
    overwriting this verdict; see the ``any_action`` comment.  Two
    readings of one residual are reported side by side, never one in
    place of the other.
    """
    if not sweep_produced_evidence:
        if bias_assessment in ("too_fast", "too_slow"):
            return bias_assessment
        return "insufficient_data"
    if recommended_decay == current_decay:
        return "ok"
    if withheld_reason:
        return withheld_reason
    return f"consider_decay_{recommended_decay}"


class DiagnosticsEngine:
    """Hosts the diagnose_model and diagnose_solar service implementations."""

    def __init__(self, coordinator) -> None:
        self.coordinator = coordinator

    def diagnose_model(self, days_back: int = 30) -> dict:
        """Analyze the learned model and history for data quality issues.

        Returns a diagnostic dict with monotonicity check, bucket
        population, mode contamination, solar correlation, and
        Track B specifics.  Designed as a service response.
        """
        from .const import MODES_EXCLUDED_FROM_GLOBAL_LEARNING

        result: dict = {}
        # Pre-insert summary so it renders first in the response — populated
        # at the bottom of this method once all sections are computed.
        # Mutating an existing key preserves its insertion-order position
        # (Python 3.7+ dict semantics), so the final assignment does not
        # move it.
        result["summary"] = {}

        # --- 1. Monotonicity check per wind bucket ---
        monotonicity: dict[str, dict] = {}
        for wind_bucket in ("normal", "high_wind", "extreme_wind"):
            # Collect (temp_key_int, kwh) pairs for this bucket.
            points: list[tuple[int, float]] = []
            for temp_key_str, buckets in self.coordinator._correlation_data.items():
                if wind_bucket in buckets:
                    try:
                        points.append((int(temp_key_str), buckets[wind_bucket]))
                    except (ValueError, TypeError):
                        continue
            if len(points) < 2:
                monotonicity[wind_bucket] = {
                    "status": "insufficient_data",
                    "points": len(points),
                }
                continue

            points.sort(key=lambda p: p[0], reverse=True)  # warmest first
            inversions = []
            for i in range(len(points) - 1):
                t_warm, kwh_warm = points[i]
                t_cold, kwh_cold = points[i + 1]
                if kwh_cold < kwh_warm:
                    inversions.append({
                        "from_temp": t_warm,
                        "to_temp": t_cold,
                        "kwh_warm": round(kwh_warm, 4),
                        "kwh_cold": round(kwh_cold, 4),
                        "delta": round(kwh_warm - kwh_cold, 4),
                    })
            monotonicity[wind_bucket] = {
                "status": "monotonic" if not inversions else "inversions_found",
                "points": len(points),
                "temp_range": [points[-1][0], points[0][0]],
                "inversions": inversions,
            }
        result["monotonicity"] = monotonicity

        # --- 2. Bucket population ---
        bucket_pop: dict[str, dict] = {}
        total_observations = 0
        for temp_key_str, buckets in self.coordinator._correlation_data.items():
            for wind_bucket, kwh in buckets.items():
                if wind_bucket not in bucket_pop:
                    bucket_pop[wind_bucket] = {"count": 0, "temp_keys": []}
                bucket_pop[wind_bucket]["count"] += 1
                bucket_pop[wind_bucket]["temp_keys"].append(temp_key_str)
                total_observations += 1

        # Check for under-sampled buckets via learning buffers.
        buffered_buckets = 0
        for temp_key_str, buckets in self.coordinator._learning_buffer_global.items():
            for wind_bucket, samples in buckets.items():
                if samples:
                    buffered_buckets += 1

        bucket_summary = {
            "total_buckets_learned": total_observations,
            "buffered_pending": buffered_buckets,
            "per_wind_bucket": {
                wb: {"bucket_count": info["count"], "temp_range": [
                    min(info["temp_keys"], key=int) if info["temp_keys"] else None,
                    max(info["temp_keys"], key=int) if info["temp_keys"] else None,
                ]}
                for wb, info in bucket_pop.items()
            },
        }
        result["bucket_population"] = bucket_summary

        # --- 3. Mode contamination (from hourly log) ---
        from homeassistant.util import dt as dt_util
        now = dt_util.now()
        cutoff_str = (now - timedelta(days=days_back)).date().isoformat() if days_back else None

        mode_stats: dict[str, dict[str, int]] = {}
        total_hours = 0
        excluded_kwh_total = 0.0

        for entry in self.coordinator._hourly_log:
            ts = entry.get("timestamp", "")
            if cutoff_str and ts[:10] < cutoff_str:
                continue
            total_hours += 1
            day = ts[:10]
            if day not in mode_stats:
                mode_stats[day] = {"total_hours": 0, "excluded_hours": 0, "excluded_kwh": 0.0, "modes": {}}
            mode_stats[day]["total_hours"] += 1

            unit_modes = entry.get("unit_modes", {})
            breakdown = entry.get("unit_breakdown", {})
            for sid, kwh in breakdown.items():
                mode = unit_modes.get(sid, "heating")
                if mode != "heating":
                    mode_stats[day]["modes"][mode] = mode_stats[day]["modes"].get(mode, 0) + 1
                if mode in MODES_EXCLUDED_FROM_GLOBAL_LEARNING:
                    mode_stats[day]["excluded_hours"] += 1
                    mode_stats[day]["excluded_kwh"] += kwh
                    excluded_kwh_total += kwh

        # Summarize — only report days with excluded energy.
        contaminated_days = {
            day: stats for day, stats in mode_stats.items()
            if stats["excluded_hours"] > 0
        }
        result["mode_contamination"] = {
            "days_analyzed": len(mode_stats),
            "total_hours_analyzed": total_hours,
            "contaminated_days": len(contaminated_days),
            "total_excluded_kwh": round(excluded_kwh_total, 2),
            "details": dict(sorted(contaminated_days.items())[-10:]),  # Last 10 affected days
        }

        # --- 4. Solar correlation ---
        # Use a low threshold (> 0.01) to capture any hour with measurable solar.
        # The 0.05 threshold used in learning is too strict for diagnostics —
        # we want to see if there's ANY solar signal in the data.
        solar_errors: list[tuple[float, float]] = []
        for entry in self.coordinator._hourly_log:
            ts = entry.get("timestamp", "")
            if cutoff_str and ts[:10] < cutoff_str:
                continue
            solar_f = entry.get("solar_factor")
            actual = entry.get("actual_kwh")
            expected = entry.get("expected_kwh")
            if solar_f is None or actual is None or expected is None or solar_f <= 0.01:
                continue
            # Subtract excluded-mode (DHW / OFF / guest) energy from actual
            # before computing the residual — mirrors the base_model_health
            # fix.  Without this, DHW or guest hours that overlap with solar
            # hours bias the correlation positive: ``expected`` is a heating-
            # only prediction (excluded modes are not in the learning loop)
            # but ``actual`` is the raw meter sum that includes DHW / guest
            # contributions.  OFF contributes 0 kWh so the subtraction is a
            # no-op for OFF — which is correct (OFF is a stable state, not
            # contamination).
            unit_modes = entry.get("unit_modes", {}) or {}
            unit_breakdown = entry.get("unit_breakdown", {}) or {}
            excluded_kwh = sum(
                kwh for sid, kwh in unit_breakdown.items()
                if unit_modes.get(sid, MODE_HEATING) in MODES_EXCLUDED_FROM_GLOBAL_LEARNING
            )
            adjusted_actual = max(0.0, actual - excluded_kwh)
            solar_errors.append((solar_f, adjusted_actual - expected))

        solar_diag: dict = {"qualifying_hours": len(solar_errors)}

        # Report max solar_factor seen in the period regardless of qualification.
        all_solar_factors = [
            entry.get("solar_factor", 0.0) for entry in self.coordinator._hourly_log
            if (not cutoff_str or entry.get("timestamp", "")[:10] >= cutoff_str)
            and entry.get("solar_factor") is not None
        ]
        if all_solar_factors:
            solar_diag["max_solar_factor_in_period"] = round(max(all_solar_factors), 3)
            solar_diag["hours_with_any_solar"] = sum(1 for f in all_solar_factors if f > 0.0)

        if len(solar_errors) >= 10:
            avg_solar = sum(s for s, _ in solar_errors) / len(solar_errors)
            avg_error = sum(e for _, e in solar_errors) / len(solar_errors)
            # Simple correlation: positive = solar hours have positive error (under-predicted solar reduction)
            n = len(solar_errors)
            mean_s = avg_solar
            mean_e = avg_error
            cov = sum((s - mean_s) * (e - mean_e) for s, e in solar_errors) / n
            var_s = sum((s - mean_s) ** 2 for s, _ in solar_errors) / n
            var_e = sum((e - mean_e) ** 2 for _, e in solar_errors) / n
            denom = (var_s * var_e) ** 0.5
            correlation = round(cov / denom, 3) if denom > 0 else 0.0
            solar_diag["correlation_solar_vs_error"] = correlation
            solar_diag["avg_solar_factor"] = round(avg_solar, 3)
            solar_diag["avg_error_kwh"] = round(avg_error, 3)
            solar_diag["interpretation"] = (
                "Positive correlation: solar hours have higher-than-expected consumption — solar model may be under-subtracting."
                if correlation > 0.15 else
                "Negative correlation: solar hours have lower-than-expected consumption — solar model may be over-subtracting."
                if correlation < -0.15 else
                "No significant correlation — solar model appears well-calibrated."
            )
        result["solar_correlation"] = solar_diag

        # --- 6. Base model health (dark-hour replay) ---
        # For each (temp_key, wind_bucket) pair in the stored correlation
        # model, compute the mean reference kWh over hourly_log entries that
        # are (a) dark (solar_factor < 0.05 → no solar contamination),
        # (b) not aux-active (aux has its own learning path), and (c) not
        # contaminated by excluded modes in any unit.  Compare that empirical
        # dark-hour mean to the stored bucket value.
        #
        # Why dark-hour replay: the stored bucket is the output of a learning
        # loop that includes normalisation of solar impact.  If solar learning
        # is biased, normalisation is biased, and the stored bucket absorbs
        # that bias — a regression against contaminated history would
        # circularly validate the bias.  Dark hours have no solar signal to
        # normalise, so the mean reference kWh on those hours is an
        # independent ground-truth for what the bucket SHOULD have converged
        # to.  This surfaces the contamination pattern a user would otherwise
        # only detect by observing live over-prediction (e.g. cloudy mild day
        # running 20% above forecast).
        #
        # Track C semantics: bucket values for Track C installations are
        # written from synthetic_kwh_el (MPC thermal delivery / per-hour
        # COP), not raw electrical.  Comparing them to actual_kwh from the
        # meter would be apple-to-pear: a partial electrical sensor on a
        # Track C install (where MPC supplements the thermal picture)
        # under-represents what the bucket models, producing spuriously
        # large "inflated" verdicts.  When Track C is enabled AND a per-day
        # track_c_distribution exists, the reference becomes
        # synthetic_kwh_el[hour] for the MPC-managed sensor plus raw
        # actual_kwh contributions from non-MPC sensors.  Days that fell
        # back to Track B (no distribution) keep the raw-actual path.
        # The per-bucket result reports track_c_aware_hours so callers can
        # see the mix.
        DARK_SOLAR_FACTOR_THRESHOLD = 0.05
        MIN_DARK_HOURS_FOR_VERDICT = 10
        BUCKET_DEVIATION_THRESHOLD_PCT = 15.0

        is_track_c_install = bool(
            getattr(self.coordinator, "track_c_enabled", False)
            and getattr(self.coordinator, "mpc_managed_sensor", None)
        )
        mpc_sid = getattr(self.coordinator, "mpc_managed_sensor", None)

        # Per-day distribution cache: {date_str: {hour: synthetic_kwh_el}}.
        # Built lazily on first access per date so non-Track-C installs pay
        # nothing.  An empty dict marks "no distribution available".
        dist_by_day_hour: dict[str, dict[int, float]] = {}

        def _resolve_distribution(day_key: str) -> dict[int, float] | None:
            """Return {hour: synthetic_kwh_el} for a Track C day, or None."""
            if day_key in dist_by_day_hour:
                return dist_by_day_hour[day_key] or None
            day_history = self.coordinator._daily_history.get(day_key, {}) or {}
            raw_dist = day_history.get("track_c_distribution")
            if not raw_dist:
                dist_by_day_hour[day_key] = {}
                return None
            hour_map: dict[int, float] = {}
            for d in raw_dist:
                try:
                    dt_str = d.get("datetime", "")
                    # ISO format: "2026-04-15T13:00:00..."  Hour at chars 11-13.
                    if len(dt_str) >= 13 and dt_str[10] == "T":
                        h = int(dt_str[11:13])
                        hour_map[h] = float(d.get("synthetic_kwh_el", 0.0))
                except (ValueError, TypeError):
                    continue
            dist_by_day_hour[day_key] = hour_map
            return hour_map or None

        def _reference_dark_kwh(entry: dict) -> tuple[float, bool]:
            """Return (kWh value to compare against stored bucket, used_track_c)."""
            actual = entry.get("actual_kwh")
            if actual is None:
                return 0.0, False
            if not is_track_c_install:
                return float(actual), False
            day_key = entry.get("timestamp", "")[:10]
            hour_map = _resolve_distribution(day_key)
            if hour_map is None:
                return float(actual), False
            hour = entry.get("hour", -1)
            synthetic = hour_map.get(hour)
            if synthetic is None:
                # Track C day but this specific hour missing from the
                # distribution.  Fall back to raw rather than guess; the
                # caller sees the mix in track_c_aware_hours.
                return float(actual), False
            breakdown = entry.get("unit_breakdown", {}) or {}
            non_mpc_kwh = sum(
                kwh for sid, kwh in breakdown.items()
                if sid != mpc_sid
            )
            return synthetic + non_mpc_kwh, True

        # (temp_key, wind_bucket) -> {"values": [...], "track_c_count": int}
        dark_accum: dict[tuple[str, str], dict] = {}
        for entry in self.coordinator._hourly_log:
            ts = entry.get("timestamp", "")
            if cutoff_str and ts[:10] < cutoff_str:
                continue
            solar_f = entry.get("solar_factor")
            if solar_f is None or solar_f >= DARK_SOLAR_FACTOR_THRESHOLD:
                continue
            if entry.get("auxiliary_active", False):
                continue
            temp_key = entry.get("temp_key")
            wind_bucket = entry.get("wind_bucket")
            if temp_key is None or wind_bucket is None or entry.get("actual_kwh") is None:
                continue
            ref_kwh, used_track_c = _reference_dark_kwh(entry)
            # Subtract excluded-mode (DHW / OFF / guest) energy rather than
            # dropping the whole hour — mirrors retrain.py:401-408.  An OFF
            # unit contributes 0 kWh so the subtraction is a no-op for it
            # (which is correct: a permanently-off auxiliary VP is a stable
            # state, not contamination).  For DHW / guest the unit's actual
            # contribution is removed so the residual matches the
            # heating-only semantic of the stored bucket.  Track-C's
            # synthetic + non_mpc_kwh path inherits the same correction:
            # any non-MPC unit in an excluded mode has its kWh subtracted.
            unit_modes = entry.get("unit_modes", {}) or {}
            unit_breakdown = entry.get("unit_breakdown", {}) or {}
            excluded_kwh = sum(
                kwh for sid, kwh in unit_breakdown.items()
                if unit_modes.get(sid, MODE_HEATING) in MODES_EXCLUDED_FROM_GLOBAL_LEARNING
            )
            ref_kwh = max(0.0, ref_kwh - excluded_kwh)
            slot = dark_accum.setdefault(
                (temp_key, wind_bucket), {"values": [], "track_c_count": 0}
            )
            slot["values"].append(ref_kwh)
            if used_track_c:
                slot["track_c_count"] += 1

        base_health: dict[str, dict] = {}
        base_flags = {"inflated": [], "deflated": [], "unverifiable": []}
        for temp_key_str, buckets in self.coordinator._correlation_data.items():
            for wind_bucket, stored_kwh in buckets.items():
                slot = dark_accum.get(
                    (temp_key_str, wind_bucket),
                    {"values": [], "track_c_count": 0},
                )
                dark_hours = slot["values"]
                entry_out: dict = {
                    "stored_kwh": round(stored_kwh, 4),
                    "n_dark_hours": len(dark_hours),
                    "track_c_aware_hours": slot["track_c_count"],
                }
                if len(dark_hours) < MIN_DARK_HOURS_FOR_VERDICT:
                    entry_out["verdict"] = "unverifiable"
                    base_flags["unverifiable"].append(
                        {"temp_key": temp_key_str, "wind_bucket": wind_bucket}
                    )
                else:
                    dark_mean = sum(dark_hours) / len(dark_hours)
                    entry_out["actual_dark_mean_kwh"] = round(dark_mean, 4)
                    delta = stored_kwh - dark_mean
                    entry_out["delta_kwh"] = round(delta, 4)
                    # Percentage deviation guarded against near-zero stored
                    # (avoids division blowup at or near balance_point).
                    if abs(dark_mean) > 0.01:
                        delta_pct = 100.0 * delta / dark_mean
                    elif abs(stored_kwh) <= 0.01:
                        delta_pct = 0.0
                    else:
                        delta_pct = float("inf")
                    entry_out["delta_pct"] = round(delta_pct, 1) if delta_pct != float("inf") else None
                    if delta_pct == float("inf"):
                        entry_out["verdict"] = "inflated"
                        base_flags["inflated"].append(
                            {"temp_key": temp_key_str, "wind_bucket": wind_bucket,
                             "delta_pct": None, "stored_kwh": round(stored_kwh, 4),
                             "actual_dark_mean_kwh": round(dark_mean, 4)}
                        )
                    elif delta_pct > BUCKET_DEVIATION_THRESHOLD_PCT:
                        entry_out["verdict"] = "inflated"
                        base_flags["inflated"].append(
                            {"temp_key": temp_key_str, "wind_bucket": wind_bucket,
                             "delta_pct": round(delta_pct, 1)}
                        )
                    elif delta_pct < -BUCKET_DEVIATION_THRESHOLD_PCT:
                        entry_out["verdict"] = "deflated"
                        base_flags["deflated"].append(
                            {"temp_key": temp_key_str, "wind_bucket": wind_bucket,
                             "delta_pct": round(delta_pct, 1)}
                        )
                    else:
                        entry_out["verdict"] = "ok"
                base_health.setdefault(temp_key_str, {})[wind_bucket] = entry_out

        # Sort flag lists by |delta_pct| descending so worst offenders surface
        # first when the caller inspects a truncated view.
        for key in ("inflated", "deflated"):
            base_flags[key].sort(
                key=lambda x: abs(x.get("delta_pct") or 0.0),
                reverse=True,
            )

        result["base_model_health"] = {
            "config": {
                "dark_solar_factor_threshold": DARK_SOLAR_FACTOR_THRESHOLD,
                "min_dark_hours_for_verdict": MIN_DARK_HOURS_FOR_VERDICT,
                "bucket_deviation_threshold_pct": BUCKET_DEVIATION_THRESHOLD_PCT,
            },
            "buckets": base_health,
            "flags": base_flags,
            "summary": {
                "inflated_count": len(base_flags["inflated"]),
                "deflated_count": len(base_flags["deflated"]),
                "unverifiable_count": len(base_flags["unverifiable"]),
            },
        }

        # --- 7. Track B diagnostics ---
        track_b_days: list[dict] = []
        for day_key, day_data in sorted(self.coordinator._daily_history.items()):
            if cutoff_str and day_key < cutoff_str:
                continue
            if "kwh" not in day_data:
                continue
            track_b_entry: dict = {
                "date": day_key,
                "raw_kwh": round(day_data["kwh"], 2),
                "temp_avg": round(day_data.get("temp", 0.0), 1),
                "tdd": round(day_data.get("tdd", 0.0), 2),
                "wind_avg": round(day_data.get("wind", 0.0), 1),
            }
            if "track_c_kwh" in day_data:
                track_b_entry["track_c_kwh"] = round(day_data["track_c_kwh"], 2)
                track_b_entry["track"] = "C"
            elif "track_b_cop_distribution" in day_data:
                track_b_entry["track"] = "B_cop_smeared"
            else:
                track_b_entry["track"] = "B_flat" if self.coordinator.daily_learning_mode else "A"
            if "midnight_indoor_temp" in day_data:
                track_b_entry["midnight_indoor_temp"] = day_data["midnight_indoor_temp"]
            track_b_days.append(track_b_entry)

        result["daily_history"] = {
            "days": len(track_b_days),
            "entries": track_b_days[-30:],  # Last 30 days
        }

        result["config_summary"] = {
            "daily_learning_mode": self.coordinator.daily_learning_mode,
            "track_c_enabled": self.coordinator.track_c_enabled,
            "balance_point": self.coordinator.balance_point,
            "learning_rate": self.coordinator.learning_rate,
            "wind_threshold": self.coordinator.wind_threshold,
            "extreme_wind_threshold": self.coordinator.extreme_wind_threshold,
            "solar_enabled": self.coordinator.solar_enabled,
            "learned_u_coefficient": round(self.coordinator._learned_u_coefficient, 4) if self.coordinator._learned_u_coefficient else None,
            "energy_sensors": self.coordinator.energy_sensors,
        }

        # #855 Option B: runtime counter of Track C MPC outages (days where
        # midnight sync found no distribution and skipped bucket+U updates).
        # Runtime-only; resets on HA restart.  Diagnostic signal for whether
        # the install is seeing frequent MPC unavailability.
        result["track_c_outage_session_count"] = self.coordinator._track_c_outage_count_session

        # --- 0. Summary (top-of-response, populated last) ---
        # Aggregates the headline numbers from each section so callers can
        # read a one-screen verdict before drilling into per-bucket detail.
        # The summary's "verdict" reflects model health only — it ignores
        # mode_contamination (informational) and bucket_population (the
        # underlying bucket counts surface elsewhere).  Noise floor for
        # monotonicity inversions is 0.05 kWh — sub-noise inversions near
        # the balance point (where stored kWh ≈ 0) are excluded from the
        # verdict driver but still counted in ``inversion_count``.
        MONOTONICITY_NOISE_FLOOR_KWH = 0.05
        SOLAR_CORRELATION_NEUTRAL_BAND = 0.15

        total_inv = 0
        total_inv_above_noise = 0
        max_delta = 0.0
        for wb_diag in monotonicity.values():
            for inv in wb_diag.get("inversions", []):
                total_inv += 1
                d = float(inv.get("delta", 0.0))
                if d > max_delta:
                    max_delta = d
                if d > MONOTONICITY_NOISE_FLOOR_KWH:
                    total_inv_above_noise += 1

        base_summary = result["base_model_health"]["summary"]
        total_buckets = (
            base_summary["inflated_count"]
            + base_summary["deflated_count"]
            + base_summary["unverifiable_count"]
            + sum(
                1
                for buckets in base_health.values()
                for entry_out in buckets.values()
                if entry_out.get("verdict") == "ok"
            )
        )
        ok_count = total_buckets - (
            base_summary["inflated_count"]
            + base_summary["deflated_count"]
            + base_summary["unverifiable_count"]
        )
        verifiable_count = total_buckets - base_summary["unverifiable_count"]

        solar_corr = solar_diag.get("correlation_solar_vs_error")
        solar_qual_hours = solar_diag.get("qualifying_hours", 0)

        if total_buckets > 0 and base_summary["unverifiable_count"] / total_buckets > 0.5:
            verdict = "model_unverifiable"
        elif (
            total_inv_above_noise > 0
            or base_summary["inflated_count"] > 0
            or base_summary["deflated_count"] > 0
            or (solar_corr is not None and abs(solar_corr) > SOLAR_CORRELATION_NEUTRAL_BAND)
        ):
            verdict = "issues_found"
        else:
            verdict = "ok"

        result["summary"] = {
            "verdict": verdict,
            "data_window": {
                "days": days_back,
                "hours_analyzed": result["mode_contamination"]["total_hours_analyzed"],
            },
            "monotonicity": {
                "inversion_count": total_inv,
                "inversion_count_above_noise": total_inv_above_noise,
                "max_delta_kwh": round(max_delta, 4),
                "noise_floor_kwh": MONOTONICITY_NOISE_FLOOR_KWH,
            },
            "base_model": {
                "total_buckets": total_buckets,
                "verifiable": verifiable_count,
                "ok": ok_count,
                "inflated": base_summary["inflated_count"],
                "deflated": base_summary["deflated_count"],
                "unverifiable": base_summary["unverifiable_count"],
            },
            "mode_contamination": {
                "contaminated_days": result["mode_contamination"]["contaminated_days"],
                "total_excluded_kwh": result["mode_contamination"]["total_excluded_kwh"],
            },
            "solar": {
                "qualifying_hours": solar_qual_hours,
                "correlation": solar_corr,
                "interpretation": solar_diag.get("interpretation"),
            },
        }

        return result

    def _format_last_batch_fit(self, entity_id: str) -> dict | None:
        """Format the most recent batch-fit-solar summary for diagnose_solar (#884).

        Returns ``None`` when the unit has never been included in any
        batch-fit run.  Persisted across HA restarts via the standard
        save path (``last_batch_fit_per_unit`` in storage).  When the
        most recent run was a top-level skip for this entity (e.g.
        ``weighted_smear_excluded``), the entry includes a ``skip_reason``
        field with empty ``regimes`` — distinguishing "skipped, here's why"
        from "never run".  Successful fits expose timestamp + per-(regime)
        sample count, residual RMSE, before/after coefficients, damping
        applied, and any per-regime skip reason.
        """
        last_per_unit = getattr(self.coordinator, "_last_batch_fit_per_unit", None)
        if not isinstance(last_per_unit, dict):
            return None
        entry = last_per_unit.get(entity_id)
        if not isinstance(entry, dict):
            return None
        out: dict = {
            "timestamp": entry.get("timestamp"),
            "regimes": entry.get("regimes") or {},
        }
        if "skip_reason" in entry:
            out["skip_reason"] = entry["skip_reason"]
        return out


    def diagnose_solar(self, days_back: int = 30, apply_battery_decay: bool = False) -> dict:
        """Analyze per-unit solar coefficient health and global solar model quality.

        Single-pass over hourly_log. Returns a diagnostic dict with per-unit
        coefficient validation, global battery/screen health, and temporal bias.
        Includes battery decay calibration: sweeps decay rates 0.50-0.95 to find
        the optimal value for this building. Designed as a service response (#810).
        """
        from datetime import timedelta, date as _date
        from homeassistant.util import dt as dt_util
        from .const import (
            MODE_HEATING, MODE_COOLING, MODE_OFF, MODE_DHW,
            MODE_GUEST_HEATING, MODE_GUEST_COOLING,
            HARD_OUTLIER_SANITY_MULTIPLIER,
        )
        from .solar import SolarCalculator as _SC

        now = dt_util.now()
        cutoff = (now - timedelta(days=days_back)).date().isoformat()

        # --- Accumulators ---
        # Per-unit: normal equations for implied coefficient (3 windows + 30d total)
        window_boundaries = [days_back, int(days_back * 2 / 3), int(days_back / 3), 0]
        unit_accum: dict[str, dict] = {}  # entity_id -> accumulators

        def _get_unit_accum(eid: str) -> dict:
            if eid not in unit_accum:
                unit_accum[eid] = {
                    # Normal equations for 30d implied coefficient (3x3: S, E, W)
                    "ss": 0.0, "ee": 0.0, "ww": 0.0,
                    "se": 0.0, "sw": 0.0, "ew": 0.0,
                    "sI": 0.0, "eI": 0.0, "wI": 0.0, "n": 0,
                    # 3-window stability (each has own normal eqs)
                    "windows": [
                        {"ss": 0.0, "ee": 0.0, "ww": 0.0,
                         "se": 0.0, "sw": 0.0, "ew": 0.0,
                         "sI": 0.0, "eI": 0.0, "wI": 0.0, "n": 0}
                        for _ in range(3)
                    ],
                    # Delta accumulator
                    "sum_delta": 0.0, "delta_n": 0,
                    # Temporal bias
                    "morning_delta": 0.0, "morning_n": 0,
                    "afternoon_delta": 0.0, "afternoon_n": 0,
                    # Saturation counter
                    "saturated": 0, "qualifying": 0,
                    # Solar shutdown (#838): count + parallel normal equations
                    # that exclude shutdown hours, so we can compare the
                    # implied coefficient with and without shutdown bias.
                    "shutdown_hours": 0,
                    "no_shutdown": {
                        "ss": 0.0, "ee": 0.0, "ww": 0.0,
                        "se": 0.0, "sw": 0.0, "ew": 0.0,
                        "sI": 0.0, "eI": 0.0, "wI": 0.0, "n": 0,
                    },
                    # Screen-position stratification (#826 validation).  Split
                    # modeled-vs-implied delta by correction_percent bucket so
                    # systematic bias at closed screens surfaces directly.
                    # open:   correction ≥ 80   (screens fully open-ish)
                    # mid:    40 ≤ correction < 80
                    # closed: correction < 40   (screens mostly deployed)
                    "correction_buckets": {
                        "open":   {"delta_sum": 0.0, "modeled_sum": 0.0, "implied_sum": 0.0, "n": 0},
                        "mid":    {"delta_sum": 0.0, "modeled_sum": 0.0, "implied_sum": 0.0, "n": 0},
                        "closed": {"delta_sum": 0.0, "modeled_sum": 0.0, "implied_sum": 0.0, "n": 0},
                    },
                    # Per-hour tuples for transmittance sensitivity sweep.
                    # Stored as effective vectors + correction so we can
                    # re-reconstruct potential under hypothesis transmittances.
                    # Each entry: (eff_s, eff_e, eff_w, correction, implied)
                    "sensitivity_tuples": [],
                    # Temperature-regime stratification relative to the
                    # configured balance_point.  Splits the modeled-vs-implied
                    # delta by COP-regime so we can empirically test the
                    # coefficient's documented COP-blindness (#826 follow-up):
                    # - heating_deep: T < BP-8   (low COP, defrost, aux-regime)
                    # - heating_mild: BP-8 ≤ T < BP-2 (optimal heat pump)
                    # - cooling:      T > BP+2   (inverted solar semantics)
                    # The ±2° transition zone around BP is deliberately dropped;
                    # mode flips hour-to-hour there and the signal is noise.
                    "temp_buckets": {
                        "heating_deep": {"delta_sum": 0.0, "n": 0},
                        "heating_mild": {"delta_sum": 0.0, "n": 0},
                        "cooling":      {"delta_sum": 0.0, "n": 0},
                    },
                    # Elevation-stratified residual buckets (#927 Tier 1).
                    # Heating regime only — qualifying cooling samples are
                    # rarer at this latitude and the user-facing hypothesis
                    # (hotspot loss vs. thermal-mass battery) shows up
                    # primarily in heating-mode shoulder-season afternoons.
                    # Per-bucket lists kept in full so median + MAD can be
                    # computed at response time; bounded by `qualifying`,
                    # which is itself bounded by 30 days × ~12 daylight
                    # hours = ~360 entries per entity in the worst case.
                    "elevation_buckets": {
                        f"{lo}-{hi}": {
                            "unsaturated": {"residuals": [], "potential_mags": []},
                            "saturated": {"residuals": [], "potential_mags": []},
                        }
                        for lo, hi in ELEVATION_BUCKETS
                    },
                    # Lag-stratified residuals for Tier 2.  Per-bucket dict
                    # of `lag_{k}` lists where each list holds `base_{H+k}
                    # − actual_{H+k}` values for entries whose originator
                    # hour H landed in that elevation bucket.  Same
                    # bookkeeping as Tier 1's residuals but indexed by
                    # forward-time offset rather than elevation only.
                    # Skipped entries (mode change, missing tail, self-
                    # qualifying solar at H+k) drop out of the lag they'd
                    # have populated — n_per_lag varies with data sparsity.
                    "elevation_lag_buckets": {
                        f"{lo}-{hi}": {f"lag_{k}": [] for k in range(7)}
                        for lo, hi in ELEVATION_BUCKETS
                    },
                    # Evening-tail walk for Tier 2.5: bypasses the
                    # self-qualifying tail gate by searching forward
                    # from the originator hour for the first dark
                    # hour (solar_factor < 0.05), then walking up to
                    # 6 h forward from that dark anchor.  Resolves the
                    # afternoon-elevation data starvation that the
                    # ordinary lag walk suffers on west-facade-
                    # dominated entities — for a 14:00 originator at
                    # 35° elevation, the lag walk hits 15-19:00 tails
                    # which often still register solar_factor > 0.05
                    # and are skipped to avoid double-counting.  The
                    # evening walk skips past those hours to e.g.
                    # 21:00 dark anchor and walks 21-03:00, freeing
                    # the tail for accumulation.
                    #
                    # Per-bucket: flat list of (residual, dark_offset)
                    # tuples across all originators × evening hours.
                    # Each tuple contributes to the bucket's mean
                    # residual at response time.  ``dark_offset`` is
                    # the number of hours forward from the originator
                    # to the first dark anchor — useful for diagnosing
                    # what fraction of originators found a dark anchor
                    # at all.
                    "elevation_evening_tail_buckets": {
                        f"{lo}-{hi}": {"residuals": [], "dark_offsets": []}
                        for lo, hi in ELEVATION_BUCKETS
                    },
                }
            return unit_accum[eid]

        # Global accumulators
        excluded = {"aux": 0, "guest": 0, "saturated": 0, "low_vector": 0, "no_base": 0, "legacy": 0, "outlier": 0}
        total_qualifying = 0

        # Battery calibration: collect per-day hourly sequences for decay sweep.
        # Key = date string, value = list of (hour, solar_impact_raw, actual, expected)
        day_sequences: dict[str, list[tuple]] = {}
        # Battery health: post-sunset hours (using current decay)
        battery_residuals: list[float] = []
        battery_expected: list[float] = []

        # Screen stratification
        screen_closed_errors: list[float] = []  # correction < 50, solar_factor > 0.3
        screen_open_errors: list[float] = []    # correction > 80, solar_factor > 0.3

        # Hour-of-day residual curve (hours 6-18)
        hourly_residuals: dict[int, list[float]] = {h: [] for h in range(6, 19)}

        # Timestamp → entry index for the lag walk in
        # `elevation_diagnostics.lag` (#927 Tier 2).  Built once over the
        # cutoff-filtered log so each per-(entity, sample) lag walk does
        # constant-time lookups without re-scanning the log.  Cost on a
        # 90-day log is ~2160 entries × dict insertion ≈ <1 ms.  Cleared
        # implicitly when diagnose_solar returns; rebuilt next call.
        log_by_timestamp: dict[str, dict] = {}
        for _entry in self.coordinator._hourly_log:
            _ts = _entry.get("timestamp", "")
            if _ts and _ts[:10] >= cutoff:
                log_by_timestamp[_ts] = _entry

        # Battery thermal-feedback sweep (#896): chronological tuples for
        # replay of the EMA under candidate k values.  Data shape mirrors the
        # production EMA input plus enough metadata to stratify residuals by
        # (hour-of-day × temp-regime × screen-position).  Population happens
        # outside the per-entity inner loop (one tuple per hour, not one per
        # entity), so it lives directly in the entry-level pass below.
        # Each tuple:
        #   (solar_impact_raw, solar_wasted, actual, expected,
        #    has_heating_unit, hour_bucket, temp_bucket, screen_bucket)
        sweep_tuples: list[tuple] = []

        # --- Single pass ---
        for entry in self.coordinator._hourly_log:
            ts = entry.get("timestamp", "")
            if ts[:10] < cutoff:
                continue

            hour = entry.get("hour", -1)
            solar_factor = entry.get("solar_factor") or 0.0
            solar_s = entry.get("solar_vector_s", 0.0)
            solar_e = entry.get("solar_vector_e", 0.0)
            solar_w = entry.get("solar_vector_w", 0.0)
            vector_mag = (solar_s ** 2 + solar_e ** 2 + solar_w ** 2) ** 0.5
            correction = entry.get("correction_percent", 100.0)
            # Outdoor temperature for BP-relative regime stratification.
            # Fall back through the same chain the learning path uses so we
            # stay consistent with the inertia-adjusted temperature when
            # present, and raw temp otherwise.
            temp_entry = (
                entry.get("inertia_temp")
                if entry.get("inertia_temp") is not None
                else entry.get("temp")
            )
            solar_impact_raw = entry.get("solar_impact_raw_kwh", 0.0)
            solar_impact_eff = entry.get("solar_impact_kwh", 0.0)
            aux_active = entry.get("auxiliary_active", False)
            guest_kwh = entry.get("guest_impact_kwh", 0.0)
            unit_modes = entry.get("unit_modes", {})
            unit_breakdown = entry.get("unit_breakdown", {})
            unit_expected_base = entry.get("unit_expected_breakdown", {})
            # Solar shutdown (#838): missing on legacy logs → empty list means
            # no shutdown hours recorded, which is treated as "all qualifying
            # hours are non-shutdown" below.
            log_shutdown_entities = set(entry.get("solar_dominant_entities", []) or [])

            # Battery health: post-sunset hours with residual battery charge
            if solar_factor < 0.01 and solar_impact_eff > 0.05:
                actual_total = entry.get("actual_kwh", 0.0)
                expected_total = entry.get("expected_kwh", 0.0)
                if expected_total > 0.05:
                    battery_residuals.append(actual_total - expected_total)
                    # Paired expectation, for the relative-deviation gate
                    # in ``_assess_battery_bias``.  Index-aligned with
                    # ``battery_residuals`` by construction — both append
                    # inside this one branch.
                    battery_expected.append(expected_total)

            # Collect day sequences for joint (decay, k) calibration sweep.
            # Per-hour wasted is needed alongside raw_solar so the carryover
            # battery EMA can be replayed under each k candidate without
            # re-iterating the hourly_log.  Read the same wasted field the
            # k-only sweep uses (preferring the heating-gated field, falling
            # back to the legacy aggregate).
            day_key = ts[:10]
            actual_total = entry.get("actual_kwh", 0.0)
            expected_total = entry.get("expected_kwh", 0.0)
            day_wasted = entry.get(
                "solar_heating_wasted_kwh",
                entry.get("solar_wasted_kwh", 0.0),
            )
            if day_key not in day_sequences:
                day_sequences[day_key] = []
            day_sequences[day_key].append(
                (hour, solar_impact_raw, day_wasted, actual_total, expected_total)
            )

            # Battery thermal-feedback sweep (#896).  Build chronological
            # tuples matching the production EMA input shape; stratification
            # happens at residual-bucketing time post-loop.
            #
            # Wasted source: prefer ``solar_heating_wasted_kwh`` (heating-
            # gated at write time, #896) and fall back to the total
            # ``solar_wasted_kwh`` for legacy entries written before that
            # field existed.  Today's saturation logic returns wasted=0 for
            # cooling/OFF/DHW, so the legacy aggregate is structurally
            # heating-only — but we prefer the explicit field when present.
            #
            # Heating-active gate mirrors the live EMA gate: the unit_modes
            # log entry only stores non-heating modes (heating is the default
            # to reduce log clutter, see HourlyProcessor.process), so missing
            # entity → MODE_HEATING.  ``has_heating_unit`` is True when at
            # least one configured energy sensor was in heating-regime that
            # hour — an absent unit_modes block (legacy or pre-#838 log)
            # falls back to True since heating is the default.
            sweep_solar_wasted = entry.get(
                "solar_heating_wasted_kwh",
                entry.get("solar_wasted_kwh", 0.0),
            )
            if sweep_solar_wasted > 0.0:
                # Cooling/OFF/DHW saturation returns wasted=0; positive wasted
                # is a structural witness of at least one heating unit having
                # contributed to the aggregate.  Avoids depending on the
                # filtered unit_modes log (which can omit explicit-heating).
                has_heating_unit = True
            elif unit_modes:
                stored_heating = any(
                    m in (MODE_HEATING, MODE_GUEST_HEATING)
                    for m in unit_modes.values()
                )
                stored_non_default = bool(unit_modes)
                # If the stored map omits some sensors entirely, those
                # sensors were in MODE_HEATING (the default-omitted regime).
                missing_count = sum(
                    1 for sid in self.coordinator.energy_sensors if sid not in unit_modes
                )
                has_heating_unit = stored_heating or missing_count > 0
                # If no sensors are configured, treat as no heating active.
                if not self.coordinator.energy_sensors and not stored_non_default:
                    has_heating_unit = False
            else:
                has_heating_unit = bool(self.coordinator.energy_sensors)

            # Hour-of-day bucket: morning captures the issue's reported
            # symptom (transition-regime over-prediction at ~7-10 °C, mid
            # morning).  Night included for completeness — the EMA carries
            # state across midnight and post-sunset error is the diagnostic
            # signal.
            if 6 <= hour < 11:
                hour_bucket = "morning"
            elif 11 <= hour < 15:
                hour_bucket = "midday"
            elif 15 <= hour < 22:
                hour_bucket = "afternoon"
            else:
                hour_bucket = "night"

            # Temp regime: 4 buckets including a ``transition`` zone
            # (BP-2 ≤ T ≤ BP+2).  Issue #896's headline symptom — sunny
            # mornings at 7-10 °C with BP=15 — actually lies inside
            # ``heating_mild`` ([7, 13)) for typical Norwegian BP=15,
            # but for higher-BP installs (older buildings, BP≈17-18) the
            # symptom hours can drift up into the BP±2 window.  Keeping
            # ``transition`` as its own bucket ensures that cell is
            # always visible in ``per_cell_at_optimum``; collapsing it
            # into None (the previous behaviour) hid the headline
            # symptom from the user-facing diagnostic for high-BP
            # installs.  The other diagnose_solar.temperature_stratified
            # block still drops transition for its own purpose
            # (mode-flip noise on the COP-blindness check); the sweep
            # has different needs and stratifies independently.
            if temp_entry is not None:
                bp = self.coordinator.balance_point
                if temp_entry < bp - 8.0:
                    temp_bucket = "heating_deep"
                elif temp_entry < bp - 2.0:
                    temp_bucket = "heating_mild"
                elif temp_entry <= bp + 2.0:
                    temp_bucket = "transition"
                else:
                    temp_bucket = "cooling"
            else:
                temp_bucket = None

            # Screen-position bucket: matches per-unit screen_stratified.
            if correction is None:
                screen_bucket = "open"
            elif correction >= 80.0:
                screen_bucket = "open"
            elif correction >= 40.0:
                screen_bucket = "mid"
            else:
                screen_bucket = "closed"

            sweep_tuples.append((
                solar_impact_raw, sweep_solar_wasted, actual_total, expected_total,
                has_heating_unit, hour_bucket, temp_bucket, screen_bucket,
            ))

            # Days-ago for window assignment
            try:
                days_ago = (now.date() - _date.fromisoformat(ts[:10])).days
            except (ValueError, TypeError):
                days_ago = 0

            # Sun elevation for this entry — computed once outside the
            # per-entity loop and reused across all entities (#927 Tier 1).
            # Uses hour midpoint so the elevation reflects the average
            # sun position across the hour, matching how solar_vector
            # was accumulated in the live coordinator.  Returns None on
            # parse failure or when astral is unavailable; downstream
            # bucketing skips those entries.
            entry_dt = dt_util.parse_datetime(ts) if ts else None
            sun_elev_entry: float | None = None
            if entry_dt is not None:
                mid_dt = entry_dt + timedelta(minutes=30)
                try:
                    sun_pos = self.coordinator.solar.get_approx_sun_pos(mid_dt)
                    sun_elev_entry = float(sun_pos[0])
                except (TypeError, ValueError, IndexError):
                    sun_elev_entry = None

            # Per-unit analysis
            for entity_id in self.coordinator.energy_sensors:
                mode = unit_modes.get(entity_id, MODE_HEATING)
                if mode in (MODE_OFF, MODE_DHW, MODE_GUEST_HEATING, MODE_GUEST_COOLING):
                    if mode in (MODE_GUEST_HEATING, MODE_GUEST_COOLING):
                        excluded["guest"] += 1
                    continue

                if aux_active:
                    excluded["aux"] += 1
                    continue

                if vector_mag < 0.01:
                    excluded["low_vector"] += 1
                    continue

                actual_unit = unit_breakdown.get(entity_id, 0.0)
                base_unit = unit_expected_base.get(entity_id)
                if base_unit is None:
                    excluded["no_base"] += 1
                    continue

                # Prior-free sanity check (#919 Part 5)
                # Mirrors the match_diagnose=True path in _collect_batch_fit_samples
                if base_unit > 0 and abs(actual_unit - base_unit) > HARD_OUTLIER_SANITY_MULTIPLIER * base_unit:
                    excluded["outlier"] += 1
                    continue

                # Implied solar reduction
                if mode == MODE_HEATING:
                    implied_solar = base_unit - actual_unit
                elif mode == MODE_COOLING:
                    implied_solar = actual_unit - base_unit
                else:
                    continue

                # Check saturation
                acc = _get_unit_accum(entity_id)
                acc["qualifying"] += 1
                # Heating saturation: ``actual_unit < 0.05 * base_unit`` is
                # equivalent to ``implied_solar >= 0.95 * base_unit`` which
                # is ``BATCH_FIT_SATURATION_RATIO``.
                is_saturated_heating = (
                    mode == MODE_HEATING
                    and base_unit > 0.05
                    and actual_unit < 0.05 * base_unit
                )
                if is_saturated_heating:
                    acc["saturated"] += 1
                    excluded["saturated"] += 1
                    # Emit the elevation residual into the saturated sub-bucket
                    # before bailing — needed to disambiguate HP-capacity
                    # censoring from genuine Kasten elevation×airmass bias in
                    # elevation_diagnostics.
                    if sun_elev_entry is not None and 0.0 <= sun_elev_entry < 90.0:
                        _implied_sat = max(0.0, base_unit - actual_unit)
                        _potential_sat = _SC.reconstruct_potential_vector(
                            (solar_s, solar_e, solar_w),
                            correction if correction is not None else 100.0,
                            self.coordinator.screen_config_for_entity(entity_id),
                        )
                        _coeff_sat = self.coordinator.solar.calculate_unit_coefficient(
                            entity_id, entry.get("temp_key", "10"), mode
                        )
                        _modeled_sat = self.coordinator.solar.calculate_unit_solar_impact(
                            _potential_sat, _coeff_sat
                        )
                        for lo, hi in ELEVATION_BUCKETS:
                            if lo <= sun_elev_entry < hi:
                                ev_sat = acc["elevation_buckets"][f"{lo}-{hi}"]["saturated"]
                                ev_sat["residuals"].append(_implied_sat - _modeled_sat)
                                ev_sat["potential_mags"].append(vector_mag)
                                break
                    continue

                implied_solar = max(0.0, implied_solar)

                # Modeled solar for this unit.
                # Reconstruct potential vector from effective + screen transmittance.
                # When ``correction`` is missing on a legacy log entry, fall back
                # to 100 % (screens fully open) so the helper's per-direction
                # transmittance returns 1.0 for every direction — matching the
                # pre-#876 ``t = 1.0`` short-circuit.  The equivalence relies on
                # the ramp identity ``mn + (1 - mn) * 1.0 = 1.0``, which holds
                # for both the legacy composite floor (mn=0.30) and the screened
                # floor (mn=0.08); unscreened directions are fixed at 1.0.
                effective = (solar_s, solar_e, solar_w)
                diag_potential = _SC.reconstruct_potential_vector(
                    effective,
                    correction if correction is not None else 100.0,
                    self.coordinator.screen_config_for_entity(entity_id),
                )
                unit_coeff = self.coordinator.solar.calculate_unit_coefficient(
                    entity_id, entry.get("temp_key", "10"), mode
                )
                modeled_solar = self.coordinator.solar.calculate_unit_solar_impact(diag_potential, unit_coeff)
                delta = modeled_solar - implied_solar

                total_qualifying += 1

                # Screen-position stratification (#826).  The per-direction
                # transmittance model introduced in 1.3.3 uses values (0.08
                # per-direction, 0.30 composite legacy) that were chosen from
                # literature, not measured at this building.  If either value
                # is wrong, modeled_solar will systematically differ from
                # implied_solar at the screen-position extremes while matching
                # well at fully-open screens (where transmittance = 1.0
                # regardless).  The correction-bucket split surfaces that
                # directly: a bias pattern of (open ≈ 0, closed ≠ 0) points
                # to a wrong transmittance floor; a uniform bias points to a
                # coefficient calibration issue.
                if correction is None:
                    bucket_key = "open"
                elif correction >= 80.0:
                    bucket_key = "open"
                elif correction >= 40.0:
                    bucket_key = "mid"
                else:
                    bucket_key = "closed"
                bucket = acc["correction_buckets"][bucket_key]
                bucket["delta_sum"] += delta
                bucket["modeled_sum"] += modeled_solar
                bucket["implied_sum"] += implied_solar
                bucket["n"] += 1

                # Temperature-regime stratification (BP-relative).  Splits
                # the same delta into COP-regime buckets so coefficient bias
                # correlated with heat-pump efficiency becomes visible.  The
                # ±2° transition zone around balance_point is dropped: mode
                # flips hour-to-hour there and the signal would be noise.
                if temp_entry is not None:
                    bp = self.coordinator.balance_point
                    if mode == MODE_HEATING:
                        if temp_entry < bp - 8.0:
                            tb = acc["temp_buckets"]["heating_deep"]
                            tb["delta_sum"] += delta
                            tb["n"] += 1
                        elif temp_entry < bp - 2.0:
                            tb = acc["temp_buckets"]["heating_mild"]
                            tb["delta_sum"] += delta
                            tb["n"] += 1
                        # else: transition zone (BP-2 ≤ T ≤ BP), dropped
                    elif mode == MODE_COOLING and temp_entry > bp + 2.0:
                        tb = acc["temp_buckets"]["cooling"]
                        tb["delta_sum"] += delta
                        tb["n"] += 1

                # Sensitivity sweep tuple — effective vector + correction lets
                # us re-reconstruct potential under hypothesis transmittances
                # post-loop without replaying the whole log.  Shutdown hours
                # excluded to avoid feeding the same bias into the sweep that
                # no_shutdown already subtracts from the baseline fit.
                if entity_id not in log_shutdown_entities:
                    acc["sensitivity_tuples"].append(
                        (solar_s, solar_e, solar_w,
                         correction if correction is not None else 100.0,
                         implied_solar)
                    )

                # Solar shutdown tracking (#838): count shutdown hours and
                # also accumulate a separate set of normal equations from
                # non-shutdown hours, so we can compare the implied
                # coefficient with vs without shutdown bias.
                is_shutdown_hour = entity_id in log_shutdown_entities
                if is_shutdown_hour:
                    acc["shutdown_hours"] += 1
                else:
                    ns = acc["no_shutdown"]
                    ns["ss"] += solar_s * solar_s
                    ns["ee"] += solar_e * solar_e
                    ns["ww"] += solar_w * solar_w
                    ns["se"] += solar_s * solar_e
                    ns["sw"] += solar_s * solar_w
                    ns["ew"] += solar_e * solar_w
                    ns["sI"] += solar_s * implied_solar
                    ns["eI"] += solar_e * implied_solar
                    ns["wI"] += solar_w * implied_solar
                    ns["n"] += 1

                # Normal equations (30d total, 3x3)
                acc["ss"] += solar_s * solar_s
                acc["ee"] += solar_e * solar_e
                acc["ww"] += solar_w * solar_w
                acc["se"] += solar_s * solar_e
                acc["sw"] += solar_s * solar_w
                acc["ew"] += solar_e * solar_w
                acc["sI"] += solar_s * implied_solar
                acc["eI"] += solar_e * implied_solar
                acc["wI"] += solar_w * implied_solar
                acc["n"] += 1

                # Window assignment
                for w_idx in range(3):
                    w_start = window_boundaries[w_idx]
                    w_end = window_boundaries[w_idx + 1]
                    if w_end <= days_ago < w_start:
                        w = acc["windows"][w_idx]
                        w["ss"] += solar_s * solar_s
                        w["ee"] += solar_e * solar_e
                        w["ww"] += solar_w * solar_w
                        w["se"] += solar_s * solar_e
                        w["sw"] += solar_s * solar_w
                        w["ew"] += solar_e * solar_w
                        w["sI"] += solar_s * implied_solar
                        w["eI"] += solar_e * implied_solar
                        w["wI"] += solar_w * implied_solar
                        w["n"] += 1
                        break

                # Delta
                acc["sum_delta"] += delta
                acc["delta_n"] += 1

                # Temporal bias
                if 6 <= hour <= 11:
                    acc["morning_delta"] += delta
                    acc["morning_n"] += 1
                elif 12 <= hour <= 17:
                    acc["afternoon_delta"] += delta
                    acc["afternoon_n"] += 1

                # Screen stratification (global)
                if solar_factor > 0.3:
                    if correction < 50:
                        screen_closed_errors.append(delta)
                    elif correction > 80:
                        screen_open_errors.append(delta)

                # Hour-of-day residual
                if 6 <= hour <= 18:
                    hourly_residuals[hour].append(delta)

                # Elevation-stratified residual (#927 Tier 1).  Heating
                # regime only — see _get_unit_accum init comment for
                # rationale.  Spec semantics (#927): residual sign is
                # ``actual_impact - predicted = implied_solar - modeled_solar``,
                # i.e. the negation of ``delta``.  Negative residual
                # therefore means the model OVER-predicts solar reduction
                # in that bucket (hotspot signature when concentrated at
                # high elevation); positive means UNDER-predicts (battery
                # signature, but Tier 1 alone cannot distinguish from
                # plain calibration error — the lag-stratified Tier 2
                # closes that gap).
                if (
                    sun_elev_entry is not None
                    and 0.0 <= sun_elev_entry < 90.0
                    and mode == MODE_HEATING
                ):
                    for lo, hi in ELEVATION_BUCKETS:
                        if lo <= sun_elev_entry < hi:
                            bucket_key = f"{lo}-{hi}"
                            ev_bucket = acc["elevation_buckets"][bucket_key]["unsaturated"]
                            ev_bucket["residuals"].append(implied_solar - modeled_solar)
                            ev_bucket["potential_mags"].append(vector_mag)

                            # Tier 2 lag walk (#927).  For each forward
                            # offset k ∈ [0, 6], pull the H+k entry from
                            # the timestamp index and append `tail_base −
                            # tail_actual` for this entity to the
                            # appropriate `lag_{k}` list.  Skip:
                            #   • missing entry (gap in log),
                            #   • mode change between H and H+k for this
                            #     entity (different physical regime),
                            #   • aux active at H+k (compromises base),
                            #   • self-qualifying tail with
                            #     solar_factor > 0.05 at H+k for k > 0
                            #     (would double-count its own originator
                            #     residual — the threshold mirrors the
                            #     `signal_agreement` stability gate).
                            # Lag 0 is the originator's own
                            # `base − actual = implied_solar`, included
                            # without the self-qualify gate.
                            if entry_dt is not None:
                                ev_lag = acc["elevation_lag_buckets"][bucket_key]
                                for _k in range(7):
                                    _tail_ts = (entry_dt + timedelta(hours=_k)).isoformat()
                                    _tail = log_by_timestamp.get(_tail_ts)
                                    if _tail is None:
                                        continue
                                    _tail_modes = _tail.get("unit_modes") or {}
                                    _tail_mode = _tail_modes.get(entity_id, MODE_HEATING)
                                    if _tail_mode != mode:
                                        continue
                                    if _tail.get("auxiliary_active", False):
                                        continue
                                    if _k > 0 and (_tail.get("solar_factor") or 0.0) > 0.05:
                                        continue
                                    _tail_base = (_tail.get("unit_expected_breakdown") or {}).get(entity_id)
                                    _tail_actual = (_tail.get("unit_breakdown") or {}).get(entity_id)
                                    if _tail_base is None or _tail_actual is None:
                                        continue
                                    ev_lag[f"lag_{_k}"].append(
                                        float(_tail_base) - float(_tail_actual)
                                    )

                                # Evening-tail walk (Tier 2.5).  Find
                                # the first dark hour (solar_factor <
                                # 0.05) within 12 h of the originator,
                                # then accumulate residuals over the 6
                                # hours from that anchor.  No self-
                                # qualifying gate inside the evening
                                # window — the whole point is to
                                # bypass the still-sunny afternoon
                                # tails that the ordinary lag walk
                                # cannot use on heavy-west installs.
                                # Skips: missing entry, mode change,
                                # aux active.
                                _ev_acc = acc["elevation_evening_tail_buckets"][bucket_key]
                                _dark_offset = None
                                for _search in range(1, 13):
                                    _search_ts = (
                                        entry_dt + timedelta(hours=_search)
                                    ).isoformat()
                                    _candidate = log_by_timestamp.get(_search_ts)
                                    if _candidate is None:
                                        continue
                                    if (_candidate.get("solar_factor") or 0.0) < 0.05:
                                        _dark_offset = _search
                                        break
                                if _dark_offset is not None:
                                    _ev_acc["dark_offsets"].append(_dark_offset)
                                    for _ek in range(6):
                                        _ev_ts = (
                                            entry_dt
                                            + timedelta(hours=_dark_offset + _ek)
                                        ).isoformat()
                                        _ev_entry = log_by_timestamp.get(_ev_ts)
                                        if _ev_entry is None:
                                            continue
                                        _ev_modes = _ev_entry.get("unit_modes") or {}
                                        if (
                                            _ev_modes.get(entity_id, MODE_HEATING)
                                            != mode
                                        ):
                                            continue
                                        if _ev_entry.get("auxiliary_active", False):
                                            continue
                                        _ev_base = (
                                            _ev_entry.get("unit_expected_breakdown") or {}
                                        ).get(entity_id)
                                        _ev_actual = (
                                            _ev_entry.get("unit_breakdown") or {}
                                        ).get(entity_id)
                                        if _ev_base is None or _ev_actual is None:
                                            continue
                                        _ev_acc["residuals"].append(
                                            float(_ev_base) - float(_ev_actual)
                                        )
                            break

        # --- Build results ---

        def _median(values: list[float]) -> float:
            n_v = len(values)
            if n_v == 0:
                return 0.0
            s = sorted(values)
            mid = n_v // 2
            if n_v % 2 == 1:
                return s[mid]
            return (s[mid - 1] + s[mid]) / 2.0

        def _mad(values: list[float], med: float) -> float:
            if not values:
                return 0.0
            return _median([abs(v - med) for v in values])

        def _build_elevation_evening_tail_block(
            evening_acc: dict[str, dict[str, list[float]]],
            min_samples: int = 10,
        ) -> dict[str, dict]:
            """Build the `elevation_diagnostics.evening_tail` response.

            Per-bucket: emits ``n_originators_with_dark_anchor``,
            ``mean_dark_offset_hours``, ``n_evening_hours`` and a
            ``mean_residual_kwh`` scalar.  ``mean_residual_kwh`` is
            the per-evening-hour reduction below baseline; positive
            means consumption sustained below baseline through the
            6-hour post-sunset window for that bucket's originators.

            Compare directly against ``elevation_diagnostics.lag``
            tail-sums by multiplying ``mean_residual_kwh × 6`` —
            same units as ``tail_sum_lag1_to_6_kwh``, modulo that
            the evening walk starts from sunset rather than H+1.

            Decisive interpretation for the high-elevation hotspot
            question:

            - **Positive** ``mean_residual_kwh`` on 30-45° / 45-60°
              buckets → energy reaches thermal mass after all,
              just on a longer time-scale than the lag walk could
              measure.  Hotspot loss as the dominant mechanism is
              **refuted**; the high-elev Tier 1 over-prediction is
              timing redistribution, not amplitude error.
            - **Near-zero** ``mean_residual_kwh`` on the same
              buckets → energy is genuinely lost (re-radiation
              through glass, ceiling stratification bypassing
              thermal mass).  Hotspot loss **confirmed** as
              dominant mechanism; high-elev coefficient should be
              modulated downward, not redistributed in time.

            Set to None when ``n_evening_hours < min_samples``.
            """
            out: dict[str, dict] = {}
            for bucket_key, data in evening_acc.items():
                residuals = data["residuals"]
                offsets = data["dark_offsets"]
                n_h = len(residuals)
                n_o = len(offsets)
                if n_h < min_samples:
                    out[bucket_key] = {
                        "n_originators_with_dark_anchor": n_o,
                        "n_evening_hours": n_h,
                        "mean_residual_kwh": None,
                        "mean_dark_offset_hours": (
                            round(sum(offsets) / n_o, 2) if n_o > 0 else None
                        ),
                    }
                    continue
                out[bucket_key] = {
                    "n_originators_with_dark_anchor": n_o,
                    "n_evening_hours": n_h,
                    "mean_residual_kwh": round(sum(residuals) / n_h, 4),
                    "mean_dark_offset_hours": (
                        round(sum(offsets) / n_o, 2) if n_o > 0 else None
                    ),
                }
            return out

        def _build_elevation_lag_block(
            elev_lag_acc: dict[str, dict[str, list[float]]],
            min_lag_samples: int = 10,
        ) -> dict[str, dict]:
            """Build the `elevation_diagnostics.lag` response (Tier 2).

            Per-bucket: emits {lag_0, ..., lag_6} sub-blocks plus two
            tail-sum scalars at different horizons.  Each sub-block
            carries `n` and (when n ≥ ``min_lag_samples``)
            `mean_residual_kwh` — the hours-mean of `base_{H+k} −
            actual_{H+k}` across all samples whose originator landed
            in this elevation bucket.

            Two tail-sum windows are emitted side-by-side:

            - **`tail_sum_lag1_to_3_kwh`** — short-horizon sum.
              Captures the 1-3 h release window where most thermal-
              mass battery effects peak (Toshiba's 0-15° release
              empirically peaked at lag_2 in field data).  Crucially,
              this scalar remains computable for afternoon-elevation
              originators on west-facade-dominated installs where
              the 6 h walk reaches into evening hours that fail the
              self-qualifying tail gate (still-sunny tails on heavy-
              west houses skip the lag walk and starve the longer
              window).
            - **`tail_sum_lag1_to_6_kwh`** — long-horizon sum.
              Picks up the slow tail (4-6 h) on installs where the
              afternoon-tail-starvation problem doesn't apply
              (south-facing, no-screen, low-elevation originators).
              Returns None on buckets where any of the longer lags
              is under-sampled.

            Interpretation of the tail-sum scalars:

            - **Positive** (consumption stays below baseline 1-3 h or
              1-6 h after the solar input) → thermal-mass battery:
              stored energy releases gradually, Tier 1 alone would
              under-credit the originator's actual contribution.
            - **Near zero** alongside a large positive `lag_0` →
              hotspot regime: instantaneous reduction without
              sustained effect.  Beam-concentrated energy lost to
              glass re-emission and ceiling-stratification before
              reaching thermal mass.
            - **Negative** is rare and usually means the lag walk is
              picking up post-sunset rebound where the HP catches up
              on missed setpoint after a sunny daytime.

            Both scalars set to None when any lag in their window
            has fewer than ``min_lag_samples`` populated entries —
            partial sums would silently exaggerate the tail by
            dropping under-sampled lags.  The 1-3 h scalar is
            populated more often than the 1-6 h scalar by design;
            both are reported so the consumer sees the trade-off.
            """
            out: dict[str, dict] = {}
            for bucket_key, lag_data in elev_lag_acc.items():
                per_lag: dict = {}
                # Per-lag mean computation.
                lag_means: dict[int, float | None] = {}
                for k in range(7):
                    rows = lag_data[f"lag_{k}"]
                    n_l = len(rows)
                    if n_l < min_lag_samples:
                        per_lag[f"lag_{k}"] = {"n": n_l}
                        lag_means[k] = None
                    else:
                        mean_r = sum(rows) / n_l
                        per_lag[f"lag_{k}"] = {
                            "n": n_l,
                            "mean_residual_kwh": round(mean_r, 4),
                        }
                        lag_means[k] = mean_r

                def _window_sum(start: int, end_inclusive: int) -> float | None:
                    vals = [lag_means[k] for k in range(start, end_inclusive + 1)]
                    if any(v is None for v in vals):
                        return None
                    return sum(vals)

                short_sum = _window_sum(1, 3)
                long_sum = _window_sum(1, 6)
                per_lag["tail_sum_lag1_to_3_kwh"] = (
                    round(short_sum, 4) if short_sum is not None else None
                )
                per_lag["tail_sum_lag1_to_6_kwh"] = (
                    round(long_sum, 4) if long_sum is not None else None
                )
                out[bucket_key] = per_lag
            return out

        def _build_elevation_block(
            elev_acc: dict[str, dict[str, dict[str, list[float]]]],
            min_samples: int = 5,
        ) -> dict[str, dict]:
            """Build the `elevation_diagnostics.instantaneous` response.

            Each elevation bucket is split into ``unsaturated`` and
            ``saturated`` sub-blocks so the reader can distinguish
            HP-capacity censoring (saturation: ``actual < 0.05·base``,
            equivalent to ``implied_solar ≥ BATCH_FIT_SATURATION_RATIO``)
            from genuine Kasten elevation×airmass bias (unsaturated
            modulating-regime hours).  Heating regime only; cooling
            samples are excluded upstream.

            Sub-blocks with fewer than ``min_samples`` qualifying hours
            return ``{"n": <count>}`` only — the median / MAD / normalised
            ratio would be too noisy to surface usefully on small ``n``.
            ``mean_potential`` ≤ 1e-3 in a populated sub-block means the
            sample's solar input was effectively zero; return None for
            ``median_residual_normalised`` rather than divide by ~0.
            """
            def _summarise(residuals: list[float], pots: list[float]) -> dict:
                n_b = len(residuals)
                if n_b < min_samples:
                    return {"n": n_b}
                med = _median(residuals)
                mad = _mad(residuals, med)
                mean_pot = sum(pots) / n_b
                if mean_pot > 1e-3:
                    norm = round(med / mean_pot, 4)
                else:
                    norm = None
                return {
                    "n": n_b,
                    "median_residual": round(med, 4),
                    "mad_residual": round(mad, 4),
                    "mean_potential": round(mean_pot, 4),
                    "median_residual_normalised": norm,
                }

            out: dict[str, dict] = {}
            for bucket_key, data in elev_acc.items():
                unsat = data["unsaturated"]
                sat = data["saturated"]
                out[bucket_key] = {
                    "unsaturated": _summarise(unsat["residuals"], unsat["potential_mags"]),
                    "saturated": _summarise(sat["residuals"], sat["potential_mags"]),
                }
            return out

        def _solve_normal(a, n, min_samples=10):
            """Solve normal equations for implied coefficient.

            Attempts a 3x3 solve (S, E, W).  When the west dimension has
            no variance (sum_ww ≈ 0, typical for legacy logs without
            solar_vector_w), falls back to the 2x2 (S, E) system so that
            diagnose_solar remains useful immediately after upgrading.

            Args:
                a: dict with keys ss, ee, ww, se, sw, ew, sI, eI, wI.
                n: number of qualifying samples.
            """
            if n < min_samples:
                return None
            ss, ee, ww = a["ss"], a["ee"], a["ww"]
            se, sw, ew = a["se"], a["sw"], a["ew"]
            sI, eI, wI = a["sI"], a["eI"], a["wI"]
            # 3x3 determinant via cofactor expansion
            det = (
                ss * (ee * ww - ew * ew)
                - se * (se * ww - ew * sw)
                + sw * (se * ew - ee * sw)
            )
            if abs(det) > 1e-6:
                # Full 3D solution
                det_s = (
                    sI * (ee * ww - ew * ew)
                    - se * (eI * ww - ew * wI)
                    + sw * (eI * ew - ee * wI)
                )
                det_e = (
                    ss * (eI * ww - ew * wI)
                    - sI * (se * ww - ew * sw)
                    + sw * (se * wI - eI * sw)
                )
                det_w = (
                    ss * (ee * wI - ew * eI)
                    - se * (se * wI - eI * sw)
                    + sI * (se * ew - ee * sw)
                )
                return {
                    "s": round(det_s / det, 4),
                    "e": round(det_e / det, 4),
                    "w": round(det_w / det, 4),
                }
            # Fallback: 2D solve (S, E) when W dimension has no variance
            # (legacy logs or insufficient afternoon data)
            det_2d = ss * ee - se * se
            if abs(det_2d) > 1e-6:
                return {
                    "s": round((ee * sI - se * eI) / det_2d, 4),
                    "e": round((ss * eI - se * sI) / det_2d, 4),
                    "w": 0.0,
                }
            return None

        # Shadow replay with inequality learner (#865).  Runs a clean
        # NLMS + inequality replay from zero coefficients over the same
        # window to show what the learner would produce if the user
        # reset and refit now.  This is the pre-validation ankerpunkt
        # The inequality learner's effect becomes visible as the delta
        # between this and ``implied_coefficient_30d_no_shutdown``.
        # Single shadow pass, no side effects on live state (all dicts
        # are local copies).
        window_entries = [
            e for e in self.coordinator._hourly_log
            if e.get("timestamp", "") >= cutoff
        ]
        shadow_coeffs: dict = {}
        shadow_buffers: dict = {}
        # Seed shadow per-unit correlation from the current coordinator
        # state — inequality needs a base reference.  If empty (fresh
        # install), the replay falls back to NLMS's threshold gate.
        shadow_diag = self.coordinator.learning.replay_solar_nlms(
            window_entries,
            solar_calculator=self.coordinator.solar,
            screen_config=getattr(self.coordinator, "screen_config", None),
            correlation_data_per_unit=self.coordinator._correlation_data_per_unit,
            solar_coefficients_per_unit=shadow_coeffs,
            learning_buffer_solar_per_unit=shadow_buffers,
            energy_sensors=self.coordinator.energy_sensors,
            learning_rate=self.coordinator.learning_rate,
            balance_point=self.coordinator.balance_point,
            aux_affected_entities=self.coordinator.aux_affected_entities,
            unit_strategies=self.coordinator._unit_strategies,
            daily_history=self.coordinator._daily_history,
            unit_min_base=self.coordinator._per_unit_min_base_thresholds or None,
            solar_affected_entities=(
                self.coordinator._solar_affected_set
                if isinstance(
                    getattr(self.coordinator, "_solar_affected_set", None),
                    (frozenset, set),
                )
                else None
            ),
            wind_threshold=self.coordinator.wind_threshold,
            extreme_wind_threshold=self.coordinator.extreme_wind_threshold,
            return_diagnostics=True,
        )

        per_unit = {}
        # Per-entity solar-scope (#962): excluded entities get a one-line stub
        # so the dict still contains every configured energy_sensor (consumers
        # that walk per_unit can find them) without the verbose coefficient /
        # stability / lag blocks for entities the user has declared do not
        # respond to solar.
        is_solar_affected_fn = getattr(self.coordinator, "is_solar_affected", None)
        for entity_id, acc in unit_accum.items():
            if callable(is_solar_affected_fn) and not is_solar_affected_fn(entity_id):
                per_unit[entity_id] = {"excluded_from_solar": True}
                continue
            # #868: report both regimes separately.  ``current_coefficient``
            # remains the prediction-time view (heating regime + default
            # fallback) for backwards compatibility with consumers that
            # haven't migrated.  The split-aware fields and
            # ``coefficient_split_delta_pct`` read raw storage instead —
            # an unlearned regime must show ``{0,0,0}``, not a default
            # decomposition.  Otherwise the validation criterion ("split
            # delta > N means the split captures real physics") is
            # muddled by every heating-only install reporting a small
            # divergence purely from the cooling default.
            current = self.coordinator.solar.calculate_unit_coefficient(
                entity_id, "10", MODE_HEATING
            )
            stored_entry = (
                self.coordinator.model.solar_coefficients_per_unit.get(
                    entity_id, {}
                )
            )
            if not isinstance(stored_entry, dict):
                stored_entry = {}
            current_heating = stored_entry.get("heating") or {
                "s": 0.0, "e": 0.0, "w": 0.0
            }
            current_cooling = stored_entry.get("cooling") or {
                "s": 0.0, "e": 0.0, "w": 0.0
            }

            implied_30d = _solve_normal(acc, acc["n"])

            # Physical-space implied (undo per-direction screen transmittance, #826)
            implied_physical = None
            if implied_30d is not None:
                # We can't perfectly recover avg transmittance from the log,
                # but screen_closed + screen_open counts give us a proxy.
                # Use the formula: physical = effective / transmittance, applied
                # per cardinal direction so unscreened facades are not divided
                # by an irrelevant factor.
                current_correction = self.coordinator.solar_correction_percent
                t_s, t_e, t_w = _SC._screen_transmittance_vector(
                    current_correction, self.coordinator.screen_config_for_entity(entity_id)
                )
                implied_physical = {
                    "s": round(implied_30d["s"] / t_s, 4) if t_s > 0.01 else implied_30d["s"],
                    "e": round(implied_30d["e"] / t_e, 4) if t_e > 0.01 else implied_30d["e"],
                    "w": round(implied_30d["w"] / t_w, 4) if t_w > 0.01 else implied_30d["w"],
                }

            # Stability windows
            stability = []
            for w in acc["windows"]:
                coeff = _solve_normal(w, w["n"], min_samples=5)
                stability.append({"coefficient": coeff, "qualifying_hours": w["n"]})

            # Flags
            flags = []
            if acc["qualifying"] > 0 and acc["saturated"] / acc["qualifying"] > 0.3:
                flags.append("high_saturation")
            # Coefficient stability: check if S component varies >2x between windows
            s_values = [w["coefficient"]["s"] for w in stability if w["coefficient"] is not None and abs(w["coefficient"]["s"]) > 0.01]
            if len(s_values) >= 2 and max(abs(v) for v in s_values) > 2 * min(abs(v) for v in s_values):
                flags.append("coefficient_unstable")
            # Under-prediction
            mean_delta = acc["sum_delta"] / acc["delta_n"] if acc["delta_n"] > 0 else 0.0
            if mean_delta < -0.1:
                flags.append("under_predicting_solar")
            elif mean_delta > 0.1:
                flags.append("over_predicting_solar")

            # Dominant component
            cs = abs(current.get("s", 0.0))
            ce = abs(current.get("e", 0.0))
            cw = abs(current.get("w", 0.0))
            total_c = cs + ce + cw
            dominant = "balanced"
            if total_c > 0.01:
                if cs / total_c > 0.9:
                    dominant = "south"
                elif ce / total_c > 0.9:
                    dominant = "east"
                elif cw / total_c > 0.9:
                    dominant = "west"

            # Solar shutdown diagnostics (#838): compare the 30-day implied
            # coefficient with and without shutdown hours.  A large gap
            # indicates those hours were biasing the learned coefficient.
            implied_no_shutdown = _solve_normal(acc["no_shutdown"], acc["no_shutdown"]["n"])
            if acc["shutdown_hours"] > 0 and acc["qualifying"] > 0:
                shutdown_pct = round(100 * acc["shutdown_hours"] / acc["qualifying"], 1)
            else:
                shutdown_pct = 0.0
            if acc["shutdown_hours"] >= 5:
                flags.append("solar_shutdown_detected")

            # Temperature-regime stratification (BP-relative, #826 follow-up).
            # First pass reports means only — no threshold flags until
            # empirical distributions from European summer data tell us what
            # "significant bias" looks like for this metric.  Buckets with
            # zero qualifying hours are emitted as {"n": 0} so the JSON
            # shape is stable across installations.
            temperature_stratified = {}
            for tb_key, tb in acc["temp_buckets"].items():
                if tb["n"] > 0:
                    temperature_stratified[tb_key] = {
                        "n": tb["n"],
                        "mean_delta_kwh": round(tb["delta_sum"] / tb["n"], 4),
                    }
                else:
                    temperature_stratified[tb_key] = {"n": 0}

            # Screen stratification (#826).  Report mean delta per correction
            # bucket along with n, so downstream (and humans) can distinguish
            # "tiny sample, noisy" from "real bias".
            #
            # Per-entity screen-scope (#963).  Entities outside
            # ``screen_affected_entities`` get a fixed transmittance of 1.0
            # at learn / predict time regardless of correction_percent.  The
            # screen_stratified binning is then a binning by something
            # correlated with screen position rather than by transmittance
            # itself — typically sun availability (closed = night / cloudy,
            # open = sunny day).  Reporting bias_gap_kwh + flagging
            # ``transmittance_floor_*`` on those entities surfaces a
            # confound (Simpson-style), not a model failure: a non-zero
            # bias_gap is the expected signature of binning by sun
            # magnitude.  Mark the block ``screen_config_active: false``
            # and suppress flag emission for those entities so downstream
            # consumers can interpret the numbers correctly.
            screen_config_for_entity_fn = getattr(
                self.coordinator, "screen_config_for_entity", None
            )
            if callable(screen_config_for_entity_fn):
                try:
                    _ent_screen_cfg = screen_config_for_entity_fn(entity_id)
                except (TypeError, ValueError):
                    _ent_screen_cfg = None
            else:
                _ent_screen_cfg = None
            screen_config_active = bool(_ent_screen_cfg) and any(_ent_screen_cfg)

            screen_stratified = {"screen_config_active": screen_config_active}
            for bkey, b in acc["correction_buckets"].items():
                if b["n"] > 0:
                    # Trimmed (#896 follow-up): only ``n`` and
                    # ``mean_delta_kwh`` are actionable.  ``mean_modeled_kwh``
                    # and ``mean_implied_kwh`` are reconstructable from the
                    # current coefficient + log breakdown if needed and
                    # added ~3× the bytes per bucket on every diagnose
                    # response.
                    screen_stratified[bkey] = {
                        "n": b["n"],
                        "mean_delta_kwh": round(b["delta_sum"] / b["n"], 4),
                    }
                else:
                    screen_stratified[bkey] = {"n": 0}
            # Bias trend: does |mean_delta| grow as screens close?  Only
            # meaningful when both extremes have enough samples AND the
            # model is actually applying transmittance to this entity.
            if (
                screen_config_active
                and acc["correction_buckets"]["open"]["n"] >= 10
                and acc["correction_buckets"]["closed"]["n"] >= 10
            ):
                open_bias = (
                    acc["correction_buckets"]["open"]["delta_sum"]
                    / acc["correction_buckets"]["open"]["n"]
                )
                closed_bias = (
                    acc["correction_buckets"]["closed"]["delta_sum"]
                    / acc["correction_buckets"]["closed"]["n"]
                )
                bias_gap = closed_bias - open_bias
                screen_stratified["bias_gap_kwh"] = round(bias_gap, 4)
                # bias_gap > 0.05 → closed over-predicts relative to open
                # → transmittance_model at closed is TOO LOW.  The model
                # assumes less sun passes through than reality; reconstructed
                # potential is inflated; coeff_learned absorbs a mix that
                # over-predicts on fully-closed hours.  The sensitivity
                # sweep below typically points at a higher optimal in this
                # regime, which is the fix: raise SCREEN_DIRECT_TRANSMITTANCE.
                # bias_gap < −0.05 → transmittance TOO HIGH (symmetric case).
                # Prior to the 1.3.3 fix these two flag names were swapped.
                if bias_gap > 0.05:
                    flags.append("transmittance_floor_too_low")
                elif bias_gap < -0.05:
                    flags.append("transmittance_floor_too_high")

            # Transmittance sensitivity sweep.  For each candidate value of
            # SCREEN_DIRECT_TRANSMITTANCE, re-solve the 3×3 normal equations
            # using potential reconstructed under that hypothesis, then
            # compute the residual RMSE against implied_solar.  The
            # hypothesis minimising RMSE is the empirically optimal floor
            # for this installation's data.
            sensitivity = None
            tuples = acc["sensitivity_tuples"]
            if len(tuples) >= 20:
                candidates = [0.05, 0.08, 0.12, 0.15, 0.20, 0.25]
                cfg = self.coordinator.screen_config_for_entity(entity_id)
                results = []
                # Correction-variance gate: if slider barely changes, the
                # sweep is uninformative — all candidates will yield similar
                # RMSE because transmittance(100%) == 1.0 for every candidate.
                corrections = [t[3] for t in tuples]
                corr_var = (
                    max(corrections) - min(corrections) if corrections else 0.0
                )
                for cand in candidates:
                    # Build 3×3 normal equations with potential = eff / t_cand
                    A_ss = A_ee = A_ww = 0.0
                    A_se = A_sw = A_ew = 0.0
                    b_s = b_e = b_w = 0.0
                    for eff_s, eff_e, eff_w, corr, implied in tuples:
                        # Per-direction transmittance under candidate floor
                        pct = max(0.0, min(100.0, corr)) / 100.0
                        t_screened = cand + (1.0 - cand) * pct
                        t_sc_s = t_screened if (cfg is None or cfg[0]) else 1.0
                        t_sc_e = t_screened if (cfg is None or cfg[1]) else 1.0
                        t_sc_w = t_screened if (cfg is None or cfg[2]) else 1.0
                        p_s = eff_s / t_sc_s if t_sc_s > 0.01 else eff_s
                        p_e = eff_e / t_sc_e if t_sc_e > 0.01 else eff_e
                        p_w = eff_w / t_sc_w if t_sc_w > 0.01 else eff_w
                        A_ss += p_s * p_s
                        A_ee += p_e * p_e
                        A_ww += p_w * p_w
                        A_se += p_s * p_e
                        A_sw += p_s * p_w
                        A_ew += p_e * p_w
                        b_s += p_s * implied
                        b_e += p_e * implied
                        b_w += p_w * implied
                    coeff_h = _solve_normal(
                        {"ss": A_ss, "ee": A_ee, "ww": A_ww,
                         "se": A_se, "sw": A_sw, "ew": A_ew,
                         "sI": b_s, "eI": b_e, "wI": b_w, "n": len(tuples)},
                        len(tuples),
                    )
                    if coeff_h is None:
                        continue
                    # Residual RMSE under the fitted coefficient
                    sse = 0.0
                    for eff_s, eff_e, eff_w, corr, implied in tuples:
                        pct = max(0.0, min(100.0, corr)) / 100.0
                        t_screened = cand + (1.0 - cand) * pct
                        t_sc_s = t_screened if (cfg is None or cfg[0]) else 1.0
                        t_sc_e = t_screened if (cfg is None or cfg[1]) else 1.0
                        t_sc_w = t_screened if (cfg is None or cfg[2]) else 1.0
                        p_s = eff_s / t_sc_s if t_sc_s > 0.01 else eff_s
                        p_e = eff_e / t_sc_e if t_sc_e > 0.01 else eff_e
                        p_w = eff_w / t_sc_w if t_sc_w > 0.01 else eff_w
                        pred = (
                            coeff_h["s"] * p_s
                            + coeff_h["e"] * p_e
                            + coeff_h["w"] * p_w
                        )
                        sse += (implied - pred) ** 2
                    rmse = (sse / len(tuples)) ** 0.5
                    results.append({
                        "screen_direct_transmittance": cand,
                        "implied_coefficient": coeff_h,
                        "residual_rmse_kwh": round(rmse, 4),
                    })
                if results:
                    best = min(results, key=lambda r: r["residual_rmse_kwh"])
                    # Trim (#896 follow-up): if every candidate produces
                    # essentially the same coefficient and RMSE, the sweep
                    # is uninformative — emit only the ``best`` row plus a
                    # ``verdict`` flag.  Saves 6 nearly-identical rows per
                    # entity on installs with low solar signal (small
                    # non-VP loads where implied coefficient is near zero
                    # so the candidate floor cannot move the fit).  Tests
                    # that need the full ``candidates`` list use a known-
                    # transmittance generative log where the sweep is
                    # genuinely informative.
                    rmses = [r["residual_rmse_kwh"] for r in results]
                    rmse_uniform = (max(rmses) - min(rmses)) < 0.01
                    first = results[0]["implied_coefficient"]
                    coeff_uniform = all(
                        abs(r["implied_coefficient"]["s"] - first["s"]) < 1e-3
                        and abs(r["implied_coefficient"]["e"] - first["e"]) < 1e-3
                        and abs(r["implied_coefficient"]["w"] - first["w"]) < 1e-3
                        for r in results
                    )
                    sensitivity = {
                        "n_hours": len(tuples),
                        "correction_range_pct": round(corr_var, 1),
                        "informative": corr_var >= 40.0,  # ≥40 pct points of slider variance
                        "best": best,
                        # Per-entity screen-scope (#963).  When the entity
                        # is unscreened the sweep is structurally a no-op
                        # — its best.screen_direct_transmittance is not a
                        # recommendation that the model would act on.
                        "screen_config_active": screen_config_active,
                    }
                    if rmse_uniform and coeff_uniform:
                        sensitivity["verdict"] = "uniform_across_candidates"
                    else:
                        sensitivity["candidates"] = results
                    if (
                        screen_config_active
                        and sensitivity["informative"]
                        and abs(best["screen_direct_transmittance"] - 0.08) > 0.04
                    ):
                        flags.append("sensitivity_suggests_transmittance_retune")

            # Inequality-replay coefficient (#865) — what the learner would
            # produce if retrained from zero over the same window.  Absence
            # means the unit did not qualify for any update (no shutdown
            # hours, or base below SOLAR_SHUTDOWN_MIN_BASE everywhere).
            # Mode-stratified per #868 — replay writes per-regime; inequality
            # is heating-only by #865 design, so we report the heating regime
            # of the shadow output.  ``None`` means no inequality update fired.
            shadow_entry = shadow_coeffs.get(entity_id)
            implied_inequality_coeff = None
            if isinstance(shadow_entry, dict):
                heating_shadow = shadow_entry.get("heating")
                if isinstance(heating_shadow, dict) and any(
                    heating_shadow.get(k) for k in ("s", "e", "w")
                ):
                    implied_inequality_coeff = {
                        k: round(v, 4) for k, v in heating_shadow.items()
                    }

            # Mode-stratified split (#868): scalar percentage divergence
            # between the heating and cooling regimes, averaged over the
            # three directions.  Stable across regime swaps because we
            # take absolute differences and normalise by the symmetric
            # mean.  ``None`` when both regimes are zero (cooling never
            # learned on a heating-only install — the seeded copy from
            # migration drifts away as cooling-mode hours arrive).
            h_dir = current_heating
            c_dir = current_cooling
            denom = sum(
                abs(h_dir.get(k, 0.0)) + abs(c_dir.get(k, 0.0))
                for k in ("s", "e", "w")
            )
            if denom > 0.001:
                split_delta_pct = (
                    100.0
                    * sum(
                        abs(h_dir.get(k, 0.0) - c_dir.get(k, 0.0))
                        for k in ("s", "e", "w")
                    )
                    / denom
                )
                coefficient_split_delta = round(split_delta_pct, 1)
            else:
                coefficient_split_delta = None

            # Tobit MLE (#904 stage 0+1, shadow-only).  Censoring-aware
            # estimator surfaced alongside ``implied_coefficient_30d``
            # (which drops saturated rows) and ``implied_coefficient_inequality``
            # (which lower-bounds via shutdown).  Modulating-regime fit
            # only — shutdown rows excluded per CHOICE 3, saturated
            # rows kept as right-censored at ``T = 0.95×base`` per
            # CHOICE 2.  No production wiring; informational diagnostic
            # for stage-1 evidence collection.  Heating regime only at
            # this stage (mirrors the existing implied_30d display
            # convention; cooling Tobit deferred until the heating
            # path validates).
            try:
                tobit_fit = self.coordinator.learning.compute_tobit_for_diagnose(
                    self.coordinator._hourly_log,
                    entity_id,
                    "heating",
                    self.coordinator,
                    unit_min_base=self.coordinator._per_unit_min_base_thresholds or None,
                    days_back=days_back,
                )
            except Exception as exc:  # noqa: BLE001
                # Defensive: shadow diagnostic must never break diagnose_solar.
                # Record the exception class/message so a swallowed failure
                # is diagnosable from the service response alone — a bare
                # "exception" label hid a structural NameError for weeks.
                tobit_fit = {
                    "coefficient": None,
                    "skip_reason": "exception",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                }
            tobit_coeff = tobit_fit.get("coefficient")
            tobit_diagnostics = {
                "iterations": tobit_fit.get("iterations", 0),
                "converged": tobit_fit.get("converged", False),
                "failure_reason": tobit_fit.get("failure_reason"),
                "sigma": tobit_fit.get("sigma"),
                "log_likelihood": tobit_fit.get("log_likelihood"),
                "n_uncensored": tobit_fit.get("n_uncensored", 0),
                "n_censored": tobit_fit.get("n_censored", 0),
                "censored_fraction": tobit_fit.get("censored_fraction", 0.0),
                "n_eff": tobit_fit.get("n_eff", 0.0),
                "skip_reason": tobit_fit.get("skip_reason"),
            }

            # Live Tobit-learner state (#904 stage 3).  Surfaces the
            # running sufficient-statistic snapshot for the maintainer's
            # validation window — without this there's no per-hour
            # observability into the live learner's convergence.
            # Always emitted (even when the master flag is off) so
            # callers walking the per_unit dict don't need conditional
            # presence checks.  The ``enabled`` and ``allow_listed``
            # fields disambiguate active vs dormant state.
            from .const import SOLAR_MODEL_VERSION as _CURRENT_SOLAR_VERSION
            tobit_live_stats = (
                self.coordinator._tobit_sufficient_stats.get(entity_id, {})
                if isinstance(getattr(self.coordinator, "_tobit_sufficient_stats", None), dict)
                else {}
            )
            shadow_entry = (
                self.coordinator._nlms_shadow_coefficients.get(entity_id)
                if isinstance(getattr(self.coordinator, "_nlms_shadow_coefficients", None), dict)
                else None
            )

            def _build_regime_block(regime_name: str) -> dict:
                slot = tobit_live_stats.get(regime_name) or {}
                samples = slot.get("samples", [])
                n_unc = sum(1 for s in samples if not s[4])
                n_cens = sum(1 for s in samples if s[4])
                last_step = slot.get("last_step", {})
                shadow_regime = (
                    shadow_entry.get(regime_name) if isinstance(shadow_entry, dict) else None
                )
                return {
                    "in_cold_start": (
                        last_step.get("skip_reason") in (
                            "insufficient_uncensored",
                            "insufficient_effective_samples",
                        )
                        or last_step.get("converged") is not True
                    ),
                    "n_uncensored": n_unc,
                    "n_censored": n_cens,
                    "n_eff": last_step.get("n_eff", float(n_unc)),
                    "last_step_iterations": last_step.get("iterations", 0),
                    "last_step_failure_reason": last_step.get("failure_reason"),
                    "last_step_norm": last_step.get("step_norm", 0.0),
                    "last_step_skip_reason": last_step.get("skip_reason"),
                    "sigma": last_step.get("sigma"),
                    "current_coefficient_nlms_shadow": (
                        {k: round(float(shadow_regime.get(k, 0.0)), 4) for k in ("s", "e", "w")}
                        if isinstance(shadow_regime, dict)
                        else None
                    ),
                    "solar_model_version": slot.get(
                        "solar_model_version", _CURRENT_SOLAR_VERSION
                    ),
                    "samples_since_reset": slot.get("samples_since_reset", 0),
                }

            # Both regimes surfaced (review I4, #912).  Cooling-active
            # entities and dual-mode VPs need observability into both
            # slots; previously only heating was reported and cooling
            # was invisible.  Top-level ``enabled`` / scope state apply
            # to both regimes equally so they live at the parent.
            #
            # Scope semantics (1.3.5+ default-on):
            # ``in_scope_override`` reflects whether the entity is in
            # the optional scope-restriction list (non-empty list =
            # only listed entities try Tobit).  ``tobit_routed_to_live``
            # reflects whether Tobit is actually running for this
            # entity at this hour given all gates: master flag enabled
            # AND (auto-mode OR in scope) AND not MPC-managed.  A user
            # walking ``per_unit[entity_id].live_tobit_state`` should
            # consult ``tobit_routed_to_live`` for "is Tobit on for me",
            # not the legacy ``allow_listed`` field.
            tobit_flag = bool(getattr(
                self.coordinator, "_experimental_tobit_live_learner", False
            ))
            scope_list = getattr(
                self.coordinator, "_tobit_live_entities", frozenset()
            )
            mpc_managed = frozenset(
                eid
                for eid, strat in (
                    getattr(self.coordinator, "_unit_strategies", {}) or {}
                ).items()
                if (
                    strat is not None
                    and strat.__class__.__name__ == "WeightedSmear"
                    and getattr(strat, "use_synthetic", False)
                )
            )
            in_scope = (not scope_list) or (entity_id in scope_list)
            tobit_routed = (
                tobit_flag and in_scope and entity_id not in mpc_managed
            )
            live_tobit_state = {
                "enabled": tobit_flag,
                "scope_mode": "auto" if not scope_list else "override",
                "in_scope_override": (
                    entity_id in scope_list if scope_list else None
                ),
                "tobit_routed_to_live": tobit_routed,
                # Legacy field name preserved for backward compatibility
                # with consumers that walked ``allow_listed`` under the
                # pre-1.3.5 opt-in semantic — now reports the effective
                # routed state.  New consumers should use
                # ``tobit_routed_to_live`` directly.
                "allow_listed": tobit_routed,
                "heating": _build_regime_block("heating"),
                "cooling": _build_regime_block("cooling"),
            }

            # Inactive-unit collapse (#896 follow-up).  Sensors with no
            # learned coefficient, no saturation, no shutdown signal, and
            # no flags carry no actionable information in the verbose
            # blocks (transmittance_sensitivity is structurally uniform
            # because the implied coefficient is ~0 so candidate floors
            # cannot move the fit; screen_stratified and stability_windows
            # are noise around zero; temporal_bias adds nothing the global
            # block doesn't already report).  Emit a minimal record so the
            # entity stays addressable via ``per_unit[entity_id]`` for any
            # consumer that walks the dict, but drop the verbose tail that
            # was bloating diagnose_solar output by ~70 % on installs with
            # many small non-VP energy sensors.  Backward-compatible: every
            # field that existing tests assert on for zero-coeff fixtures
            # (current_coefficient_*, coefficient_split_delta_pct,
            # implied_coefficient_30d, qualifying_hours, mean_delta_kwh,
            # flags) is preserved.
            heating_zero = all(abs(v) < 1e-6 for v in current_heating.values())
            cooling_zero = all(abs(v) < 1e-6 for v in current_cooling.values())
            is_inactive = (
                heating_zero
                and cooling_zero
                and acc["saturated"] == 0
                and acc["shutdown_hours"] == 0
                and not flags
            )
            if is_inactive:
                per_unit[entity_id] = {
                    "inactive": True,
                    "current_coefficient": {k: round(v, 4) for k, v in current.items()},
                    "current_coefficient_heating": {
                        k: round(v, 4) for k, v in current_heating.items()
                    },
                    "current_coefficient_cooling": {
                        k: round(v, 4) for k, v in current_cooling.items()
                    },
                    "coefficient_split_delta_pct": coefficient_split_delta,
                    "implied_coefficient_30d": implied_30d,
                    "qualifying_hours": acc["n"],
                    "mean_delta_kwh": round(mean_delta, 4),
                    # Stage 3 (#912) review I5: live_tobit_state must
                    # appear on the inactive branch too.  An entity
                    # that's allow-listed but has no qualifying hours
                    # collapses here, and ``allow_listed`` /
                    # ``enabled`` are the only place to confirm the
                    # gate is actually active for it.  Hides the
                    # verbose per-regime detail (sample lists are
                    # always empty on inactive units) but keeps the
                    # gate-status fields.
                    "live_tobit_state": {
                        "enabled": live_tobit_state["enabled"],
                        "allow_listed": live_tobit_state["allow_listed"],
                        "heating": {
                            "n_uncensored": live_tobit_state["heating"]["n_uncensored"],
                            "n_censored": live_tobit_state["heating"]["n_censored"],
                            "samples_since_reset": live_tobit_state["heating"]["samples_since_reset"],
                        },
                        "cooling": {
                            "n_uncensored": live_tobit_state["cooling"]["n_uncensored"],
                            "n_censored": live_tobit_state["cooling"]["n_censored"],
                            "samples_since_reset": live_tobit_state["cooling"]["samples_since_reset"],
                        },
                    },
                    "flags": flags,
                }
            else:
                per_unit[entity_id] = {
                    "current_coefficient": {k: round(v, 4) for k, v in current.items()},
                    "current_coefficient_heating": {
                        k: round(v, 4) for k, v in current_heating.items()
                    },
                    "current_coefficient_cooling": {
                        k: round(v, 4) for k, v in current_cooling.items()
                    },
                    "coefficient_split_delta_pct": coefficient_split_delta,
                    "implied_coefficient_30d": implied_30d,
                    "implied_coefficient_30d_no_shutdown": implied_no_shutdown,
                    "implied_coefficient_inequality": implied_inequality_coeff,
                    "implied_coefficient_physical": implied_physical,
                    "implied_coefficient_tobit_30d": tobit_coeff,
                    "tobit_diagnostics": tobit_diagnostics,
                    "live_tobit_state": live_tobit_state,
                    "stability_windows": stability,
                    "mean_delta_kwh": round(mean_delta, 4),
                    "saturation_pct": round(100 * acc["saturated"] / acc["qualifying"], 1) if acc["qualifying"] > 0 else 0.0,
                    "shutdown_hours_30d": acc["shutdown_hours"],
                    "shutdown_pct_of_qualifying": shutdown_pct,
                    "dominant_component": dominant,
                    "qualifying_hours": acc["n"],
                    "temperature_stratified": temperature_stratified,
                    "screen_stratified": screen_stratified,
                    "transmittance_sensitivity": sensitivity,
                    "temporal_bias": {
                        "morning_mean_delta": round(acc["morning_delta"] / acc["morning_n"], 4) if acc["morning_n"] > 0 else None,
                        "afternoon_mean_delta": round(acc["afternoon_delta"] / acc["afternoon_n"], 4) if acc["afternoon_n"] > 0 else None,
                    },
                    "elevation_diagnostics": {
                        "instantaneous": _build_elevation_block(acc["elevation_buckets"]),
                        "lag": _build_elevation_lag_block(acc["elevation_lag_buckets"]),
                        "evening_tail": _build_elevation_evening_tail_block(
                            acc["elevation_evening_tail_buckets"]
                        ),
                    },
                    "last_batch_fit": self._format_last_batch_fit(entity_id),
                    "flags": flags,
                }

        # Global metrics
        global_flags = []

        # Battery health
        battery_health = {}
        if battery_residuals:
            # Dispersion, sample-size gate and the relative-deviation
            # filter all live in ``_assess_battery_bias`` (#1066) so the
            # decision is one named, testable mapping rather than a
            # chain of elifs inside a 2000-line method.
            bias = _assess_battery_bias(battery_residuals, battery_expected)
            assessment = bias["assessment"]

            battery_health = {
                "mean_residual_kwh": bias["mean_residual_kwh"],
                # Mean prediction over the same hours, and the resulting
                # fractional miss.  Reported, not just used: a reader who
                # disagrees with the threshold needs the number it was
                # compared against.
                "mean_expected_kwh": bias["mean_expected_kwh"],
                "relative_deviation": bias["relative_deviation"],
                "std_residual_kwh": bias["std_residual_kwh"],
                "std_error_kwh": bias["std_error_kwh"],
                "qualifying_post_sunset_hours": bias["n_hours"],
                "min_hours_for_assessment": BATTERY_BIAS_MIN_HOURS,
                "bias_threshold_kwh": BATTERY_RESIDUAL_BIAS_KWH,
                "bias_threshold_relative": BATTERY_RESIDUAL_BIAS_RELATIVE,
                "decay_rate": self.coordinator.solar_battery_decay,
                "assessment": assessment,
            }
            if assessment in ("too_fast", "too_slow"):
                global_flags.append(f"battery_decay_{assessment}")

        # Joint (decay, k) battery calibration with counterfactual residuals.
        #
        # Replaces the prior 1-D decay sweep, which had three statistical
        # defects (#902 statistics review):
        #
        #   1. The estimator was biased toward the live decay: ``actual −
        #      expected`` is the residual against the LIVE model, not a
        #      counterfactual residual under the candidate decay.  Hours
        #      where the live model already credited enough battery were
        #      filtered out before the candidate saw them, biasing the
        #      recommendation toward status quo.
        #   2. Mean-residual minimisation is the wrong loss — a candidate
        #      that systematically over-credits some hours and under-credits
        #      others equally can score zero mean while tracking the data
        #      poorly.  Use RMSE.
        #   3. ``decay`` and ``k`` are jointly unidentified from post-sunset
        #      residuals.  Sequential calibration converges to order-
        #      dependent local optima — both must be swept jointly.
        #
        # Method: for each (decay_alt, k_alt) candidate, replay BOTH batteries
        # (main solar EMA + carryover EMA) starting from 0 over each day's
        # hours, AND replay the live (decay, k) the same way.  The difference
        # in release between the two replays is the counterfactual delta;
        # adding it to the live residual yields the residual the system
        # would have produced under the candidate.  Score by RMSE on
        # post-sunset hours.
        #
        # Post-sunset definition: per day, the ``POST_SUNSET_REPLAY_HOURS``
        # hours immediately after the last hour with raw_solar > 0.  Uses
        # the raw signal (``solar_impact_raw_kwh``) — the post-coefficient
        # × raw-vector value, which is 0 by construction when the sun is
        # below the horizon — NOT the post-battery ``solar_factor`` which
        # carries battery residue across midnight.  Restricting to the
        # window where the battery is observably charged improves SNR
        # (pre-dawn hours have battery ≈ 0 and tell us nothing about
        # decay).
        #
        # Initial-state caveat: from-0 daily replay underestimates the
        # live system's actual battery state on the morning of each day
        # by an exponentially decaying residual from yesterday's evening.
        # After ~3 half-lives (~9 h at decay 0.80) the bias is < 12 %.
        # POST_SUNSET_REPLAY_HOURS = 6 captures the 1-2 half-life window
        # where signal-to-noise is highest; the from-0 approximation is
        # acceptable here because the post-sunset evening is far past the
        # morning when the bias was largest.
        DECAY_GRID = [round(0.50 + 0.05 * i, 2) for i in range(10)]   # 0.50..0.95
        # Locked to the single value the system can actually run.
        #
        # ``battery_thermal_feedback_k`` was retired in 1.3.5: the UI was
        # removed and ``coordinator.__init__`` strips the key from
        # ``entry.data`` on every init, forcing 0.0.  Sweeping k anyway
        # meant this argmin searched a space the system has decided not to
        # occupy — and because the apply path wrote the whole ``best``
        # pair, an install could be handed ``(decay=0.70, k=0.40)``, keep
        # the decay (not stripped) and lose the k (stripped on the next
        # restart).  The result was a decay fitted *conditional on k=0.40*
        # running permanently at k=0.0: a combination the sweep evaluated
        # and did not select.  Nothing guarantees
        # ``argmin decay | k=0.4 == argmin decay | k=0.0``.
        #
        # The k dimension is not deleted, only pinned — the tuple shape,
        # ``_replay_score``'s k argument and the ``"decay,k"`` surface keys
        # all survive, so reviving the feature is a one-line change and the
        # surface keys still document which k the fit is conditional on.
        # The evidence base for revisiting the retirement lives in
        # ``battery_feedback_sweep``, which still sweeps k as observability
        # (it just no longer votes — see ``battery_feedback_verdict``).
        K_GRID = [0.0]
        POST_SUNSET_REPLAY_HOURS = 6
        MIN_POST_SUNSET_HOURS_FOR_RECOMMENDATION = 5

        calibration: dict = {}
        if day_sequences:
            decay_live = self.coordinator.solar_battery_decay
            # ``k_live`` feeds ``k_live > 0.0`` inside ``_replay_score``,
            # the ``changed`` flag, and the apply gate, so a non-numeric
            # value raises TypeError mid-sweep and a string makes
            # ``changed`` permanently True against the float grid.  All
            # four reads of this attribute in ``diagnose_solar`` go
            # through ``_coerce_scalar`` so they cannot disagree.
            #
            # For *this* attribute the guard is consistency, not hardening
            # against real input: ``coordinator.__init__`` strips
            # ``CONF_BATTERY_THERMAL_FEEDBACK_K`` from ``entry.data`` on
            # every init and assigns the module default, and the only
            # other writer is the apply path below, which writes from
            # ``K_GRID``.  ``solar_battery_decay`` is the read of this
            # shape that *does* arrive from config; it is coerced at the
            # coordinator boundary instead, which is the better place when
            # the value has consumers outside this method.
            k_live = _coerce_scalar(
                self.coordinator.battery_thermal_feedback_k, 0.0
            )

            # Build per-day post-sunset hour set: the N hours immediately
            # after the last hour with raw_solar > 0.01.  Days with no
            # qualifying sunny hour (e.g. fully overcast) contribute nothing.
            post_sunset_set_by_day: dict[str, set[int]] = {}
            # Build per-day morning hour set: from the first hour with
            # raw_solar > 0.01 through the hour where raw_solar peaks
            # (inclusive on both ends).  This is the rising-sun phase
            # where charge-side dynamics differ between models — instant-
            # respons would credit fast, EMA accumulates slowly.  Plateau
            # hours past the peak are deliberately excluded: both models
            # converge to steady state there, so they carry no
            # discriminating information about decay vs instant credit.
            #
            # The two windows together separate the two physical regimes
            # the battery model captures: post-sunset = pure decay tail
            # (current model fits this); morning = charge ramp (the
            # asymmetric-charge gap from #896's deferred scope).  The
            # tail/morning RMSE disagreement at any single (decay, k) is
            # the diagnostic that confirms whether asymmetric handling is
            # needed (large gap → yes) or not (small gap → no, single-
            # decay model is sufficient).
            morning_set_by_day: dict[str, set[int]] = {}
            for day_key, hours in day_sequences.items():
                last_sunny_h = -1
                first_sunny_h = -1
                peak_solar = 0.0
                peak_hour = -1
                for h, raw, _wasted, _act, _exp in hours:
                    if raw > 0.01:
                        if first_sunny_h < 0:
                            first_sunny_h = h
                        if h > last_sunny_h:
                            last_sunny_h = h
                        if raw > peak_solar:
                            peak_solar = raw
                            peak_hour = h
                if last_sunny_h >= 0:
                    post_sunset_set_by_day[day_key] = {
                        last_sunny_h + i for i in range(1, POST_SUNSET_REPLAY_HOURS + 1)
                    }
                if first_sunny_h >= 0 and peak_hour >= first_sunny_h:
                    morning_set_by_day[day_key] = set(
                        range(first_sunny_h, peak_hour + 1)
                    )

            def _replay_score(
                decay_alt: float,
                k_alt: float,
                window_by_day: dict[str, set[int]],
            ) -> tuple[float, int, list[float]]:
                """Counterfactual replay scored over ``window_by_day``.

                Returns (rmse, n_hours_evaluated, per_hour_residuals).
                Window-agnostic — the replay recurrence and counterfactual
                residual formula are the same regardless of which hours
                feed into the SSE.

                The residual list is returned rather than only its SSE so
                the significance gate (#1066) can run a paired test
                against another candidate's residuals.  Iteration order is
                deterministic — ``day_sequences`` insertion order, then
                ``hours_sorted`` — so two calls over the same window
                produce index-aligned lists describing the same hours,
                which is what makes the pairing exact.
                """
                sse = 0.0
                n = 0
                residuals: list[float] = []
                for day_key, hours in day_sequences.items():
                    window = window_by_day.get(day_key)
                    if not window:
                        continue
                    hours_sorted = sorted(hours, key=lambda x: x[0])
                    main_alt = main_live = 0.0
                    carry_alt = carry_live = 0.0
                    for h, raw, wasted, actual, expected in hours_sorted:
                        # Live replay
                        main_live = main_live * decay_live + raw * (1 - decay_live)
                        live_carry_in = (
                            k_live * wasted if k_live > 0.0 else 0.0
                        )
                        carry_live = (
                            carry_live * decay_live + live_carry_in * (1 - decay_live)
                        )
                        live_release = (
                            main_live + k_live * carry_live * (1 - decay_live)
                        )
                        # Alt replay
                        main_alt = main_alt * decay_alt + raw * (1 - decay_alt)
                        alt_carry_in = k_alt * wasted if k_alt > 0.0 else 0.0
                        carry_alt = (
                            carry_alt * decay_alt + alt_carry_in * (1 - decay_alt)
                        )
                        alt_release = (
                            main_alt + k_alt * carry_alt * (1 - decay_alt)
                        )
                        if h in window:
                            # Counterfactual derivation:
                            #   base[t]      = expected[t] + live_release[t]
                            #   expected_alt = base[t] − alt_release[t]
                            #               = expected + (live_release − alt_release)
                            #   residual_alt = actual − expected_alt
                            #               = (actual − expected) + (alt_release − live_release)
                            # Minimised at alt = truth where alt_release matches the
                            # release that produced ``actual``.
                            residual_alt = (actual - expected) + (alt_release - live_release)
                            sse += residual_alt * residual_alt
                            n += 1
                            residuals.append(residual_alt)
                rmse = (sse / n) ** 0.5 if n > 0 else float("inf")
                return rmse, n, residuals

            # Post-sunset surface — the original tail-decay scoring.
            # Recommendation is driven by this surface (the live battery
            # model is parameterised for tail behaviour; morning is read-
            # only diagnostic until asymmetric-charge support lands).
            surface: dict[str, float] = {}
            best = (decay_live, k_live)
            best_rmse = float("inf")
            n_post_sunset = 0
            best_residuals: list[float] = []
            for d_alt in DECAY_GRID:
                for k_alt in K_GRID:
                    rmse, n_post_sunset, resid = _replay_score(
                        d_alt, k_alt, post_sunset_set_by_day
                    )
                    if n_post_sunset < MIN_POST_SUNSET_HOURS_FOR_RECOMMENDATION:
                        continue
                    surface[f"{d_alt},{k_alt}"] = round(rmse, 4)
                    # ``1e-6`` is retained deliberately.  It guards float
                    # equality in *selection* — argmin should pick the
                    # minimum, and that was never the bug.  Whether the
                    # winning margin is worth reporting is a separate
                    # question, answered by the significance gate below.
                    if rmse < best_rmse - 1e-6:
                        best_rmse = rmse
                        best = (d_alt, k_alt)
                        best_residuals = resid

            # Morning surface — read-only diagnostic.  Same grid + same
            # counterfactual, but scored over the rising-sun window per
            # day.  Reveals whether any (decay, k) candidate would also
            # fit charge-side behaviour, or whether tail-best and morning-
            # best diverge (= asymmetric-charge gap).
            morning_surface: dict[str, float] = {}
            morning_best = (decay_live, k_live)
            morning_best_rmse = float("inf")
            n_morning = 0
            for d_alt in DECAY_GRID:
                for k_alt in K_GRID:
                    rmse, n_morning, _resid = _replay_score(
                        d_alt, k_alt, morning_set_by_day
                    )
                    if n_morning < MIN_POST_SUNSET_HOURS_FOR_RECOMMENDATION:
                        continue
                    morning_surface[f"{d_alt},{k_alt}"] = round(rmse, 4)
                    if rmse < morning_best_rmse - 1e-6:
                        morning_best_rmse = rmse
                        morning_best = (d_alt, k_alt)

            # Live config's own RMSE on the same post-sunset hours, computed
            # via the same replay path so the comparison is apples-to-apples.
            # When the live (decay_live, k_live) sits inside the swept grids,
            # this equals surface[f"{decay_live},{k_live}"]; computing it
            # explicitly handles the case where live values fall between
            # grid points (e.g. decay 0.82).
            live_rmse, _, live_residuals = _replay_score(
                decay_live, k_live, post_sunset_set_by_day
            )
            morning_live_rmse, _, _ = _replay_score(
                decay_live, k_live, morning_set_by_day
            )

            # Tail/morning disagreement at the post-sunset-recommended
            # candidate.  If small (≲ 0.05 kWh) the post-sunset
            # recommendation also fits morning — single-decay model is
            # sufficient.  If large (≳ 0.10 kWh) post-sunset and morning
            # want different decay + k — asymmetric handling is what's
            # left to fix.  Computed as |best_post_sunset_rmse −
            # rmse_at_morning(post_sunset_best)|: how much worse the
            # tail-optimised candidate is on morning RMSE compared to
            # what's achievable on morning.
            morning_at_tail_best = (
                morning_surface.get(f"{best[0]},{best[1]}")
                if best_rmse != float("inf") else None
            )
            tail_morning_disagreement = (
                round(morning_at_tail_best - morning_best_rmse, 4)
                if (
                    morning_at_tail_best is not None
                    and morning_best_rmse != float("inf")
                )
                else None
            )

            # --- Is the recommendation worth showing? (#1066) ---------
            # Three independent reasons to withhold it, all computed
            # here so the verdict below reads as a lookup rather than a
            # second decision procedure.
            changed = (best[0] != decay_live or best[1] != k_live)
            # ``len(surface)`` is the number of candidates that actually
            # qualified and competed in the argmin — not len(DECAY_GRID) ×
            # len(K_GRID), since candidates below the hour floor are
            # skipped and never had a chance to win.  Penalising for
            # comparisons that did not happen would be as wrong as
            # penalising for none.
            significance = paired_loss_improvement(
                live_residuals,
                best_residuals,
                n_candidates_considered=max(1, len(surface)),
            )

            # Did the sweep produce evidence at all?  Two ways it can run
            # and conclude nothing, and neither is visible in the hour
            # count (which is candidate-independent, so it stays truthy
            # while the surface is empty):
            #
            #   1. No candidate cleared MIN_POST_SUNSET_HOURS_FOR_
            #      RECOMMENDATION, so ``surface`` is empty and ``best``
            #      is still the live seed.
            #   2. The surface populated but fewer than
            #      BATTERY_RECOMMENDATION_MIN_PAIRED_HOURS paired hours
            #      exist, so the screen declined before measuring
            #      anything.  ``below_noise_floor`` would imply a
            #      measurement was made and came up short; it was not.
            #
            # Consumed by ``battery_decay_verdict``, which reports
            # ``insufficient_data`` and lets the bias reading speak.
            sweep_produced_evidence = bool(surface) and (
                significance.get("declined_reason") != "too_few_paired_hours"
            )

            # A minimum at the edge of the swept grid means the sweep did
            # not bracket it — the surface is flat, monotone, or the true
            # optimum lies outside.  Reporting the edge value as "the
            # empirical optimum" presents "the sweep found nothing" as
            # "the sweep found 0.95".
            decay_at_boundary = changed and best[0] in (
                DECAY_GRID[0], DECAY_GRID[-1]
            )

            # Two windows disagreeing about the optimum is evidence that
            # the recommendation is window-specific overfitting rather
            # than a property of the building.  Previously computed,
            # reported next to the recommendation, and not acted on.
            windows_disagree = bool(
                changed
                and morning_best_rmse != float("inf")
                and morning_best[0] != best[0]
            )

            calibration = {
                "current_decay": decay_live,
                "current_k": k_live,
                "current_rmse_kwh": round(live_rmse, 4) if live_rmse != float("inf") else None,
                "recommended_decay": best[0],
                "recommended_k": best[1],
                "recommended_rmse_kwh": round(best_rmse, 4) if best_rmse != float("inf") else None,
                "rmse_improvement_kwh": (
                    round(live_rmse - best_rmse, 4)
                    if (live_rmse != float("inf") and best_rmse != float("inf"))
                    else None
                ),
                "rmse_surface": surface,
                "post_sunset_hours_evaluated": n_post_sunset,
                "post_sunset_replay_hours_per_day": POST_SUNSET_REPLAY_HOURS,
                # Morning-window diagnostic block (read-only — does NOT
                # drive recommendation).  Surfaces the asymmetric-charge
                # gap from #896's deferred scope.
                "morning_current_rmse_kwh": (
                    round(morning_live_rmse, 4)
                    if morning_live_rmse != float("inf") else None
                ),
                "morning_recommended_decay": morning_best[0],
                "morning_recommended_k": morning_best[1],
                "morning_recommended_rmse_kwh": (
                    round(morning_best_rmse, 4)
                    if morning_best_rmse != float("inf") else None
                ),
                "morning_rmse_surface": morning_surface,
                "morning_hours_evaluated": n_morning,
                "tail_morning_disagreement_kwh": tail_morning_disagreement,
                # Recommendation gating (#1066).  ``recommended_decay``
                # above remains the raw argmin — these say whether it
                # should be surfaced as advice.
                "recommendation_significance": significance,
                "sweep_produced_evidence": sweep_produced_evidence,
                "qualifying_candidates": len(surface),
                "optimum_at_sweep_boundary": decay_at_boundary,
                "windows_disagree": windows_disagree,
                "recommendation_withheld_reason": (
                    None if not changed
                    else "windows_disagree" if windows_disagree
                    else "optimum_at_sweep_boundary" if decay_at_boundary
                    else "below_noise_floor" if not significance["significant"]
                    else None
                ),
                # No longer a joint sweep: k is pinned at the retired
                # feature's fixed value, so this searches decay alone and
                # the name says so.  ``k_swept`` echoes what was actually
                # searched, since the ``"decay,k"`` surface keys otherwise
                # suggest two free dimensions.
                "method": "decay_counterfactual_replay_at_fixed_k",
                "k_swept": list(K_GRID),
                "loss": "rmse_post_sunset",
                # Named a screen, not a significance test: the candidate
                # is the argmin over the grid and is tested on the
                # residuals that selected it, so the t statistic is not a
                # p-value.  See const.BATTERY_RECOMMENDATION_MIN_T.
                "recommendation_screen": (
                    "paired_diff_of_squared_residuals; base_t="
                    f"{BATTERY_RECOMMENDATION_MIN_T}"
                    " + sqrt(2 ln m) selection penalty"
                    " + lag1 serial-correlation SE inflation;"
                    " NOT a calibrated p-value"
                ),
            }
            # Apply gate now respects the withholding chain (#1066 review).
            # Previously a single call could emit
            # ``recommendation_withheld_reason: "windows_disagree"`` and
            # persist that same value to ``entry.data`` in the same pass —
            # the payload saying "do not act" while acting.  The service
            # flag remains explicit and user-driven; what changed is that
            # it now applies the recommendation the diagnostic actually
            # made, rather than the raw argmin the diagnostic declined to
            # make.  ``apply_skipped_reason`` is surfaced below so a user
            # who passed the flag and saw nothing change learns why.
            apply_skipped_reason = None
            if apply_battery_decay and changed and calibration.get(
                "recommendation_withheld_reason"
            ):
                apply_skipped_reason = calibration[
                    "recommendation_withheld_reason"
                ]
            calibration["apply_requested"] = bool(apply_battery_decay)
            calibration["apply_skipped_reason"] = apply_skipped_reason
            if apply_battery_decay and (
                best[0] != decay_live or best[1] != k_live
            ) and best_rmse != float("inf") and apply_skipped_reason is None:
                old_decay = decay_live
                self.coordinator.solar_battery_decay = best[0]
                # Decay ONLY.  ``battery_thermal_feedback_k`` is retired
                # (see ``K_GRID``) and must not be written here: the key
                # is stripped from ``entry.data`` by
                # ``coordinator.__init__`` on the next restart, so writing
                # it produced a value that worked for hours or days and
                # then silently reverted — while the decay written in the
                # same call, which is NOT stripped, survived as a value
                # fitted under a k the install no longer had.
                #
                # With ``K_GRID = [0.0]`` this is belt and braces:
                # ``best[1]`` cannot be non-zero.  Kept explicit anyway so
                # that widening the grid for research cannot silently
                # restore the write.
                new_data = {
                    **self.coordinator.entry.data,
                    "solar_battery_decay": best[0],
                }
                self.coordinator.hass.config_entries.async_update_entry(
                    self.coordinator.entry, data=new_data
                )
                calibration["applied"] = True
                _LOGGER.info(
                    "Battery decay calibration applied: %.2f → %.2f (k fixed at %.2f)",
                    old_decay, best[0], k_live,
                )

        # Carry-over reservoir feedback sweep (#896 follow-up).  Replays
        # the carryover-state EMA over the window for each k candidate
        # and reports per-cell residual delta vs the live (k=0) baseline.
        #
        # OBSERVABILITY ONLY.  ``battery_thermal_feedback_k`` is retired
        # (see ``K_GRID`` above); this sweep still searches k because it is
        # the evidence base for that retirement, but its verdict is
        # ``research_optimum_k_*`` and does not raise ``any_action``.  It
        # answers "would k have helped", never "set k to this".
        #
        # As of split-state implementation, this sweep models the LIVE
        # wiring: ``_solar_carryover_state`` is charged by ``k × wasted``
        # and its release ``state × (1 − decay)`` subtracts from
        # heating-mode demand prediction in ``calculate_total_power``.
        # The previous "hypothetical 1:1 wiring" disclaimer is removed —
        # the wiring exists.
        #
        # Counterfactual residual derivation:
        #
        #   residual_live[t] = actual[t] − expected_live[t]   (logged)
        #   residual_alt[t]  = residual_live[t] + Δrelease[t]
        #
        # where Δrelease[t] = (B_kα[t] − B_k0[t]) × (1 − decay).  Both
        # replays start from B=0 over the analysis window; the EMA's
        # initial-condition term is identical between replays and
        # cancels in the difference.  Coefficients are held at their
        # currently-learned values (frozen-coefficient mode); real
        # adoption of k > 0 will see ~2-6 % NLMS coefficient drift over
        # 2-3 weeks of qualifying hours, which this replay does not
        # model.  Use ``empirical_optimum_k`` as a calibration hint;
        # validate against actual prediction RMSE after enabling
        # k > 0 for 2-4 sunny weeks.
        battery_feedback_sweep: dict = {}
        if sweep_tuples:
            decay_for_sweep = self.coordinator.solar_battery_decay
            k_candidates = [round(0.1 * i, 1) for i in range(11)]  # 0.0..1.0

            # Replay battery for each k.  trajectories[k] is a list of
            # battery states aligned 1:1 with sweep_tuples.
            trajectories: dict[float, list[float]] = {}
            for k_cand in k_candidates:
                B = 0.0
                trace: list[float] = []
                for (impact_raw, wasted, _act, _exp, heating_active,
                     _hb, _tb, _sb) in sweep_tuples:
                    feedback = (k_cand * wasted) if (k_cand > 0.0 and heating_active) else 0.0
                    B = B * decay_for_sweep + (impact_raw + feedback) * (1 - decay_for_sweep)
                    trace.append(B)
                trajectories[k_cand] = trace

            # Counterfactual anchor MUST be the live k, not k = 0 (#1066
            # review).  ``residual_live`` below is ``actual - expected``
            # where ``expected`` is the *logged* prediction — already net
            # of whatever carryover release the live k produced.  Anchoring
            # the delta at k = 0 therefore offsets every candidate by
            # ``release_live - release_0``, so on an install running k > 0
            # the candidate equal to the live setting is scored as a
            # change and the sweep can "discover" the value already in
            # use.  Anchoring at k_live makes ``residual_alt(k_live)``
            # reduce to the logged residual exactly, which is the property
            # the decay sweep's ``_replay_score`` already had.
            #
            # ``k_live`` may sit off the candidate grid (it is a free
            # config value), so its trajectory is replayed here rather
            # than looked up.  Bit-identical to the old behaviour whenever
            # k_live == 0.0, which is the default and the case for every
            # install that has not opted in.
            k_live_sweep = _coerce_scalar(
                self.coordinator.battery_thermal_feedback_k, 0.0
            )
            _live_trace: list[float] = []
            _B = 0.0
            for (impact_raw, wasted, _act, _exp, heating_active,
                 _hb, _tb, _sb) in sweep_tuples:
                feedback = (
                    (k_live_sweep * wasted)
                    if (k_live_sweep > 0.0 and heating_active) else 0.0
                )
                _B = _B * decay_for_sweep + (impact_raw + feedback) * (1 - decay_for_sweep)
                _live_trace.append(_B)
            baseline_trace = _live_trace

            # Per-cell residuals.  Cells dropped when they lack a temp
            # bucket (transition zone) — kept in global aggregate via the
            # ``global`` cell key so the user still sees an overall RMSE.
            per_k_results: dict[str, dict] = {}
            residuals_by_k: dict[str, list[float]] = {}
            for k_cand in k_candidates:
                k_trace = trajectories[k_cand]
                cell_residuals: dict[tuple, list[float]] = {}
                global_residuals: list[float] = []
                for idx, (
                    _impact_raw, _wasted, actual, expected,
                    _heating, hour_bucket, temp_bucket, screen_bucket,
                ) in enumerate(sweep_tuples):
                    # Convert state-trajectory delta to release-delta:
                    # release[t] = state[t] × (1 − decay) is what
                    # ``calculate_total_power`` subtracts from heating
                    # demand under the live wiring (split-state, post-#896
                    # follow-up).  Multiplying the state delta by
                    # ``(1 − decay)`` projects the sweep from "what state
                    # would be" to "what release would be subtracted from
                    # prediction" — which matches the live model's
                    # observable effect on prediction error.
                    delta_release = (k_trace[idx] - baseline_trace[idx]) * (1 - decay_for_sweep)
                    residual_live = actual - expected
                    residual_alt = residual_live + delta_release
                    global_residuals.append(residual_alt)
                    if temp_bucket is None:
                        continue
                    cell_key = (hour_bucket, temp_bucket, screen_bucket)
                    cell_residuals.setdefault(cell_key, []).append(residual_alt)

                cells = {}
                for (hb, tb, sb), residuals in cell_residuals.items():
                    n = len(residuals)
                    sse = sum(r * r for r in residuals)
                    rmse = (sse / n) ** 0.5 if n > 0 else 0.0
                    mean = sum(residuals) / n if n > 0 else 0.0
                    cells[f"{hb}__{tb}__{sb}"] = {
                        "n": n,
                        "rmse_kwh": round(rmse, 4),
                        "mean_residual_kwh": round(mean, 4),
                    }
                global_n = len(global_residuals)
                global_sse = sum(r * r for r in global_residuals)
                global_rmse = (global_sse / global_n) ** 0.5 if global_n > 0 else 0.0
                per_k_results[str(k_cand)] = {
                    "global": {
                        "n": global_n,
                        "rmse_kwh": round(global_rmse, 4),
                    },
                    "cells": cells,
                }
                # Kept out of per_k_results so the service payload does
                # not grow by one float per hour per candidate.  Index
                # order is ``sweep_tuples`` order for every candidate —
                # ``global_residuals.append`` runs before the
                # ``temp_bucket is None`` continue — so the lists are
                # index-aligned across candidates and the pairing against
                # the live-k baseline is exact (#1066).
                residuals_by_k[str(k_cand)] = global_residuals

            # Baseline residuals = the live k's own replay.  With
            # ``baseline_trace`` anchored at k_live the release delta is
            # identically zero for the live setting, so these reduce to the
            # logged residuals — which is exactly the property that makes
            # the anchor correct.  Built in ``sweep_tuples`` order so they
            # are index-aligned with every ``residuals_by_k`` entry.
            live_k_residuals: list[float] = [
                actual - expected
                for (_ir, _w, actual, expected, _ha, _hb, _tb, _sb) in sweep_tuples
            ]
            live_k_sse = sum(r * r for r in live_k_residuals)
            live_k_rmse = (
                (live_k_sse / len(live_k_residuals)) ** 0.5
                if live_k_residuals else 0.0
            )

            # Empirical optimum: lowest global RMSE.  Tie-break in favour
            # of smaller k (more conservative — closer to the disabled
            # default).  Reported as a recommendation, not auto-applied.
            # Seeded from the LIVE k, not from 0.0 (#1066 review).  On an
            # install running k > 0 the old seed meant the sweep started
            # from a value the install is not using, so "no candidate
            # beats the baseline" could not be expressed and the argmin
            # always reported a change.
            best_k = k_live_sweep
            # Rounded to match what the candidates are compared as.  The
            # candidate RMSEs come out of ``per_k_results`` already
            # rounded to 4 dp, so seeding with the raw value compares a
            # full-precision baseline against quantised candidates: a
            # candidate that ties the baseline exactly can still round
            # DOWN by up to 5e-5 and clear the 1e-6 argmin margin, which
            # is 50× the epsilon.  On a sweep where every candidate ties
            # — an install with no wasted solar in the window, so k
            # changes nothing — that reported a spurious change away from
            # the live setting.  Compare like with like.
            best_global_rmse = round(live_k_rmse, 4)
            for k_cand in k_candidates:
                cand_rmse = per_k_results[str(k_cand)]["global"]["rmse_kwh"]
                if cand_rmse < best_global_rmse - 1e-6:
                    best_global_rmse = cand_rmse
                    best_k = k_cand

            # Per-cell delta-RMSE table relative to k=0.  Lets the user
            # see which (hour × temp × screen) combinations actually
            # benefit at the recommended k, vs which ones the global
            # optimum is averaging over.  Cells with n < 5 are emitted
            # but flagged so the reader does not over-interpret thin
            # data — particularly relevant at 10-day windows where many
            # cells carry only 1-3 hours.
            #
            # Sweep collapse (#896 follow-up).  When the optimum IS the
            # live setting every per_cell_at_optimum row would be
            # identical to the baseline (delta_rmse = 0 everywhere), and
            # every non-baseline ``sweep[k]["cells"]`` table is
            # informational fluff with no actionable signal.  Emit only
            # the baseline cells and a single ``per_k_global_rmse``
            # summary for the other candidates.  Otherwise the full detail
            # is preserved on baseline + optimum k; intermediate k values
            # still drop ``cells`` because they are not the recommended
            # target.
            #
            # Keyed on ``best_k == k_live_sweep``, not ``== 0.0``: with the
            # baseline anchored at the live k, "no change recommended" is
            # the live value, and on a k > 0 install the old test collapsed
            # the wrong branch.  ``baseline_key`` follows for the same
            # reason — the per-cell baseline column must be the anchor the
            # headline numbers are quoted against, or the two disagree
            # inside one payload.
            #
            # ``k_live_sweep`` may sit off the candidate grid (it is a free
            # config value), in which case there is no ``per_k_results``
            # row for it.  Both lookups below are guarded; an unguarded
            # ``per_k_results[str(best_k)]`` here previously raised
            # KeyError and killed the whole diagnose_solar service.
            baseline_key = (
                str(k_live_sweep) if str(k_live_sweep) in per_k_results else None
            )
            optimum_key = (
                str(best_k) if str(best_k) in per_k_results else None
            )
            if best_k == k_live_sweep or optimum_key is None or baseline_key is None:
                per_cell_at_optimum = None
                for k_str, k_data in per_k_results.items():
                    if k_str != baseline_key:
                        k_data.pop("cells", None)
            else:
                per_cell_at_optimum = {}
                baseline_cells = per_k_results[baseline_key]["cells"]
                optimum_cells = per_k_results[optimum_key]["cells"]
                all_cell_keys = set(baseline_cells) | set(optimum_cells)
                for cell_key in sorted(all_cell_keys):
                    base_cell = baseline_cells.get(cell_key, {"n": 0, "rmse_kwh": 0.0})
                    opt_cell = optimum_cells.get(cell_key, {"n": 0, "rmse_kwh": 0.0})
                    delta_rmse = opt_cell["rmse_kwh"] - base_cell["rmse_kwh"]
                    per_cell_at_optimum[cell_key] = {
                        "n": base_cell["n"],
                        "baseline_rmse_kwh": base_cell["rmse_kwh"],
                        "optimum_rmse_kwh": opt_cell["rmse_kwh"],
                        "delta_rmse_kwh": round(delta_rmse, 4),
                        "thin_sample": base_cell["n"] < 5,
                    }
                # Drop cells from intermediate k values; only baseline
                # and optimum are actionable for the user.  Keyed on the
                # same two variables the branch above reads, NOT on the
                # ``"0.0"`` literal: with the baseline anchored at the
                # live k, the literal stripped the row that
                # ``per_cell_at_optimum``'s baseline column and
                # ``global_rmse_at_baseline_kwh`` are both quoted from,
                # while retaining a k=0 row that is neither baseline nor
                # optimum.  Same payload, two disagreeing anchors — the
                # exact condition ``baseline_key`` exists to prevent.
                for k_str, k_data in per_k_results.items():
                    if k_str != baseline_key and k_str != optimum_key:
                        k_data.pop("cells", None)

            battery_feedback_sweep = {
                # The coerced value, not the raw attribute.  One payload
                # must not report three different answers for one setting
                # — ``baseline_k`` and ``calibration.current_k`` are both
                # coerced, and this field feeds ``summary`` directly.
                "current_k": k_live_sweep,
                "decay_used": decay_for_sweep,
                "n_hours_in_window": len(sweep_tuples),
                "n_hours_with_heating_active": sum(
                    1 for t in sweep_tuples if t[4]
                ),
                "n_hours_with_heating_wasted": sum(
                    1 for t in sweep_tuples if t[1] > 0.0 and t[4]
                ),
                "k_candidates": k_candidates,
                "sweep": per_k_results,
                "empirical_optimum_k": best_k,
                # Recommendation gating (#1066).  ``empirical_optimum_k``
                # stays the raw argmin; these say whether it is advice.
                #
                # A monotone sweep whose minimum sits on the last
                # candidate has not bracketed an optimum — it has run out
                # of grid.  Observed on a real install: 11 candidates
                # decreasing from 0.1959 to 0.1931, minimum at k=1.0, and
                # the boundary value was reported as ``empirical_optimum_k``
                # as though the sweep had located it.  Compared against
                # ``k_live_sweep`` rather than 0.0 so an install already
                # running k > 0 is not told its own value is a finding.
                "optimum_at_sweep_boundary": bool(
                    best_k != k_live_sweep
                    and len(k_candidates) > 1
                    and best_k == k_candidates[-1]
                ),
                # Baseline is the LIVE k, not the k=0 grid corner.
                "recommendation_significance": paired_loss_improvement(
                    live_k_residuals,
                    residuals_by_k.get(str(best_k), live_k_residuals),
                    n_candidates_considered=max(1, len(k_candidates)),
                ),
                "baseline_k": k_live_sweep,
                # Named a screen, not a significance test: the candidate
                # is the argmin over the grid and is tested on the
                # residuals that selected it, so the t statistic is not a
                # p-value.  See const.BATTERY_RECOMMENDATION_MIN_T.
                "recommendation_screen": (
                    "paired_diff_of_squared_residuals; base_t="
                    f"{BATTERY_RECOMMENDATION_MIN_T}"
                    " + sqrt(2 ln m) selection penalty"
                    " + lag1 serial-correlation SE inflation;"
                    " NOT a calibrated p-value"
                ),
                "global_rmse_at_optimum_kwh": round(best_global_rmse, 4),
                # Baseline is the live k's replay, so on a k = 0 install
                # this is bit-identical to the old ``per_k_results["0.0"]``
                # reading, and on a k > 0 install it is the value that was
                # previously wrong (#1066 review).
                "global_rmse_at_baseline_kwh": round(live_k_rmse, 4),
                # ``+ 0.0`` normalises ``-0.0``: the baseline is rounded
                # to 4 dp and the optimum is not, so a no-change sweep
                # leaves a residual bounded by 5e-5 that can round to
                # negative zero and serialise as ``-0.0`` in the service
                # response.  Cosmetic, but "-0.0 improvement" reads as a
                # regression.
                "rmse_improvement_kwh": round(live_k_rmse - best_global_rmse, 4) + 0.0,
                "per_cell_at_optimum": per_cell_at_optimum,
                # Methodology — read this before interpreting numbers.
                "method": "carryover_release_replay",
                "interpretation": "calibration_hint",
                "notes": (
                    "This sweep models the live wiring (split-state "
                    "implementation): _solar_carryover_state is charged "
                    "by k × wasted, and its release × (1 - decay) "
                    "subtracts from heating-mode demand prediction. "
                    "Each k candidate's hypothetical RMSE is computed "
                    "by replaying the carryover EMA over the window and "
                    "adding the release-delta vs the LIVE k baseline "
                    "(reported as baseline_k) to the logged residual — "
                    "so the candidate equal to the live setting scores "
                    "exactly the logged residual, and no change is "
                    "reported when none is available.  Coefficients are "
                    "held at their "
                    "currently-learned values (frozen-coefficient mode) "
                    "— real adoption of k > 0 will trigger ~2-6 % NLMS "
                    "coefficient drift over 2-3 weeks, which this "
                    "replay does not model.  Use empirical_optimum_k "
                    "as a calibration hint, then validate against "
                    "actual prediction RMSE after running with k > 0 "
                    "for 2-4 sunny weeks.  Transition-zone hours "
                    "(BP±2 °C) are included in both `global` aggregates "
                    "and the per-cell table under "
                    "temp_bucket=\"transition\"; expect that cell to "
                    "carry the strongest signal for the headline "
                    "symptom on high-BP installs."
                ),
            }

        # Screen impact
        screen_impact = {}
        if screen_closed_errors and screen_open_errors:
            mean_closed = sum(screen_closed_errors) / len(screen_closed_errors)
            mean_open = sum(screen_open_errors) / len(screen_open_errors)
            screen_impact = {
                "mean_error_screens_closed": round(mean_closed, 4),
                "mean_error_screens_open": round(mean_open, 4),
                "qualifying_hours_closed": len(screen_closed_errors),
                "qualifying_hours_open": len(screen_open_errors),
            }
            if abs(mean_closed - mean_open) > 0.1:
                global_flags.append("screen_drift_detected")

        # Temporal bias (global)
        all_morning = [acc["morning_delta"] / acc["morning_n"] for acc in unit_accum.values() if acc["morning_n"] > 5]
        all_afternoon = [acc["afternoon_delta"] / acc["afternoon_n"] for acc in unit_accum.values() if acc["afternoon_n"] > 5]

        # Hour-of-day curve
        hour_curve = {}
        for h in range(6, 19):
            vals = hourly_residuals[h]
            if vals:
                hour_curve[str(h)] = round(sum(vals) / len(vals), 4)

        # Context block (#826 validation).  Surfaces the values currently
        # applied so a remote analyser can tie a diagnose payload to the
        # installation's geography and configuration without needing separate
        # entity inspection.
        from .const import (
            DEFAULT_SOLAR_MIN_TRANSMITTANCE as _DEFAULT_FLOOR,
            SCREEN_DIRECT_TRANSMITTANCE as _SCREEN_DIRECT,
        )
        try:
            lat = self.coordinator.hass.config.latitude
            lon = self.coordinator.hass.config.longitude
        except AttributeError:
            lat = lon = None
        context = {
            "latitude": lat,
            "longitude": lon,
            "screen_config": {
                "south": bool(self.coordinator.screen_config[0]),
                "east":  bool(self.coordinator.screen_config[1]),
                "west":  bool(self.coordinator.screen_config[2]),
            },
            "constants": {
                "screen_direct_transmittance": _SCREEN_DIRECT,
                "composite_legacy_floor": _DEFAULT_FLOOR,
                "solar_battery_decay": self.coordinator.solar_battery_decay,
                "solar_azimuth": self.coordinator.solar_azimuth,
            },
            "days_analyzed": days_back,
        }

        # Per-unit min-base thresholds (#871).  Exposes the effective gate
        # each unit sees in NLMS / inequality / shutdown detection so the
        # user can distinguish auto-calibrated values from the global
        # fallback.  ``method`` reports the source; ``effective`` is the
        # value actually applied at the gate sites.
        from .const import (
            SOLAR_LEARNING_MIN_BASE as _GLOBAL_LEARNING_MIN_BASE,
            SOLAR_SHUTDOWN_MIN_BASE as _GLOBAL_SHUTDOWN_MIN_BASE,
        )
        per_unit_thresholds = {}
        for sid in self.coordinator.energy_sensors:
            calibrated = self.coordinator._per_unit_min_base_thresholds.get(sid)
            per_unit_thresholds[sid] = {
                "effective_nlms": round(
                    calibrated if calibrated is not None else _GLOBAL_LEARNING_MIN_BASE,
                    5,
                ),
                "effective_shutdown": round(
                    calibrated if calibrated is not None else _GLOBAL_SHUTDOWN_MIN_BASE,
                    5,
                ),
                "method": "auto" if calibrated is not None else "fallback",
                "calibrated_value": calibrated,
            }

        # Top-level summary digest (#896 follow-up).  Human-readable
        # at-a-glance overview that lets the user decide whether to read
        # the verbose blocks below.  Computed strictly from already-built
        # blocks — no new arithmetic, just pivots and counts.  Verdict
        # logic is conservative: ``no_action_needed`` only when EVERY
        # signal source agrees nothing is actionable; otherwise
        # ``review_recommended`` and the user reads ``units_with_flags``,
        # ``global_flags``, and the battery sub-blocks for specifics.
        # Excluded entities (#962) are not counted toward either active or
        # inactive — they have no solar role at all, so reporting them as
        # "inactive" would conflate "no solar signal yet" with "deliberately
        # excluded from solar".
        active_count = sum(
            1 for u in per_unit.values()
            if not u.get("inactive", False) and not u.get("excluded_from_solar", False)
        )
        inactive_count = sum(
            1 for u in per_unit.values()
            if u.get("inactive", False) and not u.get("excluded_from_solar", False)
        )
        units_with_flags = [
            {"entity_id": eid, "flags": u["flags"]}
            for eid, u in per_unit.items()
            if u.get("flags")
        ]
        # Battery feedback verdict.
        if battery_feedback_sweep:
            opt_k = battery_feedback_sweep.get("empirical_optimum_k", 0.0)
            improvement = battery_feedback_sweep.get("rmse_improvement_kwh", 0.0)
            bf_sig = battery_feedback_sweep.get("recommendation_significance") or {}
            bf_boundary = battery_feedback_sweep.get(
                "optimum_at_sweep_boundary", False
            )
            # ``improvement`` used to be bound here, reported, and never
            # read by the verdict (#1066).  It now gates, via the paired
            # significance test rather than its own magnitude.
            bf_verdict = battery_feedback_verdict(
                opt_k,
                bf_boundary,
                bf_sig.get("significant", False),
                current_k=battery_feedback_sweep.get(
                    "baseline_k", battery_feedback_sweep.get("current_k", 0.0)
                ),
            )
            battery_feedback_summary = {
                "current_k": battery_feedback_sweep.get("current_k", 0.0),
                "optimum_k": opt_k,
                "rmse_improvement_kwh": improvement,
                "t_statistic": bf_sig.get("t_statistic"),
                "optimum_at_sweep_boundary": bf_boundary,
                "verdict": bf_verdict,
            }
        else:
            battery_feedback_summary = {
                # Coerced for the same reason the sweep's own
                # ``current_k`` is: one attribute, one reported answer.
                "current_k": _coerce_scalar(
                    self.coordinator.battery_thermal_feedback_k, 0.0
                ),
                "verdict": "no_data",
            }
        # Battery decay verdict pivots on assessment from battery_health
        # and the calibration block; "ok" means no recommendation pending.
        if calibration:
            # ``recommended_decay != current_decay`` is necessary but not
            # sufficient (#1066).  ``recommendation_withheld_reason``
            # carries the three suppressing conditions in priority order,
            # computed where the sweep ran; the verdict is a lookup.
            decay_verdict = battery_decay_verdict(
                calibration.get("current_decay"),
                calibration.get("recommended_decay"),
                calibration.get("recommendation_withheld_reason"),
                sweep_produced_evidence=calibration.get(
                    "sweep_produced_evidence", True
                ),
                bias_assessment=battery_health.get("assessment"),
            )
            battery_decay_summary = {
                "current_decay": calibration.get("current_decay"),
                "recommended_decay": calibration.get("recommended_decay"),
                "t_statistic": (
                    calibration.get("recommendation_significance", {})
                    .get("t_statistic")
                ),
                # Both of these are load-bearing in the summary, not
                # detail-block duplication.  ``recommended_decay`` above
                # is the raw argmin and is populated even when the sweep
                # declined to advise it — without the reason beside it a
                # reader takes the number as endorsed and sets it by
                # hand, which is exactly what the withholding chain
                # exists to prevent.  ``bias_assessment`` is the second
                # reading of the same residual; it raises ``any_action``
                # on its own, so the summary must say what raised it.
                "recommendation_withheld_reason": calibration.get(
                    "recommendation_withheld_reason"
                ),
                "bias_assessment": battery_health.get("assessment"),
                "verdict": decay_verdict,
            }
        elif battery_health:
            # No calibration block at all, so the bias reading IS the
            # verdict here.  ``bias_assessment`` is carried alongside it
            # anyway, so the ``battery_bias_raises`` operand reads one
            # key regardless of which branch built this dict.
            battery_decay_summary = {
                "current_decay": battery_health.get("decay_rate"),
                "bias_assessment": battery_health.get("assessment"),
                "verdict": battery_health.get("assessment", "no_data"),
            }
        else:
            battery_decay_summary = {
                "current_decay": self.coordinator.solar_battery_decay,
                "verdict": "no_data",
            }
        # Computed before the summary so its verdict can raise the
        # top-level one — ``enabled_but_unsupported`` is a live
        # misconfiguration (4D routed on an install whose input cannot
        # feed it, silently degrading predictions), and a summary that
        # reports ``no_action_needed`` over it defeats the point of
        # having the gate.
        source_mix = self._compute_dni_dhi_source_mix(days_back)
        readiness_4d = self._compute_4d_readiness(days_back, source_mix=source_mix)
        # Top-level verdict: only ``no_action_needed`` when every signal
        # source is clean.  Otherwise the user should look at one of the
        # detail blocks the summary points at.
        #
        # Three 4D conditions raise the verdict; the history matters
        # because one of them reverses an earlier decision (#1062
        # reversing #1061), and without the reason written down the
        # reversal reads as a regression:
        #
        # * ``enabled_but_unsupported`` (input half) — a live
        #   misconfiguration: 4D routed on an install whose input cannot
        #   feed it, silently degrading predictions.  A summary reporting
        #   ``no_action_needed`` over it defeats the gate.
        # * ``enabled_but_not_ready`` (both halves) — strictly wider, and
        #   catches the case the input half cannot see: flag on, provider
        #   fine, but 4D untrained, so the read path returns the
        #   zero-vector and prediction runs with **no solar at all**.
        #   Because the combiner is three-valued, a definitive
        #   ``supports_4d = False`` always reaches this verdict, so it
        #   now *subsumes* the input-half condition above rather than
        #   merely overlapping it.  The narrower condition is kept
        #   deliberately as defence-in-depth: the two are computed from
        #   different predicates, and a future change to either shape
        #   should not be able to silence a live misconfiguration on its
        #   own.  (It was originally retained on the different grounds
        #   that an undeterminable learning half would collapse the
        #   composite to ``insufficient_data`` and swallow the input
        #   misconfiguration — true before the combiner was made
        #   three-valued, no longer the reason it is here.)
        # * ``ready_to_enable`` — #1061 deliberately did NOT fire on the
        #   old ``disabled_but_supported``, on the grounds that it merely
        #   meant "your weather provider is good" and would pin every
        #   well-sourced install at ``review_recommended`` forever,
        #   nagging users toward a flag they had not opted into.  That
        #   reasoning is weakened once the learning half is added: the
        #   signal no longer means "your provider is fine", it means
        #   "your input is good *and* your 4D model is trained — you are
        #   ready", which is actionable in a way the input half alone was
        #   not.  It is NOT, however, rare: ``learned`` is stamped on the
        #   first coefficient write and the 4D cold-start buffer is 5
        #   samples per (entity, regime), so a well-sourced install
        #   satisfies condition 2 within about a week of sunny weather
        #   and never un-satisfies it.  ``ready_to_enable`` is therefore
        #   an *absorbing* state whose only exit is enabling the flag —
        #   #1061's objection is delayed, not removed, and an install
        #   that deliberately declines 4D sits at ``review_recommended``
        #   indefinitely with no dismiss mechanism.  Accepted (the action
        #   is a single setting) rather than overlooked; a suppression
        #   path is the lever if the nag proves unwelcome.
        # The post-sunset residual is read twice — as a decay
        # recommendation and as a bias flag — and BOTH reach
        # ``any_action``, each through its own operand (#1066).
        #
        # The ``battery_decay_*`` entries are filtered out of the generic
        # ``bool(global_flags)`` term and re-enter as
        # ``battery_bias_raises`` below.  That is ROUTING, not
        # suppression: it puts the bias behind a named operand that says
        # what it is, instead of inside an anonymous flag-list truth test.
        # Do not "simplify" it back by dropping the operand.  An earlier
        # revision did exactly that, justified as stopping one signal from
        # raising the verdict "twice" — which a boolean OR cannot do, two
        # true operands raise it exactly as much as one — and the effect
        # was that a well-sampled bias vanished from the summary whenever
        # the sweep merely ran.
        #
        # The bias is a separate operand rather than folded into
        # ``battery_decay_summary["verdict"]`` because the verdict has to
        # stay free to report ``windows_disagree`` and the other
        # withholding reasons.  Those sit next to a populated
        # ``recommended_decay``, and replacing them with the bias leaves
        # a raw argmin in the payload looking endorsed.
        non_battery_flags = [
            f for f in global_flags if not f.startswith("battery_decay_")
        ]
        battery_bias_raises = battery_decay_summary.get("bias_assessment") in (
            "too_fast", "too_slow",
        )
        any_action = (
            bool(non_battery_flags)
            or bool(units_with_flags)
            or battery_feedback_summary["verdict"].startswith("consider_")
            or battery_decay_summary["verdict"].startswith("consider_")
            # Two operands, one residual: the sweep's recommendation and
            # the bias reading.  The second covers the case the first
            # cannot — the sweep ran, found no better decay, and the
            # residual is biased anyway.
            or battery_decay_summary["verdict"] in ("too_fast", "too_slow")
            or battery_bias_raises
            or source_mix.get("verdict") == "enabled_but_unsupported"
            or readiness_4d.get("verdict") in ("ready_to_enable", "enabled_but_not_ready")
        )
        summary = {
            "verdict": "review_recommended" if any_action else "no_action_needed",
            "global_flags": global_flags,
            "active_solar_units": active_count,
            "inactive_units": inactive_count,
            "units_with_flags": units_with_flags,
            "battery_feedback": battery_feedback_summary,
            "battery_decay": battery_decay_summary,
            "dni_dhi_source": {
                "dominant_source": source_mix.get("dominant_source"),
                "experimental_4d_primary_enabled": source_mix.get(
                    "experimental_4d_primary_enabled"
                ),
                "supports_4d_primary": source_mix.get("supports_4d_primary"),
                "verdict": source_mix.get("verdict", "unavailable"),
            },
            # Both halves of the gate: input support AND 4D having
            # actually learned.  ``dni_dhi_source`` above reports the
            # input half alone and is not sufficient on its own.
            "four_d_readiness": {
                "ready": readiness_4d.get("ready"),
                "verdict": readiness_4d.get("verdict"),
                "learning_ready": readiness_4d.get("learning", {}).get("ready"),
                "entities_not_learned": readiness_4d.get("learning", {}).get(
                    "entities_not_learned", []
                ),
            },
        }

        return {
            "summary": summary,
            "context": context,
            "global": {
                "qualifying_hours": total_qualifying,
                "excluded": excluded,
                "battery_decay_health": battery_health,
                "battery_calibration": calibration,
                "battery_feedback_sweep": battery_feedback_sweep,
                "screen_impact": screen_impact,
                "temporal_bias": {
                    "morning_mean_delta": round(sum(all_morning) / len(all_morning), 4) if all_morning else None,
                    "afternoon_mean_delta": round(sum(all_afternoon) / len(all_afternoon), 4) if all_afternoon else None,
                },
                "hour_of_day_residual": hour_curve,
                "flags": global_flags,
                # Inequality-replay diagnostics (#865).  Aggregate counters
                # from the shadow replay used to populate per-unit
                # ``implied_coefficient_inequality`` above.  ``inequality_updates``
                # shows how many (unit, hour) samples passed through the
                # one-sided constraint update; ``inequality_non_binding`` shows
                # how many samples found the constraint already satisfied.
                "inequality_replay": {
                    "updates": shadow_diag.get("inequality_updates", 0),
                    "non_binding": shadow_diag.get("inequality_non_binding", 0),
                    "skipped_low_battery": shadow_diag.get("inequality_skipped_low_battery", 0),
                    "skipped_mode": shadow_diag.get("inequality_skipped_mode", 0),
                    "skipped_base": shadow_diag.get("inequality_skipped_base", 0),
                },
            },
            "per_unit": per_unit,
            "per_unit_thresholds": per_unit_thresholds,
            "dni_dhi_source_mix": source_mix,
            "four_d_readiness": readiness_4d,
            "dni_dhi_shadow": self._compute_dni_dhi_shadow_report(days_back),
            "base_model_4d_shadow": self._compute_base_model_4d_shadow_report(days_back),
            "total_power_4d_divergence": self._compute_total_power_4d_divergence_report(days_back),
            "shoulder_saturation_blast_radius": self._compute_shoulder_saturation_blast_radius(days_back),
            "ghi_signal_agreement": self._compute_ghi_signal_agreement(days_back),
        }

    def _compute_dni_dhi_source_mix(self, days_back: int) -> dict:
        """Which ladder branch this install actually resolves through.

        The per-install gate for ``experimental_4d_primary``.  4D is the
        superior pipeline wherever a real DNI/DHI source exists and is
        *inferior* on the ``kasten_synthetic`` branch, where its fourth
        degree of freedom re-encodes ``cloud_coverage`` and the Erbs
        split trades a ~1 % constant bias for a ``kT``-dependent one (see
        CLAUDE.md > Solar Model > 4D shadow learner).  Since supplying
        DNI/DHI is never a requirement, that branch is permanent — so
        the only question a per-install gate has to answer is which
        branch *this* install is on.

        Counts the ``dni_dhi_source`` label written on every
        ``hourly_log`` entry, restricted to **daylight** hours.  The
        daylight filter is load-bearing, not hygiene: the label is
        derived from collector sample counts and never emits
        ``"no_sun"`` (see :func:`solar.derive_dni_dhi_source_label`), so
        an unfiltered count reports every night hour as
        ``kasten_synthetic`` on any install with cloud-coverage data and
        would make a perfectly-sourced install look mixed.
        ``solar_factor > 0`` is the daylight test — it is zero below the
        horizon by construction and is logged on every entry.

        Entries predating the ``dni_dhi_source`` field are counted as
        ``unlabelled_hours`` rather than silently dropped, so a thin mix
        on a freshly-upgraded install is visibly thin rather than
        looking like a verdict.

        Strict diagnostic — reads ``hourly_log`` only, writes nothing.
        """
        cutoff = (dt_util.now() - timedelta(days=days_back)).date().isoformat()

        counts: dict[str, int] = {}
        unlabelled = 0
        n_night_skipped = 0

        for entry in self.coordinator._hourly_log:
            ts = entry.get("timestamp", "")
            if ts[:10] < cutoff:
                continue
            try:
                if float(entry.get("solar_factor") or 0.0) <= 0.0:
                    n_night_skipped += 1
                    continue
            except (TypeError, ValueError):
                n_night_skipped += 1
                continue
            source = entry.get("dni_dhi_source")
            if not source:
                unlabelled += 1
                continue
            counts[source] = counts.get(source, 0) + 1

        total = sum(counts.values())
        if total == 0:
            return {
                "available": False,
                "reason": "no_labelled_daylight_hours",
                "days_back": days_back,
                "daylight_hours_total": 0,
                "unlabelled_hours": unlabelled,
                "night_hours_excluded": n_night_skipped,
            }

        by_source = {
            src: {"hours": n, "pct": round(100.0 * n / total, 1)}
            for src, n in sorted(counts.items(), key=lambda kv: -kv[1])
        }
        real_hours = counts.get("native", 0) + counts.get("erbs_from_ghi", 0)
        real_share = real_hours / total
        dominant_source = max(counts.items(), key=lambda kv: kv[1])[0]

        flag_on = bool(
            getattr(self.coordinator, "experimental_4d_primary", False)
        )

        if total < DNI_DHI_SOURCE_MIX_MIN_HOURS:
            supports_4d = None
            verdict = "insufficient_data"
        else:
            supports_4d = real_share >= DNI_DHI_REAL_SOURCE_DOMINANCE_MIN
            if supports_4d and flag_on:
                verdict = "enabled_and_supported"
            elif supports_4d and not flag_on:
                verdict = "disabled_but_supported"
            elif not supports_4d and flag_on:
                verdict = "enabled_but_unsupported"
            else:
                verdict = "disabled_and_unsupported"

        return {
            "available": True,
            "days_back": days_back,
            "daylight_hours_total": total,
            "unlabelled_hours": unlabelled,
            "night_hours_excluded": n_night_skipped,
            "by_source": by_source,
            "dominant_source": dominant_source,
            "real_source_share": round(real_share, 4),
            "experimental_4d_primary_enabled": flag_on,
            "supports_4d_primary": supports_4d,
            "verdict": verdict,
            "verdict_thresholds": {
                "real_source_dominance_min": DNI_DHI_REAL_SOURCE_DOMINANCE_MIN,
                "min_labelled_daylight_hours": DNI_DHI_SOURCE_MIX_MIN_HOURS,
                "real_sources": ["native", "erbs_from_ghi"],
            },
        }

    def _compute_dni_dhi_outage(self) -> dict:
        """Has the irradiance input gone away while 4D is live? (#1070)

        An **alert**, not a routing decision, and deliberately not built
        on :meth:`_compute_dni_dhi_source_mix`.  That gate is slow on
        purpose — 30 days, 80 % dominance, 50 hours minimum — because
        routing an install between pipelines is a considered choice.  The
        question here is the opposite: did this provider stop publishing
        irradiance in the last day or two?  A 30-day window would take
        weeks to notice.  Two questions, two windows.

        Walks ``_hourly_log`` backwards, counting **daylight** hours only
        (``solar_factor > 0``) until the window is full.  The daylight
        filter is load-bearing for the same reason it is in the source
        mix: :func:`solar.derive_dni_dhi_source_label` never emits
        ``"no_sun"``, so night hours carry a ``kasten_synthetic`` label on
        any install with cloud data, and a wall-clock window would fire
        every night on a healthy install.

        Verdicts:
            ``raise``  — real-source share below the raise bar; the
                provider has effectively stopped supplying irradiance
                while 4D is routed live.
            ``clear``  — share at or above the clear bar; recovered.
            ``hold``   — share between the two bars.  Sticky by design:
                asymmetric thresholds are what stop an intermittent
                provider from creating and deleting the repair on
                alternating days.
            ``not_applicable`` — 4D not routed live, or solar disabled.
                Nothing to warn about; any existing issue is cleared.
            ``insufficient_data`` — window not yet full.  Distinct from
                ``clear``: no evidence is not evidence of health, and a
                fresh install must not raise a repair on day one.

        Strict read — walks ``hourly_log`` and writes nothing.  The
        create/delete side-effect lives in ``repairs.py``.
        """
        flag_on = (
            getattr(self.coordinator, "experimental_4d_primary", False) is True
        )
        solar_on = bool(getattr(self.coordinator, "solar_enabled", False))
        if not flag_on or not solar_on:
            # Gated on solar_enabled as well as the flag: with solar off
            # the 4D path never runs, so a warning about its input
            # quality is noise about a pipeline that is not executing.
            return {
                "verdict": "not_applicable",
                "experimental_4d_primary_enabled": flag_on,
                "solar_enabled": solar_on,
                "daylight_hours_examined": 0,
                "real_source_hours": 0,
                "real_source_share": None,
            }

        real_hours = 0
        examined = 0
        for entry in reversed(self.coordinator._hourly_log):
            if examined >= REPAIR_DNI_DHI_OUTAGE_WINDOW_HOURS:
                break
            try:
                if float(entry.get("solar_factor") or 0.0) <= 0.0:
                    continue
            except (TypeError, ValueError):
                continue
            source = entry.get("dni_dhi_source")
            if not source:
                # Pre-#1058 entries carry no label.  Skipped rather than
                # counted either way — an unlabelled hour is not evidence
                # of an outage, and counting it as healthy would mask one.
                continue
            examined += 1
            if source in ("native", "erbs_from_ghi"):
                real_hours += 1

        result = {
            "experimental_4d_primary_enabled": flag_on,
            "solar_enabled": solar_on,
            "daylight_hours_examined": examined,
            "real_source_hours": real_hours,
            "window_hours": REPAIR_DNI_DHI_OUTAGE_WINDOW_HOURS,
            "thresholds": {
                "raise_below": REPAIR_DNI_DHI_OUTAGE_RAISE_BELOW,
                "clear_at": REPAIR_DNI_DHI_OUTAGE_CLEAR_AT,
                "min_hours": REPAIR_DNI_DHI_OUTAGE_MIN_HOURS,
                "real_sources": ["native", "erbs_from_ghi"],
            },
        }

        if examined < REPAIR_DNI_DHI_OUTAGE_MIN_HOURS:
            result["verdict"] = "insufficient_data"
            result["real_source_share"] = None
            return result

        share = real_hours / examined
        result["real_source_share"] = round(share, 4)
        if share < REPAIR_DNI_DHI_OUTAGE_RAISE_BELOW:
            result["verdict"] = "raise"
        elif share >= REPAIR_DNI_DHI_OUTAGE_CLEAR_AT:
            result["verdict"] = "clear"
        else:
            result["verdict"] = "hold"
        return result

    def _compute_4d_readiness(self, days_back: int, source_mix: dict | None = None) -> dict:
        """Both halves of the ``experimental_4d_primary`` gate (#1062).

        Readiness is two conditions, and either one alone is misleading:

        1. **Input supports it** — ``dni_dhi_source_mix.supports_4d_primary``.
           4D is superior wherever a real DNI/DHI source exists and
           inferior on the permanent ``kasten_synthetic`` branch.
        2. **4D has actually learned** —
           :func:`learning.evaluate_4d_learning_readiness`, which fires the
           live read path's own predicate.  Without this, an install with a
           perfect weather provider can be routed to 4D before its 4D
           coefficients exist and predict zero solar, silently.

        Verdicts pair the two conditions against the flag's current state:

        ``enabled_and_ready``
            Flag on, both conditions met.  Nothing to do.
        ``ready_to_enable``
            Flag off, both conditions met.  Actionable — this raises the
            summary verdict (see ``diagnose_solar``).  Note it is an
            **absorbing state**: the only exit is enabling the flag, so a
            well-sourced install that declines 4D sits at
            ``review_recommended`` indefinitely.  Accepted, not overlooked.
        ``enabled_but_not_ready``
            Flag on, a condition definitively unmet.  A **live**
            degradation, not an opportunity: prediction is running
            through an input that cannot feed it, an untrained model that
            returns the zero-vector, or both.  Raises the summary verdict.
        ``not_ready``
            Flag off, a condition definitively unmet.  The ordinary
            resting state.
        ``insufficient_data``
            Nothing definitive on either side — too few labelled daylight
            hours *and* no in-scope entity in an active solar regime.  A
            definitive ``False`` from either half outranks an unknown
            from the other; see the combiner below.

        Strict diagnostic: reads state, writes nothing.  Kept as a pure
        combiner over the two halves so the same call can serve the
        config flow (via ``coordinator.evaluate_4d_readiness``) and, later,
        an automatic router.
        """
        if source_mix is None:
            source_mix = self._compute_dni_dhi_source_mix(days_back)

        supports_4d = source_mix.get("supports_4d_primary")
        flag_on = bool(getattr(self.coordinator, "experimental_4d_primary", False))

        learning = evaluate_4d_learning_readiness(
            getattr(self.coordinator, "_solar_coefficients_4d_per_unit", None),
            getattr(self.coordinator, "_solar_affected_set", None),
            getattr(self.coordinator, "_unit_modes", None),
        )
        learned_ready = learning.get("ready")

        # Three-valued AND, not "any operand unknown → unknown".  A
        # definitive ``False`` on either half is decisive regardless of
        # the other, and collapsing it to ``insufficient_data`` would
        # hide the exact live degradation this gate exists to catch: the
        # learning half needs no history window, so it can say "4D is
        # untrained" — meaning the read path is returning the
        # zero-vector and prediction is running with no solar at all —
        # during the first 3-9 days of an install, while the input half
        # is still accumulating its 50 labelled daylight hours.  That
        # window also covers *every* upgrading install, whose retained
        # log predates the ``dni_dhi_source`` field entirely.  Only when
        # nothing definitive is known on either side is the answer
        # genuinely unknown.
        ready: bool | None
        if supports_4d is False or learned_ready is False:
            ready = False
        elif supports_4d is None or learned_ready is None:
            ready = None
        else:
            ready = True

        if ready is None:
            verdict = "insufficient_data"
        elif ready:
            verdict = "enabled_and_ready" if flag_on else "ready_to_enable"
        else:
            verdict = "enabled_but_not_ready" if flag_on else "not_ready"

        return {
            "ready": ready,
            "verdict": verdict,
            "experimental_4d_primary_enabled": flag_on,
            "input": {
                "supports_4d_primary": supports_4d,
                "dominant_source": source_mix.get("dominant_source"),
                "real_source_share": source_mix.get("real_source_share"),
                "daylight_hours_total": source_mix.get("daylight_hours_total"),
                "verdict": source_mix.get("verdict", "unavailable"),
            },
            "learning": learning,
            "accepted_limitation": (
                "Only each entity's currently active regime is checked.  A unit "
                "switching into a regime 4D has not trained yet predicts zero "
                "solar for roughly 5 qualifying sunny hours (the 4D cold-start "
                "buffer per entity and regime) before NLMS takes over.  Bounded "
                "and self-healing; gating on all regimes instead would never "
                "pass for a unit that only ever cools."
            ),
        }

    def _compute_ghi_signal_agreement(self, days_back: int) -> dict:
        """Compare Kasten-derived solar_factor against measured GHI.

        Pre-geometry scalar comparison, mirroring the
        ``signal_agreement`` block in the DNI/DHI shadow report but
        with a genuinely independent input — a local pyranometer or
        scraped weather-station GHI value, not a cloud_cover-derived
        re-encoding from a public API.

        Per-hour signals (both ∈ [0, 1.5], pre-geometry):

        - ``kasten_cloud_factor = potential_solar_factor /
          no_cloud_reference(elev, azim)`` — recovered from the same
          field used by the DNI signal_agreement block.
        - ``ghi_normalized = ghi_wm2 / (1361 × sin(elev) × 0.7^airmass)``
          — Beer-Lambert clear-sky horizontal-flux estimate.  Diffuse-
          on-clear-sky is intentionally not modelled, mirroring the
          DNI normalisation; the resulting ``ghi_normalized`` peaks
          slightly above 1.0 on clear days, which is informative as
          mean_bias rather than an error.

        High Pearson r AND low RMSE across all regimes means
        cloud_coverage × Kasten already explains what the pyranometer
        measures — measured GHI carries no new information and the
        cloud_coverage pipeline is information-bound by definition.
        Lower correlation on the broken-cloud regime is the signature
        that justifies eventually replacing the ``cloud_factor`` step
        with ``ghi_normalised`` in ``solar.calculate_solar_factor``;
        an MPC-grade fix gated on this evidence.

        Returns ``{"available": False, ...}`` when no GHI data has
        been logged yet (sensor unconfigured or unavailable for the
        whole window) or when fewer than 30 hours qualify after the
        stability gate.

        Same stability gate as ``signal_agreement``: ``elev ≥ 5°``,
        ``no_cloud_reference > 0.1``, ``potential_solar_factor >
        0.05``.  Restricts comparison to numerically stable midday
        hours where both signals carry sub-1 % relative uncertainty.

        Regime classification reads ``cloud_coverage`` from the log
        when present (clear < 30, broken 30-70, overcast ≥ 70).
        Falls back to GHI-magnitude classification when cloud_coverage
        is missing on legacy entries: < 200 W/m² overcast, > 700 W/m²
        clear, otherwise broken.
        """
        cutoff = (dt_util.now() - timedelta(days=days_back)).date().isoformat()

        pairs: list[tuple[float, float, str]] = []
        n_total_with_ghi = 0
        n_skipped_unstable = 0

        for entry in self.coordinator._hourly_log:
            ts = entry.get("timestamp", "")
            if ts[:10] < cutoff:
                continue
            ghi_val = entry.get("ghi_wm2")
            if ghi_val is None:
                continue
            n_total_with_ghi += 1

            entry_dt = dt_util.parse_datetime(ts) if ts else None
            if entry_dt is None:
                continue
            try:
                sun_pos = self.coordinator.solar.get_approx_sun_pos(
                    entry_dt + timedelta(minutes=30)
                )
                elev = float(sun_pos[0])
                azim = float(sun_pos[1])
            except (TypeError, ValueError, IndexError):
                continue
            if elev < 5.0:
                continue

            sin_elev = math.sin(math.radians(elev))
            airmass = 1.0 / max(sin_elev, 0.087)
            ghi_clear = 1361.0 * sin_elev * (0.7 ** airmass)
            if ghi_clear < 1.0:
                continue
            try:
                ghi_norm = max(0.0, min(1.5, float(ghi_val) / ghi_clear))
            except (TypeError, ValueError):
                continue

            potential_sf = entry.get("potential_solar_factor")
            if not isinstance(potential_sf, (int, float)) or potential_sf < 0:
                continue
            try:
                no_cloud_ref = self.coordinator.solar.calculate_solar_factor(
                    elev, azim, 0.0
                )
            except (TypeError, ValueError):
                continue
            if no_cloud_ref <= 0.1 or float(potential_sf) <= 0.05:
                if no_cloud_ref > 1e-6:
                    n_skipped_unstable += 1
                continue
            kasten_cf = max(0.0, min(1.5, float(potential_sf) / no_cloud_ref))

            cloud = entry.get("cloud_coverage")
            if isinstance(cloud, (int, float)):
                if cloud < 30:
                    regime = "clear"
                elif cloud < 70:
                    regime = "broken"
                else:
                    regime = "overcast"
            else:
                if ghi_val < 200.0:
                    regime = "overcast"
                elif ghi_val > 700.0:
                    regime = "clear"
                else:
                    regime = "broken"

            pairs.append((kasten_cf, ghi_norm, regime))

        if n_total_with_ghi == 0:
            return {
                "available": False,
                "reason": "no_ghi_data",
                "ghi_sensor_configured": (
                    getattr(self.coordinator, "ghi_sensor", None) is not None
                ),
            }
        if len(pairs) < 30:
            return {
                "available": False,
                "reason": "insufficient_qualifying_hours",
                "n_hours_with_ghi": n_total_with_ghi,
                "n_qualifying": len(pairs),
                "n_skipped_unstable_recovery": n_skipped_unstable,
            }

        def _stats(rows: list[tuple[float, float]]) -> dict:
            n = len(rows)
            if n < 5:
                return {
                    "n_hours": n,
                    "correlation": None,
                    "rmse": None,
                    "mean_bias_kasten_minus_ghi": None,
                }
            mx = sum(r[0] for r in rows) / n
            my = sum(r[1] for r in rows) / n
            num = sum((r[0] - mx) * (r[1] - my) for r in rows)
            dx = math.sqrt(sum((r[0] - mx) ** 2 for r in rows))
            dy = math.sqrt(sum((r[1] - my) ** 2 for r in rows))
            corr = num / (dx * dy) if dx > 1e-9 and dy > 1e-9 else None
            rmse = math.sqrt(sum((r[0] - r[1]) ** 2 for r in rows) / n)
            return {
                "n_hours": n,
                "correlation": round(corr, 4) if corr is not None else None,
                "rmse": round(rmse, 4),
                "mean_bias_kasten_minus_ghi": round(mx - my, 4),
            }

        return {
            "available": True,
            "definition": (
                "kasten_cloud_factor = potential_solar_factor / "
                "no_cloud_reference(elev, azim); "
                "ghi_normalized = ghi_wm2 / (1361 * sin(elev) * 0.7^airmass)"
            ),
            "n_hours_with_ghi": n_total_with_ghi,
            "n_skipped_unstable_recovery": n_skipped_unstable,
            "stability_gate": (
                "elev >= 5° AND no_cloud_reference > 0.1 AND potential_solar_factor > 0.05"
            ),
            "all": _stats([(p[0], p[1]) for p in pairs]),
            "by_regime": {
                reg: _stats([(p[0], p[1]) for p in pairs if p[2] == reg])
                for reg in ("clear", "broken", "overcast")
            },
        }

    def _compute_dni_dhi_shadow_report(self, days_back: int) -> dict:
        """Shadow-comparison of cloud_factor vs DNI/DHI signal (#933).

        Walks recent hourly_log entries that carry both the legacy
        3-vector (``solar_vector_*``) AND the DNI/DHI fields populated by
        1.3.6 live logging or CSV import, fits two unconstrained LS
        models against a shared target (``solar_impact_kwh``), and
        reports residual standard deviation per cloud regime.

        Regime classification is descriptive only:
          * ``clear``     — DNI > 400 W/m² and DHI/DNI < 0.3
          * ``overcast``  — DNI < 50 W/m²
          * ``broken``    — everything else (the regime where the
                            cloud_factor scalar is structurally
                            mis-attributed per the issue rationale)

        The 4D model uses the geometric beam projection
        ``dni × max(0, cos(elev)·{−cos(az), sin(az), −sin(az)})`` per
        facade plus raw DHI as a facade-agnostic diffuse signal,
        consistent with the per-facade-coupling proposal in #933.

        Returns ``{"available": False, ...}`` when the overlap set is
        too small to fit reliably.  Pure diagnostic — no model writes.

        **Two targets emitted side-by-side.**

        * ``target_field = "solar_impact_kwh"`` (top-level fields) —
          legacy self-referential target.  Live model's own output
          (``unit_coeff × solar_vector`` battery-smoothed), so the 3D
          regression has a built-in self-correlation advantage; it is
          essentially recovering the implied coefficient over the
          window.  A genuine 4D win on the broken-cloud regime would
          have to overcome this advantage; a tie does not falsify the
          #933 hypothesis.  Kept as the headline report for backward
          compatibility with the 1.3.6 wave-1 surface.
        * ``cross_check_actual`` (sub-block) — non-self-referential
          target derived from the meter and the global dark-sky base
          bucket: ``y_actual = correlation_data[temp_key][wind_bucket]
          − actual_kwh``.  This is the same target NLMS uses for its
          per-unit base-EMA learning, so neither pipeline has a self-
          correlation advantage — both signals must explain the same
          implied solar reduction from first principles.  Hours are
          gated to heating-mode-dominant non-shutdown samples (skip
          ``auxiliary_active``, ``guest_impact_kwh > 0``,
          ``solar_dominant_entities`` non-empty, any unit in cooling
          mode, missing global base bucket).  This is the read that
          informs the #933 promotion decision: if 4D still wins on
          ``broken`` under the actual target, the cloud_factor →
          DNI/DHI signal-replacement hypothesis is supported.
        """
        cutoff = (dt_util.now() - timedelta(days=days_back)).date().isoformat()
        correlation_data = self.coordinator._correlation_data or {}
        correlation_data_per_unit = (
            getattr(self.coordinator, "_correlation_data_per_unit", None) or {}
        )
        energy_sensors = list(getattr(self.coordinator, "energy_sensors", []) or [])

        # Cooling-mode set: any unit in {COOLING, GUEST_COOLING} disqualifies
        # the hour for the actual-target gate (mixed-mode contamination).
        # OFF / DHW / heating units are tolerated.
        _COOLING_MODES = frozenset((MODE_COOLING, MODE_GUEST_COOLING))
        _LEARNABLE_PER_UNIT_MODES = frozenset(
            (MODE_HEATING, MODE_GUEST_HEATING)
        )

        samples: list[dict] = []
        for entry in self.coordinator._hourly_log:
            ts = entry.get("timestamp", "")
            if ts[:10] < cutoff:
                continue
            dni = entry.get("dni")
            dhi = entry.get("dhi")
            sv_s = entry.get("solar_vector_s")
            sv_e = entry.get("solar_vector_e")
            sv_w = entry.get("solar_vector_w")
            target_self = entry.get("solar_impact_kwh")
            if dni is None or dhi is None or target_self is None:
                continue
            try:
                dni_f = float(dni)
                dhi_f = float(dhi)
                y_self = float(target_self)
                # Empty-string / None on early 2D→3D-padding entries
                # collapses to 0.0 — the legacy log shape has W absent.
                sv_s_f = float(sv_s) if sv_s not in (None, "") else 0.0
                sv_e_f = float(sv_e) if sv_e not in (None, "") else 0.0
                sv_w_f = float(sv_w) if sv_w not in (None, "") else 0.0
            except (TypeError, ValueError):
                continue
            try:
                dt_obj = datetime.fromisoformat(ts)
            except (TypeError, ValueError):
                continue
            elev, azim = self.coordinator.solar.get_approx_sun_pos(dt_obj + timedelta(minutes=30))
            if elev <= 0.0:
                continue
            elev_rad = math.radians(elev)
            az_rad = math.radians(azim)
            cos_elev = math.cos(elev_rad)
            sin_elev = math.sin(elev_rad)
            geom_s = max(0.0, cos_elev * -math.cos(az_rad))
            geom_e = max(0.0, cos_elev * math.sin(az_rad))
            geom_w = max(0.0, cos_elev * -math.sin(az_rad))

            # Pre-geometry scalar comparison signals (signal_agreement
            # block).  ``kasten_cf`` recovered from logged
            # ``potential_solar_factor`` divided by the no-cloud
            # equivalent at this sun position; ``dni_normalized`` is
            # ``dni / dni_clear_sky(elev)`` using a Beer-Lambert clear-
            # sky beam estimate (1361 W/m² × 0.7^airmass).  Both are in
            # [0, 1] and pre-geometry — directly comparable.  Either
            # may be None when the entry lacks ``potential_solar_factor``
            # (older log format) or when sun is too low for a stable
            # clear-sky reference.
            kasten_cf: float | None = None
            dni_normalized: float | None = None
            kasten_skipped_unstable_recovery = False
            potential_sf = entry.get("potential_solar_factor")
            if elev >= 5.0:
                # Skip very-low-sun hours where airmass blows up and
                # both signals are near-zero ratios of near-zero values.
                airmass = 1.0 / max(sin_elev, 0.087)  # min sin(5°)
                dni_clear = 1361.0 * (0.7 ** airmass)
                if dni_clear > 1.0:
                    dni_normalized = max(0.0, min(1.5, dni_f / dni_clear))
                if isinstance(potential_sf, (int, float)) and potential_sf >= 0:
                    try:
                        no_cloud_ref = self.coordinator.solar.calculate_solar_factor(
                            elev, azim, 0.0
                        )
                    except (TypeError, ValueError):
                        no_cloud_ref = 0.0
                    # Stable-recovery gate.  ``calculate_solar_factor``
                    # multiplies in the Kelvin-Twist ``az_factor`` whose
                    # Zone 2/3 floor is 0.05–0.1 when sun is far from
                    # the configured ``solar_azimuth``.  Combined with
                    # ``potential_solar_factor``'s 3-decimal rounding
                    # in the hourly log, that turns the inversion
                    # ``potential_sf / no_cloud_ref`` into noise
                    # amplification on early-morning / late-evening
                    # hours — observed empirically as negative-corr
                    # artifacts in early field reports.  Gating to
                    # ``no_cloud_ref > 0.1 ∧ potential_sf > 0.05``
                    # restricts the comparison to hours where both
                    # numerator and denominator carry sub-1 % relative
                    # uncertainty, leaving a clean read on the midday
                    # hours where the cloud_factor recovery is
                    # numerically stable.  Skipped hours are counted
                    # and surfaced via ``n_skipped_unstable_recovery``
                    # so the caller can see the gate's reach.
                    if no_cloud_ref > 0.1 and float(potential_sf) > 0.05:
                        kasten_cf = max(0.0, min(1.5, float(potential_sf) / no_cloud_ref))
                    elif no_cloud_ref > 1e-6:
                        kasten_skipped_unstable_recovery = True
            # Skip hours with no signal at all in either pipeline — they
            # bias the residual toward zero and tell us nothing about
            # which model fits the regime better.
            if (
                y_self == 0.0
                and dni_f == 0.0
                and dhi_f == 0.0
                and sv_s_f == 0.0
                and sv_e_f == 0.0
                and sv_w_f == 0.0
            ):
                continue
            if dni_f < 50.0:
                regime = "overcast"
            elif dni_f > 400.0 and (dhi_f / max(dni_f, 1.0)) < 0.3:
                regime = "clear"
            else:
                regime = "broken"

            # ---- Actual-target gate (non-self-referential) ----
            # y_actual = global_base − actual_kwh, qualified to heating-
            # mode-dominant non-shutdown hours.  Same target NLMS uses
            # for its dark-equivalent base learning, so neither pipeline
            # has a self-correlation advantage.
            y_actual: float | None = None
            if (
                not entry.get("auxiliary_active", False)
                and float(entry.get("guest_impact_kwh") or 0.0) <= 0.0
                and not entry.get("solar_dominant_entities")
            ):
                unit_modes = entry.get("unit_modes") or {}
                if not any(m in _COOLING_MODES for m in unit_modes.values()):
                    temp_key = entry.get("temp_key")
                    wind_bucket = entry.get("wind_bucket")
                    base_house = (
                        correlation_data.get(temp_key, {}).get(wind_bucket)
                        if temp_key is not None and wind_bucket is not None
                        else None
                    )
                    actual = entry.get("actual_kwh")
                    if (
                        isinstance(base_house, (int, float))
                        and base_house > 0.0
                        and isinstance(actual, (int, float))
                    ):
                        y_actual = float(base_house) - float(actual)

            # ---- Per-entity actual target (cross_check_actual.per_entity) ----
            # Same target shape as house cross-check but using each
            # entity's own ``unit_base − unit_actual``.  Eliminates two
            # confounds the house aggregate inherits: (a) cross-unit
            # mixing where a single screen-affected entity's signal
            # contaminates the regression's S/E/W columns via
            # ``actual_kwh = sum(unit_breakdown)``, and (b) global base
            # bucket bias from the legacy solar pipeline (per-entity
            # buckets are smaller in absolute terms but built from the
            # same SNR-weighted dark-equivalent path so the bias is
            # entity-localised, not house-aggregated).
            #
            # Per-entity gates:
            #  * skip if ``auxiliary_active`` (aux reduces unit actual
            #    independent of solar — contaminates the per-unit
            #    regression target same as the house version)
            #  * skip if entity flagged in ``solar_dominant_entities``
            #    for this hour (shutdown — actual ≈ 0, makes
            #    ``unit_base − actual ≈ unit_base`` regardless of sun)
            #  * skip if entity's mode is not in
            #    ``{HEATING, GUEST_HEATING}`` (cooling/OFF/DHW are
            #    out-of-regime for the heating-side coefficient)
            #  * skip if per-entity base bucket is missing or
            #    non-positive
            y_per_entity: dict[str, float] = {}
            unit_breakdown = entry.get("unit_breakdown") or {}
            solar_dominant = set(entry.get("solar_dominant_entities") or [])
            unit_modes_full = entry.get("unit_modes") or {}
            temp_key = entry.get("temp_key")
            wind_bucket = entry.get("wind_bucket")
            if (
                not entry.get("auxiliary_active", False)
                and temp_key is not None
                and wind_bucket is not None
            ):
                for eid in energy_sensors:
                    if eid in solar_dominant:
                        continue
                    mode = unit_modes_full.get(eid, MODE_HEATING)
                    if mode not in _LEARNABLE_PER_UNIT_MODES:
                        continue
                    unit_actual = unit_breakdown.get(eid)
                    if not isinstance(unit_actual, (int, float)):
                        continue
                    unit_base = (
                        correlation_data_per_unit.get(eid, {})
                        .get(temp_key, {})
                        .get(wind_bucket)
                    )
                    if not isinstance(unit_base, (int, float)) or unit_base <= 0.0:
                        continue
                    y_per_entity[eid] = float(unit_base) - float(unit_actual)

            samples.append({
                "regime": regime,
                "x3": (sv_s_f, sv_e_f, sv_w_f),
                "x4": (dni_f * geom_s, dni_f * geom_e, dni_f * geom_w, dhi_f),
                "y_self": y_self,
                "y_actual": y_actual,
                "y_per_entity": y_per_entity,
                "kasten_cf": kasten_cf,
                "dni_normalized": dni_normalized,
                "kasten_skipped_unstable_recovery": kasten_skipped_unstable_recovery,
            })

        # Need enough overlap to fit a 4-parameter model robustly.  Below
        # this floor, surface the count so the caller can trust-but-verify
        # but skip the regression.
        if len(samples) < 60:
            return {
                "available": False,
                "n_hours": len(samples),
                "reason": "insufficient_overlap_data",
            }

        def _solve_lstsq(
            dim: int, key: str, sample_subset: list[dict], y_field: str,
        ) -> tuple[list[float], list[float]] | None:
            """Solve X^T X β = X^T y via Gaussian elimination with partial pivot."""
            xtx = [[0.0] * dim for _ in range(dim)]
            xty = [0.0] * dim
            for s in sample_subset:
                x = s[key]
                y = s[y_field]
                for i in range(dim):
                    xty[i] += x[i] * y
                    for j in range(dim):
                        xtx[i][j] += x[i] * x[j]
            # Tikhonov ridge for collinearity (geometry vectors degenerate
            # at solar noon when sun is exactly south, etc.).  Magnitude
            # tied to mean diagonal so it scales with the data.
            mean_diag = sum(xtx[i][i] for i in range(dim)) / dim if dim else 0.0
            ridge = max(1e-9, 1e-6 * mean_diag)
            for i in range(dim):
                xtx[i][i] += ridge
            # Gaussian elimination
            a = [row[:] + [xty[i]] for i, row in enumerate(xtx)]
            for i in range(dim):
                pivot_row = max(range(i, dim), key=lambda r: abs(a[r][i]))
                a[i], a[pivot_row] = a[pivot_row], a[i]
                if abs(a[i][i]) < 1e-12:
                    return None
                for r in range(i + 1, dim):
                    factor = a[r][i] / a[i][i]
                    for c in range(i, dim + 1):
                        a[r][c] -= factor * a[i][c]
            beta = [0.0] * dim
            for i in range(dim - 1, -1, -1):
                s_val = a[i][dim] - sum(a[i][j] * beta[j] for j in range(i + 1, dim))
                beta[i] = s_val / a[i][i]
            residuals = []
            for s in sample_subset:
                x = s[key]
                yhat = sum(beta[i] * x[i] for i in range(dim))
                residuals.append(s[y_field] - yhat)
            return beta, residuals

        def _std(vals: list[float]) -> float:
            if not vals:
                return 0.0
            mean = sum(vals) / len(vals)
            return (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5

        regimes = ("clear", "broken", "overcast")

        def _build_block(
            sample_subset: list[dict],
            y_field: str,
            target_field: str,
            target_caveat: str,
        ) -> dict:
            if len(sample_subset) < 60:
                return {
                    "available": False,
                    "target_field": target_field,
                    "n_hours": len(sample_subset),
                    "reason": "insufficient_overlap_data",
                }
            fit3 = _solve_lstsq(3, "x3", sample_subset, y_field)
            fit4 = _solve_lstsq(4, "x4", sample_subset, y_field)
            if fit3 is None or fit4 is None:
                return {
                    "available": False,
                    "target_field": target_field,
                    "n_hours": len(sample_subset),
                    "reason": "singular_design_matrix",
                }
            b3, r3 = fit3
            b4, r4 = fit4
            per_regime: dict[str, dict] = {}
            for reg in regimes:
                r3r = [r3[i] for i, s in enumerate(sample_subset) if s["regime"] == reg]
                r4r = [r4[i] for i, s in enumerate(sample_subset) if s["regime"] == reg]
                std3 = _std(r3r)
                std4 = _std(r4r)
                improvement = (
                    round(100.0 * (std3 - std4) / std3, 1)
                    if std3 > 1e-9 else None
                )
                per_regime[reg] = {
                    "n_hours": len(r3r),
                    "residual_std_3d_kwh": round(std3, 4),
                    "residual_std_4d_kwh": round(std4, 4),
                    "improvement_pct": improvement,
                }
            std3_all = _std(r3)
            std4_all = _std(r4)
            overall_improvement = (
                round(100.0 * (std3_all - std4_all) / std3_all, 1)
                if std3_all > 1e-9 else None
            )
            return {
                "available": True,
                "target_field": target_field,
                "target_caveat": target_caveat,
                "n_hours": len(sample_subset),
                "regime_counts": {
                    reg: sum(1 for s in sample_subset if s["regime"] == reg)
                    for reg in regimes
                },
                "shadow_4d_coefficient": {
                    "s_direct": round(b4[0], 6),
                    "e_direct": round(b4[1], 6),
                    "w_direct": round(b4[2], 6),
                    "diffuse":  round(b4[3], 6),
                },
                "shadow_3d_coefficient": {
                    "s": round(b3[0], 4),
                    "e": round(b3[1], 4),
                    "w": round(b3[2], 4),
                },
                "residuals": {
                    "all": {
                        "residual_std_3d_kwh": round(std3_all, 4),
                        "residual_std_4d_kwh": round(std4_all, 4),
                        "improvement_pct": overall_improvement,
                    },
                    "by_regime": per_regime,
                },
                "broken_regime_improvement_pct": (
                    per_regime.get("broken", {}).get("improvement_pct")
                ),
            }

        # Headline self-referential block (kept top-level for backward
        # compatibility with the 1.3.6 wave-1 surface).
        self_block = _build_block(
            samples, "y_self", "solar_impact_kwh",
            "self_referential_3d_advantage",
        )

        # Cross-check on actual-meter target (non-self-referential).
        actual_samples = [s for s in samples if s["y_actual"] is not None]
        cross_check = _build_block(
            actual_samples,
            "y_actual",
            "global_base_minus_actual_kwh",
            "non_self_referential",
        )

        # Per-entity cross-check.  Iterates each configured energy
        # sensor and runs the same 3D / 4D regression against that
        # entity's own ``unit_base − unit_actual`` target.  Sub-blocks
        # land on ``cross_check.per_entity[entity_id]`` so a caller
        # walking the dict can answer "which entities show real 4D
        # gain?" without having to interpret a house-aggregate fit
        # contaminated by mixing.
        per_entity_blocks: dict[str, dict] = {}
        # Source for the ``learned`` flag: the live coordinator's
        # ``solar_coefficients_per_unit[eid]["heating"]["learned"]`` is
        # set by ``_update_unit_solar_coefficient`` on the first write
        # by any solar-learner path (NLMS / inequality / cold-start /
        # batch / apply-implied).  Migration-seeded coefficients carry
        # no flag and read as unlearned — exactly the entities whose
        # per-entity blocks would otherwise contaminate the filtered
        # aggregation with default-coefficient noise.
        coeff_store = (
            getattr(self.coordinator, "_solar_coefficients_per_unit", None) or {}
        )

        def _entity_has_learned(eid: str) -> bool:
            entity_coeffs = coeff_store.get(eid)
            if not isinstance(entity_coeffs, dict):
                return False
            heating = entity_coeffs.get("heating")
            if not isinstance(heating, dict):
                return False
            return bool(heating.get("learned"))

        # Per-entity solar-scope (#962): excluded entities collapse to a
        # one-line stub here too, matching the main per_unit block.
        is_solar_affected_fn = getattr(self.coordinator, "is_solar_affected", None)

        for eid in energy_sensors:
            if callable(is_solar_affected_fn) and not is_solar_affected_fn(eid):
                per_entity_blocks[eid] = {"excluded_from_solar": True}
                continue
            entity_samples = [
                {**s, "_y_eid": s["y_per_entity"][eid]}
                for s in samples
                if eid in s["y_per_entity"]
            ]
            block = _build_block(
                entity_samples,
                "_y_eid",
                "unit_base_minus_unit_actual_kwh",
                "non_self_referential_per_entity",
            )
            screen_config_fn = getattr(
                self.coordinator, "screen_config_for_entity", None
            )
            if callable(screen_config_fn):
                try:
                    sc = screen_config_fn(eid)
                except (TypeError, ValueError):
                    sc = None
                if sc is not None:
                    try:
                        block["screen_config"] = [bool(v) for v in sc]
                    except (TypeError, ValueError):
                        pass
            block["learned"] = _entity_has_learned(eid)
            per_entity_blocks[eid] = block
        if per_entity_blocks:
            cross_check["per_entity"] = per_entity_blocks

            # ``per_entity_filtered`` aggregates only entities that have
            # actually learned a heating coefficient (per the live
            # ``learned`` flag) AND have an available block.  Drops
            # default-coefficient entities whose per-entity fits are
            # rein noise from the headline numbers.  Hours-weighted
            # average of the per-entity ``improvement_pct`` per regime.
            eligible = [
                (eid, b) for eid, b in per_entity_blocks.items()
                if b.get("learned") and b.get("available")
            ]
            if eligible:
                def _weighted_pct(regime: str | None) -> tuple[float | None, int]:
                    weighted_sum = 0.0
                    n_total = 0
                    for _eid, b in eligible:
                        if regime is None:
                            n = b.get("n_hours", 0)
                            imp = b.get("residuals", {}).get("all", {}).get("improvement_pct")
                        else:
                            reg_block = (
                                b.get("residuals", {}).get("by_regime", {}).get(regime, {})
                            )
                            n = reg_block.get("n_hours", 0)
                            imp = reg_block.get("improvement_pct")
                        if imp is None or n <= 0:
                            continue
                        weighted_sum += float(imp) * n
                        n_total += n
                    if n_total == 0:
                        return None, 0
                    return round(weighted_sum / n_total, 1), n_total

                regime_pct: dict[str, dict] = {}
                for reg in regimes:
                    pct, n = _weighted_pct(reg)
                    regime_pct[reg] = {"improvement_pct": pct, "n_hours": n}
                all_pct, all_n = _weighted_pct(None)
                cross_check["per_entity_filtered"] = {
                    "n_entities": len(eligible),
                    "entities": [eid for eid, _ in eligible],
                    "all": {"improvement_pct": all_pct, "n_hours": all_n},
                    "by_regime": regime_pct,
                    "broken_regime_improvement_pct": (
                        regime_pct.get("broken", {}).get("improvement_pct")
                    ),
                }

        # ``signal_agreement`` — pre-geometry scalar comparison of the
        # Kasten-derived cloud_factor against ``dni / dni_clear_sky``.
        # Both signals live in [0, 1] and capture "how much beam survived
        # the atmosphere this hour".  High Pearson r AND low RMSE means
        # DNI re-encodes what cloud_coverage × Kasten already says (no
        # new information); divergence on broken-cloud regime is the
        # signature that the issue #933 hypothesis predicts.
        agreement_pairs = [
            (s["kasten_cf"], s["dni_normalized"], s["regime"])
            for s in samples
            if s.get("kasten_cf") is not None and s.get("dni_normalized") is not None
        ]

        def _pearson_rmse_bias(pairs: list[tuple[float, float]]) -> dict:
            n = len(pairs)
            if n < 5:
                return {"n_hours": n, "correlation": None, "rmse": None, "mean_bias": None}
            mean_x = sum(p[0] for p in pairs) / n
            mean_y = sum(p[1] for p in pairs) / n
            num = sum((p[0] - mean_x) * (p[1] - mean_y) for p in pairs)
            den_x = (sum((p[0] - mean_x) ** 2 for p in pairs)) ** 0.5
            den_y = (sum((p[1] - mean_y) ** 2 for p in pairs)) ** 0.5
            corr = num / (den_x * den_y) if den_x > 1e-9 and den_y > 1e-9 else None
            rmse = (sum((p[0] - p[1]) ** 2 for p in pairs) / n) ** 0.5
            # Positive bias = Kasten higher than DNI-normalized (Kasten
            # over-attributes transmittance, predicts more sun than DNI
            # actually shows).  Negative bias = Kasten under-attributes.
            bias = mean_x - mean_y
            return {
                "n_hours": n,
                "correlation": round(corr, 4) if corr is not None else None,
                "rmse": round(rmse, 4),
                "mean_bias_kasten_minus_dni": round(bias, 4),
            }

        signal_agreement = {
            "definition": (
                "kasten_cloud_factor = potential_solar_factor / "
                "no_cloud_reference(elev, azim); "
                "dni_normalized = dni / (1361 * 0.7^airmass)"
            ),
            "n_skipped_unstable_recovery": sum(
                1 for s in samples if s.get("kasten_skipped_unstable_recovery")
            ),
            "stability_gate": (
                "no_cloud_reference > 0.1 AND potential_solar_factor > 0.05"
            ),
            "all": _pearson_rmse_bias([(p[0], p[1]) for p in agreement_pairs]),
            "by_regime": {
                reg: _pearson_rmse_bias(
                    [(p[0], p[1]) for p in agreement_pairs if p[2] == reg]
                )
                for reg in regimes
            },
        }

        # Flatten the headline block at the top level (matches 1.3.6
        # wave-1 surface) and attach cross_check_actual as a sibling.
        return {
            **self_block,
            "signal_agreement": signal_agreement,
            "cross_check_actual": cross_check,
        }

    def calibrate_per_unit_min_base_thresholds(
        self,
        *,
        sample_days: int = 30,
        require_min_hours_of_log: int | None = None,
    ) -> dict:
        """Compute per-unit min-base noise floor from dark-hour actuals (#871).

        Replaces the global 0.15 kWh gate with a per-sensor p10 of
        dark-hour (``solar_factor < PER_UNIT_MIN_BASE_DARK_SOLAR_FACTOR``)
        metered consumption.  Dark-hour filtering isolates the non-solar
        base-demand distribution; p10 captures the operating-noise floor
        without being skewed by the tail of idle samples.

        Safety guards (in order):
            1. Requires ≥ ``PER_UNIT_MIN_BASE_MIN_HOURS_OF_LOG`` hours of
               log data overall (14 × 24 by default).  Fresh installs
               skip calibration and continue on the global fallback.
            2. Requires ≥ ``PER_UNIT_MIN_BASE_MIN_SAMPLES`` dark-hour
               samples per unit.  Under-sampled units keep their prior
               value (or skip entirely if never calibrated).
            3. Rejects when ``p10 / median(dark_samples)`` exceeds
               ``PER_UNIT_MIN_BASE_MAX_P10_MEDIAN_RATIO``.  A legitimate
               noise floor sits far below typical consumption; a ratio
               near 1.0 indicates an always-on load (electric boiler
               mislabeled as heat-pump heating, sensor scoped to a
               shared circuit) where p10 is not a noise floor at all.
               Primary physics-grounded filter.
            4. Clamps p10 from below to ``PER_UNIT_MIN_BASE_FLOOR``;
               absolute ceiling ``PER_UNIT_MIN_BASE_CEILING`` acts as
               a safety net behind the ratio-guard.
            5. Limits rate-of-change to
               ``PER_UNIT_MIN_BASE_MAX_RATE_OF_CHANGE`` per run (±50 %
               vs previous value).  Protects against a single anomalous
               week flipping the threshold.

        Only heating-mode samples contribute.  Aux-active hours and guest
        modes are excluded — matches the live learning exclusion set.

        Returns a diagnostic dict usable by the ``calibrate_unit_thresholds``
        service and the startup log; ``self.coordinator._per_unit_min_base_thresholds``
        is updated in-place.
        """
        from datetime import timedelta
        from homeassistant.util import dt as dt_util
        from .const import (
            MODE_HEATING,
            PER_UNIT_MIN_BASE_CEILING,
            PER_UNIT_MIN_BASE_DARK_SOLAR_FACTOR,
            PER_UNIT_MIN_BASE_FLOOR,
            PER_UNIT_MIN_BASE_MAX_P10_MEDIAN_RATIO,
            PER_UNIT_MIN_BASE_MAX_RATE_OF_CHANGE,
            PER_UNIT_MIN_BASE_MIN_HOURS_OF_LOG,
            PER_UNIT_MIN_BASE_MIN_SAMPLES,
        )

        min_hours = (
            PER_UNIT_MIN_BASE_MIN_HOURS_OF_LOG
            if require_min_hours_of_log is None
            else require_min_hours_of_log
        )
        total_hours = len(self.coordinator._hourly_log)
        result = {
            "total_log_hours": total_hours,
            "required_log_hours": min_hours,
            "sample_days": sample_days,
            "status": "ok",
            "units": {},
            "updated": {},
            "rejected": {},
            "skipped": {},
        }
        if total_hours < min_hours:
            result["status"] = "insufficient_log_data"
            return result

        cutoff_iso = (dt_util.now() - timedelta(days=sample_days)).date().isoformat()

        # Collect dark-hour actuals per unit.
        samples: dict[str, list[float]] = {sid: [] for sid in self.coordinator.energy_sensors}
        for entry in self.coordinator._hourly_log:
            ts = entry.get("timestamp", "")
            if ts[:10] < cutoff_iso:
                continue
            if entry.get("auxiliary_active", False):
                continue
            solar_factor = entry.get("solar_factor") or 0.0
            if solar_factor >= PER_UNIT_MIN_BASE_DARK_SOLAR_FACTOR:
                continue
            unit_modes = entry.get("unit_modes", {}) or {}
            unit_breakdown = entry.get("unit_breakdown", {}) or {}
            for sid in self.coordinator.energy_sensors:
                mode = unit_modes.get(sid, MODE_HEATING)
                if mode != MODE_HEATING:
                    continue
                if sid not in unit_breakdown:
                    continue
                actual = unit_breakdown.get(sid, 0.0)
                if actual is None or actual < 0.0:
                    continue
                samples[sid].append(float(actual))

        def _p10(values: list[float]) -> float:
            if not values:
                return 0.0
            s = sorted(values)
            # Nearest-rank p10, 1-indexed: idx = ceil(0.10 × n) - 1 in 0-indexed.
            # math.ceil is required (not round): round underestimates the
            # rank for n ∈ {21..25} — and Python's banker's rounding flips
            # n=25 to 2 rather than 3 — which would bias calibrated
            # thresholds downward and let noisy low-base hours through
            # the NLMS gate.
            idx = max(0, math.ceil(0.10 * len(s)) - 1)
            return s[idx]

        for sid in self.coordinator.energy_sensors:
            dark = samples.get(sid, [])
            n = len(dark)
            prior = self.coordinator._per_unit_min_base_thresholds.get(sid)
            unit_report = {
                "dark_samples": n,
                "prior": prior,
                "p10_actual": None,
                "effective": prior,
                "method": "prior" if prior is not None else "fallback",
            }
            if n < PER_UNIT_MIN_BASE_MIN_SAMPLES:
                unit_report["status"] = "skipped_low_samples"
                result["skipped"][sid] = unit_report
                result["units"][sid] = unit_report
                continue

            p10 = _p10(dark)
            unit_report["p10_actual"] = round(p10, 5)

            # Ratio-guard (primary filter): a legitimate noise floor
            # sits far below typical consumption.  A sorted-dark-sample
            # distribution where p10 approaches the median means the
            # sensor is measuring an always-on load rather than a
            # modulating heat pump — no noise floor exists to calibrate.
            sorted_dark = sorted(dark)
            median = sorted_dark[len(sorted_dark) // 2]
            unit_report["median_actual"] = round(median, 5)
            if median > 0.0 and p10 > PER_UNIT_MIN_BASE_MAX_P10_MEDIAN_RATIO * median:
                unit_report["status"] = "rejected_constant_load"
                unit_report["p10_over_median"] = round(p10 / median, 3)
                _LOGGER.warning(
                    "Per-unit min-base calibration rejected for %s: "
                    "p10=%.3f kWh is %.0f%% of median=%.3f — distribution "
                    "suggests an always-on load, not a modulating heat pump. "
                    "Keeping %s.",
                    sid, p10, 100.0 * p10 / median, median,
                    f"prior {prior:.3f}" if prior else "global fallback",
                )
                result["rejected"][sid] = unit_report
                result["units"][sid] = unit_report
                continue

            # Absolute ceiling (safety net behind the ratio-guard).
            if p10 > PER_UNIT_MIN_BASE_CEILING:
                unit_report["status"] = "rejected_above_ceiling"
                _LOGGER.warning(
                    "Per-unit min-base calibration rejected for %s: p10=%.3f kWh "
                    "exceeds ceiling %.3f — keeping %s.",
                    sid, p10, PER_UNIT_MIN_BASE_CEILING,
                    f"prior {prior:.3f}" if prior else "global fallback",
                )
                result["rejected"][sid] = unit_report
                result["units"][sid] = unit_report
                continue

            candidate = max(PER_UNIT_MIN_BASE_FLOOR, p10)

            # Rate-of-change clamp vs prior value.
            if prior is not None and prior > 0.0:
                lo = prior * (1.0 - PER_UNIT_MIN_BASE_MAX_RATE_OF_CHANGE)
                hi = prior * (1.0 + PER_UNIT_MIN_BASE_MAX_RATE_OF_CHANGE)
                clamped = min(hi, max(lo, candidate))
                if clamped != candidate:
                    unit_report["rate_clamped_from"] = round(candidate, 5)
                candidate = max(PER_UNIT_MIN_BASE_FLOOR, clamped)

            new_value = round(candidate, 5)
            self.coordinator._per_unit_min_base_thresholds[sid] = new_value
            unit_report["effective"] = new_value
            unit_report["method"] = "auto"
            unit_report["status"] = "updated"
            result["updated"][sid] = unit_report
            result["units"][sid] = unit_report

        _LOGGER.info(
            "Per-unit min-base calibration complete: updated=%d, rejected=%d, skipped=%d",
            len(result["updated"]), len(result["rejected"]), len(result["skipped"]),
        )
        return result

    def _compute_base_model_4d_shadow_report(self, days_back: int) -> dict:
        """Path-B promotion metric for the 4D solar shadow learner (#954).

        Re-aggregates the last ``days_back`` days of base-bucket EMA twice
        in parallel — once with ``solar_normalization_delta`` (3D, status
        quo) and once with ``solar_normalization_delta_4d`` (4D shadow,
        per-hour field landed in 7b8cb0a) — and reports per-cell
        ``drift_kwh`` plus the per-step EMA jitter RMS for each path.

        A bucket whose normalisation delta correctly captures the hourly
        solar contribution converges to a stable value and per-step EMA
        jitter trends toward zero.  A noisy / biased delta keeps nudging
        the bucket inconsistently and step jitter stays high.  Lower
        ``step_*_rms`` therefore means flatter base buckets — better
        physics.  The headline ``shoulder_ratio = 4d_rms / 3d_rms`` over
        ``[7-14°C, normal-wind]`` cells is the dominant promotion gate.

        Strict diagnostic — reads only ``hourly_log`` and the live
        ``correlation_data`` seed; writes nothing back to model state.
        Returns ``{"available": False, "reason":
        "no_4d_tagged_hours", ...}`` when the 4D learner has not yet
        produced any tagged hours (fresh install, or 4D potential never
        qualified).  No synthetic-delta fallback — the absence is the
        signal.
        """
        cutoff = (dt_util.now() - timedelta(days=days_back)).date().isoformat()
        correlation_data = self.coordinator._correlation_data or {}
        learning_rate = float(getattr(self.coordinator, "learning_rate", 0.0) or 0.0)
        energy_sensors = list(getattr(self.coordinator, "energy_sensors", []) or [])
        total_units = max(1, len(energy_sensors))

        _COOLING_MODES = frozenset((MODE_COOLING, MODE_GUEST_COOLING))

        # Walk hourly_log once, gather qualifying samples grouped by cell
        # in chronological order (the log itself is chronological).
        per_cell: dict[tuple, list[dict]] = {}
        n_skipped_no_4d = 0
        for entry in self.coordinator._hourly_log:
            ts = entry.get("timestamp", "")
            if ts[:10] < cutoff:
                continue
            d3 = entry.get("solar_normalization_delta")
            d4 = entry.get("solar_normalization_delta_4d")
            actual = entry.get("actual_kwh")
            temp_key = entry.get("temp_key")
            wind_bucket = entry.get("wind_bucket")
            if (
                d3 is None
                or actual is None
                or temp_key is None
                or wind_bucket is None
            ):
                # Missing essentials — silently skip; not the 4D gate.
                continue
            if d4 is None:
                n_skipped_no_4d += 1
                continue
            # Same gates as base learning: aux active, guest impact,
            # cooling-mode hours all excluded.
            if entry.get("auxiliary_active", False):
                continue
            try:
                if float(entry.get("guest_impact_kwh") or 0.0) > 0.0:
                    continue
            except (TypeError, ValueError):
                continue
            unit_modes = entry.get("unit_modes") or {}
            if any(m in _COOLING_MODES for m in unit_modes.values()):
                continue
            try:
                actual_f = float(actual)
                d3_f = float(d3)
                d4_f = float(d4)
                sf_f = float(entry.get("solar_factor") or 0.0)
            except (TypeError, ValueError):
                continue
            sde = entry.get("solar_dominant_entities") or []
            per_cell.setdefault((temp_key, wind_bucket), []).append(
                {
                    "actual": actual_f,
                    "d3": d3_f,
                    "d4": d4_f,
                    "sf": sf_f,
                    "sde": sde,
                }
            )

        total_hours = sum(len(v) for v in per_cell.values())
        if total_hours == 0:
            return {
                "available": False,
                "n_hours": 0,
                "n_hours_skipped_no_4d_delta": n_skipped_no_4d,
                "days_back": days_back,
                "reason": "no_4d_tagged_hours",
            }

        # Walk each cell's samples chronologically, applying the live
        # base-EMA update twice (once per delta).  Both sims start from
        # the same seed (the cell's current live value) — fair comparison.
        per_cell_out: dict[str, dict] = {}
        shoulder_temps = list(range(7, 15))  # [7..14]
        shoulder_step_3d_sq: list[float] = []
        shoulder_step_4d_sq: list[float] = []
        shoulder_cell_count = 0

        for (temp_key, wind_bucket), samples in per_cell.items():
            if not samples:
                continue
            seed = correlation_data.get(temp_key, {}).get(wind_bucket, 0.0)
            try:
                seed_f = float(seed) if seed is not None else 0.0
            except (TypeError, ValueError):
                seed_f = 0.0

            bucket_3d = seed_f
            bucket_4d = seed_f
            step_3d_sq_sum = 0.0
            step_4d_sq_sum = 0.0

            for s in samples:
                # #967: EMA step routed through ``helpers.compute_base_ema_step``
                # so the diagnostic simulation uses the same arithmetic
                # source-of-truth the live writer will when its inline
                # form (learning.py:990) is migrated.  Target construction
                # (``max(0, actual + delta)``) stays caller-side per the
                # helper's contract.
                target_3d = max(0.0, s["actual"] + s["d3"])
                target_4d = max(0.0, s["actual"] + s["d4"])
                weight = compute_snr_weight(s["sf"], s["sde"], total_units)
                bucket_3d, step_3d = compute_base_ema_step(
                    bucket_3d, target_3d, learning_rate, weight,
                )
                bucket_4d, step_4d = compute_base_ema_step(
                    bucket_4d, target_4d, learning_rate, weight,
                )
                step_3d_sq_sum += step_3d * step_3d
                step_4d_sq_sum += step_4d * step_4d

            n = len(samples)
            step_3d_rms = math.sqrt(step_3d_sq_sum / n)
            step_4d_rms = math.sqrt(step_4d_sq_sum / n)
            ratio: float | None
            if step_3d_rms > 0.0:
                ratio = round(step_4d_rms / step_3d_rms, 5)
            else:
                ratio = None

            cell_key = f"{temp_key}/{wind_bucket}"
            per_cell_out[cell_key] = {
                "n": n,
                "bucket_3d_initial": round(seed_f, 5),
                "bucket_3d_final": round(bucket_3d, 5),
                "bucket_4d_final": round(bucket_4d, 5),
                "drift_kwh": round(bucket_4d - bucket_3d, 5),
                "step_3d_rms": round(step_3d_rms, 6),
                "step_4d_rms": round(step_4d_rms, 6),
                "step_4d_to_3d_ratio": ratio,
            }

            # Shoulder aggregate: integer temp_key in [7..14] AND
            # normal-wind bucket.  temp_key is logged as a string of the
            # rounded integer temperature; coerce defensively.
            try:
                tk_int = int(temp_key)
            except (TypeError, ValueError):
                tk_int = None
            if (
                tk_int is not None
                and tk_int in shoulder_temps
                and wind_bucket == "normal"
            ):
                shoulder_step_3d_sq.append(step_3d_rms * step_3d_rms)
                shoulder_step_4d_sq.append(step_4d_rms * step_4d_rms)
                shoulder_cell_count += 1

        if shoulder_cell_count > 0:
            rms_3d = math.sqrt(sum(shoulder_step_3d_sq) / shoulder_cell_count)
            rms_4d = math.sqrt(sum(shoulder_step_4d_sq) / shoulder_cell_count)
            shoulder_ratio: float | None
            if rms_3d > 0.0:
                shoulder_ratio = round(rms_4d / rms_3d, 5)
            else:
                shoulder_ratio = None
            headline = {
                "bucket_drift_rms_3d_shoulder": round(rms_3d, 6),
                "bucket_drift_rms_4d_shoulder": round(rms_4d, 6),
                "shoulder_ratio": shoulder_ratio,
                "shoulder_cell_count": shoulder_cell_count,
            }
        else:
            headline = {
                "bucket_drift_rms_3d_shoulder": 0.0,
                "bucket_drift_rms_4d_shoulder": 0.0,
                "shoulder_ratio": None,
                "shoulder_cell_count": 0,
            }

        return {
            "available": True,
            "n_hours": total_hours,
            "n_hours_skipped_no_4d_delta": n_skipped_no_4d,
            "days_back": days_back,
            "shoulder_temps": shoulder_temps,
            "headline": headline,
            "per_cell": per_cell_out,
        }

    def _replay_4d_solar_total(self, entry: dict) -> float | None:
        """Post-mortem replay of whole-house 4D solar impact for one hour.

        Reuses the live 4D pipeline (``resolve_dni_dhi`` +
        ``calculate_unit_potential_4d``) with current 4D coefficients to
        reconstruct an estimated 4D solar total for hours that pre-date
        the shadow learner being active OR where 4D logging skipped.

        Returns ``None`` when inputs are insufficient — no logged
        ``dni``/``dhi`` and no logged ``cloud_coverage`` for the entry,
        or no parseable timestamp.

        Caveat: uses CURRENT learned 4D coefficients applied to historical
        hours (same temporal-mismatch assumption the 3D path implicitly
        makes in this diagnose block).
        """
        coord = self.coordinator
        solar_calc = getattr(coord, "solar", None)
        if not isinstance(solar_calc, SolarCalculator):
            # Real coordinators wire a SolarCalculator instance; MagicMock-
            # based test coordinators that don't exercise the replay get
            # a no-op fallback here.
            return None

        inputs, fail = reconstruct_hour_inputs(entry, solar_calc)
        if inputs is None:
            # Night → 0 (no solar). All other failures → None.
            return 0.0 if fail == HOUR_INPUT_FAIL_SUN_BELOW_HORIZON else None
        sun_elev = inputs.sun_elev
        sun_az = inputs.sun_az
        dni = inputs.dni
        dhi = inputs.dhi

        correction = float(entry.get("correction_percent", 0.0) or 0.0)
        unit_modes = entry.get("unit_modes") or {}
        coeffs_4d = getattr(coord, "_solar_coefficients_4d_per_unit", {}) or {}
        if not isinstance(coeffs_4d, dict):
            return None
        is_solar_affected = getattr(coord, "is_solar_affected", None)

        total_4d = 0.0
        for entity_id in (getattr(coord, "energy_sensors", None) or []):
            if callable(is_solar_affected) and not is_solar_affected(entity_id):
                continue
            mode = unit_modes.get(entity_id, MODE_HEATING)
            regime = _solar_coeff_regime(mode)
            if regime is None:
                continue
            entity_coeffs = coeffs_4d.get(entity_id, {})
            regime_coeffs = (
                entity_coeffs.get(regime, {})
                if isinstance(entity_coeffs, dict)
                else {}
            )
            c_s = float(regime_coeffs.get("s", 0.0))
            c_e = float(regime_coeffs.get("e", 0.0))
            c_w = float(regime_coeffs.get("w", 0.0))
            c_d = float(regime_coeffs.get("diffuse", 0.0))
            if c_s == 0.0 and c_e == 0.0 and c_w == 0.0 and c_d == 0.0:
                continue  # entity has no 4D learning yet — would skew toward 0

            screen_cfg = (
                coord.screen_config_for_entity(entity_id)
                if hasattr(coord, "screen_config_for_entity")
                else (False, False, False)
            )
            try:
                p_s, p_e, p_w, p_d = solar_calc.calculate_unit_potential_4d(
                    entity_id=entity_id,
                    dni=dni,
                    dhi=dhi,
                    sun_elev_deg=sun_elev,
                    sun_azimuth_deg=sun_az,
                    screen_config=screen_cfg,
                    correction_percent=correction,
                )
            except Exception:  # noqa: BLE001
                return None
            total_4d += c_s * p_s + c_e * p_e + c_w * p_w + c_d * p_d

        return total_4d

    def _compute_total_power_4d_divergence_report(self, days_back: int) -> dict:
        """Side-by-side replay of 3D vs 4D ``calculate_total_power`` (#962).

        Walks each qualifying entry in ``hourly_log`` over the last
        ``days_back`` days, replays both ``calculate_total_power`` (3D,
        forced via ``override_solar_factor`` + ``override_solar_vector``)
        and ``calculate_total_power_4d`` (forced via ``override_dni_dhi``
        + ``override_sun_pos``) on identical base / aux / mode inputs,
        and aggregates per-hour deltas grouped by cloud regime.

        Cloud regime classification matches
        ``_compute_dni_dhi_shadow_report`` exactly:
          * ``clear``     — DNI > 400 W/m² and DHI/DNI < 0.3
          * ``overcast``  — DNI < 50 W/m²
          * ``broken``    — everything else

        Strict diagnostic — reads only ``hourly_log``; writes nothing.
        Both replay calls run with ``detailed=False`` to skip the
        per-unit breakdown dict on a ~360 daylight-hour window.

        Returns ``{"available": False, "reason": ...}`` when there are
        too few eligible hours to characterise any regime reliably.
        """
        # Verdict thresholds — diagnostic only, kept local so future
        # tuning is visible in the report itself rather than buried in
        # const.py.  See the ``verdict_thresholds`` block in the
        # returned dict.
        _MIN_REGIME_HOURS = 5
        _VERDICT_MIN_BROKEN_HOURS = 20
        _VERDICT_ALIGNED_REL_MAX = 0.05  # < 5 % median rel delta
        _VERDICT_DIVERGE_REL_MIN = 0.10  # ≥ 10 % median rel delta

        log_iter = getattr(self.coordinator, "_hourly_log", None) or []
        cutoff = (dt_util.now() - timedelta(days=days_back)).date().isoformat()

        # Bucketed per-hour samples per regime; ``all`` aggregates
        # everything for the headline row.
        regime_samples: dict[str, list[dict]] = {
            "clear": [], "broken": [], "overcast": [],
        }
        n_total_window = 0
        n_skipped_missing_fields = 0
        n_skipped_below_horizon = 0
        n_skipped_errors = 0

        for entry in log_iter:
            ts = entry.get("timestamp", "")
            if ts[:10] < cutoff:
                continue
            n_total_window += 1

            # Required schema — pre-#933 entries lack DNI/DHI and early
            # 2D→3D-padding entries lack ``solar_vector_w``.  Both
            # populations are silently dropped via
            # ``n_skipped_missing_fields``.
            dni = entry.get("dni")
            dhi = entry.get("dhi")
            sv_s = entry.get("solar_vector_s")
            sv_e = entry.get("solar_vector_e")
            sv_w = entry.get("solar_vector_w")
            solar_factor = entry.get("solar_factor")
            temp = entry.get("temp")
            eff_wind = entry.get("effective_wind")
            correction = entry.get("correction_percent")
            if (
                dni is None or dhi is None
                or sv_s in (None, "") or sv_e in (None, "")
                or sv_w in (None, "")
                or solar_factor is None
                or temp is None or eff_wind is None
                or correction is None
            ):
                n_skipped_missing_fields += 1
                continue

            try:
                dni_f = float(dni)
                dhi_f = float(dhi)
                sv_s_f = float(sv_s)
                sv_e_f = float(sv_e)
                sv_w_f = float(sv_w)
                sf_f = float(solar_factor)
                temp_f = float(temp)
                wind_f = float(eff_wind)
            except (TypeError, ValueError):
                n_skipped_missing_fields += 1
                continue

            # Parse timestamp; compute sun position once.
            try:
                dt_obj = datetime.fromisoformat(ts)
            except (TypeError, ValueError):
                n_skipped_missing_fields += 1
                continue
            try:
                elev, azim = self.coordinator.solar.get_approx_sun_pos(dt_obj)
            except Exception:  # noqa: BLE001 — defensive against mocks
                n_skipped_errors += 1
                continue
            if elev <= 0.0:
                n_skipped_below_horizon += 1
                continue

            # Cloud regime — same convention as
            # ``_compute_dni_dhi_shadow_report``.
            if dni_f < 50.0:
                regime = "overcast"
            elif dni_f > 400.0 and (dhi_f / max(dni_f, 1.0)) < 0.3:
                regime = "clear"
            else:
                regime = "broken"

            is_aux = bool(entry.get("auxiliary_active", False))
            unit_modes = entry.get("unit_modes") or None

            # Replay both pipelines.  ``detailed=False`` — the per-unit
            # breakdown is hot for a 360-hour window.  Defensive try /
            # except so one bad hour cannot kill the whole report.
            try:
                r3 = self.coordinator.statistics.calculate_total_power(
                    temp=temp_f,
                    effective_wind=wind_f,
                    solar_impact=0.0,
                    is_aux_active=is_aux,
                    unit_modes=unit_modes,
                    override_solar_factor=sf_f,
                    override_solar_vector=(sv_s_f, sv_e_f, sv_w_f),
                    detailed=False,
                    override_now=dt_obj,
                )
                r4 = self.coordinator.statistics.calculate_total_power_4d(
                    temp=temp_f,
                    effective_wind=wind_f,
                    solar_impact=0.0,
                    is_aux_active=is_aux,
                    unit_modes=unit_modes,
                    detailed=False,
                    override_now=dt_obj,
                    override_dni_dhi=(dni_f, dhi_f),
                    override_sun_pos=(elev, azim),
                    override_correction_percent=float(correction),
                )
            except Exception:  # noqa: BLE001
                n_skipped_errors += 1
                continue

            try:
                delta_total = float(r4["total_kwh"]) - float(r3["total_kwh"])
                delta_solar = (
                    float(r4["breakdown"]["solar_reduction_kwh"])
                    - float(r3["breakdown"]["solar_reduction_kwh"])
                )
                delta_solar_heating = (
                    float(r4["breakdown"]["solar_heating_applied_kwh"])
                    - float(r3["breakdown"]["solar_heating_applied_kwh"])
                )
                gbase = float(r3.get("global_base_kwh", 0.0))
            except (KeyError, TypeError, ValueError):
                n_skipped_errors += 1
                continue

            regime_samples[regime].append({
                "delta_total": delta_total,
                "delta_solar_applied": delta_solar,
                "delta_solar_heating_applied": delta_solar_heating,
                "global_base_kwh": gbase,
            })

        n_eligible = sum(len(v) for v in regime_samples.values())

        if n_eligible == 0:
            return {
                "available": False,
                "reason": "no_hours_with_dni_dhi" if n_skipped_missing_fields else "no_eligible_hours",
                "days_back": days_back,
                "window_cutoff": cutoff,
                "n_total_log_entries": n_total_window,
                "n_eligible_hours": 0,
                "n_skipped_missing_fields": n_skipped_missing_fields,
                "n_skipped_below_horizon": n_skipped_below_horizon,
                "n_skipped_errors": n_skipped_errors,
                "regime_counts": {k: 0 for k in ("clear", "broken", "overcast")},
            }

        def _median(values: list[float]) -> float:
            n_v = len(values)
            if n_v == 0:
                return 0.0
            s = sorted(values)
            mid = n_v // 2
            if n_v % 2 == 1:
                return s[mid]
            return (s[mid - 1] + s[mid]) / 2.0

        def _mean(values: list[float]) -> float:
            if not values:
                return 0.0
            return sum(values) / len(values)

        def _stats_for(samples: list[dict]) -> dict:
            if not samples:
                return {
                    "n_hours": 0,
                    "median_abs_delta_total_kwh": None,
                    "median_signed_delta_total_kwh": None,
                    "median_abs_delta_solar_applied_kwh": None,
                    "median_signed_delta_solar_applied_kwh": None,
                    "median_abs_delta_solar_heating_applied_kwh": None,
                    "median_signed_delta_solar_heating_applied_kwh": None,
                    "median_relative_delta_total": None,
                    "mean_abs_delta_total_kwh": None,
                }
            d_total = [s["delta_total"] for s in samples]
            d_solar = [s["delta_solar_applied"] for s in samples]
            d_solar_h = [s["delta_solar_heating_applied"] for s in samples]
            rel = [
                abs(s["delta_total"]) / max(s["global_base_kwh"], 0.1)
                for s in samples
            ]
            return {
                "n_hours": len(samples),
                "median_abs_delta_total_kwh": round(_median([abs(x) for x in d_total]), 4),
                "median_signed_delta_total_kwh": round(_median(d_total), 4),
                "median_abs_delta_solar_applied_kwh": round(_median([abs(x) for x in d_solar]), 4),
                "median_signed_delta_solar_applied_kwh": round(_median(d_solar), 4),
                "median_abs_delta_solar_heating_applied_kwh": round(_median([abs(x) for x in d_solar_h]), 4),
                "median_signed_delta_solar_heating_applied_kwh": round(_median(d_solar_h), 4),
                "median_relative_delta_total": round(_median(rel), 4),
                "mean_abs_delta_total_kwh": round(_mean([abs(x) for x in d_total]), 4),
            }

        per_regime = {k: _stats_for(v) for k, v in regime_samples.items()}
        all_samples = [s for v in regime_samples.values() for s in v]
        per_regime["all"] = _stats_for(all_samples)

        # Verdict on the broken-cloud regime — this is where the 4D
        # pipeline is hypothesised to diverge from 3D (per #933 / #962).
        broken = per_regime["broken"]
        any_undersampled = any(
            per_regime[k]["n_hours"] < _MIN_REGIME_HOURS
            for k in ("clear", "broken", "overcast")
        )
        if any_undersampled:
            verdict = "insufficient_data"
        elif broken["n_hours"] >= _VERDICT_MIN_BROKEN_HOURS:
            rel = broken["median_relative_delta_total"] or 0.0
            if rel >= _VERDICT_DIVERGE_REL_MIN:
                verdict = "4d_meaningfully_diverges_on_broken"
            elif rel < _VERDICT_ALIGNED_REL_MAX:
                verdict = "4d_aligned_with_3d"
            else:
                verdict = "4d_diverges_modestly_on_broken"
        else:
            verdict = "4d_diverges_modestly_on_broken"

        return {
            "available": True,
            "days_back": days_back,
            "window_cutoff": cutoff,
            "n_total_log_entries": n_total_window,
            "n_eligible_hours": n_eligible,
            "n_skipped_missing_fields": n_skipped_missing_fields,
            "n_skipped_below_horizon": n_skipped_below_horizon,
            "n_skipped_errors": n_skipped_errors,
            "regime_counts": {
                "clear": per_regime["clear"]["n_hours"],
                "broken": per_regime["broken"]["n_hours"],
                "overcast": per_regime["overcast"]["n_hours"],
            },
            "per_regime": per_regime,
            "verdict": verdict,
            "verdict_thresholds": {
                "min_regime_hours_for_verdict": _MIN_REGIME_HOURS,
                "min_broken_hours_for_threshold_verdict": _VERDICT_MIN_BROKEN_HOURS,
                "aligned_relative_delta_max": _VERDICT_ALIGNED_REL_MAX,
                "diverges_relative_delta_min": _VERDICT_DIVERGE_REL_MIN,
            },
        }

    def _compute_shoulder_saturation_blast_radius(self, days_back: int) -> dict:
        """Quantify saturation-clamp blast radius on shoulder buckets (#928).

        For each shoulder hour (``temp ∈ [BP-3, BP+1]``) over the last
        ``days_back`` days, compute four whole-house "expected" variants
        and the absolute residual against ``actual_kwh``:

          - ``expected_3d_clamped``    — logged ``expected_kwh``
                                         (status quo; per-unit clamped)
          - ``expected_3d_unclamped``  — ``expected_kwh - solar_wasted_kwh``
                                         (whole-house proxy; can be negative)
          - ``expected_4d_clamped``    — ``max(0, sum_base - solar_impact_4d_kwh)``
                                         where ``sum_base = expected_kwh + solar_impact_kwh``
          - ``expected_4d_unclamped``  — ``sum_base - solar_impact_4d_kwh``

        Stratifies into ``all_shoulder`` and ``saturation_events_only``
        (hours where 3D clamp fired, i.e. ``solar_wasted_kwh > 0``) and
        reports median |residual| per variant per group.

        4D source ladder per hour: prefer logged ``solar_impact_4d_kwh``
        from the live shadow learner; otherwise post-mortem replay via
        ``_replay_4d_solar_total`` from logged DNI/DHI (or synthetic
        Kasten when only ``cloud_coverage`` is available).  This keeps
        the 4D variants populated across the rollout window of the
        shadow learner instead of waiting K weeks for tagging coverage,
        AND keeps the verdict comparison fair (matched populations).

        Whole-house proxies — per-unit clamp behaviour cannot be exactly
        recovered from logged whole-house aggregates, but for AGGREGATE
        medians the proxy is faithful enough to decide whether the
        saturation discontinuity is the dominant driver of shoulder
        deviation (vs. Kasten cloud-bias driving over-estimated solar
        input in the first place).

        Strict diagnostic — reads only ``hourly_log``; writes nothing.
        Returns ``{"available": False, "reason": ...}`` when there are
        too few qualifying samples.
        """
        cutoff = (dt_util.now() - timedelta(days=days_back)).date().isoformat()
        bp = float(getattr(self.coordinator, "balance_point", 17.0) or 17.0)
        shoulder_lo = bp - 3.0
        shoulder_hi = bp + 1.0

        all_residuals: dict[str, list[float]] = {
            "expected_3d_clamped": [],
            "expected_3d_unclamped": [],
            "expected_4d_clamped": [],
            "expected_4d_unclamped": [],
        }
        sat_residuals: dict[str, list[float]] = {
            "expected_3d_clamped": [],
            "expected_3d_unclamped": [],
            "expected_4d_clamped": [],
            "expected_4d_unclamped": [],
        }
        n_shoulder = 0
        n_saturation = 0
        n_4d_tagged = 0
        n_4d_replayed = 0
        n_4d_unavailable = 0

        for entry in self.coordinator._hourly_log:
            ts = entry.get("timestamp", "")
            if ts[:10] < cutoff:
                continue
            temp = entry.get("temp")
            actual = entry.get("actual_kwh")
            expected_3d = entry.get("expected_kwh")
            solar_eff = entry.get("solar_impact_kwh")
            solar_wasted = entry.get("solar_wasted_kwh", 0.0)
            if temp is None or actual is None or expected_3d is None or solar_eff is None:
                continue
            if not (shoulder_lo <= float(temp) <= shoulder_hi):
                continue

            n_shoulder += 1
            sum_base = float(expected_3d) + float(solar_eff)
            saturation_fired = float(solar_wasted) > ENERGY_GUARD_THRESHOLD
            if saturation_fired:
                n_saturation += 1

            variants: dict[str, float] = {
                "expected_3d_clamped": float(expected_3d),
                "expected_3d_unclamped": float(expected_3d) - float(solar_wasted),
            }

            solar_4d_logged = entry.get("solar_impact_4d_kwh")
            solar_4d_value: float | None = None
            if solar_4d_logged is not None:
                solar_4d_value = float(solar_4d_logged)
                n_4d_tagged += 1
            else:
                replay = self._replay_4d_solar_total(entry)
                if replay is not None:
                    solar_4d_value = replay
                    n_4d_replayed += 1
                else:
                    n_4d_unavailable += 1

            if solar_4d_value is not None:
                e4_unclamped = sum_base - solar_4d_value
                variants["expected_4d_unclamped"] = e4_unclamped
                variants["expected_4d_clamped"] = max(0.0, e4_unclamped)

            for name, exp in variants.items():
                resid = abs(float(actual) - exp)
                all_residuals[name].append(resid)
                if saturation_fired:
                    sat_residuals[name].append(resid)

        if n_shoulder < 5:
            return {
                "available": False,
                "reason": "no_shoulder_hours",
                "n_shoulder_hours": n_shoulder,
                "days_back": days_back,
                "balance_point": bp,
                "shoulder_window": [shoulder_lo, shoulder_hi],
            }

        def _median(values: list[float]) -> float:
            n_v = len(values)
            if n_v == 0:
                return 0.0
            s = sorted(values)
            mid = n_v // 2
            if n_v % 2 == 1:
                return s[mid]
            return (s[mid - 1] + s[mid]) / 2.0

        def _med_block(group: dict[str, list[float]]) -> dict:
            return {
                k: round(_median(v), 4) if v else None
                for k, v in group.items()
            }

        median_abs = {
            "all_shoulder": _med_block(all_residuals),
            "saturation_events_only": _med_block(sat_residuals),
        }

        # Verdict ref must be matched to whatever 4D-coverage population
        # we have; with replay populating most/all hours the matched-3D
        # baseline is also computed on (almost) all hours, but if some
        # entries were skipped (no DNI/DHI/cloud), we restrict the
        # comparison to the 4D-populated subset.
        verdict = "no_meaningful_difference"
        sat_4d_count = len(sat_residuals["expected_4d_clamped"])
        sat_3d_count = len(sat_residuals["expected_3d_clamped"])
        if sat_4d_count > 0 and sat_4d_count < sat_3d_count:
            # Partial 4D coverage — recompute matched 3D ref on the same
            # subset by walking again.  Cheap; we already filtered the
            # populations, but did not retain the mapping back to which
            # hour's 3D residual corresponds to a 4D-populated hour.
            # Rather than tracking that mapping, rerun the inner walk
            # to gather a matched 3D baseline.
            matched_3d: list[float] = []
            for entry in self.coordinator._hourly_log:
                ts = entry.get("timestamp", "")
                if ts[:10] < cutoff:
                    continue
                temp = entry.get("temp")
                actual = entry.get("actual_kwh")
                expected_3d = entry.get("expected_kwh")
                solar_wasted = entry.get("solar_wasted_kwh", 0.0)
                if temp is None or actual is None or expected_3d is None:
                    continue
                if not (shoulder_lo <= float(temp) <= shoulder_hi):
                    continue
                if not (float(solar_wasted) > ENERGY_GUARD_THRESHOLD):
                    continue
                # Same 4D-availability check as the main loop.
                if entry.get("solar_impact_4d_kwh") is None:
                    if self._replay_4d_solar_total(entry) is None:
                        continue
                matched_3d.append(abs(float(actual) - float(expected_3d)))
            ref = _median(matched_3d) if matched_3d else None
        else:
            ref = median_abs["saturation_events_only"].get("expected_3d_clamped")

        sat_med = median_abs["saturation_events_only"]
        if ref is not None and ref > ENERGY_GUARD_THRESHOLD:
            e4u = sat_med.get("expected_4d_unclamped")
            e3u = sat_med.get("expected_3d_unclamped")
            improvements: list[tuple[str, float]] = []
            if e4u is not None:
                improvements.append(("4d_unclamped_meaningfully_better",
                                     (ref - e4u) / ref))
            if e3u is not None:
                improvements.append(("soft_saturation_alone_meaningfully_better",
                                     (ref - e3u) / ref))
            improvements.sort(key=lambda kv: kv[1], reverse=True)
            if improvements and improvements[0][1] >= 0.20:
                verdict = improvements[0][0]

        return {
            "available": True,
            "days_back": days_back,
            "balance_point": bp,
            "shoulder_window": [shoulder_lo, shoulder_hi],
            "n_shoulder_hours": n_shoulder,
            "n_saturation_events": n_saturation,
            "n_4d_tagged_hours": n_4d_tagged,
            "n_4d_replayed_hours": n_4d_replayed,
            "n_4d_unavailable_hours": n_4d_unavailable,
            "saturation_event_share": (
                round(n_saturation / n_shoulder, 3) if n_shoulder else 0.0
            ),
            "median_abs_residual": median_abs,
            "matched_3d_ref_kwh": (
                round(ref, 4) if ref is not None else None
            ),
            "verdict": verdict,
        }

