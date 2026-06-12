"""Learning Manager Service."""
from __future__ import annotations

import bisect
import logging
import math
import statistics
from types import SimpleNamespace
from typing import Callable

from .const import (
    BATCH_FIT_DAMPING,
    BATCH_FIT_SATURATION_RATIO,
    COLD_START_SOLAR_DAMPING,
    COOLING_WIND_BUCKET,
    ENERGY_GUARD_THRESHOLD,
    INEQUALITY_MARGIN,
    INEQUALITY_STEP_SIZE,
    LEARNING_BUFFER_THRESHOLD,
    MODE_HEATING,
    MODE_COOLING,
    MODE_OFF,
    MODE_GUEST_HEATING,
    MODE_GUEST_COOLING,
    MODE_DHW,
    MODES_EXCLUDED_FROM_GLOBAL_LEARNING,
    NLMS_REGULARIZATION,
    NLMS_STEP_SIZE,
    PER_UNIT_LEARNING_RATE_CAP,
    GLOBAL_BASE_SATURATION_SKIP_KWH,
    SNR_WEIGHT_FLOOR,
    SNR_WEIGHT_K,
    SOLAR_BATTERY_DECAY,
    SOLAR_COEFF_CAP,
    SOLAR_DEAD_ZONE_THRESHOLD,
    SOLAR_LEARNING_MIN_BASE,
    SOLAR_SHUTDOWN_MIN_BASE,
    SOLAR_SHUTDOWN_MIN_MAGNITUDE,
    OBSTRUCTION_FIT_LOW_MIN_ELEV_DEG,
    OBSTRUCTION_FIT_LOW_MAX_ELEV_DEG,
    OBSTRUCTION_FIT_HIGH_MIN_ELEV_DEG,
    OBSTRUCTION_FIT_HIGH_MAX_ELEV_DEG,
    OBSTRUCTION_FIT_STEP_DEG,
    OBSTRUCTION_FIT_MIN_SAMPLES_PER_SIDE,
    OBSTRUCTION_FIT_SSE_IMPROVEMENT_THRESHOLD,
    OBSTRUCTION_FIT_DOMINANCE_RATIO,
    OBSTRUCTION_FIT_MIN_RMS_REDUCTION_KWH,
    OBSTRUCTION_FIT_MIN_WINDOW_DEG,
    OBSTRUCTION_FIT_SHUTDOWN_CONSTRAINT_MIN_N,
    OBSTRUCTION_FIT_WINDOW_K_PER_SIDE,
    OBSTRUCTION_FIT_WINDOW_MAX_ANGULAR_DEG,
    OBSTRUCTION_LOW_PLAUSIBLE_RANGE,
    OBSTRUCTION_HIGH_PLAUSIBLE_RANGE,
    OBSTRUCTION_FIT_MIN_COOLING_SAMPLES_FOR_HIGH,
    TOBIT_CONV_TOL,
    TOBIT_MAX_ITER,
    TOBIT_MIN_NEFF,
    TOBIT_MIN_UNCENSORED,
    TOBIT_Q_CLIP,
    OUTLIER_RESIDUAL_WINDOW,
    OUTLIER_K_THRESHOLD,
    OUTLIER_MIN_SAMPLES,
    OUTLIER_REJECTED_POOL_SIZE,
    OUTLIER_PROMOTION_THRESHOLD,
    HARD_OUTLIER_CAP_FACTOR,
    HARD_OUTLIER_SANITY_MULTIPLIER,
)
from .helpers import solve_gauss_jordan
from .observation import HourlyObservation, ModelState, LearningConfig
from .solar import SolarCalculator

_LOGGER = logging.getLogger(__name__)


# Obstruction apply-gate (#1020).  Replaces the cross-boundary
# OBSTRUCTION_SKIP_EXPLORED / OBSTRUCTION_SKIP_BLIND tables introduced
# in #1016.  Rationale: under the tightened primary thresholds
# (0.30 SSE + physical-plausibility prior + multi-window stability),
# the primary validation is strong enough that the opposite-side
# evidence requirement becomes redundant.  It also incorrectly
# blocked the dominant northern-latitude case of single-sided LOW
# gates — heating-only data at 60°N structurally cannot populate the
# HIGH regime, so the opposite-side requirement permanently locked
# those installs out.  New rule: four local checks per boundary, no
# cross-boundary lookup.
def _within_plausibility_range(value: float | None, side: str) -> bool:
    """Return True when ``value`` sits inside the configured plausibility
    range for ``side`` (``"low"`` or ``"high"``).  ``None`` → False
    (no value to evaluate)."""
    if value is None:
        return False
    if side == "low":
        lo, hi = OBSTRUCTION_LOW_PLAUSIBLE_RANGE
    elif side == "high":
        lo, hi = OBSTRUCTION_HIGH_PLAUSIBLE_RANGE
    else:
        return False
    return lo <= value <= hi


def _boundary_applicable(
    boundary: dict,
    side: str,
    *,
    stable_across_windows: bool = True,
) -> tuple[bool, str | None]:
    """Apply-gate classification for one obstruction boundary (#1020).

    Four independent local checks; no opposite-side dependency:
      1. ``learned``: data produced a fit.
      2. SSE improvement ≥ ``OBSTRUCTION_FIT_SSE_IMPROVEMENT_THRESHOLD``.
      3. ``best_critical_elev`` inside the plausibility range for the
         side (LOW: ``[2°, 20°]``, HIGH: ``[20°, 60°]``).
      4. ``stable_across_windows``: same boundary appears in
         ≥ ``OBSTRUCTION_STABILITY_MIN_AGREEING`` of the orchestrated
         time windows (default 30/60/90 d) within ±3°.  Single-window
         callers (tests, ad-hoc debug runs) pass ``True`` to bypass.

    Returns ``(applicable, applicable_reason)``.  ``applicable_reason``
    is the first failing check or ``None`` when all pass.  Stability
    is the LAST gate so that the diagnostic clearly distinguishes
    "fit is bad" (SSE / plausibility) from "fit is good but unstable
    across windows" (single_window_only).
    """
    if not boundary.get("learned"):
        return False, "not_learned"
    sse = boundary.get("sse_improvement_ratio") or 0.0
    if sse < OBSTRUCTION_FIT_SSE_IMPROVEMENT_THRESHOLD:
        return False, "below_sse_threshold"
    if not _within_plausibility_range(
        boundary.get("best_critical_elev"), side,
    ):
        return False, "physically_implausible"
    if not stable_across_windows:
        return False, "single_window_only"
    return True, None


def _filter_log_by_days_back(
    hourly_log: list[dict], days_back: int | None
) -> list[dict]:
    """Filter ``hourly_log`` to entries within the last ``days_back`` days.

    ``days_back`` ``None`` or ``<= 0`` falls through to "no filtering"
    (returns the input list reference unchanged) for backwards compat
    with callers that pass missing-or-zero values.

    Compares parsed ``datetime`` objects, NOT ISO strings.  Lex
    comparison of ISO strings with different tz-offsets is not
    chronologically equivalent: e.g. ``"2026-03-27T11:00:00+02:00"``
    (= 09:00 UTC) lex-compares greater than
    ``"2026-03-27T10:33:53+00:00"`` even though the first is
    chronologically earlier.  Production log timestamps may carry
    non-UTC offsets depending on how they were written; parsing both
    sides removes the ambiguity.

    Naive timestamps in the log (legacy entries without an offset)
    are interpreted as UTC for comparison purposes.  Entries that
    fail to parse are dropped silently — they would have been dropped
    by the downstream sample collector anyway.
    """
    if days_back is None or days_back <= 0:
        return hourly_log
    from datetime import timedelta
    from homeassistant.util import dt as dt_util

    cutoff_dt = dt_util.utcnow() - timedelta(days=days_back)
    filtered: list[dict] = []
    for entry in hourly_log:
        ts = entry.get("timestamp")
        if not ts:
            continue
        parsed = dt_util.parse_datetime(ts)
        if parsed is None:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt_util.UTC)
        if parsed >= cutoff_dt:
            filtered.append(entry)
    return filtered


def _pad_solar_vector(v: tuple) -> tuple[float, float, float]:
    """Ensure a solar vector is a 3-tuple (S, E, W).

    Legacy callers may pass a 2-tuple (S, E); pad with W=0.0.
    """
    if len(v) >= 3:
        return (v[0], v[1], v[2])
    return (v[0], v[1], 0.0)


def _solar_coeff_regime(unit_mode: str) -> str | None:
    """Map a unit operating mode to the solar coefficient regime key.

    Returns ``"heating"`` for ``MODE_HEATING`` / ``MODE_GUEST_HEATING``,
    ``"cooling"`` for ``MODE_COOLING`` / ``MODE_GUEST_COOLING``, and
    ``None`` for modes that don't learn solar (``MODE_OFF``,
    ``MODE_DHW``, unknown).

    Used by all per-unit solar paths (NLMS, cold-start, inequality,
    prediction) to route reads and writes to the correct
    ``solar_coefficients_per_unit[entity][regime]`` slot.

    Heating and cooling regimes converge to physically distinct
    coefficients: ``phys_coeff × E[1/COP_mode]``.  Mixing modes into one
    coefficient (pre-#868 behaviour) produced a weighted mean of the
    two, biasing whichever mode dominated by the COP ratio.
    """
    if unit_mode in (MODE_HEATING, MODE_GUEST_HEATING):
        return "heating"
    if unit_mode in (MODE_COOLING, MODE_GUEST_COOLING):
        return "cooling"
    return None


def _resolve_min_base(
    entity_id: str,
    unit_min_base: dict[str, float] | None,
    fallback: float,
) -> float:
    """Return the effective min-base threshold for one unit (#871).

    Per-unit overrides take precedence over the global fallback.  Empty
    dict / None preserves legacy behaviour.  Centralises the resolution
    so NLMS, inequality, and shutdown gates stay in lock-step.

    Non-dict inputs (including MagicMock in tests and any malformed
    storage load) fall back to the global constant rather than raising,
    mirroring the defensive posture of the storage-load path.
    """
    if not isinstance(unit_min_base, dict) or not unit_min_base:
        return fallback
    v = unit_min_base.get(entity_id)
    if isinstance(v, (int, float)) and v > 0.0:
        return float(v)
    return fallback


def compute_snr_weight(
    solar_factor: float,
    solar_dominant_entities,
    total_units: int,
    *,
    floor: float = SNR_WEIGHT_FLOOR,
    k: float = SNR_WEIGHT_K,
) -> float:
    """Signal-to-noise weight for base-model EMA (#866).

    Exogenous weight — a function of observed sun geometry and shutdown
    flags only, never of a learned state.  The weighted EMA converges to
    ``E[w·Y] / E[w]``, NOT to ``E[Y]``: because ``w`` depends on
    ``solar_factor`` and ``E[Y | solar_factor]`` is non-constant (sunny
    hours have systematically lower electrical heating load), the
    stationary point is *dark-equivalent* expected demand, not the
    arithmetic mean of observed demand.  This is the desired semantic for
    a base bucket consumed by `predict − solar_impact` downstream.  Do
    not call this estimator "unbiased for E[Y]" — it isn't, by design.

    Shape:
        w = max(floor, 1 − k × solar_factor) × (n_clean / n_total)

    - Dark hour (solar_factor ≈ 0) → weight ≈ 1.0
    - Sunny hour decays linearly toward ``floor`` as solar_factor grows
    - Per-unit shutdown scaling: fraction of units NOT in the hour's
      ``solar_dominant_entities`` list.  All-shutdown → 0.  Prevents
      a single shut-down unit from zeroing base learning for the whole
      hour when other units are still observable (the over-aggressive
      global w=0 flagged by the physics analysis in #862).

    Parameters
    ----------
    solar_factor:
        Hour's Kasten cloud-corrected geometric solar factor, in [0, 1].
    solar_dominant_entities:
        Iterable of entity_ids flagged as shutdown for this hour (from
        :func:`detect_solar_shutdown_entities`).  May be None/empty.
    total_units:
        Count of learnable energy sensors for this hour.  Used to compute
        the clean fraction.  Zero or negative → unit-scaling disabled.
    floor, k:
        Weight-function parameters (defaults from const.py).
    """
    sf = max(0.0, float(solar_factor or 0.0))
    w = max(floor, 1.0 - k * sf)
    if total_units > 0 and solar_dominant_entities:
        n_shutdown = len(solar_dominant_entities)
        n_clean = max(0, total_units - n_shutdown)
        if n_clean == 0:
            return 0.0
        w *= n_clean / total_units
    return w


def count_active_learnable_units(
    energy_sensors,
    unit_modes: dict,
    expected_base_per_unit: dict,
    *,
    min_base: float = SOLAR_LEARNING_MIN_BASE,
    unit_min_base: dict[str, float] | None = None,
) -> int:
    """Count units that *could* be observable as a base-learning signal this hour.

    Used as the denominator for the SNR weight's per-unit shutdown
    scaling.  Without this, an hour where all VPs are in shutdown but
    several small loads (termostat / varmekabel / socket) are also
    configured under ``energy_sensors`` would still get a non-zero
    weight because the denominator sees the small loads too.  But the
    small loads are not the signal-bearers — the VPs are.  When the
    VPs shut down, the hour is genuinely contaminated and the base
    EMA should freeze.

    Active = heating-mode + ``expected_base ≥ min_base``.  Cooling, OFF,
    DHW, and guest modes are excluded (they don't contribute to global
    base learning anyway).  Units below ``SOLAR_LEARNING_MIN_BASE``
    are excluded because they cycle below the level where shutdown
    detection meaningfully applies.

    Falls back to ``len(energy_sensors)`` when called with empty
    expected-base mapping (e.g. cold-start before any base data exists)
    so existing call sites don't change behaviour during initial
    warm-up.
    """
    if not expected_base_per_unit:
        return len(energy_sensors) if energy_sensors else 0
    n = 0
    for sid in energy_sensors:
        mode = unit_modes.get(sid, MODE_HEATING)
        if mode != MODE_HEATING:
            continue
        threshold = _resolve_min_base(sid, unit_min_base, min_base)
        if expected_base_per_unit.get(sid, 0.0) >= threshold:
            n += 1
    # Defensive: if nothing matches, fall back to total count rather than
    # zero out the SNR denominator.
    return n if n > 0 else (len(energy_sensors) if energy_sensors else 0)


class LearningManager:
    """Manages machine learning logic for Heating Analytics."""

    def __init__(self):
        # Dead zone counter: tracks consecutive qualifying hours where
        # actual_impact is clamped to 0 despite sun shining.  When a unit
        # is stuck (base model too low → no learnable signal), the counter
        # triggers a coefficient reset so cold-start can re-learn.
        self._dead_zone_counts: dict[str, int] = {}
        # Outlier state (#919): per-unit robust residual window for filtering.
        # Maps (entity_id) -> (regime, is_shutdown) -> {"baseline": [], "rejected": []}
        self._outlier_state: dict[str, dict[tuple[str, bool], dict[str, list[float]]]] = {}

    def reset_outlier_state(self, entity_id: str | None = None) -> None:
        """Clear the MAD outlier state for a specific entity or all entities (#919).
        
        Called when solar learning is reset so the new coefficient does not get
        evaluated against the baseline noise profile of the old setup.
        """
        if entity_id:
            if entity_id in self._outlier_state:
                del self._outlier_state[entity_id]
                _LOGGER.debug("Cleared outlier state for %s", entity_id)
        else:
            self._outlier_state.clear()
            _LOGGER.debug("Cleared outlier state for all entities")

    def _is_outlier_residual(
        self,
        entity_id: str,
        regime: str,
        residual: float,
        outlier_state: dict,
        is_shutdown: bool = False,
        actual: float = 0.0,
        expected_base: float = 0.0,
        unit_min_base: dict[str, float] | None = None,
    ) -> bool:
        """Check if a residual is an outlier based on robust history (#919).

        Maintains a per-unit sliding window of recent residuals and calculates
        the Median Absolute Deviation (MAD) as a robust estimator of scale.
        Samples where |residual| > k * sigma_robust are flagged as outliers.

        We use 1.4826 * MAD as a robust estimator of the standard deviation
        (sigma) for normally distributed data.

        Returns True if the residual is an outlier, False otherwise.
        NOTE: When a genuine shift is promoted to baseline, this method
        returns False to allow learning, but the sample is ALREADY
        appended to the new baseline internally.
        """
        # Part 1 & 2: Use (regime, is_shutdown) as the key to separate 
        # modulating vs shutdown behavior.
        entity_outlier = outlier_state.setdefault(entity_id, {})
        key = (regime, is_shutdown)
        regime_state = entity_outlier.setdefault(key, {"baseline": [], "rejected": []})
        baseline = regime_state["baseline"]
        rejected = regime_state["rejected"]

        # Part 5: Prior-free sanity check (guards against extreme glitches before 
        # baseline is even formed).
        if expected_base > 0 and abs(actual - expected_base) > HARD_OUTLIER_SANITY_MULTIPLIER * expected_base:
            _LOGGER.warning(
                "Hard outlier detected for %s [%s, shutdown=%s]: |act - exp|=%.3f > %.1f * exp(%.3f). Prior-free rejection.",
                entity_id, regime, is_shutdown, abs(actual - expected_base), HARD_OUTLIER_SANITY_MULTIPLIER, expected_base
            )
            return True

        # Check if we have enough samples to filter
        n = len(baseline)
        is_outlier = False
        if n >= OUTLIER_MIN_SAMPLES:
            median_res = statistics.median(baseline)
            
            # Absolute deviations from median
            abs_devs = [abs(r - median_res) for r in baseline]
            mad = statistics.median(abs_devs)
            
            # Part 4: Per-entity sigma floor.
            # max(0.02, 0.05 * max(unit_min_base.get(entity, 0.5), 0.1))
            min_base_val = _resolve_min_base(entity_id, unit_min_base, 0.5)
            sigma_floor = max(0.02, 0.05 * max(min_base_val, 0.1))
            sigma_robust = max(sigma_floor, 1.4826 * mad)
            
            # Compare distance from the robust center (median) rather than zero (#919).
            if abs(residual - median_res) > OUTLIER_K_THRESHOLD * sigma_robust:
                is_outlier = True
                _LOGGER.warning(
                    "Outlier detected for %s [%s, shutdown=%s]: |res - median|=%.3f > %.1f * sigma_robust(%.3f). Skipping learning.",
                    entity_id, regime, is_shutdown, abs(residual - median_res), OUTLIER_K_THRESHOLD, sigma_robust
                )
        else:
            # Part 3: Warm-up hard cap.
            # While baseline is forming, reject residuals > 20 kWh (HARD_OUTLIER_CAP_FACTOR).
            if abs(residual) > HARD_OUTLIER_CAP_FACTOR:
                is_outlier = True
                _LOGGER.warning(
                    "Warm-up outlier detected for %s [%s, shutdown=%s]: |residual|=%.3f > %.1f. Skipping learning.",
                    entity_id, regime, is_shutdown, abs(residual), HARD_OUTLIER_CAP_FACTOR
                )

        # Append to window regardless
        if not is_outlier:
            baseline.append(residual)
            if len(baseline) > OUTLIER_RESIDUAL_WINDOW:
                baseline.pop(0)
            # If we just accepted a sample, clear the rejected pool (it wasn't a shift)
            rejected.clear()
        else:
            # Part 2: Recovery from genuine shift (rejected pool promotion)
            rejected.append(residual)
            if len(rejected) >= OUTLIER_PROMOTION_THRESHOLD:
                # If enough consecutive rejected samples are consistent with 
                # EACH OTHER, promote them to baseline.
                def get_mad(data: list[float]) -> float:
                    if not data: return 0.0
                    med = statistics.median(data)
                    return statistics.median([abs(x - med) for x in data])

                mad_rejected = get_mad(rejected)
                # Use current baseline MAD if available, otherwise 0.5 as fallback.
                # Use a small floor (0.01) to handle pure-zero baselines (0 < 2*0 is False).
                mad_baseline = max(0.01, get_mad(baseline) if baseline else 0.5)
                
                if mad_rejected < 2 * mad_baseline:
                    _LOGGER.info(
                        "Genuine shift detected for %s [%s, shutdown=%s]: promoting %d rejected samples to baseline.",
                        entity_id, regime, is_shutdown, len(rejected)
                    )
                    baseline.clear()
                    baseline.extend(rejected)
                    rejected.clear()
                    # Allow current sample to be learned
                    return False

            if len(rejected) > OUTLIER_REJECTED_POOL_SIZE:
                rejected.pop(0)
                
        return is_outlier

    @staticmethod
    def _compute_4d_normalization_delta(
        obs,
        config,
        solar_coefficients_4d_per_unit: dict | None,
    ) -> float | None:
        """Pre-compute the 4D solar normalization delta for this hour (#962).

        Mirrors the per-entity saturation that
        ``calculate_total_power`` performs (and that
        ``hourly_processor`` consumes via
        ``solar_heating_applied_kwh − solar_cooling_applied_kwh`` to
        produce the canonical 3D ``solar_normalization_delta``).

        Returns the signed sum

            Σ_(heating) min(c_4d · pot_4d, base_e) − Σ_(cooling) c_4d · pot_4d

        across qualifying entities, where ``base_e`` is
        ``obs.unit_expected_base[entity_id]`` (per-entity dark-sky base
        prediction).  Heating-mode saturation prevents 4D-solar
        overshoot on a single entity from inflating the global delta
        beyond what the entity's actual base could absorb; cooling-mode
        is additive (matches ``calculate_saturation`` semantics).
        Returns ``None`` when the 4D path is not exercisable
        (missing state, sun below horizon, DNI/DHI unresolvable, no
        qualifying entities).  The caller decides what to substitute
        when None — under ``experimental_4d_primary=True``, the read
        site should fall back to ``obs.solar_normalization_delta`` (3D)
        so the global base path always has a delta to consume.

        Pre-#992 c3 this returned the RAW signed sum without per-entity
        saturation.  On installs running with ``experimental_4d_primary
        = True``, a single south-facing entity with strong 4D potential
        could push the delta past its own base; the inflated delta then
        replaced the 3D-saturated value at ``learning.py:701``, biasing
        the base-EMA target upward (``normalized_actual = max(0,
        total_energy_kwh + solar_normalization_delta)``) on every sunny
        hour.  See issue #992 violation 4.

        Outlier filtering and Tobit / NLMS gating are intentionally
        NOT replicated here — those are learner-internal state
        machines.  This helper is the read-side accumulation only.
        """
        if (
            solar_coefficients_4d_per_unit is None
            or obs is None
            or config is None
            or config.solar_calculator is None
            or getattr(obs, "timestamp", None) is None
            or not config.solar_enabled
        ):
            return None

        from datetime import timedelta as _td
        from .solar import resolve_dni_dhi

        try:
            mid_dt = obs.timestamp + _td(minutes=30)
            sun_elev, sun_az = config.solar_calculator.get_approx_sun_pos(mid_dt)
        except Exception:  # pragma: no cover — defensive against mocks
            return None
        if not (sun_elev and sun_elev > 0):
            return None

        try:
            day_of_year = obs.timestamp.timetuple().tm_yday
            dni, dhi, _src = resolve_dni_dhi(
                getattr(obs, "dni_avg", None),
                getattr(obs, "dhi_avg", None),
                getattr(obs, "ghi_avg", None),
                getattr(obs, "cloud_avg", None),
                sun_elev,
                day_of_year,
            )
        except Exception:  # pragma: no cover
            return None
        if not (dni or dhi):
            return None

        screen_config = config.screen_config
        screen_affected = config.screen_affected_entities
        solar_affected = config.solar_affected_entities
        correction_percent = obs.correction_percent
        unit_modes = obs.unit_modes or {}
        hourly_delta_per_unit = obs.unit_breakdown or {}
        # Per-entity dark-sky base prediction for saturation gate (#992
        # c3).  When absent (e.g. cold-start before per-entity buckets
        # populated), default to 0 — the heating-side ``min(predicted,
        # base=0) = 0`` then correctly contributes nothing, matching
        # the 3D path where empty buckets also produce no solar effect.
        unit_expected_base = getattr(obs, "unit_expected_base", None) or {}

        delta_4d = 0.0
        any_qualified = False
        for entity_id in config.energy_sensors:
            unit_mode = unit_modes.get(entity_id, MODE_HEATING)
            if unit_mode in (MODE_OFF, MODE_GUEST_HEATING, MODE_GUEST_COOLING):
                continue
            # Match _process_per_unit_learning: skip entity if sensor
            # was offline this hour (matches actual_unit assignment
            # logic; DHW is treated as actual=0 there, so DHW is
            # NOT skipped here — but DHW's regime is None anyway).
            if unit_mode != MODE_DHW and entity_id not in hourly_delta_per_unit:
                continue
            regime_4d = _solar_coeff_regime(unit_mode)
            if regime_4d is None:
                continue
            if solar_affected is not None and entity_id not in solar_affected:
                continue

            entity_screen_config = (
                screen_config
                if screen_affected is None or entity_id in screen_affected
                else (False, False, False)
            )
            try:
                p4d = config.solar_calculator.calculate_unit_potential_4d(
                    entity_id, dni, dhi, sun_elev, sun_az,
                    entity_screen_config, correction_percent,
                )
            except Exception:  # pragma: no cover
                continue

            entity_coeffs_4d = solar_coefficients_4d_per_unit.get(entity_id, {}) or {}
            if not isinstance(entity_coeffs_4d, dict):
                continue
            regime_coeffs_4d = entity_coeffs_4d.get(regime_4d, {}) or {}
            if not isinstance(regime_coeffs_4d, dict):
                continue
            c_s = float(regime_coeffs_4d.get("s", 0.0))
            c_e = float(regime_coeffs_4d.get("e", 0.0))
            c_w = float(regime_coeffs_4d.get("w", 0.0))
            c_d = float(regime_coeffs_4d.get("diffuse", 0.0))
            try:
                p_s, p_e, p_w, p_d = p4d
            except Exception:  # pragma: no cover
                continue
            predicted_4d = c_s * p_s + c_e * p_e + c_w * p_w + c_d * p_d
            # Per-entity saturation via canonical dispatch (#1008).
            entity_base = (
                float(unit_expected_base.get(entity_id, 0.0))
                if regime_4d == "heating"
                else 0.0
            )
            applied_4d, _wasted_4d, _net_4d = config.solar_calculator.calculate_saturation(
                entity_base, predicted_4d, unit_mode
            )
            delta_4d += applied_4d if regime_4d == "heating" else -applied_4d
            any_qualified = True

        if not any_qualified:
            return None
        return delta_4d

    @staticmethod
    def _compute_4d_solar_factor_for_snr(obs, config) -> float | None:
        """Compute the 4D-anchored solar_factor for SNR weighting (#981).

        SNR weighting on the global base bucket uses ``obs.solar_factor``,
        which is Kasten-from-cloud_coverage.  When 4D is promoted to the
        live read-path, cloud_coverage can disagree with native DNI/DHI on
        broken-cloud hours — Kasten-misclassified clear hours then get
        over-weighted into the EMA and pull the bucket down.  This helper
        rebuilds solar_factor with the same DNI/DHI signal the 4D read-
        path consumes, by passing a ladder-derived cloud-transmission
        override into ``calculate_solar_factor``.

        Returns ``None`` when the 4D path is not exercisable for this
        hour; the caller falls back to ``obs.solar_factor`` (3D).  On the
        ``kasten_synthetic`` ladder branch the result is bit-identical to
        the 3D path by construction.
        """
        if (
            obs is None
            or config is None
            or config.solar_calculator is None
            or getattr(obs, "timestamp", None) is None
            or not config.solar_enabled
        ):
            return None

        from datetime import timedelta as _td
        from .solar import resolve_dni_dhi, _eccentricity

        try:
            mid_dt = obs.timestamp + _td(minutes=30)
            sun_elev, sun_az = config.solar_calculator.get_approx_sun_pos(mid_dt)
        except Exception:  # pragma: no cover — defensive against mocks
            return None
        if not (sun_elev and sun_elev > 0):
            return 0.0

        try:
            day_of_year = obs.timestamp.timetuple().tm_yday
            dni, dhi, _src = resolve_dni_dhi(
                getattr(obs, "dni_avg", None),
                getattr(obs, "dhi_avg", None),
                getattr(obs, "ghi_avg", None),
                getattr(obs, "cloud_avg", None),
                sun_elev,
                day_of_year,
            )
        except Exception:  # pragma: no cover
            return None
        if not (dni or dhi):
            return None

        sin_elev = math.sin(math.radians(sun_elev))
        if sin_elev <= 0:
            return 0.0
        air_mass = 1.0 / sin_elev
        ghi_clear = 1367.0 * _eccentricity(day_of_year) * sin_elev * (0.7 ** air_mass)
        if ghi_clear <= 0:
            return None
        cloud_trans_4d = max(0.0, min(1.0, (dni * sin_elev + dhi) / ghi_clear))

        try:
            return config.solar_calculator.calculate_solar_factor(
                sun_elev, sun_az, 0.0,
                cloud_attenuation_override=cloud_trans_4d,
            )
        except Exception:  # pragma: no cover
            return None

    def process_learning(
        self,
        obs: HourlyObservation | None = None,
        model: ModelState | None = None,
        config: LearningConfig | None = None,
        **kwargs,
    ) -> dict:
        """Process learning for the completed hour.

        Accepts either the new contract-based interface (obs, model, config)
        or legacy keyword arguments for backward compatibility with tests.

        Returns a dictionary with learning results for logging:
        {
            "model_updated": bool,
            "model_base_before": float,
            "model_base_after": float,
            "aux_model_updated": bool,
            "aux_model_before": float,
            "aux_model_after": float,
            "learning_status": str
        }
        """
        if kwargs:
            # Legacy call path — build contracts from kwargs
            return self._process_learning_legacy(**kwargs)

        # --- Destructure data contracts into local variables ---
        # This preserves all existing code below without changes.
        # Once internal methods are migrated to use obs/model/config
        # directly, these locals can be removed.

        # From HourlyObservation
        temp_key = obs.temp_key
        wind_bucket = obs.wind_bucket
        avg_temp = obs.avg_temp
        total_energy_kwh = obs.learning_energy_kwh
        base_expected_kwh = obs.base_expected_kwh
        solar_impact = obs.effective_solar_impact
        avg_solar_vector = obs.solar_vector
        is_aux_active = obs.is_aux_dominant
        hourly_bucket_counts = obs.bucket_counts
        hourly_sample_count = obs.sample_count
        hourly_delta_per_unit = obs.unit_breakdown
        hourly_expected_per_unit = obs.unit_expected
        hourly_expected_base_per_unit = obs.unit_expected_base
        unit_modes = obs.unit_modes
        is_cooldown_active = obs.was_cooldown_active
        solar_normalization_delta = obs.solar_normalization_delta
        solar_factor = obs.solar_factor
        battery_filtered_potential = obs.battery_filtered_potential
        correction_percent = obs.correction_percent
        solar_dominant_entities = obs.solar_dominant_entities

        # From ModelState
        correlation_data = model.correlation_data
        correlation_data_per_unit = model.correlation_data_per_unit
        aux_coefficients = model.aux_coefficients
        learning_buffer_global = model.learning_buffer_global
        learning_buffer_per_unit = model.learning_buffer_per_unit
        observation_counts = model.observation_counts
        aux_coefficients_per_unit = model.aux_coefficients_per_unit
        learning_buffer_aux_per_unit = model.learning_buffer_aux_per_unit
        solar_coefficients_per_unit = model.solar_coefficients_per_unit
        learning_buffer_solar_per_unit = model.learning_buffer_solar_per_unit
        # 4D shadow learner state (#954).  Optional — None on legacy
        # ModelState construction sites; the 4D learner is a no-op when
        # either dict is None.
        solar_coefficients_4d_per_unit = getattr(
            model, "solar_coefficients_4d_per_unit", None
        )
        learning_buffer_solar_4d_per_unit = getattr(
            model, "learning_buffer_solar_4d_per_unit", None
        )

        # From LearningConfig
        learning_enabled = config.learning_enabled
        solar_enabled = config.solar_enabled
        learning_rate = config.learning_rate
        balance_point = config.balance_point
        energy_sensors = config.energy_sensors
        aux_impact = config.aux_impact
        solar_calculator = config.solar_calculator
        get_predicted_unit_base_fn = config.get_predicted_unit_base_fn
        aux_affected_entities = config.aux_affected_entities
        has_guest_activity = config.has_guest_activity
        per_unit_learning_enabled = config.per_unit_learning_enabled
        unit_min_base = config.unit_min_base  # #871: per-unit threshold overrides

        # Experimental 4D primary read-path (#962): when the flag is on,
        # the global base, per-#930 dark-equivalent lift, and aux paths
        # all consume the 4D-derived delta in place of the 3D one.
        # Falls back to the 3D delta when the 4D path is not
        # exercisable for this hour (missing state, sun below horizon,
        # etc.) so the global base block always has a delta to consume.
        if getattr(config, "experimental_4d_primary", False):
            delta_4d_pre = self._compute_4d_normalization_delta(
                obs, config, solar_coefficients_4d_per_unit,
            )
            if delta_4d_pre is not None:
                solar_normalization_delta = delta_4d_pre

        # --- Original logic (unchanged) ---

        if hourly_sample_count == 0:
            return {
                "model_updated": False,
                "model_base_before": base_expected_kwh,
                "model_base_after": base_expected_kwh,
                "aux_model_updated": False,
                "aux_model_before": None,
                "aux_model_after": None,
                "learning_status": "skipped_no_data",
            }

        model_base_before = base_expected_kwh
        model_base_after = base_expected_kwh
        model_updated = False

        aux_model_before = None
        aux_model_after = None
        aux_model_updated = False

        learning_status = "unknown"
        should_run_per_unit = False

        # Resolve per-unit override: when Track B/C owns the global model,
        # the coordinator passes learning_enabled=False but may allow per-unit
        # learning to continue for non-MPC units via per_unit_learning_enabled.
        _per_unit_enabled = per_unit_learning_enabled if per_unit_learning_enabled is not None else learning_enabled

        if not learning_enabled:
            if _per_unit_enabled:
                # Global blocked (Track B/C), per-unit allowed.
                model_updated = False
                should_run_per_unit = True
                learning_status = "disabled_global_only"
            else:
                learning_status = "disabled"
        elif is_cooldown_active:
            # COOLDOWN LOCK: Global Base Model is frozen.
            # Only non-affected units will learn (in per-unit step).
            model_updated = False
            should_run_per_unit = True # Allow per-unit (selective)
            learning_status = "cooldown_post_aux"
            _LOGGER.debug(f"Learning Cooldown Active: Global Base Model locked (T={temp_key} W={wind_bucket})")
        else:
            model_updated = True
            should_run_per_unit = True
            learning_status = "active"

        if model_updated:
            # --- Step 1: Learn Models ---

            # Saturation-aware solar normalization (#801).
            # The solar_normalization_delta is pre-computed from per-unit
            # saturation-aware solar impacts with correct mode signs:
            #   delta = heating_solar_applied - cooling_solar_applied
            # This eliminates the need for global_mode inference and correctly
            # handles simultaneous heating+cooling units.
            if solar_enabled:
                normalized_actual = max(0.0, total_energy_kwh + solar_normalization_delta)
            else:
                normalized_actual = total_energy_kwh

            if is_aux_active:
                # --- AUX ACTIVE MODE ---
                # Global Base Model is LOCKED. This is intentional:
                # 1. Global Base is the reference for calculating aux_impact
                # 2. Changing it mid-aux would create circular dependency
                # 3. Excluded per-unit models CAN still learn (see _process_per_unit_learning)
                #    but Global does NOT need to sync with them - they are independent
                # 4. Any difference between Global and sum-of-units is handled by
                #    the Aux Coefficient and built-in model inertia
                #
                # Only update Global Aux Coefficient here.

                # Guest Mode Guard: Skip aux learning if guest units are active
                # Guest consumption pollutes the actual energy, making aux learning inaccurate
                # Base and solar learning can continue as they use guest-excluded energy
                if has_guest_activity:
                    _LOGGER.info(f"Aux learning skipped due to guest mode activity (T={temp_key} W={wind_bucket})")
                    learning_status = "aux_skipped_guest_mode"
                    # Don't update aux model - keep current values
                    if temp_key in aux_coefficients and wind_bucket in aux_coefficients[temp_key]:
                        aux_model_before = aux_coefficients[temp_key][wind_bucket]
                        aux_model_after = aux_model_before
                    # Skip to per-unit learning
                else:
                    # Expected consumption IF Aux was contributing perfectly:
                    # Expected = Base - Aux_Impact
                    # Actual is normalized_actual.

                    # We want to learn Aux_Impact.
                    # Implied Aux Impact = Base - Actual.
                    # Example: Base=10, Actual=7 -> Implied Aux=3.
                    implied_aux_kw = base_expected_kwh - normalized_actual

                    # Capture Aux Before state
                    # Aux coefficients are now nested by wind bucket
                    if temp_key not in aux_coefficients:
                        aux_coefficients[temp_key] = {}

                    # Retrieve current coefficient for this wind bucket
                    if wind_bucket in aux_coefficients[temp_key]:
                        current_aux_coeff = aux_coefficients[temp_key][wind_bucket]
                    else:
                        # New bucket: Seed with fallback value (Normal/High) to avoid drop
                        # This prevents the "Forgot old category" bug where a new bucket starts at 0.0
                        current_aux_coeff = 0.0
                        bucket_data = aux_coefficients[temp_key]

                        if wind_bucket == "extreme_wind":
                            if "high_wind" in bucket_data:
                                current_aux_coeff = bucket_data["high_wind"]
                            elif "normal" in bucket_data:
                                current_aux_coeff = bucket_data["normal"]
                        elif wind_bucket == "high_wind":
                            if "normal" in bucket_data:
                                current_aux_coeff = bucket_data["normal"]

                    aux_model_before = current_aux_coeff

                    # Guard against noise
                    if abs(implied_aux_kw) > ENERGY_GUARD_THRESHOLD:
                        # Use provided global learning rate
                        new_aux_coeff = current_aux_coeff + learning_rate * (implied_aux_kw - current_aux_coeff)
                        # Clamp: Aux impact is always non-negative (auxiliary *heating* reduces consumption)
                        new_aux_coeff = max(0.0, new_aux_coeff)

                        # Update specific wind bucket
                        aux_coefficients[temp_key][wind_bucket] = round(new_aux_coeff, 3)

                        aux_model_after = aux_coefficients[temp_key][wind_bucket]
                        aux_model_updated = True

                        learning_status = f"active_aux_update ({new_aux_coeff:.2f}kW)"
                    else:
                        aux_model_after = current_aux_coeff
                        learning_status = "active_aux_no_change"

            else:
                # --- NORMAL MODE ---
                # Update Base Model.
                #
                # Target is raw ``total_energy_kwh`` (no delta compensation);
                # step size is scaled by the hour's signal-to-noise weight.
                # Aux-path above is unchanged — aux learning is a different
                # physical mechanism and does not share the COP-ceiling /
                # shutdown-contamination failure modes that motivated SNR.

                # BP-2 cold-cooling-regime shield (#885 follow-up): when
                # every active learnable unit is in cooling mode AND the
                # outdoor temp is below ``balance_point − 2``, the hour
                # reflects seasonal-misconfig idle (AC left on through
                # winter), not the installation's real heating-regime
                # baseline.  Skip the global base update so transient
                # misconfig periods do not contaminate cold-temp heating
                # buckets with compressor-standby values.  Mixed-mode
                # hours (any heating unit active) write normally — the
                # U-curve shape of ``correlation_data`` absorbs those as
                # legitimate building consumption.
                active_modes = [
                    m for m in unit_modes.values()
                    if m not in MODES_EXCLUDED_FROM_GLOBAL_LEARNING
                ]
                is_cold_cooling_regime = (
                    bool(active_modes)
                    and all(m == MODE_COOLING for m in active_modes)
                    and avg_temp < (balance_point - 2.0)
                )

                # One-sided dark-equivalent floor (#930) gated by per-regime
                # plausibility.  The lift target dark_target = actual + delta
                # is only trustworthy if at least one solar coefficient in an
                # active learning regime has actually been learned — otherwise
                # delta is dominated by seeded/default coefficients and can
                # inflate the target.  When no coefficient in the active
                # regime is learned, fall back to legacy paths (saturation
                # skip on saturated hours; raw-actual EMA otherwise).  The
                # gate is loose by design (any learned coefficient unlocks
                # the lift): mixed-fleet aggregation is still possible but
                # the upward bound is dark_target which is itself bounded by
                # SOLAR_COEFF_CAP × pot.
                lift_gate_open = solar_enabled and any(
                    solar_coefficients_per_unit.get(eid, {})
                        .get(_solar_coeff_regime(unit_modes.get(eid, MODE_HEATING)) or "", {})
                        .get("learned")
                    for eid in energy_sensors
                    if _solar_coeff_regime(unit_modes.get(eid, MODE_HEATING)) is not None
                )
                is_one_sided_lift = (
                    lift_gate_open
                    and base_expected_kwh > 0.0
                    and base_expected_kwh < normalized_actual
                )
                # Solar-saturation skip (#929): when the global model is fully
                # clipped (base ≤ solar_applied), the raw-actual EMA target is
                # solar-reduced and pulls the bucket downward.  Suppress that
                # update so dark hours own the downward correction.  When the
                # one-sided lift fires, it owns the hour instead — but if the
                # plausibility gate keeps the lift closed, this skip remains
                # reachable so the legacy wrong-direction step is still avoided.
                is_globally_saturated = (
                    solar_enabled
                    and base_expected_kwh > 0.0
                    and solar_normalization_delta > 0.0
                    and (base_expected_kwh - solar_normalization_delta)
                        < GLOBAL_BASE_SATURATION_SKIP_KWH
                    and not is_one_sided_lift
                )

                if is_globally_saturated:
                    _LOGGER.debug(
                        "Global base: skipping solar-saturated hour "
                        "(base=%.3f kWh <= solar=%.3f kWh, T=%s W=%s)",
                        base_expected_kwh, solar_normalization_delta,
                        temp_key, wind_bucket,
                    )
                    learning_status = "skipped_global_saturation"
                elif is_cold_cooling_regime:
                    _LOGGER.debug(
                        "Global base: skipping cold-cooling-regime hour "
                        "(T=%.1f, BP=%.1f, %d cooling unit(s))",
                        avg_temp, balance_point, len(active_modes),
                    )
                elif solar_enabled:
                    # Lift only fires when gate is open AND bucket sits below.
                    # Otherwise the legacy raw-actual EMA path applies — which
                    # under SNR weighting still attenuates sunny-hour pull and
                    # is corrected by dark hours.
                    if is_one_sided_lift:
                        base_target = normalized_actual
                    else:
                        base_target = total_energy_kwh

                    # #981: anchor SNR weighting in 4D DNI/DHI when 4D is
                    # the live read-path consumer, so the base bucket
                    # doesn't get pulled down by Kasten-misclassified
                    # clear hours.
                    sf_for_snr = solar_factor
                    if getattr(config, "experimental_4d_primary", False):
                        sf_4d = self._compute_4d_solar_factor_for_snr(obs, config)
                        if sf_4d is not None:
                            sf_for_snr = sf_4d
                    snr_w = compute_snr_weight(
                        sf_for_snr,
                        solar_dominant_entities,
                        total_units=count_active_learnable_units(
                            energy_sensors,
                            unit_modes,
                            hourly_expected_base_per_unit,
                            unit_min_base=unit_min_base,
                        ),
                    )
                    base_effective_rate = learning_rate * snr_w
                else:
                    base_target = total_energy_kwh
                    base_effective_rate = learning_rate

                if not is_cold_cooling_regime and not is_globally_saturated:
                    if base_expected_kwh == 0.0:
                        # Cold Start: Use Buffered Learning.
                        # Shutdown hours (weight = 0) must NOT contaminate the
                        # cold-start buffer with near-zero actuals.
                        if temp_key not in learning_buffer_global:
                            learning_buffer_global[temp_key] = {}
                        if wind_bucket not in learning_buffer_global[temp_key]:
                            learning_buffer_global[temp_key][wind_bucket] = []

                        buffer_list = learning_buffer_global[temp_key][wind_bucket]
                        if base_effective_rate > 0.0:
                            buffer_list.append(base_target)

                        if len(buffer_list) >= LEARNING_BUFFER_THRESHOLD:
                            avg_val = sum(buffer_list) / len(buffer_list)
                            new_base_prediction = avg_val
                            _LOGGER.info(f"Global Buffered Learning [Jump Start]: T={temp_key} W={wind_bucket} -> {new_base_prediction:.3f} kWh (Avg of {len(buffer_list)} samples)")
                            if temp_key not in correlation_data:
                                correlation_data[temp_key] = {}
                            correlation_data[temp_key][wind_bucket] = round(new_base_prediction, 5)
                            buffer_list.clear()
                        else:
                            _LOGGER.debug(f"Global Buffered Learning [Collecting]: T={temp_key} W={wind_bucket} -> Sample {len(buffer_list)}/{LEARNING_BUFFER_THRESHOLD} ({base_target:.3f} kWh)")
                            # While buffering, keep prediction at 0 until jump start
                            new_base_prediction = 0.0

                    else:
                        # EMA Update
                        # #967: inline EMA step.  ``helpers.compute_base_ema_step``
                        # is the named source-of-truth this expression matches;
                        # diagnostics already routes through it.  Migration of
                        # the live writer is deferred until 4D-primary
                        # observation has stabilised (refactor here during
                        # the observation window would confound attribution
                        # of any deviation drift).  When migrating, the
                        # round(_, 5) post-step clamp stays caller-side
                        # per the helper's contract.
                        new_base_prediction = base_expected_kwh + base_effective_rate * (base_target - base_expected_kwh)
                        if temp_key not in correlation_data:
                            correlation_data[temp_key] = {}
                        correlation_data[temp_key][wind_bucket] = round(new_base_prediction, 5)

                    model_base_after = new_base_prediction

        # Update Per-Unit Models (Both Normal and Aux modes)
        # We run this even in cooldown (for non-affected units)
        per_unit_result: dict = {}
        if should_run_per_unit:
            per_unit_result = self._process_per_unit_learning(
                temp_key, wind_bucket, avg_temp,
                avg_solar_vector, # Pass 3D vector for per-unit learning
                total_energy_kwh, base_expected_kwh,
                energy_sensors, hourly_delta_per_unit,
                solar_enabled, learning_rate,
                solar_calculator, get_predicted_unit_base_fn,
                learning_buffer_per_unit, correlation_data_per_unit, observation_counts,
                is_aux_active, aux_coefficients_per_unit, learning_buffer_aux_per_unit,
                solar_coefficients_per_unit, learning_buffer_solar_per_unit, balance_point,
                unit_modes, # Pass unit modes
                hourly_expected_per_unit,
                hourly_expected_base_per_unit,
                aux_affected_entities, # Pass aux exclusion list
                is_cooldown_active=is_cooldown_active,
                correction_percent=correction_percent,
                solar_dominant_entities=solar_dominant_entities,
                screen_config=config.screen_config,
                screen_affected_entities=config.screen_affected_entities,
                solar_affected_entities=config.solar_affected_entities,
                unit_min_base=unit_min_base,
                solar_factor=solar_factor,
                battery_filtered_potential=battery_filtered_potential,
                experimental_tobit_live_learner=getattr(
                    config, "experimental_tobit_live_learner", False
                ),
                tobit_live_entities=getattr(
                    config, "tobit_live_entities", frozenset()
                ),
                mpc_managed_entities=getattr(
                    config, "mpc_managed_entities", frozenset()
                ),
                tobit_sufficient_stats=getattr(
                    model, "tobit_sufficient_stats", None
                ),
                nlms_shadow_coefficients=getattr(
                    model, "nlms_shadow_coefficients", None
                ),
                shadow_learning_buffer_solar_per_unit=getattr(
                    model, "shadow_learning_buffer_solar_per_unit", None
                ),
                solar_coefficients_4d_per_unit=solar_coefficients_4d_per_unit,
                learning_buffer_solar_4d_per_unit=learning_buffer_solar_4d_per_unit,
                obs=obs,
            )

        return {
            "model_updated": model_updated,
            "model_base_before": model_base_before,
            "model_base_after": model_base_after,
            "aux_model_updated": aux_model_updated,
            "aux_model_before": aux_model_before,
            "aux_model_after": aux_model_after,
            "learning_status": learning_status,
            **per_unit_result,
        }

    def _process_learning_legacy(self, **kwargs) -> dict:
        """Backward-compatible entry point using flat keyword arguments.

        Builds HourlyObservation, ModelState, and LearningConfig from the
        legacy parameter dict, then delegates to the contract-based path.
        This exists solely to support existing tests; new code should use
        the (obs, model, config) interface.
        """
        from datetime import datetime

        obs = HourlyObservation(
            timestamp=datetime.now(),
            hour=0,
            avg_temp=kwargs["avg_temp"],
            inertia_temp=kwargs["avg_temp"],
            temp_key=kwargs["temp_key"],
            effective_wind=0.0,
            wind_bucket=kwargs["wind_bucket"],
            bucket_counts=kwargs.get("hourly_bucket_counts", {}),
            avg_humidity=None,
            solar_factor=0.0,
            solar_vector=_pad_solar_vector(kwargs.get("avg_solar_vector", (0.0, 0.0, 0.0))),
            solar_impact_raw=kwargs.get("solar_impact", 0.0),
            effective_solar_impact=kwargs.get("solar_impact", 0.0),
            total_energy_kwh=kwargs.get("total_energy_kwh", 0.0),
            learning_energy_kwh=kwargs.get("total_energy_kwh", 0.0),
            guest_impact_kwh=0.0,
            expected_kwh=kwargs.get("base_expected_kwh", 0.0),
            base_expected_kwh=kwargs.get("base_expected_kwh", 0.0),
            unit_breakdown=kwargs.get("hourly_delta_per_unit", {}),
            unit_expected=kwargs.get("hourly_expected_per_unit", {}),
            unit_expected_base=kwargs.get("hourly_expected_base_per_unit", {}),
            aux_impact_kwh=kwargs.get("aux_impact", 0.0),
            aux_fraction=1.0 if kwargs.get("is_aux_active", False) else 0.0,
            is_aux_dominant=kwargs.get("is_aux_active", False),
            sample_count=kwargs.get("hourly_sample_count", 0),
            unit_modes=kwargs.get("unit_modes", {}),
            was_cooldown_active=kwargs.get("is_cooldown_active", False),
        )

        model = ModelState(
            correlation_data=kwargs.get("correlation_data", {}),
            correlation_data_per_unit=kwargs.get("correlation_data_per_unit", {}),
            observation_counts=kwargs.get("observation_counts", {}),
            aux_coefficients=kwargs.get("aux_coefficients", {}),
            aux_coefficients_per_unit=kwargs.get("aux_coefficients_per_unit", {}),
            solar_coefficients_per_unit=kwargs.get("solar_coefficients_per_unit", {}),
            learned_u_coefficient=None,
            learning_buffer_global=kwargs.get("learning_buffer_global", {}),
            learning_buffer_per_unit=kwargs.get("learning_buffer_per_unit", {}),
            learning_buffer_aux_per_unit=kwargs.get("learning_buffer_aux_per_unit", {}),
            learning_buffer_solar_per_unit=kwargs.get("learning_buffer_solar_per_unit", {}),
        )

        config = LearningConfig(
            learning_enabled=kwargs.get("learning_enabled", True),
            solar_enabled=kwargs.get("solar_enabled", False),
            learning_rate=kwargs.get("learning_rate", 0.01),
            balance_point=kwargs.get("balance_point", 17.0),
            energy_sensors=kwargs.get("energy_sensors", []),
            aux_impact=kwargs.get("aux_impact", 0.0),
            solar_calculator=kwargs.get("solar_calculator"),
            get_predicted_unit_base_fn=kwargs.get("get_predicted_unit_base_fn"),
            aux_affected_entities=kwargs.get("aux_affected_entities"),
            has_guest_activity=kwargs.get("has_guest_activity", False),
            per_unit_learning_enabled=kwargs.get("per_unit_learning_enabled"),
        )

        return self.process_learning(obs=obs, model=model, config=config)

    def _process_per_unit_learning(
        self,
        temp_key: str,
        wind_bucket: str,
        avg_temp: float,
        avg_solar_vector: tuple[float, float, float],
        total_energy_kwh: float,
        base_expected_kwh: float,
        energy_sensors: list[str],
        hourly_delta_per_unit: dict,
        solar_enabled: bool,
        learning_rate: float,
        solar_calculator,
        get_predicted_unit_base_fn,
        learning_buffer_per_unit: dict,
        correlation_data_per_unit: dict,
        observation_counts: dict,
        is_aux_active: bool,
        aux_coefficients_per_unit: dict,
        learning_buffer_aux_per_unit: dict,
        solar_coefficients_per_unit: dict,
        learning_buffer_solar_per_unit: dict,
        balance_point: float,
        unit_modes: dict,
        hourly_expected_per_unit: dict,
        hourly_expected_base_per_unit: dict,
        aux_affected_entities: list[str] | None,
        is_cooldown_active: bool = False,
        correction_percent: float = 100.0,
        solar_dominant_entities: tuple[str, ...] = (),
        screen_config: tuple[bool, bool, bool] | None = None,
        screen_affected_entities: frozenset[str] | None = None,
        solar_affected_entities: frozenset[str] | None = None,
        solar_factor: float = 0.0,
        battery_filtered_potential: tuple[float, float, float] | None = None,
        unit_min_base: dict[str, float] | None = None,
        experimental_tobit_live_learner: bool = False,
        tobit_live_entities: frozenset[str] = frozenset(),
        mpc_managed_entities: frozenset[str] = frozenset(),
        tobit_sufficient_stats: dict | None = None,
        nlms_shadow_coefficients: dict | None = None,
        shadow_learning_buffer_solar_per_unit: dict | None = None,
        solar_coefficients_4d_per_unit: dict | None = None,
        learning_buffer_solar_4d_per_unit: dict | None = None,
        obs: "HourlyObservation | None" = None,
    ):
        """Process learning for individual units."""
        # --- 4D shadow learner per-hour preparation (#954) ---
        # Compute hour-midpoint sun position + DNI/DHI ONCE outside the
        # per-entity loop (these are entity-independent for this hour).
        # The per-entity 4D potential lives inside the loop because
        # screen_config narrows per entity.  When either 4D dict is None
        # (legacy path / test fixtures that didn't set them up), or
        # the obs lacks a timestamp, all 4D work is skipped — this is
        # the strict-shadow opt-in semantic.
        shadow_4d_enabled = (
            solar_coefficients_4d_per_unit is not None
            and learning_buffer_solar_4d_per_unit is not None
            and obs is not None
            and getattr(obs, "timestamp", None) is not None
            and solar_calculator is not None
        )
        # Read-path arbitration for dead-zone (#981).  Whichever solar pipeline
        # is the live read-path consumer needs the dead-zone safety net; the
        # other path is shadow.  Pre-promotion (flag off) → 3D is live, 4D is
        # shadow.  Post-promotion (flag on) → 4D is live, 3D is shadow.  The
        # ``is_shadow_path`` parameter on ``_learn_unit_solar_coefficient``
        # already gates the dead-zone block; we just need to thread the flag
        # to the two call sites below.
        _flag_4d_primary = bool(
            getattr(
                getattr(solar_calculator, "coordinator", None),
                "experimental_4d_primary",
                False,
            )
        )
        hour_dni_4d = 0.0
        hour_dhi_4d = 0.0
        hour_elev_4d = 0.0
        hour_az_4d = 0.0
        if shadow_4d_enabled:
            try:
                from datetime import timedelta as _td
                mid_dt = obs.timestamp + _td(minutes=30)
                hour_elev_4d, hour_az_4d = solar_calculator.get_approx_sun_pos(mid_dt)
                if hour_elev_4d > 0:
                    from .solar import resolve_dni_dhi
                    day_of_year = obs.timestamp.timetuple().tm_yday
                    dni_v, dhi_v, _src = resolve_dni_dhi(
                        getattr(obs, "dni_avg", None),
                        getattr(obs, "dhi_avg", None),
                        getattr(obs, "ghi_avg", None),
                        getattr(obs, "cloud_avg", None),
                        hour_elev_4d,
                        day_of_year,
                    )
                    hour_dni_4d = dni_v
                    hour_dhi_4d = dhi_v
                else:
                    shadow_4d_enabled = False
            except Exception:  # pragma: no cover - defensive against mocks
                shadow_4d_enabled = False

        # 4D shadow per-hour aggregates (#954 commit 9).  Accumulated
        # across qualifying entities and surfaced via the return dict
        # so the hourly_processor can log them to ``hourly_log``.
        # CAVEAT: unlike the live 3D ``solar_impact_kwh`` (which is
        # saturation-clipped via ``applied = min(c·potential, base)``
        # inside ``statistics.calculate_total_power``), these 4D values
        # are RAW: ``Σ c_4d · pot_4d`` per entity, no saturation cap.
        # A parallel saturation-aware 4D ``calculate_total_power`` would
        # double the implementation surface and is not needed for the
        # bucket-drift comparison Commit 10 builds — magnitudes will
        # diverge from 3D in saturated hours by design.
        # ``solar_normalization_delta_4d`` carries the same caveat.
        # Only populated when the 4D shadow path is enabled; if disabled
        # the keys are omitted from the return dict (the hourly_processor
        # gates on ``dict.get(...) is not None``).
        total_4d_impact = 0.0
        delta_4d = 0.0
        # ``shadow_4d_fired`` flips True the first time an entity in the
        # per-unit loop actually invokes the 4D learner.  Gating the
        # result-dict emission on this (rather than the always-truthy
        # ``solar_coefficients_4d_per_unit is not None``) keeps the log
        # fields ABSENT on hours where no entity exercised the 4D path
        # (below-horizon hours, DNI/DHI-unresolvable hours, hours where
        # every entity was outlier-suppressed, etc.).  Required by the
        # diagnose_solar.base_model_4d_shadow consumer, which uses
        # field-absence as the "no 4D-tagged hours" signal — see the
        # ``entry.get("solar_normalization_delta_4d") is None`` check in
        # ``_compute_base_model_4d_shadow_report``.
        shadow_4d_fired = False

        for entity_id in energy_sensors:
            unit_mode = unit_modes.get(entity_id, MODE_HEATING)
            if unit_mode in (MODE_OFF, MODE_GUEST_HEATING, MODE_GUEST_COOLING):
                # Skip learning for non-tracked/temporary modes
                continue

            # Per-entity solar-scope gate (#962).  Entities outside the
            # user-configured solar_affected_entities list have base/aux
            # learning continue normally but skip every solar branch
            # (NLMS, inequality, cold-start buffer accumulation).  None
            # sentinel = legacy / test coordinator path = all entities
            # affected (preserves prior behaviour).
            entity_solar_enabled = solar_enabled and (
                solar_affected_entities is None
                or entity_id in solar_affected_entities
            )

            # Mode-stratified per-unit buckets (#885): cooling-mode samples
            # land in a dedicated "cooling" wind-bucket regardless of the
            # actual hour's wind.  Heating-mode samples continue to use
            # normal/high_wind/extreme_wind.  The two sample spaces coexist
            # inside the same [entity][temp][wind] structure without
            # collision because _get_wind_bucket() never produces "cooling".
            effective_wind_bucket = (
                COOLING_WIND_BUCKET if unit_mode == MODE_COOLING else wind_bucket
            )

            # Check Cooldown Exclusion
            # If cooldown is active, we SKIP learning for units that were affected by Aux (thermal lag bias)
            # Units NOT affected by Aux can continue to learn normally.
            if is_cooldown_active:
                is_affected = True
                if aux_affected_entities is not None:
                    if entity_id not in aux_affected_entities:
                        is_affected = False

                if is_affected:
                    # Skip this unit
                    continue

            # DHW mode: unit is active but not contributing to space heating.
            # Force actual_unit = 0 so the per-unit model learns the correct
            # zero contribution rather than skipping the update entirely.
            # This also covers heat pump idle/standby cycles mapped to DHW by
            # the heat_pump_mode_sync blueprint.
            if unit_mode == MODE_DHW:
                actual_unit = 0.0
            else:
                # Only update model if sensor actually reported during this hour
                if entity_id not in hourly_delta_per_unit:
                    # Sensor was offline/unavailable entire hour - skip learning
                    continue
                actual_unit = hourly_delta_per_unit[entity_id]
            # Dual-Track Learning: prefer the per-unit baseline from Track B calculations
            if entity_id in hourly_expected_base_per_unit:
                expected_unit_base = hourly_expected_base_per_unit[entity_id]
            else:
                expected_unit_base = get_predicted_unit_base_fn(entity_id, temp_key, effective_wind_bucket, avg_temp)
            unit_mode = unit_modes.get(entity_id, MODE_HEATING)

            # Step 1: Learn Unit Solar (if enabled, sunny, and NOT aux)
            # We learn solar in Normal mode to establish the relationship.
            # Solar Formula: Unit_Solar_Impact = Global_Factor * Unit_Coeff
            # Unit_Coeff = (Unit_Base - Unit_Actual) / Global_Factor (for Heating)

            unit_solar_impact = 0.0

            # Reconstruct potential (pre-screen) solar vector for learning
            # (#809, per-direction since #826).  Coefficients learn against raw
            # irradiance so they encode window physics, not screen state.
            # Screen correction is applied at prediction time only.
            # Per-entity scope: if the entity is NOT in screen_affected_entities,
            # reconstruct with an all-False screen_config (transmittance=1.0).
            # NLMS for an unscreened entity then converges to
            # `coeff ≈ phys / install_avg_trans` — the coefficient is not "pure
            # phys" but the install_trans factor cancels at prediction time
            # (effective × (phys / install_trans) = phys × true_potential for
            # the unscreened zone).  End-to-end prediction stays consistent;
            # the purpose is to avoid confounding the coefficient with a
            # screen state the unit doesn't physically experience.
            entity_screen_config = (
                screen_config
                if screen_affected_entities is None or entity_id in screen_affected_entities
                else (False, False, False)
            )
            potential_s, potential_e, potential_w = SolarCalculator.reconstruct_potential_vector(
                avg_solar_vector, correction_percent, entity_screen_config
            )

            # Vector magnitude check for "sunny enough" threshold
            # Use potential vector — learning should happen when sun shines,
            # regardless of screen position.
            vector_magnitude = (potential_s**2 + potential_e**2 + potential_w**2) ** 0.5

            # Solar shutdown skip (#838): when a VP unit was detected as
            # solar-dominant this hour (thermostat cut compressor in sun),
            # its actual_impact = base - 0 = base would inflate the NLMS
            # coefficient.  Skip solar learning for this entity; the other
            # entities in the same hour continue to learn normally.
            is_solar_shutdown = entity_id in solar_dominant_entities

            # Per-unit override applies uniformly to both gates; only the
            # fallback constants differ.  Compute the override lookup once
            # so that a future divergence between SOLAR_LEARNING_MIN_BASE
            # and SOLAR_SHUTDOWN_MIN_BASE cannot pass silently as two
            # identical-by-coincidence _resolve_min_base() returns.
            unit_override = (unit_min_base or {}).get(entity_id)
            if unit_override is not None and unit_override > 0.0:
                nlms_threshold = shutdown_threshold = float(unit_override)
            else:
                nlms_threshold = SOLAR_LEARNING_MIN_BASE
                shutdown_threshold = SOLAR_SHUTDOWN_MIN_BASE
            if (
                entity_solar_enabled
                and vector_magnitude > 0.1
                and not is_aux_active
                and not is_solar_shutdown
                and expected_unit_base >= nlms_threshold
            ):
                # Stage 3 (#912) routing: when the experimental Tobit
                # live learner is enabled AND this entity is in the
                # allow-list AND the regime is heating/cooling, route
                # the modulating-regime sample through the running-
                # window Tobit Newton step.  When Tobit is the live
                # writer (post-cold-start) NLMS continues to fire in
                # shadow mode against ``nlms_shadow_coefficients`` for
                # comparison during the validation window — see locked
                # decision (2) in #912.  In Tobit cold-start (n_eff <
                # TOBIT_MIN_NEFF) NLMS is the live writer; the running
                # statistic accumulates in parallel so when Tobit
                # warms up it has a fully-formed window (decision (3)).
                regime_for_routing = _solar_coeff_regime(unit_mode)
                # Allow-list semantic shift (#918, 1.3.5 default-on):
                # empty list = "auto-mode, plausibility-gate decides per
                # entity"; non-empty list = "scope override — only listed
                # entities try Tobit, plausibility still applies within
                # those".  Pre-1.3.5 semantics required entity_id to be
                # explicitly in the list (default-off + manual opt-in).
                # Existing maintainer installs with non-empty allow-lists
                # keep the override behaviour; new users on default-on
                # get auto-mode without configuration.
                tobit_live_active = (
                    experimental_tobit_live_learner
                    and (
                        not tobit_live_entities
                        or entity_id in tobit_live_entities
                    )
                    and entity_id not in mpc_managed_entities
                    and regime_for_routing is not None
                    and tobit_sufficient_stats is not None
                )
                
                # --- Outlier Robustness (#919) ---
                # Check for outliers before they enter either path.
                # Use current coefficient to estimate residual.
                entity_coeffs = solar_coefficients_per_unit.get(entity_id, {})
                regime_coeffs = entity_coeffs.get(regime_for_routing, {}) if isinstance(entity_coeffs, dict) else {}
                
                c_s = regime_coeffs.get("s", 0.0)
                c_e = regime_coeffs.get("e", 0.0)
                c_w = regime_coeffs.get("w", 0.0)
                
                predicted_impact = c_s * potential_s + c_e * potential_e + c_w * potential_w
                if regime_for_routing == "heating":
                    actual_impact_for_filter = expected_unit_base - actual_unit
                else:
                    actual_impact_for_filter = actual_unit - expected_unit_base
                
                residual = actual_impact_for_filter - predicted_impact
                is_outlier = self._is_outlier_residual(
                    entity_id, regime_for_routing, residual, self._outlier_state,
                    is_shutdown=False,
                    actual=actual_unit,
                    expected_base=expected_unit_base,
                    unit_min_base=unit_min_base,
                )

                tobit_applied = False
                if tobit_live_active and not is_outlier:
                    if regime_for_routing == "heating":
                        actual_impact_for_tobit = expected_unit_base - actual_unit
                    else:
                        actual_impact_for_tobit = actual_unit - expected_unit_base
                    # Match the batch_fit / compute_tobit_for_diagnose
                    # filter for non-positive impacts — Tobit relies on
                    # the censoring threshold being above the model's
                    # noise floor, which a non-positive impact violates.
                    if actual_impact_for_tobit > 0.0:
                        tobit_result = self._update_unit_tobit_live(
                            entity_id,
                            regime_for_routing,
                            (potential_s, potential_e, potential_w),
                            actual_impact_for_tobit,
                            expected_unit_base,
                            tobit_sufficient_stats,
                            solar_coefficients_per_unit,
                        )
                        tobit_applied = bool(tobit_result.get("applied"))

                if tobit_applied and nlms_shadow_coefficients is not None and not is_outlier:
                    # NLMS shadow: independent NLMS trajectory that
                    # learns against its own state (separate from the
                    # main / Tobit-written coefficient).  Used as a
                    # reference signal in diagnose during the Stage 3
                    # validation window.  Removed at default-on promote.
                    # Shadow MUST use a SEPARATE cold-start buffer
                    # (review I1, #912): sharing with main produces an
                    # aliasing footgun where shadow's zero-impact hours
                    # could trigger a dead-zone reset that wipes main's
                    # buffer.  Dead-zone-reset is also suppressed for
                    # the shadow path — it's a recovery mechanism for
                    # the writer-of-record, and shadow is reference-
                    # only.  ``shadow_learning_buffer_solar_per_unit``
                    # is provided by the coordinator (#912).  None
                    # fallback for older test fixtures that don't pass
                    # it — defaults to a fresh dict (acceptable for
                    # tests; production always provides it).
                    shadow_buffer = (
                        shadow_learning_buffer_solar_per_unit
                        if shadow_learning_buffer_solar_per_unit is not None
                        else {}
                    )
                    self._seed_shadow_from_main_if_empty(
                        entity_id,
                        solar_coefficients_per_unit,
                        nlms_shadow_coefficients,
                    )
                    self._learn_unit_solar_coefficient(
                        entity_id, temp_key,
                        expected_unit_base, actual_unit, (potential_s, potential_e, potential_w),
                        learning_rate, nlms_shadow_coefficients, shadow_buffer,
                        avg_temp, balance_point,
                        unit_mode,
                        is_shadow_path=True,
                    )
                elif not tobit_applied and not is_outlier:
                    # Tobit cold-start / disabled / not allow-listed —
                    # NLMS is the live writer (current behaviour).  Under
                    # ``experimental_4d_primary``, 3D becomes shadow and
                    # the dead-zone net moves to the 4D call below (#981).
                    self._learn_unit_solar_coefficient(
                        entity_id, temp_key,
                        expected_unit_base, actual_unit, (potential_s, potential_e, potential_w),
                        learning_rate, solar_coefficients_per_unit, learning_buffer_solar_per_unit,
                        avg_temp, balance_point,
                        unit_mode,
                        is_shadow_path=_flag_4d_primary,
                    )

                # --- 4D shadow learner (#954) ---
                # Runs alongside the 3D NLMS / Tobit / shadow paths on
                # any qualifying, non-outlier hour.  Strict shadow —
                # writes only to ``solar_coefficients_4d_per_unit``.
                if shadow_4d_enabled and not is_outlier:
                    shadow_4d_fired = True
                    potential_4d = solar_calculator.calculate_unit_potential_4d(
                        entity_id,
                        hour_dni_4d,
                        hour_dhi_4d,
                        hour_elev_4d,
                        hour_az_4d,
                        entity_screen_config,
                        correction_percent,
                    )
                    # Accumulate 4D shadow predicted impact BEFORE
                    # the learning step so the logged value reflects
                    # coefficients at the start of the hour — matches
                    # the live 3D ``solar_impact_kwh`` temporal
                    # convention.  Per-entity saturation matches 3D /
                    # ``solar.calculate_saturation`` (#992 commit 4):
                    # heating clamps to ``expected_unit_base``,
                    # cooling is additive without clamp.  Both written
                    # values (``solar_impact_4d_kwh`` and
                    # ``solar_normalization_delta_4d``) now mirror the
                    # 3D pipeline's saturation semantics — pre-c4 they
                    # were raw c·p sums that diverged from 3D by the
                    # entity-base censoring gap on every saturated hour.
                    regime_4d = _solar_coeff_regime(unit_mode)
                    if regime_4d is not None:
                        entity_coeffs_4d = (
                            solar_coefficients_4d_per_unit.get(entity_id, {})
                            if solar_coefficients_4d_per_unit is not None
                            else {}
                        )
                        regime_coeffs_4d = (
                            entity_coeffs_4d.get(regime_4d, {})
                            if isinstance(entity_coeffs_4d, dict)
                            else {}
                        )
                        c4d_s = float(regime_coeffs_4d.get("s", 0.0))
                        c4d_e = float(regime_coeffs_4d.get("e", 0.0))
                        c4d_w = float(regime_coeffs_4d.get("w", 0.0))
                        c4d_d = float(regime_coeffs_4d.get("diffuse", 0.0))
                        p4d_s, p4d_e, p4d_w, p4d_d = potential_4d
                        predicted_4d = (
                            c4d_s * p4d_s
                            + c4d_e * p4d_e
                            + c4d_w * p4d_w
                            + c4d_d * p4d_d
                        )
                        # Per-entity saturation via canonical dispatch (#1008).
                        applied_4d, _wasted_4d, _net_4d = solar_calculator.calculate_saturation(
                            expected_unit_base, predicted_4d, unit_mode
                        )
                        total_4d_impact += applied_4d
                        delta_4d += applied_4d if regime_4d == "heating" else -applied_4d
                    self._learn_unit_solar_coefficient(
                        entity_id=entity_id,
                        temp_key=temp_key,
                        expected_unit_base=expected_unit_base,
                        actual_unit=actual_unit,
                        avg_solar_vector=potential_4d,
                        learning_rate=learning_rate,
                        solar_coefficients_per_unit=solar_coefficients_4d_per_unit,
                        learning_buffer_solar_per_unit=learning_buffer_solar_4d_per_unit,
                        avg_temp=avg_temp,
                        balance_point=balance_point,
                        unit_mode=unit_mode,
                        components=("s", "e", "w", "diffuse"),
                        # Read-path arbitration (#981): 4D is shadow until
                        # ``experimental_4d_primary`` flips, then it becomes
                        # the live writer and inherits the dead-zone net.
                        is_shadow_path=not _flag_4d_primary,
                    )
            elif (
                # Inequality learning for shutdown hours.
                # Runs in parallel to NLMS: while NLMS is gated on
                # ``not is_solar_shutdown`` above, this branch fires
                # specifically WHEN the unit was flagged shutdown.  The
                # two paths converge toward the same physical coefficient
                # from opposite sides (NLMS from above, inequality from
                # below).
                #
                # Scope restriction (#xxx per-entity screen): inequality
                # is only applied to screen-affected entities.
                # ``battery_filtered_potential`` is coordinator-level and
                # reconstructed with the installation screen_config —
                # using it for unscreened entities would yield an inflated
                # battery that satisfies the constraint too easily (~20-60 %
                # under-lift during screen-closed hours).  Unscreened
                # entities already have correct NLMS signal per the live
                # path, so skipping inequality here is behaviourally safe.
                entity_solar_enabled
                and is_solar_shutdown
                and not is_aux_active
                and expected_unit_base >= shutdown_threshold
                and unit_mode == MODE_HEATING  # cooling semantics inverted — out of scope
                and battery_filtered_potential is not None
                and (
                    screen_affected_entities is None
                    or entity_id in screen_affected_entities
                )
            ):
                # Outlier check for inequality (#919 Part 1)
                regime_ineq = "heating"  # Inequality is heating only
                entity_coeffs = solar_coefficients_per_unit.get(entity_id, {})
                regime_coeffs = entity_coeffs.get(regime_ineq, {}) if isinstance(entity_coeffs, dict) else {}
                c_s = regime_coeffs.get("s", 0.0)
                c_e = regime_coeffs.get("e", 0.0)
                c_w = regime_coeffs.get("w", 0.0)

                pot_s, pot_e, pot_w = battery_filtered_potential
                predicted_impact = c_s * pot_s + c_e * pot_e + c_w * pot_w
                # For shutdown, implied actual impact is expected_unit_base
                residual = expected_unit_base - predicted_impact

                if not self._is_outlier_residual(
                    entity_id, regime_ineq, residual, self._outlier_state,
                    is_shutdown=True,
                    actual=actual_unit,
                    expected_base=expected_unit_base,
                    unit_min_base=unit_min_base,
                ):
                    self._update_unit_solar_inequality(
                        entity_id=entity_id,
                        expected_unit_base=expected_unit_base,
                        battery_filtered_potential=battery_filtered_potential,
                        solar_coefficients_per_unit=solar_coefficients_per_unit,
                    )

                    # 4D shadow inequality (#954 commit 7).  Same gating
                    # as the 3D path (heating regime, screen_affected,
                    # not an outlier).  Strictly one-sided — invariant #5.
                    if (
                        solar_coefficients_4d_per_unit is not None
                        and obs is not None
                    ):
                        battery_4d = getattr(
                            obs, "battery_filtered_potential_4d",
                            (0.0, 0.0, 0.0, 0.0),
                        )
                        self._update_unit_solar_inequality(
                            entity_id=entity_id,
                            expected_unit_base=expected_unit_base,
                            battery_filtered_potential=battery_4d,
                            solar_coefficients_per_unit=solar_coefficients_4d_per_unit,
                            components=("s", "e", "w", "diffuse"),
                        )

            # Step 2: Calculate Solar Impact using (possibly updated) coefficients
            # Use potential vector (not effective) because the coefficient already
            # absorbs screen transmittance via NLMS learning against actual_impact
            # which includes the screen effect.  Using effective here would
            # double-count transmittance: coeff(≈phys×trans) × effective(=pot×trans)
            # = phys×pot×trans², leaving residual solar in the normalized data.
            #
            # Per-unit base EMA uses raw ``actual_unit`` as target (same
            # rationale as the global path): the hour's SNR weight is
            # combined into the existing headroom multiplier further
            # below.  ``unit_solar_impact`` is retained for the headroom
            # calculation.
            if solar_enabled:
                unit_coeff = solar_calculator.calculate_unit_coefficient(
                    entity_id, temp_key, unit_mode
                )
                unit_solar_impact = solar_calculator.calculate_unit_solar_impact(
                    (potential_s, potential_e, potential_w), unit_coeff
                )
            else:
                unit_solar_impact = 0.0
            unit_normalized = actual_unit

            # Step 3: Learn Base or Aux Model
            if is_aux_active:
                # DHW + Aux simultaneously: the zero contribution is caused by DHW,
                # not by the aux system. Skip aux-coefficient learning to avoid
                # corrupting the per-unit aux coefficient.
                if unit_mode == MODE_DHW:
                    pass
                else:
                    # Learn Individual Aux Reduction (kW)
                    # Check Exclusion: Only learn if unit is affected by aux
                    is_affected = True
                    if aux_affected_entities is not None:
                        if entity_id not in aux_affected_entities:
                            is_affected = False

                    if is_affected:
                        self._learn_unit_aux_coefficient(
                            entity_id, temp_key, effective_wind_bucket,
                            expected_unit_base, unit_normalized,
                            learning_rate,  # Use global rate
                            aux_coefficients_per_unit, learning_buffer_aux_per_unit,
                            correlation_data_per_unit
                        )
                    else:
                        # Excluded unit: Not affected by aux, so learn Base Model normally.
                        # Note: This does NOT require Global Base to update - Global is locked
                        # during aux (see process_learning). The models are independent.
                        self._learn_unit_model(
                            entity_id, temp_key, effective_wind_bucket,
                            expected_unit_base, unit_normalized,
                            learning_rate,
                            learning_buffer_per_unit, correlation_data_per_unit, observation_counts
                        )
            else:
                # Learn Normal Model.
                # Headroom-weighted EMA rate (#838): when the unsaturated
                # unit_solar_impact approaches or exceeds expected_unit_base,
                # the normalization target (actual + solar_impact) reflects
                # an inflated coefficient, not reality.  Slowing the EMA in
                # proportion to the remaining headroom breaks the second
                # link of the feedback loop without touching the formula.
                # At full saturation (solar >= base) → multiplier = 0.
                # Dark hours (solar ≈ 0) → multiplier = 1 (unchanged).
                #
                # The headroom multiplier is composed with the hour's SNR
                # weight.  Both attenuate the step size for the same class
                # of noisy hours (high solar impact) but for different
                # reasons — headroom on saturation-per-unit, SNR on
                # hour-level solar presence.  Multiplying them is
                # conservative: any hour downweighted by either mechanism
                # stays downweighted.
                if solar_enabled and expected_unit_base > 0.0:
                    headroom_multiplier = max(
                        0.0,
                        (expected_unit_base - unit_solar_impact) / expected_unit_base,
                    )
                    headroom_multiplier = min(1.0, headroom_multiplier)
                else:
                    headroom_multiplier = 1.0

                # #985: anchor per-unit SNR in 4D DNI/DHI when 4D is the
                # live read-path consumer, mirroring the global base fix
                # from #28e0125.  Per-unit base coefficients suffer the
                # same Kasten-misclassification failure mode as the global
                # base — see the 2026-05-17 production log captured in
                # #985 for the mitsubishi_vp_2 case study.  The helper
                # takes a config-shaped object; ``_process_per_unit_learning``
                # receives flat-unpacked params instead of ``config``, so
                # we shim the three attrs the helper actually reads.
                sf_for_snr = solar_factor
                if _flag_4d_primary:
                    _config_shim = SimpleNamespace(
                        solar_calculator=solar_calculator,
                        solar_enabled=solar_enabled,
                        experimental_4d_primary=True,
                    )
                    sf_4d = self._compute_4d_solar_factor_for_snr(obs, _config_shim)
                    if sf_4d is not None:
                        sf_for_snr = sf_4d
                snr_w = compute_snr_weight(
                    sf_for_snr,
                    solar_dominant_entities,
                    total_units=count_active_learnable_units(
                        energy_sensors,
                        unit_modes,
                        hourly_expected_base_per_unit,
                        unit_min_base=unit_min_base,
                    ),
                )
                rate_multiplier = headroom_multiplier * snr_w

                # Per #885: cooling-at-cold no longer needs a guard here.
                # Mode-stratified per-unit buckets route cooling samples
                # to a dedicated "cooling" wind-bucket (see
                # effective_wind_bucket above), so cold-hour cooling
                # standby populates cooling[temp]["cooling"] — which is
                # semantically correct (those ARE cooling-mode
                # observations) and cannot contaminate heating buckets.
                self._learn_unit_model(
                    entity_id, temp_key, effective_wind_bucket,
                    expected_unit_base, unit_normalized,
                    learning_rate,
                    learning_buffer_per_unit, correlation_data_per_unit, observation_counts,
                    rate_multiplier=rate_multiplier,
                )

        # Surface 4D shadow per-hour aggregates (#954 commit 9) only when
        # the 4D shadow path ACTUALLY fired for at least one entity in
        # this hour.  ``shadow_4d_fired`` is set inside the per-entity
        # 4D-learner branch above; gating on it (rather than the
        # always-truthy ``solar_coefficients_4d_per_unit is not None``,
        # since the coordinator initialises the dict to ``{}``) keeps
        # the keys absent on below-horizon / DNI-DHI-unresolved / all-
        # outlier hours.  The hourly_processor gates with
        # ``dict.get(...) is not None``, and
        # diagnose_solar.base_model_4d_shadow uses field-absence as the
        # "no 4D-tagged hours" signal.
        result: dict = {}
        if shadow_4d_fired:
            result["solar_impact_4d_kwh"] = total_4d_impact
            result["solar_normalization_delta_4d"] = delta_4d
        return result

    @staticmethod
    def _seed_shadow_from_main_if_empty(
        entity_id: str,
        solar_coefficients_per_unit: dict,
        nlms_shadow_coefficients: dict,
    ) -> bool:
        """Seed NLMS shadow from main at handover (#912 review I2).

        Without this, shadow's first qualifying hour as writer-of-record
        finds an empty entry and enters cold-start while main has
        months of NLMS-warm-up baked into its coefficient.  The
        compare-shadow-to-main exercise would be cold-start-vs-converged
        for ~2 weeks — uninformative.  Seeding copies the current main
        coefficient so shadow starts at the same state and diverges
        from there under independent NLMS evolution.

        Subsequent calls find a populated shadow entry and skip the
        seed (one-shot operation).  Returns True if a seed was applied
        on this call, False if shadow was already populated.
        """
        if entity_id in nlms_shadow_coefficients:
            return False
        main_entry = solar_coefficients_per_unit.get(entity_id)
        if not isinstance(main_entry, dict):
            return False
        nlms_shadow_coefficients[entity_id] = {
            r: dict(main_entry.get(r, {})) for r in ("heating", "cooling")
        }
        return True

    def _learn_unit_solar_coefficient(
        self,
        entity_id: str,
        temp_key: str,
        expected_unit_base: float,
        actual_unit: float,
        avg_solar_vector: tuple[float, ...],
        learning_rate: float,
        solar_coefficients_per_unit: dict | None,
        learning_buffer_solar_per_unit: dict | None,
        avg_temp: float,
        balance_point: float,
        unit_mode: str,
        *,
        components: tuple[str, ...] = ("s", "e", "w"),
        is_shadow_path: bool = False,
    ):
        """Update solar coefficient for one (entity, mode) slot.

        Mode-stratified per #868: heating-mode hours update
        ``solar_coefficients_per_unit[entity]["heating"]``; cooling-mode
        hours update ``["cooling"]``.  Each regime absorbs its own
        ``E[1/COP]`` and converges to a physically distinct value.
        OFF/DHW/unknown modes return early — no solar learning signal.

        Dimension-agnostic via ``components``.  Default ``("s","e","w")``
        is the live 3D solar coefficient — pre-existing callers see
        bit-identical behaviour.  The 4D shadow path (#954) passes
        ``("s","e","w","diffuse")`` along with a 4-tuple solar/potential
        vector and the 4D coefficient / buffer dicts.  The parameter is
        still named ``avg_solar_vector`` for backwards-compat with all
        positional 3D callers; for the 4D path it carries the per-hour
        potential vector with the diffuse component appended.

        Dimension-specific behaviour preserved verbatim:

        * magnitude gate — Euclidean ``sqrt(Σ vec_k²) ≤ 0.01`` for
          ``n_dim == 3``; L1 ``Σ vec_k ≤ 0.01`` for ``n_dim ≥ 4``
        * cold-start buffer threshold — ``LEARNING_BUFFER_THRESHOLD``
          for ``n_dim == 3``; ``max(THRESHOLD, n_dim + 1)`` otherwise
        * cold-start LS — 3×3 Gram + Gauss-Jordan with the
          ``|det| > 1e-6`` trigger for the 1D collinear fallback at
          ``n_dim == 3``; ``_solve_4x4_normal_equations`` at
          ``n_dim == 4`` (no fallback — collinear sun lets the buffer
          keep collecting); generic ``A=XᵀX`` solve at ``n_dim ≥ 5``
        * dead-zone reset — runs only when ``n_dim == 3`` because the
          4D shadow path has no read-path consumer yet (no stuck
          coefficient blocking base recovery to recover from)
        """
        if solar_coefficients_per_unit is None or learning_buffer_solar_per_unit is None:
            return

        regime = _solar_coeff_regime(unit_mode)
        if regime is None:
            return

        n_dim = len(components)

        actual_impact = 0.0
        if regime == "heating":
            # Heating: Sun reduces consumption
            actual_impact = expected_unit_base - actual_unit
        else:  # regime == "cooling"
            # Cooling: Sun increases consumption
            actual_impact = actual_unit - expected_unit_base

        # Clamping
        raw_impact = actual_impact
        actual_impact = max(0.0, actual_impact)

        vec = tuple(float(v) for v in avg_solar_vector)
        # Magnitude gate dispatched by dimension to preserve bit-identity
        # with the prior split implementations (3D used Euclidean norm;
        # 4D used L1 sum).
        if n_dim == 3:
            vector_magnitude = (vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2) ** 0.5
        else:
            vector_magnitude = sum(vec)
        if vector_magnitude <= 0.01:
            return

        # Get current (regime-specific) coefficient — None means cold-start.
        # The entity dict carries both regimes; we route to the active one.
        entity_coeffs = solar_coefficients_per_unit.get(entity_id)
        current_coeff = None
        if isinstance(entity_coeffs, dict):
            regime_coeff = entity_coeffs.get(regime)
            if isinstance(regime_coeff, dict) and any(
                regime_coeff.get(k) for k in components
            ):
                current_coeff = regime_coeff

        # --- Dead Zone Detection ---
        # When the base model is too low (expected < actual during sun),
        # actual_impact clamps to 0 and NLMS receives no signal.  The
        # coefficient is trapped — it can't learn because the base model
        # is wrong, and the base model stays wrong because the coefficient
        # is wrong (no solar normalization).  After SOLAR_DEAD_ZONE_THRESHOLD
        # consecutive qualifying hours with zero impact, force a coefficient
        # reset so cold-start can re-learn from fresh data.  Dead-zone
        # counter is keyed by (entity, regime) so a stuck cooling coefficient
        # resets without disturbing heating, and vice versa.
        # Stage 3 (#912) review I1: shadow path skips dead-zone reset.
        # The recovery mechanism is for the writer-of-record (whose
        # wrong coefficient prevents the base model from recovering);
        # shadow is reference-only.  Skipping the dead-zone reset on
        # shadow also avoids the aliasing footgun where shadow's
        # zero-impact hours could trigger a counter-driven reset that
        # wipes its own buffer mid-cold-start (the counter is on
        # ``self._dead_zone_counts`` which IS shared, but the reset's
        # effect — clearing the buffer — operates on the shadow's
        # private buffer dict, so the shared counter just becomes
        # noise rather than a footgun).  Cleaner to short-circuit.
        # #981: dead-zone is dimension-agnostic — it protects whichever
        # pipeline is the live read-path writer.  Callers in
        # ``_process_per_unit_learning`` thread ``experimental_4d_primary``
        # into ``is_shadow_path`` so the live path (3D pre-promotion,
        # 4D post-promotion) participates and the shadow does not.
        # The trigger condition (``actual_impact == 0.0`` AND
        # ``raw_impact < 0.0``) measures consumption signal and is
        # itself dimension-agnostic; only one of the two pipelines is
        # live at a time, so the shared counter has no aliasing.
        dead_key = (entity_id, regime)
        if not is_shadow_path:
            if (
                actual_impact == 0.0
                and raw_impact < 0.0
                and current_coeff is not None
            ):
                count = self._dead_zone_counts.get(dead_key, 0) + 1
                self._dead_zone_counts[dead_key] = count
                if count >= SOLAR_DEAD_ZONE_THRESHOLD:
                    # Reset only this regime — preserve the other one.
                    if isinstance(entity_coeffs, dict):
                        entity_coeffs[regime] = {c: 0.0 for c in components}
                    # Clear regime buffer so cold-start begins fresh
                    buf_entry = learning_buffer_solar_per_unit.get(entity_id)
                    if isinstance(buf_entry, dict):
                        buf_entry[regime] = []
                    self._dead_zone_counts[dead_key] = 0
                    _LOGGER.warning(
                        "Solar dead zone detected: reset %s [%s] after %d "
                        "consecutive zero-impact qualifying hours (base model too "
                        "low to produce learnable signal)",
                        entity_id, regime, count,
                    )
                return
            else:
                # Any non-zero impact resets the counter
                self._dead_zone_counts.pop(dead_key, None)

        # Cold-start buffer threshold — 3D uses LEARNING_BUFFER_THRESHOLD
        # (4); higher-dim paths need at least n_dim + 1 samples for an
        # identifiable LS fit (matches the prior 4D ``max(THRESHOLD, 5)``
        # gate at n_dim == 4).
        if n_dim == 3:
            buffer_threshold = LEARNING_BUFFER_THRESHOLD
        else:
            buffer_threshold = max(LEARNING_BUFFER_THRESHOLD, n_dim + 1)

        # --- Buffered Learning Logic (Cold Start) ---
        if current_coeff is None:
            buf_entry = learning_buffer_solar_per_unit.setdefault(entity_id, {})
            if not isinstance(buf_entry, dict):
                # Defensive: legacy in-memory shape — rewrap to per-regime.
                buf_entry = {"heating": [], "cooling": []}
                learning_buffer_solar_per_unit[entity_id] = buf_entry
            buffer_list = buf_entry.setdefault(regime, [])
            # Append (vec_0, ..., vec_{n-1}, actual_impact) as (n_dim+1)-tuple.
            buffer_list.append(tuple(vec) + (actual_impact,))

            if len(buffer_list) >= buffer_threshold:
                # Guard: if all impact values are zero, the base model is too
                # low to produce signal (dead zone during cold-start).  Discard
                # the buffer and keep collecting — the default coefficient
                # provides solar normalization for the base model via
                # calculate_unit_coefficient, which will gradually raise the
                # base model until actual_impact becomes positive.
                if all(item[n_dim] == 0.0 for item in buffer_list):
                    buffer_list.clear()
                    if n_dim == 3:
                        _LOGGER.debug(
                            "Solar cold-start buffer discarded for %s: all %d "
                            "samples had zero impact (base model recovering)",
                            entity_id, buffer_threshold,
                        )
                    return

                # Cold-start LS dispatched by dimension to preserve bit-
                # identity with the prior split implementations.
                new_coeff: dict[str, float] | None = None
                if n_dim == 3:
                    new_coeff = self._cold_start_solar_ls_3d(buffer_list)
                    if new_coeff is None:
                        _LOGGER.debug(
                            "Buffered Unit Solar [Collecting]: %s -> degenerate, skipping.",
                            entity_id,
                        )
                        return
                    _LOGGER.info(
                        "Buffered Unit Solar Learning [Jump Start]: %s [%s] -> %s "
                        "(3D Least Squares, %d samples, damping=%s)",
                        entity_id, regime, new_coeff, len(buffer_list),
                        COLD_START_SOLAR_DAMPING,
                    )
                elif n_dim == 4:
                    X_rows = [tuple(it[:n_dim]) for it in buffer_list]
                    y_vals = [it[n_dim] for it in buffer_list]
                    solution = self._solve_4x4_normal_equations(X_rows, y_vals)
                    if solution is None:
                        # Singular matrix (collinear sun, or one component
                        # identically zero).  Don't fall back to 1D — let
                        # the buffer keep collecting until sun-diversity
                        # makes the system solvable.  Keep the buffer.
                        _LOGGER.debug(
                            "4D shadow cold-start: singular LS for %s [%s], "
                            "keeping buffer (n=%d)",
                            entity_id, regime, len(buffer_list),
                        )
                        return
                    new_coeff = {
                        components[k]: max(
                            -SOLAR_COEFF_CAP,
                            min(SOLAR_COEFF_CAP, solution[k]),
                        )
                        * COLD_START_SOLAR_DAMPING
                        for k in range(n_dim)
                    }
                    _LOGGER.info(
                        "4D shadow cold-start jump start: %s [%s] -> %s "
                        "(4x4 LS, n=%d, damping=%.2f)",
                        entity_id, regime, new_coeff, len(buffer_list),
                        COLD_START_SOLAR_DAMPING,
                    )
                else:
                    # Generic n_dim — build A = XᵀX, b = Xᵀy, solve via
                    # helpers.solve_gauss_jordan.  No 1D fallback.
                    A_mat = [[0.0] * n_dim for _ in range(n_dim)]
                    b_vec = [0.0] * n_dim
                    for item in buffer_list:
                        impact = item[n_dim]
                        for i in range(n_dim):
                            xi = item[i]
                            for j in range(n_dim):
                                A_mat[i][j] += xi * item[j]
                            b_vec[i] += xi * impact
                    sol = solve_gauss_jordan(A_mat, b_vec)
                    if sol is None:
                        return
                    new_coeff = {
                        components[k]: max(
                            -SOLAR_COEFF_CAP, min(SOLAR_COEFF_CAP, sol[k])
                        )
                        * COLD_START_SOLAR_DAMPING
                        for k in range(n_dim)
                    }

                self._update_unit_solar_coefficient(
                    entity_id,
                    new_coeff,
                    solar_coefficients_per_unit,
                    regime,
                    components=components,
                )
                buffer_list.clear()
            else:
                _LOGGER.debug(
                    "Buffered Unit Solar [Collecting]: %s -> Sample %d/%d (%.3f kW)",
                    entity_id, len(buffer_list), buffer_threshold, actual_impact,
                )

        else:
            # Post-Jump Start: Normalized LMS (NLMS) (#809 A3)
            # Standard LMS has gain-dependent instability: high-solar units
            # (e.g. Toshiba with large south windows) get updates proportional
            # to solar_s², causing oscillation. NLMS normalizes by input power,
            # making convergence rate independent of solar magnitude.
            # Epsilon acts as L2 regularization — when solar power is low,
            # the effective step shrinks, preventing noise-chasing on
            # poorly-constrained components.
            coeffs = [
                float(current_coeff.get(components[k], 0.0)) for k in range(n_dim)
            ]

            predicted_impact = 0.0
            input_power = 0.0
            for k in range(n_dim):
                predicted_impact += coeffs[k] * vec[k]
                input_power += vec[k] ** 2
            error = actual_impact - predicted_impact
            step = NLMS_STEP_SIZE * error / (input_power + NLMS_REGULARIZATION)

            new_coeff = {
                components[k]: max(
                    -SOLAR_COEFF_CAP,
                    min(SOLAR_COEFF_CAP, coeffs[k] + step * vec[k]),
                )
                for k in range(n_dim)
            }

            _LOGGER.debug(f"Per-Unit Solar Learning [NLMS]: {entity_id} [{regime}] -> {new_coeff} (was {current_coeff})")
            self._update_unit_solar_coefficient(
                entity_id,
                new_coeff,
                solar_coefficients_per_unit,
                regime,
                components=components,
            )

    @staticmethod
    def _cold_start_solar_ls_3d(
        buffer_list: list[tuple[float, float, float, float]],
    ) -> dict[str, float] | None:
        """3D cold-start LS with 1D collinear fallback (preserved verbatim).

        Returns the damped coefficient dict ``{"s","e","w"}`` on success,
        or ``None`` when both joint LS and the 1D fallback are degenerate
        (the caller keeps the buffer so cold-start retries on the next
        qualifying hour).  Bit-identical to the inline pre-unification
        block — extracted here so the unified ``_learn_unit_solar_coefficient``
        body stays readable.
        """
        sum_s2 = sum(item[0] ** 2 for item in buffer_list)
        sum_e2 = sum(item[1] ** 2 for item in buffer_list)
        sum_w2 = sum(item[2] ** 2 for item in buffer_list)
        sum_se = sum(item[0] * item[1] for item in buffer_list)
        sum_sw = sum(item[0] * item[2] for item in buffer_list)
        sum_ew = sum(item[1] * item[2] for item in buffer_list)
        sum_s_I = sum(item[0] * item[3] for item in buffer_list)
        sum_e_I = sum(item[1] * item[3] for item in buffer_list)
        sum_w_I = sum(item[2] * item[3] for item in buffer_list)

        determinant = (
            sum_s2 * (sum_e2 * sum_w2 - sum_ew ** 2)
            - sum_se * (sum_se * sum_w2 - sum_ew * sum_sw)
            + sum_sw * (sum_se * sum_ew - sum_e2 * sum_sw)
        )

        # Full 3D solution when sun angles are diverse enough.  Trigger
        # `|det| > 1e-6` is preserved so the boundary between joint LS
        # and the 1D collinear fallback is unchanged.
        sol = None
        if abs(determinant) > 1e-6:
            A_mat = [
                [sum_s2, sum_se, sum_sw],
                [sum_se, sum_e2, sum_ew],
                [sum_sw, sum_ew, sum_w2],
            ]
            b_vec = [sum_s_I, sum_e_I, sum_w_I]
            sol = solve_gauss_jordan(A_mat, b_vec)

        if sol is not None:
            new_coeff_s = max(-SOLAR_COEFF_CAP, min(SOLAR_COEFF_CAP, sol[0]))
            new_coeff_e = max(-SOLAR_COEFF_CAP, min(SOLAR_COEFF_CAP, sol[1]))
            new_coeff_w = max(-SOLAR_COEFF_CAP, min(SOLAR_COEFF_CAP, sol[2]))
            return {
                "s": new_coeff_s * COLD_START_SOLAR_DAMPING,
                "e": new_coeff_e * COLD_START_SOLAR_DAMPING,
                "w": new_coeff_w * COLD_START_SOLAR_DAMPING,
            }

        # Collinear fallback: sun observed from a narrow angle range
        # (e.g. winter).  Project all observations onto the dominant
        # direction and solve 1D LS, then decompose back.  NLMS updates
        # will refine all components over time.
        dir_s = sum(item[0] for item in buffer_list)
        dir_e = sum(item[1] for item in buffer_list)
        dir_w = sum(item[2] for item in buffer_list)
        dir_norm = (dir_s ** 2 + dir_e ** 2 + dir_w ** 2) ** 0.5
        if dir_norm < 1e-6:
            return None

        d_s = dir_s / dir_norm
        d_e = dir_e / dir_norm
        d_w = dir_w / dir_norm

        sum_proj_I = sum(
            (item[0] * d_s + item[1] * d_e + item[2] * d_w) * item[3]
            for item in buffer_list
        )
        sum_proj2 = sum(
            (item[0] * d_s + item[1] * d_e + item[2] * d_w) ** 2
            for item in buffer_list
        )
        if sum_proj2 < 1e-6:
            return None

        c_scalar = max(-SOLAR_COEFF_CAP, min(SOLAR_COEFF_CAP, sum_proj_I / sum_proj2))
        return {
            "s": c_scalar * d_s * COLD_START_SOLAR_DAMPING,
            "e": c_scalar * d_e * COLD_START_SOLAR_DAMPING,
            "w": c_scalar * d_w * COLD_START_SOLAR_DAMPING,
        }

    def _learn_unit_model(
        self,
        entity_id: str,
        temp_key: str,
        wind_bucket: str,
        expected_unit_base: float,
        unit_normalized: float,
        learning_rate: float,
        learning_buffer_per_unit: dict,
        correlation_data_per_unit: dict,
        observation_counts: dict,
        rate_multiplier: float = 1.0,
    ):
        """Update correlation model for a specific unit (Buffered or EMA).

        rate_multiplier (#838): applied to the EMA rate (not the buffer
        phase) to support solar-headroom throttling.  Defaults to 1.0
        so aux-fallback and legacy callers keep existing behaviour.
        """

        # Check if we have actual learned data for this exact temp/wind combination
        has_exact_model = (
            entity_id in correlation_data_per_unit and
            temp_key in correlation_data_per_unit[entity_id] and
            wind_bucket in correlation_data_per_unit[entity_id][temp_key]
        )

        # --- Buffered Learning Logic (Cold Start Only) ---
        if not has_exact_model:
            # Cold Start Phase: Collect samples before initializing model
            # This ensures we don't create phantom data from wind fallbacks or TDD extrapolation
            if entity_id not in learning_buffer_per_unit:
                learning_buffer_per_unit[entity_id] = {}
            if temp_key not in learning_buffer_per_unit[entity_id]:
                learning_buffer_per_unit[entity_id][temp_key] = {}
            if wind_bucket not in learning_buffer_per_unit[entity_id][temp_key]:
                learning_buffer_per_unit[entity_id][temp_key][wind_bucket] = []

            buffer_list = learning_buffer_per_unit[entity_id][temp_key][wind_bucket]
            buffer_list.append(unit_normalized)

            # Check Threshold for Jump Start
            if len(buffer_list) >= LEARNING_BUFFER_THRESHOLD:
                avg_val = sum(buffer_list) / len(buffer_list)
                new_pred_unit = avg_val

                _LOGGER.info(f"Buffered Learning [Jump Start]: {entity_id} T={temp_key} W={wind_bucket} -> {new_pred_unit:.3f} kWh (Avg of {len(buffer_list)} samples)")

                self._update_unit_correlation(entity_id, temp_key, wind_bucket, new_pred_unit, correlation_data_per_unit)
                self._increment_observation_count(entity_id, temp_key, wind_bucket, observation_counts)

                # Clear buffer (no longer needed after jump start)
                buffer_list.clear()
            else:
                _LOGGER.debug(f"Buffered Learning [Collecting]: {entity_id} T={temp_key} W={wind_bucket} -> Sample {len(buffer_list)}/{LEARNING_BUFFER_THRESHOLD} ({unit_normalized:.3f} kWh)")

        else:
            # Post-Jump Start: Normal hourly EMA updates (no buffering)
            # Only runs if we have actual learned data for this temp/wind combo
            # Get the actual learned value (not fallback)
            current_model_val = correlation_data_per_unit[entity_id][temp_key][wind_bucket]

            # Cap per-unit learning rate at 3% to prevent oscillation on high-hysteresis units.
            # Headroom multiplier (#838) further throttles the rate when
            # unit_solar_impact approaches expected_unit_base — preventing
            # inflated solar normalization from drifting the base model up.
            unit_learning_rate = min(learning_rate, PER_UNIT_LEARNING_RATE_CAP) * rate_multiplier
            new_pred_unit = current_model_val + unit_learning_rate * (unit_normalized - current_model_val)

            _LOGGER.debug(f"Per-Unit Learning [EMA]: {entity_id} T={temp_key} W={wind_bucket} -> {new_pred_unit:.3f} kWh (was {current_model_val:.3f}, rate={unit_learning_rate:.1%})")

            self._update_unit_correlation(entity_id, temp_key, wind_bucket, new_pred_unit, correlation_data_per_unit)
            self._increment_observation_count(entity_id, temp_key, wind_bucket, observation_counts)

    def _learn_unit_aux_coefficient(
        self,
        entity_id: str,
        temp_key: str,
        wind_bucket: str,
        expected_unit_base: float,
        unit_normalized: float,
        learning_rate: float,
        aux_coefficients_per_unit: dict,
        learning_buffer_aux_per_unit: dict,
        correlation_data_per_unit: dict,
    ):
        """Update aux coefficient for a specific unit (Buffered or EMA)."""
        # Calculate Implied Reduction
        # Reduction = Base - Actual
        # If Base=0 (not learned yet), we can't learn reduction reliably (thermodynamically undefined).
        if expected_unit_base <= ENERGY_GUARD_THRESHOLD:
            return

        # CRITICAL: Only learn aux if we have actual base model data for this temp/wind
        # Don't learn aux based on fallback/extrapolated base values
        has_base_model = (
            entity_id in correlation_data_per_unit and
            temp_key in correlation_data_per_unit[entity_id] and
            wind_bucket in correlation_data_per_unit[entity_id][temp_key]
        )
        if not has_base_model:
            _LOGGER.debug(f"Skipping Unit Aux Learning: {entity_id} T={temp_key} W={wind_bucket} - No base model yet")
            return

        implied_reduction = expected_unit_base - unit_normalized
        # Clamp: Cannot be negative (Aux shouldn't INCREASE usage)
        implied_reduction = max(0.0, implied_reduction)

        # Get the actual base model value for this exact bucket (for clamping)
        base_model_value = correlation_data_per_unit[entity_id][temp_key][wind_bucket]
        # Clamp: Aux reduction cannot exceed base model (can't reduce below zero consumption)
        implied_reduction = min(implied_reduction, base_model_value)

        # Get Current Coefficient
        current_coeff = None
        if entity_id in aux_coefficients_per_unit:
            if temp_key in aux_coefficients_per_unit[entity_id]:
                if wind_bucket in aux_coefficients_per_unit[entity_id][temp_key]:
                    current_coeff = aux_coefficients_per_unit[entity_id][temp_key][wind_bucket]

        # --- Buffered Learning Logic (Cold Start) ---
        if current_coeff is None:
            # Cold Start Phase
            if entity_id not in learning_buffer_aux_per_unit:
                learning_buffer_aux_per_unit[entity_id] = {}
            if temp_key not in learning_buffer_aux_per_unit[entity_id]:
                learning_buffer_aux_per_unit[entity_id][temp_key] = {}
            if wind_bucket not in learning_buffer_aux_per_unit[entity_id][temp_key]:
                learning_buffer_aux_per_unit[entity_id][temp_key][wind_bucket] = []

            buffer_list = learning_buffer_aux_per_unit[entity_id][temp_key][wind_bucket]
            buffer_list.append(implied_reduction)

            if len(buffer_list) >= LEARNING_BUFFER_THRESHOLD:
                avg_val = sum(buffer_list) / len(buffer_list)
                new_coeff = min(avg_val, base_model_value)  # Clamp to base model

                _LOGGER.info(f"Buffered Unit Aux Learning [Jump Start]: {entity_id} T={temp_key} W={wind_bucket} -> {new_coeff:.3f} kW Reduction")
                self._update_unit_aux_coefficient(entity_id, temp_key, wind_bucket, new_coeff, aux_coefficients_per_unit)
                buffer_list.clear()
            else:
                 _LOGGER.debug(f"Buffered Unit Aux [Collecting]: {entity_id} T={temp_key} W={wind_bucket} -> Sample {len(buffer_list)}/{LEARNING_BUFFER_THRESHOLD} ({implied_reduction:.3f} kW)")

        else:
            # Post-Jump Start: EMA using global learning rate
            # Cap per-unit learning rate at 3% to prevent oscillation on high-hysteresis units
            unit_learning_rate = min(learning_rate, PER_UNIT_LEARNING_RATE_CAP)
            new_coeff = current_coeff + unit_learning_rate * (implied_reduction - current_coeff)
            new_coeff = max(0.0, new_coeff)
            new_coeff = min(new_coeff, base_model_value)  # Clamp to base model

            _LOGGER.debug(f"Per-Unit Aux Learning [EMA]: {entity_id} T={temp_key} W={wind_bucket} -> {new_coeff:.3f} kW (was {current_coeff:.3f}, rate={unit_learning_rate:.1%})")
            self._update_unit_aux_coefficient(entity_id, temp_key, wind_bucket, new_coeff, aux_coefficients_per_unit)

    def _update_unit_correlation(self, entity_id, temp_key, wind_bucket, value, correlation_data_per_unit):
        """Update the correlation data structure."""
        if entity_id not in correlation_data_per_unit:
            correlation_data_per_unit[entity_id] = {}
        if temp_key not in correlation_data_per_unit[entity_id]:
            correlation_data_per_unit[entity_id][temp_key] = {}

        correlation_data_per_unit[entity_id][temp_key][wind_bucket] = round(value, 5)

    def _update_unit_aux_coefficient(self, entity_id, temp_key, wind_bucket, value, aux_coefficients_per_unit):
        """Update the aux coefficient data structure."""
        if entity_id not in aux_coefficients_per_unit:
            aux_coefficients_per_unit[entity_id] = {}
        if temp_key not in aux_coefficients_per_unit[entity_id]:
            aux_coefficients_per_unit[entity_id][temp_key] = {}

        aux_coefficients_per_unit[entity_id][temp_key][wind_bucket] = round(value, 3)

    def _update_unit_solar_coefficient(
        self,
        entity_id: str,
        value: dict[str, float],
        solar_coefficients_per_unit: dict,
        regime: str,
        *,
        components: tuple[str, ...] = ("s", "e", "w"),
    ) -> None:
        """Update the solar coefficient for one (entity, regime) slot.

        ``regime`` is ``"heating"`` or ``"cooling"`` (#868).  The other
        regime is preserved unchanged on every write.  Initial entry for
        an entity creates both regimes as zero-vectors so subsequent
        reads on the unwritten regime return a stable shape.

        All components are non-negative-clamped (invariant #4): each
        represents solar gain through windows facing one cardinal
        direction, and a window can only receive gain.  Heating and
        cooling regimes share this clamp but converge to different
        absolute values (each absorbs its own ``E[1/COP]``).
        ``SOLAR_COEFF_CAP`` is applied upstream by the live learners
        (NLMS cold-start, inequality CAP clamp, batch-fit, apply-
        implied) before the value reaches this gate.

        ``components`` selects which keys to write through the gate.
        The 3D callers (NLMS, inequality, cold-start, batch-fit,
        apply-implied) pass the default 3-tuple and observe bit-
        identical pre-#954 behaviour.  The 4D shadow learner (#954)
        passes ``("s", "e", "w", "diffuse")`` and gets the diffuse
        component clamped identically.  All entries in ``components``
        must be present in ``value`` (default 0.0 if missing).

        Stamps ``learned: True`` on the written regime (#921) so the
        cooling cold-start gate (``_is_cooling_solar_cold_start``) can
        distinguish learner-written values from migration-seeded
        copies of the heating coefficient.  Legacy storage without
        the flag reads as unlearned and self-heals on the first
        learner write to that regime; no storage-version bump
        required (additive field, ``.get("learned", False)``
        defaulting works for legacy data).
        """
        if regime not in ("heating", "cooling"):
            raise ValueError(f"Invalid solar coefficient regime: {regime!r}")
        entry = solar_coefficients_per_unit.get(entity_id)
        if not isinstance(entry, dict):
            entry = {
                "heating": {"s": 0.0, "e": 0.0, "w": 0.0},
                "cooling": {"s": 0.0, "e": 0.0, "w": 0.0},
            }
            solar_coefficients_per_unit[entity_id] = entry
        else:
            if "heating" not in entry or not isinstance(entry["heating"], dict):
                entry["heating"] = {"s": 0.0, "e": 0.0, "w": 0.0}
            if "cooling" not in entry or not isinstance(entry["cooling"], dict):
                entry["cooling"] = {"s": 0.0, "e": 0.0, "w": 0.0}
        new_regime: dict = {
            comp: round(max(0.0, value.get(comp, 0.0)), 5)
            for comp in components
        }
        new_regime["learned"] = True
        entry[regime] = new_regime

    def _update_unit_solar_inequality(
        self,
        entity_id: str,
        expected_unit_base: float,
        battery_filtered_potential: tuple[float, ...],
        solar_coefficients_per_unit: dict | None,
        *,
        components: tuple[str, ...] = ("s", "e", "w"),
        margin: float = INEQUALITY_MARGIN,
        step_size: float = INEQUALITY_STEP_SIZE,
    ) -> str:
        """Inequality-constraint update for solar-shutdown hours (#865).

        A shutdown hour (``actual ≈ 0`` on a unit that has meaningful base
        demand while solar is high) carries a lower-bound signal:

            coeff · potential ≥ base

        The NLMS equality path discards these hours because ``actual_impact =
        base − actual ≈ base`` would inflate the coefficient unrealistically.
        This method uses the same hours with a different semantics: enforce
        the inequality via projected gradient, raising coefficients only,
        never lowering.  NLMS on modulating hours handles the downward
        correction — together, the two paths converge from below (inequality)
        and above (NLMS).

        Signal source is **battery-filtered** per-direction potential, not
        the instantaneous vector: shutdown is an accumulated phenomenon
        (the room overheated over hours), and the battery EMA matches that
        timescale (decay 0.80, half-life ~3.1h).

        Dimension-agnostic via ``components`` — defaults to the 3D solar
        coefficient ``("s", "e", "w")``; the 4D shadow path passes
        ``("s", "e", "w", "diffuse")`` along with the matching 4-tuple
        ``battery_filtered_potential`` and the 4D coefficient dict.  The
        4D variant is strict shadow (no live read-path consumer); both
        share the same one-sided, heating-only semantics (invariant #5).

        Parameters
        ----------
        entity_id:
            Unit being updated.
        expected_unit_base:
            Current per-unit bucket value (kWh).  Gates the update via
            ``SOLAR_SHUTDOWN_MIN_BASE`` at the call site.
        battery_filtered_potential:
            Per-direction battery EMA — ``(pot_s, pot_e, pot_w)`` for 3D
            or ``(pot_s, pot_e, pot_w, pot_d)`` for 4D.  Length must match
            ``len(components)``.
        solar_coefficients_per_unit:
            Current coefficients; mutated in place.  ``None`` returns
            ``"shadow_disabled"`` (for the 4D shadow path when the
            container hasn't been initialised yet).
        components:
            Dimension and key labels for the coefficient slot.  Default
            ``("s", "e", "w")`` preserves all pre-existing 3D call sites.
        margin:
            Conservative factor on the constraint (default 0.9).  Requires
            ``coeff·potential ≥ 0.9 · base`` rather than strict ``≥ base``.
            Prevents the learner from chasing the hard edge where
            shutdown detection might be a false positive.
        step_size:
            Projected-gradient step (default ``INEQUALITY_STEP_SIZE = 0.05``,
            half of NLMS_STEP_SIZE).  Conservative — a single anomaly hour
            cannot drive the coefficient far.

        Returns
        -------
        One of: ``"updated"`` (constraint violated, update applied),
        ``"non_binding"`` (constraint satisfied, no update),
        ``"zero_magnitude"`` (battery-filtered potential too small),
        ``"shadow_disabled"`` (coefficient container is None — 4D shadow
        path before the container is initialised).
        """
        if solar_coefficients_per_unit is None:
            return "shadow_disabled"

        n_dim = len(components)
        mag_total = sum(battery_filtered_potential)  # all non-negative by construction
        if mag_total <= 0.01:
            # Battery not yet populated, or very low sun — inequality has
            # no direction to learn against.  Skip rather than update.
            return "zero_magnitude"

        # Inequality is heating-only by design (#865).  Cooling shutdown
        # semantics invert the constraint and remain out of scope until a
        # cooling-heavy install requests it.  Read and write the heating
        # regime only — never touch cooling here.
        entity_coeffs = solar_coefficients_per_unit.get(entity_id)
        if isinstance(entity_coeffs, dict):
            heating_coeff = entity_coeffs.get("heating") or {}
        else:
            heating_coeff = {}
        coeffs = [float(heating_coeff.get(components[k], 0.0)) for k in range(n_dim)]

        predicted_impact = 0.0
        for k in range(n_dim):
            predicted_impact += coeffs[k] * battery_filtered_potential[k]
        constraint_target = margin * expected_unit_base

        if predicted_impact >= constraint_target:
            # One-sided (invariant #5): never lower coefficients.
            return "non_binding"

        # Constraint violated — distribute the deficit proportional to the
        # non-negative potential magnitude in each direction.  A direction
        # with zero potential this hour gets no update (the battery carries
        # history across hours, so a direction with recent activity still
        # gets updated when its own pot component is above zero).
        deficit = constraint_target - predicted_impact
        new_value: dict[str, float] = {}
        for k in range(n_dim):
            updated = coeffs[k] + step_size * deficit * (
                battery_filtered_potential[k] / mag_total
            )
            # Non-negative + CAP clamps consistent with NLMS path.
            new_value[components[k]] = max(0.0, min(SOLAR_COEFF_CAP, updated))

        self._update_unit_solar_coefficient(
            entity_id,
            new_value,
            solar_coefficients_per_unit,
            "heating",
            components=components,
        )
        return "updated"

    # ------------------------------------------------------------------
    # 4D shadow learner (#954)
    # ------------------------------------------------------------------
    @staticmethod
    def _solve_4x4_normal_equations(
        X_rows: list[tuple[float, float, float, float]],
        y_vals: list[float],
    ) -> tuple[float, float, float, float] | None:
        """Solve A x = b for a 4-variable LS problem via Gauss-Jordan.

        ``X_rows`` are samples [(s, e, w, d), ...] and ``y_vals`` are
        the corresponding scalar targets.  Builds the 4x4 normal matrix
        ``A = X^T X`` and vector ``b = X^T y``, then solves with
        partial-pivot Gauss-Jordan.  Returns ``None`` when the matrix
        is singular (``abs(pivot) < 1e-10`` on any step) so the caller
        can fall through without writing a non-physical coefficient.

        Pure Python — the codebase has no numpy dependency.
        """
        if len(X_rows) < 4 or len(X_rows) != len(y_vals):
            return None

        # Build the 4×4 normal matrix A = XᵀX and target b = Xᵀy.
        A = [[0.0] * 4 for _ in range(4)]
        b = [0.0] * 4
        for row, y in zip(X_rows, y_vals):
            for i in range(4):
                xi = row[i]
                for j in range(4):
                    A[i][j] += xi * row[j]
                b[i] += xi * y

        sol = solve_gauss_jordan(A, b, pivot_eps=1e-10)
        if sol is None:
            return None
        return (sol[0], sol[1], sol[2], sol[3])

    def _increment_observation_count(self, entity_id, temp_key, wind_bucket, observation_counts):
        """Increment observation count."""
        if entity_id not in observation_counts:
            observation_counts[entity_id] = {}
        if temp_key not in observation_counts[entity_id]:
            observation_counts[entity_id][temp_key] = {}
        if wind_bucket not in observation_counts[entity_id][temp_key]:
            observation_counts[entity_id][temp_key][wind_bucket] = 0

        observation_counts[entity_id][temp_key][wind_bucket] += 1

    def learn_from_historical_import(
        self,
        temp_key: str,
        wind_bucket: str,
        actual_kwh: float,
        is_aux_active: bool,
        correlation_data: dict,
        aux_coefficients: dict,
        learning_rate: float,
        get_predicted_kwh_fn: Callable[[str, str, float], float],
        actual_temp: float,
        solar_normalization_delta: float = 0.0,
        snr_weight: float = 1.0,
        solar_coefficients_per_unit: dict | None = None,
        energy_sensors: list[str] | None = None,
        unit_modes: dict[str, str] | None = None,
    ) -> str:
        """Process a single historical data point to train global models.

        Target is dark-equivalent ``actual_kwh + delta`` and the EMA step
        is scaled by ``snr_weight`` (#xxx).  Dark hours retain full rate,
        sunny hours contribute proportionally, shutdown hours zero.

        Base path uses a one-sided dark-equivalent floor (#930) gated by
        per-regime plausibility.  When at least one solar coefficient in
        an active learning regime is marked ``learned``, the lift toward
        ``actual_kwh + delta`` is enabled; otherwise the lift is
        suppressed and the legacy raw-actual EMA path applies.  The
        upward direction is bounded by ``dark_target`` and SOLAR_COEFF_CAP
        so a single inflated-coefficient hour cannot drive runaway.

        ``actual_kwh`` for the aux path is taken at face value — aux
        learning does not share the COP-ceiling / shutdown-contamination
        failure modes that motivated the SNR formulation.
        """
        dark_target = max(0.0, actual_kwh + solar_normalization_delta)

        lift_gate_open = True
        if solar_coefficients_per_unit is not None and energy_sensors:
            modes = unit_modes or {}
            lift_gate_open = any(
                solar_coefficients_per_unit.get(eid, {})
                    .get(_solar_coeff_regime(modes.get(eid, MODE_HEATING)) or "", {})
                    .get("learned")
                for eid in energy_sensors
                if _solar_coeff_regime(modes.get(eid, MODE_HEATING)) is not None
            )

        effective_rate = learning_rate * max(0.0, snr_weight)

        if is_aux_active:
            base_prediction = get_predicted_kwh_fn(temp_key, wind_bucket, actual_temp)
            if base_prediction <= ENERGY_GUARD_THRESHOLD:
                return "skipped_no_base_model"

            # Aux path always uses delta-compensated actual, see docstring.
            aux_target = max(0.0, actual_kwh + solar_normalization_delta)
            implied_aux_reduction = base_prediction - aux_target
            implied_aux_reduction = max(0.0, implied_aux_reduction)

            if temp_key not in aux_coefficients:
                aux_coefficients[temp_key] = {}
            current_coeff = aux_coefficients[temp_key].get(wind_bucket, 0.0)

            new_coeff = current_coeff + learning_rate * (implied_aux_reduction - current_coeff) if current_coeff != 0.0 else implied_aux_reduction
            aux_coefficients[temp_key][wind_bucket] = round(new_coeff, 3)
            return "updated_aux_model"

        else:
            current_pred = correlation_data.get(temp_key, {}).get(wind_bucket, 0.0)

            if effective_rate <= 0.0 and current_pred == 0.0:
                # SNR-weighted cold-start protection: a zero-weight hour
                # (all-shutdown) must not seed the bucket with actual≈0.
                # Return early without mutating correlation_data so tests
                # can assert "no bucket written".
                return "skipped_zero_weight"

            if temp_key not in correlation_data:
                correlation_data[temp_key] = {}

            if current_pred != 0.0:
                if lift_gate_open and current_pred < dark_target:
                    target = dark_target
                else:
                    target = actual_kwh
                # #967: inline EMA step.  ``helpers.compute_base_ema_step`` is the
                # named source-of-truth; this retrain path will be migrated in the
                # same pass as ``process_learning`` once 4D-primary observation
                # has stabilised.
                new_pred = current_pred + effective_rate * (target - current_pred)
            else:
                # Cold start: first non-zero sample seeds the bucket.
                # Seed with the dark-equivalent when the lift gate is open;
                # otherwise seed with the raw actual to avoid trusting an
                # unlearned coefficient on the very first sample.
                new_pred = dark_target if lift_gate_open else actual_kwh
            correlation_data[temp_key][wind_bucket] = round(new_pred, 5)
            return "updated_base_model"

    def apply_strategies_to_global_model(
        self,
        day_logs: list[dict],
        track_c_distribution: list[dict] | None,
        strategies: dict,
        model: "ModelState",
        learning_rate: float,
        balance_point: float,
        wind_threshold: float,
        extreme_wind_threshold: float,
        parse_datetime_fn,
    ) -> int:
        """Apply per-unit strategies to write hourly sums to the global model.

        Each sensor's strategy produces a kWh contribution per hour.
        The global model receives the sum of all contributions for each
        (temp_key, wind_bucket) pair via the standard buffer → jump-start
        → EMA pipeline.

        Moved from coordinator.py (#784) — pure model-writing logic with
        no Home Assistant dependencies.

        Returns the number of bucket updates written.
        """
        from .observation import WeightedSmear

        log_by_hour: dict[int, dict] = {e.get("hour", -1): e for e in day_logs}

        # Clear stale distribution from previous day on all synthetic
        # WeightedSmear strategies.
        for strategy in strategies.values():
            if isinstance(strategy, WeightedSmear) and strategy.use_synthetic:
                strategy.set_distribution(None)

        # Prepare WeightedSmear strategies with their data for this day.
        if track_c_distribution:
            dist_by_hour: dict[int, dict] = {}
            for entry in track_c_distribution:
                try:
                    entry_dt = parse_datetime_fn(entry["datetime"])
                    if entry_dt is not None:
                        dist_by_hour[entry_dt.hour] = entry
                except (KeyError, TypeError, ValueError):
                    continue

            for strategy in strategies.values():
                if isinstance(strategy, WeightedSmear) and strategy.use_synthetic:
                    strategy.set_distribution(dist_by_hour)

        # Compute normalised loss weights from hourly log weather data.
        # Uses abs(delta_t) so cooling hours (outdoor > balance_point) also
        # receive proportional weight (#792).  Solar multiplier is inverted
        # for cooling: sun increases cooling load instead of reducing it.
        raw_weights: list[float] = []
        for h in range(24):
            log_h = log_by_hour.get(h, {})
            inertia_t = log_h.get("inertia_temp")
            raw_t = log_h.get("temp")
            outdoor = inertia_t if inertia_t is not None else (raw_t if raw_t is not None else balance_point)
            delta_t = abs(balance_point - outdoor)
            eff_wind = log_h.get("effective_wind")
            effective_wind = eff_wind if eff_wind is not None else 0.0
            if effective_wind >= extreme_wind_threshold:
                wind_factor = 1.6
            elif effective_wind >= wind_threshold:
                wind_factor = 1.3
            else:
                wind_factor = 1.0
            solar_f = log_h.get("solar_factor")
            solar_f = solar_f if solar_f is not None else 0.0
            is_cooling_hour = outdoor > balance_point
            if is_cooling_hour:
                # Sun adds to cooling load — more sun = more weight
                solar_mult = max(0.1, 1.0 + solar_f * 0.5)
            else:
                # Sun reduces heating load — more sun = less weight
                solar_mult = max(0.0, 1.0 - solar_f)
            raw_weights.append(delta_t * wind_factor * solar_mult)
        total_weight = sum(raw_weights)
        norm_weights = [w / total_weight if total_weight > 0 else 1.0 / 24 for w in raw_weights]

        # Set daily totals for non-MPC WeightedSmear strategies.
        # Mode filtering (#789): only sum hours where the unit is in a
        # learning-eligible mode so excluded energy is not smeared.
        for strategy in strategies.values():
            if isinstance(strategy, WeightedSmear) and not strategy.use_synthetic:
                daily_total = 0.0
                for h in range(24):
                    entry = log_by_hour.get(h, {})
                    unit_modes = entry.get("unit_modes", {})
                    mode = unit_modes.get(strategy.sensor_id, MODE_HEATING)
                    if mode not in MODES_EXCLUDED_FROM_GLOBAL_LEARNING:
                        daily_total += entry.get("unit_breakdown", {}).get(strategy.sensor_id, 0.0)
                strategy.set_daily_total(daily_total)

        # Iterate hours and sum all strategy contributions per bucket.
        correlation_data = model.correlation_data
        learning_buffer = model.learning_buffer_global
        bucket_updates = 0

        for h in range(24):
            log_entry = log_by_hour.get(h, {})
            h_temp_key = log_entry.get("temp_key")
            if h_temp_key is None:
                continue
            # Re-bucketize live; stored wind_bucket is a cache of the
            # threshold at write-time.  See retrain.py:422 for rationale.
            # Fall back to the stored label when ``effective_wind`` is
            # missing (very old logs / partial imports).
            eff_w = log_entry.get("effective_wind")
            if isinstance(eff_w, (int, float)):
                if eff_w >= extreme_wind_threshold:
                    h_wind_bucket = "extreme_wind"
                elif eff_w >= wind_threshold:
                    h_wind_bucket = "high_wind"
                else:
                    h_wind_bucket = "normal"
            else:
                h_wind_bucket = log_entry.get("wind_bucket")
                if h_wind_bucket is None:
                    continue

            weight = norm_weights[h]

            # #854 F1: split strategy contributions by type so solar
            # normalization is applied only to DirectMeter (raw-electrical)
            # values.  WeightedSmear already embeds solar attenuation in its
            # synthetic values via the smearing weights — adding
            # `solar_normalization_delta` on top would be double-correction.
            #
            # On pure Track C installs this guard is a no-op in practice
            # (no non-MPC units → delta is always 0), but it protects
            # mixed installs (MPC + unmetered resistive backup) where
            # delta is nonzero and would otherwise contaminate the
            # MPC-synthetic portion of the hourly sum.
            direct_kwh = 0.0
            smear_kwh = 0.0
            has_direct = False
            for strategy in strategies.values():
                contrib = strategy.get_hourly_contribution(h, weight, log_entry)
                if contrib is None:
                    continue
                if isinstance(strategy, WeightedSmear):
                    smear_kwh += contrib
                else:
                    direct_kwh += contrib
                    has_direct = True

            # Apply saturation-aware solar normalization (#792) to the
            # DirectMeter sum only.  The delta was pre-computed from
            # non-MPC units' NLMS coefficients in Track A and stored in
            # the log; by construction it represents the solar correction
            # needed for raw-electrical values, not smeared-synthetic.
            # Suppressed entirely when no DirectMeter contributed this
            # hour — otherwise a nonzero delta would be applied against
            # direct_kwh=0 and leak into the total, inflating smear-only
            # (pure Track C) buckets.
            # #968: aggregation paths prefer the 4D delta whenever the log
            # entry carries it, falling back to 3D otherwise — independent
            # of ``experimental_4d_primary``.  The live read-path stays
            # flag-gated; this consumer of historical hourly_log entries
            # uses the best available signal per hour.
            solar_delta = log_entry.get(
                "solar_normalization_delta_4d",
                log_entry.get("solar_normalization_delta", 0.0),
            )
            if solar_delta and has_direct:
                direct_kwh = max(0.0, direct_kwh + solar_delta)

            total_kwh = direct_kwh + smear_kwh

            if total_kwh <= 0.0:
                continue

            # Buffer → jump-start → EMA.
            if h_temp_key not in correlation_data:
                correlation_data[h_temp_key] = {}
            current_pred = correlation_data[h_temp_key].get(h_wind_bucket, 0.0)

            if current_pred == 0.0:
                if h_temp_key not in learning_buffer:
                    learning_buffer[h_temp_key] = {}
                if h_wind_bucket not in learning_buffer[h_temp_key]:
                    learning_buffer[h_temp_key][h_wind_bucket] = []
                buffer_list = learning_buffer[h_temp_key][h_wind_bucket]
                buffer_list.append(total_kwh)
                if len(buffer_list) >= LEARNING_BUFFER_THRESHOLD:
                    avg_val = sum(buffer_list) / len(buffer_list)
                    correlation_data[h_temp_key][h_wind_bucket] = round(avg_val, 5)
                    buffer_list.clear()
                    bucket_updates += 1
                    _LOGGER.info(f"Strategy Learning [Jump Start]: T={h_temp_key} W={h_wind_bucket} -> {avg_val:.3f} kWh")
                else:
                    _LOGGER.debug(f"Strategy Learning [Buffering]: T={h_temp_key} W={h_wind_bucket} -> Sample {len(buffer_list)}/{LEARNING_BUFFER_THRESHOLD}")
            else:
                new_pred = current_pred + learning_rate * (total_kwh - current_pred)
                correlation_data[h_temp_key][h_wind_bucket] = round(new_pred, 5)
                bucket_updates += 1
                _LOGGER.debug(f"Strategy Learning [EMA]: T={h_temp_key} W={h_wind_bucket} -> {new_pred:.5f} kWh (was {current_pred:.5f})")

        return bucket_updates

    def replay_per_unit_models(
        self,
        day_entries: list[dict],
        strategies: dict,
        model: "ModelState",
        learning_rate: float,
        *,
        target_entity: str | None = None,
        dry_run: bool = False,
        reset_first: bool = False,
        wind_threshold: float | None = None,
        extreme_wind_threshold: float | None = None,
    ) -> dict | None:
        """Replay per-unit correlation models from hourly log entries.

        Each DirectMeter sensor's actual kWh is written to its per-unit
        correlation table via buffer → jump-start → EMA.  Needed so that
        ``isolate_sensor`` subtraction works after ``retrain_from_history(reset_first=True)``.
        WeightedSmear sensors are skipped.

        ``target_entity`` restricts the replay to a single DirectMeter
        sensor (used by the ``retrain_unit_from_history`` service).
        Default ``None`` preserves the original whole-installation behaviour.

        ``dry_run`` routes writes through deep copies and returns a
        diagnostic ``{"diff_summary", "buckets_changed", "entries_processed"}``
        without mutating ``model``.  Only meaningful with ``target_entity``
        set (the diff is computed for that entity's buckets).

        ``reset_first`` (only honoured when ``target_entity`` is set) clears
        the target entity's slice of ``correlation_per_unit`` and
        ``buffer_per_unit`` BEFORE the replay loop.  Cleared on the
        deep-copies in dry-run mode and on live state otherwise, so the
        reported diff matches what a real reset-then-replay would produce.

        Moved from coordinator.py (#784) — pure model-writing logic.
        """
        from copy import deepcopy

        from .observation import DirectMeter

        direct_sensors = [
            s for s in strategies.values()
            if isinstance(s, DirectMeter)
        ]
        if target_entity is not None:
            direct_sensors = [s for s in direct_sensors if s.sensor_id == target_entity]
            if not direct_sensors:
                return {
                    "diff_summary": {},
                    "buckets_changed": 0,
                    "entries_processed": 0,
                }
        if not direct_sensors:
            return None

        # Snapshot for diff diagnostic when we're reporting on a target
        # entity.  In dry-run we additionally route writes through copies
        # so model state is untouched.
        if target_entity is not None:
            before_snapshot = deepcopy(model.correlation_data_per_unit.get(target_entity, {}))
        else:
            before_snapshot = None
        if dry_run:
            correlation_per_unit = deepcopy(model.correlation_data_per_unit)
            buffer_per_unit = deepcopy(model.learning_buffer_per_unit)
        else:
            correlation_per_unit = model.correlation_data_per_unit
            buffer_per_unit = model.learning_buffer_per_unit

        if reset_first and target_entity is not None:
            correlation_per_unit.pop(target_entity, None)
            buffer_per_unit.pop(target_entity, None)

        for log_entry in day_entries:
            h_temp_key = log_entry.get("temp_key")
            if h_temp_key is None:
                continue
            # Re-bucketize live from stored effective_wind when thresholds
            # are supplied and the entry carries ``effective_wind``;
            # otherwise fall back to the stored label (legacy callers in
            # tests; very old logs without ``effective_wind``).
            # See retrain.py:422 for rationale.
            eff_w = log_entry.get("effective_wind")
            if (
                wind_threshold is not None
                and extreme_wind_threshold is not None
                and isinstance(eff_w, (int, float))
            ):
                if eff_w >= extreme_wind_threshold:
                    h_wind_bucket = "extreme_wind"
                elif eff_w >= wind_threshold:
                    h_wind_bucket = "high_wind"
                else:
                    h_wind_bucket = "normal"
            else:
                h_wind_bucket = log_entry.get("wind_bucket")
                if h_wind_bucket is None:
                    continue
            breakdown = log_entry.get("unit_breakdown", {})
            # Per #885: route cooling-mode samples to the dedicated
            # "cooling" wind-bucket, mirroring live-write semantics in
            # _process_per_unit_learning.  Without this, retrain from a
            # log containing cooling hours would pollute heating buckets.
            entry_unit_modes = log_entry.get("unit_modes", {}) or {}
            for strategy in direct_sensors:
                sid = strategy.sensor_id
                unit_kwh = breakdown.get(sid, 0.0)
                if unit_kwh <= 0.0:
                    continue
                effective_bucket = (
                    COOLING_WIND_BUCKET
                    if entry_unit_modes.get(sid) == MODE_COOLING
                    else h_wind_bucket
                )
                if sid not in correlation_per_unit:
                    correlation_per_unit[sid] = {}
                if h_temp_key not in correlation_per_unit[sid]:
                    correlation_per_unit[sid][h_temp_key] = {}
                cur = correlation_per_unit[sid][h_temp_key].get(effective_bucket, 0.0)
                if cur == 0.0:
                    if sid not in buffer_per_unit:
                        buffer_per_unit[sid] = {}
                    if h_temp_key not in buffer_per_unit[sid]:
                        buffer_per_unit[sid][h_temp_key] = {}
                    if effective_bucket not in buffer_per_unit[sid][h_temp_key]:
                        buffer_per_unit[sid][h_temp_key][effective_bucket] = []
                    buf = buffer_per_unit[sid][h_temp_key][effective_bucket]
                    buf.append(unit_kwh)
                    if len(buf) >= LEARNING_BUFFER_THRESHOLD:
                        correlation_per_unit[sid][h_temp_key][effective_bucket] = round(
                            sum(buf) / len(buf), 5
                        )
                        buf.clear()
                else:
                    new_val = cur + learning_rate * (unit_kwh - cur)
                    correlation_per_unit[sid][h_temp_key][effective_bucket] = round(new_val, 5)

        if target_entity is None:
            return None

        # Build diff diagnostic for the targeted entity.
        sid = target_entity
        before_buckets = before_snapshot or {}
        after_buckets = correlation_per_unit.get(sid, {})
        diffs: list[tuple[str, float, float, float]] = []
        all_temp_keys = set(before_buckets) | set(after_buckets)
        for tk in all_temp_keys:
            before_wb = before_buckets.get(tk, {}) or {}
            after_wb = after_buckets.get(tk, {}) or {}
            for wb in set(before_wb) | set(after_wb):
                before_v = float(before_wb.get(wb, 0.0))
                after_v = float(after_wb.get(wb, 0.0))
                if before_v != after_v:
                    diffs.append((f"{tk}/{wb}", before_v, after_v, after_v - before_v))
        diffs.sort(key=lambda x: abs(x[3]), reverse=True)
        capped = diffs[:200]
        diff_summary = {
            key: {"before": b, "after": a, "delta": d}
            for key, b, a, d in capped
        }
        return {
            "diff_summary": diff_summary,
            "buckets_changed": len(diffs),
            "entries_processed": len(day_entries),
        }

    # -------------------------------------------------------------------------
    # Retrain-time helpers (NLMS replay + on-the-fly solar_normalization_delta)
    # -------------------------------------------------------------------------

    @staticmethod
    def _reconstruct_potential(
        entry: dict,
        solar_calculator,
        screen_config,
    ) -> tuple[tuple[float, float, float], float]:
        """Return ((pot_s, pot_e, pot_w), vector_magnitude) for an entry.

        Mirrors the reconstruction used in live per-unit solar learning
        (learning.py:_process_per_unit_learning): divides each cardinal
        effective component by its direction's transmittance, using the
        coordinator's screen_config.  Returns zero vector + magnitude 0
        for entries without solar vector data.
        """
        correction = entry.get("correction_percent", 100.0)
        effective = (
            entry.get("solar_vector_s", 0.0),
            entry.get("solar_vector_e", 0.0),
            entry.get("solar_vector_w", 0.0),
        )
        pot_s, pot_e, pot_w = solar_calculator.reconstruct_potential_vector(
            effective, correction, screen_config
        )
        magnitude = (pot_s * pot_s + pot_e * pot_e + pot_w * pot_w) ** 0.5
        return (pot_s, pot_e, pot_w), magnitude

    def compute_on_the_fly_solar_delta(
        self,
        entry: dict,
        *,
        solar_calculator,
        screen_config,
        solar_coefficients_per_unit: dict,
        energy_sensors: list[str],
    ) -> float:
        """Recompute ``solar_normalization_delta`` for a single hour using
        the CURRENT solar coefficients rather than the stored value.

        ``solar_normalization_delta`` is live-computed as
        ``solar_heating_applied - solar_cooling_applied`` from
        ``calculate_total_power``'s saturation-aware breakdown.  During
        retrain we don't replay that full pipeline — we approximate by
        summing per-unit ``coeff · potential`` and signing by mode.  The
        approximation ignores saturation (solar impact clamped at unit
        base demand).  In heating-dominated regimes during spring retrain
        this is accurate; saturation is rare on mild days.  The stored
        delta this replaces carried far worse error: it was computed with
        whatever (possibly wildly wrong) coefficient was live at log time.

        Mode resolution mirrors :func:`_compute_excluded_mode_energy` and
        :func:`statistics.py._get_unit_mode_for_hour`: the log's
        ``unit_modes`` dict only stores entries whose mode is NOT
        ``MODE_HEATING`` (coordinator.py filters the default out to
        reduce log clutter).  Iterating ``unit_modes.items()`` therefore
        walks only the non-heating minority — on any all-heating hour
        (the dominant case) the dict is empty and the function returns
        zero.  Historical bug: retrain's Pass 3 refinement silently
        reverted to Pass 1 priming behaviour because this function
        always returned 0 for the typical hour.  The fix iterates
        ``energy_sensors`` (the canonical list of tracked units) and
        resolves each sensor's mode by treating absence from
        ``unit_modes`` as ``MODE_HEATING`` implicitly.
        """
        (pot_s, pot_e, pot_w), magnitude = self._reconstruct_potential(
            entry, solar_calculator, screen_config
        )
        if magnitude <= 0.01:
            return 0.0
        unit_modes = entry.get("unit_modes", {}) or {}
        heating_total = 0.0
        cooling_total = 0.0
        for entity_id in energy_sensors:
            mode = unit_modes.get(entity_id, MODE_HEATING)
            regime = _solar_coeff_regime(mode)
            if regime is None:
                # OFF / DHW / unknown — no solar contribution
                continue
            entity_coeffs = solar_coefficients_per_unit.get(entity_id)
            if not isinstance(entity_coeffs, dict):
                continue
            coeff = entity_coeffs.get(regime)
            if not isinstance(coeff, dict):
                continue
            impact = max(
                0.0,
                coeff.get("s", 0.0) * pot_s
                + coeff.get("e", 0.0) * pot_e
                + coeff.get("w", 0.0) * pot_w,
            )
            if mode in (MODE_HEATING, MODE_GUEST_HEATING):
                heating_total += impact
            elif mode in (MODE_COOLING, MODE_GUEST_COOLING):
                cooling_total += impact
        return heating_total - cooling_total

    def replay_solar_nlms(
        self,
        entries: list[dict],
        *,
        solar_calculator,
        screen_config,
        correlation_data_per_unit: dict,
        solar_coefficients_per_unit: dict,
        learning_buffer_solar_per_unit: dict,
        energy_sensors: list[str],
        learning_rate: float,
        balance_point: float,
        aux_affected_entities: list | None = None,
        unit_strategies: dict | None = None,
        daily_history: dict | None = None,
        return_diagnostics: bool = False,
        unit_min_base: dict[str, float] | None = None,
        screen_affected_entities: frozenset[str] | None = None,
        solar_affected_entities: frozenset[str] | None = None,
        wind_threshold: float | None = None,
        extreme_wind_threshold: float | None = None,
    ):
        """Re-run NLMS solar coefficient learning over historical entries.

        Iterates qualifying sunny hours, reconstructs potential vectors
        from the stored effective + correction, and calls the same
        :meth:`_learn_unit_solar_coefficient` update path used by live
        learning.

        Unit base reference (``expected_unit_base``) resolution follows
        live-learning semantics per strategy type:

        - ``DirectMeter`` sensors use ``correlation_data_per_unit``
          (populated by the preceding base-replay pass).
        - ``WeightedSmear(use_synthetic=True)`` sensors (i.e. the MPC-
          managed sensor on a Track C installation) use
          ``daily_history[date_str]["track_c_distribution"][hour]
          .synthetic_kwh_el`` — the same MPC-derived thermal/COP value
          live learning would have seen via ``calculate_total_power``'s
          per-unit breakdown.  Comparing against per-unit correlation
          would be apple-to-pear because Track C never populates
          per-unit for WeightedSmear sensors (``_replay_per_unit_models``
          only handles DirectMeter).  On Track B fallback days (no
          ``track_c_distribution`` stored), we fall back to per-unit —
          which for the MPC sensor is likely partial raw electrical and
          below threshold, so NLMS skips that hour rather than guess.

        Returns the number of per-unit coefficient updates attempted.
        When ``return_diagnostics=True``, returns a dict with the update
        count plus per-reason skip counters — useful when the integer
        return looks suspicious (e.g. zero on a real install that was
        expected to have qualifying hours).
        """
        from .observation import DirectMeter, WeightedSmear
        aux_set = set(aux_affected_entities or [])
        strategies = unit_strategies or {}
        # ``daily_history`` retained in signature for call-site stability;
        # no longer consumed because WeightedSmear sensors are skipped
        # rather than attempting a synthetic-based NLMS signal (see
        # WeightedSmear-skip block below).
        _ = daily_history
        updates = 0
        # Per-reason skip counters for diagnostics.  Names describe the
        # first guard that rejected the entry/unit; a single hour can be
        # counted in at most one bucket per unit.
        diag = {
            "entries_considered": 0,
            "entry_skipped_aux": 0,
            "entry_skipped_poisoned": 0,
            "entry_skipped_disabled": 0,  # learning_status == "disabled"
            "entry_skipped_low_magnitude": 0,
            "entry_skipped_missing_temp_key": 0,
            "unit_skipped_aux_list": 0,
            "unit_skipped_shutdown": 0,
            "unit_skipped_excluded_mode": 0,
            "unit_skipped_weighted_smear": 0,  # MPC-managed; no coherent solar signal
            "unit_skipped_below_threshold": 0,
            "inequality_updates": 0,            # #865
            "inequality_non_binding": 0,        # constraint satisfied, no update
            "inequality_skipped_low_battery": 0,  # battery not yet populated
            "inequality_skipped_mode": 0,       # cooling/OFF — out of scope
            "inequality_skipped_base": 0,       # below SOLAR_SHUTDOWN_MIN_BASE
        }

        # Local per-direction potential battery (#865).  Chronological
        # EMA over the log mirrors coordinator-side state; replay starts
        # from zeros because we have no earlier context.  Decay fixed at
        # SOLAR_BATTERY_DECAY default — replay does not thread the
        # user's calibrated value (rare tuning, not worth the coupling
        # for a retrain-only path).
        battery_s = 0.0
        battery_e = 0.0
        battery_w = 0.0
        battery_decay = SOLAR_BATTERY_DECAY

        for entry in entries:
            diag["entries_considered"] += 1

            # Reconstruct hour potential and update battery FIRST, before
            # any entry-skip filter.  Live coordinator decays the battery
            # every hour regardless of aux/poisoned/dark state — replay
            # must match or aux-stretches and poisoned periods leave a
            # stale battery vector that inflates the inequality update on
            # the next qualifying shutdown hour.
            (pot_s, pot_e, pot_w), magnitude = self._reconstruct_potential(
                entry, solar_calculator, screen_config
            )
            battery_s = battery_s * battery_decay + pot_s * (1 - battery_decay)
            battery_e = battery_e * battery_decay + pot_e * (1 - battery_decay)
            battery_w = battery_w * battery_decay + pot_w * (1 - battery_decay)

            if entry.get("auxiliary_active", False):
                diag["entry_skipped_aux"] += 1
                continue
            # Match live ``_is_poisoned`` semantics (coordinator.py:922-930):
            # all three statuses must be skipped for solar NLMS replay or the
            # retrain produces a coefficient grounded in user-disabled or
            # data-poisoned hours that live learning would never have seen.
            # Counters are split so post-retrain diagnostics distinguish a
            # user-toggle ("disabled") from a data-quality skip ("poisoned").
            status = entry.get("learning_status", "unknown")
            if status == "disabled":
                diag["entry_skipped_disabled"] += 1
                continue
            if status.startswith("skipped_") or status == "cooldown_post_aux":
                diag["entry_skipped_poisoned"] += 1
                continue

            if magnitude <= 0.1:
                diag["entry_skipped_low_magnitude"] += 1
                continue
            temp_key = entry.get("temp_key")
            if temp_key is None:
                diag["entry_skipped_missing_temp_key"] += 1
                continue
            # Re-bucketize live when thresholds supplied and the entry
            # carries ``effective_wind``; fall back to the stored label
            # otherwise (legacy callers; very old logs).
            # See retrain.py:422 for rationale.
            eff_w = entry.get("effective_wind")
            if (
                wind_threshold is not None
                and extreme_wind_threshold is not None
                and isinstance(eff_w, (int, float))
            ):
                if eff_w >= extreme_wind_threshold:
                    wind_bucket = "extreme_wind"
                elif eff_w >= wind_threshold:
                    wind_bucket = "high_wind"
                else:
                    wind_bucket = "normal"
            else:
                wind_bucket = entry.get("wind_bucket", "normal")
            unit_modes = entry.get("unit_modes", {}) or {}
            unit_breakdown = entry.get("unit_breakdown", {}) or {}
            shutdown_entities = set(entry.get("solar_dominant_entities", []) or [])
            avg_temp = entry.get("temp", 0.0) or 0.0

            for entity_id in energy_sensors:
                # Per-entity solar-scope gate (#962).  Excluded entities
                # do not participate in solar replay at all — silently
                # skip without inflating shutdown counters.
                if (
                    solar_affected_entities is not None
                    and entity_id not in solar_affected_entities
                ):
                    continue
                # aux_affected_entities is NOT a solar-NLMS exclusion list.
                # Live learning only uses it for cooldown-path aux coefficient
                # handling (learning.py:_process_per_unit_learning lines
                # 446-451 and 551-556).  Hours where aux itself was active
                # are already filtered at the entry level via
                # auxiliary_active.  Historical bug: an earlier version of
                # this replay skipped entity_id in aux_affected_entities
                # unconditionally, which blocked 100 % of solar hours on
                # installs where the config-flow default (= all energy
                # sensors) had left aux_affected_entities == energy_sensors.
                # The counter below is retained (always 0 post-fix) so
                # regressions that reintroduce the bug are immediately
                # visible in diagnostics.
                if entity_id in shutdown_entities:
                    # Inequality learning: replace the NLMS skip with an
                    # inequality update that uses the battery-filtered
                    # potential.  Requires heating mode and meaningful
                    # base demand — same gates as the live wiring in
                    # _process_per_unit_learning.
                    unit_mode_sd = unit_modes.get(entity_id, MODE_HEATING)
                    if unit_mode_sd != MODE_HEATING:
                        diag["inequality_skipped_mode"] += 1
                        diag["unit_skipped_shutdown"] += 1
                        continue
                    # Per-entity screen scope: inequality is only applied to
                    # screen-affected entities.  Mirrors the live path —
                    # unscreened entities already have correct NLMS signal
                    # and the coordinator battery is reconstructed with the
                    # installation screen_config (wrong for unscreened).
                    if (
                        screen_affected_entities is not None
                        and entity_id not in screen_affected_entities
                    ):
                        diag["inequality_skipped_mode"] += 1
                        diag["unit_skipped_shutdown"] += 1
                        continue
                    # Resolve base via same per-unit path as NLMS branch
                    # below.  WeightedSmear sensors: same as NLMS — skip.
                    strategy_sd = strategies.get(entity_id)
                    if isinstance(strategy_sd, WeightedSmear) and strategy_sd.use_synthetic:
                        diag["unit_skipped_weighted_smear"] = diag.get(
                            "unit_skipped_weighted_smear", 0
                        ) + 1
                        continue
                    # Inequality path is already gated to MODE_HEATING above,
                    # so effective_bucket == wind_bucket here — defensive
                    # override retained for parity with the NLMS branch.
                    effective_bucket_sd = (
                        COOLING_WIND_BUCKET
                        if unit_modes.get(entity_id) == MODE_COOLING
                        else wind_bucket
                    )
                    unit_buckets_sd = correlation_data_per_unit.get(entity_id, {}).get(temp_key, {})
                    expected_base_sd = unit_buckets_sd.get(effective_bucket_sd, 0.0) if unit_buckets_sd else 0.0
                    shutdown_threshold_sd = _resolve_min_base(
                        entity_id, unit_min_base, SOLAR_SHUTDOWN_MIN_BASE
                    )
                    if expected_base_sd < shutdown_threshold_sd:
                        diag["inequality_skipped_base"] += 1
                        diag["unit_skipped_shutdown"] += 1
                        continue
                    ineq_status = self._update_unit_solar_inequality(
                        entity_id=entity_id,
                        expected_unit_base=expected_base_sd,
                        battery_filtered_potential=(battery_s, battery_e, battery_w),
                        solar_coefficients_per_unit=solar_coefficients_per_unit,
                    )
                    if ineq_status == "updated":
                        diag["inequality_updates"] += 1
                    elif ineq_status == "non_binding":
                        diag["inequality_non_binding"] += 1
                    else:  # "zero_magnitude"
                        diag["inequality_skipped_low_battery"] += 1
                    # Inequality path does not run NLMS on this (entity, hour).
                    continue
                unit_mode = unit_modes.get(entity_id, MODE_HEATING)
                if unit_mode in MODES_EXCLUDED_FROM_GLOBAL_LEARNING:
                    diag["unit_skipped_excluded_mode"] += 1
                    continue
                # Resolve the unit's expected base per live-learning
                # semantics.
                #
                # WeightedSmear(use_synthetic=True) sensors (MPC-managed in
                # Track C) are SKIPPED entirely to mirror live exclusion at
                # coordinator.py:4391-4397 (#776).  Rationale: solar NLMS
                # requires a dark-equivalent baseline compared against a
                # with-sun actual.  MPC provides only with-sun data —
                # ``kwh_th_sh`` (thermal delivered that day) and
                # ``kwh_el_sh`` (electrical consumed that day).  The
                # Track C smearing in :mod:`thermodynamics` applies
                # ``solar_factor`` weighting so ``synthetic_kwh_el``
                # already has solar implicit in its per-hour attribution;
                # after Jensen-rescaling (``sum(synthetic_kwh_el) ==
                # total_kwh_el``) the per-hour difference against
                # ``kwh_el_sh`` encodes COP variance, not solar.  There is
                # no coherent pure-MPC dark baseline, therefore no
                # meaningful NLMS signal for MPC sensors.  Solar
                # coefficient for these sensors falls back to
                # ``DEFAULT_SOLAR_COEFF_HEATING`` via
                # :meth:`SolarCalculator.calculate_unit_coefficient`.
                #
                # DirectMeter sensors (including any non-MPC units on a
                # Track C install) read their per-unit correlation
                # normally.
                strategy = strategies.get(entity_id)
                if isinstance(strategy, WeightedSmear) and strategy.use_synthetic:
                    diag["unit_skipped_weighted_smear"] = diag.get(
                        "unit_skipped_weighted_smear", 0
                    ) + 1
                    continue
                # Per #885: cooling-mode entities read from the dedicated
                # "cooling" wind-bucket; otherwise use the hour's actual
                # wind bucket.  Mirrors live-read semantics in
                # coordinator._get_predicted_kwh_per_unit.
                effective_bucket = (
                    COOLING_WIND_BUCKET
                    if unit_mode == MODE_COOLING
                    else wind_bucket
                )
                unit_buckets = correlation_data_per_unit.get(entity_id, {}).get(temp_key, {})
                expected_unit_base = unit_buckets.get(effective_bucket, 0.0) if unit_buckets else 0.0
                nlms_threshold = _resolve_min_base(
                    entity_id, unit_min_base, SOLAR_LEARNING_MIN_BASE
                )
                if expected_unit_base < nlms_threshold:
                    diag["unit_skipped_below_threshold"] += 1
                    continue
                actual_unit = unit_breakdown.get(entity_id, 0.0)
                # Per-entity screen routing: mirrors live learning.  Entities
                # not in screen_affected_entities learn against the effective
                # vector directly — no reconstruction needed because
                # transmittance=1.0 per direction reduces to effective/1.0.
                if (
                    screen_affected_entities is None
                    or entity_id in screen_affected_entities
                ):
                    entity_pot = (pot_s, pot_e, pot_w)
                else:
                    entity_pot = (
                        entry.get("solar_vector_s", 0.0),
                        entry.get("solar_vector_e", 0.0),
                        entry.get("solar_vector_w", 0.0),
                    )
                self._learn_unit_solar_coefficient(
                    entity_id=entity_id,
                    temp_key=temp_key,
                    expected_unit_base=expected_unit_base,
                    actual_unit=actual_unit,
                    avg_solar_vector=entity_pot,
                    learning_rate=learning_rate,
                    solar_coefficients_per_unit=solar_coefficients_per_unit,
                    learning_buffer_solar_per_unit=learning_buffer_solar_per_unit,
                    avg_temp=avg_temp,
                    balance_point=balance_point,
                    unit_mode=unit_mode,
                )
                updates += 1
        if return_diagnostics:
            diag["updates"] = updates
            return diag
        return updates

    def batch_fit_solar_coefficients(
        self,
        hourly_log: list[dict],
        solar_coefficients_per_unit: dict,
        energy_sensors: list[str],
        coordinator,
        *,
        entity_id_filter: str | None = None,
        unit_min_base: dict[str, float] | None = None,
        screen_affected_entities: frozenset[str] | None = None,
        days_back: int | None = None,
        dry_run: bool = False,
        seed_live_window: bool = False,
    ) -> dict:
        """Periodic batch Tobit MLE fit per (entity, mode) regime (#904 stage 2).

        Solves the Type-I right-censored Gaussian regression over the
        modulating-regime hourly log to extract a coefficient that
        NLMS and inequality cannot — specifically the mild-weather
        catch-22 where expected base demand is near zero (e.g. west-
        facing rooms whose solar peak coincides with the daily
        temperature maximum), and the saturation-information loss
        that the original (#884) least-squares form discarded by
        dropping right-censored samples (HP fully off because the
        room got warm).  Tobit's Mills-ratio likelihood term
        recovers slope information from the censoring point itself.
        See ``_solve_tobit`` for the solver and #904 for the
        stage-1 evidence motivating the swap from LS to Tobit.

        Per (entity, regime): heating-mode hours fit the heating regime
        only; cooling-mode hours fit the cooling regime only.  Each
        regime absorbs its own ``E[1/COP_mode]`` (#868 invariant).
        Inequality-flagged shutdown hours are excluded — they violate
        the modulating-regime assumption (CHOICE 3 in #904); saturated
        hours are kept as right-censored samples (CHOICE 2).

        ``days_back`` (default ``None`` = full log) restricts the input
        window.  Without this, a fit on a fresh upgrade would absorb
        pre-upgrade data and pull coefficients toward old model
        behaviour (e.g. pre-1.3.3 transmittance constants); a 14- or
        30-day window after a major release fits against representative
        post-upgrade data.

        ``dry_run`` (default ``False``) runs every gate and the Tobit
        solve but does not write coefficients back to
        ``solar_coefficients_per_unit``.  The diagnostic dict reports
        ``coefficient_after`` so the user can preview the result; a
        subsequent (non-dry-run) call applies the same fit.  Useful
        for sanity-checking before letting the fit drift live state.

        Sample filter (per (entity, regime)):
        - ``unit_modes[entity_id]`` resolves to ``regime`` (heating
          regime takes HEATING + GUEST_HEATING; cooling takes
          COOLING + GUEST_COOLING).  OFF / DHW are skipped.
        - Entity not in ``solar_dominant_entities`` (shutdown — owned
          by the inequality learner, CHOICE 3).
        - Auxiliary not active during the hour.
        - Reconstructed potential vector magnitude > 0.01.
        - ``expected_unit_base > 0`` (Tobit drops the per-unit min-base
          gate that LS used — the censoring threshold ``T = 0.95×base``
          must be well-defined, but Tobit needs the mild-weather
          shoulder hours that the live-NLMS gate would exclude).
        - ``actual_impact > 0`` (clamped negative impacts).
        - Saturated rows (``actual_impact ≥ BATCH_FIT_SATURATION_RATIO
          × base``) are kept and tagged ``censored_mask = True`` with
          ``value = T``; unsaturated rows kept with
          ``value = base − actual``.

        Algorithm:
        - Tobit MLE via ``_solve_tobit``: projected-Newton on
          ``(c_S, c_E, c_W, log σ)`` with active-set non-negativity
          (invariant #4), trust-region clip on Mills-ratio q, warm-
          start from 3×3 LS on uncensored rows.
        - Sample-size gate: ``|U| ≥ TOBIT_MIN_UNCENSORED (20)``
          (pre-fit; σ identifiability) and ``n_eff ≥ TOBIT_MIN_NEFF
          (40)`` (post-fit; slope identifiability) per #904 CHOICE 4.
        - Convergence required (``did_not_converge`` skip otherwise).
        - Clamp components to ``[0, SOLAR_COEFF_CAP]`` (defence-in-
          depth; the solver's active-set already enforces ≥0).
        - Damp blend: ``new = α × tobit + (1 − α) × current`` with
          ``α = BATCH_FIT_DAMPING``.  When the regime has not learned
          before (current is zero), the Tobit result is written
          directly without damping — there is no prior to blend.

        Returns
        -------
        Diagnostics dict ``{entity_id: {regime: {...}}}`` reporting
        per slot: ``sample_count``, ``residual_rmse_kwh`` (uncensored
        rows only — censored values store ``T`` not observed impact),
        ``coefficient_before``, ``coefficient_after``,
        ``damping_applied``, ``tobit_diagnostics`` (iterations,
        convergence, sigma, n_uncensored, n_censored,
        censored_fraction, n_eff, log_likelihood), and
        ``skip_reason`` ∈ {``insufficient_uncensored``,
        ``insufficient_effective_samples``, ``did_not_converge``,
        ``warm_start_failed``} when the fit was not applied.
        """
        from .observation import WeightedSmear  # local import — heavy module
        strategies = getattr(coordinator, "_unit_strategies", {}) or {}

        # Time-window filter (#884 follow-up).  When the user runs the
        # service shortly after a major release, a full-log fit absorbs
        # pre-upgrade data and pulls coefficients toward old model
        # behaviour.  Filtering once here (rather than per-entity)
        # avoids re-iterating timestamps inside each regime collector.
        # ``_filter_log_by_days_back`` parses datetimes properly —
        # don't lex-compare ISO strings, they break under non-UTC
        # offsets near the cutoff boundary.
        filtered_log = _filter_log_by_days_back(hourly_log, days_back)

        results: dict[str, dict] = {}
        target_entities = (
            [entity_id_filter] if entity_id_filter else list(energy_sensors)
        )

        # Solar-affected scope (#962): batch fit honours the same per-entity
        # gate as the live learners.  None sentinel = legacy / test path =
        # all entities included.
        solar_affected = getattr(coordinator, "_solar_affected_set", None)
        if not isinstance(solar_affected, (frozenset, set)):
            solar_affected = None

        for entity_id in target_entities:
            if entity_id not in energy_sensors:
                results[entity_id] = {
                    "skip_reason": "unknown_entity",
                }
                continue

            if solar_affected is not None and entity_id not in solar_affected:
                results[entity_id] = {
                    "skip_reason": "excluded_from_solar",
                }
                continue

            # MPC-managed sensors have no coherent dark-equivalent
            # baseline — same exclusion rule as ``replay_solar_nlms``.
            strategy = strategies.get(entity_id)
            if isinstance(strategy, WeightedSmear) and strategy.use_synthetic:
                results[entity_id] = {
                    "skip_reason": "weighted_smear_excluded",
                }
                continue

            unit_threshold = _resolve_min_base(
                entity_id, unit_min_base, SOLAR_LEARNING_MIN_BASE
            )

            # Pre-compute potential reconstruction once per (entity, entry).
            # The reconstruction depends on entry data + per-entity screen
            # config but is regime-independent, so hoisting it outside the
            # heating/cooling loop saves ~50 % of the reconstruction calls
            # on a 365-day-retention install.
            scr_fn = getattr(coordinator, "screen_config_for_entity", None)
            if scr_fn is not None:
                screen_cfg_for_entity = scr_fn(entity_id)
            else:
                screen_cfg_for_entity = getattr(coordinator, "screen_config", None)
            entry_potentials: list[tuple[float, float, float, float]] = []
            for entry in filtered_log:
                (pot_s, pot_e, pot_w), magnitude = self._reconstruct_potential(
                    entry,
                    getattr(coordinator, "solar", None),
                    screen_cfg_for_entity,
                )
                entry_potentials.append((pot_s, pot_e, pot_w, magnitude))

            entity_results: dict[str, dict] = {}
            for regime in ("heating", "cooling"):
                samples, censored_mask, drop_counts = self._collect_batch_fit_samples(
                    entity_id=entity_id,
                    regime=regime,
                    hourly_log=filtered_log,
                    entry_potentials=entry_potentials,
                    coordinator=coordinator,
                    unit_threshold=unit_threshold,
                    screen_affected_entities=screen_affected_entities,
                    for_tobit=True,
                    solar_coefficients_per_unit=solar_coefficients_per_unit,
                )
                n_unc = sum(1 for m in censored_mask if not m)
                n_cens = len(samples) - n_unc
                regime_diag: dict = {
                    "sample_count": len(samples),
                    "drop_counts": drop_counts,
                }

                # Snapshot the current coefficient for blending + reporting.
                entity_entry = solar_coefficients_per_unit.get(entity_id)
                if isinstance(entity_entry, dict):
                    current_dict = entity_entry.get(regime) or {}
                else:
                    current_dict = {}
                current = {
                    "s": float(current_dict.get("s", 0.0)),
                    "e": float(current_dict.get("e", 0.0)),
                    "w": float(current_dict.get("w", 0.0)),
                }
                regime_diag["coefficient_before"] = {
                    k: round(v, 5) for k, v in current.items()
                }

                # Pre-fit gate: σ identifiability requires ``|U| ≥ 20``
                # (#904 CHOICE 4-B).  Below this floor we cannot warm-
                # start Tobit's MLE — skip without invoking the solver.
                if n_unc < TOBIT_MIN_UNCENSORED:
                    regime_diag["skip_reason"] = "insufficient_uncensored"
                    regime_diag["coefficient_after"] = regime_diag[
                        "coefficient_before"
                    ]
                    regime_diag["tobit_diagnostics"] = {
                        "n_uncensored": n_unc,
                        "n_censored": n_cens,
                        "censored_fraction": (
                            round(n_cens / len(samples), 3)
                            if samples
                            else 0.0
                        ),
                    }
                    entity_results[regime] = regime_diag
                    continue

                fit = LearningManager._solve_tobit(
                    samples, censored_mask, components=("s", "e", "w")
                )
                if fit is None:
                    regime_diag["skip_reason"] = "warm_start_failed"
                    regime_diag["coefficient_after"] = regime_diag[
                        "coefficient_before"
                    ]
                    entity_results[regime] = regime_diag
                    continue

                tobit_diag = {
                    "iterations": fit["iterations"],
                    "converged": bool(fit["converged"]),
                    "failure_reason": fit.get("failure_reason"),
                    "sigma": round(fit["sigma"], 5),
                    "log_likelihood": round(fit["log_likelihood"], 4),
                    "n_uncensored": fit["n_uncensored"],
                    "n_censored": fit["n_censored"],
                    "censored_fraction": (
                        round(n_cens / len(samples), 3) if samples else 0.0
                    ),
                    "n_eff": round(fit["n_eff"], 2),
                }
                regime_diag["tobit_diagnostics"] = tobit_diag

                # Post-fit gate: slope identifiability requires
                # ``n_eff = |U| + Σ_C λ(q)(λ(q)−q) ≥ 40`` — censored
                # samples that sit far inside the censoring region
                # (q ≪ 0) contribute λ(q) ≈ 0 and add no slope info,
                # so we cannot count raw |C| toward the effective
                # sample size.  Convergence failure is treated as a
                # separate skip (the last iterate may still be
                # informative for inspection).
                if fit["n_eff"] < TOBIT_MIN_NEFF:
                    regime_diag["skip_reason"] = "insufficient_effective_samples"
                    regime_diag["coefficient_after"] = regime_diag[
                        "coefficient_before"
                    ]
                    entity_results[regime] = regime_diag
                    continue

                if not fit["converged"]:
                    regime_diag["skip_reason"] = "did_not_converge"
                    regime_diag["coefficient_after"] = regime_diag[
                        "coefficient_before"
                    ]
                    entity_results[regime] = regime_diag
                    continue

                # Clamp to [0, CAP] — invariant #4.  The solver's
                # active-set already enforces non-negativity, but the
                # CAP clamp is preserved as defence-in-depth against
                # warm-start drift on pathological synthetic inputs.
                clamped = {
                    "s": max(0.0, min(SOLAR_COEFF_CAP, float(fit["s"]))),
                    "e": max(0.0, min(SOLAR_COEFF_CAP, float(fit["e"]))),
                    "w": max(0.0, min(SOLAR_COEFF_CAP, float(fit["w"]))),
                }

                # Damp blend.  When the regime has no prior (current is
                # all zeros), write the batch result directly — there is
                # no information to preserve via damping.
                has_prior = any(current[k] > 1e-6 for k in ("s", "e", "w"))
                if has_prior:
                    blended = {
                        k: round(
                            BATCH_FIT_DAMPING * clamped[k]
                            + (1.0 - BATCH_FIT_DAMPING) * current[k],
                            5,
                        )
                        for k in ("s", "e", "w")
                    }
                    damping_applied = BATCH_FIT_DAMPING
                else:
                    blended = {k: round(clamped[k], 5) for k in ("s", "e", "w")}
                    damping_applied = 1.0

                # Residual RMSE measured against the (clamped, undamped)
                # Tobit fit on UNCENSORED samples only — censored rows
                # carry threshold values (T) instead of observed
                # ``actual_impact``, so including them would report a
                # spurious residual whenever ``c·s > T`` (which is the
                # correct prediction for a saturated hour).
                uncensored_only = [
                    samples[i] for i in range(len(samples)) if not censored_mask[i]
                ]
                regime_diag["residual_rmse_kwh"] = self._batch_fit_residual_rmse(
                    uncensored_only, clamped
                )
                regime_diag["coefficient_after"] = blended
                regime_diag["damping_applied"] = damping_applied

                if dry_run:
                    # Preview only — report ``coefficient_after`` so the
                    # user can decide whether to commit, but don't drift
                    # live state.  Diagnostic flag is explicit so the
                    # coordinator's apply-summary log doesn't claim
                    # something was applied.
                    regime_diag["applied"] = False
                    regime_diag["dry_run"] = True
                else:
                    self._update_unit_solar_coefficient(
                        entity_id, blended, solar_coefficients_per_unit, regime
                    )
                    regime_diag["applied"] = True
                    # Smooth migration path: seed the live Tobit window
                    # from the same samples used in this batch fit.
                    # Without this, after a classification change (e.g.
                    # the parasitic-floor gate fix), users would either
                    # need to ``reset_solar_learning`` and accept a 25-
                    # day cold-start, or live with a polluted sliding
                    # window that keeps writing biased coefficients.
                    # ``seed_live_window=True`` rebuilds the window
                    # in-place from the current batch's classified
                    # samples (capped at TOBIT_RUNNING_WINDOW), sets
                    # n_eff to the batch's value, and tags the slot at
                    # the current SOLAR_MODEL_VERSION.  Tobit's next
                    # qualifying hour finds a populated, current-
                    # version window and refines from there — no cold-
                    # start.  Opt-in only; default behaviour preserved.
                    if seed_live_window:
                        from .const import (
                            SOLAR_MODEL_VERSION as _CURRENT_SOLAR_VERSION,
                            TOBIT_RUNNING_WINDOW,
                        )
                        live_stats = getattr(
                            coordinator, "_tobit_sufficient_stats", None
                        )
                        if isinstance(live_stats, dict):
                            window_samples = [
                                (
                                    samples[i][0],
                                    samples[i][1],
                                    samples[i][2],
                                    samples[i][3],
                                    bool(censored_mask[i]),
                                )
                                for i in range(len(samples))
                            ]
                            # Cap from the most recent end — same as
                            # the live learner trims on append.
                            if len(window_samples) > TOBIT_RUNNING_WINDOW:
                                window_samples = window_samples[
                                    -TOBIT_RUNNING_WINDOW:
                                ]
                            entity_state = live_stats.setdefault(entity_id, {})
                            entity_state[regime] = {
                                "samples": window_samples,
                                "solar_model_version": _CURRENT_SOLAR_VERSION,
                                "samples_since_reset": len(window_samples),
                                "last_step": {
                                    "iterations": fit["iterations"],
                                    "converged": bool(fit["converged"]),
                                    "failure_reason": fit.get("failure_reason"),
                                    # Round to match live ``_update_unit_tobit_live`` writes
                                    # (review I3 on f5be736) — diagnose snapshots taken
                                    # right after a seed otherwise show un-rounded values
                                    # while later snapshots show rounded ones.
                                    "sigma": round(fit["sigma"], 5) if fit.get("sigma") is not None else None,
                                    "n_eff": round(fit.get("n_eff", float(n_unc)), 2),
                                    "step_norm": 0.0,
                                    "skip_reason": None,
                                },
                            }
                            regime_diag["seeded_live_window"] = True
                            regime_diag["seeded_window_size"] = len(window_samples)
                        else:
                            # Defensive observable signal (review N5 on
                            # f5be736): coordinator without a dict-shaped
                            # ``_tobit_sufficient_stats`` cannot be seeded.
                            # Stage-3 storage migration should always
                            # populate ``{}`` so this branch is unreachable
                            # in production, but legacy mocks and
                            # restoration paths mid-migration may present
                            # ``None`` — silent no-op would mislead users
                            # into thinking the seed succeeded.
                            regime_diag["seeded_live_window"] = False
                            regime_diag["seed_skip_reason"] = "stats_not_dict"
                entity_results[regime] = regime_diag

            results[entity_id] = entity_results

        return results

    @staticmethod
    def _is_entity_shutdown_under_current_rules(
        *,
        entity_id: str,
        entry: dict,
        pot_tuple: tuple[float, float, float],
        unit_min_base: float,
    ) -> bool:
        """Re-run shutdown detection against current code on a logged entry.

        Used by the for_tobit sample-collection path so the seed /
        batch_fit window reflects the *current* classification rules
        even when the persisted log entry was written under earlier
        (possibly buggy) gate logic.  See review B1 on f5be736 — the
        gate-ordering fix in 1.3.5 means pre-fix log entries' stored
        ``solar_dominant_entities`` may miss parasitic hours that the
        current rules correctly flag as shutdown.

        Mirrors the live ``detect_solar_shutdown_entities`` gate
        sequence per-entity (no need to handle multi-entity behaviour
        here since the caller is in a per-entity loop).  Reads:

        - ``actual`` from ``entry["unit_breakdown"][entity]``.
        - ``base`` from ``entry["unit_expected_breakdown"][entity]``;
          falls back to ``unit_min_base × 2`` × ratio-floor as a
          conservative "in-range" estimate when the field is missing
          (legacy logs).  The fallback only matters for entries old
          enough to predate ``unit_expected_breakdown``; on those the
          reclassification can't be precise, but it's better than
          trusting a possibly-buggy persisted flag.
        - ``unit_modes`` from ``entry["unit_modes"]``; non-heating
          modes return False (existing detector contract).
        - ``auxiliary_active`` from ``entry["auxiliary_active"]``;
          aux-dominant hours return False.
        - ``magnitude`` from the caller-supplied potential tuple
          (already reconstructed once, no need to redo).

        Constants used (``SOLAR_SHUTDOWN_*``) are imported from
        ``const.py`` and reflect the current rules; the function
        re-imports them at call site to avoid stale-binding from a
        potential future hot-reload.
        """
        from .const import (
            SOLAR_SHUTDOWN_ACTUAL_FLOOR,
            SOLAR_SHUTDOWN_MIN_BASE,
            SOLAR_SHUTDOWN_MIN_MAGNITUDE,
            SOLAR_SHUTDOWN_RATIO,
            MODE_HEATING,
        )

        if entry.get("auxiliary_active", False):
            return False
        unit_modes = entry.get("unit_modes", {}) or {}
        if unit_modes.get(entity_id, MODE_HEATING) != MODE_HEATING:
            return False
        magnitude = (pot_tuple[0] ** 2 + pot_tuple[1] ** 2 + pot_tuple[2] ** 2) ** 0.5
        if magnitude < SOLAR_SHUTDOWN_MIN_MAGNITUDE:
            return False
        actual = (entry.get("unit_breakdown", {}) or {}).get(entity_id, 0.0)
        base_raw = (entry.get("unit_expected_breakdown") or {}).get(entity_id)
        if base_raw is None:
            # Legacy log without unit_expected_breakdown.  Fall back to
            # current correlation_data via the same path the rest of
            # the collector uses isn't available here without coordinator
            # access.  Skip reclassification — preserve the persisted
            # flag's decision instead.
            return entity_id in (entry.get("solar_dominant_entities", []) or [])
        base = float(base_raw)
        if base <= 0.0:
            return False
        if actual < SOLAR_SHUTDOWN_ACTUAL_FLOOR:
            return True  # parasitic floor — fires regardless of base/threshold
        threshold = unit_min_base if unit_min_base and unit_min_base > 0 else SOLAR_SHUTDOWN_MIN_BASE
        if base < threshold:
            return False
        ratio = actual / base if base > 0 else 1.0
        return ratio < SOLAR_SHUTDOWN_RATIO

    def _collect_batch_fit_samples(
        self,
        *,
        entity_id: str,
        regime: str,
        hourly_log: list[dict],
        entry_potentials: list[tuple[float, float, float, float]],
        coordinator,
        unit_threshold: float,
        screen_affected_entities: frozenset[str] | None,
        match_diagnose: bool = False,
        for_tobit: bool = False,
        solar_coefficients_per_unit: dict | None = None,
    ) -> tuple[list[tuple[float, float, float, float]], list[bool], dict[str, int]]:
        """Filter + assemble samples for one (entity, regime) batch fit.

        Returns ``(samples, censored_mask, drop_counts)``.  Each sample
        is a 4-tuple ``(pot_s, pot_e, pot_w, value)`` and ``censored_mask[i]``
        is True iff ``samples[i]`` is right-censored at ``value``.
        ``drop_counts`` reports per-reason rejection counters useful for
        diagnostics — a fit that ends with too few samples can be
        inspected to see which gate did the rejecting.

        ``for_tobit=True`` (#904 stage 0+1) keeps saturated rows instead
        of dropping them: each saturated row is appended with
        ``value = BATCH_FIT_SATURATION_RATIO × expected_base`` (the
        censoring threshold ``T_i``, NOT the observed ``actual_impact``)
        and tagged ``censored_mask[i] = True``.  Shutdown rows are still
        dropped — Tobit modulating-regime fit only (CHOICE 3).
        Unit-threshold gate is dropped — Tobit needs the saturated-
        information-rich shoulder hours, and the legacy LS noise-floor
        path is unreachable after stage 2 (every caller passes either
        ``for_tobit=True`` or ``match_diagnose=True``).  Base is still
        read from the live ``correlation_data_per_unit`` for ``for_tobit``
        callers (Tobit can run on logs without ``unit_expected_breakdown``);
        ``match_diagnose=True`` reads from ``unit_expected_breakdown``
        instead, see below.  The ``unit_threshold`` parameter is
        retained for API compatibility but no longer consulted on any
        live code path; it may be removed in a future cleanup once the
        signature surgery is convenient.

        Mutual exclusivity: ``for_tobit`` dominates over
        ``match_diagnose`` for the shutdown filter (#904 CHOICE 3 —
        Tobit always excludes shutdown rows, regardless of
        ``match_diagnose``'s wish to reproduce the diagnose
        accumulator).  No current caller passes both flags True;
        documented for stage-3+ safety.

        ``entry_potentials`` is parallel to ``hourly_log``: each entry
        carries its pre-reconstructed ``(pot_s, pot_e, pot_w, magnitude)``
        for this entity (regime-independent — hoisted by the caller so
        we don't recompute across heating/cooling passes).

        ``match_diagnose=True`` (used by ``apply_implied_coefficient``)
        switches the filter set to mirror ``DiagnosticsEngine.diagnose_solar``
        so the LS fit produces the same number the user reads in the
        ``implied_coefficient_30d`` diagnostic.  Specifically:

        - Drops the ``expected_base ≥ unit_min_base`` gate (diagnose
          only checks ``base is not None``).
        - Includes shutdown samples (diagnose's headline ``implied_30d``
          accumulator does not separate them; only the parallel
          ``no_shutdown`` accumulator does).
        - Reads ``base`` from the log entry's
          ``unit_expected_breakdown[entity]`` rather than the current
          ``correlation_data_per_unit`` lookup — diagnose uses the
          log-time stored value.

        Without this flag, the default filters apply (used by
        ``batch_fit_solar_coefficients``).

        Mode filter mirrors live ``_process_per_unit_learning``: only
        ``MODE_HEATING`` and ``MODE_COOLING`` produce solar-learning
        samples.  OFF / DHW / both guest modes are excluded — guest
        in particular is documented (CLAUDE.md "Mode System") as
        excluded from solar learning, and the live learner enforces
        this.  Earlier this method also accepted guest modes; that
        was a divergence found in code review and corrected.
        """
        samples: list[tuple[float, float, float, float]] = []
        censored_mask: list[bool] = []
        drop_counts: dict[str, int] = {
            "wrong_mode": 0,
            "auxiliary_active": 0,
            "low_magnitude": 0,
            "missing_temp_key": 0,
            "shutdown": 0,
            "below_min_base": 0,
            "non_positive_impact": 0,
            "saturated": 0,
        }
        if for_tobit:
            drop_counts["censored"] = 0
            drop_counts["outlier"] = 0
        target_mode = MODE_HEATING if regime == "heating" else MODE_COOLING
        correlation_per_unit = coordinator.model.correlation_data_per_unit

        # Pre-fit MAD pass (#919 Part 1): collect all candidates first,
        # then filter by robust residual.
        candidates: list[dict] = []

        # Get current coefficients for residual calculation
        entity_coeffs = (solar_coefficients_per_unit or {}).get(entity_id, {})
        regime_coeffs = entity_coeffs.get(regime, {}) if isinstance(entity_coeffs, dict) else {}
        c_s = float(regime_coeffs.get("s", 0.0))
        c_e = float(regime_coeffs.get("e", 0.0))
        c_w = float(regime_coeffs.get("w", 0.0))

        for entry, (pot_s, pot_e, pot_w, magnitude) in zip(
            hourly_log, entry_potentials
        ):
            unit_modes = entry.get("unit_modes", {}) or {}
            entry_mode = unit_modes.get(entity_id, MODE_HEATING)
            if entry_mode != target_mode:
                drop_counts["wrong_mode"] += 1
                continue

            if entry.get("auxiliary_active", False):
                drop_counts["auxiliary_active"] += 1
                continue

            if for_tobit:
                # Shutdown rows under Tobit: do NOT drop.  ``actual = 0``
                # gives ``actual_impact = base``, which naturally clears
                # the saturation gate downstream, and the row is emitted
                # with ``value = saturation_threshold`` and
                # ``censored_mask = True``.  Tobit's Mills-ratio handles
                # the right-censoring at this boundary mathematically —
                # the lower-bound information ("c·s must be at least
                # base, else log-likelihood is penalised") is exactly
                # what shutdown hours carry.  Dropping them discards
                # signal Tobit was designed to extract.
                # Counter retained for diagnostics (the row is kept,
                # not dropped, so it does not contribute to total
                # rejections — separate key).
                if self._is_entity_shutdown_under_current_rules(
                    entity_id=entity_id,
                    entry=entry,
                    pot_tuple=(pot_s, pot_e, pot_w),
                    unit_min_base=unit_threshold,
                ):
                    drop_counts["shutdown_kept_censored"] = (
                        drop_counts.get("shutdown_kept_censored", 0) + 1
                    )
            elif not match_diagnose:
                # LS path: cannot model censoring, so shutdown rows
                # remain dropped to avoid coefficient inflation.
                shutdown_entities = set(
                    entry.get("solar_dominant_entities", []) or []
                )
                if entity_id in shutdown_entities:
                    drop_counts["shutdown"] += 1
                    continue

            temp_key = entry.get("temp_key")
            if temp_key is None:
                drop_counts["missing_temp_key"] += 1
                continue

            if magnitude <= 0.01:
                drop_counts["low_magnitude"] += 1
                continue

            # Re-bucketize live against current thresholds when the
            # coordinator exposes them; stored label is a cache of
            # write-time thresholds (see retrain.py:422).  isinstance(..., (int, float))
            # filters out MagicMock auto-attrs from test stubs that
            # never set explicit thresholds.
            wt = getattr(coordinator, "wind_threshold", None)
            ewt = getattr(coordinator, "extreme_wind_threshold", None)
            if isinstance(wt, (int, float)) and isinstance(ewt, (int, float)):
                eff_w = entry.get("effective_wind", 0.0) or 0.0
                if eff_w >= ewt:
                    wind_bucket = "extreme_wind"
                elif eff_w >= wt:
                    wind_bucket = "high_wind"
                else:
                    wind_bucket = "normal"
            else:
                wind_bucket = entry.get("wind_bucket", "normal")
            effective_bucket = (
                COOLING_WIND_BUCKET if regime == "cooling" else wind_bucket
            )
            if match_diagnose:
                stored = (entry.get("unit_expected_breakdown") or {}).get(
                    entity_id
                )
                if stored is None:
                    drop_counts["below_min_base"] += 1
                    continue
                expected_base = float(stored)
            else:
                unit_buckets = correlation_per_unit.get(entity_id, {}).get(
                    temp_key, {}
                )
                expected_base = (
                    unit_buckets.get(effective_bucket, 0.0)
                    if unit_buckets
                    else 0.0
                )
                if expected_base <= 0.0:
                    drop_counts["below_min_base"] += 1
                    continue

            actual = (entry.get("unit_breakdown", {}) or {}).get(entity_id, 0.0)
            if regime == "heating":
                actual_impact = expected_base - actual
            else:
                actual_impact = actual - expected_base

            if actual_impact <= 0.0:
                drop_counts["non_positive_impact"] += 1
                continue

            saturation_threshold = BATCH_FIT_SATURATION_RATIO * expected_base
            is_saturated = (
                regime == "heating" and actual_impact >= saturation_threshold
            )
            
            # Calculate residual against current model
            pred = c_s * pot_s + c_e * pot_e + c_w * pot_w
            residual = actual_impact - pred
            
            candidates.append({
                "pot": (pot_s, pot_e, pot_w),
                "actual_impact": actual_impact,
                "is_saturated": is_saturated,
                "saturation_threshold": saturation_threshold,
                "residual": residual,
                "expected_base": expected_base,
                "actual": actual,
            })

        if not candidates:
            return [], [], drop_counts

        # Robust residual filtering (#919 Part 1).
        # Skip MAD-baseline filtering if match_diagnose=True or coefficients are missing.
        # But ALWAYS apply the prior-free sanity check to maintain consistency
        # with diagnose_solar.
        use_mad = not match_diagnose and solar_coefficients_per_unit is not None
        threshold = None
        median_res = 0.0

        if use_mad:
            # Only use non-saturated samples for MAD calculation
            uncensored_residuals = [c["residual"] for c in candidates if not c["is_saturated"]]
            
            # Always center on median to handle stale current-coefficients (#919).
            if uncensored_residuals:
                median_res = statistics.median(uncensored_residuals)
            else:
                median_res = 0.0

            if len(uncensored_residuals) >= OUTLIER_MIN_SAMPLES:
                abs_devs = [abs(r - median_res) for r in uncensored_residuals]
                mad = statistics.median(abs_devs)
                
                # Sigma floor consistent with _is_outlier_residual
                # unit_min_base is passed as unit_threshold in this method's API.
                # Handle both float (direct value) and dict (per-entity map) cases.
                # NOTE: _resolve_min_base defaults to 0.5 for unknown entities.
                # For small loads (e.g. 0.01 kWh), this default creates an implicit
                # 25 Wh sigma-floor, keeping the MAD filter latently "off" (safe)
                # until a proper unit_min_base is configured for the sensor.
                if isinstance(unit_threshold, (int, float)) and unit_threshold > 0:
                    min_base_val = float(unit_threshold)
                elif isinstance(unit_threshold, dict):
                    min_base_val = _resolve_min_base(entity_id, unit_threshold, 0.5)
                else:
                    min_base_val = 0.5
                
                sigma_floor = max(0.02, 0.05 * max(min_base_val, 0.1))
                sigma_robust = max(sigma_floor, 1.4826 * mad)
                
                threshold = OUTLIER_K_THRESHOLD * sigma_robust
            else:
                # Fallback for small windows: no MAD filtering, just hard cap.
                # Still centered on median so stale model doesn't drop all samples.
                threshold = HARD_OUTLIER_CAP_FACTOR

        filtered_candidates = []
        for c in candidates:
            is_outlier_sample = False
            # 1. Filter by MAD residual (conditional)
            if threshold is not None and abs(c["residual"] - median_res) > threshold:
                is_outlier_sample = True
                
            # 2. Filter by prior-free sanity (unconditional, matches diagnostics.py)
            if not is_outlier_sample and c["expected_base"] > 0:
                if abs(c["actual"] - c["expected_base"]) > HARD_OUTLIER_SANITY_MULTIPLIER * c["expected_base"]:
                    is_outlier_sample = True
            
            if is_outlier_sample:
                drop_counts["outlier"] = drop_counts.get("outlier", 0) + 1
                continue
            
            filtered_candidates.append(c)
        candidates = filtered_candidates

        for c in candidates:
            if c["is_saturated"]:
                if for_tobit:
                    samples.append(
                        (c["pot"][0], c["pot"][1], c["pot"][2], c["saturation_threshold"])
                    )
                    censored_mask.append(True)
                    drop_counts["censored"] += 1
                else:
                    drop_counts["saturated"] += 1
            else:
                samples.append((c["pot"][0], c["pot"][1], c["pot"][2], c["actual_impact"]))
                censored_mask.append(False)

        return samples, censored_mask, drop_counts

    @staticmethod
    def _solve_batch_fit_normal_equations(
        samples: list[tuple[float, float, float, float]],
    ) -> dict[str, float] | None:
        """Joint 3×3 LS via Gauss-Jordan, with collinear 1D fallback.

        Mirrors the cold-start solver in ``_learn_unit_solar_coefficient``
        without the cold-start damping (the caller damps).  Returns a
        coefficient dict, or ``None`` if the fit cannot be produced
        (degenerate Gram matrix and zero-magnitude direction sum).
        """
        if not samples:
            return None
        sum_s2 = sum(s[0] ** 2 for s in samples)
        sum_e2 = sum(s[1] ** 2 for s in samples)
        sum_w2 = sum(s[2] ** 2 for s in samples)
        sum_se = sum(s[0] * s[1] for s in samples)
        sum_sw = sum(s[0] * s[2] for s in samples)
        sum_ew = sum(s[1] * s[2] for s in samples)
        sum_s_I = sum(s[0] * s[3] for s in samples)
        sum_e_I = sum(s[1] * s[3] for s in samples)
        sum_w_I = sum(s[2] * s[3] for s in samples)

        determinant = (
            sum_s2 * (sum_e2 * sum_w2 - sum_ew ** 2)
            - sum_se * (sum_se * sum_w2 - sum_ew * sum_sw)
            + sum_sw * (sum_se * sum_ew - sum_e2 * sum_sw)
        )

        # Trigger `|det| > 1e-6` is preserved so the boundary between
        # joint LS and the 1D collinear fallback is unchanged.
        if abs(determinant) > 1e-6:
            sol = solve_gauss_jordan(
                [
                    [sum_s2, sum_se, sum_sw],
                    [sum_se, sum_e2, sum_ew],
                    [sum_sw, sum_ew, sum_w2],
                ],
                [sum_s_I, sum_e_I, sum_w_I],
            )
            if sol is not None:
                return {"s": sol[0], "e": sol[1], "w": sol[2]}

        # Collinear fallback: project onto dominant direction.
        dir_s = sum(s[0] for s in samples)
        dir_e = sum(s[1] for s in samples)
        dir_w = sum(s[2] for s in samples)
        dir_norm = (dir_s ** 2 + dir_e ** 2 + dir_w ** 2) ** 0.5
        if dir_norm < 1e-6:
            return None

        d_s = dir_s / dir_norm
        d_e = dir_e / dir_norm
        d_w = dir_w / dir_norm
        sum_proj_I = sum(
            (s[0] * d_s + s[1] * d_e + s[2] * d_w) * s[3] for s in samples
        )
        sum_proj2 = sum(
            (s[0] * d_s + s[1] * d_e + s[2] * d_w) ** 2 for s in samples
        )
        if sum_proj2 < 1e-6:
            return None
        c_scalar = sum_proj_I / sum_proj2
        return {"s": c_scalar * d_s, "e": c_scalar * d_e, "w": c_scalar * d_w}

    @staticmethod
    def _batch_fit_residual_rmse(
        samples: list[tuple[float, float, float, float]],
        coeff: dict[str, float],
    ) -> float:
        """RMSE of (predicted_impact - actual_impact) over the sample set."""
        if not samples:
            return 0.0
        sse = 0.0
        for pot_s, pot_e, pot_w, impact in samples:
            pred = (
                coeff.get("s", 0.0) * pot_s
                + coeff.get("e", 0.0) * pot_e
                + coeff.get("w", 0.0) * pot_w
            )
            sse += (pred - impact) ** 2
        return round((sse / len(samples)) ** 0.5, 5)

    @staticmethod
    def _tobit_mills(q: float) -> tuple[float, float]:
        """Right-censoring Mills ratio λ(q) = φ(q) / (1 − Φ(q)) and survival.

        Returns ``(λ(q), 1 − Φ(q))``.  Uses ``erfc`` for the survival
        function to avoid catastrophic cancellation at large positive
        ``q`` (where ``1 − Φ`` underflows in the naïve form).  At
        ``q ≳ 8`` the Mills ratio is computed from the asymptotic
        expansion ``λ(q) ≈ q + 1/q − 2/q³`` (Abramowitz & Stegun
        26.2.13) — direct ``φ/erfc`` divides two near-zero numbers
        and loses precision long before either underflows.
        """
        if q > 8.0:
            # Asymptotic; survival underflows but Mills ratio is finite.
            inv = 1.0 / q
            lam = q + inv - 2.0 * inv * inv * inv
            return (lam, 0.0)
        if q < -38.0:
            # Survival ≈ 1, pdf ≈ 0 → Mills ratio negligible.
            return (0.0, 1.0)
        surv = 0.5 * math.erfc(q / math.sqrt(2.0))
        if surv < 1e-300:
            inv = 1.0 / q if q != 0 else 0.0
            return (q + inv, surv)
        pdf = math.exp(-0.5 * q * q) / math.sqrt(2.0 * math.pi)
        return (pdf / surv, surv)

    @staticmethod
    def _solve_tobit(
        samples: list[tuple[float, ...]],
        censored_mask: list[bool],
        *,
        components: tuple[str, ...],
        coeff_init: dict[str, float] | None = None,
        sigma_init: float | None = None,
        max_iter: int = TOBIT_MAX_ITER,
        tol: float = TOBIT_CONV_TOL,
        q_clip: float = TOBIT_Q_CLIP,
    ) -> dict | None:
        """Type-I right-censored Gaussian regression MLE (dimension-agnostic).

        Maximises the Tobit log-likelihood for solar-coefficient learning
        with saturation-clipped (HP-fully-off) samples treated as
        right-censored data, jointly over ``(c_0, ..., c_{n-1}, log σ)``:

            ℓ(c, σ) = Σ_U [ −log σ − ½((y_i − c·s_i)/σ)² ]
                    + Σ_C log( 1 − Φ((T_i − c·s_i)/σ) )                    (+ const)

        where ``y_i = base − actual`` for uncensored rows and ``T_i =
        BATCH_FIT_SATURATION_RATIO × base`` for censored rows.  Per
        Olsen (1978) ℓ is globally concave in ``(c, log σ)``, so the
        unique maximum can be reached by Newton iterations from any
        feasible start.  Non-negativity (invariant #4) is enforced via
        an active-set projected Newton step.

        Dimension and output keys are driven by ``components`` —
        ``("s", "e", "w")`` for the 3D solar coefficient, or
        ``("s", "e", "w", "diffuse")`` for the 4D shadow potential.
        The Newton loop, gradient/Hessian build, and active-set logic
        are fully dimension-agnostic; the warm-start LS dispatches by
        ``len(components)`` to preserve bit-identity with the prior
        ``_solve_tobit_3d`` (which used the 1D-collinear-fallback
        :meth:`_solve_batch_fit_normal_equations` path) and
        ``_solve_tobit_4d`` (which used :meth:`_solve_4x4_normal_equations`).

        Sample format
        -------------
        ``samples[i] = (c_0_i, ..., c_{n-1}_i, value_i)`` — an
        ``(n+1)``-tuple where ``n = len(components)``.  Parallel to
        ``censored_mask[i]``.  When ``censored_mask[i] = True``,
        ``value_i`` is the censoring threshold ``T_i`` (NOT the observed
        ``actual_impact``); when ``False``, it is the observed
        ``y_i = base − actual``.

        Returns
        -------
        ``{components[0]: c0, ..., "sigma", "iterations", "converged",
        "failure_reason", "log_likelihood", "n_uncensored", "n_censored",
        "n_eff"}`` on success, or ``None`` when the warm-start LS has too
        few rows to form an initial estimate.  ``n_eff = |U| + Σ_C λ(q)(λ(q)−q)``
        is the censoring-weighted effective sample size.

        ``q_clip`` floor on ``q = (T − c·s)/σ`` inside the trust region
        guards against the asymptotic regime where λ(q) ≈ 0 (model
        predicts the unit fully on while data says fully off — the
        censored sample carries no slope information here, but feeding
        the limit value into the Newton step produces a pathological
        zero contribution that destabilises the active-set logic).
        """
        n_dim = len(components)
        n = len(samples)
        if n != len(censored_mask) or n == 0:
            return None

        # Warm-start: LS on uncensored rows only.  Sigma seed = residual
        # std of LS fit; we floor at a tiny value because a perfect-fit
        # warm start with σ=0 makes the very first r_i blow up.
        uncensored_samples = [
            samples[i] for i in range(n) if not censored_mask[i]
        ]
        n_unc = len(uncensored_samples)
        if n_unc < n_dim:
            # σ identifiability requires at least n_dim uncensored rows
            # to seed the LS direction (one per unknown).  Defensive —
            # caller's gate (TOBIT_MIN_UNCENSORED = 20) should already
            # have rejected.
            return None

        # Always run the LS fit on uncensored rows.  Two uses below:
        # (1) when no warm-start c is supplied, LS becomes the c-init;
        # (2) when a warm-start c IS supplied (live-learner replay),
        # the LS residual still seeds σ — see the σ-init block.
        # If LS is degenerate (collinear samples in the uncensored
        # subset), fall back to the warm-start path.
        #
        # Dispatch by dimension to preserve bit-identity with the prior
        # split solvers: 3D used _solve_batch_fit_normal_equations
        # (with a 1D collinear fallback for narrow-angle winter sun);
        # 4D used _solve_4x4_normal_equations (no fallback — returns
        # None on singular Gram matrix, letting the caller route to
        # warm_start_failed).
        if n_dim == 3:
            ls_fit_dict = LearningManager._solve_batch_fit_normal_equations(
                uncensored_samples
            )
            if ls_fit_dict is None:
                ls_solution = None
            else:
                ls_solution = tuple(
                    float(ls_fit_dict[components[k]]) for k in range(n_dim)
                )
        elif n_dim == 4:
            X_rows = [tuple(s[:n_dim]) for s in uncensored_samples]
            y_vals = [s[n_dim] for s in uncensored_samples]
            ls_solution = LearningManager._solve_4x4_normal_equations(
                X_rows, y_vals
            )
        else:
            # Generic n_dim — build A = XᵀX, b = Xᵀy, solve via
            # helpers.solve_gauss_jordan.  No collinear fallback.
            A_mat = [[0.0] * n_dim for _ in range(n_dim)]
            b_vec = [0.0] * n_dim
            for row in uncensored_samples:
                y_i = row[n_dim]
                for i in range(n_dim):
                    xi = row[i]
                    for j in range(n_dim):
                        A_mat[i][j] += xi * row[j]
                    b_vec[i] += xi * y_i
            sol = solve_gauss_jordan(A_mat, b_vec, pivot_eps=1e-10)
            ls_solution = None if sol is None else tuple(sol)

        if coeff_init is None:
            if ls_solution is None:
                return None
            c = [max(0.0, float(ls_solution[k])) for k in range(n_dim)]
        else:
            c = [
                max(0.0, float(coeff_init.get(components[k], 0.0)))
                for k in range(n_dim)
            ]

        # σ initialisation: seed σ from the LS-fit residuals (which
        # are c-independent of the warm-start) rather than from the
        # warm-start residuals.  Naïve σ-init from SSE-against-c
        # yields a wildly inflated σ when c is biased — e.g. a live-
        # learner warm-start from an NLMS-converged-but-saturation-
        # biased coefficient gives σ ≈ 5–8× true at α=27 % censoring,
        # while LS-residual σ stays within 5–15 % of truth across
        # α ∈ [0, 50 %].  Inflated σ → Newton iter 1 over-corrects σ
        # → line-search-budget exhausts → ``did_not_converge`` on
        # iter 1.
        #
        # Important: this fix does NOT widen Newton's convergence
        # basin (verified by pre-fix vs post-fix sweep — identical
        # basin boundaries).  Newton from a biased c still fails to
        # converge; it just fails 4 iterations later instead of on
        # iter 1.  The escape from biased priors is implemented via
        # LS-fallback retry in ``_update_unit_tobit_live``, not here.
        # The σ-init fix is defense-in-depth: better diagnose
        # visibility (failure shows up at iter 5 with the σ-Newton-
        # step trajectory in the slot's last_step), and better σ
        # at convergence on near-basin priors that DO succeed.
        #
        # When LS is degenerate (no usable σ seed), fall back to the
        # warm-start residuals — preserves pre-fix behaviour for the
        # corner case where LS itself cannot be computed.
        if sigma_init is None:
            if ls_solution is not None:
                sse = 0.0
                for row in uncensored_samples:
                    y_i = row[n_dim]
                    pred = 0.0
                    for k in range(n_dim):
                        pred += ls_solution[k] * row[k]
                    sse += (y_i - pred) ** 2
            else:
                sse = 0.0
                for row in uncensored_samples:
                    y_i = row[n_dim]
                    pred = 0.0
                    for k in range(n_dim):
                        pred += c[k] * row[k]
                    sse += (y_i - pred) ** 2
            sigma = max(1e-3, (sse / max(1, n_unc)) ** 0.5)
        else:
            sigma = max(1e-3, float(sigma_init))
        gamma = math.log(sigma)

        def _loglik(c_vec: list[float], gamma_val: float) -> float:
            sig = math.exp(gamma_val)
            ll = 0.0
            log_2pi = math.log(2.0 * math.pi)
            for i in range(n):
                row = samples[i]
                val = row[n_dim]
                pred = 0.0
                for k in range(n_dim):
                    pred += c_vec[k] * row[k]
                if censored_mask[i]:
                    q_i = (val - pred) / sig
                    if q_i < q_clip:
                        q_i = q_clip
                    _, surv = LearningManager._tobit_mills(q_i)
                    if surv <= 0.0:
                        # Asymptotic: log(survival) ≈ −q²/2 − log(q√2π) for q≫0
                        if q_i > 0:
                            ll -= 0.5 * q_i * q_i + math.log(
                                q_i * math.sqrt(2.0 * math.pi)
                            )
                        else:
                            return -math.inf
                    else:
                        ll += math.log(surv)
                else:
                    r_i = (val - pred) / sig
                    ll += -0.5 * log_2pi - gamma_val - 0.5 * r_i * r_i
            return ll

        ll_curr = _loglik(c, gamma)

        n_total = n_dim + 1   # c-components + γ
        gamma_idx = n_dim

        converged = False
        failure_reason: str | None = None
        last_iter = 0
        for it in range(max_iter):
            last_iter = it + 1
            sigma = math.exp(gamma)

            # Build gradient g (n_total-vec) and Hessian H (n_total×n_total).
            # Index convention: 0..n_dim-1 = c-components; n_dim = γ.
            g = [0.0] * n_total
            H = [[0.0] * n_total for _ in range(n_total)]

            for i in range(n):
                row = samples[i]
                val = row[n_dim]
                pred = 0.0
                for k in range(n_dim):
                    pred += c[k] * row[k]
                vec = row[:n_dim]
                if censored_mask[i]:
                    q_i = (val - pred) / sigma
                    if q_i < q_clip:
                        q_i = q_clip
                    lam, _ = LearningManager._tobit_mills(q_i)
                    weight = lam * (lam - q_i)  # ≥ 0 (Greene §17.3)
                    # ∂ℓ/∂c_k = (s_ik / σ) · λ(q)
                    coef_grad = lam / sigma
                    for k in range(n_dim):
                        g[k] += coef_grad * vec[k]
                    # ∂ℓ/∂γ = q · λ(q)
                    g[gamma_idx] += q_i * lam
                    # H_cc block: −(1/σ²) · weight · s_ik s_il
                    inv_sig2 = 1.0 / (sigma * sigma)
                    for k in range(n_dim):
                        for l in range(n_dim):
                            H[k][l] -= inv_sig2 * weight * vec[k] * vec[l]
                    # H_cγ:  −(1/σ) · λ(q) · [1 + q·(λ(q) − q)]
                    cross = -(lam / sigma) * (1.0 + q_i * (lam - q_i))
                    for k in range(n_dim):
                        H[k][gamma_idx] += cross * vec[k]
                        H[gamma_idx][k] += cross * vec[k]
                    # H_γγ: −q · λ(q) · [1 + q·(λ(q) − q)]
                    H[gamma_idx][gamma_idx] -= q_i * lam * (1.0 + q_i * (lam - q_i))
                else:
                    r_i = (val - pred) / sigma
                    # ∂ℓ/∂c_k = (s_ik / σ) · r_i
                    coef_grad = r_i / sigma
                    for k in range(n_dim):
                        g[k] += coef_grad * vec[k]
                    # ∂ℓ/∂γ = −1 + r²
                    g[gamma_idx] += -1.0 + r_i * r_i
                    # H_cc: −s_ik s_il / σ²
                    inv_sig2 = 1.0 / (sigma * sigma)
                    for k in range(n_dim):
                        for l in range(n_dim):
                            H[k][l] -= inv_sig2 * vec[k] * vec[l]
                    # H_cγ: −2 r · s_ik / σ
                    cross = -2.0 * r_i / sigma
                    for k in range(n_dim):
                        H[k][gamma_idx] += cross * vec[k]
                        H[gamma_idx][k] += cross * vec[k]
                    # H_γγ: −2 r²
                    H[gamma_idx][gamma_idx] -= 2.0 * r_i * r_i

            # Active-set projection: a c-component pinned at zero with
            # gradient pushing further negative stays at zero (KKT).
            # γ is always free.  Free indices participate in the reduced
            # Newton solve; pinned components contribute zero step.
            free = [True] * n_total
            for k in range(n_dim):
                if c[k] <= 0.0 and g[k] < 0.0:
                    free[k] = False

            free_idx = [k for k in range(n_total) if free[k]]
            m = len(free_idx)
            if m == 0:
                converged = True
                break

            # First-order optimality: if the gradient is essentially zero
            # on every free direction we are at the (local) maximum and
            # there is nothing for Newton to do.  Catches the noiseless-
            # warm-start case where the LS solution is already exact —
            # without this guard the line search would halve α 10 times
            # trying to find an improvement that cannot exist (every
            # perturbation strictly decreases ll on a perfect-fit point)
            # and bail out as ``line_search_failed`` even though the
            # iterate IS the optimum.  Real data always has noise → real
            # gradient ≠ 0 at the warm-start, so this branch only fires
            # at the actual converged optimum.
            grad_norm = max(abs(g[k]) for k in free_idx)
            if grad_norm < tol:
                converged = True
                break

            # Solve reduced system: −H[free, free] · Δ = g[free]
            # We negate H so the matrix is positive-definite at maximum
            # (negative-definite Hessian → minus is PD).  Tikhonov ridge
            # on the diagonal guards against rank-deficient Newton at the
            # boundary (e.g. when a c-comp pinned at 0 leaves the
            # remaining vector collinear).
            A = [[-H[free_idx[r]][free_idx[col]] for col in range(m)]
                 for r in range(m)]
            b = [g[free_idx[r]] for r in range(m)]
            sol = solve_gauss_jordan(A, b, ridge=1e-9, pivot_eps=1e-12)
            if sol is None:
                # Singular reduced Hessian — Newton step is undefined.
                # Olsen 1978's global concavity guarantees this should
                # not happen on data with non-zero solar variance, but
                # collinear sample windows or boundary-rank-deficient
                # active sets can still trip this in practice.  Bail
                # out as a solver FAILURE — ``converged = False``
                # propagates so callers route to ``did_not_converge``.
                failure_reason = "singular_step"
                break

            delta = [0.0] * n_total
            for j, k in enumerate(free_idx):
                delta[k] = sol[j]

            # Backtracking line search.  Try full Newton, halve up to
            # 10 times if the projected step doesn't improve ℓ.  The
            # global concavity (Olsen 1978) guarantees a step exists,
            # but γ-direction Newton can over-shoot when σ is small.
            alpha = 1.0
            ll_new = -math.inf
            c_new = c[:]
            gamma_new = gamma
            for _ls in range(10):
                c_new = [
                    max(0.0, c[k] + alpha * delta[k]) for k in range(n_dim)
                ]
                gamma_new = gamma + alpha * delta[gamma_idx]
                # Bound γ to keep σ in a sane numerical range.
                if gamma_new < -10.0:
                    gamma_new = -10.0
                elif gamma_new > 10.0:
                    gamma_new = 10.0
                ll_new = _loglik(c_new, gamma_new)
                if ll_new > ll_curr - 1e-12:
                    break
                alpha *= 0.5
            else:
                # Line-search exhaustion: 10 alpha-halvings produced no
                # improving step.  This is a SOLVER FAILURE — the Newton
                # direction is unreliable on this iterate (typically
                # high-censoring + numerically stiff windows).  ``converged
                # = False`` propagates so callers route to
                # ``did_not_converge``.  Setting ``converged = True`` here
                # (pre-fix) would mask the failure as a successful exit
                # and write the unimproved iterate to live state.
                failure_reason = "line_search_failed"
                break

            step_norm = max(
                abs(c_new[k] - c[k]) for k in range(n_dim)
            ) if delta[:n_dim] else 0.0
            step_norm = max(step_norm, abs(gamma_new - gamma))

            c = c_new
            gamma = gamma_new
            ll_curr = ll_new

            if step_norm < tol:
                converged = True
                break

        sigma = math.exp(gamma)

        # Effective sample size: |U| + Σ_C λ(q)(λ(q) − q).  Report so the
        # caller (and downstream gates) can act on identifiability.
        n_eff = float(n_unc)
        n_cens = 0
        for i in range(n):
            if not censored_mask[i]:
                continue
            n_cens += 1
            row = samples[i]
            val = row[n_dim]
            pred = 0.0
            for k in range(n_dim):
                pred += c[k] * row[k]
            q_i = (val - pred) / sigma
            if q_i < q_clip:
                q_i = q_clip
            lam, _ = LearningManager._tobit_mills(q_i)
            n_eff += lam * (lam - q_i)

        result: dict = {components[k]: c[k] for k in range(n_dim)}
        result["sigma"] = sigma
        result["iterations"] = last_iter
        result["converged"] = converged
        result["failure_reason"] = failure_reason
        result["log_likelihood"] = ll_curr
        result["n_uncensored"] = n_unc
        result["n_censored"] = n_cens
        result["n_eff"] = n_eff
        return result


    def _collect_batch_fit_samples_4d(
        self,
        *,
        entity_id: str,
        regime: str,
        hourly_log: list[dict],
        coordinator,
        screen_affected_entities: frozenset[str] | None,
    ) -> tuple[list[tuple[float, float, float, float, float]], list[bool], dict[str, int]]:
        """Collect 4D Tobit samples from the hourly log (#954).

        Parallel to :meth:`_collect_batch_fit_samples` but computes the
        4D potential vector via the f41ffd8 pipeline:

            1. Resolve (DNI, DHI) from per-row GHI / native /
               cloud_coverage using ``resolve_dni_dhi``.
            2. Compute hour-midpoint sun position
               (``timestamp + 30 minutes``) via
               ``solar.get_approx_sun_pos``.  Skip if sun is below the
               horizon.
            3. Call
               ``solar.calculate_unit_potential_4d(entity_id, dni, dhi,
               sun_elev, sun_az, screen_config_for_entity,
               correction_percent)``.
            4. Magnitude gate: skip when
               ``sum(potential_4d) < 0.01`` (matches f41ffd8 live-NLMS
               gate).

        Gate filters (same as 3D): modulating regime, no aux, not
        shutdown, ``expected_unit_base > 0``, ``actual_impact > 0``.
        Saturated rows kept as right-censored with
        ``value = BATCH_FIT_SATURATION_RATIO × base``; unsaturated
        rows kept with ``value = base − actual`` (heating) or
        ``actual − base`` (cooling).

        Returns ``(samples, censored_mask, drop_counts)`` parallel to
        the 3D version.  ``drop_counts`` reuses the 3D keys plus
        ``no_dni_dhi`` and ``sun_below_horizon`` for 4D-specific gates.
        """
        from .solar import reconstruct_hour_inputs

        samples: list[tuple[float, float, float, float, float]] = []
        censored_mask: list[bool] = []
        drop_counts: dict[str, int] = {
            "wrong_mode": 0,
            "auxiliary_active": 0,
            "shutdown": 0,
            "missing_timestamp": 0,
            "sun_below_horizon": 0,
            "no_dni_dhi": 0,
            "low_magnitude": 0,
            "missing_temp_key": 0,
            "below_min_base": 0,
            "non_positive_impact": 0,
            "censored": 0,
        }
        target_mode = MODE_HEATING if regime == "heating" else MODE_COOLING
        correlation_per_unit = coordinator.model.correlation_data_per_unit

        scr_fn = getattr(coordinator, "screen_config_for_entity", None)
        if scr_fn is not None:
            screen_cfg_for_entity = scr_fn(entity_id)
        else:
            screen_cfg_for_entity = getattr(coordinator, "screen_config", None)
        solar = getattr(coordinator, "solar", None)
        if solar is None:
            return [], [], drop_counts

        for entry in hourly_log:
            unit_modes = entry.get("unit_modes", {}) or {}
            entry_mode = unit_modes.get(entity_id, MODE_HEATING)
            if entry_mode != target_mode:
                drop_counts["wrong_mode"] += 1
                continue
            if entry.get("auxiliary_active", False):
                drop_counts["auxiliary_active"] += 1
                continue
            # Shutdown rows under 4D Tobit: do NOT drop.  Same rationale
            # as the 3D collector — actual = 0 gives actual_impact = base,
            # the row clears the saturation gate downstream and is
            # emitted with censored_mask = True.  Tobit's Mills-ratio
            # extracts the lower-bound information that dropping these
            # rows would discard.  Counter retained for diagnostics
            # under a separate key (the row is kept, not dropped).
            shutdown_entities = set(
                entry.get("solar_dominant_entities", []) or []
            )
            if entity_id in shutdown_entities:
                drop_counts["shutdown_kept_censored"] = (
                    drop_counts.get("shutdown_kept_censored", 0) + 1
                )

            inputs, fail = reconstruct_hour_inputs(entry, solar)
            if inputs is None:
                # Preserve pre-#1011 bucketing: sun-pos lookup exceptions
                # were counted as "sun_below_horizon" in the inline form.
                bucket = "sun_below_horizon" if fail == "sun_pos_error" else fail
                drop_counts[bucket] += 1
                continue
            # Helper rejects only on source=="none"; preserve the
            # original stricter zero-gate here (#1011 scope note).
            if inputs.dni <= 0.0 and inputs.dhi <= 0.0:
                drop_counts["no_dni_dhi"] += 1
                continue
            sun_elev = inputs.sun_elev
            sun_az = inputs.sun_az
            dni_v = inputs.dni
            dhi_v = inputs.dhi
            correction_percent = float(entry.get("correction_percent", 100.0))
            try:
                potential_4d = solar.calculate_unit_potential_4d(
                    entity_id,
                    dni_v or 0.0,
                    dhi_v or 0.0,
                    sun_elev,
                    sun_az,
                    screen_cfg_for_entity,
                    correction_percent,
                )
            except Exception:
                drop_counts["low_magnitude"] += 1
                continue
            pot_s, pot_e, pot_w, pot_d = potential_4d
            if pot_s + pot_e + pot_w + pot_d < 0.01:
                drop_counts["low_magnitude"] += 1
                continue

            temp_key = entry.get("temp_key")
            if temp_key is None:
                drop_counts["missing_temp_key"] += 1
                continue

            # Re-bucketize live against current thresholds when the
            # coordinator exposes them AND the entry carries
            # ``effective_wind``; stored label is a cache of write-time
            # thresholds (see retrain.py:422).  Fall back to stored
            # label for legacy entries / test stubs.
            wt = getattr(coordinator, "wind_threshold", None)
            ewt = getattr(coordinator, "extreme_wind_threshold", None)
            eff_w = entry.get("effective_wind")
            if (
                isinstance(wt, (int, float))
                and isinstance(ewt, (int, float))
                and isinstance(eff_w, (int, float))
            ):
                if eff_w >= ewt:
                    wind_bucket = "extreme_wind"
                elif eff_w >= wt:
                    wind_bucket = "high_wind"
                else:
                    wind_bucket = "normal"
            else:
                wind_bucket = entry.get("wind_bucket", "normal")
            effective_bucket = (
                COOLING_WIND_BUCKET if regime == "cooling" else wind_bucket
            )
            unit_buckets = correlation_per_unit.get(entity_id, {}).get(
                temp_key, {}
            )
            expected_base = (
                unit_buckets.get(effective_bucket, 0.0) if unit_buckets else 0.0
            )
            if expected_base <= 0.0:
                drop_counts["below_min_base"] += 1
                continue

            actual = (entry.get("unit_breakdown", {}) or {}).get(entity_id, 0.0)
            if regime == "heating":
                actual_impact = expected_base - actual
            else:
                actual_impact = actual - expected_base
            if actual_impact <= 0.0:
                drop_counts["non_positive_impact"] += 1
                continue

            saturation_threshold = BATCH_FIT_SATURATION_RATIO * expected_base
            is_saturated = (
                regime == "heating" and actual_impact >= saturation_threshold
            )
            if is_saturated:
                samples.append((pot_s, pot_e, pot_w, pot_d, saturation_threshold))
                censored_mask.append(True)
                drop_counts["censored"] += 1
            else:
                samples.append((pot_s, pot_e, pot_w, pot_d, actual_impact))
                censored_mask.append(False)

        return samples, censored_mask, drop_counts

    def batch_fit_solar_coefficients_4d(
        self,
        hourly_log: list[dict],
        solar_coefficients_4d_per_unit: dict,
        energy_sensors: list[str],
        coordinator,
        *,
        entity_id_filter: str | None = None,
        unit_min_base: dict[str, float] | None = None,
        screen_affected_entities: frozenset[str] | None = None,
        days_back: int | None = None,
        dry_run: bool = False,
        seed_live_window: bool = False,
    ) -> dict:
        """Batch Tobit MLE fit on the 4D shadow potential (#954).

        Parallels :meth:`batch_fit_solar_coefficients` (3D) with the
        4D potential pipeline from commit f41ffd8.  Writes only to
        ``solar_coefficients_4d_per_unit`` — strict shadow, no
        production read-path consumer yet.

        ``seed_live_window`` is accepted for parity with the 3D
        signature but is currently a no-op for 4D (the 4D path has no
        live Tobit sliding window; #954 stage 2+ would introduce
        one).  Kept in the signature so the service-handler call site
        is symmetric with the 3D one.

        Returns the same diagnostic shape as the 3D version, with
        ``coefficient_before / coefficient_after`` carrying the 4-key
        ``{s, e, w, diffuse}`` dicts.
        """
        from .observation import WeightedSmear

        filtered_log = _filter_log_by_days_back(hourly_log, days_back)
        strategies = getattr(coordinator, "_unit_strategies", {}) or {}

        results: dict[str, dict] = {}
        target_entities = (
            [entity_id_filter] if entity_id_filter else list(energy_sensors)
        )

        solar_affected = getattr(coordinator, "_solar_affected_set", None)
        if not isinstance(solar_affected, (frozenset, set)):
            solar_affected = None

        for entity_id in target_entities:
            if entity_id not in energy_sensors:
                results[entity_id] = {"skip_reason": "unknown_entity"}
                continue
            if solar_affected is not None and entity_id not in solar_affected:
                results[entity_id] = {"skip_reason": "excluded_from_solar"}
                continue
            strategy = strategies.get(entity_id)
            if isinstance(strategy, WeightedSmear) and strategy.use_synthetic:
                results[entity_id] = {"skip_reason": "weighted_smear_excluded"}
                continue

            entity_results: dict[str, dict] = {}
            for regime in ("heating", "cooling"):
                samples, censored_mask, drop_counts = (
                    self._collect_batch_fit_samples_4d(
                        entity_id=entity_id,
                        regime=regime,
                        hourly_log=filtered_log,
                        coordinator=coordinator,
                        screen_affected_entities=screen_affected_entities,
                    )
                )
                n_unc = sum(1 for m in censored_mask if not m)
                n_cens = len(samples) - n_unc
                regime_diag: dict = {
                    "sample_count": len(samples),
                    "drop_counts": drop_counts,
                }

                entity_entry = solar_coefficients_4d_per_unit.get(entity_id)
                if isinstance(entity_entry, dict):
                    current_dict = entity_entry.get(regime) or {}
                else:
                    current_dict = {}
                current = {
                    "s": float(current_dict.get("s", 0.0)),
                    "e": float(current_dict.get("e", 0.0)),
                    "w": float(current_dict.get("w", 0.0)),
                    "diffuse": float(current_dict.get("diffuse", 0.0)),
                }
                regime_diag["coefficient_before"] = {
                    k: round(v, 5) for k, v in current.items()
                }

                if n_unc < TOBIT_MIN_UNCENSORED:
                    regime_diag["skip_reason"] = "insufficient_uncensored"
                    regime_diag["coefficient_after"] = regime_diag[
                        "coefficient_before"
                    ]
                    regime_diag["tobit_diagnostics"] = {
                        "n_uncensored": n_unc,
                        "n_censored": n_cens,
                        "censored_fraction": (
                            round(n_cens / len(samples), 3)
                            if samples
                            else 0.0
                        ),
                    }
                    entity_results[regime] = regime_diag
                    continue

                fit = LearningManager._solve_tobit(
                    samples,
                    censored_mask,
                    components=("s", "e", "w", "diffuse"),
                )
                if fit is None:
                    regime_diag["skip_reason"] = "warm_start_failed"
                    regime_diag["coefficient_after"] = regime_diag[
                        "coefficient_before"
                    ]
                    entity_results[regime] = regime_diag
                    continue

                tobit_diag = {
                    "iterations": fit["iterations"],
                    "converged": bool(fit["converged"]),
                    "failure_reason": fit.get("failure_reason"),
                    "sigma": round(fit["sigma"], 5),
                    "log_likelihood": round(fit["log_likelihood"], 4),
                    "n_uncensored": fit["n_uncensored"],
                    "n_censored": fit["n_censored"],
                    "censored_fraction": (
                        round(n_cens / len(samples), 3) if samples else 0.0
                    ),
                    "n_eff": round(fit["n_eff"], 2),
                }
                regime_diag["tobit_diagnostics"] = tobit_diag

                if fit["n_eff"] < TOBIT_MIN_NEFF:
                    regime_diag["skip_reason"] = (
                        "insufficient_effective_samples"
                    )
                    regime_diag["coefficient_after"] = regime_diag[
                        "coefficient_before"
                    ]
                    entity_results[regime] = regime_diag
                    continue

                if not fit["converged"]:
                    regime_diag["skip_reason"] = "did_not_converge"
                    regime_diag["coefficient_after"] = regime_diag[
                        "coefficient_before"
                    ]
                    entity_results[regime] = regime_diag
                    continue

                clamped = {
                    "s": max(0.0, min(SOLAR_COEFF_CAP, float(fit["s"]))),
                    "e": max(0.0, min(SOLAR_COEFF_CAP, float(fit["e"]))),
                    "w": max(0.0, min(SOLAR_COEFF_CAP, float(fit["w"]))),
                    "diffuse": max(
                        0.0, min(SOLAR_COEFF_CAP, float(fit["diffuse"]))
                    ),
                }

                # ``learned`` flag drives the no-prior detection: a
                # regime entry written by a prior batch / NLMS run
                # carries learned=True even when components have
                # decayed toward 0.  Falling back to "any component
                # > 1e-6" misclassifies a converged-near-zero diffuse
                # as no-prior and overwrites without damping.
                has_prior = bool(current_dict.get("learned", False)) or any(
                    current[k] > 1e-6 for k in ("s", "e", "w", "diffuse")
                )
                if has_prior:
                    blended = {
                        k: round(
                            BATCH_FIT_DAMPING * clamped[k]
                            + (1.0 - BATCH_FIT_DAMPING) * current[k],
                            5,
                        )
                        for k in ("s", "e", "w", "diffuse")
                    }
                    damping_applied = BATCH_FIT_DAMPING
                else:
                    blended = {
                        k: round(clamped[k], 5)
                        for k in ("s", "e", "w", "diffuse")
                    }
                    damping_applied = 1.0

                regime_diag["coefficient_after"] = blended
                regime_diag["damping_applied"] = damping_applied

                if dry_run:
                    regime_diag["applied"] = False
                    regime_diag["dry_run"] = True
                else:
                    self._update_unit_solar_coefficient(
                        entity_id,
                        blended,
                        solar_coefficients_4d_per_unit,
                        regime,
                        components=("s", "e", "w", "diffuse"),
                    )
                    regime_diag["applied"] = True
                    if seed_live_window:
                        # No live 4D Tobit window yet — flag the
                        # request for diagnostics but no-op.  Stage 3+
                        # of #954 would introduce a live sliding
                        # window analogous to the 3D one.
                        regime_diag["seeded_live_window"] = False
                        regime_diag["seed_skip_reason"] = (
                            "no_live_window_4d"
                        )
                entity_results[regime] = regime_diag

            results[entity_id] = entity_results

        return results

    def fit_solar_obstruction(
        self,
        hourly_log: list[dict],
        coordinator,
        *,
        entity_id: str | None = None,
        days_back: int | None = None,
        include_cooling: bool = True,
        dry_run: bool = False,
        stability_per_facade_per_entity: (
            dict[str, dict[tuple[str, str], bool]] | None
        ) = None,
    ) -> dict:
        """Per-entity per-facade solar-window obstruction fit (v9).

        For each facade (S, E, W) independently, find TWO critical
        elevations:

        * ``critical_elev_low`` — below this, terrain / neighbouring
          buildings block direct beam.  Samples with ``sun_elev < low``
          are gated (zeroed) in the search.
        * ``critical_elev_high`` — above this, an overhang / terrace
          blocks direct beam.  Samples with ``sun_elev > high`` are
          gated (zeroed) in the search (legacy behaviour, v7/v8).

        Two independent brute-force sweeps are run per facade.  The
        low sweep searches ``[OBSTRUCTION_FIT_LOW_MIN_ELEV_DEG,
        OBSTRUCTION_FIT_LOW_MAX_ELEV_DEG]``; the high sweep searches
        ``[OBSTRUCTION_FIT_HIGH_MIN_ELEV_DEG,
        OBSTRUCTION_FIT_HIGH_MAX_ELEV_DEG]``.

        Each candidate ``c`` is scored on an adaptive kNN window — the
        ``OBSTRUCTION_FIT_WINDOW_K_PER_SIDE`` nearest samples below and
        above ``c`` by elevation distance, subject to a maximum angular
        span of ``OBSTRUCTION_FIT_WINDOW_MAX_ANGULAR_DEG``.  The local
        gated SSE is compared against the local null SSE (no gate
        anywhere in the window).

        **Shutdown-flagged hours** (``solar_dominant_entities``) serve as
        constraints on the feasible range rather than SSE contributors.
        A shutdown event at elevation E proves unobstructed sun at E,
        so ``critical_elev_low < E < critical_elev_high`` must hold.
        Candidates that would gate a proven-unobstructed elevation are
        rejected before the SSE comparison.  This uses shutdown hours'
        strong geometry signal without their saturation bias — shutdown
        hours carry no information about coefficient magnitude
        (``actual ≈ 0`` saturates ``actual_impact = base``), so they
        cannot participate in the SSE comparison without creating
        spurious obstruction finds at shutdown-rich elevations.

        ``learned=True`` for each boundary requires both: relative
        improvement above ``OBSTRUCTION_FIT_SSE_IMPROVEMENT_THRESHOLD``
        and absolute per-sample RMS reduction above
        ``OBSTRUCTION_FIT_MIN_RMS_REDUCTION_KWH`` (both computed on
        the local window).  ``OBSTRUCTION_FIT_MIN_SAMPLES_PER_SIDE``
        is the absolute floor.

        Samples are computed with the EXISTING per-entity 4D coefficients
        held fixed.  The fit does not refit coefficients — that is the
        ``batch_fit_solar_4d`` service's job, recommended as a follow-up
        once obstruction lands.

        **#1020 — suggestion-not-auto-write.**  This function no longer
        writes to ``_critical_elev_per_facade_per_unit``.  Boundaries
        that pass all four local apply-gate checks (``learned`` ∧
        ``sse_improvement >= 0.30`` ∧ plausible per side ∧ stable
        across windows) are surfaced as ``suggested_gates`` only; the
        user accepts each suggestion explicitly via the
        ``apply_obstruction_gate`` service.  ``dry_run`` is retained as
        a passthrough field in the result for backward-compat but no
        longer affects state.  Stability is supplied by the service
        handler (multi-window orchestration); single-pass callers get
        ``stable=True`` by default and the stability gate is bypassed.

        Potentials are reconstructed inline (not via
        ``calculate_unit_potential_4d``) so the existing gate state does
        not contaminate the fit input.

        Heating and cooling samples enter the same per-facade SSE search
        (geometric shadow is mode-invariant; #1006).  Sign correction is
        applied per regime: heating ``actual_impact = expected_base −
        actual``, cooling ``actual_impact = actual − expected_base``.

        Cooling-specific deviations from the heating path:
          * Saturation exclusion (``actual_impact ≥ 0.95 × base``) is
            dropped.  Residential HP cooling capacity ≫ practical load.
          * ``actual_impact ≤ 0`` IS excluded for the obstruction fit
            (mirrors the heating skip).
          * Shutdown constraint is heating-only (cooling has no
            analogous flag — the ``solar_dominant_entities`` field
            carries heating-specific shutdown semantics).
          * Baseline lookup routes to ``COOLING_WIND_BUCKET``.

        Args:
            hourly_log: Coordinator's hourly log.
            coordinator: For ``solar`` (sun-pos + transmittance),
                ``_solar_coefficients_4d_per_unit`` (existing coeffs),
                ``screen_config_for_entity``, ``_solar_affected_set``,
                ``energy_sensors``, ``critical_elev_for_entity`` (read
                "before" state), ``_critical_elev_per_facade_per_unit``
                (written on success unless ``dry_run``).
            entity_id: When provided, restrict the fit to a single
                entity.  ``None`` fits all entities.
            days_back: Restrict to last N days of ``hourly_log``.
            include_cooling: When False, skip all cooling-mode samples.
            dry_run: When True, run the analysis but write nothing.
                #1006 behaviour.  Set False to isolate whether cooling
                contributions are driving spurious gates on a given
                fit — useful diagnostic when results disagree with
                physical knowledge of the install.
            dry_run: When True, run the analysis but write nothing.

        Returns:
            Per-entity dict::

                {
                    "<entity_id>": {
                        "s": {
                            "learned": bool,
                            "applicable": bool,
                            "applicable_reason": str | None,
                            "critical_elev_before": float | None,
                            "critical_elev_after": float | None,
                            "best_critical_elev": float,
                            "sse_improvement_ratio": float,
                            "sse_flat": float,
                            "sse_best": float,
                            "sse_null_local": float,
                            "rms_reduction_kwh": float,
                            "window_elev_lo": float,
                            "window_elev_hi": float,
                            "n_samples": int,
                            "n_heating_samples": int,
                            "n_cooling_samples": int,
                            "n_below_best": int,
                            "n_above_best": int,
                            "skip_reason": str | None,
                        },
                        "e": {...},
                        "w": {...},
                    },
                    "<other_entity_id>": {...},
                    "n_skipped_cooling_unlearned": int,
                    "dry_run": bool,
                }
        """
        from .solar import reconstruct_hour_inputs

        # Time-window filter — see ``_filter_log_by_days_back`` for the
        # parse-then-compare rationale (lex-compare on ISO strings is
        # wrong under non-UTC offsets near the cutoff boundary).
        hourly_log = _filter_log_by_days_back(hourly_log, days_back)

        solar = getattr(coordinator, "solar", None)
        coeffs_4d_per_unit = getattr(coordinator, "_solar_coefficients_4d_per_unit", {}) or {}
        energy_sensors = getattr(coordinator, "energy_sensors", None) or []
        solar_affected = getattr(coordinator, "_solar_affected_set", None)
        if not isinstance(solar_affected, (frozenset, set)):
            solar_affected = None
        scr_fn = getattr(coordinator, "screen_config_for_entity", None)
        crit_fn = getattr(coordinator, "critical_elev_for_entity", None)
        crit_state_per_unit = getattr(
            coordinator, "_critical_elev_per_facade_per_unit", None
        )
        if not isinstance(crit_state_per_unit, dict):
            crit_state_per_unit = {}

        # Recalc fallback for ``unit_expected_base`` (#1005).  Pre-1.3.8
        # ``hourly_log`` entries only carry ``unit_expected_breakdown``
        # (net, post-solar); the dark-sky baseline this fit needs was
        # never persisted.  Reconstruct it on-the-fly from the current
        # per-unit base model when the logged value is absent — the
        # geometric obstruction fit holds coefficients fixed and only
        # needs a stable baseline, so current-model drift relative to
        # log-time is acceptable (overhang shadow is a step-function in
        # sun elevation, not a function of base-bucket state).
        correlation_data_per_unit = getattr(coordinator, "_correlation_data_per_unit", {}) or {}
        statistics_mgr = getattr(coordinator, "statistics", None)
        balance_point = getattr(coordinator, "balance_point", 17.0)

        # Candidate entities: either the single ``entity_id`` argument
        # or all energy sensors filtered by ``_solar_affected_set``.
        if entity_id is not None:
            candidate_entities = [entity_id]
        else:
            candidate_entities = list(energy_sensors)
        if solar_affected is not None:
            candidate_entities = [
                e for e in candidate_entities if e in solar_affected
            ]
        candidate_set = set(candidate_entities)

        # Per-(entity, facade) sample lists, per-regime counters, and
        # shutdown-constraint elevations.  Shutdown samples participate
        # in constraint enforcement (hard bound on feasible cutoff range)
        # but NOT in SSE — see docstring.
        samples_per_entity: dict[str, dict[str, list[tuple[float, float, float]]]] = {
            eid: {"s": [], "e": [], "w": []} for eid in candidate_entities
        }
        shutdown_elevs_per_entity: dict[str, dict[str, list[float]]] = {
            eid: {"s": [], "e": [], "w": []} for eid in candidate_entities
        }
        regime_counts_per_entity: dict[str, dict[str, dict[str, int]]] = {
            eid: {
                "s": {"heating": 0, "cooling": 0},
                "e": {"heating": 0, "cooling": 0},
                "w": {"heating": 0, "cooling": 0},
            }
            for eid in candidate_entities
        }
        n_skipped_cooling_unlearned = 0

        if solar is None:
            return {
                eid: {
                    "s": {"learned": False, "skip_reason": "no_solar_calculator"},
                    "e": {"learned": False, "skip_reason": "no_solar_calculator"},
                    "w": {"learned": False, "skip_reason": "no_solar_calculator"},
                }
                for eid in candidate_entities
            } | {
                "n_skipped_cooling_unlearned": 0,
                "dry_run": dry_run,
                "suggested_gates": [],
            }

        for entry in hourly_log:
            if entry.get("auxiliary_active", False):
                continue
            unit_modes = entry.get("unit_modes", {}) or {}
            shutdown_entities = set(entry.get("solar_dominant_entities", []) or [])

            inputs, _fail = reconstruct_hour_inputs(entry, solar)
            if inputs is None:
                continue
            # Helper rejects only on source=="none"; preserve the
            # original stricter zero-gate here (#1011 scope note).
            if inputs.dni <= 0.0 and inputs.dhi <= 0.0:
                continue
            sun_elev = inputs.sun_elev
            sun_az = inputs.sun_az
            dni_v = inputs.dni
            dhi_v = inputs.dhi
            correction_percent = float(entry.get("correction_percent", 100.0))
            elev_rad = math.radians(sun_elev)
            az_rad = math.radians(sun_az)
            cos_elev = math.cos(elev_rad)
            dni_horiz = max(0.0, dni_v) * cos_elev

            unit_breakdown = entry.get("unit_breakdown", {}) or {}
            # Per-entity baseline is recalculated from the current per-unit
            # base model (#1005) — single canonical source independent of
            # which integration version wrote the log entry.  The logged
            # ``unit_expected_base`` field (post-1.3.8) is intentionally
            # NOT consulted here.
            entry_temp_key = entry.get("temp_key")
            entry_wind_bucket = entry.get("wind_bucket")
            try:
                entry_temp = float(entry.get("temp", balance_point))
            except (TypeError, ValueError):
                entry_temp = balance_point

            for eid in candidate_entities:
                if eid not in unit_breakdown:
                    continue

                unit_mode = unit_modes.get(eid, MODE_HEATING)
                regime = _solar_coeff_regime(unit_mode)
                if regime is None:
                    continue  # OFF / DHW / unknown — no learnable solar coeff

                # Cooling opt-out: when include_cooling=False, skip all
                # cooling-mode samples (heating-only obstruction fit).
                # Diagnostic flag for isolating cooling contributions.
                if regime == "cooling" and not include_cooling:
                    continue

                # Shutdown-flagged heating hours: record the elevation
                # per dominating facade as a constraint (proves
                # unobstructed sun at that elevation) but do NOT
                # collect as an SSE sample — shutdown hours carry
                # no coefficient-magnitude signal (actual ≈ 0
                # saturates actual_impact = base).  Cooling has no
                # analogous flag — see docstring.
                if regime == "heating" and eid in shutdown_entities:
                    if scr_fn is not None:
                        sd_screen = scr_fn(eid)
                    else:
                        sd_screen = getattr(coordinator, "screen_config", None)
                    sd_ts, sd_te, sd_tw = solar._screen_transmittance_vector(
                        correction_percent, sd_screen
                    )
                    sd_ps = dni_horiz * max(0.0, -math.cos(az_rad)) * sd_ts
                    sd_pe = dni_horiz * max(0.0, math.sin(az_rad)) * sd_te
                    sd_pw = dni_horiz * max(0.0, -math.sin(az_rad)) * sd_tw
                    sd_pd = max(0.0, dhi_v) * 0.5 * (sd_ts + sd_te + sd_tw) / 3.0
                    sd_sum = sd_ps + sd_pe + sd_pw + sd_pd
                    if sd_sum > 0.0:
                        for sd_f, sd_pf in (
                            ("s", sd_ps), ("e", sd_pe), ("w", sd_pw)
                        ):
                            if (sd_pf / sd_sum) > OBSTRUCTION_FIT_DOMINANCE_RATIO:
                                shutdown_elevs_per_entity[eid][sd_f].append(sun_elev)
                    continue

                entity_coeffs = coeffs_4d_per_unit.get(eid, {}) or {}
                regime_coeffs = entity_coeffs.get(regime, {}) or {}
                if not regime_coeffs.get("learned"):
                    if regime == "cooling":
                        n_skipped_cooling_unlearned += 1
                    continue
                c_s = float(regime_coeffs.get("s", 0.0))
                c_e = float(regime_coeffs.get("e", 0.0))
                c_w = float(regime_coeffs.get("w", 0.0))
                c_d = float(regime_coeffs.get("diffuse", 0.0))

                # Baseline: heating reads the entry's wind bucket; cooling
                # routes to COOLING_WIND_BUCKET regardless of the entry's
                # wind classification (cooling samples live in the
                # dedicated cooling bucket per CLAUDE.md).  Per-unit base
                # model is recalculated (#1005) — the logged
                # ``unit_expected_base`` field is intentionally NOT
                # consulted.
                lookup_bucket = (
                    COOLING_WIND_BUCKET if regime == "cooling" else entry_wind_bucket
                )
                expected_base = 0.0
                if statistics_mgr is not None and entry_temp_key and lookup_bucket:
                    unit_data = correlation_data_per_unit.get(eid, {}) or {}
                    try:
                        expected_base = float(
                            statistics_mgr._get_prediction_from_model(
                                unit_data,
                                entry_temp_key,
                                lookup_bucket,
                                entry_temp,
                                balance_point,
                            )
                            or 0.0
                        )
                    except Exception:
                        expected_base = 0.0
                if expected_base <= 0.0:
                    continue
                actual = float(unit_breakdown.get(eid, 0.0) or 0.0)

                if regime == "heating":
                    actual_impact = expected_base - actual
                    if actual_impact <= 0.0:
                        continue
                    if actual_impact >= BATCH_FIT_SATURATION_RATIO * expected_base:
                        continue  # saturated — no slope info for the fit
                else:  # cooling
                    # Sign inverted: solar increases cooling load, so
                    # actual > expected_base when sun acts normally.
                    # Saturation gate dropped (cooling capacity ≫ load).
                    # ``<= 0`` gate APPLIED here (mirrors heating) —
                    # the binary gate search would otherwise be biased
                    # toward learning a gate to "hide" negative-impact
                    # samples (deterministic SSE reduction whenever
                    # br < 0 and cf_pot > 0).  See docstring.
                    actual_impact = actual - expected_base
                    if actual_impact <= 0.0:
                        continue

                if scr_fn is not None:
                    entity_screen = scr_fn(eid)
                else:
                    entity_screen = getattr(coordinator, "screen_config", None)
                t_s, t_e, t_w = solar._screen_transmittance_vector(
                    correction_percent, entity_screen
                )
                # Ungated facade potentials (raw geometry, no obstruction gate
                # — the fit will impose candidate gates downstream).
                pot_s = dni_horiz * max(0.0, -math.cos(az_rad)) * t_s
                pot_e = dni_horiz * max(0.0, math.sin(az_rad)) * t_e
                pot_w = dni_horiz * max(0.0, -math.sin(az_rad)) * t_w
                pot_d = max(0.0, dhi_v) * 0.5 * (t_s + t_e + t_w) / 3.0

                pot_sum = pot_s + pot_e + pot_w + pot_d
                if pot_sum <= 0.0:
                    continue

                # Per-facade dominance filter: only include this sample
                # for facade f if pot_f / Σpot > dominance ratio.
                for facade, pot_f, c_f, others_pred in (
                    ("s", pot_s, c_s, c_e * pot_e + c_w * pot_w + c_d * pot_d),
                    ("e", pot_e, c_e, c_s * pot_s + c_w * pot_w + c_d * pot_d),
                    ("w", pot_w, c_w, c_s * pot_s + c_e * pot_e + c_d * pot_d),
                ):
                    if pot_sum > 0.0 and (pot_f / pot_sum) > OBSTRUCTION_FIT_DOMINANCE_RATIO:
                        baseline_residual = actual_impact - others_pred
                        # Per-entity contribution to the facade's fit:
                        # (sun_elev, c_f · pot_f, baseline_residual).
                        # Predicted_under_gate = baseline_residual_pred
                        # = (gate_active ? 0 : c_f · pot_f).
                        # We store c_f · pot_f as the facade prediction.
                        samples_per_entity[eid][facade].append(
                            (sun_elev, c_f * pot_f, baseline_residual)
                        )
                        regime_counts_per_entity[eid][facade][regime] += 1

        # Per-(entity, facade) solar-window fit (v9).
        # Two independent brute-force sweeps per facade:
        #   LOW  — gate below cutoff (lower horizon: terrain/neighbours)
        #   HIGH — gate above cutoff (upper horizon: overhang/terrace)
        # Sweep range is data-adaptive per facade (#1016 Problem 2) —
        # see ``_data_adaptive_bounds`` below.

        K = OBSTRUCTION_FIT_WINDOW_K_PER_SIDE
        K_FLOOR = OBSTRUCTION_FIT_MIN_SAMPLES_PER_SIDE
        SPAN_MAX = OBSTRUCTION_FIT_WINDOW_MAX_ANGULAR_DEG

        def _sweep_boundary(
            samples_sorted: list[tuple[float, float, float]],
            elev_arr: list[float],
            min_elev: float,
            max_elev: float,
            n_steps: int,
            *,
            is_low: bool,
            constraint_elev: float | None = None,
        ) -> dict:
            """Brute-force sweep for a single boundary (low or high horizon).

            ``is_low=True``: samples BELOW the candidate cutoff are gated
            (zeroed) — models a lower horizon where terrain/neighbours
            block direct beam at low elevations.  ``is_low=False``:
            samples ABOVE the candidate cutoff are gated — legacy upper
            horizon where overhang/terrace blocks direct beam at high
            elevations.

            ``constraint_elev``: when provided, constrains the feasible
            cutoff range via a hard constraint from shutdown hours.
            For low search: cutoff must be ≤ constraint_elev (can't
            claim sun is blocked at an elevation where shutdown proved
            it was visible).  For high search: cutoff must be ≥
            constraint_elev.
            """
            best_improvement = -math.inf
            best_crit = None
            best_below = 0
            best_above = 0
            best_sse_gated_local = None
            best_sse_null_local = None
            best_window_lo = None
            best_window_hi = None
            for i in range(n_steps):
                crit = min_elev + i * OBSTRUCTION_FIT_STEP_DEG
                # Hard constraint: reject cutoffs that would gate a
                # proven-unobstructed elevation (shutdown hours).
                if constraint_elev is not None:
                    if is_low and crit > constraint_elev:
                        continue  # can't gate above proven elevation
                    if not is_low and crit < constraint_elev:
                        continue  # can't gate below proven elevation

                split = bisect.bisect_right(elev_arr, crit)
                below_start = max(0, split - K)
                above_end = min(len(samples_sorted), split + K)
                n_below = split - below_start
                n_above = above_end - split
                if n_below < K_FLOOR or n_above < K_FLOOR:
                    continue
                elev_lo = samples_sorted[below_start][0]
                elev_hi = samples_sorted[above_end - 1][0]
                if (elev_hi - elev_lo) > SPAN_MAX:
                    continue
                sse_gated = 0.0
                sse_null = 0.0
                if is_low:
                    # Low search: samples BELOW cutoff gated (zeroed).
                    for j in range(below_start, split):
                        _, cf_pot, br = samples_sorted[j]
                        sse_gated += br * br
                        d = br - cf_pot
                        sse_null += d * d
                    for j in range(split, above_end):
                        _, cf_pot, br = samples_sorted[j]
                        d = br - cf_pot
                        sse_gated += d * d
                        sse_null += d * d
                else:
                    # High search: samples ABOVE cutoff gated (zeroed).
                    for j in range(below_start, split):
                        _, cf_pot, br = samples_sorted[j]
                        d = br - cf_pot
                        sse_gated += d * d
                        sse_null += d * d
                    for j in range(split, above_end):
                        _, cf_pot, br = samples_sorted[j]
                        sse_gated += br * br
                        d = br - cf_pot
                        sse_null += d * d
                if sse_null <= 0.0:
                    continue
                improvement = 1.0 - sse_gated / sse_null
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_crit = crit
                    best_below = n_below
                    best_above = n_above
                    best_sse_gated_local = sse_gated
                    best_sse_null_local = sse_null
                    best_window_lo = elev_lo
                    best_window_hi = elev_hi
            return {
                "best_crit": best_crit,
                "best_improvement": best_improvement,
                "best_below": best_below,
                "best_above": best_above,
                "best_sse_gated_local": best_sse_gated_local,
                "best_sse_null_local": best_sse_null_local,
                "best_window_lo": best_window_lo,
                "best_window_hi": best_window_hi,
            }

        result: dict = {
            "dry_run": dry_run,
            "n_skipped_cooling_unlearned": n_skipped_cooling_unlearned,
        }
        # #1020: ``suggested_gates`` is the authoritative output for
        # the handler — auto-write was removed, so the handler builds
        # its UI from this list rather than diff-ing the gate state.
        suggested_gates: list[dict] = []
        # #1020 stability gating: the multi-window orchestrator in
        # the service handler pre-computes per-(facade, side)
        # stability and re-runs this fit injecting it via
        # ``stability_per_facade_per_entity``.  Single-pass callers
        # (tests, single-window debug runs) pass nothing and every
        # boundary defaults to stable=True (the stability check is
        # bypassed).  Keyed per entity so a single fit covering
        # multiple entities maintains independent stability per
        # entity.
        stability_input = stability_per_facade_per_entity or {}
        for eid in candidate_entities:
            # "Before" state via the coordinator helper — v9 nested shape
            # ``{"s": {"low": ..., "high": ...}, ...}``.
            if callable(crit_fn):
                entity_crit_before = crit_fn(eid)
            else:
                entity_crit_before = {
                    "s": {"low": None, "high": None},
                    "e": {"low": None, "high": None},
                    "w": {"low": None, "high": None},
                }

            entity_result: dict = {}
            stability_per_facade = stability_input.get(eid, {}) or {}

            for facade in ("s", "e", "w"):
                samples = samples_per_entity[eid][facade]
                regime_counts = regime_counts_per_entity[eid][facade]
                shutdown_elevs = shutdown_elevs_per_entity[eid][facade]

                # Shutdown constraint: shutdown-flagged hours prove the
                # sun was visible at that elevation, so they establish
                # hard bounds on the feasible cutoff range.  Using the
                # raw min/max would let a single false-positive shutdown
                # classification at an extreme elevation permanently
                # block the gate (shutdown detection has known noise);
                # require ``OBSTRUCTION_FIT_SHUTDOWN_CONSTRAINT_MIN_N``
                # shutdown samples below/above the candidate boundary
                # before treating it as a constraint.  Below floor →
                # no constraint (None).
                n_required = OBSTRUCTION_FIT_SHUTDOWN_CONSTRAINT_MIN_N
                if len(shutdown_elevs) >= n_required:
                    sorted_sd = sorted(shutdown_elevs)
                    # ``shutdown_min`` = elevation of the ``n_required``-th
                    # lowest shutdown sample → low cutoff must be ≤ this
                    # (n_required samples have proven sun visible at or
                    # below it, so a low gate cannot sit higher).
                    shutdown_min = sorted_sd[n_required - 1]
                    # Symmetric on the high side.
                    shutdown_max = sorted_sd[-n_required]
                else:
                    shutdown_min = None
                    shutdown_max = None

                before_gate = entity_crit_before.get(facade)
                if isinstance(before_gate, dict):
                    before_low = before_gate.get("low")
                    before_high = before_gate.get("high")
                else:
                    before_low = None
                    before_high = None

                # Common fields across low + high sub-results.
                common = {
                    "n_samples": len(samples),
                    "n_heating_samples": regime_counts["heating"],
                    "n_cooling_samples": regime_counts["cooling"],
                    "n_shutdown_constraints": len(shutdown_elevs),
                    "shutdown_min_elev": shutdown_min,
                    "shutdown_max_elev": shutdown_max,
                }

                if len(samples) < 2 * K_FLOOR:
                    entity_result[facade] = {
                        **common,
                        "low": {
                            "learned": False,
                            "applicable": False,
                            "applicable_reason": "not_learned",
                            "skip_reason": "insufficient_samples",
                            "critical_elev_before": before_low,
                            "critical_elev_after": before_low,
                            "best_critical_elev": None,
                            "sse_improvement_ratio": 0.0,
                            "sse_best": None,
                            "sse_null_local": None,
                            "rms_reduction_kwh": 0.0,
                            "window_elev_lo": None,
                            "window_elev_hi": None,
                            "n_below_best": 0,
                            "n_above_best": 0,
                        },
                        "high": {
                            "learned": False,
                            "applicable": False,
                            "applicable_reason": "not_learned",
                            "skip_reason": "insufficient_samples",
                            "critical_elev_before": before_high,
                            "critical_elev_after": before_high,
                            "best_critical_elev": None,
                            "sse_improvement_ratio": 0.0,
                            "sse_best": None,
                            "sse_null_local": None,
                            "rms_reduction_kwh": 0.0,
                            "window_elev_lo": None,
                            "window_elev_hi": None,
                            "n_below_best": 0,
                            "n_above_best": 0,
                        },
                    }
                    continue

                # Global SSE under no gate — diagnostic only.
                sse_flat = sum(
                    (br - cf_pot) ** 2 for (_elev, cf_pot, br) in samples
                )

                # Sort once on elevation.
                samples_sorted = sorted(samples, key=lambda s: s[0])
                elev_arr = [s[0] for s in samples_sorted]

                def _finalize_boundary(
                    sweep: dict,
                    boundary: str,
                    learned: bool,
                    skip_reason: str | None,
                    crit_before: float | None,
                ) -> dict:
                    """Package a single boundary's sweep result."""
                    best_crit = sweep["best_crit"]
                    result_d = {
                        "learned": learned,
                        "critical_elev_before": crit_before,
                        "critical_elev_after": (
                            best_crit if learned else crit_before
                        ),
                        "best_critical_elev": best_crit,
                        "sse_improvement_ratio": (
                            sweep["best_improvement"]
                            if sweep["best_improvement"] != -math.inf
                            else 0.0
                        ),
                        "sse_best": sweep["best_sse_gated_local"],
                        "sse_null_local": sweep["best_sse_null_local"],
                        "rms_reduction_kwh": 0.0,
                        "window_elev_lo": sweep["best_window_lo"],
                        "window_elev_hi": sweep["best_window_hi"],
                        "n_below_best": sweep["best_below"],
                        "n_above_best": sweep["best_above"],
                        "skip_reason": skip_reason,
                    }
                    if best_crit is not None and sweep["best_sse_null_local"] is not None:
                        n_window = sweep["best_below"] + sweep["best_above"]
                        rms = math.sqrt(
                            max(
                                0.0,
                                sweep["best_sse_null_local"]
                                - (sweep["best_sse_gated_local"] or 0.0),
                            )
                            / max(1, n_window)
                        )
                        result_d["rms_reduction_kwh"] = rms
                    return result_d

                # Data-adaptive sweep bounds (#1016 Problem 2).  A
                # candidate cutoff can only pass the K_FLOOR constraint
                # when at least ``K_FLOOR`` samples sit on each side of
                # it; ``elev_arr[K_FLOOR - 1]`` is the lowest crit with
                # K_FLOOR samples below, and ``elev_arr[-K_FLOOR]`` is
                # the highest crit with K_FLOOR samples above.
                # Intersecting these with the configured search range
                # gives an honest per-facade range: the configured
                # range is the policy ceiling, the data range is the
                # physical floor.  When the data does not reach the
                # configured side at all (e.g. winter at 60°N with no
                # sun above ~15°), the corresponding sweep is skipped
                # explicitly with ``data_does_not_reach_*`` rather
                # than silently failing K_FLOOR on every candidate.
                data_min_for_low = elev_arr[K_FLOOR - 1]
                data_max_for_high = elev_arr[-K_FLOOR]
                low_min_eff = max(
                    OBSTRUCTION_FIT_LOW_MIN_ELEV_DEG, data_min_for_low
                )
                low_max_eff = OBSTRUCTION_FIT_LOW_MAX_ELEV_DEG
                high_min_eff = OBSTRUCTION_FIT_HIGH_MIN_ELEV_DEG
                high_max_eff = min(
                    OBSTRUCTION_FIT_HIGH_MAX_ELEV_DEG, data_max_for_high
                )

                # --- LOW boundary sweep ---
                # Shutdown constraint: low cutoff must be ≤ shutdown_min
                # (can't claim sun is blocked above where shutdown
                # proved it was visible).  Same constraint for first
                # fit and re-fit — the ``before_low`` value does not
                # change what the data is allowed to prove.
                if low_min_eff > low_max_eff:
                    low_learned = False
                    low_skip = "data_does_not_reach_low_regime"
                    low_result = _finalize_boundary(
                        {
                            "best_crit": None,
                            "best_improvement": -math.inf,
                            "best_below": 0, "best_above": 0,
                            "best_sse_gated_local": None,
                            "best_sse_null_local": None,
                            "best_window_lo": None,
                            "best_window_hi": None,
                        },
                        "low", low_learned, low_skip, before_low,
                    )
                else:
                    n_steps_low = int(
                        (low_max_eff - low_min_eff) / OBSTRUCTION_FIT_STEP_DEG
                    ) + 1
                    sweep_low = _sweep_boundary(
                        samples_sorted, elev_arr,
                        low_min_eff, low_max_eff, n_steps_low,
                        is_low=True,
                        constraint_elev=shutdown_min,
                    )
                    if sweep_low["best_crit"] is not None:
                        passes_rel = sweep_low["best_improvement"] > OBSTRUCTION_FIT_SSE_IMPROVEMENT_THRESHOLD
                        n_window_low = sweep_low["best_below"] + sweep_low["best_above"]
                        rms_low = math.sqrt(
                            max(
                                0.0,
                                (sweep_low["best_sse_null_local"] or 0.0)
                                - (sweep_low["best_sse_gated_local"] or 0.0),
                            )
                            / max(1, n_window_low)
                        )
                        passes_abs = rms_low > OBSTRUCTION_FIT_MIN_RMS_REDUCTION_KWH
                        if passes_rel and passes_abs:
                            low_learned = True
                            low_skip = None
                        elif not passes_rel:
                            low_learned = False
                            low_skip = "below_sse_threshold"
                        else:
                            low_learned = False
                            low_skip = "below_rms_amplitude_threshold"
                    else:
                        low_learned = False
                        low_skip = "no_candidate_passed_sample_floor"
                    low_result = _finalize_boundary(
                        sweep_low, "low", low_learned, low_skip, before_low,
                    )

                # --- HIGH boundary sweep ---
                # Cooling-data floor (#1020).  At northern latitudes
                # with heating-only data, sun rarely reaches the HIGH
                # regime — the fit either finds nothing or "finds" a
                # marginal artifact.  Below the configured cooling-
                # sample floor for this facade, skip the HIGH sweep
                # honestly with ``insufficient_cooling_for_high_regime``
                # rather than letting a low-N fit drive a suggestion.
                # The check sits before the data-adaptive range check
                # because data coverage and signal credibility are
                # distinct concerns — heating-only at 60°N can have
                # ample LOW-regime data while structurally lacking
                # HIGH-regime samples to validate any artifact found.
                if (
                    regime_counts["cooling"]
                    < OBSTRUCTION_FIT_MIN_COOLING_SAMPLES_FOR_HIGH
                ):
                    high_learned = False
                    high_skip = "insufficient_cooling_for_high_regime"
                    high_result = _finalize_boundary(
                        {
                            "best_crit": None,
                            "best_improvement": -math.inf,
                            "best_below": 0, "best_above": 0,
                            "best_sse_gated_local": None,
                            "best_sse_null_local": None,
                            "best_window_lo": None,
                            "best_window_hi": None,
                        },
                        "high", high_learned, high_skip, before_high,
                    )
                # Shutdown constraint mirrors LOW: high cutoff must be
                # ≥ shutdown_max.
                elif high_min_eff > high_max_eff:
                    high_learned = False
                    high_skip = "data_does_not_reach_high_regime"
                    high_result = _finalize_boundary(
                        {
                            "best_crit": None,
                            "best_improvement": -math.inf,
                            "best_below": 0, "best_above": 0,
                            "best_sse_gated_local": None,
                            "best_sse_null_local": None,
                            "best_window_lo": None,
                            "best_window_hi": None,
                        },
                        "high", high_learned, high_skip, before_high,
                    )
                else:
                    n_steps_high = int(
                        (high_max_eff - high_min_eff) / OBSTRUCTION_FIT_STEP_DEG
                    ) + 1
                    sweep_high = _sweep_boundary(
                        samples_sorted, elev_arr,
                        high_min_eff, high_max_eff, n_steps_high,
                        is_low=False,
                        constraint_elev=shutdown_max,
                    )
                    if sweep_high["best_crit"] is not None:
                        passes_rel = sweep_high["best_improvement"] > OBSTRUCTION_FIT_SSE_IMPROVEMENT_THRESHOLD
                        n_window_high = sweep_high["best_below"] + sweep_high["best_above"]
                        rms_high = math.sqrt(
                            max(
                                0.0,
                                (sweep_high["best_sse_null_local"] or 0.0)
                                - (sweep_high["best_sse_gated_local"] or 0.0),
                            )
                            / max(1, n_window_high)
                        )
                        passes_abs = rms_high > OBSTRUCTION_FIT_MIN_RMS_REDUCTION_KWH
                        if passes_rel and passes_abs:
                            high_learned = True
                            high_skip = None
                        elif not passes_rel:
                            high_learned = False
                            high_skip = "below_sse_threshold"
                        else:
                            high_learned = False
                            high_skip = "below_rms_amplitude_threshold"
                    else:
                        high_learned = False
                        high_skip = "no_candidate_passed_sample_floor"
                    high_result = _finalize_boundary(
                        sweep_high, "high", high_learned, high_skip, before_high,
                    )

                facade_result = {
                    **common,
                    "sse_flat": sse_flat,
                    "low": low_result,
                    "high": high_result,
                }
                entity_result[facade] = facade_result

                # Collect writes for this facade.  Sanity-check the
                # combined window: when both sweeps fire and the gap
                # ``high - low`` is below ``OBSTRUCTION_FIT_MIN_WINDOW_DEG``
                # the combined gate would zero direct beam across most
                # of the facade's elevation range — physically
                # implausible and a known failure mode when only one
                # true gate exists (the data above a true high gate
                # has ``br=0`` everywhere, which the LOW sweep
                # mathematically interprets as "below a low gate",
                # and vice versa).  Without shutdown samples, neither
                # sweep constrains the other.  Keep only the boundary
                # with stronger evidence (higher SSE improvement
                # ratio); discard the weaker one as the spurious
                # mirror finding.  Surface the rejection as
                # ``skip_reason = combined_window_too_narrow`` so the
                # diagnostic shows why one side was suppressed.
                new_low = low_result["best_critical_elev"] if low_learned else before_low
                new_high = high_result["best_critical_elev"] if high_learned else before_high
                if (
                    low_learned
                    and high_learned
                    and (new_high - new_low) < OBSTRUCTION_FIT_MIN_WINDOW_DEG
                ):
                    reason = "combined_window_too_narrow"
                    low_imp = low_result["sse_improvement_ratio"] or 0.0
                    high_imp = high_result["sse_improvement_ratio"] or 0.0
                    if high_imp >= low_imp:
                        low_result["learned"] = False
                        low_result["skip_reason"] = reason
                        low_result["critical_elev_after"] = before_low
                        low_learned = False
                        new_low = before_low
                    else:
                        high_result["learned"] = False
                        high_result["skip_reason"] = reason
                        high_result["critical_elev_after"] = before_high
                        high_learned = False
                        new_high = before_high

                # Apply-gate (#1020).  Four local checks per side
                # with no cross-boundary dependency: learned ∧
                # sse_improvement ≥ 0.30 ∧ plausible ∧ stable.
                # Stability is supplied by the multi-window handler
                # orchestrator; single-pass callers (tests, ad-hoc
                # debug runs) get stable=True by default and the
                # stability gate is effectively bypassed.
                stable_low = stability_per_facade.get(
                    (facade, "low"), True,
                )
                stable_high = stability_per_facade.get(
                    (facade, "high"), True,
                )
                low_applicable, low_app_reason = _boundary_applicable(
                    low_result, "low",
                    stable_across_windows=stable_low,
                )
                high_applicable, high_app_reason = _boundary_applicable(
                    high_result, "high",
                    stable_across_windows=stable_high,
                )
                low_result["applicable"] = low_applicable
                low_result["applicable_reason"] = low_app_reason
                high_result["applicable"] = high_applicable
                high_result["applicable_reason"] = high_app_reason

                # ``critical_elev_after`` reflects what the gate would
                # be IF the suggestion were accepted.  Auto-write was
                # removed in #1020 — passing gates surface as
                # ``suggested_gates`` only, and the user accepts
                # explicitly via ``apply_obstruction_gate``.  Not-
                # applicable boundaries reset ``critical_elev_after``
                # to the pre-fit value for diagnostic clarity.
                if low_learned and not low_applicable:
                    low_result["critical_elev_after"] = before_low
                if high_learned and not high_applicable:
                    high_result["critical_elev_after"] = before_high

                if low_applicable:
                    suggested_gates.append({
                        "entity_id": eid,
                        "facade": facade,
                        "side": "low",
                        "value": low_result["best_critical_elev"],
                        "before": before_low,
                        "sse_improvement_ratio": (
                            low_result["sse_improvement_ratio"]
                        ),
                    })
                if high_applicable:
                    suggested_gates.append({
                        "entity_id": eid,
                        "facade": facade,
                        "side": "high",
                        "value": high_result["best_critical_elev"],
                        "before": before_high,
                        "sse_improvement_ratio": (
                            high_result["sse_improvement_ratio"]
                        ),
                    })

            result[eid] = entity_result

        # #1020: ``fit_solar_obstruction`` no longer auto-writes to
        # ``_critical_elev_per_facade_per_unit``.  ``suggested_gates``
        # is the authoritative output for callers; the handler
        # surfaces it to the user, who accepts individual gates via
        # the ``apply_obstruction_gate`` service.  Kept the
        # ``crit_state_per_unit`` lookup above so ``before_*`` values
        # reflect the current state for diagnostic transparency.
        result["suggested_gates"] = suggested_gates
        return result

    def compute_implied_for_apply(
        self,
        hourly_log: list[dict],
        entity_id: str,
        regime: str,
        coordinator,
        *,
        unit_min_base: dict[str, float] | None = None,
        screen_affected_entities: frozenset[str] | None = None,
        n_windows: int = 3,
        days_back: int | None = None,
    ) -> dict:
        """Per-(entity, regime) implied coefficient + stability windows.

        Reuses ``_collect_batch_fit_samples`` to gather modulating-regime
        samples (same filter gates as batch_fit), then runs the joint
        3×3 LS over the full set for the headline ``implied_30d`` and
        chunks the samples chronologically into ``n_windows`` sub-fits
        for stability assessment.

        Returns ``{implied: dict|None, windows: [dict|None,...],
        sample_count: int, drop_counts: dict, days_back: int|None}``.
        ``implied`` is None when there are too few samples to fit;
        ``windows[i]`` is None when the chunk has too few samples to
        solve.

        ``days_back`` (default ``None`` = full log) restricts the input
        window — recommended after a retrain or major release so the
        fit doesn't absorb data from before the model state changed.
        Mirrors the same parameter on ``batch_fit_solar_coefficients``.

        Designed for the ``apply_implied_coefficient`` service — the
        caller decides per-component stability and writes the partial
        coefficient.  This helper does no clamping or write-through;
        purely an analysis pass.
        """
        from .const import APPLY_IMPLIED_MIN_QUALIFYING_HOURS

        # Time-window filter — see ``_filter_log_by_days_back`` for the
        # parse-then-compare rationale (lex-compare on ISO strings is
        # wrong under non-UTC offsets near the cutoff boundary).
        filtered_log = _filter_log_by_days_back(hourly_log, days_back)

        unit_threshold = _resolve_min_base(
            entity_id, unit_min_base, SOLAR_LEARNING_MIN_BASE
        )

        # Per-entity reconstruction (same hoist as batch_fit).
        scr_fn = getattr(coordinator, "screen_config_for_entity", None)
        if scr_fn is not None:
            screen_cfg_for_entity = scr_fn(entity_id)
        else:
            screen_cfg_for_entity = getattr(coordinator, "screen_config", None)
        entry_potentials: list[tuple[float, float, float, float]] = []
        for entry in filtered_log:
            (pot_s, pot_e, pot_w), magnitude = self._reconstruct_potential(
                entry,
                getattr(coordinator, "solar", None),
                screen_cfg_for_entity,
            )
            entry_potentials.append((pot_s, pot_e, pot_w, magnitude))

        # ``apply_implied_coefficient`` must compute the same coefficient
        # the user reads in ``diagnose_solar.implied_coefficient_30d`` —
        # otherwise the user-facing promise of the service ("commit what
        # diagnose shows") is broken.  ``match_diagnose=True`` switches
        # the filter set: drops the per-unit min_base gate, includes
        # shutdown samples, and reads base from the log-time
        # ``unit_expected_breakdown`` field.
        #
        # NOTE: ``match_diagnose=True`` explicitly skips the full MAD
        # outlier baseline to ensure "trust diagnose, commit verbatim"
        # consistency. Only the prior-free sanity check applies.
        samples, _censored_mask, drop_counts = self._collect_batch_fit_samples(
            entity_id=entity_id,
            regime=regime,
            hourly_log=filtered_log,
            entry_potentials=entry_potentials,
            coordinator=coordinator,
            unit_threshold=unit_threshold,
            screen_affected_entities=screen_affected_entities,
            match_diagnose=True,
        )

        result: dict = {
            "sample_count": len(samples),
            "drop_counts": drop_counts,
            "days_back": days_back,
            "implied": None,
            "windows": [None] * n_windows,
        }

        # 3D pass: gated on its own sample threshold.  Sparse 3D must not
        # short-circuit the 4D pass below — Track C / MPC installs often
        # have no usable ``unit_expected_breakdown`` for the 3D collector
        # while the 4D Tobit collector still finds plenty of samples.
        if len(samples) >= APPLY_IMPLIED_MIN_QUALIFYING_HOURS:
            implied = self._solve_batch_fit_normal_equations(samples)
            if implied is not None:
                result["implied"] = {k: round(v, 4) for k, v in implied.items()}

            # Stability windows: split chronologically into ``n_windows``
            # equal chunks and fit each.  ``samples`` is already in log
            # order from ``_collect_batch_fit_samples``.  Per-window
            # min-samples is half the global threshold so a sparse late
            # window still produces SOMETHING for the stability check
            # (or None if completely empty).
            per_window_min = max(8, APPLY_IMPLIED_MIN_QUALIFYING_HOURS // n_windows)
            chunk_size = max(1, len(samples) // n_windows)
            for i in range(n_windows):
                start = i * chunk_size
                end = start + chunk_size if i < n_windows - 1 else len(samples)
                chunk = samples[start:end]
                if len(chunk) < per_window_min:
                    continue
                window_fit = self._solve_batch_fit_normal_equations(chunk)
                if window_fit is not None:
                    result["windows"][i] = {
                        "coefficient": {k: round(v, 4) for k, v in window_fit.items()},
                        "qualifying_hours": len(chunk),
                    }

        # --- 4D Tobit MLE pass (#969 slim).  Strict parallel to the 3D
        # path above using ``_collect_batch_fit_samples_4d`` and
        # ``_solve_tobit`` with 4 components.  Diagnose does not surface
        # a 4D implied today, so there is no ``match_diagnose`` analogue
        # to thread — the 4D collector's default filter set is correct.
        # Per-window 4D fits reuse the same Tobit solver for stability
        # assessment (solver-symmetric semantics).
        samples_4d, censored_mask_4d, drop_counts_4d = (
            self._collect_batch_fit_samples_4d(
                entity_id=entity_id,
                regime=regime,
                hourly_log=filtered_log,
                coordinator=coordinator,
                screen_affected_entities=screen_affected_entities,
            )
        )
        result["sample_count_4d"] = len(samples_4d)
        result["drop_counts_4d"] = drop_counts_4d
        result["implied_4d"] = None
        result["windows_4d"] = [None] * n_windows

        n_unc_total_4d = sum(1 for m in censored_mask_4d if not m)
        if n_unc_total_4d >= TOBIT_MIN_UNCENSORED:
            fit_4d = self._solve_tobit(
                samples_4d,
                censored_mask_4d,
                components=("s", "e", "w", "diffuse"),
            )
            if fit_4d is not None and fit_4d.get("converged"):
                result["implied_4d"] = {
                    k: round(float(fit_4d[k]), 4)
                    for k in ("s", "e", "w", "diffuse")
                }

            # Per-window 4D Tobit fits.  Same chunking scheme as 3D
            # above.  Skip chunks below the uncensored-row threshold.
            chunk_size_4d = max(1, len(samples_4d) // n_windows)
            for i in range(n_windows):
                start = i * chunk_size_4d
                end = (
                    start + chunk_size_4d
                    if i < n_windows - 1
                    else len(samples_4d)
                )
                chunk_samples = samples_4d[start:end]
                chunk_mask = censored_mask_4d[start:end]
                chunk_n_unc = sum(1 for m in chunk_mask if not m)
                if chunk_n_unc < TOBIT_MIN_UNCENSORED:
                    continue
                window_fit_4d = self._solve_tobit(
                    chunk_samples,
                    chunk_mask,
                    components=("s", "e", "w", "diffuse"),
                )
                if window_fit_4d is not None and window_fit_4d.get(
                    "converged"
                ):
                    result["windows_4d"][i] = {
                        "coefficient": {
                            k: round(float(window_fit_4d[k]), 4)
                            for k in ("s", "e", "w", "diffuse")
                        },
                        "qualifying_hours": len(chunk_samples),
                    }
        return result

    @staticmethod
    def assess_apply_implied_stability(
        windows: list[dict | None],
        *,
        max_spread: float | None = None,
        near_zero: float | None = None,
        directions: tuple[str, ...] = ("s", "e", "w"),
    ) -> dict[str, dict]:
        """Per-direction stability assessment for ``apply_implied_coefficient``.

        For each direction ``s/e/w``, evaluates the values across the
        non-empty stability windows:

        - All values within ``near_zero`` magnitude → ``stable`` (the
          windows agree the component is effectively zero).
        - Sign-flip across non-trivial values → ``unstable``.
        - ``max(|v|) / min(|v|) > max_spread`` on non-trivial values →
          ``unstable``.
        - Otherwise → ``stable``.
        - Fewer than 2 non-empty windows → ``insufficient_windows``.

        Returns ``{direction: {stable: bool, reason: str, values: list}}``.
        Default thresholds come from ``APPLY_IMPLIED_MAX_SPREAD`` and
        ``APPLY_IMPLIED_NEAR_ZERO``; explicit args override (used by
        tests and callers that want different behaviour).
        """
        from .const import APPLY_IMPLIED_MAX_SPREAD, APPLY_IMPLIED_NEAR_ZERO

        if max_spread is None:
            max_spread = APPLY_IMPLIED_MAX_SPREAD
        if near_zero is None:
            near_zero = APPLY_IMPLIED_NEAR_ZERO

        result: dict[str, dict] = {}
        for d in directions:
            values = [
                w["coefficient"].get(d, 0.0)
                for w in windows
                if w is not None and isinstance(w.get("coefficient"), dict)
            ]
            entry: dict = {"values": values}
            if len(values) < 2:
                entry["stable"] = False
                entry["reason"] = "insufficient_windows"
                result[d] = entry
                continue

            # All near zero — windows consistently say "no signal here".
            # Returning stable=True lets the caller write 0.0 and clear
            # any stale prior on this component.
            if all(abs(v) < near_zero for v in values):
                entry["stable"] = True
                entry["reason"] = "near_zero_consensus"
                result[d] = entry
                continue

            # Sign flip — at least one value above near_zero has a
            # different sign from any other non-zero value.  We check
            # signs across ALL non-zero values (not just non-near-zero)
            # because the user-facing red flag is "the windows disagree
            # on the direction of the gain" — a +0.58 vs -0.035 split
            # is qualitatively a sign-flip even though the negative
            # value is small.  The near_zero_consensus check above
            # already catches the all-tiny case.
            non_zero = [v for v in values if v != 0]
            signs = {1 if v > 0 else -1 for v in non_zero}
            if len(signs) > 1:
                entry["stable"] = False
                entry["reason"] = "sign_flip"
                result[d] = entry
                continue

            # Spread check on non-trivial values.
            abs_vals = [abs(v) for v in values if abs(v) >= 0.001]
            if len(abs_vals) >= 2 and max(abs_vals) / min(abs_vals) > max_spread:
                entry["stable"] = False
                entry["reason"] = "spread_exceeds_threshold"
                result[d] = entry
                continue

            entry["stable"] = True
            entry["reason"] = "ok"
            result[d] = entry
        return result

    def _update_unit_tobit_live(
        self,
        entity_id: str,
        regime: str,
        sample_vector: tuple[float, float, float],
        actual_impact: float,
        expected_unit_base: float,
        tobit_sufficient_stats: dict,
        solar_coefficients_per_unit: dict,
    ) -> dict:
        """Live Tobit step on the running sufficient-statistic (#904 stage 3).

        Append the new (s, e, w, value, censored) sample to the running
        window, trim to ``TOBIT_RUNNING_WINDOW``, run one full Newton
        iteration of ``_solve_tobit`` over the current window, and
        write the resulting coefficient to ``solar_coefficients_per_unit``.

        The "running" aspect is a sliding window of recent raw samples
        rather than a true closed-form sufficient statistic — Tobit's
        Mills-ratio gradient at each Newton iterate depends on the
        current ``c·sᵢ`` value of every censored sample, which cannot
        be reduced to a fixed sum.  Sliding window keeps memory bounded
        (~6 KB per slot at peak) while preserving exact MLE on the
        window.

        Caller MUST have verified:
        - feature flag enabled
        - allow-list semantic satisfied: ``not tobit_live_entities`` (auto-mode,
          plausibility decides) OR ``entity_id in tobit_live_entities``
          (scope override).  Plausibility-gate v2 still applies in both modes.
        - entity_id NOT in mpc_managed_entities (Track C exclusion)
        - regime != None (heating or cooling)
        - solar_model_version on stored stats matches current
          (storage load handles version-drift reset; if we got here,
          the slot is current)

        Plausibility-gate v2 applies after fit succeeds: heating regime
        runs an unconstrained-OLS sub-fit on uncensored samples and
        verifies (a) ``ols_max ≥ PLAUSIBILITY_MIN_OLS_MAX_DIRECTION`` and
        (b) cosine similarity between Tobit and OLS vectors ≥
        ``PLAUSIBILITY_MIN_DIRECTION_COSINE``.  Cooling skips the gate
        entirely (no censoring → Tobit ≡ OLS).  First transition from
        blocked → applied is rate-limited.

        Returns
        -------
        Dict with keys: ``applied`` (bool — coefficient was written),
        ``in_cold_start`` (bool — n_eff < TOBIT_MIN_NEFF, NLMS-fallback
        should run instead), ``n_uncensored``, ``n_censored``, ``n_eff``,
        ``last_step_iterations``, ``last_step_failure_reason``,
        ``last_step_norm``, ``sigma``, ``samples_since_reset``.
        Caller uses ``in_cold_start`` to decide whether to run the
        NLMS-fallback path (which writes the live coefficient when
        Tobit isn't yet identifiable).
        """
        import math
        from .const import (
            BATCH_FIT_SATURATION_RATIO,
            PLAUSIBILITY_MIN_DIRECTION_COSINE,
            PLAUSIBILITY_MIN_OLS_MAX_DIRECTION,
            PLAUSIBILITY_MIN_TOBIT_MAGNITUDE,
            PLAUSIBILITY_RATE_LIMIT_FRACTION,
            SOLAR_COEFF_CAP,
            SOLAR_MODEL_VERSION,
            TOBIT_MIN_NEFF,
            TOBIT_MIN_UNCENSORED,
            TOBIT_RUNNING_WINDOW,
        )

        # Locate / initialise the running state for this (entity, regime).
        entity_state = tobit_sufficient_stats.setdefault(entity_id, {})
        slot = entity_state.setdefault(regime, {
            "samples": [],
            "samples_since_reset": 0,
            "last_step": {},
            "solar_model_version": SOLAR_MODEL_VERSION,
        })

        # Censoring decision matches batch_fit (#908) — saturated rows
        # carry value = T = ratio × base instead of observed actual_impact.
        # Cooling has no upper-saturation analog; censoring threshold
        # only applies on the heating regime.
        s_e_w = (
            float(sample_vector[0]),
            float(sample_vector[1]),
            float(sample_vector[2]),
        )
        is_censored = (
            regime == "heating"
            and actual_impact >= BATCH_FIT_SATURATION_RATIO * expected_unit_base
        )
        if is_censored:
            value = BATCH_FIT_SATURATION_RATIO * expected_unit_base
        else:
            value = actual_impact

        # Append + trim.  Sliding window keeps the most recent
        # TOBIT_RUNNING_WINDOW samples; older samples roll off
        # automatically — the running estimator naturally tracks
        # slow drift in install conditions without manual reset.
        slot["samples"].append((s_e_w[0], s_e_w[1], s_e_w[2], value, is_censored))
        if len(slot["samples"]) > TOBIT_RUNNING_WINDOW:
            slot["samples"] = slot["samples"][-TOBIT_RUNNING_WINDOW:]
        slot["samples_since_reset"] = slot.get("samples_since_reset", 0) + 1

        n_total = len(slot["samples"])
        n_unc = sum(1 for sample in slot["samples"] if not sample[4])
        n_cens = n_total - n_unc

        result = {
            "applied": False,
            "in_cold_start": True,
            "n_uncensored": n_unc,
            "n_censored": n_cens,
            "n_eff": float(n_unc),
            "last_step_iterations": 0,
            "last_step_failure_reason": None,
            "last_step_norm": 0.0,
            "sigma": None,
            "samples_since_reset": slot["samples_since_reset"],
        }

        if n_unc < TOBIT_MIN_UNCENSORED:
            slot["last_step"] = {
                "skip_reason": "insufficient_uncensored",
                "n_uncensored": n_unc,
            }
            return result

        # Pull the parallel arrays Tobit expects.  Storing as 4-tuples
        # in slot["samples"] saves memory; convert to (samples, mask)
        # at the call site.  ~200 elements max → trivial cost.
        samples_4tup = [
            (s[0], s[1], s[2], s[3]) for s in slot["samples"]
        ]
        censored_mask = [s[4] for s in slot["samples"]]

        # Coefficient warm-start: feed Tobit the current (entity, regime)
        # coefficient as initial point.  Per-hour Newton iteration thus
        # makes incremental progress instead of restarting from LS each
        # call — matches the "running" semantics and converges in 1-2
        # iterations after the first hour.
        entity_coeff_dict = solar_coefficients_per_unit.get(entity_id, {})
        regime_coeff = entity_coeff_dict.get(regime) if isinstance(entity_coeff_dict, dict) else None
        coeff_init = None
        if isinstance(regime_coeff, dict) and any(
            regime_coeff.get(k, 0.0) for k in ("s", "e", "w")
        ):
            coeff_init = regime_coeff

        fit = LearningManager._solve_tobit(
            samples_4tup,
            censored_mask,
            components=("s", "e", "w"),
            coeff_init=coeff_init,
        )

        # LS-fallback on ANY Newton convergence failure with a non-
        # None warm-start.  When the warm-start coefficient (NLMS-
        # converged or Tobit-from-prior-hour) sits outside Tobit's
        # local convergence basin (~±10 % of the true coefficient
        # value, basin width depends on data; sharp boundary verified
        # by basin-sweep probe), Newton fails to converge.  Typical
        # failure is ``line_search_failed`` (the line-search budget
        # exhausts when σ over-corrects on iter 1), but ``max_iter``
        # and ``singular_step`` are also possible on noisier real-
        # world data.  Retry on ANY non-converged outcome rather
        # than gating on a specific failure_reason — LS-fallback
        # runs the in-solver LS-warm-start path, which has its own
        # well-behaved convergence basin determined by the data
        # (not by the prior), and converges reliably (~4 iterations
        # to truth) on any input the original Newton would have
        # rejected.  The σ-init fix in ``_solve_tobit`` (LS-
        # residual seed instead of biased-c seed) does NOT widen the
        # basin — verified by pre-fix vs post-fix sweep showing
        # identical basin boundaries; it only extends Newton's
        # iteration count before line-search-failure (1 → ~5 iter),
        # giving better diagnose visibility but no convergence
        # benefit.  LS-fallback provides the actual basin escape.
        # Without it, production behaviour was: NLMS converged to
        # a saturation-biased value (e.g. 0.55 vs true 1.65), Tobit
        # took over at n_eff ≥ 40, every fit failed Newton, the
        # live coefficient was never written, and NLMS continued
        # writing the biased value indefinitely — defeating the
        # entire Stage-3 design.  The rate-limiter then handles
        # the resulting magnitude jump from the LS-Tobit fit.
        if (
            fit is not None
            and coeff_init is not None
            and not fit.get("converged", False)
        ):
            _LOGGER.debug(
                "Tobit warm-start outside basin for %s/%s; "
                "retrying with LS warm-start",
                entity_id, regime,
            )
            fit = LearningManager._solve_tobit(
                samples_4tup,
                censored_mask,
                components=("s", "e", "w"),
                coeff_init=None,
            )

        if fit is None:
            slot["last_step"] = {
                "skip_reason": "warm_start_failed",
                "n_uncensored": n_unc,
            }
            result["last_step_failure_reason"] = "warm_start_failed"
            return result

        result["sigma"] = round(fit["sigma"], 5)
        result["last_step_iterations"] = fit["iterations"]
        result["last_step_failure_reason"] = fit.get("failure_reason")
        result["n_eff"] = round(fit["n_eff"], 2)

        # Post-fit identifiability gate.
        if fit["n_eff"] < TOBIT_MIN_NEFF:
            slot["last_step"] = {
                "skip_reason": "insufficient_effective_samples",
                "n_eff": fit["n_eff"],
            }
            return result

        if not fit["converged"]:
            slot["last_step"] = {
                "skip_reason": "did_not_converge",
                "failure_reason": fit.get("failure_reason"),
            }
            return result

        # Compute step norm vs current coefficient (for diagnose / drift
        # detection — per-week-drift falsification).
        prev_s = float((coeff_init or {}).get("s", 0.0))
        prev_e = float((coeff_init or {}).get("e", 0.0))
        prev_w = float((coeff_init or {}).get("w", 0.0))
        new_s = max(0.0, min(SOLAR_COEFF_CAP, float(fit["s"])))
        new_e = max(0.0, min(SOLAR_COEFF_CAP, float(fit["e"])))
        new_w = max(0.0, min(SOLAR_COEFF_CAP, float(fit["w"])))

        # State carried from the previous Tobit step on this slot —
        # determines log severity (transition vs continuation) and
        # whether the rate-limiter on first-post-block step applies.
        prior_skip_reason = (slot.get("last_step") or {}).get("skip_reason")
        was_plausibility_blocked = prior_skip_reason in (
            "plausibility_no_uncensored_signal",
            "plausibility_direction_mismatch",
        )

        # Plausibility-gate v2 — automatic discrimination against noise
        # loads (small electric circuits, sockets, refrigeration
        # appliances) whose Tobit fit is censoring-pattern-driven only.
        # Heating regime only: cooling has no upper-saturation
        # (``is_censored`` set to False on every cooling sample at slot
        # write-time), so Tobit reduces to OLS exactly and the
        # discrimination is degenerate (``tobit_max ≡ ols_max`` →
        # plausibility-gate would only fire on the structurally-thin
        # 0.05 < c < 0.10 band where small AC installs naturally
        # converge).  Cooling fires through directly.
        #
        # Cold-start fires plausibility too: a noise-load entity at
        # first-fit is exactly the case the gate is guarding against.
        if regime == "heating":
            tobit_max = max(abs(new_s), abs(new_e), abs(new_w))
            ols_fit = None
            if tobit_max > PLAUSIBILITY_MIN_TOBIT_MAGNITUDE:
                uncensored_only = [
                    (s[0], s[1], s[2], s[3])
                    for s in slot["samples"]
                    if not s[4]
                ]
                ols_fit = LearningManager._solve_batch_fit_normal_equations(
                    uncensored_only
                )
            if ols_fit is not None:
                ols_s = float(ols_fit.get("s", 0.0))
                ols_e = float(ols_fit.get("e", 0.0))
                ols_w = float(ols_fit.get("w", 0.0))
                ols_max = max(abs(ols_s), abs(ols_e), abs(ols_w))

                # Magnitude check: any uncensored direction must clear
                # the floor.  Pure-noise loads have no uncensored slope
                # anywhere; their Tobit fit is censoring-pattern-driven.
                if ols_max < PLAUSIBILITY_MIN_OLS_MAX_DIRECTION:
                    _log_fn = (
                        _LOGGER.debug
                        if was_plausibility_blocked
                        else _LOGGER.info
                    )
                    _log_fn(
                        "Tobit plausibility-gate blocked %s/%s: "
                        "ols_max=%.3f < %.3f, tobit_max=%.3f "
                        "(no uncensored signal — NLMS retains write authority)",
                        entity_id, regime, ols_max,
                        PLAUSIBILITY_MIN_OLS_MAX_DIRECTION, tobit_max,
                    )
                    slot["last_step"] = {
                        "skip_reason": "plausibility_no_uncensored_signal",
                        "ols_max": round(ols_max, 5),
                        "tobit_max": round(tobit_max, 5),
                        "n_eff": round(fit["n_eff"], 2),
                        "iterations": fit["iterations"],
                        "sigma": round(fit["sigma"], 5),
                    }
                    return result

                # Direction-agreement check.  Tobit's projected-Newton
                # active-set can pin a direction at zero from a wrong
                # warm-start, producing a wrong-direction magnitude
                # that satisfies the OLS-max floor (because OLS still
                # identifies the real direction in the uncensored
                # subset).  Cosine ≥ 0.5 (≈ 60° tolerance) ensures
                # Tobit's vector points roughly with OLS's.
                tobit_norm = math.sqrt(
                    new_s * new_s + new_e * new_e + new_w * new_w
                )
                ols_norm = math.sqrt(
                    ols_s * ols_s + ols_e * ols_e + ols_w * ols_w
                )
                # The 1e-9 norm guards skip the cosine test when either
                # vector is structurally zero.  ``tobit_norm < 1e-9``
                # means active-set clamped every direction to zero —
                # the write is a no-op (zero coefficient overwriting
                # zero or near-zero) and the plausibility check has
                # nothing to discriminate.  ``ols_norm < 1e-9`` means
                # OLS itself is degenerate (already covered by the
                # magnitude floor above; defense-in-depth).  Both cases
                # let the (near-)zero write proceed — equivalent to
                # NLMS dead-zone behaviour, no cushion lost.
                if tobit_norm > 1e-9 and ols_norm > 1e-9:
                    cosine = (
                        new_s * ols_s + new_e * ols_e + new_w * ols_w
                    ) / (tobit_norm * ols_norm)
                    if cosine < PLAUSIBILITY_MIN_DIRECTION_COSINE:
                        _log_fn = (
                            _LOGGER.debug
                            if was_plausibility_blocked
                            else _LOGGER.info
                        )
                        _log_fn(
                            "Tobit plausibility-gate blocked %s/%s: "
                            "direction cosine %.3f < %.3f vs OLS-uncensored "
                            "(warm-start direction-pinning suspected — "
                            "NLMS retains write authority)",
                            entity_id, regime, cosine,
                            PLAUSIBILITY_MIN_DIRECTION_COSINE,
                        )
                        slot["last_step"] = {
                            "skip_reason": "plausibility_direction_mismatch",
                            "ols_max": round(ols_max, 5),
                            "tobit_max": round(tobit_max, 5),
                            "direction_cosine": round(cosine, 4),
                            "n_eff": round(fit["n_eff"], 2),
                            "iterations": fit["iterations"],
                            "sigma": round(fit["sigma"], 5),
                        }
                        return result

        # Rate-limit (general step-size limiter): cap each direction's
        # per-hour delta to ``PLAUSIBILITY_RATE_LIMIT_FRACTION × prior_max``
        # whenever ANY direction's proposed step exceeds that cap.
        # Triggered by step magnitude alone (not by plausibility-block
        # history): a Tobit fit that wants to jump 200 % from the
        # current coefficient is converged over 4-5 hours regardless of
        # whether the entity was just unblocked or simply has fast-
        # changing data.  Skipped on cold-start (``prior_max < ε``)
        # because there's no prior to cushion against and the first
        # bootstrap step would otherwise stay clamped at zero.
        prior_max = max(prev_s, prev_e, prev_w)
        proposed_step_max = max(
            abs(new_s - prev_s),
            abs(new_e - prev_e),
            abs(new_w - prev_w),
        )
        rate_limit_active = (
            prior_max > 0.05
            and proposed_step_max
            > PLAUSIBILITY_RATE_LIMIT_FRACTION * prior_max
        )
        if rate_limit_active:
            cap = PLAUSIBILITY_RATE_LIMIT_FRACTION * prior_max
            new_s = max(0.0, min(SOLAR_COEFF_CAP, prev_s + max(-cap, min(cap, new_s - prev_s))))
            new_e = max(0.0, min(SOLAR_COEFF_CAP, prev_e + max(-cap, min(cap, new_e - prev_e))))
            new_w = max(0.0, min(SOLAR_COEFF_CAP, prev_w + max(-cap, min(cap, new_w - prev_w))))
            _LOGGER.debug(
                "Tobit rate-limit: %s/%s step %.3f → cap %.3f (prior_max=%.3f)",
                entity_id, regime, proposed_step_max, cap, prior_max,
            )

        # Plausibility-clear transition log (decoupled from rate-limit).
        # Fires INFO once when the gate transitions from blocked to
        # passing — actionable signal for the maintainer / multi-install
        # falsification surface.  Annotates whether the rate-limiter is
        # also smoothing the transition.
        if was_plausibility_blocked:
            _LOGGER.info(
                "Tobit plausibility-gate clearing for %s/%s%s",
                entity_id, regime,
                " (rate-limited)" if rate_limit_active else "",
            )

        result["last_step_norm"] = round(
            max(abs(new_s - prev_s), abs(new_e - prev_e), abs(new_w - prev_w)),
            5,
        )

        # Write through the canonical helper so non-negativity (invariant
        # #4) and SOLAR_COEFF_CAP clamps are uniform with NLMS / inequality
        # / batch_fit / apply_implied write paths.
        self._update_unit_solar_coefficient(
            entity_id,
            {"s": new_s, "e": new_e, "w": new_w},
            solar_coefficients_per_unit,
            regime,
        )
        slot["last_step"] = {
            "iterations": fit["iterations"],
            "converged": True,
            "n_eff": round(fit["n_eff"], 2),
            "sigma": round(fit["sigma"], 5),
            "step_norm": result["last_step_norm"],
        }
        result["applied"] = True
        result["in_cold_start"] = False
        return result

    def compute_tobit_for_diagnose(
        self,
        hourly_log: list[dict],
        entity_id: str,
        regime: str,
        coordinator,
        *,
        unit_min_base: dict[str, float] | None = None,
        screen_affected_entities: frozenset[str] | None = None,
        days_back: int | None = 30,
    ) -> dict:
        """Per-(entity, regime) Tobit MLE for ``diagnose_solar`` (#904 stage 0+1).

        Surfaces a censoring-aware coefficient estimate alongside
        ``implied_coefficient_30d`` (unconstrained LS dropping saturated
        rows) and ``implied_coefficient_inequality`` (lower-bound
        replay).  Pure analysis — no production wiring, no writes to
        ``solar_coefficients_per_unit``.  Promotion to live wiring is
        gated on stage-1 evidence over the validation window.

        Mirrors ``compute_implied_for_apply`` shape but routes through
        ``_collect_batch_fit_samples(..., for_tobit=True)``: shutdown
        rows excluded (modulating-regime fit; CHOICE 3), saturated rows
        kept with ``value = T = BATCH_FIT_SATURATION_RATIO × base`` and
        flagged in ``censored_mask``.

        Returns
        -------
        ``{"coefficient", "sigma", "iterations", "converged",
        "log_likelihood", "n_uncensored", "n_censored", "n_eff",
        "censored_fraction", "drop_counts", "days_back",
        "skip_reason"}`` — ``coefficient`` is ``None`` for the
        ``insufficient_uncensored``, ``warm_start_failed``, and
        ``insufficient_effective_samples`` skip paths.  For the
        ``did_not_converge`` path the coefficient is **populated
        with the last Newton iterate** for inspection (not None) —
        callers must check ``skip_reason`` rather than
        ``coefficient is None`` to detect the non-convergent case
        and decide whether to trust the iterate.  ``log_likelihood``
        is set whenever the solver ran (i.e. all paths except
        ``insufficient_uncensored`` and ``warm_start_failed``).
        """
        filtered_log = _filter_log_by_days_back(hourly_log, days_back)

        unit_threshold = _resolve_min_base(
            entity_id, unit_min_base, SOLAR_LEARNING_MIN_BASE
        )

        scr_fn = getattr(coordinator, "screen_config_for_entity", None)
        if scr_fn is not None:
            screen_cfg_for_entity = scr_fn(entity_id)
        else:
            screen_cfg_for_entity = getattr(coordinator, "screen_config", None)

        # Current live coefficients feed the #919 MAD outlier pre-filter
        # inside the sample collector — same convention as the
        # batch_fit_solar caller (which receives the dict as a function
        # parameter; this entry point receives the coordinator instead,
        # so resolve via the model view with private-attribute fallback).
        # Falls back to None — MAD baseline skipped, prior-free sanity
        # check still applies — only when neither surface exposes a dict
        # (ad-hoc test fixtures).
        try:
            solar_coefficients_per_unit = coordinator.model.solar_coefficients_per_unit
        except AttributeError:
            solar_coefficients_per_unit = None
        if not isinstance(solar_coefficients_per_unit, dict):
            solar_coefficients_per_unit = getattr(
                coordinator, "_solar_coefficients_per_unit", None
            )
        if not isinstance(solar_coefficients_per_unit, dict):
            solar_coefficients_per_unit = None

        entry_potentials: list[tuple[float, float, float, float]] = []
        for entry in filtered_log:
            (pot_s, pot_e, pot_w), magnitude = self._reconstruct_potential(
                entry,
                getattr(coordinator, "solar", None),
                screen_cfg_for_entity,
            )
            entry_potentials.append((pot_s, pot_e, pot_w, magnitude))

        samples, censored_mask, drop_counts = self._collect_batch_fit_samples(
            entity_id=entity_id,
            regime=regime,
            hourly_log=filtered_log,
            entry_potentials=entry_potentials,
            coordinator=coordinator,
            unit_threshold=unit_threshold,
            screen_affected_entities=screen_affected_entities,
            for_tobit=True,
            solar_coefficients_per_unit=solar_coefficients_per_unit,
        )

        n_total = len(samples)
        n_unc = sum(1 for m in censored_mask if not m)
        n_cens = n_total - n_unc
        cens_frac = (n_cens / n_total) if n_total > 0 else 0.0

        result: dict = {
            "coefficient": None,
            "sigma": None,
            "iterations": 0,
            "converged": False,
            "n_uncensored": n_unc,
            "n_censored": n_cens,
            "n_eff": float(n_unc),
            "censored_fraction": round(cens_frac, 3),
            "drop_counts": drop_counts,
            "days_back": days_back,
            "skip_reason": None,
        }

        # CHOICE 4 gate: |U| ≥ 20 AND n_eff ≥ 40 (preliminary; pre-fit
        # we only check the |U| floor — the n_eff floor is post-fit).
        if n_unc < TOBIT_MIN_UNCENSORED:
            result["skip_reason"] = "insufficient_uncensored"
            return result

        fit = LearningManager._solve_tobit(
            samples, censored_mask, components=("s", "e", "w")
        )
        if fit is None:
            result["skip_reason"] = "warm_start_failed"
            return result

        result["sigma"] = round(fit["sigma"], 5)
        result["iterations"] = fit["iterations"]
        result["converged"] = bool(fit["converged"])
        result["failure_reason"] = fit.get("failure_reason")
        result["n_eff"] = round(fit["n_eff"], 2)
        result["log_likelihood"] = round(fit["log_likelihood"], 4)

        if fit["n_eff"] < TOBIT_MIN_NEFF:
            result["skip_reason"] = "insufficient_effective_samples"
            return result

        if not fit["converged"]:
            result["skip_reason"] = "did_not_converge"
            # Still return the (last) coefficient — the caller can
            # surface it for manual inspection but the skip_reason
            # signals it's not trustworthy.

        coeff = {
            "s": max(0.0, min(SOLAR_COEFF_CAP, float(fit["s"]))),
            "e": max(0.0, min(SOLAR_COEFF_CAP, float(fit["e"]))),
            "w": max(0.0, min(SOLAR_COEFF_CAP, float(fit["w"]))),
        }
        result["coefficient"] = {k: round(v, 4) for k, v in coeff.items()}
        return result
