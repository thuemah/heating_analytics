"""Learning Manager Service."""
from __future__ import annotations

import logging
from typing import Callable

from .const import (
    BATCH_FIT_DAMPING,
    BATCH_FIT_MIN_SAMPLES,
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
    SNR_WEIGHT_FLOOR,
    SNR_WEIGHT_K,
    SOLAR_BATTERY_DECAY,
    SOLAR_COEFF_CAP,
    SOLAR_DEAD_ZONE_THRESHOLD,
    SOLAR_LEARNING_MIN_BASE,
    SOLAR_SHUTDOWN_MIN_BASE,
    SOLAR_SHUTDOWN_MIN_MAGNITUDE,
)
from .observation import HourlyObservation, ModelState, LearningConfig
from .solar import SolarCalculator

_LOGGER = logging.getLogger(__name__)


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

                if is_cold_cooling_regime:
                    _LOGGER.debug(
                        "Global base: skipping cold-cooling-regime hour "
                        "(T=%.1f, BP=%.1f, %d cooling unit(s))",
                        avg_temp, balance_point, len(active_modes),
                    )
                elif solar_enabled:
                    base_target = total_energy_kwh
                    snr_w = compute_snr_weight(
                        solar_factor,
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

                if not is_cold_cooling_regime:
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
                        new_base_prediction = base_expected_kwh + base_effective_rate * (base_target - base_expected_kwh)
                        if temp_key not in correlation_data:
                            correlation_data[temp_key] = {}
                        correlation_data[temp_key][wind_bucket] = round(new_base_prediction, 5)

                    model_base_after = new_base_prediction

        # Update Per-Unit Models (Both Normal and Aux modes)
        # We run this even in cooldown (for non-affected units)
        if should_run_per_unit:
            self._process_per_unit_learning(
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
                unit_min_base=unit_min_base,
                solar_factor=solar_factor,
                battery_filtered_potential=battery_filtered_potential,
            )

        return {
            "model_updated": model_updated,
            "model_base_before": model_base_before,
            "model_base_after": model_base_after,
            "aux_model_updated": aux_model_updated,
            "aux_model_before": aux_model_before,
            "aux_model_after": aux_model_after,
            "learning_status": learning_status,
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
        solar_factor: float = 0.0,
        battery_filtered_potential: tuple[float, float, float] | None = None,
        unit_min_base: dict[str, float] | None = None,
    ):
        """Process learning for individual units."""
        for entity_id in energy_sensors:
            unit_mode = unit_modes.get(entity_id, MODE_HEATING)
            if unit_mode in (MODE_OFF, MODE_GUEST_HEATING, MODE_GUEST_COOLING):
                # Skip learning for non-tracked/temporary modes
                continue

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
                solar_enabled
                and vector_magnitude > 0.1
                and not is_aux_active
                and not is_solar_shutdown
                and expected_unit_base >= nlms_threshold
            ):
                self._learn_unit_solar_coefficient(
                    entity_id, temp_key,
                    expected_unit_base, actual_unit, (potential_s, potential_e, potential_w),
                    learning_rate, solar_coefficients_per_unit, learning_buffer_solar_per_unit,
                    avg_temp, balance_point,
                    unit_mode
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
                solar_enabled
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
                self._update_unit_solar_inequality(
                    entity_id=entity_id,
                    expected_unit_base=expected_unit_base,
                    battery_filtered_potential=battery_filtered_potential,
                    solar_coefficients_per_unit=solar_coefficients_per_unit,
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

                snr_w = compute_snr_weight(
                    solar_factor,
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

    def _learn_unit_solar_coefficient(
        self,
        entity_id: str,
        temp_key: str,
        expected_unit_base: float,
        actual_unit: float,
        avg_solar_vector: tuple[float, float, float],
        learning_rate: float,
        solar_coefficients_per_unit: dict,
        learning_buffer_solar_per_unit: dict,
        avg_temp: float,
        balance_point: float,
        unit_mode: str,
    ):
        """Update 3D solar coefficient (S, E, W) for one (entity, mode) slot.

        Mode-stratified per #868: heating-mode hours update
        ``solar_coefficients_per_unit[entity]["heating"]``; cooling-mode
        hours update ``["cooling"]``.  Each regime absorbs its own
        ``E[1/COP]`` and converges to a physically distinct value.
        OFF/DHW/unknown modes return early — no solar learning signal.
        """
        regime = _solar_coeff_regime(unit_mode)
        if regime is None:
            return

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

        solar_s, solar_e, solar_w = avg_solar_vector
        vector_magnitude = (solar_s**2 + solar_e**2 + solar_w**2) ** 0.5

        if vector_magnitude <= 0.01:
            return

        # Get current (regime-specific) coefficient — None means cold-start.
        # The entity dict carries both regimes; we route to the active one.
        entity_coeffs = solar_coefficients_per_unit.get(entity_id)
        current_coeff = None
        if isinstance(entity_coeffs, dict):
            regime_coeff = entity_coeffs.get(regime)
            if isinstance(regime_coeff, dict) and any(
                regime_coeff.get(k) for k in ("s", "e", "w")
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
        dead_key = (entity_id, regime)
        if actual_impact == 0.0 and raw_impact < 0.0 and current_coeff is not None:
            count = self._dead_zone_counts.get(dead_key, 0) + 1
            self._dead_zone_counts[dead_key] = count
            if count >= SOLAR_DEAD_ZONE_THRESHOLD:
                # Reset only this regime — preserve the other one.
                if isinstance(entity_coeffs, dict):
                    entity_coeffs[regime] = {"s": 0.0, "e": 0.0, "w": 0.0}
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

        # --- Buffered Learning Logic (Cold Start) ---
        if current_coeff is None:
            buf_entry = learning_buffer_solar_per_unit.setdefault(entity_id, {})
            if not isinstance(buf_entry, dict):
                # Defensive: legacy in-memory shape — rewrap to per-regime.
                buf_entry = {"heating": [], "cooling": []}
                learning_buffer_solar_per_unit[entity_id] = buf_entry
            buffer_list = buf_entry.setdefault(regime, [])
            # Store tuple of (s, e, w, impact)
            buffer_list.append((solar_s, solar_e, solar_w, actual_impact))

            # Need enough data for 3D least squares
            if len(buffer_list) >= LEARNING_BUFFER_THRESHOLD:
                # Guard: if all impact values are zero, the base model is too
                # low to produce signal (dead zone during cold-start).  Discard
                # the buffer and keep collecting — the default coefficient
                # (0.35) provides solar normalization for the base model via
                # calculate_unit_coefficient, which will gradually raise the
                # base model until actual_impact becomes positive.
                if all(item[3] == 0.0 for item in buffer_list):
                    buffer_list.clear()
                    _LOGGER.debug(
                        "Solar cold-start buffer discarded for %s: all %d "
                        "samples had zero impact (base model recovering)",
                        entity_id, LEARNING_BUFFER_THRESHOLD,
                    )
                    return

                # Solve 3x3 normal equations (Gram matrix A, moment vector b)
                # A = [[ss, se, sw], [se, ee, ew], [sw, ew, ww]]
                # b = [sI, eI, wI]
                # Note: sum_ew is always ~0 because E and W have disjoint support.
                sum_s2 = sum(item[0]**2 for item in buffer_list)
                sum_e2 = sum(item[1]**2 for item in buffer_list)
                sum_w2 = sum(item[2]**2 for item in buffer_list)
                sum_se = sum(item[0] * item[1] for item in buffer_list)
                sum_sw = sum(item[0] * item[2] for item in buffer_list)
                sum_ew = sum(item[1] * item[2] for item in buffer_list)
                sum_s_I = sum(item[0] * item[3] for item in buffer_list)
                sum_e_I = sum(item[1] * item[3] for item in buffer_list)
                sum_w_I = sum(item[2] * item[3] for item in buffer_list)

                # 3x3 determinant via cofactor expansion along first row
                determinant = (
                    sum_s2 * (sum_e2 * sum_w2 - sum_ew**2)
                    - sum_se * (sum_se * sum_w2 - sum_ew * sum_sw)
                    + sum_sw * (sum_se * sum_ew - sum_e2 * sum_sw)
                )

                # Full 3D solution when sun angles are diverse enough
                if abs(determinant) > 1e-6:
                    # Cramer's rule: replace each column with b and compute determinant
                    det_s = (
                        sum_s_I * (sum_e2 * sum_w2 - sum_ew**2)
                        - sum_se * (sum_e_I * sum_w2 - sum_ew * sum_w_I)
                        + sum_sw * (sum_e_I * sum_ew - sum_e2 * sum_w_I)
                    )
                    det_e = (
                        sum_s2 * (sum_e_I * sum_w2 - sum_ew * sum_w_I)
                        - sum_s_I * (sum_se * sum_w2 - sum_ew * sum_sw)
                        + sum_sw * (sum_se * sum_w_I - sum_e_I * sum_sw)
                    )
                    det_w = (
                        sum_s2 * (sum_e2 * sum_w_I - sum_ew * sum_e_I)
                        - sum_se * (sum_se * sum_w_I - sum_e_I * sum_sw)
                        + sum_s_I * (sum_se * sum_ew - sum_e2 * sum_sw)
                    )

                    new_coeff_s = max(-SOLAR_COEFF_CAP, min(SOLAR_COEFF_CAP, det_s / determinant))
                    new_coeff_e = max(-SOLAR_COEFF_CAP, min(SOLAR_COEFF_CAP, det_e / determinant))
                    new_coeff_w = max(-SOLAR_COEFF_CAP, min(SOLAR_COEFF_CAP, det_w / determinant))

                    new_coeff = {
                        "s": new_coeff_s * COLD_START_SOLAR_DAMPING,
                        "e": new_coeff_e * COLD_START_SOLAR_DAMPING,
                        "w": new_coeff_w * COLD_START_SOLAR_DAMPING,
                    }
                    _LOGGER.info(f"Buffered Unit Solar Learning [Jump Start]: {entity_id} [{regime}] -> {new_coeff} (3D Least Squares, {len(buffer_list)} samples, damping={COLD_START_SOLAR_DAMPING})")
                else:
                    # Collinear fallback: sun observed from a narrow angle range (e.g. winter).
                    # Project all observations onto the dominant direction and solve 1D LS,
                    # then decompose back. NLMS updates will refine all components over time.
                    dir_s = sum(item[0] for item in buffer_list)
                    dir_e = sum(item[1] for item in buffer_list)
                    dir_w = sum(item[2] for item in buffer_list)
                    dir_norm = (dir_s**2 + dir_e**2 + dir_w**2) ** 0.5

                    if dir_norm < 1e-6:
                        _LOGGER.debug(f"Buffered Unit Solar [Collecting]: {entity_id} -> zero-magnitude vectors, skipping.")
                        return

                    d_s = dir_s / dir_norm
                    d_e = dir_e / dir_norm
                    d_w = dir_w / dir_norm

                    sum_proj_I = sum((item[0] * d_s + item[1] * d_e + item[2] * d_w) * item[3] for item in buffer_list)
                    sum_proj2 = sum((item[0] * d_s + item[1] * d_e + item[2] * d_w) ** 2 for item in buffer_list)

                    if sum_proj2 < 1e-6:
                        _LOGGER.debug(f"Buffered Unit Solar [Collecting]: {entity_id} -> degenerate projection, skipping.")
                        return

                    c_scalar = max(-SOLAR_COEFF_CAP, min(SOLAR_COEFF_CAP, sum_proj_I / sum_proj2))
                    new_coeff = {
                        "s": c_scalar * d_s * COLD_START_SOLAR_DAMPING,
                        "e": c_scalar * d_e * COLD_START_SOLAR_DAMPING,
                        "w": c_scalar * d_w * COLD_START_SOLAR_DAMPING,
                    }
                    _LOGGER.info(f"Buffered Unit Solar Learning [Jump Start 1D]: {entity_id} [{regime}] -> {new_coeff} (collinear fallback, dir=({d_s:.2f},{d_e:.2f},{d_w:.2f}), {len(buffer_list)} samples, damping={COLD_START_SOLAR_DAMPING})")

                self._update_unit_solar_coefficient(
                    entity_id, new_coeff, solar_coefficients_per_unit, regime
                )
                buffer_list.clear()
            else:
                 _LOGGER.debug(f"Buffered Unit Solar [Collecting]: {entity_id} -> Sample {len(buffer_list)}/{LEARNING_BUFFER_THRESHOLD} ({actual_impact:.3f} kW)")

        else:
            # Post-Jump Start: Normalized LMS (NLMS) (#809 A3)
            # Standard LMS has gain-dependent instability: high-solar units
            # (e.g. Toshiba with large south windows) get updates proportional
            # to solar_s², causing oscillation. NLMS normalizes by input power,
            # making convergence rate independent of solar magnitude.
            # Epsilon acts as L2 regularization — when solar power is low,
            # the effective step shrinks, preventing noise-chasing on
            # poorly-constrained components.
            coeff_s = current_coeff.get("s", 0.0)
            coeff_e = current_coeff.get("e", 0.0)
            coeff_w = current_coeff.get("w", 0.0)

            predicted_impact = coeff_s * solar_s + coeff_e * solar_e + coeff_w * solar_w
            error = actual_impact - predicted_impact

            # NLMS: normalize step by input power + regularization
            input_power = solar_s ** 2 + solar_e ** 2 + solar_w ** 2
            step = NLMS_STEP_SIZE * error / (input_power + NLMS_REGULARIZATION)

            new_coeff_s = coeff_s + step * solar_s
            new_coeff_e = coeff_e + step * solar_e
            new_coeff_w = coeff_w + step * solar_w

            # Clamp for safety
            new_coeff_s = max(-SOLAR_COEFF_CAP, min(SOLAR_COEFF_CAP, new_coeff_s))
            new_coeff_e = max(-SOLAR_COEFF_CAP, min(SOLAR_COEFF_CAP, new_coeff_e))
            new_coeff_w = max(-SOLAR_COEFF_CAP, min(SOLAR_COEFF_CAP, new_coeff_w))

            new_coeff = {"s": new_coeff_s, "e": new_coeff_e, "w": new_coeff_w}

            _LOGGER.debug(f"Per-Unit Solar Learning [NLMS]: {entity_id} [{regime}] -> {new_coeff} (was {current_coeff})")
            self._update_unit_solar_coefficient(
                entity_id, new_coeff, solar_coefficients_per_unit, regime
            )

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
    ) -> None:
        """Update the solar coefficient for one (entity, regime) slot.

        ``regime`` is ``"heating"`` or ``"cooling"`` (#868).  The other
        regime is preserved unchanged on every write.  Initial entry for
        an entity creates both regimes as zero-vectors so subsequent
        reads on the unwritten regime return a stable shape.

        All three components (south, east, west) are clamped to >= 0
        (invariant #4): each represents solar gain through windows
        facing one cardinal direction, and a window can only receive
        gain.  Heating and cooling regimes share this clamp but
        converge to different absolute values (each absorbs its own
        ``E[1/COP]``).
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
        entry[regime] = {
            "s": round(max(0.0, value.get("s", 0.0)), 5),
            "e": round(max(0.0, value.get("e", 0.0)), 5),
            "w": round(max(0.0, value.get("w", 0.0)), 5),
        }

    def _update_unit_solar_inequality(
        self,
        entity_id: str,
        expected_unit_base: float,
        battery_filtered_potential: tuple[float, float, float],
        solar_coefficients_per_unit: dict,
        *,
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

        Parameters
        ----------
        entity_id:
            Unit being updated.
        expected_unit_base:
            Current per-unit bucket value (kWh).  Gates the update via
            ``SOLAR_SHUTDOWN_MIN_BASE`` at the call site.
        battery_filtered_potential:
            3-tuple ``(pot_s, pot_e, pot_w)`` from the coordinator's
            ``_potential_battery_s/e/w`` (live) or local replay state.
        solar_coefficients_per_unit:
            Current coefficients; mutated in place.
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
        ``"zero_magnitude"`` (battery-filtered potential too small).
        """
        pot_s, pot_e, pot_w = battery_filtered_potential
        mag_total = pot_s + pot_e + pot_w  # all non-negative by construction
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
        coeff_s = float(heating_coeff.get("s", 0.0))
        coeff_e = float(heating_coeff.get("e", 0.0))
        coeff_w = float(heating_coeff.get("w", 0.0))

        predicted_impact = coeff_s * pot_s + coeff_e * pot_e + coeff_w * pot_w
        constraint_target = margin * expected_unit_base

        if predicted_impact >= constraint_target:
            return "non_binding"

        # Constraint violated — distribute the deficit proportional to the
        # non-negative potential magnitude in each direction.  A direction
        # with zero potential this hour gets no update (the battery carries
        # history across hours, so a direction with recent activity still
        # gets updated when its own pot component is above zero).
        deficit = constraint_target - predicted_impact
        new_s = coeff_s + step_size * deficit * (pot_s / mag_total)
        new_e = coeff_e + step_size * deficit * (pot_e / mag_total)
        new_w = coeff_w + step_size * deficit * (pot_w / mag_total)

        # Apply non-negative + CAP clamps consistent with NLMS path.
        new_s = max(0.0, min(SOLAR_COEFF_CAP, new_s))
        new_e = max(0.0, min(SOLAR_COEFF_CAP, new_e))
        new_w = max(0.0, min(SOLAR_COEFF_CAP, new_w))

        self._update_unit_solar_coefficient(
            entity_id,
            {"s": new_s, "e": new_e, "w": new_w},
            solar_coefficients_per_unit,
            "heating",
        )
        return "updated"

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
    ) -> str:
        """Process a single historical data point to train global models.

        Target is raw ``actual_kwh`` and the EMA step is scaled by
        ``snr_weight``.  Dark hours retain full rate, sunny hours
        contribute proportionally, shutdown hours zero.  Caller is
        responsible for computing the weight via
        :func:`compute_snr_weight` from the hour's ``solar_factor`` and
        ``solar_dominant_entities``.

        ``actual_kwh`` for the aux path is taken at face value — aux
        learning does not share the COP-ceiling / shutdown-contamination
        failure modes that motivated the SNR formulation.  Aux on sunny
        hours is rare (aux typically fires in cold dark conditions) so
        the residual bias is small and bounded.

        ``solar_normalization_delta`` is still consumed by the aux path
        below — aux learning on sunny hours needs the dark-equivalent
        target so solar-induced reductions are not mis-attributed as aux
        reductions.  The base path ignores it entirely.
        """
        normalized_actual = actual_kwh  # raw — SNR weight carries signal-quality
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
                new_pred = current_pred + effective_rate * (normalized_actual - current_pred)
            else:
                # Cold start: first non-zero sample seeds the bucket.
                # Under SNR the first sample may be weighted down; we still
                # seed from the raw value here because EMA needs a
                # starting point.  Subsequent samples with high weight
                # will dominate quickly via higher effective_rate.
                new_pred = normalized_actual
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
            h_wind_bucket = log_entry.get("wind_bucket")
            if h_temp_key is None or h_wind_bucket is None:
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
            solar_delta = log_entry.get("solar_normalization_delta", 0.0)
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
    ) -> None:
        """Replay per-unit correlation models from hourly log entries.

        Each DirectMeter sensor's actual kWh is written to its per-unit
        correlation table via buffer → jump-start → EMA.  Needed so that
        ``isolate_sensor`` subtraction works after ``retrain_from_history(reset_first=True)``.
        WeightedSmear sensors are skipped.

        Moved from coordinator.py (#784) — pure model-writing logic.
        """
        from .observation import DirectMeter

        direct_sensors = [
            s for s in strategies.values()
            if isinstance(s, DirectMeter)
        ]
        if not direct_sensors:
            return

        correlation_per_unit = model.correlation_data_per_unit
        buffer_per_unit = model.learning_buffer_per_unit

        for log_entry in day_entries:
            h_temp_key = log_entry.get("temp_key")
            h_wind_bucket = log_entry.get("wind_bucket")
            if h_temp_key is None or h_wind_bucket is None:
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
            wind_bucket = entry.get("wind_bucket", "normal")
            if temp_key is None:
                diag["entry_skipped_missing_temp_key"] += 1
                continue
            unit_modes = entry.get("unit_modes", {}) or {}
            unit_breakdown = entry.get("unit_breakdown", {}) or {}
            shutdown_entities = set(entry.get("solar_dominant_entities", []) or [])
            avg_temp = entry.get("temp", 0.0) or 0.0

            for entity_id in energy_sensors:
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
    ) -> dict:
        """Periodic batch least-squares fit per (entity, mode) regime (#884).

        Solves the joint 3×3 normal equations over the modulating-regime
        hourly log to extract a coefficient that NLMS and inequality
        cannot — specifically the mild-weather catch-22 where expected
        base demand is near zero (e.g. west-facing rooms whose solar
        peak coincides with the daily temperature maximum).  A
        least-squares fit over N samples extracts the underlying
        coefficient even when each individual sample carries tiny SNR
        and net-zeroes in NLMS.

        Per (entity, regime): heating-mode hours fit the heating regime
        only; cooling-mode hours fit the cooling regime only.  Each
        regime absorbs its own ``E[1/COP_mode]`` (#868 invariant).
        Inequality-flagged shutdown hours and saturated samples are
        dropped — they violate the linear-regression assumptions of
        the fit.

        ``days_back`` (default ``None`` = full log) restricts the input
        window.  Without this, a fit on a fresh upgrade would absorb
        pre-upgrade data and pull coefficients toward old model
        behaviour (e.g. pre-1.3.3 transmittance constants); a 14- or
        30-day window after a major release fits against representative
        post-upgrade data.

        ``dry_run`` (default ``False``) runs every gate and the LS
        solve but does not write coefficients back to
        ``solar_coefficients_per_unit``.  The diagnostic dict reports
        ``coefficient_after`` so the user can preview the result; a
        subsequent (non-dry-run) call applies the same fit.  Useful
        for sanity-checking before letting the fit drift live state.

        Sample filter (per (entity, regime)):
        - ``unit_modes[entity_id]`` resolves to ``regime`` (heating
          regime takes HEATING + GUEST_HEATING; cooling takes
          COOLING + GUEST_COOLING).  OFF / DHW are skipped.
        - Entity not in ``solar_dominant_entities`` (shutdown
          samples are saturated — would inflate the fit).
        - Auxiliary not active during the hour.
        - Reconstructed potential vector magnitude > 0.01.
        - ``expected_unit_base ≥ unit_min_base[entity]`` (or
          ``SOLAR_LEARNING_MIN_BASE`` fallback) — same gate as live
          NLMS.
        - ``actual_impact > 0`` (clamped negative impacts).
        - ``actual_impact < BATCH_FIT_SATURATION_RATIO × base`` —
          drop saturation-clamped samples that have no information
          about the coefficient itself.

        Algorithm:
        - 3×3 normal equations + Cramer's rule (same arithmetic as
          cold-start solver in ``_learn_unit_solar_coefficient``).
        - Collinear fallback (1D projection along dominant direction)
          when the determinant is degenerate.
        - Clamp components to ``[0, SOLAR_COEFF_CAP]`` (#868 invariant
          #4).
        - Damp blend: ``new = α × batch + (1 − α) × current`` with
          ``α = BATCH_FIT_DAMPING``.  When the regime has not learned
          before (current is zero), the batch result is written
          directly without damping — there is no prior to blend.

        Returns
        -------
        Diagnostics dict ``{entity_id: {regime: {...}}}`` reporting
        per slot: ``sample_count``, ``residual_rmse_kwh``,
        ``coefficient_before``, ``coefficient_after``, ``damping``,
        and ``skip_reason`` when the fit was not applied.
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

        for entity_id in target_entities:
            if entity_id not in energy_sensors:
                results[entity_id] = {
                    "skip_reason": "unknown_entity",
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
                samples, drop_counts = self._collect_batch_fit_samples(
                    entity_id=entity_id,
                    regime=regime,
                    hourly_log=filtered_log,
                    entry_potentials=entry_potentials,
                    coordinator=coordinator,
                    unit_threshold=unit_threshold,
                    screen_affected_entities=screen_affected_entities,
                )
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

                if len(samples) < BATCH_FIT_MIN_SAMPLES:
                    regime_diag["skip_reason"] = "insufficient_samples"
                    regime_diag["coefficient_after"] = regime_diag[
                        "coefficient_before"
                    ]
                    entity_results[regime] = regime_diag
                    continue

                fit = self._solve_batch_fit_normal_equations(samples)
                if fit is None:
                    regime_diag["skip_reason"] = "degenerate_fit"
                    regime_diag["coefficient_after"] = regime_diag[
                        "coefficient_before"
                    ]
                    entity_results[regime] = regime_diag
                    continue

                # Clamp to [0, CAP] — invariant #4.  The solver itself
                # may produce negative components from noise; the clamp
                # is the physical truth.
                clamped = {
                    k: max(0.0, min(SOLAR_COEFF_CAP, v))
                    for k, v in fit.items()
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
                # batch fit — the question this answers is "how well does
                # the joint LS fit explain the samples", not "how well
                # does the damped result explain them".  Damping is a
                # caution against single-batch overshoot; it doesn't
                # affect the fit-quality signal.
                regime_diag["residual_rmse_kwh"] = self._batch_fit_residual_rmse(
                    samples, clamped
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
                entity_results[regime] = regime_diag

            results[entity_id] = entity_results

        return results

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
    ) -> tuple[list[tuple[float, float, float, float]], dict[str, int]]:
        """Filter + assemble samples for one (entity, regime) batch fit.

        Returns ``(samples, drop_counts)``.  Each sample is a 4-tuple
        ``(pot_s, pot_e, pot_w, actual_impact)`` ready for the joint
        normal equations.  ``drop_counts`` reports per-reason rejection
        counters useful for diagnostics — a fit that ends with too few
        samples can be inspected to see which gate did the rejecting.

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
        target_mode = MODE_HEATING if regime == "heating" else MODE_COOLING
        correlation_per_unit = coordinator.model.correlation_data_per_unit

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

            # Shutdown filter: batch_fit excludes (the inequality learner
            # owns those samples).  match_diagnose=True INCLUDES shutdown
            # samples to reproduce ``diagnose_solar.implied_30d`` exactly
            # — that headline accumulator includes them; only the parallel
            # ``no_shutdown`` accumulator separates them out.
            if not match_diagnose:
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

            # Base lookup: batch_fit reads CURRENT correlation_data; the
            # diagnose-match path reads the LOG-TIME ``unit_expected_breakdown``
            # so the LS sees the same base value diagnose's accumulator did.
            # Diagnose-match falls back to the current correlation_data if
            # the log entry lacks the field (legacy logs without
            # ``unit_expected_breakdown``).
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
                if expected_base < unit_threshold:
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

            # Saturation gate is heating-physics: when sun fully covers
            # heating demand, ``actual ≈ 0`` and ``actual_impact ≈ base``
            # → coefficient is clipped, the sample carries no slope
            # information.  Cooling has no equivalent — ``actual_impact =
            # actual - base`` has no upper bound from physics, so dropping
            # at ``≥ 0.95×base`` would discard the strongest cooling-load
            # samples for no reason.
            if (
                regime == "heating"
                and actual_impact >= BATCH_FIT_SATURATION_RATIO * expected_base
            ):
                drop_counts["saturated"] += 1
                continue

            samples.append((pot_s, pot_e, pot_w, actual_impact))

        return samples, drop_counts

    @staticmethod
    def _solve_batch_fit_normal_equations(
        samples: list[tuple[float, float, float, float]],
    ) -> dict[str, float] | None:
        """Joint 3×3 LS via Cramer's rule, with collinear 1D fallback.

        Same arithmetic as the cold-start solver in
        ``_learn_unit_solar_coefficient`` lines 985-1085 — extracted
        here without the cold-start damping (the caller damps).
        Returns a coefficient dict, or ``None`` if the fit cannot be
        produced (degenerate Gram matrix and zero-magnitude direction
        sum).
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

        if abs(determinant) > 1e-6:
            det_s = (
                sum_s_I * (sum_e2 * sum_w2 - sum_ew ** 2)
                - sum_se * (sum_e_I * sum_w2 - sum_ew * sum_w_I)
                + sum_sw * (sum_e_I * sum_ew - sum_e2 * sum_w_I)
            )
            det_e = (
                sum_s2 * (sum_e_I * sum_w2 - sum_ew * sum_w_I)
                - sum_s_I * (sum_se * sum_w2 - sum_ew * sum_sw)
                + sum_sw * (sum_se * sum_w_I - sum_e_I * sum_sw)
            )
            det_w = (
                sum_s2 * (sum_e2 * sum_w_I - sum_ew * sum_e_I)
                - sum_se * (sum_se * sum_w_I - sum_e_I * sum_sw)
                + sum_s_I * (sum_se * sum_ew - sum_e2 * sum_sw)
            )
            return {
                "s": det_s / determinant,
                "e": det_e / determinant,
                "w": det_w / determinant,
            }

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
        samples, drop_counts = self._collect_batch_fit_samples(
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

        if len(samples) < APPLY_IMPLIED_MIN_QUALIFYING_HOURS:
            return result

        # 30-day implied: full LS over all samples.
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
        return result

    @staticmethod
    def assess_apply_implied_stability(
        windows: list[dict | None],
        *,
        max_spread: float | None = None,
        near_zero: float | None = None,
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
        for d in ("s", "e", "w"):
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
