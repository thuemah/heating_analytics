"""Constants for the Heating Analytics integration."""

DOMAIN = "heating_analytics"

# Default Configuration Values
DEFAULT_NAME = "Heating Analytics"
DEFAULT_WIND_GUST_FACTOR = 0.6
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_DAILY_LEARNING_RATE = 0.005
DEFAULT_BALANCE_POINT = 17.0
DEFAULT_WIND_THRESHOLD = 8
DEFAULT_EXTREME_WIND_THRESHOLD = 10.8
DEFAULT_CSV_AUTO_LOGGING = False
DEFAULT_CSV_HOURLY_PATH = "/config/heating_analytics_hourly_log.csv"
DEFAULT_CSV_DAILY_PATH = "/config/heating_analytics_daily_log.csv"
DEFAULT_WIND_UNIT = "m/s"
DEFAULT_MAX_ENERGY_DELTA = 3.0

# Explanation Constants
DEFAULT_TEMP_EXTREME = 5.0      # °C delta
DEFAULT_TEMP_SIGNIFICANT = 2.5  # °C delta
DEFAULT_TEMP_MODERATE = 1.0     # °C delta

DEFAULT_WIND_RELEVANCE = 1.0    # m/s
DEFAULT_SOLAR_RELEVANCE = 0.5   # kWh

DEFAULT_CONTRADICTION_TEMP_DELTA = 2.0 # °C
DEFAULT_CONTRADICTION_WIND_DELTA = 2.5 # m/s
DEFAULT_CONTRADICTION_SOLAR_KWH = 1.5  # kWh

DEFAULT_SOLAR_SIGNIFICANT_KWH = 2.0  # Daily kWh delta
DEFAULT_SOLAR_MODERATE_KWH = 0.5     # Daily kWh delta

# Thermal Inertia Configuration (User Selectable)
CONF_THERMAL_INERTIA = "thermal_inertia"
THERMAL_INERTIA_FAST = "fast"
THERMAL_INERTIA_NORMAL = "normal"
THERMAL_INERTIA_SLOW = "slow"

DEFAULT_THERMAL_INERTIA_HOURS = 4

# Solar Defaults
DEFAULT_SOLAR_ENABLED = True
DEFAULT_SOLAR_CORRECTION = 100
ENERGY_GUARD_THRESHOLD = 0.01  # 10 Wh - Consistent guard against division by zero
DEFAULT_SOLAR_LEARNING_RATE = 0.01
DEFAULT_AUX_LEARNING_RATE = 0.01

# Default solar coefficients — starting point for per-unit EMA learning.
# Suitable for mixed installations (heat pumps + direct electric).
# The model will fine-tune per unit within 1-2 weeks of sunny weather.
DEFAULT_SOLAR_COEFF_HEATING = 0.35
DEFAULT_SOLAR_COEFF_COOLING = 0.40

# Solar thermal battery: exponential decay factor applied per hour.
# Models how solar energy absorbed by building mass is released over time.
# 0.80 → half-life ~3.1 h; appropriate for typical Norwegian construction
# with concrete floor slabs.  Per-installation calibration
# available via diagnose_solar with apply_battery_decay: true.
SOLAR_BATTERY_DECAY = 0.80

# Saturation-wasted thermal-feedback coefficient (#896).  Fraction of
# saturation-wasted solar (heating mode only) fed back into the solar
# thermal battery EMA on top of the applied solar.  When 0.0 (default),
# behaviour is bit-identical to the pre-#896 model.  When > 0.0, the
# EMA input becomes ``solar_impact + k × solar_wasted``, accounting for
# the portion of solar potential that exceeded VP demand and was clipped
# but still physically entered the building thermal mass.  Heating mode
# only — cooling-mode wasted is structurally zero (saturation returns
# wasted=0 for cooling) and gated additionally at the call site.
# Per-installation tuning via Advanced Options; expected useful range
# 0.3-0.5 for high-saturation installs based on issue research.  Field
# validation in progress.
CONF_BATTERY_THERMAL_FEEDBACK_K = "battery_thermal_feedback_k"
DEFAULT_BATTERY_THERMAL_FEEDBACK_K = 0.0

# Composite legacy floor for screen transmittance.  Used for facades configured
# WITHOUT explicit per-direction screen presence (CONF_SCREEN_SOUTH/EAST/WEST):
# represents the typical Nordic building where ~70-90 % of windows have screens
# and 10-30 % do not (north walls, utility rooms, skylights).  Raised from 0.20
# to 0.30 in v1.3.3 based on Nordic-residential analysis (issue #826):
# unscreened-window penalty dominates the floor and the previous 0.20 fit only
# fully-screened bespoke installations.
DEFAULT_SOLAR_MIN_TRANSMITTANCE = 0.30

# Per-direction floor for facades that ARE configured as screened
# (CONF_SCREEN_SOUTH/EAST/WEST = True).  Represents pure screen-fabric × glass
# transmittance: zip screen G-value ~0.06 combined with low-E triple glazing
# g-value ~0.5 yields ~0.08 effective.  Source: manufacturer datasheets
# referenced in issue #826 research.  Tunable in patch releases as summer
# diagnose_solar data accumulates.
SCREEN_DIRECT_TRANSMITTANCE = 0.08

CONF_WIND_UNIT = "wind_unit"
CONF_ENABLE_LIFETIME_TRACKING = "enable_lifetime_tracking"
CONF_SOLAR_ENABLED = "solar_enabled"
CONF_SOLAR_AZIMUTH = "solar_azimuth"
CONF_SOLAR_CORRECTION = "solar_correction"
# Per-direction screen presence flags (#826).  True = facade has external
# motorised screens that respond to the global solar_correction slider; the
# direction's transmittance falls to SCREEN_DIRECT_TRANSMITTANCE when fully
# closed.  False = facade has no screens; transmittance is fixed at 1.0
# regardless of slider position.  Default True for all three on upgrade so the
# composite behaviour stays similar to pre-1.3.3 (single global floor).
CONF_SCREEN_SOUTH = "screen_south"
CONF_SCREEN_EAST = "screen_east"
CONF_SCREEN_WEST = "screen_west"
DEFAULT_SCREEN_SOUTH = True
DEFAULT_SCREEN_EAST = True
DEFAULT_SCREEN_WEST = True

# Per-entity scope for the screen config.  List of energy_sensor entity_ids
# whose solar coefficients learn and predict against the installation-level
# `screen_config` + `solar_correction_percent`.  Entities NOT in the list
# effectively get `screen_config=(False, False, False)` — their coefficients
# learn and predict against pure transmittance=1.0 regardless of the
# slider.  Purpose: a unit in a room with no screens (e.g. a second-floor
# heat pump serving a west-facing room without motorised screens) should
# not absorb an avg transmittance it never physically experiences.  Default
# on upgrade / fresh install: all energy sensors included (preserves
# pre-#xxx behaviour).
CONF_SCREEN_AFFECTED_ENTITIES = "screen_affected_entities"

# Per-entity scope for solar coefficient learning + prediction (#962).
# Mirrors the screen_affected / aux_affected pattern: explicit user list of
# energy sensors whose consumption responds to solar gain.  Entities outside
# the list are excluded from all five solar learning paths (NLMS, inequality,
# cold-start, batch-fit, apply-implied) AND from the read path
# (calculate_unit_coefficient returns zero-vector instead of falling back to
# DEFAULT_SOLAR_COEFF_HEATING decomposition).  None / missing → default to all
# energy_sensors (legacy behaviour, no behaviour change on upgrade).
CONF_SOLAR_AFFECTED_ENTITIES = "solar_affected_entities"

DEFAULT_SOLAR_AZIMUTH = 180

WIND_UNIT_MS = "m/s"
WIND_UNIT_KMH = "km/h"
WIND_UNIT_KNOTS = "knots"

# Conversion Constants
MS_TO_KMH = 3.6
MS_TO_KNOTS = 1.94384

# Learning Constants
PER_UNIT_LEARNING_RATE_CAP = 0.03   # 3% max EMA rate for base/aux per-unit learning
SOLAR_COEFF_CAP = 5.0               # Max solar coefficient (kW per full sun)
COLD_START_SOLAR_DAMPING = 0.75     # Dampen cold-start solar estimates; base model noise inflates early samples
NLMS_STEP_SIZE = 0.10               # NLMS mu for solar coefficient learning (converges in ~10 qualifying hours)
NLMS_REGULARIZATION = 0.05          # NLMS epsilon: step-size denominator floor (mu / (||s||^2 + eps)).  Attenuates updates when input power is weak; does NOT shrink the coefficient toward zero.
SOLAR_DEAD_ZONE_THRESHOLD = 15      # Consecutive zero-impact sunny hours before forcing coefficient reset
# Solar shutdown detection (#838): identifies VP units whose thermostat cut
# the compressor because sun-heated rooms exceeded setpoint.  Such hours
# inflate the NLMS coefficient (actual_impact = base - 0 = base) and
# contaminate the base model via solar_normalization_delta.  Detection uses
# only data available at the call site (no historical tracking).
SOLAR_SHUTDOWN_ACTUAL_FLOOR = 0.03   # kWh — below this, unit is effectively off
SOLAR_SHUTDOWN_RATIO = 0.15          # actual/expected below this = shutdown
# Fallback default for shutdown-detection base gate.  Individual units may
# override this via ``_per_unit_min_base_thresholds`` populated by
# ``_calibrate_per_unit_min_base_thresholds`` (#871).  The per-unit path
# lets small loads (termostat, varmekabel) and low-min-modulation heat
# pumps (Panasonic 3 kW, Toshiba at mild temps) participate in learning
# below this global floor when their own noise floor warrants it.
SOLAR_SHUTDOWN_MIN_BASE = 0.15       # kWh — fallback; per-unit overrides preferred
SOLAR_SHUTDOWN_MIN_MAGNITUDE = 0.3   # potential vector magnitude — must be sunny
# Minimum per-unit base demand for solar NLMS learning.  Below this, the
# actual_impact = base - actual residual is dominated by VP cycling noise,
# not solar signal.  Separate from SOLAR_SHUTDOWN_MIN_BASE (which gates
# shutdown detection) because the two serve different purposes even though
# the values happen to be the same default today.
# Fallback default; per-unit overrides populated from dark-hour p10 (#871)
# take precedence.
SOLAR_LEARNING_MIN_BASE = 0.15       # kWh — fallback; per-unit overrides preferred

# Per-unit min-base auto-calibration.  Computes a per-sensor
# noise floor from dark-hour (solar_factor < 0.05) actual consumption
# and uses it as the NLMS + inequality + shutdown-detection gate,
# replacing the global 0.15 fallback when sufficient data exists.
PER_UNIT_MIN_BASE_FLOOR = 0.03              # Absolute floor on calibrated threshold
# Absolute ceiling sized for residential heat pumps up to ~10-12 kW nameplate
# running continuously at minimum modulation (~800-1000 W) in cold weather.
# Larger than this suggests a non-VP load scoped onto the sensor or an
# always-on circuit — the ratio-guard below is the primary filter for that
# class; this ceiling is a safety net.
PER_UNIT_MIN_BASE_CEILING = 1.5
PER_UNIT_MIN_BASE_MIN_SAMPLES = 20          # Min dark-hour samples for p10 to be trusted
PER_UNIT_MIN_BASE_MIN_HOURS_OF_LOG = 14 * 24  # 14 days × 24h before calibration runs
PER_UNIT_MIN_BASE_MAX_RATE_OF_CHANGE = 0.5  # Max ±50 % change per recalibration
PER_UNIT_MIN_BASE_DARK_SOLAR_FACTOR = 0.05  # solar_factor below = dark hour
# Ratio-guard: reject when p10 is too close to the median.  A legitimate
# noise floor sits far below typical consumption (off-periods in the tail,
# active modulation in the mass).  When p10/median approaches 1.0 the
# sensor is measuring an always-on load — electric boiler mislabeled as
# heat-pump heating, sensor scoped to a shared circuit, etc. — and the
# p10 is not a noise floor at all.
PER_UNIT_MIN_BASE_MAX_P10_MEDIAN_RATIO = 0.9
LEARNING_BUFFER_THRESHOLD = 4
TARGET_TDD_WINDOW = 0.5  # Minimum TDD accumulation for seamless rolling window efficiency
MIN_EXTRAPOLATION_DELTA_T = 0.5  # Minimum Delta T (Degrees) required to trust extrapolation source

# SNR-weighted base-model learning (#866).
# When True, base-bucket EMA uses signal-to-noise weighting instead of
# solar_normalization_delta compensation.  Target becomes raw actual;
# step size is scaled by snr_weight(solar_factor, shutdown_state).
# Dark hours retain full rate; sunny hours contribute proportional to
# their signal quality.
SNR_WEIGHT_FLOOR = 0.1  # Minimum weight for sunny hours (avoids bucket starvation)
SNR_WEIGHT_K = 3.0      # Slope: w = max(FLOOR, 1 − K × solar_factor)

# Global base EMA: skip threshold for solar-saturated hours.
# When the estimated global net (base − solar_normalization_delta) is below
# this value the EMA update is suppressed.  At 0.0 only fully-clipped hours
# are skipped (global_net = 0); raise to e.g. 0.010 (≈ 10 W) to also skip
# near-saturated hours where the learning signal is similarly unreliable.
GLOBAL_BASE_SATURATION_SKIP_KWH = 0.0

# Inequality learning for solar shutdown hours.
# Hours flagged as shutdown (by detect_solar_shutdown_entities) feed a
# parallel one-sided learner.  The learner enforces
# ``coeff · battery_filtered_potential ≥ INEQUALITY_MARGIN × base``
# via projected gradient with step ``INEQUALITY_STEP_SIZE``, distributing
# the deficit across non-zero components of the battery-filtered potential
# vector.  Non-negativity and ``SOLAR_COEFF_CAP`` clamps are preserved.
# Rationale: a shutdown hour carries physical information (solar gain was
# at least enough to cover base demand) that the equality-only NLMS path
# would otherwise discard.  On low-demand units the discard would create
# a permanent ceiling on west-coefficient convergence.
INEQUALITY_STEP_SIZE = 0.05   # Half of NLMS_STEP_SIZE — conservative for new mechanism
INEQUALITY_MARGIN = 0.9       # Constraint: coeff·potential ≥ MARGIN × base (10% buffer)

# Batch-fit solar coefficients (#884).  A periodic offline least-squares
# Periodic batch Tobit MLE fit over the modulating-regime hourly log
# (#884 LS introduction; #904 stage 2 swap to censoring-aware Tobit).
# Escapes the mild-weather catch-22 where NLMS and inequality both
# produce zero signal because expected base demand is near zero (e.g.
# west sun peaks during the warmest part of the day).  Saturation-
# clipped samples (HP fully off because the room got warm) are now
# kept as right-censored data with threshold ``T = ratio × base``,
# instead of being dropped as #884 did — Tobit's Mills-ratio
# likelihood term recovers slope information from the censoring
# point itself.  Per (entity, mode) — heating and cooling regimes
# are fit independently.  Damping factor revised after one season
# of data.
BATCH_FIT_DAMPING = 0.3              # new = α × tobit + (1 - α) × current
BATCH_FIT_SATURATION_RATIO = 0.95    # Censoring threshold: T_i = ratio × base_i

# Tobit MLE solver (#904).  Type-I right-censored Gaussian regression
# for solar coefficients.  Used by ``compute_tobit_for_diagnose``
# (stage 0+1 shadow surface in ``diagnose_solar``) AND by
# ``batch_fit_solar_coefficients`` (stage 2 live solver swap).
# Sample-size gates are two-pronged: ``|U| ≥ TOBIT_MIN_UNCENSORED``
# pre-fit (σ identifiability) and ``n_eff ≥ TOBIT_MIN_NEFF`` post-fit
# (slope identifiability) — censored samples that sit far inside the
# censoring region carry near-zero λ(q) and add no slope info, so we
# cannot count raw |C| toward effective sample size.
TOBIT_MIN_UNCENSORED = 20            # |U| floor — σ identifiability gate
TOBIT_RUNNING_WINDOW = 200           # #904 stage 3 — sliding-window cap for the live learner's recent-sample buffer.  Bounded memory (≈200 × 32 bytes ≈ 6 KB per (entity, regime) at peak); covers ~30 days of qualifying hours for a typical heating-active VP.  Newton iteration on the current window each hour.

# Outlier robustness (#919)
OUTLIER_RESIDUAL_WINDOW = 50        # Window size for the robust residual filter
OUTLIER_K_THRESHOLD = 5.0           # Filter samples where |residual| > k * sigma_robust
OUTLIER_MIN_SAMPLES = 20            # Wait for enough baseline before filtering
OUTLIER_REJECTED_POOL_SIZE = 20
OUTLIER_PROMOTION_THRESHOLD = 10
HARD_OUTLIER_CAP_FACTOR = 10.0
HARD_OUTLIER_SANITY_MULTIPLIER = 10.0
TOBIT_MIN_NEFF = 40                  # |U| + Σ_C λ(q)(λ(q)−q) floor — slope identifiability
TOBIT_MAX_ITER = 30                  # Projected-Newton iter cap
TOBIT_CONV_TOL = 1e-6                # ‖step‖∞ on (c, log σ)
TOBIT_Q_CLIP = -5.0                  # Lower trust-region clip on q = (T − c·s)/σ (Greene §17.3)

# Plausibility-gate v2 (#918, 1.3.5 default-on) — automatic discriminator
# applied inside ``_update_unit_tobit_live`` after the Tobit fit succeeds.
# Replaces the manual ``tobit_live_entities`` allow-list as the primary
# gate so default-on Tobit can ship without asking users to opt-in
# per-entity.
#
# Rationale: Tobit's value-add is recovering large coefficients from
# censoring information, but only meaningful when the uncensored
# samples have SOME slope to enhance.  Pure-noise loads (small
# electric circuits, wine cellars, garage sockets) have no uncensored
# slope anywhere — their Tobit fit is censoring-pattern-driven only,
# producing non-physical coefficients.  Magnitude-ratio
# ``|OLS|/|Tobit|`` failed to discriminate (Toshiba VP: 0.30,
# noise-load gjæringskjeller: 0.36 — VP ratio is LOWER because
# Tobit's amplification factor is similar regardless of the
# underlying physical reality).  OLS-max-direction across S/E/W is
# the right discriminator: legitimate VPs always show some
# uncensored signal in some direction; noise loads do not.
#
# Calibration on maintainer install (2026-04-30, 10-day window):
# Toshiba 0.33, Mitsubishi 0.12, gjæringskjeller 0.04, vinkjeller
# 0.005, garage 0.007, yaser-socket 0.009.  Threshold 0.10 sits in
# the gap.  Bump rule: revisit after N≥3 multi-install observations
# of legitimate-VP false-positives (real solar response, OLS max <
# 0.10) OR noise-load false-negatives (no real solar response, OLS
# max ≥ 0.10).  Both directions logged at info-level when the
# plausibility-gate fires so multi-install evidence accumulates
# passively post-default-on.
PLAUSIBILITY_MIN_OLS_MAX_DIRECTION = 0.10   # Largest |OLS_d| across S/E/W must clear this for Tobit to pass
PLAUSIBILITY_MIN_TOBIT_MAGNITUDE = 0.05     # Skip plausibility-check when Tobit fit is itself near-zero (no harm: zero writes through)

# Plausibility-gate v2 — direction-agreement check.  Magnitude-only
# discrimination misses a real failure mode: Tobit's projected-Newton
# active-set can pin a direction at zero from a wrong warm-start (e.g.
# NLMS-cold-start delivers ``{s: 1.0, e: 0, w: 0}`` to Tobit while real
# signal has shifted to W-dominant), producing a wrong-direction
# magnitude that satisfies ``ols_max ≥ 0.10`` because OLS-on-uncensored
# correctly identifies the W-direction signal.  The cosine check
# requires Tobit's coefficient vector to point in roughly the same
# direction as OLS-on-uncensored — catches the warm-start direction
# pinning failure.  Threshold 0.5 ≈ 60° mismatch tolerance — lenient
# enough that random noise doesn't trip it on legitimate fits.  Not
# applied on cooling regime (cooling has no censoring → Tobit ≡ OLS
# exactly → cosine ≡ 1).
PLAUSIBILITY_MIN_DIRECTION_COSINE = 0.5

# Plausibility-gate v2 — general step-size limiter on Tobit's per-hour
# delta.  Cap each direction's per-hour change to 30 % of the prior
# coefficient's maximum component whenever ANY direction's proposed
# step exceeds that cap.  Triggered by step magnitude alone (not
# plausibility-block history): fires both on the post-block recovery
# path AND on any other hour where Tobit's Newton step would produce
# a large discontinuity (e.g. fast-changing data, late-converging
# warm-start).  On the worst-case post-block jump (NLMS-converged
# 0.55 → Tobit-fit 1.65 = 200 % single-hour step), the limiter spreads
# convergence over ~5 hours: 0.715 → 0.929 → 1.207 → 1.569 → 1.65.
# Skipped on cold-start (``prior_max < 0.05``) — there's no prior to
# cushion against and the bootstrap step would otherwise stay clamped
# at zero.  Applied uniformly across heating and cooling regimes.
PLAUSIBILITY_RATE_LIMIT_FRACTION = 0.30

# Apply-implied-coefficient guard parameters (#884 follow-up).  The
# diagnose_solar implied-LS fit is precise but can be noisy on
# data-sparse installations, especially for directions where solar
# rarely arrives (e.g. west on a south-facing house).  The apply
# service evaluates per-direction stability across the diagnose
# stability_windows: a sign-flip OR > MAX_SPREAD ratio between
# windows means that component is noise-dominated and gets skipped
# (current value preserved); stable components are written.  The
# ``force`` service flag overrides per-component skipping.
APPLY_IMPLIED_MAX_SPREAD = 3.0          # max(|w|) / min(|w|) > this → unstable
APPLY_IMPLIED_NEAR_ZERO = 0.05          # all |w| below this → stable (effectively zero)
APPLY_IMPLIED_MIN_QUALIFYING_HOURS = 30 # at least this many qualifying hours required

# Cloud Coverage Default (when weather entity has unknown state)
DEFAULT_CLOUD_COVERAGE = 50.0

# Mixed Mode Detection Bounds (aux fraction for learning eligibility)
MIXED_MODE_LOW = 0.20   # Below this = mostly normal heating
MIXED_MODE_HIGH = 0.80  # Above this = mostly aux heating

# Aux Cooldown / Decay Mechanism (Prevent Thermal Lag Sampling Bias)
COOLDOWN_MIN_HOURS = 2              # Minimum hours to lock learning after Aux turns off
COOLDOWN_MAX_HOURS = 6              # Maximum safety timeout for the lock
COOLDOWN_CONVERGENCE_THRESHOLD = 0.92 # Convergence ratio (Actual/Expected) to exit early

# Dual Interference Guard (kWh threshold for both solar and aux)
DUAL_INTERFERENCE_THRESHOLD = 0.1

# Forecast Confidence Thresholds
CONFIDENCE_MIN_SAMPLES = 7          # Below this = "low" confidence
CONFIDENCE_HIGH_SAMPLES = 14        # Above this + low error = "high"
CONFIDENCE_HIGH_ERROR_MAX = 2.0     # p50 error ceiling for "high"
CONFIDENCE_MEDIUM_ERROR_MAX = 4.0   # p50 error ceiling for "medium"
FORECAST_COMPARISON_FACTOR = 0.9    # 10% better threshold for source comparison

# Thermal Load Stress Index Thresholds (% of max historical load)
STRESS_INDEX_LIGHT = 30
STRESS_INDEX_MODERATE = 60
STRESS_INDEX_HEAVY = 90

# Typical Day Matching
TYPICAL_DAY_TEMP_TOLERANCE = 1.0    # +/- degrees C for temperature matching
TYPICAL_DAY_WIND_TOLERANCE = 2.0    # m/s deviation from global average
TYPICAL_DAY_MIN_SAMPLES = 3
TYPICAL_DAY_HIGH_CONFIDENCE = 7

# TDD Stability Guard
TDD_STABILITY_THRESHOLD = 0.05     # TDD/hour minimum (~1.2C delta)

# Deviation Detection
DEVIATION_MIN_OBSERVATIONS = 5
DEVIATION_MIN_KWH = 0.2
DEVIATION_TOLERANCE_NEW = 0.75      # High tolerance for new data (0 obs)
DEVIATION_TOLERANCE_MATURE = 0.30   # Standard tolerance at maturity
DEVIATION_MATURITY_COUNT = 20.0     # Observations for full maturity

# Forecast Defaults (Safeguards for missing history)
DEFAULT_UNCERTAINTY_P50 = 1.0
DEFAULT_UNCERTAINTY_P95 = 2.0

# Storage
STORAGE_VERSION = 5  # v5: Tobit live-learner sufficient-statistic state (#904 stage 3, see storage.py:_migrate_v4_to_v5)
STORAGE_KEY = f"{DOMAIN}.storage"

# Solar model version (#904 stage 3 blocker 2 — manual reset hook).  Bump
# this whenever ``solar.py`` formulas / constants change in a way that
# affects the ``effective_solar_vector`` values we log at hour boundary.
# On Tobit live-learner load, if the stored model version differs from
# this constant the running sufficient-statistic is zeroed and rebuilt
# from cold-start (NLMS fallback fires until n_eff ≥ TOBIT_MIN_NEFF
# again).  Without this, Tobit would silently fit against logged
# vectors that no longer match the model it reconstructs against.
#
# Bump checklist (when in doubt, bump):
# - SolarCalculator azimuth-projection formula
# - Kasten cloud exponent or any cloud-factor constant
# - Air-mass formula or its base
# - Screen transmittance formula or constants (DEFAULT_SOLAR_MIN_TRANSMITTANCE,
#   SCREEN_DIRECT_TRANSMITTANCE, COMPOSITE_LEGACY_FLOOR)
# - Solar-vector decomposition (S/E/W projection logic)
# Not affected:
# - Coefficient-learning constants (NLMS step, regularization, etc.)
# - Tobit solver internals (TOBIT_MAX_ITER, TOBIT_CONV_TOL)
# - Storage / serialization changes that don't alter the logged value
SOLAR_MODEL_VERSION = 1

# Attributes
ATTR_EFFICIENCY = "efficiency_kwh_tdd"
ATTR_PREDICTED = "predicted_kwh"
ATTR_DEVIATION = "deviation_percent"
ATTR_TDD = "thermal_degree_days"
ATTR_FORECAST_TODAY = "forecast_today_kwh"
ATTR_CORRELATION_DATA = "correlation_data"
ATTR_LAST_HOUR_ACTUAL = "last_hour_actual_kwh"
ATTR_LAST_HOUR_EXPECTED = "last_hour_expected_kwh"
ATTR_LAST_HOUR_DEVIATION = "last_hour_deviation_kwh"
ATTR_LAST_HOUR_DEVIATION_PCT = "last_hour_deviation_pct"
ATTR_POTENTIAL_SAVINGS = "potential_savings"
ATTR_ENERGY_TODAY = "energy_today_kwh"
ATTR_EXPECTED_TODAY = "expected_today_kwh"
ATTR_TDD_DAILY_STABLE = "tdd_daily_stable"
ATTR_TDD_SO_FAR = "tdd_so_far_today"
ATTR_DEVIATION_BREAKDOWN = "deviation_breakdown"

# Temperature Stats Attributes
ATTR_TEMP_LAST_YEAR_DAY = "temp_last_year_day"
ATTR_TEMP_LAST_YEAR_WEEK = "temp_last_year_week"
ATTR_TEMP_LAST_YEAR_MONTH = "temp_last_year_month"
ATTR_TEMP_FORECAST_TODAY = "temp_forecast_today"
ATTR_TEMP_ACTUAL_TODAY = "temp_actual_today"
ATTR_TEMP_ACTUAL_WEEK = "temp_actual_week"
ATTR_TEMP_ACTUAL_MONTH = "temp_actual_month"

# TDD Stats Attributes
ATTR_TDD_YESTERDAY = "tdd_yesterday"
ATTR_TDD_LAST_7D = "tdd_last_7d_avg"
ATTR_TDD_LAST_30D = "tdd_last_30d_avg"

# Efficiency Stats Attributes
ATTR_EFFICIENCY_YESTERDAY = "efficiency_yesterday"
ATTR_EFFICIENCY_LAST_7D = "efficiency_last_7d_avg"
ATTR_EFFICIENCY_LAST_30D = "efficiency_last_30d_avg"
ATTR_EFFICIENCY_FORECAST_TODAY = "efficiency_forecast_today"

# Solar Attributes
ATTR_SOLAR_FACTOR = "solar_factor"
ATTR_SOLAR_IMPACT = "solar_impact_kwh"
ATTR_MIDNIGHT_FORECAST = "midnight_forecast_kwh"
ATTR_MIDNIGHT_UNIT_ESTIMATES = "midnight_unit_estimates"
ATTR_MIDNIGHT_UNIT_MODES = "midnight_unit_modes"
ATTR_FORECAST_UNCERTAINTY = "forecast_uncertainty"
ATTR_FORECAST_BLEND_CONFIG = "forecast_blend_config"
ATTR_FORECAST_ACCURACY_BY_SOURCE = "forecast_accuracy_by_source"
ATTR_FORECAST_DETAILS = "forecast_details"

ATTR_SOLAR_POTENTIAL = "solar_potential_kw"
ATTR_SOLAR_GAIN_NOW = "solar_gain_now_kw"
ATTR_HEATING_LOAD_OFFSET = "heating_load_offset"
ATTR_RECOMMENDATION_STATE = "recommendation_state"

# Recommendation States
RECOMMENDATION_MAXIMIZE_SOLAR = "maximize_solar"
RECOMMENDATION_INSULATE = "insulate"
RECOMMENDATION_MITIGATE_SOLAR = "mitigate_solar"

# Sensor Names (Suffixes)
SENSOR_EFFICIENCY = "Efficiency"
SENSOR_WEATHER_PLAN_TODAY = "Weather Plan Today"
SENSOR_DEVIATION = "Deviation"
SENSOR_EFFECTIVE_WIND = "Effective Wind"
SENSOR_CORRELATION_DATA = "Correlation Data"
SENSOR_LAST_HOUR_ACTUAL = "Last Hour Actual"
SENSOR_LAST_HOUR_EXPECTED = "Last Hour Expected"
SENSOR_LAST_HOUR_DEVIATION = "Last Hour Deviation"
SENSOR_POTENTIAL_SAVINGS = "Potential Savings"
SENSOR_ENERGY_TODAY = "Energy Consumption Today"
SENSOR_ENERGY_BASELINE_TODAY = "Energy Baseline Today"
SENSOR_ENERGY_ESTIMATE_TODAY = "Energy Estimate Today"
SENSOR_FORECAST_DETAILS = "Forecast Details"
SENSOR_THERMAL_STATE = "Thermal State"

# Cloud Coverage Map (Fallback for text states)
CLOUD_COVERAGE_MAP = {
    "clear-night": 0,
    "sunny": 0,
    "partlycloudy": 50,
    "cloudy": 85,
    "rainy": 95,
    "pouring": 100,
    "fog": 100,
    "hail": 100,
    "lightning": 95,
    "lightning-rainy": 95,
    "snowy": 100,
    "snowy-rainy": 100,
    "windy": 50,
    "windy-variant": 50,
    "exceptional": 50,
}


def convert_from_ms(value: float, unit: str) -> float:
    """Convert value from m/s to unit."""
    if unit == WIND_UNIT_KMH:
        return value * MS_TO_KMH
    if unit == WIND_UNIT_KNOTS:
        return value * MS_TO_KNOTS
    return value

def convert_to_ms(value: float, unit: str) -> float:
    """Convert value from unit to m/s."""
    if unit == WIND_UNIT_KMH:
        return value / MS_TO_KMH
    if unit == WIND_UNIT_KNOTS:
        return value / MS_TO_KNOTS
    return value

# Wind Stats Attributes
ATTR_WIND_LAST_YEAR_DAY = "wind_last_year_day"
ATTR_WIND_LAST_YEAR_WEEK = "wind_last_year_week"
ATTR_WIND_LAST_YEAR_MONTH = "wind_last_year_month"
ATTR_WIND_ACTUAL_TODAY = "wind_actual_today"
ATTR_WIND_ACTUAL_WEEK = "wind_actual_week"
ATTR_WIND_ACTUAL_MONTH = "wind_actual_month"

# New Model Comparison Sensor Names
SENSOR_MODEL_COMPARISON_DAY = "Model Comparison Day"
SENSOR_MODEL_COMPARISON_WEEK = "Model Comparison Week"
SENSOR_MODEL_COMPARISON_MONTH = "Model Comparison Month"
SENSOR_WEEK_AHEAD_FORECAST = "Week Ahead Forecast"
SENSOR_PERIOD_COMPARISON = "Period Comparison"
SENSOR_DAILY_LEARNING = "Daily Learning"

ATTR_SOLAR_PREDICTED = "solar_predicted_kwh"
ATTR_DAILY_FORECAST = "daily_forecast"
ATTR_WEEKLY_SUMMARY = "weekly_summary"
ATTR_FORECAST_RANGE_MIN = "forecast_range_min"
ATTR_FORECAST_RANGE_MAX = "forecast_range_max"
ATTR_AVG_TEMP_FORECAST = "avg_temperature"
ATTR_AVG_WIND_FORECAST = "avg_wind_speed"
ATTR_COLDEST_DAY = "coldest_day"
ATTR_WARMEST_DAY = "warmest_day"
ATTR_TYPICAL_WEEK_KWH = "typical_week_kwh"
ATTR_VS_TYPICAL_KWH = "vs_typical_kwh"
ATTR_VS_TYPICAL_PCT = "vs_typical_pct"
ATTR_PEAK_DAY = "peak_day"
ATTR_LIGHTEST_DAY = "lightest_day"
ATTR_WEEK_START_DATE = "week_start_date"
ATTR_WEEK_END_DATE = "week_end_date"

# Source Selection Constants
CONF_OUTDOOR_TEMP_SOURCE = "outdoor_temp_source"
CONF_WIND_SOURCE = "wind_source"
CONF_WIND_GUST_SOURCE = "wind_gust_source"
CONF_SECONDARY_WEATHER_ENTITY = "secondary_weather_entity"
CONF_FORECAST_CROSSOVER_DAY = "forecast_crossover_day"
CONF_AUX_AFFECTED_ENTITIES = "aux_affected_entities"
CONF_INDOOR_TEMP_SENSOR = "indoor_temp_sensor"
CONF_THERMAL_MASS = "thermal_mass_kwh_per_degree"
DEFAULT_THERMAL_MASS = 0.0
CONF_DAILY_LEARNING_MODE = "daily_learning_mode"
CONF_TRACK_C = "track_c_enabled"
CONF_MPC_ENTRY_ID = "mpc_entry_id"
CONF_MPC_MANAGED_SENSOR = "mpc_managed_sensor"

SOURCE_SENSOR = "sensor"
SOURCE_WEATHER = "weather"

DEFAULT_FORECAST_CROSSOVER_DAY = 4

# Modes
MODE_HEATING = "heating"
MODE_COOLING = "cooling"
MODE_OFF = "off"
MODE_GUEST_HEATING = "guest_heating"
MODE_GUEST_COOLING = "guest_cooling"
MODE_DHW = "dhw"

# Modes excluded from global model learning (Track B/C).
# Cooling participates in the global model since #801 introduced
# saturation-aware solar normalization that correctly handles mixed
# heating+cooling regimes via per-unit mode-signed solar deltas.
MODES_EXCLUDED_FROM_GLOBAL_LEARNING = frozenset({
    MODE_OFF,
    MODE_DHW,
    MODE_GUEST_HEATING,
    MODE_GUEST_COOLING,
})

# Canonical wind-bucket key for per-unit cooling-mode samples.  All
# cooling-mode per-unit learning writes land here regardless of actual
# wind, and per-unit cooling-mode prediction reads from here.  Never
# produced by coordinator._get_wind_bucket() (which only returns
# "normal" / "high_wind" / "extreme_wind" for heating), so the heating
# and cooling sample spaces stay cleanly separated inside the same
# [entity][temp_key][wind_bucket] structure.
COOLING_WIND_BUCKET = "cooling"

CONF_HOURLY_LOG_RETENTION_DAYS = "hourly_log_retention_days"
DEFAULT_HOURLY_LOG_RETENTION_DAYS = 90
HOURLY_LOG_RETENTION_OPTIONS = [90, 180, 365]

# --- Internal feature flags (not exposed in config flow) ---
# #793: Use COP-weighted smearing for Track B instead of flat q/24.
# When True and COP params are available (from MPC or future manual config),
# Track B distributes daily energy across 24 hours using per-hour COP weights
# instead of flat daily average. Provides Track A/C resolution without thermal sensors.
ENABLE_TRACK_B_COP_SMEARING = False
