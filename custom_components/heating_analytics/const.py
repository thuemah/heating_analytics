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

# Share of thermal demand one regime must hold before the building-level
# thermal_regime label commits to it (#1051).  Deliberately well above a bare
# majority: the label decides whether one regime's sign conventions may be
# applied to the whole building, and at a 60/40 split they may not.
THERMAL_REGIME_DOMINANCE_SHARE = 0.8
DEFAULT_SOLAR_LEARNING_RATE = 0.01
DEFAULT_AUX_LEARNING_RATE = 0.01

# Default solar coefficients — starting point for per-unit EMA learning.
# Suitable for mixed installations (heat pumps + direct electric).
# The model will fine-tune per unit within 1-2 weeks of sunny weather.
DEFAULT_SOLAR_COEFF_HEATING = 0.35
DEFAULT_SOLAR_COEFF_COOLING = 0.40

# Solar thermal battery: exponential decay factor applied per hour.
# Models how solar energy absorbed by building mass is released over time.
# 0.50 → half-life ~1.0 h.  Per-installation calibration available via
# diagnose_solar with apply_battery_decay: true.
SOLAR_BATTERY_DECAY = 0.50

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

# --- Battery recommendation noise floor (#1066) ---
# Every battery sweep picks its winner with ``< best - 1e-6``.  That
# epsilon is a guard against float equality, and it is correct for
# *selection* — argmin should pick the minimum.  What was missing is a
# separate question: is the winning margin larger than the noise it was
# measured against?  Without one, a converged install reports
# ``review_recommended`` permanently over differences in the
# milliwatt-hour range, and a verdict that is always on stops being read.
#
# The replays are PAIRED — same hours, different parameter — so the
# statistic is the per-hour difference of squared residuals:
#
#     d_h = residual_baseline[h]^2 - residual_candidate[h]^2
#     t   = mean(d) / SE(d)
#
# Computed on the squared-residual scale rather than the RMSE scale
# deliberately: that is where the loss is defined and what the replay
# minimises.  RMSE improvement is still reported; it just no longer
# decides.  Unlike a bare kWh floor this scales with the data's own
# dispersion, so it does not need per-install validation to mean
# something.
#
# THIS IS A SCREEN, NOT A CALIBRATED p-VALUE.  An earlier revision of
# this constant described 2.0 as "the conventional ~95 % two-sided bar".
# That claim was wrong as implemented and has been removed.  The
# candidate under test is the ARGMIN over the whole sweep grid, then
# tested on the very residuals that selected it — textbook post-selection
# inference.  Measured against a pure-noise null (200 trials, n=180,
# 110 candidates, zero true effect) the unadjusted t >= 2.0 bar fired on
# 56 % of trials.  Two adjustments in ``paired_loss_improvement`` bring
# that down; neither makes the number a p-value:
#
#   1. Selection penalty.  The threshold is raised by sqrt(2 ln m) for m
#      candidates considered — the leading term in the expected maximum
#      of m standard normals, which is what an argmin-then-test procedure
#      is actually up against.  At m = 110 that is ~3.07, so the
#      effective bar is ~5.07 rather than 2.0.  Deliberately
#      conservative: a missed real improvement costs the value of a
#      calibration nobody asked for, a false one costs the credibility
#      of every verdict the service emits.
#   2. Serial-correlation inflation.  The post-sunset residuals are
#      POST_SUNSET_REPLAY_HOURS consecutive hours per day driven by a
#      recursive EMA, so d_h is autocorrelated while var/n assumes
#      independence.  SE is inflated by sqrt((1+r1)/(1-r1)) on the lag-1
#      autocorrelation r1 when positive — the standard first-order
#      correction.
#
# Read the pair as "this margin survived a deliberately harsh screen",
# not as "p < 0.05".
BATTERY_RECOMMENDATION_MIN_T = 2.0
# Floor on paired hours before the screen may pass at all.  The bare
# n >= 2 guard is mathematically sufficient to compute a t statistic and
# nowhere near enough to trust one: at n = 2 the 95 % two-sided critical
# value is 12.7, so a fixed threshold is meaningless there.
BATTERY_RECOMMENDATION_MIN_PAIRED_HOURS = 10
# Post-sunset mean-residual magnitude above which the battery is judged
# to be decaying too fast / too slow.  Previously a bare ``0.05`` literal
# inline in the assessment expression.
BATTERY_RESIDUAL_BIAS_KWH = 0.05
# Minimum qualifying post-sunset hours before the bias flag may fire.
# Without it a handful of hours can produce a mean far from zero and the
# flag reads as a real bias rather than a thin-sample artefact.
BATTERY_BIAS_MIN_HOURS = 30
# Companion relative gate, applied IN ADDITION to the absolute kWh floor
# above — both must pass before the bias flag fires.
#
# The absolute floor alone means different things on different installs
# and in different seasons: 0.05 kWh against a 0.3 kWh post-sunset hour
# is a 17 % miss worth acting on, while the same 0.05 kWh against a 3 kWh
# midwinter hour is under 2 % and sits inside the base model's own noise.
# Firing on both as if they were the same finding is what made the flag
# permanent on a converged install.  Same principle as the Analysis
# Standards rule that an error is measured in its own regime — here the
# regime is the size of the hour.
#
# 0.10 is a starting point, not a calibrated value: a 10 % systematic
# miss on the post-sunset tail is the smallest deviation that survives
# being described to a user as worth a look.  ``relative_deviation`` is
# reported in ``diagnose_solar.battery_decay_health`` beside the mean, so
# an install that disagrees can read its own number and this can be
# re-tuned against evidence rather than argument.
#
# Because both gates must pass, raising this can only ever narrow what
# fires relative to the absolute-only behaviour it replaces.
BATTERY_RESIDUAL_BIAS_RELATIVE = 0.10

# Route the live solar read-path (prediction, base learning, the
# legacy battery, aux normalisation, display sensors) through the 4D
# pipeline (#962).  Default off — flag is a read-path route only.
# Toggle does NOT reset ``_solar_coefficients_per_unit``; the other
# pipeline continues to learn in parallel so rollback is symmetric.
#
# NOTE: the ``experimental_`` prefix in the key is **historical** and no
# longer describes the setting (#1062).  The condition is not maturity
# but input: 4D is superior wherever a real DNI/DHI source exists and
# inferior on the permanent ``kasten_synthetic`` branch, so this is a
# property of the user's weather provider, not a beta opt-in.  The
# user-facing strings say so; the key is deliberately left alone
# because renaming it buys nothing and costs a config-entry migration.
# Readiness (input support AND 4D having actually learned) is
# ``coordinator.evaluate_4d_readiness`` / ``diagnose_solar``'s
# ``four_d_readiness``.  See ``CLAUDE.md > Solar Model > 4D shadow
# learner > Promotion`` for the gate.
CONF_EXPERIMENTAL_4D_PRIMARY = "experimental_4d_primary"
DEFAULT_EXPERIMENTAL_4D_PRIMARY = False

# Composite legacy floor for screen transmittance.  Used for facades configured
# WITHOUT explicit per-direction screen presence (CONF_SCREEN_SOUTH/EAST/WEST):
# represents the typical Nordic building where ~70-90 % of windows have screens
# and 10-30 % do not (north walls, utility rooms, skylights).  Raised from 0.20
# to 0.30 in v1.3.3 based on Nordic-residential analysis (issue #826):
# unscreened-window penalty dominates the floor and the previous 0.20 fit only
# fully-screened bespoke installations.
DEFAULT_SOLAR_MIN_TRANSMITTANCE = 0.30

# Per-direction floor for facades that ARE configured as screened
# (CONF_SCREEN_SOUTH/EAST/WEST = True).  Represents the area-weighted
# composite transmittance of a typical Nordic "screened" facade: zip
# screen + low-E triple glazing achieve ~0.05-0.10 transmittance on
# the screened portion of glass, but architectural conventions mean
# only ~70-85 % of a screened facade's glass area is actually behind
# motorised screens — terrace doors, kitchen windows, karnapp /
# utility windows on the same facade typically remain unscreened.
# Pre-1.3.3 used a single global floor of 0.30 (composite); 1.3.3
# split per-direction and accidentally set this constant to 0.08
# (pure material physics, equivalent to assuming 100 % screen
# coverage on a screened facade — physically unreachable in typical
# residential construction).  Restored to the area-weighted Nordic
# composite range 0.28-0.33 (Nordic-residential analysis with
# manufacturer-datasheet material physics + TEK17/BBR architectural
# constraints); 0.30 is the empirically validated production value
# that matches pre-1.3.3 behaviour.  Tunable in patch releases via
# summer diagnose_solar evidence.
SCREEN_DIRECT_TRANSMITTANCE = 0.30

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

# Service name for the 4D shadow batch-fit (#954).  Strict shadow —
# writes only to ``_solar_coefficients_4d_per_unit``; no production
# read-path consumer yet.
SERVICE_BATCH_FIT_SOLAR_4D = "batch_fit_solar_4d"

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

# Week-horizon forecast accuracy.  The 7-day plan is snapshotted every
# midnight and later scored as a week SUM against daily_history actuals —
# measuring "what do we usually miss by for a week" directly instead of
# constructing it from day-ahead errors (which underrepresent days 2-7
# and ignore day-to-day error correlation).
WEEK_PLAN_RETENTION_DAYS = 120        # stored midnight 7-day plans (8 floats/day)
WEEK_HORIZON_STATS_WINDOW_DAYS = 90   # trailing percentile window — keeps errors in the current load regime
WEEK_HORIZON_MIN_WINDOWS = 14         # scorable windows before the range band is surfaced (~2 independent weeks of evidence; rolling windows overlap)

# Storage
STORAGE_VERSION = 9  # v9: solar-window low+high obstruction gate per facade per entity (see storage.py:_migrate_v8_to_v9)

# Solar-window obstruction gate (v9).  Each facade per entity carries
# two independent critical elevations: ``critical_elev_low`` (below
# which terrain / neighbouring buildings block direct beam) and
# ``critical_elev_high`` (above which an overhang / terrace blocks
# direct beam).  ``pot_dir_facade`` is zeroed in
# ``calculate_unit_potential_4d`` when ``sun_elev < low`` OR
# ``sun_elev > high``; diffuse is unaffected.  ``None`` per boundary
# = no gate on that side (default).  Two independent brute-force
# searches are run per facade by ``fit_solar_obstruction``: a low-
# horizon search (inverted gate — samples BELOW cutoff zeroed) and a
# high-horizon search (samples ABOVE cutoff zeroed, legacy behaviour).
# Shutdown-flagged hours constrain the feasible range (hard constraint:
# a shutdown sample at elevation E proves unobstructed sun at E, so
# ``low < E < high``) but do NOT participate in SSE scoring.
#
# MIN_ELEV_DEG lowered from 15° to 5° because real-world fits were
# clipping on the floor (multiple installs producing best_critical_elev
# = 15° simultaneously on west facades — pathognomonic for "search range
# truncates the true minimum").  The per-side sample-count gate
# (OBSTRUCTION_FIT_MIN_SAMPLES_PER_SIDE = 10) prevents the lower range
# from picking up noise on installs that have no low-elevation samples
# — candidates below the data's elev range fail the n_below gate.
# LOW / HIGH suffixes distinguish the two independent brute-force
# sweeps in the v9 solar-window fit.
OBSTRUCTION_FIT_LOW_MIN_ELEV_DEG = 5.0
OBSTRUCTION_FIT_LOW_MAX_ELEV_DEG = 40.0   # lower horizon typically ≤ 30-35°
OBSTRUCTION_FIT_HIGH_MIN_ELEV_DEG = 15.0  # upper horizon typically ≥ 20-25°
OBSTRUCTION_FIT_HIGH_MAX_ELEV_DEG = 70.0
OBSTRUCTION_FIT_STEP_DEG = 1.0
OBSTRUCTION_FIT_MIN_SAMPLES_PER_SIDE = 10
# Raised from 0.10 to 0.30 (#1020).  Genuine geometric step-functions
# produce SSE improvement in the 0.50-0.80 range when data covers them
# cleanly; the 0.10 floor allowed cosmetic-noise fits to pass.  0.30
# cuts out the marginal regime where artifacts dominate without
# losing real-geometry detection.  Combined with the multi-window
# stability gate, physical-plausibility prior, and suggestion-rather-
# than-auto-write default, this scopes the obstruction path to
# informed decisions rather than automatic convergence.
OBSTRUCTION_FIT_SSE_IMPROVEMENT_THRESHOLD = 0.30
OBSTRUCTION_FIT_DOMINANCE_RATIO = 0.5
# Physical-plausibility prior (#1020).  Terrain horizons above 20° are
# extremely rare (would require fjellvegg or close-neighbour wall);
# overhangs below 20° are architecturally unusual.  Fits landing
# outside these bounds are surfaced with ``applicable_reason =
# "physically_implausible"`` and never produce a suggested gate.
# Users with genuine outliers (rare) can still write manually via
# ``apply_obstruction_gate`` (also range-validated; use ``force`` if
# you really mean it — currently not implemented because no install
# in the wild has needed it).
OBSTRUCTION_LOW_PLAUSIBLE_RANGE = (2.0, 20.0)
OBSTRUCTION_HIGH_PLAUSIBLE_RANGE = (20.0, 60.0)
# Cooling-data floor for HIGH-gate auto-suggestion (#1020).  At
# northern latitudes with heating-only data, sun rarely climbs high
# enough to populate the HIGH regime — the fit will either find
# nothing or "find" a marginal artifact reflecting noise rather than
# geometry.  Below this floor of cooling-mode samples per facade,
# the HIGH sweep is skipped honestly with
# ``insufficient_cooling_for_high_regime``.  LOW sweep is unaffected
# (heating-mode data covers low elevations year-round).
OBSTRUCTION_FIT_MIN_COOLING_SAMPLES_FOR_HIGH = 50
# Multi-window stability gate (#1020).  The fit is re-run on three
# overlapping time windows; a boundary becomes ``applicable`` only
# when ≥ MIN_AGREEING of the windows produce a best_critical_elev
# within ±TOLERANCE_DEG of each other (and all values pass the
# plausibility prior).  Single-window agreement is insufficient —
# only consistency over time is artifact-immune.  Calibration-era
# transitions (e.g. the SCREEN_DIRECT_TRANSMITTANCE 0.08 → 0.30 fix)
# leave fingerprints that this gate naturally rejects without
# explicit era tracking.
OBSTRUCTION_STABILITY_WINDOWS = (30, 60, 90)
OBSTRUCTION_STABILITY_TOLERANCE_DEG = 3.0
OBSTRUCTION_STABILITY_MIN_AGREEING = 2
# Adaptive kNN-window parameters for the local SSE test.  Shared across
# both the low-horizon and high-horizon brute-force sweeps.  The window
# selects the ``K_PER_SIDE`` nearest samples below and above each
# candidate cutoff (by elevation distance).  This decouples the
# discontinuity test from samples far from the hypothesised edge — a
# sample at 65° elevation no longer participates in the score for an
# edge at 25°.  WINDOW_MAX_ANGULAR_DEG caps the angular span; if the
# K_PER_SIDE nearest samples on either side fall outside this span,
# the candidate is skipped.  K_PER_SIDE = 20 balances statistical
# power against locality; MIN_SAMPLES_PER_SIDE remains the absolute
# floor.
OBSTRUCTION_FIT_WINDOW_K_PER_SIDE = 20
OBSTRUCTION_FIT_WINDOW_MAX_ANGULAR_DEG = 25.0
# Absolute amplitude gate — defends against the relative-SSE-threshold
# being too permissive on entities with weak directional signals
# (diffuse-dominated units where sse_flat itself is in the noise-floor
# range).  Per-sample RMS reduction must exceed this value in kWh for
# the gate to be considered learnable, in addition to the relative
# improvement gate.  0.05 kWh/sample ≈ 50 Wh/hour, around 1-10 % of
# typical HP electrical input — below this is noise floor.
OBSTRUCTION_FIT_MIN_RMS_REDUCTION_KWH = 0.05
# Minimum shutdown-sample count required before the low/high cutoff
# constraint kicks in.  Shutdown detection has known noise — a single
# false-positive at an extreme elevation would otherwise permanently
# block the gate via raw ``min``/``max`` constraint.  Requiring at
# least N samples to support the constraint (and indexing into the
# sorted shutdown elevations from the N-th lowest / highest) shields
# the fit from individual outliers while preserving the hard
# constraint when the shutdown evidence is real.
OBSTRUCTION_FIT_SHUTDOWN_CONSTRAINT_MIN_N = 3
# Minimum gap between learned low and high cutoffs.  Without shutdown
# data, the two sweeps are independent and noisy small-sample fits
# can produce inverted or near-inverted windows that gate the entire
# facade.  The combined-window sanity check refuses to write when
# ``high - low < MIN_WINDOW_DEG``; conservative default of 10° rules
# out only physically implausible configurations.
OBSTRUCTION_FIT_MIN_WINDOW_DEG = 10.0
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

# --- DNI/DHI ladder source mix (diagnose_solar.dni_dhi_source_mix) ---
# The per-install gate for ``experimental_4d_primary``.  4D is superior
# wherever a real DNI/DHI source exists and inferior on the
# ``kasten_synthetic`` branch (one scalar cannot become two independent
# signals; the Erbs split trades a ~1 % constant bias for a kT-dependent
# one) — so the only question a per-install gate must answer is whether
# this install resolves through a real source.
#
# Dominance bar for "real source": share of DAYLIGHT hours resolving via
# ``native`` or ``erbs_from_ghi``.  Set high because the failure mode is
# asymmetric — enabling 4D on a predominantly-Kasten install degrades the
# model, while leaving it off on a borderline install costs only the
# improvement.  An install genuinely fed by a weather provider with
# DNI/DHI sits near 1.0; anything mixed is a sign the provider drops the
# fields intermittently.
DNI_DHI_REAL_SOURCE_DOMINANCE_MIN = 0.80
# Minimum labelled daylight hours before a verdict is offered.  Two
# daylight days is far too thin to characterise a provider that drops
# fields under some conditions; ~1 week of daylight hours is the floor.
DNI_DHI_SOURCE_MIX_MIN_HOURS = 50

# --- DNI/DHI outage repair issue (#1070) ---
# Alert, not a routing decision.  Deliberately does NOT reuse the two
# constants above: ``supports_4d_primary`` is a 30-day / 80 % / 50-hour
# gate because routing an install is a slow, considered choice.  This is
# the opposite question — "did my provider stop publishing irradiance in
# the last day or two?" — and a 30-day window would take weeks to notice.
# Two different questions, two different windows.
#
# Counted in DAYLIGHT hours, not wall-clock hours.  Mandatory, not
# hygiene: ``derive_dni_dhi_source_label`` never emits ``"no_sun"``, so a
# night hour is labelled ``kasten_synthetic`` on any install with
# cloud-coverage data and a wall-clock window would fire every night on a
# perfectly healthy install.  Same trap ``_compute_dni_dhi_source_mix``
# guards with ``solar_factor > 0``; the same filter is reused.
#
# Seasonal consequence at ~60 °N, and it is the intended behaviour: 24
# daylight hours is ~1.3 days in June (~18 h daylight) and ~4 days in
# December (~6 h).  The window measures *evidence*, not elapsed time.  Do
# not "fix" this into wall-clock hours.
REPAIR_DNI_DHI_OUTAGE_WINDOW_HOURS = 24
# Asymmetric by design, so a provider that drops the fields intermittently
# cannot create and delete the repair on alternating days.  Between the
# two shares the state is sticky: neither raised nor cleared, whatever it
# already was persists.
#
# The raise bar is a slack allowance, not a statistical claim.  A provider
# that stops publishing collapses to exactly 0.0 — the failure is sharp,
# not gradual — so 0.10 exists only to tolerate a single stray hour
# slipping through a genuine outage.  Neither number has been validated
# against a real outage on a real install; they are starting points.
REPAIR_DNI_DHI_OUTAGE_RAISE_BELOW = 0.10
REPAIR_DNI_DHI_OUTAGE_CLEAR_AT = 0.50
# No verdict at all until the window is fully populated.  Without this a
# freshly-configured 4D install raises the repair on its first day, before
# any evidence exists that anything is wrong.
REPAIR_DNI_DHI_OUTAGE_MIN_HOURS = 24
# Issue ID registered with HA's issue registry.  Stable across restarts —
# the registry persists, and re-registering the same ID is idempotent.
REPAIR_ISSUE_DNI_DHI_OUTAGE = "dni_dhi_outage_4d_active"
