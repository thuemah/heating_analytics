"""Data contracts for the observation → learning pipeline.

Defines immutable data structures that decouple observation collection
from learning and prediction. These replace the ~30 loose parameters
previously passed between coordinator, learning, statistics, and forecast.

Part of the coordinator decomposition (#775).
Per-unit learning strategies added in #776.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Protocol, runtime_checkable

from .const import (
    MODE_HEATING,
    MODES_EXCLUDED_FROM_GLOBAL_LEARNING,
    SOLAR_SHUTDOWN_ACTUAL_FLOOR,
    SOLAR_SHUTDOWN_RATIO,
    SOLAR_SHUTDOWN_MIN_BASE,
    SOLAR_SHUTDOWN_MIN_MAGNITUDE,
)

_LOGGER = logging.getLogger(__name__)


def detect_solar_shutdown_entities(
    *,
    solar_enabled: bool,
    is_aux_dominant: bool,
    potential_vector: tuple[float, float, float],
    energy_sensors: list[str],
    unit_modes: dict[str, str],
    unit_actual_kwh: dict[str, float],
    unit_expected_base_kwh: dict[str, float],
    unit_min_base: dict[str, float] | None = None,
    solar_affected_entities: frozenset[str] | None = None,
) -> tuple[str, ...]:
    """Detect VP units whose thermostat cut the compressor during sun (#838).

    A unit is flagged solar-dominant when it reports near-zero consumption
    during sunny conditions with significant expected base demand — the
    signature of a heat pump thermostat shutdown in a sun-warmed room.
    Learning on such hours would inflate the solar coefficient because
    actual_impact = base - 0 = base.

    Detection is purely a function of this-hour data — no historical
    tracking, no circular dependencies on the learned coefficient.

    Magnitude uses the potential (pre-screen) vector so closed screens
    don't hide a sunny hour from detection, matching the learning-side
    reconstruction in LearningManager._process_per_unit_learning.

    Args:
        solar_enabled: Skip detection entirely when solar is disabled.
        is_aux_dominant: Skip during aux regime — aux already freezes base.
        potential_vector: (S, E, W) pre-screen solar vector.
        energy_sensors: Candidate entity IDs.
        unit_modes: {entity_id: MODE_*}; only MODE_HEATING is considered.
        unit_actual_kwh: {entity_id: metered kWh this hour}.
        unit_expected_base_kwh: {entity_id: base-expected kWh this hour}.
        unit_min_base: Optional per-unit base thresholds (#871).  When
            provided, overrides SOLAR_SHUTDOWN_MIN_BASE per entity;
            absent entries fall back to the global constant.  Lets
            small loads and low-min-modulation heat pumps participate
            in shutdown detection at their own calibrated noise floor.

    Returns:
        Tuple of entity IDs flagged as solar-dominant (possibly empty).
    """
    if not solar_enabled or is_aux_dominant:
        return ()

    magnitude = (
        potential_vector[0] ** 2
        + potential_vector[1] ** 2
        + potential_vector[2] ** 2
    ) ** 0.5
    if magnitude < SOLAR_SHUTDOWN_MIN_MAGNITUDE:
        return ()

    # Defensive: only accept dict-shaped overrides.  Anything else
    # (None, MagicMock, malformed storage load) falls back to the global
    # constant — mirrors the same posture in learning._resolve_min_base.
    overrides = unit_min_base if isinstance(unit_min_base, dict) else {}
    # Per-entity solar-scope gate (#962): entities outside the
    # solar_affected_entities set cannot be "shut down by sun" — they have
    # no solar response by user declaration.  Their low actual_kwh relative
    # to base reflects other physics (slab-thermostat duty cycle, parasitic
    # load patterns, etc.) and must not pollute solar_dominant_entities or
    # flip solar_regime to "shutdown" downstream.  None sentinel = legacy
    # path (all entities eligible) for test fixtures.
    affected: frozenset[str] | None = (
        solar_affected_entities
        if isinstance(solar_affected_entities, (frozenset, set))
        else None
    )
    flagged: list[str] = []
    for entity_id in energy_sensors:
        if affected is not None and entity_id not in affected:
            continue
        if unit_modes.get(entity_id, MODE_HEATING) != MODE_HEATING:
            continue
        actual = unit_actual_kwh.get(entity_id, 0.0)
        base = unit_expected_base_kwh.get(entity_id, 0.0)
        override_val = overrides.get(entity_id)
        if isinstance(override_val, (int, float)) and override_val > 0.0:
            threshold = float(override_val)
        else:
            threshold = SOLAR_SHUTDOWN_MIN_BASE

        # Absolute parasitic-floor check fires when ``base > 0`` AND
        # ``actual < ACTUAL_FLOOR`` — even when ``base < threshold``.
        # Pre-fix the gate ordering put the ``base < threshold`` skip
        # BEFORE the absolute check, which meant a low-modulation HP
        # showing standby/parasitic consumption (~12 Wh) on a mild day
        # with low base (~0.08 kWh) got missed: base failed the
        # eligibility gate, the absolute actual-floor check never ran,
        # and the hour was misclassified as modulating instead of
        # shutdown.  Tobit / NLMS then ingested it with
        # ``actual_impact ≈ 0.07 kWh`` ("86 % of base disappeared"),
        # inflating the solar coefficient on what was physically a
        # fully-off compressor.  See post-stage-3 audit (#904) for the
        # field data.  ``base > 0`` is required so that an offline
        # sensor (no data, base reads zero) is NOT mistaken for a
        # shutdown — that case should remain "no signal", not
        # "shutdown".
        if base > 0.0 and actual < SOLAR_SHUTDOWN_ACTUAL_FLOOR:
            flagged.append(entity_id)
            continue

        if base < threshold:
            # Below noise floor AND not parasitic — neither modulating-
            # nor-shutdown-classifiable.  Existing behaviour preserved
            # for non-parasitic low-base hours.
            continue
        ratio = actual / base if base > 0 else 1.0
        if ratio < SOLAR_SHUTDOWN_RATIO:
            flagged.append(entity_id)
    return tuple(flagged)


@dataclass(frozen=True)
class LearningConfig:
    """Per-hour learning configuration passed to LearningManager.

    Captures the coordinator settings and eligibility decisions that
    govern how (and whether) the model is updated for a given hour.
    Frozen because these values must not change during a learning pass.
    """

    learning_enabled: bool
    solar_enabled: bool
    learning_rate: float
    balance_point: float
    energy_sensors: list  # [str] — sensor entity IDs to learn from
    aux_impact: float  # kW reduction from aux coefficients for this temp/wind

    # Service dependencies (passed as callables / instances)
    solar_calculator: object = None  # SolarCalculator instance
    get_predicted_unit_base_fn: Callable | None = None

    # Aux control
    aux_affected_entities: list | None = None
    has_guest_activity: bool = False

    # Per-unit Track A override (active under daily_learning_mode)
    # None = follow learning_enabled.
    per_unit_learning_enabled: bool | None = None

    # Per-direction screen presence (#826).  None = legacy / use composite floor.
    # When set, controls how correction_percent attenuates each cardinal
    # direction independently in :meth:`SolarCalculator._screen_transmittance_vector`.
    screen_config: tuple[bool, bool, bool] | None = None

    # Per-entity scope for the screen_config above.  Set of entity_ids whose
    # solar coefficients learn against the installation-level screen_config;
    # others get ``(False, False, False)`` (transmittance=1.0) at learn time.
    # None = legacy behaviour (all entities affected, as before).
    screen_affected_entities: frozenset[str] | None = None

    # Per-entity scope for solar coefficient learning entirely (#962).
    # Entities outside this set are excluded from all five solar learning
    # paths (NLMS, inequality, cold-start, batch-fit, apply-implied) and
    # from prediction read-path.  Distinct from screen_affected_entities
    # which only narrows the screen-transmittance treatment within the
    # solar pipeline; this set decides whether the entity participates in
    # solar at all.  Use case: floor-heating cables on slab thermostat
    # (slab dominates over solar gain regardless of windows), interior
    # loads with no solar exposure, parasitic basement consumption.
    # None = legacy behaviour (all entities solar-affected).
    solar_affected_entities: frozenset[str] | None = None

    # Per-unit min-base thresholds (#871).  Maps entity_id → calibrated
    # noise floor (kWh) used to gate NLMS, inequality, and shutdown
    # detection per unit.  None or empty → all gate sites fall back to
    # the global SOLAR_*_MIN_BASE constants (legacy behaviour).
    unit_min_base: dict[str, float] | None = None

    # Tobit live-learner gate (storage v5+, default-on on 1.3.5+).
    # The master flag must be True AND the (entity, regime) must satisfy
    # the scope predicate: empty ``tobit_live_entities`` = auto-mode
    # (every eligible entity candidates Tobit, plausibility-gate v2
    # decides per entity); non-empty list = scope-restrict to listed
    # entities only (plausibility still applies within scope —
    # explicit listing does not bypass the noise-load filter).  When
    # the master flag is False (user explicit disable), NLMS runs as
    # primary writer for every entity and the running sufficient-
    # statistic state is untouched.
    experimental_tobit_live_learner: bool = False
    tobit_live_entities: frozenset[str] = field(default_factory=frozenset)
    # MPC-managed entities (Track C, ``WeightedSmear(use_synthetic=True)``):
    # excluded from live Tobit unconditionally — even when scope-listed
    # by the user.  MPC's load-shifting produces samples that are not a
    # function of solar potential at this hour (HP runs hard at 02:00
    # without sun, idles at 12:00 with sun).  Tobit fitting against
    # such samples produces non-physical coefficients.  Mirrors the
    # ``weighted_smear_excluded`` skip path that
    # ``batch_fit_solar_coefficients`` already applies; defense-in-depth
    # so scope-list misconfiguration cannot accidentally feed Tobit
    # MPC-managed data.  Track C users continue to use
    # ``apply_implied_coefficient`` (path 5) for MPC-managed sensors,
    # which is the documented mechanism.
    mpc_managed_entities: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class HourlyObservation:
    """Immutable snapshot of one completed hour of sensor data.

    Produced by the observation layer at each hour boundary.
    Consumed by LearningManager, and logged to hourly_log.
    All values are final (averaged/aggregated) — no further mutation.
    """

    # --- Time ---
    timestamp: datetime
    hour: int

    # --- Weather (aggregated over the hour) ---
    avg_temp: float
    inertia_temp: float
    temp_key: str  # str(int(round(inertia_temp)))
    effective_wind: float
    wind_bucket: str  # "normal" | "high_wind" | "extreme_wind"
    bucket_counts: dict[str, int]  # {"normal": n, "high_wind": n, "extreme_wind": n}

    # --- Humidity (aggregated over the hour) ---
    avg_humidity: float | None  # Relative humidity (%), None if unavailable

    # --- Solar (aggregated over the hour) ---
    solar_factor: float
    solar_vector: tuple[float, float, float]  # (south, east, west)
    solar_impact_raw: float  # Raw hourly solar impact (before battery)
    effective_solar_impact: float  # Battery-smoothed solar impact

    # --- Energy ---
    total_energy_kwh: float  # Full metered energy (for logging/display)
    learning_energy_kwh: float  # Guest/DHW-excluded energy (for model learning)
    guest_impact_kwh: float
    expected_kwh: float  # Accumulated expected (minute-by-minute)
    base_expected_kwh: float  # Physics base (no aux/solar)
    unit_breakdown: dict[str, float]  # {entity_id: kwh}
    unit_expected: dict[str, float]  # {entity_id: expected_kwh}
    unit_expected_base: dict[str, float]  # {entity_id: base_expected_kwh}

    # --- Auxiliary heating ---
    aux_impact_kwh: float
    aux_fraction: float  # 0.0–1.0 (minutes active / total samples)
    is_aux_dominant: bool  # aux_fraction >= 0.80

    # --- Metadata ---
    sample_count: int
    unit_modes: dict[str, str]  # {entity_id: MODE_*}

    # --- Forecast context (computed at hour boundary) ---
    forecasted_kwh: float | None = None
    forecasted_kwh_primary: float | None = None
    forecasted_kwh_secondary: float | None = None
    forecasted_kwh_gross: float | None = None
    forecasted_kwh_gross_primary: float | None = None
    forecasted_kwh_gross_secondary: float | None = None
    forecast_source: str | None = None

    # --- Solar optimizer context ---
    recommendation_state: str = "none"
    correction_percent: float = 100.0  # 100% = screens open (no reduction)
    potential_solar_factor: float = 0.0

    # --- Solar normalization for global model (#801) ---
    # Saturation-aware delta: heating_solar_applied - cooling_solar_applied.
    # Learning uses: dark_actual = actual + solar_normalization_delta.
    # This correctly handles mixed heating+cooling without global_mode inference.
    solar_normalization_delta: float = 0.0

    # --- Cooldown state at observation time ---
    was_cooldown_active: bool = False

    # --- Solar shutdown regime (#838) ---
    # Detected at hour boundary when VP units report near-zero consumption
    # during sunny conditions with significant expected base demand.  Used
    # by LearningManager to (a) skip NLMS coefficient updates that would
    # otherwise learn actual_impact = base, and (b) slow per-unit base EMA
    # via headroom weighting.  Defaults ensure zero impact on existing code.
    is_solar_dominant: bool = False
    solar_dominant_entities: tuple[str, ...] = ()

    # Battery-filtered per-direction potential (#865). Consumed by the
    # inequality learner which operates on shutdown-flagged hours.  The
    # inequality cares about accumulated sun presence (thermal-mass time
    # constant ~3h) rather than the instantaneous hour's vector, because
    # shutdown is an accumulated phenomenon — the room overheated over
    # hours, not within one.  Lives on the observation so it's available
    # at the per-unit learning site without threading an extra argument.
    # Defaults to zeros for legacy call paths (replay builds its own via
    # a local EMA over the log; live path fills from coordinator state).
    battery_filtered_potential: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class ModelState:
    """Reference-based view of the learned model state.

    Passed to StatisticsManager, ForecastManager, and SolarCalculator
    instead of direct coordinator._field access.  Holds references to
    the coordinator's canonical dicts (not deep copies), so mutations
    to the underlying dicts are immediately visible.  Not frozen because
    the learning layer mutates the referenced dicts in-place.

    Consumers should treat the *references* as stable (don't reassign
    fields) and the *contents* as read-only outside the learning layer.
    """

    # --- Global correlation model ---
    correlation_data: dict  # {temp_key: {wind_bucket: avg_kwh}}

    # --- Per-unit models ---
    correlation_data_per_unit: dict  # {entity_id: {temp_key: {wind_bucket: avg_kwh}}}
    observation_counts: dict  # {entity_id: {temp_key: {wind_bucket: count}}}

    # --- Auxiliary heating coefficients ---
    aux_coefficients: dict  # {temp_key: kw_reduction}
    aux_coefficients_per_unit: dict  # {entity_id: {temp_key: {wind_bucket: kw_reduction}}}

    # --- Solar coefficients ---
    # Mode-stratified per #868:
    # ``{entity_id: {"heating": {s, e, w}, "cooling": {s, e, w}}}``.
    solar_coefficients_per_unit: dict

    # --- Daily learning ---
    learned_u_coefficient: float | None

    # --- Learning buffers (mutable during learning phase only) ---
    learning_buffer_global: dict = field(default_factory=dict)
    learning_buffer_per_unit: dict = field(default_factory=dict)
    learning_buffer_aux_per_unit: dict = field(default_factory=dict)
    learning_buffer_solar_per_unit: dict = field(default_factory=dict)

    # --- History (read-only references) ---
    daily_history: dict = field(default_factory=dict)
    hourly_log: list = field(default_factory=list)

    # --- Tobit live-learner state (#904 stage 3) ---
    # Mutable references owned by the coordinator; consumed by
    # ``LearningManager._update_unit_tobit_live`` at hour boundaries.
    # ``None`` is permitted on legacy ModelState construction sites
    # (replay paths, tests pre-stage-3) — learning code falls back to
    # NLMS-only when either is None.
    tobit_sufficient_stats: dict | None = None
    nlms_shadow_coefficients: dict | None = None
    # Stage 3 (#912) review I1: NLMS shadow path uses a separate
    # cold-start buffer to avoid aliasing main's buffer.  Coordinator
    # owns the dict; legacy construction sites that pre-date stage 3
    # pass None and the shadow path falls back to a fresh dict
    # (acceptable for tests; production always provides it).
    shadow_learning_buffer_solar_per_unit: dict | None = None


# ---------------------------------------------------------------------------
# Per-unit learning strategies (#776)
# ---------------------------------------------------------------------------

@runtime_checkable
class LearningStrategy(Protocol):
    """Protocol for per-unit learning strategies.

    Each energy sensor is assigned a strategy that determines how its
    kWh contribution is computed for each hour during midnight learning.
    The global model sums all units' contributions per hour.
    """

    sensor_id: str

    def get_hourly_contribution(
        self,
        hour: int,
        weight: float,
        log_entry: dict,
    ) -> float | None:
        """Return kWh contribution for this unit at the given hour.

        Args:
            hour: Hour of day (0–23).
            weight: Normalised thermodynamic loss weight for this hour
                    (from ThermodynamicEngine).  Always provided, but
                    DirectMeter ignores it.
            log_entry: The hourly log dict for this hour (contains
                       unit_breakdown, temp_key, wind_bucket, etc.).

        Returns:
            kWh value to contribute to the global model for this hour,
            or None if this hour should be skipped for this unit.
        """
        ...


class DirectMeter:
    """Track A: use actual meter reading from unit_breakdown.

    For units with reliable per-hour electrical data (panel heaters,
    A2A heat pumps, any non-MPC unit).  The meter is ground truth —
    no smearing or COP correction needed.

    Mode filtering (#789): units in OFF/DHW/Guest/Cooling modes are
    excluded from the global model to match Track A semantics.
    """

    def __init__(self, sensor_id: str) -> None:
        self.sensor_id = sensor_id

    def get_hourly_contribution(
        self,
        hour: int,
        weight: float,
        log_entry: dict,
    ) -> float | None:
        # Mode filtering (#789): skip units in excluded modes.
        unit_modes = log_entry.get("unit_modes", {})
        mode = unit_modes.get(self.sensor_id, MODE_HEATING)
        if mode in MODES_EXCLUDED_FROM_GLOBAL_LEARNING:
            return None

        breakdown = log_entry.get("unit_breakdown", {})
        kwh = breakdown.get(self.sensor_id, 0.0)
        return kwh if kwh > 0.0 else None

    def __repr__(self) -> str:
        return f"DirectMeter({self.sensor_id!r})"


class WeightedSmear:
    """Track C / weighted smearing: distribute daily total using loss weights.

    Two sub-cases, controlled by ``use_synthetic``:

    - **MPC unit** (``use_synthetic=True``): The ThermodynamicEngine has
      already computed ``synthetic_kwh_el`` per hour (smeared thermal
      divided by per-hour COP).  The weight parameter is ignored; we
      read directly from the pre-computed distribution.

    - **Non-MPC smeared unit** (``use_synthetic=False``): e.g. floor
      heating with high thermal mass.  The unit's daily electrical total
      is distributed across 24 hours using the same thermodynamic weight
      vector.  COP = 1.0 for resistive elements, so no COP division.
    """

    def __init__(
        self,
        sensor_id: str,
        *,
        use_synthetic: bool = False,
    ) -> None:
        self.sensor_id = sensor_id
        self.use_synthetic = use_synthetic
        # Set at midnight by coordinator before the learning loop.
        self._distribution: dict[int, dict] | None = None
        self._daily_total: float = 0.0

    def set_distribution(self, distribution_by_hour: dict[int, dict] | None) -> None:
        """Provide the hour-indexed Track C distribution, or None to clear.

        Args:
            distribution_by_hour: Maps hour (0–23) → HourlyDistribution dict
                with at least ``synthetic_kwh_el`` key, or None to reset.
        """
        self._distribution = distribution_by_hour

    def set_daily_total(self, total: float) -> None:
        """Provide the raw daily electrical total for non-MPC smearing."""
        self._daily_total = total

    def get_hourly_contribution(
        self,
        hour: int,
        weight: float,
        log_entry: dict,
    ) -> float | None:
        # Defensive mode guard (#789): MPC data is already mode-filtered
        # by ThermodynamicEngine, but guard for consistency / edge cases.
        unit_modes = log_entry.get("unit_modes", {})
        mode = unit_modes.get(self.sensor_id, MODE_HEATING)
        if mode in MODES_EXCLUDED_FROM_GLOBAL_LEARNING:
            return None

        if self.use_synthetic:
            # MPC path: read pre-computed synthetic kWh from distribution.
            if not self._distribution or hour not in self._distribution:
                return None
            kwh = self._distribution[hour].get("synthetic_kwh_el", 0.0)
            return kwh if kwh > 0.0 else None
        else:
            # Non-MPC smeared path: daily_total × normalised weight.
            if self._daily_total <= 0.0 or weight <= 0.0:
                return None
            kwh = self._daily_total * weight
            return kwh if kwh > 0.0 else None

    def __repr__(self) -> str:
        mode = "synthetic" if self.use_synthetic else "weighted"
        return f"WeightedSmear({self.sensor_id!r}, {mode})"


def build_strategies(
    energy_sensors: list[str],
    track_c_enabled: bool,
    mpc_managed_sensor: str | None,
) -> dict[str, LearningStrategy]:
    """Build the strategy map from configuration.

    Default assignment rules:
    - If Track C is enabled and sensor == mpc_managed_sensor → WeightedSmear(use_synthetic=True)
    - All other sensors → DirectMeter

    Returns:
        Dict mapping sensor_id → strategy instance.
    """
    strategies: dict[str, LearningStrategy] = {}
    for sensor_id in energy_sensors:
        if track_c_enabled and mpc_managed_sensor and sensor_id == mpc_managed_sensor:
            strategies[sensor_id] = WeightedSmear(sensor_id, use_synthetic=True)
        else:
            strategies[sensor_id] = DirectMeter(sensor_id)
    return strategies


class ObservationCollector:
    """Owns all hour-scoped mutable accumulators.

    Replaces ~20 individual ``self._hourly_*`` / ``self._accumulated_*``
    fields on the coordinator with a single object whose ``reset()``
    method guarantees all counters are zeroed atomically — eliminating
    the class of restart bugs caused by missing one field in the manual
    reset block.

    Lifecycle:
        1. Per-minute tick → ``accumulate_weather()``, energy deltas
           written to ``delta_per_unit``, expected via ``accumulate_expected()``.
        2. Hour boundary → coordinator reads fields to build
           ``HourlyObservation``, then calls ``reset()``.
    """

    __slots__ = (
        "wind_sum",
        "wind_values",
        "temp_sum",
        "humidity_sum",
        "humidity_count",
        "solar_sum",
        "solar_vector_s_sum",
        "solar_vector_e_sum",
        "solar_vector_w_sum",
        "bucket_counts",
        "aux_count",
        "sample_count",
        "start_time",
        "energy_hour",
        "expected_energy_hour",
        "aux_impact_hour",
        "orphaned_aux",
        "aux_breakdown",
        "delta_per_unit",
        "expected_per_unit",
        "expected_base_per_unit",
        "correction_sum",
        "dni_sum",
        "dni_count",
        "dhi_sum",
        "dhi_count",
        "cloud_coverage_sum",
        "cloud_coverage_count",
        "ghi_sum",
        "ghi_count",
        "solar_data_time_last",
        "last_minute_processed",
    )

    def __init__(self) -> None:
        # Create containers ONCE — reset() clears them in-place so that
        # any external references (aliases) remain valid.
        self.wind_sum: float = 0.0
        self.wind_values: list[float] = []
        self.temp_sum: float = 0.0
        self.humidity_sum: float = 0.0
        self.humidity_count: int = 0
        self.solar_sum: float = 0.0
        self.solar_vector_s_sum: float = 0.0
        self.solar_vector_e_sum: float = 0.0
        self.solar_vector_w_sum: float = 0.0
        self.bucket_counts: dict[str, int] = {
            "normal": 0,
            "high_wind": 0,
            "extreme_wind": 0,
        }
        self.aux_count: int = 0
        self.sample_count: int = 0
        self.start_time: datetime | None = None
        self.energy_hour: float = 0.0
        self.expected_energy_hour: float = 0.0
        self.aux_impact_hour: float = 0.0
        self.orphaned_aux: float = 0.0
        self.aux_breakdown: dict[str, dict[str, float]] = {}
        self.delta_per_unit: dict[str, float] = {}
        self.expected_per_unit: dict[str, float] = {}
        self.expected_base_per_unit: dict[str, float] = {}
        self.correction_sum: float = 0.0
        # DNI / DHI from weather entity (issue #933).  Counted independently
        # of ``sample_count`` because the source may be unavailable on some
        # ticks (primary lacks the attribute, secondary missing, value not
        # yet refreshed) — averaging only over ticks that produced a value
        # avoids dragging the mean toward zero.  ``solar_data_time_last``
        # is the most recent ``solar_data_time`` attribute observed during
        # the hour (any source); persisted to the hourly log so we can
        # later detect stale-data hours without joining against the source
        # entity history.
        self.dni_sum: float = 0.0
        self.dni_count: int = 0
        self.dhi_sum: float = 0.0
        self.dhi_count: int = 0
        # Cloud_coverage logged per hour as the third arm of the
        # DNI/DHI investigation — lets future signal_agreement readings
        # compute Kasten directly from the source value rather than
        # inverting it out of potential_solar_factor.
        self.cloud_coverage_sum: float = 0.0
        self.cloud_coverage_count: int = 0
        # Optional local GHI sensor (#933 follow-up).  Accumulated only
        # on ticks where the configured sensor returned a value;
        # ``ghi_count == 0`` at hour boundary means the sensor is
        # unconfigured or was unavailable for the whole hour, in which
        # case ``ghi_wm2`` is omitted from the log entry.
        self.ghi_sum: float = 0.0
        self.ghi_count: int = 0
        self.solar_data_time_last: str | None = None
        self.last_minute_processed: int | None = None

    def reset(self) -> None:
        """Zero all hour-scoped accumulators. Called at hour boundary.

        Mutable containers are cleared IN-PLACE so that coordinator
        aliases (e.g. ``self._hourly_delta_per_unit``) remain valid.
        """
        self.wind_sum = 0.0
        self.wind_values.clear()
        self.temp_sum = 0.0
        self.humidity_sum = 0.0
        self.humidity_count = 0
        self.solar_sum = 0.0
        self.solar_vector_s_sum = 0.0
        self.solar_vector_e_sum = 0.0
        self.solar_vector_w_sum = 0.0
        for k in self.bucket_counts:
            self.bucket_counts[k] = 0
        self.aux_count = 0
        self.sample_count = 0
        self.start_time = None
        self.energy_hour = 0.0
        self.expected_energy_hour = 0.0
        self.aux_impact_hour = 0.0
        self.orphaned_aux = 0.0
        self.aux_breakdown.clear()
        self.delta_per_unit.clear()
        self.expected_per_unit.clear()
        self.expected_base_per_unit.clear()
        self.correction_sum = 0.0
        self.dni_sum = 0.0
        self.dni_count = 0
        self.dhi_sum = 0.0
        self.dhi_count = 0
        self.cloud_coverage_sum = 0.0
        self.cloud_coverage_count = 0
        self.ghi_sum = 0.0
        self.ghi_count = 0
        self.solar_data_time_last = None
        self.last_minute_processed = None

    def accumulate_weather(
        self,
        temp: float,
        effective_wind: float,
        wind_bucket: str,
        solar_factor: float,
        solar_vector: tuple[float, float, float],
        is_aux_active: bool,
        current_time: datetime,
        humidity: float | None = None,
        correction_percent: float = 100.0,
        dni: float | None = None,
        dhi: float | None = None,
        cloud_coverage: float | None = None,
        ghi: float | None = None,
        solar_data_time: str | None = None,
    ) -> None:
        """Record one minute's weather readings."""
        self.wind_sum += effective_wind
        self.wind_values.append(effective_wind)
        self.temp_sum += temp
        if humidity is not None:
            self.humidity_sum += humidity
            self.humidity_count += 1
        self.solar_sum += solar_factor
        self.solar_vector_s_sum += solar_vector[0]
        self.solar_vector_e_sum += solar_vector[1]
        self.solar_vector_w_sum += solar_vector[2]
        self.correction_sum += correction_percent
        if dni is not None:
            self.dni_sum += dni
            self.dni_count += 1
        if dhi is not None:
            self.dhi_sum += dhi
            self.dhi_count += 1
        if cloud_coverage is not None:
            self.cloud_coverage_sum += cloud_coverage
            self.cloud_coverage_count += 1
        if ghi is not None:
            self.ghi_sum += ghi
            self.ghi_count += 1
        if solar_data_time is not None:
            self.solar_data_time_last = solar_data_time
        self.bucket_counts[wind_bucket] += 1
        if is_aux_active:
            self.aux_count += 1
        self.sample_count += 1

        if self.start_time is None:
            self.start_time = current_time.replace(
                minute=0, second=0, microsecond=0
            )

    def accumulate_expected(
        self,
        fraction: float,
        prediction_rate: float,
        aux_impact_rate: float,
        unit_breakdown: dict,
        orphaned_part: float,
    ) -> None:
        """Accumulate expected energy and aux impact for one time step.

        Args:
            fraction: Time step as fraction of an hour (minutes_step / 60).
            prediction_rate: Current prediction rate (kWh/h).
            aux_impact_rate: Current aux impact rate (kWh/h).
            unit_breakdown: Per-unit prediction breakdown from statistics.
            orphaned_part: Unattributed aux savings rate.
        """
        if prediction_rate > 0:
            self.expected_energy_hour += prediction_rate * fraction

        if aux_impact_rate > 0:
            self.aux_impact_hour += aux_impact_rate * fraction

        for entity_id, stats in unit_breakdown.items():
            unit_pred = stats["net_kwh"]
            if unit_pred > 0:
                if entity_id not in self.expected_per_unit:
                    self.expected_per_unit[entity_id] = 0.0
                self.expected_per_unit[entity_id] += unit_pred * fraction

            unit_base = stats.get("base_kwh", 0.0)
            if unit_base > 0:
                if entity_id not in self.expected_base_per_unit:
                    self.expected_base_per_unit[entity_id] = 0.0
                self.expected_base_per_unit[entity_id] += unit_base * fraction

            if entity_id not in self.aux_breakdown:
                self.aux_breakdown[entity_id] = {
                    "allocated": 0.0,
                    "overflow": 0.0,
                }
            applied_aux = stats.get("aux_reduction_kwh", 0.0)
            overflow_aux = stats.get("overflow_kwh", 0.0)
            self.aux_breakdown[entity_id]["allocated"] += applied_aux * fraction
            self.aux_breakdown[entity_id]["overflow"] += overflow_aux * fraction

        if orphaned_part > 0:
            self.orphaned_aux += orphaned_part * fraction
