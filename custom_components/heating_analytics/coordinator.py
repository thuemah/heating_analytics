"""Coordinator for Heating Analytics."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, date
import math

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import (
    DataUpdateCoordinator,
)
from homeassistant.util import dt as dt_util
from homeassistant.const import UnitOfSpeed

from .helpers import convert_speed_to_ms, generate_exponential_kernel
from .solar import SolarCalculator
from .forecast import ForecastManager
from .statistics import StatisticsManager
from .learning import LearningManager, compute_snr_weight, count_active_learnable_units
from .daily_processor import DailyProcessor
from .diagnostics import DiagnosticsEngine
from .hourly_processor import HourlyProcessor
from .retrain import RetrainEngine
from .storage import StorageManager
from .solar_optimizer import SolarOptimizer
from .const import (
    DOMAIN,
    ATTR_EFFICIENCY,
    ATTR_PREDICTED,
    ATTR_SOLAR_PREDICTED,
    ATTR_DEVIATION,
    ATTR_TDD,
    ATTR_FORECAST_TODAY,
    ATTR_CORRELATION_DATA,
    ATTR_LAST_HOUR_ACTUAL,
    ATTR_LAST_HOUR_EXPECTED,
    ATTR_LAST_HOUR_DEVIATION,
    ATTR_LAST_HOUR_DEVIATION_PCT,
    ATTR_POTENTIAL_SAVINGS,
    ATTR_ENERGY_TODAY,
    ATTR_EXPECTED_TODAY,
    ATTR_TDD_YESTERDAY,
    ATTR_TDD_LAST_7D,
    ATTR_TDD_LAST_30D,
    ATTR_EFFICIENCY_YESTERDAY,
    ATTR_EFFICIENCY_LAST_7D,
    ATTR_EFFICIENCY_LAST_30D,
    ATTR_EFFICIENCY_FORECAST_TODAY,
    ATTR_TDD_DAILY_STABLE,
    ATTR_TEMP_LAST_YEAR_DAY,
    ATTR_TEMP_LAST_YEAR_WEEK,
    ATTR_TEMP_LAST_YEAR_MONTH,
    ATTR_TEMP_FORECAST_TODAY,
    ATTR_TEMP_ACTUAL_TODAY,
    ATTR_TEMP_ACTUAL_WEEK,
    ATTR_TEMP_ACTUAL_MONTH,
    ATTR_WIND_LAST_YEAR_DAY,
    ATTR_WIND_LAST_YEAR_WEEK,
    ATTR_WIND_LAST_YEAR_MONTH,
    ATTR_WIND_ACTUAL_TODAY,
    ATTR_WIND_ACTUAL_WEEK,
    ATTR_WIND_ACTUAL_MONTH,
    ATTR_SOLAR_FACTOR,
    ATTR_SOLAR_IMPACT,
    ATTR_TDD_SO_FAR,
    ATTR_MIDNIGHT_FORECAST,
    ATTR_FORECAST_UNCERTAINTY,
    ATTR_DEVIATION_BREAKDOWN,
    ATTR_FORECAST_BLEND_CONFIG,
    ATTR_FORECAST_ACCURACY_BY_SOURCE,
    ATTR_FORECAST_DETAILS,
    ATTR_SOLAR_POTENTIAL,
    ATTR_SOLAR_GAIN_NOW,
    ATTR_HEATING_LOAD_OFFSET,
    ATTR_RECOMMENDATION_STATE,
    RECOMMENDATION_MAXIMIZE_SOLAR,
    RECOMMENDATION_INSULATE,
    RECOMMENDATION_MITIGATE_SOLAR,
    DEFAULT_WIND_THRESHOLD,
    DEFAULT_WIND_GUST_FACTOR,
    DEFAULT_EXTREME_WIND_THRESHOLD,
    DEFAULT_WIND_UNIT,
    DEFAULT_MAX_ENERGY_DELTA,
    CONF_WIND_UNIT,
    CONF_ENABLE_LIFETIME_TRACKING,
    CONF_SOLAR_ENABLED,
    CONF_SOLAR_AZIMUTH,
    CONF_SCREEN_SOUTH,
    CONF_SCREEN_EAST,
    CONF_SCREEN_WEST,
    CONF_SCREEN_AFFECTED_ENTITIES,
    DEFAULT_SOLAR_ENABLED,
    DEFAULT_SOLAR_AZIMUTH,
    DEFAULT_SOLAR_CORRECTION,
    DEFAULT_SCREEN_SOUTH,
    DEFAULT_SCREEN_EAST,
    DEFAULT_SCREEN_WEST,
    CONF_AUX_AFFECTED_ENTITIES,
    DEFAULT_SOLAR_LEARNING_RATE,
    DEFAULT_SOLAR_COEFF_HEATING,
    DEFAULT_SOLAR_COEFF_COOLING,
    ENERGY_GUARD_THRESHOLD,
    CLOUD_COVERAGE_MAP,
    WIND_UNIT_MS,
    WIND_UNIT_KMH,
    WIND_UNIT_KNOTS,
    LEARNING_BUFFER_THRESHOLD,
    CONF_OUTDOOR_TEMP_SOURCE,
    CONF_WIND_SOURCE,
    CONF_WIND_GUST_SOURCE,
    CONF_SECONDARY_WEATHER_ENTITY,
    CONF_FORECAST_CROSSOVER_DAY,
    DEFAULT_FORECAST_CROSSOVER_DAY,
    CONF_THERMAL_INERTIA,
    DEFAULT_THERMAL_INERTIA_HOURS,
    CONF_INDOOR_TEMP_SENSOR,
    CONF_THERMAL_MASS,
    DEFAULT_THERMAL_MASS,
    DEFAULT_DAILY_LEARNING_RATE,
    CONF_DAILY_LEARNING_MODE,
    CONF_TRACK_C,
    CONF_MPC_ENTRY_ID,
    CONF_MPC_MANAGED_SENSOR,
    SOURCE_SENSOR,
    SOURCE_WEATHER,
    COOLING_WIND_BUCKET,
    MODE_HEATING,
    MODE_COOLING,
    MODE_OFF,
    MODE_GUEST_HEATING,
    MODE_GUEST_COOLING,
    MODE_DHW,
    MODES_EXCLUDED_FROM_GLOBAL_LEARNING,
    SNR_WEIGHT_FLOOR,
    SNR_WEIGHT_K,
    DEFAULT_CSV_AUTO_LOGGING,
    DEFAULT_CSV_HOURLY_PATH,
    DEFAULT_CSV_DAILY_PATH,
    DEFAULT_CLOUD_COVERAGE,
    MIXED_MODE_LOW,
    MIXED_MODE_HIGH,
    DUAL_INTERFERENCE_THRESHOLD,
    COOLDOWN_MIN_HOURS,
    COOLDOWN_MAX_HOURS,
    COOLDOWN_CONVERGENCE_THRESHOLD,
    SOLAR_BATTERY_DECAY,
    CONF_BATTERY_THERMAL_FEEDBACK_K,
    DEFAULT_BATTERY_THERMAL_FEEDBACK_K,
)

_LOGGER = logging.getLogger(__name__)

class HeatingDataCoordinator(DataUpdateCoordinator):
    """Class to manage fetching data from the API."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize."""
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=timedelta(minutes=1),
        )
        self.entry = entry

        # Internal state
        self._correlation_data = {} # { "temp": { "wind_bucket": avg_kwh_per_hour } }
        self._correlation_data_per_unit = {} # { entity_id: { "temp": { "wind_bucket": avg_kwh_per_hour } } }
        # { entity_id: { "temp_key": { "wind_bucket": count } } }
        self._observation_counts = {}
        self._aux_coefficients = {} # { "temp": kw_reduction }
        self._aux_coefficients_per_unit = {} # { entity_id: { "temp": { "wind_bucket": kw_reduction } } }

        self._unit_modes = {} # { entity_id: MODE_HEATING/MODE_COOLING/MODE_OFF }

        # Mode-stratified per-unit solar coefficients (#868):
        # {entity_id: {"heating": {s, e, w}, "cooling": {s, e, w}}}
        self._solar_coefficients_per_unit = {}

        # Per-unit min-base thresholds (#871).  Populated from dark-hour
        # p10 by :meth:`_calibrate_per_unit_min_base_thresholds` at startup
        # and via the ``calibrate_unit_thresholds`` service.  Empty dict
        # means all gate sites (NLMS / inequality / shutdown) fall back
        # to the global SOLAR_*_MIN_BASE constants (0.15).  Survives
        # ``reset_solar_learning`` — these are hardware noise floors, not
        # learned coefficients.
        self._per_unit_min_base_thresholds: dict[str, float] = {}

        # Last batch-fit-solar summary per unit (#884).  Populated by
        # ``async_batch_fit_solar``; consumed by ``diagnose_solar`` for
        # observability.  In-memory only — restart clears it.  The
        # actual coefficient updates land in
        # ``_solar_coefficients_per_unit`` and persist via storage.
        self._last_batch_fit_per_unit: dict[str, dict] = {}

        # Solar thermal battery: accumulates solar impact across hours with exponential decay.
        # Carries residual solar heat (stored in building mass) into post-solar hours.
        # Reset to 0 on restart — recovers within a few hours.
        self._solar_battery_state: float = 0.0

        # Solar carry-over reservoir (#896 follow-up).  Parallel to
        # ``_solar_battery_state`` but charged ONLY from
        # ``k × solar_heating_wasted`` (no applied-solar term), so its
        # value represents "smoothed reservoir of saturation-clipped
        # energy still residing in thermal mass" rather than the
        # applied-solar effective impact.  Used exclusively for the
        # release-into-prediction subtraction in
        # ``statistics.calculate_total_power``.  Splitting state from
        # ``_solar_battery_state`` preserves all existing consumers
        # (display sensors, ``thermodynamic_gross_kwh``, aux-learning
        # input via ``effective_solar_impact``) — they continue to read
        # the unchanged scalar.  When ``battery_thermal_feedback_k == 0.0``
        # this state stays at 0.0 and the release subtraction is a
        # bit-identical no-op.  Same decay as ``_solar_battery_state``
        # for physical consistency.  Persisted via storage so a restart
        # doesn't reset the carry-over signal during a sunny spell.
        self._solar_carryover_state: float = 0.0

        # Per-direction potential battery (#865).  Separate from
        # ``_solar_battery_state`` (which accumulates post-coefficient
        # solar impact for display); these accumulate the **pre-coefficient**
        # potential vector per cardinal direction.  Fed to the inequality
        # learner which needs to see accumulated sun presence, not
        # instantaneous hour magnitude, because shutdown is an accumulated
        # thermal-mass phenomenon.  Same decay (``solar_battery_decay``,
        # default 0.80) for physical consistency — the building's thermal
        # mass responds to sun on the same timescale regardless of which
        # downstream consumer is looking at it.  Persisted via storage so
        # a restart doesn't reset the signal the inequality learner relies
        # on for its first few shutdown hours.
        self._potential_battery_s: float = 0.0
        self._potential_battery_e: float = 0.0
        self._potential_battery_w: float = 0.0

        # Hour-scoped accumulators — delegated to ObservationCollector (#775).
        # The collector owns all hourly state; the coordinator keeps
        # backward-compatible aliases that point to the SAME dict objects.
        # ObservationCollector.reset() clears containers in-place, so
        # these aliases remain valid across hour boundaries.
        from .observation import ObservationCollector
        self._collector = ObservationCollector()

        self._hourly_delta_per_unit = self._collector.delta_per_unit
        self._hourly_expected_per_unit = self._collector.expected_per_unit
        self._hourly_expected_base_per_unit = self._collector.expected_base_per_unit

        self._learning_buffer_per_unit = {} # { entity_id: { "temp": { "wind_bucket": [normalized_values] } } }
        self._learning_buffer_aux_per_unit = {} # { entity_id: { "temp": { "wind_bucket": [reduction_values] } } }

        # Mode-stratified per-unit solar cold-start buffer (#868):
        # {entity_id: {"heating": [(s, e, w, impact), ...], "cooling": [...]}}
        self._learning_buffer_solar_per_unit = {}

        self._learning_buffer_global = {} # { "temp": { "wind_bucket": [normalized_values] } }

        self._collector.last_minute_processed = None

        self.data = {
            ATTR_EFFICIENCY: 0.0,
            ATTR_PREDICTED: 0.0,
            ATTR_DEVIATION: 0.0,
            ATTR_TDD: 0.0,
            ATTR_FORECAST_TODAY: 0.0,
            ATTR_LAST_HOUR_ACTUAL: 0.0,
            ATTR_LAST_HOUR_EXPECTED: 0.0,
            ATTR_LAST_HOUR_DEVIATION: 0.0,
            ATTR_LAST_HOUR_DEVIATION_PCT: 0.0,
            ATTR_POTENTIAL_SAVINGS: 0.0,
            ATTR_ENERGY_TODAY: 0.0,
            ATTR_EXPECTED_TODAY: 0.0,
            ATTR_CORRELATION_DATA: self._correlation_data,  # Direct reference
            ATTR_TDD_YESTERDAY: None,
            ATTR_TDD_LAST_7D: None,
            ATTR_TDD_LAST_30D: None,
            ATTR_EFFICIENCY_YESTERDAY: None,
            ATTR_EFFICIENCY_LAST_7D: None,
            ATTR_EFFICIENCY_LAST_30D: None,
            ATTR_EFFICIENCY_FORECAST_TODAY: None,
            ATTR_TDD_DAILY_STABLE: 0.0,
            ATTR_TEMP_LAST_YEAR_DAY: None,
            ATTR_TEMP_LAST_YEAR_WEEK: None,
            ATTR_TEMP_LAST_YEAR_MONTH: None,
            ATTR_TEMP_FORECAST_TODAY: None,
            ATTR_TEMP_ACTUAL_TODAY: None,
            ATTR_TEMP_ACTUAL_WEEK: None,
            ATTR_TEMP_ACTUAL_MONTH: None,
            ATTR_WIND_LAST_YEAR_DAY: None,
            ATTR_WIND_LAST_YEAR_WEEK: None,
            ATTR_WIND_LAST_YEAR_MONTH: None,
            ATTR_WIND_ACTUAL_TODAY: None,
            ATTR_WIND_ACTUAL_WEEK: None,
            ATTR_WIND_ACTUAL_MONTH: None,
            ATTR_SOLAR_FACTOR: 0.0,
            ATTR_SOLAR_IMPACT: 0.0,
            "accumulated_solar_impact_kwh": 0.0,
            "accumulated_guest_impact_kwh": 0.0,
            "accumulated_aux_impact_kwh": 0.0,
            ATTR_TDD_SO_FAR: 0.0,
            ATTR_DEVIATION_BREAKDOWN: [],
            "hourly_log": [],
            "daily_individual": {},
            "savings_theory_normal": 0.0,
            "savings_theory_aux": 0.0,
            "missing_aux_data": False,
        }
        self._daily_history = {} # { "YYYY-MM-DD": { "kwh": float, "temp": float, "tdd": float } }
        self._daily_individual = {} # { entity_id: kwh_today }
        self._lifetime_individual = {} # { entity_id: kwh_lifetime }
        self._hourly_log = [] # List of dicts for hourly stats
        self._last_hour_processed = None
        self._last_day_processed = None
        self._last_energy_values = {} # { entity_id: value_at_last_update }
        self._learned_u_coefficient = None
        self._last_midnight_indoor_temp = None
        # #855 Option B: count days where Track C was enabled but MPC did not
        # produce a distribution, so the midnight sync skipped bucket + U
        # updates to avoid mixing MPC-synthetic and raw-electrical semantics.
        # Runtime counter — resets on HA restart by design (diagnostic only).
        self._track_c_outage_count_session = 0

        # Track C pre-midnight snapshot (#855 follow-up).  The midnight sync
        # runs at 00:xx of the new day and processes yesterday's data.  If
        # MPC happens to be unavailable at exactly that moment (HA startup
        # race, service loading, busy kernel), we lose the entire day.  We
        # pre-fetch the MPC buffer at 22:00, 23:00, and 23:55 the evening
        # before; the midnight sync falls back to the most recent snapshot
        # when the live call fails.  Transient state — not persisted; a
        # restart between 23:55 and 00:01 loses the snapshot and falls
        # through to the Option B skip (acceptable edge case).
        self._track_c_snapshot: dict | None = None
        self._track_c_last_snapshot_slot: str | None = None

        # Hourly aggregates are now owned by ObservationCollector (see above).
        # Only daily-scoped accumulators remain here.
        self._accumulated_energy_today = 0.0
        self._daily_aux_breakdown = {} # { entity_id: { "allocated": 0.0, "overflow": 0.0 } } - Daily accumulator
        self._daily_orphaned_aux = 0.0
        self._accumulation_start_time = None

        # Configuration parameters
        self.outdoor_temp_sensor = entry.data.get("outdoor_temp_sensor")
        self.indoor_temp_sensor = entry.data.get(CONF_INDOOR_TEMP_SENSOR)
        self.thermal_mass_kwh_per_degree = entry.data.get(CONF_THERMAL_MASS, DEFAULT_THERMAL_MASS)
        self.daily_learning_mode = entry.data.get(CONF_DAILY_LEARNING_MODE, False)
        self.track_c_enabled = entry.data.get(CONF_TRACK_C, False)
        self.mpc_entry_id = entry.data.get(CONF_MPC_ENTRY_ID)
        self.mpc_managed_sensor = entry.data.get(CONF_MPC_MANAGED_SENSOR)
        self.wind_speed_sensor = entry.data.get("wind_speed_sensor")
        self.wind_gust_sensor = entry.data.get("wind_gust_sensor")

        # Source Configuration (Default to SENSOR for backward compatibility if key missing,
        # though migration should handle it)
        self.outdoor_temp_source = entry.data.get(CONF_OUTDOOR_TEMP_SOURCE, SOURCE_SENSOR)
        self.wind_speed_source = entry.data.get(CONF_WIND_SOURCE, SOURCE_SENSOR)
        self.wind_gust_source = entry.data.get(CONF_WIND_GUST_SOURCE, SOURCE_SENSOR)

        self.weather_entity = entry.data.get("weather_entity")
        self.energy_sensors = entry.data.get("energy_sensors", [])

        # Per-unit learning strategies (#776) — must be after energy_sensors,
        # track_c_enabled, and mpc_managed_sensor are set.
        from .observation import build_strategies
        self._unit_strategies = build_strategies(
            energy_sensors=self.energy_sensors,
            track_c_enabled=self.track_c_enabled,
            mpc_managed_sensor=self.mpc_managed_sensor,
        )

        self.solar_battery_decay = entry.data.get("solar_battery_decay", SOLAR_BATTERY_DECAY)
        # Saturation-wasted thermal-feedback (#896).  The Advanced Options UI
        # was retired in the same release that documented this code path —
        # joint sweep evidence showed empirical optimum k = 0 across the
        # decay × k grid, so the feature provides no measurable benefit
        # while users could still tune coefficients via the slider.
        #
        # Stale-key cleanup: any installation that set k > 0 via the
        # pre-1.3.5 UI carries the value forward in entry.data even after
        # the slider is gone, with no way for the user to reset it.
        # Remove the key on next coordinator init so the default (0.0)
        # takes over uniformly.  One-time silent migration; no warning
        # log because the user has no control to act on it.
        if CONF_BATTERY_THERMAL_FEEDBACK_K in entry.data:
            cleaned_data = {
                k: v for k, v in entry.data.items()
                if k != CONF_BATTERY_THERMAL_FEEDBACK_K
            }
            hass.config_entries.async_update_entry(entry, data=cleaned_data)
        self.battery_thermal_feedback_k = DEFAULT_BATTERY_THERMAL_FEEDBACK_K
        self.wind_gust_factor = entry.data.get("wind_gust_factor", DEFAULT_WIND_GUST_FACTOR)
        self.balance_point = entry.data.get("balance_point", 17.0)
        self.learning_rate = entry.data.get("learning_rate", 0.01)

        # New Config Params
        self.wind_threshold = entry.data.get("wind_threshold", DEFAULT_WIND_THRESHOLD)
        self.extreme_wind_threshold = entry.data.get("extreme_wind_threshold", DEFAULT_EXTREME_WIND_THRESHOLD)
        self.wind_unit = entry.data.get(CONF_WIND_UNIT, DEFAULT_WIND_UNIT)
        self.max_energy_delta = entry.data.get("max_energy_delta", DEFAULT_MAX_ENERGY_DELTA)
        self.enable_lifetime_tracking = entry.data.get(CONF_ENABLE_LIFETIME_TRACKING, False)

        # Hourly log retention (#820): configurable via config flow
        from .const import CONF_HOURLY_LOG_RETENTION_DAYS, DEFAULT_HOURLY_LOG_RETENTION_DAYS
        retention_days = entry.data.get(CONF_HOURLY_LOG_RETENTION_DAYS, DEFAULT_HOURLY_LOG_RETENTION_DAYS)
        self._hourly_log_max_entries = int(retention_days) * 24

        # Solar Config
        self.solar_enabled = entry.data.get(CONF_SOLAR_ENABLED, DEFAULT_SOLAR_ENABLED)
        self.solar_azimuth = entry.data.get(CONF_SOLAR_AZIMUTH, DEFAULT_SOLAR_AZIMUTH)
        self.solar_correction_percent = DEFAULT_SOLAR_CORRECTION
        # Per-direction screen presence (#826).  Tuple aligned with the 3D
        # solar vector ordering (south, east, west).  Default True for all
        # three on upgrade so legacy installations keep behaviour close to
        # the pre-1.3.3 single-floor model (with the floor itself raised
        # 0.20 → 0.30).  Users can uncheck a direction in the config flow
        # if a facade has no external screens; that direction's
        # transmittance then stays at 1.0 regardless of the slider.
        self.screen_config: tuple[bool, bool, bool] = (
            bool(entry.data.get(CONF_SCREEN_SOUTH, DEFAULT_SCREEN_SOUTH)),
            bool(entry.data.get(CONF_SCREEN_EAST, DEFAULT_SCREEN_EAST)),
            bool(entry.data.get(CONF_SCREEN_WEST, DEFAULT_SCREEN_WEST)),
        )

        # Per-entity scope for the installation-level screen_config.  Entities
        # in this set see the global screen_config at learn/predict time;
        # others are treated as screen-independent (transmittance=1.0).  Empty
        # / missing list defaults to all energy_sensors — preserves legacy
        # behaviour on upgrade.
        _screen_affected = entry.data.get(CONF_SCREEN_AFFECTED_ENTITIES)
        if _screen_affected is None:
            _screen_affected = list(self.energy_sensors)
        self._screen_affected_set: frozenset[str] = frozenset(_screen_affected)

        # Load Thermal Inertia Profile (Default to Normal/4h if missing)
        inertia_setting = self.entry.data.get(CONF_THERMAL_INERTIA, DEFAULT_THERMAL_INERTIA_HOURS)

        # Migration for old string values
        if isinstance(inertia_setting, str):
            if inertia_setting == "fast":
                inertia_setting = 2
            elif inertia_setting == "slow":
                inertia_setting = 12
            else:
                inertia_setting = 4

        # Store tau for use in gap detection (_get_inertia_parameters)
        self.inertia_tau = float(inertia_setting)
        # Generate weights dynamically; cap window at 5×tau (captures 99.3% of weight)
        # to avoid requesting hundreds of hours of history for large tau values.
        self.inertia_weights = generate_exponential_kernel(
            tau=self.inertia_tau,
            window_hours=min(int(inertia_setting * 5), 168),
        )

        # Load Aux Affected Entities (Default to all energy sensors if missing)
        self.aux_affected_entities = self.entry.data.get(CONF_AUX_AFFECTED_ENTITIES)
        if self.aux_affected_entities is None:
            # Backward compatibility: Assume all units are affected if not configured
            # Note: We respect explicit empty list [] as "No Units Affected"
            self.aux_affected_entities = self.energy_sensors

        # Optimization: Maintain a set for O(1) lookups in high-frequency loops
        self._aux_affected_set = set(self.aux_affected_entities) if self.aux_affected_entities else set()

        self.solar = SolarCalculator(self)
        self.solar_optimizer = SolarOptimizer(self)
        self.forecast = ForecastManager(self)
        self.statistics = StatisticsManager(self)
        self.learning = LearningManager()
        self._diagnostics = DiagnosticsEngine(self)
        self._hourly_processor = HourlyProcessor(self)
        self._daily_processor = DailyProcessor(self)
        self._retrain = RetrainEngine(self)
        self.storage = StorageManager(self)

        # CSV Auto-logging configuration
        self.csv_auto_logging = entry.data.get("csv_auto_logging", DEFAULT_CSV_AUTO_LOGGING)
        self.csv_hourly_path = entry.data.get("csv_hourly_path", DEFAULT_CSV_HOURLY_PATH)
        self.csv_daily_path = entry.data.get("csv_daily_path", DEFAULT_CSV_DAILY_PATH)

        # Dynamic State Flags
        self.auxiliary_heating_active = False
        self.learning_enabled = True
        self._aux_cooldown_active = False
        self._aux_cooldown_start_time = None

        self._is_loaded = False

    async def _async_load_data(self):
        """Load data from storage."""
        await self.storage.async_load_data()
        updated_count = self._backfill_daily_from_hourly()
        if updated_count > 0:
            _LOGGER.info(f"Backfilled/Enriched {updated_count} daily history entries. Saving data.")
            await self._async_save_data(force=True)

        # Per-unit min-base calibration.  Runs opportunistically on
        # every startup once ≥14 days of log data exist.  Rate-of-change
        # clamp (±50 %) protects against a single anomalous week.  Results
        # persisted next save; no-op when data is insufficient.
        # Defensive: a malformed log entry (corrupt storage, manual edit)
        # must not block integration load — calibration is recoverable
        # next startup or via the ``calibrate_unit_thresholds`` service.
        try:
            cal_result = self._calibrate_per_unit_min_base_thresholds()
            if cal_result.get("status") == "ok" and (
                cal_result.get("updated") or cal_result.get("rejected")
            ):
                await self._async_save_data(force=True)
        except Exception as e:  # noqa: BLE001 — one-shot startup calibration; log-and-skip beats blocking boot
            _LOGGER.warning(
                "Per-unit min-base calibration failed at startup: %s. "
                "Falling back to global SOLAR_*_MIN_BASE constants. "
                "Run the 'calibrate_unit_thresholds' service after "
                "investigating the log data.", e,
            )

    async def _async_save_data(self, force: bool = False):
        """Save data to storage."""
        await self.storage.async_save_data(force)

    async def async_reset_learning_data(self):
        """Reset the learning data (correlation model) and refresh all sensors."""
        await self.storage.async_reset_learning_data()
        await self.async_refresh()

    async def async_reset_forecast_accuracy(self):
        """Reset the forecast accuracy history."""
        self.forecast.reset_forecast_history()
        # Save explicitly to persist the empty state (prevents backfill on reload)
        await self._async_save_data(force=True)

    async def set_auxiliary_heating_active(self, active: bool):
        """Set auxiliary heating active state and manage cooldown transitions."""
        if self.auxiliary_heating_active == active:
            return

        # Detect Transition: True -> False (Aux Turned Off)
        if self.auxiliary_heating_active and not active:
            _LOGGER.info("Auxiliary Heating turned OFF. Initiating Model Cooldown/Decay Lock.")
            self._aux_cooldown_active = True
            self._aux_cooldown_start_time = dt_util.now()

        # Update State
        if active and self._aux_cooldown_active:
            _LOGGER.info("Aux re-activated during cooldown. Cancelling cooldown.")
            self._aux_cooldown_active = False
            self._aux_cooldown_start_time = None

        self.auxiliary_heating_active = active
        await self._async_save_data()

        # Trigger immediate UI update
        self.async_set_updated_data(self.data)

    async def async_exit_cooldown(self):
        """Manually exit auxiliary cooldown state."""
        if not self._aux_cooldown_active:
            _LOGGER.info("exit_cooldown called but no active cooldown found - ignoring")
            return
        _LOGGER.info("Manually exiting auxiliary cooldown")
        self._aux_cooldown_active = False
        self._aux_cooldown_start_time = None
        await self._async_save_data()
        self.async_set_updated_data(self.data)

    def get_unit_mode(self, entity_id: str) -> str:
        """Get current mode for a unit."""
        return self._unit_modes.get(entity_id, MODE_HEATING)

    def screen_config_for_entity(self, entity_id: str) -> tuple[bool, bool, bool]:
        """Return the effective screen_config for this entity.

        Entities in ``_screen_affected_set`` see the installation-level
        ``screen_config``; others get ``(False, False, False)`` so their
        solar coefficients learn and predict against pure transmittance=1.0
        regardless of the global ``solar_correction_percent`` slider.

        Use this at per-entity learning and prediction call sites.  Global
        state (potential battery, solar_factor, correlation_data base) keeps
        using ``self.screen_config`` directly — those represent the
        installation-wide signal and do not split per unit.
        """
        if entity_id in self._screen_affected_set:
            return self.screen_config
        return (False, False, False)

    async def async_set_unit_mode(self, entity_id: str, mode: str):
        """Set mode for a unit."""
        if entity_id not in self.energy_sensors:
            _LOGGER.warning(f"Attempted to set mode for unknown entity: {entity_id}")
            return

        self._unit_modes[entity_id] = mode
        await self._async_save_data()

        self.async_set_updated_data(self.data)

    async def async_reset_unit_learning_data(self, entity_id: str):
        """Reset learning data for a specific unit."""
        if not entity_id:
            _LOGGER.warning("Attempted to reset unit learning with empty entity_id")
            return

        _LOGGER.info(f"Resetting learning data for unit: {entity_id}")

        # Clear Model Data
        if entity_id in self._correlation_data_per_unit:
            del self._correlation_data_per_unit[entity_id]
            _LOGGER.debug(f"Cleared correlation data for {entity_id}")

        # Clear Unit Aux Data
        if entity_id in self._aux_coefficients_per_unit:
            del self._aux_coefficients_per_unit[entity_id]
            _LOGGER.debug(f"Cleared aux coefficients for {entity_id}")

        # Clear Unit Solar Data
        if entity_id in self._solar_coefficients_per_unit:
            del self._solar_coefficients_per_unit[entity_id]
            _LOGGER.debug(f"Cleared solar coefficients for {entity_id}")

        # Clear Observation Counts
        if entity_id in self._observation_counts:
            del self._observation_counts[entity_id]
            _LOGGER.debug(f"Cleared observation counts for {entity_id}")

        # Note: We do NOT clear _learning_buffer_per_unit, _learning_buffer_aux_per_unit, or _learning_buffer_solar_per_unit.
        # This allows any pre-existing buffer to be applied during the next "Cold Start" cycle.
        # If the user wanted to clear the buffer, they would need a separate action,
        # but the request is specifically "so that the buffered learning can be reapplied".

        await self._async_save_data(force=True)

    async def async_reset_solar_learning_data(
        self,
        entity_id: str | None = None,
        replay_from_history: bool = False,
        days_back: int | None = None,
    ) -> dict:
        """Reset solar learning data for a specific unit or all units.

        When ``replay_from_history`` is True, the reset is followed by an
        NLMS replay over the stored hourly log so coefficients are refitted
        against the existing base model. This is the "solar-only retrain"
        path — base/aux/U-coefficient tables are never touched, only solar
        coefficients and their cold-start buffers. Useful after the base
        model has stabilised but the solar coefficients are known to be
        wrong (e.g. screen-config change, or divergence detected in
        ``diagnose_solar``).

        When ``entity_id`` is given, both the reset and the replay are
        scoped to that unit — other units' coefficients are preserved.
        The per-unit filter is applied by passing a single-element
        ``energy_sensors`` list to :meth:`LearningManager.replay_solar_nlms`.

        Returns a dict describing what happened (always — SupportsResponse.ONLY):

        - ``status``: "reset"
        - ``unit_entity_id``: the filter (or None for all units)
        - ``replay_from_history``: bool
        - ``solar_replay_diagnostics``: dict from ``replay_solar_nlms`` or None
        """
        if entity_id:
            if entity_id in self._solar_coefficients_per_unit:
                del self._solar_coefficients_per_unit[entity_id]
                _LOGGER.debug(f"Cleared solar coefficients for {entity_id}")
            if entity_id in self._learning_buffer_solar_per_unit:
                del self._learning_buffer_solar_per_unit[entity_id]
                _LOGGER.debug(f"Cleared solar learning buffer for {entity_id}")
            # Clear the unit's last batch-fit summary — its before/after
            # snapshot no longer matches the (now-empty) coefficient state.
            if entity_id in self._last_batch_fit_per_unit:
                del self._last_batch_fit_per_unit[entity_id]
            _LOGGER.info(f"Solar learning reset for unit: {entity_id}")
        else:
            self._solar_coefficients_per_unit.clear()
            self._learning_buffer_solar_per_unit.clear()
            # Last batch-fit summaries are stale once coefficients are wiped.
            self._last_batch_fit_per_unit.clear()
            # Carry-over reservoir is whole-house (#896 follow-up) — no
            # per-entity slice exists.  Clear it on any all-units reset
            # so the residual release from the pre-reset regime doesn't
            # leak into the post-reset cold-start.  Per-unit reset
            # (entity_id != None) does NOT clear it: other units are
            # still operating against the same physical reservoir.
            self._solar_carryover_state = 0.0
            _LOGGER.info("Solar learning reset for all units")

        solar_replay_diagnostics: dict | None = None
        if replay_from_history:
            # Scope replay to matching subset of the log.
            if days_back is not None:
                from homeassistant.util import dt as dt_util
                from datetime import timedelta
                cutoff_str = (dt_util.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
                entries = [e for e in self._hourly_log if e.get("timestamp", "") >= cutoff_str]
            else:
                entries = list(self._hourly_log)

            # Per-unit filter: pass a single-element energy_sensors list so
            # replay_solar_nlms naturally skips all other units.  No change
            # to replay_solar_nlms required.
            sensors_for_replay = [entity_id] if entity_id else self.energy_sensors

            solar_replay_diagnostics = self.learning.replay_solar_nlms(
                entries,
                solar_calculator=self.solar,
                screen_config=getattr(self, "screen_config", None),
                correlation_data_per_unit=self._correlation_data_per_unit,
                solar_coefficients_per_unit=self._solar_coefficients_per_unit,
                learning_buffer_solar_per_unit=self._learning_buffer_solar_per_unit,
                energy_sensors=sensors_for_replay,
                learning_rate=self.learning_rate,
                balance_point=self.balance_point,
                aux_affected_entities=self.aux_affected_entities,
                unit_strategies=self._unit_strategies,
                daily_history=self._daily_history,
                unit_min_base=self._per_unit_min_base_thresholds or None,
                return_diagnostics=True,
            )
            _LOGGER.info(
                f"Solar NLMS replay: {solar_replay_diagnostics.get('updates', 0)} updates "
                f"over {len(entries)} entries (unit={entity_id or 'all'})"
            )

        await self._async_save_data(force=True)

        return {
            "status": "reset",
            "unit_entity_id": entity_id,
            "replay_from_history": replay_from_history,
            "solar_replay_diagnostics": solar_replay_diagnostics,
        }

    async def async_batch_fit_solar(
        self,
        entity_id: str | None = None,
        *,
        days_back: int | None = None,
        dry_run: bool = False,
    ) -> dict:
        """Run a periodic batch least-squares fit on solar coefficients (#884).

        Joint 3×3 LS over the modulating-regime hourly log per
        (entity, mode) regime — bridges the mild-weather catch-22 where
        NLMS and inequality both produce zero net signal because base
        demand is near zero (e.g. west-facing rooms whose solar peak
        coincides with the daily temperature maximum).  Saturated and
        shutdown samples are excluded.

        ``days_back`` (default ``None`` = full log) restricts the input
        window.  Recommended after a major release: a 14- or 30-day
        window fits against representative post-upgrade data instead
        of being pulled toward old model behaviour by pre-upgrade
        entries.

        ``dry_run`` (default ``False``) runs every gate and the LS
        solve but does not write coefficients back nor persist.  Each
        regime's ``coefficient_after`` reports the would-be result so
        the user can preview before committing.

        Returns a dict from
        :meth:`LearningManager.batch_fit_solar_coefficients` containing
        per-(entity, regime) sample counts, residuals, before/after
        coefficients, and any skip reasons.  The last-fit summary is
        also stashed in ``self._last_batch_fit_per_unit`` for later
        consumption by ``diagnose_solar`` (dry-run results are still
        stashed so users can inspect via the diagnose path).

        Persists coefficients via the standard save path when any
        regime was updated and ``dry_run`` is ``False``.
        """
        result = self.learning.batch_fit_solar_coefficients(
            hourly_log=self._hourly_log,
            solar_coefficients_per_unit=self._solar_coefficients_per_unit,
            energy_sensors=self.energy_sensors,
            coordinator=self,
            entity_id_filter=entity_id,
            unit_min_base=self._per_unit_min_base_thresholds or None,
            screen_affected_entities=getattr(self, "_screen_affected_set", None),
            days_back=days_back,
            dry_run=dry_run,
        )

        applied_any = False
        applied_count = 0
        skipped_count = 0
        skip_reasons: dict[str, int] = {}
        last_batch_fit_changed = False
        timestamp = dt_util.utcnow().isoformat()
        for eid, regimes in result.items():
            if not isinstance(regimes, dict):
                # Top-level skip (e.g. unknown entity, weighted_smear).
                reason = (
                    regimes.get("skip_reason", "unknown")
                    if isinstance(regimes, dict)
                    else "unknown"
                )
                skipped_count += 1
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                # Record the skip so diagnose_solar can show "skipped + reason"
                # rather than null (which is indistinguishable from "never run").
                self._last_batch_fit_per_unit[eid] = {
                    "timestamp": timestamp,
                    "skip_reason": reason,
                    "regimes": {},
                }
                last_batch_fit_changed = True
                continue
            # Detect the top-level "skip_reason" branch where ``regimes``
            # is a flat dict like ``{"skip_reason": "weighted_smear_excluded"}``.
            if "skip_reason" in regimes and "heating" not in regimes:
                skipped_count += 1
                reason = regimes.get("skip_reason", "unknown")
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                self._last_batch_fit_per_unit[eid] = {
                    "timestamp": timestamp,
                    "skip_reason": reason,
                    "regimes": {},
                }
                last_batch_fit_changed = True
                continue
            self._last_batch_fit_per_unit[eid] = {
                "timestamp": timestamp,
                "regimes": regimes,
            }
            last_batch_fit_changed = True
            for regime_diag in regimes.values():
                if not isinstance(regime_diag, dict):
                    continue
                if regime_diag.get("applied"):
                    applied_count += 1
                    applied_any = True
                elif regime_diag.get("skip_reason"):
                    reason = regime_diag["skip_reason"]
                    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

        # Persist whenever a coefficient was applied OR _last_batch_fit_per_unit
        # was updated.  Pre-#902 this gate was ``applied_any`` only — but with
        # persistent last_batch_fit_per_unit, every kind of skip-record (top-
        # level or per-regime) is durable state that must survive an immediate
        # restart.  Without this, a run that produced only skip-records would
        # leave the new entries in memory and lose them on the next restart.
        if applied_any or last_batch_fit_changed:
            await self._async_save_data(force=True)

        scope = f"unit {entity_id}" if entity_id else "all units"
        if days_back is not None:
            scope += f", last {days_back}d"
        if dry_run:
            scope += ", DRY RUN"
        skip_summary = (
            ", ".join(f"{r}={n}" for r, n in sorted(skip_reasons.items()))
            or "none"
        )
        _LOGGER.info(
            "batch_fit_solar complete (%s): %d applied, %d skipped (reasons: %s)",
            scope, applied_count, skipped_count, skip_summary,
        )

        return {
            "status": "ok",
            "unit_entity_id": entity_id,
            "timestamp": timestamp,
            "days_back": days_back,
            "dry_run": dry_run,
            "applied_count": applied_count,
            "skipped_count": skipped_count,
            "per_unit": result,
        }

    async def async_apply_implied_coefficient(
        self,
        entity_id: str,
        mode: str,
        *,
        dry_run: bool = False,
        force: bool = False,
        days_back: int | None = None,
    ) -> dict:
        """Apply diagnose_solar's implied LS-fit to a unit's coefficient (#884 follow-up).

        The diagnose_solar implied-30d LS fit is precise and well-tested
        but currently has no automatic write-back path: a user reads
        ``implied_coefficient_30d`` from diagnose, decides whether to
        trust it, and copy-pastes via a manual override.  This service
        automates that — with per-direction stability guards so noisy
        components don't get auto-applied.

        Stability assessment: each direction is evaluated across the
        diagnose stability windows.  A direction with sign-flip OR
        ``max(|v|) / min(|v|) > APPLY_IMPLIED_MAX_SPREAD`` between
        windows is "noise-dominated" and gets skipped (current value
        preserved).  Stable directions are written.  ``force=True``
        overrides per-direction guards.

        ``mode`` resolves to a regime via the same mapping used
        elsewhere (HEATING + GUEST_HEATING → "heating", COOLING +
        GUEST_COOLING → "cooling"); OFF / DHW raise ``ValueError``
        because those modes have no learnable solar regime.

        ``dry_run`` runs the analysis pass but writes nothing — the
        return dict reports what *would* have been applied.

        ``days_back`` (default ``None`` = full log) restricts the input
        window.  Recommended after a retrain, post-upgrade, or when the
        user wants to spot-check whether recent data agrees with the
        full-log fit (e.g. compare ``days_back=7`` vs ``days_back=30``
        — significant divergence indicates the older data is dragging
        the fit and the recent window should be trusted).

        Returns
        -------
        ``{
            "status": "ok" | "no_data" | "no_stable_components",
            "unit_entity_id": entity_id,
            "regime": "heating" | "cooling",
            "dry_run": bool,
            "force_applied": bool,
            "days_back": int | None,
            "sample_count": int,
            "implied_30d": dict | None,
            "stability": {direction: {stable, reason, values}, ...},
            "before": {s, e, w},
            "after": {s, e, w},  # what was written (or would be)
            "applied_components": [direction, ...],
            "skipped_components": [direction, ...],
        }``
        """
        from .const import (
            MODE_HEATING,
            MODE_COOLING,
            MODE_GUEST_HEATING,
            MODE_GUEST_COOLING,
            SOLAR_COEFF_CAP,
        )

        if mode in (MODE_HEATING, MODE_GUEST_HEATING):
            regime = "heating"
        elif mode in (MODE_COOLING, MODE_GUEST_COOLING):
            regime = "cooling"
        else:
            raise ValueError(
                f"apply_implied_coefficient called with mode={mode!r}; only "
                f"HEATING / COOLING / GUEST_HEATING / GUEST_COOLING resolve "
                f"to a learnable regime."
            )

        if entity_id not in self.energy_sensors:
            return {
                "status": "no_data",
                "unit_entity_id": entity_id,
                "regime": regime,
                "days_back": days_back,
                "skip_reason": "unknown_entity",
            }

        analysis = self.learning.compute_implied_for_apply(
            hourly_log=self._hourly_log,
            entity_id=entity_id,
            regime=regime,
            coordinator=self,
            unit_min_base=self._per_unit_min_base_thresholds or None,
            screen_affected_entities=getattr(self, "_screen_affected_set", None),
            days_back=days_back,
        )

        before_dict = (
            self._solar_coefficients_per_unit.get(entity_id, {}).get(regime)
            or {"s": 0.0, "e": 0.0, "w": 0.0}
        )
        before = {k: float(before_dict.get(k, 0.0)) for k in ("s", "e", "w")}

        if analysis["implied"] is None:
            return {
                "status": "no_data",
                "unit_entity_id": entity_id,
                "regime": regime,
                "dry_run": dry_run,
                "force_applied": force,
                "days_back": days_back,
                "sample_count": analysis["sample_count"],
                "drop_counts": analysis["drop_counts"],
                "implied_30d": None,
                "stability": {},
                "before": before,
                "after": before,
                "applied_components": [],
                "skipped_components": [],
                "skip_reason": "insufficient_samples",
            }

        stability = self.learning.assess_apply_implied_stability(
            analysis["windows"]
        )

        implied_30d = analysis["implied"]
        applied_components: list[str] = []
        skipped_components: list[str] = []
        new_coeff = dict(before)
        for d in ("s", "e", "w"):
            stable = stability[d]["stable"]
            if force or stable:
                # Clamp to invariant #4: non-negative, ≤ CAP.
                new_coeff[d] = max(0.0, min(SOLAR_COEFF_CAP, implied_30d[d]))
                applied_components.append(d)
            else:
                # Preserve current value.
                skipped_components.append(d)

        if not applied_components:
            return {
                "status": "no_stable_components",
                "unit_entity_id": entity_id,
                "regime": regime,
                "dry_run": dry_run,
                "force_applied": force,
                "days_back": days_back,
                "sample_count": analysis["sample_count"],
                "implied_30d": implied_30d,
                "stability": stability,
                "before": before,
                "after": before,
                "applied_components": [],
                "skipped_components": skipped_components,
            }

        if not dry_run:
            self.learning._update_unit_solar_coefficient(
                entity_id, new_coeff, self._solar_coefficients_per_unit, regime
            )
            await self._async_save_data(force=True)

        scope = (
            f"{entity_id} [{regime}]"
            f"{f', last {days_back}d' if days_back is not None else ''}"
            f"{', DRY RUN' if dry_run else ''}"
            f"{', forced' if force else ''}"
        )
        _LOGGER.info(
            "apply_implied_coefficient (%s): applied=%s, skipped=%s "
            "(samples=%d, before=%s, after=%s)",
            scope, applied_components, skipped_components,
            analysis["sample_count"], before, new_coeff,
        )

        return {
            "status": "ok",
            "unit_entity_id": entity_id,
            "regime": regime,
            "dry_run": dry_run,
            "force_applied": force,
            "days_back": days_back,
            "sample_count": analysis["sample_count"],
            "implied_30d": implied_30d,
            "stability": stability,
            "before": before,
            "after": new_coeff,
            "applied_components": applied_components,
            "skipped_components": skipped_components,
        }

    async def async_migrate_aux_coefficients(self, new_aux_affected_entities: list[str]):
        """Migrate auxiliary coefficients when the affected entities list changes.

        Implements the 'Conservation Strategy':
        When a unit is removed from the aux list, its accumulated auxiliary reduction (coefficients)
        is redistributed to the remaining units to maintain the global Total_Aux_Reduction.

        Redistribution is proportional to the remaining units' existing coefficients (strength).
        Fallback: If no remaining units have data for a specific bucket, assign to the Primary Unit
        (highest total consumption).
        """
        old_list = self.aux_affected_entities or []
        # Identify removed units (In old but not in new)
        removed_units = [u for u in old_list if u not in new_aux_affected_entities]

        if not removed_units:
            _LOGGER.debug("Aux Migration: No units removed, skipping conservation logic.")
            # We still update the internal list to match the pending config change
            self.aux_affected_entities = new_aux_affected_entities
            # Also update the optimized set
            self._aux_affected_set = set(new_aux_affected_entities) if new_aux_affected_entities else set()
            return

        _LOGGER.info(f"Aux Migration: Removed units {removed_units}. Beginning conservation strategy.")

        # Helper to find Primary Unit among remaining
        def _get_primary_unit(candidates: list[str]) -> str | None:
            best_unit = None
            max_kwh = -1.0
            for entity_id in candidates:
                val = self._get_float_state(entity_id)
                if val is None: val = 0.0
                if val > max_kwh:
                    max_kwh = val
                    best_unit = entity_id

            if best_unit is None and candidates:
                return candidates[0]
            return best_unit

        primary_unit = _get_primary_unit(new_aux_affected_entities)

        changes_made = False

        for removed_unit in removed_units:
            if removed_unit not in self._aux_coefficients_per_unit:
                continue

            unit_data = self._aux_coefficients_per_unit[removed_unit]

            # Iterate through all temp keys and wind buckets for this unit
            for temp_key, wind_data in unit_data.items():
                for wind_bucket, coeff_value in wind_data.items():
                    if coeff_value <= 0:
                        continue

                    # We have a value to redistribute
                    # 1. Calculate total weight of remaining units for this specific bucket
                    total_weight = 0.0
                    eligible_weights = {}

                    for target_unit in new_aux_affected_entities:
                        # Get existing coeff
                        w = 0.0
                        if target_unit in self._aux_coefficients_per_unit:
                            if temp_key in self._aux_coefficients_per_unit[target_unit]:
                                w = self._aux_coefficients_per_unit[target_unit][temp_key].get(wind_bucket, 0.0)

                        eligible_weights[target_unit] = w
                        total_weight += w

                    # 2. Distribute
                    if total_weight > 0:
                        # Proportional Redistribution
                        for target_unit, weight in eligible_weights.items():
                            if weight > 0:
                                share = (weight / total_weight) * coeff_value
                                self._add_aux_coefficient(target_unit, temp_key, wind_bucket, share)
                                changes_made = True
                    else:
                        # Fallback: Assign to Primary Unit
                        if primary_unit:
                            _LOGGER.debug(f"Aux Migration: Fallback to primary {primary_unit} for {temp_key}/{wind_bucket} (Val: {coeff_value})")
                            self._add_aux_coefficient(primary_unit, temp_key, wind_bucket, coeff_value)
                            changes_made = True

            # Clean up the removed unit
            del self._aux_coefficients_per_unit[removed_unit]
            changes_made = True

        if changes_made:
            _LOGGER.info("Aux Migration: Completed successfully. Saving data.")
            await self._async_save_data(force=True)

        self.aux_affected_entities = new_aux_affected_entities
        self._aux_affected_set = set(new_aux_affected_entities) if new_aux_affected_entities else set()

    def _add_aux_coefficient(self, entity_id: str, temp_key: str, wind_bucket: str, value_to_add: float):
        """Helper to safely add value to a coefficient, clamped to base model."""
        if entity_id not in self._aux_coefficients_per_unit:
            self._aux_coefficients_per_unit[entity_id] = {}
        if temp_key not in self._aux_coefficients_per_unit[entity_id]:
            self._aux_coefficients_per_unit[entity_id][temp_key] = {}

        current = self._aux_coefficients_per_unit[entity_id][temp_key].get(wind_bucket, 0.0)
        new_val = current + value_to_add

        # Clamp: Aux coefficient cannot exceed base model for this unit/temp/wind
        # (aux reduction cannot be greater than what the unit would use without aux)
        unit_base_data = self._correlation_data_per_unit.get(entity_id, {})
        if temp_key in unit_base_data and wind_bucket in unit_base_data[temp_key]:
            base_limit = unit_base_data[temp_key][wind_bucket]
            new_val = min(new_val, base_limit)

        self._aux_coefficients_per_unit[entity_id][temp_key][wind_bucket] = round(new_val, 3)


    async def async_replace_sensor_source(self, old_entity_id: str, new_entity_id: str) -> bool:
        """Replace a heating unit's energy sensor with a new one, migrating all data."""
        if not old_entity_id or not new_entity_id:
             _LOGGER.error("Replace Sensor Source: Missing old or new entity ID.")
             return False

        if old_entity_id not in self.energy_sensors:
             _LOGGER.debug(f"Replace Sensor Source: Old entity '{old_entity_id}' not found in this coordinator. Skipping.")
             return False

        if new_entity_id in self.energy_sensors:
             _LOGGER.error(f"Replace Sensor Source: New entity '{new_entity_id}' is already configured in this coordinator. Aborting to prevent data corruption.")
             return False

        # Verify new entity exists in HA state machine
        if not self.hass.states.get(new_entity_id):
            _LOGGER.error(f"Replace Sensor Source: New entity '{new_entity_id}' does not exist in Home Assistant. Please check the entity ID.")
            return False

        _LOGGER.info(f"Replacing sensor source: '{old_entity_id}' -> '{new_entity_id}'")

        # 1. Update In-Memory Configuration
        try:
            self.energy_sensors.remove(old_entity_id)
        except ValueError:
            pass

        if new_entity_id not in self.energy_sensors:
            self.energy_sensors.append(new_entity_id)

        # Update Aux Affected Entities if applicable
        if self.aux_affected_entities is not None and old_entity_id in self.aux_affected_entities:
            try:
                self.aux_affected_entities.remove(old_entity_id)
                self._aux_affected_set.discard(old_entity_id)
            except ValueError:
                pass
            if new_entity_id not in self.aux_affected_entities:
                self.aux_affected_entities.append(new_entity_id)
                self._aux_affected_set.add(new_entity_id)

        # 2. Update Persistent Configuration (Config Entry)
        new_data = self.entry.data.copy()
        new_data["energy_sensors"] = self.energy_sensors
        if CONF_AUX_AFFECTED_ENTITIES in new_data:
            new_data[CONF_AUX_AFFECTED_ENTITIES] = self.aux_affected_entities
        self.hass.config_entries.async_update_entry(self.entry, data=new_data)
        _LOGGER.info(f"Config entry updated. New sensor list: {self.energy_sensors}")

        # 3. Migrate Dictionary Data
        def _migrate_key(d: dict, old_k: str, new_k: str):
            if old_k in d:
                # If new key exists, we overwrite it (assuming replacement intent)
                # or we could merge? For simplicity and safety of history, we overwrite/move.
                d[new_k] = d.pop(old_k)

        _migrate_key(self._correlation_data_per_unit, old_entity_id, new_entity_id)
        _migrate_key(self._aux_coefficients_per_unit, old_entity_id, new_entity_id) # New: Migrate unit aux coeffs
        _migrate_key(self._solar_coefficients_per_unit, old_entity_id, new_entity_id) # New: Migrate unit solar coeffs
        _migrate_key(self._unit_modes, old_entity_id, new_entity_id) # New: Migrate unit modes
        _migrate_key(self._observation_counts, old_entity_id, new_entity_id)
        _migrate_key(self._hourly_delta_per_unit, old_entity_id, new_entity_id)
        _migrate_key(self._hourly_expected_per_unit, old_entity_id, new_entity_id)
        _migrate_key(self._hourly_expected_base_per_unit, old_entity_id, new_entity_id)
        _migrate_key(self._learning_buffer_per_unit, old_entity_id, new_entity_id)
        _migrate_key(self._learning_buffer_aux_per_unit, old_entity_id, new_entity_id) # New: Migrate unit aux buffer
        _migrate_key(self._learning_buffer_solar_per_unit, old_entity_id, new_entity_id) # New: Migrate unit solar buffer
        _migrate_key(self._daily_individual, old_entity_id, new_entity_id)
        _migrate_key(self._lifetime_individual, old_entity_id, new_entity_id)

        # 4. Handle Last Energy Values (Baseline Reset)
        # We DELETE the old key to remove the old baseline.
        # We do NOT set the new key yet. This forces the system to treat the new sensor
        # as a "Fresh Start" on the next update loop, establishing a new baseline from its current value.
        # This prevents massive spikes from Old=5000kWh and New=0kWh.
        if old_entity_id in self._last_energy_values:
            del self._last_energy_values[old_entity_id]
            _LOGGER.info(f"Removed baseline for '{old_entity_id}'. New sensor '{new_entity_id}' will initialize baseline on next update.")

        # 5. Migrate Hourly Log History (Preserve Statistics)
        # We iterate through history and rename the keys in the breakdown dicts.
        count_migrated = 0
        for entry in self._hourly_log:
            # Unit Breakdown (Actual)
            if "unit_breakdown" in entry:
                _migrate_key(entry["unit_breakdown"], old_entity_id, new_entity_id)

            # Unit Expected Breakdown
            if "unit_expected_breakdown" in entry:
                _migrate_key(entry["unit_expected_breakdown"], old_entity_id, new_entity_id)

            count_migrated += 1

        _LOGGER.info(f"Migrated stats in {count_migrated} hourly log entries.")

        # 6. Save Everything
        await self._async_save_data(force=True)
        _LOGGER.info("Sensor replacement and data migration completed successfully.")

        # 7. Refresh to establish new baseline immediately
        # This prevents a "zero" period or drift until the next scheduled update.
        await self.async_refresh()

        # 8. Save again to persist the new baseline established by refresh
        # If we reload immediately after this without saving, the new baseline (in memory) is lost.
        await self._async_save_data(force=True)
        _LOGGER.info("New sensor baseline saved to disk.")

        return True

    async def async_backup_data(self, file_path: str):
        """Backup full system state to JSON file."""
        await self.storage.async_backup_data(file_path)

    async def async_restore_data(self, file_path: str):
        """Restore full system state from JSON file."""
        await self.storage.async_restore_data(file_path)

    async def export_csv_data(self, file_path: str, export_type: str):
        """Export data to CSV."""
        await self.storage.export_csv_data(file_path, export_type)

    async def import_csv_data(self, file_path: str, mapping: dict, update_model: bool = True):
        """Import historical data from CSV."""
        await self.storage.import_csv_data(file_path, mapping, update_model)

    async def retrain_from_history(
        self,
        days_back: int | None = None,
        reset_first: bool = False,
        experimental_cop_smear: bool = False,
    ) -> dict:
        """Delegates to :class:`retrain.RetrainEngine`."""
        return await self._retrain.retrain_from_history(
            days_back=days_back,
            reset_first=reset_first,
            experimental_cop_smear=experimental_cop_smear,
        )

    def diagnose_model(self, days_back: int = 30) -> dict:
        """Delegates to :class:`diagnostics.DiagnosticsEngine`."""
        return self._diagnostics.diagnose_model(days_back)

    def _calibrate_per_unit_min_base_thresholds(
        self,
        *,
        sample_days: int = 30,
        require_min_hours_of_log: int | None = None,
    ) -> dict:
        """Delegates to :class:`diagnostics.DiagnosticsEngine`."""
        return self._diagnostics.calibrate_per_unit_min_base_thresholds(
            sample_days=sample_days,
            require_min_hours_of_log=require_min_hours_of_log,
        )

    def diagnose_solar(self, days_back: int = 30, apply_battery_decay: bool = False) -> dict:
        """Delegates to :class:`diagnostics.DiagnosticsEngine`."""
        return self._diagnostics.diagnose_solar(days_back, apply_battery_decay)

    def _get_float_state(self, entity_id: str) -> float | None:
        """Helper to get float state from an entity."""
        if not entity_id:
            return None
        state = self.hass.states.get(entity_id)
        if state and state.state not in ("unknown", "unavailable"):
            try:
                return float(state.state)
            except ValueError:
                pass
        return None

    def _get_cloud_coverage(self) -> float:
        """Get cloud coverage in percent (0-100)."""
        if not self.weather_entity:
            return 50.0 # Conservative default

        state = self.hass.states.get(self.weather_entity)
        if not state:
            return 50.0

        # Try to find numeric attribute first
        cloud_attr = state.attributes.get("cloud_coverage")
        if cloud_attr is not None:
            try:
                val = float(cloud_attr)
                return max(0.0, min(100.0, val))
            except ValueError:
                pass

        # Fallback: Map condition state — warn once per hour.
        # Condition-text mapping (e.g. "sunny" → 10%) is too coarse for
        # reliable solar coefficient learning.  Numeric cloud_coverage from
        # the weather entity is strongly recommended.
        if self.hass.is_running:
            now = dt_util.now()
            last_warn = getattr(self, '_cloud_coverage_warn_time', None)
            if last_warn is None or (now - last_warn).total_seconds() >= 3600:
                _LOGGER.warning(
                    "Weather entity '%s' does not provide numeric cloud_coverage attribute. "
                    "Falling back to condition-text mapping which significantly reduces solar "
                    "model accuracy. Consider switching to a weather integration that provides "
                    "cloud_coverage (e.g. Met.no or the custom Open-Meteo fork).",
                    self.weather_entity,
                )
                self._cloud_coverage_warn_time = now

        condition = state.state
        if condition in CLOUD_COVERAGE_MAP:
            return float(CLOUD_COVERAGE_MAP[condition])

        return DEFAULT_CLOUD_COVERAGE

    def _get_speed_in_ms(self, entity_id: str) -> float | None:
        """Get speed in m/s from an entity, converting from km/h, mph, or knots if needed."""
        if not entity_id:
            return None
        state = self.hass.states.get(entity_id)
        if state and state.state not in ("unknown", "unavailable"):
            try:
                value = float(state.state)
                unit = state.attributes.get("unit_of_measurement")
                return convert_speed_to_ms(value, unit)
            except ValueError:
                pass
        return None

    def _get_weather_attribute(self, attribute: str) -> float | None:
        """Get attribute from weather entity, handling units if necessary."""
        if not self.weather_entity:
            return None

        state = self.hass.states.get(self.weather_entity)
        if not state:
            return None

        val = state.attributes.get(attribute)
        if val is None:
            return None

        try:
            val = float(val)
        except (ValueError, TypeError):
            return None

        # Handle Wind Units
        if attribute in ("wind_speed", "wind_gust_speed"):
            unit = state.attributes.get("wind_speed_unit")
            return convert_speed_to_ms(val, unit)

        return val

    def _get_inertia_parameters(self) -> tuple[int, int]:
        """Helper to derive requirements from weights."""
        hours_back = len(self.inertia_weights) - 1
        # max_gap is the stale-data cutoff (gap detection), tied to tau – not the kernel window.
        # Logs older than tau hours are considered a thermal discontinuity and discarded.
        max_gap = int(self.inertia_tau)
        return hours_back, max_gap

    def _calculate_weighted_inertia(self, temps: list[float]) -> float:
        """Calculate weighted inertia temperature.

        Aligns available temps (Newest -> Oldest) to weights (Newest -> Oldest).
        Normalizes weights to 1.0 based on available samples.
        """
        if not temps:
            return 0.0

        # Weights are defined Oldest -> Newest (Time increasing)
        # e.g. [0.1, 0.2, 0.3, 0.4]
        # temps list is also Oldest -> Newest (History + Current)
        # e.g. [12, 11, 10] (H-2, H-1, Current)

        # We want to match Newest Temp to Newest Weight.
        # Temp[-1] matches Weight[-1]
        # Temp[-2] matches Weight[-2]

        num_temps = len(temps)
        num_weights = len(self.inertia_weights)

        # Take the last N weights where N = num_temps
        # But limited by num_weights (shouldn't happen if logic is correct, but safe)
        count = min(num_temps, num_weights)

        active_temps = temps[-count:]
        active_weights = self.inertia_weights[-count:]

        total_weight = sum(active_weights)
        if total_weight == 0:
            return sum(active_temps) / count # Fallback to simple average

        weighted_sum = sum(t * w for t, w in zip(active_temps, active_weights))
        return weighted_sum / total_weight

    def _get_recent_log_temps(self, reference_time: datetime, hours_back: int | None = None, max_gap_hours: int | None = None) -> list[float]:
        """Get recent temperatures from log, ensuring no thermodynamic discontinuity.

        Args:
            reference_time: The point in time to look backwards from.
            hours_back: Number of past hourly samples to retrieve (target). Defaults to the
                        active inertia kernel length (derived from CONF_THERMAL_INERTIA via
                        generate_exponential_kernel; capped at 5×tau hours).
            max_gap_hours: Maximum allowable age of a log sample. If older, it's ignored.
                           This protects against using data from before a long downtime.
                           Defaults to the configured inertia tau (in hours).
        """
        def_hours, def_gap = self._get_inertia_parameters()

        if hours_back is None:
            hours_back = def_hours
        if max_gap_hours is None:
            max_gap_hours = def_gap

        temps = []
        if not self._hourly_log:
            return temps

        # Calculate threshold: Logs older than max_gap_hours should be ignored
        cutoff_time = reference_time - timedelta(hours=max_gap_hours)

        # Filter FIRST: Select all logs that are recent (within tolerance)
        valid_logs = []
        # Optimization: Iterate backwards as logs are sorted by time (newest last)
        for log in reversed(self._hourly_log):
            try:
                timestamp_str = log.get("timestamp")
                if timestamp_str:
                    log_dt = dt_util.parse_datetime(timestamp_str)
                    if log_dt:
                        # Ensure timezone awareness for comparison
                        if log_dt.tzinfo is None and reference_time.tzinfo:
                            log_dt = log_dt.replace(tzinfo=reference_time.tzinfo)

                        if log_dt >= cutoff_time:
                            valid_logs.append(log)
                        else:
                            # Since we iterate backwards, once we hit an old log, we can stop
                            # assuming logs are sorted.
                            break
            except (ValueError, TypeError):
                pass

        # Restore chronological order (oldest first)
        valid_logs.reverse()

        # Take last N valid logs
        recent_logs = valid_logs[-hours_back:]
        for log in recent_logs:
            temps.append(log["temp"])

        return temps

    def _get_inertia_list(self, current_time: datetime) -> list[float]:
        """Get full list of temperatures for inertia calculation (History + Current).

        Centralizes the logic used by forecast seeding and inertia calculations.
        Returns the raw list of temperatures [H-3, H-2, H-1, Current].
        """
        # Get valid history using helper
        temps = self._get_recent_log_temps(current_time)

        # Add current temp (Strict Source Logic)
        current_temp = None

        if self.outdoor_temp_source == SOURCE_SENSOR and self.outdoor_temp_sensor:
            # Prefer hourly rolling average for consistency with historical log entries,
            # which are stored as hourly averages. Fall back to raw sensor reading only
            # if no samples have been collected yet this hour.
            if self._collector.sample_count > 0:
                current_temp = self._collector.temp_sum / self._collector.sample_count
            else:
                current_temp = self._get_float_state(self.outdoor_temp_sensor)

        elif self.outdoor_temp_source == SOURCE_WEATHER or (
            self.outdoor_temp_source == SOURCE_SENSOR and not self.outdoor_temp_sensor
        ):
             # Use Weather (or if Sensor source selected but no sensor configured - edge case)
             current_temp = self._get_weather_attribute("temperature")

        if current_temp is not None:
            temps.append(current_temp)

        # If list is empty but we have a running average (edge case?), return that seeded
        # Handled by caller if needed.

        return temps

    def _calculate_inertia_temp(self) -> float | None:
        """Calculate the inertia temperature (Weighted Average).

        Uses the last N-1 valid hours of history from hourly_log + current/latest known temp.
        N = len(self.inertia_weights).
        """
        current_time = dt_util.now()
        temps = self._get_inertia_list(current_time)

        if not temps:
            return None

        # Short-circuit: no history available, return current temp directly
        # (avoids floating-point rounding from kernel normalisation)
        if len(temps) == 1:
            return temps[0]

        return self._calculate_weighted_inertia(temps)

    def _calculate_effective_wind(self, speed: float, gust: float | None) -> float:
        """Calculate effective wind speed.

        Formula: wind_speed + (wind_gust - wind_speed) * gust_factor
        We treat wind_speed as the base load, and add a weighted component of the gust (turbulence)
        that exceeds the average speed.
        """
        # Ensure non-negative speed
        speed = max(0.0, speed)

        if gust is None:
            return speed

        gust = max(0.0, gust)

        # Calculate turbulence component (only if gust > speed)
        turbulence = max(0.0, gust - speed)

        return speed + (turbulence * self.wind_gust_factor)

    def _get_wind_bucket(self, effective_wind: float) -> str:
        """Determine wind bucket."""
        if effective_wind >= self.extreme_wind_threshold:
            return "extreme_wind"
        elif effective_wind >= self.wind_threshold:
            return "high_wind"
        return "normal"

    def _get_weather_wind_unit(self) -> str | None:
        """Get wind speed unit from weather entity."""
        if not self.weather_entity:
            return None
        weather_state = self.hass.states.get(self.weather_entity)
        if not weather_state:
            return None
        return weather_state.attributes.get("wind_speed_unit")

    def _get_sun_info_now(self):
        """Get current sun elevation and azimuth from HA."""
        sun_state = self.hass.states.get("sun.sun")
        if not sun_state:
            if self.solar_enabled and self.hass.is_running:
                now = dt_util.now()
                last_warn = getattr(self, '_sun_entity_warn_time', None)
                if last_warn is None or (now - last_warn).total_seconds() >= 3600:
                    _LOGGER.warning(
                        "sun.sun entity is unavailable — solar model is disabled. "
                        "All solar factors will be zero until the entity recovers. "
                        "Check that the Sun integration is enabled in HA."
                    )
                    self._sun_entity_warn_time = now
            return 0.0, 0.0

        try:
            elev = float(sun_state.attributes.get("elevation", 0))
            azim = float(sun_state.attributes.get("azimuth", 0))
            return elev, azim
        except (ValueError, TypeError):
            return 0.0, 0.0

    def _update_live_predictions(self, calc_temp: float | None, temp_key: str, wind_bucket: str, current_time: datetime) -> float:
        """Update live prediction models and accumulate expected energy."""
        effective_wind = self.data.get("effective_wind", 0.0)

        # 1. Use Centralized Robust Calculation (StatisticsManager)
        # This replaces manual calls to _get_predicted_kwh, _get_aux_impact_kw, etc.
        # and automatically handles aggregation of units.

        solar_impact_global = self.data.get(ATTR_SOLAR_IMPACT, 0.0)

        current_prediction_rate = 0.0
        unit_breakdown = {}
        aux_impact_rate = 0.0
        result = {}

        if calc_temp is not None:
            result = self.statistics.calculate_total_power(
                calc_temp,
                effective_wind,
                solar_impact_global,
                is_aux_active=self.auxiliary_heating_active
            )

            current_prediction_rate = result["total_kwh"]
            unit_breakdown = result["unit_breakdown"]
            # Kelvin Protocol: Use Global Authority for Aux Impact
            # We must use the Global Model value, not the Sum of Units (which is subject to exclusions)
            # STRICT MODE: Default to 0.0 if global model is missing, do NOT fallback to unit sum.
            aux_impact_rate = result.get("global_aux_reduction_kwh", 0.0)

            # Sync solar impact to data for sensor exposure
            if self.solar_enabled:
                 self.data[ATTR_SOLAR_IMPACT] = result["breakdown"]["solar_reduction_kwh"]

        # Store Current Model Rate and Aux state for gap filling
        self.data["current_model_rate"] = current_prediction_rate
        self.data["current_calc_temp"] = calc_temp
        self.data["current_aux_impact_rate"] = aux_impact_rate
        self.data["current_unit_breakdown"] = unit_breakdown

        # Intra-hour accumulation (delegated to ObservationCollector)
        if self._collector.last_minute_processed != current_time.minute:
            # Calculate time step (minutes) to account for gaps (e.g., restart, downtime)
            minutes_step = 1  # Default to 1 minute
            if self._collector.last_minute_processed is not None:
                diff = current_time.minute - self._collector.last_minute_processed
                if diff > 0:
                    minutes_step = diff
                # If diff < 0 (hour rollover?), we rely on start_time reset logic
                # in _process_hourly_data to handle the fresh start, so minutes_step=1 is fine.
            else:
                # First run in hour (e.g., after restart)
                # Minute is 0-indexed, so :30 means 31 minutes have passed in this hour slot.
                minutes_step = current_time.minute + 1

            fraction = minutes_step / 60.0
            orphaned_part = result.get("breakdown", {}).get("orphaned_aux_savings", 0.0)

            self._collector.accumulate_expected(
                fraction=fraction,
                prediction_rate=current_prediction_rate,
                aux_impact_rate=aux_impact_rate,
                unit_breakdown=unit_breakdown,
                orphaned_part=orphaned_part,
            )

            # Mark minute as processed
            self._collector.last_minute_processed = current_time.minute

        return current_prediction_rate

    def _get_partial_log_for_current_hour(self) -> dict | None:
        """Retrieve partial log entry for the current hour if it exists (e.g. pre-restart data)."""
        if not self._hourly_log:
            return None

        current_time = dt_util.now()
        current_hour = current_time.hour
        today_iso = current_time.date().isoformat()

        # Check the most recent log entry
        last_entry = self._hourly_log[-1]
        if last_entry["hour"] == current_hour and last_entry["timestamp"].startswith(today_iso):
            return last_entry

        return None

    def _close_hour_gap(
        self,
        current_time: datetime,
        last_minute: int,
        avg_temp: float = 0.0,
        avg_wind: float = 0.0,
        avg_solar: float = 0.0,
        is_aux_active: bool = False,
        avg_solar_vector: tuple[float, float, float] | None = None
    ):
        """Close the gap at hour boundary by filling missing minutes.

        If the last processed minute was 58, and we are now in the next hour,
        we missed minute 59 (and potentially the 59->00 transition).
        This method calculates the missing expected energy for the tail of the previous hour
        to ensure the hourly log sums to exactly 60 minutes of expectation.

        Kelvin Protocol Improvement: Mean Imputation
        Uses hourly aggregates (Temp, Wind, Solar, Aux) to calculate the gap energy.
        This ensures consistency between the logged hour stats and the total accumulated energy,
        eliminating "Desync" caused by using point-in-time rates from the end of the hour.
        """
        # Calculate how many minutes were missed at the end of the previous hour
        # Example: last=58. Missed 59. (60 - (58+1)) = 1 minute.
        minutes_missing = 60 - (last_minute + 1)

        if minutes_missing <= 0:
            return

        fraction = minutes_missing / 60.0

        # Calculate Gap Power using Aggregates
        result = self.statistics.calculate_total_power(
            avg_temp,
            avg_wind,
            0.0, # solar_impact (ignored when override is used)
            is_aux_active=is_aux_active,
            override_solar_factor=avg_solar,
            override_solar_vector=avg_solar_vector,
            detailed=True
        )

        current_rate = result["total_kwh"]
        unit_breakdown = result["unit_breakdown"]
        aux_impact_rate = result.get("global_aux_reduction_kwh", 0.0)

        if current_rate > 0:
            self._collector.expected_energy_hour += (current_rate * fraction)
            _LOGGER.debug(
                f"Closed hour gap: Added {minutes_missing} min using aggregates "
                f"(Temp={avg_temp:.1f}, Wind={avg_wind:.1f}, Rate={current_rate:.2f} kW)"
            )

            # Accumulate gap aux impact
            if aux_impact_rate > 0:
                 self._collector.aux_impact_hour += (aux_impact_rate * fraction)

            # Distribute gap energy to units
            for entity_id, stats in unit_breakdown.items():
                unit_pred = stats["net_kwh"]
                if unit_pred > 0:
                    if entity_id not in self._hourly_expected_per_unit:
                        self._hourly_expected_per_unit[entity_id] = 0.0
                    self._hourly_expected_per_unit[entity_id] += (unit_pred * fraction)

                unit_base = stats.get("base_kwh", 0.0)
                if unit_base > 0:
                    if entity_id not in self._hourly_expected_base_per_unit:
                        self._hourly_expected_base_per_unit[entity_id] = 0.0
                    self._hourly_expected_base_per_unit[entity_id] += (unit_base * fraction)

                # Accumulate Gap Aux Breakdown
                if entity_id not in self._collector.aux_breakdown:
                    self._collector.aux_breakdown[entity_id] = {"allocated": 0.0, "overflow": 0.0}

                applied_aux = stats.get("aux_reduction_kwh", 0.0)
                overflow_aux = stats.get("overflow_kwh", 0.0)

                self._collector.aux_breakdown[entity_id]["allocated"] += (applied_aux * fraction)
                self._collector.aux_breakdown[entity_id]["overflow"] += (overflow_aux * fraction)

    def _calculate_daily_wind_penalty(self) -> float:
        """Calculate the total kWh penalty due to wind for the entire day (Past + Future)."""
        total_penalty = 0.0
        today_iso = dt_util.now().date().isoformat()

        # 1. Past Hours (Completed)
        for log in self._hourly_log:
            if not log["timestamp"].startswith(today_iso):
                continue

            # Recalculate model with 0 wind vs Actual Wind
            # Note: We use the stored temperature and auxiliary state for consistency.
            temp = log["temp"]
            aux_active = log.get("auxiliary_active", False)
            solar_impact = 0.0 # Solar cancels out in difference, keep 0 for simplicity

            # Use logged solar factor to ensure correct saturation logic
            s_factor = log.get("solar_factor", 0.0)

            # Use logged unit modes if available, else standard inference
            unit_modes = log.get("unit_modes")

            # Scenario 1: Actual Wind
            # Use 'effective_wind' from log
            eff_wind = log.get("effective_wind", 0.0)

            s_vector_s = log.get("solar_vector_s")
            s_vector_e = log.get("solar_vector_e")
            s_vector_w = log.get("solar_vector_w", 0.0)
            s_vector = (s_vector_s, s_vector_e, s_vector_w) if s_vector_s is not None and s_vector_e is not None else None

            res_actual = self.statistics.calculate_total_power(
                temp=temp,
                effective_wind=eff_wind,
                solar_impact=solar_impact,
                is_aux_active=aux_active,
                unit_modes=unit_modes,
                override_solar_factor=s_factor,
                override_solar_vector=s_vector,
                detailed=False
            )

            # Scenario 2: No Wind
            res_no_wind = self.statistics.calculate_total_power(
                temp=temp,
                effective_wind=0.0,
                solar_impact=solar_impact,
                is_aux_active=aux_active,
                unit_modes=unit_modes,
                override_solar_factor=s_factor,
                override_solar_vector=s_vector,
                detailed=False
            )

            # Penalty = Energy(Wind) - Energy(NoWind)
            penalty = max(0.0, res_actual["total_kwh"] - res_no_wind["total_kwh"])
            total_penalty += penalty

        # 2. Current Partial Hour
        current_temp = self.data.get("current_calc_temp")
        if current_temp is not None:
            eff_wind = self.data.get("effective_wind", 0.0)
            minutes_passed = dt_util.now().minute
            fraction = minutes_passed / 60.0

            res_curr = self.statistics.calculate_total_power(
                current_temp, eff_wind, 0.0, is_aux_active=self.auxiliary_heating_active, detailed=False
            )
            res_curr_no_wind = self.statistics.calculate_total_power(
                current_temp, 0.0, 0.0, is_aux_active=self.auxiliary_heating_active, detailed=False
            )

            penalty = max(0.0, res_curr["total_kwh"] - res_curr_no_wind["total_kwh"])
            total_penalty += (penalty * fraction)

        # 3. Future Hours
        # Compare Future Normal vs Future No Wind
        future_normal, _, _ = self.forecast.calculate_future_energy(
            dt_util.now(), ignore_aux=(not self.auxiliary_heating_active)
        )

        future_no_wind, _, _ = self.forecast.calculate_future_energy(
            dt_util.now(), ignore_aux=(not self.auxiliary_heating_active), force_no_wind=True
        )

        total_penalty += max(0.0, future_normal - future_no_wind)

        return round(total_penalty, 2)

    def _update_daily_budgets(self, current_prediction_rate: float, current_time: datetime, minutes_passed: int):
        """Update daily energy budgets and forecasts."""
        expected_hour_so_far = self._collector.expected_energy_hour

        expected_today_sum = 0.0
        forecasted_past_sum = 0.0
        gross_past_sum = 0.0
        has_past_aux = False
        solar_today_sum = 0.0
        today_date_str = current_time.date().isoformat()

        for entry in reversed(self._hourly_log):
             if entry["timestamp"].startswith(today_date_str):
                 expected_today_sum += entry.get("expected_kwh") or 0.0
                 forecasted_past_sum += entry.get("forecasted_kwh") or entry.get("expected_kwh") or 0.0
                 gross_past_sum += entry.get("forecasted_kwh_gross") or entry.get("forecasted_kwh") or entry.get("expected_kwh") or 0.0
                 solar_today_sum += entry.get("solar_impact_kwh") or 0.0
                 if entry.get("auxiliary_active", False):
                     has_past_aux = True
             else:
                 break

        expected_today_total = expected_today_sum + expected_hour_so_far
        self.data[ATTR_EXPECTED_TODAY] = round(expected_today_total, 3)

        # Calculate Future Forecast
        # The running plan (Predicted) MUST account for current aux state for future hours.
        future_forecast_kwh, future_solar_kwh, future_unit_totals = self.forecast.calculate_future_energy(
            current_time, ignore_aux=(not self.auxiliary_heating_active)
        )

        # Calculate Current Hour Plan (Forecast based, not actual)
        # Use the Reference Forecast (locked at midnight) for stability.
        current_hour_plan_rate, current_hour_unit_breakdown = self.forecast.get_plan_for_hour(
            current_time,
            source='reference',
            ignore_aux=(not self.auxiliary_heating_active)
        )

        # Fallback: If no reference forecast, use live model (best guess)
        if current_hour_plan_rate == 0.0 and current_prediction_rate > 0.0:
            current_hour_plan_rate = current_prediction_rate

        # Calculate Daily Budget (The Plan)
        # = Locked Past + Stable Current Forecast (Reference) + Live Future Forecast
        budget_total = forecasted_past_sum + current_hour_plan_rate + future_forecast_kwh
        self.data[ATTR_PREDICTED] = round(budget_total, 2)

        # Calculate Gross Daily Budget (aux-unaware) for deviation denominator.
        if self.auxiliary_heating_active:
             future_gross_kwh_budget, _, _ = self.forecast.calculate_future_energy(
                 current_time, ignore_aux=True
             )
             current_hour_gross_rate_budget, _ = self.forecast.get_plan_for_hour(
                 current_time, source='reference', ignore_aux=True
             )
             if current_hour_gross_rate_budget == 0.0:
                  current_hour_gross_rate_budget = current_hour_plan_rate + self.data.get("current_aux_impact_rate", 0.0)
             budget_total_gross = gross_past_sum + current_hour_gross_rate_budget + future_gross_kwh_budget
        elif has_past_aux:
             budget_total_gross = gross_past_sum + current_hour_plan_rate + future_forecast_kwh
        else:
             budget_total_gross = budget_total

        self.data["predicted_gross"] = round(budget_total_gross, 2)

        # Calculate Thermodynamic Projection (The Reality Check)
        # = Actuals So Far + Live Forecast for Remaining Today
        # This answers: "Where will we end up if the house behaves as modeled for the rest of the day?"
        # Uses LIVE forecast for current hour (pro-rated) and future hours.

        # 1. Get Live Plan for Current Hour
        current_hour_live_rate, _ = self.forecast.get_plan_for_hour(
            current_time,
            source='live',
            ignore_aux=(not self.auxiliary_heating_active)
        )

        # 2. Calculate remaining portion of current hour (Forecast)
        remaining_fraction = max(0.0, (60 - minutes_passed) / 60.0)
        remaining_live_current_hour = current_hour_live_rate * remaining_fraction

        # 3. Sum parts
        # expected_today_total includes expected_hour_so_far (Actuals)
        thermodynamic_projection = expected_today_total + remaining_live_current_hour + future_forecast_kwh
        self.data["thermodynamic_projection_kwh"] = round(thermodynamic_projection, 2)

        # Calculate Thermodynamic Deviation
        # actual_today vs expected_today_total (Actual So Far vs Model So Far)
        # Note: Future components cancel out when projecting End of Day.
        actual_today = self._accumulated_energy_today
        thermo_deviation = actual_today - expected_today_total
        self.data["thermodynamic_deviation_kwh"] = round(thermo_deviation, 2)

        if thermodynamic_projection > ENERGY_GUARD_THRESHOLD:
            self.data["thermodynamic_deviation_pct"] = round((thermo_deviation / thermodynamic_projection) * 100, 1)
        else:
            self.data["thermodynamic_deviation_pct"] = 0.0

        # Calculate Solar Budget
        current_solar_kw = self.data.get(ATTR_SOLAR_IMPACT, 0.0)
        solar_budget_total = solar_today_sum + current_solar_kw + future_solar_kwh
        self.data[ATTR_SOLAR_PREDICTED] = round(solar_budget_total, 2)

        # Calculate Forecast Today (Projected Consumption)
        # This is Actual Consumption So Far + Live Forecast for the rest of the day.
        # This calculation is now mode-aware (respects aux heating) and replaces the
        # previous logic that was anchored to an aux-unaware midnight snapshot, which
        # caused incorrect deviation reporting.

        # FIX: Use `max(0, forecast - actual)` instead of `forecast * remaining_fraction`
        # to prevent "Double Accounting Risk" where actuals > planned rate for the elapsed time
        # causes a phantom spike in the daily total.
        # Note: self._accumulated_energy_today ALREADY includes the current hour's actuals.
        # So we add the "Remaining Budget" for the current hour.

        # We need actuals specifically for THIS hour to subtract from the hourly plan.
        actual_this_hour = self._collector.energy_hour

        # Smart Merge: Check for partial log data (pre-restart)
        partial_log = self._get_partial_log_for_current_hour()
        if partial_log:
            actual_this_hour += partial_log.get("actual_kwh", 0.0)

        remaining_current_hour = max(0.0, current_hour_plan_rate - actual_this_hour)

        forecast_today_val = (
            self._accumulated_energy_today  # Includes actual_this_hour
            + remaining_current_hour        # The part of the hourly plan we haven't "spent" yet
            + future_forecast_kwh
        )
        self.data[ATTR_FORECAST_TODAY] = round(forecast_today_val, 2)

        # Calculate Gross Forecast Today (Thermodynamic Demand - Aux Unaware)
        # Used for Thermal Stress Index to avoid "Mild Weather Fallacy" when Aux is active.

        # 1. Gross Future
        future_gross_kwh, _, _ = self.forecast.calculate_future_energy(
            current_time, ignore_aux=True
        )

        # 2. Gross Current Hour Plan
        # We need the full hour gross plan to calculate remaining

        # OPTIMIZATION: Only recalculate if aux is active. If inactive, Gross == Net,
        # and we can reuse current_hour_plan_rate (already calculated above).
        if self.auxiliary_heating_active:
            # Calculate strictly from Reference Forecast (Aux Ignored)
            current_hour_gross_rate, _ = self.forecast.get_plan_for_hour(
                current_time,
                source='reference',
                ignore_aux=True
            )

            # Fallback if forecast missing
            if current_hour_gross_rate == 0.0:
                 # Fallback: If no forecast for current hour, use the live model rate (Net).
                 # We must restore the Aux component to get the GROSS rate for the Stress Index.
                 current_hour_gross_rate = current_hour_plan_rate # Default to Net
                 current_aux_rate = self.data.get("current_aux_impact_rate", 0.0)
                 if current_aux_rate > 0:
                     current_hour_gross_rate += current_aux_rate
        else:
            # Optimization: When aux is inactive, Gross == Net (no need to recalculate)
            current_hour_gross_rate = current_hour_plan_rate

        # 3. Gross Actuals So Far (Net + Aux)
        # accumulated_aux_impact_kwh is updated just before this method call
        gross_actuals_so_far = self._accumulated_energy_today + self.data.get("accumulated_aux_impact_kwh", 0.0)

        # 4. Gross Remaining Current Hour
        # gross_actual_this_hour = net_actual_hour + aux_actual_hour
        # Note: We must include partial logs if any
        gross_actual_this_hour = self._collector.energy_hour + self._collector.aux_impact_hour

        if partial_log:
             gross_actual_this_hour += partial_log.get("actual_kwh", 0.0)
             gross_actual_this_hour += partial_log.get("aux_impact_kwh", 0.0)

        remaining_gross_current = max(0.0, current_hour_gross_rate - gross_actual_this_hour)

        forecast_gross_val = (
            gross_actuals_so_far
            + remaining_gross_current
            + future_gross_kwh
        )
        self.data["forecast_today_gross"] = round(forecast_gross_val, 2)

        # Calculate Per-Unit Forecast Today
        # Formula: Actual So Far + Remaining Current Hour (Forecast) + Future (Forecast)
        unit_forecast_today = {}
        for entity_id in self.energy_sensors:
            # 1. Actual So Far
            actual_total = self._daily_individual.get(entity_id, 0.0)

            # 2. Remaining Current Hour
            current_unit_data = current_hour_unit_breakdown.get(entity_id, {})
            current_unit_forecast_kwh = current_unit_data.get("net_kwh", 0.0) if current_unit_data else 0.0

            actual_unit_this_hour = self._hourly_delta_per_unit.get(entity_id, 0.0)
            remaining_unit = max(0.0, current_unit_forecast_kwh - actual_unit_this_hour)

            # 3. Future
            future_unit = future_unit_totals.get(entity_id, 0.0)

            # Note: actual_total includes actual_unit_this_hour.
            # So Total = (Actual_Prev + Actual_Now) + (Forecast_Hour - Actual_Now) + Future
            # If Actual_Now < Forecast_Hour => Total = Actual_Prev + Forecast_Hour + Future.
            # If Actual_Now > Forecast_Hour => Total = Actual_Prev + Actual_Now + Future.
            total_unit = actual_total + remaining_unit + future_unit
            unit_forecast_today[entity_id] = round(total_unit, 2)

        self.data["forecast_today_per_unit"] = unit_forecast_today

        # Calculate Efficiency Forecast Today
        tdd_forecast = self.data.get(ATTR_TDD_DAILY_STABLE, 0.0)
        if tdd_forecast > ENERGY_GUARD_THRESHOLD:
            self.data[ATTR_EFFICIENCY_FORECAST_TODAY] = round(forecast_today_val / tdd_forecast, 3)
        else:
            self.data[ATTR_EFFICIENCY_FORECAST_TODAY] = None

        # Calculate Forecast Uncertainty
        stats = self.forecast.calculate_uncertainty_stats()
        self.data[ATTR_FORECAST_UNCERTAINTY] = stats

        day_progress = (current_time.hour + (current_time.minute / 60.0)) / 24.0
        remaining_day_fraction = max(0.0, 1.0 - day_progress)
        max_dev = stats.get("p95_abs_error", 2.0)
        uncertainty_margin = max_dev * remaining_day_fraction

        self.data["confidence_interval_margin"] = round(uncertainty_margin, 2)
        self.data["confidence_interval_lower"] = round(forecast_today_val - uncertainty_margin, 2)
        self.data["confidence_interval_upper"] = round(forecast_today_val + uncertainty_margin, 2)

        # Calculate Daily Wind Chill Penalty (Full Day Impact)
        self.data["daily_wind_chill_penalty"] = self._calculate_daily_wind_penalty()

        # Per-Source Accuracy & Blend Config (New Unified Attribute)
        self.data[ATTR_FORECAST_DETAILS] = {
            "blend_config": {
                "primary_entity_id": self.weather_entity,
                "secondary_entity_id": self.entry.data.get(CONF_SECONDARY_WEATHER_ENTITY),
                "crossover_day": self.entry.data.get(CONF_FORECAST_CROSSOVER_DAY, DEFAULT_FORECAST_CROSSOVER_DAY)
            },
            "accuracy_by_source": {
                source: {"daily": stats["daily"]}
                for source, stats in self.forecast.calculate_per_source_uncertainty_stats().items()
            }
        }

    def _update_tdd_calculations(self, temp: float | None, minutes_passed: int):
        """Update TDD calculations for the current partial hour."""
        current_tdd_acc = self.data.get(ATTR_TDD, 0.0)

        if temp is not None:
             current_hour_tdd = (abs(self.balance_point - temp) / 24.0) * (minutes_passed / 60.0)
             total_tdd_so_far = current_tdd_acc + current_hour_tdd
        else:
             total_tdd_so_far = current_tdd_acc

        self.data[ATTR_TDD_SO_FAR] = round(total_tdd_so_far, 3)

    def _update_deviation_stats(self):
        """Update deviation statistics."""
        # Deviation (Today)
        # Gross-domain comparison cancels live forecast drift from both sides of diff.
        # Frozen midnight_forecast denominator prevents the percentage from scaling
        # with weather forecast revisions.
        predicted_gross = self.data.get("predicted_gross", 0.0)
        forecast_today_gross = self.data.get("forecast_today_gross", 0.0)
        midnight = self.data.get(ATTR_MIDNIGHT_FORECAST, 0.0)
        denominator = midnight if midnight > ENERGY_GUARD_THRESHOLD else predicted_gross

        if denominator > ENERGY_GUARD_THRESHOLD:
             diff = forecast_today_gross - predicted_gross
             deviation = (diff / denominator) * 100
             self.data[ATTR_DEVIATION] = round(deviation, 1)
        else:
             self.data[ATTR_DEVIATION] = 0.0

        # Deviation Breakdown
        self.data[ATTR_DEVIATION_BREAKDOWN] = self.statistics.calculate_deviation_breakdown()

        # Plan Revision Impact (formerly Weather Forecast Deviation)
        self.data["plan_revision_impact"] = self.forecast.calculate_plan_revision_impact()

        # Weather-Adjusted Deviation
        weather_dev_data = self.data.get("plan_revision_impact", {})
        weather_impact_kwh = weather_dev_data.get("estimated_impact_kwh", 0.0)

        actual_kwh = self.data.get(ATTR_ENERGY_TODAY, 0.0)
        expected_kwh = self.data.get(ATTR_EXPECTED_TODAY, 0.0)
        current_deviation_kwh = actual_kwh - expected_kwh
        weather_adjusted_kwh = current_deviation_kwh + weather_impact_kwh

        weather_adjusted_pct = 0.0
        if expected_kwh > ENERGY_GUARD_THRESHOLD:
            weather_adjusted_pct = (weather_adjusted_kwh / expected_kwh) * 100

        if abs(weather_adjusted_kwh) < 0.1:
            adjusted_explanation = "Forecast Accuracy Adjustment: Consumption matches the model perfectly ✨"
        elif weather_adjusted_kwh > 0:
            adjusted_explanation = (
                f"Forecast Accuracy Adjustment: You are using {weather_adjusted_kwh:+.1f} kWh "
                f"({weather_adjusted_pct:+.1f}%) more than the model expects 📈"
            )
        else:
            adjusted_explanation = (
                f"Forecast Accuracy Adjustment: You are using {abs(weather_adjusted_kwh):.1f} kWh "
                f"({abs(weather_adjusted_pct):.1f}%) less than the model expects 📉"
            )

        self.data["weather_adjusted_deviation"] = {
            "deviation_kwh": round(weather_adjusted_kwh, 2),
            "deviation_pct": round(weather_adjusted_pct, 1),
            "current_deviation_kwh": round(current_deviation_kwh, 2),
            "weather_impact_kwh": round(weather_impact_kwh, 2),
            "explanation": adjusted_explanation
        }

    async def _async_update_data(self):
        """Update data."""
        if not self._is_loaded:
             await self._async_load_data()

        current_time = dt_util.now()

        # Initialize trackers if this is the first run
        if self._accumulation_start_time is None:
            self._accumulation_start_time = current_time
        if self._last_day_processed is None:
            self._last_day_processed = current_time.date()
        if self._last_hour_processed is None:
            self._last_hour_processed = current_time.hour

        # --- Boundary Processing ---
        # First, handle hour/day rollovers to ensure all historical data is up-to-date
        # before any new calculations for the current period are made.
        # This resolves the midnight race condition.
        completed_hour = current_time.hour != self._last_hour_processed
        if completed_hour:
            await self._process_hourly_data(current_time)
            # Recalculate stats that depend on hourly_log (moved from per-minute execution)
            self.statistics.calculate_temp_stats()
            self.statistics.update_daily_savings_cache()
            _LOGGER.debug("Stats recalculated after hourly processing")

            self._last_hour_processed = current_time.hour
            self._accumulation_start_time = current_time

        completed_day = current_time.date() != self._last_day_processed
        if completed_day:
            if self._last_day_processed is not None:
                # Process the completed day
                await self._process_daily_data(self._last_day_processed)
            self._last_day_processed = current_time.date()

        # --- Track C pre-midnight snapshot (#855 follow-up) ---
        # Three attempts per evening so the midnight sync has a usable
        # fallback when the live MPC call fails.  See _track_c_snapshot
        # docstring on __init__ for rationale.
        if self.track_c_enabled and self.daily_learning_mode:
            await self._maybe_snapshot_track_c(current_time)

        # --- Forecast Update ---
        # Now that rollovers are handled, update the forecast.
        # This ensures the new day's snapshot is created with complete inertia data.
        # Update is forced on boundaries, otherwise periodic.
        if completed_hour or completed_day or current_time.minute % 15 == 0 or self.data.get(ATTR_TEMP_FORECAST_TODAY) is None:
            await self.forecast.update_daily_forecast()


        # --- Live Data Processing ---

        # 1. Fetch Sensor Data (Strict Source Logic)

        # Temp
        temp = None
        if self.outdoor_temp_source == SOURCE_SENSOR:
             if self.outdoor_temp_sensor:
                 temp = self._get_float_state(self.outdoor_temp_sensor)
                 if temp is None and self.hass.is_running:
                     _LOGGER.warning(f"Outdoor Temp Sensor '{self.outdoor_temp_sensor}' is unavailable.")
        else:
             temp = self._get_weather_attribute("temperature")

        # Wind Speed
        wind_speed = None
        if self.wind_speed_source == SOURCE_SENSOR:
             if self.wind_speed_sensor:
                 wind_speed = self._get_speed_in_ms(self.wind_speed_sensor)
                 if wind_speed is None and self.hass.is_running:
                      _LOGGER.warning(f"Wind Speed Sensor '{self.wind_speed_sensor}' is unavailable.")
        else:
             wind_speed = self._get_weather_attribute("wind_speed")

        # Wind Gust
        wind_gust = None
        if self.wind_gust_source == SOURCE_SENSOR:
             if self.wind_gust_sensor:
                 wind_gust = self._get_speed_in_ms(self.wind_gust_sensor)
                 if wind_gust is None and self.hass.is_running:
                     # Gusts are often intermittent, debug level.  is_running
                     # guard suppresses the spurious startup log before
                     # entities are available (CLAUDE.md Startup warnings).
                     _LOGGER.debug(f"Wind Gust Sensor '{self.wind_gust_sensor}' is unavailable.")
        else:
             wind_gust = self._get_weather_attribute("wind_gust_speed")

        # Fetch Humidity (for Track C per-hour COP / defrost penalty)
        humidity = self._get_weather_attribute("humidity")
        if humidity is None and self.track_c_enabled and self.hass.is_running:
            now = dt_util.now()
            last_warn = getattr(self, '_humidity_warn_time', None)
            if last_warn is None or (now - last_warn).total_seconds() >= 3600:
                _LOGGER.warning(
                    "Weather entity '%s' does not provide humidity data. "
                    "Track C per-hour COP defrost penalty will use 50%% default, "
                    "which may over- or under-trigger defrost compensation.",
                    self.weather_entity,
                )
                self._humidity_warn_time = now

        # Fetch Solar Data
        potential_solar_factor = 0.0
        solar_factor = 0.0
        solar_vector = (0.0, 0.0, 0.0)
        if self.solar_enabled:
             elev, azim = self._get_sun_info_now()
             cloud = self._get_cloud_coverage()
             potential_solar_factor = self.solar.calculate_solar_factor(elev, azim, cloud)
             solar_factor = self.solar.calculate_effective_solar_factor(
                 potential_solar_factor, self.solar_correction_percent
             )
             self.data[ATTR_SOLAR_FACTOR] = round(solar_factor, 2)

             potential_solar_vector = self.solar.calculate_solar_vector(elev, azim, cloud)
             solar_vector = self.solar.calculate_effective_solar_vector(
                 potential_solar_vector, self.solar_correction_percent
             )

             # Expose current effective solar vector for fallback projection
             self.data["solar_vector_s"] = round(solar_vector[0], 4)
             self.data["solar_vector_e"] = round(solar_vector[1], 4)
             self.data["solar_vector_w"] = round(solar_vector[2], 4)

             if ATTR_SOLAR_IMPACT not in self.data:
                 self.data[ATTR_SOLAR_IMPACT] = 0.0

        if wind_speed is None:
            wind_speed = 0.0
            if self.hass.is_running:
                now = dt_util.now()
                last_warn = getattr(self, '_wind_speed_warn_time', None)
                if last_warn is None or (now - last_warn).total_seconds() >= 3600:
                    _LOGGER.warning(
                        "Wind speed data unavailable from configured source. "
                        "Defaulting to 0.0 m/s — all hours will land in the 'normal' "
                        "wind bucket, disabling wind differentiation in the model."
                    )
                    self._wind_speed_warn_time = now

        # Calculate Inertia & Temp Keys EARLY (Moved from Step 4)
        # We need these for Solar Impact calculation and other model lookups
        inertia_list = self._get_inertia_list(current_time)
        inertia_temp = None
        if inertia_list:
             inertia_temp = self._calculate_weighted_inertia(inertia_list)

        calc_temp = inertia_temp if inertia_temp is not None else temp
        temp_key = str(int(round(calc_temp))) if calc_temp is not None else "0"

        # Calculate Potential Solar Impact (Screens Up) for Sensors
        potential_impact_kw = 0.0
        if self.solar_enabled and temp_key:
             # Calculate what impact WOULD be if correction was 100%
             # We use the same unit coefficients as actual calculation
             total_potential = 0.0
             for entity_id in self.energy_sensors:
                 unit_coeff = self.solar.calculate_unit_coefficient(
                     entity_id, temp_key, self.get_unit_mode(entity_id)
                 )
                 total_potential += self.solar.calculate_unit_solar_impact(potential_solar_vector, unit_coeff)
             potential_impact_kw = total_potential

        self.data[ATTR_SOLAR_POTENTIAL] = round(potential_impact_kw, 3)

        # Calculate Recommendation State
        rec_state = "none"
        if self.solar_enabled and temp is not None:
             rec_state = self.solar_optimizer.get_recommendation_state(temp, potential_solar_factor)

        self.data[ATTR_RECOMMENDATION_STATE] = rec_state
        self.data["potential_solar_factor"] = round(potential_solar_factor, 3)

        # 2. Calculate Effective Wind & Conditions (if temp is available)
        if temp is not None:
            effective_wind = self._calculate_effective_wind(wind_speed, wind_gust)
            wind_bucket = self._get_wind_bucket(effective_wind)
            # Cache effective wind for sensors
            self.data["effective_wind"] = effective_wind

            # Update hourly aggregates (delegated to ObservationCollector)
            self._collector.accumulate_weather(
                temp=temp,
                effective_wind=effective_wind,
                wind_bucket=wind_bucket,
                solar_factor=solar_factor,
                solar_vector=solar_vector,
                is_aux_active=self.auxiliary_heating_active,
                current_time=current_time,
                humidity=humidity,
                correction_percent=self.solar_correction_percent,
            )
        else:
            effective_wind = 0.0
            wind_bucket = "normal"
            if (self.outdoor_temp_sensor or self.weather_entity) and self.hass.is_running:
                _LOGGER.warning(
                    "Unable to retrieve outdoor temperature. Sensor '%s' and Weather Fallback '%s' both failed.",
                    self.outdoor_temp_sensor,
                    self.weather_entity
                )

        # 3. Energy Tracking (Robust Delta Calculation)
        hourly_delta = 0.0
        for entity_id in self.energy_sensors:
            val = self._get_float_state(entity_id)
            if val is not None:
                if entity_id in self._last_energy_values:
                    prev_val = self._last_energy_values[entity_id]
                    delta = val - prev_val

                    if delta < 0:
                        if self.hass.is_running:
                            _LOGGER.warning(f"Energy meter reset detected for {entity_id}: {prev_val} -> {val}. Skipping this reading.")
                        self._last_energy_values[entity_id] = val
                        continue

                    if delta > self.max_energy_delta:
                        if self.hass.is_running:
                            _LOGGER.warning(f"Energy spike detected for {entity_id}: {delta:.2f} kWh > {self.max_energy_delta} kWh. Skipping this reading and updating baseline.")
                        self._last_energy_values[entity_id] = val
                        continue

                    # DHW mode: energy goes to hot water tank, not space heating.
                    # Exclude from all heating totals. _last_energy_values is still
                    # updated below so the delta baseline is correct when mode switches back.
                    if self.get_unit_mode(entity_id) == MODE_DHW:
                        pass
                    else:
                        hourly_delta += delta

                        if entity_id not in self._daily_individual:
                            self._daily_individual[entity_id] = 0.0
                        self._daily_individual[entity_id] += delta

                        if entity_id not in self._hourly_delta_per_unit:
                            self._hourly_delta_per_unit[entity_id] = 0.0
                        self._hourly_delta_per_unit[entity_id] += delta

                        if self.enable_lifetime_tracking:
                            if entity_id not in self._lifetime_individual:
                                self._lifetime_individual[entity_id] = 0.0
                            self._lifetime_individual[entity_id] += delta

                self._last_energy_values[entity_id] = val

        self.data["daily_individual"] = self._daily_individual
        self.data["lifetime_individual"] = self._lifetime_individual

        self._collector.energy_hour += hourly_delta
        self._accumulated_energy_today += hourly_delta
        self.data[ATTR_ENERGY_TODAY] = round(self._accumulated_energy_today, 3)

        # 4. Live Prediction / Efficiency Display
        # (inertia_list and inertia_temp calculated earlier in Step 1)

        # Calculate Thermal State Attributes
        # Normalize weights to the active subset so they sum to 1.0,
        # allowing direct verification: sum(history[i] * weights[i]) == effective_temperature.
        _active_w = list(self.inertia_weights[-len(inertia_list):]) if inertia_list else []
        _w_sum = sum(_active_w)
        _display_weights = [round(w / _w_sum, 4) for w in _active_w] if _w_sum > 0 else _active_w
        thermal_state = {
             "raw_temperature": round(temp, 1) if temp is not None else None,
             "effective_temperature": round(inertia_temp, 1) if inertia_temp is not None else None,
             "inertia_history": [round(t, 1) for t in inertia_list],
             "weights": _display_weights,
             "samples_used": len(inertia_list),
             "last_updated": current_time.isoformat(),
             "balance_point": self.balance_point,
        }

        if temp is not None and inertia_temp is not None:
             lag = inertia_temp - temp
             thermal_state["thermal_lag"] = round(lag, 1)

             if lag > 0.5:
                 thermal_state["lag_status"] = "Shedding Heat" # House is warmer than outside
             elif lag < -0.5:
                 thermal_state["lag_status"] = "Gaining Heat" # House is colder than outside
             else:
                 thermal_state["lag_status"] = "Balanced"
        else:
             thermal_state["thermal_lag"] = None
             thermal_state["lag_status"] = "Unknown"

        # Extended State: Regime, Balance, and Instant TDD
        if inertia_temp is not None:
             # Regime (Cold vs Mild) - Matches statistics.py logic
             tdd_target = abs(self.balance_point - inertia_temp)
             if tdd_target > 4.0:
                 thermal_state["regime"] = "Cold"
                 thermal_state["regime_explanation"] = "Using Ratio Extrapolation (Physics-dominated)"
             else:
                 thermal_state["regime"] = "Mild"
                 thermal_state["regime_explanation"] = "Using Nearest Neighbor (Noise-dominated)"

             # Degrees to Balance
             diff = self.balance_point - inertia_temp
             thermal_state["degrees_to_balance"] = round(diff, 1)

             # Instant TDD
             thermal_state["instant_tdd"] = round(tdd_target / 24.0, 3)
        else:
             thermal_state["regime"] = None
             thermal_state["regime_explanation"] = None
             thermal_state["degrees_to_balance"] = None
             thermal_state["instant_tdd"] = None

        # Trend Calculation
        if len(inertia_list) >= 2:
             # Look back up to 2 hours if possible (3 points: Current, H-1, H-2)
             # List is [oldest, ..., newest]
             # If len=4: [H-3, H-2, H-1, Current]
             # Compare Current (index -1) with H-2 (index -3)
             hours_back = 0
             old_val = inertia_list[-1]

             if len(inertia_list) >= 3:
                  old_val = inertia_list[-3]
                  hours_back = 2
             elif len(inertia_list) == 2:
                  old_val = inertia_list[-2]
                  hours_back = 1

             diff = inertia_list[-1] - old_val
             rate = diff / hours_back if hours_back > 0 else 0.0
             thermal_state["trend_rate"] = round(rate, 2)

             if rate < -2.0: trend = "falling_fast"
             elif rate < -0.5: trend = "falling"
             elif rate > 2.0: trend = "rising_fast"
             elif rate > 0.5: trend = "rising"
             else: trend = "stable"
             thermal_state["temperature_trend"] = trend
        else:
             thermal_state["trend_rate"] = None
             thermal_state["temperature_trend"] = "unknown"

        self.data["thermal_state"] = thermal_state

        # (calc_temp and temp_key calculated earlier in Step 1)
        minutes_passed = current_time.minute

        # 4a. Update Live Predictions & Accumulation
        current_prediction_rate = self._update_live_predictions(
            calc_temp, temp_key, wind_bucket, current_time
        )

        # 4b. Update Accumulated Impacts (Moved up to support Daily Budgets)
        self._update_accumulated_impacts(current_time)

        # 4c. Update Daily Budgets & Forecasts
        self._update_daily_budgets(
            current_prediction_rate, current_time, minutes_passed
        )

        # 4d. Update TDD Calculations
        self._update_tdd_calculations(temp, minutes_passed)

        # 4e. Update Deviation Stats
        self._update_deviation_stats()
        self.statistics.calculate_potential_savings()

        return self.data

    def _update_accumulated_impacts(self, current_time: datetime):
        """Calculate and update the daily accumulated impacts for solar, guest, and aux modes."""
        today_date_str = current_time.date().isoformat()

        # 1. Sum impacts from today's completed hourly logs
        accumulated_solar = 0.0
        accumulated_guest = 0.0
        accumulated_aux = 0.0

        for entry in self._hourly_log:
            if entry["timestamp"].startswith(today_date_str):
                accumulated_solar += entry.get("solar_impact_kwh", 0.0)
                accumulated_guest += entry.get("guest_impact_kwh", 0.0)
                accumulated_aux += entry.get("aux_impact_kwh", 0.0)

        # 2. Add live, intra-hour impact
        #    - Solar: Use the current `ATTR_SOLAR_IMPACT` (in kW) and scale by minutes passed.
        #    - Aux: Use `last_hour_aux_impact_kwh` as a rate (kW) and scale.
        #    - Guest: Calculate live guest impact.
        minutes_fraction = current_time.minute / 60.0

        # Live Solar Impact
        # NOTE: self.data["solar_impact_kwh"] stores instantaneous power in kW due to a legacy naming convention.
        current_solar_kw = self.data.get(ATTR_SOLAR_IMPACT, 0.0)
        accumulated_solar += current_solar_kw * minutes_fraction

        # Live Aux Impact
        # Use the precise minute-by-minute accumulator
        accumulated_aux += self._collector.aux_impact_hour

        # Live Guest Impact
        # Guest units are not tracked in expected - their full consumption is the impact
        live_guest_impact = 0.0
        for entity_id, actual_kwh in self._hourly_delta_per_unit.items():
            unit_mode = self.get_unit_mode(entity_id)
            if unit_mode in (MODE_GUEST_HEATING, MODE_GUEST_COOLING):
                live_guest_impact += actual_kwh
        accumulated_guest += live_guest_impact

        # 3. Update the data dictionary
        self.data["accumulated_solar_impact_kwh"] = round(accumulated_solar, 3)
        self.data["accumulated_guest_impact_kwh"] = round(accumulated_guest, 3)
        self.data["accumulated_aux_impact_kwh"] = round(accumulated_aux, 3)

        # 4. Calculate Thermodynamic Gross Today (Net + Aux +/- Solar)
        accumulated_gross = 0.0
        for entry in self._hourly_log:
             if entry["timestamp"].startswith(today_date_str):
                 accumulated_gross += entry.get("thermodynamic_gross_kwh", 0.0)

        # Add live component
        # We need the current temp context to decide solar sign
        # We use the LAST KNOWN calc_temp stored in self.data (from _update_live_predictions)
        current_calc_temp = self.data.get("current_calc_temp")
        if current_calc_temp is None:
             # Fallback if unknown
             current_calc_temp = self.balance_point

        live_net = self._collector.energy_hour
        live_aux = self._collector.aux_impact_hour
        live_solar_impact = current_solar_kw * minutes_fraction

        live_gross = live_net + live_aux
        if current_calc_temp >= self.balance_point:
             live_gross -= live_solar_impact
        else:
             live_gross += live_solar_impact

        self.data["thermodynamic_gross_today_kwh"] = round(accumulated_gross + live_gross, 3)

    def _is_model_covered(self, temp_key: str, wind_bucket: str) -> bool:
        """Check if exact model data exists for this temp/wind combo."""
        if temp_key in self._correlation_data:
            # Check exact bucket first
            if wind_bucket in self._correlation_data[temp_key]:
                return self._correlation_data[temp_key][wind_bucket] > 0
            # "Covered" = has learned data (exact or fallback bucket), no TDD extrapolation
            if wind_bucket == "extreme_wind" and "high_wind" in self._correlation_data[temp_key]:
                return True
            if wind_bucket in ("extreme_wind", "high_wind") and "normal" in self._correlation_data[temp_key]:
                return True
        return False

    def _get_predicted_kwh(self, temp_key: str, wind_bucket: str, actual_temp: float) -> float:
        """Get predicted kWh for a given temp and wind using the centralized robust model."""
        return self.statistics._get_prediction_from_model(
            self._correlation_data, temp_key, wind_bucket, actual_temp, self.balance_point
        )

    def _get_predicted_kwh_per_unit(self, entity_id: str, temp_key: str, wind_bucket: str, actual_temp: float) -> float:
        """Get predicted kWh for a given unit, temp and wind using the centralized robust model.

        Per #885: cooling-mode units read from the dedicated "cooling"
        wind-bucket regardless of the wind_bucket passed in.  Already-
        cooling requests pass through unchanged.
        """
        unit_data = self._correlation_data_per_unit.get(entity_id, {})
        if self.get_unit_mode(entity_id) == MODE_COOLING:
            wind_bucket = COOLING_WIND_BUCKET
        return self.statistics._get_prediction_from_model(
            unit_data, temp_key, wind_bucket, actual_temp, self.balance_point
        )

    def _get_aux_impact_kw(self, temp_key: str, wind_bucket: str = "normal", actual_temp: float = 0.0) -> float:
        """Get the Aux Impact (kW reduction) for a given temp key and wind bucket."""
        # Note: actual_temp is only needed if scaling were applied, which it is not for aux models.
        # We pass it to satisfy the function signature.
        if actual_temp == 0.0:
            try:
                actual_temp = float(temp_key)
            except ValueError:
                actual_temp = self.balance_point # A safe default if temp_key is invalid

        return self.statistics._get_prediction_from_model(
            self._aux_coefficients, temp_key, wind_bucket, actual_temp, self.balance_point, apply_scaling=False
        )

    def _get_aux_impact_kw_per_unit(self, entity_id: str, temp_key: str, wind_bucket: str = "normal", actual_temp: float = 0.0) -> float | None:
        """Get the Aux Impact (kW reduction) for a given unit, temp key, and wind bucket."""
        unit_data = self._aux_coefficients_per_unit.get(entity_id, {})
        if not unit_data:
            return None

        if actual_temp == 0.0:
            try:
                actual_temp = float(temp_key)
            except ValueError:
                actual_temp = self.balance_point # A safe default if temp_key is invalid

        return self.statistics._get_prediction_from_model(
            unit_data, temp_key, wind_bucket, actual_temp, self.balance_point, apply_scaling=False
        )

    def _get_unit_observation_count(
        self,
        entity_id: str,
        temp_key: str,
        wind_bucket: str,
        mode: str | None = None,
    ) -> int:
        """Get observation count for a unit/temp/bucket.

        This count represents the number of learning cycles (hours) where this specific
        combination of conditions (Temperature + Wind) has been observed and used to
        train the model for this specific unit.

        Per #885: cooling-mode units count against the dedicated "cooling"
        wind-bucket (mirrors the write-side routing in
        learning._process_per_unit_learning).

        Historical callers (evaluating a past log entry) must pass ``mode``
        explicitly — set to the unit's mode AT THE TIME that entry was
        logged — so the count is read from the bucket the sample was
        actually written to.  Without ``mode``, the helper falls back to
        the unit's CURRENT mode, which is correct for live / current-hour
        callers but wrong for historical evaluation of a mode-switching
        unit (#xxx).
        """
        effective_mode = mode if mode is not None else self.get_unit_mode(entity_id)
        if effective_mode == MODE_COOLING:
            wind_bucket = COOLING_WIND_BUCKET
        if entity_id in self._observation_counts:
            if temp_key in self._observation_counts[entity_id]:
                return self._observation_counts[entity_id][temp_key].get(wind_bucket, 0)
        return 0

    def calculate_unit_rolling_power_watts(self, entity_id: str) -> int:
        """Calculate a 60-minute rolling average power (Watts) for a specific unit.

        Uses the actual energy consumed in the current partial hour plus a
        proportional part of the previous hour's total consumption.
        This provides a continuous, non-zero power value that correctly reflects
        the last 60 minutes of activity.
        """
        now = dt_util.now()
        minutes_passed = now.minute
        fraction_passed = minutes_passed / 60.0

        # 1. Actual energy so far this hour
        kwh_now = self._hourly_delta_per_unit.get(entity_id, 0.0)

        # 2. Actual energy from the last completed hour
        kwh_last = 0.0
        if self._hourly_log:
            last_entry = self._hourly_log[-1]
            kwh_last = last_entry.get("unit_breakdown", {}).get(entity_id, 0.0)

        # 3. Combine to form a 60-minute window
        # (e.g. at :15, we take 15 mins from now + 45 mins from last hour)
        rolling_kwh = kwh_now + (kwh_last * (1.0 - fraction_passed))

        # 4. Convert kWh/h to Watts
        return int(round(rolling_kwh * 1000))

    def calculate_modeled_energy(self, start_date: date, end_date: date, pre_fetched_logs: dict | None = None) -> tuple[float, float, float | None, float | None, float]:
        """Calculate modeled energy for a date range."""
        return self.statistics.calculate_modeled_energy(start_date, end_date, pre_fetched_logs)

    async def async_compare_periods(self, p1_start: date, p1_end: date, p2_start: date, p2_end: date):
        """Handle the compare_periods service call."""
        _LOGGER.info(f"Comparing periods: {p1_start}-{p1_end} vs {p2_start}-{p2_end}")

        comparison_data = self.statistics.compare_periods(
            p1_start, p1_end, p2_start, p2_end
        )

        self.data["last_comparison"] = comparison_data

        # Trigger an update for the new sensor
        self.async_set_updated_data(self.data)

    # --- Data Contract Factory Methods (Issue #775) ---

    def _build_hourly_observation(self, *args, **kwargs):
        """Delegates to :class:`hourly_processor.HourlyProcessor`."""
        return self._hourly_processor.build_observation(*args, **kwargs)

    @property
    def model(self) -> "ModelState":
        """ModelState view holding live references to learned model data.

        Returns a fresh ModelState each call.  The instance holds
        references (not copies) to the coordinator's canonical dicts,
        so reading ``self.model.correlation_data`` is equivalent to
        reading ``self._correlation_data`` — but makes the dependency
        explicit and eliminates private-field access from external
        modules.

        No caching.  If a coordinator field is reassigned (e.g. during
        storage load) the next ``.model`` access picks up the new
        reference, so the old aliasing footgun that required
        in-place mutation of the underlying dicts is gone.
        """
        return self.get_model_state()

    def get_model_state(self) -> "ModelState":
        """Return a snapshot of the current learned model state.

        Returns references (not deep copies) for performance.
        Consumers must treat the returned object as read-only.
        """
        from .observation import ModelState

        return ModelState(
            correlation_data=self._correlation_data,
            correlation_data_per_unit=self._correlation_data_per_unit,
            observation_counts=self._observation_counts,
            aux_coefficients=self._aux_coefficients,
            aux_coefficients_per_unit=self._aux_coefficients_per_unit,
            solar_coefficients_per_unit=self._solar_coefficients_per_unit,
            learned_u_coefficient=self._learned_u_coefficient,
            learning_buffer_global=self._learning_buffer_global,
            learning_buffer_per_unit=self._learning_buffer_per_unit,
            learning_buffer_aux_per_unit=self._learning_buffer_aux_per_unit,
            learning_buffer_solar_per_unit=self._learning_buffer_solar_per_unit,
            daily_history=self._daily_history,
            hourly_log=self._hourly_log,
        )

    # ------------------------------------------------------------------
    # Public accessors for runtime state consumed by manager modules.
    # Managers (StatisticsManager, ForecastManager) read these instead of
    # reaching into ``coordinator._X`` — keeping the manager/coordinator
    # boundary explicit.  Model-state fields are exposed via ``model``
    # (ModelState) above; these accessors cover non-model runtime state
    # (collector accumulators, daily aggregates, aux cooldown flags).
    # ------------------------------------------------------------------

    @property
    def collector(self) -> ObservationCollector:
        """ObservationCollector holding the current hour's accumulators."""
        return self._collector

    @property
    def daily_individual(self) -> dict:
        """Per-sensor running daily kWh totals."""
        return self._daily_individual

    @property
    def aux_cooldown_active(self) -> bool:
        """True while the aux-heating cooldown window is active."""
        return self._aux_cooldown_active

    @property
    def aux_affected_set(self) -> set:
        """Set of entity IDs flagged as aux-affected (derived from config)."""
        return self._aux_affected_set

    async def _process_hourly_data(self, current_time: datetime):
        """Delegates to :class:`hourly_processor.HourlyProcessor`."""
        return await self._hourly_processor.process(current_time)

    def _aggregate_daily_logs(self, *args, **kwargs):
        """Delegates to :class:`daily_processor.DailyProcessor`."""
        return self._daily_processor.aggregate_logs(*args, **kwargs)

    def _backfill_daily_from_hourly(self, *args, **kwargs):
        """Delegates to :class:`daily_processor.DailyProcessor`."""
        return self._daily_processor.backfill_from_hourly(*args, **kwargs)

    async def _fetch_mpc_buffer_and_cop(self, *args, **kwargs):
        """Delegates to :class:`daily_processor.DailyProcessor`."""
        return await self._daily_processor.fetch_mpc_buffer_and_cop(*args, **kwargs)

    async def _maybe_snapshot_track_c(self, *args, **kwargs):
        """Delegates to :class:`daily_processor.DailyProcessor`."""
        return await self._daily_processor.maybe_snapshot_track_c(*args, **kwargs)

    async def _run_track_c_midnight_sync(self, *args, **kwargs):
        """Delegates to :class:`daily_processor.DailyProcessor`."""
        return await self._daily_processor.run_track_c_midnight_sync(*args, **kwargs)

    def _apply_strategies_to_global_model(self, *args, **kwargs):
        """Delegates to :class:`daily_processor.DailyProcessor`."""
        return self._daily_processor.apply_strategies_to_global_model(*args, **kwargs)

    def _replay_per_unit_models(self, *args, **kwargs):
        """Delegates to :class:`daily_processor.DailyProcessor`."""
        return self._daily_processor.replay_per_unit_models(*args, **kwargs)

    async def _try_track_b_cop_smearing(self, *args, **kwargs):
        """Delegates to :class:`daily_processor.DailyProcessor`."""
        return await self._daily_processor.try_track_b_cop_smearing(*args, **kwargs)

    @staticmethod
    def _compute_excluded_mode_energy(day_logs: list[dict]) -> float:
        """Delegates to :class:`daily_processor.DailyProcessor`."""
        return DailyProcessor.compute_excluded_mode_energy(day_logs)

    async def _process_daily_data(self, *args, **kwargs):
        """Delegates to :class:`daily_processor.DailyProcessor`."""
        return await self._daily_processor.process(*args, **kwargs)

