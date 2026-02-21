"""Constants for the Heating Analytics integration."""

DOMAIN = "heating_analytics"

# Default Configuration Values
DEFAULT_NAME = "Heating Analytics"
DEFAULT_WIND_GUST_FACTOR = 0.6
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_BALANCE_POINT = 17.0
DEFAULT_WIND_THRESHOLD = 5.5
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

# Default Inertia Profile: "Thermal Mass Balanced"
# Logic: (H-3, H-2, H-1, Current) -> Sums to 1.0
#
# Weights breakdown:
# 20% (Current): Immediate outdoor conditions.
# 80% (History): Dominant factor — retained heat in walls/floors (Thermal Mass).
#
# Symmetric arc profile: the middle hours (H-2, H-1) carry most weight,
# reflecting that a well-insulated house's heating demand correlates most
# with the 1-3 hour temperature trend, not the instantaneous reading.
DEFAULT_INERTIA_WEIGHTS = (0.20, 0.30, 0.30, 0.20)

# Thermal Inertia Configuration (User Selectable)
CONF_THERMAL_INERTIA = "thermal_inertia"
THERMAL_INERTIA_FAST = "fast"
THERMAL_INERTIA_NORMAL = "normal"
THERMAL_INERTIA_SLOW = "slow"

# Predefined Weight Profiles
# Fast (2 Hours): Highly responsive. (0.50, 0.50)
INERTIA_PROFILE_FAST = (0.50, 0.50)
# Normal (4 Hours): Standard House (Default)
INERTIA_PROFILE_NORMAL = DEFAULT_INERTIA_WEIGHTS
# Slow (12 Hours): Passive House / Concrete Foundation (0.05, ..., 0.06)
# Smooth bell curve centered on H-6/H-5
INERTIA_PROFILE_SLOW = (0.05, 0.05, 0.06, 0.08, 0.10, 0.12, 0.12, 0.12, 0.10, 0.08, 0.06, 0.06)

# Solar Defaults
DEFAULT_SOLAR_ENABLED = False
DEFAULT_SOLAR_CORRECTION = 100
ENERGY_GUARD_THRESHOLD = 0.01  # 10 Wh - Consistent guard against division by zero
DEFAULT_SOLAR_LEARNING_RATE = 0.01
DEFAULT_AUX_LEARNING_RATE = 0.01

# Solar coefficients optimized for heat pump installations (COP ~2.3-2.5)
# These values account for the fact that solar gain saves less electricity with heat pumps
# compared to direct electric heating (1 kWh solar gain saves ~1/COP kWh electricity)
# - Heat pumps (COP 2.5-3.5): Model will fine-tune to 0.10-0.14 over 1-2 weeks
# - Direct electric heating: Model will learn upward to 0.30-0.40 over 1-2 weeks
DEFAULT_SOLAR_COEFF_HEATING = 0.15  # Was 0.35 (optimized for direct electric)
DEFAULT_SOLAR_COEFF_COOLING = 0.17  # Was 0.40 (maintains ~1.13x ratio for summer impact)

CONF_WIND_UNIT = "wind_unit"
CONF_ENABLE_LIFETIME_TRACKING = "enable_lifetime_tracking"
CONF_SOLAR_ENABLED = "solar_enabled"
CONF_SOLAR_AZIMUTH = "solar_azimuth"
CONF_SOLAR_CORRECTION = "solar_correction"
# Removed: CONF_AC_CAPABLE_DEVICES - replaced by CONF_HAS_AC_UNITS global checkbox

DEFAULT_SOLAR_AZIMUTH = 180

WIND_UNIT_MS = "m/s"
WIND_UNIT_KMH = "km/h"
WIND_UNIT_KNOTS = "knots"

# Conversion Constants
MS_TO_KMH = 3.6
MS_TO_KNOTS = 1.94384

# Learning Constants
PER_UNIT_LEARNING_RATE_CAP = 0.03   # 3% max EMA rate to prevent oscillation
SOLAR_COEFF_CAP = 5.0               # Max solar coefficient (kW per full sun)
LEARNING_BUFFER_THRESHOLD = 4
TARGET_TDD_WINDOW = 0.5  # Minimum TDD accumulation for seamless rolling window efficiency
MIN_EXTRAPOLATION_DELTA_T = 0.5  # Minimum Delta T (Degrees) required to trust extrapolation source

# Cloud Coverage Default (when weather entity has unknown state)
DEFAULT_CLOUD_COVERAGE = 50.0

# Mixed Mode Detection Bounds (aux fraction for learning eligibility)
MIXED_MODE_LOW = 0.20   # Below this = mostly normal heating
MIXED_MODE_HIGH = 0.80  # Above this = mostly aux heating

# Aux Cooldown / Decay Mechanism (Prevent Thermal Lag Sampling Bias)
COOLDOWN_MIN_HOURS = 2              # Minimum hours to lock learning after Aux turns off
COOLDOWN_MAX_HOURS = 6              # Maximum safety timeout for the lock
COOLDOWN_CONVERGENCE_THRESHOLD = 0.95 # Convergence ratio (Actual/Expected) to exit early

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
STORAGE_VERSION = 2  # Incremented for Solar Correction support
STORAGE_KEY = f"{DOMAIN}.storage"

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

SOURCE_SENSOR = "sensor"
SOURCE_WEATHER = "weather"

DEFAULT_FORECAST_CROSSOVER_DAY = 4

# Modes
MODE_HEATING = "heating"
MODE_COOLING = "cooling"
MODE_OFF = "off"
MODE_GUEST_HEATING = "guest_heating"
MODE_GUEST_COOLING = "guest_cooling"
CONF_HAS_AC_UNITS = "has_ac_units"
