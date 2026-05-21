import sys
from unittest.mock import MagicMock, NonCallableMagicMock
import pytest
from datetime import timezone, datetime

# Mock Voluptuous
sys.modules["voluptuous"] = MagicMock()

# Mock Home Assistant modules
# We must mock these BEFORE any imports in tests
sys.modules["homeassistant"] = MagicMock()
sys.modules["homeassistant.core"] = MagicMock()
sys.modules["homeassistant.exceptions"] = MagicMock()
sys.modules["homeassistant.config_entries"] = MagicMock()
sys.modules["homeassistant.components"] = MagicMock()
sys.modules["homeassistant.components.sensor"] = MagicMock()
sys.modules["homeassistant.components.number"] = MagicMock()
sys.modules["homeassistant.components.switch"] = MagicMock()
sys.modules["homeassistant.components.select"] = MagicMock()
sys.modules["homeassistant.helpers"] = MagicMock()
sys.modules["homeassistant.helpers.typing"] = MagicMock()
sys.modules["homeassistant.helpers.entity"] = MagicMock()
sys.modules["homeassistant.helpers.entity_platform"] = MagicMock()
sys.modules["homeassistant.helpers.storage"] = MagicMock()
sys.modules["homeassistant.util"] = MagicMock()

# Mock specific submodules that might be imported directly
sys.modules["homeassistant.const"] = MagicMock()

# Mock UnitOfSpeed for use in code
class MockUnitOfSpeed:
    KILOMETERS_PER_HOUR = "km/h"
    MILES_PER_HOUR = "mph"
    KNOTS = "kn"
    METERS_PER_SECOND = "m/s"

sys.modules["homeassistant.const"].UnitOfSpeed = MockUnitOfSpeed

# Mock Sensor Device Class
class MockSensorDeviceClass:
    ENERGY = "energy"
    TEMPERATURE = "temperature"
    POWER = "power"
    CURRENT = "current"
    VOLTAGE = "voltage"

# Mock Sensor State Class
class MockSensorStateClass:
    MEASUREMENT = "measurement"
    TOTAL = "total"
    TOTAL_INCREASING = "total_increasing"

sys.modules["homeassistant.components.sensor"].SensorDeviceClass = MockSensorDeviceClass
sys.modules["homeassistant.components.sensor"].SensorStateClass = MockSensorStateClass

# Helper to simulate Entity properties
class MockEntityMixin:
    @property
    def name(self):
        return getattr(self, "_attr_name", None)

    @property
    def unique_id(self):
        return getattr(self, "_attr_unique_id", None)

    @property
    def native_value(self):
        return getattr(self, "_attr_native_value", None)

    @property
    def extra_state_attributes(self):
        return getattr(self, "_attr_extra_state_attributes", {})

    @property
    def device_info(self):
         return getattr(self, "_attr_device_info", None)


# We need to be careful with update_coordinator as it's a class
# Define a dummy class that accepts init args
class MockDataUpdateCoordinator:
    def __init__(self, hass, logger, name, update_interval):
        self.hass = hass
        self.logger = logger
        self.name = name
        self.update_interval = update_interval
        self.data = {}

    async def async_refresh(self):
        pass

class MockCoordinatorEntity(MockEntityMixin):
    def __init__(self, coordinator):
        self.coordinator = coordinator

mock_coord_module = MagicMock()
mock_coord_module.DataUpdateCoordinator = MockDataUpdateCoordinator
mock_coord_module.CoordinatorEntity = MockCoordinatorEntity
sys.modules["homeassistant.helpers.update_coordinator"] = mock_coord_module

class MockEntity(MockEntityMixin):
    pass

sys.modules["homeassistant.components.sensor"].SensorEntity = MockEntity
sys.modules["homeassistant.components.number"].NumberEntity = MockEntity
sys.modules["homeassistant.components.switch"].SwitchEntity = MockEntity
sys.modules["homeassistant.components.select"].SelectEntity = MockEntity

# Mock util.dt with REAL timezone and parse_datetime
mock_dt = MagicMock(name='mock_dt_real')
mock_dt.UTC = timezone.utc # Use real UTC object
mock_dt.as_utc.side_effect = lambda d: d.replace(tzinfo=timezone.utc)
mock_dt.now.return_value = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
# Implement parse_datetime to return real datetime objects
def side_effect_parse_datetime(dt_str):
    try:
        return datetime.fromisoformat(dt_str)
    except (ValueError, TypeError):
        return None
mock_dt.parse_datetime.side_effect = side_effect_parse_datetime
# Implement as_local to just return as-is (simulating UTC/same timezone)
mock_dt.as_local.side_effect = lambda d: d

# CRITICAL: Ensure imports via parent module also get the specific mock
sys.modules["homeassistant.util"].dt = mock_dt
sys.modules["homeassistant.util.dt"] = mock_dt

# Recursively link the mock modules so `patch` resolves correctly
# even when the real 'homeassistant' library is not installed.
for mod_name, mock_mod in sys.modules.items():
    if mod_name.startswith("homeassistant."):
        parent_name = mod_name.rsplit(".", 1)[0]
        child_name = mod_name.split(".")[-1]
        if parent_name in sys.modules:
            setattr(sys.modules[parent_name], child_name, mock_mod)

from tests.helpers import ModelProxy, CoordinatorModelMixin  # noqa: F401


@pytest.fixture
def mock_coordinator(mock_entry):
    """Global pre-configured coordinator mock for all tests (#962).

    Uses MagicMock(spec=HeatingDataCoordinator) to block unknown attrs
    while providing physical defaults for all critical read-paths.
    """
    from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
    from custom_components.heating_analytics.const import MODE_HEATING

    # Use NonCallableMagicMock for the base so it doesnt masquerade as a function,
    # but MagicMock(spec=...) usually handles this well.
    mock = MagicMock(spec=HeatingDataCoordinator, name="mock_coordinator")

    # --- Core Attributes ---
    mock.hass = MagicMock()
    mock.entry = mock_entry
    mock.data = {}
    mock.solar_enabled = True
    mock.learning_enabled = True
    mock.solar_azimuth = 180
    mock.solar_correction_percent = 100.0
    mock.balance_point = 18.0
    mock.energy_sensors = []
    mock.solar_battery_decay = 0.50
    mock.battery_thermal_feedback_k = 0.0
    mock.inertia_weights = [0.2, 0.8]
    mock.learning_rate = 0.05
    mock.learning_enabled = True
    mock.wind_unit = "m/s"
    mock.wind_threshold = 5.5
    mock.extreme_wind_threshold = 10.8
    mock.wind_gust_factor = 0.6

    # --- Managers ---
    mock.solar = MagicMock(name="mock_solar")
    mock.forecast = MagicMock(name="mock_forecast")
    mock.statistics = MagicMock(name="mock_statistics")
    mock.learning = MagicMock(name="mock_learning")
    mock.solar_optimizer = MagicMock(name="mock_solar_optimizer")

    # Defaults for managers
    mock.forecast.calculate_future_energy.return_value = (0.0, 0.0, {})
    mock.forecast.get_future_day_prediction.return_value = (0.0, 0.0, {})
    mock.forecast._cached_long_term_hourly = []
    mock.forecast._cached_long_term_daily = []
    mock.forecast._cached_forecast_date = None
    mock.forecast._forecast_history = []
    mock.forecast._midnight_forecast_snapshot = {}
    mock.forecast._reference_forecast = []
    mock.forecast._primary_reference_forecast = []
    mock.forecast._secondary_reference_forecast = []
    mock.forecast._live_forecast = []
    mock.forecast._last_live_update = None

    mock.statistics.calculate_hybrid_projection.return_value = (0.0, 0.0)
    mock.statistics.calculate_historical_actual_sum.return_value = 0.0

    # --- Private Attributes (common in tests) ---
    mock._hourly_log = []
    mock._hourly_delta_per_unit = {}
    mock._hourly_expected_per_unit = {}
    mock._hourly_expected_base_per_unit = {}
    mock._accumulated_energy_today = 0.0
    mock._daily_aux_breakdown = {}
    mock._lifetime_individual = {}
    mock._last_hour_processed = None
    mock._accumulation_start_time = None
    mock._last_energy_values = {}
    mock._learned_u_coefficient = None
    mock._last_midnight_indoor_temp = None
    mock._solar_battery_state = 0.0
    mock._solar_carryover_state = 0.0
    mock._potential_battery_s = 0.0
    mock._potential_battery_e = 0.0
    mock._potential_battery_w = 0.0
    mock._potential_battery_4d_s = 0.0
    mock._potential_battery_4d_e = 0.0
    mock._potential_battery_4d_w = 0.0
    mock._potential_battery_4d_diffuse = 0.0
    mock._aux_coefficients = {}
    mock._aux_coefficients_per_unit = {}
    mock._solar_coefficients_per_unit = {}
    mock._solar_coefficients_4d_per_unit = {}
    mock._critical_elev_per_facade = {"s": None, "e": None, "w": None}
    mock._learning_buffer_global = []
    mock._learning_buffer_per_unit = {}
    mock._learning_buffer_aux_per_unit = {}
    mock._learning_buffer_solar_per_unit = {}
    mock._learning_buffer_solar_4d_per_unit = {}
    mock._per_unit_min_base_thresholds = {}
    mock._unit_modes = {}
    mock._last_batch_fit_per_unit = {}
    mock._tobit_sufficient_stats = {}
    mock._experimental_tobit_live_learner = False
    mock._tobit_live_entities = set()
    mock._tobit_default_applied = False
    mock._observation_counts = {}
    mock._correlation_data = {}
    mock._correlation_data_per_unit = {}
    mock._daily_history = {}
    mock._daily_individual = {}
    mock.auxiliary_heating_active = False
    mock._aux_cooldown_active = False
    mock._aux_cooldown_start_time = None

    # --- Methods ---
    # Default: behaves as legacy 3D impact.
    mock.hourly_solar_impact_kwh = MagicMock(name="hourly_solar_impact_kwh")
    mock.hourly_solar_impact_kwh.side_effect = lambda entry: float(entry.get("solar_impact_kwh") or 0.0)

    # Defaults for other common managers/call-sites
    mock.get_unit_mode.return_value = MODE_HEATING
    mock.is_solar_affected.return_value = True
    mock.aux_affected_entities = []
    mock.screen_config_for_entity.return_value = (False, False, False)
    mock._get_unit_observation_count.return_value = 100

    # Define functional side effects for solar calculations
    def mock_solar_impact(potential_vector, coeff):
        if not coeff:
            return 0.0
        # Handle both 3D and 4D vectors
        s = coeff.get("s", 0.0) * potential_vector[0]
        e = coeff.get("e", 0.0) * potential_vector[1]
        w = coeff.get("w", 0.0) * potential_vector[2]
        d = coeff.get("diffuse", 0.0) * potential_vector[3] if len(potential_vector) > 3 else 0.0
        return s + e + w + d
    
    mock.solar.calculate_unit_solar_impact.side_effect = mock_solar_impact

    def mock_unit_coeff(entity_id, temp_key, mode):
        # Determine regime
        from custom_components.heating_analytics.const import MODE_HEATING, MODE_GUEST_HEATING, MODE_COOLING, MODE_GUEST_COOLING
        regime = "heating" if mode in (MODE_HEATING, MODE_GUEST_HEATING) else "cooling" if mode in (MODE_COOLING, MODE_GUEST_COOLING) else None
        if not regime:
            return {}
        
        # Try to get from _solar_coefficients_per_unit
        coeffs_per_unit = getattr(mock, "_solar_coefficients_per_unit", {})
        unit_coeffs = coeffs_per_unit.get(entity_id, {})
        return unit_coeffs.get(regime, {"s": 0.35, "e": 0.35, "w": 0.35} if regime == "heating" else {"s": 0.4, "e": 0.4, "w": 0.4})

    mock.solar.calculate_unit_coefficient.side_effect = mock_unit_coeff

    def mock_apply_correction(base, impact, mode_or_temp):
        from custom_components.heating_analytics.const import MODE_HEATING, MODE_COOLING
        if isinstance(mode_or_temp, (int, float)):
            mode = MODE_HEATING if mode_or_temp < mock.balance_point else MODE_COOLING
        else:
            mode = mode_or_temp
        
        if mode == MODE_HEATING:
            return max(0.0, base - impact)
        if mode == MODE_COOLING:
            return base + impact
        return base
        
    mock.solar.apply_correction.side_effect = mock_apply_correction

    # Ensure get_model_state() returns the .model proxy built by monkey-patch
    mock.get_model_state.side_effect = lambda: mock.model

    return mock


@pytest.fixture
def mock_entry():
    """Mock Home Assistant ConfigEntry."""
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.data = {}
    entry.options = {}
    return entry


@pytest.fixture
def hass():
    """Mock Home Assistant object."""
    h = MagicMock()
    h.config.units.is_metric = True
    return h


# --- Monkey-patch MagicMock to auto-create _collector (#775) ---
# Tests use MagicMock(spec=HeatingDataCoordinator) which blocks underscore
# attribute auto-creation.  The _collector is the only underscore attr that
# production code *creates in __init__* and then accesses via dotted reads
# (self._collector.sample_count).  Rather than patching every fixture, we
# monkey-patch MagicMock.__getattr__ to lazily create a real
# ObservationCollector when _collector is first accessed on any mock.
_original_mock_getattr = MagicMock.__getattr__

def _patched_mock_getattr(self, name):
    if name in ("_collector", "collector"):
        # Share one real ObservationCollector between ``_collector``
        # (legacy private attribute still used within coordinator) and
        # ``collector`` (new public accessor used by managers).  Both
        # access paths must resolve to the same object so tests that
        # prime collector state via ``coord._collector.delta_per_unit[…]``
        # are visible when managers read via ``coord.collector.…``.
        from custom_components.heating_analytics.observation import ObservationCollector
        existing = self.__dict__.get("_collector")
        if existing is None:
            existing = ObservationCollector()
            object.__setattr__(self, "_collector", existing)
            object.__setattr__(self, "collector", existing)
        elif "collector" not in self.__dict__:
            object.__setattr__(self, "collector", existing)
        return existing
    # Coordinator exposes runtime state as public @property over
    # underlying ``_X`` fields (daily_individual, aux_cooldown_active,
    # aux_affected_set).  Properties do not fire on MagicMock, so when
    # a test sets ``coord._aux_affected_set = set(...)`` and manager
    # code reads ``coord.aux_affected_set``, the latter would resolve to
    # an auto-MagicMock.  Forward public-name reads to the ``_X`` dict
    # entry when tests have primed it; otherwise fall through to
    # MagicMock's auto-attribute machinery.
    _PUBLIC_ALIASES = ("daily_individual", "aux_cooldown_active", "aux_affected_set")
    if name in _PUBLIC_ALIASES:
        underscore_name = f"_{name}"
        if underscore_name in self.__dict__:
            return self.__dict__[underscore_name]
    if name == "model":
        # Build a ModelState-like namespace that delegates to the mock's
        # own _fields.  This lets ``coordinator.model.hourly_log`` resolve
        # to the same list that tests set via ``coordinator._hourly_log = [...]``.
        from custom_components.heating_analytics.observation import ModelState
        mock_self = self
        class _LazyModel:
            """Proxy: reads coordinator._field via model.field."""
            def __getattr__(inner_self, attr):
                return getattr(mock_self, f"_{attr}")
        proxy = _LazyModel()
        object.__setattr__(self, "model", proxy)
        return proxy
    return _original_mock_getattr(self, name)

MagicMock.__getattr__ = _patched_mock_getattr
NonCallableMagicMock.__getattr__ = _patched_mock_getattr


