import sys
from unittest.mock import MagicMock
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

@pytest.fixture
def hass():
    """Mock Home Assistant object."""
    h = MagicMock()
    h.config.units.is_metric = True
    return h
