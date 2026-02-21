"""Direct Unit Tests for LearningManager."""
from unittest.mock import MagicMock
import pytest
from custom_components.heating_analytics.learning import LearningManager
from custom_components.heating_analytics.const import MODE_HEATING

@pytest.fixture
def learning_manager():
    """Fixture to create a LearningManager instance."""
    return LearningManager()

@pytest.fixture
def base_context():
    """Fixture to provide a base context dictionary for process_learning."""
    # Pre-populate correlation data to simulate existing model (skipping cold start buffering)
    correlation_data_per_unit = {
        "sensor.affected": {
            "5": {"normal": 10.0}
        },
        "sensor.unaffected": {
            "5": {"normal": 10.0}
        }
    }

    return {
        "temp_key": "5",
        "wind_bucket": "normal",
        "avg_temp": 5.0,
        "total_energy_kwh": 20.0,
        "base_expected_kwh": 20.0,
        "solar_impact": 0.0,
        "avg_solar_factor": 0.0,
        "is_aux_active": False,
        "aux_impact": 0.0,
        "learning_enabled": True,
        "solar_enabled": False,
        "learning_rate": 0.1,
        "balance_point": 17.0,
        "energy_sensors": ["sensor.affected", "sensor.unaffected"],
        # State
        "hourly_bucket_counts": {},
        "hourly_sample_count": 60,
        "correlation_data": {},
        "correlation_data_per_unit": correlation_data_per_unit,
        "aux_coefficients": {},
        "learning_buffer_global": {},
        "learning_buffer_per_unit": {},
        "observation_counts": {},
        "hourly_delta_per_unit": {
            "sensor.affected": 8.0,   # Lower than expected (10.0) -> Should decrease model
            "sensor.unaffected": 8.0  # Lower than expected (10.0) -> Should decrease model
        },
        "hourly_expected_per_unit": {},
        "hourly_expected_base_per_unit": {
             "sensor.affected": 10.0,
             "sensor.unaffected": 10.0
        },
        "aux_coefficients_per_unit": {},
        "learning_buffer_aux_per_unit": {},
        "solar_coefficients_per_unit": {},
        "learning_buffer_solar_per_unit": {},
        # Services
        "solar_calculator": MagicMock(),
        "get_predicted_unit_base_fn": MagicMock(return_value=10.0),
        "unit_modes": {
            "sensor.affected": MODE_HEATING,
            "sensor.unaffected": MODE_HEATING
        },
        # Aux Control (Default)
        "aux_affected_entities": ["sensor.affected"],
        "has_guest_activity": False,
        "is_cooldown_active": False,
    }

def test_selective_learning_during_cooldown(learning_manager, base_context):
    """Test that only non-affected units learn during cooldown."""

    # 1. Setup Cooldown Context
    base_context["is_cooldown_active"] = True
    base_context["aux_affected_entities"] = ["sensor.affected"]

    # 2. Run Learning
    learning_manager.process_learning(**base_context)

    # 3. Verify Results
    correlation_data = base_context["correlation_data_per_unit"]

    # Affected unit: Should NOT change (Locked)
    assert correlation_data["sensor.affected"]["5"]["normal"] == 10.0

    # Unaffected unit: Should change (Learned)
    # Expected 10.0, Actual 8.0, Rate 0.1 (capped at 0.03 per unit rate cap?)
    # Wait, CAP is PER_UNIT_LEARNING_RATE_CAP = 0.03
    # New = 10.0 + 0.03 * (8.0 - 10.0) = 10.0 - 0.06 = 9.94
    assert correlation_data["sensor.unaffected"]["5"]["normal"] < 10.0
    assert correlation_data["sensor.unaffected"]["5"]["normal"] == 9.94

def test_cooldown_all_affected_via_none(learning_manager, base_context):
    """Test that NO units learn if aux_affected_entities is None (Default All)."""

    # 1. Setup Cooldown Context
    base_context["is_cooldown_active"] = True
    base_context["aux_affected_entities"] = None # Implicitly ALL affected

    # 2. Run Learning
    learning_manager.process_learning(**base_context)

    # 3. Verify Results
    correlation_data = base_context["correlation_data_per_unit"]

    # Both should be locked
    assert correlation_data["sensor.affected"]["5"]["normal"] == 10.0
    assert correlation_data["sensor.unaffected"]["5"]["normal"] == 10.0

def test_cooldown_none_affected_via_empty_list(learning_manager, base_context):
    """Test that ALL units learn if aux_affected_entities is [] (None Affected)."""

    # 1. Setup Cooldown Context
    base_context["is_cooldown_active"] = True
    base_context["aux_affected_entities"] = [] # Explicitly NONE affected

    # 2. Run Learning
    learning_manager.process_learning(**base_context)

    # 3. Verify Results
    correlation_data = base_context["correlation_data_per_unit"]

    # Both should learn
    assert correlation_data["sensor.affected"]["5"]["normal"] < 10.0
    assert correlation_data["sensor.unaffected"]["5"]["normal"] < 10.0

def test_cooldown_excludes_aux_learning(learning_manager, base_context):
    """Test that cooldown logic prevents learning even if aux was active (edge case)."""
    # Ideally cooldown starts AFTER aux is inactive, but if flags conflict,
    # cooldown lock should take precedence for affected units.

    base_context["is_cooldown_active"] = True
    base_context["is_aux_active"] = True # Aux still theoretically active?
    # In reality, cooldown starts when aux turns off, so this combination
    # is unlikely in production unless manually forced, but tests logic robustness.

    learning_manager.process_learning(**base_context)

    correlation_data = base_context["correlation_data_per_unit"]

    # Affected unit locked
    assert correlation_data["sensor.affected"]["5"]["normal"] == 10.0
