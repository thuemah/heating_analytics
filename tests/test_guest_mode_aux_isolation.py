"""Tests for guest mode isolation in auxiliary heating learning."""
import pytest
from unittest.mock import MagicMock
from custom_components.heating_analytics.learning import LearningManager
from custom_components.heating_analytics.const import (
    MODE_HEATING,
    MODE_GUEST_HEATING,
    MODE_GUEST_COOLING,
)


@pytest.fixture
def learning_manager():
    """Create a learning manager instance for testing."""
    return LearningManager()


@pytest.fixture
def base_learning_params():
    """Base parameters for learning tests."""
    return {
        "temp_key": "-5",
        "wind_bucket": "normal",
        "avg_temp": -5.0,
        "total_energy_kwh": 1.0,  # Clean energy (guest excluded)
        "base_expected_kwh": 0.8,
        "solar_impact": 0.0,
        "avg_solar_factor": 0.0,
        "is_aux_active": True,
        "aux_impact": 0.5,
        "learning_enabled": True,
        "solar_enabled": False,
        "learning_rate": 0.2,
        "balance_point": 15.0,
        "energy_sensors": ["sensor.heater_1"],
        "hourly_bucket_counts": {},
        "hourly_sample_count": 60,
        "correlation_data": {},
        "correlation_data_per_unit": {},
        "aux_coefficients": {"-5": {"normal": 0.5}},
        "learning_buffer_global": {},
        "learning_buffer_per_unit": {},
        "observation_counts": {},
        "hourly_delta_per_unit": {"sensor.heater_1": 1.0},
        "hourly_expected_per_unit": {},
        "hourly_expected_base_per_unit": {},
        "aux_coefficients_per_unit": {},
        "learning_buffer_aux_per_unit": {},
        "solar_coefficients_per_unit": {},
        "learning_buffer_solar_per_unit": {},
        "solar_calculator": MagicMock(),
        "get_predicted_unit_base_fn": lambda eid, tk, wb, temp: 0.8,
        "unit_modes": {},
        "aux_affected_entities": None,
    }


def test_aux_learning_normal_mode_no_guest(learning_manager, base_learning_params):
    """Test that aux learning works normally without guest mode."""
    params = base_learning_params.copy()
    params["unit_modes"] = {"sensor.heater_1": MODE_HEATING}
    params["has_guest_activity"] = False

    result = learning_manager.process_learning(**params)

    # Aux learning should update
    assert result["aux_model_updated"] is True
    assert result["aux_model_before"] == 0.5
    assert result["aux_model_after"] != 0.5  # Should be updated
    assert result["learning_status"].startswith("active_aux")


def test_aux_learning_skipped_with_guest_mode(learning_manager, base_learning_params):
    """Test that aux learning is skipped when guest mode is active."""
    params = base_learning_params.copy()
    params["unit_modes"] = {
        "sensor.heater_1": MODE_HEATING,
        "sensor.guest_heater": MODE_GUEST_HEATING,
    }
    params["has_guest_activity"] = True

    result = learning_manager.process_learning(**params)

    # Aux learning should be skipped
    assert result["aux_model_updated"] is False
    assert result["aux_model_before"] == 0.5
    assert result["aux_model_after"] == 0.5  # Unchanged
    assert result["learning_status"] == "aux_skipped_guest_mode"


def test_aux_learning_skipped_with_guest_cooling(learning_manager, base_learning_params):
    """Test that aux learning is skipped with guest cooling mode."""
    params = base_learning_params.copy()
    params["unit_modes"] = {
        "sensor.heater_1": MODE_HEATING,
        "sensor.guest_ac": MODE_GUEST_COOLING,
    }
    params["has_guest_activity"] = True

    result = learning_manager.process_learning(**params)

    # Aux learning should be skipped
    assert result["aux_model_updated"] is False
    assert result["aux_model_before"] == 0.5
    assert result["aux_model_after"] == 0.5
    assert result["learning_status"] == "aux_skipped_guest_mode"


def test_base_learning_continues_with_guest_mode(learning_manager, base_learning_params):
    """Test that base learning continues even with guest mode active."""
    params = base_learning_params.copy()
    params["is_aux_active"] = False  # Normal mode (not aux)
    params["unit_modes"] = {
        "sensor.heater_1": MODE_HEATING,
        "sensor.guest_heater": MODE_GUEST_HEATING,
    }
    params["has_guest_activity"] = True
    params["correlation_data"] = {"-5": {"normal": 0.8}}

    result = learning_manager.process_learning(**params)

    # Base model should be updated (guest energy was excluded via learning_energy_kwh)
    assert result["model_updated"] is True
    assert result["model_base_after"] != 0.8  # Should change due to EMA learning


def test_aux_coeff_preserved_when_guest_active(learning_manager, base_learning_params):
    """Test that existing aux coefficients are preserved during guest mode."""
    params = base_learning_params.copy()

    # Set up initial aux coefficient
    initial_coeff = 0.75
    params["aux_coefficients"] = {"-5": {"normal": initial_coeff}}
    params["unit_modes"] = {
        "sensor.heater_1": MODE_HEATING,
        "sensor.guest": MODE_GUEST_HEATING,
    }
    params["has_guest_activity"] = True

    result = learning_manager.process_learning(**params)

    # Coefficient should remain unchanged
    assert params["aux_coefficients"]["-5"]["normal"] == initial_coeff
    assert result["aux_model_before"] == initial_coeff
    assert result["aux_model_after"] == initial_coeff
