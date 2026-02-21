"""Test Solar Saturation Logic with Partial Aux."""
import pytest
from unittest.mock import MagicMock
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.const import DOMAIN, MODE_HEATING

# Mock HASS and Coordinator
class MockHass:
    def __init__(self):
        self.states = MagicMock()
        self.states.get = MagicMock(return_value=None)
        self.data = {DOMAIN: {}}
        self.config_entries = MagicMock()
        self.bus = MagicMock()
        self.is_running = True

@pytest.fixture
def mock_hass():
    return MockHass()

@pytest.fixture
def coordinator(mock_hass):
    entry = MagicMock()
    entry.data = {
        "energy_sensors": ["sensor.heater_1"],
        "aux_affected_entities": ["sensor.heater_1"],
        "outdoor_temp_sensor": "sensor.outdoor_temp",
        "balance_point": 15.0,
        "wind_speed_sensor": "sensor.wind_speed",
        "solar_enabled": True
    }
    coord = HeatingDataCoordinator(mock_hass, entry)
    # We want real StatisticsManager to test the logic change
    from custom_components.heating_analytics.statistics import StatisticsManager
    coord.statistics = StatisticsManager(coord)

    # Use REAL SolarCalculator to verify the new saturation logic
    from custom_components.heating_analytics.solar import SolarCalculator
    coord.solar = SolarCalculator(coord)

    # But mock the coefficient calculation to control the "Potential"
    coord.solar.calculate_unit_coefficient = MagicMock(return_value=1.0)
    coord.solar.calculate_unit_solar_impact = MagicMock(return_value=0.81)

    # Init internal structures
    coord._hourly_delta_per_unit = {"sensor.heater_1": 0.0}
    coord._accumulated_aux_breakdown = {}

    # Init storage structures (required for calculate_total_power lookups)
    coord._correlation_data_per_unit = {"sensor.heater_1": {"_id": "unit_base"}}
    coord._aux_coefficients_per_unit = {"sensor.heater_1": {"_id": "unit_aux"}}
    coord._aux_coefficients = {"_id": "global_aux"}
    coord._correlation_data = {"_id": "global_base"}

    return coord

def test_solar_saturation_with_partial_aux(coordinator):
    """Test that solar saturation respects known partial aux impact even if is_aux_active is False."""

    # Scenario:
    # Base: 2.1
    # Partial Aux (Known): 1.7 (e.g. 60% of hour active, so is_aux_active=False)
    # Solar Potential: 0.81

    # Physics:
    # Net Demand = Base - Aux = 2.1 - 1.7 = 0.4
    # Solar Limit = 0.4
    # Applied Solar = min(0.81, 0.4) = 0.4
    # Wasted Solar = 0.41

    # Mock Models
    def mock_get_pred(data_map, temp_key, wind_bucket, temp, bp, apply_scaling=True):
        doc_id = data_map.get("_id")
        if doc_id == "unit_base": return 2.1
        if doc_id == "unit_aux": return 1.7 # Model predicts 1.7 if asked (but won't be asked if is_aux_active=False without fix)
        if doc_id == "global_aux": return 1.7
        if doc_id == "global_base": return 2.1
        return 0.0

    coordinator.statistics._get_prediction_from_model = MagicMock(side_effect=mock_get_pred)

    # Mock Solar Calculation
    coordinator.solar.calculate_unit_coefficient.return_value = 1.0
    coordinator.solar.calculate_unit_solar_impact.return_value = 0.81

    # Mode
    coordinator.get_unit_mode = MagicMock(return_value=MODE_HEATING)

    # Action: Call with known_aux_impact_kwh
    # Note: is_aux_active=False simulates the "not dominant" flag from coordinator
    result = coordinator.statistics.calculate_total_power(
        temp=5.0,
        effective_wind=0.0,
        solar_impact=0.0,
        is_aux_active=False,
        known_aux_impact_kwh=1.7  # This argument is new
    )

    bd = result["unit_breakdown"]["sensor.heater_1"]

    print(f"Breakdown: {bd}")

    # Assertions
    assert bd["base_kwh"] == 2.1

    # With the fix, aux_reduction_kwh should reflect the known value distributed
    # Since we only have 1 unit, it should take the full 1.7
    assert bd["aux_reduction_kwh"] == 1.7

    # Net should be 0.4
    assert bd["net_kwh"] == 0.0 # Wait, net_kwh in breakdown is "Net Final" (After solar).
    # Net After Aux = 0.4.
    # Applied Solar = 0.4.
    # Net Final = 0.0.

    assert bd["net_kwh"] == 0.0

    # Solar Saturation Check
    # Should be capped at 0.4
    assert bd["solar_reduction_kwh"] == 0.4
    assert bd["solar_wasted_kwh"] == pytest.approx(0.41, 0.01)
