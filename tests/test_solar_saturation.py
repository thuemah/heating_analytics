"""Test Solar Saturation Logic."""
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

    return coord

def test_solar_saturation_accounting(coordinator):
    """Test that solar saturation is correctly accounted for."""

    # Scenario from user:
    # Base: 2.1
    # Aux Applied: 1.7 -> Remaining: 0.4
    # Solar Theoretical: 0.81
    # Expected:
    #   Net Final: 0.0
    #   Solar Applied: 0.4
    #   Solar Wasted: 0.41

    # Setup Data
    coordinator.auxiliary_heating_active = True
    coordinator.data["effective_wind"] = 0.0
    coordinator.data["solar_impact"] = 0.0 # Will be ignored as we override in raw_unit_data via mocks

    # Mock Models
    # We bypass _get_prediction_from_model by mocking it,
    # or we can setup the data structures. Mocking is easier for specific numbers.

    # We need to intercept the calls in calculate_total_power.
    # But calculate_total_power calls _get_prediction_from_model.

    # Setup Unit Data with IDs for identification in mock
    coordinator._correlation_data_per_unit = {"sensor.heater_1": {"_id": "unit_base"}}
    coordinator._aux_coefficients_per_unit = {"sensor.heater_1": {"_id": "unit_aux"}}
    coordinator._aux_coefficients = {"_id": "global_aux"}
    coordinator._correlation_data = {"_id": "global_base"}

    def mock_get_pred(data_map, temp_key, wind_bucket, temp, bp, apply_scaling=True):
        # We need to return Base=2.1 and Aux=1.7
        # The method uses different maps for base and aux.
        # Check by ID if available (our injected marker)
        doc_id = data_map.get("_id")

        if doc_id == "unit_base":
            return 2.1
        if doc_id == "unit_aux":
            return 1.7
        if doc_id == "global_aux":
            return 1.7
        if doc_id == "global_base":
            return 2.1
        return 0.0

    coordinator.statistics._get_prediction_from_model = MagicMock(side_effect=mock_get_pred)

    # Mock Solar Calculation
    # We need calculate_unit_coefficient and calculate_unit_solar_impact to result in 0.81
    coordinator.solar.calculate_unit_coefficient.return_value = 1.0
    coordinator.solar.calculate_unit_solar_impact.return_value = 0.81

    # Mode
    coordinator.get_unit_mode = MagicMock(return_value=MODE_HEATING)

    # Run Calculation
    result = coordinator.statistics.calculate_total_power(
        temp=5.0,
        effective_wind=0.0,
        solar_impact=0.0,
        is_aux_active=True
    )

    bd = result["unit_breakdown"]["sensor.heater_1"]

    print(f"Breakdown: {bd}")

    # Assertions
    assert bd["base_kwh"] == 2.1
    assert bd["raw_aux_kwh"] == 1.7 # scaled if global matches unit sum

    # Logic check:
    # Net After Aux = 2.1 - 1.7 = 0.4
    # Solar = 0.81
    # Net Final = max(0, 0.4 - 0.81) = 0.0

    assert bd["net_kwh"] == 0.0

    # New Behavior Check (After Fix):
    # solar_reduction_kwh contains the APPLIED (0.4)
    # Net Demand (0.4) < Potential (0.81) => Applied = 0.4
    assert bd["solar_reduction_kwh"] == 0.4

    # solar_wasted_kwh contains the WASTED (0.41)
    # 0.81 - 0.4 = 0.41
    assert bd["solar_wasted_kwh"] == 0.41

    # Check that raw potential is also preserved if we need it
    assert bd["raw_solar_kwh"] == 0.81
