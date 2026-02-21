
import unittest
from datetime import datetime
from unittest.mock import MagicMock
import pytest
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.const import ATTR_SOLAR_FACTOR, ATTR_SOLAR_IMPACT

class TestForecastOptimization(unittest.TestCase):
    def setUp(self):
        # Mock Home Assistant and ConfigEntry
        self.hass = MagicMock()
        self.entry = MagicMock()
        self.entry.data = {}

        # Instantiate Coordinator (it will create ForecastManager internally)
        self.coordinator = HeatingDataCoordinator(self.hass, self.entry)

        # Mock internal state needed for _update_daily_budgets
        self.coordinator._accumulated_expected_energy_hour = 0.0
        self.coordinator._hourly_log = []
        self.coordinator._accumulated_energy_today = 0.0
        self.coordinator._accumulated_energy_hour = 0.0
        self.coordinator.data = {ATTR_SOLAR_FACTOR: 0.0, ATTR_SOLAR_IMPACT: 0.0}

        # Mock ForecastManager
        self.coordinator.forecast = MagicMock()
        self.coordinator.forecast.calculate_future_energy.return_value = (0.0, 0.0, {})
        # FIX: Mock get_plan_for_hour to return a 2-tuple
        self.coordinator.forecast.get_plan_for_hour.return_value = (5.0, {})

        # Mock Helper methods
        self.coordinator._calculate_inertia_temp = MagicMock(return_value=10.0)
        self.coordinator._get_cloud_coverage = MagicMock(return_value=50.0)
        self.coordinator._get_partial_log_for_current_hour = MagicMock(return_value=None)

        # Mock weather entity state lookup
        weather_state = MagicMock()
        weather_state.attributes.get.return_value = "km/h"
        self.coordinator.weather_entity = "weather.home"
        self.hass.states.get.return_value = weather_state

    def test_redundant_calculation_check(self):
        # Setup: Aux is NOT active
        self.coordinator.auxiliary_heating_active = False

        # Mock Forecast Item (so we enter the relevant block)
        forecast_item = {"datetime": "2023-10-27T12:00:00"}
        self.coordinator.forecast.get_forecast_for_hour.return_value = forecast_item

        # Mock _process_forecast_item to return dummy values
        # Returns: (predicted_kwh, solar_kwh, inertia_val, raw_temp, wind_speed_raw, wind_speed_ms, unit_breakdown, aux_impact)
        self.coordinator.forecast._process_forecast_item.return_value = (5.0, 0.0, 10.0, 10.0, 5.0, 1.4, {}, 0.0)

        # Execution
        current_time = datetime(2023, 10, 27, 12, 30)
        self.coordinator._update_daily_budgets(1.0, current_time, 30)

        # Verification
        # Expectation: 2 calls to get_plan_for_hour (1 Reference Net, 1 Live Net)
        print(f"Call count (Aux Inactive): {self.coordinator.forecast.get_plan_for_hour.call_count}")
        self.assertEqual(self.coordinator.forecast.get_plan_for_hour.call_count, 2)

    def test_aux_active_still_calls_twice(self):
        # Setup: Aux IS active
        self.coordinator.auxiliary_heating_active = True

        forecast_item = {"datetime": "2023-10-27T12:00:00"}
        self.coordinator.forecast.get_forecast_for_hour.return_value = forecast_item
        self.coordinator.forecast._process_forecast_item.return_value = (5.0, 0.0, 10.0, 10.0, 5.0, 1.4, {}, 0.0)

        current_time = datetime(2023, 10, 27, 12, 30)
        self.coordinator._update_daily_budgets(1.0, current_time, 30)

        # Should be called 3 times (1 Reference Net, 1 Live Net, 1 Reference Gross)
        self.assertEqual(self.coordinator.forecast.get_plan_for_hour.call_count, 3)
        print(f"Call count (Aux Active): {self.coordinator.forecast.get_plan_for_hour.call_count}")

if __name__ == '__main__':
    unittest.main()
