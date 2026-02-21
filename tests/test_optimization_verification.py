
import unittest
from datetime import date, timedelta
from unittest.mock import MagicMock
from custom_components.heating_analytics.statistics import StatisticsManager
from custom_components.heating_analytics.const import MODE_HEATING, MODE_COOLING, ATTR_SOLAR_FACTOR

# Mock Coordinator
class MockCoordinator:
    def __init__(self, sensors):
        self.energy_sensors = sensors
        self.balance_point = 15.0
        self.solar_enabled = True
        self._daily_history = {}
        self._hourly_log = []
        self._correlation_data_per_unit = {}
        self._aux_coefficients_per_unit = {}
        self._aux_coefficients = {}
        self.aux_affected_entities = []
        self.solar = MagicMock()
        self.solar.calculate_unit_coefficient.return_value = 1.0
        self.solar.calculate_unit_solar_impact.return_value = 0.0
        self.solar.apply_correction.side_effect = lambda net, solar, mode: net # No-op
        self.data = {ATTR_SOLAR_FACTOR: 0.0}

    def _get_wind_bucket(self, wind):
        return "normal"

    def _get_predicted_kwh(self, *args):
        return 1.0

    def get_unit_mode(self, entity_id):
        return MODE_HEATING

class TestOptimizationVerification(unittest.TestCase):
    def setUp(self):
        self.sensors = [f"sensor.heater_{i}" for i in range(5)]
        self.coord = MockCoordinator(self.sensors)
        self.stats = StatisticsManager(self.coord)

        # Mock calculate_total_power to verify it receives the correct unit_modes
        self.stats.calculate_total_power = MagicMock(return_value={
            "total_kwh": 5.0,
            "global_base_kwh": 5.0,
            "global_aux_reduction_kwh": 0.0,
            "breakdown": {
                "base_kwh": 5.0,
                "aux_reduction_kwh": 0.0,
                "solar_reduction_kwh": 0.0,
                "unassigned_aux_savings": 0.0,
                "unspecified_kwh": 0.0
            },
            "unit_breakdown": {s: {"net_kwh": 1.0} for s in self.sensors}
        })

        # Mock _get_daily_log_map to return empty (forcing fallback to simulated logic in test or just empty)
        # But wait, calculate_modeled_energy relies on daily_history if log map is empty.
        # We need to populate _daily_history to trigger the logic path.

        self.coord._daily_history = {
            "2023-01-01": {
                "temp": 10.0,
                "wind": 5.0,
                "kwh": 24.0,
                "tdd": 5.0,
                "hourly_vectors": {
                    "temp": [10.0] * 24, # 10 < 15 => Heating
                    "wind": [5.0] * 24,
                    "tdd": [0.2] * 24,
                    "actual_kwh": [1.0] * 24,
                    "solar_rad": [0.0] * 24
                }
            },
            "2023-01-02": {
                "temp": 20.0,
                "wind": 5.0,
                "kwh": 24.0,
                "tdd": 5.0,
                "hourly_vectors": {
                    "temp": [20.0] * 24, # 20 > 15 => Cooling
                    "wind": [5.0] * 24,
                    "tdd": [0.2] * 24,
                    "actual_kwh": [1.0] * 24,
                    "solar_rad": [0.0] * 24
                }
            }
        }
        # StatisticsManager checks if hourly_vectors has valid data.
        # It handles 'actual_kwh' vs 'load'.

    def test_calculate_modeled_energy_optimized_path(self):
        start = date(2023, 1, 1)
        end = date(2023, 1, 2)

        # Run calculation
        self.stats.calculate_modeled_energy(start, end)

        # Verify calculate_total_power was called
        # We have 2 days * 24 hours = 48 calls.
        self.assertEqual(self.stats.calculate_total_power.call_count, 48)

        # Verify unit_modes passed correctly
        # Day 1: Heating (Temp 10 < 15)
        # Day 2: Cooling (Temp 20 > 15)

        calls = self.stats.calculate_total_power.call_args_list

        # Check a call from Day 1
        call_args_day1 = calls[0]
        kwargs_day1 = call_args_day1.kwargs
        unit_modes_day1 = kwargs_day1['unit_modes']

        self.assertEqual(len(unit_modes_day1), 5)
        for mode in unit_modes_day1.values():
            self.assertEqual(mode, MODE_HEATING)

        # Check a call from Day 2 (should be later in the list)
        # The first 24 calls are Day 1, next 24 are Day 2.
        call_args_day2 = calls[24]
        kwargs_day2 = call_args_day2.kwargs
        unit_modes_day2 = kwargs_day2['unit_modes']

        self.assertEqual(len(unit_modes_day2), 5)
        for mode in unit_modes_day2.values():
            self.assertEqual(mode, MODE_COOLING)

    def test_identity_preservation(self):
        """Verify that pre-computed dictionaries are reused (identity check)."""
        # This confirms we are using the optimized pre-computed dicts, not creating new ones.

        start = date(2023, 1, 1)
        end = date(2023, 1, 1) # Just one day

        self.stats.calculate_modeled_energy(start, end)

        calls = self.stats.calculate_total_power.call_args_list

        first_call_modes = calls[0].kwargs['unit_modes']
        second_call_modes = calls[1].kwargs['unit_modes']

        # They should be the SAME object in memory
        self.assertIs(first_call_modes, second_call_modes)

if __name__ == '__main__':
    unittest.main()
