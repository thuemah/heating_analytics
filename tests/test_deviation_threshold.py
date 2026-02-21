"""Test Deviation Threshold Logic."""
import pytest
from custom_components.heating_analytics.statistics import StatisticsManager

class MockCoordinator:
    def __init__(self):
        self.data = {}
        self.hass = None
        self.energy_sensors = []
        self._daily_individual = {}
        self._hourly_log = []
        self._hourly_delta_per_unit = {}
        self._accumulated_energy_hour = 0.0
        # Add required attributes for StatisticsManager
        self.wind_unit = "km/h"
        self.wind_threshold = 20.0
        self.extreme_wind_threshold = 50.0

def test_deviation_threshold_continuity():
    """Verify that deviation threshold decays smoothly without jumps."""
    coord = MockCoordinator()
    stats = StatisticsManager(coord)

    # Check the critical transition point (19 -> 20 observations)
    # Previous logic had a jump from 0.50 to 0.30 here.

    _, _, t19 = stats._is_deviation_unusual(1.0, 10.0, 19)
    _, _, t20 = stats._is_deviation_unusual(1.0, 10.0, 20)

    diff = t19 - t20

    # Allow for very small floating point differences, but not 0.2
    assert diff < 0.05, f"Discontinuity detected: Jump of {diff} between obs 19 and 20"

    # Verify bounds for active range (obs >= 5)
    # At 5 obs: 0.75 - (5 * 0.0225) = 0.75 - 0.1125 = 0.6375
    _, _, t5 = stats._is_deviation_unusual(1.0, 10.0, 5)
    expected_t5 = 0.75 - (5 * 0.0225)
    assert abs(t5 - expected_t5) < 0.001, f"Threshold at 5 obs incorrect. Got {t5}, expected {expected_t5}"

    # Verify clamping at high observations
    _, _, t100 = stats._is_deviation_unusual(1.0, 10.0, 100)
    assert t100 == 0.30, "Mature threshold should be clamped at 0.30"
