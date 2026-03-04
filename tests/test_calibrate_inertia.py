import math
import pytest
from datetime import datetime, timedelta
from homeassistant.util import dt as dt_util
from custom_components.heating_analytics.statistics import StatisticsManager

class MockCoordinator:
    def __init__(self):
        self.solar_azimuth = 180
        self.balance_point = 17.0
        self._hourly_log = []

@pytest.fixture
def mock_coordinator():
    return MockCoordinator()

@pytest.fixture
def statistics_manager(mock_coordinator):
    return StatisticsManager(mock_coordinator)

def test_calibrate_inertia_not_enough_data(statistics_manager):
    result = statistics_manager.calibrate_inertia(days=30)
    assert "error" in result
    assert "total_hours_evaluated" in result
    assert "discarded_hours" in result
    assert result["total_hours_evaluated"] == 0

def test_calibrate_inertia_success(statistics_manager, mock_coordinator):
    now = dt_util.now()

    # Generate 15 days of perfect data (r^2 = 1.0)
    # y = m*x + c
    bp = mock_coordinator.balance_point

    logs = []
    temps = []

    # Generate 20 days (enough for stability check)
    start = now - timedelta(days=20)

    for i in range(20 * 24):
        t_now = start + timedelta(hours=i)

        # Simpler temp wave
        temp = 5.0 + 5.0 * math.sin(i / 12.0)
        temps.append(temp)

        if len(temps) >= 4:
            # Let's make actual consumption exactly match a 4-hour inertia
            # Kernel for 4 hours from our formula: (0.0576, 0.2583, 0.4258, 0.2583)
            # Older -> Newer
            # Actually, to make it perfectly hit "4", we need to use the exact kernel
            # the script calculates for h=4.
            w = [0.05762880217217644, 0.25827437283171767, 0.4258224521643882, 0.25827437283171767]
            eff_temp = sum(t * weight for t, weight in zip(temps[-4:], w))

            tdd = max(0.0, bp - eff_temp) / 24.0
            actual_kwh = tdd * 15.0 # slope = 15
        else:
            actual_kwh = max(0.0, bp - temp) / 24.0 * 15.0

        logs.append({
            "timestamp": t_now.isoformat(),
            "temp": temp,
            "actual_kwh": actual_kwh,
            "solar_impact_kwh": 0.0,
            "auxiliary_active": False,
            "learning_status": "active"
        })

    mock_coordinator._hourly_log = logs

    result = statistics_manager.calibrate_inertia(days=15)

    assert "error" not in result
    assert result["days_analyzed"] == 15
    assert "total_hours_evaluated" in result
    assert "discarded_hours" in result
    assert result["total_hours_evaluated"] > 0
    assert result["discarded_hours"]["total_discarded"] >= 0
    print(result["top_5_profiles"])
    assert result["best_overall"]["hours"] in [3, 4] # Allow slight variance due to simple wave
    assert result["best_overall"]["r2"] > 0.95

    # Check stability
    assert len(result["weekly_breakdown"]) > 0
    for week in result["weekly_breakdown"]:
        assert week["best_hours"] in [3, 4]


def test_calibrate_inertia_with_centered_energy(statistics_manager, mock_coordinator):
    now = dt_util.now()

    # Create logs with artificially noisy actual_kwh to see if smoothing helps
    logs = []
    temps = []
    # Need at least 14 days for the test to pass the 'Not enough data points after applying kernel history' check
    # because the function splits data to weeks, etc., actually 5 days should be enough if we just generate enough points.
    # Ah, the problem is that for 24 kernels, and 5*24=120 points, if R^2 calculation fails or something, it might error.
    # Wait, the error is `AssertionError: assert 'error' not in {'error': 'Not enough data points after applying kernel history.'}`
    # Wait! the temp is constant 10.0. That means tdd is constant.
    # `denominator = sum((tdd - mean_tdd)**2 for tdd in tdd_vals)`
    # Since tdd is constant, denominator is 0, so it skips all kernels and returns the error!
    # Let's fix this by adding variation to temp.
    start = now - timedelta(days=5)

    for i in range(5 * 24):
        t_now = start + timedelta(hours=i)
        temp = 10.0 + 5.0 * math.sin(i / 12.0)
        temps.append(temp)

        # Base consumption 5.0, but add a noise pattern that smoothing would cancel out
        # e.g. [5.0, 3.0, 7.0, 5.0, 3.0, 7.0]
        # Smoothed: (5+3+7)/3 = 5, (3+7+5)/3 = 5, etc.
        if i % 3 == 0:
            actual_kwh = 5.0
        elif i % 3 == 1:
            actual_kwh = 3.0
        else:
            actual_kwh = 7.0

        logs.append({
            "timestamp": t_now.isoformat(),
            "temp": temp,
            "actual_kwh": actual_kwh,
            "solar_impact_kwh": 0.0,
            "auxiliary_active": False,
            "learning_status": "active"
        })

    mock_coordinator._hourly_log = logs

    # Test without smoothing
    result_normal = statistics_manager.calibrate_inertia(days=5, centered_energy_average=False)

    # Test with smoothing
    result_smoothed = statistics_manager.calibrate_inertia(days=5, centered_energy_average=True)

    assert "error" not in result_normal
    assert "error" not in result_smoothed

    # The normal calculation should have a lower R^2 due to the added noise
    # The smoothed calculation should have a better fit (higher R^2 or lower RMSE)
    best_normal = max(result_normal["top_5_profiles"], key=lambda k: k["r2"])
    best_smoothed = max(result_smoothed["top_5_profiles"], key=lambda k: k["r2"])

    assert best_smoothed["rmse"] < best_normal["rmse"], "Smoothed RMSE should be lower"
