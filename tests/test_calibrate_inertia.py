import math
from tests.helpers import CoordinatorModelMixin
import pytest
from datetime import datetime, timedelta
from homeassistant.util import dt as dt_util
from custom_components.heating_analytics.statistics import StatisticsManager

class MockCoordinator(CoordinatorModelMixin):
    def __init__(self):
        self.solar_azimuth = 180
        self.balance_point = 17.0
        self._hourly_log = []
        self._correlation_data = {}

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
    bp = mock_coordinator.balance_point

    logs = []
    temps = []

    # Generate 20 days (enough for stability check)
    start = now - timedelta(days=20)

    for i in range(20 * 24):
        t_now = start + timedelta(hours=i)

        temp = 5.0 + 5.0 * math.sin(i / 12.0)
        temps.append(temp)

        if len(temps) >= 4:
            # Consumption generated from the 4-hour Gaussian kernel (Older -> Newer)
            w = [0.05762880217217644, 0.25827437283171767, 0.4258224521643882, 0.25827437283171767]
            eff_temp = sum(t * weight for t, weight in zip(temps[-4:], w))
            tdd = max(0.0, bp - eff_temp) / 24.0
            actual_kwh = tdd * 15.0
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

    # Primary result: exponential tau (data generated from 4h Gaussian → best tau nearby)
    assert "recommended_tau" in result
    assert 1 <= result["recommended_tau"] <= 8
    assert result["recommended_tau_r2"] > 0.95

    # Gaussian comparison fields present
    assert "gaussian_best_hours" in result
    assert result["gaussian_best_hours"] in [3, 4]
    assert result["gaussian_best_r2"] > 0.95

    # Stability: weekly_breakdown uses best_tau (not best_hours)
    assert len(result["weekly_breakdown"]) > 0
    for week in result["weekly_breakdown"]:
        assert "best_tau" in week
        assert 1 <= week["best_tau"] <= 24


def test_calibrate_inertia_with_centered_energy(statistics_manager, mock_coordinator):
    now = dt_util.now()

    logs = []
    temps = []
    start = now - timedelta(days=5)

    for i in range(5 * 24):
        t_now = start + timedelta(hours=i)
        temp = 10.0 + 5.0 * math.sin(i / 12.0)
        temps.append(temp)

        # Noisy consumption pattern – smoothing should cancel it out
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

    result_normal = statistics_manager.calibrate_inertia(days=5, centered_energy_average=False)
    result_smoothed = statistics_manager.calibrate_inertia(days=5, centered_energy_average=True)

    assert "error" not in result_normal
    assert "error" not in result_smoothed

    # Smoothing should reduce RMSE on the best exponential fit
    assert result_smoothed["recommended_tau_rmse"] < result_normal["recommended_tau_rmse"], (
        "Smoothed RMSE should be lower than unsmoothed"
    )
