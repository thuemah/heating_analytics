"""Tests for design compliance (DESIGN.md claims)."""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime
from custom_components.heating_analytics.statistics import StatisticsManager
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.const import (
    ATTR_TDD, ATTR_TDD_SO_FAR, ATTR_ENERGY_TODAY,
    ATTR_TDD_YESTERDAY, ATTR_EFFICIENCY_YESTERDAY,
    ATTR_FORECAST_TODAY
)

def test_rolling_efficiency_logic():
    """Test Seamless Rolling Efficiency (Design.md Section 4).
    Claim: If TDD_Today < 0.5, it borrows data from Yesterday to fill the denominator.
    """
    coordinator = MagicMock()
    # Scenario: Low TDD So Far (0.1), Good Yesterday data
    coordinator.data = {
        ATTR_TDD_SO_FAR: 0.1,
        ATTR_ENERGY_TODAY: 0.5,
        ATTR_TDD_YESTERDAY: 10.0,
        ATTR_EFFICIENCY_YESTERDAY: 3.0, # 30 kWh total yesterday
    }
    stats = StatisticsManager(coordinator)

    # Logic Verification:
    # Target Window = 0.5. Needed = 0.5 - 0.1 = 0.4.
    # Borrowed TDD = 0.4 (Available 10.0, so we take 0.4)
    # Borrowed Energy = Efficiency (3.0) * Borrowed TDD (0.4) = 1.2 kWh.
    # Total Energy = 0.5 + 1.2 = 1.7.
    # Total TDD = 0.1 + 0.4 = 0.5.
    # Expected Result: 1.7 / 0.5 = 3.4.

    result = stats.calculate_realtime_efficiency()
    assert result == 3.4

def test_typical_day_normalization():
    """Test Typical Day Normalization (Design.md Section 2C).
    Claim: Typical = Actual_kWh + Aux_Impact_kWh.
    """
    coordinator = MagicMock()
    # Setup history with Aux Impact
    coordinator._daily_history = {
        "2023-01-01": {"temp": 10.0, "wind": 2.0, "kwh": 20.0, "aux_impact_kwh": 5.0}, # Base = 25
        "2023-01-02": {"temp": 10.0, "wind": 2.0, "kwh": 24.0, "aux_impact_kwh": 0.0}, # Base = 24
        "2023-01-03": {"temp": 10.0, "wind": 2.0, "kwh": 20.0, "aux_impact_kwh": 6.0}, # Base = 26
    }
    stats = StatisticsManager(coordinator)

    # Logic Verification:
    # Samples: [25, 24, 26] -> Sorted: [24, 25, 26] -> Median: 25.

    val, count, conf = stats.get_typical_day_consumption(10.0)
    assert val == 25.0
    assert count == 3

def test_hybrid_projection_funnel():
    """Test Hybrid Projection 'The Funnel' (Design.md Section 3D).
    Claim: Forecast_Today = (Actual_kWh_So_Far) + (Predicted_kWh_Remaining).
    """
    hass = MagicMock()
    entry = MagicMock()
    entry.data = {}

    with patch("custom_components.heating_analytics.storage.Store"), \
         patch("custom_components.heating_analytics.coordinator.dt_util.now") as mock_now:

        coordinator = HeatingDataCoordinator(hass, entry)
        coordinator.forecast = MagicMock()

        # Setup State
        coordinator._accumulated_energy_today = 60.0 # Actual So Far (Includes current hour)
        coordinator._accumulated_energy_hour = 2.0 # Actual This Hour (Included in above)
        coordinator.auxiliary_heating_active = False

        # Mocks
        # Future Forecast (Rest of day after this hour)
        coordinator.forecast.calculate_future_energy.return_value = (50.0, 0.0, {})
        # Current Hour Forecast Plan
        coordinator.forecast.get_plan_for_hour.return_value = (4.0, {})

        # Time: 12:30
        current_time = datetime(2023, 1, 1, 12, 30)
        mock_now.return_value = current_time

        # Execution
        # _update_daily_budgets calculates:
        # Remaining Current Hour = Max(0, Plan(4.0) - Actual(2.0)) = 2.0
        # Forecast Today = Actual_Total(60.0) + Remaining(2.0) + Future(50.0) = 112.0

        coordinator._update_daily_budgets(0.0, current_time, 30)

        assert coordinator.data[ATTR_FORECAST_TODAY] == 112.0
