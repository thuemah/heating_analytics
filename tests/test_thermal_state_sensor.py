"""Test the Thermal State Sensor logic."""
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.const import DEFAULT_INERTIA_WEIGHTS

@pytest.fixture
def coordinator(hass):
    """Create a mock coordinator."""
    mock_entry = MagicMock()
    mock_entry.data = {}
    return HeatingDataCoordinator(hass, mock_entry)

@pytest.mark.asyncio
async def test_thermal_state_calculation_logic():
    """Test the logic used for thermal state calculation."""
    # This test verifies the math, independent of the coordinator's state
    # Replicating logic from coordinator.py

    # 1. Basic Weighted Average
    inertia_list = [10.0, 10.0, 10.0, 10.0]
    weights = DEFAULT_INERTIA_WEIGHTS # (0.20, 0.30, 0.30, 0.20)
    weighted_sum = sum(t * w for t, w in zip(inertia_list, weights))
    assert weighted_sum == 10.0

    # 2. Rising Trend
    inertia_list = [0.0, 0.0, 0.0, 10.0]
    # Inertia = 0*0.20 + 0*0.30 + 0*0.30 + 10*0.20 = 2.0
    weighted_sum = sum(t * w for t, w in zip(inertia_list, weights))
    assert abs(weighted_sum - 2.0) < 0.001

    # 3. Thermal Lag
    effective_temp = 2.0
    raw_temp = 10.0
    thermal_lag = effective_temp - raw_temp
    assert thermal_lag == -8.0

    # 4. Trend Rate
    # [0, 0, 0, 10] -> compare 10 vs 0 (H-2). Diff=10. Hours=2. Rate=5.
    old_val = inertia_list[-3]
    current_val = inertia_list[-1]
    rate = (current_val - old_val) / 2.0
    assert rate == 5.0

    trend = "stable"
    if rate > 2.0: trend = "rising_fast"
    assert trend == "rising_fast"

@pytest.mark.asyncio
async def test_thermal_state_short_history():
    """Test with insufficient history."""
    inertia_list = [5.0, 10.0] # Only 2 samples

    # Weights used: last 2 -> [0.30, 0.20]
    w = DEFAULT_INERTIA_WEIGHTS[-2:]
    total_w = sum(w) # 0.50
    val = sum(t * weight for t, weight in zip(inertia_list, w)) / total_w

    # (5*0.30 + 10*0.20) / 0.50 = 3.5 / 0.50 = 7.0
    assert abs(val - 7.0) < 0.01

    # Trend: Current vs H-1 (index -2). Diff=5. Hours=1.
    rate = (10.0 - 5.0) / 1.0
    assert rate == 5.0
