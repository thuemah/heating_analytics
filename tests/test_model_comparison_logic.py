"""Test Model Comparison Logic consistency."""
import pytest
from unittest.mock import MagicMock, patch
from datetime import date, datetime, timedelta
import sys

# Imports from custom_components (mocks provided by conftest.py)
from custom_components.heating_analytics.sensor import HeatingModelComparisonWeekSensor, HeatingModelComparisonDaySensor
from custom_components.heating_analytics.const import ATTR_PREDICTED, ATTR_SOLAR_PREDICTED

@pytest.fixture
def mock_coordinator():
    coordinator = MagicMock()
    coordinator.data = {
        ATTR_PREDICTED: 10.0, # Today's model
        ATTR_SOLAR_PREDICTED: 0.0,
    }
    # Mock calculate_modeled_energy to capture calls and return dummy values
    # Must return 5-tuple now: (kwh, solar, temp, wind, tdd)
    coordinator.calculate_modeled_energy = MagicMock(return_value=(50.0, 0.0, 10.0, 5.0, 10.0))

    # Mock forecast.get_future_day_prediction to avoid ValueError during unpacking
    # Return None to trigger fallback logic (or a valid tuple if needed)
    coordinator.forecast.get_future_day_prediction = MagicMock(return_value=None)

    # Mock forecast.calculate_future_energy to avoid ValueError during unpacking in _get_today_data
    coordinator.forecast.calculate_future_energy = MagicMock(return_value=(0.0, 0.0, None))

    return coordinator

@pytest.fixture
def mock_entry():
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.title = "Test Heating Analytics"
    return entry

@pytest.mark.asyncio
async def test_week_sensor_iso_logic(mock_coordinator, mock_entry):
    """Test that Week Sensor uses ISO week logic for Last Year."""

    # Instantiate Sensor
    sensor = HeatingModelComparisonWeekSensor(mock_coordinator, mock_entry)

    # Date: Monday Jan 1st 2024 (ISO 2024-W01-1)
    # Last Year ISO Week 1 2023 starts on Jan 2nd 2023 (Monday)

    mock_now = datetime(2024, 1, 1, 12, 0, 0)

    with patch("custom_components.heating_analytics.sensor.dt_util.now", return_value=mock_now):
        # Mock the new hybrid projection method
        mock_coordinator.statistics.calculate_hybrid_projection.return_value = (50.0, 5.0)
        # Mock historical actuals
        mock_coordinator.statistics.calculate_historical_actual_sum.return_value = 12.8

        # Trigger calculation
        curr, last, actual, model, meta = sensor._calculate_period_stats(
            start_date=date(2024, 1, 1),
            period_type="week",
            total_days_in_period=7
        )

        # Verify return values
        assert curr == 50.0
        assert actual == 12.8
        # model is the 4th element (current hybrid total)
        assert model == 50.0

    # Analyze calls to calculate_modeled_energy
    calls = mock_coordinator.calculate_modeled_energy.call_args_list

    found_ly_start = False
    used_date = None

    # We look for the call corresponding to the Last Year period
    for call in calls:
        args = call[0]
        start_d = args[0]
        if start_d.year == 2023:
            found_ly_start = True
            used_date = start_d
            break

    assert found_ly_start, "Did not find a call for last year"

    # Verify exact ISO date alignment
    # Jan 2, 2023 is the Monday of ISO Week 1 in 2023
    assert used_date == date(2023, 1, 2), f"Expected Jan 2 2023 (ISO), got {used_date}"

@pytest.mark.asyncio
async def test_day_sensor_calendar_logic(mock_coordinator, mock_entry):
    """Test that Day Sensor still uses Calendar logic."""

    sensor = HeatingModelComparisonDaySensor(mock_coordinator, mock_entry)

    # Date: Monday Jan 1st 2024
    # Last Year Day should be Jan 1st 2023 (Sunday)
    mock_now = datetime(2024, 1, 1, 12, 0, 0)

    with patch("custom_components.heating_analytics.sensor.dt_util.now", return_value=mock_now):
        mock_coordinator.statistics.calculate_hybrid_projection.return_value = (10.0, 1.0)
        mock_coordinator.statistics.calculate_historical_actual_sum.return_value = 5.5

        # Also trigger get_or_calculate_stats to verify attribute flow if we were calling properties
        # But here we test internal method logic mainly. Let's stick to existing pattern but verify values.
        curr, last, actual, model, meta = sensor._calculate_period_stats(
            start_date=date(2024, 1, 1),
            period_type="day",
            total_days_in_period=1
        )

        # Manually invoke attribute property logic to verify assignment
        # Since _calculate_period_stats is stateless regarding the sensor instance attributes in this test context,
        # we check the return values are correct for passing to attributes.
        assert actual == 5.5
        assert model == 10.0

        # Verify attributes (must be inside patch context)
        attrs = sensor.extra_state_attributes
        assert "last_year_actual_kwh" in attrs
        assert attrs["last_year_actual_kwh"] == 5.5
        assert "current_model_kwh" in attrs
        assert attrs["current_model_kwh"] == 10.0

    calls = mock_coordinator.calculate_modeled_energy.call_args_list

    found_ly_start = False
    used_date = None

    for call in calls:
        args = call[0]
        start_d = args[0]
        if start_d.year == 2023:
            found_ly_start = True
            used_date = start_d
            break

    assert found_ly_start, "Did not find a call for last year"

    # Verify Calendar alignment
    assert used_date == date(2023, 1, 1), f"Expected Jan 1 2023 (Calendar), got {used_date}"

@pytest.mark.asyncio
async def test_last_year_actual_missing_data(mock_coordinator, mock_entry):
    """Test that missing historical data is handled gracefully (None)."""
    sensor = HeatingModelComparisonDaySensor(mock_coordinator, mock_entry)
    mock_now = datetime(2024, 1, 1, 12, 0, 0)

    with patch("custom_components.heating_analytics.sensor.dt_util.now", return_value=mock_now):
        # Mock missing data (return None)
        mock_coordinator.statistics.calculate_historical_actual_sum.return_value = None
        mock_coordinator.statistics.calculate_hybrid_projection.return_value = (10.0, 1.0)

        # Trigger logic
        curr, last, actual, model, meta = sensor._calculate_period_stats(
            start_date=date(2024, 1, 1),
            period_type="day",
            total_days_in_period=1
        )

        assert actual is None

        # Verify attribute handling
        attrs = sensor.extra_state_attributes
        assert attrs["last_year_actual_kwh"] is None
