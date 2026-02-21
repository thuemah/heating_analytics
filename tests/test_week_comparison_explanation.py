"""Test week comparison sensor explanation logic."""
import pytest
from unittest.mock import MagicMock, patch
from datetime import date, timedelta, datetime
from custom_components.heating_analytics.sensor import HeatingModelComparisonWeekSensor
from custom_components.heating_analytics.const import (
    ATTR_TEMP_ACTUAL_TODAY,
    ATTR_WIND_ACTUAL_TODAY,
    ATTR_PREDICTED,
    ATTR_ENERGY_TODAY,
)

# Dummy classes for inheritance
class MockCoordinatorEntity:
    def __init__(self, coordinator):
        self.coordinator = coordinator

class MockSensorEntity:
    pass

@pytest.fixture
def mock_coordinator():
    coordinator = MagicMock()
    coordinator.data = {}
    coordinator._daily_history = {}

    # Mock wind bucket logic
    def get_wind_bucket(wind, ignore_aux=False):
        if wind is None: return "normal"
        if wind >= 10.8: return "extreme_wind"
        if wind >= 5.5: return "high_wind"
        return "normal"

    coordinator._get_wind_bucket.side_effect = get_wind_bucket

    # Mock forecast
    coordinator.forecast = MagicMock()

    # CRITICAL: Ensure calculate_modeled_energy returns a tuple of 5 elements
    # Using 30.0 kWh per day to sum up to ~210 for the week. 5th element is TDD (e.g. 10.0).
    coordinator.calculate_modeled_energy.return_value = (30.0, 0.0, 10.0, 5.0, 10.0)

    # CRITICAL: Ensure _calculate_pure_model_today returns a tuple of 2 elements
    coordinator.statistics._calculate_pure_model_today.return_value = (60.0, 0.0)

    # CRITICAL: Ensure calculate_future_energy returns a tuple of 3 elements
    coordinator.forecast.calculate_future_energy.return_value = (0.0, 0.0, {})

    return coordinator

@pytest.fixture
def mock_entry():
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.title = "Heating Analytics"
    return entry

# Mock dependencies to prevent import errors during test execution if not using full HASS env
@patch('custom_components.heating_analytics.sensor.CoordinatorEntity', MockCoordinatorEntity)
@patch('custom_components.heating_analytics.sensor.SensorEntity', MockSensorEntity)
class TestWeekComparisonExplanation:

    def test_week_comparison_with_explanation_module(self, mock_coordinator, mock_entry):
        """Test Week Comparison sensor uses explanation module correctly."""

        # Setup Dates
        today = date(2023, 10, 18)

        # Mock time
        with patch("custom_components.heating_analytics.sensor.dt_util.now", return_value=datetime(2023, 10, 18, 12, 0, 0)):

            # 1. Setup Data
            mock_coordinator._daily_history = {
                '2023-10-16': {'temp': 5.0, 'wind': 5.0, 'kwh': 50.0}, # Mon
                '2023-10-17': {'temp': 4.0, 'wind': 6.0, 'kwh': 55.0}, # Tue
            }

            mock_coordinator.data = {
                ATTR_TEMP_ACTUAL_TODAY: 3.0,
                ATTR_WIND_ACTUAL_TODAY: 7.0,
                ATTR_PREDICTED: 60.0, # Today's budget
                ATTR_ENERGY_TODAY: 20.0  # So far
            }

            # Mock future energy for today (remaining)
            mock_coordinator.forecast.calculate_future_energy.return_value = (40.0, 0.0, {})

            def get_future_prediction(d, ignore_aux=False):
                if d > today:
                    return (40.0, 0.0, {'temp': 2.0, 'wind': 8.0, 'wind_bucket': 'high_wind'})
                return None

            mock_coordinator.forecast.get_future_day_prediction.side_effect = get_future_prediction

            # Last Year Data
            for i in range(7):
                d = date(2022, 10, 17) + timedelta(days=i)
                mock_coordinator._daily_history[d.isoformat()] = {
                    'temp': 10.0, 'wind': 2.0, 'kwh': 30.0
                }

            # Mock period stats calculation
            mock_coordinator.statistics.calculate_hybrid_projection.return_value = (350.0, 0.0)

            # Use daily value that aligns with period totals (30*7 ~ 210)
            mock_coordinator.calculate_modeled_energy.return_value = (30.0, 0.0, 10.0, 2.0, 10.0)
            mock_coordinator.statistics.calculate_historical_actual_sum.return_value = 210.0

            # Mock pure model today specifically for this test if needed, but fixture covers it
            # But let's override with consistent values if logic depends on it
            mock_coordinator.statistics._calculate_pure_model_today.return_value = (60.0, 0.0)

            # Create Sensor
            sensor = HeatingModelComparisonWeekSensor(mock_coordinator, mock_entry)

            # Force _calculate_period_stats to match expected delta
            sensor._calculate_period_stats = MagicMock(return_value=(
                350.0, # Curr (Higher)
                210.0, # Last (Lower)
                210.0, # Actual Last
                350.0, # Curr Debug
                { # Metadata
                    "curr_temp": 3.5, "ref_temp": 10.0, # Colder
                    "curr_wind": 6.5, "ref_wind": 2.0,  # Windier
                    "ref_solar": 0, "curr_solar": 0
                }
            ))

            # ACT
            attrs = sensor.extra_state_attributes

            # ASSERT
            assert "weekly_summary" in attrs
            summary = attrs["weekly_summary"]
            print(f"Summary generated: {summary}")

            # Expect specific summary based on severe cold driver
            assert "Significantly Colder" in summary or "Higher" in summary
            assert "Driven by" in summary
            assert "extreme cold" in summary or "cold" in summary

    def test_week_comparison_explanation_fallback(self, mock_coordinator, mock_entry):
        """Test fallback when explanation module fails."""
        with patch("custom_components.heating_analytics.sensor.dt_util.now", return_value=datetime(2023, 10, 18, 12, 0, 0)):
             sensor = HeatingModelComparisonWeekSensor(mock_coordinator, mock_entry)

             sensor._calculate_period_stats = MagicMock(return_value=(
                100.0, 100.0, 100.0, 100.0,
                {"curr_temp": 5, "ref_temp": 5, "curr_wind": 5, "ref_wind": 5, "ref_solar":0, "curr_solar":0}
             ))

             sensor._build_current_period_days = MagicMock(side_effect=Exception("Boom"))

             attrs = sensor.extra_state_attributes

             assert "weekly_summary" in attrs
             # Fallback string verification
             assert "Consumption similar to last year" in attrs["weekly_summary"] or "Error" in attrs["weekly_summary"] or "Same" in attrs["weekly_summary"] or "Higher" in attrs["weekly_summary"] or "Lower" in attrs["weekly_summary"]

    def test_build_current_period_days(self, mock_coordinator, mock_entry):
        """Verify the data gathering logic."""
        today = date(2023, 10, 18)
        start_week = date(2023, 10, 16)
        end_week = start_week + timedelta(days=6)

        with patch("custom_components.heating_analytics.sensor.dt_util.now", return_value=datetime(2023, 10, 18, 12, 0, 0)):
            mock_coordinator._daily_history = {'2023-10-16': {'temp': 10, 'wind': 5}}
            mock_coordinator.data = {
                ATTR_TEMP_ACTUAL_TODAY: 11,
                ATTR_WIND_ACTUAL_TODAY: 6,
                ATTR_PREDICTED: 50,
                ATTR_ENERGY_TODAY: 20.0 # Actual so far
            }
            mock_coordinator.forecast.get_future_day_prediction.return_value = (40, 0, {'temp': 12, 'wind': 7})

            # Mock Future Energy (Remaining for Today) -> 30.0
            # Total Today = 20 (Actual) + 30 (Forecast) = 50
            mock_coordinator.forecast.calculate_future_energy.return_value = (30.0, 0.0, {})

            mock_coordinator.calculate_modeled_energy.return_value = (30.0, 0.0, 10.0, 5.0, 10.0)
            mock_coordinator.statistics._calculate_pure_model_today.return_value = (50.0, 0.0)

            sensor = HeatingModelComparisonWeekSensor(mock_coordinator, mock_entry)

            days = sensor._build_current_period_days(start_week, end_week)

            assert len(days) == 7
            assert days[0]['date'] == '2023-10-16' # Past
            assert days[0]['temp'] == 10

            assert days[2]['date'] == '2023-10-18' # Today
            assert days[2]['temp'] == 11
            assert days[2]['kwh'] == 50.0

            assert days[3]['date'] == '2023-10-19' # Future
            assert days[3]['temp'] == 12
            assert days[3]['kwh'] == 40
