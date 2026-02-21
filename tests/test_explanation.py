"""Tests for the explanation module."""
from datetime import date
import pytest
from custom_components.heating_analytics.explanation import (
    CategoryThresholds,
    WeatherImpactAnalyzer,
    ExplanationFormatter
)

# Dummy Coordinator for testing (mocking _get_wind_bucket)
class MockCoordinator:
    def _get_wind_bucket(self, wind_speed, ignore_aux=False):
        if wind_speed >= 10.8:
            return 'extreme_wind'
        elif wind_speed >= 5.5:
            return 'high_wind'
        return 'normal'

@pytest.fixture
def analyzer():
    return WeatherImpactAnalyzer(coordinator=MockCoordinator())

@pytest.fixture
def formatter():
    return ExplanationFormatter()

def test_category_thresholds():
    """Test threshold logic."""
    # Wind Impact
    assert CategoryThresholds.get_wind_impact('normal', 'normal') == 'normal'
    assert CategoryThresholds.get_wind_impact('high_wind', 'normal') == 'significant'
    assert CategoryThresholds.get_wind_impact('extreme_wind', 'normal') == 'extreme'
    assert CategoryThresholds.get_wind_impact('extreme_wind', 'high_wind') == 'significant'
    assert CategoryThresholds.get_wind_impact('normal', 'high_wind') == 'significant' # Significant change (calmer)

    # Combined Severity
    # Temp(Sig=2) + Wind(Sig=2) = 4 -> Challenging
    assert CategoryThresholds.get_combined_severity('significant', 'significant') == 'challenging'
    # Temp(Ext=3) + Wind(Sig=2) = 5 -> Severe
    assert CategoryThresholds.get_combined_severity('extreme', 'significant') == 'severe'
    # Temp(Mod=1) + Wind(Norm=0) = 1 -> Notable
    assert CategoryThresholds.get_combined_severity('moderate', 'normal') == 'notable'

def test_analyze_day_extreme_cold(analyzer):
    """Test analyzing a single day with extreme cold."""
    day = {'temp': -10.0, 'wind': 2.0, 'wind_bucket': 'normal', 'kwh': 50.0}
    base = {'temp': -4.0, 'wind': 2.0, 'wind_bucket': 'normal', 'kwh': 40.0}

    res = analyzer.analyze_day(day, base)

    # Delta = -6.0 -> Extreme (>5.0)
    assert res['temp_delta'] == -6.0
    assert res['temp_impact'] == 'extreme'
    assert res['wind_impact'] == 'normal'
    assert res['combined_severity'] == 'challenging' # Extreme(3) -> Challenging (>=3)

    # Causality: Colder -> More consumption
    assert res['causality']['temp_explains'] is True
    assert res['causality']['wind_explains'] is False

def test_analyze_day_wind_impact(analyzer):
    """Test analyzing a single day with high wind."""
    day = {'temp': 0.0, 'wind': 7.0, 'wind_bucket': 'high_wind', 'kwh': 45.0}
    base = {'temp': 0.0, 'wind': 3.0, 'wind_bucket': 'normal', 'kwh': 35.0}

    res = analyzer.analyze_day(day, base)

    assert res['wind_impact'] == 'significant' # Normal -> High
    assert res['causality']['wind_explains'] is True

def test_week_ahead_jan_2026(analyzer, formatter):
    """Real-world test case from production (Jan 1-7)."""
    # Test Data provided by user
    period_days = [
        {'date': '2025-12-29', 'temp': -3.8, 'wind': 3.6, 'kwh': 42.67},
        {'date': '2025-12-30', 'temp': -2.7, 'wind': 2.7, 'kwh': 41.5},
        {'date': '2025-12-31', 'temp': -3.2, 'wind': 2.2, 'kwh': 41.1},
        {'date': '2026-01-01', 'temp': -1.8, 'wind': 3.7, 'kwh': 38.6},
        {'date': '2026-01-02', 'temp': -4.9, 'wind': 6.1, 'kwh': 48.4}, # Wind jump? 6.1 -> High?
        {'date': '2026-01-03', 'temp': -11.6, 'wind': 4.7, 'kwh': 67.3}, # Extreme cold (-6.1 vs baseline -5.5) -> -11.6 is 6.1 colder
        {'date': '2026-01-04', 'temp': -12.9, 'wind': 2.8, 'kwh': 73.9}, # Extreme cold
    ]

    # Pre-process buckets for the test data (mimic coordinator)
    for d in period_days:
        d['wind_bucket'] = 'high_wind' if d['wind'] >= 5.5 else 'normal'

    # Baseline is a single aggregated dict in user example, but analyze_period expects list.
    # We must expand it to daily baselines for the period.
    base_avg = {'temp': -5.5, 'wind': 4.0, 'kwh': 45.8, 'wind_bucket': 'normal'}
    baseline_days = [base_avg.copy() for _ in range(7)]

    analysis = analyzer.analyze_period(period_days, baseline_days, context='week_ahead')

    # 1. Verify totals
    # Total kWh = 42.67+41.5+41.1+38.6+48.4+67.3+73.9 = 353.47
    # Analysis rounds to 1 decimal -> 353.5
    assert analysis['total_kwh'] == 353.5

    # 2. Verify Counts
    # Expect 2 Challenging days (Jan 3, 4) and 1 Notable (Jan 2).
    # "Variable week" based on Challenging > 1.

    assert analysis['day_counts']['severe'] == 0
    assert analysis['day_counts']['challenging'] >= 2
    assert analysis['variability'] == 'medium' # Challenging > 1 -> medium

    # 3. Verify Formatting
    text = formatter.format_week_ahead(analysis)

    # Expected: "Challenging week: 354 kWh (+10% vs typical). Driven by 2 challenging days (extreme cold)"
    # Note: 353.5 kWh rounds to 354.

    assert "Challenging week" in text
    assert "354 kWh" in text
    assert "+10%" in text
    assert "2 challenging days" in text
    assert "extreme cold" in text

def test_analyze_period_mixed_drivers(analyzer):
    """Test a week with both cold and wind drivers."""
    # 3 days Cold (Extreme), 2 days Windy (Significant)
    p_days = []
    b_days = []

    # 3 Cold days
    for _ in range(3):
        p_days.append({'temp': -10.0, 'wind': 2.0, 'wind_bucket': 'normal', 'kwh': 60})
        b_days.append({'temp': -2.0, 'wind': 2.0, 'wind_bucket': 'normal', 'kwh': 40})

    # 2 Windy days
    for _ in range(2):
        p_days.append({'temp': -2.0, 'wind': 8.0, 'wind_bucket': 'high_wind', 'kwh': 55})
        b_days.append({'temp': -2.0, 'wind': 3.0, 'wind_bucket': 'normal', 'kwh': 40})

    analysis = analyzer.analyze_period(p_days, b_days)

    # 3 Challenging (Cold), 2 Notable (Wind) -> 5 days total impact
    assert analysis['day_counts']['challenging'] == 3

    drivers = analysis['drivers']
    assert len(drivers) == 2
    assert drivers[0]['factor'] == 'temp' # Extreme overrides Significant
    assert drivers[1]['factor'] == 'wind'

def test_analyze_day_with_none_values(analyzer):
    """Test analyzing a day with None values (missing data scenario)."""
    # Scenario: Missing wind and kwh data (None values)
    day = {'temp': -5.0, 'wind': None, 'wind_bucket': 'normal', 'kwh': None}
    base = {'temp': 0.0, 'wind': 3.0, 'wind_bucket': 'normal', 'kwh': 40.0}

    # Should NOT crash
    res = analyzer.analyze_day(day, base)

    # Verify handling
    assert res['temp_delta'] == -5.0  # Temp calculation works
    assert res['wind_delta'] == -3.0  # None treated as 0.0: 0.0 - 3.0 = -3.0
    assert res['kwh_delta'] == -40.0  # None treated as 0.0: 0.0 - 40.0 = -40.0

def test_format_day_comparison(formatter):
    """Test formatting of day comparison."""
    # Cold day
    analysis = {
        'delta_kwh': 5.5,
        'causality': {'temp_explains': True},
        'temp_impact': 'significant'
    }
    text = formatter.format_day_comparison(analysis)
    # New format: "Colder weather (+5.5 kWh vs last year)"
    assert "Colder weather (+5.5 kWh vs last year)" in text

    # Windy day
    analysis = {
        'delta_kwh': 3.5,
        'causality': {'wind_explains': True},
        'wind_impact': 'significant'
    }
    text = formatter.format_day_comparison(analysis)
    # New format: "High wind (+3.5 kWh vs last year)"
    assert "High wind (+3.5 kWh vs last year)" in text

def test_format_forecast_weather_context(formatter):
    """Test formatting of absolute weather context for Forecast Today."""

    # Extreme cold
    text = formatter.format_forecast_weather_context(temp=-11.6, wind=2.0)
    assert "Driven by extreme cold (-11.6°C)" in text
    assert "stormy" not in text  # Wind not extreme

    # Very cold with strong wind
    text = formatter.format_forecast_weather_context(temp=-3.0, wind=7.0, wind_high_threshold=5.5)
    assert "Driven by very cold (-3.0°C) and strong wind" in text

    # Cold with stormy conditions
    text = formatter.format_forecast_weather_context(temp=3.0, wind=12.0, wind_extreme_threshold=10.8)
    assert "Driven by cold (3.0°C) and stormy conditions" in text

    # Chilly with breezy conditions
    text = formatter.format_forecast_weather_context(temp=8.0, wind=4.0)
    assert "Driven by chilly (8.0°C) and breezy conditions" in text

    # Mild conditions (no significant drivers)
    text = formatter.format_forecast_weather_context(temp=15.0, wind=2.0)
    assert "Mild conditions (15.0°C)" in text
    assert "Driven by" not in text

    # Warm day
    text = formatter.format_forecast_weather_context(temp=20.0, wind=1.0)
    assert "Mild conditions (20.0°C)" in text

    # Hot day
    text = formatter.format_forecast_weather_context(temp=25.0, wind=3.5)
    assert "Driven by hot (25.0°C) and breezy conditions" in text

    # Missing temperature
    text = formatter.format_forecast_weather_context(temp=None, wind=5.0)
    assert text == "Weather data unavailable"

    # Missing wind (should still work)
    text = formatter.format_forecast_weather_context(temp=-5.0, wind=None)
    assert "Driven by very cold (-5.0°C)" in text
    assert "wind" not in text.lower()

    # Boundary: exactly 12°C (just mild)
    text = formatter.format_forecast_weather_context(temp=12.0, wind=1.0)
    assert "Mild conditions (12.0°C)" in text

    # Boundary: just below 12°C (chilly, significant)
    text = formatter.format_forecast_weather_context(temp=11.9, wind=1.0)
    assert "Driven by chilly (11.9°C)" in text

def test_format_behavioral_deviation_guest_mode(formatter):
    """Test that guest mode explanation adapts based on guest impact ratio."""
    top_contributor = {"name": "Living Room Heater", "deviation": 1.0}
    deviation_pct = 50.0

    # Scenario 1: Dominant guest impact (90% = >50%)
    deviation_kwh = 10.0
    guest_impact_kwh = 9.0
    text = formatter.format_behavioral_deviation(
        deviation_kwh=deviation_kwh,
        deviation_pct=deviation_pct,
        top_contributor=top_contributor,
        weather_impact=None,
        guest_impact_kwh=guest_impact_kwh
    )
    assert "primarily due to guest heaters" in text
    assert "consuming 9.0 kWh" in text
    assert "Living Room Heater" not in text

    # Scenario 2: Significant but not dominant (40% = 30-50%)
    guest_impact_kwh = 4.0
    text = formatter.format_behavioral_deviation(
        deviation_kwh=deviation_kwh,
        deviation_pct=deviation_pct,
        top_contributor=top_contributor,
        weather_impact=None,
        guest_impact_kwh=guest_impact_kwh
    )
    assert "Guest heaters account for 4.0 kWh" in text
    assert "Living Room Heater contributing" in text
    assert "primarily" not in text

    # Scenario 3: Minor guest impact (20% = <30%)
    guest_impact_kwh = 2.0
    text = formatter.format_behavioral_deviation(
        deviation_kwh=deviation_kwh,
        deviation_pct=deviation_pct,
        top_contributor=top_contributor,
        weather_impact=None,
        guest_impact_kwh=guest_impact_kwh
    )
    assert "primarily due to guest heaters" not in text
    assert "Guest heaters account for" not in text
    assert "Living Room Heater" in text

    # Scenario 4: Negative deviation with guest heaters active
    deviation_kwh = -3.0
    guest_impact_kwh = 2.0
    text = formatter.format_behavioral_deviation(
        deviation_kwh=deviation_kwh,
        deviation_pct=-15.0,
        top_contributor=top_contributor,
        weather_impact=None,
        guest_impact_kwh=guest_impact_kwh
    )
    assert "despite guest heaters consuming 2.0 kWh" in text
    assert "excellent efficiency" in text

    # Scenario 5: Low deviation, guest mode active (should not trigger)
    deviation_kwh = 0.4
    guest_impact_kwh = 0.4
    text = formatter.format_behavioral_deviation(
        deviation_kwh=deviation_kwh,
        deviation_pct=deviation_pct,
        top_contributor=top_contributor,
        weather_impact=None,
        guest_impact_kwh=guest_impact_kwh
    )
    assert "matches expectations" in text
