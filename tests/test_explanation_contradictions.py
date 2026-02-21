"""Tests for explanation contradictions."""
from unittest.mock import MagicMock
import pytest

from custom_components.heating_analytics.explanation import (
    ExplanationFormatter,
    WeatherImpactAnalyzer,
    CategoryThresholds,
)

@pytest.fixture
def analyzer():
    return WeatherImpactAnalyzer()

@pytest.fixture
def formatter():
    return ExplanationFormatter()

def test_temp_driver_wind_contradiction(analyzer, formatter):
    """Test Case 1: Temp driver (+), Wind contradiction (-).

    Example: Colder weather (usage UP) but Less Wind (usage DOWN).
    Total Delta should be positive (Temp wins).
    """
    # Baseline: 0°C, 5 m/s, 50 kWh
    baseline = {
        'temp': 0.0,
        'wind': 5.0,
        'wind_bucket': 'normal',
        'kwh': 50.0
    }
    # Current: -5°C (Colder), 0 m/s (Calmer), 60 kWh
    current = {
        'temp': -5.0, # Delta -5 (Extreme Cold) -> Usage UP
        'wind': 0.0,  # Delta -5 (Calmer) -> Usage DOWN
        'wind_bucket': 'normal',
        'kwh': 60.0   # Delta +10
    }

    analysis = analyzer.analyze_day(current, baseline)
    text = formatter.format_day_comparison(analysis)
    text_lower = text.lower()

    assert analysis['causality']['temp_explains'] is True
    assert analysis['causality']['wind_contradicts'] is True

    # Expected: "Extreme cold, offset by very calm weather (+10.0 kWh vs last year)"
    assert "extreme cold" in text_lower or "colder weather" in text_lower
    assert "offset by" in text_lower
    # "very calm" comes from 'extreme' impact, or 'calmer' from 'moderate'/'significant' contradiction
    assert "very calm" in text_lower or "calmer" in text_lower or "calm" in text_lower
    assert "+10.0 kwh" in text_lower

def test_wind_driver_temp_contradiction(analyzer, formatter):
    """Test Case 2: Wind driver (+), Temp contradiction (-).

    Example: Warmer weather (usage DOWN) but Stormy (usage UP).
    Total Delta should be positive (Wind wins).
    """
    # Baseline: 0°C, 0 m/s, 50 kWh
    baseline = {
        'temp': 0.0,
        'wind': 0.0,
        'wind_bucket': 'normal',
        'kwh': 50.0
    }
    # Current: 5°C (Warmer), 15 m/s (Stormy), 60 kWh
    current = {
        'temp': 5.0,   # Delta +5 (Warmer) -> Usage DOWN
        'wind': 15.0,  # Delta +15 (Stormy) -> Usage UP
        'wind_bucket': 'extreme_wind', # Explicit bucket for 'stormy' label
        'kwh': 60.0    # Delta +10
    }

    analysis = analyzer.analyze_day(current, baseline)
    text = formatter.format_day_comparison(analysis)
    text_lower = text.lower()

    assert analysis['causality']['wind_explains'] is True
    assert analysis['causality']['temp_contradicts'] is True

    assert "stormy weather" in text_lower or "windy weather" in text_lower
    assert "offset by" in text_lower
    assert "warmer weather" in text_lower
    assert "+10.0 kwh" in text_lower

def test_multiple_contradictions(analyzer, formatter):
    """Test Case 3: Multiple Contradictions.

    Example: Temp (Driver UP), but Wind (Contra DOWN) + Solar (Contra DOWN).
    """
    # Baseline: 0°C, 5 m/s, 0 kWh Solar, 50 kWh Total
    baseline = {
        'temp': 0.0,
        'wind': 5.0,
        'wind_bucket': 'normal',
        'solar_kwh': 0.0,
        'kwh': 50.0
    }
    # Current: -5°C (Colder->UP), 0 m/s (Calmer->DOWN), 5 kWh Solar (Sunny->DOWN), 55 kWh Total
    current = {
        'temp': -5.0,  # Delta -5 (Cold) -> UP
        'wind': 0.0,   # Delta -5 (Calm) -> DOWN
        'wind_bucket': 'normal',
        'solar_kwh': 5.0, # Delta +5 (Sunny) -> DOWN
        'kwh': 55.0    # Delta +5
    }

    analysis = analyzer.analyze_day(current, baseline)
    text = formatter.format_day_comparison(analysis)
    text_lower = text.lower()

    assert analysis['causality']['temp_explains'] is True
    assert analysis['causality']['wind_contradicts'] is True
    assert analysis['causality']['solar_contradicts'] is True

    # "Extreme cold, offset by very calm weather + sunny weather (+5.0 kWh...)"
    assert "offset by" in text_lower
    assert " + " in text # Check for join syntax
    assert "sunny weather" in text_lower

def test_contradiction_under_threshold(analyzer, formatter):
    """Test Case 4: Contradiction under threshold (skal IKKE vises)."""

    # Baseline: 0°C, 5 m/s, 50 kWh
    baseline = {
        'temp': 0.0,
        'wind': 5.0,
        'wind_bucket': 'normal',
        'kwh': 50.0
    }
    # Current: -5°C (Colder->UP), 4.0 m/s (Slightly Calmer->DOWN), 60 kWh Total
    # Wind Delta is -1.0.
    # Threshold for RELEVANCE is 1.0 (so causality check passes)
    # Threshold for DISPLAY CONTRADICTION is 2.5 (so it should be hidden)
    current = {
        'temp': -5.0,
        'wind': 4.0,  # Delta -1.0
        'wind_bucket': 'normal',
        'kwh': 60.0
    }

    analysis = analyzer.analyze_day(current, baseline)
    text = formatter.format_day_comparison(analysis)
    text_lower = text.lower()

    # Check Logic
    assert analysis['causality']['temp_explains'] is True
    # Wind delta is 1.0, so it IS flagged as a contradiction internally
    assert analysis['causality']['wind_contradicts'] is True

    # But it should NOT appear in text because |delta| (1.0) < CONTRADICTION_WIND_DELTA (2.5)
    assert "offset by" not in text_lower
    assert "calm" not in text_lower
    assert "wind" not in text_lower

    # Text should just be about the driver
    assert "extreme cold" in text_lower or "colder weather" in text_lower

def test_solar_contradiction_display(analyzer, formatter):
    """Verify Solar contradiction display threshold."""
    # Baseline: Cloudy (0 solar)
    baseline = {'temp': 0, 'solar_kwh': 0.0, 'kwh': 50.0}

    # Current: Sunny (3 kWh solar -> DOWN), but Cold (-5 -> UP). Net UP.
    current = {
        'temp': -5.0,
        'solar_kwh': 3.0, # Delta +3.0 > 1.5 (Threshold)
        'kwh': 60.0
    }

    analysis = analyzer.analyze_day(current, baseline)
    text = formatter.format_day_comparison(analysis)
    text_lower = text.lower()

    assert "offset by" in text_lower
    assert "sunny weather" in text_lower

    # Now verify UNDER threshold
    current_small_solar = {
        'temp': -5.0,
        'solar_kwh': 1.0, # Delta +1.0 < 1.5
        'kwh': 60.0
    }
    analysis = analyzer.analyze_day(current_small_solar, baseline)
    text_small = formatter.format_day_comparison(analysis)
    text_small_lower = text_small.lower()

    assert "offset by" not in text_small_lower
    assert "sunny weather" not in text_small_lower
