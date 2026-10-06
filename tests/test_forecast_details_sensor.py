"""Tests for HeatingForecastDetailsSensor's state string.

The sensor spent months permanently stuck on "Gathering accuracy data" because
the producer (ForecastManager), the projection (coordinator) and the consumer
(sensor) disagreed about where two fields lived, and nothing pinned the output.

The first test here is the one that matters: it drives real
``calculate_per_source_uncertainty_stats`` output through the real coordinator
assembly into the real sensor, so a future re-drift at either boundary fails
rather than silently reverting the sensor to its warm-up string.
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from custom_components.heating_analytics.const import (
    ATTR_FORECAST_DETAILS,
    CONFIDENCE_MIN_SAMPLES,
    CONF_SECONDARY_WEATHER_ENTITY,
    KEY_FORECAST_ACCURACY_INTERNAL,
)
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.forecast import ForecastManager
from custom_components.heating_analytics.sensor import HeatingForecastDetailsSensor

PRIMARY = "weather.primary"
SECONDARY = "weather.secondary"

# conftest pins dt_util.now() to this instant.
NOW = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _history(days: int, primary_abs: float, secondary_abs: float) -> list[dict]:
    """Build ``days`` daily forecast-history entries with a per-source breakdown.

    One entry per day, which is what makes the upstream ``samples`` count a day
    count rather than an hour count.
    """
    entries = []
    for i in range(days, 0, -1):
        date_key = (NOW.date() - timedelta(days=i)).isoformat()
        entries.append({
            "date": date_key,
            "forecast_kwh": 20.0,
            "actual_kwh": 20.0,
            "error_kwh": 0.0,
            "abs_error_kwh": 0.0,
            "primary_entity": PRIMARY,
            "secondary_entity": SECONDARY,
            "source_breakdown": {
                "primary": {
                    "hours": 24, "forecast": 20.0, "actual": 20.0,
                    "error": primary_abs, "abs_error": primary_abs,
                },
                "secondary": {
                    "hours": 24, "forecast": 20.0, "actual": 20.0,
                    "error": secondary_abs, "abs_error": secondary_abs,
                },
            },
        })
    return entries


def _wire(mock_coordinator, mock_entry, history, secondary_entity=SECONDARY):
    """Run producer -> coordinator projection, and return the live sensor."""
    mock_entry.data = {CONF_SECONDARY_WEATHER_ENTITY: secondary_entity}
    mock_coordinator.entry = mock_entry
    mock_coordinator.weather_entity = PRIMARY
    mock_coordinator.data = {}

    forecast = ForecastManager(mock_coordinator)
    forecast._forecast_history = history
    mock_coordinator.forecast = forecast

    # mock_coordinator is a spec'd MagicMock, so call the real method unbound.
    HeatingDataCoordinator._update_forecast_details(mock_coordinator)

    return HeatingForecastDetailsSensor(mock_coordinator, mock_entry)


@pytest.mark.asyncio
async def test_sensor_reaches_comparison_verdict_end_to_end(mock_coordinator, mock_entry):
    """With enough history the sensor must leave the warm-up string.

    This is the regression guard. Both historical bugs (the coordinator
    projecting away the half the sensor read, and the sensor reading `samples`
    one level too high) independently pinned this assertion to
    "Gathering accuracy data".
    """
    days = CONFIDENCE_MIN_SAMPLES + 3
    sensor = _wire(
        mock_coordinator, mock_entry,
        _history(days, primary_abs=1.0, secondary_abs=3.0),
    )

    value = sensor.native_value

    assert value != "Gathering accuracy data"
    assert "Primary is performing better" in value
    # The numbers must be the measured errors, not the 999 fallback sentinel.
    assert "999" not in value
    assert "1.0 vs 3.0" in value


@pytest.mark.asyncio
async def test_internal_key_carries_day_count_and_error(mock_coordinator, mock_entry):
    """The projection must publish both fields the sensor needs."""
    days = CONFIDENCE_MIN_SAMPLES + 1
    _wire(mock_coordinator, mock_entry, _history(days, 1.0, 2.0))

    internal = mock_coordinator.data[KEY_FORECAST_ACCURACY_INTERNAL]

    for source in ("primary", "secondary"):
        assert internal[source]["samples"] == days
        assert internal[source]["p50_abs_error"] is not None


@pytest.mark.asyncio
async def test_user_attribute_stays_daily_only(mock_coordinator, mock_entry):
    """The hourly-derived numbers must not leak back into the attribute.

    Dropping them was a deliberate UX decision; the fix for the stuck sensor
    routes around it rather than reversing it.
    """
    _wire(mock_coordinator, mock_entry, _history(CONFIDENCE_MIN_SAMPLES + 1, 1.0, 2.0))

    accuracy = mock_coordinator.data[ATTR_FORECAST_DETAILS]["accuracy_by_source"]

    for source in ("primary", "secondary"):
        assert set(accuracy[source]) == {"daily"}
        assert "p50_abs_error" not in accuracy[source]
        assert "samples" not in accuracy[source]


@pytest.mark.asyncio
async def test_secondary_better_verdict(mock_coordinator, mock_entry):
    sensor = _wire(
        mock_coordinator, mock_entry,
        _history(CONFIDENCE_MIN_SAMPLES + 1, primary_abs=4.0, secondary_abs=1.0),
    )

    assert "Secondary is performing better" in sensor.native_value


@pytest.mark.asyncio
async def test_similar_accuracy_verdict(mock_coordinator, mock_entry):
    """Neither source clears FORECAST_COMPARISON_FACTOR against the other."""
    sensor = _wire(
        mock_coordinator, mock_entry,
        _history(CONFIDENCE_MIN_SAMPLES + 1, primary_abs=2.0, secondary_abs=2.05),
    )

    assert "Both sources have similar accuracy" in sensor.native_value


@pytest.mark.asyncio
async def test_gathering_below_threshold(mock_coordinator, mock_entry):
    """Below CONFIDENCE_MIN_SAMPLES days the warm-up string is still correct."""
    sensor = _wire(
        mock_coordinator, mock_entry,
        _history(CONFIDENCE_MIN_SAMPLES - 1, 1.0, 3.0),
    )

    assert sensor.native_value == "Gathering accuracy data"


@pytest.mark.asyncio
async def test_primary_source_only_without_secondary_entity(mock_coordinator, mock_entry):
    """The single-source path is unaffected by the accuracy plumbing."""
    sensor = _wire(
        mock_coordinator, mock_entry,
        _history(CONFIDENCE_MIN_SAMPLES + 1, 1.0, 3.0),
        secondary_entity=None,
    )

    assert sensor.native_value == "Primary source only"


@pytest.mark.asyncio
async def test_single_source_active_messages(mock_coordinator, mock_entry):
    """One source with enough days, the other without, reports a day count."""
    history = _history(CONFIDENCE_MIN_SAMPLES + 2, 1.0, 3.0)
    # Strip the secondary breakdown so only primary accumulates samples.
    for entry in history:
        del entry["source_breakdown"]["secondary"]

    sensor = _wire(mock_coordinator, mock_entry, history)

    value = sensor.native_value
    assert value == f"Primary source is active ({CONFIDENCE_MIN_SAMPLES + 2} days logged)"


@pytest.mark.asyncio
async def test_missing_internal_key_does_not_crash(mock_coordinator, mock_entry):
    """A payload without the internal key falls back, it does not format None."""
    mock_coordinator.data = {
        ATTR_FORECAST_DETAILS: {
            "blend_config": {"secondary_entity_id": SECONDARY},
            "accuracy_by_source": {"primary": {"daily": {}}, "secondary": {"daily": {}}},
        }
    }

    sensor = HeatingForecastDetailsSensor(mock_coordinator, mock_entry)

    # No internal key at all -> no secondary stats -> single-source string.
    assert sensor.native_value == "Primary source only"


@pytest.mark.asyncio
async def test_missing_p50_does_not_format_none(mock_coordinator, mock_entry):
    """Sufficient samples but absent errors must not raise on the f-string."""
    mock_coordinator.data = {
        ATTR_FORECAST_DETAILS: {"blend_config": {"secondary_entity_id": SECONDARY}},
        KEY_FORECAST_ACCURACY_INTERNAL: {
            "primary": {"samples": 30, "p50_abs_error": None},
            "secondary": {"samples": 30, "p50_abs_error": None},
        },
    }

    sensor = HeatingForecastDetailsSensor(mock_coordinator, mock_entry)

    assert sensor.native_value == "Gathering accuracy data"
