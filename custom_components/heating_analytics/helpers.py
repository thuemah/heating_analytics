"""Helper functions for Heating Analytics."""
from __future__ import annotations
import logging
from datetime import date
from homeassistant.const import UnitOfSpeed

_LOGGER = logging.getLogger(__name__)

def convert_speed_to_ms(value: float, unit: str | None) -> float:
    """Convert speed to m/s."""
    if not unit:
        return value

    # Normalize unit string (though HA constants might be mixed case, usually lowercase or symbol)
    # We check against constants first, then string variants

    # Already in m/s - no conversion needed
    if unit in (UnitOfSpeed.METERS_PER_SECOND, "m/s", "ms"):
        return value

    # km/h
    if unit in (UnitOfSpeed.KILOMETERS_PER_HOUR, "km/h", "kmh", "km/t", "kph"):
        return value / 3.6

    # mph
    if unit in (UnitOfSpeed.MILES_PER_HOUR, "mph"):
        return value * 0.44704

    # knots
    if unit in (UnitOfSpeed.KNOTS, "kn", "kt", "knots"):
        return value * 0.514444

    # Unknown unit - log warning and return value as-is (assuming m/s)
    _LOGGER.warning(f"Unknown speed unit: {unit}, assuming value is in m/s")
    return value

def get_last_year_iso_date(date_obj: date) -> date:
    """Get the corresponding date in the previous year based on ISO week and weekday.

    Handles the edge case where the current year has 53 weeks but the previous year only has 52.
    In that case, it falls back to Week 52.
    """
    year, week, weekday = date_obj.isocalendar()
    try:
        return date.fromisocalendar(year - 1, week, weekday)
    except ValueError:
        # Fallback to Week 52 if Week 53 doesn't exist in previous year
        return date.fromisocalendar(year - 1, 52, weekday)
