"""Helper functions for Heating Analytics."""
from __future__ import annotations
import logging
import math
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

def calculate_asymmetric_inertia(window: list[float]) -> tuple[float, str]:
    """Calculate effective temperature using asymmetric thermal inertia.

    Uses a slow profile (4h) when temperature is falling (heat shedding),
    a fast profile (2h) when temperature is rising (heat gaining),
    and a stable 3h profile otherwise.

    window: temperatures in chronological order (oldest first), same convention
            as the Gaussian kernel windows used in calibration.

    Returns a tuple of (effective_temperature, regime) where regime is one of
    'shedding', 'gaining', or 'stable'.
    """
    if not window:
        return 0.0, "stable"
    if len(window) == 1:
        return window[-1], "stable"

    current_temp = window[-1]
    trend_index = max(0, len(window) - 1 - 4)
    past_temp = window[trend_index]

    if current_temp < (past_temp - 0.5):
        weights = [0.20, 0.30, 0.30, 0.20]
        regime = "shedding"
    elif current_temp > (past_temp + 0.5):
        weights = [0.50, 0.50]
        regime = "gaining"
    else:
        weights = [0.34, 0.33, 0.33]
        regime = "stable"

    usable_window = window[-len(weights):]
    usable_weights = weights[-len(usable_window):]
    weight_sum = sum(usable_weights)
    eff_temp = sum(t * w for t, w in zip(usable_window, usable_weights)) / weight_sum
    return round(eff_temp, 2), regime


def generate_exponential_kernel(tau: float, window_hours: int = 168) -> tuple[float, ...]:
    """Generate a causal exponential decay kernel with time constant tau.

    Physically motivated by first-order thermal dynamics (RC-circuit analogy).
    Weights decay as e^(-t/tau) going back in time, giving a long tail with
    low but non-zero influence from days-old temperatures.

    tau: time constant in hours (higher = longer thermal memory)
    window_hours: how far back to look (default 7 days / 168 hours)
    Returns weights in oldest-to-newest order (same convention as Gaussian kernel).
    """
    # t=0 is most recent hour, t=window_hours-1 is oldest
    weights = [math.exp(-t / tau) for t in range(window_hours)]
    total = sum(weights)
    # Reverse to oldest-to-newest order
    return tuple(w / total for w in reversed(weights))


def generate_gaussian_kernel(hours: int) -> tuple[float, ...]:
    """Generate a Gaussian/Bell-curve kernel for the given number of hours."""
    if hours == 1:
        return (1.0,)
    if hours == 2:
        return (0.5, 0.5)

    weights = []
    center = (hours - 1) / 2.0
    sigma = hours / 4.0

    for i in range(hours):
        x = i - center
        weights.append(math.exp(-(x**2) / (2 * sigma**2)))

    total = sum(weights)
    return tuple(w / total for w in weights)
