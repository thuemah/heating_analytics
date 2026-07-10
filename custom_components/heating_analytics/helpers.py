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


def solve_gauss_jordan(
    A: list[list[float]],
    b: list[float],
    *,
    ridge: float = 0.0,
    pivot_eps: float = 1e-12,
) -> list[float] | None:
    """Solve ``A · x = b`` via Gauss-Jordan elimination with partial pivoting.

    Dimension-agnostic — handles any square ``N×N`` system.  Returns the
    solution as a list of length ``N``, or ``None`` if the matrix is
    singular (any pivot magnitude < ``pivot_eps``).

    ``ridge`` adds a Tikhonov term to the diagonal before elimination
    (``A[i][i] += ridge``).  Used by the Tobit Newton step to guard
    rank-deficient Hessians at active-set boundaries.

    Pure Python — no numpy.  Inputs are not mutated; a working copy is
    built internally.
    """
    n = len(A)
    if n == 0 or len(b) != n:
        return None
    # Augmented [A | b] working matrix.
    M = [list(row) + [b[i]] for i, row in enumerate(A)]
    if ridge:
        for i in range(n):
            M[i][i] += ridge

    for col in range(n):
        pivot_row = col
        pivot_val = abs(M[col][col])
        for r in range(col + 1, n):
            if abs(M[r][col]) > pivot_val:
                pivot_val = abs(M[r][col])
                pivot_row = r
        if pivot_val < pivot_eps:
            return None
        if pivot_row != col:
            M[col], M[pivot_row] = M[pivot_row], M[col]

        pivot = M[col][col]
        for j in range(col, n + 1):
            M[col][j] /= pivot

        for r in range(n):
            if r == col:
                continue
            factor = M[r][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                M[r][j] -= factor * M[col][j]

    return [M[i][n] for i in range(n)]


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


def compute_base_ema_step(
    current_bucket: float,
    target: float,
    learning_rate: float,
    snr_weight: float,
) -> tuple[float, float]:
    """Pure-math kernel for the base-model EMA step (#967).

    The single arithmetic source of truth for the formula

        step       = learning_rate × snr_weight × (target − current_bucket)
        new_bucket = current_bucket + step

    Centralised here so diagnostic simulations and live learning provably
    use the same arithmetic — see #967 for the silent-drift hazard that
    motivated the extraction (if the live formula evolves without the
    diagnostic being patched in lockstep, ``base_model_4d_shadow`` and
    the promotion-gate metrics it feeds would silently characterise a
    model that no longer matches production).

    Caller owns:

    - Target construction (e.g. ``max(0, actual + delta)`` for the lift
      path, ``total_energy_kwh`` for the legacy path).
    - ``snr_weight`` computation via :func:`learning.compute_snr_weight`.
    - Post-step rounding or clamping (live learning rounds to 5 decimals
      before storing; diagnostic simulations leave the float unrounded).
    - Buffer-jumpstart vs EMA branching for cold-start (only applies to
      the live writer; diagnostic seeds from the current bucket).

    Returns ``(new_bucket_value, applied_step_size)``.  The step is
    returned separately so step-RMS jitter diagnostics can read it
    without re-deriving from the bucket delta.

    Consumers: the live writer (``learning.process_learning``), the
    retrain path (``learning.learn_from_historical_import``), and the
    diagnostic simulation (``diagnostics._compute_base_model_4d_shadow_report``).
    Call-form convention: callers that have already folded the SNR
    weight into an effective rate pass ``(effective_rate, 1.0)`` —
    multiplication by 1.0 is exact in IEEE-754, so the result is
    bit-identical to ``bucket + effective_rate × (target − bucket)``
    no matter how the effective rate was constructed.  Callers holding
    the factors separately (diagnostics) pass them as-is; Python's
    left-to-right evaluation makes ``lr × w × diff`` identical to
    ``(lr × w) × diff``.
    """
    step = learning_rate * snr_weight * (target - current_bucket)
    return current_bucket + step, step
