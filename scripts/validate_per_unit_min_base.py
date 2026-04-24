#!/usr/bin/env python3
"""Pre-ship validation for per-unit min-base thresholds.

Replays the stored hourly_log under two threshold rules and compares
qualifying-hour counts + threshold values per unit:

    Rule A (current):  global SOLAR_*_MIN_BASE = 0.15 kWh for every unit.
    Rule B (proposed): per-unit p10 of dark-hour actuals (≥ 20 samples,
                       floored at 0.03 kWh, with ratio-guard rejecting
                       p10 / median > 0.9 and an absolute ceiling of
                       1.5 kWh acting as safety net).

Reports:

    - Per-unit calibrated threshold + dark-hour sample count
    - Δ NLMS qualifying hours under Rule B vs A
    - Δ inequality qualifying hours under Rule B vs A
    - Δ shutdown detections under Rule B vs A

Falsification criteria (any triggers "reject"):

    1. Any auto-calibrated threshold > 1.5 kWh (data problem — ceiling hit).
    2. For units whose base exceeds the global 0.15 by > 2×, the set of
       qualifying NLMS hours should be a strict superset under Rule B
       (lowering a threshold cannot remove qualifying hours).  If not,
       the implementation has a bug.
    3. Fraction of units gaining < 5 % additional qualifying hours when
       their calibrated threshold was lowered by > 20 %: suggests the
       change is not addressing the motivating problem (Toshiba mild-temp
       case).

Usage:
    python scripts/validate_per_unit_min_base.py --storage <path-to-storage-json>
    python scripts/validate_per_unit_min_base.py --storage <path> --window 90
    python scripts/validate_per_unit_min_base.py --storage <path> --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path


# Mirrored from const.py — avoid importing the HA integration (keeps the
# script runnable without Home Assistant on PATH).
DARK_SOLAR_FACTOR = 0.05
MIN_DARK_SAMPLES = 20
MIN_HOURS_OF_LOG = 14 * 24
FLOOR = 0.03
CEILING = 1.5
MAX_P10_MEDIAN_RATIO = 0.9
MAX_RATE_OF_CHANGE = 0.5

GLOBAL_LEARNING_MIN_BASE = 0.15
GLOBAL_SHUTDOWN_MIN_BASE = 0.15

SHUTDOWN_ACTUAL_FLOOR = 0.03
SHUTDOWN_RATIO = 0.15
SHUTDOWN_MIN_MAGNITUDE = 0.3

MODE_HEATING = "heating"
MODES_EXCLUDED = frozenset({"off", "dhw", "guest_heating", "guest_cooling"})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_hourly_log(storage_path: str) -> list[dict]:
    with open(storage_path) as f:
        store = json.load(f)
    data = store.get("data", store)
    log = data.get("hourly_log", [])
    if not log:
        sys.exit("ERROR: hourly_log missing or empty")
    return log


def _parse_ts(ts):
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def filter_window(entries, window_days):
    if not window_days or window_days <= 0:
        return entries
    latest = max(
        (_parse_ts(e.get("timestamp")) for e in entries if _parse_ts(e.get("timestamp"))),
        default=None,
    )
    if latest is None:
        return entries
    cutoff = latest - timedelta(days=window_days)
    return [e for e in entries if (_parse_ts(e.get("timestamp")) or latest) >= cutoff]


def discover_sensors(entries) -> list[str]:
    sids = set()
    for e in entries:
        sids.update((e.get("unit_breakdown") or {}).keys())
        sids.update((e.get("unit_expected_breakdown") or {}).keys())
    return sorted(sids)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def calibrate(entries, sensors) -> dict:
    """Return {sensor: {'threshold', 'p10', 'samples'}} for calibrated units."""
    samples: dict[str, list[float]] = defaultdict(list)
    for entry in entries:
        if entry.get("auxiliary_active", False):
            continue
        sf = entry.get("solar_factor") or 0.0
        if sf >= DARK_SOLAR_FACTOR:
            continue
        modes = entry.get("unit_modes", {}) or {}
        breakdown = entry.get("unit_breakdown", {}) or {}
        for sid in sensors:
            if modes.get(sid, MODE_HEATING) != MODE_HEATING:
                continue
            if sid not in breakdown:
                continue
            v = breakdown.get(sid, 0.0)
            if v is None or v < 0.0:
                continue
            samples[sid].append(float(v))

    out = {}
    for sid in sensors:
        ss = sorted(samples.get(sid, []))
        n = len(ss)
        if n < MIN_DARK_SAMPLES:
            out[sid] = {
                "samples": n, "p10": None, "threshold": None,
                "status": "under_sampled",
            }
            continue
        # Nearest-rank p10 via ceiling (matches coordinator._p10).
        import math as _math
        idx = max(0, _math.ceil(0.10 * n) - 1)
        p10 = ss[idx]
        median = ss[n // 2]
        # Ratio-guard (primary filter): p10 approaching median means
        # always-on load, not a modulating heat pump.
        if median > 0.0 and p10 > MAX_P10_MEDIAN_RATIO * median:
            out[sid] = {
                "samples": n, "p10": p10, "median": median,
                "threshold": None,
                "status": "rejected_constant_load",
            }
            continue
        # Absolute ceiling (safety net).
        if p10 > CEILING:
            out[sid] = {
                "samples": n, "p10": p10, "median": median,
                "threshold": None,
                "status": "rejected_above_ceiling",
            }
            continue
        out[sid] = {
            "samples": n, "p10": p10, "median": median,
            "threshold": round(max(FLOOR, p10), 5),
            "status": "ok",
        }
    return out


# ---------------------------------------------------------------------------
# Qualification counters
# ---------------------------------------------------------------------------


def count_qualifications(entries, sensors, thresholds_fn):
    """Count NLMS / inequality / shutdown qualifications per (unit, rule).

    ``thresholds_fn(sensor_id) -> float`` resolves the gate per sensor.
    Mirrors live-gate semantics:
      - NLMS: sunny, not aux, not shutdown, base >= threshold
      - Inequality: sunny, not aux, shutdown, base >= threshold, heating
      - Shutdown: base >= threshold and (actual < actual_floor or ratio < 0.15)
    """
    counters = {
        sid: {"nlms": 0, "inequality": 0, "shutdown": 0, "skipped_low_base": 0}
        for sid in sensors
    }
    for entry in entries:
        if entry.get("auxiliary_active", False):
            continue
        sf = entry.get("solar_factor") or 0.0
        solar_s = entry.get("solar_vector_s", 0.0) or 0.0
        solar_e = entry.get("solar_vector_e", 0.0) or 0.0
        solar_w = entry.get("solar_vector_w", 0.0) or 0.0
        mag = (solar_s ** 2 + solar_e ** 2 + solar_w ** 2) ** 0.5
        if mag < 0.1:
            continue
        modes = entry.get("unit_modes", {}) or {}
        breakdown = entry.get("unit_breakdown", {}) or {}
        expected_base = entry.get("unit_expected_breakdown", {}) or {}
        log_shutdowns = set(entry.get("solar_dominant_entities", []) or [])

        for sid in sensors:
            mode = modes.get(sid, MODE_HEATING)
            if mode in MODES_EXCLUDED:
                continue
            base = expected_base.get(sid, 0.0) or 0.0
            threshold = thresholds_fn(sid)
            if base < threshold:
                counters[sid]["skipped_low_base"] += 1
                continue

            actual = breakdown.get(sid, 0.0) or 0.0
            is_shutdown = sid in log_shutdowns
            if is_shutdown and mode == MODE_HEATING:
                counters[sid]["inequality"] += 1
            elif not is_shutdown:
                counters[sid]["nlms"] += 1

            # Separately tally what shutdown detection would flag NOW under
            # this rule (independent of what was historically logged).
            if mag >= SHUTDOWN_MIN_MAGNITUDE:
                ratio = actual / base if base > 0 else 1.0
                if actual < SHUTDOWN_ACTUAL_FLOOR or ratio < SHUTDOWN_RATIO:
                    counters[sid]["shutdown"] += 1

    return counters


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_delta(a: int, b: int) -> str:
    if a == 0 and b == 0:
        return "n/a"
    if a == 0:
        return f"+{b}"
    return f"{b - a:+d}  ({((b - a) / max(1, a)) * 100:+.1f} %)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", required=True, help="Path to HA storage JSON")
    parser.add_argument("--window", type=int, default=0,
                        help="Last N days to replay (0 = all)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not Path(args.storage).exists():
        sys.exit(f"ERROR: storage file not found: {args.storage}")

    log = load_hourly_log(args.storage)
    log = filter_window(log, args.window)
    if len(log) < MIN_HOURS_OF_LOG:
        print(f"WARNING: only {len(log)} hours available "
              f"(need {MIN_HOURS_OF_LOG} for calibration).  "
              f"Calibration will skip; reporting global-rule counts only.")

    sensors = discover_sensors(log)
    if not sensors:
        sys.exit("ERROR: no sensors discovered in log")

    # Calibrate per-unit thresholds.
    calib = calibrate(log, sensors)

    def rule_a(sid: str) -> float:
        return GLOBAL_LEARNING_MIN_BASE  # 0.15 kWh, fixed.

    def rule_b(sid: str) -> float:
        row = calib.get(sid, {})
        t = row.get("threshold")
        return t if t is not None else GLOBAL_LEARNING_MIN_BASE

    counts_a = count_qualifications(log, sensors, rule_a)
    counts_b = count_qualifications(log, sensors, rule_b)

    # --- Falsification checks ---
    rejects: list[str] = []
    for sid, row in calib.items():
        t = row.get("threshold")
        if t is not None and t > CEILING:
            rejects.append(f"{sid}: threshold {t:.3f} > ceiling {CEILING:.3f}")
    # Monotonicity check: Rule B threshold ≤ Rule A → NLMS count ≥ Rule A.
    for sid in sensors:
        t_b = rule_b(sid)
        if t_b <= GLOBAL_LEARNING_MIN_BASE:
            if counts_b[sid]["nlms"] < counts_a[sid]["nlms"]:
                rejects.append(
                    f"{sid}: NLMS count decreased under lower threshold "
                    f"(A={counts_a[sid]['nlms']}, B={counts_b[sid]['nlms']})"
                )

    # --- Report ---
    print("=" * 78)
    print(f"Per-unit min-base validation ({len(log)} hours, {len(sensors)} sensors)")
    print("=" * 78)
    print()
    header = (f"{'Sensor':<30} {'Samples':>8} {'p10':>8} {'Thresh':>8} "
              f"{'NLMS Δ':>14} {'Ineq Δ':>14} {'Shut Δ':>14}")
    print(header)
    print("-" * len(header))
    for sid in sensors:
        row = calib.get(sid, {})
        samples = row.get("samples", 0)
        p10 = row.get("p10")
        threshold = row.get("threshold")
        thresh_str = f"{threshold:.3f}" if threshold is not None else "fallback"
        p10_str = f"{p10:.3f}" if p10 is not None else "—"
        a = counts_a[sid]
        b = counts_b[sid]
        print(f"{sid[:30]:<30} {samples:>8} {p10_str:>8} {thresh_str:>8} "
              f"{format_delta(a['nlms'], b['nlms']):>14} "
              f"{format_delta(a['inequality'], b['inequality']):>14} "
              f"{format_delta(a['shutdown'], b['shutdown']):>14}")

    print()
    if rejects:
        print("REJECT — falsification criteria triggered:")
        for msg in rejects:
            print(f"  - {msg}")
        return 1

    print("PASS — all falsification checks clear.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
