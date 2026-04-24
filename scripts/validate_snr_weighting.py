#!/usr/bin/env python3
"""
Option 2 validation: SNR-weighted vs. delta-normalized base-model learning.

Replays the stored hourly_log under two learning rules and compares per-bucket
convergence against dark-hour empirical ground truth.

Rule A (current): target = learning_energy + solar_normalization_delta
                  bucket += lr * (target - bucket)

Rule B (Option 2): target = learning_energy
                   bucket += lr * snr_weight(solar_factor, shutdown) * (target - bucket)

Ground truth: per-(temp_key, wind_bucket) empirical mean over hours with
solar_factor < 0.05, non-aux, non-excluded-mode. Requires >= 10 samples
per bucket to be eligible for comparison (parity with base_model_health).

Two convergence modes:
  live   — single chronological pass, EMA starting at 0 (simulates what
           would have actually happened if the rule were live)
  conv   — analytical steady state: (weighted) mean of targets. For EMA
           with fixed learning rate over many passes, this is the limit.

Usage:
    python scripts/validate_snr_weighting.py --storage <path-to-storage-json>
    python scripts/validate_snr_weighting.py --storage <path> --window 90 --mode both
    python scripts/validate_snr_weighting.py --storage <path> --sweep --out results/sweep

Falsification criteria (any triggers "reject"):
  - aggregate RMSE degrades > 5% vs current
  - > 30% of comparable buckets degrade > 10%
  - any cold bucket (temp < 5°C) degrades > 10%
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path


# Constants mirrored from const.py (avoid importing custom_components)
DARK_SOLAR_FACTOR_THRESHOLD = 0.05
MIN_DARK_HOURS_FOR_VERDICT = 10
DEFAULT_LEARNING_RATE = 0.01
MODES_EXCLUDED = frozenset({"off", "dhw", "guest_heating", "guest_cooling"})
MODE_HEATING_DEFAULT = "heating"

# Falsification thresholds
AGGREGATE_RMSE_DEGRADE_PCT = 5.0
BUCKET_DEGRADE_PCT = 10.0
BUCKET_DEGRADE_ABS_KWH = 0.05  # Absolute floor — avoids astronomical rel% on near-zero current error
BUCKET_DEGRADE_FRACTION_MAX = 0.30
COLD_BUCKET_TEMP_THRESHOLD = 5.0

# Default weight-function parameters (can be overridden via CLI)
DEFAULT_FLOOR = 0.1
DEFAULT_K = 3.0

# Sweep grid
SWEEP_FLOOR = [0.0, 0.05, 0.1, 0.2]
SWEEP_K = [1.5, 2.0, 3.0, 5.0]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_hourly_log(storage_path):
    """Load hourly_log from HA storage JSON file."""
    with open(storage_path) as f:
        store = json.load(f)
    # HA Store wraps payload as {"version": N, "key": "...", "data": {...}}
    data = store.get("data", store)
    log = data.get("hourly_log", [])
    if not log:
        sys.exit("ERROR: hourly_log is empty or missing in storage file")
    return log


def filter_window(entries, window_days):
    """Keep entries within the last N days of the latest timestamp."""
    if not window_days or window_days <= 0:
        return entries
    latest = None
    for e in entries:
        ts = _parse_ts(e.get("timestamp"))
        if ts and (latest is None or ts > latest):
            latest = ts
    if latest is None:
        return entries
    cutoff = latest - timedelta(days=window_days)
    return [e for e in entries if (_parse_ts(e.get("timestamp")) or latest) >= cutoff]


def _parse_ts(ts):
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Entry interpretation (parity with live learning)
# ---------------------------------------------------------------------------

def learning_energy(entry):
    """Reconstruct learning_energy_kwh from unit_breakdown + unit_modes.

    Live learning excludes OFF/DHW/GUEST_HEATING/GUEST_COOLING units from the
    global base-learning target. We mirror that here. unit_modes stores only
    non-HEATING modes (log-size optimization); absence == HEATING by default.
    """
    breakdown = entry.get("unit_breakdown") or {}
    modes = entry.get("unit_modes") or {}
    if not breakdown:
        # Legacy fallback
        return float(entry.get("actual_kwh", 0.0))
    total = 0.0
    for entity, kwh in breakdown.items():
        mode = modes.get(entity, MODE_HEATING_DEFAULT)
        if mode not in MODES_EXCLUDED:
            total += float(kwh or 0.0)
    return total


def is_learnable(entry):
    """Gate matching live-learning: skip aux-dominated hours."""
    if entry.get("auxiliary_active"):
        return False
    if entry.get("temp_key") is None or entry.get("wind_bucket") is None:
        return False
    return True


def is_dark(entry):
    return float(entry.get("solar_factor", 0.0)) < DARK_SOLAR_FACTOR_THRESHOLD


# ---------------------------------------------------------------------------
# SNR weight
# ---------------------------------------------------------------------------

def snr_weight(solar_factor, dominant_entities, total_units, *, floor, k):
    """Weight a learning sample by signal-to-noise ratio.

    w = max(floor, 1 - k*solar_factor) × (fraction of units not in shutdown)

    - Dark hour (solar_factor ≈ 0) → w ≈ 1.0
    - Sunny hour → w decays toward floor as k*solar_factor grows
    - Per-unit shutdown scaling: if some units are in shutdown, other units'
      contribution is still valid (avoids the agent-flagged over-aggressive
      global w=0).  If ALL units are in shutdown → w = 0.
    """
    sf = max(0.0, float(solar_factor or 0.0))
    w = max(floor, 1.0 - k * sf)
    if total_units > 0 and dominant_entities:
        n_shutdown = len(dominant_entities)
        n_clean = max(0, total_units - n_shutdown)
        if n_clean == 0:
            return 0.0
        w *= n_clean / total_units
    return w


# ---------------------------------------------------------------------------
# Replay (live-like EMA)
# ---------------------------------------------------------------------------

def replay_live(entries, lr, rule, *, floor=DEFAULT_FLOOR, k=DEFAULT_K):
    """Single chronological EMA pass from zero. Returns (bucket_values, counts).

    rule = "current"  → target = learning_energy + delta,      rate = lr
    rule = "option2"  → target = learning_energy,              rate = lr * w(hour)
    """
    ordered = sorted(
        (e for e in entries if is_learnable(e)),
        key=lambda e: _parse_ts(e.get("timestamp")) or datetime.min,
    )
    bucket = defaultdict(float)
    count = defaultdict(int)
    for e in ordered:
        key = (e["temp_key"], e["wind_bucket"])
        le = learning_energy(e)
        if rule == "current":
            target = le + float(e.get("solar_normalization_delta", 0.0) or 0.0)
            step = lr
        elif rule == "option2":
            target = le
            total_units = len(e.get("unit_breakdown") or {})
            dominant = e.get("solar_dominant_entities") or []
            w = snr_weight(
                e.get("solar_factor", 0.0), dominant, total_units, floor=floor, k=k
            )
            step = lr * w
        else:
            raise ValueError(f"unknown rule: {rule}")
        if step > 0:
            bucket[key] += step * (target - bucket[key])
        count[key] += 1
    return dict(bucket), dict(count)


# ---------------------------------------------------------------------------
# Analytical steady-state (converged)
# ---------------------------------------------------------------------------

def converged_buckets(entries, *, floor=DEFAULT_FLOOR, k=DEFAULT_K):
    """Analytical limit of EMA: (weighted) mean of targets per bucket.

    Returns (current_mean, option2_mean, sample_counts).
    """
    cur_sum = defaultdict(float)
    cur_cnt = defaultdict(float)
    opt_sum = defaultdict(float)
    opt_w = defaultdict(float)
    raw_cnt = defaultdict(int)
    for e in entries:
        if not is_learnable(e):
            continue
        key = (e["temp_key"], e["wind_bucket"])
        le = learning_energy(e)
        cur_sum[key] += le + float(e.get("solar_normalization_delta", 0.0) or 0.0)
        cur_cnt[key] += 1.0
        total_units = len(e.get("unit_breakdown") or {})
        dominant = e.get("solar_dominant_entities") or []
        w = snr_weight(
            e.get("solar_factor", 0.0), dominant, total_units, floor=floor, k=k
        )
        if w > 0:
            opt_sum[key] += w * le
            opt_w[key] += w
        raw_cnt[key] += 1
    cur_mean = {k_: cur_sum[k_] / cur_cnt[k_] for k_ in cur_sum if cur_cnt[k_] > 0}
    opt_mean = {k_: opt_sum[k_] / opt_w[k_] for k_ in opt_sum if opt_w[k_] > 0}
    return cur_mean, opt_mean, dict(raw_cnt)


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def compute_dark_mean(entries):
    """Per-bucket empirical mean over dark (solar_factor < 0.05), learnable hours."""
    samples = defaultdict(list)
    for e in entries:
        if not is_learnable(e) or not is_dark(e):
            continue
        key = (e["temp_key"], e["wind_bucket"])
        samples[key].append(learning_energy(e))
    dark_mean = {
        k_: sum(v) / len(v) for k_, v in samples.items() if len(v) >= MIN_DARK_HOURS_FOR_VERDICT
    }
    dark_count = {k_: len(v) for k_, v in samples.items()}
    return dark_mean, dark_count


# ---------------------------------------------------------------------------
# Comparison + falsification
# ---------------------------------------------------------------------------

def _temp_key_to_float(temp_key):
    try:
        s = str(temp_key)
        # Expect something like "T_5" or "-3"; take the last numeric segment
        last = s.split("_")[-1] if "_" in s else s
        return float(last)
    except (ValueError, AttributeError):
        return float("nan")


def compare(bucket_current, bucket_option2, dark_mean, dark_count, sample_count):
    """Build per-bucket comparison rows."""
    rows = []
    for key, truth in dark_mean.items():
        temp_key, wind_bucket = key
        c_val = bucket_current.get(key, 0.0)
        o_val = bucket_option2.get(key, 0.0)
        c_err = c_val - truth
        o_err = o_val - truth
        c_abs = abs(c_err)
        o_abs = abs(o_err)
        rows.append({
            "temp_key": temp_key,
            "wind_bucket": wind_bucket,
            "temp_c": _temp_key_to_float(temp_key),
            "n_total": sample_count.get(key, 0),
            "n_dark": dark_count.get(key, 0),
            "dark_mean": round(truth, 4),
            "current_value": round(c_val, 4),
            "option2_value": round(o_val, 4),
            "current_err": round(c_err, 4),
            "option2_err": round(o_err, 4),
            "current_abs_err": round(c_abs, 4),
            "option2_abs_err": round(o_abs, 4),
            "abs_err_change": round(o_abs - c_abs, 4),
            "rel_err_change_pct": round(100.0 * (o_abs - c_abs) / max(c_abs, 1e-6), 2),
            "winner": "option2" if o_abs < c_abs else ("tie" if o_abs == c_abs else "current"),
        })
    return rows


def aggregate(rows):
    """Population-weighted RMSE and falsification verdict."""
    if not rows:
        return {
            "error": "no comparable buckets",
            "n_buckets_compared": 0,
            "verdict": "insufficient_data",
        }

    def rmse(pairs):
        num = sum(w * (e * e) for e, w in pairs)
        den = sum(w for _, w in pairs)
        return math.sqrt(num / den) if den > 0 else 0.0

    w_pairs_c = [(r["current_err"], r["n_dark"]) for r in rows]
    w_pairs_o = [(r["option2_err"], r["n_dark"]) for r in rows]
    c_rmse = rmse(w_pairs_c)
    o_rmse = rmse(w_pairs_o)

    # A bucket counts as "degraded" only if both relative AND absolute thresholds are crossed.
    # This avoids false rejection when current_err is microscopic — a near-perfect bucket
    # moving from 1e-9 to 0.01 produces an enormous rel%, but the absolute change is trivial.
    degrading = [
        r for r in rows
        if r["rel_err_change_pct"] > BUCKET_DEGRADE_PCT
        and r["abs_err_change"] > BUCKET_DEGRADE_ABS_KWH
    ]
    cold_degraded = [
        r for r in degrading
        if not math.isnan(r["temp_c"]) and r["temp_c"] < COLD_BUCKET_TEMP_THRESHOLD
    ]

    aggregate_degraded = o_rmse > c_rmse * (1.0 + AGGREGATE_RMSE_DEGRADE_PCT / 100.0)
    too_many_degraded = len(degrading) > len(rows) * BUCKET_DEGRADE_FRACTION_MAX
    cold_degraded_any = bool(cold_degraded)

    if aggregate_degraded or too_many_degraded or cold_degraded_any:
        verdict = "reject"
    elif o_rmse < c_rmse and len(degrading) == 0:
        verdict = "accept"
    else:
        verdict = "mixed"

    return {
        "n_buckets_compared": len(rows),
        "n_buckets_option2_wins": sum(1 for r in rows if r["winner"] == "option2"),
        "n_buckets_degraded_10pct": len(degrading),
        "cold_buckets_degraded": len(cold_degraded),
        "aggregate_rmse_current": round(c_rmse, 4),
        "aggregate_rmse_option2": round(o_rmse, 4),
        "aggregate_rmse_change_pct": round(100.0 * (o_rmse - c_rmse) / max(c_rmse, 1e-6), 2),
        "falsification": {
            "aggregate_rmse_degrade_gt_5pct": aggregate_degraded,
            "degraded_buckets_gt_30pct": too_many_degraded,
            "cold_buckets_degraded": cold_degraded_any,
        },
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def emit_results(results, args):
    """Either print to stdout or write to --out prefix."""
    if args.out:
        out_prefix = Path(args.out)
        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        for mode_name, r in results.items():
            csv_path = out_prefix.with_name(out_prefix.name + f".{mode_name}.csv")
            if r["rows"]:
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(r["rows"][0].keys()))
                    writer.writeheader()
                    writer.writerows(r["rows"])
        json_path = out_prefix.with_name(out_prefix.name + ".aggregate.json")
        with open(json_path, "w") as f:
            json.dump(
                {m: r["aggregate"] for m, r in results.items()},
                f, indent=2,
            )
        print(f"# Wrote: {out_prefix}.<mode>.csv + {json_path}", file=sys.stderr)
    else:
        for mode_name, r in results.items():
            print(f"\n========= {mode_name.upper()} =========")
            print(json.dumps(r["aggregate"], indent=2))
            if r["rows"]:
                worst = sorted(r["rows"], key=lambda x: x["rel_err_change_pct"], reverse=True)[:10]
                print(f"\nTop-10 worst-degrading buckets ({mode_name}):")
                print(f"  {'temp':>6} {'wind':>5} {'n_dark':>7} {'dark':>8} {'cur':>8} {'opt2':>8} {'Δabs':>8} {'Δ%':>7} win")
                for row in worst:
                    print(
                        f"  {row['temp_key']!s:>6} {row['wind_bucket']!s:>5} "
                        f"{row['n_dark']:>7} {row['dark_mean']:>8.3f} "
                        f"{row['current_value']:>8.3f} {row['option2_value']:>8.3f} "
                        f"{row['abs_err_change']:>+8.3f} {row['rel_err_change_pct']:>+6.1f}% "
                        f"{row['winner']}"
                    )


# ---------------------------------------------------------------------------
# Sweep (grid over floor × k)
# ---------------------------------------------------------------------------

def run_sweep(entries, dark_mean, dark_count, args):
    """Grid search over (floor, k), analytical steady-state only (fast)."""
    cur_mean, _, sample_count = converged_buckets(entries, floor=DEFAULT_FLOOR, k=DEFAULT_K)
    # cur_mean is independent of (floor, k); compute once.
    sweep_rows = []
    header = (
        f"{'floor':>6} {'k':>5}  {'rmse_cur':>10} {'rmse_opt2':>10} "
        f"{'Δ%':>7}  {'wins':>5} {'degr':>5} {'cold':>5}  verdict"
    )
    print(header)
    print("-" * len(header))
    for floor in SWEEP_FLOOR:
        for k in SWEEP_K:
            _, opt_mean, _ = converged_buckets(entries, floor=floor, k=k)
            rows = compare(cur_mean, opt_mean, dark_mean, dark_count, sample_count)
            agg = aggregate(rows)
            print(
                f"{floor:>6.2f} {k:>5.1f}  "
                f"{agg['aggregate_rmse_current']:>10.4f} {agg['aggregate_rmse_option2']:>10.4f} "
                f"{agg['aggregate_rmse_change_pct']:>+6.2f}%  "
                f"{agg['n_buckets_option2_wins']:>5} "
                f"{agg['n_buckets_degraded_10pct']:>5} "
                f"{agg['cold_buckets_degraded']:>5}  "
                f"{agg['verdict']}"
            )
            sweep_rows.append({
                "floor": floor, "k": k,
                **{kk: vv for kk, vv in agg.items() if kk != "falsification"},
                **{f"fals_{kk}": vv for kk, vv in agg["falsification"].items()},
            })
    if args.out:
        out_prefix = Path(args.out)
        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        sweep_path = out_prefix.with_name(out_prefix.name + ".sweep.json")
        with open(sweep_path, "w") as f:
            json.dump(sweep_rows, f, indent=2)
        print(f"\n# Wrote: {sweep_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_single(entries, dark_mean, dark_count, args):
    """Run a single (floor, k) at the selected mode(s)."""
    results = {}
    _, _, sample_count = converged_buckets(entries, floor=args.floor, k=args.k)

    if args.mode in ("live", "both"):
        bc_live, _ = replay_live(entries, args.learning_rate, "current")
        bo_live, _ = replay_live(
            entries, args.learning_rate, "option2", floor=args.floor, k=args.k
        )
        rows = compare(bc_live, bo_live, dark_mean, dark_count, sample_count)
        results["live"] = {"rows": rows, "aggregate": aggregate(rows)}

    if args.mode in ("conv", "both"):
        cur_mean, opt_mean, _ = converged_buckets(entries, floor=args.floor, k=args.k)
        rows = compare(cur_mean, opt_mean, dark_mean, dark_count, sample_count)
        results["conv"] = {"rows": rows, "aggregate": aggregate(rows)}

    emit_results(results, args)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--storage", required=True,
                    help="Path to HA .storage/heating_analytics.storage_<entry_id> JSON")
    ap.add_argument("--window", type=int, default=90,
                    help="Days of history to replay (default 90). 0 = all.")
    ap.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE,
                    help=f"EMA rate for live mode (default {DEFAULT_LEARNING_RATE})")
    ap.add_argument("--floor", type=float, default=DEFAULT_FLOOR,
                    help=f"SNR weight floor (default {DEFAULT_FLOOR})")
    ap.add_argument("--k", type=float, default=DEFAULT_K,
                    help=f"SNR weight slope (default {DEFAULT_K})")
    ap.add_argument("--mode", choices=["live", "conv", "both"], default="both",
                    help="live = single-pass EMA, conv = analytical steady state")
    ap.add_argument("--sweep", action="store_true",
                    help="Grid search over (floor, k); overrides --mode")
    ap.add_argument("--out", default=None,
                    help="Output path prefix. Writes .<mode>.csv + .aggregate.json "
                         "(+ .sweep.json with --sweep). Default: print to stdout.")
    args = ap.parse_args()

    all_entries = load_hourly_log(args.storage)
    entries = filter_window(all_entries, args.window)
    print(
        f"# Loaded {len(all_entries)} entries; {len(entries)} in "
        f"{'all' if args.window == 0 else f'{args.window}-day'} window",
        file=sys.stderr,
    )

    dark_mean, dark_count = compute_dark_mean(entries)
    print(
        f"# Ground truth: {len(dark_mean)} buckets with >= "
        f"{MIN_DARK_HOURS_FOR_VERDICT} dark-hour samples "
        f"(out of {len(dark_count)} total dark-populated buckets)",
        file=sys.stderr,
    )
    if not dark_mean:
        sys.exit("ERROR: no buckets have enough dark-hour samples for ground truth")

    if args.sweep:
        run_sweep(entries, dark_mean, dark_count, args)
    else:
        run_single(entries, dark_mean, dark_count, args)


if __name__ == "__main__":
    main()
