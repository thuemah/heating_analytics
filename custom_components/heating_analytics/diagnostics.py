"""DiagnosticsEngine — hosts diagnose_model() and diagnose_solar() extracted from coordinator.py.

Thin-delegate pattern: the engine holds a reference to the coordinator
and reaches back for state.  Public methods are called via delegates
on the coordinator so the external API is unchanged.
"""
from __future__ import annotations

import logging
from datetime import date as _date, datetime, timedelta

from homeassistant.util import dt as dt_util

from .const import (
    MODE_COOLING,
    MODE_DHW,
    MODE_GUEST_COOLING,
    MODE_GUEST_HEATING,
    MODE_HEATING,
    MODE_OFF,
    SOLAR_BATTERY_DECAY,
)
from .solar import SolarCalculator

_LOGGER = logging.getLogger(__name__)


class DiagnosticsEngine:
    """Hosts the diagnose_model and diagnose_solar service implementations."""

    def __init__(self, coordinator) -> None:
        self.coordinator = coordinator

    def diagnose_model(self, days_back: int = 30) -> dict:
        """Analyze the learned model and history for data quality issues.

        Returns a diagnostic dict with monotonicity check, bucket
        population, mode contamination, solar correlation, and
        Track B specifics.  Designed as a service response.
        """
        from .const import MODES_EXCLUDED_FROM_GLOBAL_LEARNING

        result: dict = {}

        # --- 1. Monotonicity check per wind bucket ---
        monotonicity: dict[str, dict] = {}
        for wind_bucket in ("normal", "high_wind", "extreme_wind"):
            # Collect (temp_key_int, kwh) pairs for this bucket.
            points: list[tuple[int, float]] = []
            for temp_key_str, buckets in self.coordinator._correlation_data.items():
                if wind_bucket in buckets:
                    try:
                        points.append((int(temp_key_str), buckets[wind_bucket]))
                    except (ValueError, TypeError):
                        continue
            if len(points) < 2:
                monotonicity[wind_bucket] = {
                    "status": "insufficient_data",
                    "points": len(points),
                }
                continue

            points.sort(key=lambda p: p[0], reverse=True)  # warmest first
            inversions = []
            for i in range(len(points) - 1):
                t_warm, kwh_warm = points[i]
                t_cold, kwh_cold = points[i + 1]
                if kwh_cold < kwh_warm:
                    inversions.append({
                        "from_temp": t_warm,
                        "to_temp": t_cold,
                        "kwh_warm": round(kwh_warm, 4),
                        "kwh_cold": round(kwh_cold, 4),
                        "delta": round(kwh_warm - kwh_cold, 4),
                    })
            monotonicity[wind_bucket] = {
                "status": "monotonic" if not inversions else "inversions_found",
                "points": len(points),
                "temp_range": [points[-1][0], points[0][0]],
                "inversions": inversions,
            }
        result["monotonicity"] = monotonicity

        # --- 2. Bucket population ---
        bucket_pop: dict[str, dict] = {}
        total_observations = 0
        for temp_key_str, buckets in self.coordinator._correlation_data.items():
            for wind_bucket, kwh in buckets.items():
                if wind_bucket not in bucket_pop:
                    bucket_pop[wind_bucket] = {"count": 0, "temp_keys": []}
                bucket_pop[wind_bucket]["count"] += 1
                bucket_pop[wind_bucket]["temp_keys"].append(temp_key_str)
                total_observations += 1

        # Check for under-sampled buckets via learning buffers.
        buffered_buckets = 0
        for temp_key_str, buckets in self.coordinator._learning_buffer_global.items():
            for wind_bucket, samples in buckets.items():
                if samples:
                    buffered_buckets += 1

        bucket_summary = {
            "total_buckets_learned": total_observations,
            "buffered_pending": buffered_buckets,
            "per_wind_bucket": {
                wb: {"bucket_count": info["count"], "temp_range": [
                    min(info["temp_keys"], key=int) if info["temp_keys"] else None,
                    max(info["temp_keys"], key=int) if info["temp_keys"] else None,
                ]}
                for wb, info in bucket_pop.items()
            },
        }
        result["bucket_population"] = bucket_summary

        # --- 3. Mode contamination (from hourly log) ---
        from homeassistant.util import dt as dt_util
        now = dt_util.now()
        cutoff_str = (now - timedelta(days=days_back)).date().isoformat() if days_back else None

        mode_stats: dict[str, dict[str, int]] = {}
        total_hours = 0
        excluded_kwh_total = 0.0

        for entry in self.coordinator._hourly_log:
            ts = entry.get("timestamp", "")
            if cutoff_str and ts[:10] < cutoff_str:
                continue
            total_hours += 1
            day = ts[:10]
            if day not in mode_stats:
                mode_stats[day] = {"total_hours": 0, "excluded_hours": 0, "excluded_kwh": 0.0, "modes": {}}
            mode_stats[day]["total_hours"] += 1

            unit_modes = entry.get("unit_modes", {})
            breakdown = entry.get("unit_breakdown", {})
            for sid, kwh in breakdown.items():
                mode = unit_modes.get(sid, "heating")
                if mode != "heating":
                    mode_stats[day]["modes"][mode] = mode_stats[day]["modes"].get(mode, 0) + 1
                if mode in MODES_EXCLUDED_FROM_GLOBAL_LEARNING:
                    mode_stats[day]["excluded_hours"] += 1
                    mode_stats[day]["excluded_kwh"] += kwh
                    excluded_kwh_total += kwh

        # Summarize — only report days with excluded energy.
        contaminated_days = {
            day: stats for day, stats in mode_stats.items()
            if stats["excluded_hours"] > 0
        }
        result["mode_contamination"] = {
            "days_analyzed": len(mode_stats),
            "total_hours_analyzed": total_hours,
            "contaminated_days": len(contaminated_days),
            "total_excluded_kwh": round(excluded_kwh_total, 2),
            "details": dict(sorted(contaminated_days.items())[-10:]),  # Last 10 affected days
        }

        # --- 4. Solar correlation ---
        # Use a low threshold (> 0.01) to capture any hour with measurable solar.
        # The 0.05 threshold used in learning is too strict for diagnostics —
        # we want to see if there's ANY solar signal in the data.
        solar_errors: list[tuple[float, float]] = []
        for entry in self.coordinator._hourly_log:
            ts = entry.get("timestamp", "")
            if cutoff_str and ts[:10] < cutoff_str:
                continue
            solar_f = entry.get("solar_factor")
            actual = entry.get("actual_kwh")
            expected = entry.get("expected_kwh")
            if solar_f is not None and actual is not None and expected is not None and solar_f > 0.01:
                solar_errors.append((solar_f, actual - expected))

        solar_diag: dict = {"qualifying_hours": len(solar_errors)}

        # Report max solar_factor seen in the period regardless of qualification.
        all_solar_factors = [
            entry.get("solar_factor", 0.0) for entry in self.coordinator._hourly_log
            if (not cutoff_str or entry.get("timestamp", "")[:10] >= cutoff_str)
            and entry.get("solar_factor") is not None
        ]
        if all_solar_factors:
            solar_diag["max_solar_factor_in_period"] = round(max(all_solar_factors), 3)
            solar_diag["hours_with_any_solar"] = sum(1 for f in all_solar_factors if f > 0.0)

        if len(solar_errors) >= 10:
            avg_solar = sum(s for s, _ in solar_errors) / len(solar_errors)
            avg_error = sum(e for _, e in solar_errors) / len(solar_errors)
            # Simple correlation: positive = solar hours have positive error (under-predicted solar reduction)
            n = len(solar_errors)
            mean_s = avg_solar
            mean_e = avg_error
            cov = sum((s - mean_s) * (e - mean_e) for s, e in solar_errors) / n
            var_s = sum((s - mean_s) ** 2 for s, _ in solar_errors) / n
            var_e = sum((e - mean_e) ** 2 for _, e in solar_errors) / n
            denom = (var_s * var_e) ** 0.5
            correlation = round(cov / denom, 3) if denom > 0 else 0.0
            solar_diag["correlation_solar_vs_error"] = correlation
            solar_diag["avg_solar_factor"] = round(avg_solar, 3)
            solar_diag["avg_error_kwh"] = round(avg_error, 3)
            solar_diag["interpretation"] = (
                "Positive correlation: solar hours have higher-than-expected consumption — solar model may be under-subtracting."
                if correlation > 0.15 else
                "Negative correlation: solar hours have lower-than-expected consumption — solar model may be over-subtracting."
                if correlation < -0.15 else
                "No significant correlation — solar model appears well-calibrated."
            )
        result["solar_correlation"] = solar_diag

        # --- 6. Base model health (dark-hour replay) ---
        # For each (temp_key, wind_bucket) pair in the stored correlation
        # model, compute the mean reference kWh over hourly_log entries that
        # are (a) dark (solar_factor < 0.05 → no solar contamination),
        # (b) not aux-active (aux has its own learning path), and (c) not
        # contaminated by excluded modes in any unit.  Compare that empirical
        # dark-hour mean to the stored bucket value.
        #
        # Why dark-hour replay: the stored bucket is the output of a learning
        # loop that includes normalisation of solar impact.  If solar learning
        # is biased, normalisation is biased, and the stored bucket absorbs
        # that bias — a regression against contaminated history would
        # circularly validate the bias.  Dark hours have no solar signal to
        # normalise, so the mean reference kWh on those hours is an
        # independent ground-truth for what the bucket SHOULD have converged
        # to.  This surfaces the contamination pattern a user would otherwise
        # only detect by observing live over-prediction (e.g. cloudy mild day
        # running 20% above forecast).
        #
        # Track C semantics: bucket values for Track C installations are
        # written from synthetic_kwh_el (MPC thermal delivery / per-hour
        # COP), not raw electrical.  Comparing them to actual_kwh from the
        # meter would be apple-to-pear: a partial electrical sensor on a
        # Track C install (where MPC supplements the thermal picture)
        # under-represents what the bucket models, producing spuriously
        # large "inflated" verdicts.  When Track C is enabled AND a per-day
        # track_c_distribution exists, the reference becomes
        # synthetic_kwh_el[hour] for the MPC-managed sensor plus raw
        # actual_kwh contributions from non-MPC sensors.  Days that fell
        # back to Track B (no distribution) keep the raw-actual path.
        # The per-bucket result reports track_c_aware_hours so callers can
        # see the mix.
        DARK_SOLAR_FACTOR_THRESHOLD = 0.05
        MIN_DARK_HOURS_FOR_VERDICT = 10
        BUCKET_DEVIATION_THRESHOLD_PCT = 15.0

        is_track_c_install = bool(
            getattr(self.coordinator, "track_c_enabled", False)
            and getattr(self.coordinator, "mpc_managed_sensor", None)
        )
        mpc_sid = getattr(self.coordinator, "mpc_managed_sensor", None)

        # Per-day distribution cache: {date_str: {hour: synthetic_kwh_el}}.
        # Built lazily on first access per date so non-Track-C installs pay
        # nothing.  An empty dict marks "no distribution available".
        dist_by_day_hour: dict[str, dict[int, float]] = {}

        def _resolve_distribution(day_key: str) -> dict[int, float] | None:
            """Return {hour: synthetic_kwh_el} for a Track C day, or None."""
            if day_key in dist_by_day_hour:
                return dist_by_day_hour[day_key] or None
            day_history = self.coordinator._daily_history.get(day_key, {}) or {}
            raw_dist = day_history.get("track_c_distribution")
            if not raw_dist:
                dist_by_day_hour[day_key] = {}
                return None
            hour_map: dict[int, float] = {}
            for d in raw_dist:
                try:
                    dt_str = d.get("datetime", "")
                    # ISO format: "2026-04-15T13:00:00..."  Hour at chars 11-13.
                    if len(dt_str) >= 13 and dt_str[10] == "T":
                        h = int(dt_str[11:13])
                        hour_map[h] = float(d.get("synthetic_kwh_el", 0.0))
                except (ValueError, TypeError):
                    continue
            dist_by_day_hour[day_key] = hour_map
            return hour_map or None

        def _reference_dark_kwh(entry: dict) -> tuple[float, bool]:
            """Return (kWh value to compare against stored bucket, used_track_c)."""
            actual = entry.get("actual_kwh")
            if actual is None:
                return 0.0, False
            if not is_track_c_install:
                return float(actual), False
            day_key = entry.get("timestamp", "")[:10]
            hour_map = _resolve_distribution(day_key)
            if hour_map is None:
                return float(actual), False
            hour = entry.get("hour", -1)
            synthetic = hour_map.get(hour)
            if synthetic is None:
                # Track C day but this specific hour missing from the
                # distribution.  Fall back to raw rather than guess; the
                # caller sees the mix in track_c_aware_hours.
                return float(actual), False
            breakdown = entry.get("unit_breakdown", {}) or {}
            non_mpc_kwh = sum(
                kwh for sid, kwh in breakdown.items()
                if sid != mpc_sid
            )
            return synthetic + non_mpc_kwh, True

        # (temp_key, wind_bucket) -> {"values": [...], "track_c_count": int}
        dark_accum: dict[tuple[str, str], dict] = {}
        for entry in self.coordinator._hourly_log:
            ts = entry.get("timestamp", "")
            if cutoff_str and ts[:10] < cutoff_str:
                continue
            solar_f = entry.get("solar_factor")
            if solar_f is None or solar_f >= DARK_SOLAR_FACTOR_THRESHOLD:
                continue
            if entry.get("auxiliary_active", False):
                continue
            # Exclude hours where any unit was in an excluded mode — the
            # stored bucket represents all-heating energy, mixing in DHW or
            # guest hours would bias the empirical mean.
            unit_modes = entry.get("unit_modes", {}) or {}
            if any(m in MODES_EXCLUDED_FROM_GLOBAL_LEARNING for m in unit_modes.values()):
                continue
            temp_key = entry.get("temp_key")
            wind_bucket = entry.get("wind_bucket")
            if temp_key is None or wind_bucket is None or entry.get("actual_kwh") is None:
                continue
            ref_kwh, used_track_c = _reference_dark_kwh(entry)
            slot = dark_accum.setdefault(
                (temp_key, wind_bucket), {"values": [], "track_c_count": 0}
            )
            slot["values"].append(ref_kwh)
            if used_track_c:
                slot["track_c_count"] += 1

        base_health: dict[str, dict] = {}
        base_flags = {"inflated": [], "deflated": [], "unverifiable": []}
        for temp_key_str, buckets in self.coordinator._correlation_data.items():
            for wind_bucket, stored_kwh in buckets.items():
                slot = dark_accum.get(
                    (temp_key_str, wind_bucket),
                    {"values": [], "track_c_count": 0},
                )
                dark_hours = slot["values"]
                entry_out: dict = {
                    "stored_kwh": round(stored_kwh, 4),
                    "n_dark_hours": len(dark_hours),
                    "track_c_aware_hours": slot["track_c_count"],
                }
                if len(dark_hours) < MIN_DARK_HOURS_FOR_VERDICT:
                    entry_out["verdict"] = "unverifiable"
                    base_flags["unverifiable"].append(
                        {"temp_key": temp_key_str, "wind_bucket": wind_bucket}
                    )
                else:
                    dark_mean = sum(dark_hours) / len(dark_hours)
                    entry_out["actual_dark_mean_kwh"] = round(dark_mean, 4)
                    delta = stored_kwh - dark_mean
                    entry_out["delta_kwh"] = round(delta, 4)
                    # Percentage deviation guarded against near-zero stored
                    # (avoids division blowup at or near balance_point).
                    if abs(dark_mean) > 0.01:
                        delta_pct = 100.0 * delta / dark_mean
                    elif abs(stored_kwh) <= 0.01:
                        delta_pct = 0.0
                    else:
                        delta_pct = float("inf")
                    entry_out["delta_pct"] = round(delta_pct, 1) if delta_pct != float("inf") else None
                    if delta_pct == float("inf"):
                        entry_out["verdict"] = "inflated"
                        base_flags["inflated"].append(
                            {"temp_key": temp_key_str, "wind_bucket": wind_bucket,
                             "delta_pct": None, "stored_kwh": round(stored_kwh, 4),
                             "actual_dark_mean_kwh": round(dark_mean, 4)}
                        )
                    elif delta_pct > BUCKET_DEVIATION_THRESHOLD_PCT:
                        entry_out["verdict"] = "inflated"
                        base_flags["inflated"].append(
                            {"temp_key": temp_key_str, "wind_bucket": wind_bucket,
                             "delta_pct": round(delta_pct, 1)}
                        )
                    elif delta_pct < -BUCKET_DEVIATION_THRESHOLD_PCT:
                        entry_out["verdict"] = "deflated"
                        base_flags["deflated"].append(
                            {"temp_key": temp_key_str, "wind_bucket": wind_bucket,
                             "delta_pct": round(delta_pct, 1)}
                        )
                    else:
                        entry_out["verdict"] = "ok"
                base_health.setdefault(temp_key_str, {})[wind_bucket] = entry_out

        # Sort flag lists by |delta_pct| descending so worst offenders surface
        # first when the caller inspects a truncated view.
        for key in ("inflated", "deflated"):
            base_flags[key].sort(
                key=lambda x: abs(x.get("delta_pct") or 0.0),
                reverse=True,
            )

        result["base_model_health"] = {
            "config": {
                "dark_solar_factor_threshold": DARK_SOLAR_FACTOR_THRESHOLD,
                "min_dark_hours_for_verdict": MIN_DARK_HOURS_FOR_VERDICT,
                "bucket_deviation_threshold_pct": BUCKET_DEVIATION_THRESHOLD_PCT,
            },
            "buckets": base_health,
            "flags": base_flags,
            "summary": {
                "inflated_count": len(base_flags["inflated"]),
                "deflated_count": len(base_flags["deflated"]),
                "unverifiable_count": len(base_flags["unverifiable"]),
            },
        }

        # --- 7. Track B diagnostics ---
        track_b_days: list[dict] = []
        for day_key, day_data in sorted(self.coordinator._daily_history.items()):
            if cutoff_str and day_key < cutoff_str:
                continue
            if "kwh" not in day_data:
                continue
            track_b_entry: dict = {
                "date": day_key,
                "raw_kwh": round(day_data["kwh"], 2),
                "temp_avg": round(day_data.get("temp", 0.0), 1),
                "tdd": round(day_data.get("tdd", 0.0), 2),
                "wind_avg": round(day_data.get("wind", 0.0), 1),
            }
            if "track_c_kwh" in day_data:
                track_b_entry["track_c_kwh"] = round(day_data["track_c_kwh"], 2)
                track_b_entry["track"] = "C"
            elif "track_b_cop_distribution" in day_data:
                track_b_entry["track"] = "B_cop_smeared"
            else:
                track_b_entry["track"] = "B_flat" if self.coordinator.daily_learning_mode else "A"
            if "midnight_indoor_temp" in day_data:
                track_b_entry["midnight_indoor_temp"] = day_data["midnight_indoor_temp"]
            track_b_days.append(track_b_entry)

        result["daily_history"] = {
            "days": len(track_b_days),
            "entries": track_b_days[-30:],  # Last 30 days
        }

        result["config_summary"] = {
            "daily_learning_mode": self.coordinator.daily_learning_mode,
            "track_c_enabled": self.coordinator.track_c_enabled,
            "balance_point": self.coordinator.balance_point,
            "learning_rate": self.coordinator.learning_rate,
            "wind_threshold": self.coordinator.wind_threshold,
            "extreme_wind_threshold": self.coordinator.extreme_wind_threshold,
            "solar_enabled": self.coordinator.solar_enabled,
            "learned_u_coefficient": round(self.coordinator._learned_u_coefficient, 4) if self.coordinator._learned_u_coefficient else None,
            "energy_sensors": self.coordinator.energy_sensors,
        }

        # #855 Option B: runtime counter of Track C MPC outages (days where
        # midnight sync found no distribution and skipped bucket+U updates).
        # Runtime-only; resets on HA restart.  Diagnostic signal for whether
        # the install is seeing frequent MPC unavailability.
        result["track_c_outage_session_count"] = self.coordinator._track_c_outage_count_session

        return result


    def diagnose_solar(self, days_back: int = 30, apply_battery_decay: bool = False) -> dict:
        """Analyze per-unit solar coefficient health and global solar model quality.

        Single-pass over hourly_log. Returns a diagnostic dict with per-unit
        coefficient validation, global battery/screen health, and temporal bias.
        Includes battery decay calibration: sweeps decay rates 0.50-0.95 to find
        the optimal value for this building. Designed as a service response (#810).
        """
        from datetime import timedelta, date as _date
        from homeassistant.util import dt as dt_util
        from .const import (
            MODE_HEATING, MODE_COOLING, MODE_OFF, MODE_DHW,
            MODE_GUEST_HEATING, MODE_GUEST_COOLING,
        )
        from .solar import SolarCalculator as _SC

        now = dt_util.now()
        cutoff = (now - timedelta(days=days_back)).date().isoformat()

        # --- Accumulators ---
        # Per-unit: normal equations for implied coefficient (3 windows + 30d total)
        window_boundaries = [days_back, int(days_back * 2 / 3), int(days_back / 3), 0]
        unit_accum: dict[str, dict] = {}  # entity_id -> accumulators

        def _get_unit_accum(eid: str) -> dict:
            if eid not in unit_accum:
                unit_accum[eid] = {
                    # Normal equations for 30d implied coefficient (3x3: S, E, W)
                    "ss": 0.0, "ee": 0.0, "ww": 0.0,
                    "se": 0.0, "sw": 0.0, "ew": 0.0,
                    "sI": 0.0, "eI": 0.0, "wI": 0.0, "n": 0,
                    # 3-window stability (each has own normal eqs)
                    "windows": [
                        {"ss": 0.0, "ee": 0.0, "ww": 0.0,
                         "se": 0.0, "sw": 0.0, "ew": 0.0,
                         "sI": 0.0, "eI": 0.0, "wI": 0.0, "n": 0}
                        for _ in range(3)
                    ],
                    # Delta accumulator
                    "sum_delta": 0.0, "delta_n": 0,
                    # Temporal bias
                    "morning_delta": 0.0, "morning_n": 0,
                    "afternoon_delta": 0.0, "afternoon_n": 0,
                    # Saturation counter
                    "saturated": 0, "qualifying": 0,
                    # Solar shutdown (#838): count + parallel normal equations
                    # that exclude shutdown hours, so we can compare the
                    # implied coefficient with and without shutdown bias.
                    "shutdown_hours": 0,
                    "no_shutdown": {
                        "ss": 0.0, "ee": 0.0, "ww": 0.0,
                        "se": 0.0, "sw": 0.0, "ew": 0.0,
                        "sI": 0.0, "eI": 0.0, "wI": 0.0, "n": 0,
                    },
                    # Screen-position stratification (#826 validation).  Split
                    # modeled-vs-implied delta by correction_percent bucket so
                    # systematic bias at closed screens surfaces directly.
                    # open:   correction ≥ 80   (screens fully open-ish)
                    # mid:    40 ≤ correction < 80
                    # closed: correction < 40   (screens mostly deployed)
                    "correction_buckets": {
                        "open":   {"delta_sum": 0.0, "modeled_sum": 0.0, "implied_sum": 0.0, "n": 0},
                        "mid":    {"delta_sum": 0.0, "modeled_sum": 0.0, "implied_sum": 0.0, "n": 0},
                        "closed": {"delta_sum": 0.0, "modeled_sum": 0.0, "implied_sum": 0.0, "n": 0},
                    },
                    # Per-hour tuples for transmittance sensitivity sweep.
                    # Stored as effective vectors + correction so we can
                    # re-reconstruct potential under hypothesis transmittances.
                    # Each entry: (eff_s, eff_e, eff_w, correction, implied)
                    "sensitivity_tuples": [],
                    # Temperature-regime stratification relative to the
                    # configured balance_point.  Splits the modeled-vs-implied
                    # delta by COP-regime so we can empirically test the
                    # coefficient's documented COP-blindness (#826 follow-up):
                    # - heating_deep: T < BP-8   (low COP, defrost, aux-regime)
                    # - heating_mild: BP-8 ≤ T < BP-2 (optimal heat pump)
                    # - cooling:      T > BP+2   (inverted solar semantics)
                    # The ±2° transition zone around BP is deliberately dropped;
                    # mode flips hour-to-hour there and the signal is noise.
                    "temp_buckets": {
                        "heating_deep": {"delta_sum": 0.0, "n": 0},
                        "heating_mild": {"delta_sum": 0.0, "n": 0},
                        "cooling":      {"delta_sum": 0.0, "n": 0},
                    },
                }
            return unit_accum[eid]

        # Global accumulators
        excluded = {"aux": 0, "guest": 0, "saturated": 0, "low_vector": 0, "no_base": 0, "legacy": 0}
        total_qualifying = 0

        # Battery calibration: collect per-day hourly sequences for decay sweep.
        # Key = date string, value = list of (hour, solar_impact_raw, actual, expected)
        day_sequences: dict[str, list[tuple]] = {}
        # Battery health: post-sunset hours (using current decay)
        battery_residuals: list[float] = []

        # Screen stratification
        screen_closed_errors: list[float] = []  # correction < 50, solar_factor > 0.3
        screen_open_errors: list[float] = []    # correction > 80, solar_factor > 0.3

        # Hour-of-day residual curve (hours 6-18)
        hourly_residuals: dict[int, list[float]] = {h: [] for h in range(6, 19)}

        # --- Single pass ---
        for entry in self.coordinator._hourly_log:
            ts = entry.get("timestamp", "")
            if ts[:10] < cutoff:
                continue

            hour = entry.get("hour", -1)
            solar_factor = entry.get("solar_factor") or 0.0
            solar_s = entry.get("solar_vector_s", 0.0)
            solar_e = entry.get("solar_vector_e", 0.0)
            solar_w = entry.get("solar_vector_w", 0.0)
            vector_mag = (solar_s ** 2 + solar_e ** 2 + solar_w ** 2) ** 0.5
            correction = entry.get("correction_percent", 100.0)
            # Outdoor temperature for BP-relative regime stratification.
            # Fall back through the same chain the learning path uses so we
            # stay consistent with the inertia-adjusted temperature when
            # present, and raw temp otherwise.
            temp_entry = (
                entry.get("inertia_temp")
                if entry.get("inertia_temp") is not None
                else entry.get("temp")
            )
            solar_impact_raw = entry.get("solar_impact_raw_kwh", 0.0)
            solar_impact_eff = entry.get("solar_impact_kwh", 0.0)
            aux_active = entry.get("auxiliary_active", False)
            guest_kwh = entry.get("guest_impact_kwh", 0.0)
            unit_modes = entry.get("unit_modes", {})
            unit_breakdown = entry.get("unit_breakdown", {})
            unit_expected_base = entry.get("unit_expected_breakdown", {})
            # Solar shutdown (#838): missing on legacy logs → empty list means
            # no shutdown hours recorded, which is treated as "all qualifying
            # hours are non-shutdown" below.
            log_shutdown_entities = set(entry.get("solar_dominant_entities", []) or [])

            # Battery health: post-sunset hours with residual battery charge
            if solar_factor < 0.01 and solar_impact_eff > 0.05:
                actual_total = entry.get("actual_kwh", 0.0)
                expected_total = entry.get("expected_kwh", 0.0)
                if expected_total > 0.05:
                    battery_residuals.append(actual_total - expected_total)

            # Collect day sequences for battery decay calibration sweep
            day_key = ts[:10]
            actual_total = entry.get("actual_kwh", 0.0)
            expected_total = entry.get("expected_kwh", 0.0)
            if day_key not in day_sequences:
                day_sequences[day_key] = []
            day_sequences[day_key].append((hour, solar_impact_raw, actual_total, expected_total))

            # Days-ago for window assignment
            try:
                days_ago = (now.date() - _date.fromisoformat(ts[:10])).days
            except (ValueError, TypeError):
                days_ago = 0

            # Per-unit analysis
            for entity_id in self.coordinator.energy_sensors:
                mode = unit_modes.get(entity_id, MODE_HEATING)
                if mode in (MODE_OFF, MODE_DHW, MODE_GUEST_HEATING, MODE_GUEST_COOLING):
                    if mode in (MODE_GUEST_HEATING, MODE_GUEST_COOLING):
                        excluded["guest"] += 1
                    continue

                if aux_active:
                    excluded["aux"] += 1
                    continue

                if vector_mag < 0.01:
                    excluded["low_vector"] += 1
                    continue

                actual_unit = unit_breakdown.get(entity_id, 0.0)
                base_unit = unit_expected_base.get(entity_id)
                if base_unit is None:
                    excluded["no_base"] += 1
                    continue

                # Implied solar reduction
                if mode == MODE_HEATING:
                    implied_solar = base_unit - actual_unit
                elif mode == MODE_COOLING:
                    implied_solar = actual_unit - base_unit
                else:
                    continue

                # Check saturation
                acc = _get_unit_accum(entity_id)
                acc["qualifying"] += 1
                if mode == MODE_HEATING and actual_unit < 0.05 * base_unit and base_unit > 0.05:
                    acc["saturated"] += 1
                    excluded["saturated"] += 1
                    continue

                implied_solar = max(0.0, implied_solar)

                # Modeled solar for this unit.
                # Reconstruct potential vector from effective + screen transmittance.
                # When ``correction`` is missing on a legacy log entry, fall back
                # to 100 % (screens fully open) so the helper's per-direction
                # transmittance returns 1.0 for every direction — matching the
                # pre-#876 ``t = 1.0`` short-circuit.  The equivalence relies on
                # the ramp identity ``mn + (1 - mn) * 1.0 = 1.0``, which holds
                # for both the legacy composite floor (mn=0.30) and the screened
                # floor (mn=0.08); unscreened directions are fixed at 1.0.
                effective = (solar_s, solar_e, solar_w)
                diag_potential = _SC.reconstruct_potential_vector(
                    effective,
                    correction if correction is not None else 100.0,
                    self.coordinator.screen_config,
                )
                unit_coeff = self.coordinator.solar.calculate_unit_coefficient(entity_id, entry.get("temp_key", "10"))
                modeled_solar = self.coordinator.solar.calculate_unit_solar_impact(diag_potential, unit_coeff)
                delta = modeled_solar - implied_solar

                total_qualifying += 1

                # Screen-position stratification (#826).  The per-direction
                # transmittance model introduced in 1.3.3 uses values (0.08
                # per-direction, 0.30 composite legacy) that were chosen from
                # literature, not measured at this building.  If either value
                # is wrong, modeled_solar will systematically differ from
                # implied_solar at the screen-position extremes while matching
                # well at fully-open screens (where transmittance = 1.0
                # regardless).  The correction-bucket split surfaces that
                # directly: a bias pattern of (open ≈ 0, closed ≠ 0) points
                # to a wrong transmittance floor; a uniform bias points to a
                # coefficient calibration issue.
                if correction is None:
                    bucket_key = "open"
                elif correction >= 80.0:
                    bucket_key = "open"
                elif correction >= 40.0:
                    bucket_key = "mid"
                else:
                    bucket_key = "closed"
                bucket = acc["correction_buckets"][bucket_key]
                bucket["delta_sum"] += delta
                bucket["modeled_sum"] += modeled_solar
                bucket["implied_sum"] += implied_solar
                bucket["n"] += 1

                # Temperature-regime stratification (BP-relative).  Splits
                # the same delta into COP-regime buckets so coefficient bias
                # correlated with heat-pump efficiency becomes visible.  The
                # ±2° transition zone around balance_point is dropped: mode
                # flips hour-to-hour there and the signal would be noise.
                if temp_entry is not None:
                    bp = self.coordinator.balance_point
                    if mode == MODE_HEATING:
                        if temp_entry < bp - 8.0:
                            tb = acc["temp_buckets"]["heating_deep"]
                            tb["delta_sum"] += delta
                            tb["n"] += 1
                        elif temp_entry < bp - 2.0:
                            tb = acc["temp_buckets"]["heating_mild"]
                            tb["delta_sum"] += delta
                            tb["n"] += 1
                        # else: transition zone (BP-2 ≤ T ≤ BP), dropped
                    elif mode == MODE_COOLING and temp_entry > bp + 2.0:
                        tb = acc["temp_buckets"]["cooling"]
                        tb["delta_sum"] += delta
                        tb["n"] += 1

                # Sensitivity sweep tuple — effective vector + correction lets
                # us re-reconstruct potential under hypothesis transmittances
                # post-loop without replaying the whole log.  Shutdown hours
                # excluded to avoid feeding the same bias into the sweep that
                # no_shutdown already subtracts from the baseline fit.
                if entity_id not in log_shutdown_entities:
                    acc["sensitivity_tuples"].append(
                        (solar_s, solar_e, solar_w,
                         correction if correction is not None else 100.0,
                         implied_solar)
                    )

                # Solar shutdown tracking (#838): count shutdown hours and
                # also accumulate a separate set of normal equations from
                # non-shutdown hours, so we can compare the implied
                # coefficient with vs without shutdown bias.
                is_shutdown_hour = entity_id in log_shutdown_entities
                if is_shutdown_hour:
                    acc["shutdown_hours"] += 1
                else:
                    ns = acc["no_shutdown"]
                    ns["ss"] += solar_s * solar_s
                    ns["ee"] += solar_e * solar_e
                    ns["ww"] += solar_w * solar_w
                    ns["se"] += solar_s * solar_e
                    ns["sw"] += solar_s * solar_w
                    ns["ew"] += solar_e * solar_w
                    ns["sI"] += solar_s * implied_solar
                    ns["eI"] += solar_e * implied_solar
                    ns["wI"] += solar_w * implied_solar
                    ns["n"] += 1

                # Normal equations (30d total, 3x3)
                acc["ss"] += solar_s * solar_s
                acc["ee"] += solar_e * solar_e
                acc["ww"] += solar_w * solar_w
                acc["se"] += solar_s * solar_e
                acc["sw"] += solar_s * solar_w
                acc["ew"] += solar_e * solar_w
                acc["sI"] += solar_s * implied_solar
                acc["eI"] += solar_e * implied_solar
                acc["wI"] += solar_w * implied_solar
                acc["n"] += 1

                # Window assignment
                for w_idx in range(3):
                    w_start = window_boundaries[w_idx]
                    w_end = window_boundaries[w_idx + 1]
                    if w_end <= days_ago < w_start:
                        w = acc["windows"][w_idx]
                        w["ss"] += solar_s * solar_s
                        w["ee"] += solar_e * solar_e
                        w["ww"] += solar_w * solar_w
                        w["se"] += solar_s * solar_e
                        w["sw"] += solar_s * solar_w
                        w["ew"] += solar_e * solar_w
                        w["sI"] += solar_s * implied_solar
                        w["eI"] += solar_e * implied_solar
                        w["wI"] += solar_w * implied_solar
                        w["n"] += 1
                        break

                # Delta
                acc["sum_delta"] += delta
                acc["delta_n"] += 1

                # Temporal bias
                if 6 <= hour <= 11:
                    acc["morning_delta"] += delta
                    acc["morning_n"] += 1
                elif 12 <= hour <= 17:
                    acc["afternoon_delta"] += delta
                    acc["afternoon_n"] += 1

                # Screen stratification (global)
                if solar_factor > 0.3:
                    if correction < 50:
                        screen_closed_errors.append(delta)
                    elif correction > 80:
                        screen_open_errors.append(delta)

                # Hour-of-day residual
                if 6 <= hour <= 18:
                    hourly_residuals[hour].append(delta)

        # --- Build results ---
        def _solve_normal(a, n, min_samples=10):
            """Solve normal equations for implied coefficient.

            Attempts a 3x3 solve (S, E, W).  When the west dimension has
            no variance (sum_ww ≈ 0, typical for legacy logs without
            solar_vector_w), falls back to the 2x2 (S, E) system so that
            diagnose_solar remains useful immediately after upgrading.

            Args:
                a: dict with keys ss, ee, ww, se, sw, ew, sI, eI, wI.
                n: number of qualifying samples.
            """
            if n < min_samples:
                return None
            ss, ee, ww = a["ss"], a["ee"], a["ww"]
            se, sw, ew = a["se"], a["sw"], a["ew"]
            sI, eI, wI = a["sI"], a["eI"], a["wI"]
            # 3x3 determinant via cofactor expansion
            det = (
                ss * (ee * ww - ew * ew)
                - se * (se * ww - ew * sw)
                + sw * (se * ew - ee * sw)
            )
            if abs(det) > 1e-6:
                # Full 3D solution
                det_s = (
                    sI * (ee * ww - ew * ew)
                    - se * (eI * ww - ew * wI)
                    + sw * (eI * ew - ee * wI)
                )
                det_e = (
                    ss * (eI * ww - ew * wI)
                    - sI * (se * ww - ew * sw)
                    + sw * (se * wI - eI * sw)
                )
                det_w = (
                    ss * (ee * wI - ew * eI)
                    - se * (se * wI - eI * sw)
                    + sI * (se * ew - ee * sw)
                )
                return {
                    "s": round(det_s / det, 4),
                    "e": round(det_e / det, 4),
                    "w": round(det_w / det, 4),
                }
            # Fallback: 2D solve (S, E) when W dimension has no variance
            # (legacy logs or insufficient afternoon data)
            det_2d = ss * ee - se * se
            if abs(det_2d) > 1e-6:
                return {
                    "s": round((ee * sI - se * eI) / det_2d, 4),
                    "e": round((ss * eI - se * sI) / det_2d, 4),
                    "w": 0.0,
                }
            return None

        # Shadow replay with inequality learner (#865).  Runs a clean
        # NLMS + inequality replay from zero coefficients over the same
        # window to show what the learner would produce if the user
        # reset and refit now.  This is the pre-validation ankerpunkt
        # The inequality learner's effect becomes visible as the delta
        # between this and ``implied_coefficient_30d_no_shutdown``.
        # Single shadow pass, no side effects on live state (all dicts
        # are local copies).
        window_entries = [
            e for e in self.coordinator._hourly_log
            if e.get("timestamp", "") >= cutoff
        ]
        shadow_coeffs: dict = {}
        shadow_buffers: dict = {}
        # Seed shadow per-unit correlation from the current coordinator
        # state — inequality needs a base reference.  If empty (fresh
        # install), the replay falls back to NLMS's threshold gate.
        shadow_diag = self.coordinator.learning.replay_solar_nlms(
            window_entries,
            solar_calculator=self.coordinator.solar,
            screen_config=getattr(self.coordinator, "screen_config", None),
            correlation_data_per_unit=self.coordinator._correlation_data_per_unit,
            solar_coefficients_per_unit=shadow_coeffs,
            learning_buffer_solar_per_unit=shadow_buffers,
            energy_sensors=self.coordinator.energy_sensors,
            learning_rate=self.coordinator.learning_rate,
            balance_point=self.coordinator.balance_point,
            aux_affected_entities=self.coordinator.aux_affected_entities,
            unit_strategies=self.coordinator._unit_strategies,
            daily_history=self.coordinator._daily_history,
            unit_min_base=self.coordinator._per_unit_min_base_thresholds or None,
            return_diagnostics=True,
        )

        per_unit = {}
        for entity_id, acc in unit_accum.items():
            current = self.coordinator.solar.calculate_unit_coefficient(entity_id, "10")

            implied_30d = _solve_normal(acc, acc["n"])

            # Physical-space implied (undo per-direction screen transmittance, #826)
            implied_physical = None
            if implied_30d is not None:
                # We can't perfectly recover avg transmittance from the log,
                # but screen_closed + screen_open counts give us a proxy.
                # Use the formula: physical = effective / transmittance, applied
                # per cardinal direction so unscreened facades are not divided
                # by an irrelevant factor.
                current_correction = self.coordinator.solar_correction_percent
                t_s, t_e, t_w = _SC._screen_transmittance_vector(
                    current_correction, self.coordinator.screen_config
                )
                implied_physical = {
                    "s": round(implied_30d["s"] / t_s, 4) if t_s > 0.01 else implied_30d["s"],
                    "e": round(implied_30d["e"] / t_e, 4) if t_e > 0.01 else implied_30d["e"],
                    "w": round(implied_30d["w"] / t_w, 4) if t_w > 0.01 else implied_30d["w"],
                }

            # Stability windows
            stability = []
            for w in acc["windows"]:
                coeff = _solve_normal(w, w["n"], min_samples=5)
                stability.append({"coefficient": coeff, "qualifying_hours": w["n"]})

            # Flags
            flags = []
            if acc["qualifying"] > 0 and acc["saturated"] / acc["qualifying"] > 0.3:
                flags.append("high_saturation")
            # Coefficient stability: check if S component varies >2x between windows
            s_values = [w["coefficient"]["s"] for w in stability if w["coefficient"] is not None and abs(w["coefficient"]["s"]) > 0.01]
            if len(s_values) >= 2 and max(abs(v) for v in s_values) > 2 * min(abs(v) for v in s_values):
                flags.append("coefficient_unstable")
            # Under-prediction
            mean_delta = acc["sum_delta"] / acc["delta_n"] if acc["delta_n"] > 0 else 0.0
            if mean_delta < -0.1:
                flags.append("under_predicting_solar")
            elif mean_delta > 0.1:
                flags.append("over_predicting_solar")

            # Dominant component
            cs = abs(current.get("s", 0.0))
            ce = abs(current.get("e", 0.0))
            cw = abs(current.get("w", 0.0))
            total_c = cs + ce + cw
            dominant = "balanced"
            if total_c > 0.01:
                if cs / total_c > 0.9:
                    dominant = "south"
                elif ce / total_c > 0.9:
                    dominant = "east"
                elif cw / total_c > 0.9:
                    dominant = "west"

            # Solar shutdown diagnostics (#838): compare the 30-day implied
            # coefficient with and without shutdown hours.  A large gap
            # indicates those hours were biasing the learned coefficient.
            implied_no_shutdown = _solve_normal(acc["no_shutdown"], acc["no_shutdown"]["n"])
            if acc["shutdown_hours"] > 0 and acc["qualifying"] > 0:
                shutdown_pct = round(100 * acc["shutdown_hours"] / acc["qualifying"], 1)
            else:
                shutdown_pct = 0.0
            if acc["shutdown_hours"] >= 5:
                flags.append("solar_shutdown_detected")

            # Temperature-regime stratification (BP-relative, #826 follow-up).
            # First pass reports means only — no threshold flags until
            # empirical distributions from European summer data tell us what
            # "significant bias" looks like for this metric.  Buckets with
            # zero qualifying hours are emitted as {"n": 0} so the JSON
            # shape is stable across installations.
            temperature_stratified = {}
            for tb_key, tb in acc["temp_buckets"].items():
                if tb["n"] > 0:
                    temperature_stratified[tb_key] = {
                        "n": tb["n"],
                        "mean_delta_kwh": round(tb["delta_sum"] / tb["n"], 4),
                    }
                else:
                    temperature_stratified[tb_key] = {"n": 0}

            # Screen stratification (#826).  Report mean delta per correction
            # bucket along with n, so downstream (and humans) can distinguish
            # "tiny sample, noisy" from "real bias".
            screen_stratified = {}
            for bkey, b in acc["correction_buckets"].items():
                if b["n"] > 0:
                    screen_stratified[bkey] = {
                        "n": b["n"],
                        "mean_delta_kwh": round(b["delta_sum"] / b["n"], 4),
                        "mean_modeled_kwh": round(b["modeled_sum"] / b["n"], 4),
                        "mean_implied_kwh": round(b["implied_sum"] / b["n"], 4),
                    }
                else:
                    screen_stratified[bkey] = {"n": 0}
            # Bias trend: does |mean_delta| grow as screens close?  Only
            # meaningful when both extremes have enough samples.
            if (
                acc["correction_buckets"]["open"]["n"] >= 10
                and acc["correction_buckets"]["closed"]["n"] >= 10
            ):
                open_bias = (
                    acc["correction_buckets"]["open"]["delta_sum"]
                    / acc["correction_buckets"]["open"]["n"]
                )
                closed_bias = (
                    acc["correction_buckets"]["closed"]["delta_sum"]
                    / acc["correction_buckets"]["closed"]["n"]
                )
                bias_gap = closed_bias - open_bias
                screen_stratified["bias_gap_kwh"] = round(bias_gap, 4)
                # bias_gap > 0.05 → closed over-predicts relative to open
                # → transmittance_model at closed is TOO LOW.  The model
                # assumes less sun passes through than reality; reconstructed
                # potential is inflated; coeff_learned absorbs a mix that
                # over-predicts on fully-closed hours.  The sensitivity
                # sweep below typically points at a higher optimal in this
                # regime, which is the fix: raise SCREEN_DIRECT_TRANSMITTANCE.
                # bias_gap < −0.05 → transmittance TOO HIGH (symmetric case).
                # Prior to the 1.3.3 fix these two flag names were swapped.
                if bias_gap > 0.05:
                    flags.append("transmittance_floor_too_low")
                elif bias_gap < -0.05:
                    flags.append("transmittance_floor_too_high")

            # Transmittance sensitivity sweep.  For each candidate value of
            # SCREEN_DIRECT_TRANSMITTANCE, re-solve the 3×3 normal equations
            # using potential reconstructed under that hypothesis, then
            # compute the residual RMSE against implied_solar.  The
            # hypothesis minimising RMSE is the empirically optimal floor
            # for this installation's data.
            sensitivity = None
            tuples = acc["sensitivity_tuples"]
            if len(tuples) >= 20:
                candidates = [0.05, 0.08, 0.12, 0.15, 0.20, 0.25]
                cfg = self.coordinator.screen_config
                results = []
                # Correction-variance gate: if slider barely changes, the
                # sweep is uninformative — all candidates will yield similar
                # RMSE because transmittance(100%) == 1.0 for every candidate.
                corrections = [t[3] for t in tuples]
                corr_var = (
                    max(corrections) - min(corrections) if corrections else 0.0
                )
                for cand in candidates:
                    # Build 3×3 normal equations with potential = eff / t_cand
                    A_ss = A_ee = A_ww = 0.0
                    A_se = A_sw = A_ew = 0.0
                    b_s = b_e = b_w = 0.0
                    for eff_s, eff_e, eff_w, corr, implied in tuples:
                        # Per-direction transmittance under candidate floor
                        pct = max(0.0, min(100.0, corr)) / 100.0
                        t_screened = cand + (1.0 - cand) * pct
                        t_sc_s = t_screened if (cfg is None or cfg[0]) else 1.0
                        t_sc_e = t_screened if (cfg is None or cfg[1]) else 1.0
                        t_sc_w = t_screened if (cfg is None or cfg[2]) else 1.0
                        p_s = eff_s / t_sc_s if t_sc_s > 0.01 else eff_s
                        p_e = eff_e / t_sc_e if t_sc_e > 0.01 else eff_e
                        p_w = eff_w / t_sc_w if t_sc_w > 0.01 else eff_w
                        A_ss += p_s * p_s
                        A_ee += p_e * p_e
                        A_ww += p_w * p_w
                        A_se += p_s * p_e
                        A_sw += p_s * p_w
                        A_ew += p_e * p_w
                        b_s += p_s * implied
                        b_e += p_e * implied
                        b_w += p_w * implied
                    coeff_h = _solve_normal(
                        {"ss": A_ss, "ee": A_ee, "ww": A_ww,
                         "se": A_se, "sw": A_sw, "ew": A_ew,
                         "sI": b_s, "eI": b_e, "wI": b_w, "n": len(tuples)},
                        len(tuples),
                    )
                    if coeff_h is None:
                        continue
                    # Residual RMSE under the fitted coefficient
                    sse = 0.0
                    for eff_s, eff_e, eff_w, corr, implied in tuples:
                        pct = max(0.0, min(100.0, corr)) / 100.0
                        t_screened = cand + (1.0 - cand) * pct
                        t_sc_s = t_screened if (cfg is None or cfg[0]) else 1.0
                        t_sc_e = t_screened if (cfg is None or cfg[1]) else 1.0
                        t_sc_w = t_screened if (cfg is None or cfg[2]) else 1.0
                        p_s = eff_s / t_sc_s if t_sc_s > 0.01 else eff_s
                        p_e = eff_e / t_sc_e if t_sc_e > 0.01 else eff_e
                        p_w = eff_w / t_sc_w if t_sc_w > 0.01 else eff_w
                        pred = (
                            coeff_h["s"] * p_s
                            + coeff_h["e"] * p_e
                            + coeff_h["w"] * p_w
                        )
                        sse += (implied - pred) ** 2
                    rmse = (sse / len(tuples)) ** 0.5
                    results.append({
                        "screen_direct_transmittance": cand,
                        "implied_coefficient": coeff_h,
                        "residual_rmse_kwh": round(rmse, 4),
                    })
                if results:
                    best = min(results, key=lambda r: r["residual_rmse_kwh"])
                    sensitivity = {
                        "n_hours": len(tuples),
                        "correction_range_pct": round(corr_var, 1),
                        "informative": corr_var >= 40.0,  # ≥40 pct points of slider variance
                        "candidates": results,
                        "best": best,
                    }
                    if (
                        sensitivity["informative"]
                        and abs(best["screen_direct_transmittance"] - 0.08) > 0.04
                    ):
                        flags.append("sensitivity_suggests_transmittance_retune")

            # Inequality-replay coefficient (#865) — what the learner would
            # produce if retrained from zero over the same window.  Absence
            # means the unit did not qualify for any update (no shutdown
            # hours, or base below SOLAR_SHUTDOWN_MIN_BASE everywhere).
            implied_inequality_coeff = shadow_coeffs.get(entity_id)
            if implied_inequality_coeff is not None:
                implied_inequality_coeff = {
                    k: round(v, 4) for k, v in implied_inequality_coeff.items()
                }

            per_unit[entity_id] = {
                "current_coefficient": {k: round(v, 4) for k, v in current.items()},
                "implied_coefficient_30d": implied_30d,
                "implied_coefficient_30d_no_shutdown": implied_no_shutdown,
                "implied_coefficient_inequality": implied_inequality_coeff,
                "implied_coefficient_physical": implied_physical,
                "stability_windows": stability,
                "mean_delta_kwh": round(mean_delta, 4),
                "saturation_pct": round(100 * acc["saturated"] / acc["qualifying"], 1) if acc["qualifying"] > 0 else 0.0,
                "shutdown_hours_30d": acc["shutdown_hours"],
                "shutdown_pct_of_qualifying": shutdown_pct,
                "dominant_component": dominant,
                "qualifying_hours": acc["n"],
                "temperature_stratified": temperature_stratified,
                "screen_stratified": screen_stratified,
                "transmittance_sensitivity": sensitivity,
                "temporal_bias": {
                    "morning_mean_delta": round(acc["morning_delta"] / acc["morning_n"], 4) if acc["morning_n"] > 0 else None,
                    "afternoon_mean_delta": round(acc["afternoon_delta"] / acc["afternoon_n"], 4) if acc["afternoon_n"] > 0 else None,
                },
                "flags": flags,
            }

        # Global metrics
        global_flags = []

        # Battery health
        battery_health = {}
        if battery_residuals:
            mean_residual = sum(battery_residuals) / len(battery_residuals)
            battery_health = {
                "mean_residual_kwh": round(mean_residual, 4),
                "qualifying_post_sunset_hours": len(battery_residuals),
                "decay_rate": self.coordinator.solar_battery_decay,
                # Negative residual = actual < expected = battery under-credits
                # post-sunset = decays too fast. Positive = opposite.
                "assessment": "too_slow" if mean_residual > 0.05 else ("too_fast" if mean_residual < -0.05 else "ok"),
            }
            if battery_health["assessment"] != "ok":
                global_flags.append(f"battery_decay_{battery_health['assessment']}")

        # Battery decay calibration: sweep decay rates to find optimal
        calibration = {}
        candidates = [round(0.50 + i * 0.05, 2) for i in range(10)]  # 0.50 to 0.95
        if day_sequences:
            best_decay = self.coordinator.solar_battery_decay
            best_abs_residual = float("inf")
            sweep_results = {}
            for candidate in candidates:
                total_residual = 0.0
                n_post_sunset = 0
                for _day_key, hours in day_sequences.items():
                    hours_sorted = sorted(hours, key=lambda x: x[0])
                    sim_battery = 0.0
                    for h, raw_solar, actual, expected in hours_sorted:
                        sim_battery = sim_battery * candidate + raw_solar * (1 - candidate)
                        # Post-sunset: solar_factor ~ 0 but battery > 0
                        if raw_solar < 0.01 and sim_battery > 0.05 and expected > 0.05:
                            # expected already includes current battery effect.
                            # Residual from this candidate = what the candidate battery
                            # would credit minus what the current battery credited.
                            # Simpler: just track if actual-expected improves.
                            total_residual += actual - expected
                            n_post_sunset += 1
                if n_post_sunset >= 5:
                    mean_res = total_residual / n_post_sunset
                    sweep_results[str(candidate)] = round(mean_res, 4)
                    if abs(mean_res) < best_abs_residual:
                        best_abs_residual = abs(mean_res)
                        best_decay = candidate
            calibration = {
                "current_decay": self.coordinator.solar_battery_decay,
                "recommended_decay": best_decay,
                "sweep_results": sweep_results,
                "post_sunset_hours_evaluated": max(
                    (sum(1 for h, raw, a, e in hrs if raw < 0.01)
                     for hrs in day_sequences.values()), default=0
                ),
            }
            if apply_battery_decay and best_decay != self.coordinator.solar_battery_decay:
                old_decay = self.coordinator.solar_battery_decay
                self.coordinator.solar_battery_decay = best_decay
                # Persist to entry.data
                new_data = {**self.coordinator.entry.data, "solar_battery_decay": best_decay}
                self.coordinator.hass.config_entries.async_update_entry(self.coordinator.entry, data=new_data)
                calibration["applied"] = True
                _LOGGER.info(
                    "Solar battery decay calibrated: %.2f → %.2f (applied)",
                    old_decay, best_decay,
                )

        # Screen impact
        screen_impact = {}
        if screen_closed_errors and screen_open_errors:
            mean_closed = sum(screen_closed_errors) / len(screen_closed_errors)
            mean_open = sum(screen_open_errors) / len(screen_open_errors)
            screen_impact = {
                "mean_error_screens_closed": round(mean_closed, 4),
                "mean_error_screens_open": round(mean_open, 4),
                "qualifying_hours_closed": len(screen_closed_errors),
                "qualifying_hours_open": len(screen_open_errors),
            }
            if abs(mean_closed - mean_open) > 0.1:
                global_flags.append("screen_drift_detected")

        # Temporal bias (global)
        all_morning = [acc["morning_delta"] / acc["morning_n"] for acc in unit_accum.values() if acc["morning_n"] > 5]
        all_afternoon = [acc["afternoon_delta"] / acc["afternoon_n"] for acc in unit_accum.values() if acc["afternoon_n"] > 5]

        # Hour-of-day curve
        hour_curve = {}
        for h in range(6, 19):
            vals = hourly_residuals[h]
            if vals:
                hour_curve[str(h)] = round(sum(vals) / len(vals), 4)

        # Context block (#826 validation).  Surfaces the values currently
        # applied so a remote analyser can tie a diagnose payload to the
        # installation's geography and configuration without needing separate
        # entity inspection.
        from .const import (
            DEFAULT_SOLAR_MIN_TRANSMITTANCE as _DEFAULT_FLOOR,
            SCREEN_DIRECT_TRANSMITTANCE as _SCREEN_DIRECT,
        )
        try:
            lat = self.coordinator.hass.config.latitude
            lon = self.coordinator.hass.config.longitude
        except AttributeError:
            lat = lon = None
        context = {
            "latitude": lat,
            "longitude": lon,
            "screen_config": {
                "south": bool(self.coordinator.screen_config[0]),
                "east":  bool(self.coordinator.screen_config[1]),
                "west":  bool(self.coordinator.screen_config[2]),
            },
            "constants": {
                "screen_direct_transmittance": _SCREEN_DIRECT,
                "composite_legacy_floor": _DEFAULT_FLOOR,
                "solar_battery_decay": self.coordinator.solar_battery_decay,
                "solar_azimuth": self.coordinator.solar_azimuth,
            },
            "days_analyzed": days_back,
        }

        # Per-unit min-base thresholds (#871).  Exposes the effective gate
        # each unit sees in NLMS / inequality / shutdown detection so the
        # user can distinguish auto-calibrated values from the global
        # fallback.  ``method`` reports the source; ``effective`` is the
        # value actually applied at the gate sites.
        from .const import (
            SOLAR_LEARNING_MIN_BASE as _GLOBAL_LEARNING_MIN_BASE,
            SOLAR_SHUTDOWN_MIN_BASE as _GLOBAL_SHUTDOWN_MIN_BASE,
        )
        per_unit_thresholds = {}
        for sid in self.coordinator.energy_sensors:
            calibrated = self.coordinator._per_unit_min_base_thresholds.get(sid)
            per_unit_thresholds[sid] = {
                "effective_nlms": round(
                    calibrated if calibrated is not None else _GLOBAL_LEARNING_MIN_BASE,
                    5,
                ),
                "effective_shutdown": round(
                    calibrated if calibrated is not None else _GLOBAL_SHUTDOWN_MIN_BASE,
                    5,
                ),
                "method": "auto" if calibrated is not None else "fallback",
                "calibrated_value": calibrated,
            }

        return {
            "context": context,
            "global": {
                "qualifying_hours": total_qualifying,
                "excluded": excluded,
                "battery_decay_health": battery_health,
                "battery_calibration": calibration,
                "screen_impact": screen_impact,
                "temporal_bias": {
                    "morning_mean_delta": round(sum(all_morning) / len(all_morning), 4) if all_morning else None,
                    "afternoon_mean_delta": round(sum(all_afternoon) / len(all_afternoon), 4) if all_afternoon else None,
                },
                "hour_of_day_residual": hour_curve,
                "flags": global_flags,
                # Inequality-replay diagnostics (#865).  Aggregate counters
                # from the shadow replay used to populate per-unit
                # ``implied_coefficient_inequality`` above.  ``inequality_updates``
                # shows how many (unit, hour) samples passed through the
                # one-sided constraint update; ``inequality_non_binding`` shows
                # how many samples found the constraint already satisfied.
                "inequality_replay": {
                    "updates": shadow_diag.get("inequality_updates", 0),
                    "non_binding": shadow_diag.get("inequality_non_binding", 0),
                    "skipped_low_battery": shadow_diag.get("inequality_skipped_low_battery", 0),
                    "skipped_mode": shadow_diag.get("inequality_skipped_mode", 0),
                    "skipped_base": shadow_diag.get("inequality_skipped_base", 0),
                },
            },
            "per_unit": per_unit,
            "per_unit_thresholds": per_unit_thresholds,
        }

