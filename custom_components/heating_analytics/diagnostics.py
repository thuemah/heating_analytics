"""DiagnosticsEngine — hosts diagnose_model() and diagnose_solar() extracted from coordinator.py.

Thin-delegate pattern: the engine holds a reference to the coordinator
and reaches back for state.  Public methods are called via delegates
on the coordinator so the external API is unchanged.
"""
from __future__ import annotations

import logging
import math
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
        # Pre-insert summary so it renders first in the response — populated
        # at the bottom of this method once all sections are computed.
        # Mutating an existing key preserves its insertion-order position
        # (Python 3.7+ dict semantics), so the final assignment does not
        # move it.
        result["summary"] = {}

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
            if solar_f is None or actual is None or expected is None or solar_f <= 0.01:
                continue
            # Subtract excluded-mode (DHW / OFF / guest) energy from actual
            # before computing the residual — mirrors the base_model_health
            # fix.  Without this, DHW or guest hours that overlap with solar
            # hours bias the correlation positive: ``expected`` is a heating-
            # only prediction (excluded modes are not in the learning loop)
            # but ``actual`` is the raw meter sum that includes DHW / guest
            # contributions.  OFF contributes 0 kWh so the subtraction is a
            # no-op for OFF — which is correct (OFF is a stable state, not
            # contamination).
            unit_modes = entry.get("unit_modes", {}) or {}
            unit_breakdown = entry.get("unit_breakdown", {}) or {}
            excluded_kwh = sum(
                kwh for sid, kwh in unit_breakdown.items()
                if unit_modes.get(sid, MODE_HEATING) in MODES_EXCLUDED_FROM_GLOBAL_LEARNING
            )
            adjusted_actual = max(0.0, actual - excluded_kwh)
            solar_errors.append((solar_f, adjusted_actual - expected))

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
            temp_key = entry.get("temp_key")
            wind_bucket = entry.get("wind_bucket")
            if temp_key is None or wind_bucket is None or entry.get("actual_kwh") is None:
                continue
            ref_kwh, used_track_c = _reference_dark_kwh(entry)
            # Subtract excluded-mode (DHW / OFF / guest) energy rather than
            # dropping the whole hour — mirrors retrain.py:401-408.  An OFF
            # unit contributes 0 kWh so the subtraction is a no-op for it
            # (which is correct: a permanently-off auxiliary VP is a stable
            # state, not contamination).  For DHW / guest the unit's actual
            # contribution is removed so the residual matches the
            # heating-only semantic of the stored bucket.  Track-C's
            # synthetic + non_mpc_kwh path inherits the same correction:
            # any non-MPC unit in an excluded mode has its kWh subtracted.
            unit_modes = entry.get("unit_modes", {}) or {}
            unit_breakdown = entry.get("unit_breakdown", {}) or {}
            excluded_kwh = sum(
                kwh for sid, kwh in unit_breakdown.items()
                if unit_modes.get(sid, MODE_HEATING) in MODES_EXCLUDED_FROM_GLOBAL_LEARNING
            )
            ref_kwh = max(0.0, ref_kwh - excluded_kwh)
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

        # --- 0. Summary (top-of-response, populated last) ---
        # Aggregates the headline numbers from each section so callers can
        # read a one-screen verdict before drilling into per-bucket detail.
        # The summary's "verdict" reflects model health only — it ignores
        # mode_contamination (informational) and bucket_population (the
        # underlying bucket counts surface elsewhere).  Noise floor for
        # monotonicity inversions is 0.05 kWh — sub-noise inversions near
        # the balance point (where stored kWh ≈ 0) are excluded from the
        # verdict driver but still counted in ``inversion_count``.
        MONOTONICITY_NOISE_FLOOR_KWH = 0.05
        SOLAR_CORRELATION_NEUTRAL_BAND = 0.15

        total_inv = 0
        total_inv_above_noise = 0
        max_delta = 0.0
        for wb_diag in monotonicity.values():
            for inv in wb_diag.get("inversions", []):
                total_inv += 1
                d = float(inv.get("delta", 0.0))
                if d > max_delta:
                    max_delta = d
                if d > MONOTONICITY_NOISE_FLOOR_KWH:
                    total_inv_above_noise += 1

        base_summary = result["base_model_health"]["summary"]
        total_buckets = (
            base_summary["inflated_count"]
            + base_summary["deflated_count"]
            + base_summary["unverifiable_count"]
            + sum(
                1
                for buckets in base_health.values()
                for entry_out in buckets.values()
                if entry_out.get("verdict") == "ok"
            )
        )
        ok_count = total_buckets - (
            base_summary["inflated_count"]
            + base_summary["deflated_count"]
            + base_summary["unverifiable_count"]
        )
        verifiable_count = total_buckets - base_summary["unverifiable_count"]

        solar_corr = solar_diag.get("correlation_solar_vs_error")
        solar_qual_hours = solar_diag.get("qualifying_hours", 0)

        if total_buckets > 0 and base_summary["unverifiable_count"] / total_buckets > 0.5:
            verdict = "model_unverifiable"
        elif (
            total_inv_above_noise > 0
            or base_summary["inflated_count"] > 0
            or base_summary["deflated_count"] > 0
            or (solar_corr is not None and abs(solar_corr) > SOLAR_CORRELATION_NEUTRAL_BAND)
        ):
            verdict = "issues_found"
        else:
            verdict = "ok"

        result["summary"] = {
            "verdict": verdict,
            "data_window": {
                "days": days_back,
                "hours_analyzed": result["mode_contamination"]["total_hours_analyzed"],
            },
            "monotonicity": {
                "inversion_count": total_inv,
                "inversion_count_above_noise": total_inv_above_noise,
                "max_delta_kwh": round(max_delta, 4),
                "noise_floor_kwh": MONOTONICITY_NOISE_FLOOR_KWH,
            },
            "base_model": {
                "total_buckets": total_buckets,
                "verifiable": verifiable_count,
                "ok": ok_count,
                "inflated": base_summary["inflated_count"],
                "deflated": base_summary["deflated_count"],
                "unverifiable": base_summary["unverifiable_count"],
            },
            "mode_contamination": {
                "contaminated_days": result["mode_contamination"]["contaminated_days"],
                "total_excluded_kwh": result["mode_contamination"]["total_excluded_kwh"],
            },
            "solar": {
                "qualifying_hours": solar_qual_hours,
                "correlation": solar_corr,
                "interpretation": solar_diag.get("interpretation"),
            },
        }

        return result

    def _format_last_batch_fit(self, entity_id: str) -> dict | None:
        """Format the most recent batch-fit-solar summary for diagnose_solar (#884).

        Returns ``None`` when the unit has never been included in any
        batch-fit run.  Persisted across HA restarts via the standard
        save path (``last_batch_fit_per_unit`` in storage).  When the
        most recent run was a top-level skip for this entity (e.g.
        ``weighted_smear_excluded``), the entry includes a ``skip_reason``
        field with empty ``regimes`` — distinguishing "skipped, here's why"
        from "never run".  Successful fits expose timestamp + per-(regime)
        sample count, residual RMSE, before/after coefficients, damping
        applied, and any per-regime skip reason.
        """
        last_per_unit = getattr(self.coordinator, "_last_batch_fit_per_unit", None)
        if not isinstance(last_per_unit, dict):
            return None
        entry = last_per_unit.get(entity_id)
        if not isinstance(entry, dict):
            return None
        out: dict = {
            "timestamp": entry.get("timestamp"),
            "regimes": entry.get("regimes") or {},
        }
        if "skip_reason" in entry:
            out["skip_reason"] = entry["skip_reason"]
        return out


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
            HARD_OUTLIER_SANITY_MULTIPLIER,
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
        excluded = {"aux": 0, "guest": 0, "saturated": 0, "low_vector": 0, "no_base": 0, "legacy": 0, "outlier": 0}
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

        # Battery thermal-feedback sweep (#896): chronological tuples for
        # replay of the EMA under candidate k values.  Data shape mirrors the
        # production EMA input plus enough metadata to stratify residuals by
        # (hour-of-day × temp-regime × screen-position).  Population happens
        # outside the per-entity inner loop (one tuple per hour, not one per
        # entity), so it lives directly in the entry-level pass below.
        # Each tuple:
        #   (solar_impact_raw, solar_wasted, actual, expected,
        #    has_heating_unit, hour_bucket, temp_bucket, screen_bucket)
        sweep_tuples: list[tuple] = []

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

            # Collect day sequences for joint (decay, k) calibration sweep.
            # Per-hour wasted is needed alongside raw_solar so the carryover
            # battery EMA can be replayed under each k candidate without
            # re-iterating the hourly_log.  Read the same wasted field the
            # k-only sweep uses (preferring the heating-gated field, falling
            # back to the legacy aggregate).
            day_key = ts[:10]
            actual_total = entry.get("actual_kwh", 0.0)
            expected_total = entry.get("expected_kwh", 0.0)
            day_wasted = entry.get(
                "solar_heating_wasted_kwh",
                entry.get("solar_wasted_kwh", 0.0),
            )
            if day_key not in day_sequences:
                day_sequences[day_key] = []
            day_sequences[day_key].append(
                (hour, solar_impact_raw, day_wasted, actual_total, expected_total)
            )

            # Battery thermal-feedback sweep (#896).  Build chronological
            # tuples matching the production EMA input shape; stratification
            # happens at residual-bucketing time post-loop.
            #
            # Wasted source: prefer ``solar_heating_wasted_kwh`` (heating-
            # gated at write time, #896) and fall back to the total
            # ``solar_wasted_kwh`` for legacy entries written before that
            # field existed.  Today's saturation logic returns wasted=0 for
            # cooling/OFF/DHW, so the legacy aggregate is structurally
            # heating-only — but we prefer the explicit field when present.
            #
            # Heating-active gate mirrors the live EMA gate: the unit_modes
            # log entry only stores non-heating modes (heating is the default
            # to reduce log clutter, see HourlyProcessor.process), so missing
            # entity → MODE_HEATING.  ``has_heating_unit`` is True when at
            # least one configured energy sensor was in heating-regime that
            # hour — an absent unit_modes block (legacy or pre-#838 log)
            # falls back to True since heating is the default.
            sweep_solar_wasted = entry.get(
                "solar_heating_wasted_kwh",
                entry.get("solar_wasted_kwh", 0.0),
            )
            if sweep_solar_wasted > 0.0:
                # Cooling/OFF/DHW saturation returns wasted=0; positive wasted
                # is a structural witness of at least one heating unit having
                # contributed to the aggregate.  Avoids depending on the
                # filtered unit_modes log (which can omit explicit-heating).
                has_heating_unit = True
            elif unit_modes:
                stored_heating = any(
                    m in (MODE_HEATING, MODE_GUEST_HEATING)
                    for m in unit_modes.values()
                )
                stored_non_default = bool(unit_modes)
                # If the stored map omits some sensors entirely, those
                # sensors were in MODE_HEATING (the default-omitted regime).
                missing_count = sum(
                    1 for sid in self.coordinator.energy_sensors if sid not in unit_modes
                )
                has_heating_unit = stored_heating or missing_count > 0
                # If no sensors are configured, treat as no heating active.
                if not self.coordinator.energy_sensors and not stored_non_default:
                    has_heating_unit = False
            else:
                has_heating_unit = bool(self.coordinator.energy_sensors)

            # Hour-of-day bucket: morning captures the issue's reported
            # symptom (transition-regime over-prediction at ~7-10 °C, mid
            # morning).  Night included for completeness — the EMA carries
            # state across midnight and post-sunset error is the diagnostic
            # signal.
            if 6 <= hour < 11:
                hour_bucket = "morning"
            elif 11 <= hour < 15:
                hour_bucket = "midday"
            elif 15 <= hour < 22:
                hour_bucket = "afternoon"
            else:
                hour_bucket = "night"

            # Temp regime: 4 buckets including a ``transition`` zone
            # (BP-2 ≤ T ≤ BP+2).  Issue #896's headline symptom — sunny
            # mornings at 7-10 °C with BP=15 — actually lies inside
            # ``heating_mild`` ([7, 13)) for typical Norwegian BP=15,
            # but for higher-BP installs (older buildings, BP≈17-18) the
            # symptom hours can drift up into the BP±2 window.  Keeping
            # ``transition`` as its own bucket ensures that cell is
            # always visible in ``per_cell_at_optimum``; collapsing it
            # into None (the previous behaviour) hid the headline
            # symptom from the user-facing diagnostic for high-BP
            # installs.  The other diagnose_solar.temperature_stratified
            # block still drops transition for its own purpose
            # (mode-flip noise on the COP-blindness check); the sweep
            # has different needs and stratifies independently.
            if temp_entry is not None:
                bp = self.coordinator.balance_point
                if temp_entry < bp - 8.0:
                    temp_bucket = "heating_deep"
                elif temp_entry < bp - 2.0:
                    temp_bucket = "heating_mild"
                elif temp_entry <= bp + 2.0:
                    temp_bucket = "transition"
                else:
                    temp_bucket = "cooling"
            else:
                temp_bucket = None

            # Screen-position bucket: matches per-unit screen_stratified.
            if correction is None:
                screen_bucket = "open"
            elif correction >= 80.0:
                screen_bucket = "open"
            elif correction >= 40.0:
                screen_bucket = "mid"
            else:
                screen_bucket = "closed"

            sweep_tuples.append((
                solar_impact_raw, sweep_solar_wasted, actual_total, expected_total,
                has_heating_unit, hour_bucket, temp_bucket, screen_bucket,
            ))

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

                # Prior-free sanity check (#919 Part 5)
                # Mirrors the match_diagnose=True path in _collect_batch_fit_samples
                if base_unit > 0 and abs(actual_unit - base_unit) > HARD_OUTLIER_SANITY_MULTIPLIER * base_unit:
                    excluded["outlier"] += 1
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
                    self.coordinator.screen_config_for_entity(entity_id),
                )
                unit_coeff = self.coordinator.solar.calculate_unit_coefficient(
                    entity_id, entry.get("temp_key", "10"), mode
                )
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
            # #868: report both regimes separately.  ``current_coefficient``
            # remains the prediction-time view (heating regime + default
            # fallback) for backwards compatibility with consumers that
            # haven't migrated.  The split-aware fields and
            # ``coefficient_split_delta_pct`` read raw storage instead —
            # an unlearned regime must show ``{0,0,0}``, not a default
            # decomposition.  Otherwise the validation criterion ("split
            # delta > N means the split captures real physics") is
            # muddled by every heating-only install reporting a small
            # divergence purely from the cooling default.
            current = self.coordinator.solar.calculate_unit_coefficient(
                entity_id, "10", MODE_HEATING
            )
            stored_entry = (
                self.coordinator.model.solar_coefficients_per_unit.get(
                    entity_id, {}
                )
            )
            if not isinstance(stored_entry, dict):
                stored_entry = {}
            current_heating = stored_entry.get("heating") or {
                "s": 0.0, "e": 0.0, "w": 0.0
            }
            current_cooling = stored_entry.get("cooling") or {
                "s": 0.0, "e": 0.0, "w": 0.0
            }

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
                    current_correction, self.coordinator.screen_config_for_entity(entity_id)
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
                    # Trimmed (#896 follow-up): only ``n`` and
                    # ``mean_delta_kwh`` are actionable.  ``mean_modeled_kwh``
                    # and ``mean_implied_kwh`` are reconstructable from the
                    # current coefficient + log breakdown if needed and
                    # added ~3× the bytes per bucket on every diagnose
                    # response.
                    screen_stratified[bkey] = {
                        "n": b["n"],
                        "mean_delta_kwh": round(b["delta_sum"] / b["n"], 4),
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
                cfg = self.coordinator.screen_config_for_entity(entity_id)
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
                    # Trim (#896 follow-up): if every candidate produces
                    # essentially the same coefficient and RMSE, the sweep
                    # is uninformative — emit only the ``best`` row plus a
                    # ``verdict`` flag.  Saves 6 nearly-identical rows per
                    # entity on installs with low solar signal (small
                    # non-VP loads where implied coefficient is near zero
                    # so the candidate floor cannot move the fit).  Tests
                    # that need the full ``candidates`` list use a known-
                    # transmittance generative log where the sweep is
                    # genuinely informative.
                    rmses = [r["residual_rmse_kwh"] for r in results]
                    rmse_uniform = (max(rmses) - min(rmses)) < 0.01
                    first = results[0]["implied_coefficient"]
                    coeff_uniform = all(
                        abs(r["implied_coefficient"]["s"] - first["s"]) < 1e-3
                        and abs(r["implied_coefficient"]["e"] - first["e"]) < 1e-3
                        and abs(r["implied_coefficient"]["w"] - first["w"]) < 1e-3
                        for r in results
                    )
                    sensitivity = {
                        "n_hours": len(tuples),
                        "correction_range_pct": round(corr_var, 1),
                        "informative": corr_var >= 40.0,  # ≥40 pct points of slider variance
                        "best": best,
                    }
                    if rmse_uniform and coeff_uniform:
                        sensitivity["verdict"] = "uniform_across_candidates"
                    else:
                        sensitivity["candidates"] = results
                    if (
                        sensitivity["informative"]
                        and abs(best["screen_direct_transmittance"] - 0.08) > 0.04
                    ):
                        flags.append("sensitivity_suggests_transmittance_retune")

            # Inequality-replay coefficient (#865) — what the learner would
            # produce if retrained from zero over the same window.  Absence
            # means the unit did not qualify for any update (no shutdown
            # hours, or base below SOLAR_SHUTDOWN_MIN_BASE everywhere).
            # Mode-stratified per #868 — replay writes per-regime; inequality
            # is heating-only by #865 design, so we report the heating regime
            # of the shadow output.  ``None`` means no inequality update fired.
            shadow_entry = shadow_coeffs.get(entity_id)
            implied_inequality_coeff = None
            if isinstance(shadow_entry, dict):
                heating_shadow = shadow_entry.get("heating")
                if isinstance(heating_shadow, dict) and any(
                    heating_shadow.get(k) for k in ("s", "e", "w")
                ):
                    implied_inequality_coeff = {
                        k: round(v, 4) for k, v in heating_shadow.items()
                    }

            # Mode-stratified split (#868): scalar percentage divergence
            # between the heating and cooling regimes, averaged over the
            # three directions.  Stable across regime swaps because we
            # take absolute differences and normalise by the symmetric
            # mean.  ``None`` when both regimes are zero (cooling never
            # learned on a heating-only install — the seeded copy from
            # migration drifts away as cooling-mode hours arrive).
            h_dir = current_heating
            c_dir = current_cooling
            denom = sum(
                abs(h_dir.get(k, 0.0)) + abs(c_dir.get(k, 0.0))
                for k in ("s", "e", "w")
            )
            if denom > 0.001:
                split_delta_pct = (
                    100.0
                    * sum(
                        abs(h_dir.get(k, 0.0) - c_dir.get(k, 0.0))
                        for k in ("s", "e", "w")
                    )
                    / denom
                )
                coefficient_split_delta = round(split_delta_pct, 1)
            else:
                coefficient_split_delta = None

            # Tobit MLE (#904 stage 0+1, shadow-only).  Censoring-aware
            # estimator surfaced alongside ``implied_coefficient_30d``
            # (which drops saturated rows) and ``implied_coefficient_inequality``
            # (which lower-bounds via shutdown).  Modulating-regime fit
            # only — shutdown rows excluded per CHOICE 3, saturated
            # rows kept as right-censored at ``T = 0.95×base`` per
            # CHOICE 2.  No production wiring; informational diagnostic
            # for stage-1 evidence collection.  Heating regime only at
            # this stage (mirrors the existing implied_30d display
            # convention; cooling Tobit deferred until the heating
            # path validates).
            try:
                tobit_fit = self.coordinator.learning.compute_tobit_for_diagnose(
                    self.coordinator._hourly_log,
                    entity_id,
                    "heating",
                    self.coordinator,
                    unit_min_base=self.coordinator._per_unit_min_base_thresholds or None,
                    days_back=days_back,
                )
            except Exception:  # noqa: BLE001
                # Defensive: shadow diagnostic must never break diagnose_solar.
                tobit_fit = {
                    "coefficient": None,
                    "skip_reason": "exception",
                }
            tobit_coeff = tobit_fit.get("coefficient")
            tobit_diagnostics = {
                "iterations": tobit_fit.get("iterations", 0),
                "converged": tobit_fit.get("converged", False),
                "failure_reason": tobit_fit.get("failure_reason"),
                "sigma": tobit_fit.get("sigma"),
                "log_likelihood": tobit_fit.get("log_likelihood"),
                "n_uncensored": tobit_fit.get("n_uncensored", 0),
                "n_censored": tobit_fit.get("n_censored", 0),
                "censored_fraction": tobit_fit.get("censored_fraction", 0.0),
                "n_eff": tobit_fit.get("n_eff", 0.0),
                "skip_reason": tobit_fit.get("skip_reason"),
            }

            # Live Tobit-learner state (#904 stage 3).  Surfaces the
            # running sufficient-statistic snapshot for the maintainer's
            # validation window — without this there's no per-hour
            # observability into the live learner's convergence.
            # Always emitted (even when the master flag is off) so
            # callers walking the per_unit dict don't need conditional
            # presence checks.  The ``enabled`` and ``allow_listed``
            # fields disambiguate active vs dormant state.
            from .const import SOLAR_MODEL_VERSION as _CURRENT_SOLAR_VERSION
            tobit_live_stats = (
                self.coordinator._tobit_sufficient_stats.get(entity_id, {})
                if isinstance(getattr(self.coordinator, "_tobit_sufficient_stats", None), dict)
                else {}
            )
            shadow_entry = (
                self.coordinator._nlms_shadow_coefficients.get(entity_id)
                if isinstance(getattr(self.coordinator, "_nlms_shadow_coefficients", None), dict)
                else None
            )

            def _build_regime_block(regime_name: str) -> dict:
                slot = tobit_live_stats.get(regime_name) or {}
                samples = slot.get("samples", [])
                n_unc = sum(1 for s in samples if not s[4])
                n_cens = sum(1 for s in samples if s[4])
                last_step = slot.get("last_step", {})
                shadow_regime = (
                    shadow_entry.get(regime_name) if isinstance(shadow_entry, dict) else None
                )
                return {
                    "in_cold_start": (
                        last_step.get("skip_reason") in (
                            "insufficient_uncensored",
                            "insufficient_effective_samples",
                        )
                        or last_step.get("converged") is not True
                    ),
                    "n_uncensored": n_unc,
                    "n_censored": n_cens,
                    "n_eff": last_step.get("n_eff", float(n_unc)),
                    "last_step_iterations": last_step.get("iterations", 0),
                    "last_step_failure_reason": last_step.get("failure_reason"),
                    "last_step_norm": last_step.get("step_norm", 0.0),
                    "last_step_skip_reason": last_step.get("skip_reason"),
                    "sigma": last_step.get("sigma"),
                    "current_coefficient_nlms_shadow": (
                        {k: round(float(shadow_regime.get(k, 0.0)), 4) for k in ("s", "e", "w")}
                        if isinstance(shadow_regime, dict)
                        else None
                    ),
                    "solar_model_version": slot.get(
                        "solar_model_version", _CURRENT_SOLAR_VERSION
                    ),
                    "samples_since_reset": slot.get("samples_since_reset", 0),
                }

            # Both regimes surfaced (review I4, #912).  Cooling-active
            # entities and dual-mode VPs need observability into both
            # slots; previously only heating was reported and cooling
            # was invisible.  Top-level ``enabled`` / scope state apply
            # to both regimes equally so they live at the parent.
            #
            # Scope semantics (1.3.5+ default-on):
            # ``in_scope_override`` reflects whether the entity is in
            # the optional scope-restriction list (non-empty list =
            # only listed entities try Tobit).  ``tobit_routed_to_live``
            # reflects whether Tobit is actually running for this
            # entity at this hour given all gates: master flag enabled
            # AND (auto-mode OR in scope) AND not MPC-managed.  A user
            # walking ``per_unit[entity_id].live_tobit_state`` should
            # consult ``tobit_routed_to_live`` for "is Tobit on for me",
            # not the legacy ``allow_listed`` field.
            tobit_flag = bool(getattr(
                self.coordinator, "_experimental_tobit_live_learner", False
            ))
            scope_list = getattr(
                self.coordinator, "_tobit_live_entities", frozenset()
            )
            mpc_managed = frozenset(
                eid
                for eid, strat in (
                    getattr(self.coordinator, "_unit_strategies", {}) or {}
                ).items()
                if (
                    strat is not None
                    and strat.__class__.__name__ == "WeightedSmear"
                    and getattr(strat, "use_synthetic", False)
                )
            )
            in_scope = (not scope_list) or (entity_id in scope_list)
            tobit_routed = (
                tobit_flag and in_scope and entity_id not in mpc_managed
            )
            live_tobit_state = {
                "enabled": tobit_flag,
                "scope_mode": "auto" if not scope_list else "override",
                "in_scope_override": (
                    entity_id in scope_list if scope_list else None
                ),
                "tobit_routed_to_live": tobit_routed,
                # Legacy field name preserved for backward compatibility
                # with consumers that walked ``allow_listed`` under the
                # pre-1.3.5 opt-in semantic — now reports the effective
                # routed state.  New consumers should use
                # ``tobit_routed_to_live`` directly.
                "allow_listed": tobit_routed,
                "heating": _build_regime_block("heating"),
                "cooling": _build_regime_block("cooling"),
            }

            # Inactive-unit collapse (#896 follow-up).  Sensors with no
            # learned coefficient, no saturation, no shutdown signal, and
            # no flags carry no actionable information in the verbose
            # blocks (transmittance_sensitivity is structurally uniform
            # because the implied coefficient is ~0 so candidate floors
            # cannot move the fit; screen_stratified and stability_windows
            # are noise around zero; temporal_bias adds nothing the global
            # block doesn't already report).  Emit a minimal record so the
            # entity stays addressable via ``per_unit[entity_id]`` for any
            # consumer that walks the dict, but drop the verbose tail that
            # was bloating diagnose_solar output by ~70 % on installs with
            # many small non-VP energy sensors.  Backward-compatible: every
            # field that existing tests assert on for zero-coeff fixtures
            # (current_coefficient_*, coefficient_split_delta_pct,
            # implied_coefficient_30d, qualifying_hours, mean_delta_kwh,
            # flags) is preserved.
            heating_zero = all(abs(v) < 1e-6 for v in current_heating.values())
            cooling_zero = all(abs(v) < 1e-6 for v in current_cooling.values())
            is_inactive = (
                heating_zero
                and cooling_zero
                and acc["saturated"] == 0
                and acc["shutdown_hours"] == 0
                and not flags
            )
            if is_inactive:
                per_unit[entity_id] = {
                    "inactive": True,
                    "current_coefficient": {k: round(v, 4) for k, v in current.items()},
                    "current_coefficient_heating": {
                        k: round(v, 4) for k, v in current_heating.items()
                    },
                    "current_coefficient_cooling": {
                        k: round(v, 4) for k, v in current_cooling.items()
                    },
                    "coefficient_split_delta_pct": coefficient_split_delta,
                    "implied_coefficient_30d": implied_30d,
                    "qualifying_hours": acc["n"],
                    "mean_delta_kwh": round(mean_delta, 4),
                    # Stage 3 (#912) review I5: live_tobit_state must
                    # appear on the inactive branch too.  An entity
                    # that's allow-listed but has no qualifying hours
                    # collapses here, and ``allow_listed`` /
                    # ``enabled`` are the only place to confirm the
                    # gate is actually active for it.  Hides the
                    # verbose per-regime detail (sample lists are
                    # always empty on inactive units) but keeps the
                    # gate-status fields.
                    "live_tobit_state": {
                        "enabled": live_tobit_state["enabled"],
                        "allow_listed": live_tobit_state["allow_listed"],
                        "heating": {
                            "n_uncensored": live_tobit_state["heating"]["n_uncensored"],
                            "n_censored": live_tobit_state["heating"]["n_censored"],
                            "samples_since_reset": live_tobit_state["heating"]["samples_since_reset"],
                        },
                        "cooling": {
                            "n_uncensored": live_tobit_state["cooling"]["n_uncensored"],
                            "n_censored": live_tobit_state["cooling"]["n_censored"],
                            "samples_since_reset": live_tobit_state["cooling"]["samples_since_reset"],
                        },
                    },
                    "flags": flags,
                }
            else:
                per_unit[entity_id] = {
                    "current_coefficient": {k: round(v, 4) for k, v in current.items()},
                    "current_coefficient_heating": {
                        k: round(v, 4) for k, v in current_heating.items()
                    },
                    "current_coefficient_cooling": {
                        k: round(v, 4) for k, v in current_cooling.items()
                    },
                    "coefficient_split_delta_pct": coefficient_split_delta,
                    "implied_coefficient_30d": implied_30d,
                    "implied_coefficient_30d_no_shutdown": implied_no_shutdown,
                    "implied_coefficient_inequality": implied_inequality_coeff,
                    "implied_coefficient_physical": implied_physical,
                    "implied_coefficient_tobit_30d": tobit_coeff,
                    "tobit_diagnostics": tobit_diagnostics,
                    "live_tobit_state": live_tobit_state,
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
                    "last_batch_fit": self._format_last_batch_fit(entity_id),
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

        # Joint (decay, k) battery calibration with counterfactual residuals.
        #
        # Replaces the prior 1-D decay sweep, which had three statistical
        # defects (#902 statistics review):
        #
        #   1. The estimator was biased toward the live decay: ``actual −
        #      expected`` is the residual against the LIVE model, not a
        #      counterfactual residual under the candidate decay.  Hours
        #      where the live model already credited enough battery were
        #      filtered out before the candidate saw them, biasing the
        #      recommendation toward status quo.
        #   2. Mean-residual minimisation is the wrong loss — a candidate
        #      that systematically over-credits some hours and under-credits
        #      others equally can score zero mean while tracking the data
        #      poorly.  Use RMSE.
        #   3. ``decay`` and ``k`` are jointly unidentified from post-sunset
        #      residuals.  Sequential calibration converges to order-
        #      dependent local optima — both must be swept jointly.
        #
        # Method: for each (decay_alt, k_alt) candidate, replay BOTH batteries
        # (main solar EMA + carryover EMA) starting from 0 over each day's
        # hours, AND replay the live (decay, k) the same way.  The difference
        # in release between the two replays is the counterfactual delta;
        # adding it to the live residual yields the residual the system
        # would have produced under the candidate.  Score by RMSE on
        # post-sunset hours.
        #
        # Post-sunset definition: per day, the ``POST_SUNSET_REPLAY_HOURS``
        # hours immediately after the last hour with raw_solar > 0.  Uses
        # the raw signal (``solar_impact_raw_kwh``) — the post-coefficient
        # × raw-vector value, which is 0 by construction when the sun is
        # below the horizon — NOT the post-battery ``solar_factor`` which
        # carries battery residue across midnight.  Restricting to the
        # window where the battery is observably charged improves SNR
        # (pre-dawn hours have battery ≈ 0 and tell us nothing about
        # decay).
        #
        # Initial-state caveat: from-0 daily replay underestimates the
        # live system's actual battery state on the morning of each day
        # by an exponentially decaying residual from yesterday's evening.
        # After ~3 half-lives (~9 h at decay 0.80) the bias is < 12 %.
        # POST_SUNSET_REPLAY_HOURS = 6 captures the 1-2 half-life window
        # where signal-to-noise is highest; the from-0 approximation is
        # acceptable here because the post-sunset evening is far past the
        # morning when the bias was largest.
        DECAY_GRID = [round(0.50 + 0.05 * i, 2) for i in range(10)]   # 0.50..0.95
        K_GRID = [round(0.1 * i, 1) for i in range(11)]               # 0.0..1.0
        POST_SUNSET_REPLAY_HOURS = 6
        MIN_POST_SUNSET_HOURS_FOR_RECOMMENDATION = 5

        calibration: dict = {}
        if day_sequences:
            decay_live = self.coordinator.solar_battery_decay
            k_live = self.coordinator.battery_thermal_feedback_k

            # Build per-day post-sunset hour set: the N hours immediately
            # after the last hour with raw_solar > 0.01.  Days with no
            # qualifying sunny hour (e.g. fully overcast) contribute nothing.
            post_sunset_set_by_day: dict[str, set[int]] = {}
            # Build per-day morning hour set: from the first hour with
            # raw_solar > 0.01 through the hour where raw_solar peaks
            # (inclusive on both ends).  This is the rising-sun phase
            # where charge-side dynamics differ between models — instant-
            # respons would credit fast, EMA accumulates slowly.  Plateau
            # hours past the peak are deliberately excluded: both models
            # converge to steady state there, so they carry no
            # discriminating information about decay vs instant credit.
            #
            # The two windows together separate the two physical regimes
            # the battery model captures: post-sunset = pure decay tail
            # (current model fits this); morning = charge ramp (the
            # asymmetric-charge gap from #896's deferred scope).  The
            # tail/morning RMSE disagreement at any single (decay, k) is
            # the diagnostic that confirms whether asymmetric handling is
            # needed (large gap → yes) or not (small gap → no, single-
            # decay model is sufficient).
            morning_set_by_day: dict[str, set[int]] = {}
            for day_key, hours in day_sequences.items():
                last_sunny_h = -1
                first_sunny_h = -1
                peak_solar = 0.0
                peak_hour = -1
                for h, raw, _wasted, _act, _exp in hours:
                    if raw > 0.01:
                        if first_sunny_h < 0:
                            first_sunny_h = h
                        if h > last_sunny_h:
                            last_sunny_h = h
                        if raw > peak_solar:
                            peak_solar = raw
                            peak_hour = h
                if last_sunny_h >= 0:
                    post_sunset_set_by_day[day_key] = {
                        last_sunny_h + i for i in range(1, POST_SUNSET_REPLAY_HOURS + 1)
                    }
                if first_sunny_h >= 0 and peak_hour >= first_sunny_h:
                    morning_set_by_day[day_key] = set(
                        range(first_sunny_h, peak_hour + 1)
                    )

            def _replay_score(
                decay_alt: float,
                k_alt: float,
                window_by_day: dict[str, set[int]],
            ) -> tuple[float, int]:
                """Counterfactual replay scored over ``window_by_day``.

                Returns (rmse, n_hours_evaluated).  Window-agnostic — the
                replay recurrence and counterfactual residual formula are
                the same regardless of which hours feed into the SSE.
                """
                sse = 0.0
                n = 0
                for day_key, hours in day_sequences.items():
                    window = window_by_day.get(day_key)
                    if not window:
                        continue
                    hours_sorted = sorted(hours, key=lambda x: x[0])
                    main_alt = main_live = 0.0
                    carry_alt = carry_live = 0.0
                    for h, raw, wasted, actual, expected in hours_sorted:
                        # Live replay
                        main_live = main_live * decay_live + raw * (1 - decay_live)
                        live_carry_in = (
                            k_live * wasted if k_live > 0.0 else 0.0
                        )
                        carry_live = (
                            carry_live * decay_live + live_carry_in * (1 - decay_live)
                        )
                        live_release = (
                            main_live + k_live * carry_live * (1 - decay_live)
                        )
                        # Alt replay
                        main_alt = main_alt * decay_alt + raw * (1 - decay_alt)
                        alt_carry_in = k_alt * wasted if k_alt > 0.0 else 0.0
                        carry_alt = (
                            carry_alt * decay_alt + alt_carry_in * (1 - decay_alt)
                        )
                        alt_release = (
                            main_alt + k_alt * carry_alt * (1 - decay_alt)
                        )
                        if h in window:
                            # Counterfactual derivation:
                            #   base[t]      = expected[t] + live_release[t]
                            #   expected_alt = base[t] − alt_release[t]
                            #               = expected + (live_release − alt_release)
                            #   residual_alt = actual − expected_alt
                            #               = (actual − expected) + (alt_release − live_release)
                            # Minimised at alt = truth where alt_release matches the
                            # release that produced ``actual``.
                            residual_alt = (actual - expected) + (alt_release - live_release)
                            sse += residual_alt * residual_alt
                            n += 1
                rmse = (sse / n) ** 0.5 if n > 0 else float("inf")
                return rmse, n

            # Post-sunset surface — the original tail-decay scoring.
            # Recommendation is driven by this surface (the live battery
            # model is parameterised for tail behaviour; morning is read-
            # only diagnostic until asymmetric-charge support lands).
            surface: dict[str, float] = {}
            best = (decay_live, k_live)
            best_rmse = float("inf")
            n_post_sunset = 0
            for d_alt in DECAY_GRID:
                for k_alt in K_GRID:
                    rmse, n_post_sunset = _replay_score(
                        d_alt, k_alt, post_sunset_set_by_day
                    )
                    if n_post_sunset < MIN_POST_SUNSET_HOURS_FOR_RECOMMENDATION:
                        continue
                    surface[f"{d_alt},{k_alt}"] = round(rmse, 4)
                    if rmse < best_rmse - 1e-6:
                        best_rmse = rmse
                        best = (d_alt, k_alt)

            # Morning surface — read-only diagnostic.  Same grid + same
            # counterfactual, but scored over the rising-sun window per
            # day.  Reveals whether any (decay, k) candidate would also
            # fit charge-side behaviour, or whether tail-best and morning-
            # best diverge (= asymmetric-charge gap).
            morning_surface: dict[str, float] = {}
            morning_best = (decay_live, k_live)
            morning_best_rmse = float("inf")
            n_morning = 0
            for d_alt in DECAY_GRID:
                for k_alt in K_GRID:
                    rmse, n_morning = _replay_score(
                        d_alt, k_alt, morning_set_by_day
                    )
                    if n_morning < MIN_POST_SUNSET_HOURS_FOR_RECOMMENDATION:
                        continue
                    morning_surface[f"{d_alt},{k_alt}"] = round(rmse, 4)
                    if rmse < morning_best_rmse - 1e-6:
                        morning_best_rmse = rmse
                        morning_best = (d_alt, k_alt)

            # Live config's own RMSE on the same post-sunset hours, computed
            # via the same replay path so the comparison is apples-to-apples.
            # When the live (decay_live, k_live) sits inside the swept grids,
            # this equals surface[f"{decay_live},{k_live}"]; computing it
            # explicitly handles the case where live values fall between
            # grid points (e.g. decay 0.82).
            live_rmse, _ = _replay_score(
                decay_live, k_live, post_sunset_set_by_day
            )
            morning_live_rmse, _ = _replay_score(
                decay_live, k_live, morning_set_by_day
            )

            # Tail/morning disagreement at the post-sunset-recommended
            # candidate.  If small (≲ 0.05 kWh) the post-sunset
            # recommendation also fits morning — single-decay model is
            # sufficient.  If large (≳ 0.10 kWh) post-sunset and morning
            # want different decay + k — asymmetric handling is what's
            # left to fix.  Computed as |best_post_sunset_rmse −
            # rmse_at_morning(post_sunset_best)|: how much worse the
            # tail-optimised candidate is on morning RMSE compared to
            # what's achievable on morning.
            morning_at_tail_best = (
                morning_surface.get(f"{best[0]},{best[1]}")
                if best_rmse != float("inf") else None
            )
            tail_morning_disagreement = (
                round(morning_at_tail_best - morning_best_rmse, 4)
                if (
                    morning_at_tail_best is not None
                    and morning_best_rmse != float("inf")
                )
                else None
            )

            calibration = {
                "current_decay": decay_live,
                "current_k": k_live,
                "current_rmse_kwh": round(live_rmse, 4) if live_rmse != float("inf") else None,
                "recommended_decay": best[0],
                "recommended_k": best[1],
                "recommended_rmse_kwh": round(best_rmse, 4) if best_rmse != float("inf") else None,
                "rmse_improvement_kwh": (
                    round(live_rmse - best_rmse, 4)
                    if (live_rmse != float("inf") and best_rmse != float("inf"))
                    else None
                ),
                "rmse_surface": surface,
                "post_sunset_hours_evaluated": n_post_sunset,
                "post_sunset_replay_hours_per_day": POST_SUNSET_REPLAY_HOURS,
                # Morning-window diagnostic block (read-only — does NOT
                # drive recommendation).  Surfaces the asymmetric-charge
                # gap from #896's deferred scope.
                "morning_current_rmse_kwh": (
                    round(morning_live_rmse, 4)
                    if morning_live_rmse != float("inf") else None
                ),
                "morning_recommended_decay": morning_best[0],
                "morning_recommended_k": morning_best[1],
                "morning_recommended_rmse_kwh": (
                    round(morning_best_rmse, 4)
                    if morning_best_rmse != float("inf") else None
                ),
                "morning_rmse_surface": morning_surface,
                "morning_hours_evaluated": n_morning,
                "tail_morning_disagreement_kwh": tail_morning_disagreement,
                "method": "joint_decay_k_counterfactual_replay",
                "loss": "rmse_post_sunset",
            }
            if apply_battery_decay and (
                best[0] != decay_live or best[1] != k_live
            ) and best_rmse != float("inf"):
                old_decay, old_k = decay_live, k_live
                self.coordinator.solar_battery_decay = best[0]
                self.coordinator.battery_thermal_feedback_k = best[1]
                # Persist BOTH to entry.data
                new_data = {
                    **self.coordinator.entry.data,
                    "solar_battery_decay": best[0],
                    "battery_thermal_feedback_k": best[1],
                }
                self.coordinator.hass.config_entries.async_update_entry(
                    self.coordinator.entry, data=new_data
                )
                calibration["applied"] = True
                _LOGGER.info(
                    "Joint battery calibration applied: decay %.2f → %.2f, k %.2f → %.2f",
                    old_decay, best[0], old_k, best[1],
                )

        # Carry-over reservoir feedback sweep (#896 follow-up).  Replays
        # the carryover-state EMA over the window for each k candidate
        # and reports per-cell residual delta vs the live (k=0) baseline.
        #
        # As of split-state implementation, this sweep models the LIVE
        # wiring: ``_solar_carryover_state`` is charged by ``k × wasted``
        # and its release ``state × (1 − decay)`` subtracts from
        # heating-mode demand prediction in ``calculate_total_power``.
        # The previous "hypothetical 1:1 wiring" disclaimer is removed —
        # the wiring exists.
        #
        # Counterfactual residual derivation:
        #
        #   residual_live[t] = actual[t] − expected_live[t]   (logged)
        #   residual_alt[t]  = residual_live[t] + Δrelease[t]
        #
        # where Δrelease[t] = (B_kα[t] − B_k0[t]) × (1 − decay).  Both
        # replays start from B=0 over the analysis window; the EMA's
        # initial-condition term is identical between replays and
        # cancels in the difference.  Coefficients are held at their
        # currently-learned values (frozen-coefficient mode); real
        # adoption of k > 0 will see ~2-6 % NLMS coefficient drift over
        # 2-3 weeks of qualifying hours, which this replay does not
        # model.  Use ``empirical_optimum_k`` as a calibration hint;
        # validate against actual prediction RMSE after enabling
        # k > 0 for 2-4 sunny weeks.
        battery_feedback_sweep: dict = {}
        if sweep_tuples:
            decay_for_sweep = self.coordinator.solar_battery_decay
            k_candidates = [round(0.1 * i, 1) for i in range(11)]  # 0.0..1.0

            # Replay battery for each k.  trajectories[k] is a list of
            # battery states aligned 1:1 with sweep_tuples.
            trajectories: dict[float, list[float]] = {}
            for k_cand in k_candidates:
                B = 0.0
                trace: list[float] = []
                for (impact_raw, wasted, _act, _exp, heating_active,
                     _hb, _tb, _sb) in sweep_tuples:
                    feedback = (k_cand * wasted) if (k_cand > 0.0 and heating_active) else 0.0
                    B = B * decay_for_sweep + (impact_raw + feedback) * (1 - decay_for_sweep)
                    trace.append(B)
                trajectories[k_cand] = trace

            baseline_trace = trajectories[0.0]

            # Per-cell residuals.  Cells dropped when they lack a temp
            # bucket (transition zone) — kept in global aggregate via the
            # ``global`` cell key so the user still sees an overall RMSE.
            per_k_results: dict[str, dict] = {}
            for k_cand in k_candidates:
                k_trace = trajectories[k_cand]
                cell_residuals: dict[tuple, list[float]] = {}
                global_residuals: list[float] = []
                for idx, (
                    _impact_raw, _wasted, actual, expected,
                    _heating, hour_bucket, temp_bucket, screen_bucket,
                ) in enumerate(sweep_tuples):
                    # Convert state-trajectory delta to release-delta:
                    # release[t] = state[t] × (1 − decay) is what
                    # ``calculate_total_power`` subtracts from heating
                    # demand under the live wiring (split-state, post-#896
                    # follow-up).  Multiplying the state delta by
                    # ``(1 − decay)`` projects the sweep from "what state
                    # would be" to "what release would be subtracted from
                    # prediction" — which matches the live model's
                    # observable effect on prediction error.
                    delta_release = (k_trace[idx] - baseline_trace[idx]) * (1 - decay_for_sweep)
                    residual_live = actual - expected
                    residual_alt = residual_live + delta_release
                    global_residuals.append(residual_alt)
                    if temp_bucket is None:
                        continue
                    cell_key = (hour_bucket, temp_bucket, screen_bucket)
                    cell_residuals.setdefault(cell_key, []).append(residual_alt)

                cells = {}
                for (hb, tb, sb), residuals in cell_residuals.items():
                    n = len(residuals)
                    sse = sum(r * r for r in residuals)
                    rmse = (sse / n) ** 0.5 if n > 0 else 0.0
                    mean = sum(residuals) / n if n > 0 else 0.0
                    cells[f"{hb}__{tb}__{sb}"] = {
                        "n": n,
                        "rmse_kwh": round(rmse, 4),
                        "mean_residual_kwh": round(mean, 4),
                    }
                global_n = len(global_residuals)
                global_sse = sum(r * r for r in global_residuals)
                global_rmse = (global_sse / global_n) ** 0.5 if global_n > 0 else 0.0
                per_k_results[str(k_cand)] = {
                    "global": {
                        "n": global_n,
                        "rmse_kwh": round(global_rmse, 4),
                    },
                    "cells": cells,
                }

            # Empirical optimum: lowest global RMSE.  Tie-break in favour
            # of smaller k (more conservative — closer to the disabled
            # default).  Reported as a recommendation, not auto-applied.
            best_k = 0.0
            best_global_rmse = per_k_results["0.0"]["global"]["rmse_kwh"]
            for k_cand in k_candidates:
                cand_rmse = per_k_results[str(k_cand)]["global"]["rmse_kwh"]
                if cand_rmse < best_global_rmse - 1e-6:
                    best_global_rmse = cand_rmse
                    best_k = k_cand

            # Per-cell delta-RMSE table relative to k=0.  Lets the user
            # see which (hour × temp × screen) combinations actually
            # benefit at the recommended k, vs which ones the global
            # optimum is averaging over.  Cells with n < 5 are emitted
            # but flagged so the reader does not over-interpret thin
            # data — particularly relevant at 10-day windows where many
            # cells carry only 1-3 hours.
            #
            # Sweep collapse (#896 follow-up).  When ``best_k == 0.0``
            # every per_cell_at_optimum row would be identical to the
            # baseline (delta_rmse = 0 everywhere), and every non-baseline
            # ``sweep[k]["cells"]`` table is informational fluff with no
            # actionable signal.  Emit only the baseline cells in
            # ``sweep["0.0"]`` and a single ``per_k_global_rmse`` summary
            # for the other candidates.  When ``best_k > 0`` the full
            # detail is preserved on baseline + optimum k; intermediate k
            # values still drop ``cells`` because they are not the
            # recommended target.
            if best_k == 0.0:
                per_cell_at_optimum = None
                for k_str, k_data in per_k_results.items():
                    if k_str != "0.0":
                        k_data.pop("cells", None)
            else:
                per_cell_at_optimum = {}
                baseline_cells = per_k_results["0.0"]["cells"]
                optimum_cells = per_k_results[str(best_k)]["cells"]
                all_cell_keys = set(baseline_cells) | set(optimum_cells)
                for cell_key in sorted(all_cell_keys):
                    base_cell = baseline_cells.get(cell_key, {"n": 0, "rmse_kwh": 0.0})
                    opt_cell = optimum_cells.get(cell_key, {"n": 0, "rmse_kwh": 0.0})
                    delta_rmse = opt_cell["rmse_kwh"] - base_cell["rmse_kwh"]
                    per_cell_at_optimum[cell_key] = {
                        "n": base_cell["n"],
                        "baseline_rmse_kwh": base_cell["rmse_kwh"],
                        "optimum_rmse_kwh": opt_cell["rmse_kwh"],
                        "delta_rmse_kwh": round(delta_rmse, 4),
                        "thin_sample": base_cell["n"] < 5,
                    }
                # Drop cells from intermediate k values; only baseline
                # and optimum are actionable for the user.
                for k_str, k_data in per_k_results.items():
                    if k_str != "0.0" and k_str != str(best_k):
                        k_data.pop("cells", None)

            battery_feedback_sweep = {
                "current_k": self.coordinator.battery_thermal_feedback_k,
                "decay_used": decay_for_sweep,
                "n_hours_in_window": len(sweep_tuples),
                "n_hours_with_heating_active": sum(
                    1 for t in sweep_tuples if t[4]
                ),
                "n_hours_with_heating_wasted": sum(
                    1 for t in sweep_tuples if t[1] > 0.0 and t[4]
                ),
                "k_candidates": k_candidates,
                "sweep": per_k_results,
                "empirical_optimum_k": best_k,
                "global_rmse_at_optimum_kwh": round(best_global_rmse, 4),
                "global_rmse_at_baseline_kwh": round(
                    per_k_results["0.0"]["global"]["rmse_kwh"], 4
                ),
                "rmse_improvement_kwh": round(
                    per_k_results["0.0"]["global"]["rmse_kwh"] - best_global_rmse, 4
                ),
                "per_cell_at_optimum": per_cell_at_optimum,
                # Methodology — read this before interpreting numbers.
                "method": "carryover_release_replay",
                "interpretation": "calibration_hint",
                "notes": (
                    "This sweep models the live wiring (split-state "
                    "implementation): _solar_carryover_state is charged "
                    "by k × wasted, and its release × (1 - decay) "
                    "subtracts from heating-mode demand prediction. "
                    "Each k candidate's hypothetical RMSE is computed "
                    "by replaying the carryover EMA over the window and "
                    "adding the release-delta vs k=0 baseline to the "
                    "logged residual.  Coefficients are held at their "
                    "currently-learned values (frozen-coefficient mode) "
                    "— real adoption of k > 0 will trigger ~2-6 % NLMS "
                    "coefficient drift over 2-3 weeks, which this "
                    "replay does not model.  Use empirical_optimum_k "
                    "as a calibration hint, then validate against "
                    "actual prediction RMSE after running with k > 0 "
                    "for 2-4 sunny weeks.  Transition-zone hours "
                    "(BP±2 °C) are included in both `global` aggregates "
                    "and the per-cell table under "
                    "temp_bucket=\"transition\"; expect that cell to "
                    "carry the strongest signal for the headline "
                    "symptom on high-BP installs."
                ),
            }

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

        # Top-level summary digest (#896 follow-up).  Human-readable
        # at-a-glance overview that lets the user decide whether to read
        # the verbose blocks below.  Computed strictly from already-built
        # blocks — no new arithmetic, just pivots and counts.  Verdict
        # logic is conservative: ``no_action_needed`` only when EVERY
        # signal source agrees nothing is actionable; otherwise
        # ``review_recommended`` and the user reads ``units_with_flags``,
        # ``global_flags``, and the battery sub-blocks for specifics.
        active_count = sum(
            1 for u in per_unit.values() if not u.get("inactive", False)
        )
        inactive_count = sum(
            1 for u in per_unit.values() if u.get("inactive", False)
        )
        units_with_flags = [
            {"entity_id": eid, "flags": u["flags"]}
            for eid, u in per_unit.items()
            if u.get("flags")
        ]
        # Battery feedback verdict.
        if battery_feedback_sweep:
            opt_k = battery_feedback_sweep.get("empirical_optimum_k", 0.0)
            improvement = battery_feedback_sweep.get("rmse_improvement_kwh", 0.0)
            if opt_k == 0.0:
                bf_verdict = "no_improvement_available"
            else:
                bf_verdict = f"consider_k_{opt_k}"
            battery_feedback_summary = {
                "current_k": battery_feedback_sweep.get("current_k", 0.0),
                "optimum_k": opt_k,
                "rmse_improvement_kwh": improvement,
                "verdict": bf_verdict,
            }
        else:
            battery_feedback_summary = {
                "current_k": self.coordinator.battery_thermal_feedback_k,
                "verdict": "no_data",
            }
        # Battery decay verdict pivots on assessment from battery_health
        # and the calibration block; "ok" means no recommendation pending.
        if calibration:
            decay_verdict = (
                "ok"
                if calibration.get("recommended_decay") == calibration.get("current_decay")
                else f"consider_decay_{calibration.get('recommended_decay')}"
            )
            battery_decay_summary = {
                "current_decay": calibration.get("current_decay"),
                "recommended_decay": calibration.get("recommended_decay"),
                "verdict": decay_verdict,
            }
        elif battery_health:
            battery_decay_summary = {
                "current_decay": battery_health.get("decay_rate"),
                "verdict": battery_health.get("assessment", "no_data"),
            }
        else:
            battery_decay_summary = {
                "current_decay": self.coordinator.solar_battery_decay,
                "verdict": "no_data",
            }
        # Top-level verdict: only ``no_action_needed`` when every signal
        # source is clean.  Otherwise the user should look at one of the
        # detail blocks the summary points at.
        any_action = (
            bool(global_flags)
            or bool(units_with_flags)
            or battery_feedback_summary["verdict"].startswith("consider_")
            or battery_decay_summary["verdict"].startswith("consider_")
            or battery_decay_summary["verdict"] in ("too_fast", "too_slow")
        )
        summary = {
            "verdict": "review_recommended" if any_action else "no_action_needed",
            "global_flags": global_flags,
            "active_solar_units": active_count,
            "inactive_units": inactive_count,
            "units_with_flags": units_with_flags,
            "battery_feedback": battery_feedback_summary,
            "battery_decay": battery_decay_summary,
        }

        return {
            "summary": summary,
            "context": context,
            "global": {
                "qualifying_hours": total_qualifying,
                "excluded": excluded,
                "battery_decay_health": battery_health,
                "battery_calibration": calibration,
                "battery_feedback_sweep": battery_feedback_sweep,
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

    def calibrate_per_unit_min_base_thresholds(
        self,
        *,
        sample_days: int = 30,
        require_min_hours_of_log: int | None = None,
    ) -> dict:
        """Compute per-unit min-base noise floor from dark-hour actuals (#871).

        Replaces the global 0.15 kWh gate with a per-sensor p10 of
        dark-hour (``solar_factor < PER_UNIT_MIN_BASE_DARK_SOLAR_FACTOR``)
        metered consumption.  Dark-hour filtering isolates the non-solar
        base-demand distribution; p10 captures the operating-noise floor
        without being skewed by the tail of idle samples.

        Safety guards (in order):
            1. Requires ≥ ``PER_UNIT_MIN_BASE_MIN_HOURS_OF_LOG`` hours of
               log data overall (14 × 24 by default).  Fresh installs
               skip calibration and continue on the global fallback.
            2. Requires ≥ ``PER_UNIT_MIN_BASE_MIN_SAMPLES`` dark-hour
               samples per unit.  Under-sampled units keep their prior
               value (or skip entirely if never calibrated).
            3. Rejects when ``p10 / median(dark_samples)`` exceeds
               ``PER_UNIT_MIN_BASE_MAX_P10_MEDIAN_RATIO``.  A legitimate
               noise floor sits far below typical consumption; a ratio
               near 1.0 indicates an always-on load (electric boiler
               mislabeled as heat-pump heating, sensor scoped to a
               shared circuit) where p10 is not a noise floor at all.
               Primary physics-grounded filter.
            4. Clamps p10 from below to ``PER_UNIT_MIN_BASE_FLOOR``;
               absolute ceiling ``PER_UNIT_MIN_BASE_CEILING`` acts as
               a safety net behind the ratio-guard.
            5. Limits rate-of-change to
               ``PER_UNIT_MIN_BASE_MAX_RATE_OF_CHANGE`` per run (±50 %
               vs previous value).  Protects against a single anomalous
               week flipping the threshold.

        Only heating-mode samples contribute.  Aux-active hours and guest
        modes are excluded — matches the live learning exclusion set.

        Returns a diagnostic dict usable by the ``calibrate_unit_thresholds``
        service and the startup log; ``self.coordinator._per_unit_min_base_thresholds``
        is updated in-place.
        """
        from datetime import timedelta
        from homeassistant.util import dt as dt_util
        from .const import (
            MODE_HEATING,
            PER_UNIT_MIN_BASE_CEILING,
            PER_UNIT_MIN_BASE_DARK_SOLAR_FACTOR,
            PER_UNIT_MIN_BASE_FLOOR,
            PER_UNIT_MIN_BASE_MAX_P10_MEDIAN_RATIO,
            PER_UNIT_MIN_BASE_MAX_RATE_OF_CHANGE,
            PER_UNIT_MIN_BASE_MIN_HOURS_OF_LOG,
            PER_UNIT_MIN_BASE_MIN_SAMPLES,
        )

        min_hours = (
            PER_UNIT_MIN_BASE_MIN_HOURS_OF_LOG
            if require_min_hours_of_log is None
            else require_min_hours_of_log
        )
        total_hours = len(self.coordinator._hourly_log)
        result = {
            "total_log_hours": total_hours,
            "required_log_hours": min_hours,
            "sample_days": sample_days,
            "status": "ok",
            "units": {},
            "updated": {},
            "rejected": {},
            "skipped": {},
        }
        if total_hours < min_hours:
            result["status"] = "insufficient_log_data"
            return result

        cutoff_iso = (dt_util.now() - timedelta(days=sample_days)).date().isoformat()

        # Collect dark-hour actuals per unit.
        samples: dict[str, list[float]] = {sid: [] for sid in self.coordinator.energy_sensors}
        for entry in self.coordinator._hourly_log:
            ts = entry.get("timestamp", "")
            if ts[:10] < cutoff_iso:
                continue
            if entry.get("auxiliary_active", False):
                continue
            solar_factor = entry.get("solar_factor") or 0.0
            if solar_factor >= PER_UNIT_MIN_BASE_DARK_SOLAR_FACTOR:
                continue
            unit_modes = entry.get("unit_modes", {}) or {}
            unit_breakdown = entry.get("unit_breakdown", {}) or {}
            for sid in self.coordinator.energy_sensors:
                mode = unit_modes.get(sid, MODE_HEATING)
                if mode != MODE_HEATING:
                    continue
                if sid not in unit_breakdown:
                    continue
                actual = unit_breakdown.get(sid, 0.0)
                if actual is None or actual < 0.0:
                    continue
                samples[sid].append(float(actual))

        def _p10(values: list[float]) -> float:
            if not values:
                return 0.0
            s = sorted(values)
            # Nearest-rank p10, 1-indexed: idx = ceil(0.10 × n) - 1 in 0-indexed.
            # math.ceil is required (not round): round underestimates the
            # rank for n ∈ {21..25} — and Python's banker's rounding flips
            # n=25 to 2 rather than 3 — which would bias calibrated
            # thresholds downward and let noisy low-base hours through
            # the NLMS gate.
            idx = max(0, math.ceil(0.10 * len(s)) - 1)
            return s[idx]

        for sid in self.coordinator.energy_sensors:
            dark = samples.get(sid, [])
            n = len(dark)
            prior = self.coordinator._per_unit_min_base_thresholds.get(sid)
            unit_report = {
                "dark_samples": n,
                "prior": prior,
                "p10_actual": None,
                "effective": prior,
                "method": "prior" if prior is not None else "fallback",
            }
            if n < PER_UNIT_MIN_BASE_MIN_SAMPLES:
                unit_report["status"] = "skipped_low_samples"
                result["skipped"][sid] = unit_report
                result["units"][sid] = unit_report
                continue

            p10 = _p10(dark)
            unit_report["p10_actual"] = round(p10, 5)

            # Ratio-guard (primary filter): a legitimate noise floor
            # sits far below typical consumption.  A sorted-dark-sample
            # distribution where p10 approaches the median means the
            # sensor is measuring an always-on load rather than a
            # modulating heat pump — no noise floor exists to calibrate.
            sorted_dark = sorted(dark)
            median = sorted_dark[len(sorted_dark) // 2]
            unit_report["median_actual"] = round(median, 5)
            if median > 0.0 and p10 > PER_UNIT_MIN_BASE_MAX_P10_MEDIAN_RATIO * median:
                unit_report["status"] = "rejected_constant_load"
                unit_report["p10_over_median"] = round(p10 / median, 3)
                _LOGGER.warning(
                    "Per-unit min-base calibration rejected for %s: "
                    "p10=%.3f kWh is %.0f%% of median=%.3f — distribution "
                    "suggests an always-on load, not a modulating heat pump. "
                    "Keeping %s.",
                    sid, p10, 100.0 * p10 / median, median,
                    f"prior {prior:.3f}" if prior else "global fallback",
                )
                result["rejected"][sid] = unit_report
                result["units"][sid] = unit_report
                continue

            # Absolute ceiling (safety net behind the ratio-guard).
            if p10 > PER_UNIT_MIN_BASE_CEILING:
                unit_report["status"] = "rejected_above_ceiling"
                _LOGGER.warning(
                    "Per-unit min-base calibration rejected for %s: p10=%.3f kWh "
                    "exceeds ceiling %.3f — keeping %s.",
                    sid, p10, PER_UNIT_MIN_BASE_CEILING,
                    f"prior {prior:.3f}" if prior else "global fallback",
                )
                result["rejected"][sid] = unit_report
                result["units"][sid] = unit_report
                continue

            candidate = max(PER_UNIT_MIN_BASE_FLOOR, p10)

            # Rate-of-change clamp vs prior value.
            if prior is not None and prior > 0.0:
                lo = prior * (1.0 - PER_UNIT_MIN_BASE_MAX_RATE_OF_CHANGE)
                hi = prior * (1.0 + PER_UNIT_MIN_BASE_MAX_RATE_OF_CHANGE)
                clamped = min(hi, max(lo, candidate))
                if clamped != candidate:
                    unit_report["rate_clamped_from"] = round(candidate, 5)
                candidate = max(PER_UNIT_MIN_BASE_FLOOR, clamped)

            new_value = round(candidate, 5)
            self.coordinator._per_unit_min_base_thresholds[sid] = new_value
            unit_report["effective"] = new_value
            unit_report["method"] = "auto"
            unit_report["status"] = "updated"
            result["updated"][sid] = unit_report
            result["units"][sid] = unit_report

        _LOGGER.info(
            "Per-unit min-base calibration complete: updated=%d, rejected=%d, skipped=%d",
            len(result["updated"]), len(result["rejected"]), len(result["skipped"]),
        )
        return result

