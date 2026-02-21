"""Statistics Manager Service."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, date

from homeassistant.util import dt as dt_util

from .helpers import get_last_year_iso_date
from .explanation import WeatherImpactAnalyzer, ExplanationFormatter
from .const import (
    ATTR_TEMP_ACTUAL_TODAY,
    ATTR_TEMP_ACTUAL_WEEK,
    ATTR_TEMP_ACTUAL_MONTH,
    ATTR_WIND_ACTUAL_TODAY,
    ATTR_WIND_ACTUAL_WEEK,
    ATTR_WIND_ACTUAL_MONTH,
    ATTR_TEMP_LAST_YEAR_DAY,
    ATTR_TEMP_LAST_YEAR_WEEK,
    ATTR_TEMP_LAST_YEAR_MONTH,
    ATTR_WIND_LAST_YEAR_DAY,
    ATTR_WIND_LAST_YEAR_WEEK,
    ATTR_WIND_LAST_YEAR_MONTH,
    ATTR_TDD_YESTERDAY,
    ATTR_TDD_LAST_7D,
    ATTR_TDD_LAST_30D,
    ATTR_EFFICIENCY_YESTERDAY,
    ATTR_EFFICIENCY_LAST_7D,
    ATTR_EFFICIENCY_LAST_30D,
    ATTR_EFFICIENCY_FORECAST_TODAY,
    ATTR_POTENTIAL_SAVINGS,
    ATTR_SOLAR_IMPACT,
    ATTR_ENERGY_TODAY,
    ATTR_PREDICTED,
    ATTR_SOLAR_PREDICTED,
    ATTR_TDD_SO_FAR,
    ATTR_SOLAR_FACTOR,
    ENERGY_GUARD_THRESHOLD,
    TARGET_TDD_WINDOW,
    MIN_EXTRAPOLATION_DELTA_T,
    MODE_HEATING,
    MODE_COOLING,
    MODE_OFF,
    MODE_GUEST_HEATING,
    MODE_GUEST_COOLING,
    TDD_STABILITY_THRESHOLD,
    TYPICAL_DAY_TEMP_TOLERANCE,
    TYPICAL_DAY_WIND_TOLERANCE,
    TYPICAL_DAY_MIN_SAMPLES,
    TYPICAL_DAY_HIGH_CONFIDENCE,
    DEVIATION_MIN_OBSERVATIONS,
    DEVIATION_MIN_KWH,
    DEVIATION_TOLERANCE_NEW,
    DEVIATION_TOLERANCE_MATURE,
    DEVIATION_MATURITY_COUNT,
)

_LOGGER = logging.getLogger(__name__)

# Minimum TDD required to trust the calculation fully without supplement
# 0.1 TDD ~ 2.4 Degree-Hours (e.g., 2.4C delta for 1 hour, or 12C delta for 12 mins)
MIN_STABLE_TDD = 0.1

class StatisticsManager:
    """Manages statistics and analytics calculations."""

    def __init__(self, coordinator) -> None:
        """Initialize with reference to coordinator."""
        self.coordinator = coordinator
        self._daily_savings_cache = {}

    def calculate_total_power(
        self,
        temp: float,
        effective_wind: float,
        solar_impact: float,
        is_aux_active: bool,
        unit_modes: dict[str, str] | None = None,
        override_solar_factor: float | None = None,
        detailed: bool = True,
        known_aux_impact_kwh: float | None = None,
    ) -> dict:
        """Calculate total power and breakdown using Hybrid "Global Stabilizer" logic.

        Architecture:
        1. Global Model (Master): Determines the Total Power (Anchor).
        2. Unit Models (Detail): Determines the Breakdown.
        3. Deviation: Any difference is flagged as 'unspecified'.
        4. Kelvin Protocol Update: When Exclusions are active, redistribute global aux reduction
           proportionally to remaining units to ensure Sum of Parts == Whole.

        Args:
            temp: Outdoor temperature (C)
            effective_wind: Effective wind speed (m/s)
            solar_impact: Global solar impact (kWh) - Legacy/Unused for calculation, kept for compat
            is_aux_active: Whether auxiliary heating is active
            unit_modes: Dictionary of {entity_id: mode}, defaults to coordinator state if None
            override_solar_factor: Optional solar factor (0.0-1.0) to use instead of current state
            detailed: Whether to include per-unit breakdown in the response. Defaults to True.
                      If False, unit_breakdown will be empty to save performance.
            known_aux_impact_kwh: Optional known auxiliary impact (kWh). If provided, it overrides
                                  the model-predicted aux reduction and forces distribution logic.

        Returns:
             dict: {
                "total_kwh": float (Global Model),
                "global_base_kwh": float (For Learning Track A),
                "global_aux_reduction_kwh": float (Global Authority for Aux Impact),
                "breakdown": {
                    "base_kwh": float (Unit Sum Base),
                    "aux_reduction_kwh": float (Unit Sum Aux - Scaled),
                    "unassigned_aux_savings": float (Aux overflow due to clamping),
                    "solar_reduction_kwh": float (Unit Sum Solar),
                    "unspecified_kwh": float (Global - Unit Sum)
                },
                "unit_breakdown": { ... }
             }
        """
        temp_key = str(int(round(temp)))
        wind_bucket = self.coordinator._get_wind_bucket(effective_wind)

        # Retrieve per-unit data structures
        correlation_per_unit = self.coordinator._correlation_data_per_unit
        aux_coeffs_per_unit = self.coordinator._aux_coefficients_per_unit

        # Pre-calculate Solar Factor if needed for unit distribution
        if override_solar_factor is not None:
            curr_solar_factor = override_solar_factor
        else:
            curr_solar_factor = self.coordinator.data.get(ATTR_SOLAR_FACTOR, 0.0)

        # --- Track A: Global Model (Top-Down / Master) ---
        # 1. Global Base Prediction
        global_base = self.coordinator._get_predicted_kwh(temp_key, wind_bucket, temp)

        # 2. Global Aux Reduction
        global_aux_reduction = 0.0
        effective_aux_active = is_aux_active

        if known_aux_impact_kwh is not None:
            # Physics Override: Use known value directly
            global_aux_reduction = known_aux_impact_kwh
            if global_aux_reduction > 0:
                effective_aux_active = True
        elif is_aux_active:
            # Kelvin Protocol: Global Model is the TRUTH for Total Reduction.
            # Note: If aux_affected_entities is explicitly empty (user removed all units),
            # the Global Model still returns its learned prediction. Track B will mark no
            # units as affected, so the full reduction becomes unassigned. This surfaces
            # via unassigned_kwh / leak_status in the Potential Savings sensor — that is
            # the intended feedback loop for this configuration.
            global_aux_reduction = self._get_prediction_from_model(
                self.coordinator._aux_coefficients, temp_key, wind_bucket, temp, self.coordinator.balance_point
            )

        # --- Track B: Unit Models (Bottom-Up) ---
        # Pass 1: Collect Raw Values
        raw_unit_data = {}
        sum_affected_aux = 0.0
        unit_sum_base = 0.0
        heating_solar_reduction_sum = 0.0
        cooling_solar_gain_sum = 0.0

        # Optimization: Use set for O(1) lookup of affected entities
        aux_affected_set = None
        if effective_aux_active:
            if hasattr(self.coordinator, "_aux_affected_set"):
                aux_affected_set = self.coordinator._aux_affected_set
            elif self.coordinator.aux_affected_entities:
                aux_affected_set = set(self.coordinator.aux_affected_entities)

        for entity_id in self.coordinator.energy_sensors:
            # Base Prediction
            unit_data = correlation_per_unit.get(entity_id, {})
            base_kwh = self._get_prediction_from_model(unit_data, temp_key, wind_bucket, temp, self.coordinator.balance_point)
            unit_sum_base += base_kwh

            # Aux Reduction (Raw)
            aux_reduction = 0.0
            is_affected = False
            if effective_aux_active:
                if aux_affected_set:
                     if entity_id in aux_affected_set:
                         is_affected = True
                elif entity_id in self.coordinator.aux_affected_entities:
                    is_affected = True

                if is_affected:
                    unit_aux_data = aux_coeffs_per_unit.get(entity_id, {})
                    aux_reduction = self._get_prediction_from_model(unit_aux_data, temp_key, wind_bucket, temp, self.coordinator.balance_point)
                    sum_affected_aux += aux_reduction

            # Solar Reduction
            unit_solar_reduction = 0.0
            if self.coordinator.solar_enabled:
                 unit_coeff = self.coordinator.solar.calculate_unit_coefficient(entity_id, temp_key)
                 unit_solar_reduction = self.coordinator.solar.calculate_unit_solar_impact(curr_solar_factor, unit_coeff)

            # Determine Mode
            mode = MODE_HEATING
            if unit_modes and entity_id in unit_modes:
                mode = unit_modes[entity_id]
            else:
                mode = self.coordinator.get_unit_mode(entity_id)

            if self.coordinator.solar_enabled:
                if mode == MODE_COOLING:
                    cooling_solar_gain_sum += unit_solar_reduction
                else:
                    heating_solar_reduction_sum += unit_solar_reduction

            raw_unit_data[entity_id] = {
                "base": base_kwh,
                "raw_aux": aux_reduction,
                "solar": unit_solar_reduction,
                "mode": mode,
                "affected": is_affected
            }

        # --- Pass 2: Finalize Breakdown ---
        unit_sum_net = 0.0
        unit_sum_aux_final = 0.0
        unit_sum_solar_final = 0.0  # This now tracks APPLIED solar
        unit_sum_solar_wasted = 0.0  # This tracks WASTED solar
        unassigned_aux_savings = 0.0
        unit_breakdown = {}

        # 3. Global Solar Adjustment (Respect Heating vs Cooling contributions)
        # Re-calculate global solar effect using the APPLIED (saturated) values to ensure consistency
        sum_applied_heating = 0.0
        sum_applied_cooling = 0.0

        for entity_id, data in raw_unit_data.items():
            # Use raw learned aux reduction (No Scaling)
            # We no longer force the sum of unit reductions to match the global model.
            # Any discrepancy is reported as 'orphaned_aux_savings' (positive gap)
            # or implicitly accepted as model disagreement.
            final_aux = data["raw_aux"]
            if not data["affected"]:
                final_aux = 0.0

            # Calculate Net: Base - Aux (Clamp for safety)
            applied_aux = min(final_aux, data["base"])
            overflow_aux = final_aux - applied_aux
            if overflow_aux > 0:
                unassigned_aux_savings += overflow_aux
            net_after_aux = data["base"] - applied_aux

            # Apply Solar to Net (with Saturation Logic)
            solar_applied, solar_wasted, net_final = self.coordinator.solar.calculate_saturation(
                net_after_aux, data["solar"], data["mode"]
            )

            # Store Breakdown (Only if detailed)
            if detailed:
                unit_breakdown[entity_id] = {
                    "net_kwh": round(net_final, 3),
                    "base_kwh": round(data["base"], 3),
                    "aux_reduction_kwh": round(applied_aux, 3),
                    "raw_aux_kwh": round(final_aux, 3),
                    "overflow_kwh": round(overflow_aux, 3),
                    "clamped": overflow_aux > 0.001,
                    "solar_reduction_kwh": round(solar_applied, 3),  # Changed from Potential to Applied
                    "raw_solar_kwh": round(data["solar"], 3),        # New: Potential
                    "solar_wasted_kwh": round(solar_wasted, 3)       # New: Wasted
                }

            # Accumulate Global Stats
            # Optimization: Calculate these here to avoid iterating unit_breakdown later
            # This also fixes a bug where detailed=False resulted in 0 solar effect
            if data["mode"] in (MODE_COOLING, MODE_GUEST_COOLING):
                sum_applied_cooling += solar_applied
            else:
                sum_applied_heating += solar_applied

            unit_sum_net += net_final
            unit_sum_aux_final += applied_aux
            unit_sum_solar_final += solar_applied
            unit_sum_solar_wasted += solar_wasted

        # Check for unassigned global savings (orphaned or overflow not captured by units)
        # This handles the case where Global Model predicts savings (e.g. known_aux_impact_kwh or learned model)
        # but unit models fail to capture it (e.g. no affected units, or unit models predict 0 and fallback fails).
        orphaned_aux_savings = 0.0
        if effective_aux_active and global_aux_reduction > 0:
            remaining = global_aux_reduction - unit_sum_aux_final - unassigned_aux_savings
            if remaining > 0.001:
                orphaned_aux_savings = remaining
                unassigned_aux_savings += remaining

        global_solar_effect = sum_applied_cooling - sum_applied_heating

        # 4. Global Net Calculation
        global_net_after_aux = max(0.0, global_base - global_aux_reduction)
        global_net = max(0.0, global_net_after_aux + global_solar_effect)

        # 5. Calculate Deviation (Unspecified)
        unspecified_kwh = global_net - unit_sum_net

        return {
            "total_kwh": round(global_net, 3),
            "global_base_kwh": round(global_base, 3),
            "global_aux_reduction_kwh": round(global_aux_reduction, 3),
            "breakdown": {
                "base_kwh": round(unit_sum_base, 3),
                "aux_reduction_kwh": round(unit_sum_aux_final, 3),
                "solar_reduction_kwh": round(unit_sum_solar_final, 3),
                "solar_wasted_kwh": round(unit_sum_solar_wasted, 3),
                "unassigned_aux_savings": round(unassigned_aux_savings, 3),
                "orphaned_aux_savings": round(orphaned_aux_savings, 3),
                "unspecified_kwh": round(unspecified_kwh, 3)
            },
            "unit_breakdown": unit_breakdown
        }

    def _resolve_bucket_for_extrapolation(self, bucket_data: dict, requested_bucket: str) -> str | None:
        """Resolve the best available bucket for extrapolation, preventing recursion loops.

        Priority:
        1. Requested Bucket (Exact Match)
        2. Normal (Most robust baseline)
        3. High Wind (If normal missing)
        4. Extreme Wind (Last resort)
        """
        if requested_bucket in bucket_data:
            return requested_bucket
        if "normal" in bucket_data:
            return "normal"
        if "high_wind" in bucket_data:
            return "high_wind"
        if "extreme_wind" in bucket_data:
            return "extreme_wind"
        return None

    def _get_prediction_from_model(self, data_map: dict, temp_key: str, wind_bucket: str, actual_temp: float, balance_point: float, apply_scaling: bool = True) -> float:
        """Retrieve value from a model map with robust fallback logic.

        Regime-Aware Priority:
        1. Exact Match (Temp, Wind) - Step 1 in all regimes.
        2. Cold Regime (Delta T > 4.0): Skip averaging/wind fallback. Force Thermodynamic Scaling
           (recursive extrapolation) with a strict guard (source_delta_t > 1.0).
        3. Mild Regime (Delta T <= 4.0):
           - Nearest Neighbor (Temp +/- 1, Same Wind)
           - Wind Fallback (Temp, Fallback Wind)
           - Thermodynamic Scaling (Extrapolation) with standard guard (source_delta_t > 0.5).

        Args:
            data_map: Dictionary { "temp_str": { "wind_bucket": value } }
            temp_key: Key for current temperature (e.g. "5")
            wind_bucket: Current wind bucket
            actual_temp: Exact current temperature (float) for TDD calculation
            balance_point: The balance point temperature for TDD scaling.
            apply_scaling: If False, extrapolation will not scale the result by Delta T.

        Returns:
            float: Predicted value
        """
        # 1. Exact Match
        if temp_key in data_map:
            bucket_data = data_map[temp_key]
            if wind_bucket in bucket_data:
                return bucket_data[wind_bucket]

        # Calculate regime thresholds
        delta_t_target = abs(balance_point - actual_temp)
        is_cold_regime = delta_t_target > 4.0

        # Determine guard for extrapolation
        # For Cold Regime: Require at least 1.0 degree delta (prevent noise from mild keys)
        # For Mild Regime: Require at least MIN_EXTRAPOLATION_DELTA_T (0.5 degree)
        min_source_delta_t = 1.0 if is_cold_regime else MIN_EXTRAPOLATION_DELTA_T

        if not is_cold_regime:
            _LOGGER.debug(
                "Regime-aware prediction: Mild Regime (Delta T=%.2f <= 4.0) for temp %.1f",
                delta_t_target, actual_temp
            )
            # 2. Nearest Neighbor (Temp +/- 1, Same Wind)
            try:
                t_int = int(temp_key)
                neighbors = []
                # Check T-1 and T+1
                for offset in [-1, 1]:
                    n_key = str(t_int + offset)
                    if n_key in data_map and wind_bucket in data_map[n_key]:
                        neighbors.append(data_map[n_key][wind_bucket])

                if neighbors:
                    return sum(neighbors) / len(neighbors)
            except ValueError:
                pass

            # 3. Wind Fallback (Same Temp)
            # Check if we have data for the requested temp but different wind
            if temp_key in data_map:
                bucket_data = data_map[temp_key]
                if wind_bucket == "extreme_wind":
                    if "high_wind" in bucket_data: return bucket_data["high_wind"]
                    if "normal" in bucket_data: return bucket_data["normal"]
                elif wind_bucket == "high_wind":
                    if "normal" in bucket_data: return bucket_data["normal"]

                # If requesting Normal and missing, we can't fallback to High/Extreme (unsafe).
                if "normal" in bucket_data:
                     return bucket_data["normal"]
        else:
            _LOGGER.debug(
                "Regime-aware prediction: Cold Regime (Delta T=%.2f > 4.0) for temp %.1f. Forcing Delta T scaling.",
                delta_t_target, actual_temp
            )

        # 4. Extrapolation (Thermodynamic Scaling)
        # Find nearest available temperature key
        try:
            target_t = int(temp_key)

            # Strategy: Prefer extrapolating from a NEIGHBOR first (to handle wind gaps in current key).
            # If no neighbors exist, fall back to current key and use its available buckets.

            # Step 4a: Find nearest key excluding self
            # Optimization: Iterate data_map directly to avoid O(N) list allocation and double iteration.
            best_key = None
            min_diff = 999
            has_any_data = False

            for k, buckets in data_map.items():
                if not buckets:
                    continue
                has_any_data = True

                if k == temp_key:
                    continue

                try:
                    k_int = int(k)
                    diff = abs(target_t - k_int)
                    if diff < min_diff:
                        min_diff = diff
                        best_key = k
                except ValueError:
                    continue

            # Check if we found any data at all
            if not has_any_data:
                return 0.0

            # Step 4b: If no neighbor found, fallback to self (Local Wind Fallback)
            if best_key is None:
                if data_map.get(temp_key):
                    best_key = temp_key
                else:
                    return 0.0

            # Get value from nearest neighbor or self (Recursive call handles its wind buckets)
            # If best_key == temp_key, recursive call will hit "Exact Match" if bucket exists,
            # or it will fall through to Wind Fallback logic (Mild) or come back here (Cold).
            # To prevent infinite recursion when best_key == temp_key, we must ensure we don't just loop.
            # We must resolve the bucket locally using the robust fallback logic.

            if best_key == temp_key:
                 bucket_data = data_map[temp_key]
                 fallback_bucket = self._resolve_bucket_for_extrapolation(bucket_data, wind_bucket)
                 return bucket_data[fallback_bucket] if fallback_bucket else 0.0

            # Before recursing to neighbor, ensure we use an available bucket to prevent ping-pong recursion
            neighbor_bucket_data = data_map.get(best_key, {})
            effective_bucket = self._resolve_bucket_for_extrapolation(neighbor_bucket_data, wind_bucket)

            if not effective_bucket:
                # Neighbor has no buckets at all (shouldn't happen if key is in valid_keys, but safe)
                return 0.0

            neighbor_val = self._get_prediction_from_model(data_map, best_key, effective_bucket, float(best_key), balance_point, apply_scaling)

            if neighbor_val <= 0.001:
                return 0.0

            # If scaling is disabled (e.g., for Aux models), return the neighbor value directly.
            if not apply_scaling:
                return neighbor_val

            # Scale by Delta T
            # Ratio = Delta T(target) / Delta T(source)
            # delta_t_target already calculated above
            delta_t_source = abs(balance_point - float(best_key))

            # Safety Guards
            # Prevent extrapolation from extreme noise (very close to Balance Point)
            # If source Delta T is negligible, the value is likely noise or idle load.
            # Extrapolating from noise with a huge ratio creates massive overprediction.
            if delta_t_source < min_source_delta_t:
                return neighbor_val

            ratio = delta_t_target / delta_t_source

            return round(neighbor_val * ratio, 3)

        except ValueError:
            return 0.0

    def calculate_realtime_efficiency(self) -> float | None:
        """Calculate real-time efficiency with Seamless Rolling Window.

        Algorithm:
        Efficiency = (Energy_Today + Energy_Yesterday_Fraction) / (TDD_Today + TDD_Yesterday_Fraction)

        Ensures the denominator (Total TDD) is at least TARGET_TDD_WINDOW (default 0.5).
        If Today's TDD is sufficient, we rely solely on today.
        If not, we "borrow" just enough from yesterday to reach the target window.

        Returns:
            Efficiency (kWh/TDD) or None/Fallback if data is insufficient.
        """
        tdd_so_far = self.coordinator.data.get(ATTR_TDD_SO_FAR, 0.0)
        energy_accumulated = self.coordinator.data.get(ATTR_ENERGY_TODAY, 0.0)

        # 1. Sufficient TDD Today? (Ideal Case)
        if tdd_so_far >= TARGET_TDD_WINDOW:
            # Safe division guaranteed by TARGET_TDD_WINDOW > 0
            return round(energy_accumulated / tdd_so_far, 3)

        # 2. Supplement with Yesterday (Rolling Window)
        needed_tdd = TARGET_TDD_WINDOW - tdd_so_far
        tdd_yesterday = self.coordinator.data.get(ATTR_TDD_YESTERDAY)
        eff_yesterday = self.coordinator.data.get(ATTR_EFFICIENCY_YESTERDAY)

        added_energy = 0.0
        added_tdd = 0.0

        if tdd_yesterday is not None and eff_yesterday is not None and tdd_yesterday > ENERGY_GUARD_THRESHOLD:
            available_tdd = tdd_yesterday
            # Take only what is needed to reach target, or all if less than needed
            take_tdd = min(needed_tdd, available_tdd)

            total_energy_yesterday = eff_yesterday * tdd_yesterday
            fraction = take_tdd / available_tdd

            added_energy = total_energy_yesterday * fraction
            added_tdd = take_tdd

        total_energy = energy_accumulated + added_energy
        total_tdd = tdd_so_far + added_tdd

        # 3. Stability Check & Blending
        # If we couldn't fill the window (e.g. yesterday missing/warm), and total_tdd is still small,
        # the calculation "Energy / small_number" is extremely volatile.
        # We implement a weighted blend between "Theoretical Model" and "Actual Data"
        # as TDD grows from 0.0 to MIN_STABLE_TDD.

        actual_eff = None
        if total_tdd > 0.001:
            actual_eff = total_energy / total_tdd

        if total_tdd < MIN_STABLE_TDD:
             model_inst = self.calculate_instantaneous_efficiency()

             # If no model available (warm weather), rely on actual if possible, else None
             if model_inst is None:
                 return round(actual_eff, 3) if actual_eff is not None and total_tdd > 0.05 else None

             # If no actual available (start of day), return model
             if actual_eff is None:
                 return round(model_inst, 3)

             # Blend Logic
             # Confidence in Actual Data grows linearly from 0 to 1 as TDD goes 0 -> MIN_STABLE_TDD
             # Quadratic Blending: Suppress noise at low TDD where division by near-zero is volatile.
             # e.g. At 10% TDD, confidence is 1% (not 10%). This clamps the noise significantly.
             ratio = total_tdd / MIN_STABLE_TDD
             confidence = ratio * ratio

             blended_eff = (model_inst * (1.0 - confidence)) + (actual_eff * confidence)
             return round(blended_eff, 3)

        # Safe division (guaranteed by MIN_STABLE_TDD > 0)
        return round(actual_eff, 3)

    def calculate_instantaneous_efficiency(self) -> float | None:
        """Calculate instantaneous efficiency when accumulated TDD is insufficient.

        Returns:
            Efficiency (kWh/TDD) or None if heating demand is negligible.
        """
        current_rate = self.coordinator.data.get("current_model_rate", 0.0)
        current_temp = self.coordinator.data.get("current_calc_temp")

        if current_temp is None:
            return None

        # Calculate hourly TDD rate
        degree_days_hourly = abs(self.coordinator.balance_point - current_temp) / 24.0

        # Guard: If TDD rate is very low (near balance point), efficiency is undefined/unstable.
        if degree_days_hourly <= TDD_STABILITY_THRESHOLD:
            return None

        return current_rate / degree_days_hourly

    def get_max_historical_daily_kwh(self) -> float:
        """Get the maximum daily energy consumption observed in history."""
        max_kwh = 0.0
        if not self.coordinator._daily_history:
            return 0.0

        for day_data in self.coordinator._daily_history.values():
            if day_data and "kwh" in day_data:
                kwh = day_data["kwh"]
                if kwh > max_kwh:
                    max_kwh = kwh

        return max_kwh

    def get_typical_day_consumption(self, temp: float) -> tuple[float | None, int, str]:
        """Find typical (median) daily consumption for a given temperature under normal conditions.

        Criteria:
        - Temp within +/- 1.0 C
        - Wind deviation from global average < 2.0 m/s (Rolige forhold)

        Returns:
            (typical_kwh, sample_count, confidence_level)
        """
        if not self.coordinator._daily_history:
            return None, 0, "low"

        # Calculate global average wind
        winds = [d["wind"] for d in self.coordinator._daily_history.values() if d and "wind" in d]
        if not winds:
            return None, 0, "low"

        avg_wind_global = sum(winds) / len(winds)

        samples = []
        for day_data in self.coordinator._daily_history.values():
            if not day_data or "temp" not in day_data or "kwh" not in day_data:
                continue

            d_temp = day_data["temp"]
            d_wind = day_data.get("wind", 0.0)

            # Normalise consumption to "what would it be without aux"
            # Actual + Saved = Base Consumption
            base_kwh = day_data["kwh"] + day_data.get("aux_impact_kwh", 0.0)

            if abs(d_temp - temp) <= TYPICAL_DAY_TEMP_TOLERANCE:
                if abs(d_wind - avg_wind_global) < TYPICAL_DAY_WIND_TOLERANCE:
                    samples.append(base_kwh)

        count = len(samples)
        if count < TYPICAL_DAY_MIN_SAMPLES:
            return None, count, "low"

        # Calculate Median
        samples.sort()
        mid = count // 2
        if count % 2 == 0:
            median = (samples[mid - 1] + samples[mid]) / 2.0
        else:
            median = samples[mid]

        confidence = "high" if count >= TYPICAL_DAY_HIGH_CONFIDENCE else "medium"
        return round(median, 1), count, confidence

    def calculate_temp_stats(self):
        """Calculate temperature and wind statistics for sensors."""
        current_time = dt_util.now()
        today = current_time.date()

        def _avg(values):
            return sum(values) / len(values) if values else None

        today_iso = today.isoformat()
        start_of_week_iso = (today - timedelta(days=today.weekday())).isoformat()
        start_of_month_iso = today.replace(day=1).isoformat()

        today_temp_sum = 0.0
        today_wind_sum = 0.0
        today_count = 0

        week_temp_sum = 0.0
        week_wind_sum = 0.0
        week_count = 0

        month_temp_sum = 0.0
        month_wind_sum = 0.0
        month_count = 0

        # Optimization (Method B): Iterate in reverse to find recent logs efficiently.
        # This avoids scanning the entire 90-day history when we only need the current month.
        # Reduces complexity from O(Total History) to O(Current Month).
        # Optimization: Use running sums to avoid O(N) list allocation and secondary iteration.
        for entry in reversed(self.coordinator._hourly_log):
            ts = entry["timestamp"]

            if ts < start_of_month_iso:
                break

            t = entry["temp"]
            w = entry["effective_wind"]

            month_temp_sum += t
            month_wind_sum += w
            month_count += 1

            if ts >= start_of_week_iso:
                week_temp_sum += t
                week_wind_sum += w
                week_count += 1
                if ts.startswith(today_iso):
                    today_temp_sum += t
                    today_wind_sum += w
                    today_count += 1

        self.coordinator.data[ATTR_TEMP_ACTUAL_TODAY] = round(today_temp_sum / today_count, 1) if today_count > 0 else None
        self.coordinator.data[ATTR_WIND_ACTUAL_TODAY] = round(today_wind_sum / today_count, 1) if today_count > 0 else 0.0

        self.coordinator.data[ATTR_TEMP_ACTUAL_WEEK] = round(week_temp_sum / week_count, 1) if week_count > 0 else None
        self.coordinator.data[ATTR_WIND_ACTUAL_WEEK] = round(week_wind_sum / week_count, 1) if week_count > 0 else 0.0

        self.coordinator.data[ATTR_TEMP_ACTUAL_MONTH] = round(month_temp_sum / month_count, 1) if month_count > 0 else None
        self.coordinator.data[ATTR_WIND_ACTUAL_MONTH] = round(month_wind_sum / month_count, 1) if month_count > 0 else 0.0

        try:
            last_year_today = today.replace(year=today.year - 1)
        except ValueError:
            last_year_today = today.replace(year=today.year - 1, day=28)

        ly_day_key = last_year_today.isoformat()
        entry_ly_day = self.coordinator._daily_history.get(ly_day_key)

        if entry_ly_day:
            self.coordinator.data[ATTR_TEMP_LAST_YEAR_DAY] = entry_ly_day.get("temp")
            self.coordinator.data[ATTR_WIND_LAST_YEAR_DAY] = entry_ly_day.get("wind", 0.0)
        else:
            self.coordinator.data[ATTR_TEMP_LAST_YEAR_DAY] = None
            self.coordinator.data[ATTR_WIND_LAST_YEAR_DAY] = None

        # Get start of current ISO week (Monday)
        current_iso_week_start = today - timedelta(days=today.isoweekday() - 1)
        ly_week_start = get_last_year_iso_date(current_iso_week_start)

        ly_week_t, ly_week_w = self._calculate_stats_for_period(ly_week_start, 7)
        self.coordinator.data[ATTR_TEMP_LAST_YEAR_WEEK] = ly_week_t
        self.coordinator.data[ATTR_WIND_LAST_YEAR_WEEK] = ly_week_w

        try:
            ly_month_start = date(today.year - 1, today.month, 1)
        except ValueError:
            ly_month_start = date(today.year - 1, today.month, 28)

        if ly_month_start.month == 12:
            ly_next_month = date(ly_month_start.year + 1, 1, 1)
        else:
            ly_next_month = date(ly_month_start.year, ly_month_start.month + 1, 1)

        days_in_ly_month = (ly_next_month - ly_month_start).days

        ly_month_t, ly_month_w = self._calculate_stats_for_period(ly_month_start, days_in_ly_month)
        self.coordinator.data[ATTR_TEMP_LAST_YEAR_MONTH] = ly_month_t
        self.coordinator.data[ATTR_WIND_LAST_YEAR_MONTH] = ly_month_w

        yesterday = today - timedelta(days=1)
        yesterday_key = yesterday.isoformat()
        if yesterday_key in self.coordinator._daily_history:
             entry = self.coordinator._daily_history[yesterday_key]
             if entry is not None:
                 tdd_yest = entry.get("tdd", 0.0)
                 kwh_yest = entry.get("kwh", 0.0)
             else:
                 tdd_yest = 0.0
                 kwh_yest = 0.0
             self.coordinator.data[ATTR_TDD_YESTERDAY] = tdd_yest

             if tdd_yest > ENERGY_GUARD_THRESHOLD:
                 self.coordinator.data[ATTR_EFFICIENCY_YESTERDAY] = round(kwh_yest / tdd_yest, 3)
             else:
                 self.coordinator.data[ATTR_EFFICIENCY_YESTERDAY] = None
        else:
             self.coordinator.data[ATTR_TDD_YESTERDAY] = None
             self.coordinator.data[ATTR_EFFICIENCY_YESTERDAY] = None

        last_7_keys = []
        last_30_keys = []

        for i in range(1, 31):
            d_key = (today - timedelta(days=i)).isoformat()
            if d_key in self.coordinator._daily_history:
                if i <= 7:
                    last_7_keys.append(d_key)
                last_30_keys.append(d_key)

        tdd_7d, eff_7d = self._calculate_efficiency_stats(last_7_keys)
        self.coordinator.data[ATTR_TDD_LAST_7D] = tdd_7d
        self.coordinator.data[ATTR_EFFICIENCY_LAST_7D] = eff_7d

        tdd_30d, eff_30d = self._calculate_efficiency_stats(last_30_keys)
        self.coordinator.data[ATTR_TDD_LAST_30D] = tdd_30d
        self.coordinator.data[ATTR_EFFICIENCY_LAST_30D] = eff_30d

    def _calculate_stats_for_period(self, start_date: date, days: int) -> tuple[float | None, float | None]:
        """Calculate average temp and wind for a historical period."""
        sum_temp = 0.0
        sum_wind = 0.0
        count = 0

        # Optimization: Local variable for history and iteration
        history = self.coordinator._daily_history
        current_date = start_date
        one_day = timedelta(days=1)

        for _ in range(days):
            d_key = current_date.isoformat()
            current_date += one_day

            if d_key in history:
                entry = history[d_key]
                if entry is not None:
                    sum_temp += entry.get("temp", 0.0)
                    sum_wind += entry.get("wind", 0.0)
                    count += 1

        avg_temp = round(sum_temp / count, 1) if count > 0 else None
        avg_wind = round(sum_wind / count, 1) if count > 0 else None
        return avg_temp, avg_wind

    def _calculate_efficiency_stats(self, keys: list[str]) -> tuple[float | None, float | None]:
        """Calculate average TDD and Efficiency for a given list of history keys."""
        if not keys:
            return None, None

        sum_tdd = 0.0
        sum_kwh = 0.0
        count = 0
        history = self.coordinator._daily_history

        # Optimization: Iterate once, avoid list allocation
        for k in keys:
            entry = history.get(k)
            if entry is not None:
                sum_tdd += entry.get("tdd", 0.0)
                sum_kwh += entry.get("kwh", 0.0)
                count += 1

        if count == 0:
            return None, None

        avg_tdd = round(sum_tdd / count, 1)

        if sum_tdd > ENERGY_GUARD_THRESHOLD:
            efficiency = round(sum_kwh / sum_tdd, 3)
        else:
            efficiency = None

        return avg_tdd, efficiency

    def update_daily_savings_cache(self):
        """Update the cache for potential savings based on completed hours.

        This method should be called once per hour (after hourly log processing)
        to cache the stable portion of the day's savings calculation.
        """
        today_iso = dt_util.now().date().isoformat()

        theory_normal = 0.0
        theory_aux = 0.0
        missing_aux_data = False
        aux_hours_count = 0.0
        actual_savings_kwh = 0.0
        aux_hours_list = []

        today_logs = [e for e in self.coordinator._hourly_log if e["timestamp"].startswith(today_iso)]

        for entry in today_logs:
            temp = entry["temp"]
            eff_wind = entry["effective_wind"]
            solar_impact = entry.get("solar_impact_kwh", 0.0)

            t_norm, t_aux, is_missing, _ = self._calculate_savings_component(temp, eff_wind, solar_impact, detailed=False)

            if is_missing:
                missing_aux_data = True

            theory_normal += t_norm
            theory_aux += t_aux

            aux_impact = entry.get("aux_impact_kwh", 0.0)
            actual_savings_kwh += aux_impact

            if entry.get("auxiliary_active") or aux_impact > 0.001:
                aux_hours_count += 1.0
                aux_hours_list.append(entry["hour"])

        self._daily_savings_cache = {
            "date": today_iso,
            "theory_normal": theory_normal,
            "theory_aux": theory_aux,
            "aux_hours": aux_hours_count,
            "actual_savings": actual_savings_kwh,
            "aux_list": aux_hours_list,
            "missing_data": missing_aux_data
        }

    def calculate_potential_savings(self):
        """Calculate daily potential savings (Cache + Live)."""
        today_iso = dt_util.now().date().isoformat()

        # Ensure cache is initialized for today
        if self._daily_savings_cache.get("date") != today_iso:
             self.update_daily_savings_cache()

        # Start with cached values (Completed Hours)
        theory_normal = self._daily_savings_cache["theory_normal"]
        theory_aux = self._daily_savings_cache["theory_aux"]
        missing_aux_data = self._daily_savings_cache["missing_data"]

        aux_hours_count = self._daily_savings_cache["aux_hours"]
        actual_savings_kwh = self._daily_savings_cache["actual_savings"]
        # Copy list to avoid mutating cache
        aux_hours_list = list(self._daily_savings_cache["aux_list"])

        # Add Live Component (Current Partial Hour)
        current_temp = self.coordinator.data.get("current_calc_temp")

        # Initialize remainders (defaults if no current temp)
        remainder_normal = 0.0
        remainder_aux = 0.0

        if current_temp is not None:
             eff_wind = self.coordinator.data.get("effective_wind", 0.0)
             solar_impact = self.coordinator.data.get(ATTR_SOLAR_IMPACT, 0.0)
             minutes_passed = dt_util.now().minute
             fraction = minutes_passed / 60.0

             t_norm, t_aux, is_missing, potential_breakdown = self._calculate_savings_component(current_temp, eff_wind, solar_impact, detailed=True)

             # Store Potential Breakdown for Sensor visibility
             self.coordinator.data["potential_savings_breakdown"] = potential_breakdown

             if is_missing:
                 missing_aux_data = True

             theory_normal += t_norm * fraction
             theory_aux += t_aux * fraction

             # --- Calculate Whole Day Projection (Theoretical Max) ---
             # Completed Hours (Cache) + Current Hour (Live Extrapolation) + Future Hours (Forecast)
             remaining_fraction = max(0.0, 1.0 - fraction)
             remainder_normal = t_norm * remaining_fraction
             remainder_aux = t_aux * remaining_fraction

             # Check if Aux is active OR has produced impact (Mixed Mode)
             live_aux_impact = self.coordinator._accumulated_aux_impact_hour
             if self.coordinator.auxiliary_heating_active or live_aux_impact > 0.001:
                 aux_hours_count += fraction
                 aux_hours_list.append(dt_util.now().hour)

        # Actual Savings (The Reward)
        # Use the robust global accumulator from Coordinator which sums (Logs + Live)
        # This solves the "Mixed Hour" issue where a flag flip might zero out the savings
        # if we calculated it dynamically here.
        actual_savings_kwh = self.coordinator.data.get("accumulated_aux_impact_kwh", 0.0)

        # Calculate Future Component (Forecast Weather) - ALWAYS
        # Force Ignore Aux for Normal, Force Aux for Aux
        future_normal, _, _ = self.coordinator.forecast.calculate_future_energy(
             dt_util.now(), ignore_aux=True
        )
        future_aux, _, _ = self.coordinator.forecast.calculate_future_energy(
             dt_util.now(), force_aux=True
        )

        # Total Projected (Full Day)
        # theory_normal/aux here contains ONLY "So Far" (Cache + Current Passed Fraction)
        projected_total_normal = theory_normal + remainder_normal + future_normal
        projected_total_aux = theory_aux + remainder_aux + future_aux

        # IMPORTANT: savings_theory_normal/aux must reflect "So Far" to align with Actual Savings
        self.coordinator.data["savings_theory_normal"] = round(theory_normal, 3)
        self.coordinator.data["savings_theory_aux"] = round(theory_aux, 3)
        self.coordinator.data["missing_aux_data"] = missing_aux_data

        self.coordinator.data["savings_aux_hours_today"] = round(aux_hours_count, 1)
        self.coordinator.data["savings_actual_kwh"] = round(actual_savings_kwh, 3)
        self.coordinator.data["savings_aux_hours_list"] = aux_hours_list

        # POTENTIAL SAVINGS (Theoretical Max) = Full Day Projection
        # This attribute specifically requested to be "Weather Plan Today"-like (Whole Day)
        # while keeping the consumption attributes "So Far".
        self.coordinator.data[ATTR_POTENTIAL_SAVINGS] = round(projected_total_normal - projected_total_aux, 3)

    def _calculate_savings_component(self, temp: float, eff_wind: float, solar_impact: float, detailed: bool = False) -> tuple[float, float, bool, dict]:
        """Calculate theoretical energy consumption for normal and aux modes."""
        # Use robust calculation for both Normal and Aux
        # This replaces the old specific calls with the unified logic

        # Scenario 1: Normal Mode (Aux Inactive)
        res_normal = self.calculate_total_power(
            temp, eff_wind, solar_impact, is_aux_active=False, detailed=False
        )
        pred_norm = res_normal["total_kwh"]

        # Scenario 2: Aux Mode (Aux Active)
        res_aux = self.calculate_total_power(
            temp, eff_wind, solar_impact, is_aux_active=True, detailed=detailed
        )
        pred_aux = res_aux["total_kwh"]

        # Instantaneous savings rate
        savings_rate = pred_norm - pred_aux
        if pred_aux == 0 and pred_norm == 0:
             self.coordinator.data["current_savings_rate"] = None
        else:
             self.coordinator.data["current_savings_rate"] = round(savings_rate, 2)

        # Check for missing aux data
        missing_aux_data = False
        if savings_rate == 0 and pred_norm > 0:
             if res_aux["breakdown"]["aux_reduction_kwh"] == 0:
                 missing_aux_data = True

        return pred_norm, pred_aux, missing_aux_data, res_aux["unit_breakdown"]

    def calculate_modeled_energy(self, start_date: date, end_date: date, pre_fetched_logs: dict | None = None) -> tuple[float, float, float | None, float | None, float]:
        """Calculate modeled energy and weather stats for a date range."""
        total_kwh = 0.0
        total_solar_impact = 0.0
        sum_temp = 0.0
        sum_wind = 0.0
        total_tdd = 0.0
        count_days = 0

        # Optimization: Pre-fetch logs for the range (Method B: O(Range) vs O(N))
        if pre_fetched_logs is not None:
             daily_log_map = pre_fetched_logs
        else:
             daily_log_map = self._get_daily_log_map(start_date, end_date)

        # Optimization: Pre-compute mode maps to avoid O(N*M) dictionary creation in loop
        modes_heating = {eid: MODE_HEATING for eid in self.coordinator.energy_sensors}
        modes_cooling = {eid: MODE_COOLING for eid in self.coordinator.energy_sensors}

        current = start_date
        while current <= end_date:
            date_iso = current.isoformat()
            daily_logs = daily_log_map.get(date_iso)

            # --- Source Selection ---
            # 1. Prefer Hourly Logs (Most accurate, captures intra-day variation)
            # 2. Fallback to Daily History (Averages, less accurate due to Jensen's Inequality)

            day_data_points = []

            if daily_logs:
                for log in daily_logs:
                    day_data_points.append({
                        "temp": log["temp"],
                        "wind": log.get("effective_wind", 0.0),
                        "load": log.get("actual_kwh", 0.0), # Use actual load for weighting
                        "solar_factor": log.get("solar_factor"),
                        "timestamp": log["timestamp"],
                        "multiplier": 1.0, # Hourly logs have 1h weight
                        "unit_modes": log.get("unit_modes") # Capture unit modes
                    })
            elif date_iso in self.coordinator._daily_history:
                entry = self.coordinator._daily_history[date_iso]
                # Guard against None entries in legacy storage
                if entry is not None:
                    # Strategy: Use Vectors (Precision) -> Fallback to Daily (Average)
                    vectors = entry.get("hourly_vectors")

                    # Check if vectors exist AND contain actual data (not just Nones)
                    has_valid_vectors = False
                    if vectors:
                         # Handle both legacy 'load' and new 'actual_kwh'
                         vec_load = vectors.get("actual_kwh", vectors.get("load"))

                         if vectors.get("temp") and vec_load:
                            # Ensure we have at least 20 valid temperature readings to reconstruct
                            # (Fixes Partial Day Data Corruption where <20h data is upscaled by 1.0)
                            valid_hours = sum(1 for t in vectors["temp"] if t is not None)
                            if valid_hours >= 20:
                                has_valid_vectors = True

                    # Only proceed with reconstruction if we passed the check
                    if has_valid_vectors:
                        # Get correct key for load
                        vec_load = vectors.get("actual_kwh", vectors.get("load"))

                        # Reconstruct hourly points from vectors
                        for h in range(24):
                            v_temp = vectors["temp"][h]
                            v_load = vec_load[h]
                            if v_temp is not None:
                                day_data_points.append({
                                    "temp": v_temp,
                                    "wind": vectors["wind"][h] if vectors.get("wind") else entry.get("wind", 0.0),
                                    "load": v_load if v_load is not None else 0.0,
                                    "tdd": vectors["tdd"][h] if vectors.get("tdd") else None,
                                    "solar_factor": vectors["solar_rad"][h] if vectors.get("solar_rad") else entry.get("solar_factor"),
                                    "timestamp": None,
                                    "multiplier": 1.0, # Hourly weight
                                    "unit_modes": None # Vectors don't store unit modes yet, inferred later
                                })
                    else:
                        # Fallback to Daily Averages
                        day_data_points.append({
                            "temp": entry.get("temp", 0.0),
                            "tdd": entry.get("tdd"), # Capture TDD for reconstruction
                            "wind": entry.get("wind", 0.0),
                            "load": entry.get("kwh", 0.0), # Daily average doesn't have intra-day shape
                            "solar_factor": entry.get("solar_factor"),
                            "timestamp": None,
                            "multiplier": 24.0 # Daily history has 24h weight
                        })

            # --- Process Day ---
            if day_data_points:
                day_temp_weighted_sum = 0.0
                day_wind_weighted_sum = 0.0
                day_load_sum = 0.0
                day_temp_simple_sum = 0.0
                day_wind_simple_sum = 0.0
                day_sample_count = 0

                for point in day_data_points:
                    temp = point["temp"]
                    wind = point["wind"]
                    load = point.get("load", 0.0)
                    s_factor = point["solar_factor"]
                    multiplier = point["multiplier"]

                    # Thermodynamic Reconstruction (Jensen's Inequality Fix)
                    # This logic is now applied to BOTH hourly logs and daily history
                    # to ensure consistent calculations.
                    calc_temp = temp
                    tdd_contribution = 0.0

                    if point.get("tdd") is not None:
                        tdd_contribution = float(point["tdd"])
                        tdd_val = tdd_contribution
                        # If it's an hourly log, scale the TDD value up to represent the hour's delta
                        if multiplier == 1.0:
                            tdd_val = tdd_val * 24.0

                        if tdd_val > 0.1:  # Threshold to avoid noise
                            # Recover effective temperature from TDD
                            if temp >= self.coordinator.balance_point:
                                calc_temp = self.coordinator.balance_point + tdd_val
                            else:
                                calc_temp = self.coordinator.balance_point - tdd_val
                    else:
                        # Fallback if TDD missing: Calculate from temp (Approximate)
                        # If Hourly (1.0): abs(BP - temp) / 24.0
                        # If Daily (24.0): abs(BP - temp)
                        delta = abs(self.coordinator.balance_point - temp)
                        if multiplier == 1.0:
                            tdd_contribution = delta / 24.0
                        else:
                            tdd_contribution = delta

                    total_tdd += tdd_contribution

                    if self.coordinator.solar_enabled:
                        if multiplier > 1.0 and s_factor is None:  # Only for daily history
                            s_factor = self.coordinator.solar.estimate_daily_avg_solar_factor(
                                current
                            )

                    # Use Robust Lookup manually via calculate_total_power
                    # This supports thermodynamic reconstruction (calc_temp) and history
                    # We infer mode from temperature (Heating vs Cooling).
                    unit_modes = point.get("unit_modes")
                    if not unit_modes:
                        # Use pre-computed maps for massive speedup in historical processing
                        if calc_temp < self.coordinator.balance_point:
                            unit_modes = modes_heating
                        else:
                            unit_modes = modes_cooling

                    res = self.calculate_total_power(
                        temp=calc_temp,
                        effective_wind=wind,
                        solar_impact=0.0,  # Unused
                        is_aux_active=False,
                        unit_modes=unit_modes,
                        override_solar_factor=s_factor,
                        detailed=False,
                    )

                    # Multiply by multiplier (24 for daily history, 1 for hourly)
                    day_total_kwh = res["total_kwh"] * multiplier
                    day_total_solar = (
                        res["breakdown"]["solar_reduction_kwh"] * multiplier
                    )

                    total_kwh += day_total_kwh
                    total_solar_impact += day_total_solar

                    # Accumulate for weighting
                    # Use multiplier to normalize daily vs hourly
                    # Ideally, hourly logs have multiplier 1, daily has 24.
                    # Load is total for that slot.
                    # If daily average (multiplier 24), load is daily total.
                    # If hourly (multiplier 1), load is hourly total.

                    # Weighted Accumulation
                    day_temp_weighted_sum += calc_temp * load
                    day_wind_weighted_sum += wind * load
                    day_load_sum += load

                    # Simple Accumulation (Fallback)
                    day_temp_simple_sum += calc_temp
                    day_wind_simple_sum += wind
                    day_sample_count += 1

                # Accumulate Daily Stats for Global Average
                if day_sample_count > 0:
                    # Prefer Weighted Average if meaningful load exists
                    if day_load_sum > 0.001:
                        sum_temp += day_temp_weighted_sum / day_load_sum
                        sum_wind += day_wind_weighted_sum / day_load_sum
                    else:
                        sum_temp += day_temp_simple_sum / day_sample_count
                        sum_wind += day_wind_simple_sum / day_sample_count
                    count_days += 1

            current += timedelta(days=1)

        avg_temp = round(sum_temp / count_days, 1) if count_days > 0 else None
        avg_wind = round(sum_wind / count_days, 1) if count_days > 0 else None

        return round(total_kwh, 2), round(total_solar_impact, 2), avg_temp, avg_wind, round(total_tdd, 1)

    def _get_daily_log_map(self, start_date: date, end_date: date) -> dict:
        """Get a map of date_iso -> list[log_entries] for the date range.

        Optimization (Method B):
        - Uses reversed iteration to efficiently find recent logs.
        - Checks bounds (start_iso, end_iso) to break early.
        - Reduces complexity from O(N) to O(Range) for recent queries (which are most common).
        """
        daily_log_map = {}

        if not self.coordinator._hourly_log:
            return daily_log_map

        start_iso = start_date.isoformat()
        end_iso = end_date.isoformat()

        for entry in reversed(self.coordinator._hourly_log):
            date_key = entry["timestamp"][:10]

            if date_key > end_iso:
                continue

            if date_key < start_iso:
                break

            if date_key not in daily_log_map:
                daily_log_map[date_key] = []

            daily_log_map[date_key].append(entry)

        return daily_log_map

    def calculate_historical_actual_sum(self, start_date: date, end_date: date) -> float | None:
        """
        Summerer faktisk kwh fra _daily_history for gitt periode.
        Brukes til å hente last year actuals.

        Args:
            start_date: Start dato (inklusiv)
            end_date: Slutt dato (inklusiv)

        Returns:
            float | None: Total kwh fra historical data, eller None hvis ingen data
        """
        total = 0.0
        found_data = False
        current = start_date
        while current <= end_date:
            day_str = current.isoformat()
            if day_str in self.coordinator._daily_history:
                entry = self.coordinator._daily_history[day_str]
                if entry is not None:
                    total += entry.get("kwh", 0.0)
                    found_data = True
            current += timedelta(days=1)
        return round(total, 3) if found_data else None

    def _calculate_model_value_from_log(self, log: dict) -> tuple[float, float]:
        """Calculate model kWh and solar kWh for a single log entry.

        Handles both legacy logs (global calculation) and modern logs (per-unit calculation).
        Uses 'unit_modes' from log if available to correctly apply solar correction (Heating/Cooling).
        """
        temp = log["temp"]
        eff_wind = log.get("effective_wind", 0.0)
        s_factor = log.get("solar_factor") # Can be None
        unit_modes = log.get("unit_modes")

        # For pure model reconstruction, ALWAYS assume aux is inactive.
        res = self.calculate_total_power(
            temp=temp,
            effective_wind=eff_wind,
            solar_impact=0.0,  # Legacy, unused
            is_aux_active=False, # CRUCIAL for pure model
            unit_modes=unit_modes,
            override_solar_factor=s_factor,
            detailed=False,
        )

        net_kwh = res["total_kwh"]
        total_solar_impact_kwh = res["breakdown"]["solar_reduction_kwh"]

        return net_kwh, total_solar_impact_kwh

    def _calculate_pure_model_today(self) -> tuple[float, float]:
        """Calculate pure model energy for today (Past Actual Weather + Future Forecast).

        Uses learned model parameters on actual weather for past hours,
        and forecast (ignoring aux) for remaining hours.
        """
        total_kwh = 0.0
        total_solar = 0.0
        today_iso = dt_util.now().date().isoformat()

        # 1. Past Hours (from Log) - Reconstruct Model Value
        # This ensures we use the model even if actual consumption was different (or aux affected it)
        for log in self.coordinator._hourly_log:
            if log["timestamp"].startswith(today_iso):
                model_kwh, solar_kwh = self._calculate_model_value_from_log(log)
                total_kwh += model_kwh
                total_solar += solar_kwh

        # 2. Future Hours (Forecast)
        # Use ForecastManager to predict remaining energy with ignore_aux=True
        # This ensures future part is also "Pure Model"
        future_kwh, future_solar, _ = self.coordinator.forecast.calculate_future_energy(
            dt_util.now(), ignore_aux=True
        )

        total_kwh += future_kwh
        total_solar += future_solar

        return total_kwh, total_solar

    def calculate_hybrid_projection(self, start_date: date, end_date: date) -> tuple[float, float]:
        """Calculate hybrid projection: Past(Actual) + Today(Budget) + Future(Forecast/Fallback).

        Generalized for arbitrary date ranges (Week/Month).

        Returns:
            (total_kwh, total_solar_impact_kwh)
        """
        now = dt_util.now()
        today = now.date()

        total_kwh = 0.0
        total_solar = 0.0

        # Initialize Inertia Tracking for Future Projections
        # We need "End of Today" inertia to seed "Tomorrow".
        # 1. Start with "Now" inertia (Actuals)
        # Use centralized helper to avoid duplication and ensure consistency
        running_inertia = self.coordinator._get_inertia_list(now)

        # If today is included in the range (or passed), we must simulate Today's forecast
        # to get the "End of Today" inertia for Tomorrow.
        # This is a bit computationally redundant with "today_model" calculation below,
        # but necessary for correct thermodynamic bridging to future days.
        # We only do this if we actually need to project into the future.
        has_future_days = end_date > today
        if has_future_days:
            # Run simulation for Today to get final inertia
            prediction_today = self.coordinator.forecast.get_future_day_prediction(today, running_inertia, ignore_aux=False)
            if prediction_today:
                _, _, w_stats_today = prediction_today
                if "final_inertia" in w_stats_today:
                    running_inertia = w_stats_today["final_inertia"]

        past_end_date = min(end_date, today - timedelta(days=1))

        if start_date <= past_end_date:
            p_kwh, p_solar, _, _, _ = self.calculate_modeled_energy(start_date, past_end_date)
            total_kwh += p_kwh
            total_solar += p_solar

            current = past_end_date + timedelta(days=1)
        else:
            current = start_date

        while current <= end_date:
            if current == today:
                # Use Pure Model calculation for Today to ensure consistent comparison
                # (Model vs Model, not Actual vs Model)
                today_model, today_solar = self._calculate_pure_model_today()
                total_kwh += today_model
                total_solar += today_solar

            elif current > today:
                # FORCE IGNORE AUX FOR FUTURE DAYS
                # Pass running_inertia to ensure thermodynamic continuity
                prediction = self.coordinator.forecast.get_future_day_prediction(
                    current, initial_inertia=running_inertia, ignore_aux=True
                )

                if prediction:
                    p_kwh, p_solar, w_stats = prediction
                    total_kwh += p_kwh
                    total_solar += p_solar

                    # Update inertia for next iteration
                    if "final_inertia" in w_stats:
                        running_inertia = w_stats["final_inertia"]
                else:
                    ly_date = get_last_year_iso_date(current)

                    ly_kwh, ly_solar, _, _, _ = self.coordinator.calculate_modeled_energy(ly_date, ly_date)
                    total_kwh += ly_kwh
                    total_solar += ly_solar

            current += timedelta(days=1)

        return round(total_kwh, 1), round(total_solar, 1)

    def calculate_deviation_breakdown(self):
        """Calculate deviation breakdown per unit."""
        breakdown = []
        now = dt_util.now()

        processed_logs = self._get_processed_logs(now)

        current_temp_val = self.coordinator._calculate_inertia_temp()
        if current_temp_val is None:
             current_temp_val = self.coordinator._get_float_state(self.coordinator.outdoor_temp_sensor)

        current_eff_wind = self.coordinator.data.get("effective_wind", 0.0)
        current_wind_bucket = self.coordinator._get_wind_bucket(current_eff_wind)
        curr_impact = self.coordinator.data.get(ATTR_SOLAR_IMPACT, 0.0)
        minutes_passed = now.minute

        for entity_id in self.coordinator.energy_sensors:
            actual_so_far = self.coordinator._daily_individual.get(entity_id, 0.0)

            hist_expected, hist_obs_count, hist_hours = self._calculate_historical_expectations(
                entity_id, processed_logs
            )

            curr_expected, curr_obs_count, curr_hours = self._calculate_current_hour_expectations(
                entity_id,
                current_temp_val,
                current_wind_bucket,
                curr_impact,
                minutes_passed
            )

            expected_so_far = hist_expected + curr_expected
            total_obs_count = hist_obs_count + curr_obs_count
            total_hours = hist_hours + curr_hours

            deviation = actual_so_far - expected_so_far
            state = self.coordinator.hass.states.get(entity_id)
            name = state.name if state else entity_id

            if total_hours > 0:
                avg_obs_count = total_obs_count / total_hours
            else:
                avg_obs_count = 0

            confidence = "low"
            if avg_obs_count >= 20:
                confidence = "high"
            elif avg_obs_count >= 5:
                confidence = "medium"

            is_unusual, dev_score, dev_threshold = self._is_deviation_unusual(deviation, expected_so_far, avg_obs_count)

            # Suppress deviation warnings for aux-affected units during cooldown
            # (thermal lag bias makes the base model unreliable for these units)
            cooldown_suppressed = False
            if is_unusual and self.coordinator._aux_cooldown_active:
                affected_set = self.coordinator.aux_affected_entities or []
                if entity_id in affected_set:
                    cooldown_suppressed = True

            if is_unusual and not cooldown_suppressed:
                _LOGGER.warning(
                    f"Unusual Deviation Detected for {name} ({entity_id}): "
                    f"Deviation={deviation:.2f} kWh, Expected={expected_so_far:.2f} kWh, "
                    f"Confidence={confidence} (Avg Obs: {avg_obs_count:.1f}), Score={dev_score:.2f} > {dev_threshold:.2f}"
                )

            breakdown.append({
                "entity_id": entity_id,
                "name": name,
                "deviation": round(deviation, 2),
                "actual": round(actual_so_far, 2),
                "expected": round(expected_so_far, 2),
                "observation_count": round(avg_obs_count, 1),
                "confidence": confidence,
                "unusual": is_unusual,
                "deviation_score": round(dev_score, 3) if dev_score is not None else None,
                "deviation_threshold": dev_threshold,
            })


        breakdown.sort(key=lambda x: abs(x["deviation"]), reverse=True)
        return breakdown

    def _get_processed_logs(self, now_dt: datetime) -> list[dict]:
        """Retrieve and pre-process hourly logs for the current day."""
        today_iso = now_dt.date().isoformat()
        today_logs = []

        for entry in reversed(self.coordinator._hourly_log):
             if entry["timestamp"].startswith(today_iso):
                 today_logs.append(entry)
             else:
                 break
        today_logs.reverse()

        processed_logs = []
        for log in today_logs:
            temp_key = log.get("temp_key", "0")
            wind_bucket = log["wind_bucket"]
            eff_wind = log.get("effective_wind", 0.0)
            global_base = self.coordinator._get_predicted_kwh(temp_key, wind_bucket, log["temp"])

            # Check if log contains exact unit expectation (New format: Mixed Mode Support)
            # If missing (Legacy), we reconstruct it using the robust calculation method (Kelvin Protocol).
            unit_expected_breakdown = log.get("unit_expected_breakdown")

            if not unit_expected_breakdown:
                # Reconstruct for Legacy Logs
                # Infer unit modes from temp (Best effort)
                current_temp = log["temp"]

                # Use temp_key or inertia_temp if available to match historical model lookup
                calc_temp = current_temp
                if "inertia_temp" in log:
                    calc_temp = log["inertia_temp"]
                elif "temp_key" in log:
                    try:
                        calc_temp = float(log["temp_key"])
                    except ValueError:
                        pass

                unit_modes = {}
                for eid in self.coordinator.energy_sensors:
                    unit_modes[eid] = (
                        MODE_HEATING
                        if calc_temp < self.coordinator.balance_point
                        else MODE_COOLING
                    )

                res = self.calculate_total_power(
                    temp=calc_temp,
                    effective_wind=eff_wind,
                    solar_impact=0.0, # Solar handled internally by calculate_total_power using factor
                    is_aux_active=log.get("auxiliary_active", False),
                    unit_modes=unit_modes,
                    override_solar_factor=log.get("solar_factor", 0.0)
                )

                # Extract the breakdown (Net Expected)
                unit_expected_breakdown = {}
                for eid, stats in res["unit_breakdown"].items():
                    unit_expected_breakdown[eid] = stats["net_kwh"]

            processed_logs.append({
                "log": log,
                "global_base": global_base,
                "reconstructed_breakdown": unit_expected_breakdown
            })
        return processed_logs

    def _calculate_historical_expectations(self, entity_id: str, processed_logs: list[dict]) -> tuple[float, int, int]:
        """Calculate expected energy for completed hours based on logs."""
        expected_sum = 0.0
        obs_count_sum = 0
        hours_counted = 0

        for item in processed_logs:
            log = item["log"]
            reconstructed = item.get("reconstructed_breakdown", {})

            temp_key = log.get("temp_key", "0")
            wind_bucket = log["wind_bucket"]

            # Handle Legacy Logs where wind_bucket might be "with_auxiliary_heating"
            if wind_bucket == "with_auxiliary_heating":
                # Fallback to physical bucket derived from effective wind if available
                # Or map to Normal if unknown
                eff_wind_legacy = log.get("effective_wind", 0.0)
                wind_bucket = self.coordinator._get_wind_bucket(eff_wind_legacy)

            count = self.coordinator._get_unit_observation_count(entity_id, temp_key, wind_bucket)
            obs_count_sum += count
            hours_counted += 1

            # Use the pre-processed breakdown (Original or Reconstructed)
            # This handles both modern logs and legacy reconstruction transparently
            unit_expected = reconstructed.get(entity_id, 0.0)
            expected_sum += unit_expected

        return expected_sum, obs_count_sum, hours_counted

    def _calculate_current_hour_expectations(
        self,
        entity_id: str,
        current_temp_val: float | None,
        current_wind_bucket: str,
        curr_impact: float,
        minutes_passed: int
    ) -> tuple[float, int, int]:
        """Calculate expected energy for the current partial hour."""
        if current_temp_val is None:
            return 0.0, 0, 0

        temp_key_curr = str(int(round(current_temp_val)))
        count = self.coordinator._get_unit_observation_count(entity_id, temp_key_curr, current_wind_bucket)

        if entity_id in self.coordinator._hourly_expected_per_unit:
            expected_val = self.coordinator._hourly_expected_per_unit[entity_id]
        else:
            # We need the current effective wind for fallback projection
            eff_wind = self.coordinator.data.get("effective_wind", 0.0)
            expected_val = self._calculate_fallback_projection(
                entity_id, temp_key_curr, current_wind_bucket,
                current_temp_val, minutes_passed, eff_wind
            )

        return expected_val, count, 1

    def _calculate_fallback_projection(
        self,
        entity_id: str,
        temp_key: str,
        wind_bucket: str,
        current_temp: float,
        minutes_passed: int,
        effective_wind: float | None = None
    ) -> float:
        """Calculate fallback projection if intra-hour accumulation is missing."""
        unit_data = self.coordinator._correlation_data_per_unit.get(entity_id, {})
        unit_base_curr = self._get_prediction_from_model(unit_data, temp_key, wind_bucket, current_temp, self.coordinator.balance_point)

        if self.coordinator.solar_enabled:
             curr_solar_factor = self.coordinator.data.get(ATTR_SOLAR_FACTOR, 0.0)
             unit_coeff = self.coordinator.solar.calculate_unit_coefficient(entity_id, temp_key)
             unit_solar_curr_kw = self.coordinator.solar.calculate_unit_solar_impact(curr_solar_factor, unit_coeff)

             mode = MODE_HEATING if current_temp < self.coordinator.balance_point else MODE_COOLING
             unit_rate_curr = self.coordinator.solar.apply_correction(unit_base_curr, unit_solar_curr_kw, mode)
        else:
             unit_rate_curr = unit_base_curr

        return unit_rate_curr * (minutes_passed / 60.0)

    def _is_deviation_unusual(self, deviation_kwh, expected_kwh, obs_count):
        """Determine if a deviation is statistically unusual.

        Returns:
            (is_unusual: bool | None, score: float | None, threshold: float | None)
        """
        if obs_count < DEVIATION_MIN_OBSERVATIONS:
            return None, None, None

        if abs(deviation_kwh) < DEVIATION_MIN_KWH:
            return False, 0.0, None

        # Dynamic Threshold (Continuous Decay):
        # New data gets high tolerance, decaying to mature tolerance over time.
        decay_rate = (DEVIATION_TOLERANCE_NEW - DEVIATION_TOLERANCE_MATURE) / DEVIATION_MATURITY_COUNT
        threshold = max(DEVIATION_TOLERANCE_MATURE, DEVIATION_TOLERANCE_NEW - (obs_count * decay_rate))

        offset = 1.0
        damped_score = abs(deviation_kwh) / (expected_kwh + offset)

        is_unusual = damped_score > threshold
        return is_unusual, damped_score, threshold

    def compare_periods(
        self, p1_start: date, p1_end: date, p2_start: date, p2_end: date
    ) -> dict:
        """Compare two historical periods.

        NOTE: This function returns two types of totals:
        1. 'actual_kwh' (in period_1/2 data): Aggregate of historical logs only.
           This may be incomplete if the period includes today or future dates.
        2. 'hybrid_total_kwh' (top-level): Actionable total including historical actuals,
           current live usage + forecast for today, and pure forecast for future dates.
           This is preferred for "Week Ahead" or ongoing period analysis.
        """

        # === 1. Fetch Aggregates (Existing Logic) ===
        def _get_period_data(start_date: date, end_date: date) -> dict:
            """Fetch and structure data for a single period."""
            actual_kwh = self.calculate_historical_actual_sum(start_date, end_date)
            modeled_kwh, solar_impact, avg_temp, avg_wind, total_tdd = self.calculate_modeled_energy(
                start_date, end_date
            )

            # Sum aux impact from daily history
            aux_impact = 0.0
            current = start_date
            while current <= end_date:
                day_str = current.isoformat()
                entry = self.coordinator._daily_history.get(day_str)
                if entry:
                    aux_impact += entry.get("aux_impact_kwh", 0.0)
                current += timedelta(days=1)

            efficiency = None
            if actual_kwh is not None and total_tdd > ENERGY_GUARD_THRESHOLD:
                efficiency = round(actual_kwh / total_tdd, 3)

            deviation_kwh = (
                (actual_kwh - modeled_kwh) if actual_kwh is not None else None
            )
            deviation_pct = (
                (deviation_kwh / modeled_kwh * 100)
                if deviation_kwh is not None and modeled_kwh > 0.1
                else None
            )

            num_days = (end_date - start_date).days + 1
            avg_daily_tdd = round(total_tdd / num_days, 1) if num_days > 0 else 0.0

            return {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": num_days,
                "actual_kwh": actual_kwh,
                "modeled_kwh": modeled_kwh,
                "deviation_kwh": round(deviation_kwh, 2) if deviation_kwh is not None else None,
                "deviation_pct": round(deviation_pct, 1) if deviation_pct is not None else None,
                "efficiency": efficiency,
                "total_tdd": total_tdd,
                "avg_daily_tdd": avg_daily_tdd,
                "solar_impact_kwh": solar_impact,
                "aux_impact_kwh": round(aux_impact, 2),
                "avg_temp": avg_temp,
                "avg_wind_speed": avg_wind,
            }

        period1_data = _get_period_data(p1_start, p1_end)
        period2_data = _get_period_data(p2_start, p2_end)

        # === 2. Build Detailed Daily Data (New Logic) ===
        def _get_period_daily_data(start_date: date, end_date: date) -> list[dict]:
            """Build daily data list for period with Hybrid logic (Actual + Forecast)."""
            days = []
            current = start_date
            now = dt_util.now()
            today = now.date()

            while current <= end_date:
                day_data = None

                if current < today:
                    # Past: Use daily_history
                    day_str = current.isoformat()
                    entry = self.coordinator._daily_history.get(day_str)

                    if entry:
                        temp = entry.get('temp')
                        wind = entry.get('wind', 0.0)

                        # Fallback Model value if actuals missing
                        model_kwh, solar_kwh, _, _, _ = self.coordinator.calculate_modeled_energy(current, current)

                        actual_kwh = entry.get('kwh')
                        if actual_kwh is None:
                            actual_kwh = model_kwh

                        wind_bucket = self.coordinator._get_wind_bucket(wind) if wind is not None else 'normal'

                        day_data = {
                            'date': day_str,
                            'temp': temp,
                            'wind': wind,
                            'wind_bucket': wind_bucket,
                            'kwh': round(actual_kwh, 2),
                            'solar_kwh': round(solar_kwh, 2)
                        }
                    else:
                        # Missing Data
                        day_data = {
                            'date': day_str,
                            'temp': None,
                            'wind': None,
                            'wind_bucket': None,
                            'kwh': 0.0,
                            'solar_kwh': 0.0
                        }

                elif current == today:
                    # Today: Hybrid (Actual So Far + Forecast Remaining)
                    temp = self.coordinator.data.get(ATTR_TEMP_ACTUAL_TODAY)
                    wind = self.coordinator.data.get(ATTR_WIND_ACTUAL_TODAY)
                    solar = self.coordinator.data.get(ATTR_SOLAR_PREDICTED, 0.0)

                    actual_so_far = self.coordinator.data.get(ATTR_ENERGY_TODAY, 0.0)
                    future_kwh, _, _ = self.coordinator.forecast.calculate_future_energy(now)

                    kwh = actual_so_far + future_kwh
                    wind_bucket = self.coordinator._get_wind_bucket(wind or 0.0)

                    day_data = {
                        'date': current.isoformat(),
                        'temp': temp,
                        'wind': wind,
                        'wind_bucket': wind_bucket,
                        'kwh': round(kwh, 2),
                        'solar_kwh': round(solar, 2)
                    }

                else:
                    # Future: Forecast
                    prediction = self.coordinator.forecast.get_future_day_prediction(current)
                    if prediction:
                        p_kwh, p_solar, w_stats = prediction
                        day_data = {
                            'date': current.isoformat(),
                            'temp': w_stats.get('temp'),
                            'wind': w_stats.get('wind'),
                            'wind_bucket': self.coordinator._get_wind_bucket(w_stats.get('wind', 0.0)),
                            'kwh': p_kwh,
                            'solar_kwh': p_solar
                        }
                    else:
                        # Fallback: Last Year
                        ly_date = get_last_year_iso_date(current)
                        model_kwh, solar_kwh, _, _, _ = self.coordinator.calculate_modeled_energy(ly_date, ly_date)
                        day_data = {
                            'date': current.isoformat(),
                            'temp': None,
                            'wind': None,
                            'wind_bucket': None,
                            'kwh': round(model_kwh, 2),
                            'solar_kwh': round(solar_kwh, 2)
                        }

                if day_data:
                    days.append(day_data)
                current += timedelta(days=1)

            return days

        p1_days = _get_period_daily_data(p1_start, p1_end)
        p2_days = _get_period_daily_data(p2_start, p2_end)

        # === 3. Analyze & Explain (New Logic) ===
        # Use existing aggregates for totals, but allow hybrid override if needed
        # Actually, let's pass the aggregates we calculated in Step 1 to keep consistency
        # BUT if P1 is hybrid, the 'actual_kwh' in period1_data might be None or partial.

        # Calculate Hybrid Totals from Daily Data
        p1_hybrid_kwh = sum(d['kwh'] for d in p1_days)
        p2_hybrid_kwh = sum(d['kwh'] for d in p2_days)

        # Determine which totals to use for analysis
        # If P1 is strictly past, period1_data['actual_kwh'] matches p1_hybrid_kwh (approx)
        # If P1 is hybrid, p1_hybrid_kwh is the "Actionable Total".

        analyzer = WeatherImpactAnalyzer(self.coordinator)
        analysis = analyzer.analyze_period(
            p1_days,
            p2_days,
            'period_comparison',
            current_total_kwh=p1_hybrid_kwh,
            last_year_total_kwh=p2_hybrid_kwh
        )

        formatter = ExplanationFormatter()
        summary_text = formatter.format_period_comparison(analysis)

        # === 4. Calculate Deltas (Existing Logic) ===
        deltas = {}

        def _calc_delta(key, precision=1):
            v1 = period1_data.get(key)
            v2 = period2_data.get(key)
            if v1 is not None and v2 is not None:
                return round(v1 - v2, precision)
            return None

        deltas["delta_actual_kwh"] = _calc_delta("actual_kwh", 2)
        deltas["delta_modeled_kwh"] = _calc_delta("modeled_kwh", 2)
        deltas["delta_solar_impact_kwh"] = _calc_delta("solar_impact_kwh", 2)
        deltas["delta_aux_impact_kwh"] = _calc_delta("aux_impact_kwh", 2)
        deltas["delta_temp"] = _calc_delta("avg_temp", 1)
        deltas["delta_wind"] = _calc_delta("avg_wind_speed", 1)
        deltas["delta_tdd"] = _calc_delta("total_tdd", 1)
        deltas["delta_avg_daily_tdd"] = _calc_delta("avg_daily_tdd", 1)
        deltas["delta_efficiency"] = _calc_delta("efficiency", 3)

        # Cross-comparison: P1 actual vs P2 modeled
        # Useful when P2 has no actuals (e.g. system wasn't running last year)
        p1_actual = period1_data.get("actual_kwh")
        p2_modeled = period2_data.get("modeled_kwh")
        if p1_actual is not None and p2_modeled and p2_modeled > 0.1:
            cross_delta = round(p1_actual - p2_modeled, 2)
            cross_pct = round((cross_delta / p2_modeled) * 100, 1)
            deltas["actual_vs_reference_model_kwh"] = cross_delta
            deltas["actual_vs_reference_model_pct"] = cross_pct

        # === 5. Return Expanded Result ===
        return {
            "period_1": period1_data,
            "period_2": period2_data,
            **deltas,
            # New Analytical Fields
            "summary": summary_text,
            "drivers": analysis.get('drivers', []),
            "characterization": analysis.get('characterization'),
            "hybrid_total_kwh": round(p1_hybrid_kwh, 1),
            "hybrid_reference_kwh": round(p2_hybrid_kwh, 1)
        }
