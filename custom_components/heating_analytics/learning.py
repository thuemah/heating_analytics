"""Learning Manager Service."""
from __future__ import annotations

import logging
from typing import Callable

from .const import (
    ENERGY_GUARD_THRESHOLD,
    LEARNING_BUFFER_THRESHOLD,
    MODE_HEATING,
    MODE_COOLING,
    MODE_OFF,
    MODE_GUEST_HEATING,
    MODE_GUEST_COOLING,
    PER_UNIT_LEARNING_RATE_CAP,
    SOLAR_COEFF_CAP,
)

_LOGGER = logging.getLogger(__name__)

class LearningManager:
    """Manages machine learning logic for Heating Analytics."""

    def process_learning(
        self,
        # Inputs
        temp_key: str,
        wind_bucket: str,
        avg_temp: float,
        total_energy_kwh: float,
        base_expected_kwh: float,
        solar_impact: float,
        avg_solar_factor: float,
        is_aux_active: bool,
        aux_impact: float,
        # Configuration
        learning_enabled: bool,
        solar_enabled: bool,
        learning_rate: float,
        balance_point: float,
        energy_sensors: list[str],
        # State (Mutable)
        hourly_bucket_counts: dict,
        hourly_sample_count: int,
        correlation_data: dict,
        correlation_data_per_unit: dict,
        aux_coefficients: dict,
        learning_buffer_global: dict,
        learning_buffer_per_unit: dict,
        observation_counts: dict,
        hourly_delta_per_unit: dict,
        hourly_expected_per_unit: dict,
        hourly_expected_base_per_unit: dict,
        aux_coefficients_per_unit: dict,
        learning_buffer_aux_per_unit: dict,
        # NEW State Arguments for Unit Solar Learning
        solar_coefficients_per_unit: dict,
        learning_buffer_solar_per_unit: dict,
        # Services / Callbacks
        solar_calculator,  # Passing the instance
        get_predicted_unit_base_fn: Callable[[str, str, str], float],
        # Mode Control
        unit_modes: dict = {},
        # Aux Control
        aux_affected_entities: list[str] | None = None,
        # Guest Mode Flag
        has_guest_activity: bool = False,
        # Cooldown State
        is_cooldown_active: bool = False,
    ) -> dict:
        """Process learning for the completed hour.

        Returns a dictionary with learning results for logging:
        {
            "model_updated": bool,
            "model_base_before": float,
            "model_base_after": float,
            "aux_model_updated": bool,
            "aux_model_before": float,
            "aux_model_after": float,
            "learning_status": str
        }
        """
        # Aux mode determined by is_aux_active from coordinator

        if hourly_sample_count == 0:
            return {
                "model_updated": False,
                "model_base_before": base_expected_kwh,
                "model_base_after": base_expected_kwh,
                "aux_model_updated": False,
                "aux_model_before": None,
                "aux_model_after": None,
                "learning_status": "skipped_no_data",
            }

        model_base_before = base_expected_kwh
        model_base_after = base_expected_kwh
        model_updated = False

        aux_model_before = None
        aux_model_after = None
        aux_model_updated = False

        learning_status = "unknown"
        should_run_per_unit = False

        if not learning_enabled:
            learning_status = "disabled"
        elif is_cooldown_active:
            # COOLDOWN LOCK: Global Base Model is frozen.
            # Only non-affected units will learn (in per-unit step).
            model_updated = False
            should_run_per_unit = True # Allow per-unit (selective)
            learning_status = "cooldown_post_aux"
            _LOGGER.debug(f"Learning Cooldown Active: Global Base Model locked (T={temp_key} W={wind_bucket})")
        else:
            model_updated = True
            should_run_per_unit = True
            learning_status = "active"

        if model_updated:
            # --- Step 1: Learn Models ---

            # Determine Global Mode for aggregate normalization
            # Fallback to temp-based inference for global stats
            global_mode = MODE_HEATING if avg_temp < balance_point else MODE_COOLING

            if solar_enabled:
                normalized_actual = solar_calculator.normalize_for_learning(total_energy_kwh, solar_impact, global_mode)
            else:
                normalized_actual = total_energy_kwh

            if is_aux_active:
                # --- AUX ACTIVE MODE ---
                # Global Base Model is LOCKED. This is intentional:
                # 1. Global Base is the reference for calculating aux_impact
                # 2. Changing it mid-aux would create circular dependency
                # 3. Excluded per-unit models CAN still learn (see _process_per_unit_learning)
                #    but Global does NOT need to sync with them - they are independent
                # 4. Any difference between Global and sum-of-units is handled by
                #    the Aux Coefficient and built-in model inertia
                #
                # Only update Global Aux Coefficient here.

                # Guest Mode Guard: Skip aux learning if guest units are active
                # Guest consumption pollutes the actual energy, making aux learning inaccurate
                # Base and solar learning can continue as they use guest-excluded energy
                if has_guest_activity:
                    _LOGGER.info(f"Aux learning skipped due to guest mode activity (T={temp_key} W={wind_bucket})")
                    learning_status = "aux_skipped_guest_mode"
                    # Don't update aux model - keep current values
                    if temp_key in aux_coefficients and wind_bucket in aux_coefficients[temp_key]:
                        aux_model_before = aux_coefficients[temp_key][wind_bucket]
                        aux_model_after = aux_model_before
                    # Skip to per-unit learning
                else:
                    # Expected consumption IF Aux was contributing perfectly:
                    # Expected = Base - Aux_Impact
                    # Actual is normalized_actual.

                    # We want to learn Aux_Impact.
                    # Implied Aux Impact = Base - Actual.
                    # Example: Base=10, Actual=7 -> Implied Aux=3.
                    implied_aux_kw = base_expected_kwh - normalized_actual

                    # Capture Aux Before state
                    # Aux coefficients are now nested by wind bucket
                    if temp_key not in aux_coefficients:
                        aux_coefficients[temp_key] = {}

                    # Retrieve current coefficient for this wind bucket
                    if wind_bucket in aux_coefficients[temp_key]:
                        current_aux_coeff = aux_coefficients[temp_key][wind_bucket]
                    else:
                        # New bucket: Seed with fallback value (Normal/High) to avoid drop
                        # This prevents the "Forgot old category" bug where a new bucket starts at 0.0
                        current_aux_coeff = 0.0
                        bucket_data = aux_coefficients[temp_key]

                        if wind_bucket == "extreme_wind":
                            if "high_wind" in bucket_data:
                                current_aux_coeff = bucket_data["high_wind"]
                            elif "normal" in bucket_data:
                                current_aux_coeff = bucket_data["normal"]
                        elif wind_bucket == "high_wind":
                            if "normal" in bucket_data:
                                current_aux_coeff = bucket_data["normal"]

                    aux_model_before = current_aux_coeff

                    # Guard against noise
                    if abs(implied_aux_kw) > ENERGY_GUARD_THRESHOLD:
                        # Use provided global learning rate
                        new_aux_coeff = current_aux_coeff + learning_rate * (implied_aux_kw - current_aux_coeff)
                        # Clamp: Aux impact is always non-negative (auxiliary *heating* reduces consumption)
                        new_aux_coeff = max(0.0, new_aux_coeff)

                        # Update specific wind bucket
                        aux_coefficients[temp_key][wind_bucket] = round(new_aux_coeff, 3)

                        aux_model_after = aux_coefficients[temp_key][wind_bucket]
                        aux_model_updated = True

                        learning_status = f"active_aux_update ({new_aux_coeff:.2f}kW)"
                    else:
                        aux_model_after = current_aux_coeff
                        learning_status = "active_aux_no_change"

            else:
                # --- NORMAL MODE ---
                # Update Base Model
                if base_expected_kwh == 0.0:
                    # Cold Start: Use Buffered Learning
                    if temp_key not in learning_buffer_global:
                        learning_buffer_global[temp_key] = {}
                    if wind_bucket not in learning_buffer_global[temp_key]:
                        learning_buffer_global[temp_key][wind_bucket] = []

                    buffer_list = learning_buffer_global[temp_key][wind_bucket]
                    buffer_list.append(normalized_actual)

                    if len(buffer_list) >= LEARNING_BUFFER_THRESHOLD:
                        avg_val = sum(buffer_list) / len(buffer_list)
                        new_base_prediction = avg_val
                        _LOGGER.info(f"Global Buffered Learning [Jump Start]: T={temp_key} W={wind_bucket} -> {new_base_prediction:.3f} kWh (Avg of {len(buffer_list)} samples)")
                        if temp_key not in correlation_data:
                            correlation_data[temp_key] = {}
                        correlation_data[temp_key][wind_bucket] = round(new_base_prediction, 5)
                        buffer_list.clear()
                    else:
                        _LOGGER.debug(f"Global Buffered Learning [Collecting]: T={temp_key} W={wind_bucket} -> Sample {len(buffer_list)}/{LEARNING_BUFFER_THRESHOLD} ({normalized_actual:.3f} kWh)")
                        # While buffering, keep prediction at 0 until jump start
                        new_base_prediction = 0.0

                else:
                    # EMA Update
                    new_base_prediction = base_expected_kwh + learning_rate * (normalized_actual - base_expected_kwh)
                    if temp_key not in correlation_data:
                        correlation_data[temp_key] = {}
                    correlation_data[temp_key][wind_bucket] = round(new_base_prediction, 5)

                model_base_after = new_base_prediction

        # Update Per-Unit Models (Both Normal and Aux modes)
        # We run this even in cooldown (for non-affected units)
        if should_run_per_unit:
            self._process_per_unit_learning(
                temp_key, wind_bucket, avg_temp,
                avg_solar_factor, # Replaced solar_impact with factor for recalculation
                total_energy_kwh, base_expected_kwh,
                energy_sensors, hourly_delta_per_unit,
                solar_enabled, learning_rate,
                solar_calculator, get_predicted_unit_base_fn,
                learning_buffer_per_unit, correlation_data_per_unit, observation_counts,
                is_aux_active, aux_coefficients_per_unit, learning_buffer_aux_per_unit,
                solar_coefficients_per_unit, learning_buffer_solar_per_unit, balance_point,
                unit_modes, # Pass unit modes
                hourly_expected_per_unit,
                hourly_expected_base_per_unit,
                aux_affected_entities, # Pass aux exclusion list
                is_cooldown_active=is_cooldown_active,
            )

        return {
            "model_updated": model_updated,
            "model_base_before": model_base_before,
            "model_base_after": model_base_after,
            "aux_model_updated": aux_model_updated,
            "aux_model_before": aux_model_before,
            "aux_model_after": aux_model_after,
            "learning_status": learning_status,
        }

    def _process_per_unit_learning(
        self,
        temp_key: str,
        wind_bucket: str,
        avg_temp: float,
        avg_solar_factor: float,
        total_energy_kwh: float,
        base_expected_kwh: float,
        energy_sensors: list[str],
        hourly_delta_per_unit: dict,
        solar_enabled: bool,
        learning_rate: float,
        solar_calculator,
        get_predicted_unit_base_fn,
        learning_buffer_per_unit: dict,
        correlation_data_per_unit: dict,
        observation_counts: dict,
        is_aux_active: bool,
        aux_coefficients_per_unit: dict,
        learning_buffer_aux_per_unit: dict,
        solar_coefficients_per_unit: dict,
        learning_buffer_solar_per_unit: dict,
        balance_point: float,
        unit_modes: dict,
        hourly_expected_per_unit: dict,
        hourly_expected_base_per_unit: dict,
        aux_affected_entities: list[str] | None,
        is_cooldown_active: bool = False,
    ):
        """Process learning for individual units."""
        for entity_id in energy_sensors:
            unit_mode = unit_modes.get(entity_id, MODE_HEATING)
            if unit_mode in (MODE_OFF, MODE_GUEST_HEATING, MODE_GUEST_COOLING):
                # Skip learning for non-tracked/temporary modes
                continue

            # Check Cooldown Exclusion
            # If cooldown is active, we SKIP learning for units that were affected by Aux (thermal lag bias)
            # Units NOT affected by Aux can continue to learn normally.
            if is_cooldown_active:
                is_affected = True
                if aux_affected_entities is not None:
                    if entity_id not in aux_affected_entities:
                        is_affected = False

                if is_affected:
                    # Skip this unit
                    continue

            # Only update model if sensor actually reported during this hour
            if entity_id not in hourly_delta_per_unit:
                # Sensor was offline/unavailable entire hour - skip learning
                continue

            actual_unit = hourly_delta_per_unit[entity_id]
            # Dual-Track Learning: prefer the per-unit baseline from Track B calculations
            if entity_id in hourly_expected_base_per_unit:
                expected_unit_base = hourly_expected_base_per_unit[entity_id]
            else:
                expected_unit_base = get_predicted_unit_base_fn(entity_id, temp_key, wind_bucket, avg_temp)
            unit_mode = unit_modes.get(entity_id, MODE_HEATING)

            # Step 1: Learn Unit Solar (if enabled, sunny, and NOT aux)
            # We learn solar in Normal mode to establish the relationship.
            # Solar Formula: Unit_Solar_Impact = Global_Factor * Unit_Coeff
            # Unit_Coeff = (Unit_Base - Unit_Actual) / Global_Factor (for Heating)

            unit_solar_impact = 0.0

            if solar_enabled and avg_solar_factor > 0.1 and not is_aux_active:
                self._learn_unit_solar_coefficient(
                    entity_id, temp_key,
                    expected_unit_base, actual_unit, avg_solar_factor,
                    learning_rate, solar_coefficients_per_unit, learning_buffer_solar_per_unit,
                    avg_temp, balance_point,
                    unit_mode
                )

            # Step 2: Calculate Solar Impact using (possibly updated) coefficients
            if solar_enabled:
                 unit_coeff = solar_calculator.calculate_unit_coefficient(entity_id, temp_key)
                 unit_solar_impact = solar_calculator.calculate_unit_solar_impact(avg_solar_factor, unit_coeff)
                 # Use unit_mode for normalization
                 unit_normalized = solar_calculator.normalize_for_learning(actual_unit, unit_solar_impact, unit_mode)
            else:
                 unit_normalized = actual_unit

            # Step 3: Learn Base or Aux Model
            if is_aux_active:
                # Learn Individual Aux Reduction (kW)
                # Check Exclusion: Only learn if unit is affected by aux
                is_affected = True
                if aux_affected_entities is not None:
                    if entity_id not in aux_affected_entities:
                        is_affected = False

                if is_affected:
                    self._learn_unit_aux_coefficient(
                        entity_id, temp_key, wind_bucket,
                        expected_unit_base, unit_normalized,
                        learning_rate,  # Use global rate
                        aux_coefficients_per_unit, learning_buffer_aux_per_unit,
                        correlation_data_per_unit
                    )
                else:
                    # Excluded unit: Not affected by aux, so learn Base Model normally.
                    # Note: This does NOT require Global Base to update - Global is locked
                    # during aux (see process_learning). The models are independent.
                    self._learn_unit_model(
                        entity_id, temp_key, wind_bucket,
                        expected_unit_base, unit_normalized,
                        learning_rate,
                        learning_buffer_per_unit, correlation_data_per_unit, observation_counts
                    )
            else:
                # Learn Normal Model
                self._learn_unit_model(
                    entity_id, temp_key, wind_bucket,
                    expected_unit_base, unit_normalized,
                    learning_rate,
                    learning_buffer_per_unit, correlation_data_per_unit, observation_counts
                )

    def _learn_unit_solar_coefficient(
        self,
        entity_id: str,
        temp_key: str,
        expected_unit_base: float,
        actual_unit: float,
        avg_solar_factor: float,
        learning_rate: float,
        solar_coefficients_per_unit: dict,
        learning_buffer_solar_per_unit: dict,
        avg_temp: float,
        balance_point: float,
        unit_mode: str,
    ):
        """Update solar coefficient for a specific unit (Buffered or EMA).

        Formula: Impact = Base - Actual (heating) or Actual - Base (cooling).
        Coefficient = Impact / Solar Factor.
        """

        impact = 0.0
        if unit_mode == MODE_HEATING:
            # Heating: Sun reduces consumption
            impact = expected_unit_base - actual_unit
        elif unit_mode == MODE_COOLING:
            # Cooling: Sun increases consumption
            impact = actual_unit - expected_unit_base
        else:
            # OFF or unknown: Cannot learn solar coefficient
            return

        # Negative impact (consumed MORE than base despite sun) is clamped to 0.
        # This is intentional:
        # 1. Negative solar impact is physically impossible
        # 2. Near balance point, heating/cooling modes alternate - negative impacts
        #    have opposite meanings in each mode, causing oscillation without clamping
        # 3. Clamping provides noise filtering; coefficient still decreases via EMA
        #    when implied (0) < current, just more gradually

        if avg_solar_factor <= 0.01:
            return

        implied_coeff = impact / avg_solar_factor
        implied_coeff = max(0.0, implied_coeff)
        implied_coeff = min(SOLAR_COEFF_CAP, implied_coeff)

        # Get Current Coefficient
        current_coeff = None
        if entity_id in solar_coefficients_per_unit:
            if temp_key in solar_coefficients_per_unit[entity_id]:
                current_coeff = solar_coefficients_per_unit[entity_id][temp_key]

        # --- Buffered Learning Logic ---
        if current_coeff is None:
            # Cold Start
            if entity_id not in learning_buffer_solar_per_unit:
                learning_buffer_solar_per_unit[entity_id] = {}
            if temp_key not in learning_buffer_solar_per_unit[entity_id]:
                learning_buffer_solar_per_unit[entity_id][temp_key] = []

            buffer_list = learning_buffer_solar_per_unit[entity_id][temp_key]
            buffer_list.append(implied_coeff)

            if len(buffer_list) >= LEARNING_BUFFER_THRESHOLD:
                avg_val = sum(buffer_list) / len(buffer_list)
                new_coeff = avg_val

                _LOGGER.info(f"Buffered Unit Solar Learning [Jump Start]: {entity_id} T={temp_key} -> {new_coeff:.3f} (Avg of {len(buffer_list)})")
                self._update_unit_solar_coefficient(entity_id, temp_key, new_coeff, solar_coefficients_per_unit)
                buffer_list.clear()
            else:
                 _LOGGER.debug(f"Buffered Unit Solar [Collecting]: {entity_id} T={temp_key} -> Sample {len(buffer_list)}/{LEARNING_BUFFER_THRESHOLD} ({implied_coeff:.3f})")

        else:
            # Post-Jump Start: EMA
            # Cap per-unit learning rate at 3% to prevent oscillation on high-hysteresis units
            unit_learning_rate = min(learning_rate, PER_UNIT_LEARNING_RATE_CAP)
            new_coeff = current_coeff + unit_learning_rate * (implied_coeff - current_coeff)
            new_coeff = max(0.0, new_coeff)

            _LOGGER.debug(f"Per-Unit Solar Learning [EMA]: {entity_id} T={temp_key} -> {new_coeff:.3f} (was {current_coeff:.3f}, rate={unit_learning_rate:.1%})")
            self._update_unit_solar_coefficient(entity_id, temp_key, new_coeff, solar_coefficients_per_unit)

    def _learn_unit_model(
        self,
        entity_id: str,
        temp_key: str,
        wind_bucket: str,
        expected_unit_base: float,
        unit_normalized: float,
        learning_rate: float,
        learning_buffer_per_unit: dict,
        correlation_data_per_unit: dict,
        observation_counts: dict
    ):
        """Update correlation model for a specific unit (Buffered or EMA)."""

        # Check if we have actual learned data for this exact temp/wind combination
        has_exact_model = (
            entity_id in correlation_data_per_unit and
            temp_key in correlation_data_per_unit[entity_id] and
            wind_bucket in correlation_data_per_unit[entity_id][temp_key]
        )

        # --- Buffered Learning Logic (Cold Start Only) ---
        if not has_exact_model:
            # Cold Start Phase: Collect samples before initializing model
            # This ensures we don't create phantom data from wind fallbacks or TDD extrapolation
            if entity_id not in learning_buffer_per_unit:
                learning_buffer_per_unit[entity_id] = {}
            if temp_key not in learning_buffer_per_unit[entity_id]:
                learning_buffer_per_unit[entity_id][temp_key] = {}
            if wind_bucket not in learning_buffer_per_unit[entity_id][temp_key]:
                learning_buffer_per_unit[entity_id][temp_key][wind_bucket] = []

            buffer_list = learning_buffer_per_unit[entity_id][temp_key][wind_bucket]
            buffer_list.append(unit_normalized)

            # Check Threshold for Jump Start
            if len(buffer_list) >= LEARNING_BUFFER_THRESHOLD:
                avg_val = sum(buffer_list) / len(buffer_list)
                new_pred_unit = avg_val

                _LOGGER.info(f"Buffered Learning [Jump Start]: {entity_id} T={temp_key} W={wind_bucket} -> {new_pred_unit:.3f} kWh (Avg of {len(buffer_list)} samples)")

                self._update_unit_correlation(entity_id, temp_key, wind_bucket, new_pred_unit, correlation_data_per_unit)
                self._increment_observation_count(entity_id, temp_key, wind_bucket, observation_counts)

                # Clear buffer (no longer needed after jump start)
                buffer_list.clear()
            else:
                _LOGGER.debug(f"Buffered Learning [Collecting]: {entity_id} T={temp_key} W={wind_bucket} -> Sample {len(buffer_list)}/{LEARNING_BUFFER_THRESHOLD} ({unit_normalized:.3f} kWh)")

        else:
            # Post-Jump Start: Normal hourly EMA updates (no buffering)
            # Only runs if we have actual learned data for this temp/wind combo
            # Get the actual learned value (not fallback)
            current_model_val = correlation_data_per_unit[entity_id][temp_key][wind_bucket]

            # Cap per-unit learning rate at 3% to prevent oscillation on high-hysteresis units
            unit_learning_rate = min(learning_rate, PER_UNIT_LEARNING_RATE_CAP)
            new_pred_unit = current_model_val + unit_learning_rate * (unit_normalized - current_model_val)

            _LOGGER.debug(f"Per-Unit Learning [EMA]: {entity_id} T={temp_key} W={wind_bucket} -> {new_pred_unit:.3f} kWh (was {current_model_val:.3f}, rate={unit_learning_rate:.1%})")

            self._update_unit_correlation(entity_id, temp_key, wind_bucket, new_pred_unit, correlation_data_per_unit)
            self._increment_observation_count(entity_id, temp_key, wind_bucket, observation_counts)

    def _learn_unit_aux_coefficient(
        self,
        entity_id: str,
        temp_key: str,
        wind_bucket: str,
        expected_unit_base: float,
        unit_normalized: float,
        learning_rate: float,
        aux_coefficients_per_unit: dict,
        learning_buffer_aux_per_unit: dict,
        correlation_data_per_unit: dict,
    ):
        """Update aux coefficient for a specific unit (Buffered or EMA)."""
        # Calculate Implied Reduction
        # Reduction = Base - Actual
        # If Base=0 (not learned yet), we can't learn reduction reliably (thermodynamically undefined).
        if expected_unit_base <= ENERGY_GUARD_THRESHOLD:
            return

        # CRITICAL: Only learn aux if we have actual base model data for this temp/wind
        # Don't learn aux based on fallback/extrapolated base values
        has_base_model = (
            entity_id in correlation_data_per_unit and
            temp_key in correlation_data_per_unit[entity_id] and
            wind_bucket in correlation_data_per_unit[entity_id][temp_key]
        )
        if not has_base_model:
            _LOGGER.debug(f"Skipping Unit Aux Learning: {entity_id} T={temp_key} W={wind_bucket} - No base model yet")
            return

        implied_reduction = expected_unit_base - unit_normalized
        # Clamp: Cannot be negative (Aux shouldn't INCREASE usage)
        implied_reduction = max(0.0, implied_reduction)

        # Get the actual base model value for this exact bucket (for clamping)
        base_model_value = correlation_data_per_unit[entity_id][temp_key][wind_bucket]
        # Clamp: Aux reduction cannot exceed base model (can't reduce below zero consumption)
        implied_reduction = min(implied_reduction, base_model_value)

        # Get Current Coefficient
        current_coeff = None
        if entity_id in aux_coefficients_per_unit:
            if temp_key in aux_coefficients_per_unit[entity_id]:
                if wind_bucket in aux_coefficients_per_unit[entity_id][temp_key]:
                    current_coeff = aux_coefficients_per_unit[entity_id][temp_key][wind_bucket]

        # --- Buffered Learning Logic (Cold Start) ---
        if current_coeff is None:
            # Cold Start Phase
            if entity_id not in learning_buffer_aux_per_unit:
                learning_buffer_aux_per_unit[entity_id] = {}
            if temp_key not in learning_buffer_aux_per_unit[entity_id]:
                learning_buffer_aux_per_unit[entity_id][temp_key] = {}
            if wind_bucket not in learning_buffer_aux_per_unit[entity_id][temp_key]:
                learning_buffer_aux_per_unit[entity_id][temp_key][wind_bucket] = []

            buffer_list = learning_buffer_aux_per_unit[entity_id][temp_key][wind_bucket]
            buffer_list.append(implied_reduction)

            if len(buffer_list) >= LEARNING_BUFFER_THRESHOLD:
                avg_val = sum(buffer_list) / len(buffer_list)
                new_coeff = min(avg_val, base_model_value)  # Clamp to base model

                _LOGGER.info(f"Buffered Unit Aux Learning [Jump Start]: {entity_id} T={temp_key} W={wind_bucket} -> {new_coeff:.3f} kW Reduction")
                self._update_unit_aux_coefficient(entity_id, temp_key, wind_bucket, new_coeff, aux_coefficients_per_unit)
                buffer_list.clear()
            else:
                 _LOGGER.debug(f"Buffered Unit Aux [Collecting]: {entity_id} T={temp_key} W={wind_bucket} -> Sample {len(buffer_list)}/{LEARNING_BUFFER_THRESHOLD} ({implied_reduction:.3f} kW)")

        else:
            # Post-Jump Start: EMA using global learning rate
            # Cap per-unit learning rate at 3% to prevent oscillation on high-hysteresis units
            unit_learning_rate = min(learning_rate, PER_UNIT_LEARNING_RATE_CAP)
            new_coeff = current_coeff + unit_learning_rate * (implied_reduction - current_coeff)
            new_coeff = max(0.0, new_coeff)
            new_coeff = min(new_coeff, base_model_value)  # Clamp to base model

            _LOGGER.debug(f"Per-Unit Aux Learning [EMA]: {entity_id} T={temp_key} W={wind_bucket} -> {new_coeff:.3f} kW (was {current_coeff:.3f}, rate={unit_learning_rate:.1%})")
            self._update_unit_aux_coefficient(entity_id, temp_key, wind_bucket, new_coeff, aux_coefficients_per_unit)

    def _update_unit_correlation(self, entity_id, temp_key, wind_bucket, value, correlation_data_per_unit):
        """Update the correlation data structure."""
        if entity_id not in correlation_data_per_unit:
            correlation_data_per_unit[entity_id] = {}
        if temp_key not in correlation_data_per_unit[entity_id]:
            correlation_data_per_unit[entity_id][temp_key] = {}

        correlation_data_per_unit[entity_id][temp_key][wind_bucket] = round(value, 5)

    def _update_unit_aux_coefficient(self, entity_id, temp_key, wind_bucket, value, aux_coefficients_per_unit):
        """Update the aux coefficient data structure."""
        if entity_id not in aux_coefficients_per_unit:
            aux_coefficients_per_unit[entity_id] = {}
        if temp_key not in aux_coefficients_per_unit[entity_id]:
            aux_coefficients_per_unit[entity_id][temp_key] = {}

        aux_coefficients_per_unit[entity_id][temp_key][wind_bucket] = round(value, 3)

    def _update_unit_solar_coefficient(self, entity_id, temp_key, value, solar_coefficients_per_unit):
        """Update the solar coefficient data structure."""
        if entity_id not in solar_coefficients_per_unit:
            solar_coefficients_per_unit[entity_id] = {}

        solar_coefficients_per_unit[entity_id][temp_key] = round(value, 5)

    def _increment_observation_count(self, entity_id, temp_key, wind_bucket, observation_counts):
        """Increment observation count."""
        if entity_id not in observation_counts:
            observation_counts[entity_id] = {}
        if temp_key not in observation_counts[entity_id]:
            observation_counts[entity_id][temp_key] = {}
        if wind_bucket not in observation_counts[entity_id][temp_key]:
            observation_counts[entity_id][temp_key][wind_bucket] = 0

        observation_counts[entity_id][temp_key][wind_bucket] += 1

    def learn_from_historical_import(
        self,
        temp_key: str,
        wind_bucket: str,
        actual_kwh: float,
        is_aux_active: bool,
        correlation_data: dict,
        aux_coefficients: dict,
        learning_rate: float,
        get_predicted_kwh_fn: Callable[[str, str, float], float],
        actual_temp: float,
    ) -> str:
        """Process a single historical data point to train global models."""
        normalized_actual = actual_kwh

        if is_aux_active:
            base_prediction = get_predicted_kwh_fn(temp_key, wind_bucket, actual_temp)
            if base_prediction <= ENERGY_GUARD_THRESHOLD:
                return "skipped_no_base_model"

            implied_aux_reduction = base_prediction - normalized_actual
            implied_aux_reduction = max(0.0, implied_aux_reduction)

            if temp_key not in aux_coefficients:
                aux_coefficients[temp_key] = {}
            current_coeff = aux_coefficients[temp_key].get(wind_bucket, 0.0)

            new_coeff = current_coeff + learning_rate * (implied_aux_reduction - current_coeff) if current_coeff != 0.0 else implied_aux_reduction
            aux_coefficients[temp_key][wind_bucket] = round(new_coeff, 3)
            return "updated_aux_model"

        else:
            if temp_key not in correlation_data:
                correlation_data[temp_key] = {}
            current_pred = correlation_data[temp_key].get(wind_bucket, 0.0)

            new_pred = current_pred + learning_rate * (normalized_actual - current_pred) if current_pred != 0.0 else normalized_actual
            correlation_data[temp_key][wind_bucket] = round(new_pred, 5)
            return "updated_base_model"
