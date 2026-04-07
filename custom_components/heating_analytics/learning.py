"""Learning Manager Service."""
from __future__ import annotations

import logging
from typing import Callable

from .const import (
    COLD_START_SOLAR_DAMPING,
    ENERGY_GUARD_THRESHOLD,
    LEARNING_BUFFER_THRESHOLD,
    MODE_HEATING,
    MODE_COOLING,
    MODE_OFF,
    MODE_GUEST_HEATING,
    MODE_GUEST_COOLING,
    MODE_DHW,
    MODES_EXCLUDED_FROM_GLOBAL_LEARNING,
    PER_UNIT_LEARNING_RATE_CAP,
    SOLAR_COEFF_CAP,
)
from .observation import HourlyObservation, ModelState, LearningConfig

_LOGGER = logging.getLogger(__name__)

class LearningManager:
    """Manages machine learning logic for Heating Analytics."""

    def process_learning(
        self,
        obs: HourlyObservation | None = None,
        model: ModelState | None = None,
        config: LearningConfig | None = None,
        **kwargs,
    ) -> dict:
        """Process learning for the completed hour.

        Accepts either the new contract-based interface (obs, model, config)
        or legacy keyword arguments for backward compatibility with tests.

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
        if kwargs:
            # Legacy call path — build contracts from kwargs
            return self._process_learning_legacy(**kwargs)

        # --- Destructure data contracts into local variables ---
        # This preserves all existing code below without changes.
        # Once internal methods are migrated to use obs/model/config
        # directly, these locals can be removed.

        # From HourlyObservation
        temp_key = obs.temp_key
        wind_bucket = obs.wind_bucket
        avg_temp = obs.avg_temp
        total_energy_kwh = obs.learning_energy_kwh
        base_expected_kwh = obs.base_expected_kwh
        solar_impact = obs.effective_solar_impact
        avg_solar_vector = obs.solar_vector
        is_aux_active = obs.is_aux_dominant
        hourly_bucket_counts = obs.bucket_counts
        hourly_sample_count = obs.sample_count
        hourly_delta_per_unit = obs.unit_breakdown
        hourly_expected_per_unit = obs.unit_expected
        hourly_expected_base_per_unit = obs.unit_expected_base
        unit_modes = obs.unit_modes
        is_cooldown_active = obs.was_cooldown_active
        solar_normalization_delta = obs.solar_normalization_delta

        # From ModelState
        correlation_data = model.correlation_data
        correlation_data_per_unit = model.correlation_data_per_unit
        aux_coefficients = model.aux_coefficients
        learning_buffer_global = model.learning_buffer_global
        learning_buffer_per_unit = model.learning_buffer_per_unit
        observation_counts = model.observation_counts
        aux_coefficients_per_unit = model.aux_coefficients_per_unit
        learning_buffer_aux_per_unit = model.learning_buffer_aux_per_unit
        solar_coefficients_per_unit = model.solar_coefficients_per_unit
        learning_buffer_solar_per_unit = model.learning_buffer_solar_per_unit

        # From LearningConfig
        learning_enabled = config.learning_enabled
        solar_enabled = config.solar_enabled
        learning_rate = config.learning_rate
        balance_point = config.balance_point
        energy_sensors = config.energy_sensors
        aux_impact = config.aux_impact
        solar_calculator = config.solar_calculator
        get_predicted_unit_base_fn = config.get_predicted_unit_base_fn
        aux_affected_entities = config.aux_affected_entities
        has_guest_activity = config.has_guest_activity
        per_unit_learning_enabled = config.per_unit_learning_enabled

        # --- Original logic (unchanged) ---

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

        # Resolve per-unit override: when Track B/C owns the global model,
        # the coordinator passes learning_enabled=False but may allow per-unit
        # learning to continue for non-MPC units via per_unit_learning_enabled.
        _per_unit_enabled = per_unit_learning_enabled if per_unit_learning_enabled is not None else learning_enabled

        if not learning_enabled:
            if _per_unit_enabled:
                # Global blocked (Track B/C), per-unit allowed.
                model_updated = False
                should_run_per_unit = True
                learning_status = "disabled_global_only"
            else:
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

            # Saturation-aware solar normalization (#801).
            # The solar_normalization_delta is pre-computed from per-unit
            # saturation-aware solar impacts with correct mode signs:
            #   delta = heating_solar_applied - cooling_solar_applied
            # This eliminates the need for global_mode inference and correctly
            # handles simultaneous heating+cooling units.
            if solar_enabled:
                normalized_actual = max(0.0, total_energy_kwh + solar_normalization_delta)
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
                avg_solar_vector, # Pass 2D vector for per-unit learning
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

    def _process_learning_legacy(self, **kwargs) -> dict:
        """Backward-compatible entry point using flat keyword arguments.

        Builds HourlyObservation, ModelState, and LearningConfig from the
        legacy parameter dict, then delegates to the contract-based path.
        This exists solely to support existing tests; new code should use
        the (obs, model, config) interface.
        """
        from datetime import datetime

        obs = HourlyObservation(
            timestamp=datetime.now(),
            hour=0,
            avg_temp=kwargs["avg_temp"],
            inertia_temp=kwargs["avg_temp"],
            temp_key=kwargs["temp_key"],
            effective_wind=0.0,
            wind_bucket=kwargs["wind_bucket"],
            bucket_counts=kwargs.get("hourly_bucket_counts", {}),
            avg_humidity=None,
            solar_factor=0.0,
            solar_vector=kwargs.get("avg_solar_vector", (0.0, 0.0)),
            solar_impact_raw=kwargs.get("solar_impact", 0.0),
            effective_solar_impact=kwargs.get("solar_impact", 0.0),
            total_energy_kwh=kwargs.get("total_energy_kwh", 0.0),
            learning_energy_kwh=kwargs.get("total_energy_kwh", 0.0),
            guest_impact_kwh=0.0,
            expected_kwh=kwargs.get("base_expected_kwh", 0.0),
            base_expected_kwh=kwargs.get("base_expected_kwh", 0.0),
            unit_breakdown=kwargs.get("hourly_delta_per_unit", {}),
            unit_expected=kwargs.get("hourly_expected_per_unit", {}),
            unit_expected_base=kwargs.get("hourly_expected_base_per_unit", {}),
            aux_impact_kwh=kwargs.get("aux_impact", 0.0),
            aux_fraction=1.0 if kwargs.get("is_aux_active", False) else 0.0,
            is_aux_dominant=kwargs.get("is_aux_active", False),
            sample_count=kwargs.get("hourly_sample_count", 0),
            unit_modes=kwargs.get("unit_modes", {}),
            was_cooldown_active=kwargs.get("is_cooldown_active", False),
        )

        model = ModelState(
            correlation_data=kwargs.get("correlation_data", {}),
            correlation_data_per_unit=kwargs.get("correlation_data_per_unit", {}),
            observation_counts=kwargs.get("observation_counts", {}),
            aux_coefficients=kwargs.get("aux_coefficients", {}),
            aux_coefficients_per_unit=kwargs.get("aux_coefficients_per_unit", {}),
            solar_coefficients_per_unit=kwargs.get("solar_coefficients_per_unit", {}),
            learned_u_coefficient=None,
            learning_buffer_global=kwargs.get("learning_buffer_global", {}),
            learning_buffer_per_unit=kwargs.get("learning_buffer_per_unit", {}),
            learning_buffer_aux_per_unit=kwargs.get("learning_buffer_aux_per_unit", {}),
            learning_buffer_solar_per_unit=kwargs.get("learning_buffer_solar_per_unit", {}),
        )

        config = LearningConfig(
            learning_enabled=kwargs.get("learning_enabled", True),
            solar_enabled=kwargs.get("solar_enabled", False),
            learning_rate=kwargs.get("learning_rate", 0.01),
            balance_point=kwargs.get("balance_point", 17.0),
            energy_sensors=kwargs.get("energy_sensors", []),
            aux_impact=kwargs.get("aux_impact", 0.0),
            solar_calculator=kwargs.get("solar_calculator"),
            get_predicted_unit_base_fn=kwargs.get("get_predicted_unit_base_fn"),
            aux_affected_entities=kwargs.get("aux_affected_entities"),
            has_guest_activity=kwargs.get("has_guest_activity", False),
            per_unit_learning_enabled=kwargs.get("per_unit_learning_enabled"),
        )

        return self.process_learning(obs=obs, model=model, config=config)

    def _process_per_unit_learning(
        self,
        temp_key: str,
        wind_bucket: str,
        avg_temp: float,
        avg_solar_vector: tuple[float, float],
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

            # DHW mode: unit is active but not contributing to space heating.
            # Force actual_unit = 0 so the per-unit model learns the correct
            # zero contribution rather than skipping the update entirely.
            # This also covers heat pump idle/standby cycles mapped to DHW by
            # the heat_pump_mode_sync blueprint.
            if unit_mode == MODE_DHW:
                actual_unit = 0.0
            else:
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

            # Vector magnitude check for "sunny enough" threshold
            s, e = avg_solar_vector
            vector_magnitude = (s**2 + e**2) ** 0.5

            if solar_enabled and vector_magnitude > 0.1 and not is_aux_active:
                self._learn_unit_solar_coefficient(
                    entity_id, temp_key,
                    expected_unit_base, actual_unit, avg_solar_vector,
                    learning_rate, solar_coefficients_per_unit, learning_buffer_solar_per_unit,
                    avg_temp, balance_point,
                    unit_mode
                )

            # Step 2: Calculate Solar Impact using (possibly updated) coefficients
            if solar_enabled:
                 unit_coeff = solar_calculator.calculate_unit_coefficient(entity_id, temp_key)
                 unit_solar_impact = solar_calculator.calculate_unit_solar_impact(avg_solar_vector, unit_coeff)
                 # Use unit_mode for normalization
                 unit_normalized = solar_calculator.normalize_for_learning(actual_unit, unit_solar_impact, unit_mode)
            else:
                 unit_normalized = actual_unit

            # Step 3: Learn Base or Aux Model
            if is_aux_active:
                # DHW + Aux simultaneously: the zero contribution is caused by DHW,
                # not by the aux system. Skip aux-coefficient learning to avoid
                # corrupting the per-unit aux coefficient.
                if unit_mode == MODE_DHW:
                    pass
                else:
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
        avg_solar_vector: tuple[float, float],
        learning_rate: float,
        solar_coefficients_per_unit: dict,
        learning_buffer_solar_per_unit: dict,
        avg_temp: float,
        balance_point: float,
        unit_mode: str,
    ):
        """Update 2D solar coefficient vector for a specific unit (Buffered or EMA)."""
        actual_impact = 0.0
        if unit_mode == MODE_HEATING:
            # Heating: Sun reduces consumption
            actual_impact = expected_unit_base - actual_unit
        elif unit_mode == MODE_COOLING:
            # Cooling: Sun increases consumption
            actual_impact = actual_unit - expected_unit_base
        else:
            # OFF or unknown: Cannot learn solar coefficient
            return

        # Clamping
        actual_impact = max(0.0, actual_impact)

        solar_s, solar_e = avg_solar_vector
        vector_magnitude = (solar_s**2 + solar_e**2) ** 0.5

        if vector_magnitude <= 0.01:
            return

        # Get Current Coefficient Vector (global per unit — solar gain is temperature-independent)
        current_coeff = solar_coefficients_per_unit.get(entity_id)

        # --- Buffered Learning Logic (Cold Start) ---
        if current_coeff is None:
            if entity_id not in learning_buffer_solar_per_unit:
                learning_buffer_solar_per_unit[entity_id] = []

            buffer_list = learning_buffer_solar_per_unit[entity_id]
            # Store tuple of (s, e, impact)
            buffer_list.append((solar_s, solar_e, actual_impact))

            # Need more data for 2D least squares (minimum 2 points, usually want more for stability)
            if len(buffer_list) >= LEARNING_BUFFER_THRESHOLD:
                # Solve 2x2 normal equations
                sum_s2 = sum(item[0]**2 for item in buffer_list)
                sum_e2 = sum(item[1]**2 for item in buffer_list)
                sum_se = sum(item[0] * item[1] for item in buffer_list)
                sum_s_I = sum(item[0] * item[2] for item in buffer_list)
                sum_e_I = sum(item[1] * item[2] for item in buffer_list)

                determinant = (sum_s2 * sum_e2) - (sum_se**2)

                # Full 2D solution when sun angles are diverse enough
                if abs(determinant) > 1e-6:
                    new_coeff_s = ((sum_e2 * sum_s_I) - (sum_se * sum_e_I)) / determinant
                    new_coeff_e = ((sum_s2 * sum_e_I) - (sum_se * sum_s_I)) / determinant

                    new_coeff_s = max(-SOLAR_COEFF_CAP, min(SOLAR_COEFF_CAP, new_coeff_s))
                    new_coeff_e = max(-SOLAR_COEFF_CAP, min(SOLAR_COEFF_CAP, new_coeff_e))

                    new_coeff = {"s": new_coeff_s * COLD_START_SOLAR_DAMPING, "e": new_coeff_e * COLD_START_SOLAR_DAMPING}
                    _LOGGER.info(f"Buffered Unit Solar Learning [Jump Start]: {entity_id} -> {new_coeff} (2D Least Squares, {len(buffer_list)} samples, damping={COLD_START_SOLAR_DAMPING})")
                else:
                    # Collinear fallback: sun observed from a narrow angle range (e.g. winter).
                    # We can only identify the coefficient along the dominant direction.
                    # Project all observations onto that direction and solve 1D LS,
                    # then decompose back. LMS updates will refine both components over time.
                    dir_s = sum(item[0] for item in buffer_list)
                    dir_e = sum(item[1] for item in buffer_list)
                    dir_norm = (dir_s**2 + dir_e**2) ** 0.5

                    if dir_norm < 1e-6:
                        _LOGGER.debug(f"Buffered Unit Solar [Collecting]: {entity_id} -> zero-magnitude vectors, skipping.")
                        return

                    d_s = dir_s / dir_norm
                    d_e = dir_e / dir_norm

                    sum_proj_I = sum((item[0] * d_s + item[1] * d_e) * item[2] for item in buffer_list)
                    sum_proj2 = sum((item[0] * d_s + item[1] * d_e) ** 2 for item in buffer_list)

                    if sum_proj2 < 1e-6:
                        _LOGGER.debug(f"Buffered Unit Solar [Collecting]: {entity_id} -> degenerate projection, skipping.")
                        return

                    c_scalar = max(-SOLAR_COEFF_CAP, min(SOLAR_COEFF_CAP, sum_proj_I / sum_proj2))
                    new_coeff = {"s": c_scalar * d_s * COLD_START_SOLAR_DAMPING, "e": c_scalar * d_e * COLD_START_SOLAR_DAMPING}
                    _LOGGER.info(f"Buffered Unit Solar Learning [Jump Start 1D]: {entity_id} -> {new_coeff} (collinear fallback, dir=({d_s:.2f},{d_e:.2f}), {len(buffer_list)} samples, damping={COLD_START_SOLAR_DAMPING})")

                self._update_unit_solar_coefficient(entity_id, new_coeff, solar_coefficients_per_unit)
                buffer_list.clear()
            else:
                 _LOGGER.debug(f"Buffered Unit Solar [Collecting]: {entity_id} -> Sample {len(buffer_list)}/{LEARNING_BUFFER_THRESHOLD} ({actual_impact:.3f} kW)")

        else:
            # Post-Jump Start: LMS Gradient Descent
            # Cap per-unit learning rate at 3% to prevent oscillation
            unit_learning_rate = min(learning_rate, PER_UNIT_LEARNING_RATE_CAP)

            coeff_s = current_coeff.get("s", 0.0)
            coeff_e = current_coeff.get("e", 0.0)

            predicted_impact = coeff_s * solar_s + coeff_e * solar_e
            error = actual_impact - predicted_impact

            # Gradient update
            new_coeff_s = coeff_s + unit_learning_rate * error * solar_s
            new_coeff_e = coeff_e + unit_learning_rate * error * solar_e

            # Clamp individually for safety, but allow negative components
            new_coeff_s = max(-SOLAR_COEFF_CAP, min(SOLAR_COEFF_CAP, new_coeff_s))
            new_coeff_e = max(-SOLAR_COEFF_CAP, min(SOLAR_COEFF_CAP, new_coeff_e))

            new_coeff = {"s": new_coeff_s, "e": new_coeff_e}

            _LOGGER.debug(f"Per-Unit Solar Learning [LMS EMA]: {entity_id} -> {new_coeff} (was {current_coeff})")
            self._update_unit_solar_coefficient(entity_id, new_coeff, solar_coefficients_per_unit)

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

    def _update_unit_solar_coefficient(self, entity_id, value: dict[str, float], solar_coefficients_per_unit):
        """Update the solar coefficient data structure (global per unit, not temp-stratified).

        Coefficients are clamped to >= 0: solar gain can only reduce heating
        demand, never increase it.  A negative value indicates a confounding
        variable (e.g. a scheduled pre-heating routine that coincides with
        morning sun) and must not propagate into the model.
        """
        solar_coefficients_per_unit[entity_id] = {
            "s": round(max(0.0, value.get("s", 0.0)), 5),
            "e": round(max(0.0, value.get("e", 0.0)), 5)
        }

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

    def apply_strategies_to_global_model(
        self,
        day_logs: list[dict],
        track_c_distribution: list[dict] | None,
        strategies: dict,
        model: "ModelState",
        learning_rate: float,
        balance_point: float,
        wind_threshold: float,
        extreme_wind_threshold: float,
        parse_datetime_fn,
    ) -> int:
        """Apply per-unit strategies to write hourly sums to the global model.

        Each sensor's strategy produces a kWh contribution per hour.
        The global model receives the sum of all contributions for each
        (temp_key, wind_bucket) pair via the standard buffer → jump-start
        → EMA pipeline.

        Moved from coordinator.py (#784) — pure model-writing logic with
        no Home Assistant dependencies.

        Returns the number of bucket updates written.
        """
        from .observation import WeightedSmear

        log_by_hour: dict[int, dict] = {e.get("hour", -1): e for e in day_logs}

        # Clear stale distribution from previous day on all synthetic
        # WeightedSmear strategies.
        for strategy in strategies.values():
            if isinstance(strategy, WeightedSmear) and strategy.use_synthetic:
                strategy.set_distribution(None)

        # Prepare WeightedSmear strategies with their data for this day.
        if track_c_distribution:
            dist_by_hour: dict[int, dict] = {}
            for entry in track_c_distribution:
                try:
                    entry_dt = parse_datetime_fn(entry["datetime"])
                    if entry_dt is not None:
                        dist_by_hour[entry_dt.hour] = entry
                except (KeyError, TypeError, ValueError):
                    continue

            for strategy in strategies.values():
                if isinstance(strategy, WeightedSmear) and strategy.use_synthetic:
                    strategy.set_distribution(dist_by_hour)

        # Compute normalised loss weights from hourly log weather data.
        # Uses abs(delta_t) so cooling hours (outdoor > balance_point) also
        # receive proportional weight (#792).  Solar multiplier is inverted
        # for cooling: sun increases cooling load instead of reducing it.
        raw_weights: list[float] = []
        for h in range(24):
            log_h = log_by_hour.get(h, {})
            inertia_t = log_h.get("inertia_temp")
            raw_t = log_h.get("temp")
            outdoor = inertia_t if inertia_t is not None else (raw_t if raw_t is not None else balance_point)
            delta_t = abs(balance_point - outdoor)
            eff_wind = log_h.get("effective_wind")
            effective_wind = eff_wind if eff_wind is not None else 0.0
            if effective_wind >= extreme_wind_threshold:
                wind_factor = 1.6
            elif effective_wind >= wind_threshold:
                wind_factor = 1.3
            else:
                wind_factor = 1.0
            solar_f = log_h.get("solar_factor")
            solar_f = solar_f if solar_f is not None else 0.0
            is_cooling_hour = outdoor > balance_point
            if is_cooling_hour:
                # Sun adds to cooling load — more sun = more weight
                solar_mult = max(0.1, 1.0 + solar_f * 0.5)
            else:
                # Sun reduces heating load — more sun = less weight
                solar_mult = max(0.0, 1.0 - solar_f)
            raw_weights.append(delta_t * wind_factor * solar_mult)
        total_weight = sum(raw_weights)
        norm_weights = [w / total_weight if total_weight > 0 else 1.0 / 24 for w in raw_weights]

        # Set daily totals for non-MPC WeightedSmear strategies.
        # Mode filtering (#789): only sum hours where the unit is in a
        # learning-eligible mode so excluded energy is not smeared.
        for strategy in strategies.values():
            if isinstance(strategy, WeightedSmear) and not strategy.use_synthetic:
                daily_total = 0.0
                for h in range(24):
                    entry = log_by_hour.get(h, {})
                    unit_modes = entry.get("unit_modes", {})
                    mode = unit_modes.get(strategy.sensor_id, MODE_HEATING)
                    if mode not in MODES_EXCLUDED_FROM_GLOBAL_LEARNING:
                        daily_total += entry.get("unit_breakdown", {}).get(strategy.sensor_id, 0.0)
                strategy.set_daily_total(daily_total)

        # Iterate hours and sum all strategy contributions per bucket.
        correlation_data = model.correlation_data
        learning_buffer = model.learning_buffer_global
        bucket_updates = 0

        for h in range(24):
            log_entry = log_by_hour.get(h, {})
            h_temp_key = log_entry.get("temp_key")
            h_wind_bucket = log_entry.get("wind_bucket")
            if h_temp_key is None or h_wind_bucket is None:
                continue

            weight = norm_weights[h]

            total_kwh = 0.0
            for strategy in strategies.values():
                contrib = strategy.get_hourly_contribution(h, weight, log_entry)
                if contrib is not None:
                    total_kwh += contrib

            # Apply saturation-aware solar normalization (#792).
            # The delta was pre-computed during Track A and stored in the log.
            # dark_actual = actual + (heating_solar_applied - cooling_solar_applied)
            solar_delta = log_entry.get("solar_normalization_delta", 0.0)
            if solar_delta:
                total_kwh = max(0.0, total_kwh + solar_delta)

            if total_kwh <= 0.0:
                continue

            # Buffer → jump-start → EMA.
            if h_temp_key not in correlation_data:
                correlation_data[h_temp_key] = {}
            current_pred = correlation_data[h_temp_key].get(h_wind_bucket, 0.0)

            if current_pred == 0.0:
                if h_temp_key not in learning_buffer:
                    learning_buffer[h_temp_key] = {}
                if h_wind_bucket not in learning_buffer[h_temp_key]:
                    learning_buffer[h_temp_key][h_wind_bucket] = []
                buffer_list = learning_buffer[h_temp_key][h_wind_bucket]
                buffer_list.append(total_kwh)
                if len(buffer_list) >= LEARNING_BUFFER_THRESHOLD:
                    avg_val = sum(buffer_list) / len(buffer_list)
                    correlation_data[h_temp_key][h_wind_bucket] = round(avg_val, 5)
                    buffer_list.clear()
                    bucket_updates += 1
                    _LOGGER.info(f"Strategy Learning [Jump Start]: T={h_temp_key} W={h_wind_bucket} -> {avg_val:.3f} kWh")
                else:
                    _LOGGER.debug(f"Strategy Learning [Buffering]: T={h_temp_key} W={h_wind_bucket} -> Sample {len(buffer_list)}/{LEARNING_BUFFER_THRESHOLD}")
            else:
                new_pred = current_pred + learning_rate * (total_kwh - current_pred)
                correlation_data[h_temp_key][h_wind_bucket] = round(new_pred, 5)
                bucket_updates += 1
                _LOGGER.debug(f"Strategy Learning [EMA]: T={h_temp_key} W={h_wind_bucket} -> {new_pred:.5f} kWh (was {current_pred:.5f})")

        return bucket_updates

    def replay_per_unit_models(
        self,
        day_entries: list[dict],
        strategies: dict,
        model: "ModelState",
        learning_rate: float,
    ) -> None:
        """Replay per-unit correlation models from hourly log entries.

        Each DirectMeter sensor's actual kWh is written to its per-unit
        correlation table via buffer → jump-start → EMA.  Needed so that
        ``isolate_sensor`` subtraction works after ``retrain_from_history(reset_first=True)``.
        WeightedSmear sensors are skipped.

        Moved from coordinator.py (#784) — pure model-writing logic.
        """
        from .observation import DirectMeter

        direct_sensors = [
            s for s in strategies.values()
            if isinstance(s, DirectMeter)
        ]
        if not direct_sensors:
            return

        correlation_per_unit = model.correlation_data_per_unit
        buffer_per_unit = model.learning_buffer_per_unit

        for log_entry in day_entries:
            h_temp_key = log_entry.get("temp_key")
            h_wind_bucket = log_entry.get("wind_bucket")
            if h_temp_key is None or h_wind_bucket is None:
                continue
            breakdown = log_entry.get("unit_breakdown", {})
            for strategy in direct_sensors:
                sid = strategy.sensor_id
                unit_kwh = breakdown.get(sid, 0.0)
                if unit_kwh <= 0.0:
                    continue
                if sid not in correlation_per_unit:
                    correlation_per_unit[sid] = {}
                if h_temp_key not in correlation_per_unit[sid]:
                    correlation_per_unit[sid][h_temp_key] = {}
                cur = correlation_per_unit[sid][h_temp_key].get(h_wind_bucket, 0.0)
                if cur == 0.0:
                    if sid not in buffer_per_unit:
                        buffer_per_unit[sid] = {}
                    if h_temp_key not in buffer_per_unit[sid]:
                        buffer_per_unit[sid][h_temp_key] = {}
                    if h_wind_bucket not in buffer_per_unit[sid][h_temp_key]:
                        buffer_per_unit[sid][h_temp_key][h_wind_bucket] = []
                    buf = buffer_per_unit[sid][h_temp_key][h_wind_bucket]
                    buf.append(unit_kwh)
                    if len(buf) >= LEARNING_BUFFER_THRESHOLD:
                        correlation_per_unit[sid][h_temp_key][h_wind_bucket] = round(
                            sum(buf) / len(buf), 5
                        )
                        buf.clear()
                else:
                    new_val = cur + learning_rate * (unit_kwh - cur)
                    correlation_per_unit[sid][h_temp_key][h_wind_bucket] = round(new_val, 5)
