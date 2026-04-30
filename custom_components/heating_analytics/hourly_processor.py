"""HourlyProcessor — hosts the hour-boundary processing extracted from coordinator.py.

Thin-delegate pattern: the processor holds a reference to the coordinator
and reaches back for state.  Public methods on the coordinator delegate
to this engine so the external API is unchanged.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta

from homeassistant.util import dt as dt_util

from .const import (
    ATTR_LAST_HOUR_ACTUAL,
    ATTR_LAST_HOUR_DEVIATION,
    ATTR_LAST_HOUR_DEVIATION_PCT,
    ATTR_LAST_HOUR_EXPECTED,
    ATTR_SOLAR_IMPACT,
    ATTR_TDD,
    CONF_FORECAST_CROSSOVER_DAY,
    CONF_SECONDARY_WEATHER_ENTITY,
    COOLDOWN_CONVERGENCE_THRESHOLD,
    COOLDOWN_MAX_HOURS,
    COOLDOWN_MIN_HOURS,
    DEFAULT_FORECAST_CROSSOVER_DAY,
    DUAL_INTERFERENCE_THRESHOLD,
    ENERGY_GUARD_THRESHOLD,
    MIXED_MODE_HIGH,
    MIXED_MODE_LOW,
    MODE_COOLING,
    MODE_DHW,
    MODE_GUEST_COOLING,
    MODE_GUEST_HEATING,
    MODE_HEATING,
    MODE_OFF,
)

_LOGGER = logging.getLogger(__name__)


class HourlyProcessor:
    """Hour-boundary processing engine.

    Owns :meth:`process` (the hour-boundary entry) and
    :meth:`build_observation` (snapshot the completed hour as a
    frozen ``HourlyObservation``).  All state lives on the coordinator.
    """

    def __init__(self, coordinator) -> None:
        self.coordinator = coordinator

    @staticmethod
    def _compute_battery_ema_input(
        solar_impact: float,
        solar_wasted: float,
        k_feedback: float,
        has_heating_unit: bool,
    ) -> float:
        """Return the EMA input for the solar thermal battery (#896).

        When ``k_feedback == 0.0`` returns ``solar_impact`` unchanged —
        the pre-#896 formula collapses bit-for-bit, including for any
        positive ``solar_wasted``.  When ``k_feedback > 0.0`` AND a
        heating-mode unit is active that hour, returns
        ``solar_impact + k_feedback * solar_wasted``.  Otherwise the
        feedback term is gated out (cooling-only / OFF / DHW hours).

        Note: ``solar_wasted`` is structurally zero in cooling-only
        hours because ``solar.calculate_saturation`` returns wasted=0
        for non-heating modes.  The explicit ``has_heating_unit`` gate
        is defense-in-depth so the contract is local to this helper.
        """
        if k_feedback > 0.0 and has_heating_unit:
            return solar_impact + k_feedback * solar_wasted
        return solar_impact

    def build_observation(
        self,
        current_time: datetime,
        *,
        avg_temp: float,
        inertia_temp: float,
        temp_key: str,
        effective_wind: float,
        wind_bucket: str,
        avg_solar_factor: float,
        avg_solar_vector: tuple[float, float, float],
        solar_impact_raw: float,
        effective_solar_impact: float,
        total_energy_kwh: float,
        learning_energy_kwh: float,
        guest_impact_kwh: float,
        expected_kwh: float,
        base_expected_kwh: float,
        aux_impact_kwh: float,
        aux_fraction: float,
        is_aux_dominant: bool,
        was_cooldown_active: bool,
        forecasted_kwh: float | None = None,
        forecasted_kwh_primary: float | None = None,
        forecasted_kwh_secondary: float | None = None,
        forecasted_kwh_gross: float | None = None,
        forecasted_kwh_gross_primary: float | None = None,
        forecasted_kwh_gross_secondary: float | None = None,
        forecast_source: str | None = None,
        recommendation_state: str = "none",
        correction_percent: float = 0.0,
        potential_solar_factor: float = 0.0,
        solar_normalization_delta: float = 0.0,
        is_solar_dominant: bool = False,
        solar_dominant_entities: tuple[str, ...] = (),
    ) -> "HourlyObservation":
        """Build an immutable HourlyObservation from current coordinator state.

        Called at hour boundary after all aggregates are computed.
        Captures the frozen snapshot before accumulators are reset.
        """
        from .observation import HourlyObservation

        n = self.coordinator._collector.sample_count
        hc = self.coordinator._collector.humidity_count
        avg_humidity = round(self.coordinator._collector.humidity_sum / hc, 1) if hc > 0 else None

        timestamp = self.coordinator._collector.start_time if self.coordinator._collector.start_time else current_time
        return HourlyObservation(
            timestamp=timestamp,
            hour=timestamp.hour,
            avg_temp=avg_temp,
            inertia_temp=inertia_temp,
            temp_key=temp_key,
            effective_wind=effective_wind,
            wind_bucket=wind_bucket,
            bucket_counts=dict(self.coordinator._collector.bucket_counts),
            avg_humidity=avg_humidity,
            solar_factor=avg_solar_factor,
            solar_vector=avg_solar_vector,
            solar_impact_raw=solar_impact_raw,
            effective_solar_impact=effective_solar_impact,
            total_energy_kwh=total_energy_kwh,
            learning_energy_kwh=learning_energy_kwh,
            guest_impact_kwh=guest_impact_kwh,
            expected_kwh=expected_kwh,
            base_expected_kwh=base_expected_kwh,
            unit_breakdown={
                eid: round(kwh, 3)
                for eid, kwh in self.coordinator._hourly_delta_per_unit.items()
                if kwh > 0
            },
            unit_expected={
                eid: round(kwh, 3)
                for eid, kwh in self.coordinator._hourly_expected_per_unit.items()
                if kwh > 0
            },
            unit_expected_base={
                eid: round(kwh, 3)
                for eid, kwh in self.coordinator._hourly_expected_base_per_unit.items()
                if kwh > 0
            },
            aux_impact_kwh=aux_impact_kwh,
            aux_fraction=aux_fraction,
            is_aux_dominant=is_aux_dominant,
            sample_count=self.coordinator._collector.sample_count,
            unit_modes={
                entity_id: mode
                for entity_id, mode in self.coordinator._unit_modes.items()
            },
            forecasted_kwh=forecasted_kwh,
            forecasted_kwh_primary=forecasted_kwh_primary,
            forecasted_kwh_secondary=forecasted_kwh_secondary,
            forecasted_kwh_gross=forecasted_kwh_gross,
            forecasted_kwh_gross_primary=forecasted_kwh_gross_primary,
            forecasted_kwh_gross_secondary=forecasted_kwh_gross_secondary,
            forecast_source=forecast_source,
            recommendation_state=recommendation_state,
            correction_percent=correction_percent,
            potential_solar_factor=potential_solar_factor,
            was_cooldown_active=was_cooldown_active,
            solar_normalization_delta=solar_normalization_delta,
            is_solar_dominant=is_solar_dominant,
            solar_dominant_entities=solar_dominant_entities,
            battery_filtered_potential=(
                self.coordinator._potential_battery_s,
                self.coordinator._potential_battery_e,
                self.coordinator._potential_battery_w,
            ),
        )

    async def process(self, current_time: datetime):
        """Process the accumulated data for the past hour.

        Triggered at the start of a new hour (or restart).
        1. Calculates final stats for the completed hour (Temp, Effective Wind).
        2. Updates learning models (Correlation & Solar Coefficients).
        3. Persists hourly log entry.
        4. Saves data to storage.
        """
        # --- Aux Cooldown / Decay Management ---
        # Capture initial state for learning (so convergent hour is still protected)
        was_cooldown_active = self.coordinator._aux_cooldown_active

        if self.coordinator._aux_cooldown_active and self.coordinator._aux_cooldown_start_time:
            # Calculate elapsed time in hours
            # Ensure timezone awareness compatibility
            start_ts = self.coordinator._aux_cooldown_start_time
            if start_ts.tzinfo is None and current_time.tzinfo:
                start_ts = start_ts.replace(tzinfo=current_time.tzinfo)
            elif start_ts.tzinfo and current_time.tzinfo is None:
                start_ts = start_ts.replace(tzinfo=None)

            elapsed = (current_time - start_ts).total_seconds() / 3600.0

            # Condition 1: Max Timeout
            if elapsed >= COOLDOWN_MAX_HOURS:
                _LOGGER.info(f"Aux Cooldown: Max duration exceeded ({elapsed:.1f}h). Exiting lock.")
                self.coordinator._aux_cooldown_active = False
                self.coordinator._aux_cooldown_start_time = None

            # Condition 2: Min Duration + Convergence
            elif elapsed >= COOLDOWN_MIN_HOURS:
                # Calculate Convergence Ratio for Affected Units
                actual_sum = 0.0
                expected_sum = 0.0

                targets = self.coordinator.aux_affected_entities or []

                for entity_id in targets:
                    actual_sum += self.coordinator._hourly_delta_per_unit.get(entity_id, 0.0)
                    expected_sum += self.coordinator._hourly_expected_base_per_unit.get(entity_id, 0.0)

                # Check Convergence
                if expected_sum > ENERGY_GUARD_THRESHOLD:
                    ratio = actual_sum / expected_sum
                    if ratio >= COOLDOWN_CONVERGENCE_THRESHOLD:
                        _LOGGER.info(f"Aux Cooldown: Convergence reached ({ratio:.1%}). Exiting lock.")
                        self.coordinator._aux_cooldown_active = False
                        self.coordinator._aux_cooldown_start_time = None
                    else:
                        _LOGGER.debug(f"Aux Cooldown: Active. Ratio {ratio:.1%} < {COOLDOWN_CONVERGENCE_THRESHOLD:.0%}")
                else:
                    _LOGGER.debug("Aux Cooldown: Active. No expected consumption to verify convergence.")

        # Calculate what was expected for the full hour (using aggregates)
        avg_temp = 0.0
        calculated_effective_wind = 0.0
        avg_solar_factor = 0.0
        avg_solar_vector = (0.0, 0.0, 0.0)
        wind_bucket = "normal"
        aux_fraction = 0.0
        is_aux_dominant = False  # For learning purposes

        if self.coordinator._collector.sample_count > 0:
             # Calculate averages from aggregates
             avg_temp = self.coordinator._collector.temp_sum / self.coordinator._collector.sample_count

             # Calculate 90th percentile for effective wind (Nearest Rank)
             eff_winds = sorted(self.coordinator._collector.wind_values)
             idx = math.ceil(0.9 * len(eff_winds)) - 1
             calculated_effective_wind = eff_winds[idx]

             # Determine bucket for the passed hour
             if self.coordinator.solar_enabled:
                 avg_solar_factor = self.coordinator._collector.solar_sum / self.coordinator._collector.sample_count
                 avg_solar_vector = (
                     self.coordinator._collector.solar_vector_s_sum / self.coordinator._collector.sample_count,
                     self.coordinator._collector.solar_vector_e_sum / self.coordinator._collector.sample_count,
                     self.coordinator._collector.solar_vector_w_sum / self.coordinator._collector.sample_count,
                 )

             # Determine base wind bucket (always physical now)
             wind_bucket = self.coordinator._get_wind_bucket(calculated_effective_wind)

             # Calculate Aux Fraction
             aux_fraction = self.coordinator._collector.aux_count / self.coordinator._collector.sample_count

             # Check auxiliary heating dominant? (For Learning & Log)
             # Use 80% threshold for pure modes to ensure clean learning data (tolerant to restart)
             if aux_fraction >= 0.80:
                 is_aux_dominant = True

        # 0. Close any gap from the end of the previous hour
        # Use Mean Imputation (Aggregates) to ensure consistency with logged hour stats
        if self.coordinator._collector.last_minute_processed is not None:
             self.coordinator._close_hour_gap(
                 current_time,
                 self.coordinator._collector.last_minute_processed,
                 avg_temp=avg_temp,
                 avg_wind=calculated_effective_wind,
                 avg_solar=avg_solar_factor,
                 is_aux_active=is_aux_dominant
             )

        # Save Last Hour Stats for Sensors

        # Solar Optimization Learning
        rec_state_avg = "none"
        # Use hour-averaged correction instead of end-of-hour snapshot.
        # Screen automation may change correction mid-hour; the average
        # is consistent with the accumulated solar_factor and solar_vector.
        if self.coordinator._collector.sample_count > 0:
            actual_correction = self.coordinator._collector.correction_sum / self.coordinator._collector.sample_count
        else:
            actual_correction = self.coordinator.solar_correction_percent
        potential_factor_avg = 0.0

        if self.coordinator.solar_enabled and self.coordinator._collector.sample_count > 0:
             # Calculate average potential factor for the hour
             target_dt = self.coordinator._collector.start_time if self.coordinator._collector.start_time else current_time
             mid_point = target_dt + timedelta(minutes=30)
             elev, azimuth = self.coordinator.solar.get_approx_sun_pos(mid_point)

             # Fetch current cloud for learning constraint (and potential calc fallback)
             current_cloud = self.coordinator._get_cloud_coverage()

             if actual_correction >= 5.0:
                 potential_factor_avg = (avg_solar_factor / (actual_correction / 100.0))
             else:
                 # Fallback if screens were closed (can't derive potential from effective)
                 potential_factor_avg = self.coordinator.solar.calculate_solar_factor(elev, azimuth, current_cloud)

             potential_factor_avg = max(0.0, min(1.0, potential_factor_avg))

             rec_state_avg = self.coordinator.solar_optimizer.get_recommendation_state(avg_temp, potential_factor_avg)
             self.coordinator.solar_optimizer.learn_correction_percent(rec_state_avg, elev, azimuth, actual_correction, cloud_cover=current_cloud)

        # Calculate Inertia Temp for Learning (3 previous hours + this hour's average)
        # We look at hourly_log (previous hours) + avg_temp (this hour)
        # Use helper to ensure we don't pick up stale logs after a gap
        inertia_temps = self.coordinator._get_recent_log_temps(current_time)
        inertia_temps.append(avg_temp)

        inertia_avg = self.coordinator._calculate_weighted_inertia(inertia_temps)

        # Use inertia average for correlation key
        temp_key = str(int(round(inertia_avg)))

        # Total Energy (for logging & display)
        total_energy_kwh = self.coordinator._collector.energy_hour

        # Energy for Global Model Learning (exclude guest and DHW modes)
        learning_energy_kwh = 0.0
        guest_impact_kwh = 0.0
        for entity_id, actual_kwh in self.coordinator._hourly_delta_per_unit.items():
            unit_mode = self.coordinator.get_unit_mode(entity_id)
            if unit_mode in (MODE_GUEST_HEATING, MODE_GUEST_COOLING):
                # Guest units are not tracked in expected - their full consumption is the impact
                guest_impact_kwh += actual_kwh
            elif unit_mode in (MODE_OFF, MODE_DHW):
                # Off and DHW energy do not inform the space-heating model
                pass
            else:
                learning_energy_kwh += actual_kwh


        # Expected (Physics / Reality)
        # Use the accumulated expected energy (Weighted sum of Normal/Aux minutes)
        # This ensures the log matches the "Expected So Far" calculation and respects mixed hours.
        expected_kwh = self.coordinator._collector.expected_energy_hour

        # Forecasted (Plan) - Shadow Forecasting Logic
        # Calculate predictions for ALL sources using REFERENCE weather locked at midnight.
        forecasted_kwh = 0.0
        forecasted_kwh_primary = 0.0
        forecasted_kwh_secondary = 0.0

        target_dt = self.coordinator._collector.start_time if self.coordinator._collector.start_time else current_time

        # Get items for all sources
        f_item_blended = self.coordinator.forecast.get_forecast_for_hour(target_dt, source='reference')
        f_item_primary = self.coordinator.forecast.get_forecast_for_hour(target_dt, source='primary_reference')
        f_item_secondary = self.coordinator.forecast.get_forecast_for_hour(target_dt, source='secondary_reference')

        # Setup context for calculation
        local_inertia_seed = list(inertia_temps[:-1])
        weather_wind_unit = self.coordinator._get_weather_wind_unit()
        current_cloud = self.coordinator._get_cloud_coverage()

        # Helper to process an item
        def _get_f_kwh(item, ignore_aux=False):
            if not item: return None
            # 9-tuple return as of #899 trajectory threading; we only need
            # the predicted kWh here (single-point query, no trajectory).
            val, _, _, _, _, _, _, _, _ = self.coordinator.forecast._process_forecast_item(
                item, list(local_inertia_seed), weather_wind_unit, current_cloud, ignore_aux=ignore_aux
            )
            return val

        # Calculate values
        # Net Forecasts (Shadow) - Tracks expected consumption including Aux reduction
        forecasted_kwh = _get_f_kwh(f_item_blended, ignore_aux=False)
        forecasted_kwh_primary = _get_f_kwh(f_item_primary, ignore_aux=False)
        forecasted_kwh_secondary = _get_f_kwh(f_item_secondary, ignore_aux=False)

        # Gross Forecasts (Thermodynamic) - Tracks pure thermodynamic demand (Plan)
        # Used for accurate "Gross vs Gross" error calculation
        forecasted_kwh_gross = _get_f_kwh(f_item_blended, ignore_aux=True)
        forecasted_kwh_gross_primary = _get_f_kwh(f_item_primary, ignore_aux=True)
        forecasted_kwh_gross_secondary = _get_f_kwh(f_item_secondary, ignore_aux=True)

        # Fallback if no forecast available (Assume perfect forecast for baseline)
        if not f_item_blended: forecasted_kwh = expected_kwh
        if not f_item_primary: forecasted_kwh_primary = forecasted_kwh
        if not f_item_secondary: forecasted_kwh_secondary = forecasted_kwh

        # Calculate Base Expected (for Learning Reference)
        # This is the PHYSICAL BASE (without Aux or Solar)
        # NEW: Use robust calculation for base reference too
        # We need "Base" (Normal) prediction for the given conditions.

        # Calculate and store aux_impact in self.coordinator.data
        # We want the *theoretical* impact of aux if it were active
        # calculate_total_power for aux active gives us: Base - Aux_Impact
        # So Aux_Impact = Base - (Net)
        # Or we can read it from the breakdown!

        # Proportional Aux Impact:
        # Use the precise accumulated value (No threshold) to ensure accurate
        # accounting of auxiliary impact even during mixed-mode hours.
        aux_impact_kwh = round(self.coordinator._collector.aux_impact_hour, 3)

        # OPTIMIZATION: Call once with correct aux/solar context.
        # This returns BOTH the base values (unaffected by aux) AND the aux/solar reduction.
        res_analysis = self.coordinator.statistics.calculate_total_power(
            inertia_avg, # Use inertia temp (float)
            calculated_effective_wind,
            0.0, # This arg is ignored by calculate_total_power
            is_aux_active=is_aux_dominant,
            detailed=False,
            override_solar_factor=avg_solar_factor,
            override_solar_vector=avg_solar_vector,
            known_aux_impact_kwh=aux_impact_kwh,
        )

        # Global Stabilizer: Use Global Base for learning Track A
        # Prevents feedback loop from Unit Sums
        base_expected_kwh = res_analysis.get("global_base_kwh", res_analysis["breakdown"]["base_kwh"])

        self.coordinator.data["last_hour_aux_impact_kwh"] = aux_impact_kwh

        # Calculate Solar Impact (Saturated)
        # Use the authoritative calculation from statistics which correctly applies saturation
        # (Solar cannot reduce consumption below zero or below base - aux)
        solar_impact = 0.0
        solar_wasted = 0.0
        solar_heating_wasted = 0.0
        solar_normalization_delta = 0.0
        if self.coordinator.solar_enabled:
            solar_impact = res_analysis["breakdown"]["solar_reduction_kwh"]
            solar_wasted = res_analysis["breakdown"].get("solar_wasted_kwh", 0.0)
            # Heating-only wasted (#896).  ``solar_wasted_kwh`` is the
            # total aggregate across all units; cooling-mode wasted is
            # structurally zero today (saturation returns 0 for cooling)
            # but the explicit heating-only field protects the battery
            # EMA against any future change that lets cooling produce
            # non-zero wasted.  Falls back to the total aggregate for
            # legacy log-replay callers that pre-date this field.
            solar_heating_wasted = res_analysis["breakdown"].get(
                "solar_heating_wasted_kwh", solar_wasted
            )
            # Saturation-aware solar delta for global model normalization (#801).
            # Heating solar was subtracted from demand; cooling solar was added.
            # To normalize to dark-sky: add heating solar back, subtract cooling solar.
            solar_normalization_delta = (
                res_analysis["breakdown"].get("solar_heating_applied_kwh", 0.0)
                - res_analysis["breakdown"].get("solar_cooling_applied_kwh", 0.0)
            )
            # Update global attr for consistency
            self.coordinator.data[ATTR_SOLAR_IMPACT] = round(solar_impact, 3)

        # Solar Thermal Battery: EMA model of building thermal mass.
        # The battery smooths solar impact across hours via exponential decay,
        # modelling how thermal mass absorbs solar heat and releases it slowly.
        # The (1 - decay) input factor is the exact discretisation of Newton's
        # law of cooling: at steady state, effective solar = raw solar (no
        # amplification).  The decay parameter controls only the time constant.
        #
        # Saturation-wasted thermal feedback (#896, opt-in, default off):
        # ``battery_thermal_feedback_k`` ∈ [0, 1] adds ``k × solar_wasted`` to
        # the EMA input on heating-mode hours, accounting for solar potential
        # that exceeded VP demand and was clipped but still entered the
        # building thermal mass.  Defense-in-depth gate on
        # ``MODE_HEATING/MODE_GUEST_HEATING`` — ``solar_wasted`` is already
        # zero in cooling/OFF/DHW (saturation returns wasted=0 for those
        # modes) so the gate is redundant in steady state, but the explicit
        # check makes the contract local to this line.  When ``k == 0.0`` the
        # formula collapses to the pre-#896 form bit-for-bit.
        if self.coordinator.solar_enabled:
            # Heating-active gate iterates configured energy sensors and
            # routes each through ``get_unit_mode`` rather than reading
            # ``_unit_modes.values()`` directly.  ``_unit_modes`` is sparse —
            # entries are written only when the user explicitly sets a mode
            # via the select entity, so on default heating-only installs
            # (the issue's target population) the dict is empty.  Reading
            # ``.values()`` would evaluate ``any()`` over an empty iterable
            # and gate the feedback off in exactly the case it should be
            # on.  ``get_unit_mode(eid)`` returns ``MODE_HEATING`` for
            # missing entries (coordinator.py:540-542), matching the
            # documented "default mode is heating" semantics.
            has_heating_unit = any(
                self.coordinator.get_unit_mode(eid) in (MODE_HEATING, MODE_GUEST_HEATING)
                for eid in self.coordinator.energy_sensors
            )
            ema_input = self._compute_battery_ema_input(
                solar_impact=solar_impact,
                solar_wasted=solar_heating_wasted,
                k_feedback=self.coordinator.battery_thermal_feedback_k,
                has_heating_unit=has_heating_unit,
            )
            self.coordinator._solar_battery_state = (
                self.coordinator._solar_battery_state * self.coordinator.solar_battery_decay
                + ema_input * (1 - self.coordinator.solar_battery_decay)
            )
            # Solar carry-over reservoir (#896 follow-up).  Parallel EMA
            # to ``_solar_battery_state`` but charged ONLY from
            # ``k × solar_heating_wasted`` — no applied-solar term.  The
            # carry-over reservoir represents the saturation-clipped
            # energy still residing in thermal mass; release from it
            # subtracts from heating-mode demand prediction in
            # ``statistics.calculate_total_power`` (mode-gated, blind
            # decay / option-a).  When ``k == 0.0`` the input is always
            # 0 and the state stays at 0 — bit-identical no-op for the
            # release subtraction.  Splitting state from
            # ``_solar_battery_state`` preserves all five existing
            # consumers of the scalar battery (display sensors,
            # ``thermodynamic_gross_kwh``, aux learning input via
            # ``effective_solar_impact``, hourly log
            # ``solar_impact_kwh``).  Same gate + same decay as the
            # scalar battery for physical consistency.
            carryover_input = (
                self.coordinator.battery_thermal_feedback_k * solar_heating_wasted
                if (self.coordinator.battery_thermal_feedback_k > 0.0 and has_heating_unit)
                else 0.0
            )
            self.coordinator._solar_carryover_state = (
                self.coordinator._solar_carryover_state * self.coordinator.solar_battery_decay
                + carryover_input * (1 - self.coordinator.solar_battery_decay)
            )
            # Per-direction potential battery (#865).  Reconstruct the
            # hour's potential vector from the effective ``avg_solar_vector``
            # and ``actual_correction`` via the canonical per-direction
            # transmittance, then feed into per-direction EMAs.  Same
            # decay as the scalar battery for physical consistency.
            #
            # NOTE (#896): the wasted-feedback term k × solar_wasted is
            # intentionally NOT mixed into the per-direction potential
            # batteries.  Inequality learning (invariant #5) consumes
            # ``_potential_battery_s/e/w`` as raw potential to enforce
            # ``coeff · battery ≥ 0.9 × base``; introducing a saturation-
            # derived feedback here would inflate the constraint signal
            # and let coefficients satisfy the constraint at lower
            # values, breaking the one-sided semantics that anchors
            # invariant #5.  The k feedback affects only the scalar
            # impact battery (used for aux learning + display); the
            # per-direction batteries remain pure raw-potential EMAs.
            #
            # Guarded: test mocks may return a non-iterable from the
            # transmittance helper.  On any unpacking error we simply skip
            # this hour's battery update — the battery decays toward zero
            # over a few hours rather than corrupting the EMA on bad input.
            try:
                screen_cfg = getattr(self.coordinator, "screen_config", None)
                pot_s, pot_e, pot_w = self.coordinator.solar.reconstruct_potential_vector(
                    avg_solar_vector, actual_correction, screen_cfg
                )
                decay = self.coordinator.solar_battery_decay
                self.coordinator._potential_battery_s = (
                    self.coordinator._potential_battery_s * decay + pot_s * (1 - decay)
                )
                self.coordinator._potential_battery_e = (
                    self.coordinator._potential_battery_e * decay + pot_e * (1 - decay)
                )
                self.coordinator._potential_battery_w = (
                    self.coordinator._potential_battery_w * decay + pot_w * (1 - decay)
                )
            except (TypeError, ValueError, ZeroDivisionError):
                # Mock-friendly fallback.  Real screen_transmittance_vector
                # cannot raise — it returns a fixed-length tuple of floats.
                pass
        effective_solar_impact = self.coordinator._solar_battery_state

        # Update Last Hour Data in Coordinator
        self.coordinator.data[ATTR_LAST_HOUR_ACTUAL] = round(total_energy_kwh, 3)
        self.coordinator.data[ATTR_LAST_HOUR_EXPECTED] = round(expected_kwh, 3)
        self.coordinator.data[ATTR_LAST_HOUR_DEVIATION] = round(total_energy_kwh - expected_kwh, 3)
        self.coordinator.data["last_hour_wind_bucket"] = wind_bucket
        self.coordinator.data["last_hour_solar_impact_kwh"] = round(effective_solar_impact, 3)
        self.coordinator.data["last_hour_guest_impact_kwh"] = round(guest_impact_kwh, 3)

        if expected_kwh > ENERGY_GUARD_THRESHOLD:
            self.coordinator.data[ATTR_LAST_HOUR_DEVIATION_PCT] = round(((total_energy_kwh - expected_kwh) / expected_kwh) * 100, 1)
        else:
            self.coordinator.data[ATTR_LAST_HOUR_DEVIATION_PCT] = 0.0

        if self.coordinator._collector.sample_count > 0:
            # Determine Learning Eligibility
            # Skip learning if Mixed Mode to avoid model corruption
            # Skip entirely if Daily Learning Mode is active (strategies own midnight writes)
            is_mixed_mode = (MIXED_MODE_LOW < aux_fraction < MIXED_MODE_HIGH)
            should_learn = self.coordinator.learning_enabled and not is_mixed_mode and not self.coordinator.daily_learning_mode

            # Per-unit learning: DirectMeter sensors continue via Track A even when
            # daily_learning_mode blocks global learning.  WeightedSmear sensors are
            # excluded — their meter data is MPC-tainted. (#776)
            should_learn_per_unit = self.coordinator.learning_enabled and not is_mixed_mode if self.coordinator.daily_learning_mode else None

            # Dual Interference Guard:
            # If both Solar and Aux are significant, we cannot reliably attribute deviation.
            is_dual_interference = (solar_impact > DUAL_INTERFERENCE_THRESHOLD) and (aux_impact_kwh > DUAL_INTERFERENCE_THRESHOLD)
            if is_dual_interference:
                should_learn = False

            # Detect Guest Mode Activity
            # If any unit is in Guest Mode, aux learning must be disabled to prevent pollution
            # Base and solar learning can continue as they use learning_energy_kwh (guest-excluded)
            has_guest_activity = any(
                mode in (MODE_GUEST_HEATING, MODE_GUEST_COOLING)
                for mode in self.coordinator._unit_modes.values()
            )

            # Solar shutdown detection (#838).
            # Identifies VP units whose thermostat cut the compressor because
            # sun-heated rooms exceeded setpoint.  Such hours produce
            # actual_impact = base - 0 = base in the NLMS target, inflating
            # the solar coefficient and feeding contaminated values into the
            # base model via solar_normalization_delta.
            from .solar import SolarCalculator as _SC
            from .observation import detect_solar_shutdown_entities
            _potential_vector = _SC.reconstruct_potential_vector(
                avg_solar_vector, actual_correction, self.coordinator.screen_config
            )
            solar_dominant_entities = detect_solar_shutdown_entities(
                solar_enabled=self.coordinator.solar_enabled,
                is_aux_dominant=is_aux_dominant,
                potential_vector=_potential_vector,
                energy_sensors=self.coordinator.energy_sensors,
                unit_modes=self.coordinator._unit_modes,
                unit_actual_kwh=self.coordinator._hourly_delta_per_unit,
                unit_expected_base_kwh=self.coordinator._hourly_expected_base_per_unit,
                unit_min_base=self.coordinator._per_unit_min_base_thresholds or None,
            )
            is_solar_dominant = bool(solar_dominant_entities)

            # Correct normalization delta: exclude shutdown-flagged units whose
            # saturated solar contribution was included in solar_heating_applied
            # before detection ran.  Without this, the global base model learns
            # from an inflated delta during shutdown hours.
            if solar_dominant_entities and self.coordinator.solar_enabled:
                for _sd_eid in solar_dominant_entities:
                    _sd_mode = self.coordinator._unit_modes.get(_sd_eid, MODE_HEATING)
                    _sd_coeff = self.coordinator.solar.calculate_unit_coefficient(
                        _sd_eid, temp_key, _sd_mode
                    )
                    _sd_solar = self.coordinator.solar.calculate_unit_solar_impact(
                        _potential_vector, _sd_coeff
                    )
                    _sd_base = self.coordinator._hourly_expected_base_per_unit.get(_sd_eid, 0.0)
                    _sd_applied, _, _ = self.coordinator.solar.calculate_saturation(
                        _sd_base, _sd_solar, _sd_mode
                    )
                    if _sd_mode in (MODE_HEATING, MODE_GUEST_HEATING):
                        solar_normalization_delta -= _sd_applied
                    elif _sd_mode in (MODE_COOLING, MODE_GUEST_COOLING):
                        solar_normalization_delta += _sd_applied

            # --- Build immutable observation snapshot (Issue #775) ---
            obs = self.build_observation(
                current_time,
                avg_temp=avg_temp,
                inertia_temp=inertia_avg,
                temp_key=temp_key,
                effective_wind=calculated_effective_wind,
                wind_bucket=wind_bucket,
                avg_solar_factor=avg_solar_factor,
                avg_solar_vector=avg_solar_vector,
                solar_impact_raw=solar_impact,
                effective_solar_impact=effective_solar_impact,
                total_energy_kwh=total_energy_kwh,
                learning_energy_kwh=learning_energy_kwh,
                guest_impact_kwh=guest_impact_kwh,
                expected_kwh=expected_kwh,
                base_expected_kwh=base_expected_kwh,
                aux_impact_kwh=aux_impact_kwh,
                aux_fraction=aux_fraction,
                is_aux_dominant=is_aux_dominant,
                was_cooldown_active=was_cooldown_active,
                forecasted_kwh=forecasted_kwh,
                forecasted_kwh_primary=forecasted_kwh_primary,
                forecasted_kwh_secondary=forecasted_kwh_secondary,
                forecasted_kwh_gross=forecasted_kwh_gross,
                forecasted_kwh_gross_primary=forecasted_kwh_gross_primary,
                forecasted_kwh_gross_secondary=forecasted_kwh_gross_secondary,
                forecast_source=f_item_blended.get("_source") if f_item_blended else None,
                recommendation_state=rec_state_avg,
                correction_percent=round(actual_correction, 1),
                potential_solar_factor=round(potential_factor_avg, 3),
                solar_normalization_delta=solar_normalization_delta,
                is_solar_dominant=is_solar_dominant,
                solar_dominant_entities=solar_dominant_entities,
            )

            # Delegate Learning to LearningManager (#775 Phase 3)
            from .observation import LearningConfig

            # Only DirectMeter sensors participate in hourly per-unit learning;
            # WeightedSmear sensors are excluded (MPC-tainted meter data). (#776)
            from .observation import DirectMeter
            hourly_sensors = [
                sid for sid, strat in self.coordinator._unit_strategies.items()
                if isinstance(strat, DirectMeter)
            ] if self.coordinator.daily_learning_mode else self.coordinator.energy_sensors

            learning_config = LearningConfig(
                learning_enabled=should_learn,
                solar_enabled=self.coordinator.solar_enabled,
                learning_rate=self.coordinator.learning_rate,
                balance_point=self.coordinator.balance_point,
                energy_sensors=hourly_sensors,
                aux_impact=self.coordinator._get_aux_impact_kw(temp_key, wind_bucket),
                solar_calculator=self.coordinator.solar,
                get_predicted_unit_base_fn=self.coordinator._get_predicted_kwh_per_unit,
                aux_affected_entities=self.coordinator.aux_affected_entities,
                has_guest_activity=has_guest_activity,
                per_unit_learning_enabled=should_learn_per_unit,
                screen_config=self.coordinator.screen_config,
                screen_affected_entities=self.coordinator._screen_affected_set,
                unit_min_base=self.coordinator._per_unit_min_base_thresholds or None,
            )

            learning_result = self.coordinator.learning.process_learning(
                obs=obs,
                model=self.coordinator.get_model_state(),
                config=learning_config,
            )

            # Accumulate Real TDD (Thermal Degree Days)
            # Use ABSOLUTE difference to handle both Heating and Cooling
            tdd_contribution = abs(self.coordinator.balance_point - avg_temp) / 24.0
            current_tdd_acc = self.coordinator.data.get(ATTR_TDD, 0.0)
            self.coordinator.data[ATTR_TDD] = round(current_tdd_acc + tdd_contribution, 3)

            # Prepare Unit Breakdown for Log (rounded)
            unit_breakdown = {
                eid: round(kwh, 3)
                for eid, kwh in self.coordinator._hourly_delta_per_unit.items()
                if kwh > 0
            }

            # Prepare Unit Expected Breakdown for Log
            # This captures the true "mixed" expectation for the hour (solving "Majority Rule" contamination)
            unit_expected_breakdown = {
                eid: round(kwh, 3)
                for eid, kwh in self.coordinator._hourly_expected_per_unit.items()
                if kwh > 0
            }

            # Hourly Log Entry
            # Use _hourly_start_time to represent the START of the hour period, not the END

            # If skipped due to mixed mode, clarify status
            final_learning_status = learning_result.get("learning_status", "unknown")
            if is_mixed_mode and self.coordinator.learning_enabled:
                if final_learning_status != "cooldown_post_aux":
                    final_learning_status = "skipped_mixed_mode"
            elif is_dual_interference and self.coordinator.learning_enabled:
                if final_learning_status != "cooldown_post_aux":
                    final_learning_status = "skipped_dual_interference"

            # Calculate Thermodynamic Gross (Actual + Aux + Solar Adjustment)
            # Make mode-aware: Add in Heating, Subtract in Cooling
            # Use effective_solar_impact (battery-smoothed) so the gross reflects
            # the full residual solar heat carried in building mass.
            solar_adjustment = effective_solar_impact
            if avg_temp >= self.coordinator.balance_point:
                solar_adjustment = -effective_solar_impact

            thermodynamic_gross_kwh = total_energy_kwh + aux_impact_kwh + solar_adjustment

            log_entry = {
                "timestamp": self.coordinator._collector.start_time.isoformat() if self.coordinator._collector.start_time else current_time.isoformat(),
                "hour": self.coordinator._collector.start_time.hour if self.coordinator._collector.start_time else current_time.hour,
                "temp": round(avg_temp, 1),
                "tdd": round(tdd_contribution, 3),
                "unit_breakdown": unit_breakdown,
                "unit_expected_breakdown": unit_expected_breakdown,
                "temp_key": temp_key,
                "inertia_temp": round(inertia_avg, 2),
                "effective_wind": round(calculated_effective_wind, 2),
                "wind_bucket": wind_bucket,
                "humidity": obs.avg_humidity,
                "actual_kwh": round(total_energy_kwh, 3),
                "expected_kwh": round(expected_kwh, 3),
                "thermodynamic_gross_kwh": round(thermodynamic_gross_kwh, 3),
                "forecasted_kwh": round(forecasted_kwh, 3),
                "forecasted_kwh_primary": round(forecasted_kwh_primary, 3),
                "forecasted_kwh_secondary": round(forecasted_kwh_secondary, 3),
                "forecasted_kwh_gross": round(forecasted_kwh_gross, 3) if forecasted_kwh_gross is not None else None,
                "forecasted_kwh_gross_primary": round(forecasted_kwh_gross_primary, 3) if forecasted_kwh_gross_primary is not None else None,
                "forecasted_kwh_gross_secondary": round(forecasted_kwh_gross_secondary, 3) if forecasted_kwh_gross_secondary is not None else None,
                "forecast_source": f_item_blended.get("_source") if f_item_blended else None,
                "deviation": round(total_energy_kwh - expected_kwh, 3),
                "deviation_pct": self.coordinator.data.get(ATTR_LAST_HOUR_DEVIATION_PCT, 0.0),
                "auxiliary_active": is_aux_dominant,
                "aux_impact_kwh": aux_impact_kwh,
                "guest_impact_kwh": round(guest_impact_kwh, 3),
                "solar_factor": round(avg_solar_factor, 3),
                "solar_vector_s": round(avg_solar_vector[0], 3),
                "solar_vector_e": round(avg_solar_vector[1], 3),
                "solar_vector_w": round(avg_solar_vector[2], 3),
                "solar_impact_kwh": round(effective_solar_impact, 3),
                "solar_impact_raw_kwh": round(solar_impact, 3),
                "solar_wasted_kwh": round(solar_wasted, 3),
                # Heating-only wasted (#896).  Structurally equal to
                # ``solar_wasted_kwh`` today (cooling saturation returns 0)
                # but persisted as its own field so the feedback-sweep
                # replay in diagnose_solar can use the explicitly-gated
                # value without re-deriving mode from unit_modes.
                "solar_heating_wasted_kwh": round(solar_heating_wasted, 3),
                "primary_entity": self.coordinator.weather_entity,
                "secondary_entity": self.coordinator.entry.data.get(CONF_SECONDARY_WEATHER_ENTITY),
                "crossover_day": self.coordinator.entry.data.get(CONF_FORECAST_CROSSOVER_DAY, DEFAULT_FORECAST_CROSSOVER_DAY),
                # Model Update Info
                "model_temp_key": temp_key,
                "model_base_before": round(learning_result["model_base_before"], 5),
                "model_base_after": round(learning_result["model_base_after"], 5),
                "model_updated": learning_result["model_updated"],
                "aux_model_updated": learning_result.get("aux_model_updated", False),
                "aux_model_before": learning_result.get("aux_model_before"),
                "aux_model_after": learning_result.get("aux_model_after"),
                "learning_status": final_learning_status,
                "recommendation_state": rec_state_avg,
                "correction_percent": round(actual_correction, 1),
                "potential_solar_factor": round(potential_factor_avg, 3),
                "solar_normalization_delta": round(solar_normalization_delta, 5),
                "solar_regime": "shutdown" if is_solar_dominant else "normal",
                "solar_dominant_entities": list(solar_dominant_entities),
                # Balance point active when this entry was logged (#856).  BP is
                # user-configurable via the reconfigure flow; recording it per
                # entry lets diagnostics detect BP transitions in the analysis
                # window and flag the U-coefficient as mixed-BP when the stored
                # tdd values were computed under different BPs.
                "bp_at_log_time": self.coordinator.balance_point,
                # Only filter out MODE_HEATING (the true default) to reduce log clutter
                # Cooling, off, and guest modes MUST be logged for correct historical reconstruction
                "unit_modes": {
                    entity_id: mode
                    for entity_id, mode in self.coordinator._unit_modes.items()
                    if mode != MODE_HEATING
                },
            }
            # Guard against duplicate entries (e.g., crash-at-boundary + restart scenario).
            # Live appends are monotonic in timestamp, so comparing the tail is enough — O(1)
            # instead of scanning ~8760 entries every hour-boundary.  CSV import re-sorts the
            # log by timestamp (storage.py), which preserves the tail-is-latest invariant
            # because any new real-time entry is at least an hour after every imported entry.
            entry_ts = log_entry["timestamp"]
            if self.coordinator._hourly_log and self.coordinator._hourly_log[-1].get("timestamp") == entry_ts:
                _LOGGER.warning(f"Duplicate hourly entry detected for {entry_ts}, skipping append.")
            else:
                self.coordinator._hourly_log.append(log_entry)

            # Retention Policy (#820): configurable via hourly_log_retention_days.
            max_entries = self.coordinator._hourly_log_max_entries
            if len(self.coordinator._hourly_log) > max_entries:
                del self.coordinator._hourly_log[:-max_entries]

            _LOGGER.info(f"Hourly Update: Temp={avg_temp:.1f}, Wind={wind_bucket}, Energy={total_energy_kwh:.2f}, Solar={avg_solar_factor:.2f}")

            # CSV Auto-logging (if enabled)
            await self.coordinator.storage.append_hourly_log_csv(log_entry)

        # Accumulate hourly aux breakdown into daily stats
        for entity_id, stats in self.coordinator._collector.aux_breakdown.items():
            if entity_id not in self.coordinator._daily_aux_breakdown:
                self.coordinator._daily_aux_breakdown[entity_id] = {"allocated": 0.0, "overflow": 0.0}
            self.coordinator._daily_aux_breakdown[entity_id]["allocated"] += stats.get("allocated", 0.0)
            self.coordinator._daily_aux_breakdown[entity_id]["overflow"] += stats.get("overflow", 0.0)

        # Accumulate orphaned savings into daily total
        self.coordinator._daily_orphaned_aux += self.coordinator._collector.orphaned_aux

        # Save Logic (force save on hourly boundary)
        await self.coordinator._async_save_data(force=True)

        # Reset all hour-scoped accumulators atomically (#775).
        # In-place clearing preserves coordinator aliases for dict fields.
        self.coordinator._collector.reset()

