"""Solar Calculator Service."""
from __future__ import annotations

import math
import logging
from datetime import datetime, timedelta, date

from homeassistant.util import dt as dt_util

try:
    from astral import Observer
    from astral.sun import elevation as sun_elevation, azimuth as sun_azimuth
    HAS_ASTRAL = True
except ImportError:
    HAS_ASTRAL = False

from .const import (
    DEFAULT_SOLAR_COEFF_HEATING,
    DEFAULT_SOLAR_COEFF_COOLING,
    ENERGY_GUARD_THRESHOLD,
    MODE_HEATING,
    MODE_COOLING,
    MODE_OFF,
    MODE_GUEST_HEATING,
    MODE_GUEST_COOLING,
)

_LOGGER = logging.getLogger(__name__)

class SolarCalculator:
    """Calculates solar impact on heating/cooling."""

    def __init__(self, coordinator) -> None:
        """Initialize with reference to coordinator (for configuration/state)."""
        self.coordinator = coordinator

    def calculate_solar_factor(self, elevation: float, azimuth: float, cloud_coverage: float) -> float:
        """Calculate solar factor (0.0 - 1.0)."""
        # 1. Elevation Factor
        # Atmospheric Attenuation Logic:
        # - Ignore < 5 degrees (too much atmosphere)
        # - Fade in linearly 5-10 degrees
        # - Full impact > 10 degrees
        if elevation < 5.0:
            return 0.0

        elev_rad = math.radians(elevation)
        # Switch to Vertical Geometry: cos(elevation)
        # This gives higher factors for low sun (Winter) and lower for high sun (Summer)
        raw_elev_factor = max(0.0, math.cos(elev_rad))

        # Apply Atmospheric Fade
        if elevation < 10.0:
            attenuation = (elevation - 5.0) / 5.0
            elev_factor = raw_elev_factor * attenuation
        else:
            elev_factor = raw_elev_factor

        # 2. Azimuth Factor (Peak at Configured Azimuth)
        # Kelvin Twist: Uses a 3-zone logic to account for self-shading (egenskygge)
        # Zone 1 (0-75°): Rescaled Cosine (1.0 -> 0.1)
        # Zone 2 (75-90°): Glancing Diffuse (0.1)
        # Zone 3 (90-180°): Backside Diffuse (0.05)

        target_azimuth = self.coordinator.solar_azimuth

        # Calculate delta degrees
        delta = abs(azimuth - target_azimuth)
        # Normalize to 0-180 (shortest path)
        if delta > 180:
            delta = 360 - delta

        # Constants for "Kelvin Twist"
        BUFFER_ANGLE = 15.0
        DIFFUSE_FLOOR = 0.1
        BACKSIDE_FLOOR = 0.05

        cutoff = 90.0 - BUFFER_ANGLE  # 75.0
        az_factor = 0.0

        if delta <= cutoff:
            # Zone 1: Direct Sun
            # Maps 0..75 degrees to 0..90 degrees (conceptually) for the cosine curve
            # ensuring it hits the floor exactly at cutoff.
            normalized_pos = delta / cutoff
            # cos(0) = 1, cos(PI/2) = 0
            direct_component = math.cos(normalized_pos * (math.pi / 2))
            az_factor = direct_component * (1.0 - DIFFUSE_FLOOR) + DIFFUSE_FLOOR
        elif delta <= 90.0:
            # Zone 2: Glancing
            az_factor = DIFFUSE_FLOOR
        else:
            # Zone 3: Backside
            az_factor = BACKSIDE_FLOOR

        # 3. Cloud Factor
        cloud_factor = 1.0 - (cloud_coverage / 100.0)

        return elev_factor * az_factor * cloud_factor

    def calculate_effective_solar_factor(self, potential_solar_factor: float, correction_percent: float) -> float:
        """Calculate effective solar factor after applying screens/correction.

        Args:
            potential_solar_factor: The raw calculated solar factor (Screens Up).
            correction_percent: The percentage of solar gain allowed (0-100).
                                100 = No screens (Full Gain).
                                0 = Full screens (No Gain).

        Returns:
            The effective solar factor used for impact calculations.
        """
        return potential_solar_factor * (correction_percent / 100.0)

    def calculate_unit_solar_impact(self, global_solar_factor: float, unit_coeff: float) -> float:
        """Calculate solar impact in kW for a specific unit using its learned coefficient.

        Formula: Impact = Global_Factor * Unit_Coeff
        The Unit_Coeff absorbs the "Effective Window Area" and "Efficiency".
        """
        return global_solar_factor * unit_coeff

    def calculate_unit_coefficient(self, entity_id: str, temp_key: str) -> float:
        """Calculate solar coefficient for a specific unit and temp.

        Uses the following priority:
        1. Exact match for unit and temperature bucket.
        2. Closest learned neighbor within the same thermal mode (Heating/Cooling).
        3. Mode-appropriate global default (optimized for heat pumps).
        """
        # Check storage structure: {unit: {temp_key: coeff}}
        coeffs = self.coordinator._solar_coefficients_per_unit.get(entity_id, {})

        # 1. Exact Match
        if temp_key in coeffs:
            return coeffs[temp_key]

        # 2. Mode-aware Neighborhood Search
        try:
            target_t = int(temp_key)
            bp = self.coordinator.balance_point
            # Determine target mode for the requested temp
            target_mode = MODE_HEATING if target_t < bp else MODE_COOLING

            # Find all learned coeffs for this unit and filter by mode
            mode_coeffs: dict[int, float] = {}
            for t_str, val in coeffs.items():
                try:
                    t = int(t_str)
                    m = MODE_HEATING if t < bp else MODE_COOLING
                    if m == target_mode:
                        mode_coeffs[t] = val
                except ValueError:
                    continue

            if mode_coeffs:
                # Check for +/- 1 neighbors for averaging
                t_minus = target_t - 1
                t_plus = target_t + 1

                if t_minus in mode_coeffs and t_plus in mode_coeffs:
                    return (mode_coeffs[t_minus] + mode_coeffs[t_plus]) / 2.0

                # Find closest temperature bucket in the same mode
                closest_t = min(mode_coeffs.keys(), key=lambda t: abs(t - target_t))
                return mode_coeffs[closest_t]

        except ValueError:
            # Handle non-numeric temp_key gracefully
            pass

        # 3. Mode-appropriate Default
        # If no coefficients learned yet for this mode, use global defaults.
        # We derive the mode from temp_key or fall back to current unit mode.
        mode = None
        try:
            target_t = int(temp_key)
            mode = MODE_HEATING if target_t < self.coordinator.balance_point else MODE_COOLING
        except ValueError:
            mode = self.coordinator.get_unit_mode(entity_id)

        if mode in (MODE_HEATING, MODE_GUEST_HEATING):
            return DEFAULT_SOLAR_COEFF_HEATING
        if mode in (MODE_COOLING, MODE_GUEST_COOLING):
            return DEFAULT_SOLAR_COEFF_COOLING

        return 0.0

    def apply_correction(self, base_kwh: float, solar_impact: float, val: str | float) -> float:
        """Apply solar correction to predicted energy.

        Args:
            base_kwh: Base energy prediction.
            solar_impact: Solar impact to apply.
            val: Either mode (str) or temperature (float). If temp, mode is derived.

        - Heating: Solar gain reduces heating need (subtract).
        - Cooling: Solar gain increases cooling load (add).
        - Result is clamped to 0.0.
        """
        _, _, final_net = self.calculate_saturation(base_kwh, solar_impact, val)
        return final_net

    def calculate_saturation(self, net_demand: float, solar_potential: float, val: str | float) -> tuple[float, float, float]:
        """Calculate solar saturation (applied vs wasted).

        Args:
            net_demand: Remaining demand (Base - Aux).
            solar_potential: Theoretical solar impact (kW/kWh).
            val: Mode or Temperature.

        Returns:
            (applied_solar, wasted_solar, final_net)
        """
        mode = val
        if isinstance(val, (int, float)):
            if val < self.coordinator.balance_point:
                mode = MODE_HEATING
            else:
                mode = MODE_COOLING

        applied = 0.0
        wasted = 0.0
        final_net = net_demand

        if mode in (MODE_HEATING, MODE_GUEST_HEATING):
            # Solar reduces heating demand.
            # Saturation Limit = Net Demand (Cannot reduce below 0).
            # If Net Demand < 0 (Aux Overkill), Limit is 0.
            limit = max(0.0, net_demand)

            applied = min(solar_potential, limit)
            wasted = solar_potential - applied
            final_net = max(0.0, net_demand - applied)

        elif mode in (MODE_COOLING, MODE_GUEST_COOLING):
            # Solar increases cooling demand (Additive).
            # No saturation concept here.
            applied = solar_potential
            wasted = 0.0
            final_net = net_demand + applied

        elif mode == MODE_OFF:
            applied = 0.0
            wasted = 0.0
            final_net = 0.0

        else:
            # Unknown mode -> No correction
            applied = 0.0
            wasted = 0.0
            final_net = net_demand

        return round(applied, 3), round(wasted, 3), round(final_net, 3)

    def normalize_for_learning(self, actual_kwh: float, solar_impact: float, val: str | float) -> float:
        """Normalize actual energy to 'dark' conditions for learning.

        Args:
            actual_kwh: Actual measured energy.
            solar_impact: Estimated solar impact.
            val: Either mode (str) or temperature (float). If temp, mode is derived.

        This removes the solar effect from the actual reading so we can train the base model.
        - Heating: Actual was reduced by solar. Dark = Actual + Solar.
        - Cooling: Actual was increased by solar. Dark = Actual - Solar.
        - Result is clamped to 0.0.
        """
        mode = val
        if isinstance(val, (int, float)):
            if val < self.coordinator.balance_point:
                mode = MODE_HEATING
            else:
                mode = MODE_COOLING

        if mode in (MODE_HEATING, MODE_GUEST_HEATING):
            normalized = actual_kwh + solar_impact
        elif mode in (MODE_COOLING, MODE_GUEST_COOLING):
            normalized = actual_kwh - solar_impact
        else:
            # MODE_OFF or unknown -> No correction
            normalized = actual_kwh

        return max(0.0, normalized)

    def distribute_solar_impact(
        self,
        total_solar_impact_kw: float,
        predicted_kwh_breakdown: dict[str, float],
        actual_kwh_breakdown: dict[str, float] | None = None
    ) -> dict[str, float]:
        """Distribute global solar impact across heating units.

        Prioritizes distribution based on predicted consumption. If prediction is zero
        (e.g., during shoulder seasons), it falls back to actual consumption ratios.

        Args:
            total_solar_impact_kw: The total solar impact to distribute (in kW).
            predicted_kwh_breakdown: A dict of {entity_id: predicted_kwh}.
            actual_kwh_breakdown: An optional dict of {entity_id: actual_kwh} for fallback.

        Returns:
            A dict of {entity_id: distributed_solar_impact_kw}.
        """
        if total_solar_impact_kw < ENERGY_GUARD_THRESHOLD:
            return {entity_id: 0.0 for entity_id in predicted_kwh_breakdown}

        # 1. Try distribution by predicted consumption
        total_predicted = sum(predicted_kwh_breakdown.values())
        if total_predicted > ENERGY_GUARD_THRESHOLD:
            return {
                entity_id: (pred_kwh / total_predicted) * total_solar_impact_kw
                for entity_id, pred_kwh in predicted_kwh_breakdown.items()
            }

        # 2. Fallback to distribution by actual consumption
        if actual_kwh_breakdown:
            total_actual = sum(actual_kwh_breakdown.values())
            if total_actual > ENERGY_GUARD_THRESHOLD:
                # Ensure we only distribute to units present in the prediction breakdown
                # to maintain consistency in keys.
                return {
                    entity_id: (actual_kwh_breakdown.get(entity_id, 0.0) / total_actual) * total_solar_impact_kw
                    for entity_id in predicted_kwh_breakdown
                }

        # 3. If both are zero, distribute evenly (or return zero)
        # Returning zero is safer to avoid division by zero.
        return {entity_id: 0.0 for entity_id in predicted_kwh_breakdown}

    def get_approx_sun_pos(self, dt_obj: datetime) -> tuple[float, float]:
        """Get sun position (Elevation, Azimuth) for any datetime.

        Uses the astral library directly (same library used by HA's sun.sun entity)
        for high-precision astronomical calculations. This ensures consistency
        and eliminates custom PSA algorithm drift.

        Args:
            dt_obj: Datetime object to calculate sun position for

        Returns:
            Tuple of (elevation, azimuth) in degrees
        """
        if self.coordinator.hass.config.latitude is None or self.coordinator.hass.config.longitude is None:
            return 0.0, 0.0

        if not HAS_ASTRAL:
            _LOGGER.error("Astral library not available. Cannot calculate sun position.")
            return 0.0, 0.0

        try:
            # Create Observer with HA's configured location
            observer = Observer(
                latitude=self.coordinator.hass.config.latitude,
                longitude=self.coordinator.hass.config.longitude,
                elevation=self.coordinator.hass.config.elevation or 0
            )

            # Ensure datetime is timezone-aware (astral requires it)
            if dt_obj.tzinfo is None:
                dt_obj = dt_util.as_utc(dt_obj)

            # Calculate sun position using astral library
            elevation = sun_elevation(observer, dt_obj)
            azimuth = sun_azimuth(observer, dt_obj)

            return elevation, azimuth
        except Exception as e:
            _LOGGER.warning(f"Failed to calculate sun position for {dt_obj}: {e}")
            return 0.0, 0.0

    def estimate_daily_avg_solar_factor(self, date_obj: date, cloud_coverage: float = 50.0) -> float:
        """Estimate the average solar factor for a given day (24h).

        Useful for backfilling historical data where solar factor was not logged.
        """
        total_factor = 0.0
        start_dt = dt_util.start_of_local_day(dt_util.now().replace(year=date_obj.year, month=date_obj.month, day=date_obj.day))

        # Iterate 24 hours
        for i in range(24):
            check_dt = start_dt + timedelta(hours=i)
            elev, azim = self.get_approx_sun_pos(check_dt)
            factor = self.calculate_solar_factor(elev, azim, cloud_coverage)
            total_factor += factor

        return total_factor / 24.0
