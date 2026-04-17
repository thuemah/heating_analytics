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
    DEFAULT_SOLAR_MIN_TRANSMITTANCE,
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
        # Air Mass and Atmospheric Transmittance Logic
        if elevation <= 0.0:
            return 0.0

        # Cap elevation at 1° minimum to keep air mass finite (sin of near-zero angles
        # produces astronomically large AM values that carry no physical meaning).
        safe_elev = max(1.0, elevation)
        elev_rad = math.radians(safe_elev)

        # Calculate Air Mass (AM) and atmospheric transmittance intensity
        am = 1.0 / math.sin(elev_rad)
        intensity = 0.7 ** am

        # Switch to Vertical Geometry: cos(elevation)
        # This gives higher factors for low sun (Winter) and lower for high sun (Summer)
        raw_elev_factor = max(0.0, math.cos(elev_rad))

        elev_factor = raw_elev_factor * intensity

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

        # 3. Cloud Factor (Kasten & Czeplak 1980)
        # G/G_clear = 1 - 0.75 × (N/8)^3.4 where N is oktas.
        # With cloud_frac = cloud_coverage/100 as fractional sky cover,
        # the exponent 3.4 produces a nearly constant bias (~1%) across
        # all cloud levels when using satellite/model-derived percentages,
        # allowing per-unit coefficients to represent window physics
        # rather than compensating for cloud-model error.
        cloud_frac = cloud_coverage / 100.0
        cloud_factor = 1.0 - 0.75 * cloud_frac ** 3.4

        return elev_factor * az_factor * cloud_factor

    def calculate_solar_vector(self, elevation: float, azimuth: float, cloud_coverage: float) -> tuple[float, float, float]:
        """Calculate 3D solar vector (South, East, West) components.

        Each component uses max(0, ...) to produce non-negative basis functions:
        - South: max(0, -cos(az)) — positive when sun is south of E-W line
        - East:  max(0,  sin(az)) — positive in morning (az 0-180)
        - West:  max(0, -sin(az)) — positive in afternoon (az 180-360)

        This allows all three per-unit coefficients to be physically clamped
        to >= 0 (each window direction can only receive solar gain, never
        produce negative gain).  East and West are orthogonal by construction
        (disjoint temporal support).
        """
        if elevation <= 0.0:
            return 0.0, 0.0, 0.0

        # Elevation Factor
        safe_elev = max(1.0, elevation)
        elev_rad = math.radians(safe_elev)

        am = 1.0 / math.sin(elev_rad)
        intensity = 0.7 ** am

        raw_elev_factor = max(0.0, math.cos(elev_rad))
        elev_factor = raw_elev_factor * intensity

        # Cloud Factor (Kasten & Czeplak 1980 — matches calculate_solar_factor)
        cloud_frac = cloud_coverage / 100.0
        cloud_factor = 1.0 - 0.75 * cloud_frac ** 3.4

        base_intensity = elev_factor * cloud_factor

        # 3D Decomposition — non-negative basis functions
        az_rad = math.radians(azimuth)
        solar_south = base_intensity * max(0.0, -math.cos(az_rad))
        solar_east = base_intensity * max(0.0, math.sin(az_rad))
        solar_west = base_intensity * max(0.0, -math.sin(az_rad))

        return solar_south, solar_east, solar_west

    @staticmethod
    def _screen_transmittance(correction_percent: float) -> float:
        """Map screen open-percentage to effective transmittance fraction.

        Applies a physically-motivated floor (DEFAULT_SOLAR_MIN_TRANSMITTANCE)
        so that a fully-closed screen (correction_percent = 0) still passes a
        baseline of solar energy through:

            effective = MIN + (1 - MIN) * (correction_percent / 100)

        At 0 %:   effective = MIN   (~screen G-value + unmonitored windows)
        At 100 %: effective = 1.0   (fully open, unchanged behaviour)

        The floor keeps the effective solar vector non-zero even when all
        screens are down, which prevents solar-coefficient learning from
        stalling (vector_magnitude guard in _learn_unit_solar_coefficient)
        and stops the base thermal model from absorbing residual solar gain.
        """
        mn = DEFAULT_SOLAR_MIN_TRANSMITTANCE
        return mn + (1.0 - mn) * (correction_percent / 100.0)

    def calculate_effective_solar_vector(self, potential_solar_vector: tuple[float, float, float], correction_percent: float) -> tuple[float, float, float]:
        """Calculate effective solar vector after applying screens/correction."""
        s, e, w = potential_solar_vector
        factor = self._screen_transmittance(correction_percent)
        return s * factor, e * factor, w * factor

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
        return potential_solar_factor * self._screen_transmittance(correction_percent)

    def calculate_unit_solar_impact(
        self,
        global_solar_vector: tuple[float, float, float],
        unit_coeff: dict[str, float],
        screen_transmittance: float = 1.0,
    ) -> float:
        """Calculate solar impact in kW for a specific unit.

        Since #809, coefficients are learned against the potential (pre-screen)
        solar vector. The prediction path passes the potential vector and
        screen_transmittance separately:

            Impact = (Coeff_S × Pot_S + Coeff_E × Pot_E + Coeff_W × Pot_W) × transmittance

        With the 3-component decomposition (S, E, W), all basis functions and
        coefficients are non-negative, so the dot product is always >= 0.
        The max(0, ...) clamp is retained as defense-in-depth.
        """
        solar_s, solar_e, solar_w = global_solar_vector
        coeff_s = unit_coeff.get("s", 0.0)
        coeff_e = unit_coeff.get("e", 0.0)
        coeff_w = unit_coeff.get("w", 0.0)

        impact = (coeff_s * solar_s + coeff_e * solar_e + coeff_w * solar_w) * screen_transmittance
        return max(0.0, impact)

    def calculate_unit_coefficient(self, entity_id: str, temp_key: str) -> dict[str, float]:
        """Calculate 3D solar coefficient vector (S, E, W) for a specific unit.

        Solar gain is a physical property of the window (area, orientation, shading) and is
        independent of outdoor temperature. The coefficient is therefore stored and looked up
        globally per unit rather than stratified by temperature bucket.

        Uses the following priority:
        1. Learned coefficient for this unit (global, not temp-stratified).
        2. Mode-appropriate global default (optimized for heat pumps).
        """
        # Check storage structure: {unit: {"s": float, "e": float, "w": float}}
        coeff = self.coordinator.model.solar_coefficients_per_unit.get(entity_id)
        if coeff is not None:
            return coeff

        # 2. Mode-appropriate Default
        # If no coefficients learned yet for this mode, use global defaults.
        # We derive the mode from temp_key or fall back to current unit mode.
        mode = None
        try:
            target_t = int(temp_key)
            mode = MODE_HEATING if target_t < self.coordinator.balance_point else MODE_COOLING
        except ValueError:
            mode = self.coordinator.get_unit_mode(entity_id)

        default_scalar = 0.0
        if mode in (MODE_HEATING, MODE_GUEST_HEATING):
            default_scalar = DEFAULT_SOLAR_COEFF_HEATING
        elif mode in (MODE_COOLING, MODE_GUEST_COOLING):
            default_scalar = DEFAULT_SOLAR_COEFF_COOLING

        # Decompose global scalar default along the configured primary azimuth
        az_rad = math.radians(self.coordinator.solar_azimuth)
        return {
            "s": default_scalar * max(0.0, -math.cos(az_rad)),
            "e": default_scalar * max(0.0, math.sin(az_rad)),
            "w": default_scalar * max(0.0, -math.sin(az_rad)),
        }

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

    def estimate_daily_avg_solar_vector(self, date_obj: date, cloud_coverage: float = 50.0) -> tuple[float, float, float]:
        """Estimate the average solar vector (S, E, W) for a given day (24h).

        Each component is averaged independently across 24 hourly samples.
        The result represents the expected mean solar contribution per hour
        from each cardinal direction.
        """
        total_s, total_e, total_w = 0.0, 0.0, 0.0
        start_dt = dt_util.start_of_local_day(dt_util.now().replace(year=date_obj.year, month=date_obj.month, day=date_obj.day))

        for i in range(24):
            check_dt = start_dt + timedelta(hours=i)
            elev, azim = self.get_approx_sun_pos(check_dt)
            s, e, w = self.calculate_solar_vector(elev, azim, cloud_coverage)
            total_s += s
            total_e += e
            total_w += w

        return total_s / 24.0, total_e / 24.0, total_w / 24.0
