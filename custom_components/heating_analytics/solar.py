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
    SCREEN_DIRECT_TRANSMITTANCE,
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
    def _screen_transmittance_vector(
        correction_percent: float,
        screen_config: tuple[bool, bool, bool] | None = None,
    ) -> tuple[float, float, float]:
        """Map screen open-percentage to per-direction transmittance (S, E, W).

        Per-direction model (#826).  Each cardinal direction is treated
        independently based on whether its facade has external screens:

            screened    : t = SCREEN_DIRECT_TRANSMITTANCE + (1 - mn) * pct/100
                          (mn ≈ 0.08, pure screen-fabric × glass at 0 %)

            unscreened  : t = 1.0  (always, regardless of slider)

        When ``screen_config`` is None (legacy / pre-1.3.3 storage), all three
        directions fall back to the composite floor DEFAULT_SOLAR_MIN_TRANSMITTANCE
        (≈ 0.30) which represents a typical Nordic residential building with
        partial screen coverage.

        Args:
            correction_percent: Slider position 0–100.  100 = screens open
                (no reduction), 0 = screens fully closed.
            screen_config: (south_has_screen, east_has_screen, west_has_screen).
                None = legacy composite floor for all directions.

        Returns:
            (t_south, t_east, t_west) each in [floor, 1.0].
        """
        pct = max(0.0, min(100.0, correction_percent))
        ratio = pct / 100.0
        # Treat malformed / missing screen_config as legacy.  Defensive
        # because mock coordinators in tests may yield empty tuples or
        # truthy MagicMock values; falling back keeps unrelated tests
        # green while the proper config path stays exercised.
        if (
            screen_config is None
            or not hasattr(screen_config, "__len__")
            or len(screen_config) != 3
        ):
            mn = DEFAULT_SOLAR_MIN_TRANSMITTANCE
            t = mn + (1.0 - mn) * ratio
            return t, t, t
        mn = SCREEN_DIRECT_TRANSMITTANCE
        t_screened = mn + (1.0 - mn) * ratio
        s_has, e_has, w_has = screen_config
        return (
            t_screened if s_has else 1.0,
            t_screened if e_has else 1.0,
            t_screened if w_has else 1.0,
        )

    @staticmethod
    def reconstruct_potential_vector(
        effective_vec: tuple[float, float, float],
        correction_percent: float,
        screen_config: tuple[bool, bool, bool] | None = None,
        *,
        min_transmittance: float = 0.01,
    ) -> tuple[float, float, float]:
        """Reconstruct the pre-screen potential vector from the effective vector.

        Per CLAUDE.md invariant #2 — when potential is constant within the
        hour, ``effective_avg / transmittance(correction_avg) == potential``
        because both vector and correction_percent are linearly accumulated
        per-minute by the collector.  Per direction since #826: each cardinal
        direction undoes its own transmittance (1.0 for unscreened facades).

        Below ``min_transmittance`` the reconstruction is undefined; the
        component is returned unchanged.  Matches the historical guard at
        every call site (pre-#876).
        """
        t_s, t_e, t_w = SolarCalculator._screen_transmittance_vector(
            correction_percent, screen_config
        )
        return (
            effective_vec[0] / t_s if t_s > min_transmittance else effective_vec[0],
            effective_vec[1] / t_e if t_e > min_transmittance else effective_vec[1],
            effective_vec[2] / t_w if t_w > min_transmittance else effective_vec[2],
        )

    @staticmethod
    def _screen_transmittance(
        correction_percent: float,
        screen_config: tuple[bool, bool, bool] | None = None,
    ) -> float:
        """Scalar transmittance — average across the three directions.

        Retained for diagnostics, factor-style fallbacks, and any code path
        that does not have a per-direction vector to operate on.  Per-direction
        callers MUST use :meth:`_screen_transmittance_vector` to avoid the
        cross-direction coupling that motivated #826.

        With ``screen_config=None`` this returns the composite legacy floor
        ramp identical to pre-1.3.3 behaviour but with floor 0.30 instead of
        0.20.
        """
        s, e, w = SolarCalculator._screen_transmittance_vector(
            correction_percent, screen_config
        )
        return (s + e + w) / 3.0

    def _resolve_screen_config(
        self,
        screen_config: tuple[bool, bool, bool] | None,
    ) -> tuple[bool, bool, bool] | None:
        """Resolve a per-call override against the coordinator default."""
        if screen_config is not None:
            return screen_config
        cfg = getattr(self.coordinator, "screen_config", None)
        return cfg

    def calculate_effective_solar_vector(
        self,
        potential_solar_vector: tuple[float, float, float],
        correction_percent: float,
        screen_config: tuple[bool, bool, bool] | None = None,
    ) -> tuple[float, float, float]:
        """Calculate effective solar vector after per-direction screen attenuation."""
        s, e, w = potential_solar_vector
        cfg = self._resolve_screen_config(screen_config)
        t_s, t_e, t_w = self._screen_transmittance_vector(correction_percent, cfg)
        return s * t_s, e * t_e, w * t_w

    def calculate_effective_solar_factor(
        self,
        potential_solar_factor: float,
        correction_percent: float,
        screen_config: tuple[bool, bool, bool] | None = None,
    ) -> float:
        """Calculate effective scalar solar factor (legacy, direction-agnostic).

        Uses the average of the three per-direction transmittances since the
        scalar factor has no direction information.  Direction-aware callers
        should use :meth:`calculate_effective_solar_vector`.
        """
        cfg = self._resolve_screen_config(screen_config)
        return potential_solar_factor * self._screen_transmittance(
            correction_percent, cfg
        )

    def calculate_unit_solar_impact(
        self,
        global_solar_vector: tuple[float, float, float],
        unit_coeff: dict[str, float],
    ) -> float:
        """Calculate solar impact in kWh for a specific unit.

        Per CLAUDE.md invariant #1: prediction uses ``coeff × potential``
        with no extra transmittance factor.  The coefficient absorbs
        ``avg_transmittance`` via the NLMS learning target (``base − actual``),
        so multiplying by the current transmittance here would yield
        ``phys × trans² × potential`` — the trans² bug.  Callers must pass
        the *potential* (pre-screen reconstructed) vector.

            Impact = Coeff_S × Pot_S + Coeff_E × Pot_E + Coeff_W × Pot_W

        With the 3-component decomposition (S, E, W), all basis functions and
        coefficients are non-negative, so the dot product is always >= 0.
        The ``max(0, ...)`` clamp is retained as defense-in-depth.
        """
        solar_s, solar_e, solar_w = global_solar_vector
        coeff_s = unit_coeff.get("s", 0.0)
        coeff_e = unit_coeff.get("e", 0.0)
        coeff_w = unit_coeff.get("w", 0.0)

        impact = coeff_s * solar_s + coeff_e * solar_e + coeff_w * solar_w
        return max(0.0, impact)

    def calculate_unit_coefficient(
        self, entity_id: str, temp_key: str, mode: str
    ) -> dict[str, float]:
        """Calculate 3D solar coefficient vector (S, E, W) for one (unit, mode).

        Mode-stratified per #868: heating-mode lookups read
        ``solar_coefficients_per_unit[entity]["heating"]``; cooling-mode
        lookups read ``["cooling"]``.  Each regime absorbs its own
        ``E[1/COP]`` and converges to a physically distinct value.

        Mode is required (not derived from ``temp_key``).  Strict
        signature catches accidental mode-blind call sites; ``temp_key``
        is preserved for future temp-stratified extensions but currently
        unused for coefficient lookup.

        OFF / DHW / unknown modes route to the heating regime as a safe
        fallback — these modes don't drive a real solar prediction (a
        unit in OFF contributes 0 kWh; DHW prediction is separate), so
        the regime choice is semantically irrelevant but stable.

        Priority:
        1. Learned coefficient for ``[entity][regime]`` if non-zero.
        2. Mode-appropriate global default (heating: 0.35, cooling:
           0.40) decomposed along the configured primary azimuth.
        """
        del temp_key  # reserved; coefficients are temperature-blind by design
        # Per-entity solar-scope gate (#962).  Excluded entities return a
        # zero-vector directly — bypassing the default-fallback decomposition
        # at the bottom of this method that would otherwise inject a phantom
        # coefficient (DEFAULT_SOLAR_COEFF × azimuth_decomposition) onto
        # interior loads, slab-thermostat floor heating, etc.  Test
        # coordinators without the helper read as legacy (all entities
        # affected) so existing tests keep passing.
        is_solar_affected_fn = getattr(self.coordinator, "is_solar_affected", None)
        if callable(is_solar_affected_fn) and not is_solar_affected_fn(entity_id):
            return {"s": 0.0, "e": 0.0, "w": 0.0}
        regime = "cooling" if mode in (MODE_COOLING, MODE_GUEST_COOLING) else "heating"
        entity_coeffs = self.coordinator.model.solar_coefficients_per_unit.get(entity_id)
        if isinstance(entity_coeffs, dict):
            regime_coeff = entity_coeffs.get(regime)
            if isinstance(regime_coeff, dict) and any(
                regime_coeff.get(k) for k in ("s", "e", "w")
            ):
                return regime_coeff

        # Mode-appropriate default — same scalar / azimuth decomposition
        # path as before, just regime-keyed instead of mode-keyed.
        if regime == "cooling":
            default_scalar = DEFAULT_SOLAR_COEFF_COOLING
        else:
            default_scalar = DEFAULT_SOLAR_COEFF_HEATING

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
        except (TypeError, ValueError) as e:
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
