"""Solar Calculator Service."""
from __future__ import annotations

import math
import logging
from datetime import datetime, timedelta, date
from typing import NamedTuple

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


def _eccentricity(day_of_year: int) -> float:
    """Earth-orbit eccentricity correction factor E_0 (Spencer-style approx)."""
    return 1.0 + 0.033 * math.cos(2.0 * math.pi * day_of_year / 365.0)


def erbs_decomposition(
    ghi: float,
    sun_elev_deg: float,
    day_of_year: int,
) -> tuple[float, float]:
    """Erbs (1982) GHI -> (DNI, DHI) decomposition.

    Returns (DNI, DHI) in W/m^2.  Splits the measured global horizontal
    irradiance into a direct-normal component and a diffuse-horizontal
    component using the diffuse-fraction model of Erbs, Klein & Duffie
    (1982).  The model is the meteorological standard reference for
    single-sensor pyranometer / lux-sensor fallback when neither native
    DNI nor native DHI is available.

    Documented broken-cloud bias: kT in [0.6, 0.8] tends to over-estimate
    diffuse fraction by ~10 % on partly-cloudy hours.  This bias is itself
    measurable downstream via the 4D shadow learner - it is not opaque
    like the Kasten 3.4 elevation x airmass distortion that motivated the
    DNI/DHI work in the first place.

    Algorithm:
        sin_elev = sin(max(0, sun_elev_deg))
        I_sc     = 1367 W/m^2 (solar constant)
        E_0      = 1 + 0.033*cos(2*pi*n/365)  (eccentricity correction)
        GHI_ext  = I_sc * E_0 * sin_elev    (extraterrestrial horizontal)
        kT       = ghi / GHI_ext            (clearness index)
        kd       = piecewise-polynomial in kT (see below)
        DHI      = kd * ghi
        DNI      = (ghi - DHI) / sin_elev

    Piecewise diffuse fraction (Erbs et al. 1982):
        kT <= 0.22:        kd = 1.0 - 0.09*kT
        0.22 < kT <= 0.80: kd = 0.9511 - 0.1604*kT + 4.388*kT^2
                                - 16.638*kT^3 + 12.336*kT^4
        kT > 0.80:         kd = 0.165

    Edge cases:
        sun_elev_deg <= 0 or ghi <= 0 -> return (0.0, 0.0)
        GHI_ext < 1 W/m^2 (numerical floor near sunrise/sunset) ->
            return (0.0, ghi)  (treat all measured irradiance as diffuse;
            DNI is meaningless when the sun is below the horizon)
        kT > 1.0 (sensor noise / non-physical) -> clamp to 1.0 for kd
            lookup and treat all surplus as DNI.

    Args:
        ghi: Global horizontal irradiance in W/m^2 (typically 0-1100).
        sun_elev_deg: Sun elevation angle in degrees (negative below horizon).
        day_of_year: 1-366, used for eccentricity correction.

    Returns:
        (dni, dhi) tuple in W/m^2.  Both >= 0.  DNI clamped to >= 0 in
        case of numerical rounding when kT ~ 1.
    """
    if sun_elev_deg <= 0 or ghi <= 0:
        return (0.0, 0.0)

    sin_elev = math.sin(math.radians(sun_elev_deg))
    e_0 = _eccentricity(day_of_year)
    ghi_ext = 1367.0 * e_0 * sin_elev

    if ghi_ext < 1.0:
        # Sun too low for meaningful DNI; everything is diffuse.
        return (0.0, ghi)

    kT = ghi / ghi_ext
    kT_for_kd = min(kT, 1.0)

    if kT_for_kd <= 0.22:
        kd = 1.0 - 0.09 * kT_for_kd
    elif kT_for_kd <= 0.80:
        kd = (
            0.9511
            - 0.1604 * kT_for_kd
            + 4.388 * kT_for_kd ** 2
            - 16.638 * kT_for_kd ** 3
            + 12.336 * kT_for_kd ** 4
        )
    else:
        kd = 0.165

    dhi = kd * ghi
    dni = max(0.0, (ghi - dhi) / sin_elev)
    return (dni, dhi)


def derive_dni_dhi_source_label(
    ghi_avg: float | None,
    dni_avg: float | None,
    dhi_avg: float | None,
    cloud_avg: float | None,
) -> str:
    """Label which ladder branch an hour's inputs resolve through.

    Sun-blind counterpart to :func:`resolve_dni_dhi`, for the
    hour-boundary logger: it answers "which branch did this hour take?"
    without needing the sun position.

    Takes the **same averaged values** the 4D learner feeds to
    :func:`resolve_dni_dhi` (``obs.ghi_avg`` / ``dni_avg`` / ``dhi_avg``
    / ``cloud_avg``) and applies the **same branch conditions in the
    same order**, so the label cannot disagree with the branch actually
    taken.  In particular the GHI branch requires a *positive* reading,
    not merely a configured sensor: a stuck or covered pyranometer
    averaging 0 during daylight falls through to native / Kasten inside
    ``resolve_dni_dhi``, and labelling those hours ``erbs_from_ghi``
    would report a real DNI/DHI source to
    ``diagnose_solar.dni_dhi_source_mix`` on an install whose pipeline
    is running Kasten.

    Lives next to :func:`resolve_dni_dhi` so the two stay read together;
    ``test_dni_dhi_source_mix`` pins their agreement branch-for-branch,
    including the zero-GHI case.

    **Never returns** ``"no_sun"``.  The sun gate belongs to
    :func:`resolve_dni_dhi` (and to the 4D learner, which disables
    itself below the horizon), so a night hour is labelled by whatever
    inputs were sampled — typically ``"kasten_synthetic"`` on an install
    with cloud-coverage data.  Consumers counting these labels MUST
    filter to daylight hours first, or the mix is diluted with darkness
    on every install.

    Returns:
        One of ``"erbs_from_ghi"``, ``"native"``, ``"kasten_synthetic"``,
        ``"none"`` — the value-path subset of :func:`resolve_dni_dhi`'s
        return excluding ``"no_sun"``.
    """
    if ghi_avg is not None and ghi_avg > 0:
        return "erbs_from_ghi"
    if dni_avg is not None and dhi_avg is not None:
        return "native"
    if cloud_avg is not None:
        return "kasten_synthetic"
    return "none"


def resolve_dni_dhi(
    dni_in: float | None,
    dhi_in: float | None,
    ghi_in: float | None,
    cloud_coverage_pct: float | None,
    sun_elev_deg: float,
    day_of_year: int,
) -> tuple[float, float, str]:
    """Resolve (DNI, DHI) from whichever signals are available.

    Priority ladder (per user choice in design discussion):

        1. GHI sensor -> Erbs decomposition       (source = "erbs_from_ghi")
        2. Native DNI + DHI                        (source = "native")
        3. cloud_coverage -> synthetic GHI -> Erbs (source = "kasten_synthetic")
        4. Nothing available                       (source = "none")

    Rationale for putting GHI ahead of native: a local pyranometer
    measures the actual sky at the building.  Open-Meteo native DNI/DHI
    is satellite/model-derived and carries the same regional smoothing
    bias as the cloud_coverage signal.  Erbs introduces a documented
    broken-cloud diffuse-fraction bias (see ``erbs_decomposition``),
    but on installs that have a real GHI sensor the local-truth
    advantage dominates.  Installs without a GHI sensor get the native
    path (no Erbs bias), and installs without either fall back to the
    cloud-cover synthetic path.

    Synthetic GHI from cloud_coverage uses the same Kasten 3.4 attenuation
    as the live pipeline applied to a simple clear-sky model:

        GHI_clear    = 1367 * E_0 * sin_elev * 0.7^(1/sin_elev)
        cloud_frac   = cloud_coverage_pct / 100
        cloud_factor = 1 - 0.75 * cloud_frac^3.4
        GHI_synth    = GHI_clear * cloud_factor

    This deliberately collapses the same information that the legacy
    pipeline already has - the kasten_synthetic path is mainly there
    to keep the downstream 4D code path uniform.  It is NOT expected
    to improve over the 3D Kasten path; its role is to keep the ladder
    well-defined when no real DNI/DHI/GHI is configured.

    Args:
        dni_in, dhi_in: From hourly_log["dni"] / ["dhi"], if available.
            Both must be non-None for the native path.
        ghi_in: From hourly_log["ghi"], if a GHI sensor is configured.
        cloud_coverage_pct: From hourly_log["cloud_coverage_avg"] or
            equivalent.  Falls through if None.
        sun_elev_deg: Sun elevation for this hour.  <= 0 -> no sun, return
            (0, 0, "no_sun").
        day_of_year: 1-366, threaded into Erbs.

    Returns:
        (dni, dhi, source) where source in {"erbs_from_ghi", "native",
        "kasten_synthetic", "no_sun", "none"}.
    """
    if sun_elev_deg <= 0:
        return (0.0, 0.0, "no_sun")

    if ghi_in is not None and ghi_in > 0:
        dni, dhi = erbs_decomposition(ghi_in, sun_elev_deg, day_of_year)
        return (dni, dhi, "erbs_from_ghi")

    if dni_in is not None and dhi_in is not None:
        return (max(0.0, dni_in), max(0.0, dhi_in), "native")

    if cloud_coverage_pct is not None:
        sin_elev = math.sin(math.radians(sun_elev_deg))
        air_mass = 1.0 / sin_elev
        ghi_clear = 1367.0 * _eccentricity(day_of_year) * sin_elev * (0.7 ** air_mass)
        ghi_synth = max(0.0, ghi_clear * _kasten_cloud_attenuation(cloud_coverage_pct))
        dni, dhi = erbs_decomposition(ghi_synth, sun_elev_deg, day_of_year)
        return (dni, dhi, "kasten_synthetic")

    return (0.0, 0.0, "none")


def resolve_dni_dhi_for_forecast(
    weather_hour: dict,
    sun_elev_deg: float,
    day_of_year: int,
    ghi_in: float | None = None,
) -> tuple[float, float, str]:
    """Resolve ``(DNI, DHI, source)`` for a forecast-hour dict (#978).

    Wrapper around :func:`resolve_dni_dhi` that knows how to pull native
    DNI/DHI and cloud-coverage out of a per-hour forecast item dict (as
    produced by HA's ``weather.get_forecasts`` for an Open-Meteo-style
    entity).  ``ghi_in`` is plumbed through for completeness but defaults
    to ``None``: Open-Meteo forecast does not expose forecast GHI per
    hour on the standard ``weather.get_forecasts`` payload, so installs
    with a local GHI sensor have live-only data and forecasted hours
    fall through to the native or Kasten paths.

    The cloud_coverage fallback mirrors the inline pattern already used
    twice in ``forecast.py`` (around lines ~147-150 / ~1148-1151) so the
    Kasten leg of the ladder fires off the same condition→% map the
    legacy 3D scalar path uses.

    Args:
        weather_hour: One forecast item from ``weather.get_forecasts``.
            Reads ``direct_normal_irradiance``, ``diffuse_radiation``,
            ``cloud_coverage`` (with ``condition`` → ``CLOUD_COVERAGE_MAP``
            fallback) from the dict; missing keys collapse to ``None``.
        sun_elev_deg: Sun elevation at the forecast hour midpoint.
        day_of_year: 1-366, threaded into Erbs.
        ghi_in: Optional explicit GHI; defaults to None (see above).

    Returns:
        ``(dni, dhi, source)`` per :func:`resolve_dni_dhi`.
    """
    dni_in = weather_hour.get("direct_normal_irradiance")
    dhi_in = weather_hour.get("diffuse_radiation")
    cloud_cov = weather_hour.get("cloud_coverage")
    if cloud_cov is None:
        # Fall back to condition-derived cloud cover, mirroring the
        # inline pattern in forecast.py for the 3D scalar path.
        from .const import CLOUD_COVERAGE_MAP
        cond = weather_hour.get("condition")
        mapped = CLOUD_COVERAGE_MAP.get(cond) if cond else None
        cloud_cov = float(mapped) if mapped is not None else None
    elif cloud_cov is not None:
        try:
            cloud_cov = float(cloud_cov)
        except (TypeError, ValueError):
            cloud_cov = None
    return resolve_dni_dhi(dni_in, dhi_in, ghi_in, cloud_cov, sun_elev_deg, day_of_year)


class HourInputs(NamedTuple):
    """Canonical reconstructed inputs for one ``hourly_log`` entry (#1011).

    Produced by :func:`reconstruct_hour_inputs`. Carries the four
    fields every consumer of the 4D replay path needs before it
    diverges into consumer-specific post-processing (potential
    reconstruction, coefficient lookup, saturation, regime routing).
    """

    ts_dt: datetime
    mid_dt: datetime
    sun_elev: float
    sun_az: float
    dni: float
    dhi: float
    dni_source: str


HOUR_INPUT_FAIL_MISSING_TIMESTAMP = "missing_timestamp"
HOUR_INPUT_FAIL_SUN_BELOW_HORIZON = "sun_below_horizon"
HOUR_INPUT_FAIL_SUN_POS_ERROR = "sun_pos_error"
HOUR_INPUT_FAIL_NO_DNI_DHI = "no_dni_dhi"


def reconstruct_hour_inputs(
    entry: dict,
    solar,
) -> tuple[HourInputs | None, str | None]:
    """Rebuild the opening sequence shared by every log-entry replay (#1011).

    Consolidates the parse-timestamp -> hour-midpoint offset ->
    ``get_approx_sun_pos`` -> ``resolve_dni_dhi`` chain. Was duplicated
    inline at three call sites (``_collect_batch_fit_samples_4d``,
    ``fit_solar_obstruction``,
    ``_compute_total_power_4d_divergence_replay``); see issue #1011 for
    the motivating #994-class failure (one missing ``+30min`` offset
    caused silent divergence at 7 sites).

    Args:
        entry: One ``hourly_log`` dict (``timestamp``, ``dni``, ``dhi``,
            ``ghi_wm2``, ``cloud_coverage``, ``correction_percent`` —
            missing keys treated per ``resolve_dni_dhi``'s semantics).
        solar: ``SolarCalculator`` instance (any object exposing
            ``get_approx_sun_pos(dt) -> (elev, az)``).

    Returns:
        ``(inputs, None)`` on success. ``(None, reason)`` on skip,
        where ``reason`` is one of:

            * ``HOUR_INPUT_FAIL_MISSING_TIMESTAMP``
            * ``HOUR_INPUT_FAIL_SUN_BELOW_HORIZON``
            * ``HOUR_INPUT_FAIL_SUN_POS_ERROR`` — exception raised by
              ``get_approx_sun_pos`` (astral / timezone / mock issue).
              Distinguished from ``SUN_BELOW_HORIZON`` because a lookup
              failure has no physical "0 solar" answer; callers should
              treat as skip, not as a 0-solar hour.
            * ``HOUR_INPUT_FAIL_NO_DNI_DHI`` — ``resolve_dni_dhi``
              returned ``source == "none"`` (no signal at all).
              Callers needing the stricter "both DNI and DHI <= 0"
              gate apply it themselves on the returned values.

        Callers map these to their own drop-count categories or
        return-shape conventions (e.g. the divergence replay returns
        ``0.0`` on ``SUN_BELOW_HORIZON``, ``None`` on the others).
    """
    ts_raw = entry.get("timestamp")
    if not isinstance(ts_raw, str):
        return (None, HOUR_INPUT_FAIL_MISSING_TIMESTAMP)
    try:
        ts_dt = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return (None, HOUR_INPUT_FAIL_MISSING_TIMESTAMP)

    mid_dt = ts_dt + timedelta(minutes=30)
    try:
        sun_elev, sun_az = solar.get_approx_sun_pos(mid_dt)
    except Exception:  # noqa: BLE001 — defensive against mocks / no-astral
        # Distinguished from real below-horizon (which has a deterministic
        # 0-solar physical answer); lookup failure has no physical answer
        # and callers should skip rather than count as a 0-solar hour.
        return (None, HOUR_INPUT_FAIL_SUN_POS_ERROR)
    if sun_elev is None or sun_elev <= 0.0:
        return (None, HOUR_INPUT_FAIL_SUN_BELOW_HORIZON)

    day_of_year = ts_dt.timetuple().tm_yday
    try:
        dni, dhi, source = resolve_dni_dhi(
            entry.get("dni"),
            entry.get("dhi"),
            entry.get("ghi_wm2"),
            entry.get("cloud_coverage"),
            sun_elev,
            day_of_year,
        )
    except Exception:  # noqa: BLE001
        return (None, HOUR_INPUT_FAIL_NO_DNI_DHI)
    if source == "none":
        return (None, HOUR_INPUT_FAIL_NO_DNI_DHI)
    dni_f = dni or 0.0
    dhi_f = dhi or 0.0

    return (
        HourInputs(
            ts_dt=ts_dt,
            mid_dt=mid_dt,
            sun_elev=float(sun_elev),
            sun_az=float(sun_az),
            dni=dni_f,
            dhi=dhi_f,
            dni_source=source,
        ),
        None,
    )


def _kasten_cloud_attenuation(cloud_coverage_pct: float) -> float:
    """Kasten & Czeplak (1980) cloud-factor: 1 - 0.75 * (N/8)^3.4.

    Single source for the Kasten exponent 3.4 — see CLAUDE.md > Solar Model
    for why this specific exponent is invariant.
    """
    cloud_frac = max(0.0, min(1.0, cloud_coverage_pct / 100.0))
    return 1.0 - 0.75 * cloud_frac ** 3.4


def _clear_sky_elevation_factor(elevation_deg: float) -> float:
    """Vertical-geometry clear-sky factor: cos(elev) * 0.7^airmass.

    Elevation clamped at 1° minimum to keep air mass finite (sin of near-zero
    angles produces astronomically large AM values with no physical meaning).
    Returns 0.0 below the horizon.
    """
    if elevation_deg <= 0.0:
        return 0.0
    safe_elev = max(1.0, elevation_deg)
    elev_rad = math.radians(safe_elev)
    am = 1.0 / math.sin(elev_rad)
    intensity = 0.7 ** am
    raw_elev_factor = max(0.0, math.cos(elev_rad))
    return raw_elev_factor * intensity


def coefficients_4d_are_learned(regime_coeff) -> bool:
    """Does this 4D (entity, regime) slot fire on the live read path?

    **The** predicate — not a copy of one.
    :meth:`SolarCalculator.calculate_unit_coefficient_4d` calls this to
    decide between the stored coefficients and the zero-vector, and
    every readiness / gating consumer calls the same function rather
    than restating the condition.  That is deliberate: an unlearned 4D
    regime predicts *zero solar* (unlike 3D, which falls back to a
    seeded ``DEFAULT_SOLAR_COEFF_HEATING`` / ``_COOLING``), so a
    readiness check whose predicate drifted from the read path's would
    silently stop matching the cliff it exists to prevent.  If the fire
    condition below changes, every consumer follows automatically.

    Fires only when the slot is explicitly marked ``learned`` AND
    carries at least one non-zero component — an empty or
    freshly-initialised dict is not a model.

    Accepts any object; a non-dict (missing entity, missing regime,
    corrupt storage) is simply not learned.
    """
    if not isinstance(regime_coeff, dict):
        return False
    if not regime_coeff.get("learned"):
        return False
    return any(regime_coeff.get(k) for k in ("s", "e", "w", "diffuse"))


class SolarCalculator:
    """Calculates solar impact on heating/cooling."""

    def __init__(self, coordinator) -> None:
        """Initialize with reference to coordinator (for configuration/state)."""
        self.coordinator = coordinator

    def calculate_solar_factor(
        self,
        elevation: float,
        azimuth: float,
        cloud_coverage: float,
        *,
        cloud_attenuation_override: float | None = None,
    ) -> float:
        """Calculate solar factor (0.0 - 1.0).

        ``cloud_attenuation_override`` (kw-only) lets the caller substitute
        a precomputed cloud-transmission factor in [0, 1] for the default
        Kasten-from-cloud_coverage value.  Used by the 4D-anchored SNR path
        (#981) so base-EMA weighting consumes the same DNI/DHI signal the
        live read-path uses, instead of the regional cloud_coverage signal
        that can diverge from native DNI/DHI on broken-cloud hours.  When
        DNI/DHI come from the ``kasten_synthetic`` ladder branch the
        override equals ``_kasten_cloud_attenuation(cloud_coverage)`` by
        construction, so output is bit-identical to the default path on
        installs without native DNI/DHI or a local GHI sensor.
        """
        if elevation <= 0.0:
            return 0.0

        elev_factor = _clear_sky_elevation_factor(elevation)

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

        if cloud_attenuation_override is not None:
            cloud_factor = max(0.0, min(1.0, cloud_attenuation_override))
        else:
            cloud_factor = _kasten_cloud_attenuation(cloud_coverage)

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

        base_intensity = _clear_sky_elevation_factor(elevation) * _kasten_cloud_attenuation(cloud_coverage)

        # 3D Decomposition — non-negative basis functions
        az_rad = math.radians(azimuth)
        solar_south = base_intensity * max(0.0, -math.cos(az_rad))
        solar_east = base_intensity * max(0.0, math.sin(az_rad))
        solar_west = base_intensity * max(0.0, -math.sin(az_rad))

        return solar_south, solar_east, solar_west

    def calculate_unit_potential_4d(
        self,
        entity_id: str,
        dni: float,
        dhi: float,
        sun_elev_deg: float,
        sun_azimuth_deg: float,
        screen_config: tuple[bool, bool, bool] | None,
        correction_percent: float,
    ) -> tuple[float, float, float, float]:
        """4D solar potential for a unit (#954 shadow learner).

        Decomposes (DNI, DHI) into per-facade direct components plus an
        isotropic diffuse component, both attenuated by the per-direction
        screen transmittance.  Returns the four-component potential
        vector that the shadow-learner NLMS / Tobit will fit against.

        Geometry (consistent with the 3D ``calculate_solar_vector``
        decomposition):
            cos(elev) factor projects DNI onto the horizontal plane;
            per-facade horizontal cosine selects the component aligned
            with that wall's outward normal.

            pot_s_dir = DNI * max(0, cos(elev) * (-cos(az))) * t_s
            pot_e_dir = DNI * max(0, cos(elev) *   sin(az) ) * t_e
            pot_w_dir = DNI * max(0, cos(elev) * (-sin(az))) * t_w

            pot_diffuse = DHI * 0.5 * mean(t_s, t_e, t_w)

        The fixed 0.5 represents a vertical window seeing half the
        hemisphere; per-facade asymmetry is absorbed by ``c_diff``.

        **Solar-window obstruction gate (v9).**
        When ``coordinator.critical_elev_for_entity(entity_id)[f]`` is
        ``{"low": float|None, "high": float|None}``, the corresponding
        ``pot_dir_facade`` is zeroed whenever ``sun_elev_deg < low`` OR
        ``sun_elev_deg > high``.  This models a solar window: a lower
        horizon (terrain / neighbouring buildings blocks direct beam
        below) and an upper horizon (overhang / terrace blocks above).
        Both boundaries are optional (``None`` = no gate on that side).
        Per-entity because shading geometry differs across windows owned
        by different units.  Diffuse term is intentionally unaffected:
        a scalar gate cannot reproduce diffuse's smooth hemisphere-
        fraction dependence on obstruction geometry.

        Args:
            entity_id: Used only for screen_config lookup at call sites
                via ``coordinator.screen_config_for_entity(entity_id)``;
                this method receives the resolved tuple directly.  Kept
                in the signature for API symmetry with
                ``calculate_unit_coefficient``.
            dni, dhi: Direct-normal and diffuse-horizontal irradiance in
                W/m^2 (typically from ``resolve_dni_dhi``).
            sun_elev_deg: Sun elevation in degrees.  <= 0 -> zero output.
            sun_azimuth_deg: Sun azimuth in degrees, 0=N, 90=E, 180=S, 270=W.
            screen_config: Per-direction screen presence (S, E, W).
            correction_percent: 0-100 screen slider position.

        Returns:
            (pot_s_dir, pot_e_dir, pot_w_dir, pot_diffuse), all >= 0.
        """
        if sun_elev_deg <= 0.0:
            return (0.0, 0.0, 0.0, 0.0)

        t_s, t_e, t_w = self._screen_transmittance_vector(
            correction_percent, screen_config
        )

        elev_rad = math.radians(sun_elev_deg)
        az_rad = math.radians(sun_azimuth_deg)
        cos_elev = math.cos(elev_rad)

        dni_horiz = max(0.0, dni) * cos_elev
        pot_s_dir = dni_horiz * max(0.0, -math.cos(az_rad)) * t_s
        pot_e_dir = dni_horiz * max(0.0, math.sin(az_rad)) * t_e
        pot_w_dir = dni_horiz * max(0.0, -math.sin(az_rad)) * t_w

        crit_fn = getattr(self.coordinator, "critical_elev_for_entity", None)
        if callable(crit_fn) and entity_id is not None:
            crit = crit_fn(entity_id)
            if isinstance(crit, dict):
                pot_by_facade = {"s": pot_s_dir, "e": pot_e_dir, "w": pot_w_dir}
                for facade in ("s", "e", "w"):
                    gate = crit.get(facade)
                    if isinstance(gate, dict):
                        low = gate.get("low")
                        high = gate.get("high")
                        below_low = isinstance(low, (int, float)) and sun_elev_deg < low
                        above_high = isinstance(high, (int, float)) and sun_elev_deg > high
                        if below_low or above_high:
                            pot_by_facade[facade] = 0.0
                    elif isinstance(gate, (int, float)):
                        # Legacy v8 single-float gate -> treated as high-only
                        if sun_elev_deg > gate:
                            pot_by_facade[facade] = 0.0
                pot_s_dir = pot_by_facade["s"]
                pot_e_dir = pot_by_facade["e"]
                pot_w_dir = pot_by_facade["w"]

        pot_diffuse = max(0.0, dhi) * 0.5 * (t_s + t_e + t_w) / 3.0

        return (pot_s_dir, pot_e_dir, pot_w_dir, pot_diffuse)

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

    def calculate_unit_coefficient_4d(
        self, entity_id: str, temp_key: str, mode: str
    ) -> dict[str, float]:
        """4D shadow read of the per-(entity, regime) solar coefficient (#962).

        Strict-shadow counterpart to :meth:`calculate_unit_coefficient`.
        Reads ``ModelState.solar_coefficients_4d_per_unit`` (falls back to
        the coordinator's private ``_solar_coefficients_4d_per_unit`` when
        the model view is missing — happens on test fixtures that build
        ad-hoc ``ModelProxy`` without 4D awareness).

        Semantics differ from the 3D variant in one important way: there
        is NO default-azimuth-decomposition fallback.  4D coefficients
        fire only when explicitly learned — an unlearned regime returns
        the zero-vector ``{"s": 0, "e": 0, "w": 0, "diffuse": 0}``.
        Rationale: 3D has a years-long installed base whose live
        prediction must keep working through a fresh install (hence the
        seeded default).  4D is shadow-only; it has no installed base and
        no consumer relying on a sensible "first-hour" prediction.  A
        silent default would also confound the shadow-vs-live drift
        diagnostic by treating untrained regimes as if they had a real
        coefficient.

        Mode → regime mapping mirrors the 3D method exactly (heating /
        cooling regimes; OFF / DHW route to heating as a stable fallback,
        but those modes contribute 0 solar at higher layers anyway).
        Per-entity solar-scope gate (``is_solar_affected``) is honoured —
        excluded entities return the zero-vector.
        """
        del temp_key  # reserved; coefficients are temperature-blind by design

        # Per-entity solar-scope gate (#962) — same pattern as 3D.
        is_solar_affected_fn = getattr(self.coordinator, "is_solar_affected", None)
        if callable(is_solar_affected_fn) and not is_solar_affected_fn(entity_id):
            return {"s": 0.0, "e": 0.0, "w": 0.0, "diffuse": 0.0}

        regime = "cooling" if mode in (MODE_COOLING, MODE_GUEST_COOLING) else "heating"

        # Prefer the ModelState view; fall back to the coordinator's private
        # attribute for legacy / test ModelProxy fixtures that don't expose
        # the 4D dict via the model surface.
        # TODO: once every test ModelProxy/ModelState construction site
        # threads ``solar_coefficients_4d_per_unit`` through the model
        # view, drop the private-attribute fallback.
        coeffs_4d_map = None
        try:
            coeffs_4d_map = self.coordinator.model.solar_coefficients_4d_per_unit
        except AttributeError:
            coeffs_4d_map = None
        if coeffs_4d_map is None:
            coeffs_4d_map = getattr(
                self.coordinator, "_solar_coefficients_4d_per_unit", None
            )
        if not isinstance(coeffs_4d_map, dict):
            return {"s": 0.0, "e": 0.0, "w": 0.0, "diffuse": 0.0}

        entity_coeffs = coeffs_4d_map.get(entity_id)
        if not isinstance(entity_coeffs, dict):
            return {"s": 0.0, "e": 0.0, "w": 0.0, "diffuse": 0.0}

        regime_coeff = entity_coeffs.get(regime)
        if not isinstance(regime_coeff, dict):
            return {"s": 0.0, "e": 0.0, "w": 0.0, "diffuse": 0.0}

        # Fire only when explicitly learned AND at least one non-zero
        # component.  Empty / freshly-initialised dicts return zero.
        # The condition lives in :func:`coefficients_4d_are_learned` so
        # the 4D readiness gate mirrors this read path instead of
        # re-deriving it — see that function's docstring.
        if coefficients_4d_are_learned(regime_coeff):
            return {
                "s": float(regime_coeff.get("s", 0.0)),
                "e": float(regime_coeff.get("e", 0.0)),
                "w": float(regime_coeff.get("w", 0.0)),
                "diffuse": float(regime_coeff.get("diffuse", 0.0)),
            }
        return {"s": 0.0, "e": 0.0, "w": 0.0, "diffuse": 0.0}

    @staticmethod
    def calculate_unit_solar_impact_4d(
        potential_4d: tuple[float, float, float, float],
        unit_coeff_4d: dict[str, float],
    ) -> float:
        """4D solar impact for a single unit (#962).

        Dot product over four components ``s + e + w + diffuse``.  Per
        CLAUDE.md invariant #1: the caller passes the per-facade-
        attenuated *potential* from :meth:`calculate_unit_potential_4d`
        (which already includes per-direction screen transmittance).  No
        extra transmittance factor is applied here — multiplying again
        would yield the same trans² bug the 3D path was hardened against.

        Components and coefficients are individually non-negative
        (invariant #4); the ``max(0, ...)`` clamp is defence-in-depth.
        """
        pot_s, pot_e, pot_w, pot_diff = potential_4d
        impact = (
            unit_coeff_4d.get("s", 0.0) * pot_s
            + unit_coeff_4d.get("e", 0.0) * pot_e
            + unit_coeff_4d.get("w", 0.0) * pot_w
            + unit_coeff_4d.get("diffuse", 0.0) * pot_diff
        )
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
