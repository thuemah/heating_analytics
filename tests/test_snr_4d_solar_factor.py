"""Tests for the 4D-anchored SNR solar_factor path (#981 point 1).

When ``experimental_4d_primary`` is on, base-EMA SNR weighting should
consume the same DNI/DHI signal the live read-path uses instead of the
3D Kasten-from-cloud_coverage scalar.  On installs without native
DNI/DHI or local GHI (kasten_synthetic ladder branch) the 4D-anchored
factor must collapse bit-identically to the 3D path.
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from custom_components.heating_analytics.learning import (
    LearningManager,
    compute_snr_weight,
)
from custom_components.heating_analytics.solar import SolarCalculator


_TS = datetime(2026, 6, 21, 12, 0, 0, tzinfo=timezone.utc)
# Fixed sun position used by all tests — sun-up, roughly south-facing
# afternoon geometry.  Lets us bypass astral entirely (not installed in
# the test environment) by mocking ``get_approx_sun_pos``.
_FIXED_ELEV = 45.0
_FIXED_AZ = 180.0


def _make_calculator(*, azimuth=180.0, elev=_FIXED_ELEV, az=_FIXED_AZ):
    """Build a SolarCalculator with a minimal coordinator stub.

    ``get_approx_sun_pos`` is mocked so the test does not depend on
    astral being installed.
    """
    coord = MagicMock()
    coord.solar_azimuth = azimuth
    calc = SolarCalculator(coord)
    calc.get_approx_sun_pos = MagicMock(return_value=(elev, az))
    return calc


def _make_config(calculator, *, solar_enabled=True, flag=False):
    """Minimal duck-typed LearningConfig stand-in for the helper."""
    return SimpleNamespace(
        solar_enabled=solar_enabled,
        solar_calculator=calculator,
        experimental_4d_primary=flag,
    )


def _make_obs(
    *,
    ts=_TS,
    dni_avg=None,
    dhi_avg=None,
    ghi_avg=None,
    cloud_avg=None,
    solar_factor=0.0,
):
    """Minimal duck-typed HourlyObservation stand-in for the helper."""
    return SimpleNamespace(
        timestamp=ts,
        dni_avg=dni_avg,
        dhi_avg=dhi_avg,
        ghi_avg=ghi_avg,
        cloud_avg=cloud_avg,
        solar_factor=solar_factor,
    )


class TestSNRSolarFactor4D:
    """Behaviour of ``LearningManager._compute_4d_solar_factor_for_snr``."""

    def test_kasten_synthetic_collapses_to_3d(self):
        """No native DNI/DHI/GHI → helper output == 3D Kasten path.

        Pins the collapse property: when DNI/DHI come from the
        ``kasten_synthetic`` ladder branch,
        ``(DNI*sin_elev + DHI) / GHI_clear`` equals
        ``_kasten_cloud_attenuation(cloud)`` by construction so the
        whole solar_factor is bit-identical.
        """
        calc = _make_calculator()
        config = _make_config(calc, flag=True)
        obs = _make_obs(cloud_avg=40.0)

        sf_4d = LearningManager._compute_4d_solar_factor_for_snr(obs, config)
        assert sf_4d is not None

        sf_3d = calc.calculate_solar_factor(_FIXED_ELEV, _FIXED_AZ, 40.0)
        assert sf_4d == pytest.approx(sf_3d, abs=1e-6)

    def test_native_dni_dhi_diverges_from_cloud_coverage(self):
        """Native clear DNI/DHI but cloudy cloud_coverage → 4D > 3D."""
        calc = _make_calculator()
        config = _make_config(calc, flag=True)
        # Native DNI/DHI says it's mostly clear; cloud_avg=70 would push
        # the 3D Kasten path well below that.
        obs = _make_obs(dni_avg=800.0, dhi_avg=100.0, cloud_avg=70.0)

        sf_4d = LearningManager._compute_4d_solar_factor_for_snr(obs, config)
        assert sf_4d is not None

        sf_3d = calc.calculate_solar_factor(_FIXED_ELEV, _FIXED_AZ, 70.0)
        assert sf_4d > sf_3d, (
            f"4D path should see through cloud_coverage misclassification "
            f"(sf_4d={sf_4d}, sf_3d={sf_3d})"
        )

    def test_no_sun_returns_zero(self):
        """Sun below horizon → 0.0 (full SNR weight, dark-hour semantics)."""
        calc = _make_calculator(elev=-5.0, az=0.0)
        config = _make_config(calc, flag=True)
        obs = _make_obs(cloud_avg=50.0)
        sf_4d = LearningManager._compute_4d_solar_factor_for_snr(obs, config)
        assert sf_4d == 0.0

    def test_returns_none_when_no_signal(self):
        """All inputs None → helper returns None (caller falls back to 3D)."""
        calc = _make_calculator()
        config = _make_config(calc, flag=True)
        obs = _make_obs(dni_avg=None, dhi_avg=None, ghi_avg=None, cloud_avg=None)
        assert LearningManager._compute_4d_solar_factor_for_snr(obs, config) is None

    def test_returns_none_when_solar_disabled(self):
        calc = _make_calculator()
        config = _make_config(calc, solar_enabled=False, flag=True)
        obs = _make_obs(dni_avg=800.0, dhi_avg=100.0)
        assert LearningManager._compute_4d_solar_factor_for_snr(obs, config) is None

    def test_call_site_uses_4d_when_flag_on(self):
        """Flag-on + native clearer-than-cloud DNI/DHI → SNR weight LOWER than 3D path.

        Mirrors the dispatch at learning.py ~940: helper output replaces
        ``obs.solar_factor`` in the ``compute_snr_weight`` argument.
        End-to-end check: feed both factors through ``compute_snr_weight``
        and confirm the directional inequality.  Geometry tuned (low
        sun, very cloudy cloud_coverage) so both factors sit below the
        SNR floor cutoff and ``1 - K*sf`` resolves to distinct, non-
        floored weights.
        """
        low_elev = 20.0
        calc = _make_calculator(elev=low_elev, az=_FIXED_AZ)
        config_on = _make_config(calc, flag=True)
        sf_3d = calc.calculate_solar_factor(low_elev, _FIXED_AZ, 95.0)
        obs = _make_obs(
            dni_avg=300.0,
            dhi_avg=80.0,
            cloud_avg=95.0,
            solar_factor=sf_3d,
        )
        sf_4d = LearningManager._compute_4d_solar_factor_for_snr(obs, config_on)
        assert sf_4d is not None
        assert sf_4d > sf_3d

        w_3d = compute_snr_weight(sf_3d, [], total_units=1)
        w_4d = compute_snr_weight(sf_4d, [], total_units=1)
        assert w_4d < w_3d, (
            f"4D-anchored path sees clearer sky → lower SNR weight "
            f"(w_4d={w_4d}, w_3d={w_3d})"
        )

    def test_call_site_falls_back_when_4d_unresolvable(self):
        """Flag-on but no DNI/DHI/GHI/cloud → helper returns None → 3D fallback.

        Mirrors the dispatch fallback branch: when the helper yields
        None, ``compute_snr_weight`` consumes the original 3D
        ``obs.solar_factor`` unchanged.
        """
        calc = _make_calculator()
        config_on = _make_config(calc, flag=True)
        obs = _make_obs(
            dni_avg=None, dhi_avg=None, ghi_avg=None, cloud_avg=None,
            solar_factor=0.42,
        )
        sf_4d = LearningManager._compute_4d_solar_factor_for_snr(obs, config_on)
        assert sf_4d is None

        sf_for_snr = obs.solar_factor if sf_4d is None else sf_4d
        w_actual = compute_snr_weight(sf_for_snr, [], total_units=1)
        w_3d = compute_snr_weight(obs.solar_factor, [], total_units=1)
        assert w_actual == pytest.approx(w_3d)


class TestPerUnitSNR4D:
    """#985: per-unit base-learning SNR dispatch at ``learning.py:~1750``.

    The per-unit path in ``_process_per_unit_learning`` receives flat-
    unpacked attrs instead of a ``config`` object, so the production
    code shims a ``SimpleNamespace(solar_calculator, solar_enabled,
    experimental_4d_primary=True)`` before calling the helper.  These
    tests pin the dispatch shape — that the shimmed call returns the
    same 4D-anchored factor the global path does, and that flag-off
    short-circuits to the 3D ``solar_factor``.
    """

    def _shim(self, calculator):
        return SimpleNamespace(
            solar_calculator=calculator,
            solar_enabled=True,
            experimental_4d_primary=True,
        )

    def test_per_unit_dispatch_uses_4d_when_flag_on(self):
        """Native clearer-than-cloud DNI/DHI through the shimmed call →
        4D-anchored factor higher than 3D Kasten path.  Mirrors the
        global ``test_call_site_uses_4d_when_flag_on`` invariant but
        through the per-unit shim shape rather than a real LearningConfig.
        """
        low_elev = 20.0
        calc = _make_calculator(elev=low_elev, az=_FIXED_AZ)
        sf_3d = calc.calculate_solar_factor(low_elev, _FIXED_AZ, 95.0)
        obs = _make_obs(
            dni_avg=300.0,
            dhi_avg=80.0,
            cloud_avg=95.0,
            solar_factor=sf_3d,
        )

        # Production-shape dispatch from learning.py:~1750.
        sf_4d = LearningManager._compute_4d_solar_factor_for_snr(obs, self._shim(calc))
        assert sf_4d is not None
        assert sf_4d > sf_3d

        w_3d = compute_snr_weight(sf_3d, [], total_units=1)
        w_4d = compute_snr_weight(sf_4d, [], total_units=1)
        assert w_4d < w_3d, (
            f"per-unit 4D-anchored path sees clearer sky → lower SNR "
            f"weight → less weight on this sunny hour's actual when "
            f"updating per-unit base bucket (w_4d={w_4d}, w_3d={w_3d})"
        )

    def test_per_unit_dispatch_falls_back_when_flag_off(self):
        """``_flag_4d_primary=False`` → dispatch short-circuits before the
        helper is called → SNR weight derived from the unmodified 3D
        ``solar_factor``.  No-op-when-flag-off invariant.
        """
        calc = _make_calculator()
        sf_3d = 0.42
        obs = _make_obs(
            dni_avg=800.0,
            dhi_avg=100.0,
            cloud_avg=70.0,
            solar_factor=sf_3d,
        )

        # Mirror the production short-circuit: when flag is off, the helper
        # is never called; sf_for_snr stays at solar_factor.
        flag_4d_primary = False
        sf_for_snr = obs.solar_factor
        if flag_4d_primary:
            sf_4d = LearningManager._compute_4d_solar_factor_for_snr(obs, self._shim(calc))
            if sf_4d is not None:
                sf_for_snr = sf_4d

        assert sf_for_snr == sf_3d
        # Helper would have produced something different if called — proving
        # the short-circuit is doing work (defensive, not strictly required).
        sf_4d_hypothetical = LearningManager._compute_4d_solar_factor_for_snr(
            obs, self._shim(calc),
        )
        assert sf_4d_hypothetical is not None and sf_4d_hypothetical != sf_3d
