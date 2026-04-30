"""Tests for 3D solar vector usage in forecast paths (#832).

Verifies:
1. override_solar_vector takes priority over override_solar_factor
   in calculate_total_power
2. Forecast hourly path uses coeff_w for west-exposed buildings
3. estimate_daily_avg_solar_vector produces meaningful 3-tuples
"""
import pytest
import math
from unittest.mock import MagicMock
from datetime import date, timedelta

from custom_components.heating_analytics.solar import SolarCalculator
from custom_components.heating_analytics.statistics import StatisticsManager
from tests.helpers import CoordinatorModelMixin


class _MockCoord(CoordinatorModelMixin):
    solar_azimuth = 180
    balance_point = 17.0
    solar_enabled = True
    solar_correction_percent = 100.0
    auxiliary_heating_active = False

    def __init__(self):
        self.data = {"solar_factor": 0.0}
        self._solar_coefficients_per_unit = {}
        self._correlation_data = {}
        self._correlation_data_per_unit = {}
        self._aux_coefficients = {}
        self._aux_coefficients_per_unit = {}
        self._observation_counts = {}
        self._learned_u_coefficient = None
        self._learning_buffer_global = {}
        self._learning_buffer_per_unit = {}
        self._learning_buffer_aux_per_unit = {}
        self._learning_buffer_solar_per_unit = {}
        self._daily_history = {}
        self._hourly_log = []
        self._unit_modes = {}
        self.energy_sensors = []
        self.solar = SolarCalculator(self)
        self._model_cache = None

    def _get_predicted_kwh(self, temp_key, wind_bucket, temp):
        return 1.0

    def _get_wind_bucket(self, wind):
        return "normal"

    def get_unit_mode(self, entity_id):
        return "heating"


class TestOverrideSolarVectorPriority:
    """Verify override_solar_vector takes priority over override_solar_factor."""

    def test_vector_overrides_factor(self):
        """When both override_solar_vector and override_solar_factor are provided,
        the vector should be used and the factor ignored."""
        coord = _MockCoord()
        coord.energy_sensors = ["unit_a"]
        coord._solar_coefficients_per_unit = {"unit_a": {
            "heating": {"s": 1.0, "e": 0.0, "w": 0.0},
            "cooling": {"s": 0.0, "e": 0.0, "w": 0.0},
        }}
        coord._correlation_data_per_unit = {"unit_a": {"10": {"normal": 1.0}}}
        coord._observation_counts = {"unit_a": {"10": {"normal": 10}}}
        stats = StatisticsManager(coord)

        # Factor says solar_factor=0.5 → with azimuth 180, that gives vector (0.5, 0, 0)
        # Vector says (0.0, 0.0, 0.0) → zero solar
        res_vector_zero = stats.calculate_total_power(
            temp=10.0, effective_wind=0.0, solar_impact=0.0,
            is_aux_active=False,
            override_solar_factor=0.5,
            override_solar_vector=(0.0, 0.0, 0.0),
        )

        # Only factor, no vector → factor reconstructs to (0.5, 0, 0)
        res_factor_only = stats.calculate_total_power(
            temp=10.0, effective_wind=0.0, solar_impact=0.0,
            is_aux_active=False,
            override_solar_factor=0.5,
        )

        # With vector=(0,0,0), solar impact should be zero
        solar_with_vector = res_vector_zero["breakdown"]["solar_reduction_kwh"]
        # With factor=0.5 and coeff_s=1.0, solar impact should be positive
        solar_with_factor = res_factor_only["breakdown"]["solar_reduction_kwh"]

        assert solar_with_vector == 0.0, "Vector override should produce zero solar"
        assert solar_with_factor > 0.0, "Factor-only should produce positive solar"

    def test_vector_none_falls_back_to_factor(self):
        """When override_solar_vector is None, factor is used."""
        coord = _MockCoord()
        coord.energy_sensors = ["unit_a"]
        coord._solar_coefficients_per_unit = {"unit_a": {
            "heating": {"s": 1.0, "e": 0.0, "w": 0.0},
            "cooling": {"s": 0.0, "e": 0.0, "w": 0.0},
        }}
        coord._correlation_data_per_unit = {"unit_a": {"10": {"normal": 1.0}}}
        coord._observation_counts = {"unit_a": {"10": {"normal": 10}}}
        stats = StatisticsManager(coord)

        res = stats.calculate_total_power(
            temp=10.0, effective_wind=0.0, solar_impact=0.0,
            is_aux_active=False,
            override_solar_factor=0.5,
            override_solar_vector=None,
        )

        solar = res["breakdown"]["solar_reduction_kwh"]
        assert solar > 0.0, "With vector=None, factor should be used"


class TestForecastUsesWestCoefficient:
    """Forecast should use coeff_w for buildings with west window exposure."""

    def test_afternoon_forecast_uses_coeff_w(self):
        """A west-facing building should see higher solar impact in afternoon
        forecast than if coeff_w were zero."""
        coord = _MockCoord()
        coord.energy_sensors = ["unit_west"]
        coord._correlation_data_per_unit = {"unit_west": {"10": {"normal": 1.0}}}
        coord._observation_counts = {"unit_west": {"10": {"normal": 10}}}
        stats = StatisticsManager(coord)

        # Afternoon sun at azimuth 240 (SW), elevation 30, clear sky
        elev, azim, cloud = 30.0, 240.0, 0.0
        solar_vector = coord.solar.calculate_solar_vector(elev, azim, cloud)

        # With west coefficient (mode-stratified per #868).
        coord._solar_coefficients_per_unit = {"unit_west": {
            "heating": {"s": 0.3, "e": 0.0, "w": 0.5},
            "cooling": {"s": 0.0, "e": 0.0, "w": 0.0},
        }}
        res_with_w = stats.calculate_total_power(
            temp=10.0, effective_wind=0.0, solar_impact=0.0,
            is_aux_active=False,
            override_solar_vector=solar_vector,
        )

        # Without west coefficient
        coord._solar_coefficients_per_unit = {"unit_west": {
            "heating": {"s": 0.3, "e": 0.0, "w": 0.0},
            "cooling": {"s": 0.0, "e": 0.0, "w": 0.0},
        }}
        coord._model_cache = None  # Invalidate cache
        res_without_w = stats.calculate_total_power(
            temp=10.0, effective_wind=0.0, solar_impact=0.0,
            is_aux_active=False,
            override_solar_vector=solar_vector,
        )

        solar_with = res_with_w["breakdown"]["solar_reduction_kwh"]
        solar_without = res_without_w["breakdown"]["solar_reduction_kwh"]

        assert solar_with > solar_without, (
            f"West coefficient should increase afternoon solar impact: "
            f"with_w={solar_with:.4f}, without_w={solar_without:.4f}"
        )

        # Verify the vector itself has a significant W component
        assert solar_vector[2] > 0.1, f"West component should be significant at az=240: {solar_vector}"


class _HassConfig:
    latitude = 59.9  # Oslo, Norway
    longitude = 10.75
    time_zone = "Europe/Oslo"
    # Real ``HassConfig`` instances always carry ``elevation``.  Pre-#878
    # the broad ``except Exception`` in ``solar.get_approx_sun_pos``
    # swallowed any AttributeError silently; after #878 narrowed the
    # except to ``(TypeError, ValueError)`` an AttributeError here
    # propagates and the test fails.  Provide the attribute explicitly
    # to mirror the production HA contract.
    elevation = 0


class _MockCoordWithHass(CoordinatorModelMixin):
    solar_azimuth = 180
    balance_point = 17.0
    solar_enabled = True
    solar_correction_percent = 100.0

    def __init__(self):
        self.hass = MagicMock()
        self.hass.config = _HassConfig()
        self._solar_coefficients_per_unit = {}
        self._correlation_data = {}
        self._correlation_data_per_unit = {}
        self._aux_coefficients = {}
        self._aux_coefficients_per_unit = {}
        self._observation_counts = {}
        self._learned_u_coefficient = None
        self._learning_buffer_global = {}
        self._learning_buffer_per_unit = {}
        self._learning_buffer_aux_per_unit = {}
        self._learning_buffer_solar_per_unit = {}
        self._daily_history = {}
        self._hourly_log = []
        self._model_cache = None
        self.solar = SolarCalculator(self)


class TestEstimateDailyAvgSolarVector:
    """estimate_daily_avg_solar_vector should produce meaningful 3-tuples."""

    def test_south_dominant_winter(self):
        """On a short winter day, south component should dominate.

        Simulates a winter-like sun track: 6 hours near due south.
        """
        calc = SolarCalculator(_MockCoord())
        # Winter day at 60N: sun is up ~6 hours, low elevation, narrow arc
        # Azimuths roughly 150-210 degrees, elevation 5-10 degrees
        winter_hours = [
            (5.0, 155.0),   # morning: SE
            (8.0, 165.0),   # late morning
            (10.0, 180.0),  # noon
            (8.0, 195.0),   # early afternoon
            (5.0, 205.0),   # afternoon: SW
        ]
        total_s, total_e, total_w = 0.0, 0.0, 0.0
        for elev, azim in winter_hours:
            s, e, w = calc.calculate_solar_vector(elev, azim, 0.0)
            total_s += s
            total_e += e
            total_w += w

        assert total_s > total_e, f"South should dominate in winter: s={total_s:.4f}, e={total_e:.4f}"
        assert total_s > total_w, f"South should dominate in winter: s={total_s:.4f}, w={total_w:.4f}"
        assert total_s > 0.0

    def test_east_west_both_present_summer(self):
        """On a long summer day, both east and west should be non-zero.

        Simulates a summer-like sun track: 16 hours, wide arc.
        """
        calc = SolarCalculator(_MockCoord())
        # Summer day at 60N: sun up ~16 hours, wide arc from NE to NW
        summer_hours = [
            (10.0, 70.0),   # early morning: ENE
            (25.0, 100.0),  # morning: E
            (40.0, 130.0),  # late morning: SE
            (50.0, 160.0),  # approaching noon
            (53.0, 180.0),  # noon: S
            (50.0, 200.0),  # early afternoon
            (40.0, 230.0),  # afternoon: SW
            (25.0, 260.0),  # late afternoon: W
            (10.0, 290.0),  # evening: WNW
        ]
        total_s, total_e, total_w = 0.0, 0.0, 0.0
        for elev, azim in summer_hours:
            s, e, w = calc.calculate_solar_vector(elev, azim, 0.0)
            total_s += s
            total_e += e
            total_w += w

        assert total_e > 0.0, f"East should be positive in summer: e={total_e:.4f}"
        assert total_w > 0.0, f"West should be positive in summer: w={total_w:.4f}"
        # E and W should be roughly balanced for a symmetric day
        if total_e > 0.01 and total_w > 0.01:
            ratio = max(total_e, total_w) / min(total_e, total_w)
            assert ratio < 2.0, f"E/W should be roughly symmetric: e={total_e:.4f}, w={total_w:.4f}"

    def test_overcast_reduces_all_components(self):
        """Heavy cloud coverage should reduce all components proportionally."""
        coord = _MockCoordWithHass()
        calc = coord.solar
        test_date = date(2026, 4, 15)
        s_clear, e_clear, w_clear = calc.estimate_daily_avg_solar_vector(test_date, cloud_coverage=0.0)
        s_cloudy, e_cloudy, w_cloudy = calc.estimate_daily_avg_solar_vector(test_date, cloud_coverage=80.0)

        if s_clear > 0.01:
            assert s_cloudy < s_clear, "Cloudy south should be less than clear"
        if e_clear > 0.01:
            assert e_cloudy < e_clear, "Cloudy east should be less than clear"
        if w_clear > 0.01:
            assert w_cloudy < w_clear, "Cloudy west should be less than clear"

    def test_matches_scalar_magnitude(self):
        """The vector magnitude should be related to the scalar factor."""
        coord = _MockCoordWithHass()
        calc = coord.solar
        test_date = date(2026, 4, 15)
        scalar = calc.estimate_daily_avg_solar_factor(test_date, cloud_coverage=30.0)
        s, e, w = calc.estimate_daily_avg_solar_vector(test_date, cloud_coverage=30.0)

        # The scalar uses the Kelvin Twist azimuth weighting which is different
        # from the 3D decomposition, so they won't match exactly.
        # But both should be zero/non-zero together.
        if scalar > 0.0:
            assert (s + e + w) > 0.0, "If scalar is positive, vector sum should be too"
