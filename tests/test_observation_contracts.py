"""Tests for the observation data contracts, strategies, and ObservationCollector (#775, #776).

Covers the critical invariants:
1. ObservationCollector.reset() preserves dict aliases (in-place clearing)
2. HourlyObservation is truly frozen
3. Legacy kwargs → contract roundtrip produces identical results
4. LearningStrategy implementations conform to protocol
5. build_strategies produces correct assignment from config
"""
from datetime import datetime
from unittest.mock import MagicMock
import pytest

from custom_components.heating_analytics.observation import (
    HourlyObservation,
    ModelState,
    LearningConfig,
    ObservationCollector,
    LearningStrategy,
    DirectMeter,
    WeightedSmear,
    build_strategies,
)
from custom_components.heating_analytics.learning import LearningManager
from custom_components.heating_analytics.const import MODE_HEATING


# --- ObservationCollector ---


class TestObservationCollector:
    """Tests for ObservationCollector state management."""

    def test_reset_clears_all_scalars(self):
        """After accumulation and reset, all scalars are zeroed."""
        c = ObservationCollector()
        c.accumulate_weather(
            temp=5.0, effective_wind=3.0, wind_bucket="normal",
            solar_factor=0.5, solar_vector=(0.1, 0.2),
            is_aux_active=True, current_time=datetime(2023, 1, 1, 12, 5),
        )
        c.energy_hour = 1.5
        c.expected_energy_hour = 1.3
        c.aux_impact_hour = 0.2
        c.orphaned_aux = 0.1
        c.last_minute_processed = 5

        c.reset()

        assert c.wind_sum == 0.0
        assert c.temp_sum == 0.0
        assert c.solar_sum == 0.0
        assert c.sample_count == 0
        assert c.aux_count == 0
        assert c.energy_hour == 0.0
        assert c.expected_energy_hour == 0.0
        assert c.aux_impact_hour == 0.0
        assert c.orphaned_aux == 0.0
        assert c.start_time is None
        assert c.last_minute_processed is None

    def test_reset_preserves_dict_identity(self):
        """reset() must clear dicts IN-PLACE so external aliases stay valid.

        This is the most critical invariant: the coordinator holds aliases
        like self._hourly_delta_per_unit = self._collector.delta_per_unit.
        If reset() replaced the dict, the alias would point to stale data.
        """
        c = ObservationCollector()

        # Grab references BEFORE any mutation (same as coordinator __init__)
        delta_ref = c.delta_per_unit
        expected_ref = c.expected_per_unit
        expected_base_ref = c.expected_base_per_unit
        aux_breakdown_ref = c.aux_breakdown
        wind_values_ref = c.wind_values
        bucket_ref = c.bucket_counts

        # Simulate accumulation
        c.delta_per_unit["sensor.test"] = 1.5
        c.expected_per_unit["sensor.test"] = 1.3
        c.expected_base_per_unit["sensor.test"] = 1.6
        c.aux_breakdown["sensor.test"] = {"allocated": 0.1, "overflow": 0.0}
        c.wind_values.append(5.0)
        c.bucket_counts["normal"] = 30

        # Reset
        c.reset()

        # References must be the SAME objects (identity, not just equality)
        assert c.delta_per_unit is delta_ref
        assert c.expected_per_unit is expected_ref
        assert c.expected_base_per_unit is expected_base_ref
        assert c.aux_breakdown is aux_breakdown_ref
        assert c.wind_values is wind_values_ref
        assert c.bucket_counts is bucket_ref

        # And they must be empty
        assert len(delta_ref) == 0
        assert len(expected_ref) == 0
        assert len(expected_base_ref) == 0
        assert len(aux_breakdown_ref) == 0
        assert len(wind_values_ref) == 0
        assert bucket_ref["normal"] == 0
        assert bucket_ref["high_wind"] == 0
        assert bucket_ref["extreme_wind"] == 0

    def test_accumulate_weather_increments(self):
        """Weather accumulation increments all counters correctly."""
        c = ObservationCollector()
        t = datetime(2023, 1, 1, 12, 0)

        c.accumulate_weather(5.0, 3.0, "normal", 0.5, (0.1, 0.2), False, t)
        c.accumulate_weather(6.0, 4.0, "high_wind", 0.6, (0.3, 0.4), True, t)

        assert c.sample_count == 2
        assert c.temp_sum == pytest.approx(11.0)
        assert c.wind_sum == pytest.approx(7.0)  # Not stored in wind_sum; stored separately
        assert c.solar_sum == pytest.approx(1.1)
        assert c.solar_vector_s_sum == pytest.approx(0.4)
        assert c.solar_vector_e_sum == pytest.approx(0.6)
        assert c.bucket_counts["normal"] == 1
        assert c.bucket_counts["high_wind"] == 1
        assert c.aux_count == 1  # Only second call had is_aux_active=True
        assert c.start_time == datetime(2023, 1, 1, 12, 0)

    def test_accumulate_expected_distributes_to_units(self):
        """Expected accumulation populates per-unit dicts correctly."""
        c = ObservationCollector()
        breakdown = {
            "sensor.a": {"net_kwh": 1.0, "base_kwh": 1.2, "aux_reduction_kwh": 0.1, "overflow_kwh": 0.0},
            "sensor.b": {"net_kwh": 0.5, "base_kwh": 0.6, "aux_reduction_kwh": 0.0, "overflow_kwh": 0.0},
        }

        c.accumulate_expected(
            fraction=0.5, prediction_rate=2.0, aux_impact_rate=0.3,
            unit_breakdown=breakdown, orphaned_part=0.05,
        )

        assert c.expected_energy_hour == pytest.approx(1.0)  # 2.0 * 0.5
        assert c.aux_impact_hour == pytest.approx(0.15)  # 0.3 * 0.5
        assert c.expected_per_unit["sensor.a"] == pytest.approx(0.5)  # 1.0 * 0.5
        assert c.expected_per_unit["sensor.b"] == pytest.approx(0.25)  # 0.5 * 0.5
        assert c.expected_base_per_unit["sensor.a"] == pytest.approx(0.6)  # 1.2 * 0.5
        assert c.orphaned_aux == pytest.approx(0.025)  # 0.05 * 0.5


# --- HourlyObservation ---


class TestHourlyObservation:
    """Tests for HourlyObservation immutability."""

    def _make_obs(self, **overrides):
        defaults = dict(
            timestamp=datetime(2023, 1, 1, 12, 0),
            hour=12, avg_temp=5.0, inertia_temp=5.5, temp_key="6",
            effective_wind=3.0, wind_bucket="normal",
            bucket_counts={"normal": 30}, avg_humidity=65.0, solar_factor=0.1,
            solar_vector=(0.1, 0.2), solar_impact_raw=0.05,
            effective_solar_impact=0.04, total_energy_kwh=1.5,
            learning_energy_kwh=1.4, guest_impact_kwh=0.1,
            expected_kwh=1.3, base_expected_kwh=1.6,
            unit_breakdown={}, unit_expected={}, unit_expected_base={},
            aux_impact_kwh=0.3, aux_fraction=0.0, is_aux_dominant=False,
            sample_count=30, unit_modes={},
        )
        defaults.update(overrides)
        return HourlyObservation(**defaults)

    def test_frozen(self):
        """HourlyObservation must reject attribute mutation."""
        obs = self._make_obs()
        with pytest.raises(AttributeError):
            obs.avg_temp = 99.0

    def test_all_fields_accessible(self):
        """All required fields are accessible after construction."""
        obs = self._make_obs(avg_temp=7.5, wind_bucket="high_wind")
        assert obs.avg_temp == 7.5
        assert obs.wind_bucket == "high_wind"
        assert obs.hour == 12

    def test_optional_defaults(self):
        """Optional fields have correct defaults."""
        obs = self._make_obs()
        assert obs.forecasted_kwh is None
        assert obs.forecast_source is None
        assert obs.recommendation_state == "none"
        assert obs.was_cooldown_active is False


# --- LearningManager legacy roundtrip ---


class TestLearningLegacyRoundtrip:
    """Verify that the legacy kwargs path produces identical results."""

    def _make_kwargs(self):
        """Standard learning kwargs (Track A, no aux, no solar)."""
        return {
            "temp_key": "5",
            "wind_bucket": "normal",
            "avg_temp": 5.0,
            "total_energy_kwh": 1.0,
            "base_expected_kwh": 1.0,
            "solar_impact": 0.0,
            "avg_solar_vector": (0.0, 0.0),
            "is_aux_active": False,
            "aux_impact": 0.0,
            "learning_enabled": True,
            "solar_enabled": False,
            "learning_rate": 0.1,
            "balance_point": 17.0,
            "energy_sensors": ["sensor.heater"],
            "hourly_bucket_counts": {"normal": 60},
            "hourly_sample_count": 60,
            "correlation_data": {"5": {"normal": 0.8}},
            "correlation_data_per_unit": {"sensor.heater": {"5": {"normal": 0.8}}},
            "aux_coefficients": {},
            "learning_buffer_global": {},
            "learning_buffer_per_unit": {},
            "observation_counts": {},
            "hourly_delta_per_unit": {"sensor.heater": 1.0},
            "hourly_expected_per_unit": {},
            "hourly_expected_base_per_unit": {"sensor.heater": 1.0},
            "aux_coefficients_per_unit": {},
            "learning_buffer_aux_per_unit": {},
            "solar_coefficients_per_unit": {},
            "learning_buffer_solar_per_unit": {},
            "solar_calculator": MagicMock(),
            "get_predicted_unit_base_fn": MagicMock(return_value=0.8),
            "unit_modes": {"sensor.heater": MODE_HEATING},
            "aux_affected_entities": None,
            "has_guest_activity": False,
            "is_cooldown_active": False,
        }

    def test_legacy_and_contract_produce_same_status(self):
        """Legacy kwargs path and contract path yield the same learning_status."""
        lm = LearningManager()
        kwargs = self._make_kwargs()

        # Legacy path
        result_legacy = lm.process_learning(**kwargs)

        assert result_legacy["learning_status"] == "active"
        assert result_legacy["model_updated"] is True

    def test_legacy_skipped_when_no_samples(self):
        """Legacy path correctly skips learning when sample_count is 0."""
        lm = LearningManager()
        kwargs = self._make_kwargs()
        kwargs["hourly_sample_count"] = 0

        result = lm.process_learning(**kwargs)
        assert result["learning_status"] == "skipped_no_data"
        assert result["model_updated"] is False

    def test_legacy_cooldown_state(self):
        """Legacy path correctly passes cooldown state."""
        lm = LearningManager()
        kwargs = self._make_kwargs()
        kwargs["is_cooldown_active"] = True

        result = lm.process_learning(**kwargs)
        assert result["learning_status"] == "cooldown_post_aux"
        assert result["model_updated"] is False


# --- Learning Strategies (#776) ---


class TestDirectMeter:
    """Tests for DirectMeter strategy."""

    def test_returns_kwh_from_unit_breakdown(self):
        dm = DirectMeter("sensor.heater1")
        log = {"unit_breakdown": {"sensor.heater1": 1.5, "sensor.heater2": 0.8}}
        assert dm.get_hourly_contribution(12, 0.05, log) == 1.5

    def test_returns_none_when_zero(self):
        dm = DirectMeter("sensor.heater1")
        log = {"unit_breakdown": {"sensor.heater1": 0.0}}
        assert dm.get_hourly_contribution(12, 0.05, log) is None

    def test_returns_none_when_missing(self):
        dm = DirectMeter("sensor.heater1")
        log = {"unit_breakdown": {"sensor.other": 1.0}}
        assert dm.get_hourly_contribution(12, 0.05, log) is None

    def test_returns_none_when_no_breakdown(self):
        dm = DirectMeter("sensor.heater1")
        log = {}
        assert dm.get_hourly_contribution(12, 0.05, log) is None

    def test_ignores_weight(self):
        """DirectMeter must ignore the weight parameter."""
        dm = DirectMeter("sensor.heater1")
        log = {"unit_breakdown": {"sensor.heater1": 2.0}}
        assert dm.get_hourly_contribution(5, 0.001, log) == 2.0
        assert dm.get_hourly_contribution(5, 0.999, log) == 2.0

    def test_conforms_to_protocol(self):
        assert isinstance(DirectMeter("x"), LearningStrategy)


class TestWeightedSmear:
    """Tests for WeightedSmear strategy."""

    def test_synthetic_mode_reads_distribution(self):
        ws = WeightedSmear("sensor.mpc_pump", use_synthetic=True)
        ws.set_distribution({
            10: {"synthetic_kwh_el": 0.8},
            11: {"synthetic_kwh_el": 1.2},
        })
        assert ws.get_hourly_contribution(10, 0.05, {}) == 0.8
        assert ws.get_hourly_contribution(11, 0.05, {}) == 1.2
        assert ws.get_hourly_contribution(12, 0.05, {}) is None

    def test_synthetic_mode_returns_none_without_distribution(self):
        ws = WeightedSmear("sensor.mpc_pump", use_synthetic=True)
        assert ws.get_hourly_contribution(10, 0.05, {}) is None

    def test_weighted_mode_uses_weight(self):
        ws = WeightedSmear("sensor.floor", use_synthetic=False)
        ws.set_daily_total(24.0)
        assert ws.get_hourly_contribution(5, 0.1, {}) == pytest.approx(2.4)
        assert ws.get_hourly_contribution(5, 0.0, {}) is None

    def test_weighted_mode_returns_none_zero_total(self):
        ws = WeightedSmear("sensor.floor", use_synthetic=False)
        ws.set_daily_total(0.0)
        assert ws.get_hourly_contribution(5, 0.1, {}) is None

    def test_conforms_to_protocol(self):
        assert isinstance(WeightedSmear("x"), LearningStrategy)


class TestBuildStrategies:
    """Tests for build_strategies factory."""

    def test_all_direct_when_no_track_c(self):
        strategies = build_strategies(
            ["sensor.a", "sensor.b"], track_c_enabled=False, mpc_managed_sensor=None
        )
        assert len(strategies) == 2
        assert all(isinstance(s, DirectMeter) for s in strategies.values())

    def test_mpc_gets_weighted_smear(self):
        strategies = build_strategies(
            ["sensor.mpc", "sensor.panel"],
            track_c_enabled=True,
            mpc_managed_sensor="sensor.mpc",
        )
        assert isinstance(strategies["sensor.mpc"], WeightedSmear)
        assert strategies["sensor.mpc"].use_synthetic is True
        assert isinstance(strategies["sensor.panel"], DirectMeter)

    def test_track_c_without_mpc_sensor(self):
        """Track C enabled but no mpc_managed_sensor → all DirectMeter."""
        strategies = build_strategies(
            ["sensor.a"], track_c_enabled=True, mpc_managed_sensor=None
        )
        assert isinstance(strategies["sensor.a"], DirectMeter)

    def test_empty_sensors(self):
        strategies = build_strategies([], track_c_enabled=False, mpc_managed_sensor=None)
        assert strategies == {}
