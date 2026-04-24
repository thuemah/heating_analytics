"""Tests for F1 (#854): split DirectMeter vs WeightedSmear for solar-delta.

`apply_strategies_to_global_model` sums per-hour contributions from all
strategies and then historically applied ``solar_normalization_delta``
to the combined total.  That's correct for DirectMeter (raw electrical
values that need dark-sky normalization) but wrong for WeightedSmear:
synthetic_kwh_el is already solar-weighted via the smearing weights,
so adding the delta is double-correction.

After F1 the delta applies only to the DirectMeter portion.  On pure
Track C installs delta is typically 0 anyway (no non-MPC NLMS coeffs),
so the guard is primarily a semantic safeguard for mixed installs.
"""
from unittest.mock import MagicMock

import pytest

from custom_components.heating_analytics.const import MODE_HEATING
from custom_components.heating_analytics.learning import LearningManager
from custom_components.heating_analytics.observation import (
    DirectMeter,
    ModelState,
    WeightedSmear,
)


def _log_entry(h: int, *, solar_delta: float = 0.0, unit_breakdown: dict | None = None):
    return {
        "hour": h,
        "timestamp": f"2026-04-20T{h:02d}:00:00",
        "temp_key": "10",
        "wind_bucket": "normal",
        "inertia_temp": 10.0,
        "temp": 10.0,
        "effective_wind": 2.0,
        "solar_factor": 0.0,
        "solar_normalization_delta": solar_delta,
        "unit_modes": {},
        "unit_breakdown": unit_breakdown or {},
    }


def _model(correlation_data=None):
    return ModelState(
        correlation_data=correlation_data or {},
        correlation_data_per_unit={},
        observation_counts={},
        aux_coefficients={},
        aux_coefficients_per_unit={},
        solar_coefficients_per_unit={},
        learned_u_coefficient=None,
    )


def _noop_parse(ts: str):
    """Minimal parse_datetime stub that extracts the hour only."""
    from datetime import datetime
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _track_c_dist(synth_el_per_hour: float):
    """Build a track_c_distribution arg matching apply_strategies_to_global_model.

    ``apply_strategies_to_global_model`` clears the WeightedSmear distribution
    at the top of the call and only re-sets it when ``track_c_distribution``
    is provided.  Tests that want smear contributions must route through
    this argument, not call ``strategy.set_distribution`` directly.
    """
    return [
        {
            "datetime": f"2026-04-20T{h:02d}:00:00",
            "synthetic_kwh_el": synth_el_per_hour,
        }
        for h in range(24)
    ]


# ---------------------------------------------------------------------------
# Pure Track C (only WeightedSmear) — delta applies to NOTHING
# ---------------------------------------------------------------------------


class TestPureTrackCDeltaAgnostic:
    """A pure Track C install has no DirectMeter — delta never applies."""

    def test_zero_delta_preserves_smear_sum(self):
        """Baseline: delta=0, sum is exactly the smeared contribution."""
        lm = LearningManager()
        smear = WeightedSmear("sensor.mpc", use_synthetic=True)

        day_logs = [_log_entry(h) for h in range(24)]
        model = _model()
        lm.apply_strategies_to_global_model(
            day_logs=day_logs,
            track_c_distribution=_track_c_dist(0.5),
            strategies={"sensor.mpc": smear},
            model=model,
            learning_rate=1.0,
            balance_point=15.0,
            wind_threshold=8.0,
            extreme_wind_threshold=10.8,
            parse_datetime_fn=_noop_parse,
        )

        # 24 hours × 0.5 kWh, bucket settles at 0.5.
        assert model.correlation_data["10"]["normal"] == pytest.approx(0.5, abs=0.001)

    def test_nonzero_delta_does_not_inflate_smear(self):
        """If delta is somehow nonzero on a pure Track C install, WeightedSmear
        values must still flow through unchanged — double-correction guard."""
        lm = LearningManager()
        smear = WeightedSmear("sensor.mpc", use_synthetic=True)

        # Hypothetical nonzero delta (shouldn't happen in practice on pure
        # Track C, but guards the code path).
        day_logs = [_log_entry(h, solar_delta=0.2) for h in range(24)]
        model = _model()
        lm.apply_strategies_to_global_model(
            day_logs=day_logs,
            track_c_distribution=_track_c_dist(0.5),
            strategies={"sensor.mpc": smear},
            model=model,
            learning_rate=1.0,
            balance_point=15.0,
            wind_threshold=8.0,
            extreme_wind_threshold=10.8,
            parse_datetime_fn=_noop_parse,
        )

        # Pre-F1: 24 × (0.5 + 0.2) = 16.8 → bucket 0.7. Double-correction.
        # Post-F1: delta applies to direct_kwh=0, so bucket stays 0.5.
        assert model.correlation_data["10"]["normal"] == pytest.approx(0.5, abs=0.001)


# ---------------------------------------------------------------------------
# Pure Track A / Track B (only DirectMeter) — delta applies as before
# ---------------------------------------------------------------------------


class TestDirectMeterUnaffected:
    """F1 must not change behaviour when only DirectMeter strategies exist."""

    def test_zero_delta_passthrough(self):
        lm = LearningManager()
        dm = DirectMeter("sensor.heater")

        day_logs = [
            _log_entry(h, unit_breakdown={"sensor.heater": 0.5})
            for h in range(24)
        ]
        model = _model()
        lm.apply_strategies_to_global_model(
            day_logs=day_logs,
            track_c_distribution=None,
            strategies={"sensor.heater": dm},
            model=model,
            learning_rate=1.0,
            balance_point=15.0,
            wind_threshold=8.0,
            extreme_wind_threshold=10.8,
            parse_datetime_fn=_noop_parse,
        )

        assert model.correlation_data["10"]["normal"] == pytest.approx(0.5, abs=0.001)

    def test_delta_lifts_direct_unchanged_post_f1(self):
        """Delta still normalises DirectMeter contributions to dark-sky."""
        lm = LearningManager()
        dm = DirectMeter("sensor.heater")

        day_logs = [
            _log_entry(h, solar_delta=0.2, unit_breakdown={"sensor.heater": 0.3})
            for h in range(24)
        ]
        model = _model()
        lm.apply_strategies_to_global_model(
            day_logs=day_logs,
            track_c_distribution=None,
            strategies={"sensor.heater": dm},
            model=model,
            learning_rate=1.0,
            balance_point=15.0,
            wind_threshold=8.0,
            extreme_wind_threshold=10.8,
            parse_datetime_fn=_noop_parse,
        )

        # direct_kwh = 0.3 + delta 0.2 = 0.5 — unchanged from pre-F1.
        assert model.correlation_data["10"]["normal"] == pytest.approx(0.5, abs=0.001)


# ---------------------------------------------------------------------------
# Mixed install — delta applies to DirectMeter, not to WeightedSmear
# ---------------------------------------------------------------------------


class TestMixedInstallSplit:
    """The key fix: on mixed installs, delta must not inflate smear values."""

    def test_delta_applies_only_to_direct_portion(self):
        lm = LearningManager()
        smear = WeightedSmear("sensor.mpc", use_synthetic=True)
        dm = DirectMeter("sensor.panel")

        # On every hour: smear contributes 0.4, panel contributes 0.1, delta 0.3.
        # Post-F1: direct = 0.1 + 0.3 = 0.4, smear = 0.4, total = 0.8.
        # Pre-F1: total = (0.4 + 0.1) + 0.3 = 0.8 — identical in sum but
        # conceptually different.  Keep this test as a behavioural baseline;
        # the saturation test below actually distinguishes pre/post.
        day_logs = [
            _log_entry(h, solar_delta=0.3, unit_breakdown={"sensor.panel": 0.1})
            for h in range(24)
        ]
        model = _model()
        lm.apply_strategies_to_global_model(
            day_logs=day_logs,
            track_c_distribution=_track_c_dist(0.4),
            strategies={"sensor.mpc": smear, "sensor.panel": dm},
            model=model,
            learning_rate=1.0,
            balance_point=15.0,
            wind_threshold=8.0,
            extreme_wind_threshold=10.8,
            parse_datetime_fn=_noop_parse,
        )

        assert model.correlation_data["10"]["normal"] == pytest.approx(0.8, abs=0.001)

    def test_smear_saturation_not_double_corrected(self):
        """When delta > direct's headroom, pre-F1 would have lifted smear.

        Construct a scenario where direct_kwh + delta goes NEGATIVE if the
        clamp is applied to DirectMeter alone — confirming delta stays on
        the DirectMeter side and doesn't spill over into smear.
        """
        lm = LearningManager()
        smear = WeightedSmear("sensor.mpc", use_synthetic=True)
        dm = DirectMeter("sensor.panel")

        # Panel contributes 0.05, delta is -0.3 (cooling-dominant or saturated).
        # Post-F1: direct_kwh = max(0, 0.05 + (-0.3)) = 0. smear stays 0.5.
        #          total = 0.5.
        # Pre-F1:  total = max(0, (0.05 + 0.5) + (-0.3)) = 0.25 — wrong, because
        #          the negative correction ate into smear's value.
        day_logs = [
            _log_entry(h, solar_delta=-0.3, unit_breakdown={"sensor.panel": 0.05})
            for h in range(24)
        ]
        model = _model()
        lm.apply_strategies_to_global_model(
            day_logs=day_logs,
            track_c_distribution=_track_c_dist(0.5),
            strategies={"sensor.mpc": smear, "sensor.panel": dm},
            model=model,
            learning_rate=1.0,
            balance_point=15.0,
            wind_threshold=8.0,
            extreme_wind_threshold=10.8,
            parse_datetime_fn=_noop_parse,
        )

        # Post-F1: smear (0.5) protected from negative delta spill.
        assert model.correlation_data["10"]["normal"] == pytest.approx(0.5, abs=0.001)

    def test_mixed_zero_delta_matches_pre_f1(self):
        """With delta=0 the split is equivalent to the old sum — regression."""
        lm = LearningManager()
        smear = WeightedSmear("sensor.mpc", use_synthetic=True)
        dm = DirectMeter("sensor.panel")

        day_logs = [
            _log_entry(h, unit_breakdown={"sensor.panel": 0.15})
            for h in range(24)
        ]
        model = _model()
        lm.apply_strategies_to_global_model(
            day_logs=day_logs,
            track_c_distribution=_track_c_dist(0.4),
            strategies={"sensor.mpc": smear, "sensor.panel": dm},
            model=model,
            learning_rate=1.0,
            balance_point=15.0,
            wind_threshold=8.0,
            extreme_wind_threshold=10.8,
            parse_datetime_fn=_noop_parse,
        )

        assert model.correlation_data["10"]["normal"] == pytest.approx(0.55, abs=0.001)
