"""One-sided dark-equivalent floor for global base EMA (#930).

The floor lifts the bucket toward ``actual + solar_normalization_delta``
only when the existing bucket sits below; the legacy raw-actual EMA
owns the downward path.  This is structurally immune to COP-ceiling
feedback (overestimated coefficients can only inflate dark_target,
which lifts the bucket — never pulls it down) so no plausibility gate
on per-unit ``learned`` flags is required.
"""
import pytest
from unittest.mock import MagicMock
from custom_components.heating_analytics.learning import LearningManager
from custom_components.heating_analytics.observation import HourlyObservation, ModelState, LearningConfig
from custom_components.heating_analytics.const import MODE_HEATING


def _obs(temp, actual, solar_factor, delta, base_expected):
    return HourlyObservation(
        timestamp=MagicMock(),
        hour=12,
        avg_temp=temp,
        inertia_temp=temp,
        temp_key=str(int(round(temp))),
        effective_wind=0.0,
        wind_bucket="normal",
        bucket_counts={"normal": 60},
        avg_humidity=50.0,
        solar_factor=solar_factor,
        solar_vector=(solar_factor, 0.0, 0.0),
        solar_impact_raw=delta,
        effective_solar_impact=delta,
        total_energy_kwh=actual,
        learning_energy_kwh=actual,
        guest_impact_kwh=0.0,
        expected_kwh=0.0,
        base_expected_kwh=base_expected,
        unit_breakdown={"sensor.vp": actual},
        unit_expected={"sensor.vp": 0.0},
        unit_expected_base={"sensor.vp": base_expected},
        aux_impact_kwh=0.0,
        aux_fraction=0.0,
        is_aux_dominant=False,
        sample_count=60,
        unit_modes={"sensor.vp": MODE_HEATING},
        solar_normalization_delta=delta,
        correction_percent=100.0,
    )


def _setup_model(base_val, coeff_val, learned=True):
    correlation_data = {"12": {"normal": base_val}}
    return ModelState(
        correlation_data=correlation_data,
        correlation_data_per_unit={"sensor.vp": {"12": {"normal": base_val}}},
        observation_counts={},
        aux_coefficients={},
        aux_coefficients_per_unit={},
        solar_coefficients_per_unit={"sensor.vp": {"heating": {"s": coeff_val, "e": 0.0, "w": 0.0, "learned": learned}}},
        learned_u_coefficient=None,
        learning_buffer_global={},
        learning_buffer_per_unit={},
        learning_buffer_aux_per_unit={},
        learning_buffer_solar_per_unit={},
    )


def _setup_config(model, solar_calc, unit_min_base=None):
    return LearningConfig(
        learning_enabled=True,
        solar_enabled=True,
        learning_rate=0.1,
        balance_point=17.0,
        energy_sensors=["sensor.vp"],
        aux_impact=0.0,
        get_predicted_unit_base_fn=lambda eid, t, w, at: model.correlation_data_per_unit[eid].get(t, {}).get(w, 0.0),
        solar_calculator=solar_calc,
        unit_min_base=unit_min_base,
    )


def test_lifts_underconverged_bucket():
    """Bucket below dark_target on sunny hours is lifted toward it."""
    lm = LearningManager()
    # Bucket starts at 0.25 (drifted down).  True demand 0.5, solar 0.3, actual 0.2.
    # dark_target = 0.5; current 0.25 < 0.5 → lift.
    model = _setup_model(0.25, 0.3, learned=True)
    solar_calc = MagicMock()
    solar_calc.calculate_unit_coefficient.return_value = {"s": 0.3, "e": 0.0, "w": 0.0, "learned": True}
    solar_calc.calculate_unit_solar_impact.return_value = 0.3
    config = _setup_config(model, solar_calc)

    for _ in range(50):
        current_base = model.correlation_data["12"]["normal"]
        obs = _obs(12.2, 0.2, 1.0, 0.3, current_base)
        lm.process_learning(obs, model, config)

    final_base = model.correlation_data["12"]["normal"]
    assert final_base > 0.30, f"expected lift toward 0.5, got {final_base}"
    assert final_base <= 0.5 + 1e-9, f"must not overshoot dark_target, got {final_base}"


def test_no_lift_when_bucket_above_dark_target():
    """Bucket already above dark_target uses legacy raw-actual EMA — drifts down."""
    lm = LearningManager()
    # Bucket starts at 0.6, dark_target = 0.5 → legacy path → drift toward 0.2.
    model = _setup_model(0.6, 0.3, learned=True)
    solar_calc = MagicMock()
    solar_calc.calculate_unit_coefficient.return_value = {"s": 0.3, "e": 0.0, "w": 0.0, "learned": True}
    solar_calc.calculate_unit_solar_impact.return_value = 0.3
    config = _setup_config(model, solar_calc)

    for _ in range(50):
        current_base = model.correlation_data["12"]["normal"]
        obs = _obs(12.2, 0.2, 1.0, 0.3, current_base)
        lm.process_learning(obs, model, config)

    final_base = model.correlation_data["12"]["normal"]
    # SNR-attenuated downward step on sunny hours pulls toward 0.2 from 0.6.
    assert final_base < 0.6, f"legacy downward path should apply, got {final_base}"


def test_no_runaway_under_overestimated_coefficient():
    """Overestimated coeff inflates dark_target but cannot push bucket past it.

    Coeff=0.5 (true 0.3), so delta=0.5 from per-hour pipeline.  dark_target =
    0.2 + 0.5 = 0.7.  Bucket starts at 0.5 and climbs toward 0.7 from below;
    once it reaches 0.7 the legacy raw-actual path takes over and pulls it
    back down toward 0.2.  Bounded above by 0.7 — no runaway.
    """
    lm = LearningManager()
    model = _setup_model(0.5, 0.5, learned=True)
    solar_calc = MagicMock()
    solar_calc.calculate_unit_coefficient.return_value = {"s": 0.5, "e": 0.0, "w": 0.0, "learned": True}
    solar_calc.calculate_unit_solar_impact.return_value = 0.5
    config = _setup_config(model, solar_calc, unit_min_base={"sensor.vp": 1.0})

    for _ in range(200):
        current_base = model.correlation_data["12"]["normal"]
        obs = _obs(12.2, 0.2, 1.0, 0.5, current_base)
        lm.process_learning(obs, model, config)

    final_base = model.correlation_data["12"]["normal"]
    assert final_base > 0.5, f"expected lift above starting 0.5, got {final_base}"
    assert final_base <= 0.7 + 1e-9, f"bounded by dark_target=0.7, got {final_base}"


def test_lift_gate_blocks_unlearned_coefficient():
    """Plausibility gate suppresses the lift when no regime coefficient is
    learned.  Prevents seeded/default coefficients from inflating dark_target
    and drifting the bucket upward on day-1 of a fresh install or post-reset.
    Saturation skip remains reachable under this branch (#929 wrong-direction
    protection); modulating-regime hours fall back to legacy raw-actual EMA.
    """
    lm = LearningManager()
    model = _setup_model(0.25, 0.3, learned=False)
    solar_calc = MagicMock()
    solar_calc.calculate_unit_coefficient.return_value = {"s": 0.3, "e": 0.0, "w": 0.0, "learned": False}
    solar_calc.calculate_unit_solar_impact.return_value = 0.3
    # Modulating-regime hour (base > delta) — gate closed, falls to legacy path.
    config = _setup_config(model, solar_calc, unit_min_base={"sensor.vp": 1.0})

    for _ in range(50):
        current_base = model.correlation_data["12"]["normal"]
        # actual=0.3, delta=0.1 → dark_target=0.4, base ≥ 0.25 (legacy modulating)
        obs = _obs(12.2, 0.3, 1.0, 0.1, current_base)
        lm.process_learning(obs, model, config)

    final_base = model.correlation_data["12"]["normal"]
    # No upward lift through the floor; legacy raw-actual EMA applies.
    assert final_base <= 0.30 + 1e-6, f"lift suppressed when no coeff learned, got {final_base}"


def test_saturation_skip_reachable_when_lift_gate_closed():
    """When no coefficient is learned AND the hour is solar-saturated,
    the #929 saturation skip must still fire — the lift gate does not
    swallow it.  Bucket must not drift on these hours.
    """
    lm = LearningManager()
    # base=0.05, delta=0.3 → saturated (base < delta).  No coeff learned.
    model = _setup_model(0.05, 0.3, learned=False)
    solar_calc = MagicMock()
    solar_calc.calculate_unit_coefficient.return_value = {"s": 0.3, "e": 0.0, "w": 0.0, "learned": False}
    solar_calc.calculate_unit_solar_impact.return_value = 0.3
    config = _setup_config(model, solar_calc, unit_min_base={"sensor.vp": 1.0})

    for _ in range(50):
        current_base = model.correlation_data["12"]["normal"]
        obs = _obs(12.2, 0.0, 1.0, 0.3, current_base)
        lm.process_learning(obs, model, config)

    final_base = model.correlation_data["12"]["normal"]
    # Bucket frozen at initial value: lift suppressed by gate, raw-actual
    # path skipped by saturation guard.
    assert final_base == pytest.approx(0.05), f"saturation skip should freeze bucket, got {final_base}"


if __name__ == "__main__":
    pytest.main([__file__])
