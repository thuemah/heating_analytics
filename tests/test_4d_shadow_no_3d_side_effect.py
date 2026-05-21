"""Pin the bit-identical 3D path with 4D shadow enabled (#954).

The independent code review on the #954 4D-shadow PR flagged that the
"bit-identical 3D path" claim was previously asserted only by absence
of edits to existing 3D tests.  This test closes that gap directly:
run ``LearningManager.process_learning`` twice from identical 3D state
— once with the 4D shadow dicts left as ``None`` (legacy / disabled),
once with them passed as empty dicts (4D shadow path enabled and
exercised by the per-unit loop) — and assert that every 3D-targeted
dict ends up bit-identical post-call.

A sentinel ``MagicMock`` on the unified ``_learn_unit_solar_coefficient``
filters calls by ``components`` kwarg to verify Run B actually entered
the 4D branching code path (``components=("s","e","w","diffuse")``) so
the equality assertion can't pass vacuously.

Ships single-hour variant only; multi-hour convergence-trajectory
variant deferred as a follow-up.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from custom_components.heating_analytics.const import MODE_HEATING
from custom_components.heating_analytics.learning import LearningManager
from custom_components.heating_analytics.observation import (
    HourlyObservation,
    LearningConfig,
    ModelState,
)


ENTITY = "sensor.vp"


def _make_obs() -> HourlyObservation:
    """Build an obs that opens the 4D gate (timestamp + DNI/DHI present)."""
    return HourlyObservation(
        timestamp=datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
        hour=12,
        avg_temp=12.0,
        inertia_temp=12.0,
        temp_key="12",
        effective_wind=0.0,
        wind_bucket="normal",
        bucket_counts={"normal": 60},
        avg_humidity=50.0,
        solar_factor=0.5,
        solar_vector=(0.5, 0.2, 0.1),
        solar_impact_raw=0.3,
        effective_solar_impact=0.3,
        total_energy_kwh=0.4,
        learning_energy_kwh=0.4,
        guest_impact_kwh=0.0,
        expected_kwh=0.0,
        base_expected_kwh=1.0,
        unit_breakdown={ENTITY: 0.4},
        unit_expected={ENTITY: 0.0},
        unit_expected_base={ENTITY: 1.0},
        aux_impact_kwh=0.0,
        aux_fraction=0.0,
        is_aux_dominant=False,
        sample_count=60,
        unit_modes={ENTITY: MODE_HEATING},
        solar_normalization_delta=0.3,
        correction_percent=100.0,
        # 4D-gate inputs:
        dni_avg=600.0,
        dhi_avg=150.0,
        ghi_avg=None,
        cloud_avg=20.0,
    )


def _make_3d_state() -> dict:
    """Independent (deep) copy of all 3D-targeted dicts the learner mutates."""
    return {
        "correlation_data": {"12": {"normal": 0.5}},
        "correlation_data_per_unit": {ENTITY: {"12": {"normal": 0.5}}},
        "observation_counts": {},
        "aux_coefficients": {},
        "aux_coefficients_per_unit": {},
        "solar_coefficients_per_unit": {
            ENTITY: {"heating": {"s": 0.3, "e": 0.0, "w": 0.0, "learned": True}}
        },
        "learning_buffer_global": {},
        "learning_buffer_per_unit": {},
        "learning_buffer_aux_per_unit": {},
        "learning_buffer_solar_per_unit": {},
    }


def _make_model(state: dict, *, with_4d: bool) -> ModelState:
    return ModelState(
        correlation_data=state["correlation_data"],
        correlation_data_per_unit=state["correlation_data_per_unit"],
        observation_counts=state["observation_counts"],
        aux_coefficients=state["aux_coefficients"],
        aux_coefficients_per_unit=state["aux_coefficients_per_unit"],
        solar_coefficients_per_unit=state["solar_coefficients_per_unit"],
        learned_u_coefficient=None,
        learning_buffer_global=state["learning_buffer_global"],
        learning_buffer_per_unit=state["learning_buffer_per_unit"],
        learning_buffer_aux_per_unit=state["learning_buffer_aux_per_unit"],
        learning_buffer_solar_per_unit=state["learning_buffer_solar_per_unit"],
        solar_coefficients_4d_per_unit=({} if with_4d else None),
        learning_buffer_solar_4d_per_unit=({} if with_4d else None),
    )


def _make_solar_calc() -> MagicMock:
    """SolarCalculator stub that satisfies both the 3D and 4D code paths."""
    calc = MagicMock()
    # 3D coefficient read (used by base-floor lift gate / impact calc).
    calc.calculate_unit_coefficient.return_value = {
        "s": 0.3, "e": 0.0, "w": 0.0, "learned": True,
    }
    calc.calculate_unit_solar_impact.return_value = 0.3
    # 4D-gate dependencies.
    calc.get_approx_sun_pos.return_value = (45.0, 180.0)  # elev > 0 → gate stays open
    calc.calculate_unit_potential_4d.return_value = (0.5, 0.2, 0.1, 0.3)
    # screen helpers (defensive — different code paths consult these).
    calc._screen_transmittance_vector = MagicMock(return_value=(1.0, 1.0, 1.0))
    calc._screen_transmittance = MagicMock(return_value=1.0)
    return calc


def _make_config(model: ModelState, solar_calc: MagicMock) -> LearningConfig:
    return LearningConfig(
        learning_enabled=True,
        solar_enabled=True,
        learning_rate=0.1,
        balance_point=17.0,
        energy_sensors=[ENTITY],
        aux_impact=0.0,
        get_predicted_unit_base_fn=(
            lambda eid, t, w, at: model.correlation_data_per_unit
            .get(eid, {}).get(t, {}).get(w, 0.0)
        ),
        solar_calculator=solar_calc,
    )


def test_3d_state_unchanged_by_4d_shadow():
    """Bit-identical 3D state regardless of whether the 4D shadow runs.

    Closes the verification gap from the #954 independent review.
    """
    obs = _make_obs()

    state_A = _make_3d_state()  # 4D disabled
    state_B = _make_3d_state()  # 4D enabled
    # Sanity: identical inputs.
    assert state_A == state_B

    lm_A = LearningManager()
    lm_B = LearningManager()  # separate instance — independent internal state

    # Sentinel on the unified learner: filter calls by components kwarg
    # to identify 4D-branch invocations (components has length 4).
    sentinel_B = MagicMock(wraps=lm_B._learn_unit_solar_coefficient)
    lm_B._learn_unit_solar_coefficient = sentinel_B
    sentinel_A = MagicMock(wraps=lm_A._learn_unit_solar_coefficient)
    lm_A._learn_unit_solar_coefficient = sentinel_A

    def _called_4d(sentinel: MagicMock) -> bool:
        return any(
            len(call.kwargs.get("components", ("s", "e", "w"))) == 4
            for call in sentinel.call_args_list
        )

    model_A = _make_model(state_A, with_4d=False)
    model_B = _make_model(state_B, with_4d=True)

    solar_calc_A = _make_solar_calc()
    solar_calc_B = _make_solar_calc()
    config_A = _make_config(model_A, solar_calc_A)
    config_B = _make_config(model_B, solar_calc_B)

    lm_A.process_learning(obs=obs, model=model_A, config=config_A)
    lm_B.process_learning(obs=obs, model=model_B, config=config_B)

    # --- Sentinel-mock sanity: Run B exercised the 4D branching code
    # path; Run A did not.  Without this, the equality check below
    # could pass vacuously if both runs unexpectedly skipped 4D for an
    # unrelated reason (gate closed, obs invalid, mock surface drift).
    assert _called_4d(sentinel_B), (
        "Run B did not invoke the 4D learner — the 4D gate closed "
        "(check obs.timestamp / dni_avg / dhi_avg and solar_calc mocks)."
    )
    assert not _called_4d(sentinel_A), (
        "Run A invoked the 4D learner despite 4D dicts being None — "
        "the strict-shadow opt-in semantic is broken."
    )

    # --- Bit-identical 3D state.
    # Every dict the per-unit learner is documented to potentially
    # mutate; if 4D leakage exists, one of these will differ.
    keys = [
        "correlation_data",
        "correlation_data_per_unit",
        "solar_coefficients_per_unit",
        "learning_buffer_per_unit",
        "learning_buffer_solar_per_unit",
        "aux_coefficients",
        "aux_coefficients_per_unit",
        "learning_buffer_aux_per_unit",
        "observation_counts",
        "learning_buffer_global",
    ]
    for key in keys:
        assert state_A[key] == state_B[key], (
            f"3D state diverged at {key!r} when 4D shadow ran:\n"
            f"  A (4D=None) = {state_A[key]}\n"
            f"  B (4D={{}}) = {state_B[key]}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
