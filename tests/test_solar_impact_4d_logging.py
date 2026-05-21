"""Tests for per-hour 4D shadow impact logging (#954 commit 9).

`_process_per_unit_learning` accumulates per-entity ``c_4d · pot_4d``
across the per-unit loop and surfaces the totals via the result dict
keys ``solar_impact_4d_kwh`` and ``solar_normalization_delta_4d``.
The hourly_processor reads those keys and writes them to
``hourly_log`` when the 4D shadow path is enabled.

These tests target the unit-level accumulator: with the 4D dict
populated the keys appear and carry the expected sums; with the 4D
dict ``None`` the keys are absent (gating the log emission).
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

from custom_components.heating_analytics.learning import LearningManager
from custom_components.heating_analytics.const import MODE_HEATING


def _solar_calc_stub(potential_4d: tuple[float, float, float, float]):
    """Solar-calculator mock returning a fixed 4D potential per call."""
    sc = MagicMock()
    sc.get_approx_sun_pos = MagicMock(return_value=(45.0, 180.0))
    sc.calculate_unit_potential_4d = MagicMock(return_value=potential_4d)
    sc.calculate_unit_coefficient = MagicMock(return_value={"s": 0.0, "e": 0.0, "w": 0.0})
    sc.calculate_unit_solar_impact = MagicMock(return_value=0.0)
    sc._screen_transmittance_vector = MagicMock(return_value=(1.0, 1.0, 1.0))
    sc.coordinator = MagicMock()
    return sc


def _obs_stub(timestamp: datetime):
    obs = MagicMock()
    obs.timestamp = timestamp
    obs.battery_filtered_potential_4d = (0.0, 0.0, 0.0, 0.0)
    # Real numeric values so resolve_dni_dhi() inside the 4D
    # preamble doesn't raise on MagicMock arithmetic and trip
    # the broad-except that disables ``shadow_4d_enabled``.
    obs.dni_avg = 600.0
    obs.dhi_avg = 200.0
    obs.ghi_avg = 700.0
    obs.cloud_avg = 20.0
    return obs


def _base_kwargs(solar_calc, obs, coeffs_4d, buffers_4d):
    return {
        "temp_key": "5",
        "wind_bucket": "normal",
        "avg_temp": 5.0,
        "avg_solar_vector": (0.6, 0.3, 0.0),
        "total_energy_kwh": 2.0,
        "base_expected_kwh": 4.0,
        "energy_sensors": ["sensor.u1"],
        "hourly_delta_per_unit": {"sensor.u1": 2.0},
        "solar_enabled": True,
        "learning_rate": 0.1,
        "solar_calculator": solar_calc,
        "get_predicted_unit_base_fn": MagicMock(return_value=4.0),
        "learning_buffer_per_unit": {},
        "correlation_data_per_unit": {"sensor.u1": {"5": {"normal": 4.0}}},
        "observation_counts": {},
        "is_aux_active": False,
        "aux_coefficients_per_unit": {},
        "learning_buffer_aux_per_unit": {},
        "solar_coefficients_per_unit": {},
        "learning_buffer_solar_per_unit": {},
        "balance_point": 17.0,
        "unit_modes": {"sensor.u1": MODE_HEATING},
        "hourly_expected_per_unit": {"sensor.u1": 4.0},
        "hourly_expected_base_per_unit": {"sensor.u1": 4.0},
        "aux_affected_entities": [],
        "correction_percent": 100.0,
        "screen_config": (False, False, False),
        "solar_factor": 0.5,
        "solar_coefficients_4d_per_unit": coeffs_4d,
        "learning_buffer_solar_4d_per_unit": buffers_4d,
        "obs": obs,
    }


def test_solar_impact_4d_logged_with_seeded_coeffs():
    """With 4D dict populated, result carries the predicted impact sum.

    Seeded coefficients (0.4, 0.2, 0.1, 0.05) against a fixed potential
    (1.0, 0.5, 0.0, 0.8) should produce predicted_4d = 0.4*1.0 + 0.2*0.5 +
    0.1*0.0 + 0.05*0.8 = 0.54.  Single heating unit -> total = 0.54;
    delta also = 0.54 (heating sign positive).
    """
    manager = LearningManager()
    coeffs_4d = {
        "sensor.u1": {
            "heating": {
                "s": 0.4, "e": 0.2, "w": 0.1, "diffuse": 0.05,
                "learned": True,
            },
            "cooling": {},
        }
    }
    buffers_4d: dict = {}
    solar_calc = _solar_calc_stub(potential_4d=(1.0, 0.5, 0.0, 0.8))
    obs = _obs_stub(datetime(2026, 6, 21, 12, 0, tzinfo=timezone.utc))

    result = manager._process_per_unit_learning(
        **_base_kwargs(solar_calc, obs, coeffs_4d, buffers_4d)
    )

    assert "solar_impact_4d_kwh" in result
    assert "solar_normalization_delta_4d" in result
    assert abs(result["solar_impact_4d_kwh"] - 0.54) < 1e-6
    assert abs(result["solar_normalization_delta_4d"] - 0.54) < 1e-6


def test_solar_impact_4d_absent_when_sun_below_horizon():
    """Bug regression (#954 Codex review): with the 4D dicts present
    (the production state — coordinator initialises both to ``{}``)
    but the sun below horizon, the gate previously fired on
    ``solar_coefficients_4d_per_unit is not None`` (always True) and
    emitted ``0.0`` for both log fields.  Diagnose's base_model_4d_shadow
    treated those zeros as legitimate "tagged" hours.

    Fix: gate emission on ``shadow_4d_fired`` (set only when an entity
    actually invokes the 4D learner).  Below-horizon hours never reach
    that branch — fields must be ABSENT.
    """
    manager = LearningManager()
    coeffs_4d: dict = {}  # production-shape: present but empty
    buffers_4d: dict = {}
    solar_calc = _solar_calc_stub(potential_4d=(1.0, 0.5, 0.0, 0.8))
    # Force sun below horizon: stub returns elev=-5.
    solar_calc.get_approx_sun_pos = MagicMock(return_value=(-5.0, 180.0))
    # Winter midnight timestamp (also makes intent visible).
    obs = _obs_stub(datetime(2026, 1, 15, 0, 0, tzinfo=timezone.utc))

    result = manager._process_per_unit_learning(
        **_base_kwargs(solar_calc, obs, coeffs_4d, buffers_4d)
    )

    assert result.get("solar_impact_4d_kwh") is None
    assert result.get("solar_normalization_delta_4d") is None


def test_solar_impact_4d_absent_when_shadow_disabled():
    """Without the 4D dict, the result dict omits both keys.

    The hourly_processor's gate ``learning_result.get(...) is not None``
    then leaves the log fields out — preserving legacy log shape on
    installs where the 4D shadow has not been activated.
    """
    manager = LearningManager()
    solar_calc = _solar_calc_stub(potential_4d=(0.0, 0.0, 0.0, 0.0))
    obs = _obs_stub(datetime(2026, 6, 21, 12, 0, tzinfo=timezone.utc))

    kwargs = _base_kwargs(solar_calc, obs, None, None)
    result = manager._process_per_unit_learning(**kwargs)

    assert "solar_impact_4d_kwh" not in result
    assert "solar_normalization_delta_4d" not in result
