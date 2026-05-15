"""Tests for the experimental tail-aware solar redistribution (#948).

Redistribution kernel applied at hour H:

    predicted_solar_H = (1 − α_now) · coef · potential_H
                      + Σ_{k=1..K} w_k · coef · potential_at_H-k

where:
  - K = 3 (lookback hours)
  - α is the configured redistribution fraction
  - α_now = α if H is low-elev eligible (sun_elev < 30 AND solar_factor > 0.05),
            else 0 (no scaling of current credit on high-elev hours)
  - w_k = α · exp(-k/τ) / Σ_{j=1..K} exp(-j/τ)
  - Past-hour contributions are only included when that past hour was
    low-elev eligible.

α = 0.0 is a bit-identical no-op.
"""
from __future__ import annotations

import math
from datetime import timedelta
from unittest.mock import MagicMock

import pytest

from homeassistant.util import dt as dt_util

from custom_components.heating_analytics.const import (
    ATTR_SOLAR_FACTOR,
    DOMAIN,
    MODE_HEATING,
)
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.solar import SolarCalculator
from custom_components.heating_analytics.statistics import StatisticsManager


class MockHass:
    def __init__(self):
        self.states = MagicMock()
        self.states.get = MagicMock(return_value=None)
        self.data = {DOMAIN: {}}
        self.config_entries = MagicMock()
        self.bus = MagicMock()
        self.is_running = True


@pytest.fixture
def mock_hass():
    return MockHass()


# Fixed potential vector used in all tests; coef · pot = 1.0 · 1.0 = 1.0.
_POT_VEC = (1.0, 0.0, 0.0)


def _build_coord(
    mock_hass,
    *,
    alpha: float,
    tau: float = 2.0,
):
    entry = MagicMock()
    entry.data = {
        "energy_sensors": ["sensor.heater_1"],
        "aux_affected_entities": [],
        "screen_affected_entities": [],
        "screen_south": False,
        "screen_east": False,
        "screen_west": False,
        "outdoor_temp_sensor": "sensor.outdoor_temp",
        "balance_point": 15.0,
        "wind_speed_sensor": "sensor.wind_speed",
        "solar_enabled": True,
        "solar_redistribution_alpha": alpha,
        "solar_redistribution_tau_hours": tau,
    }
    coord = HeatingDataCoordinator(mock_hass, entry)
    coord.statistics = StatisticsManager(coord)
    coord.solar = SolarCalculator(coord)

    # Constant unit coef = 1.0 on south, 0 elsewhere.
    coord.solar.calculate_unit_coefficient = MagicMock(
        return_value={"s": 1.0, "e": 0.0, "w": 0.0}
    )
    # Real calculate_unit_solar_impact uses coef·pot dot product; the
    # redistribution path calls it twice (current + past).  We need a
    # response that depends on the passed potential vector so past-hour
    # reconstruction shows up in the result.
    def _impact(pot_vec, coef):
        try:
            return (
                pot_vec[0] * coef.get("s", 0.0)
                + pot_vec[1] * coef.get("e", 0.0)
                + pot_vec[2] * coef.get("w", 0.0)
            )
        except Exception:
            return 0.0

    coord.solar.calculate_unit_solar_impact = MagicMock(side_effect=_impact)
    # Always reconstruct identity (we feed the test "potential" as the
    # logged effective vector + correction=100 → trans=1.0 → pot == eff).
    coord.solar.reconstruct_potential_vector = MagicMock(
        side_effect=lambda eff, corr, cfg: eff
    )

    # Storage hooks (mirror the hotspot test).
    coord._correlation_data_per_unit = {"sensor.heater_1": {"_id": "unit_base"}}
    coord._aux_coefficients_per_unit = {"sensor.heater_1": {"_id": "unit_aux"}}
    coord._aux_coefficients = {"_id": "global_aux"}
    coord._correlation_data = {"_id": "global_base"}
    coord._hourly_delta_per_unit = {"sensor.heater_1": 0.0}
    coord._collector.aux_breakdown = {}

    # Base prediction 5.0 kWh — large enough that solar (≤ 1.0 here) does
    # not get clipped by saturation, so the redistribution effect surfaces
    # directly in the breakdown.
    def _mock_pred(data_map, temp_key, wind_bucket, temp, bp, apply_scaling=True):
        doc_id = data_map.get("_id") if isinstance(data_map, dict) else None
        if doc_id in ("unit_base", "global_base"):
            return 5.0
        return 0.0

    coord.statistics._get_prediction_from_model = MagicMock(side_effect=_mock_pred)
    coord.get_unit_mode = MagicMock(return_value=MODE_HEATING)
    coord.data["effective_wind"] = 0.0
    coord.data[ATTR_SOLAR_FACTOR] = 0.5  # current eligible by default
    # Force potential vector to a known value via override_solar_vector at
    # call time; correction=100 → trans=1.0 → pot == effective.
    coord.solar_correction_percent = 100.0

    return coord


def _build_log_entry(ts_iso: str, solar_factor: float, vec=_POT_VEC):
    return {
        "timestamp": ts_iso,
        "solar_factor": solar_factor,
        "solar_vector_s": vec[0],
        "solar_vector_e": vec[1],
        "solar_vector_w": vec[2],
        "correction_percent": 100.0,
    }


def _run(coord, *, sun_elev_now: float, past_elev: float = 20.0,
         now_override: datetime | None = None):
    """Invoke calculate_total_power with controllable sun positions.

    get_approx_sun_pos is called once for "now" and once per past hour at
    midpoint.  We use a counter-based stub so the first call returns the
    current elevation and subsequent calls return the past elevation.

    The forecast-path guard added for #948/#950 P1 disables interventions
    when overrides are supplied without an explicit ``override_now``.
    Tests pin ``override_now`` to ``dt_util.now()`` (or a caller-supplied
    value) so interventions remain active under the override path.
    """
    calls = {"n": 0}

    def _pos(dt_obj):
        calls["n"] += 1
        if calls["n"] == 1:
            return (sun_elev_now, 180.0)
        return (past_elev, 180.0)

    coord.solar.get_approx_sun_pos = MagicMock(side_effect=_pos)
    return coord.statistics.calculate_total_power(
        temp=5.0,
        effective_wind=0.0,
        solar_impact=0.0,
        is_aux_active=False,
        override_solar_factor=coord.data[ATTR_SOLAR_FACTOR],
        override_solar_vector=_POT_VEC,
        override_now=now_override if now_override is not None else dt_util.now(),
    )


def _expected_weights(alpha: float, tau: float, K: int = 3):
    raw = [math.exp(-(k + 1) / tau) for k in range(K)]
    z = sum(raw)
    return [alpha * r / z for r in raw]


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------


def test_alpha_zero_is_noop(mock_hass):
    """α = 0.0 produces bit-identical predictions vs the legacy path."""
    coord = _build_coord(mock_hass, alpha=0.0)
    coord._hourly_log = []  # empty log
    res = _run(coord, sun_elev_now=20.0)
    bd = res["unit_breakdown"]["sensor.heater_1"]
    # coef·pot = 1.0; no scaling at α=0.
    assert bd["raw_solar_kwh"] == pytest.approx(1.0)


def test_eligible_current_no_past_entries(mock_hass):
    """α=0.5, eligible current, empty log → solar = (1 − α) · coef·pot."""
    coord = _build_coord(mock_hass, alpha=0.5, tau=2.0)
    coord._hourly_log = []
    res = _run(coord, sun_elev_now=20.0)
    bd = res["unit_breakdown"]["sensor.heater_1"]
    assert bd["raw_solar_kwh"] == pytest.approx(0.5)


def test_eligible_current_with_one_past(mock_hass):
    """α=0.5, eligible current + one eligible past at H-1 with same vec.

    Expected = (1 − α)·1.0 + w_1·1.0 where w_1 = α·e^(−1/τ)/Σ.
    """
    alpha, tau = 0.5, 2.0
    coord = _build_coord(mock_hass, alpha=alpha, tau=tau)
    now = dt_util.now()
    h_minus_1 = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    coord._hourly_log = [_build_log_entry(h_minus_1.isoformat(), 0.4)]
    res = _run(coord, sun_elev_now=20.0, past_elev=20.0)
    bd = res["unit_breakdown"]["sensor.heater_1"]
    w = _expected_weights(alpha, tau)
    expected = (1.0 - alpha) * 1.0 + w[0] * 1.0
    assert bd["raw_solar_kwh"] == pytest.approx(expected, abs=1e-3)


def test_high_elev_current_keeps_full_credit_plus_past(mock_hass):
    """Current NOT eligible (high elev) + eligible past at H-1.

    Current credit is NOT scaled (α_now = 0), but past contributions still
    add their kernel weight on top.  Expected = 1.0 + w_1·1.0.
    """
    alpha, tau = 0.5, 2.0
    coord = _build_coord(mock_hass, alpha=alpha, tau=tau)
    now = dt_util.now()
    h_minus_1 = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    coord._hourly_log = [_build_log_entry(h_minus_1.isoformat(), 0.4)]
    res = _run(coord, sun_elev_now=45.0, past_elev=20.0)
    bd = res["unit_breakdown"]["sensor.heater_1"]
    w = _expected_weights(alpha, tau)
    expected = 1.0 + w[0] * 1.0
    assert bd["raw_solar_kwh"] == pytest.approx(expected, abs=1e-3)


def test_past_dark_skipped(mock_hass):
    """Past entry with solar_factor ≤ 0.05 → past skipped."""
    alpha, tau = 0.5, 2.0
    coord = _build_coord(mock_hass, alpha=alpha, tau=tau)
    now = dt_util.now()
    h_minus_1 = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    coord._hourly_log = [_build_log_entry(h_minus_1.isoformat(), 0.04)]
    res = _run(coord, sun_elev_now=20.0)
    bd = res["unit_breakdown"]["sensor.heater_1"]
    # No past contribution; only the current scaling.
    assert bd["raw_solar_kwh"] == pytest.approx(0.5)


def test_past_missing_skipped(mock_hass):
    """Gap in log (no entry at H-1) → past skipped."""
    alpha, tau = 0.5, 2.0
    coord = _build_coord(mock_hass, alpha=alpha, tau=tau)
    # Only an H-2 entry exists.
    now = dt_util.now()
    h_minus_2 = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=2)
    coord._hourly_log = [_build_log_entry(h_minus_2.isoformat(), 0.4)]
    res = _run(coord, sun_elev_now=20.0, past_elev=20.0)
    bd = res["unit_breakdown"]["sensor.heater_1"]
    w = _expected_weights(alpha, tau)
    # Only w_2 contributes; w_1 and w_3 skipped.
    expected = (1.0 - alpha) * 1.0 + w[1] * 1.0
    assert bd["raw_solar_kwh"] == pytest.approx(expected, abs=1e-3)


def test_current_elev_boundary_strict(mock_hass):
    """sun_elev = 30.0 exactly → strict `<`, treated as NOT eligible.

    With α=0.5 and no past entries, the result should be the full unscaled
    coef·pot = 1.0 (no current scaling).
    """
    coord = _build_coord(mock_hass, alpha=0.5, tau=2.0)
    coord._hourly_log = []
    res = _run(coord, sun_elev_now=30.0)
    bd = res["unit_breakdown"]["sensor.heater_1"]
    assert bd["raw_solar_kwh"] == pytest.approx(1.0)


def test_kernel_weights_sum_to_alpha():
    """Kernel weights w_1, w_2, w_3 sum to α at τ = 2.0, K = 3."""
    alpha, tau = 0.5, 2.0
    w = _expected_weights(alpha, tau, K=3)
    assert sum(w) == pytest.approx(alpha)
    # Decay ordering: w_1 > w_2 > w_3.
    assert w[0] > w[1] > w[2]


# -----------------------------------------------------------------------
# Regression tests for #948 / #950 review feedback
# -----------------------------------------------------------------------


def test_forecast_path_without_override_now_disables_interventions(mock_hass):
    """Forecast-path guard: when override_solar_factor or override_solar_vector
    is supplied WITHOUT a matching override_now, interventions skip.  The
    prior bug fired the elevation gate against dt_util.now() — the wall
    clock at execution time, not the target forecast hour — and looked up
    the wrong past-hour window in _hourly_log.  The guard makes that case
    a no-op instead.
    """
    alpha, tau = 0.5, 2.0
    coord = _build_coord(mock_hass, alpha=alpha, tau=tau)
    # Stage a log entry at H-1 so that, if interventions WERE active, we'd
    # see a tail contribution of w_1 × 1.0 added on top of (1 − α) × 1.0.
    now = dt_util.now()
    h_minus_1 = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    coord._hourly_log = [_build_log_entry(h_minus_1.isoformat(), 0.4)]
    coord.solar.get_approx_sun_pos = MagicMock(return_value=(20.0, 180.0))
    # Call with overrides but NO override_now — should skip interventions.
    res = coord.statistics.calculate_total_power(
        temp=5.0,
        effective_wind=0.0,
        solar_impact=0.0,
        is_aux_active=False,
        override_solar_factor=0.5,
        override_solar_vector=_POT_VEC,
    )
    bd = res["unit_breakdown"]["sensor.heater_1"]
    # Legacy path: solar = coef · pot = 1.0 (no α scaling, no tail).
    assert bd["raw_solar_kwh"] == pytest.approx(1.0)


def test_override_now_routes_to_target_hour(mock_hass):
    """When override_now is provided, the elevation gate and past-hour
    lookup are anchored to the target prediction hour, not wall clock.
    Forecasting hour H at any wall-clock time produces the same result
    as if dt_util.now() == H.
    """
    alpha, tau = 0.5, 2.0
    coord = _build_coord(mock_hass, alpha=alpha, tau=tau)
    # Stage past entry relative to a synthetic prediction target far from
    # wall clock.  If the implementation used dt_util.now() it would miss
    # this entry entirely (timestamps don't align with now − 1h).
    target = dt_util.now().replace(minute=0, second=0, microsecond=0) - timedelta(days=3)
    past = target - timedelta(hours=1)
    coord._hourly_log = [_build_log_entry(past.isoformat(), 0.4)]
    coord.solar.get_approx_sun_pos = MagicMock(return_value=(20.0, 180.0))
    res = coord.statistics.calculate_total_power(
        temp=5.0,
        effective_wind=0.0,
        solar_impact=0.0,
        is_aux_active=False,
        override_solar_factor=0.5,
        override_solar_vector=_POT_VEC,
        override_now=target,
    )
    bd = res["unit_breakdown"]["sensor.heater_1"]
    w = _expected_weights(alpha, tau)
    # (1 − α) × 1.0 + w_1 × 1.0 — current eligible + one eligible past.
    expected = (1.0 - alpha) * 1.0 + w[0] * 1.0
    assert bd["raw_solar_kwh"] == pytest.approx(expected, abs=1e-3)


def test_hotspot_does_not_attenuate_redistributed_tail(mock_hass):
    """When both #948 and #950 are enabled and the current hour is
    high-elev screened, #950 must attenuate ONLY the originator term —
    NOT the redistributed tail.  The tail contributions come from past
    low-elev hours where the hotspot loss mechanism doesn't apply.
    Prior bug: scaling unit_solar_reduction AFTER tail addition double-
    attenuated low-elev energy.
    """
    alpha, tau, gamma = 0.5, 2.0, 0.3
    coord = _build_coord(mock_hass, alpha=alpha, tau=tau)
    coord.solar_hotspot_attenuation_gamma = gamma
    # Screen partially deployed so the #950 gate (correction_percent < 80)
    # fires at the current high-elev originator hour.
    coord.solar_correction_percent = 50.0
    # Ensure the entity has at least one screened facade so #950 applies.
    coord.screen_config = (False, False, True)
    coord.screen_config_for_entity = MagicMock(return_value=(False, False, True))
    # Stage one eligible past entry at H-1 (low-elev contribution).
    now = dt_util.now()
    h_minus_1 = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    coord._hourly_log = [_build_log_entry(h_minus_1.isoformat(), 0.4)]
    # Current hour: high-elev (50°) — triggers #950 — past hour at 20°.
    res = _run(coord, sun_elev_now=50.0, past_elev=20.0)
    bd = res["unit_breakdown"]["sensor.heater_1"]
    w = _expected_weights(alpha, tau)
    # Originator: high-elev + screened → (1 − γ) × 1.0 = 0.7 (α NOT
    # applied because current_eligible is False at high elev).
    # Tail: w_1 × 1.0, NOT scaled by (1 − γ).
    expected = (1.0 - gamma) * 1.0 + w[0] * 1.0
    assert bd["raw_solar_kwh"] == pytest.approx(expected, abs=1e-3)
    # Negative-check: under the prior bug the value would be
    # (1 − γ) × (1.0 + w_1) — verify we're NOT computing that.
    buggy = (1.0 - gamma) * (1.0 + w[0])
    assert not (abs(bd["raw_solar_kwh"] - buggy) < 1e-3)


def test_hotspot_and_redistribution_low_elev_current(mock_hass):
    """When both interventions are enabled and current is LOW-elev:
    #948's α scaling fires on originator, #950's γ scaling does NOT
    (elev < 30 → hotspot gate False), tail terms add normally.
    Confirms low-elev current routes to #948 only.
    """
    alpha, tau, gamma = 0.5, 2.0, 0.3
    coord = _build_coord(mock_hass, alpha=alpha, tau=tau)
    coord.solar_hotspot_attenuation_gamma = gamma
    now = dt_util.now()
    h_minus_1 = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    coord._hourly_log = [_build_log_entry(h_minus_1.isoformat(), 0.4)]
    res = _run(coord, sun_elev_now=20.0, past_elev=20.0)
    bd = res["unit_breakdown"]["sensor.heater_1"]
    w = _expected_weights(alpha, tau)
    # Originator: (1 − α) × 1.0 (no γ — low-elev never triggers hotspot).
    # Tail: w_1 × 1.0.
    expected = (1.0 - alpha) * 1.0 + w[0] * 1.0
    assert bd["raw_solar_kwh"] == pytest.approx(expected, abs=1e-3)
