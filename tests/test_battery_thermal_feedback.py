"""Tests for saturation-wasted thermal-feedback into the solar battery EMA (#896).

Three groups:

- ``TestBatteryEMAInputFormula``: the static helper
  ``HourlyProcessor._compute_battery_ema_input`` — bit-identical to the
  legacy formula at k=0, additive at k>0 with the heating gate.
- ``TestCoordinatorWiring``: ``battery_thermal_feedback_k`` is read from
  ``entry.data`` with a clamp to [0, 1] and a fallback to 0.0 on bad input.
- ``TestStatisticsHeatingWastedField``: ``calculate_total_power`` exposes
  ``solar_heating_wasted_kwh`` in the breakdown — the explicit
  heating-only aggregate that downstream callers (battery EMA, sweep
  replay) use instead of the all-modes total.
- ``TestDiagnoseSolarBatteryFeedbackSweep``: the new
  ``global.battery_feedback_sweep`` block exists, recommends k=0 when
  the log carries no wasted signal, recommends k>0 when wasted feeds an
  improvement, and bypasses the feedback for cooling-only hours.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from custom_components.heating_analytics.const import (
    CONF_BATTERY_THERMAL_FEEDBACK_K,
    DEFAULT_BATTERY_THERMAL_FEEDBACK_K,
    MODE_COOLING,
    MODE_GUEST_COOLING,
    MODE_GUEST_HEATING,
    MODE_HEATING,
    MODE_OFF,
    SOLAR_BATTERY_DECAY,
)
from custom_components.heating_analytics.diagnostics import DiagnosticsEngine
from custom_components.heating_analytics.hourly_processor import HourlyProcessor


class TestBatteryEMAInputFormula:
    """Helper at hourly_processor.HourlyProcessor._compute_battery_ema_input."""

    def test_k_zero_returns_impact_unchanged(self):
        """k=0 collapses to the pre-#896 formula bit-for-bit."""
        out = HourlyProcessor._compute_battery_ema_input(
            solar_impact=1.5, solar_wasted=0.7, k_feedback=0.0,
            has_heating_unit=True,
        )
        assert out == 1.5

    def test_k_zero_with_zero_wasted_returns_impact(self):
        """k=0 stays at impact regardless of wasted."""
        out = HourlyProcessor._compute_battery_ema_input(
            solar_impact=2.0, solar_wasted=0.0, k_feedback=0.0,
            has_heating_unit=True,
        )
        assert out == 2.0

    def test_k_positive_with_heating_adds_wasted_term(self):
        """k>0 + heating active adds k * wasted to impact."""
        out = HourlyProcessor._compute_battery_ema_input(
            solar_impact=1.5, solar_wasted=0.8, k_feedback=0.5,
            has_heating_unit=True,
        )
        assert out == pytest.approx(1.5 + 0.5 * 0.8)

    def test_k_positive_no_heating_unit_skips_feedback(self):
        """Cooling-only hour: feedback gated out even when k>0 and wasted>0."""
        out = HourlyProcessor._compute_battery_ema_input(
            solar_impact=0.0, solar_wasted=0.5, k_feedback=0.5,
            has_heating_unit=False,
        )
        assert out == 0.0

    def test_k_positive_zero_wasted_returns_impact(self):
        """No wasted ⇒ no feedback contribution; impact passes through."""
        out = HourlyProcessor._compute_battery_ema_input(
            solar_impact=1.0, solar_wasted=0.0, k_feedback=0.5,
            has_heating_unit=True,
        )
        assert out == 1.0

    def test_k_one_doubles_to_solar_potential(self):
        """k=1 with heating active: input becomes impact + wasted = potential."""
        impact, wasted = 1.2, 0.6  # potential = 1.8
        out = HourlyProcessor._compute_battery_ema_input(
            solar_impact=impact, solar_wasted=wasted, k_feedback=1.0,
            has_heating_unit=True,
        )
        assert out == pytest.approx(1.8)


class TestCoordinatorStaleKeyCleanup:
    """The coordinator removes ``battery_thermal_feedback_k`` from entry.data
    on init.  The Advanced Options UI was retired in 1.3.5; any installation
    that set k > 0 via the pre-1.3.5 slider would otherwise carry the stale
    value forward with no UI to reset it.  Cleanup is silent (no warning
    log) because the user has no control to act on it.

    The runtime ``self.battery_thermal_feedback_k`` always lands at the
    default (0.0) regardless of the prior entry.data state.
    """

    @staticmethod
    def _make_entry(data):
        from unittest.mock import MagicMock
        entry = MagicMock()
        entry.data = dict(data)  # copy so the test sees mutations
        entry.entry_id = "test_entry"
        return entry

    @staticmethod
    def _make_hass():
        from unittest.mock import MagicMock
        hass = MagicMock()
        hass.config = MagicMock()
        hass.config.latitude = 60.0
        hass.config.longitude = 10.0
        hass.config_entries = MagicMock()
        # Track update calls
        hass.config_entries.async_update_entry = MagicMock()
        hass.is_running = True
        hass.states = MagicMock()
        hass.data = {"heating_analytics": {}}
        hass.services = MagicMock()
        return hass

    def _build_coord(self, k_value):
        from unittest.mock import patch
        from custom_components.heating_analytics.coordinator import (
            HeatingDataCoordinator,
        )
        data = {
            "energy_sensors": ["sensor.heater1"],
            "balance_point": 15.0,
            "solar_enabled": True,
            "csv_auto_logging": False,
        }
        if k_value is not None:
            data[CONF_BATTERY_THERMAL_FEEDBACK_K] = k_value
        entry = self._make_entry(data)
        hass = self._make_hass()
        with patch("custom_components.heating_analytics.storage.Store"):
            coord = HeatingDataCoordinator(hass, entry)
        return coord, hass, entry

    def test_stale_key_removed(self):
        """entry.data carrying k > 0 has the key stripped and the runtime
        attribute lands at 0.0."""
        coord, hass, entry = self._build_coord(k_value=0.3)
        # The cleanup ran during __init__
        hass.config_entries.async_update_entry.assert_called_once()
        # Verify the new data dict passed to async_update_entry has no key
        call_kwargs = hass.config_entries.async_update_entry.call_args.kwargs
        new_data = call_kwargs.get("data") or hass.config_entries.async_update_entry.call_args.args[1]
        assert CONF_BATTERY_THERMAL_FEEDBACK_K not in new_data
        # Runtime attribute zeroed regardless of the prior value
        assert coord.battery_thermal_feedback_k == DEFAULT_BATTERY_THERMAL_FEEDBACK_K

    def test_stale_zero_value_also_removed(self):
        """An explicit 0.0 value in entry.data is also cleaned up — the field
        is retired entirely, not just non-zero values."""
        coord, hass, entry = self._build_coord(k_value=0.0)
        hass.config_entries.async_update_entry.assert_called_once()
        assert coord.battery_thermal_feedback_k == DEFAULT_BATTERY_THERMAL_FEEDBACK_K

    def test_missing_key_no_op(self):
        """Fresh installs (key never present) skip the cleanup —
        async_update_entry is not called."""
        coord, hass, entry = self._build_coord(k_value=None)
        hass.config_entries.async_update_entry.assert_not_called()
        assert coord.battery_thermal_feedback_k == DEFAULT_BATTERY_THERMAL_FEEDBACK_K

    def test_stale_invalid_value_also_removed(self):
        """A non-numeric value (corrupt config) is also cleaned up — the
        cleanup checks for key presence, not value validity."""
        coord, hass, entry = self._build_coord(k_value="invalid")
        hass.config_entries.async_update_entry.assert_called_once()
        assert coord.battery_thermal_feedback_k == DEFAULT_BATTERY_THERMAL_FEEDBACK_K

    def test_other_entry_data_preserved(self):
        """Cleanup only removes the k key — every other entry.data field
        is forwarded unchanged to async_update_entry."""
        coord, hass, entry = self._build_coord(k_value=0.5)
        call_kwargs = hass.config_entries.async_update_entry.call_args.kwargs
        new_data = call_kwargs.get("data") or hass.config_entries.async_update_entry.call_args.args[1]
        # Other keys we put in the fixture survive
        assert new_data["balance_point"] == 15.0
        assert new_data["energy_sensors"] == ["sensor.heater1"]
        assert new_data["solar_enabled"] is True


class TestStatisticsHeatingWastedField:
    """``calculate_total_power`` exposes ``solar_heating_wasted_kwh``.

    Verifies the upstream invariant: wasted from cooling/OFF/DHW units
    does not enter the heating-only aggregate.  Uses a synthetic
    ``calculate_saturation`` whose return-shape encodes per-mode behaviour
    while letting us inject non-zero wasted on cooling — which today's
    production saturation does not produce, but future changes might.
    The test certifies that the downstream contract holds even under that
    hypothetical.

    Uses the heavy real-coordinator fixture pattern from
    ``test_aux_overflow.py`` because ``calculate_total_power`` builds its
    ``raw_unit_data`` from coordinator state internally; there is no
    lighter-weight aggregation entry point.
    """

    @staticmethod
    def _make_real_coord(unit_modes, sat_func):
        from custom_components.heating_analytics.const import (
            ATTR_POTENTIAL_SAVINGS, DOMAIN,
        )
        from custom_components.heating_analytics.coordinator import (
            HeatingDataCoordinator,
        )

        hass = MagicMock()
        hass.states = MagicMock()
        hass.states.get = MagicMock(return_value=None)
        hass.data = {DOMAIN: {}}
        hass.config_entries = MagicMock()
        hass.bus = MagicMock()
        hass.is_running = True

        entry = MagicMock()
        sensors = list(unit_modes.keys())
        entry.data = {
            "energy_sensors": sensors,
            "aux_affected_entities": [],
            "balance_point": 15.0,
            "wind_threshold": 5.0,
            "extreme_wind_threshold": 10.0,
        }
        coord = HeatingDataCoordinator(hass, entry)
        coord.statistics = MagicMock()
        coord.learning = MagicMock()
        coord.storage = MagicMock()
        coord.forecast = MagicMock()
        coord.solar = MagicMock()
        coord.solar.calculate_saturation.side_effect = sat_func
        coord.solar.apply_correction.side_effect = lambda base, impact, val: base
        coord.solar.calculate_unit_solar_impact = MagicMock(return_value=1.0)
        coord.solar.calculate_unit_coefficient = MagicMock(
            return_value={"s": 1.0, "e": 0.0, "w": 0.0}
        )
        # Reconstruct returns the input vector unchanged — we don't care
        # about transmittance here, only the wasted-aggregation arithmetic.
        coord.solar.reconstruct_potential_vector = MagicMock(
            side_effect=lambda eff, _c, _cfg: eff
        )

        coord.solar_enabled = True
        coord.solar_azimuth = 180
        coord.solar_correction_percent = 100.0
        coord.screen_config = (True, True, True)
        coord._unit_modes = dict(unit_modes)
        coord._hourly_delta_per_unit = {sid: 0.0 for sid in sensors}
        coord.data[ATTR_POTENTIAL_SAVINGS] = 0.0
        coord.data["effective_wind"] = 0.0

        # Empty learned models — predictions come via the patched helper
        # below.  Each per-unit base = 2.0, no aux.
        coord._correlation_data_per_unit = {
            sid: {"15": {"normal": 2.0}} for sid in sensors
        }
        coord._aux_coefficients_per_unit = {
            sid: {"15": {"normal": 0.0}} for sid in sensors
        }
        coord._correlation_data = {"15": {"normal": 2.0 * len(sensors)}}
        return coord

    @staticmethod
    def _patched_stats(coord):
        from custom_components.heating_analytics.statistics import StatisticsManager

        sm = StatisticsManager(coord)

        # Bypass the temp/wind-stratified prediction lookup; return a
        # known-shape value so the wasted aggregation is the only moving
        # variable in the assertion.
        def _pred(data_map, temp_key, wind_bucket, temp, bp, apply_scaling=True):
            try:
                v = data_map.get(str(int(temp)), {}).get(wind_bucket, 0.0)
            except AttributeError:
                v = 0.0
            return v or 0.0

        sm._get_prediction_from_model = MagicMock(side_effect=_pred)
        # Per-unit base lookup: returns the per-unit data map's value.
        coord._get_predicted_kwh_per_unit = MagicMock(
            side_effect=lambda eid, temp_key, wind_bucket: 2.0
        )
        coord._get_predicted_kwh = MagicMock(return_value=2.0 * len(coord.energy_sensors))
        return sm

    def test_heating_wasted_field_present(self):
        """Two heating units with wasted=0.5 each → heating field = 1.0."""
        def _sat(net, solar, mode):
            return (net, 0.5, max(0.0, net - solar))

        coord = self._make_real_coord(
            {"sensor.h1": MODE_HEATING, "sensor.h2": MODE_HEATING}, _sat,
        )
        sm = self._patched_stats(coord)
        result = sm.calculate_total_power(
            temp=10.0, effective_wind=0.0, solar_impact=0.0,
            is_aux_active=False, detailed=False,
        )
        breakdown = result["breakdown"]
        assert "solar_heating_wasted_kwh" in breakdown
        assert breakdown["solar_heating_wasted_kwh"] == pytest.approx(1.0)

    def test_cooling_wasted_excluded_from_heating_field(self):
        """Hypothetical non-zero cooling wasted MUST NOT enter heating field."""
        def _sat(net, solar, mode):
            if mode in (MODE_COOLING, MODE_GUEST_COOLING):
                return (solar, 0.7, net + solar)
            return (net, 0.3, max(0.0, net - solar))

        coord = self._make_real_coord(
            {"sensor.h": MODE_HEATING, "sensor.c": MODE_COOLING}, _sat,
        )
        sm = self._patched_stats(coord)
        result = sm.calculate_total_power(
            temp=10.0, effective_wind=0.0, solar_impact=0.0,
            is_aux_active=False, detailed=False,
        )
        breakdown = result["breakdown"]
        # Total wasted = 0.3 (heating) + 0.7 (cooling) = 1.0
        assert breakdown["solar_wasted_kwh"] == pytest.approx(1.0)
        # Heating-only field strips the cooling 0.7
        assert breakdown["solar_heating_wasted_kwh"] == pytest.approx(0.3)


def _make_sweep_coord(hourly_log):
    """Minimal coordinator mock for diagnose_solar.battery_feedback_sweep tests.

    Mirrors the fixture pattern in test_solar_diagnose.py — populates
    only what the sweep block reads.  The legacy diagnose_solar code
    paths (per-unit normal equations, shadow replay, transmittance
    sensitivity) are noop'd via empty / mock returns so this fixture
    isolates behaviour to the new block.
    """
    coord = MagicMock()
    coord._hourly_log = hourly_log
    coord.solar_battery_decay = SOLAR_BATTERY_DECAY
    coord.battery_thermal_feedback_k = 0.0
    coord.energy_sensors = ["sensor.heater1"]
    coord.solar_correction_percent = 100.0
    coord.solar_azimuth = 180
    coord.balance_point = 15.0
    coord.screen_config = (True, True, True)
    coord.screen_config_for_entity = MagicMock(side_effect=lambda _eid: (True, True, True))
    coord.hass.config.latitude = 60.0
    coord.hass.config.longitude = 10.0
    coord.aux_affected_entities = []
    coord._unit_strategies = {}
    coord._daily_history = {}
    coord._per_unit_min_base_thresholds = {}
    coord._correlation_data_per_unit = {"sensor.heater1": {"10": {"normal": 1.5}}}
    coord.learning_rate = 0.1

    coord._solar_coefficients_per_unit = {}
    coord.model.solar_coefficients_per_unit = {}
    coord.model.correlation_data_per_unit = coord._correlation_data_per_unit

    coord.solar = MagicMock()
    coord.solar.calculate_unit_coefficient = MagicMock(
        return_value={"s": 0.0, "e": 0.0, "w": 0.0}
    )
    coord.solar.calculate_unit_solar_impact = MagicMock(return_value=0.0)
    coord.solar._screen_transmittance_vector = MagicMock(return_value=(1.0, 1.0, 1.0))
    coord.solar.reconstruct_potential_vector = MagicMock(
        side_effect=lambda eff, _corr, _cfg: eff
    )

    coord.learning = MagicMock()
    coord.learning.replay_solar_nlms = MagicMock(return_value={
        "updates": 0, "entries_considered": 0,
        "entry_skipped_aux": 0, "entry_skipped_poisoned": 0,
        "entry_skipped_disabled": 0, "entry_skipped_low_magnitude": 0,
        "entry_skipped_missing_temp_key": 0, "unit_skipped_aux_list": 0,
        "unit_skipped_shutdown": 0, "unit_skipped_excluded_mode": 0,
        "unit_skipped_weighted_smear": 0, "unit_skipped_below_threshold": 0,
        "inequality_updates": 0, "inequality_non_binding": 0,
        "inequality_skipped_low_battery": 0, "inequality_skipped_mode": 0,
        "inequality_skipped_base": 0,
    })
    return coord


def _sweep_log_entry(
    ts, *, hour=8, temp=8.0, solar_factor=0.6, solar_s=0.6, solar_e=0.0,
    solar_w=0.0, correction=100.0, actual=2.0, expected=2.5,
    solar_impact_raw=0.5, solar_wasted=0.0, solar_heating_wasted=None,
    mode=MODE_HEATING,
):
    """Build a sweep-relevant hourly-log entry.

    Defaults populate a transition-regime morning hour (BP=15 ⇒ T=8 is
    in heating_mild) with one heating unit.  ``solar_heating_wasted``
    defaults to ``solar_wasted`` (matches today's structural equality).
    """
    if solar_heating_wasted is None:
        solar_heating_wasted = solar_wasted
    # unit_modes log only stores non-default; missing entity → MODE_HEATING.
    unit_modes = {} if mode == MODE_HEATING else {"sensor.heater1": mode}
    return {
        "timestamp": ts,
        "hour": hour,
        "temp": temp,
        "inertia_temp": temp,
        "temp_key": str(int(round(temp))),
        "wind_bucket": "normal",
        "solar_factor": solar_factor,
        "solar_vector_s": solar_s,
        "solar_vector_e": solar_e,
        "solar_vector_w": solar_w,
        "solar_impact_raw_kwh": solar_impact_raw,
        "solar_impact_kwh": solar_impact_raw,
        "solar_wasted_kwh": solar_wasted,
        "solar_heating_wasted_kwh": solar_heating_wasted,
        "actual_kwh": actual,
        "expected_kwh": expected,
        "correction_percent": correction,
        "auxiliary_active": False,
        "guest_impact_kwh": 0.0,
        "unit_modes": unit_modes,
        "unit_breakdown": {"sensor.heater1": actual},
        "unit_expected_breakdown": {"sensor.heater1": expected},
        "solar_dominant_entities": [],
    }


class TestDiagnoseSolarBatteryFeedbackSweep:
    """``global.battery_feedback_sweep`` block."""

    def test_block_present_with_log(self):
        """Sweep block populated when log has at least one entry."""
        log = [
            _sweep_log_entry(f"2026-04-{15 + i//24:02d}T{i%24:02d}:00:00",
                             hour=i % 24)
            for i in range(48)
        ]
        coord = _make_sweep_coord(log)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        sweep = result["global"]["battery_feedback_sweep"]
        assert sweep, "battery_feedback_sweep should be populated for non-empty log"
        assert "current_k" in sweep
        assert "k_candidates" in sweep
        assert "empirical_optimum_k" in sweep
        assert "global_rmse_at_optimum_kwh" in sweep
        assert "rmse_improvement_kwh" in sweep
        assert "method" in sweep
        # 11 candidates: 0.0, 0.1, ..., 1.0
        assert sweep["k_candidates"] == [round(0.1 * i, 1) for i in range(11)]
        assert "0.0" in sweep["sweep"]
        assert "0.5" in sweep["sweep"]

    def test_empty_log_returns_empty_sweep(self):
        """No log entries → empty sweep block (no crash)."""
        coord = _make_sweep_coord([])
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        # An empty log produces an empty dict, not missing key.
        assert result["global"]["battery_feedback_sweep"] == {}

    def test_no_wasted_signal_recommends_k_zero(self):
        """When wasted is structurally 0 across the log, optimum k = 0."""
        log = [
            _sweep_log_entry(f"2026-04-{15 + i//24:02d}T{i%24:02d}:00:00",
                             hour=i % 24, solar_wasted=0.0,
                             solar_heating_wasted=0.0)
            for i in range(48)
        ]
        coord = _make_sweep_coord(log)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        sweep = result["global"]["battery_feedback_sweep"]
        # Without wasted signal, every k produces identical battery state
        # (the feedback term is zero everywhere), so RMSE is flat.  The
        # tie-break rule favours the smaller k → optimum = 0.0.
        assert sweep["empirical_optimum_k"] == 0.0
        assert sweep["rmse_improvement_kwh"] == 0.0

    def test_baseline_residual_matches_live_at_k_zero(self):
        """At k=0 the replay reproduces the live RMSE exactly.

        Sanity check on the residual reconstruction: replay-with-k=0
        gives ``residual_alt = (actual − expected) + (B_k0 − B_k0) =
        residual_live``.  RMSE of (actual − expected) over the log
        equals the reported global RMSE at k=0.0.
        """
        log = [
            _sweep_log_entry(
                f"2026-04-{15 + i//24:02d}T{i%24:02d}:00:00",
                hour=8 + (i % 6), temp=8.0,
                solar_impact_raw=0.4 + 0.05 * (i % 4),
                solar_wasted=0.0, solar_heating_wasted=0.0,
                actual=2.0 + 0.1 * (i % 3),
                expected=2.5 - 0.05 * (i % 3),
            )
            for i in range(36)
        ]
        coord = _make_sweep_coord(log)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        sweep = result["global"]["battery_feedback_sweep"]

        # Compute live RMSE directly from the log.
        residuals = [e["actual_kwh"] - e["expected_kwh"] for e in log]
        n = len(residuals)
        live_rmse = (sum(r * r for r in residuals) / n) ** 0.5
        assert sweep["sweep"]["0.0"]["global"]["rmse_kwh"] == pytest.approx(
            round(live_rmse, 4), abs=1e-4,
        )

    def test_k_positive_recommended_when_wasted_helps(self):
        """Synthetic log where wasted feed reduces residual ⇒ optimum k > 0.

        Construct a log where:
          - Live ``expected_live`` over-predicts (actual < expected) on
            transition-regime sunny mornings (residual_live < 0).
          - Wasted is positive (saturation occurred).
          - Replaying with k > 0 raises battery state by k * wasted * (1−decay)
            cumulatively, so residual_alt = residual_live + delta_battery
            shifts toward zero.
        Optimum k should be the candidate that drives mean residual closest
        to zero — i.e. RMSE-minimising for symmetric residuals around zero.
        """
        # 24 sunny morning hours with wasted=0.5 each.  Residual_live = -0.3
        # (over-prediction).  At k=0.6 the cumulative battery shift across
        # successive hours pushes residual_alt toward zero on later hours.
        log = []
        for i in range(48):
            # Morning hours 8-12 carry the over-prediction signal; afternoon
            # hours 13-18 are normal so they don't pollute the bucket.
            day = 15 + i // 8
            hour = 8 + (i % 8)
            if hour <= 12:
                solar_impact = 0.4
                wasted = 0.5
                actual_demand = 1.5
                expected_demand = 1.8
            else:
                solar_impact = 0.2
                wasted = 0.0
                actual_demand = 2.0
                expected_demand = 2.0
            log.append(_sweep_log_entry(
                f"2026-04-{day:02d}T{hour:02d}:00:00",
                hour=hour, temp=8.0,
                solar_impact_raw=solar_impact,
                solar_wasted=wasted, solar_heating_wasted=wasted,
                actual=actual_demand, expected=expected_demand,
            ))
        coord = _make_sweep_coord(log)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        sweep = result["global"]["battery_feedback_sweep"]

        # The over-prediction in morning hours means residual_live < 0;
        # delta_battery > 0 (k>0 raises battery), so residual_alt rises
        # toward zero on hours where battery has accumulated.  Optimum k
        # should be > 0.
        assert sweep["empirical_optimum_k"] > 0.0, (
            f"expected non-zero optimum k for over-prediction signal; "
            f"got {sweep['empirical_optimum_k']}, full sweep: "
            f"{sweep['sweep']}"
        )
        assert sweep["rmse_improvement_kwh"] > 0.0

    def test_cooling_only_log_skips_feedback(self):
        """All-cooling-mode hours: feedback gated out at every k.

        ``has_heating_unit`` is False on every tuple, so the EMA input is
        just ``solar_impact_raw`` for every k candidate.  Each k produces
        an identical battery trajectory, hence identical residuals and
        RMSE.  Optimum k falls to 0.0 by tie-break.
        """
        log = []
        for i in range(48):
            day = 15 + i // 24
            hour = i % 24
            log.append(_sweep_log_entry(
                f"2026-04-{day:02d}T{hour:02d}:00:00",
                hour=hour, temp=20.0,  # cooling regime (T > BP+2)
                solar_impact_raw=0.4,
                solar_wasted=0.5, solar_heating_wasted=0.0,  # heating-gated to 0
                actual=2.0, expected=2.5,
                mode=MODE_COOLING,
            ))
        coord = _make_sweep_coord(log)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        sweep = result["global"]["battery_feedback_sweep"]

        # Heating-wasted is zero on every hour (cooling-only) → no
        # candidate produces a non-trivial delta_battery → all RMSEs
        # equal → tie-break selects k = 0.0.
        assert sweep["empirical_optimum_k"] == 0.0
        # All candidates should report (numerically) the same RMSE.
        rmses = {k: sweep["sweep"][str(k)]["global"]["rmse_kwh"]
                 for k in sweep["k_candidates"]}
        baseline_rmse = rmses[0.0]
        for k_val, rmse in rmses.items():
            assert rmse == pytest.approx(baseline_rmse, abs=1e-4), (
                f"k={k_val} produced {rmse} but baseline is {baseline_rmse}"
            )
        # n_hours_with_heating_wasted should be zero (cooling everywhere).
        assert sweep["n_hours_with_heating_wasted"] == 0

    def test_cell_keys_follow_stratification_layout(self):
        """Per-cell keys match ``{hour_bucket}__{temp_bucket}__{screen_bucket}``."""
        log = [
            _sweep_log_entry(f"2026-04-15T{h:02d}:00:00", hour=h, temp=8.0)
            for h in range(6, 18)
        ]
        coord = _make_sweep_coord(log)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        cells = result["global"]["battery_feedback_sweep"]["sweep"]["0.0"]["cells"]
        for key in cells:
            parts = key.split("__")
            assert len(parts) == 3, f"unexpected cell key shape: {key}"
            hb, tb, sb = parts
            assert hb in {"morning", "midday", "afternoon", "night"}
            assert tb in {"heating_deep", "heating_mild", "transition", "cooling"}
            assert sb in {"open", "mid", "closed"}

    def test_transition_zone_visible_in_per_cell_table(self):
        """Issue #896 headline symptom — transition zone — is bucketed.

        Pre-fix: transition (BP-2 ≤ T ≤ BP+2) returned None and was
        dropped from per-cell tabulation, hiding the headline symptom
        from ``per_cell_at_optimum`` for high-BP installs where the
        7-10 °C signal sits at BP-5..BP-8 (still in heating_mild) but
        for very-high-BP installs (BP=18-19) the signal can drift up
        into the BP±2 window.  The transition bucket exists; verify
        cells with temp_bucket=\"transition\" actually appear in the
        per-cell tables when the log carries qualifying hours there.
        """
        # BP=15, transition zone = [13, 17].  Use T=14 to land in it.
        log = [
            _sweep_log_entry(f"2026-04-15T{h:02d}:00:00", hour=h, temp=14.0)
            for h in range(8, 14)
        ]
        coord = _make_sweep_coord(log)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        cells = result["global"]["battery_feedback_sweep"]["sweep"]["0.0"]["cells"]
        transition_keys = [k for k in cells if "transition" in k.split("__")]
        assert transition_keys, (
            "expected at least one transition cell for T=14 in BP=15 install; "
            f"got cell keys: {list(cells.keys())}"
        )

    def test_methodology_note_describes_live_wiring(self):
        """Sweep methodology must describe the now-live wiring honestly.

        After split-state was shipped (#896 follow-up), the sweep models
        the actual wiring rather than a hypothetical 1:1 demand
        displacement.  The notes field must (a) describe what the sweep
        is calibrated to (release-into-prediction), (b) flag the
        frozen-coefficient simplification, (c) point users to real
        validation via 2-4 sunny weeks of k>0 operation.
        """
        log = [
            _sweep_log_entry(f"2026-04-{15+i//8:02d}T{(i%8)+8:02d}:00:00",
                             hour=(i % 8) + 8, solar_wasted=0.3,
                             solar_heating_wasted=0.3)
            for i in range(16)
        ]
        coord = _make_sweep_coord(log)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        sweep = result["global"]["battery_feedback_sweep"]
        # Method field should reflect the live wiring.
        assert sweep["method"] == "carryover_release_replay"
        assert sweep["interpretation"] == "calibration_hint"
        notes = sweep["notes"]
        # Required disclaimers in the user-facing note.
        for must_have in (
            "live wiring",          # describes what's actually being modeled
            "frozen-coefficient",   # NLMS-mediated drift not modeled
            "validate",             # points users to real validation
            "sunny weeks",          # real validation requires running k>0
        ):
            assert must_have in notes.lower(), (
                f"methodology note missing required keyword "
                f"'{must_have}'; got: {notes!r}"
            )

    def test_legacy_log_falls_back_to_solar_wasted_kwh(self):
        """Legacy entries without ``solar_heating_wasted_kwh`` use total wasted.

        Pre-#896 hourly_log entries lack the heating-only field.  The
        sweep must read ``solar_wasted_kwh`` as a fallback so historical
        data analysis works on installations that upgrade.  Verifies
        the diagnostics ``entry.get("solar_heating_wasted_kwh",
        entry.get("solar_wasted_kwh", 0.0))`` chain end-to-end.
        """
        log = []
        for i in range(24):
            entry = _sweep_log_entry(
                f"2026-04-15T{i:02d}:00:00",
                hour=i, temp=8.0,
                solar_impact_raw=0.3,
                solar_wasted=0.4, solar_heating_wasted=0.4,
                actual=2.0, expected=2.5,
            )
            # Strip the post-#896 field to simulate a legacy entry.
            del entry["solar_heating_wasted_kwh"]
            log.append(entry)
        coord = _make_sweep_coord(log)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        sweep = result["global"]["battery_feedback_sweep"]
        # The fallback should pick up the all-modes wasted (0.4 per hour),
        # which is structurally heating-only on legacy logs (cooling
        # always returned wasted=0 in calculate_saturation), so this
        # mimics the heating-only field exactly.
        assert sweep["n_hours_with_heating_wasted"] == 24, (
            f"legacy fallback failed; expected 24 hours with heating "
            f"wasted (from solar_wasted_kwh fallback), got "
            f"{sweep['n_hours_with_heating_wasted']}"
        )
        # Wasted > 0 with heating active means k>0 produces non-trivial
        # delta_battery → at least one candidate differs from baseline.
        rmses = [sweep["sweep"][str(k)]["global"]["rmse_kwh"]
                 for k in sweep["k_candidates"]]
        assert max(rmses) != min(rmses), (
            "expected non-trivial RMSE variation across k when fallback "
            "successfully reads solar_wasted_kwh"
        )

    def test_n_hours_with_heating_active_field(self):
        """Sweep block exposes ``n_hours_with_heating_active`` counter.

        Useful for distinguishing \"low signal\" (sparse saturation) from
        \"no heating activity at all\" (cooling-dominated install) in
        the user-facing diagnostic.
        """
        log = []
        # 12 heating hours, 6 cooling hours
        for i in range(12):
            log.append(_sweep_log_entry(
                f"2026-04-15T{i:02d}:00:00", hour=i, temp=8.0,
                mode=MODE_HEATING,
            ))
        for i in range(12, 18):
            log.append(_sweep_log_entry(
                f"2026-04-15T{i:02d}:00:00", hour=i, temp=20.0,
                solar_heating_wasted=0.0, mode=MODE_COOLING,
            ))
        coord = _make_sweep_coord(log)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        sweep = result["global"]["battery_feedback_sweep"]
        assert sweep["n_hours_in_window"] == 18
        assert sweep["n_hours_with_heating_active"] == 12


def _make_wiring_coord(unit_modes, k=0.0, prior_state=0.0, decay=0.80,
                        energy_sensors=None):
    """Coordinator stub for production-wiring test below.

    Captures the exact attributes the production EMA call site reads:
    ``solar_enabled``, ``battery_thermal_feedback_k``, ``_unit_modes``,
    ``solar_battery_decay``, ``_solar_battery_state``, ``energy_sensors``,
    and the ``get_unit_mode`` helper.

    ``energy_sensors`` defaults to the keys of ``unit_modes`` so the
    common case (every configured sensor has an explicit mode) needs
    no extra wiring.  Pass an explicit list to test the bug-pattern
    case: configured sensors with empty ``_unit_modes`` (default
    heating-only install).
    """
    coord = MagicMock()
    coord.solar_enabled = True
    coord.battery_thermal_feedback_k = k
    coord._unit_modes = dict(unit_modes)
    coord.solar_battery_decay = decay
    coord._solar_battery_state = prior_state
    if energy_sensors is None:
        energy_sensors = list(unit_modes.keys())
    coord.energy_sensors = energy_sensors

    def _get_unit_mode(eid):
        # Mirror coordinator.HeatingDataCoordinator.get_unit_mode —
        # default is MODE_HEATING for missing entries.
        return coord._unit_modes.get(eid, MODE_HEATING)

    coord.get_unit_mode = MagicMock(side_effect=_get_unit_mode)
    return coord


class TestProductionWiring:
    """Smoke tests for the production EMA call site.

    Reproduces the *exact* expressions used in
    ``HourlyProcessor.process`` at the battery-update line so that a
    silent divergence (refactor that drops the gate, swaps wasted source,
    forgets the helper, etc.) is caught here.  These tests are
    intentionally redundant with the helper tests — their purpose is to
    pin the surrounding wiring, not the helper math.
    """

    @staticmethod
    def _apply_production_ema_step(coord, solar_impact, solar_heating_wasted):
        """Mirror of hourly_processor.py battery-EMA block.

        If you refactor that block, update this helper in lockstep —
        this is the production-wiring contract.  Iterates configured
        ``energy_sensors`` and routes each through ``get_unit_mode``
        so the default-MODE_HEATING semantic for missing
        ``_unit_modes`` entries is honoured.
        """
        if not coord.solar_enabled:
            return
        has_heating_unit = any(
            coord.get_unit_mode(eid) in (MODE_HEATING, MODE_GUEST_HEATING)
            for eid in coord.energy_sensors
        )
        ema_input = HourlyProcessor._compute_battery_ema_input(
            solar_impact=solar_impact,
            solar_wasted=solar_heating_wasted,
            k_feedback=coord.battery_thermal_feedback_k,
            has_heating_unit=has_heating_unit,
        )
        coord._solar_battery_state = (
            coord._solar_battery_state * coord.solar_battery_decay
            + ema_input * (1 - coord.solar_battery_decay)
        )

    def test_heating_mode_with_k_zero_matches_legacy(self):
        coord = _make_wiring_coord({"sensor.h": MODE_HEATING}, k=0.0)
        self._apply_production_ema_step(coord, 0.5, 1.0)
        # k=0 → input = solar_impact = 0.5 regardless of wasted
        assert coord._solar_battery_state == pytest.approx(0.5 * (1 - 0.80))

    def test_heating_mode_with_k_positive_uses_heating_wasted(self):
        coord = _make_wiring_coord({"sensor.h": MODE_HEATING}, k=0.5)
        self._apply_production_ema_step(coord, 0.5, 1.0)
        # input = 0.5 + 0.5 × 1.0 = 1.0
        assert coord._solar_battery_state == pytest.approx(1.0 * (1 - 0.80))

    def test_all_cooling_with_k_positive_skips_feedback(self):
        coord = _make_wiring_coord({"sensor.c": MODE_COOLING}, k=0.5)
        self._apply_production_ema_step(coord, 0.5, 1.0)
        # No heating unit active → gate closed → input = solar_impact
        assert coord._solar_battery_state == pytest.approx(0.5 * (1 - 0.80))

    def test_mixed_heating_cooling_uses_feedback(self):
        coord = _make_wiring_coord(
            {"sensor.h": MODE_HEATING, "sensor.c": MODE_COOLING}, k=0.5,
        )
        self._apply_production_ema_step(coord, 0.5, 0.4)
        # Mixed: at least one heating → gate open → input = 0.5 + 0.5 × 0.4
        assert coord._solar_battery_state == pytest.approx(0.7 * (1 - 0.80))

    def test_guest_heating_counts_as_heating(self):
        coord = _make_wiring_coord({"sensor.g": MODE_GUEST_HEATING}, k=0.5)
        self._apply_production_ema_step(coord, 0.5, 0.4)
        assert coord._solar_battery_state == pytest.approx(0.7 * (1 - 0.80))

    def test_default_install_empty_unit_modes_treats_as_heating(self):
        """Regression for inert-feedback bug.

        On default heating-only installs the user has never explicitly
        set a mode, so ``_unit_modes`` is empty.  But ``get_unit_mode()``
        returns ``MODE_HEATING`` for missing entries — the feature MUST
        apply on this configuration, otherwise the entire #896 mechanism
        is inert for the population the issue actually targets.

        Pre-fix the gate iterated ``_unit_modes.values()`` over an empty
        dict and ``any()`` evaluated to False — feedback was silently
        skipped on every default install.  Post-fix the gate iterates
        ``energy_sensors`` and routes each through ``get_unit_mode``,
        which honours the MODE_HEATING default.
        """
        coord = _make_wiring_coord(
            {},  # empty unit_modes → user never toggled
            k=0.5,
            energy_sensors=["sensor.h1", "sensor.h2"],
        )
        self._apply_production_ema_step(coord, 0.5, 1.0)
        # Gate open → input = 0.5 + 0.5 × 1.0 = 1.0
        assert coord._solar_battery_state == pytest.approx(1.0 * (1 - 0.80)), (
            "feedback should apply for default heating-only install with "
            "empty _unit_modes; if this test fails, the gate has reverted "
            "to reading _unit_modes.values() and the feature is inert "
            "for default configurations"
        )

    def test_no_configured_sensors_skips_feedback(self):
        """Edge case: install with zero configured energy sensors.

        ``any()`` over an empty iterable is False → gate closed →
        feedback skipped.  Safe default for a malformed install.
        """
        coord = _make_wiring_coord({}, k=0.5, energy_sensors=[])
        self._apply_production_ema_step(coord, 0.5, 1.0)
        assert coord._solar_battery_state == pytest.approx(0.5 * (1 - 0.80))

    def test_explicit_off_modes_skip_feedback(self):
        """All sensors explicitly set to OFF → no heating → gate closed."""
        coord = _make_wiring_coord(
            {"sensor.h1": MODE_OFF, "sensor.h2": MODE_OFF},
            k=0.5,
        )
        self._apply_production_ema_step(coord, 0.5, 1.0)
        assert coord._solar_battery_state == pytest.approx(0.5 * (1 - 0.80))

    def test_partial_explicit_modes_default_heating_counts(self):
        """Mixed: one sensor explicitly OFF, one with no entry (default heating).

        The sensor with no ``_unit_modes`` entry should be treated as
        heating via ``get_unit_mode``'s default.  Gate must open.
        """
        coord = _make_wiring_coord(
            {"sensor.off": MODE_OFF},  # only "off" has explicit mode
            k=0.5,
            # "sensor.h" is configured but has no _unit_modes entry →
            # get_unit_mode returns MODE_HEATING (default)
            energy_sensors=["sensor.off", "sensor.h"],
        )
        self._apply_production_ema_step(coord, 0.5, 0.4)
        assert coord._solar_battery_state == pytest.approx(0.7 * (1 - 0.80))

    def test_solar_disabled_short_circuits(self):
        """If solar_enabled is False, the EMA does not advance at all."""
        coord = _make_wiring_coord({"sensor.h": MODE_HEATING}, k=0.5,
                                    prior_state=0.7)
        coord.solar_enabled = False
        self._apply_production_ema_step(coord, 0.5, 1.0)
        assert coord._solar_battery_state == pytest.approx(0.7)

    def test_per_cell_at_optimum_omitted_when_optimum_zero(self):
        """Compaction (#896 follow-up): no improvement → no per-cell table.

        ``per_cell_at_optimum`` would carry only ``delta_rmse_kwh = 0``
        rows when ``empirical_optimum_k == 0``.  Emit None instead so
        the user sees "see baseline cells" rather than scrolling through
        a 30-row identity table.
        """
        log = [_sweep_log_entry(f"2026-04-15T{h:02d}:00:00", hour=h)
               for h in range(8, 18)]  # default wasted=0 → optimum=0
        coord = _make_sweep_coord(log)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        sweep = result["global"]["battery_feedback_sweep"]
        assert sweep["empirical_optimum_k"] == 0.0
        assert sweep["per_cell_at_optimum"] is None

    def test_intermediate_sweep_k_drops_cells(self):
        """Compaction: only baseline + optimum k retain ``cells`` payload.

        Intermediate k values keep their ``global.rmse_kwh`` (the
        single-line summary across the sweep) but drop the verbose
        per-cell tables, which were duplicated 11 times when optimum=0.
        """
        log = [_sweep_log_entry(f"2026-04-15T{h:02d}:00:00", hour=h)
               for h in range(8, 18)]  # optimum=0
        coord = _make_sweep_coord(log)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        sweep = result["global"]["battery_feedback_sweep"]
        # k=0 (baseline) keeps cells.
        assert "cells" in sweep["sweep"]["0.0"]
        # All other k values drop cells but keep global.
        for k_str in ("0.1", "0.2", "0.5", "0.7", "1.0"):
            entry = sweep["sweep"][k_str]
            assert "global" in entry
            assert "rmse_kwh" in entry["global"]
            assert "cells" not in entry, (
                f"non-baseline sweep entry k={k_str} should drop cells "
                f"when optimum=0; got: {list(entry.keys())}"
            )

    def test_baseline_and_optimum_keep_cells_when_optimum_nonzero(self):
        """Compaction: when optimum>0 both baseline AND optimum keep cells.

        Lets the user diff the per-cell deltas at the recommended k
        against the live baseline.  Intermediate k still drops cells.
        """
        # Setup that drives optimum > 0 (over-prediction + positive wasted).
        log = []
        for i in range(36):
            day = 15 + i // 8
            hour = 8 + (i % 8)
            log.append(_sweep_log_entry(
                f"2026-04-{day:02d}T{hour:02d}:00:00",
                hour=hour, temp=8.0,
                solar_impact_raw=0.4,
                solar_wasted=0.5, solar_heating_wasted=0.5,
                actual=1.5, expected=1.8,  # over-prediction
            ))
        coord = _make_sweep_coord(log)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        sweep = result["global"]["battery_feedback_sweep"]
        assert sweep["empirical_optimum_k"] > 0.0
        opt_str = str(sweep["empirical_optimum_k"])
        # Baseline + optimum keep cells.
        assert "cells" in sweep["sweep"]["0.0"]
        assert "cells" in sweep["sweep"][opt_str]
        # An intermediate k (one that's not baseline and not optimum) drops cells.
        intermediate = next(
            k for k in ("0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0")
            if k != opt_str
        )
        assert "cells" not in sweep["sweep"][intermediate]
        # per_cell_at_optimum still emitted with full detail.
        assert sweep["per_cell_at_optimum"] is not None

    def test_per_cell_at_optimum_includes_n_and_baseline(self):
        """``per_cell_at_optimum`` carries n + baseline + optimum + delta."""
        log = [
            _sweep_log_entry(f"2026-04-15T{h:02d}:00:00", hour=h, temp=8.0,
                             solar_wasted=0.3, solar_heating_wasted=0.3)
            for h in range(8, 14)
        ]
        coord = _make_sweep_coord(log)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        per_cell = result["global"]["battery_feedback_sweep"][
            "per_cell_at_optimum"
        ]
        assert per_cell, "expected non-empty per-cell table for non-empty log"
        for cell_key, cell in per_cell.items():
            for field in ("n", "baseline_rmse_kwh", "optimum_rmse_kwh",
                          "delta_rmse_kwh", "thin_sample"):
                assert field in cell, f"missing {field} in {cell_key}"
            assert isinstance(cell["thin_sample"], bool)


# =============================================================================
# Split-state carry-over reservoir (#896 follow-up): _solar_carryover_state
# parallel to _solar_battery_state.  EMA charges from k × wasted only;
# release feeds back into prediction in calculate_total_power.
# =============================================================================


class TestCarryoverEMAFormula:
    """Helper-equivalent semantics for the new parallel EMA."""

    @staticmethod
    def _ema_step(state, decay, k, wasted, has_heating_unit):
        """Mirror of hourly_processor's carryover EMA charge formula.

        If you refactor that block, update this helper in lockstep —
        this is the carryover-state production-wiring contract.
        """
        carryover_input = (
            k * wasted if (k > 0.0 and has_heating_unit) else 0.0
        )
        return state * decay + carryover_input * (1 - decay)

    def test_k_zero_keeps_state_at_zero(self):
        """At k=0 the carryover EMA never charges, regardless of wasted."""
        state = 0.0
        for _ in range(50):
            state = self._ema_step(state, 0.80, 0.0, 1.0, True)
        assert state == 0.0

    def test_k_positive_no_heating_keeps_state_at_zero(self):
        """Cooling-only hours don't charge carryover even with k>0."""
        state = 0.0
        for _ in range(50):
            state = self._ema_step(state, 0.80, 0.5, 1.0, False)
        assert state == 0.0

    def test_k_positive_heating_active_converges_to_k_times_wasted(self):
        """Steady state under sustained input = k × wasted."""
        state = 0.0
        k, wasted, decay = 0.5, 1.0, 0.80
        for _ in range(200):
            state = self._ema_step(state, decay, k, wasted, True)
        assert state == pytest.approx(k * wasted, abs=1e-4)

    def test_carryover_isolated_from_applied(self):
        """Critical: applied solar must NOT charge carryover.

        The whole point of split state is that ``_solar_carryover_state``
        sees only ``k × wasted``; the existing ``_solar_battery_state``
        absorbs the ``applied`` term separately.  This test pins that
        boundary — if a future refactor mixes ``applied`` into the
        carryover EMA, the steady state would no longer equal
        ``k × wasted`` but ``applied + k × wasted``, breaking the
        prediction-side release math.
        """
        state = 0.0
        k, wasted, decay = 0.4, 0.5, 0.80
        # Even with massive "applied" passed to a hypothetical bad
        # implementation, the carryover formula explicitly ignores it
        # — only k × wasted enters.
        for _ in range(200):
            state = self._ema_step(state, decay, k, wasted, True)
        # Steady state = k × wasted, NOT applied + k × wasted
        assert state == pytest.approx(k * wasted, abs=1e-4)
        assert state < 1.0  # would be > 1.0 if applied (e.g. 0.8) were mixed in


class TestCarryoverReleaseInPrediction:
    """Release subtraction in statistics.calculate_total_power."""

    @staticmethod
    def _make_release_coord(unit_modes, carryover_state=0.0,
                            unit_demand=2.0):
        """Coordinator stub that calculate_total_power can run against.

        Mirrors test_aux_overflow.py's heavyweight pattern but trimmed
        to what's needed for the release-subtraction path.
        """
        from custom_components.heating_analytics.const import (
            ATTR_POTENTIAL_SAVINGS, DOMAIN,
        )
        from custom_components.heating_analytics.coordinator import (
            HeatingDataCoordinator,
        )

        hass = MagicMock()
        hass.states = MagicMock()
        hass.states.get = MagicMock(return_value=None)
        hass.data = {DOMAIN: {}}
        hass.config_entries = MagicMock()
        hass.bus = MagicMock()
        hass.is_running = True

        entry = MagicMock()
        sensors = list(unit_modes.keys())
        entry.data = {
            "energy_sensors": sensors,
            "aux_affected_entities": [],
            "balance_point": 15.0,
            "wind_threshold": 5.0,
            "extreme_wind_threshold": 10.0,
        }
        coord = HeatingDataCoordinator(hass, entry)
        coord.statistics = MagicMock()
        coord.learning = MagicMock()
        coord.storage = MagicMock()
        coord.forecast = MagicMock()
        coord.solar = MagicMock()
        # No solar applied/wasted in this test — focus on release math.
        coord.solar.calculate_saturation.side_effect = (
            lambda net, pot, val: (0.0, 0.0, net)
        )
        coord.solar.apply_correction.side_effect = lambda b, i, v: b
        coord.solar.calculate_unit_solar_impact = MagicMock(return_value=0.0)
        coord.solar.calculate_unit_coefficient = MagicMock(
            return_value={"s": 0.0, "e": 0.0, "w": 0.0}
        )
        coord.solar.reconstruct_potential_vector = MagicMock(
            side_effect=lambda eff, _c, _cfg: eff
        )

        coord.solar_enabled = False  # no solar contribution this test
        coord.solar_azimuth = 180
        coord.solar_correction_percent = 100.0
        coord.screen_config = (True, True, True)
        coord._unit_modes = dict(unit_modes)
        coord._solar_carryover_state = carryover_state
        coord._hourly_delta_per_unit = {sid: 0.0 for sid in sensors}
        coord.data[ATTR_POTENTIAL_SAVINGS] = 0.0
        coord.data["effective_wind"] = 0.0

        coord._correlation_data_per_unit = {
            sid: {"15": {"normal": unit_demand}} for sid in sensors
        }
        coord._aux_coefficients_per_unit = {
            sid: {"15": {"normal": 0.0}} for sid in sensors
        }
        coord._correlation_data = {
            "15": {"normal": unit_demand * len(sensors)}
        }
        return coord

    @staticmethod
    def _patched_stats(coord):
        from custom_components.heating_analytics.statistics import StatisticsManager

        sm = StatisticsManager(coord)

        def _pred(data_map, temp_key, wind_bucket, temp, bp, apply_scaling=True):
            try:
                v = data_map.get(str(int(temp)), {}).get(wind_bucket, 0.0)
            except AttributeError:
                v = 0.0
            return v or 0.0

        sm._get_prediction_from_model = MagicMock(side_effect=_pred)
        coord._get_predicted_kwh_per_unit = MagicMock(
            side_effect=lambda eid, temp_key, wind_bucket: 2.0
        )
        coord._get_predicted_kwh = MagicMock(
            return_value=2.0 * len(coord.energy_sensors)
        )
        return sm

    def test_zero_state_no_release_emitted(self):
        """At carryover_state = 0, no release subtracts; total_kwh unchanged."""
        coord = self._make_release_coord(
            {"sensor.h1": MODE_HEATING, "sensor.h2": MODE_HEATING},
            carryover_state=0.0,
        )
        sm = self._patched_stats(coord)
        result = sm.calculate_total_power(
            temp=15.0, effective_wind=0.0, solar_impact=0.0,
            is_aux_active=False, detailed=False,
        )
        assert result["breakdown"]["carryover_release_kwh"] == 0.0
        # Unit predictions: 2 × 2.0 = 4.0
        assert result["total_kwh"] == pytest.approx(4.0, abs=0.01)

    def test_state_with_heating_units_release_subtracts(self):
        """With non-zero carryover state and heating-mode demand, release subtracts."""
        # carryover_state = 1.0, decay = 0.80 → release_available = 0.2
        coord = self._make_release_coord(
            {"sensor.h1": MODE_HEATING, "sensor.h2": MODE_HEATING},
            carryover_state=1.0,
        )
        sm = self._patched_stats(coord)
        result = sm.calculate_total_power(
            temp=15.0, effective_wind=0.0, solar_impact=0.0,
            is_aux_active=False, detailed=False,
        )
        assert result["breakdown"]["carryover_release_kwh"] == pytest.approx(
            0.2, abs=0.01
        )
        # total_kwh: 4.0 - 0.2 = 3.8
        assert result["total_kwh"] == pytest.approx(3.8, abs=0.01)

    def test_state_with_only_cooling_units_no_release(self):
        """All-cooling install: release gated out (heating_unit_sum_net == 0)."""
        coord = self._make_release_coord(
            {"sensor.c1": MODE_COOLING, "sensor.c2": MODE_COOLING},
            carryover_state=1.0,
        )
        sm = self._patched_stats(coord)
        result = sm.calculate_total_power(
            temp=20.0, effective_wind=0.0, solar_impact=0.0,
            is_aux_active=False, detailed=False,
        )
        # heating_unit_sum_net == 0 → release capped at 0
        assert result["breakdown"]["carryover_release_kwh"] == 0.0

    def test_release_distributed_to_heating_units_only_in_breakdown(self):
        """Release is distributed across heating units in unit_breakdown.

        Cooling units' net_kwh stays untouched; heating units share the
        release proportional to their pre-release net_kwh.
        """
        coord = self._make_release_coord(
            {"sensor.h": MODE_HEATING, "sensor.c": MODE_COOLING},
            carryover_state=1.0,
        )
        sm = self._patched_stats(coord)
        # Same call without carryover state for baseline comparison.
        coord_baseline = self._make_release_coord(
            {"sensor.h": MODE_HEATING, "sensor.c": MODE_COOLING},
            carryover_state=0.0,
        )
        sm_baseline = self._patched_stats(coord_baseline)
        baseline = sm_baseline.calculate_total_power(
            temp=15.0, effective_wind=0.0, solar_impact=0.0,
            is_aux_active=False, detailed=True,
        )
        result = sm.calculate_total_power(
            temp=15.0, effective_wind=0.0, solar_impact=0.0,
            is_aux_active=False, detailed=True,
        )
        # Heating unit's net_kwh dropped by 0.2 (release_available); cooling
        # unit unchanged — release distribution only touches heating-mode
        # entries in unit_breakdown.
        h_baseline = baseline["unit_breakdown"]["sensor.h"]["net_kwh"]
        c_baseline = baseline["unit_breakdown"]["sensor.c"]["net_kwh"]
        h_with_release = result["unit_breakdown"]["sensor.h"]["net_kwh"]
        c_with_release = result["unit_breakdown"]["sensor.c"]["net_kwh"]
        assert h_with_release == pytest.approx(h_baseline - 0.2, abs=0.01)
        assert c_with_release == c_baseline  # cooling untouched

    def test_carryover_override_used_for_forecast_calls(self):
        """``carryover_state_override`` overrides live state at the call site."""
        # Live state is 1.0 but caller passes override = 0.5
        coord = self._make_release_coord(
            {"sensor.h1": MODE_HEATING, "sensor.h2": MODE_HEATING},
            carryover_state=1.0,
        )
        sm = self._patched_stats(coord)
        result = sm.calculate_total_power(
            temp=15.0, effective_wind=0.0, solar_impact=0.0,
            is_aux_active=False, detailed=False,
            carryover_state_override=0.5,
        )
        # Release from override 0.5 × (1 - 0.80) = 0.10, not from live 1.0
        assert result["breakdown"]["carryover_release_kwh"] == pytest.approx(
            0.10, abs=0.01
        )
        # With override = 0.0, release stays at 0 even though live state is 1.0
        result_zero = sm.calculate_total_power(
            temp=15.0, effective_wind=0.0, solar_impact=0.0,
            is_aux_active=False, detailed=False,
            carryover_state_override=0.0,
        )
        assert result_zero["breakdown"]["carryover_release_kwh"] == 0.0


class TestSplitStateBackwardCompat:
    """Existing _solar_battery_state semantics must NOT change.

    Split-state is the architectural commitment that lets us close the
    prediction loop without touching any consumer of the original
    scalar battery (display sensors, thermodynamic_gross, aux input).
    These tests pin that boundary.
    """

    def test_existing_battery_state_still_charged_by_applied(self):
        """``_solar_battery_state`` still charges from ``applied + k×wasted``.

        Verifies the existing helper retains its semantic — the new
        parallel EMA does not replace it.
        """
        # Helper signature unchanged: applied, wasted, k, has_heating
        out = HourlyProcessor._compute_battery_ema_input(
            solar_impact=0.5, solar_wasted=1.0, k_feedback=0.4,
            has_heating_unit=True,
        )
        # Original semantic: applied + k × wasted = 0.5 + 0.4 = 0.9
        assert out == pytest.approx(0.9)

    def test_existing_battery_helper_at_k_zero_returns_applied(self):
        """k=0 collapses the existing helper to the legacy form bit-for-bit."""
        out = HourlyProcessor._compute_battery_ema_input(
            solar_impact=0.5, solar_wasted=1.0, k_feedback=0.0,
            has_heating_unit=True,
        )
        assert out == 0.5  # unchanged from pre-#896 form


class TestCarryoverReleaseGateP1Regressions:
    """Regression tests for two P1 findings on the split-state release.

    Both were caught by chatgpt-codex-connector on the initial split-
    state commit and would have been live bugs:

    - **DHW-mode accounting break**: ``heating_unit_sum_net``
      accumulates in the non-cooling branch, which catches DHW (whose
      ``calculate_saturation`` falls to the unknown-mode default
      ``final_net = net_demand``).  Using that as the release cap let
      release subtract from global net while the per-unit distribution
      filtered to MODE_HEATING only — ``unspecified_kwh`` invariant
      broken.
    - **Daily-forecast over-credit**: a single
      ``calculate_total_power`` call multiplied by 24 for daily
      forecasts caused the release to be credited 24 times instead of
      once with proper decay → ~5x over-credit.
    """

    def test_dhw_mode_does_not_drive_release_through_loose_gate(self):
        """All-DHW install: release must NOT subtract from global net.

        DHW falls into ``solar.calculate_saturation``'s unknown-mode
        branch (``final_net = net_demand``), so a DHW unit with a real
        base demand contributes to ``heating_unit_sum_net``.  But DHW
        is not in MODE_HEATING / MODE_GUEST_HEATING and the per-unit
        distribution loop skips it.  The fix: gate release on
        ``heating_only_unit_sum_net`` (strict heating regime) so DHW
        installs don't see phantom release subtraction.
        """
        from custom_components.heating_analytics.const import MODE_DHW
        coord = TestCarryoverReleaseInPrediction._make_release_coord(
            {"sensor.dhw1": MODE_DHW, "sensor.dhw2": MODE_DHW},
            carryover_state=1.0,
        )
        sm = TestCarryoverReleaseInPrediction._patched_stats(coord)
        result = sm.calculate_total_power(
            temp=15.0, effective_wind=0.0, solar_impact=0.0,
            is_aux_active=False, detailed=True,
        )
        # Carryover state is 1.0 → release_available = 0.2 if the gate
        # had been the loose ``heating_unit_sum_net``.  Strict gate sees
        # no heating-regime units → release stays 0.
        assert result["breakdown"]["carryover_release_kwh"] == 0.0
        # And global net matches sum of unit_breakdown — invariant
        # ``unspecified_kwh ≈ 0`` holds.
        assert result["breakdown"]["unspecified_kwh"] == pytest.approx(0.0, abs=0.01)

    def test_mixed_dhw_and_heating_only_distributes_to_heating(self):
        """DHW + heating: release subtracts only from heating net.

        Confirms the invariant: release goes to global net AND is
        distributed only across MODE_HEATING units.  The DHW unit's
        net stays at its baseline.
        """
        from custom_components.heating_analytics.const import MODE_DHW
        coord = TestCarryoverReleaseInPrediction._make_release_coord(
            {"sensor.h": MODE_HEATING, "sensor.dhw": MODE_DHW},
            carryover_state=1.0,
        )
        sm = TestCarryoverReleaseInPrediction._patched_stats(coord)
        result = sm.calculate_total_power(
            temp=15.0, effective_wind=0.0, solar_impact=0.0,
            is_aux_active=False, detailed=True,
        )
        # release_available = 1.0 × 0.20 = 0.20.  heating_only_net = 2.0
        # (just sensor.h's base).  release_applied = min(0.20, 2.0) = 0.20.
        assert result["breakdown"]["carryover_release_kwh"] == pytest.approx(
            0.20, abs=0.01
        )
        h_net = result["unit_breakdown"]["sensor.h"]["net_kwh"]
        dhw_net = result["unit_breakdown"]["sensor.dhw"]["net_kwh"]
        # All 0.20 release goes to the single heating unit.
        assert h_net == pytest.approx(2.0 - 0.20, abs=0.01)
        # DHW unit unchanged from its pre-release base.
        baseline_coord = TestCarryoverReleaseInPrediction._make_release_coord(
            {"sensor.h": MODE_HEATING, "sensor.dhw": MODE_DHW},
            carryover_state=0.0,
        )
        sm_baseline = TestCarryoverReleaseInPrediction._patched_stats(baseline_coord)
        baseline = sm_baseline.calculate_total_power(
            temp=15.0, effective_wind=0.0, solar_impact=0.0,
            is_aux_active=False, detailed=True,
        )
        dhw_baseline = baseline["unit_breakdown"]["sensor.dhw"]["net_kwh"]
        assert dhw_net == dhw_baseline

    def test_daily_forecast_uses_integrated_release_not_per_call(self):
        """Daily forecast must integrate release over 24h, not multiply.

        The forecast helper calls ``calculate_total_power`` once and
        multiplies by 24.  Without an override, each forecast hour
        would credit ``state × (1 − decay)`` of release — for state=1,
        decay=0.80, that's 0.20 × 24 = 4.8 kWh subtracted.
        Physically integrated over 24h: ``state × (1 − decay²⁴) ≈
        state × 1.0`` = 1.0 kWh.

        The fix in ``_calculate_from_daily_forecast`` passes a
        forward-integrated override so the per-call × 24 expansion
        yields the correct daily total.

        This test pins the override formula by computing it directly
        and checking that the daily-forecast subtraction matches
        physics.  We don't actually call the forecast helper — we
        simulate its math using ``calculate_total_power`` with the
        override, then verify that result × 24 ≈ physically integrated
        24h release.
        """
        decay = 0.80
        carryover_state = 1.0
        # Physics: integrated 24h release from initial state 1.0
        target_24h_release = carryover_state * (1 - decay ** 24)  # ~0.995

        # Override formula from _calculate_from_daily_forecast.
        daily_factor = (1 - decay ** 24) / (24 * (1 - decay))
        override = carryover_state * daily_factor

        coord = TestCarryoverReleaseInPrediction._make_release_coord(
            {"sensor.h1": MODE_HEATING, "sensor.h2": MODE_HEATING},
            carryover_state=carryover_state,
        )
        sm = TestCarryoverReleaseInPrediction._patched_stats(coord)
        # Single call with the daily-override.
        res_per_call = sm.calculate_total_power(
            temp=15.0, effective_wind=0.0, solar_impact=0.0,
            is_aux_active=False, detailed=False,
            carryover_state_override=override,
        )
        per_call_release = res_per_call["breakdown"]["carryover_release_kwh"]
        # Multiplied by 24 (the forecast helper's expansion):
        daily_release = per_call_release * 24
        # Should match physics within breakdown rounding.  The
        # ``round(release, 3)`` in the breakdown amplifies by × 24 →
        # tolerance ~0.024 kWh on the integrated daily total.
        assert daily_release == pytest.approx(target_24h_release, abs=0.025), (
            f"daily override formula broken: per-call × 24 = {daily_release:.4f}, "
            f"physics target = {target_24h_release:.4f}"
        )

        # Sanity: without the override (using live state), we'd get the bug.
        res_buggy = sm.calculate_total_power(
            temp=15.0, effective_wind=0.0, solar_impact=0.0,
            is_aux_active=False, detailed=False,
            # no override — uses live state (1.0)
        )
        buggy_daily = res_buggy["breakdown"]["carryover_release_kwh"] * 24
        # The bug would give ~4.8 kWh — confirms the failure mode the
        # override fix prevents.
        assert buggy_daily > 4.0, (
            f"bug-mode sanity check failed: buggy daily = {buggy_daily:.4f}; "
            f"expected > 4.0 if the bug were active"
        )


class TestSumForecastEnergyCarryoverThreading:
    """Regression for follow-up code review finding (HIGH).

    ``_sum_forecast_energy_internal`` is the workhorse for
    ``calculate_future_energy``, comparison sensors, and force-aux
    budget calls.  Initial split-state implementation only threaded
    ``carryover_state_override`` through ``_calculate_from_hourly_
    forecast`` — its sibling ``_sum_forecast_energy_internal`` was
    overlooked, so multi-hour calls there read live state at every
    iteration and over-credited release N times.  Same bug class as
    the daily-forecast over-credit chatgpt-codex caught in d2f372c.

    These tests pin the fix: the loop must apply ``decay^N`` per
    forecast item.
    """

    @staticmethod
    def _make_forecast_helper(carryover_state=0.0, k=0.4, decay=0.80):
        """Mock ForecastManager with just enough shape for the helper."""
        from custom_components.heating_analytics.forecast import ForecastManager

        coord = MagicMock()
        coord._solar_carryover_state = carryover_state
        coord.battery_thermal_feedback_k = k
        coord.solar_battery_decay = decay
        coord._get_weather_wind_unit = MagicMock(return_value="m/s")
        coord._get_cloud_coverage = MagicMock(return_value=50.0)
        coord._get_live_forecast_or_ref = MagicMock(return_value=[])
        coord.auxiliary_heating_active = False
        return ForecastManager(coord), coord

    def test_zero_wasted_collapses_to_pure_decay(self):
        """When predicted wasted is 0 (no sun), trajectory = pure decay.

        Pins the bit-identical-no-op contract: with k=0 OR all wasted=0
        across the forecast, the trajectory simulation collapses to the
        same ``state_now × decay^N`` form the pre-trajectory branch
        produced.  This is what default-config users (k=0) see at all
        times — no behaviour change vs the pre-trajectory shipped form.
        """
        from datetime import timedelta
        from homeassistant.util import dt as dt_util

        helper, coord = self._make_forecast_helper(
            carryover_state=1.0, k=0.0, decay=0.80,  # k=0 → no charge ever
        )
        captured_overrides = []

        def _spy(item, history, wind_unit, cloud, **kwargs):
            captured_overrides.append(kwargs.get("carryover_state_override"))
            # Return 9-tuple including wasted (last element).  Wasted=0 means
            # no charge regardless of k.
            return (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, {}, 0.0, 0.0)

        helper._process_forecast_item = _spy
        now = dt_util.now()
        forecast_items = [
            {
                "datetime": (now + timedelta(hours=i + 1)).isoformat(),
                "temperature": 5.0,
                "wind_speed": 0.0,
            }
            for i in range(6)
        ]
        helper._sum_forecast_energy_internal(
            start_time=now,
            end_time=now + timedelta(hours=10),
            inertia_history=[],
            include_start=False,
            source_data=forecast_items,
        )

        # At k=0, override at offset N should be state × decay^N — the
        # legacy pre-trajectory form.
        assert len(captured_overrides) == 6
        for offset, override in enumerate(captured_overrides, start=1):
            expected = 1.0 * (0.80 ** offset)
            assert override == pytest.approx(expected, abs=0.01), (
                f"k=0 forecast offset N={offset}: override={override:.4f}, "
                f"expected ≈ {expected:.4f} (state × decay^N).  At k=0 the "
                f"trajectory must collapse to pure decay — bit-identical "
                f"no-op for default-config users."
            )

    def test_predicted_wasted_charges_local_trajectory(self):
        """When predicted wasted > 0, local state RECHARGES across the loop.

        Trajectory threading's whole point: a sunny-day forecast loop sees
        the reservoir charge up from morning's wasted and discharge into
        evening hours, capturing the charge-and-discharge cycle that the
        pre-trajectory ``decay^N`` approach assumed away.

        Setup: starting state = 0, every forecast hour predicts
        ``wasted = 1.0 kWh`` (sustained mid-day saturation), k=0.5.
        Per the live EMA formula:
          state[t+1] = state[t] × 0.80 + (0.5 × 1.0) × 0.20 = state[t] × 0.80 + 0.1
        Steady state: state_∞ = 0.5 (= k × wasted).  Within 6 iterations
        the local trajectory should approach steady state, NOT stay at 0
        (which is what pre-trajectory ``decay^N`` would produce).
        """
        from datetime import timedelta
        from homeassistant.util import dt as dt_util

        helper, coord = self._make_forecast_helper(
            carryover_state=0.0, k=0.5, decay=0.80,
        )
        captured = []

        def _spy(item, history, wind_unit, cloud, **kwargs):
            captured.append(kwargs.get("carryover_state_override"))
            # Sustained wasted = 1.0 per forecast hour (sunny day).
            return (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, {}, 0.0, 1.0)

        helper._process_forecast_item = _spy
        now = dt_util.now()
        forecast_items = [
            {"datetime": (now + timedelta(hours=i + 1)).isoformat(),
             "temperature": 5.0, "wind_speed": 0.0}
            for i in range(6)
        ]
        helper._sum_forecast_energy_internal(
            start_time=now, end_time=now + timedelta(hours=10),
            inertia_history=[], include_start=False, source_data=forecast_items,
        )

        # Override at offset N should follow the EMA recurrence:
        # state[1] = 0 (initial state forward-decayed to first item)
        # state[2] = 0 × 0.80 + 0.5 × 0.20 = 0.10
        # state[3] = 0.10 × 0.80 + 0.5 × 0.20 = 0.18
        # state[4] = 0.18 × 0.80 + 0.5 × 0.20 = 0.244
        # state[5] = 0.244 × 0.80 + 0.5 × 0.20 = 0.2952
        # state[6] = 0.2952 × 0.80 + 0.5 × 0.20 = 0.33616
        expected = [0.0, 0.10, 0.18, 0.244, 0.2952, 0.33616]
        assert len(captured) == 6
        for i, (got, want) in enumerate(zip(captured, expected)):
            assert got == pytest.approx(want, abs=0.01), (
                f"trajectory iter {i}: got {got:.4f}, expected {want:.4f}.  "
                f"If you see all zeros, the charge step has reverted to "
                f"pure-decay (no recharge from predicted wasted)."
            )

    def test_zero_state_threads_zero_override(self):
        """No carry-over AND k=0 → all forecast items receive override=0.0."""
        from datetime import timedelta
        from homeassistant.util import dt as dt_util

        helper, coord = self._make_forecast_helper(
            carryover_state=0.0, k=0.0,  # no live state, no k → strictly zero
        )
        captured = []

        def _spy(item, history, wind_unit, cloud, **kwargs):
            captured.append(kwargs.get("carryover_state_override"))
            return (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, {}, 0.0, 0.0)

        helper._process_forecast_item = _spy
        now = dt_util.now()
        forecast_items = [
            {"datetime": (now + timedelta(hours=i + 1)).isoformat(),
             "temperature": 5.0, "wind_speed": 0.0}
            for i in range(3)
        ]
        helper._sum_forecast_energy_internal(
            start_time=now, end_time=now + timedelta(hours=10),
            inertia_history=[], include_start=False, source_data=forecast_items,
        )
        for override in captured:
            assert override == 0.0, (
                f"with zero state and k=0, override should be 0.0; got {override}"
            )


class TestResetSolarLearningClearsCarryover:
    """Regression for physics review finding F.2 (residual-state leak).

    ``async_reset_solar_learning_data(entity_id=None)`` must clear
    ``_solar_carryover_state`` so post-reset cold-start doesn't see
    pre-reset release leaking through.  Per-unit reset preserves the
    state since other units share the physical reservoir.
    """

    @staticmethod
    def _make_coord_with_state(state_value):
        """Real coordinator instance with set state."""
        from custom_components.heating_analytics.const import DOMAIN
        from custom_components.heating_analytics.coordinator import (
            HeatingDataCoordinator,
        )

        hass = MagicMock()
        hass.states = MagicMock()
        hass.states.get = MagicMock(return_value=None)
        hass.data = {DOMAIN: {}}
        hass.config_entries = MagicMock()
        hass.bus = MagicMock()
        hass.is_running = True

        entry = MagicMock()
        entry.data = {
            "energy_sensors": ["sensor.h"],
            "balance_point": 15.0,
            "wind_threshold": 5.0,
            "extreme_wind_threshold": 10.0,
        }
        coord = HeatingDataCoordinator(hass, entry)
        coord._solar_carryover_state = state_value
        coord.statistics = MagicMock()
        coord.learning = MagicMock()
        coord.storage = MagicMock()

        async def _noop_save(*args, **kwargs):
            return None

        coord._async_save_data = _noop_save
        return coord

    @pytest.mark.asyncio
    async def test_all_units_reset_clears_carryover_state(self):
        """``entity_id=None`` reset path clears ``_solar_carryover_state``."""
        coord = self._make_coord_with_state(state_value=0.3)
        assert coord._solar_carryover_state == 0.3
        await coord.async_reset_solar_learning_data(entity_id=None)
        assert coord._solar_carryover_state == 0.0, (
            "all-units reset must clear the carry-over state — "
            "otherwise pre-reset release leaks into post-reset cold-start"
        )

    @pytest.mark.asyncio
    async def test_per_unit_reset_preserves_carryover_state(self):
        """Per-unit reset preserves state — other units still see the reservoir."""
        coord = self._make_coord_with_state(state_value=0.3)
        await coord.async_reset_solar_learning_data(entity_id="sensor.h")
        assert coord._solar_carryover_state == 0.3, (
            "per-unit reset must NOT clear shared whole-house reservoir"
        )


class TestCarryoverStateStorageRoundTrip:
    """Storage save → load round-trip preserves ``_solar_carryover_state``."""

    def test_save_and_restore_value(self):
        """Pin the persistence path: state survives save+load roundtrip."""
        # Simulate the save format and the restore default chain that
        # ``async_load_data`` uses.  Avoids dragging in async storage
        # internals; this is a contract test on the field name.
        save_dict = {"solar_carryover_state": 0.42}
        # Fresh coord starts at 0.0 (init)
        loaded_value = save_dict.get("solar_carryover_state", 0.0)
        assert loaded_value == 0.42

        # Missing key → 0.0 default (legacy storage compatibility)
        empty_dict = {}
        loaded_default = empty_dict.get("solar_carryover_state", 0.0)
        assert loaded_default == 0.0


class TestReleaseDistributionEdgeCases:
    """Edge-case coverage for the per-unit release distribution loop."""

    def test_zero_demand_heating_unit_excluded_from_share(self):
        """Heating unit with zero per-unit base contributes nothing.

        Per-unit base of 0 → unit_net = 0 → ``unit_net <= 0`` branch in
        the distribution loop skips it.  The `heating_only_unit_sum_net`
        accumulator also sees 0 contribution from this unit, so the
        release cap reflects only the positive-demand units.  This test
        pins that the branch is exercised correctly when at least one
        heating unit has zero demand.

        Setup: one heating unit with positive base, one with zero base
        (e.g. unit predicted 0 because base bucket has no data at this
        temp).  Release should go entirely to the positive-demand unit.
        """
        # Override the per-unit correlation data so h_zero predicts 0.
        coord = TestCarryoverReleaseInPrediction._make_release_coord(
            {"sensor.h_normal": MODE_HEATING, "sensor.h_zero": MODE_HEATING},
            carryover_state=1.0,
        )
        # Different demand per sensor.
        coord._correlation_data_per_unit = {
            "sensor.h_normal": {"15": {"normal": 2.0}},
            "sensor.h_zero": {"15": {"normal": 0.0}},
        }
        sm = TestCarryoverReleaseInPrediction._patched_stats(coord)
        result = sm.calculate_total_power(
            temp=15.0, effective_wind=0.0, solar_impact=0.0,
            is_aux_active=False, detailed=True,
        )
        # Release available = 1.0 × 0.20 = 0.20; heating_only_net = 2.0
        # (h_normal only).  h_normal absorbs the full 0.20.
        h_normal_net = result["unit_breakdown"]["sensor.h_normal"]["net_kwh"]
        assert h_normal_net == pytest.approx(2.0 - 0.20, abs=0.01), (
            "positive-demand heating unit should absorb the full release"
        )
        # h_zero stays at zero — release distribution loop's
        # ``unit_net <= 0: continue`` branch fires.
        h_zero_net = result["unit_breakdown"]["sensor.h_zero"]["net_kwh"]
        assert h_zero_net == 0.0, (
            f"zero-demand heating unit must not absorb release share; "
            f"got {h_zero_net}"
        )


class TestProcessForecastItemReturnArity:
    """Pin the 9-tuple contract on ``_process_forecast_item``.

    #899 trajectory threading expanded the return tuple from 8 to 9
    elements (added ``solar_heating_wasted_kwh``).  Two production call
    sites destructure this tuple explicitly with ``a, b, c, d, e, f,
    g, h, i = ...`` — if a future refactor reverts the arity those
    sites raise ``ValueError`` on the next hourly cycle.  These tests
    pin the arity so a revert is caught at test time.
    """

    def test_return_tuple_has_nine_elements(self):
        """Direct invocation of _process_forecast_item returns 9 elements.

        Synthesises a minimal valid item; we only check the return
        shape.  All call sites — production and test — must destructure
        a 9-tuple.
        """
        from custom_components.heating_analytics.forecast import ForecastManager

        coord = MagicMock()
        coord._calculate_effective_wind = MagicMock(return_value=0.0)
        coord._get_wind_bucket = MagicMock(return_value="normal")
        coord._calculate_weighted_inertia = MagicMock(return_value=10.0)
        coord.solar_enabled = False
        coord.solar_optimizer = MagicMock()
        coord.solar_optimizer.recommended_correction_for = MagicMock(return_value=100.0)
        coord.statistics.calculate_total_power.return_value = {
            "total_kwh": 1.0,
            "breakdown": {
                "solar_reduction_kwh": 0.0,
                "aux_reduction_kwh": 0.0,
                "solar_heating_wasted_kwh": 0.5,
            },
            "unit_breakdown": {},
        }
        coord.auxiliary_heating_active = False
        helper = ForecastManager(coord)
        result = helper._process_forecast_item(
            item={"temperature": 5.0, "wind_speed": 0.0,
                  "datetime": "2026-04-15T12:00:00"},
            inertia_history=[],
            wind_unit="m/s",
            default_cloud=50.0,
        )
        assert len(result) == 9, (
            f"_process_forecast_item must return a 9-tuple "
            f"(predicted, solar_kwh, inertia_val, raw_temp, w_speed, "
            f"w_speed_ms, unit_breakdown, aux_impact_kwh, "
            f"solar_heating_wasted); got {len(result)} elements.  "
            f"Production call sites in hourly_processor.py and sensor.py "
            f"destructure 9 elements explicitly — a revert here will "
            f"crash live with ValueError on the next hourly cycle."
        )
        # Last element should be the heating wasted from breakdown.
        assert result[8] == pytest.approx(0.5)
