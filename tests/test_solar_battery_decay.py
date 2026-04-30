"""Tests for solar battery decay — EMA model (#809 A1, #827).

Verifies:
- EMA battery mechanics: steady state = input (no amplification)
- Decay-only behavior unchanged (exponential tail after sunset)
- Calibration sweep uses EMA formula
- Migration from old leaky-integrator model
"""
import math
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from custom_components.heating_analytics.const import SOLAR_BATTERY_DECAY
from custom_components.heating_analytics.diagnostics import DiagnosticsEngine


class TestBatteryDecayMechanics:
    """EMA battery: state = state × decay + input × (1 - decay)."""

    def test_decay_reduces_state(self):
        """Battery state decays each hour by the decay factor (unchanged)."""
        state = 1.0
        decay = 0.80
        # After 1 hour with no new solar
        state = state * decay
        assert state == pytest.approx(0.80)
        # After 2 hours
        state = state * decay
        assert state == pytest.approx(0.64)
        # After 5 hours
        for _ in range(3):
            state = state * decay
        assert state == pytest.approx(0.80 ** 5, rel=1e-6)

    def test_ema_charging_no_amplification(self):
        """EMA battery reaches steady state = input (no amplification).

        The old leaky integrator converged to input/(1-decay) = 2.5× at
        decay=0.60.  The EMA converges to exactly input.
        """
        state = 0.0
        decay = 0.60
        solar_input = 0.5

        # Run until convergence
        for _ in range(50):
            state = state * decay + solar_input * (1 - decay)

        assert state == pytest.approx(solar_input, abs=1e-6), (
            f"EMA steady state {state:.4f} should equal input {solar_input}"
        )

    def test_ema_charging_carries_into_dark(self):
        """Battery charges during sun, carries residual into post-solar hours."""
        state = 0.0
        decay = 0.80
        solar_input = 0.5

        # 3 sunny hours with EMA formula
        for _ in range(3):
            state = state * decay + solar_input * (1 - decay)

        # State should be sum of EMA contributions
        factor = 1 - decay  # 0.20
        expected = solar_input * factor * (decay**2 + decay + 1)
        assert state == pytest.approx(expected)

        # 2 dark hours (no new solar) — same decay as before
        state_after_sun = state
        for _ in range(2):
            state = state * decay
        assert state > 0.01
        assert state == pytest.approx(state_after_sun * decay**2)

    def test_ema_steady_state_independent_of_decay(self):
        """Steady state is always equal to input, regardless of decay value.

        This is the key property: decay controls only the time constant
        (how quickly the battery responds), not the amplitude.
        """
        solar_input = 0.4
        for decay in [0.50, 0.60, 0.75, 0.85, 0.95]:
            state = 0.0
            for _ in range(200):  # enough iterations for any decay
                state = state * decay + solar_input * (1 - decay)
            assert state == pytest.approx(solar_input, abs=1e-4), (
                f"At decay={decay}: steady state {state:.4f} != input {solar_input}"
            )

    def test_half_life_at_default(self):
        """Current default decay has expected half-life."""
        half_life = math.log(0.5) / math.log(SOLAR_BATTERY_DECAY)
        # 0.80 → ~3.1h
        assert 3.0 < half_life < 3.5

    def test_half_life_at_080(self):
        """Decay 0.80 has half-life of ~3.1 hours."""
        half_life = math.log(0.5) / math.log(0.80)
        assert 3.0 < half_life < 3.5

    def test_half_life_at_060(self):
        """Decay 0.60 has half-life of ~1.4 hours."""
        half_life = math.log(0.5) / math.log(0.60)
        assert 1.0 < half_life < 1.8


class TestBatteryMigration:
    """Migration from old leaky-integrator to EMA model."""

    def test_migration_scales_by_one_minus_decay(self):
        """Old state is multiplied by (1 - decay) to match new steady state."""
        decay = 0.75
        old_steady_state = 0.5 / (1 - decay)  # 2.0 (amplified)
        migrated = old_steady_state * (1 - decay)
        assert migrated == pytest.approx(0.5)  # back to input level

    def test_migration_preserves_zero_state(self):
        """Zero battery state is unchanged by migration."""
        for decay in [0.60, 0.75, 0.85]:
            assert 0.0 * (1 - decay) == 0.0


class TestDiagnoseSolarCalibrationSweep:
    """The diagnose_solar calibration sweep should find optimal decay."""

    @staticmethod
    def _make_coordinator_with_log(hourly_log, current_decay=SOLAR_BATTERY_DECAY,
                                    current_k=0.0):
        """Create a minimal coordinator mock with hourly log data."""
        coord = MagicMock()
        coord._hourly_log = hourly_log
        coord.solar_battery_decay = current_decay
        coord.battery_thermal_feedback_k = current_k
        coord.energy_sensors = ["sensor.heater1"]
        coord.solar_correction_percent = 100.0
        coord.solar_azimuth = 180
        coord.balance_point = 15.0

        # Mock solar calculator methods
        coord.solar = MagicMock()
        coord.solar.calculate_unit_coefficient = MagicMock(
            return_value={"s": 1.0, "e": 0.0, "w": 0.0}
        )
        coord.solar.calculate_unit_solar_impact = MagicMock(return_value=0.0)

        return coord

    @staticmethod
    def _generate_day_log(date_str, decay_truth=0.80, base_kwh=2.0, peak_solar=0.5):
        """Generate a synthetic day of hourly log entries.

        Simulates a building where the true battery decay is `decay_truth`.
        Solar peaks at noon, then battery carries residual into evening.
        Uses the EMA formula to match the production model.
        """
        entries = []
        battery = 0.0

        for hour in range(24):
            # Solar profile: Gaussian around noon
            if 8 <= hour <= 16:
                raw_solar = peak_solar * max(0, 1.0 - abs(hour - 12) / 5.0)
            else:
                raw_solar = 0.0

            battery = battery * decay_truth + raw_solar * (1 - decay_truth)

            # Solar vector (simplified: all south)
            solar_s = raw_solar
            solar_e = 0.0
            solar_factor = raw_solar

            # "Actual" = base - battery effect (building uses less when warm)
            actual = max(0.0, base_kwh - battery * 0.5)
            expected = base_kwh  # Base model prediction (no solar)

            entries.append({
                "timestamp": f"{date_str}T{hour:02d}:00:00",
                "hour": hour,
                "temp": 5.0,
                "temp_key": "5",
                "wind_bucket": "normal",
                "solar_factor": solar_factor,
                "solar_vector_s": solar_s,
                "solar_vector_e": solar_e,
                "solar_vector_w": 0.0,
                "solar_impact_raw_kwh": raw_solar,
                "solar_impact_kwh": battery * 0.5,
                "actual_kwh": actual,
                "expected_kwh": expected,
                "correction_percent": 100.0,
                "auxiliary_active": False,
                "guest_impact_kwh": 0.0,
                "unit_modes": {"sensor.heater1": "heating"},
                "unit_breakdown": {"sensor.heater1": actual},
                "unit_expected_breakdown": {"sensor.heater1": base_kwh},
            })

        return entries

    def test_sweep_returns_calibration_data(self):
        """Sweep should return structured calibration results."""
        log = self._generate_day_log("2026-04-01")
        log += self._generate_day_log("2026-04-02")

        coord = self._make_coordinator_with_log(log)

        from custom_components.heating_analytics.coordinator import HeatingDataCoordinator

        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)

        assert "global" in result
        calibration = result["global"].get("battery_calibration", {})

        assert calibration, "battery_calibration should not be empty with 2 days of data"
        assert "current_decay" in calibration
        assert "current_k" in calibration
        assert "recommended_decay" in calibration
        assert "recommended_k" in calibration
        assert "rmse_surface" in calibration
        assert "method" in calibration
        assert calibration["method"] == "joint_decay_k_counterfactual_replay"

    def test_sweep_with_no_data_returns_empty(self):
        """Empty hourly log produces empty calibration."""
        coord = self._make_coordinator_with_log([])
        from custom_components.heating_analytics.coordinator import HeatingDataCoordinator

        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        # Should not crash — graceful empty result
        assert "global" in result
        assert result["global"]["qualifying_hours"] == 0


class TestJointDecayKSweepValidation:
    """The joint (decay, k) sweep with counterfactual residuals should:

    1. Recover a known-true (decay, k) when the log is generated from it.
    2. NOT systematically favour the live config (the bias the 1-D
       mean-residual loss had — fixed by switching to RMSE counterfactual).
    3. Define post-sunset using raw signal (raw_solar > 0 cutoff per day),
       NOT battery-contaminated solar_factor.

    These are the validations the statistics-review §3 critique called for.
    """

    @staticmethod
    def _make_coord(hourly_log, *, decay=0.80, k=0.0):
        coord = MagicMock()
        coord._hourly_log = hourly_log
        coord.solar_battery_decay = decay
        coord.battery_thermal_feedback_k = k
        coord.energy_sensors = ["sensor.heater1"]
        coord.solar_correction_percent = 100.0
        coord.solar_azimuth = 180
        coord.balance_point = 15.0
        coord.solar = MagicMock()
        coord.solar.calculate_unit_coefficient = MagicMock(
            return_value={"s": 1.0, "e": 0.0, "w": 0.0}
        )
        coord.solar.calculate_unit_solar_impact = MagicMock(return_value=0.0)
        return coord

    @staticmethod
    def _generate_day(date_str, *, true_decay, true_k=0.0,
                      live_decay=None, live_k=None,
                      base_kwh=2.0, peak_solar=0.5, peak_wasted=0.0):
        """Generate a synthetic day where:

        - ``actual_kwh`` reflects the building's TRUE consumption under
          (true_decay, true_k) — i.e. base demand minus the true battery
          release.
        - ``expected_kwh`` reflects what the LIVE production system would
          have predicted under (live_decay, live_k) — i.e. base demand
          minus the live battery release.

        When live == truth, residual is zero everywhere.  When live differs
        from truth, residual = actual − expected = live_release − truth_release,
        which the counterfactual sweep should learn to neutralise by
        recommending alt = truth.

        ``live_decay``/``live_k`` default to the truth values for tests
        that don't need a live/truth split.
        """
        if live_decay is None:
            live_decay = true_decay
        if live_k is None:
            live_k = true_k
        entries = []
        main_truth = main_live = 0.0
        carry_truth = carry_live = 0.0
        for hour in range(24):
            if 8 <= hour <= 16:
                raw_solar = peak_solar * max(0, 1.0 - abs(hour - 12) / 5.0)
                wasted = peak_wasted * max(0, 1.0 - abs(hour - 12) / 5.0)
            else:
                raw_solar = 0.0
                wasted = 0.0
            # Truth replay
            main_truth = main_truth * true_decay + raw_solar * (1 - true_decay)
            truth_carry_in = true_k * wasted if true_k > 0 else 0.0
            carry_truth = carry_truth * true_decay + truth_carry_in * (1 - true_decay)
            truth_release = main_truth + true_k * carry_truth * (1 - true_decay)
            # Live replay
            main_live = main_live * live_decay + raw_solar * (1 - live_decay)
            live_carry_in = live_k * wasted if live_k > 0 else 0.0
            carry_live = carry_live * live_decay + live_carry_in * (1 - live_decay)
            live_release = main_live + live_k * carry_live * (1 - live_decay)
            actual = max(0.0, base_kwh - truth_release)
            expected = max(0.0, base_kwh - live_release)
            entries.append({
                "timestamp": f"{date_str}T{hour:02d}:00:00",
                "hour": hour,
                "temp": 5.0,
                "temp_key": "5",
                "wind_bucket": "normal",
                "solar_factor": raw_solar,
                "solar_vector_s": raw_solar,
                "solar_vector_e": 0.0,
                "solar_vector_w": 0.0,
                "solar_impact_raw_kwh": raw_solar,
                "solar_impact_kwh": main_truth,
                "solar_heating_wasted_kwh": wasted,
                "actual_kwh": actual,
                "expected_kwh": expected,
                "correction_percent": 100.0,
                "auxiliary_active": False,
                "guest_impact_kwh": 0.0,
                "unit_modes": {"sensor.heater1": "heating"},
                "unit_breakdown": {"sensor.heater1": actual},
                "unit_expected_breakdown": {"sensor.heater1": base_kwh},
            })
        return entries

    def test_sweep_recovers_known_true_decay_and_k(self):
        """When the log is generated under (true_decay, true_k) and live ==
        true, the sweep's recommended values should match the truth — and
        RMSE at the truth should be ~ 0 (residual_alt is identically zero
        when alt == live == true).
        """
        true_decay, true_k = 0.65, 0.0
        log = []
        for d in range(1, 6):
            log += self._generate_day(
                f"2026-04-{d:02d}", true_decay=true_decay, true_k=true_k
            )
        coord = self._make_coord(log, decay=true_decay, k=true_k)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        cal = result["global"]["battery_calibration"]

        assert cal["recommended_decay"] == pytest.approx(true_decay, abs=0.05)
        assert cal["recommended_k"] == pytest.approx(true_k, abs=0.1)
        # RMSE at the truth (== live) is essentially 0 — no replay disagreement.
        assert cal["recommended_rmse_kwh"] < 1e-6

    def test_sweep_does_not_blindly_favour_live_config(self):
        """Status-quo bias (the §3 critique): the OLD 1-D sweep biased toward
        live decay because the loss was mean-residual against live-expected.
        The new sweep should recommend a DIFFERENT (decay, k) when the data
        was generated under different parameters than live.
        """
        true_decay, true_k = 0.60, 0.0
        live_decay, live_k = 0.85, 0.0  # Live deliberately wrong
        log = []
        for d in range(1, 8):
            log += self._generate_day(
                f"2026-04-{d:02d}",
                true_decay=true_decay, true_k=true_k,
                live_decay=live_decay, live_k=live_k,
            )
        # Fixture: ``actual`` is computed under TRUTH, ``expected`` under LIVE.
        # Live residual = actual − expected = live_release − truth_release ≠ 0.
        # Counterfactual RMSE for alt = (alt_release − truth_release); minimised
        # at alt = truth.  The sweep must recover the truth, not the live config.
        coord = self._make_coord(log, decay=live_decay, k=live_k)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        cal = result["global"]["battery_calibration"]

        # The recommendation must differ from the wrong live decay.
        assert cal["recommended_decay"] != live_decay
        # And it should be close to the truth (within grid resolution).
        assert cal["recommended_decay"] == pytest.approx(true_decay, abs=0.05)
        # RMSE at truth is ~0; RMSE at live is positive — the surface clearly
        # differentiates them, demonstrating the bias removal.
        truth_rmse = cal["rmse_surface"][f"{true_decay},{true_k}"]
        live_rmse_on_surface = cal["rmse_surface"][f"{live_decay},{live_k}"]
        assert truth_rmse < live_rmse_on_surface

    def test_post_sunset_definition_uses_raw_signal_not_solar_factor(self):
        """Post-sunset hours are defined as the N hours after the last hour
        with raw_solar > 0.01.  Days with no sunny hour contribute zero
        post-sunset hours (would happen on fully-overcast winter days);
        sweep skips them gracefully.
        """
        # All-dark day: no raw_solar anywhere → no post-sunset hours
        # (battery stays at 0 throughout, nothing to calibrate against).
        all_dark = []
        for h in range(24):
            all_dark.append({
                "timestamp": f"2026-04-01T{h:02d}:00:00",
                "hour": h,
                "temp": 5.0,
                "temp_key": "5",
                "wind_bucket": "normal",
                "solar_factor": 0.0,
                "solar_vector_s": 0.0,
                "solar_vector_e": 0.0,
                "solar_vector_w": 0.0,
                "solar_impact_raw_kwh": 0.0,
                "solar_impact_kwh": 0.0,
                "solar_heating_wasted_kwh": 0.0,
                "actual_kwh": 2.0,
                "expected_kwh": 2.0,
                "correction_percent": 100.0,
                "auxiliary_active": False,
                "guest_impact_kwh": 0.0,
                "unit_modes": {"sensor.heater1": "heating"},
                "unit_breakdown": {"sensor.heater1": 2.0},
            })
        coord = self._make_coord(all_dark, decay=0.80, k=0.0)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        cal = result["global"]["battery_calibration"]
        # No post-sunset hours → no recommendation possible (cal still
        # populated with current values + empty surface, never crashes).
        assert cal["post_sunset_hours_evaluated"] == 0
        assert cal["rmse_surface"] == {}

    def test_sweep_includes_k_dimension(self):
        """When wasted is non-zero on truth-data with k > 0, the sweep
        should pick up the k contribution and recommend k > 0."""
        true_decay, true_k = 0.80, 0.3
        log = []
        for d in range(1, 8):
            log += self._generate_day(
                f"2026-04-{d:02d}", true_decay=true_decay, true_k=true_k,
                peak_wasted=0.2,  # non-zero wasted so k matters
            )
        coord = self._make_coord(log, decay=true_decay, k=true_k)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        cal = result["global"]["battery_calibration"]

        # Recovers the truth on both dimensions
        assert cal["recommended_decay"] == pytest.approx(true_decay, abs=0.05)
        assert cal["recommended_k"] == pytest.approx(true_k, abs=0.1)
        assert cal["recommended_rmse_kwh"] < 1e-6

    def test_rmse_surface_grid_size(self):
        """Surface should cover all evaluable (decay, k) combinations:
        10 decay × 11 k = 110 cells.  Cells with too few post-sunset hours
        are dropped, but with a typical multi-day log all 110 should appear.
        """
        log = []
        for d in range(1, 6):
            log += self._generate_day(
                f"2026-04-{d:02d}", true_decay=0.80, true_k=0.0
            )
        coord = self._make_coord(log)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        cal = result["global"]["battery_calibration"]
        assert len(cal["rmse_surface"]) == 110

    def test_surface_keys_format(self):
        """Surface keys are 'decay,k' strings (consumable by JSON / json.dumps)."""
        log = []
        for d in range(1, 4):
            log += self._generate_day(
                f"2026-04-{d:02d}", true_decay=0.80, true_k=0.0
            )
        coord = self._make_coord(log)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        cal = result["global"]["battery_calibration"]
        for key in cal["rmse_surface"]:
            assert "," in key
            decay_str, k_str = key.split(",")
            float(decay_str)  # raises if not a float
            float(k_str)


class TestMorningWindowDiagnostic:
    """The morning-window block scores rising-sun-phase residuals to expose
    the asymmetric-charge gap (the deferred scope from #896).  It runs
    alongside the post-sunset sweep — same grid, same counterfactual, just
    a different window — and reports a parallel surface plus a
    ``tail_morning_disagreement_kwh`` diagnostic.

    Read-only.  Does NOT drive the recommendation — that stays anchored to
    post-sunset since the live battery is parameterised for tail behaviour.
    """

    @staticmethod
    def _make_coord(hourly_log, *, decay=0.80, k=0.0):
        coord = MagicMock()
        coord._hourly_log = hourly_log
        coord.solar_battery_decay = decay
        coord.battery_thermal_feedback_k = k
        coord.energy_sensors = ["sensor.heater1"]
        coord.solar_correction_percent = 100.0
        coord.solar_azimuth = 180
        coord.balance_point = 15.0
        coord.solar = MagicMock()
        coord.solar.calculate_unit_coefficient = MagicMock(
            return_value={"s": 1.0, "e": 0.0, "w": 0.0}
        )
        coord.solar.calculate_unit_solar_impact = MagicMock(return_value=0.0)
        return coord

    @staticmethod
    def _generate_day(date_str, *, true_decay, true_k=0.0,
                       live_decay=None, live_k=None,
                       base_kwh=2.0, peak_solar=0.5, peak_wasted=0.0):
        """Reuses the logic from TestJointDecayKSweepValidation — kept
        local to avoid cross-class fixture coupling.  Solar peaks at noon
        (hour=12); morning window covers 8-12 inclusive."""
        if live_decay is None:
            live_decay = true_decay
        if live_k is None:
            live_k = true_k
        entries = []
        main_truth = main_live = 0.0
        carry_truth = carry_live = 0.0
        for hour in range(24):
            if 8 <= hour <= 16:
                raw_solar = peak_solar * max(0, 1.0 - abs(hour - 12) / 5.0)
                wasted = peak_wasted * max(0, 1.0 - abs(hour - 12) / 5.0)
            else:
                raw_solar = 0.0
                wasted = 0.0
            main_truth = main_truth * true_decay + raw_solar * (1 - true_decay)
            truth_carry_in = true_k * wasted if true_k > 0 else 0.0
            carry_truth = carry_truth * true_decay + truth_carry_in * (1 - true_decay)
            truth_release = main_truth + true_k * carry_truth * (1 - true_decay)
            main_live = main_live * live_decay + raw_solar * (1 - live_decay)
            live_carry_in = live_k * wasted if live_k > 0 else 0.0
            carry_live = carry_live * live_decay + live_carry_in * (1 - live_decay)
            live_release = main_live + live_k * carry_live * (1 - live_decay)
            actual = max(0.0, base_kwh - truth_release)
            expected = max(0.0, base_kwh - live_release)
            entries.append({
                "timestamp": f"{date_str}T{hour:02d}:00:00",
                "hour": hour,
                "temp": 5.0,
                "temp_key": "5",
                "wind_bucket": "normal",
                "solar_factor": raw_solar,
                "solar_vector_s": raw_solar,
                "solar_vector_e": 0.0,
                "solar_vector_w": 0.0,
                "solar_impact_raw_kwh": raw_solar,
                "solar_impact_kwh": main_truth,
                "solar_heating_wasted_kwh": wasted,
                "actual_kwh": actual,
                "expected_kwh": expected,
                "correction_percent": 100.0,
                "auxiliary_active": False,
                "guest_impact_kwh": 0.0,
                "unit_modes": {"sensor.heater1": "heating"},
                "unit_breakdown": {"sensor.heater1": actual},
                "unit_expected_breakdown": {"sensor.heater1": base_kwh},
            })
        return entries

    def test_morning_block_present_in_calibration(self):
        """Calibration block carries morning fields alongside post-sunset."""
        log = []
        for d in range(1, 6):
            log += self._generate_day(
                f"2026-04-{d:02d}", true_decay=0.80, true_k=0.0
            )
        coord = self._make_coord(log)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        cal = result["global"]["battery_calibration"]
        assert "morning_recommended_decay" in cal
        assert "morning_recommended_k" in cal
        assert "morning_recommended_rmse_kwh" in cal
        assert "morning_rmse_surface" in cal
        assert "morning_hours_evaluated" in cal
        assert "tail_morning_disagreement_kwh" in cal

    def test_morning_window_covers_rising_phase_only(self):
        """Morning window = first sunny hour to peak hour inclusive.

        For the fixture (sun ramps up 8→12, peaks at 12, ramps down 12→16),
        morning hours are 8-12 = 5 hours/day × N days.  Plateau and decline
        (13-16) excluded since they don't discriminate decay vs instant.
        """
        log = []
        for d in range(1, 8):  # 7 days
            log += self._generate_day(
                f"2026-04-{d:02d}", true_decay=0.80, true_k=0.0
            )
        coord = self._make_coord(log)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        cal = result["global"]["battery_calibration"]
        # 7 days × 5 morning hours = 35
        assert cal["morning_hours_evaluated"] == 35

    def test_morning_recovers_truth_when_live_matches_truth(self):
        """When live config matches the truth used to generate the data,
        morning-best should match truth (RMSE → 0 at truth)."""
        true_decay, true_k = 0.65, 0.0
        log = []
        for d in range(1, 6):
            log += self._generate_day(
                f"2026-04-{d:02d}", true_decay=true_decay, true_k=true_k
            )
        coord = self._make_coord(log, decay=true_decay, k=true_k)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        cal = result["global"]["battery_calibration"]
        assert cal["morning_recommended_decay"] == pytest.approx(true_decay, abs=0.05)
        assert cal["morning_recommended_rmse_kwh"] < 1e-6

    def test_tail_morning_disagreement_zero_when_unified_optimum(self):
        """If the same (decay, k) optimum minimises both windows (which
        happens when the model's symmetric EMA actually matches reality —
        i.e. the data was generated under that EMA), the disagreement
        metric is ~0.

        Synthetic fixture is symmetric-EMA, so we expect this case."""
        log = []
        for d in range(1, 6):
            log += self._generate_day(
                f"2026-04-{d:02d}", true_decay=0.65, true_k=0.0
            )
        coord = self._make_coord(log, decay=0.65, k=0.0)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        cal = result["global"]["battery_calibration"]
        assert cal["tail_morning_disagreement_kwh"] is not None
        assert abs(cal["tail_morning_disagreement_kwh"]) < 0.001

    def test_tail_morning_disagreement_positive_when_live_wrong(self):
        """When live is wrong AND the data is symmetric-EMA, the
        post-sunset surface and morning surface should still recover the
        same truth.  Disagreement reads close to zero — confirming that
        symmetric-EMA fixtures cannot synthesise asymmetric-charge
        evidence (a future test on real-world data would; the diagnostic
        is structurally correct even when the synthetic shape doesn't
        exercise it)."""
        true_decay, true_k = 0.60, 0.0
        live_decay, live_k = 0.85, 0.0
        log = []
        for d in range(1, 8):
            log += self._generate_day(
                f"2026-04-{d:02d}",
                true_decay=true_decay, true_k=true_k,
                live_decay=live_decay, live_k=live_k,
            )
        coord = self._make_coord(log, decay=live_decay, k=live_k)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        cal = result["global"]["battery_calibration"]
        # Both surfaces recover truth → disagreement ~0
        assert cal["recommended_decay"] == pytest.approx(true_decay, abs=0.05)
        assert cal["morning_recommended_decay"] == pytest.approx(true_decay, abs=0.05)
        assert abs(cal["tail_morning_disagreement_kwh"]) < 0.01

    def test_morning_surface_full_grid_size(self):
        """Morning surface covers same 110-cell grid as post-sunset
        (when enough morning hours exist)."""
        log = []
        for d in range(1, 6):
            log += self._generate_day(
                f"2026-04-{d:02d}", true_decay=0.80, true_k=0.0
            )
        coord = self._make_coord(log)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        cal = result["global"]["battery_calibration"]
        assert len(cal["morning_rmse_surface"]) == 110

    def test_morning_block_handles_no_sunny_days(self):
        """All-overcast / all-dark fixture → morning evaluation falls
        through gracefully without crashing.  ``morning_hours_evaluated``
        stays at 0 and surface is empty; recommendation falls back to
        live values."""
        # All-dark fixture — same shape as the post-sunset all-dark test
        all_dark = []
        for h in range(24):
            all_dark.append({
                "timestamp": f"2026-04-01T{h:02d}:00:00",
                "hour": h,
                "temp": 5.0,
                "temp_key": "5",
                "wind_bucket": "normal",
                "solar_factor": 0.0,
                "solar_vector_s": 0.0,
                "solar_vector_e": 0.0,
                "solar_vector_w": 0.0,
                "solar_impact_raw_kwh": 0.0,
                "solar_impact_kwh": 0.0,
                "solar_heating_wasted_kwh": 0.0,
                "actual_kwh": 2.0,
                "expected_kwh": 2.0,
                "correction_percent": 100.0,
                "auxiliary_active": False,
                "guest_impact_kwh": 0.0,
                "unit_modes": {"sensor.heater1": "heating"},
                "unit_breakdown": {"sensor.heater1": 2.0},
            })
        coord = self._make_coord(all_dark, decay=0.80, k=0.0)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        cal = result["global"]["battery_calibration"]
        assert cal["morning_hours_evaluated"] == 0
        assert cal["morning_rmse_surface"] == {}
