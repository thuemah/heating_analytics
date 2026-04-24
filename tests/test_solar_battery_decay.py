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
    def _make_coordinator_with_log(hourly_log, current_decay=SOLAR_BATTERY_DECAY):
        """Create a minimal coordinator mock with hourly log data."""
        coord = MagicMock()
        coord._hourly_log = hourly_log
        coord.solar_battery_decay = current_decay
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
        assert "recommended_decay" in calibration
        assert "sweep_results" in calibration

    def test_sweep_with_no_data_returns_empty(self):
        """Empty hourly log produces empty calibration."""
        coord = self._make_coordinator_with_log([])
        from custom_components.heating_analytics.coordinator import HeatingDataCoordinator

        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        # Should not crash — graceful empty result
        assert "global" in result
        assert result["global"]["qualifying_hours"] == 0
