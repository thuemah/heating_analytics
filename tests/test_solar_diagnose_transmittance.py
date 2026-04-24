"""Tests for #826 diagnostic additions in ``diagnose_solar``.

Covers:
- ``context`` block (lat/lon, screen_config, constants)
- ``screen_stratified`` per correction-bucket delta and bias-gap flag
- ``transmittance_sensitivity`` sweep over candidate SCREEN_DIRECT_TRANSMITTANCE

These sections exist so the dev can read 3 Antwerpen houses' diagnose output
and decide whether the literature-derived 0.08 / 0.30 values match reality,
without guessing from the user's own (solar-poor) installation.
"""
from unittest.mock import MagicMock

import pytest

from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.const import (
    SCREEN_DIRECT_TRANSMITTANCE,
    DEFAULT_SOLAR_MIN_TRANSMITTANCE,
    SOLAR_BATTERY_DECAY,
)
from custom_components.heating_analytics.diagnostics import DiagnosticsEngine


def _make_coord(hourly_log, screen_config=(True, True, True), latitude=51.2,
                longitude=4.4, learned_coeff=None):
    """Minimal coordinator mock for calling diagnose_solar as an unbound method."""
    coord = MagicMock()
    coord._hourly_log = hourly_log
    coord.solar_battery_decay = SOLAR_BATTERY_DECAY
    coord.energy_sensors = ["sensor.heater1"]
    coord.solar_correction_percent = 100.0
    coord.solar_azimuth = 180
    coord.balance_point = 15.0
    coord.screen_config = screen_config
    coord.hass.config.latitude = latitude
    coord.hass.config.longitude = longitude

    learned = learned_coeff or {"s": 1.0, "e": 0.0, "w": 0.0}
    coord.solar = MagicMock()
    coord.solar.calculate_unit_coefficient = MagicMock(return_value=learned)

    # Real impact: coeff · potential (matches production path)
    def _impact(potential, coeff, screen_transmittance=1.0):
        return max(
            0.0,
            (coeff["s"] * potential[0]
             + coeff["e"] * potential[1]
             + coeff["w"] * potential[2]) * screen_transmittance,
        )

    coord.solar.calculate_unit_solar_impact = MagicMock(side_effect=_impact)
    return coord


def _hour_entry(ts, *, solar_s=0.5, solar_e=0.0, solar_w=0.0, correction=100.0,
                actual=1.5, base=2.0, solar_factor=None, hour=12, temp=10.0,
                mode="heating"):
    """Build one hourly-log entry with all fields diagnose_solar reads."""
    return {
        "timestamp": ts,
        "hour": hour,
        "temp": temp,
        "inertia_temp": temp,
        "temp_key": str(int(round(temp))),
        "wind_bucket": "normal",
        "solar_factor": solar_factor if solar_factor is not None else (solar_s + solar_e + solar_w),
        "solar_vector_s": solar_s,
        "solar_vector_e": solar_e,
        "solar_vector_w": solar_w,
        "solar_impact_raw_kwh": 0.0,
        "solar_impact_kwh": 0.0,
        "actual_kwh": actual,
        "expected_kwh": base,
        "correction_percent": correction,
        "auxiliary_active": False,
        "guest_impact_kwh": 0.0,
        "unit_modes": {"sensor.heater1": mode},
        "unit_breakdown": {"sensor.heater1": actual},
        "unit_expected_breakdown": {"sensor.heater1": base},
        "solar_dominant_entities": [],
    }


class TestTemperatureRegimeStratification:
    """BP-relative temperature buckets surface COP-regime bias.

    Layout with balance_point = 15:
      heating_deep: T < 7    (BP - 8)
      heating_mild: 7 ≤ T < 13   (BP - 2 boundary)
      cooling:      T > 17   (BP + 2)
      ±2°C around BP is dropped (mode flips hour-to-hour there).
    """

    @staticmethod
    def _log_across_regimes(bp=15.0, n_per_bucket=12, delta_cooling=0.0,
                             delta_deep=0.0, delta_mild=0.0):
        """Build a log with qualifying hours in each regime.

        Temperatures are chosen well inside each bucket so the ±2°C
        transition zone is not touched accidentally.  ``delta_*`` scales the
        implied-solar bias for each bucket independently (kWh added to
        base-actual); this lets tests verify that bias in one regime
        surfaces in the corresponding bucket without contaminating others.
        """
        entries = []
        from datetime import datetime, timedelta
        base_dt = datetime(2026, 4, 1, 12, 0)
        regimes = [
            # (temp, mode, delta_bias, bucket_label)
            (bp - 12.0, "heating", delta_deep, "heating_deep"),
            (bp - 5.0,  "heating", delta_mild, "heating_mild"),
            (bp + 5.0,  "cooling", delta_cooling, "cooling"),
        ]
        hour_offset = 0
        for t, mode, bias, _label in regimes:
            for i in range(n_per_bucket):
                ts = (base_dt + timedelta(hours=hour_offset)).isoformat()
                hour_offset += 1
                # Calibrate to zero baseline delta: modeled = coeff·potential
                # = 1.0 × 0.5 = 0.5, so pick actual such that implied = 0.5
                # at bias=0.  Bias>0 then shifts implied up by exactly bias,
                # producing mean_delta = -bias in the affected bucket.
                if mode == "cooling":
                    # implied = actual - base; want implied = 0.5 + bias
                    actual = 2.5 + bias
                    base = 2.0
                else:
                    # implied = base - actual; want implied = 0.5 + bias
                    actual = 1.5 - bias
                    base = 2.0
                entries.append(_hour_entry(
                    ts, solar_s=0.5,
                    solar_e=0.15 if i % 2 == 0 else 0.05,
                    correction=100.0, actual=actual, base=base,
                    temp=t, mode=mode,
                ))
        return entries

    def test_all_three_buckets_populated_with_balanced_regimes(self):
        coord = _make_coord(self._log_across_regimes(), )
        coord.balance_point = 15.0
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)

        temp_strat = result["per_unit"]["sensor.heater1"]["temperature_stratified"]
        assert temp_strat["heating_deep"]["n"] == 12
        assert temp_strat["heating_mild"]["n"] == 12
        assert temp_strat["cooling"]["n"] == 12

    def test_transition_zone_dropped(self):
        """Hours within BP±2 are excluded from all three buckets."""
        entries = []
        from datetime import datetime, timedelta
        base_dt = datetime(2026, 4, 1, 12, 0)
        # 15 hours all inside the transition zone (BP-2 ≤ T ≤ BP+2)
        for i in range(15):
            ts = (base_dt + timedelta(hours=i)).isoformat()
            entries.append(_hour_entry(
                ts, solar_s=0.5, solar_e=0.15, correction=100.0,
                actual=1.7, base=2.0, temp=14.0, mode="heating",  # BP-1
            ))
        coord = _make_coord(entries)
        coord.balance_point = 15.0
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)

        temp_strat = result["per_unit"]["sensor.heater1"]["temperature_stratified"]
        # All hours consumed by qualifying logic, none ended up in a bucket
        assert temp_strat["heating_deep"]["n"] == 0
        assert temp_strat["heating_mild"]["n"] == 0
        assert temp_strat["cooling"]["n"] == 0

    def test_deep_bias_isolated_to_deep_bucket(self):
        """A bias in heating_deep must not leak into mild or cooling."""
        coord = _make_coord(self._log_across_regimes(delta_deep=0.3))
        coord.balance_point = 15.0
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)

        temp_strat = result["per_unit"]["sensor.heater1"]["temperature_stratified"]
        # delta_deep=0.3 shifted actual DOWN, so implied_solar rose by 0.3
        # (heating: implied = base - actual).  modeled stays at
        # coeff×potential.  So mean_delta = modeled - implied shifts NEGATIVE
        # by 0.3 in deep bucket only.
        assert temp_strat["heating_deep"]["mean_delta_kwh"] == pytest.approx(-0.3, abs=0.05)
        assert abs(temp_strat["heating_mild"]["mean_delta_kwh"]) < 0.05
        assert abs(temp_strat["cooling"]["mean_delta_kwh"]) < 0.05

    def test_cooling_bias_isolated_to_cooling_bucket(self):
        """Cooling-mode bias shows only in the cooling bucket."""
        coord = _make_coord(self._log_across_regimes(delta_cooling=0.3))
        coord.balance_point = 15.0
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)

        temp_strat = result["per_unit"]["sensor.heater1"]["temperature_stratified"]
        # Cooling: implied = actual - base, so delta_bias=0.3 raises implied by 0.3,
        # and mean_delta = modeled - implied shifts NEGATIVE.
        assert temp_strat["cooling"]["mean_delta_kwh"] == pytest.approx(-0.3, abs=0.05)
        assert abs(temp_strat["heating_deep"]["mean_delta_kwh"]) < 0.05
        assert abs(temp_strat["heating_mild"]["mean_delta_kwh"]) < 0.05

    def test_missing_temperature_skips_stratification(self):
        """Entries without temp data are excluded from temperature buckets only."""
        entries = []
        from datetime import datetime, timedelta
        base_dt = datetime(2026, 4, 1, 12, 0)
        for i in range(15):
            ts = (base_dt + timedelta(hours=i)).isoformat()
            e = _hour_entry(
                ts, solar_s=0.5, solar_e=0.15,
                correction=100.0, actual=1.7, base=2.0,
            )
            e["temp"] = None
            e["inertia_temp"] = None
            entries.append(e)
        coord = _make_coord(entries)
        coord.balance_point = 15.0
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)

        temp_strat = result["per_unit"]["sensor.heater1"]["temperature_stratified"]
        assert temp_strat["heating_deep"]["n"] == 0
        assert temp_strat["heating_mild"]["n"] == 0
        assert temp_strat["cooling"]["n"] == 0
        # Other sections still work
        assert result["per_unit"]["sensor.heater1"]["qualifying_hours"] == 15


class TestContextBlock:
    """Context block identifies installation for remote analysis."""

    def test_context_present_and_populated(self):
        coord = _make_coord([], screen_config=(True, False, True),
                             latitude=51.2167, longitude=4.4)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)

        assert "context" in result
        ctx = result["context"]
        assert ctx["latitude"] == pytest.approx(51.2167)
        assert ctx["longitude"] == pytest.approx(4.4)
        assert ctx["screen_config"] == {"south": True, "east": False, "west": True}
        assert ctx["constants"]["screen_direct_transmittance"] == SCREEN_DIRECT_TRANSMITTANCE
        assert ctx["constants"]["composite_legacy_floor"] == DEFAULT_SOLAR_MIN_TRANSMITTANCE
        assert ctx["days_analyzed"] == 30


class TestScreenStratification:
    """correction-bucket split surfaces systematic bias at closed screens."""

    @staticmethod
    def _log_with_screen_mix(n_open, n_closed, closed_actual_bias=0.0):
        """Build a log with n hours at correction=100 and n hours at correction=10.

        ``closed_actual_bias`` adds extra implied-solar at closed-screen hours,
        simulating a mismatch between the transmittance model and reality.
        """
        entries = []
        from datetime import datetime, timedelta
        base_dt = datetime(2026, 4, 1, 12, 0)
        # Small E component (non-constant) so normal equations don't degenerate
        # if a later test reads implied_coefficient_30d from this helper's
        # output.  Bucket split is driven by correction_percent regardless.
        for i in range(n_open):
            ts = (base_dt + timedelta(hours=i)).isoformat()
            entries.append(_hour_entry(
                ts, solar_s=0.6, solar_e=0.15 if i % 2 == 0 else 0.05,
                correction=100.0, actual=1.5, base=2.0,
            ))
        for i in range(n_closed):
            ts = (base_dt + timedelta(hours=n_open + i)).isoformat()
            # Correction=0 → transmittance=SCREEN_DIRECT_TRANSMITTANCE for
            # each screened direction → potential reconstructs cleanly.
            eff_s = 0.6 * SCREEN_DIRECT_TRANSMITTANCE
            eff_e = (0.15 if i % 2 == 0 else 0.05) * SCREEN_DIRECT_TRANSMITTANCE
            entries.append(_hour_entry(
                ts, solar_s=eff_s, solar_e=eff_e, correction=0.0,
                actual=1.5 - closed_actual_bias,
                base=2.0,
            ))
        return entries

    def test_stratified_buckets_report_separate_means(self):
        log = self._log_with_screen_mix(n_open=15, n_closed=15, closed_actual_bias=0.0)
        coord = _make_coord(log)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)

        strat = result["per_unit"]["sensor.heater1"]["screen_stratified"]
        assert strat["open"]["n"] == 15
        assert strat["closed"]["n"] == 15
        # With zero synthetic bias and the coefficient matching the data,
        # bias_gap should be small.
        assert abs(strat["bias_gap_kwh"]) < 0.1

    def test_too_low_floor_flagged_when_closed_over_predicts(self):
        """Bias implied-solar DOWN at closed screens → model over-predicts → floor too LOW.

        Positive bias_gap (closed delta > open delta) means the model over-
        predicts when screens are deployed.  The cause is that
        ``reconstructed_potential = effective / transmittance_model`` is
        inflated because the assumed transmittance is LOWER than reality
        (too much sun is pouring through the screens in practice compared
        to our 0.08 floor assumption).  Fix is to RAISE the transmittance
        floor.  Flag name was swapped prior to the 1.3.3 fix.
        """
        log = self._log_with_screen_mix(
            n_open=15, n_closed=15, closed_actual_bias=-0.4
        )
        coord = _make_coord(log)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)

        unit = result["per_unit"]["sensor.heater1"]
        assert unit["screen_stratified"]["bias_gap_kwh"] > 0.05
        assert "transmittance_floor_too_low" in unit["flags"]
        assert "transmittance_floor_too_high" not in unit["flags"]

    def test_too_high_floor_flagged_when_closed_under_predicts(self):
        """Bias implied-solar UP at closed screens → model under-predicts → floor too HIGH.

        Negative bias_gap (closed delta < open delta) is the symmetric case:
        the transmittance floor is too high, reconstructed potential is
        compressed, and the model under-predicts on closed hours.
        """
        log = self._log_with_screen_mix(
            n_open=15, n_closed=15, closed_actual_bias=+0.4
        )
        coord = _make_coord(log)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)

        unit = result["per_unit"]["sensor.heater1"]
        assert unit["screen_stratified"]["bias_gap_kwh"] < -0.05
        assert "transmittance_floor_too_high" in unit["flags"]
        assert "transmittance_floor_too_low" not in unit["flags"]

    def test_no_gap_flag_without_both_extremes(self):
        """Fewer than 10 hours in either extreme → no bias_gap reported."""
        log = self._log_with_screen_mix(n_open=5, n_closed=15)
        coord = _make_coord(log)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)

        unit = result["per_unit"]["sensor.heater1"]
        assert "bias_gap_kwh" not in unit["screen_stratified"]
        assert "transmittance_floor_too_high" not in unit["flags"]
        assert "transmittance_floor_too_low" not in unit["flags"]


class TestTransmittanceSensitivity:
    """Sweep over SCREEN_DIRECT_TRANSMITTANCE candidates."""

    @staticmethod
    def _log_with_known_transmittance(true_t, n=30, true_coeff=1.0):
        """Generate hours where the true floor is ``true_t``.

        For each hour the effective vector is potential × transmittance(true_t, correction),
        implied_solar is coeff · potential.  Sweeping candidates should
        minimise RMSE at the true value.

        Includes a small east component so the 3×3 normal equations have
        enough dimensional variance to resolve (real installations always
        have morning/afternoon sun bringing E/W components non-zero; a
        pure-south synthetic test would hit the 2D-solve degeneracy path).
        """
        entries = []
        from datetime import datetime, timedelta
        base_dt = datetime(2026, 4, 1, 12, 0)
        for i in range(n):
            # Vary correction across the full range for informative sweep
            correction = (i * 100 / n) if n > 1 else 50.0
            pct = correction / 100.0
            t_screened = true_t + (1.0 - true_t) * pct
            potential_s = 0.5
            # Rotate a small E component through the day so (S, E) has rank 2
            potential_e = 0.15 if i % 2 == 0 else 0.05
            eff_s = potential_s * t_screened
            eff_e = potential_e * t_screened
            implied = true_coeff * potential_s  # only S contributes to implied
            ts = (base_dt + timedelta(hours=i)).isoformat()
            entries.append(_hour_entry(
                ts, solar_s=eff_s, solar_e=eff_e, correction=correction,
                actual=2.0 - implied, base=2.0,
            ))
        return entries

    def test_sweep_reports_informative_when_correction_varies(self):
        log = self._log_with_known_transmittance(true_t=0.08, n=30)
        coord = _make_coord(log)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)

        sens = result["per_unit"]["sensor.heater1"]["transmittance_sensitivity"]
        assert sens is not None
        assert sens["informative"] is True
        assert sens["correction_range_pct"] >= 40.0
        assert sens["n_hours"] == 30
        assert len(sens["candidates"]) > 0

    def test_sweep_identifies_true_transmittance(self):
        """Sweep's best should match the true generative transmittance."""
        log = self._log_with_known_transmittance(true_t=0.08, n=40)
        coord = _make_coord(log)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)

        sens = result["per_unit"]["sensor.heater1"]["transmittance_sensitivity"]
        assert sens["best"]["screen_direct_transmittance"] == pytest.approx(0.08)

    def test_sweep_identifies_different_true_value(self):
        """If true floor is 0.15, sweep picks 0.15 (not 0.08)."""
        log = self._log_with_known_transmittance(true_t=0.15, n=40)
        coord = _make_coord(log)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)

        sens = result["per_unit"]["sensor.heater1"]["transmittance_sensitivity"]
        assert sens["best"]["screen_direct_transmittance"] == pytest.approx(0.15)
        unit = result["per_unit"]["sensor.heater1"]
        # Deviates from current 0.08 by > 0.04 → retune suggestion
        assert "sensitivity_suggests_transmittance_retune" in unit["flags"]

    def test_sweep_not_informative_when_correction_constant(self):
        """If slider never moves, sweep cannot distinguish transmittances."""
        entries = []
        from datetime import datetime, timedelta
        base_dt = datetime(2026, 4, 1, 12, 0)
        # All 25 hours at correction=100 → transmittance effect identical.
        # Vary E slightly so normal equations don't degenerate.
        for i in range(25):
            ts = (base_dt + timedelta(hours=i)).isoformat()
            entries.append(_hour_entry(
                ts, solar_s=0.5, solar_e=0.15 if i % 2 == 0 else 0.05,
                correction=100.0, actual=1.5, base=2.0,
            ))
        coord = _make_coord(entries)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)

        sens = result["per_unit"]["sensor.heater1"]["transmittance_sensitivity"]
        assert sens is not None
        assert sens["informative"] is False

    def test_sweep_omitted_when_too_few_hours(self):
        """Under 20 usable hours → no sensitivity section."""
        log = self._log_with_known_transmittance(true_t=0.08, n=10)
        coord = _make_coord(log)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)

        sens = result["per_unit"]["sensor.heater1"]["transmittance_sensitivity"]
        assert sens is None
