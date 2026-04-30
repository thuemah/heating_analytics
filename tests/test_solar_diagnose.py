"""Tests for diagnose_solar diagnostic output.

Two sections:

- **Transmittance / context / screen-stratification.** Validates the
  ``context`` block (lat/lon, screen_config, constants), the
  ``screen_stratified`` per correction-bucket delta + bias-gap flag,
  and the ``transmittance_sensitivity`` sweep over candidate
  ``SCREEN_DIRECT_TRANSMITTANCE`` values.

- **Mode-stratified coefficient split.** Validates the per-unit
  fields ``current_coefficient_heating``, ``current_coefficient_cooling``,
  and ``coefficient_split_delta_pct`` so the user/dev can observe whether
  the heating/cooling regimes converge to the same coefficient (split is
  theatrical) or diverge (split captures real COP separation).
"""
from unittest.mock import MagicMock

import pytest

from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.const import (
    SCREEN_DIRECT_TRANSMITTANCE,
    DEFAULT_SOLAR_MIN_TRANSMITTANCE,
    SOLAR_BATTERY_DECAY,
    DEFAULT_SOLAR_COEFF_HEATING,
    DEFAULT_SOLAR_COEFF_COOLING,
    MODE_HEATING,
    MODE_COOLING,
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
    # Per-entity routing helper: default all entities to the installation
    # screen_config (pre-`screen_affected_entities` behaviour).  Tests that
    # want an entity-specific config can override this attribute.
    coord.screen_config_for_entity = MagicMock(side_effect=lambda _eid: screen_config)
    coord.hass.config.latitude = latitude
    coord.hass.config.longitude = longitude

    learned = learned_coeff or {"s": 1.0, "e": 0.0, "w": 0.0}
    coord.solar = MagicMock()
    coord.solar.calculate_unit_coefficient = MagicMock(return_value=learned)

    # Mirror live storage so the inactive-unit collapse (#896 follow-up)
    # treats this fixture as an *active* sensor — without this, every
    # test that asserts on `transmittance_sensitivity` /
    # `temperature_stratified` / other verbose blocks fails because the
    # diagnostic emits a minimal record for inactive units (zero stored
    # coefficient → inactive).  The model proxy mirrors the underscored
    # attribute the live coordinator exposes via `model.solar_coefficients_per_unit`.
    learned_storage = {"sensor.heater1": {"heating": dict(learned), "cooling": {"s": 0.0, "e": 0.0, "w": 0.0}}}
    coord._solar_coefficients_per_unit = learned_storage
    coord.model.solar_coefficients_per_unit = learned_storage

    # Real impact: coeff · potential (matches production path — invariant #1,
    # no extra transmittance factor; coefficient absorbs avg_transmittance
    # via the NLMS learning target).
    def _impact(potential, coeff):
        return max(
            0.0,
            coeff["s"] * potential[0]
            + coeff["e"] * potential[1]
            + coeff["w"] * potential[2],
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


# =============================================================================
# Mode-stratified coefficient split — per-unit fields
# current_coefficient_heating / current_coefficient_cooling /
# coefficient_split_delta_pct so the user can observe whether the heating
# and cooling regimes converged to the same physical coefficient.
# =============================================================================

def _hour_entry_modesplit(ts, *, solar_s=0.5, solar_e=0.0, solar_w=0.0, correction=100.0,
                actual=1.5, base=2.0, hour=12, temp=10.0, mode="heating"):
    """Minimal hourly-log entry that diagnose_solar will accumulate."""
    return {
        "timestamp": ts,
        "hour": hour,
        "temp": temp,
        "inertia_temp": temp,
        "temp_key": str(int(round(temp))),
        "wind_bucket": "normal",
        "solar_factor": solar_s + solar_e + solar_w,
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


def _make_coord_modesplit(stratified_coeff: dict | None = None, *, hourly_log=None):
    """Coordinator mock that returns regime-specific coefficients via
    ``solar.calculate_unit_coefficient(entity, temp_key, mode)``."""
    coord = MagicMock()
    if hourly_log is None:
        # Default: 30 sunny heating-mode entries so the entity gets
        # accumulated into ``unit_accum`` and surfaces in per_unit output.
        hourly_log = [
            _hour_entry_modesplit(f"2026-04-{10 + i // 24:02d}T{i % 24:02d}:00:00")
            for i in range(30)
        ]
    coord._hourly_log = hourly_log
    coord.solar_battery_decay = SOLAR_BATTERY_DECAY
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

    # Raw storage drives the split-delta computation in diagnose_solar
    # (#868 + post-merge fix).  Empty regime → {0,0,0}, NOT a default
    # decomposition — otherwise the validation criterion would be muddled
    # by every heating-only install reporting a small spurious split.
    if stratified_coeff is None:
        raw_solar = {}
    else:
        raw_solar = {"sensor.heater1": stratified_coeff}
    coord._solar_coefficients_per_unit = raw_solar
    # The ``model`` proxy in the real coordinator returns live references
    # to the underscored attrs.  In this MagicMock-based fixture we mirror
    # that explicitly so the diagnose_solar code path that reads
    # ``model.solar_coefficients_per_unit`` sees the same dict.
    coord.model.solar_coefficients_per_unit = raw_solar
    coord.model.correlation_data_per_unit = coord._correlation_data_per_unit

    # Mode-aware mock: heating mode returns heating regime,
    # cooling mode returns cooling regime, defaults otherwise.
    def _coeff(entity_id, temp_key, mode):
        if stratified_coeff is None:
            return {"s": 0.0, "e": 0.0, "w": 0.0}
        regime = "cooling" if mode == MODE_COOLING else "heating"
        return stratified_coeff.get(regime, {"s": 0.0, "e": 0.0, "w": 0.0})

    coord.solar = MagicMock()
    coord.solar.calculate_unit_coefficient = MagicMock(side_effect=_coeff)
    coord.solar.calculate_unit_solar_impact = MagicMock(return_value=0.0)
    coord.solar._screen_transmittance_vector = MagicMock(return_value=(1.0, 1.0, 1.0))
    coord.solar.reconstruct_potential_vector = MagicMock(
        side_effect=lambda eff, _corr, _cfg: eff
    )

    # Replay shadow returns no inequality updates (empty log).
    coord.learning = MagicMock()
    coord.learning.replay_solar_nlms = MagicMock(return_value={
        "updates": 0,
        "entries_considered": 0,
        "entry_skipped_aux": 0,
        "entry_skipped_poisoned": 0,
        "entry_skipped_disabled": 0,
        "entry_skipped_low_magnitude": 0,
        "entry_skipped_missing_temp_key": 0,
        "unit_skipped_aux_list": 0,
        "unit_skipped_shutdown": 0,
        "unit_skipped_excluded_mode": 0,
        "unit_skipped_weighted_smear": 0,
        "unit_skipped_below_threshold": 0,
        "inequality_updates": 0,
        "inequality_non_binding": 0,
        "inequality_skipped_low_battery": 0,
        "inequality_skipped_mode": 0,
        "inequality_skipped_base": 0,
    })
    return coord


class TestModeSplitFields:
    """The new per-unit fields exist and reflect the stored regimes."""

    def test_split_delta_zero_when_regimes_equal(self):
        """Cooling seeded from heating (post-migration) → split delta ≈ 0."""
        coeff = {
            "heating": {"s": 0.40, "e": 0.10, "w": 0.05},
            "cooling": {"s": 0.40, "e": 0.10, "w": 0.05},
        }
        coord = _make_coord_modesplit(coeff)
        engine = DiagnosticsEngine(coord)
        result = engine.diagnose_solar()
        per_unit = result["per_unit"]["sensor.heater1"]
        assert per_unit["current_coefficient_heating"] == {
            "s": 0.4, "e": 0.1, "w": 0.05
        }
        assert per_unit["current_coefficient_cooling"] == {
            "s": 0.4, "e": 0.1, "w": 0.05
        }
        assert per_unit["coefficient_split_delta_pct"] == 0.0

    def test_split_delta_nonzero_when_regimes_diverge(self):
        """Cooling drifted (e.g. cooling-active install) → non-zero delta."""
        coeff = {
            "heating": {"s": 0.40, "e": 0.10, "w": 0.05},
            "cooling": {"s": 0.20, "e": 0.05, "w": 0.025},  # half — large COP gap
        }
        coord = _make_coord_modesplit(coeff)
        engine = DiagnosticsEngine(coord)
        result = engine.diagnose_solar()
        per_unit = result["per_unit"]["sensor.heater1"]
        # 50 % divergence on each direction → ~33 % under symmetric-mean
        # normalisation: |a-b| / ((a+b)/2 × n).  Approximately the L1
        # relative gap, halved.
        assert per_unit["coefficient_split_delta_pct"] is not None
        assert per_unit["coefficient_split_delta_pct"] > 20.0
        assert per_unit["coefficient_split_delta_pct"] < 50.0

    def test_split_delta_none_when_both_regimes_zero(self):
        """No coefficient learned → split delta is None (not divide-by-zero)."""
        coeff = {
            "heating": {"s": 0.0, "e": 0.0, "w": 0.0},
            "cooling": {"s": 0.0, "e": 0.0, "w": 0.0},
        }
        coord = _make_coord_modesplit(coeff)
        engine = DiagnosticsEngine(coord)
        result = engine.diagnose_solar()
        per_unit = result["per_unit"]["sensor.heater1"]
        assert per_unit["coefficient_split_delta_pct"] is None

    def test_current_coefficient_backwards_compat_field_present(self):
        """``current_coefficient`` (pre-#868 field) reflects the heating
        regime so existing dashboards and consumers continue working."""
        coeff = {
            "heating": {"s": 0.40, "e": 0.10, "w": 0.05},
            "cooling": {"s": 0.20, "e": 0.05, "w": 0.025},
        }
        coord = _make_coord_modesplit(coeff)
        engine = DiagnosticsEngine(coord)
        result = engine.diagnose_solar()
        per_unit = result["per_unit"]["sensor.heater1"]
        assert per_unit["current_coefficient"] == {"s": 0.4, "e": 0.1, "w": 0.05}
        # The split-aware fields must also be present
        assert "current_coefficient_heating" in per_unit
        assert "current_coefficient_cooling" in per_unit

    def test_split_delta_only_heating_learned(self):
        """Heating-only install: cooling at 0 → split delta = 100 %."""
        coeff = {
            "heating": {"s": 0.40, "e": 0.10, "w": 0.05},
            "cooling": {"s": 0.0, "e": 0.0, "w": 0.0},
        }
        coord = _make_coord_modesplit(coeff)
        engine = DiagnosticsEngine(coord)
        result = engine.diagnose_solar()
        per_unit = result["per_unit"]["sensor.heater1"]
        # |a - 0| / (|a| + 0) = 1.0 in every direction → 100 %.
        assert per_unit["coefficient_split_delta_pct"] == 100.0

    def test_split_delta_reads_raw_storage_not_prediction_default(self):
        """Regression for the post-#868 review finding: the metric must
        compare *stored* regimes — an unlearned cooling regime is
        ``{0,0,0}``, not the default decomposition (0.40 along the
        configured azimuth).  Otherwise every heating-only install
        reports a small spurious "split divergence" purely from the
        cooling default, muddling the validation criterion.
        """
        coeff = {
            "heating": {"s": 0.40, "e": 0.10, "w": 0.05},
            "cooling": {"s": 0.0, "e": 0.0, "w": 0.0},
        }
        coord = _make_coord_modesplit(coeff)

        # Make the prediction-time fallback return a non-trivial cooling
        # default — if the metric were reading from this path, it would
        # see two non-zero regimes and report a small split-delta.
        # Storage path correctly reports 100 %.
        def _coeff_with_default(entity_id, temp_key, mode):
            if mode == MODE_COOLING:
                # Production default for cooling, decomposed at south.
                return {"s": 0.40, "e": 0.0, "w": 0.0}
            return {"s": 0.40, "e": 0.10, "w": 0.05}
        coord.solar.calculate_unit_coefficient = MagicMock(side_effect=_coeff_with_default)

        engine = DiagnosticsEngine(coord)
        result = engine.diagnose_solar()
        per_unit = result["per_unit"]["sensor.heater1"]

        # Storage-driven: cooling regime is zero, heating learned.
        # |0.4-0|+|0.1-0|+|0.05-0| / (0.4+0.1+0.05) × 100 = 100 %.
        assert per_unit["coefficient_split_delta_pct"] == 100.0
        # Per-regime fields also reflect raw storage, not prediction
        # defaults.
        assert per_unit["current_coefficient_cooling"] == {
            "s": 0.0, "e": 0.0, "w": 0.0
        }
        # Backwards-compat field still uses the prediction-time view.
        assert per_unit["current_coefficient"]["s"] == 0.4


class TestInequalityShadowReplayHeatingOnly:
    """Inequality is heating-only (#865); the diagnose_solar shadow
    replay's stratified output reports the heating-regime coefficient
    when an inequality update fired, else None."""

    def test_no_inequality_update_returns_none(self):
        """Empty replay state → implied_coefficient_inequality is None."""
        coeff = {
            "heating": {"s": 0.40, "e": 0.10, "w": 0.05},
            "cooling": {"s": 0.0, "e": 0.0, "w": 0.0},
        }
        coord = _make_coord_modesplit(coeff)
        engine = DiagnosticsEngine(coord)
        result = engine.diagnose_solar()
        per_unit = result["per_unit"]["sensor.heater1"]
        assert per_unit["implied_coefficient_inequality"] is None

    def test_heating_shadow_with_values_reports_heating_regime(self):
        """Shadow replay produces a heating-regime coefficient → reported."""
        coeff = {
            "heating": {"s": 0.40, "e": 0.10, "w": 0.05},
            "cooling": {"s": 0.0, "e": 0.0, "w": 0.0},
        }
        coord = _make_coord_modesplit(coeff)

        # Patch the replay to populate shadow_coeffs with stratified shape.
        def _replay(*args, **kwargs):
            shadow = kwargs["solar_coefficients_per_unit"]
            shadow["sensor.heater1"] = {
                "heating": {"s": 0.30, "e": 0.05, "w": 0.10},
                "cooling": {"s": 0.0, "e": 0.0, "w": 0.0},
            }
            return {
                "updates": 0,
                "entries_considered": 0,
                "entry_skipped_aux": 0,
                "entry_skipped_poisoned": 0,
                "entry_skipped_disabled": 0,
                "entry_skipped_low_magnitude": 0,
                "entry_skipped_missing_temp_key": 0,
                "unit_skipped_aux_list": 0,
                "unit_skipped_shutdown": 0,
                "unit_skipped_excluded_mode": 0,
                "unit_skipped_weighted_smear": 0,
                "unit_skipped_below_threshold": 0,
                "inequality_updates": 1,
                "inequality_non_binding": 0,
                "inequality_skipped_low_battery": 0,
                "inequality_skipped_mode": 0,
                "inequality_skipped_base": 0,
            }
        coord.learning.replay_solar_nlms = MagicMock(side_effect=_replay)

        engine = DiagnosticsEngine(coord)
        result = engine.diagnose_solar()
        per_unit = result["per_unit"]["sensor.heater1"]
        assert per_unit["implied_coefficient_inequality"] == {
            "s": 0.3, "e": 0.05, "w": 0.1
        }


# =============================================================================
# Output compaction (#896 follow-up): top-level summary + inactive-unit collapse
# + screen_stratified field trim + transmittance candidate-list omission when
# uniform.  These cuts brought a 12-sensor 10-day diagnose response from ~3000
# JSON lines down to ~600.  The full verbose detail is still emitted for any
# unit / surface where the diagnostic actually carries actionable signal.
# =============================================================================


class TestDiagnoseSolarSummary:
    """Top-level ``summary`` block as the human-readable digest entry point."""

    def test_summary_block_present(self):
        """Every diagnose_solar response emits a ``summary`` at the top."""
        log = [_hour_entry(f"2026-04-15T{h:02d}:00:00", hour=h) for h in range(8, 14)]
        coord = _make_coord(log)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        assert "summary" in result
        summary = result["summary"]
        for field in (
            "verdict", "global_flags", "active_solar_units",
            "inactive_units", "units_with_flags",
            "battery_feedback", "battery_decay",
        ):
            assert field in summary, f"missing summary field: {field}"

    def test_summary_verdict_review_recommended_with_global_flags(self):
        """Any global flag → review_recommended verdict."""
        # A log that triggers `screen_drift_detected`: high error
        # difference between closed-screen and open-screen hours.  Reuse
        # the screen-stratification fixture's bias-injection pattern.
        from datetime import datetime, timedelta
        base_dt = datetime(2026, 4, 1, 8, 0)
        entries = []
        for i in range(15):
            ts = (base_dt + timedelta(hours=i * 4)).isoformat()
            entries.append(_hour_entry(
                ts, solar_s=0.6, solar_e=0.0, correction=20.0,  # closed
                actual=1.0, base=2.0, hour=12,  # large error
            ))
        for i in range(15):
            ts = (base_dt + timedelta(hours=i * 4 + 200)).isoformat()
            entries.append(_hour_entry(
                ts, solar_s=0.6, solar_e=0.0, correction=100.0,  # open
                actual=1.95, base=2.0, hour=12,  # tiny error
            ))
        coord = _make_coord(entries)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        summary = result["summary"]
        assert summary["verdict"] == "review_recommended"
        assert "screen_drift_detected" in summary["global_flags"]

    def test_summary_units_with_flags_lists_only_flagged(self):
        """``units_with_flags`` excludes units whose flags == []."""
        log = [_hour_entry(f"2026-04-15T{h:02d}:00:00", hour=h) for h in range(8, 14)]
        coord = _make_coord(log)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        summary = result["summary"]
        # Each entry in units_with_flags must have non-empty flags.
        for entry in summary["units_with_flags"]:
            assert "entity_id" in entry
            assert "flags" in entry
            assert entry["flags"], f"flagged unit {entry['entity_id']} has empty flags"

    def test_summary_battery_feedback_no_data_when_log_empty(self):
        """Empty log → battery_feedback verdict reports `no_data`."""
        coord = _make_coord([])
        # Battery thermal feedback flag must exist on the coord — default 0.0.
        coord.battery_thermal_feedback_k = 0.0
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        bf = result["summary"]["battery_feedback"]
        assert bf["verdict"] == "no_data"
        assert bf["current_k"] == 0.0


class TestInactiveUnitCollapse:
    """Sensors with no learned coefficient + no flags get a minimal record."""

    @staticmethod
    def _make_inactive_coord():
        """Coordinator with one sensor, no stored solar coefficient."""
        log = [_hour_entry(f"2026-04-15T{h:02d}:00:00", hour=h, actual=1.95, base=2.0)
               for h in range(8, 18)]
        coord = _make_coord(log, learned_coeff={"s": 0.0, "e": 0.0, "w": 0.0})
        # Override storage to be explicitly empty (no learned coefficient
        # stored anywhere) — the helper's default seeds storage with the
        # learned_coeff which would otherwise look "active" structurally.
        coord._solar_coefficients_per_unit = {}
        coord.model.solar_coefficients_per_unit = {}
        return coord

    def test_inactive_unit_marked_inactive(self):
        """``inactive: true`` flag on units that meet the collapse criteria."""
        coord = self._make_inactive_coord()
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        unit = result["per_unit"]["sensor.heater1"]
        assert unit.get("inactive") is True

    def test_inactive_unit_drops_verbose_blocks(self):
        """Verbose blocks omitted to compress noise on non-VP loads."""
        coord = self._make_inactive_coord()
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        unit = result["per_unit"]["sensor.heater1"]
        for dropped in (
            "transmittance_sensitivity",
            "screen_stratified",
            "temperature_stratified",
            "stability_windows",
            "temporal_bias",
            "implied_coefficient_30d_no_shutdown",
            "implied_coefficient_inequality",
            "implied_coefficient_physical",
            "saturation_pct",
            "shutdown_hours_30d",
            "shutdown_pct_of_qualifying",
            "dominant_component",
            "last_batch_fit",
        ):
            assert dropped not in unit, (
                f"inactive unit emission still carries '{dropped}' — "
                f"compaction missed this field"
            )

    def test_inactive_unit_keeps_backward_compat_fields(self):
        """Backward-compat: identifying + summary fields preserved."""
        coord = self._make_inactive_coord()
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        unit = result["per_unit"]["sensor.heater1"]
        for kept in (
            "current_coefficient",
            "current_coefficient_heating",
            "current_coefficient_cooling",
            "coefficient_split_delta_pct",
            "implied_coefficient_30d",
            "qualifying_hours",
            "mean_delta_kwh",
            "flags",
        ):
            assert kept in unit, (
                f"inactive unit missing backward-compat field '{kept}' — "
                f"existing consumers may break"
            )

    def test_active_unit_keeps_full_emission(self):
        """Sensors with non-zero learned coefficient retain verbose blocks."""
        log = [_hour_entry(f"2026-04-15T{h:02d}:00:00", hour=h) for h in range(8, 18)]
        coord = _make_coord(log)  # default learned_coeff is non-zero
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        unit = result["per_unit"]["sensor.heater1"]
        assert unit.get("inactive") is not True
        # Spot-check that the verbose fields are present.
        assert "transmittance_sensitivity" in unit
        assert "screen_stratified" in unit
        assert "temperature_stratified" in unit
        assert "stability_windows" in unit

    def test_inactive_collapse_does_not_fire_with_flag_present(self):
        """A unit with any flag stays active even with zero coefficient."""
        # Reuse the under-prediction scenario: actual << base + sun → mean_delta < -0.1
        # This generates an `under_predicting_solar` flag that should keep the
        # unit in the active emission path.
        log = [_hour_entry(f"2026-04-15T{h:02d}:00:00", hour=h,
                           actual=0.5, base=2.0, solar_s=0.5)
               for h in range(8, 18)]
        coord = _make_coord(log, learned_coeff={"s": 0.0, "e": 0.0, "w": 0.0})
        coord._solar_coefficients_per_unit = {}
        coord.model.solar_coefficients_per_unit = {}
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        unit = result["per_unit"]["sensor.heater1"]
        # Flag should fire AND compaction must NOT collapse.
        if unit.get("flags"):
            assert unit.get("inactive") is not True, (
                "inactive collapse must not fire when the unit has flags"
            )


class TestScreenStratifiedTrim:
    """``screen_stratified`` no longer carries mean_modeled / mean_implied."""

    def test_modeled_and_implied_means_dropped(self):
        log = [_hour_entry(f"2026-04-15T{h:02d}:00:00", hour=h) for h in range(8, 18)]
        coord = _make_coord(log)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        strat = result["per_unit"]["sensor.heater1"]["screen_stratified"]
        for bucket_key in ("open", "mid", "closed"):
            bucket = strat[bucket_key]
            if bucket.get("n", 0) > 0:
                assert "mean_modeled_kwh" not in bucket, (
                    "screen_stratified should not emit mean_modeled_kwh anymore"
                )
                assert "mean_implied_kwh" not in bucket, (
                    "screen_stratified should not emit mean_implied_kwh anymore"
                )
                assert "mean_delta_kwh" in bucket  # the actionable field stays


class TestTransmittanceUniformCollapse:
    """``transmittance_sensitivity.candidates`` omitted when uninformative."""

    def test_uniform_candidates_collapse_to_best_only(self):
        """When all candidates produce same RMSE+coefficient, drop the list.

        Setup: stored coefficient non-zero (so the unit stays active and
        emits ``transmittance_sensitivity`` instead of getting collapsed
        by the inactive-unit cut), but ``actual ≈ base`` so the *implied*
        solar derived per hour is ≈ 0 — every candidate transmittance
        floor fits the same zero-coefficient and produces the same RMSE.
        """
        from datetime import datetime, timedelta
        base_dt = datetime(2026, 4, 1, 8, 0)
        entries = []
        for i in range(30):
            ts = (base_dt + timedelta(hours=i)).isoformat()
            corr = 100.0 if i % 2 == 0 else 20.0  # vary so corr_var ≥ 40
            # Vary solar_e so the LS normal equations don't degenerate
            # (uniform vectors collapse the determinant to zero).  The
            # implied solar stays 0 because actual==base.
            entries.append(_hour_entry(
                ts, solar_s=0.4, solar_e=0.1 + 0.05 * (i % 3),
                correction=corr,
                actual=2.0, base=2.0,  # implied = base - actual = 0
            ))
        # Stored coefficient non-zero → unit stays active under the
        # inactive-collapse criteria; implied per-hour is still 0 so
        # the transmittance sweep is degenerate.
        coord = _make_coord(entries, learned_coeff={"s": 0.5, "e": 0.0, "w": 0.0})
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        unit = result["per_unit"]["sensor.heater1"]
        assert unit.get("inactive") is not True, (
            "test setup error: unit must be active for "
            "transmittance_sensitivity to be emitted"
        )
        sens = unit["transmittance_sensitivity"]
        assert sens is not None
        assert "best" in sens
        assert sens.get("verdict") == "uniform_across_candidates"
        assert "candidates" not in sens, (
            "uniform sweep should drop candidates list — saves "
            "6 nearly-identical rows per unit"
        )
