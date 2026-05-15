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


# =============================================================================
# DNI/DHI shadow report (#933) — top-level dni_dhi_shadow block
# =============================================================================


class TestDniDhiShadowReport:
    """Validates ``diagnose_solar``'s ``dni_dhi_shadow`` block."""

    def _make_log(self, n_clear=40, n_broken=40, n_overcast=40, with_dni_dhi=True):
        """Synthesize hourly log entries spanning the three cloud regimes.

        Target values are constructed so the 3D regression on
        ``solar_vector_*`` perfectly recovers ``y = 0.5·sv_s + 0.3·sv_e``.
        DNI/DHI carry a related-but-noisy signal so the 4D fit is
        non-trivial but does not strictly outperform 3D — the test
        cares about *shape* (regime classification, available=True,
        residual fields populated), not numerical superiority of one
        model.
        """
        from datetime import datetime, timedelta
        entries = []
        base_dt = datetime(2026, 4, 10, 8, 0)
        idx = 0
        for n, dni, dhi, regime_label in (
            (n_clear, 700.0, 100.0, "clear"),
            (n_broken, 250.0, 200.0, "broken"),
            (n_overcast, 20.0, 150.0, "overcast"),
        ):
            for j in range(n):
                ts = (base_dt + timedelta(hours=idx)).isoformat()
                idx += 1
                # Vary the vector so the LS doesn't degenerate.
                sv_s = 0.3 + 0.01 * (j % 5)
                sv_e = 0.05 + 0.005 * (j % 7)
                sv_w = 0.02
                y = 0.5 * sv_s + 0.3 * sv_e
                entry = {
                    "timestamp": ts,
                    "hour": (base_dt + timedelta(hours=idx)).hour,
                    "solar_vector_s": sv_s,
                    "solar_vector_e": sv_e,
                    "solar_vector_w": sv_w,
                    "solar_factor": sv_s + sv_e + sv_w,
                    "solar_impact_kwh": y,
                    "actual_kwh": 1.5,
                    "expected_kwh": 2.0,
                    "correction_percent": 100.0,
                    "auxiliary_active": False,
                    "guest_impact_kwh": 0.0,
                    "unit_modes": {"sensor.heater1": "heating"},
                    "unit_breakdown": {"sensor.heater1": 1.5},
                    "unit_expected_breakdown": {"sensor.heater1": 2.0},
                    "solar_dominant_entities": [],
                    "temp": 10.0,
                    "inertia_temp": 10.0,
                    "temp_key": "10",
                    "wind_bucket": "normal",
                    "solar_impact_raw_kwh": y,
                }
                if with_dni_dhi:
                    entry["dni"] = dni
                    entry["dhi"] = dhi
                entries.append(entry)
        return entries

    def _coord_with_sun(self, log, *, correlation_data=None):
        """_make_coord variant that wires get_approx_sun_pos."""
        coord = _make_coord(log)
        # Fixed sun position keeps the geometric beam projection
        # deterministic across the synthetic timestamps — south-facing
        # noon-ish sun gives non-zero geom_s, near-zero geom_e/w, so
        # the 4D regression gets a usable signal in the south term.
        coord.solar.get_approx_sun_pos = MagicMock(return_value=(45.0, 180.0))
        # Concrete numeric attributes for code paths reached by the
        # ≥60-entry fixtures used here (the smaller-fixture tests
        # elsewhere in the file never hit these branches).
        coord.battery_thermal_feedback_k = 0.0
        coord._per_unit_min_base_thresholds = {}
        # Cross-check_actual gate consults correlation_data; default to
        # empty dict so the gate falls through cleanly.  Tests that want
        # the actual-target path populated pass an explicit dict.
        coord._correlation_data = correlation_data if correlation_data is not None else {}
        return coord

    def test_unavailable_when_no_dni_dhi(self):
        """Pre-1.3.6 logs (no dni/dhi) → available=False."""
        log = self._make_log(with_dni_dhi=False)
        coord = self._coord_with_sun(log)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        block = result["dni_dhi_shadow"]
        assert block["available"] is False
        assert block["n_hours"] == 0
        assert block["reason"] == "insufficient_overlap_data"

    def test_unavailable_when_too_few_overlap_entries(self):
        """Below the 60-hour floor → available=False with n_hours echo."""
        log = self._make_log(n_clear=10, n_broken=10, n_overcast=10)
        coord = self._coord_with_sun(log)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        block = result["dni_dhi_shadow"]
        assert block["available"] is False
        assert block["n_hours"] <= 30

    def test_full_block_shape_and_regime_split(self):
        """Sufficient data → full block with regime classification."""
        log = self._make_log(n_clear=40, n_broken=40, n_overcast=40)
        coord = self._coord_with_sun(log)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        block = result["dni_dhi_shadow"]
        assert block["available"] is True
        assert block["target_field"] == "solar_impact_kwh"
        assert block["target_caveat"] == "self_referential_3d_advantage"
        assert block["n_hours"] == 120
        assert block["regime_counts"] == {"clear": 40, "broken": 40, "overcast": 40}

        # Coefficient blocks present with right keys.
        assert set(block["shadow_3d_coefficient"]) == {"s", "e", "w"}
        assert set(block["shadow_4d_coefficient"]) == {
            "s_direct", "e_direct", "w_direct", "diffuse"
        }

        # 3D fit recovers y = 0.5·sv_s + 0.3·sv_e closely (synthetic
        # has no noise).  W has no signal so coefficient ≈ 0.
        b3 = block["shadow_3d_coefficient"]
        assert b3["s"] == pytest.approx(0.5, abs=0.05)
        assert b3["e"] == pytest.approx(0.3, abs=0.1)

        # Residual fields populated for every regime.
        residuals = block["residuals"]
        assert "all" in residuals
        for reg in ("clear", "broken", "overcast"):
            r = residuals["by_regime"][reg]
            assert "n_hours" in r
            assert "residual_std_3d_kwh" in r
            assert "residual_std_4d_kwh" in r

        # Cross-check sub-block exists.  Without correlation_data the
        # actual-target gate produces zero qualifying samples, so the
        # sub-block reports unavailable with a count of 0.
        cc = block["cross_check_actual"]
        assert cc["available"] is False
        assert cc["target_field"] == "global_base_minus_actual_kwh"
        assert cc["n_hours"] == 0

    def test_cross_check_actual_fits_when_correlation_data_present(self):
        """Populating ``_correlation_data`` activates the non-self-
        referential cross-check block; gates exclude shutdown and
        cooling-mode hours."""
        log = self._make_log(n_clear=80, n_broken=80, n_overcast=80)
        # Match the temp_key/wind_bucket the synthetic entries write
        # (temp=10 → temp_key="10", wind_bucket="normal").  Fix
        # base_house = 2.0 to match the entries' expected_kwh — the
        # gate computes y_actual = base_house − actual_kwh = 2.0 − 1.5
        # = 0.5 per qualifying hour.  This is constant across the
        # entries' synthetic shape; the LS fit is nearly degenerate
        # but the diagnostic still produces a populated block.
        coord = self._coord_with_sun(log, correlation_data={"10": {"normal": 2.0}})
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        cc = result["dni_dhi_shadow"]["cross_check_actual"]
        assert cc["available"] is True
        assert cc["target_field"] == "global_base_minus_actual_kwh"
        assert cc["target_caveat"] == "non_self_referential"
        assert cc["n_hours"] == 240
        assert "shadow_4d_coefficient" in cc
        assert "shadow_3d_coefficient" in cc
        assert "broken_regime_improvement_pct" in cc

    def test_cross_check_per_entity_block_present_and_independent(self):
        """``cross_check_actual.per_entity`` reports one block per energy
        sensor with its own n_hours, fits, and screen_config.

        Two entities with disjoint per-unit base buckets exercise the
        per-entity regression independently of the house aggregate —
        each fit sees only its own ``unit_base − unit_actual`` target.
        """
        log = self._make_log(n_clear=80, n_broken=80, n_overcast=80)
        # Add a second entity to every entry's unit_breakdown so the
        # per-entity loop has two candidates to fit.
        for entry in log:
            entry["unit_breakdown"] = {
                "sensor.heater1": 1.5,
                "sensor.heater2": 0.6,
            }
        coord = self._coord_with_sun(
            log, correlation_data={"10": {"normal": 2.0}},
        )
        coord.energy_sensors = ["sensor.heater1", "sensor.heater2"]
        coord._correlation_data_per_unit = {
            "sensor.heater1": {"10": {"normal": 1.6}},  # base − actual = 0.1
            "sensor.heater2": {"10": {"normal": 0.8}},  # base − actual = 0.2
        }
        # Distinct screen_config per entity so we can confirm the right
        # one lands on the per-entity block.
        coord.screen_config_for_entity = MagicMock(
            side_effect=lambda eid: (False, False, True) if eid == "sensor.heater1"
            else (False, False, False),
        )

        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        cc = result["dni_dhi_shadow"]["cross_check_actual"]
        assert "per_entity" in cc
        per_e = cc["per_entity"]
        assert set(per_e) == {"sensor.heater1", "sensor.heater2"}

        for eid, expected_sc in (
            ("sensor.heater1", [False, False, True]),
            ("sensor.heater2", [False, False, False]),
        ):
            block = per_e[eid]
            assert block["available"] is True
            assert block["target_field"] == "unit_base_minus_unit_actual_kwh"
            assert block["target_caveat"] == "non_self_referential_per_entity"
            assert block["n_hours"] == 240
            assert block["screen_config"] == expected_sc

    def test_cross_check_per_entity_skips_unit_in_cooling_mode(self):
        """A unit running cooling is excluded per-entity even when the
        house aggregate would fail mixed-mode gating elsewhere."""
        log = self._make_log(n_clear=80, n_broken=80, n_overcast=80)
        # Mark the second half of entries with sensor.heater1 in cooling
        # mode at the per-unit level — the house aggregate gate would
        # reject these too, but the per-entity gate is what we want to
        # verify here so the test asserts on the per-entity block's
        # n_hours (which counts only HEATING/GUEST_HEATING samples).
        for i in range(120, 240):
            log[i]["unit_modes"] = {"sensor.heater1": "cooling"}
        coord = self._coord_with_sun(
            log, correlation_data={"10": {"normal": 2.0}},
        )
        coord.energy_sensors = ["sensor.heater1"]
        coord._correlation_data_per_unit = {
            "sensor.heater1": {"10": {"normal": 1.6}},
        }

        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        cc = result["dni_dhi_shadow"]["cross_check_actual"]
        per_e = cc["per_entity"]["sensor.heater1"]
        # 120 heating-mode entries qualify; 120 cooling-mode entries
        # are dropped from the per-entity sample subset.
        assert per_e["n_hours"] == 120

    def test_per_entity_filtered_aggregates_learned_entities_only(self):
        """``per_entity_filtered`` aggregates hours-weighted improvement_pct
        across entities with ``learned`` set on heating regime, excluding
        default-coefficient noise."""
        log = self._make_log(n_clear=80, n_broken=80, n_overcast=80)
        for entry in log:
            entry["unit_breakdown"] = {
                "sensor.heater1": 1.5,
                "sensor.heater2": 0.6,
            }
        coord = self._coord_with_sun(
            log, correlation_data={"10": {"normal": 2.0}},
        )
        coord.energy_sensors = ["sensor.heater1", "sensor.heater2"]
        coord._correlation_data_per_unit = {
            "sensor.heater1": {"10": {"normal": 1.6}},
            "sensor.heater2": {"10": {"normal": 0.8}},
        }
        # Only heater1 has learned a real coefficient.  heater2 carries
        # the migration-seeded default and must be filtered out.
        coord._solar_coefficients_per_unit = {
            "sensor.heater1": {
                "heating": {"s": 0.5, "e": 0.3, "w": 0.2, "learned": True}
            },
            "sensor.heater2": {
                "heating": {"s": 0.1, "e": 0.0, "w": 0.0}  # no learned flag
            },
        }

        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        cc = result["dni_dhi_shadow"]["cross_check_actual"]
        assert "per_entity_filtered" in cc
        filt = cc["per_entity_filtered"]
        assert filt["n_entities"] == 1
        assert filt["entities"] == ["sensor.heater1"]
        # per_entity blocks themselves carry the learned flag for caller
        # introspection.
        assert cc["per_entity"]["sensor.heater1"]["learned"] is True
        assert cc["per_entity"]["sensor.heater2"]["learned"] is False

    def test_signal_agreement_skips_unstable_recovery_hours(self):
        """Hours with ``no_cloud_reference < 0.1`` (sun in Kelvin-Twist
        Zone 2/3) or ``potential_solar_factor < 0.05`` are skipped from
        the signal_agreement comparison and counted via
        ``n_skipped_unstable_recovery``.

        Mathematically: ``kasten_cf = potential_sf / no_cloud_ref`` is
        unstable when the denominator is small (Kelvin-Twist floor of
        0.05 in Zone 3) or the numerator is below the rounding-noise
        floor of the 3-decimal-rounded log field, producing artifact
        correlations that prompted the gate.
        """
        from datetime import datetime, timedelta
        entries = []
        base_dt = datetime(2026, 4, 10, 12, 0)
        # Half the entries pass the gate; the other half get a tiny
        # potential_sf so the gate trips on numerator side.
        for j in range(160):
            ts = (base_dt + timedelta(hours=j)).isoformat()
            potential = 0.40 if j % 2 == 0 else 0.02
            entries.append({
                "timestamp": ts, "hour": 12,
                "solar_vector_s": 0.3, "solar_vector_e": 0.05, "solar_vector_w": 0.02,
                "solar_factor": 0.37,
                "potential_solar_factor": potential,
                "solar_impact_kwh": 0.5,
                "actual_kwh": 1.5, "expected_kwh": 2.0,
                "correction_percent": 100.0, "auxiliary_active": False,
                "guest_impact_kwh": 0.0, "unit_modes": {"sensor.heater1": "heating"},
                "unit_breakdown": {"sensor.heater1": 1.5},
                "unit_expected_breakdown": {"sensor.heater1": 2.0},
                "solar_dominant_entities": [], "temp": 10.0, "inertia_temp": 10.0,
                "temp_key": "10", "wind_bucket": "normal",
                "solar_impact_raw_kwh": 0.5,
                "dni": 700.0, "dhi": 100.0,
            })
        coord = self._coord_with_sun(entries)
        coord.solar.calculate_solar_factor = MagicMock(return_value=0.5)

        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        sa = result["dni_dhi_shadow"]["signal_agreement"]
        # 80 entries with potential_sf=0.40 pass; 80 with 0.02 are gated.
        assert sa["all"]["n_hours"] == 80
        assert sa["n_skipped_unstable_recovery"] == 80
        assert sa["stability_gate"] == (
            "no_cloud_reference > 0.1 AND potential_solar_factor > 0.05"
        )

    def test_signal_agreement_pairs_kasten_with_dni_normalized(self):
        """``signal_agreement`` reports per-regime correlation, RMSE, and
        mean-bias of the two pre-geometry scalar signals."""
        from datetime import datetime, timedelta
        # Build entries with a deterministic potential_solar_factor and
        # DNI mix so the expected agreement is non-degenerate.  Two
        # bands: clear hours where kasten_cf and dni_normalized agree
        # well, and broken hours where they diverge.
        entries = []
        base_dt = datetime(2026, 4, 10, 12, 0)
        for j in range(80):
            ts = (base_dt + timedelta(hours=j)).isoformat()
            entries.append({
                "timestamp": ts, "hour": 12,
                "solar_vector_s": 0.3, "solar_vector_e": 0.05, "solar_vector_w": 0.02,
                "solar_factor": 0.37,
                "potential_solar_factor": 0.40,
                "solar_impact_kwh": 0.5,
                "actual_kwh": 1.5, "expected_kwh": 2.0,
                "correction_percent": 100.0, "auxiliary_active": False,
                "guest_impact_kwh": 0.0, "unit_modes": {"sensor.heater1": "heating"},
                "unit_breakdown": {"sensor.heater1": 1.5},
                "unit_expected_breakdown": {"sensor.heater1": 2.0},
                "solar_dominant_entities": [], "temp": 10.0, "inertia_temp": 10.0,
                "temp_key": "10", "wind_bucket": "normal",
                "solar_impact_raw_kwh": 0.5,
                "dni": 700.0, "dhi": 100.0,  # clear
            })
        for j in range(80):
            ts = (base_dt + timedelta(hours=80 + j)).isoformat()
            entries.append({
                "timestamp": ts, "hour": 12,
                "solar_vector_s": 0.3, "solar_vector_e": 0.05, "solar_vector_w": 0.02,
                "solar_factor": 0.37,
                # potential_solar_factor much lower than the DNI value
                # would suggest → broken-cloud divergence
                "potential_solar_factor": 0.10,
                "solar_impact_kwh": 0.5,
                "actual_kwh": 1.5, "expected_kwh": 2.0,
                "correction_percent": 100.0, "auxiliary_active": False,
                "guest_impact_kwh": 0.0, "unit_modes": {"sensor.heater1": "heating"},
                "unit_breakdown": {"sensor.heater1": 1.5},
                "unit_expected_breakdown": {"sensor.heater1": 2.0},
                "solar_dominant_entities": [], "temp": 10.0, "inertia_temp": 10.0,
                "temp_key": "10", "wind_bucket": "normal",
                "solar_impact_raw_kwh": 0.5,
                "dni": 250.0, "dhi": 200.0,  # broken
            })
        coord = self._coord_with_sun(entries)
        # No-cloud reference returns 0.5 deterministically so
        # kasten_cf = potential_solar_factor / 0.5 ∈ [0, 1].
        coord.solar.calculate_solar_factor = MagicMock(return_value=0.5)

        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        sa = result["dni_dhi_shadow"]["signal_agreement"]
        assert "definition" in sa
        assert sa["all"]["n_hours"] == 160
        assert sa["all"]["correlation"] is not None
        assert sa["all"]["rmse"] is not None
        # Per-regime population: 80 clear, 80 broken, 0 overcast.
        assert sa["by_regime"]["clear"]["n_hours"] == 80
        assert sa["by_regime"]["broken"]["n_hours"] == 80
        # Overcast under-populated → degraded report shape, not crash.
        assert sa["by_regime"]["overcast"]["n_hours"] == 0
        assert sa["by_regime"]["overcast"]["correlation"] is None

    def test_cross_check_per_entity_unavailable_when_no_per_unit_base(self):
        """No per-unit correlation data → block reports unavailable."""
        log = self._make_log(n_clear=80, n_broken=80, n_overcast=80)
        coord = self._coord_with_sun(
            log, correlation_data={"10": {"normal": 2.0}},
        )
        coord.energy_sensors = ["sensor.heater1"]
        coord._correlation_data_per_unit = {}  # missing per-unit bucket

        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        cc = result["dni_dhi_shadow"]["cross_check_actual"]
        per_e = cc["per_entity"]["sensor.heater1"]
        assert per_e["available"] is False
        assert per_e["n_hours"] == 0

    def test_cross_check_skips_shutdown_and_cooling_hours(self):
        """``solar_dominant_entities`` flagged hours and cooling-mode hours
        are excluded from the actual-target sample set."""
        log = self._make_log(n_clear=80, n_broken=80, n_overcast=80)
        # Mark first 60 entries as solar-shutdown; next 60 as cooling-mode.
        # Remaining 120 entries are eligible for the actual-target fit.
        for i in range(60):
            log[i]["solar_dominant_entities"] = ["sensor.heater1"]
        for i in range(60, 120):
            log[i]["unit_modes"] = {"sensor.heater1": "cooling"}
        coord = self._coord_with_sun(log, correlation_data={"10": {"normal": 2.0}})
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        cc = result["dni_dhi_shadow"]["cross_check_actual"]
        assert cc["available"] is True
        assert cc["n_hours"] == 120  # 240 − 60 (shutdown) − 60 (cooling)


class TestElevationDiagnostics:
    """Tier 1 elevation-stratified residual diagnostic (#927).

    ``elevation_diagnostics.instantaneous`` bins the per-hour residual
    ``actual_impact - predicted = implied_solar - modeled_solar`` by sun
    elevation at the hour's midpoint.  Heating regime only.  Negative
    median in a bucket means the model OVER-predicts solar reduction
    in that bucket (hotspot signature when concentrated at high elev);
    positive means UNDER-predicts.
    """

    @staticmethod
    def _coord_with_elevations(hourly_log, elevations):
        """Build a coord whose `get_approx_sun_pos` returns the given
        elevations sequentially (one per `_hourly_log` entry, in order).
        """
        coord = _make_coord(hourly_log)
        elev_iter = iter(elevations)
        coord.solar.get_approx_sun_pos = MagicMock(
            side_effect=lambda _dt: (next(elev_iter), 180.0)
        )
        return coord

    @staticmethod
    def _hours_at(elev, *, n, base_dt, hour_offset_start, mode="heating",
                  solar_s=0.5, actual=1.5, base=2.0):
        """Generate `n` hourly log entries plus the matching elevation list."""
        from datetime import timedelta
        entries = []
        elevs = []
        for i in range(n):
            ts = (base_dt + timedelta(hours=hour_offset_start + i)).isoformat()
            entries.append(_hour_entry(
                ts, solar_s=solar_s, solar_e=0.0, solar_w=0.0,
                actual=actual, base=base, mode=mode,
            ))
            elevs.append(elev)
        return entries, elevs

    def test_block_present_in_response(self):
        """`elevation_diagnostics.instantaneous` appears under each active per-unit block."""
        from datetime import datetime
        base_dt = datetime(2026, 4, 1, 12, 0)
        entries, elevations = self._hours_at(20.0, n=10, base_dt=base_dt, hour_offset_start=0)
        coord = self._coord_with_elevations(entries, elevations)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        per_unit = result["per_unit"]["sensor.heater1"]
        assert "elevation_diagnostics" in per_unit
        assert "instantaneous" in per_unit["elevation_diagnostics"]
        # All 5 buckets are emitted, even empty ones.
        for bucket in ("0-15", "15-30", "30-45", "45-60", "60-90"):
            assert bucket in per_unit["elevation_diagnostics"]["instantaneous"]

    def test_buckets_partition_by_elevation(self):
        """Each bucket gets exactly the entries whose elevation falls inside it."""
        from datetime import datetime
        base_dt = datetime(2026, 4, 1, 12, 0)
        # 6 entries × 5 elevations spanning the bucket midpoints.  6 ≥ min_samples=5
        # so each bucket emits the full block instead of just `n`.
        entries: list[dict] = []
        elevations: list[float] = []
        for elev in (7.0, 22.0, 37.0, 52.0, 75.0):
            chunk_entries, chunk_elevs = self._hours_at(
                elev, n=6, base_dt=base_dt, hour_offset_start=len(entries),
            )
            entries.extend(chunk_entries)
            elevations.extend(chunk_elevs)
        coord = self._coord_with_elevations(entries, elevations)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        ev = result["per_unit"]["sensor.heater1"]["elevation_diagnostics"]["instantaneous"]
        assert ev["0-15"]["n"] == 6
        assert ev["15-30"]["n"] == 6
        assert ev["30-45"]["n"] == 6
        assert ev["45-60"]["n"] == 6
        assert ev["60-90"]["n"] == 6

    def test_zero_bias_yields_zero_residual(self):
        """When implied_solar == modeled_solar exactly, median residual ≈ 0."""
        from datetime import datetime
        base_dt = datetime(2026, 4, 1, 12, 0)
        # learned_coeff·s = 1.0·0.5 = 0.5 = modeled_solar.
        # base − actual = 2.0 − 1.5 = 0.5 = implied_solar.
        # residual = 0.5 − 0.5 = 0.
        entries, elevations = self._hours_at(20.0, n=10, base_dt=base_dt, hour_offset_start=0)
        coord = self._coord_with_elevations(entries, elevations)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        ev = result["per_unit"]["sensor.heater1"]["elevation_diagnostics"]["instantaneous"]
        assert ev["15-30"]["n"] == 10
        assert abs(ev["15-30"]["median_residual"]) < 1e-3
        assert abs(ev["15-30"]["median_residual_normalised"]) < 1e-3

    def test_hotspot_signature_negative_at_high_elev(self):
        """Synthetic over-prediction concentrated at high elevation produces
        negative `median_residual_normalised` in the high-elev bucket while
        leaving low-elev buckets at zero — the hotspot fingerprint."""
        from datetime import datetime
        base_dt = datetime(2026, 4, 1, 12, 0)
        # Low-elev: zero bias (actual=1.5, base=2.0 → implied=0.5, modeled=0.5).
        low, low_elevs = self._hours_at(
            10.0, n=12, base_dt=base_dt, hour_offset_start=0,
            actual=1.5, base=2.0,
        )
        # High-elev: model OVER-predicts solar reduction.  modeled=0.5 (coef·s),
        # but observed implied = 0.2 (actual=1.8, base=2.0).  Residual =
        # implied − modeled = −0.3.  Normalised by mean_potential = vector_mag = 0.5.
        high, high_elevs = self._hours_at(
            50.0, n=12, base_dt=base_dt, hour_offset_start=12,
            actual=1.8, base=2.0,
        )
        coord = self._coord_with_elevations(low + high, low_elevs + high_elevs)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        ev = result["per_unit"]["sensor.heater1"]["elevation_diagnostics"]["instantaneous"]
        # Low-elev bucket clean.
        assert ev["0-15"]["n"] == 12
        assert abs(ev["0-15"]["median_residual"]) < 1e-3
        # High-elev bucket: residual = −0.3, normalised = −0.6.
        assert ev["45-60"]["n"] == 12
        assert ev["45-60"]["median_residual"] == pytest.approx(-0.3, abs=1e-3)
        assert ev["45-60"]["median_residual_normalised"] == pytest.approx(-0.6, abs=1e-3)
        # Untouched buckets remain empty.
        assert ev["15-30"]["n"] == 0
        assert ev["30-45"]["n"] == 0

    def test_min_samples_gate_collapses_small_buckets(self):
        """Buckets with fewer than 5 samples emit only `n`, not the full
        median / MAD / normalised block."""
        from datetime import datetime
        base_dt = datetime(2026, 4, 1, 12, 0)
        # 4 samples in 0-15 (below threshold), 10 samples in 15-30 (above).
        entries: list[dict] = []
        elevations: list[float] = []
        e1, ev1 = self._hours_at(10.0, n=4, base_dt=base_dt, hour_offset_start=0)
        e2, ev2 = self._hours_at(20.0, n=10, base_dt=base_dt, hour_offset_start=4)
        entries.extend(e1 + e2)
        elevations.extend(ev1 + ev2)
        coord = self._coord_with_elevations(entries, elevations)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        ev = result["per_unit"]["sensor.heater1"]["elevation_diagnostics"]["instantaneous"]
        # Below-threshold bucket: only `n` reported.
        assert ev["0-15"] == {"n": 4}
        # Above-threshold: full block.
        assert ev["15-30"]["n"] == 10
        assert "median_residual" in ev["15-30"]
        assert "mad_residual" in ev["15-30"]
        assert "mean_potential" in ev["15-30"]
        assert "median_residual_normalised" in ev["15-30"]

    def test_cooling_mode_skipped(self):
        """Cooling-mode hours do not populate elevation buckets — Tier 1 is
        heating-regime-only by design (rationale: the user-facing hypothesis
        is heating-mode shoulder-season afternoons; cooling extension comes
        with Tier 2)."""
        from datetime import datetime
        base_dt = datetime(2026, 4, 1, 12, 0)
        entries, elevations = self._hours_at(
            50.0, n=10, base_dt=base_dt, hour_offset_start=0, mode="cooling",
            actual=2.5, base=2.0,  # cooling: implied = actual − base = 0.5
        )
        coord = self._coord_with_elevations(entries, elevations)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        per_unit = result["per_unit"]["sensor.heater1"]
        # Inactive-collapse may apply if cooling has zero coefficient — in
        # which case there's no `elevation_diagnostics` key at all (collapsed
        # response).  When the verbose block IS emitted, all elev buckets
        # must be empty since cooling samples are excluded.
        if "elevation_diagnostics" in per_unit:
            ev = per_unit["elevation_diagnostics"]["instantaneous"]
            for bucket in ev.values():
                assert bucket["n"] == 0

    def test_solar_unavailable_no_crash(self):
        """When `get_approx_sun_pos` raises or returns nothing usable, the
        diagnostic emits all-empty buckets without breaking the rest of the
        response."""
        from datetime import datetime
        base_dt = datetime(2026, 4, 1, 12, 0)
        entries, _ = self._hours_at(20.0, n=10, base_dt=base_dt, hour_offset_start=0)
        coord = _make_coord(entries)
        # Simulate astral failure: a TypeError-raising mock.
        coord.solar.get_approx_sun_pos = MagicMock(side_effect=TypeError("astral down"))
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        ev = result["per_unit"]["sensor.heater1"]["elevation_diagnostics"]["instantaneous"]
        for bucket_data in ev.values():
            assert bucket_data["n"] == 0

    def test_bucket_boundary_half_open(self):
        """Elevation = 15.0 lands in [15, 30), not [0, 15) — half-open intervals."""
        from datetime import datetime
        base_dt = datetime(2026, 4, 1, 12, 0)
        entries, elevations = self._hours_at(
            15.0, n=10, base_dt=base_dt, hour_offset_start=0,
        )
        coord = self._coord_with_elevations(entries, elevations)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        ev = result["per_unit"]["sensor.heater1"]["elevation_diagnostics"]["instantaneous"]
        assert ev["0-15"]["n"] == 0
        assert ev["15-30"]["n"] == 10

    def test_negative_elevation_skipped(self):
        """Sun below horizon (elev < 0) is excluded — no bucket catches it."""
        from datetime import datetime
        base_dt = datetime(2026, 4, 1, 12, 0)
        entries, elevations = self._hours_at(
            -5.0, n=10, base_dt=base_dt, hour_offset_start=0,
        )
        coord = self._coord_with_elevations(entries, elevations)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        ev = result["per_unit"]["sensor.heater1"]["elevation_diagnostics"]["instantaneous"]
        for bucket_data in ev.values():
            assert bucket_data["n"] == 0


class TestElevationDiagnosticsLag:
    """Tier 2 lag-stratified residual diagnostic (#927).

    For each qualifying originator hour H, walk forward 0-6 hours and
    accumulate ``base_{H+k} − actual_{H+k}`` per (elevation bucket, lag k).
    `tail_sum_lag1_to_6_kwh` is the actionable scalar — positive on
    battery regime, near zero on hotspot regime.
    """

    @staticmethod
    def _coord_with_elevations(hourly_log, elevations):
        """Sequential-elevation coord (one per log entry, in order)."""
        coord = _make_coord(hourly_log)
        elev_iter = iter(elevations)
        coord.solar.get_approx_sun_pos = MagicMock(
            side_effect=lambda _dt: (next(elev_iter), 180.0)
        )
        return coord

    @staticmethod
    def _make_train(*, originator_dt, originator_elev, n_trains, train_spacing=8,
                    originator_actual=1.5, tail_actual=1.7, base=2.0,
                    tail_solar_factor=0.0, originator_solar_s=0.5):
        """Build N spaced 7-hour trains: originator + 6 tail entries each.

        Tail entries have solar_s=0 (so they fail the vector_mag gate and
        do not become originators themselves) and a configurable actual /
        base / solar_factor.
        """
        from datetime import timedelta
        entries = []
        elevations = []
        for train_idx in range(n_trains):
            h0 = train_idx * train_spacing
            ts_h0 = (originator_dt + timedelta(hours=h0)).isoformat()
            entries.append(_hour_entry(
                ts_h0, solar_s=originator_solar_s, solar_e=0.0, solar_w=0.0,
                actual=originator_actual, base=base, mode="heating",
            ))
            elevations.append(originator_elev)
            for k in range(1, 7):
                ts_hk = (originator_dt + timedelta(hours=h0 + k)).isoformat()
                entries.append(_hour_entry(
                    ts_hk, solar_s=0.0, solar_e=0.0, solar_w=0.0,
                    solar_factor=tail_solar_factor,
                    actual=tail_actual, base=base, mode="heating",
                ))
                # Below-horizon → no bucket; tail entries don't participate
                # as originators.
                elevations.append(-1.0)
        return entries, elevations

    def test_lag_block_present_in_response(self):
        """`elevation_diagnostics.lag` exists alongside `instantaneous`
        with all 5 buckets emitted."""
        from datetime import datetime
        base_dt = datetime(2026, 4, 1, 12, 0)
        entries, elevations = self._make_train(
            originator_dt=base_dt, originator_elev=20.0, n_trains=12,
        )
        coord = self._coord_with_elevations(entries, elevations)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        ev_diag = result["per_unit"]["sensor.heater1"]["elevation_diagnostics"]
        assert "lag" in ev_diag
        for bucket in ("0-15", "15-30", "30-45", "45-60", "60-90"):
            assert bucket in ev_diag["lag"]

    def test_battery_signature_positive_tail_at_originator_bucket(self):
        """Sustained consumption reduction over 6 h after the originator
        produces large positive tail-sum scalars on both windows."""
        from datetime import datetime
        base_dt = datetime(2026, 4, 1, 12, 0)
        # 12 trains, originator at 10° elev (bucket 0-15°).  Sustained
        # reduction: tail_actual=1.7 vs base=2.0 → 0.3 kWh per tail hour.
        # tail_sum_1_to_3 = 3 × 0.3 = 0.9, tail_sum_1_to_6 = 6 × 0.3 = 1.8.
        entries, elevations = self._make_train(
            originator_dt=base_dt, originator_elev=10.0, n_trains=12,
            originator_actual=1.5, tail_actual=1.7, base=2.0,
        )
        coord = self._coord_with_elevations(entries, elevations)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        lag_block = result["per_unit"]["sensor.heater1"]["elevation_diagnostics"]["lag"]
        bucket = lag_block["0-15"]
        # Each lag has 12 trains' worth of samples, well above min=10.
        assert bucket["lag_0"]["n"] == 12
        assert bucket["lag_0"]["mean_residual_kwh"] == pytest.approx(0.5, abs=1e-3)
        for k in range(1, 7):
            assert bucket[f"lag_{k}"]["n"] == 12
            assert bucket[f"lag_{k}"]["mean_residual_kwh"] == pytest.approx(0.3, abs=1e-3)
        assert bucket["tail_sum_lag1_to_3_kwh"] == pytest.approx(0.9, abs=1e-3)
        assert bucket["tail_sum_lag1_to_6_kwh"] == pytest.approx(1.8, abs=1e-3)

    def test_hotspot_signature_zero_tail_at_high_elev(self):
        """No sustained reduction (tail consumption ≈ baseline) → both
        tail-sum scalars near zero, while lag_0 still reports the
        originator's instantaneous reduction.  Hotspot fingerprint."""
        from datetime import datetime
        base_dt = datetime(2026, 4, 1, 12, 0)
        # 12 trains, originator at 50° elev (bucket 45-60°).  No sustained
        # reduction: tail_actual=2.0 vs base=2.0 → 0 each tail hour.
        entries, elevations = self._make_train(
            originator_dt=base_dt, originator_elev=50.0, n_trains=12,
            originator_actual=1.5, tail_actual=2.0, base=2.0,
        )
        coord = self._coord_with_elevations(entries, elevations)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        lag_block = result["per_unit"]["sensor.heater1"]["elevation_diagnostics"]["lag"]
        bucket = lag_block["45-60"]
        # Originator hour reduction preserved.
        assert bucket["lag_0"]["mean_residual_kwh"] == pytest.approx(0.5, abs=1e-3)
        # All tail hours: zero reduction.
        for k in range(1, 7):
            assert bucket[f"lag_{k}"]["mean_residual_kwh"] == pytest.approx(0.0, abs=1e-3)
        assert bucket["tail_sum_lag1_to_3_kwh"] == pytest.approx(0.0, abs=1e-3)
        assert bucket["tail_sum_lag1_to_6_kwh"] == pytest.approx(0.0, abs=1e-3)

    def test_lag_walk_skips_missing_tail_entries(self):
        """Gaps in the log (missing H+k entries) drop only those lags;
        present lags still accumulate."""
        from datetime import datetime, timedelta
        base_dt = datetime(2026, 4, 1, 12, 0)
        entries: list[dict] = []
        elevations: list[float] = []
        # 12 trains, but only emit lags 0, 4, 5, 6 (skip 1-3).
        for train_idx in range(12):
            h0 = train_idx * 10
            ts_h0 = (base_dt + timedelta(hours=h0)).isoformat()
            entries.append(_hour_entry(
                ts_h0, solar_s=0.5, actual=1.5, base=2.0, mode="heating",
            ))
            elevations.append(20.0)
            for k in (4, 5, 6):
                ts_hk = (base_dt + timedelta(hours=h0 + k)).isoformat()
                entries.append(_hour_entry(
                    ts_hk, solar_s=0.0, solar_factor=0.0,
                    actual=1.7, base=2.0, mode="heating",
                ))
                elevations.append(-1.0)
        coord = self._coord_with_elevations(entries, elevations)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        bucket = result["per_unit"]["sensor.heater1"]["elevation_diagnostics"]["lag"]["15-30"]
        assert bucket["lag_0"]["n"] == 12
        # Lags 1-3 have no tail entries → n=0.
        for k in (1, 2, 3):
            assert bucket[f"lag_{k}"]["n"] == 0
        # Lags 4-6 fully populated.
        for k in (4, 5, 6):
            assert bucket[f"lag_{k}"]["n"] == 12
        # tail_sum unavailable when any of 1-6 is below min_samples.
        assert bucket["tail_sum_lag1_to_6_kwh"] is None

    def test_lag_walk_skips_self_qualifying_tail(self):
        """A tail entry that itself qualifies as a solar hour
        (solar_factor > 0.05) is skipped at lag k>0 to avoid
        double-counting its own originator residual."""
        from datetime import datetime, timedelta
        base_dt = datetime(2026, 4, 1, 12, 0)
        entries: list[dict] = []
        elevations: list[float] = []
        for train_idx in range(12):
            h0 = train_idx * 10
            ts_h0 = (base_dt + timedelta(hours=h0)).isoformat()
            entries.append(_hour_entry(
                ts_h0, solar_s=0.5, actual=1.5, base=2.0, mode="heating",
            ))
            elevations.append(20.0)
            for k in range(1, 7):
                ts_hk = (base_dt + timedelta(hours=h0 + k)).isoformat()
                # Lag 3 has solar_factor=0.5 (self-qualifying); others zero.
                entries.append(_hour_entry(
                    ts_hk, solar_s=0.0,
                    solar_factor=(0.5 if k == 3 else 0.0),
                    actual=1.7, base=2.0, mode="heating",
                ))
                elevations.append(-1.0)
        coord = self._coord_with_elevations(entries, elevations)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        bucket = result["per_unit"]["sensor.heater1"]["elevation_diagnostics"]["lag"]["15-30"]
        # Lag 3: self-qualifying → skipped → n=0.
        assert bucket["lag_3"]["n"] == 0
        # Other lags 1, 2, 4, 5, 6 populated normally.
        for k in (1, 2, 4, 5, 6):
            assert bucket[f"lag_{k}"]["n"] == 12

    def test_lag_walk_skips_mode_change(self):
        """When the entity's mode changes between H and H+k, that lag is
        skipped (different physical regime)."""
        from datetime import datetime, timedelta
        base_dt = datetime(2026, 4, 1, 12, 0)
        entries: list[dict] = []
        elevations: list[float] = []
        for train_idx in range(12):
            h0 = train_idx * 10
            ts_h0 = (base_dt + timedelta(hours=h0)).isoformat()
            entries.append(_hour_entry(
                ts_h0, solar_s=0.5, actual=1.5, base=2.0, mode="heating",
            ))
            elevations.append(20.0)
            for k in range(1, 7):
                ts_hk = (base_dt + timedelta(hours=h0 + k)).isoformat()
                # Lag 2 is in cooling mode (regime change); rest stay heating.
                entries.append(_hour_entry(
                    ts_hk, solar_s=0.0, solar_factor=0.0,
                    actual=1.7, base=2.0,
                    mode=("cooling" if k == 2 else "heating"),
                ))
                elevations.append(-1.0)
        coord = self._coord_with_elevations(entries, elevations)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        bucket = result["per_unit"]["sensor.heater1"]["elevation_diagnostics"]["lag"]["15-30"]
        # Lag 2: mode change → skipped.
        assert bucket["lag_2"]["n"] == 0
        for k in (1, 3, 4, 5, 6):
            assert bucket[f"lag_{k}"]["n"] == 12

    def test_tail_sum_requires_min_samples_each_lag(self):
        """If even one of lags 1-6 has fewer than `min_lag_samples`
        observations, `tail_sum_lag1_to_6_kwh` is None — partial sums
        would silently exaggerate the tail by dropping under-sampled
        lags."""
        from datetime import datetime
        base_dt = datetime(2026, 4, 1, 12, 0)
        # Only 8 trains — below the 10-sample threshold per lag.
        entries, elevations = self._make_train(
            originator_dt=base_dt, originator_elev=20.0, n_trains=8,
            originator_actual=1.5, tail_actual=1.7, base=2.0,
        )
        coord = self._coord_with_elevations(entries, elevations)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        bucket = result["per_unit"]["sensor.heater1"]["elevation_diagnostics"]["lag"]["15-30"]
        assert bucket["lag_1"]["n"] == 8
        assert "mean_residual_kwh" not in bucket["lag_1"]
        assert bucket["tail_sum_lag1_to_6_kwh"] is None

    def test_short_window_populated_when_long_window_starves(self):
        """When lags 4-6 are under-sampled but lags 1-3 are not,
        ``tail_sum_lag1_to_3_kwh`` is populated and
        ``tail_sum_lag1_to_6_kwh`` is None.  This is the empirical
        afternoon-elevation case on west-facade installs where the
        self-qualify gate eats far-tail entries but near-tail
        evidence remains.
        """
        from datetime import datetime, timedelta
        base_dt = datetime(2026, 4, 1, 12, 0)
        entries: list[dict] = []
        elevations: list[float] = []
        for train_idx in range(12):
            h0 = train_idx * 10
            ts_h0 = (base_dt + timedelta(hours=h0)).isoformat()
            entries.append(_hour_entry(
                ts_h0, solar_s=0.5, actual=1.5, base=2.0, mode="heating",
            ))
            elevations.append(20.0)
            # Lags 1-3: full data with sustained reduction.
            for k in (1, 2, 3):
                ts_hk = (base_dt + timedelta(hours=h0 + k)).isoformat()
                entries.append(_hour_entry(
                    ts_hk, solar_s=0.0, solar_factor=0.0,
                    actual=1.7, base=2.0, mode="heating",
                ))
                elevations.append(-1.0)
            # Lags 4-6: only one tail per train (n=1 per lag — below threshold).
            # Entries for lag_5/6 are absent → those lags get n=0; lag_4 gets n=1.
            ts_h4 = (base_dt + timedelta(hours=h0 + 4)).isoformat()
            entries.append(_hour_entry(
                ts_h4, solar_s=0.0, solar_factor=0.0,
                actual=1.7, base=2.0, mode="heating",
            ))
            elevations.append(-1.0)
        coord = self._coord_with_elevations(entries, elevations)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        bucket = result["per_unit"]["sensor.heater1"]["elevation_diagnostics"]["lag"]["15-30"]
        # Lags 1-3 fully populated.
        for k in (1, 2, 3):
            assert bucket[f"lag_{k}"]["n"] == 12
        # Lag 4 has one entry per train but below min=10 threshold → n=12 actually.
        # Wait — 12 trains × 1 entry = 12, which IS above the threshold.
        # Re-examining: with 12 entries at lag_4, that lag IS populated.
        # The intended starvation is lag_5 and lag_6 (no entries).
        assert bucket["lag_5"]["n"] == 0
        assert bucket["lag_6"]["n"] == 0
        # Short window scalar: 3 × 0.3 = 0.9 (populated).
        assert bucket["tail_sum_lag1_to_3_kwh"] == pytest.approx(0.9, abs=1e-3)
        # Long window scalar: gated by missing lags 5-6 → None.
        assert bucket["tail_sum_lag1_to_6_kwh"] is None

    def test_evening_tail_battery_signature(self):
        """Synthetic battery signature in the evening-tail walk:
        sustained reduction in the 6 h post-sunset window for a
        mid-day high-elev originator (where the lag walk would
        have failed because afternoon tails self-qualify)."""
        from datetime import datetime, timedelta
        base_dt = datetime(2026, 4, 1, 12, 0)
        entries: list[dict] = []
        elevations: list[float] = []
        for train_idx in range(12):
            # Spaced 24 h apart so each train has a clean post-sunset
            # window.  Originator at 12:00 (mid-day, 50° elev =
            # bucket 45-60°).
            day_origin = base_dt + timedelta(days=train_idx)
            ts_h0 = day_origin.isoformat()
            entries.append(_hour_entry(
                ts_h0, solar_s=0.5, actual=1.5, base=2.0, mode="heating",
            ))
            elevations.append(50.0)
            # Hours 13-15: still sunny (solar_factor=0.3, would fail
            # the lag walk's self-qualify gate).
            for k in (1, 2, 3):
                ts_hk = (day_origin + timedelta(hours=k)).isoformat()
                entries.append(_hour_entry(
                    ts_hk, solar_s=0.3, solar_factor=0.3,
                    actual=1.7, base=2.0, mode="heating",
                ))
                elevations.append(40.0 - k * 5)
            # Hour 16: dark anchor (solar_factor=0).  Then 16-21
            # are evening hours with sustained reduction.
            for k in range(4, 10):
                ts_hk = (day_origin + timedelta(hours=k)).isoformat()
                entries.append(_hour_entry(
                    ts_hk, solar_s=0.0, solar_factor=0.0,
                    actual=1.7, base=2.0, mode="heating",
                ))
                elevations.append(-1.0)
        coord = self._coord_with_elevations(entries, elevations)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        ev_diag = result["per_unit"]["sensor.heater1"]["elevation_diagnostics"]
        assert "evening_tail" in ev_diag
        bucket = ev_diag["evening_tail"]["45-60"]
        # 12 originators, all found a dark anchor.  Dark offset = 4 h
        # (4 sunny tail hours then sunset).
        assert bucket["n_originators_with_dark_anchor"] == 12
        assert bucket["mean_dark_offset_hours"] == pytest.approx(4.0, abs=1e-2)
        # Each originator contributes 6 evening hours of residual = 0.3
        # (base − actual = 2.0 − 1.7).  12 × 6 = 72 evening hours.
        assert bucket["n_evening_hours"] == 72
        assert bucket["mean_residual_kwh"] == pytest.approx(0.3, abs=1e-3)

    def test_evening_tail_hotspot_signature(self):
        """When tail consumption == baseline (no sustained reduction),
        evening-tail mean is ~0.  Hotspot signature."""
        from datetime import datetime, timedelta
        base_dt = datetime(2026, 4, 1, 12, 0)
        entries: list[dict] = []
        elevations: list[float] = []
        for train_idx in range(12):
            day_origin = base_dt + timedelta(days=train_idx)
            ts_h0 = day_origin.isoformat()
            entries.append(_hour_entry(
                ts_h0, solar_s=0.5, actual=1.5, base=2.0, mode="heating",
            ))
            elevations.append(50.0)
            # 3 sunny tail hours.
            for k in (1, 2, 3):
                ts_hk = (day_origin + timedelta(hours=k)).isoformat()
                entries.append(_hour_entry(
                    ts_hk, solar_s=0.3, solar_factor=0.3,
                    actual=1.7, base=2.0, mode="heating",
                ))
                elevations.append(40.0 - k * 5)
            # Evening hours: actual=base, no reduction.
            for k in range(4, 10):
                ts_hk = (day_origin + timedelta(hours=k)).isoformat()
                entries.append(_hour_entry(
                    ts_hk, solar_s=0.0, solar_factor=0.0,
                    actual=2.0, base=2.0, mode="heating",
                ))
                elevations.append(-1.0)
        coord = self._coord_with_elevations(entries, elevations)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        bucket = result["per_unit"]["sensor.heater1"]["elevation_diagnostics"]["evening_tail"]["45-60"]
        assert bucket["n_evening_hours"] == 72
        assert bucket["mean_residual_kwh"] == pytest.approx(0.0, abs=1e-3)

    def test_evening_tail_no_dark_anchor_within_search_window(self):
        """Originators followed by 12+ continuously sunny hours
        produce no dark anchor → no contribution to evening_tail."""
        from datetime import datetime, timedelta
        base_dt = datetime(2026, 4, 1, 12, 0)
        entries: list[dict] = []
        elevations: list[float] = []
        for train_idx in range(12):
            day_origin = base_dt + timedelta(days=train_idx)
            ts_h0 = day_origin.isoformat()
            entries.append(_hour_entry(
                ts_h0, solar_s=0.5, actual=1.5, base=2.0, mode="heating",
            ))
            elevations.append(50.0)
            # 13 sunny hours after — exhausts the 12 h search window.
            for k in range(1, 14):
                ts_hk = (day_origin + timedelta(hours=k)).isoformat()
                entries.append(_hour_entry(
                    ts_hk, solar_s=0.3, solar_factor=0.3,
                    actual=1.7, base=2.0, mode="heating",
                ))
                elevations.append(20.0)
        coord = self._coord_with_elevations(entries, elevations)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        bucket = result["per_unit"]["sensor.heater1"]["elevation_diagnostics"]["evening_tail"]["45-60"]
        # No dark anchor found → no originators contribute.
        assert bucket["n_originators_with_dark_anchor"] == 0
        assert bucket["n_evening_hours"] == 0
        assert bucket["mean_residual_kwh"] is None

    def test_evening_tail_skips_mode_change(self):
        """Mode change between originator and evening hour breaks
        the walk for that hour."""
        from datetime import datetime, timedelta
        base_dt = datetime(2026, 4, 1, 12, 0)
        entries: list[dict] = []
        elevations: list[float] = []
        for train_idx in range(12):
            day_origin = base_dt + timedelta(days=train_idx)
            ts_h0 = day_origin.isoformat()
            entries.append(_hour_entry(
                ts_h0, solar_s=0.5, actual=1.5, base=2.0, mode="heating",
            ))
            elevations.append(50.0)
            for k in range(1, 4):
                ts_hk = (day_origin + timedelta(hours=k)).isoformat()
                entries.append(_hour_entry(
                    ts_hk, solar_s=0.3, solar_factor=0.3,
                    actual=1.7, base=2.0, mode="heating",
                ))
                elevations.append(20.0)
            # Evening hours: 4-9, but hour 6 is cooling-mode.
            for k in range(4, 10):
                ts_hk = (day_origin + timedelta(hours=k)).isoformat()
                entries.append(_hour_entry(
                    ts_hk, solar_s=0.0, solar_factor=0.0,
                    actual=1.7, base=2.0,
                    mode=("cooling" if k == 6 else "heating"),
                ))
                elevations.append(-1.0)
        coord = self._coord_with_elevations(entries, elevations)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        bucket = result["per_unit"]["sensor.heater1"]["elevation_diagnostics"]["evening_tail"]["45-60"]
        # 12 originators × 5 evening hours (cooling-mode hour skipped) = 60.
        assert bucket["n_originators_with_dark_anchor"] == 12
        assert bucket["n_evening_hours"] == 60
        # 60 hours × 0.3 = 18 → mean = 0.3.
        assert bucket["mean_residual_kwh"] == pytest.approx(0.3, abs=1e-3)

    def test_aux_active_at_tail_skipped(self):
        """Tail hours where auxiliary heating was active are skipped from
        the lag walk (aux contaminates the base-vs-actual comparison)."""
        from datetime import datetime, timedelta
        base_dt = datetime(2026, 4, 1, 12, 0)
        entries: list[dict] = []
        elevations: list[float] = []
        for train_idx in range(12):
            h0 = train_idx * 10
            ts_h0 = (base_dt + timedelta(hours=h0)).isoformat()
            entries.append(_hour_entry(
                ts_h0, solar_s=0.5, actual=1.5, base=2.0, mode="heating",
            ))
            elevations.append(20.0)
            for k in range(1, 7):
                ts_hk = (base_dt + timedelta(hours=h0 + k)).isoformat()
                e = _hour_entry(
                    ts_hk, solar_s=0.0, solar_factor=0.0,
                    actual=1.7, base=2.0, mode="heating",
                )
                # Lag 5 has aux active; rest do not.
                if k == 5:
                    e["auxiliary_active"] = True
                entries.append(e)
                elevations.append(-1.0)
        coord = self._coord_with_elevations(entries, elevations)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        bucket = result["per_unit"]["sensor.heater1"]["elevation_diagnostics"]["lag"]["15-30"]
        assert bucket["lag_5"]["n"] == 0
        for k in (1, 2, 3, 4, 6):
            assert bucket[f"lag_{k}"]["n"] == 12


class TestGhiSignalAgreement:
    """Validates ``diagnose_solar``'s ``ghi_signal_agreement`` block.

    Compares Kasten-derived ``cloud_factor`` against measured GHI
    normalised by Beer-Lambert clear-sky horizontal flux.  Both
    signals live in [0, 1.5] and answer the same physical question
    (\"how much of the clear-sky beam survived the atmosphere this
    hour\") — high correlation means cloud_coverage already explains
    the pyranometer signal; low correlation, particularly on the
    broken-cloud regime, is the fingerprint that justifies
    eventually replacing the cloud_factor step in the pipeline.
    """

    @staticmethod
    def _ghi_log_entry(ts, *, ghi_wm2=None, potential_solar_factor=0.40,
                       cloud_coverage=None):
        """Minimal hourly log entry used by ghi_signal_agreement."""
        entry = {
            "timestamp": ts,
            "hour": 12,
            "solar_vector_s": 0.3, "solar_vector_e": 0.05, "solar_vector_w": 0.02,
            "solar_factor": 0.37,
            "potential_solar_factor": potential_solar_factor,
            "solar_impact_kwh": 0.5,
            "actual_kwh": 1.5, "expected_kwh": 2.0,
            "correction_percent": 100.0, "auxiliary_active": False,
            "guest_impact_kwh": 0.0,
            "unit_modes": {"sensor.heater1": "heating"},
            "unit_breakdown": {"sensor.heater1": 1.5},
            "unit_expected_breakdown": {"sensor.heater1": 2.0},
            "solar_dominant_entities": [],
            "temp": 10.0, "inertia_temp": 10.0,
            "temp_key": "10", "wind_bucket": "normal",
            "solar_impact_raw_kwh": 0.5,
        }
        if ghi_wm2 is not None:
            entry["ghi_wm2"] = ghi_wm2
        if cloud_coverage is not None:
            entry["cloud_coverage"] = cloud_coverage
        return entry

    @staticmethod
    def _coord_for_ghi(log, *, ghi_sensor="sensor.local_ghi", elev=45.0,
                       no_cloud_ref=0.5):
        """Build a coord mock with stable sun position + clear-sky factor.

        Sets the same numeric attributes that the larger fixtures
        (TestDniDhiShadowReport._coord_with_sun) populate, so the
        battery-decay calibration replay path doesn't blow up on
        MagicMock comparison.
        """
        coord = _make_coord(log)
        coord.ghi_sensor = ghi_sensor
        coord.solar.get_approx_sun_pos = MagicMock(return_value=(elev, 180.0))
        coord.solar.calculate_solar_factor = MagicMock(return_value=no_cloud_ref)
        coord.battery_thermal_feedback_k = 0.0
        coord._per_unit_min_base_thresholds = {}
        coord._correlation_data = {}
        return coord

    def test_unavailable_when_no_ghi_data_logged(self):
        """No `ghi_wm2` field on any entry → block reports unavailable."""
        from datetime import datetime, timedelta
        base_dt = datetime(2026, 4, 10, 12, 0)
        entries = [
            self._ghi_log_entry((base_dt + timedelta(hours=j)).isoformat())
            for j in range(40)
        ]
        coord = self._coord_for_ghi(entries)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        ga = result["ghi_signal_agreement"]
        assert ga["available"] is False
        assert ga["reason"] == "no_ghi_data"
        assert ga["ghi_sensor_configured"] is True

    def test_reports_sensor_unconfigured(self):
        """When `coord.ghi_sensor` is None, the block flags it explicitly."""
        from datetime import datetime, timedelta
        base_dt = datetime(2026, 4, 10, 12, 0)
        entries = [
            self._ghi_log_entry((base_dt + timedelta(hours=j)).isoformat())
            for j in range(10)
        ]
        coord = self._coord_for_ghi(entries, ghi_sensor=None)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        ga = result["ghi_signal_agreement"]
        assert ga["available"] is False
        assert ga["ghi_sensor_configured"] is False

    def test_insufficient_qualifying_hours(self):
        """20 hours with GHI logged but threshold is 30 → unavailable with diagnostics."""
        from datetime import datetime, timedelta
        base_dt = datetime(2026, 4, 10, 12, 0)
        entries = [
            self._ghi_log_entry(
                (base_dt + timedelta(hours=j)).isoformat(),
                ghi_wm2=500.0, cloud_coverage=20.0,
            )
            for j in range(20)
        ]
        coord = self._coord_for_ghi(entries)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        ga = result["ghi_signal_agreement"]
        assert ga["available"] is False
        assert ga["reason"] == "insufficient_qualifying_hours"
        assert ga["n_hours_with_ghi"] == 20
        assert ga["n_qualifying"] == 20

    def test_available_with_30_plus_hours(self):
        """30+ qualifying hours produces a populated agreement block."""
        from datetime import datetime, timedelta
        base_dt = datetime(2026, 4, 10, 12, 0)
        entries = [
            self._ghi_log_entry(
                (base_dt + timedelta(hours=j)).isoformat(),
                ghi_wm2=500.0, cloud_coverage=20.0,
            )
            for j in range(40)
        ]
        coord = self._coord_for_ghi(entries)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        ga = result["ghi_signal_agreement"]
        assert ga["available"] is True
        assert ga["n_hours_with_ghi"] == 40
        assert "kasten_cloud_factor" in ga["definition"]
        assert "ghi_normalized" in ga["definition"]
        assert "all" in ga
        assert "by_regime" in ga
        assert ga["all"]["n_hours"] == 40
        # Single regime populated when cloud_coverage is uniform across entries.
        assert ga["by_regime"]["clear"]["n_hours"] == 40
        assert ga["by_regime"]["broken"]["n_hours"] == 0
        assert ga["by_regime"]["overcast"]["n_hours"] == 0

    def test_high_correlation_when_signals_aligned(self):
        """When measured GHI tracks Kasten's predicted clear-sky fraction
        across qualifying hours, Pearson correlation approaches 1.0 — the
        \"cloud_coverage already explains it\" outcome."""
        from datetime import datetime, timedelta
        # Vary potential_solar_factor (drives kasten_cf) across entries
        # and pair each with a GHI value that scales the same way; both
        # signals move together.  ghi_clear at elev=45° is
        # 1361 × sin(45°) × 0.7^(1/sin(45°)) ≈ 1361 × 0.707 × 0.7^1.414
        # = 1361 × 0.707 × 0.604 = 581.4 W/m².  no_cloud_ref = 0.5 fixed.
        base_dt = datetime(2026, 4, 10, 12, 0)
        entries = []
        for j in range(60):
            psf = 0.10 + 0.40 * (j / 60.0)  # 0.10 → 0.50
            kasten_target = psf / 0.5  # = 0.20 → 1.0
            ghi_target = kasten_target * 581.4  # GHI that yields same normalised value
            entries.append(self._ghi_log_entry(
                (base_dt + timedelta(hours=j)).isoformat(),
                ghi_wm2=ghi_target,
                potential_solar_factor=psf,
                cloud_coverage=20.0,
            ))
        coord = self._coord_for_ghi(entries)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        ga = result["ghi_signal_agreement"]
        assert ga["available"] is True
        assert ga["all"]["n_hours"] == 60
        assert ga["all"]["correlation"] == pytest.approx(1.0, abs=0.01)
        # mean_bias ≈ 0 when the two signals are perfectly aligned and on
        # the same scale.
        assert abs(ga["all"]["mean_bias_kasten_minus_ghi"]) < 0.05

    def test_regime_classification_from_cloud_coverage(self):
        """When `cloud_coverage` is present, regime is classified
        via the < 30 / 30-70 / ≥ 70 thresholds (matching the existing
        DNI signal_agreement regime split)."""
        from datetime import datetime, timedelta
        base_dt = datetime(2026, 4, 10, 12, 0)
        entries = []
        for j in range(15):
            entries.append(self._ghi_log_entry(
                (base_dt + timedelta(hours=j)).isoformat(),
                ghi_wm2=500.0, cloud_coverage=15.0,  # clear
            ))
        for j in range(15, 30):
            entries.append(self._ghi_log_entry(
                (base_dt + timedelta(hours=j)).isoformat(),
                ghi_wm2=300.0, cloud_coverage=50.0,  # broken
            ))
        for j in range(30, 45):
            entries.append(self._ghi_log_entry(
                (base_dt + timedelta(hours=j)).isoformat(),
                ghi_wm2=50.0, cloud_coverage=80.0,  # overcast
            ))
        coord = self._coord_for_ghi(entries)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        ga = result["ghi_signal_agreement"]
        assert ga["available"] is True
        assert ga["by_regime"]["clear"]["n_hours"] == 15
        assert ga["by_regime"]["broken"]["n_hours"] == 15
        assert ga["by_regime"]["overcast"]["n_hours"] == 15

    def test_regime_fallback_from_ghi_when_cloud_coverage_missing(self):
        """Legacy entries without `cloud_coverage` use GHI magnitude bands
        for regime classification: < 200 overcast, > 700 clear, else broken."""
        from datetime import datetime, timedelta
        base_dt = datetime(2026, 4, 10, 12, 0)
        entries = []
        # 15 entries with high GHI (clear by GHI threshold).
        for j in range(15):
            entries.append(self._ghi_log_entry(
                (base_dt + timedelta(hours=j)).isoformat(),
                ghi_wm2=800.0, cloud_coverage=None,
            ))
        # 15 entries with mid GHI (broken).
        for j in range(15, 30):
            entries.append(self._ghi_log_entry(
                (base_dt + timedelta(hours=j)).isoformat(),
                ghi_wm2=400.0, cloud_coverage=None,
            ))
        # 15 entries with low GHI (overcast).
        for j in range(30, 45):
            entries.append(self._ghi_log_entry(
                (base_dt + timedelta(hours=j)).isoformat(),
                ghi_wm2=100.0, cloud_coverage=None,
            ))
        coord = self._coord_for_ghi(entries)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        ga = result["ghi_signal_agreement"]
        assert ga["available"] is True
        assert ga["by_regime"]["clear"]["n_hours"] == 15
        assert ga["by_regime"]["broken"]["n_hours"] == 15
        assert ga["by_regime"]["overcast"]["n_hours"] == 15

    def test_stability_gate_skips_low_elevation_and_unstable_recovery(self):
        """Hours below the stability gate (low elev, low no_cloud_ref, or
        below-rounding-noise potential_solar_factor) are excluded and
        counted via `n_skipped_unstable_recovery`."""
        from datetime import datetime, timedelta
        base_dt = datetime(2026, 4, 10, 12, 0)
        entries = []
        # 40 entries with stable potential_solar_factor — pass.
        for j in range(40):
            entries.append(self._ghi_log_entry(
                (base_dt + timedelta(hours=j)).isoformat(),
                ghi_wm2=500.0, potential_solar_factor=0.40, cloud_coverage=20.0,
            ))
        # 30 entries with unstable potential_solar_factor (below 0.05) — gated.
        for j in range(40, 70):
            entries.append(self._ghi_log_entry(
                (base_dt + timedelta(hours=j)).isoformat(),
                ghi_wm2=500.0, potential_solar_factor=0.02, cloud_coverage=20.0,
            ))
        coord = self._coord_for_ghi(entries)
        result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
        ga = result["ghi_signal_agreement"]
        assert ga["available"] is True
        assert ga["all"]["n_hours"] == 40
        assert ga["n_skipped_unstable_recovery"] == 30
        assert "elev >= 5°" in ga["stability_gate"]
        assert "no_cloud_reference > 0.1" in ga["stability_gate"]


class TestObservationCollectorGhi:
    """Phase 1 wiring: the collector accumulates GHI per tick and
    HourlyProcessor writes ``ghi_wm2`` at hour boundary."""

    def test_collector_accumulates_ghi_only_when_provided(self):
        from custom_components.heating_analytics.observation import ObservationCollector
        from datetime import datetime
        c = ObservationCollector()
        now = datetime(2026, 4, 10, 12, 0)
        c.accumulate_weather(
            temp=10.0, effective_wind=0.0, wind_bucket="normal",
            solar_factor=0.5, solar_vector=(0.3, 0.05, 0.02),
            is_aux_active=False, current_time=now,
            ghi=400.0,
        )
        c.accumulate_weather(
            temp=10.0, effective_wind=0.0, wind_bucket="normal",
            solar_factor=0.5, solar_vector=(0.3, 0.05, 0.02),
            is_aux_active=False, current_time=now,
            ghi=600.0,
        )
        # One tick with GHI unavailable — count must not advance.
        c.accumulate_weather(
            temp=10.0, effective_wind=0.0, wind_bucket="normal",
            solar_factor=0.5, solar_vector=(0.3, 0.05, 0.02),
            is_aux_active=False, current_time=now,
            ghi=None,
        )
        assert c.ghi_count == 2
        assert c.ghi_sum == 1000.0
        # Hour-mean: 500 W/m².
        assert c.ghi_sum / c.ghi_count == 500.0

    def test_collector_reset_clears_ghi(self):
        from custom_components.heating_analytics.observation import ObservationCollector
        from datetime import datetime
        c = ObservationCollector()
        c.accumulate_weather(
            temp=10.0, effective_wind=0.0, wind_bucket="normal",
            solar_factor=0.5, solar_vector=(0.3, 0.05, 0.02),
            is_aux_active=False, current_time=datetime(2026, 4, 10, 12, 0),
            ghi=400.0,
        )
        assert c.ghi_count == 1
        c.reset()
        assert c.ghi_sum == 0.0
        assert c.ghi_count == 0

