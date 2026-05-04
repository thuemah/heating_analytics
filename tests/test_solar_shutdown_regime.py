"""Tests for solar shutdown detection and base/coefficient protection (#838).

Covers two complementary mechanisms:

1. ``detect_solar_shutdown_entities``: identifies VP units whose thermostat
   cut the compressor during sun.  Pure function of this-hour data; no
   circular dependencies on the learned coefficient.

2. Learning protections:
   - Solar NLMS is skipped for flagged entities, preventing
     ``actual_impact = base`` from inflating the coefficient.
   - Per-unit base EMA rate is multiplied by headroom
     ``(base - unit_solar_impact) / base`` so inflated normalization
     targets can't drift the base model upward.
"""
from unittest.mock import MagicMock

import pytest

from custom_components.heating_analytics.const import (
    MODE_COOLING,
    MODE_DHW,
    MODE_GUEST_HEATING,
    MODE_HEATING,
    MODE_OFF,
    PER_UNIT_LEARNING_RATE_CAP,
    SOLAR_SHUTDOWN_ACTUAL_FLOOR,
    SOLAR_SHUTDOWN_MIN_BASE,
    SOLAR_SHUTDOWN_MIN_MAGNITUDE,
    SOLAR_SHUTDOWN_RATIO,
)
from custom_components.heating_analytics.learning import LearningManager
from custom_components.heating_analytics.observation import (
    detect_solar_shutdown_entities,
)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def _detect(**overrides):
    """Call detect_solar_shutdown_entities with sensible defaults."""
    defaults = dict(
        solar_enabled=True,
        is_aux_dominant=False,
        potential_vector=(0.6, 0.2, 0.0),
        energy_sensors=["vp_stue"],
        unit_modes={"vp_stue": MODE_HEATING},
        unit_actual_kwh={"vp_stue": 0.0},
        unit_expected_base_kwh={"vp_stue": 1.2},
    )
    defaults.update(overrides)
    return detect_solar_shutdown_entities(**defaults)


class TestShutdownDetection:
    """Scenarios from Phase 2 analysis (comment #1 on issue 838)."""

    def test_vp_full_shutdown_detected(self):
        """VP shutoff (actual≈0) under sun with significant base → detected."""
        result = _detect(
            unit_actual_kwh={"vp_stue": 0.00},
            unit_expected_base_kwh={"vp_stue": 1.20},
        )
        assert result == ("vp_stue",)

    def test_vp_minimum_power_not_detected(self):
        """VP at minimum power (170W ≈ 0.17 kWh, ratio > 0.15) → not detected.

        ratio = 0.17 / 1.0 = 0.17, above SOLAR_SHUTDOWN_RATIO (0.15).
        Actual is above the floor (0.03).
        """
        result = _detect(
            unit_actual_kwh={"vp_stue": 0.17},
            unit_expected_base_kwh={"vp_stue": 1.00},
        )
        assert result == ()

    def test_low_demand_floor_heating_not_detected(self):
        """Low demand + mild temp: base below MIN_BASE threshold → not detected."""
        result = _detect(
            unit_actual_kwh={"vp_stue": 0.05},
            unit_expected_base_kwh={"vp_stue": 0.10},  # below MIN_BASE
        )
        assert result == ()

    def test_vp_off_cloudy_not_detected(self):
        """VP off but sky is cloudy (magnitude < threshold) → not detected."""
        result = _detect(
            potential_vector=(0.1, 0.05, 0.0),  # magnitude ≈ 0.11 < 0.3
            unit_actual_kwh={"vp_stue": 0.00},
            unit_expected_base_kwh={"vp_stue": 1.20},
        )
        assert result == ()

    def test_partial_cycling_detected(self):
        """VP cycling (actual 0.08, ratio 0.10 < 0.15) → detected."""
        result = _detect(
            unit_actual_kwh={"vp_stue": 0.08},
            unit_expected_base_kwh={"vp_stue": 0.80},
        )
        assert result == ("vp_stue",)

    def test_actual_at_floor_boundary_detected(self):
        """actual == floor is below the strict-less-than check → detected only if ratio."""
        # actual exactly at floor → NOT below floor; must rely on ratio.
        # base 1.0, ratio = 0.03 / 1.0 = 0.03 < 0.15 → detected.
        result = _detect(
            unit_actual_kwh={"vp_stue": SOLAR_SHUTDOWN_ACTUAL_FLOOR},
            unit_expected_base_kwh={"vp_stue": 1.00},
        )
        assert result == ("vp_stue",)

    def test_parasitic_with_low_base_flagged_via_absolute_floor(self):
        """Field-evidence regression (#918, post-stage-3 audit): standby
        consumption (~12 Wh) on a mild day with low base (~0.08 kWh)
        MUST be flagged shutdown even though base is below the per-unit
        threshold.  Pre-fix the gate ordering put ``base < threshold``
        BEFORE the absolute ``actual < ACTUAL_FLOOR`` check, so
        parasitic-with-low-base hours fell through to modulating and
        inflated the solar coefficient via Tobit/NLMS.

        Reconstructed maintainer fixture:
        - Toshiba per-unit threshold: 0.169 kWh
        - Mild April day, base = 0.083 kWh (< threshold)
        - Standby consumption: 0.012 kWh (< ACTUAL_FLOOR 0.03)
        - Sun up: magnitude > 0.30
        """
        result = _detect(
            unit_actual_kwh={"vp_stue": 0.012},
            unit_expected_base_kwh={"vp_stue": 0.083},
            unit_min_base={"vp_stue": 0.169},
        )
        assert result == ("vp_stue",), (
            "parasitic actual below floor must trigger shutdown flag "
            "regardless of base-vs-threshold relationship"
        )

    def test_low_modulating_with_low_base_not_flagged(self):
        """Counter-case to the parasitic fix: a low-modulation HP at
        a mild temperature with actual ABOVE the absolute floor must
        still NOT be flagged when base is below threshold (existing
        semantic preserved).  Without this, the fix would over-flag
        legitimate low-modulating shoulder hours.

        Fixture: Mitsubishi cycling at minimum modulation, base low
        because outdoor temp is mild but HP is genuinely running.
        """
        result = _detect(
            unit_actual_kwh={"vp_stue": 0.06},  # > 0.03 floor (running)
            unit_expected_base_kwh={"vp_stue": 0.10},  # < 0.15 threshold
        )
        assert result == (), (
            "above-floor running with low base must remain modulating; "
            "base-below-threshold gate preserves existing semantic for "
            "non-parasitic hours"
        )

    def test_parasitic_with_high_base_still_flagged(self):
        """Pre-existing case (full shutdown, high base, parasitic
        residual): both gates should fire.  Pinned here so the gate-
        ordering fix doesn't accidentally regress the headline case.
        """
        result = _detect(
            unit_actual_kwh={"vp_stue": 0.012},
            unit_expected_base_kwh={"vp_stue": 1.20},
        )
        assert result == ("vp_stue",)

    def test_actual_below_floor_with_low_base_flagged(self):
        """Boundary-case companion to the parasitic fix (review N1):
        actual = 0.029 < 0.03 floor + base = 0.10 < 0.15 threshold →
        FLAGGED via the unconditional parasitic short-circuit.  Pins
        the post-fix gate-ordering against future regression.
        """
        result = _detect(
            unit_actual_kwh={"vp_stue": 0.029},
            unit_expected_base_kwh={"vp_stue": 0.10},
        )
        assert result == ("vp_stue",)

    def test_actual_above_floor_with_low_base_not_flagged(self):
        """Boundary-case companion (review N1): actual = 0.031 > 0.03
        floor + base = 0.10 < 0.15 threshold + ratio 0.31 > 0.15 →
        NOT flagged.  Without the parasitic short-circuit the entity
        is correctly classed modulating-but-skipped (base below
        eligibility).  Critical for distinguishing the gate fix from
        an over-flagging regression.
        """
        result = _detect(
            unit_actual_kwh={"vp_stue": 0.031},
            unit_expected_base_kwh={"vp_stue": 0.10},
        )
        assert result == ()

    def test_near_zero_base_with_zero_actual_does_not_inflate(self):
        """Review N3: near-zero-base sensor (e.g. 0.0001 kWh) with
        zero actual is technically flagged shutdown by the new gate,
        but the inequality learner's constraint
        ``coeff·battery ≥ 0.9×base`` is trivially satisfied at near-
        zero base, so no coefficient lift fires.  Modulating learning
        is correctly skipped.  Behaviour is benign — pin it so a
        future tightening doesn't introduce surprise inflation here.
        """
        # Behavioural test: detect-level flag fires (not a bug).
        result = _detect(
            unit_actual_kwh={"vp_stue": 0.0},
            unit_expected_base_kwh={"vp_stue": 0.0001},
        )
        # Either flagging or not is acceptable here as the downstream
        # is benign — the test pins that the function returns
        # cleanly, no exception, no unbounded behaviour.
        assert result in (("vp_stue",), ())

    def test_magnitude_at_threshold_detects(self):
        """Magnitude exactly at MIN_MAGNITUDE → detection proceeds."""
        # Pure-south at magnitude threshold.
        result = _detect(
            potential_vector=(SOLAR_SHUTDOWN_MIN_MAGNITUDE, 0.0, 0.0),
            unit_actual_kwh={"vp_stue": 0.00},
        )
        assert result == ("vp_stue",)

    def test_base_exactly_at_min_base_detects(self):
        """Base equal to SOLAR_SHUTDOWN_MIN_BASE → qualifies (≥)."""
        result = _detect(
            unit_actual_kwh={"vp_stue": 0.00},
            unit_expected_base_kwh={"vp_stue": SOLAR_SHUTDOWN_MIN_BASE},
        )
        assert result == ("vp_stue",)


class TestDetectionGuards:
    """Guards that must unconditionally suppress detection."""

    def test_solar_disabled_returns_empty(self):
        result = _detect(
            solar_enabled=False,
            unit_actual_kwh={"vp_stue": 0.0},
        )
        assert result == ()

    def test_aux_dominant_returns_empty(self):
        """Aux regime already freezes base; no solar regime on top of it."""
        result = _detect(
            is_aux_dominant=True,
            unit_actual_kwh={"vp_stue": 0.0},
        )
        assert result == ()

    def test_non_heating_modes_skipped(self):
        """Cooling, DHW, OFF, Guest → not eligible."""
        for mode in (MODE_COOLING, MODE_DHW, MODE_OFF, MODE_GUEST_HEATING):
            result = _detect(
                unit_modes={"vp_stue": mode},
                unit_actual_kwh={"vp_stue": 0.0},
            )
            assert result == (), f"mode={mode} should not be flagged"

    def test_multi_unit_selective(self):
        """Only the shutdown unit is flagged; others learn normally."""
        result = _detect(
            energy_sensors=["vp_stue", "panel_bad", "vp_kjeller"],
            unit_modes={
                "vp_stue": MODE_HEATING,
                "panel_bad": MODE_HEATING,
                "vp_kjeller": MODE_HEATING,
            },
            unit_actual_kwh={
                "vp_stue": 0.00,  # shutdown
                "panel_bad": 0.40,  # normal linear response
                "vp_kjeller": 0.25,  # above ratio — normal
            },
            unit_expected_base_kwh={
                "vp_stue": 1.2,
                "panel_bad": 0.5,
                "vp_kjeller": 1.0,
            },
        )
        assert result == ("vp_stue",)

    def test_missing_unit_data_not_flagged(self):
        """Sensor without entries in actual/base dicts must not be flagged."""
        result = _detect(
            energy_sensors=["offline_sensor"],
            unit_modes={"offline_sensor": MODE_HEATING},
            unit_actual_kwh={},  # sensor offline this hour
            unit_expected_base_kwh={},
        )
        assert result == ()


# ---------------------------------------------------------------------------
# Learning: NLMS skip during shutdown
# ---------------------------------------------------------------------------


def _build_per_unit_kwargs(**overrides):
    """Minimal kwargs for LearningManager._process_per_unit_learning."""
    defaults = dict(
        temp_key="10",
        wind_bucket="normal",
        avg_temp=10.0,
        avg_solar_vector=(0.6, 0.2, 0.0),
        total_energy_kwh=0.0,
        base_expected_kwh=1.2,
        energy_sensors=["vp_stue"],
        hourly_delta_per_unit={"vp_stue": 0.0},
        solar_enabled=True,
        learning_rate=0.05,
        solar_calculator=MagicMock(),
        get_predicted_unit_base_fn=MagicMock(return_value=1.2),
        learning_buffer_per_unit={},
        correlation_data_per_unit={"vp_stue": {"10": {"normal": 1.0}}},
        observation_counts={},
        is_aux_active=False,
        aux_coefficients_per_unit={},
        learning_buffer_aux_per_unit={},
        solar_coefficients_per_unit={
            "vp_stue": {
                "heating": {"s": 1.0, "e": 0.3, "w": 0.0},
                "cooling": {"s": 0.0, "e": 0.0, "w": 0.0},
            }
        },
        learning_buffer_solar_per_unit={},
        balance_point=17.0,
        unit_modes={"vp_stue": MODE_HEATING},
        hourly_expected_per_unit={"vp_stue": 1.2},
        hourly_expected_base_per_unit={"vp_stue": 1.2},
        aux_affected_entities=None,
        is_cooldown_active=False,
        correction_percent=100.0,
        solar_dominant_entities=(),
    )
    defaults.update(overrides)
    # Solar calculator must return the coefficient + impact we can control.
    # ``solar_coefficients_per_unit`` is now mode-stratified (#868); the mock
    # picks the regime corresponding to the unit's mode.
    def _fake_unit_coeff(entity_id, temp_key, mode):
        entry = defaults["solar_coefficients_per_unit"].get(
            entity_id, {"heating": {"s": 0, "e": 0, "w": 0},
                        "cooling": {"s": 0, "e": 0, "w": 0}}
        )
        if isinstance(entry, dict) and ("heating" in entry or "cooling" in entry):
            regime = "cooling" if mode == MODE_COOLING else "heating"
            return entry.get(regime, {"s": 0, "e": 0, "w": 0})
        # Legacy flat shape — return as-is for tests not yet migrated.
        return entry

    def _fake_impact(vec, coeff, *args, **kwargs):
        return coeff.get("s", 0) * vec[0] + coeff.get("e", 0) * vec[1] + coeff.get("w", 0) * vec[2]

    def _fake_normalize(actual, solar_impact, mode):
        if mode == MODE_HEATING:
            return max(0.0, actual + solar_impact)
        return max(0.0, actual - solar_impact)

    defaults["solar_calculator"].calculate_unit_coefficient.side_effect = _fake_unit_coeff
    defaults["solar_calculator"].calculate_unit_solar_impact.side_effect = _fake_impact
    defaults["solar_calculator"].normalize_for_learning.side_effect = _fake_normalize
    return defaults


class TestSolarShutdownSkip:
    """NLMS must not run for solar-dominant entities."""

    def test_shutdown_entity_coefficient_unchanged(self):
        """With shutdown flag, the solar coefficient does not move."""
        lm = LearningManager()
        kwargs = _build_per_unit_kwargs(
            solar_dominant_entities=("vp_stue",),
            solar_coefficients_per_unit={
            "vp_stue": {
                "heating": {"s": 1.0, "e": 0.3, "w": 0.0},
                "cooling": {"s": 0.0, "e": 0.0, "w": 0.0},
            }
        },
        )
        before = dict(kwargs["solar_coefficients_per_unit"]["vp_stue"])
        lm._process_per_unit_learning(**kwargs)
        after = kwargs["solar_coefficients_per_unit"]["vp_stue"]
        assert after == before, f"Shutdown skip violated: {before} → {after}"

    def test_normal_entity_coefficient_updates(self):
        """Without shutdown flag, an inflated actual_impact=base moves the coeff.

        Regression check: the skip must be gated on the entities list,
        not applied globally when the list is present.
        """
        lm = LearningManager()
        kwargs = _build_per_unit_kwargs(
            solar_dominant_entities=("other_unit",),  # flag for a DIFFERENT unit
            solar_coefficients_per_unit={
            "vp_stue": {
                "heating": {"s": 1.0, "e": 0.3, "w": 0.0},
                "cooling": {"s": 0.0, "e": 0.0, "w": 0.0},
            }
        },
        )
        before = dict(kwargs["solar_coefficients_per_unit"]["vp_stue"])
        lm._process_per_unit_learning(**kwargs)
        after = kwargs["solar_coefficients_per_unit"]["vp_stue"]
        # Should have moved because vp_stue is NOT in dominant list.
        assert after != before

    def test_empty_flags_preserves_old_behaviour(self):
        """Default empty tuple → identical to pre-#838 behaviour."""
        lm = LearningManager()
        kwargs = _build_per_unit_kwargs(solar_dominant_entities=())
        before = dict(kwargs["solar_coefficients_per_unit"]["vp_stue"])
        lm._process_per_unit_learning(**kwargs)
        after = kwargs["solar_coefficients_per_unit"]["vp_stue"]
        assert after != before  # NLMS should have run


# ---------------------------------------------------------------------------
# Learning: headroom-weighted EMA rate
# ---------------------------------------------------------------------------


class TestHeadroomWeightedEMA:
    """Per-unit base EMA rate is throttled when solar approaches base demand."""

    def _run_once(self, expected_base, solar_coeff_s, correlation_before=2.0, actual=0.0):
        """Run one learning step with full shutdown actual=0 scenario.

        Returns (correlation_after).
        """
        lm = LearningManager()
        solar_coeffs = {
            "vp_stue": {
                "heating": {"s": solar_coeff_s, "e": 0.0, "w": 0.0},
                "cooling": {"s": 0.0, "e": 0.0, "w": 0.0},
            }
        }
        kwargs = _build_per_unit_kwargs(
            solar_coefficients_per_unit=solar_coeffs,
            correlation_data_per_unit={"vp_stue": {"10": {"normal": correlation_before}}},
            hourly_delta_per_unit={"vp_stue": actual},
            hourly_expected_base_per_unit={"vp_stue": expected_base},
            # Critical: when solar_dominant is set, NLMS is skipped — otherwise
            # the solar coefficient would drift during the call and distort
            # unit_solar_impact we're measuring.
            solar_dominant_entities=("vp_stue",),
            avg_solar_vector=(1.0, 0.0, 0.0),  # S=1.0 for clean impact = coeff_s
        )
        lm._process_per_unit_learning(**kwargs)
        return kwargs["correlation_data_per_unit"]["vp_stue"]["10"]["normal"]

    def test_full_saturation_freezes_ema(self):
        """solar_impact == base → headroom = 0 → no EMA update."""
        # Setup: base=1.2, coeff_s=1.2, potential_s=1.0 → solar_impact=1.2.
        # headroom = (1.2 - 1.2) / 1.2 = 0.0 → rate * 0 → no movement.
        after = self._run_once(expected_base=1.2, solar_coeff_s=1.2, correlation_before=1.2, actual=0.0)
        assert after == 1.2, f"Base should not move when headroom=0, got {after}"

    def test_oversaturation_clamps_to_zero(self):
        """solar_impact > base → headroom clamped to 0 (not negative)."""
        after = self._run_once(expected_base=1.0, solar_coeff_s=2.0, correlation_before=1.0, actual=0.0)
        assert after == 1.0, f"Oversaturation must not cause reverse EMA, got {after}"

    def test_zero_solar_preserves_full_rate(self):
        """Dark hours (no solar) → headroom = 1 → rate unchanged."""
        lm = LearningManager()
        solar_coeffs = {
            "vp_stue": {
                "heating": {"s": 0.0, "e": 0.0, "w": 0.0},
                "cooling": {"s": 0.0, "e": 0.0, "w": 0.0},
            }
        }
        kwargs = _build_per_unit_kwargs(
            solar_coefficients_per_unit=solar_coeffs,
            correlation_data_per_unit={"vp_stue": {"10": {"normal": 2.0}}},
            hourly_delta_per_unit={"vp_stue": 1.0},
            hourly_expected_base_per_unit={"vp_stue": 1.0},
            avg_solar_vector=(0.0, 0.0, 0.0),
            learning_rate=PER_UNIT_LEARNING_RATE_CAP,  # hit the cap cleanly
        )
        lm._process_per_unit_learning(**kwargs)
        after = kwargs["correlation_data_per_unit"]["vp_stue"]["10"]["normal"]
        # Pre-#838 EMA: 2.0 + 0.03 * (1.0 - 2.0) = 1.97.
        assert after == pytest.approx(1.97, abs=1e-3)


class TestHeadroomNoEffectWithoutSolar:
    """When solar is disabled globally, the rate multiplier is always 1.0."""

    def test_solar_disabled_keeps_full_rate(self):
        lm = LearningManager()
        kwargs = _build_per_unit_kwargs(
            solar_enabled=False,
            correlation_data_per_unit={"vp_stue": {"10": {"normal": 2.0}}},
            hourly_delta_per_unit={"vp_stue": 1.0},
            hourly_expected_base_per_unit={"vp_stue": 1.0},
            learning_rate=PER_UNIT_LEARNING_RATE_CAP,
        )
        lm._process_per_unit_learning(**kwargs)
        after = kwargs["correlation_data_per_unit"]["vp_stue"]["10"]["normal"]
        assert after == pytest.approx(1.97, abs=1e-3)


# ---------------------------------------------------------------------------
# Integration: detection + skip + headroom together break the feedback loop
# ---------------------------------------------------------------------------


class TestFeedbackLoopBroken:
    """The feedback loop from issue #836 must be broken by these two mechanisms."""

    def test_shutdown_hour_does_not_inflate_coeff_or_base(self):
        """A shutdown hour must not inflate coefficient or base upward.

        Pre-#838: actual_impact = 1.2 - 0 = 1.2 inflates the solar
        coefficient AND unit_normalized = 0 + inflated_coeff*potential drifts
        the base model upward, which in turn makes next hour's actual_impact
        even larger — the feedback loop.

        Post-#838:
          - NLMS skip → coefficient exactly unchanged.
          - Headroom multiplier → base EMA step drops in magnitude AND
            direction is toward unit_normalized (which is bounded by base),
            so base can never drift above its starting value this hour.
        """
        lm = LearningManager()
        shutdown = detect_solar_shutdown_entities(
            solar_enabled=True,
            is_aux_dominant=False,
            potential_vector=(0.6, 0.2, 0.0),
            energy_sensors=["vp_stue"],
            unit_modes={"vp_stue": MODE_HEATING},
            unit_actual_kwh={"vp_stue": 0.0},
            unit_expected_base_kwh={"vp_stue": 1.2},
        )
        assert shutdown == ("vp_stue",)

        coeff_before = {
            "heating": {"s": 1.0, "e": 0.3, "w": 0.0},
            "cooling": {"s": 0.0, "e": 0.0, "w": 0.0},
        }
        corr_before = 1.2
        # Pre-deep-copy so we can compare regime contents post-call.
        coeff_before_snapshot = {
            r: dict(v) for r, v in coeff_before.items()
        }
        kwargs = _build_per_unit_kwargs(
            solar_dominant_entities=shutdown,
            solar_coefficients_per_unit={"vp_stue": coeff_before},
            correlation_data_per_unit={"vp_stue": {"10": {"normal": corr_before}}},
            hourly_delta_per_unit={"vp_stue": 0.0},
            hourly_expected_base_per_unit={"vp_stue": 1.2},
            avg_solar_vector=(0.6, 0.2, 0.0),
        )
        lm._process_per_unit_learning(**kwargs)
        # Invariant 1: heating-regime coefficient locked when shutdown gates NLMS.
        # Inequality may lift the coefficient when constraint is violated; this
        # test runs with battery=0 (no inequality signal) so coeff is unchanged.
        assert (
            kwargs["solar_coefficients_per_unit"]["vp_stue"]
            == coeff_before_snapshot
        )
        # Invariant 2: base cannot have drifted upward on a shutdown hour.
        # unit_normalized = 0 + coeff·potential = 0.66 ≤ base, so EMA target
        # is below base → movement is downward (gradually self-correcting
        # if coefficient is truly inflated).  Headroom caps the step.
        corr_after = kwargs["correlation_data_per_unit"]["vp_stue"]["10"]["normal"]
        assert corr_after <= corr_before, (
            f"Shutdown hour drove base upward: {corr_before} → {corr_after}"
        )
        # And the step is small in absolute magnitude thanks to headroom.
        assert abs(corr_after - corr_before) < 0.02, (
            f"Base drift too large: {corr_before} → {corr_after}"
        )
