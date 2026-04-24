"""Tests for NLMS-replay + 3-pass EM-lite in retrain_from_history (#847).

Verifies that:

1. ``LearningManager.replay_solar_nlms`` iterates qualifying sunny hours from
   a historical log, reconstructs potential vectors using the canonical
   per-direction transmittance (#826), and updates solar coefficients via
   the same NLMS path used by live learning.

2. ``LearningManager.compute_on_the_fly_solar_delta`` recomputes
   ``solar_normalization_delta`` for an hour using CURRENT coefficients,
   signing by unit mode (heating positive, cooling negative, DHW/OFF zero).

3. ``retrain_from_history`` Track A with ``reset_first=True`` runs the
   three-pass EM-lite: base priming → NLMS replay → base refinement.
   The priming pass uses solar_norm=0 (avoiding stored-delta contamination);
   the refinement pass uses on-the-fly delta from the refreshed coefficients.

4. ``retrain_from_history`` Track A with ``reset_first=False`` keeps the
   existing single-pass behaviour and appends NLMS replay at the end so
   post-retrain coefficients are refreshed against the final base.

5. Response payload reports ``solar_replay_updates`` and ``em_passes``.
"""
from unittest.mock import MagicMock, AsyncMock

import pytest

from custom_components.heating_analytics.learning import LearningManager
from custom_components.heating_analytics.observation import DirectMeter, WeightedSmear
from custom_components.heating_analytics.solar import SolarCalculator
from custom_components.heating_analytics.const import (
    SOLAR_LEARNING_MIN_BASE,
    MODE_HEATING,
    MODE_COOLING,
    MODE_DHW,
)
from custom_components.heating_analytics.retrain import RetrainEngine


# -----------------------------------------------------------------------------
# Unit-level: compute_on_the_fly_solar_delta
# -----------------------------------------------------------------------------

class TestOnTheFlyDelta:
    """Recompute solar_normalization_delta from current coefficients."""

    def _solar_calc(self):
        coord = MagicMock()
        coord.screen_config = (True, True, True)
        return SolarCalculator(coord)

    def test_heating_mode_positive_delta(self):
        """Heating unit with solar gain → positive delta.

        Note: log ``unit_modes`` for a heating unit is *empty* by design
        (coordinator.py filters MODE_HEATING out).  The function must
        resolve heating via the default when a sensor is absent from
        ``unit_modes``.  Test reflects this by leaving unit_modes={}.
        """
        lm = LearningManager()
        entry = {
            "correction_percent": 100.0,
            "solar_vector_s": 0.5,
            "solar_vector_e": 0.0,
            "solar_vector_w": 0.0,
            "unit_modes": {},  # matches real log shape for all-heating hour
        }
        coeffs = {"sensor.heater1": {"s": 1.0, "e": 0.0, "w": 0.0}}
        delta = lm.compute_on_the_fly_solar_delta(
            entry,
            solar_calculator=self._solar_calc(),
            screen_config=(True, True, True),
            solar_coefficients_per_unit=coeffs,
            energy_sensors=["sensor.heater1"],
        )
        assert delta == pytest.approx(0.5)

    def test_regression_empty_unit_modes_does_not_zero_delta(self):
        """Explicit regression for the B1 bug: pre-fix, an all-heating
        hour (empty unit_modes in log) returned 0 and silently neutralised
        Pass 3 refinement.  Post-fix, the function iterates energy_sensors
        and the coefficient is applied."""
        lm = LearningManager()
        entry = {
            "correction_percent": 100.0,
            "solar_vector_s": 0.4,
            "solar_vector_e": 0.1,
            "solar_vector_w": 0.0,
            "unit_modes": {},  # all-heating hour
        }
        coeffs = {"sensor.heater1": {"s": 0.8, "e": 0.0, "w": 0.0}}
        delta = lm.compute_on_the_fly_solar_delta(
            entry,
            solar_calculator=self._solar_calc(),
            screen_config=(True, True, True),
            solar_coefficients_per_unit=coeffs,
            energy_sensors=["sensor.heater1"],
        )
        assert delta > 0.0, "B1 regression: empty unit_modes must not zero out heating delta"
        assert delta == pytest.approx(0.8 * 0.4)

    def test_cooling_mode_negates_contribution(self):
        """Cooling unit → impact subtracted from delta (inverted semantics)."""
        lm = LearningManager()
        entry = {
            "correction_percent": 100.0,
            "solar_vector_s": 0.5,
            "solar_vector_e": 0.0,
            "solar_vector_w": 0.0,
            "unit_modes": {"sensor.heater1": MODE_COOLING},
        }
        coeffs = {"sensor.heater1": {"s": 1.0, "e": 0.0, "w": 0.0}}
        delta = lm.compute_on_the_fly_solar_delta(
            entry,
            solar_calculator=self._solar_calc(),
            screen_config=(True, True, True),
            solar_coefficients_per_unit=coeffs,
            energy_sensors=["sensor.heater1"],
        )
        # Cooling: impact = 0.5 but subtracted (sun adds demand in cooling)
        assert delta == pytest.approx(-0.5)

    def test_dhw_off_modes_contribute_zero(self):
        """DHW / OFF modes contribute nothing to delta."""
        lm = LearningManager()
        entry = {
            "correction_percent": 100.0,
            "solar_vector_s": 0.5,
            "solar_vector_e": 0.0,
            "solar_vector_w": 0.0,
            "unit_modes": {"sensor.heater1": MODE_DHW},
        }
        coeffs = {"sensor.heater1": {"s": 1.0}}
        delta = lm.compute_on_the_fly_solar_delta(
            entry,
            solar_calculator=self._solar_calc(),
            screen_config=(True, True, True),
            solar_coefficients_per_unit=coeffs,
            energy_sensors=["sensor.heater1"],
        )
        assert delta == pytest.approx(0.0)

    def test_missing_coefficient_skips_unit(self):
        """Units without learned coeffs contribute zero to delta."""
        lm = LearningManager()
        entry = {
            "correction_percent": 100.0,
            "solar_vector_s": 0.5,
            "solar_vector_e": 0.0,
            "solar_vector_w": 0.0,
            "unit_modes": {},  # both heating (log omits heating)
        }
        coeffs = {"sensor.heater1": {"s": 1.0}}  # heater2 missing
        delta = lm.compute_on_the_fly_solar_delta(
            entry,
            solar_calculator=self._solar_calc(),
            screen_config=(True, True, True),
            solar_coefficients_per_unit=coeffs,
            energy_sensors=["sensor.heater1", "sensor.heater2"],
        )
        # Only heater1 contributes (0.5); heater2 skipped (no coeff)
        assert delta == pytest.approx(0.5)

    def test_zero_solar_returns_zero_delta(self):
        """Dark hour (zero vector) → zero delta regardless of coefficients."""
        lm = LearningManager()
        entry = {
            "correction_percent": 100.0,
            "solar_vector_s": 0.0,
            "solar_vector_e": 0.0,
            "solar_vector_w": 0.0,
            "unit_modes": {},
        }
        coeffs = {"sensor.heater1": {"s": 5.0, "e": 5.0, "w": 5.0}}
        delta = lm.compute_on_the_fly_solar_delta(
            entry,
            solar_calculator=self._solar_calc(),
            screen_config=(True, True, True),
            solar_coefficients_per_unit=coeffs,
            energy_sensors=["sensor.heater1"],
        )
        assert delta == pytest.approx(0.0)


# -----------------------------------------------------------------------------
# Unit-level: replay_solar_nlms
# -----------------------------------------------------------------------------

class TestReplaySolarNLMS:
    """NLMS replay iterates historical hours and updates coefficients."""

    def _build_env(self, unit_base_kwh=1.0):
        coord = MagicMock()
        coord.screen_config = (True, True, True)
        calc = SolarCalculator(coord)
        return calc

    def test_converges_toward_true_coefficient(self):
        """A log of sunny hours with known true coefficient produces NLMS
        updates that pull the solar coeff toward that truth."""
        lm = LearningManager()
        calc = self._build_env()
        # 20 sunny hours: potential_s varying, implied_solar = 1.0 × potential_s
        true_coeff = 1.0
        base_kwh = 1.0  # per-unit base
        entries = []
        from datetime import datetime, timedelta
        base_dt = datetime(2026, 4, 15, 10, 0)
        for i in range(20):
            pot_s = 0.4 + (i % 5) * 0.1  # 0.4..0.8
            impact = true_coeff * pot_s
            entries.append({
                "timestamp": (base_dt + timedelta(hours=i)).isoformat(),
                "temp_key": "10",
                "wind_bucket": "normal",
                "correction_percent": 100.0,  # t=1.0 → pot == eff
                "solar_vector_s": pot_s,
                "solar_vector_e": 0.05 + (i % 3) * 0.02,
                "solar_vector_w": 0.02,
                "auxiliary_active": False,
                "temp": 10.0,
                "unit_modes": {"sensor.heater1": MODE_HEATING},
                "unit_breakdown": {"sensor.heater1": max(0.0, base_kwh - impact)},
                "solar_dominant_entities": [],
            })

        correlation_per_unit = {"sensor.heater1": {"10": {"normal": base_kwh}}}
        solar_coeffs = {}
        solar_buffers = {}

        updates = lm.replay_solar_nlms(
            entries,
            solar_calculator=calc,
            screen_config=(True, True, True),
            correlation_data_per_unit=correlation_per_unit,
            solar_coefficients_per_unit=solar_coeffs,
            learning_buffer_solar_per_unit=solar_buffers,
            energy_sensors=["sensor.heater1"],
            learning_rate=0.1,
            balance_point=15.0,
        )
        assert updates >= 10
        assert "sensor.heater1" in solar_coeffs
        learned_s = solar_coeffs["sensor.heater1"].get("s", 0.0)
        # NLMS with 20 hours should land within ~30% of truth (buffered
        # cold-start damps the initial 4-sample LS by 0.75 and then NLMS
        # closes the gap; exact convergence depends on sample spread)
        assert 0.6 <= learned_s <= 1.3

    def test_aux_affected_entities_list_does_not_block_solar_learning(self):
        """aux_affected_entities is NOT a solar-NLMS exclusion list.

        Regression guard for an earlier bug: replay_solar_nlms skipped any
        entity_id in ``aux_affected_entities`` unconditionally.  The config-
        flow default is ``aux_affected_entities = energy_sensors`` (all of
        them), so the bug blocked 100 % of solar hours on installations
        that never customised the setting.  Live learning only uses the
        list for cooldown-path aux coefficient handling — solar NLMS runs
        normally on non-aux-active hours regardless of membership.
        """
        lm = LearningManager()
        calc = self._build_env()
        entries = [{
            "timestamp": f"2026-04-15T{h:02d}:00:00",
            "temp_key": "10",
            "wind_bucket": "normal",
            "correction_percent": 100.0,
            "solar_vector_s": 0.5,
            "solar_vector_e": 0.1,
            "solar_vector_w": 0.02,
            "auxiliary_active": False,
            "temp": 10.0,
            "unit_modes": {"sensor.heater1": MODE_HEATING},
            "unit_breakdown": {"sensor.heater1": 0.5},
            "solar_dominant_entities": [],
        } for h in range(20)]
        correlation_per_unit = {"sensor.heater1": {"10": {"normal": 1.0}}}
        solar_coeffs: dict = {}
        # Sensor is in aux_affected_entities (default when user never
        # customised the wizard).  Pre-fix: every hour would hit
        # unit_skipped_aux_list and updates would be 0.
        updates = lm.replay_solar_nlms(
            entries,
            solar_calculator=calc,
            screen_config=(True, True, True),
            correlation_data_per_unit=correlation_per_unit,
            solar_coefficients_per_unit=solar_coeffs,
            learning_buffer_solar_per_unit={},
            energy_sensors=["sensor.heater1"],
            learning_rate=0.1,
            balance_point=15.0,
            aux_affected_entities=["sensor.heater1"],
        )
        assert updates >= 10, (
            "Solar NLMS must proceed for units listed in "
            "aux_affected_entities on non-aux-active hours."
        )
        assert "sensor.heater1" in solar_coeffs

    def test_aux_active_hours_still_skipped_regardless_of_aux_list(self):
        """Entry-level aux_active filter remains the correct block; being
        in aux_affected_entities does NOT add a second-layer block."""
        lm = LearningManager()
        calc = self._build_env()
        entries = [{
            "timestamp": f"2026-04-15T{h:02d}:00:00",
            "temp_key": "10",
            "wind_bucket": "normal",
            "correction_percent": 100.0,
            "solar_vector_s": 0.5,
            "solar_vector_e": 0.1,
            "solar_vector_w": 0.02,
            "auxiliary_active": True,  # aux active → entry-level skip
            "temp": 10.0,
            "unit_modes": {"sensor.heater1": MODE_HEATING},
            "unit_breakdown": {"sensor.heater1": 0.5},
            "solar_dominant_entities": [],
        } for h in range(20)]
        solar_coeffs: dict = {}
        updates = lm.replay_solar_nlms(
            entries,
            solar_calculator=calc,
            screen_config=(True, True, True),
            correlation_data_per_unit={"sensor.heater1": {"10": {"normal": 1.0}}},
            solar_coefficients_per_unit=solar_coeffs,
            learning_buffer_solar_per_unit={},
            energy_sensors=["sensor.heater1"],
            learning_rate=0.1,
            balance_point=15.0,
            aux_affected_entities=["sensor.heater1"],
        )
        assert updates == 0
        assert "sensor.heater1" not in solar_coeffs

    def test_disabled_status_hours_are_skipped(self):
        """``learning_status='disabled'`` entries must not feed NLMS replay.

        Live ``_is_poisoned`` (coordinator.py:922-930) treats the status
        as poisoned in non-daily mode.  An earlier replay implementation
        matched the if-condition on ``status == 'disabled'`` but failed
        to ``continue`` for it, leaking user-disabled hours into solar
        coefficient learning during retrain.  Pin the post-fix counter
        and zero-update outcome so the fall-through cannot regress.
        """
        lm = LearningManager()
        calc = self._build_env()
        entries = [{
            "timestamp": f"2026-04-15T{h:02d}:00:00",
            "temp_key": "10",
            "wind_bucket": "normal",
            "correction_percent": 100.0,
            "solar_vector_s": 0.5,
            "solar_vector_e": 0.1,
            "solar_vector_w": 0.02,
            "auxiliary_active": False,
            "learning_status": "disabled",
            "temp": 10.0,
            "unit_modes": {"sensor.heater1": MODE_HEATING},
            "unit_breakdown": {"sensor.heater1": 0.5},
            "solar_dominant_entities": [],
        } for h in range(20)]
        solar_coeffs: dict = {}
        diag = lm.replay_solar_nlms(
            entries,
            solar_calculator=calc,
            screen_config=(True, True, True),
            correlation_data_per_unit={"sensor.heater1": {"10": {"normal": 1.0}}},
            solar_coefficients_per_unit=solar_coeffs,
            learning_buffer_solar_per_unit={},
            energy_sensors=["sensor.heater1"],
            learning_rate=0.1,
            balance_point=15.0,
            aux_affected_entities=[],
            return_diagnostics=True,
        )
        # All 20 entries skipped under the dedicated counter; none reach
        # the per-unit loop, so the coefficient dict stays empty and
        # entry_skipped_poisoned stays at 0 (counter split avoids
        # conflating user-disabled with data-quality skips).
        assert diag["entry_skipped_disabled"] == 20
        assert diag["entry_skipped_poisoned"] == 0
        assert diag["updates"] == 0
        assert "sensor.heater1" not in solar_coeffs

    def test_aux_hours_skipped(self):
        """Aux-active hours are not fed into NLMS."""
        lm = LearningManager()
        calc = self._build_env()
        entries = [{
            "timestamp": f"2026-04-15T{h:02d}:00:00",
            "temp_key": "10",
            "wind_bucket": "normal",
            "correction_percent": 100.0,
            "solar_vector_s": 0.5,
            "solar_vector_e": 0.0,
            "solar_vector_w": 0.0,
            "auxiliary_active": True,  # aux active
            "temp": 10.0,
            "unit_modes": {"sensor.heater1": MODE_HEATING},
            "unit_breakdown": {"sensor.heater1": 0.5},
            "solar_dominant_entities": [],
        } for h in range(20)]

        correlation_per_unit = {"sensor.heater1": {"10": {"normal": 1.0}}}
        solar_coeffs = {}

        updates = lm.replay_solar_nlms(
            entries,
            solar_calculator=calc,
            screen_config=(True, True, True),
            correlation_data_per_unit=correlation_per_unit,
            solar_coefficients_per_unit=solar_coeffs,
            learning_buffer_solar_per_unit={},
            energy_sensors=["sensor.heater1"],
            learning_rate=0.1,
            balance_point=15.0,
        )
        assert updates == 0
        assert "sensor.heater1" not in solar_coeffs

    def test_insufficient_base_hours_skipped(self):
        """Units with base < SOLAR_LEARNING_MIN_BASE are skipped (live parity)."""
        lm = LearningManager()
        calc = self._build_env()
        entries = [{
            "timestamp": f"2026-04-15T{h:02d}:00:00",
            "temp_key": "10",
            "wind_bucket": "normal",
            "correction_percent": 100.0,
            "solar_vector_s": 0.5,
            "solar_vector_e": 0.0,
            "solar_vector_w": 0.0,
            "auxiliary_active": False,
            "temp": 10.0,
            "unit_modes": {"sensor.heater1": MODE_HEATING},
            "unit_breakdown": {"sensor.heater1": 0.05},
            "solar_dominant_entities": [],
        } for h in range(20)]

        # Base is 0.1 which is below SOLAR_LEARNING_MIN_BASE (0.15)
        correlation_per_unit = {"sensor.heater1": {"10": {"normal": 0.1}}}
        solar_coeffs = {}

        updates = lm.replay_solar_nlms(
            entries,
            solar_calculator=calc,
            screen_config=(True, True, True),
            correlation_data_per_unit=correlation_per_unit,
            solar_coefficients_per_unit=solar_coeffs,
            learning_buffer_solar_per_unit={},
            energy_sensors=["sensor.heater1"],
            learning_rate=0.1,
            balance_point=15.0,
        )
        assert updates == 0

# -----------------------------------------------------------------------------
# Integration: retrain_from_history 3-pass EM
# -----------------------------------------------------------------------------

def _retrain_coord(hourly_log):
    """Build a coord for retrain tests (Track A, real SolarCalculator)."""
    coord = MagicMock()
    coord._hourly_log = hourly_log
    coord.daily_learning_mode = False
    coord.learning_rate = 1.0
    coord.balance_point = 15.0
    coord.energy_sensors = ["sensor.heater1"]
    coord.aux_affected_entities = []
    coord.screen_config = (True, True, True)
    coord._correlation_data = {}
    coord._correlation_data_per_unit = {}
    coord._aux_coefficients = {}
    coord._aux_coefficients_per_unit = {}
    coord._learning_buffer_global = {}
    coord._learning_buffer_per_unit = {}
    coord._learning_buffer_aux_per_unit = {}
    coord._solar_coefficients_per_unit = {}
    coord._learning_buffer_solar_per_unit = {}
    coord._observation_counts = {}
    coord._learned_u_coefficient = None
    coord.storage.async_save_data = AsyncMock()
    coord._get_predicted_kwh = MagicMock(return_value=0.0)
    coord.learning = LearningManager()
    coord.solar = SolarCalculator(coord)
    # Real replay for per-unit (matches production behaviour)
    from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
    coord._replay_per_unit_models = MagicMock(
        side_effect=lambda entries: HeatingDataCoordinator._replay_per_unit_models(
            coord, entries
        )
    )
    coord._unit_strategies = {}
    from custom_components.heating_analytics.observation import build_strategies
    coord._unit_strategies = build_strategies(
        energy_sensors=["sensor.heater1"],
        track_c_enabled=False,
        mpc_managed_sensor=None,
    )
    return coord


def _sunny_entry(ts, *, actual=0.3, solar_delta=0.2, solar_s=0.5,
                 temp=10.0, unit_breakdown=None):
    return {
        "timestamp": ts,
        "hour": int(ts[11:13]),
        "temp": temp,
        "temp_key": str(int(round(temp))),
        "wind_bucket": "normal",
        "actual_kwh": actual,
        "auxiliary_active": False,
        "solar_normalization_delta": solar_delta,
        "solar_factor": solar_s,
        "solar_vector_s": solar_s,
        "solar_vector_e": 0.1,
        "solar_vector_w": 0.05,
        "correction_percent": 100.0,
        "unit_modes": {"sensor.heater1": MODE_HEATING},
        "unit_breakdown": unit_breakdown or {"sensor.heater1": actual},
        "solar_dominant_entities": [],
    }


class TestRetrainTrackAThreePass:
    """Track A with reset_first=True runs the three-pass EM-lite."""

    @pytest.mark.asyncio
    async def test_single_pass_when_not_resetting(self):
        from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
        entries = [
            _sunny_entry(f"2026-04-10T{h:02d}:00:00", actual=0.3, solar_delta=0.2)
            for h in range(24)
        ]
        coord = _retrain_coord(entries)
        result = await RetrainEngine(coord).retrain_from_history(reset_first=False
        )
        assert result["em_passes"] == 1
        # NLMS replay still runs at the end to refresh coefficients
        assert "solar_replay_updates" in result

    @pytest.mark.asyncio
    async def test_priming_pass_ignores_stored_delta(self):
        """Pass 1 of 3-pass EM uses solar_norm=0, NOT the (contaminated)
        stored delta.  The final bucket value therefore does NOT inherit
        the stored-delta bias from the log."""
        from custom_components.heating_analytics.coordinator import HeatingDataCoordinator

        # 24 dark-ish hours where actual=1.0 but stored delta says +2.0
        # (pretending solar coeff was hugely over-estimated).  If retrain
        # used stored delta we'd EMA the bucket toward 1.0+2.0 = 3.0.
        # With priming (delta=0) and no solar at all (vector=0), bucket
        # should settle close to 1.0.
        entries = [{
            "timestamp": f"2026-04-10T{h:02d}:00:00",
            "hour": h,
            "temp": 10.0,
            "temp_key": "10",
            "wind_bucket": "normal",
            "actual_kwh": 1.0,
            "auxiliary_active": False,
            "solar_normalization_delta": 2.0,  # "stored" is hugely contaminated
            "solar_factor": 0.0,  # but actually dark
            "solar_vector_s": 0.0,
            "solar_vector_e": 0.0,
            "solar_vector_w": 0.0,
            "correction_percent": 100.0,
            "unit_modes": {"sensor.heater1": MODE_HEATING},
            "unit_breakdown": {"sensor.heater1": 1.0},
            "solar_dominant_entities": [],
        } for h in range(24)]
        coord = _retrain_coord(entries)

        await RetrainEngine(coord).retrain_from_history(reset_first=True
        )
        # Refinement pass uses on-the-fly delta; solar coeffs were learned
        # against dark hours (no signal) so coeff stays tiny/unset → delta=0
        # in refinement pass.  Bucket converges to raw 1.0, not 3.0.
        assert coord._correlation_data["10"]["normal"] == pytest.approx(1.0, abs=0.1)


class TestReplaySolarNLMSWeightedSmearSkip:
    """WeightedSmear (MPC-managed) sensors are skipped by NLMS replay.

    Rationale documented in ``replay_solar_nlms``: MPC provides no
    dark-equivalent baseline, and Track C smearing bakes solar into
    ``synthetic_kwh_el`` per-hour shape while Jensen-rescaling anchors
    the daily total.  ``synthetic - kwh_el_sh`` therefore encodes COP
    variance, not solar reduction.  Live learning already excludes
    these sensors (coordinator.py:4391-4397, #776) — retrain mirrors
    that.  Solar coefficient for MPC sensors falls back to
    ``DEFAULT_SOLAR_COEFF_HEATING`` via
    :meth:`SolarCalculator.calculate_unit_coefficient`.
    """

    def _calc(self):
        coord = MagicMock()
        coord.screen_config = (True, True, True)
        return SolarCalculator(coord)

    def _entry(self, hour, *, sensor_id, actual=0.5, date="2026-04-15"):
        return {
            "timestamp": f"{date}T{hour:02d}:00:00",
            "hour": hour,
            "temp_key": "10",
            "wind_bucket": "normal",
            "correction_percent": 100.0,
            "solar_vector_s": 0.5,
            "solar_vector_e": 0.05 + (hour % 3) * 0.02,
            "solar_vector_w": 0.02,
            "auxiliary_active": False,
            "temp": 10.0,
            "unit_modes": {},  # heating-only logged implicitly
            "unit_breakdown": {sensor_id: actual},
            "solar_dominant_entities": [],
        }

    def test_weighted_smear_sensor_is_skipped(self):
        """Every qualifying hour lands in ``unit_skipped_weighted_smear``."""
        lm = LearningManager()
        calc = self._calc()
        mpc_sid = "sensor.heater_mpc"
        entries = [self._entry(h, sensor_id=mpc_sid) for h in range(20)]
        strategies = {mpc_sid: WeightedSmear(mpc_sid, use_synthetic=True)}
        solar_coeffs: dict = {}
        diag = lm.replay_solar_nlms(
            entries,
            solar_calculator=calc,
            screen_config=(True, True, True),
            correlation_data_per_unit={},
            solar_coefficients_per_unit=solar_coeffs,
            learning_buffer_solar_per_unit={},
            energy_sensors=[mpc_sid],
            learning_rate=0.1,
            balance_point=15.0,
            unit_strategies=strategies,
            return_diagnostics=True,
        )
        assert diag["updates"] == 0
        assert diag["unit_skipped_weighted_smear"] == 20
        assert mpc_sid not in solar_coeffs

    def test_direct_meter_sensor_learns_normally(self):
        """DirectMeter sensors (incl. non-MPC units on a Track C install)
        are unaffected by the WeightedSmear skip."""
        lm = LearningManager()
        calc = self._calc()
        direct_sid = "sensor.heater_direct"
        entries = [self._entry(h, sensor_id=direct_sid) for h in range(20)]
        strategies = {direct_sid: DirectMeter(direct_sid)}
        correlation_per_unit = {direct_sid: {"10": {"normal": 1.0}}}
        solar_coeffs: dict = {}
        updates = lm.replay_solar_nlms(
            entries,
            solar_calculator=calc,
            screen_config=(True, True, True),
            correlation_data_per_unit=correlation_per_unit,
            solar_coefficients_per_unit=solar_coeffs,
            learning_buffer_solar_per_unit={},
            energy_sensors=[direct_sid],
            learning_rate=0.1,
            balance_point=15.0,
            unit_strategies=strategies,
        )
        assert updates >= 10
        assert direct_sid in solar_coeffs

    def test_mixed_install_direct_learns_mpc_skipped(self):
        """Multi-unit install: DirectMeter learns; WeightedSmear skipped.

        Exercises the key Track C principle: non-MPC units unaffected by
        the MPC-sensor's exclusion.
        """
        lm = LearningManager()
        calc = self._calc()
        mpc_sid = "sensor.heater_mpc"
        direct_sid = "sensor.heater_direct"
        entries = []
        for h in range(20):
            e = self._entry(h, sensor_id=mpc_sid)
            e["unit_breakdown"][direct_sid] = 0.5
            entries.append(e)
        strategies = {
            mpc_sid: WeightedSmear(mpc_sid, use_synthetic=True),
            direct_sid: DirectMeter(direct_sid),
        }
        correlation_per_unit = {direct_sid: {"10": {"normal": 1.0}}}
        solar_coeffs: dict = {}
        diag = lm.replay_solar_nlms(
            entries,
            solar_calculator=calc,
            screen_config=(True, True, True),
            correlation_data_per_unit=correlation_per_unit,
            solar_coefficients_per_unit=solar_coeffs,
            learning_buffer_solar_per_unit={},
            energy_sensors=[mpc_sid, direct_sid],
            learning_rate=0.1,
            balance_point=15.0,
            unit_strategies=strategies,
            return_diagnostics=True,
        )
        assert direct_sid in solar_coeffs
        assert mpc_sid not in solar_coeffs
        assert diag["unit_skipped_weighted_smear"] == 20
