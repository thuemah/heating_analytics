"""Tests for SNR-weighted base-model learning (#866).

Validates the four behaviours promised by the issue:

1. ``compute_snr_weight`` mathematical behaviour (weight curve + shutdown
   scaling) in isolation.
2. Track A retrain: under flag=True the target is raw actual and the
   rate is scaled by per-entry snr_weight.  ``em_passes == 1``.
3. Track B retrain: under flag=True uses day-average solar_factor for the
   weight and drops delta from the daily target.  ``em_passes == 1``.
4. ``learn_from_historical_import`` direct-call contract: snr_weight
   parameter scales EMA step when flag is on; solar_normalization_delta
   is ignored in that regime.

Tests that assert legacy delta-based behaviour live alongside these in
the pre-existing test files and pin ``legacy_delta_rule`` fixture.
"""
from unittest.mock import MagicMock, AsyncMock

import pytest

from custom_components.heating_analytics.learning import (
    LearningManager,
    compute_snr_weight,
)
from custom_components.heating_analytics.solar import SolarCalculator
from custom_components.heating_analytics.observation import build_strategies
from custom_components.heating_analytics.const import (
    SNR_WEIGHT_FLOOR,
    SNR_WEIGHT_K,
    MODE_HEATING,
)
from custom_components.heating_analytics.retrain import RetrainEngine


# -----------------------------------------------------------------------------
# 1. compute_snr_weight — isolated math
# -----------------------------------------------------------------------------

class TestSnrWeightFunction:

    def test_dark_hour_full_weight(self):
        assert compute_snr_weight(0.0, [], total_units=1) == pytest.approx(1.0)

    def test_high_solar_hits_floor(self):
        # sf=1.0, k=3.0 → max(floor, 1-3) = floor
        assert compute_snr_weight(1.0, [], total_units=1) == pytest.approx(SNR_WEIGHT_FLOOR)

    def test_mid_solar_linear_decay(self):
        # sf=0.2, k=3.0 → 1 - 0.6 = 0.4
        assert compute_snr_weight(0.2, [], total_units=1) == pytest.approx(0.4)

    def test_negative_sf_coerced_to_zero(self):
        """Defensive: malformed entries with negative solar_factor → full weight."""
        assert compute_snr_weight(-0.5, [], total_units=1) == pytest.approx(1.0)

    def test_all_units_shutdown_zero_weight(self):
        assert compute_snr_weight(0.0, ["a", "b"], total_units=2) == 0.0

    def test_partial_shutdown_proportional(self):
        # 1 of 2 units shut down, dark hour → weight = 1.0 * 0.5 = 0.5
        assert compute_snr_weight(0.0, ["a"], total_units=2) == pytest.approx(0.5)

    def test_combined_solar_and_partial_shutdown(self):
        # sf=0.2 → base 0.4; 1 of 3 down → scale × 2/3
        w = compute_snr_weight(0.2, ["a"], total_units=3)
        assert w == pytest.approx(0.4 * 2 / 3)

    def test_zero_total_units_disables_shutdown_scaling(self):
        # Edge: no known total → can't compute fraction, just use base weight
        assert compute_snr_weight(0.2, ["a"], total_units=0) == pytest.approx(0.4)

    def test_custom_floor_and_k(self):
        # floor=0.5 overrides default — clear-sky hour retains 0.5
        assert compute_snr_weight(1.0, [], total_units=1, floor=0.5, k=3.0) == pytest.approx(0.5)
        # k=10 → very steep. sf=0.1 → 1 - 1 = 0 → floor
        assert compute_snr_weight(0.1, [], total_units=1, floor=0.1, k=10.0) == pytest.approx(0.1)


# -----------------------------------------------------------------------------
# 2. Track A retrain — flag=True: raw target + per-entry snr_weight + em_passes=1
# -----------------------------------------------------------------------------

def _track_a_coord(hourly_log, energy_sensors=("sensor.heater",)):
    """Minimal Track A retrain coord with real LearningManager + SolarCalculator."""
    coord = MagicMock()
    coord._hourly_log = hourly_log
    coord.daily_learning_mode = False
    coord.learning_rate = 1.0  # amplify for observable convergence
    coord.balance_point = 15.0
    coord.energy_sensors = list(energy_sensors)
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
    coord._daily_history = {}
    coord.storage.async_save_data = AsyncMock()
    coord._get_predicted_kwh = MagicMock(return_value=0.0)
    coord.learning = LearningManager()
    coord.solar = SolarCalculator(coord)
    coord._unit_strategies = build_strategies(
        energy_sensors=list(energy_sensors),
        track_c_enabled=False,
        mpc_managed_sensor=None,
    )
    from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
    coord._replay_per_unit_models = MagicMock(
        side_effect=lambda entries: HeatingDataCoordinator._replay_per_unit_models(
            coord, entries
        )
    )
    return coord


def _entry(ts, *, actual=0.3, solar_factor=0.0, solar_delta=0.0, sensor_id="sensor.heater"):
    return {
        "timestamp": ts,
        "hour": int(ts[11:13]),
        "temp": 10.0,
        "temp_key": "10",
        "wind_bucket": "normal",
        "actual_kwh": actual,
        "auxiliary_active": False,
        "solar_factor": solar_factor,
        "solar_vector_s": solar_factor,
        "solar_vector_e": 0.0,
        "solar_vector_w": 0.0,
        "correction_percent": 100.0,
        "unit_modes": {},
        "unit_breakdown": {sensor_id: actual},
        "solar_dominant_entities": [],
        "solar_normalization_delta": solar_delta,
    }


class TestTrackARetrainUnderSnr:

    @pytest.mark.asyncio
    async def test_em_passes_collapses_to_one(self):
        """3-pass EM-lite collapses to 1-pass + orthogonal NLMS replay."""
        from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
        entries = [_entry(f"2026-04-10T{h:02d}:00:00") for h in range(24)]
        coord = _track_a_coord(entries)
        result = await RetrainEngine(coord).retrain_from_history(reset_first=True)
        assert result["em_passes"] == 1

    @pytest.mark.asyncio
    async def test_dark_hour_bucket_equals_raw_actual(self):
        """Dark-hour replay: weight=1.0, target=raw → bucket ≈ raw actual.

        Stored delta of +2.0 must NOT contaminate the bucket under SNR.
        Legacy path would push bucket to 3.0; SNR ignores the delta.
        """
        from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
        entries = [
            _entry(f"2026-04-10T{h:02d}:00:00", actual=1.0, solar_delta=2.0, solar_factor=0.0)
            for h in range(24)
        ]
        coord = _track_a_coord(entries)
        await RetrainEngine(coord).retrain_from_history(reset_first=True)
        # With lr=1.0 and 24 dark samples all at actual=1.0, bucket = 1.0
        assert coord._correlation_data["10"]["normal"] == pytest.approx(1.0, abs=0.05)

    @pytest.mark.asyncio
    async def test_sunny_hour_downweighted_not_amplified(self):
        """Mixed dark + sunny day: SNR weights dark hours more than sunny.

        At steady state, SNR-weighted mean of (12 × 1.0 @ w=1.0, 12 × 0.3 @ w=0.1)
        is (12 + 0.36) / (12 + 1.2) = 0.936.  Naive unweighted mean is 0.65.
        A single chronological pass with lr=0.5 converges well past the naive
        mean toward the SNR steady state.  The weight ratio 10:1 means sunny
        hours contribute 10× less per sample than dark hours — even on a single
        pass the bucket settles strictly above 0.65.
        """
        from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
        entries = []
        for h in range(12):
            entries.append(_entry(f"2026-04-10T{h:02d}:00:00", actual=1.0, solar_factor=0.0))
        for h in range(12):
            entries.append(
                _entry(
                    f"2026-04-10T{12 + h:02d}:00:00",
                    actual=0.3,
                    solar_delta=0.7,  # ignored under SNR — left to prove delta doesn't leak
                    solar_factor=1.0,
                )
            )
        coord = _track_a_coord(entries)
        coord.learning_rate = 0.5
        await RetrainEngine(coord).retrain_from_history(reset_first=True)
        bucket = coord._correlation_data["10"]["normal"]
        # Naive unweighted mean of the 24 targets = (12+3.6)/24 = 0.65.
        # SNR must land ABOVE the unweighted mean (dark weight > sunny weight).
        assert bucket > 0.65, f"SNR should dominate-weight dark hours; bucket={bucket}"
        # And strictly below pure-dark truth (sunny hours still carry some weight).
        assert bucket < 1.0, f"Sunny hours must still contribute something; bucket={bucket}"


# -----------------------------------------------------------------------------
# 3. learn_from_historical_import — direct-call contract
# -----------------------------------------------------------------------------

class TestLearnFromHistoricalImportUnderSnr:

    def _call(self, *, actual_kwh, snr_weight, solar_delta=0.0, current_bucket=0.0, lr=0.5):
        lm = LearningManager()
        correlation_data = {"10": {"normal": current_bucket}} if current_bucket else {}
        aux_coefficients = {}
        lm.learn_from_historical_import(
            temp_key="10",
            wind_bucket="normal",
            actual_kwh=actual_kwh,
            is_aux_active=False,
            correlation_data=correlation_data,
            aux_coefficients=aux_coefficients,
            learning_rate=lr,
            get_predicted_kwh_fn=lambda t, w, at: 0.0,
            actual_temp=10.0,
            solar_normalization_delta=solar_delta,
            snr_weight=snr_weight,
        )
        return correlation_data.get("10", {}).get("normal", 0.0)

    def test_full_weight_dark_hour_seeds_from_raw(self):
        """w=1.0 on empty bucket → bucket seeded with raw actual."""
        v = self._call(actual_kwh=1.0, snr_weight=1.0, solar_delta=99.0)
        # Delta ignored — bucket reflects raw actual
        assert v == pytest.approx(1.0)

    def test_full_weight_ema_uses_raw_target(self):
        """w=1.0 on existing bucket → EMA toward raw actual, not actual+delta."""
        # bucket=2.0, actual=1.0, delta=0.5 → SNR target=1.0, new=2.0+0.5*(1.0-2.0)=1.5
        v = self._call(actual_kwh=1.0, snr_weight=1.0, solar_delta=0.5, current_bucket=2.0)
        assert v == pytest.approx(1.5)

    def test_half_weight_moves_half_as_much(self):
        """w=0.5 halves the effective EMA step."""
        v_full = self._call(actual_kwh=1.0, snr_weight=1.0, current_bucket=2.0)
        v_half = self._call(actual_kwh=1.0, snr_weight=0.5, current_bucket=2.0)
        # Full: 2.0 + 0.5*1.0*(1.0-2.0) = 1.5 → δ=0.5
        # Half: 2.0 + 0.5*0.5*(1.0-2.0) = 1.75 → δ=0.25
        assert (2.0 - v_full) == pytest.approx(2 * (2.0 - v_half))

    def test_zero_weight_skips_seeding_empty_bucket(self):
        """w=0 (full shutdown) on empty bucket → no update, returns skipped."""
        lm = LearningManager()
        correlation_data = {}
        status = lm.learn_from_historical_import(
            temp_key="10", wind_bucket="normal",
            actual_kwh=0.0, is_aux_active=False,
            correlation_data=correlation_data,
            aux_coefficients={},
            learning_rate=1.0,
            get_predicted_kwh_fn=lambda *a: 0.0,
            actual_temp=10.0,
            solar_normalization_delta=0.0,
            snr_weight=0.0,
        )
        assert status == "skipped_zero_weight"
        assert correlation_data == {}

    def test_zero_weight_freezes_existing_bucket(self):
        """w=0 on non-empty bucket → no movement."""
        v = self._call(actual_kwh=0.0, snr_weight=0.0, current_bucket=2.0)
        assert v == pytest.approx(2.0)


# -----------------------------------------------------------------------------
# 4. Track B retrain — day-level SNR weighting
# -----------------------------------------------------------------------------

def _daily_coord(hourly_log):
    coord = _track_a_coord(hourly_log)
    coord.daily_learning_mode = True
    coord.track_c_enabled = False
    coord.mpc_managed_sensor = None
    coord.thermal_mass_kwh_per_degree = 0.0
    coord._daily_history = {}
    coord._get_wind_bucket = MagicMock(return_value="normal")
    coord._compute_excluded_mode_energy = MagicMock(return_value=0.0)
    coord._try_track_b_cop_smearing = AsyncMock(return_value=None)
    coord._apply_strategies_to_global_model = MagicMock()
    return coord


def _full_day_entries(date_str, *, actual_per_hour=0.5, solar_factor=0.0,
                       solar_delta=0.0, tdd_per_hour=5.0 / 24):
    return [
        {
            "timestamp": f"{date_str}T{h:02d}:00:00",
            "hour": h,
            "temp": 10.0,
            "temp_key": "10",
            "wind_bucket": "normal",
            "actual_kwh": actual_per_hour,
            "tdd": tdd_per_hour,
            "effective_wind": 3.0,
            "auxiliary_active": False,
            "solar_factor": solar_factor,
            "solar_vector_s": solar_factor,
            "solar_vector_e": 0.0,
            "solar_vector_w": 0.0,
            "correction_percent": 100.0,
            "unit_modes": {},
            "unit_breakdown": {"sensor.heater": actual_per_hour},
            "solar_dominant_entities": [],
            "solar_normalization_delta": solar_delta,
        }
        for h in range(24)
    ]


class TestTrackBRetrainUnderSnr:

    @pytest.mark.asyncio
    async def test_em_passes_collapses_to_one(self):
        from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
        entries = _full_day_entries("2026-04-10", actual_per_hour=0.5, solar_factor=0.0)
        coord = _daily_coord(entries)
        result = await RetrainEngine(coord).retrain_from_history(reset_first=True)
        assert result["em_passes"] == 1
        assert result["mode"] == "strategy_dispatch"

    @pytest.mark.asyncio
    async def test_dark_day_bucket_uses_raw_not_delta(self):
        """Dark-day retrain: bucket = q_adjusted/24, stored delta ignored."""
        from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
        # Each hour actual=0.5, solar_delta=2.0, solar_factor=0.0 (dark).
        # Legacy: q_normalized = 12 + 24*2.0 = 60 → bucket = 2.5 kWh/hr
        # SNR (dark): day_weight=1.0, q_hourly_avg = 12/24 = 0.5 → bucket = 0.5
        entries = _full_day_entries("2026-04-10", actual_per_hour=0.5,
                                    solar_factor=0.0, solar_delta=2.0)
        coord = _daily_coord(entries)
        await RetrainEngine(coord).retrain_from_history(reset_first=True)
        # First-day write seeds bucket with raw hourly average
        assert coord._correlation_data["10"]["normal"] == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_sunny_day_weight_at_floor(self):
        """Clear-sky day: avg solar_factor=1.0 → weight = FLOOR (0.1).

        Uses reset_first=False so the pre-seeded bucket survives (reset_first
        clears correlation_data at the start of retrain).
        """
        from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
        entries = _full_day_entries("2026-04-10", actual_per_hour=0.3,
                                    solar_factor=1.0, solar_delta=0.7)
        coord = _daily_coord(entries)
        coord._correlation_data = {"10": {"normal": 1.0}}
        await RetrainEngine(coord).retrain_from_history(reset_first=False)
        # day_weight = max(FLOOR, 1-3×1) = FLOOR = 0.1
        # q_hourly_avg = (24 × 0.3) / 24 = 0.3
        # new = 1.0 + (lr=1.0 × 0.1) × (0.3 - 1.0) = 1.0 - 0.07 = 0.93
        bucket = coord._correlation_data["10"]["normal"]
        assert bucket == pytest.approx(1.0 - 1.0 * SNR_WEIGHT_FLOOR * 0.7, abs=0.02)
