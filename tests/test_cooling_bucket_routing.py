"""Tests for mode-stratified per-unit base buckets (#885).

Per-unit cooling samples route to a dedicated "cooling" wind-bucket key
at the write boundary (`learning._process_per_unit_learning`) and the
read boundary (`coordinator._get_predicted_kwh_per_unit`,
`statistics._calculate_fallback_projection`,
`statistics.calculate_total_power`).  Global `correlation_data` is
unchanged — cooling continues to fold in via sign-flipped
`solar_normalization_delta` as before.

Covers:
1. Write-side routing: cooling samples land in `[entity][temp]["cooling"]`
   regardless of actual wind; heating samples use normal/high_wind/
   extreme_wind as before.
2. Cooling-at-cold no longer skipped (1.3.3 guard removed).
3. D2a guard on prediction: empty cooling bucket → return 0, not
   heating-bucket value (which would be physically wrong at cooling
   temperatures).
"""
from unittest.mock import MagicMock

import pytest

from custom_components.heating_analytics.const import (
    COOLING_WIND_BUCKET,
    MODE_COOLING,
    MODE_HEATING,
)
from custom_components.heating_analytics.learning import LearningManager
from custom_components.heating_analytics.statistics import StatisticsManager


def _call_process_per_unit_learning(
    manager: LearningManager,
    *,
    entity_id: str,
    unit_mode: str,
    wind_bucket: str,
    avg_temp: float,
    actual_kwh: float,
    correlation_data_per_unit: dict,
    learning_buffer_per_unit: dict,
    observation_counts: dict,
    temp_key: str = "5",
    balance_point: float = 15.0,
):
    """Run _process_per_unit_learning with a minimal mock coordinator state.

    Solar is disabled, aux is inactive, cooldown is off — isolates the
    base-EMA write path so bucket routing is the only active signal.
    """
    manager._process_per_unit_learning(
        temp_key=temp_key,
        wind_bucket=wind_bucket,
        avg_temp=avg_temp,
        avg_solar_vector=(0.0, 0.0, 0.0),
        total_energy_kwh=actual_kwh,
        base_expected_kwh=actual_kwh,  # unused when solar disabled
        energy_sensors=[entity_id],
        hourly_delta_per_unit={entity_id: actual_kwh},
        solar_enabled=False,
        learning_rate=1.0,  # immediate convergence for assertion simplicity
        solar_calculator=None,
        get_predicted_unit_base_fn=lambda *_a, **_k: 0.0,
        learning_buffer_per_unit=learning_buffer_per_unit,
        correlation_data_per_unit=correlation_data_per_unit,
        observation_counts=observation_counts,
        is_aux_active=False,
        aux_coefficients_per_unit={},
        learning_buffer_aux_per_unit={},
        solar_coefficients_per_unit={},
        learning_buffer_solar_per_unit={},
        balance_point=balance_point,
        unit_modes={entity_id: unit_mode},
        hourly_expected_per_unit={},
        hourly_expected_base_per_unit={entity_id: 0.0},
        aux_affected_entities=[],
        is_cooldown_active=False,
    )


class TestWriteSideRouting:
    """Cooling samples route to `"cooling"` wind-bucket regardless of actual wind."""

    def test_cooling_mode_writes_cooling_bucket_under_high_wind(self):
        """Cooling sample with high_wind hour → lands in `["cooling"]`."""
        manager = LearningManager()
        correlation: dict = {}
        buffers: dict = {}
        counts: dict = {}

        # Fill buffer with 4 samples to trigger jump-start + first EMA write.
        for _ in range(5):
            _call_process_per_unit_learning(
                manager,
                entity_id="ac.unit",
                unit_mode=MODE_COOLING,
                wind_bucket="high_wind",  # actual hour is windy
                avg_temp=25.0,
                actual_kwh=0.80,
                correlation_data_per_unit=correlation,
                learning_buffer_per_unit=buffers,
                observation_counts=counts,
                temp_key="25",
            )

        assert "ac.unit" in correlation
        assert "25" in correlation["ac.unit"]
        # The write landed in "cooling", not "high_wind".
        assert COOLING_WIND_BUCKET in correlation["ac.unit"]["25"]
        assert "high_wind" not in correlation["ac.unit"]["25"]
        assert "normal" not in correlation["ac.unit"]["25"]

    def test_heating_mode_writes_actual_wind_bucket(self):
        """Heating sample with high_wind hour → lands in `["high_wind"]`."""
        manager = LearningManager()
        correlation: dict = {}
        buffers: dict = {}
        counts: dict = {}

        for _ in range(5):
            _call_process_per_unit_learning(
                manager,
                entity_id="hp.unit",
                unit_mode=MODE_HEATING,
                wind_bucket="high_wind",
                avg_temp=-5.0,
                actual_kwh=2.50,
                correlation_data_per_unit=correlation,
                learning_buffer_per_unit=buffers,
                observation_counts=counts,
                temp_key="-5",
            )

        assert "high_wind" in correlation["hp.unit"]["-5"]
        assert COOLING_WIND_BUCKET not in correlation["hp.unit"]["-5"]

    def test_cooling_at_cold_no_longer_skipped(self):
        """Removed 1.3.3 guard: cooling + temp < BP-2 populates cooling bucket."""
        manager = LearningManager()
        correlation: dict = {}
        buffers: dict = {}
        counts: dict = {}

        # balance_point=15 → BP-2=13; avg_temp=5 is well below.
        for _ in range(5):
            _call_process_per_unit_learning(
                manager,
                entity_id="ac.unit",
                unit_mode=MODE_COOLING,
                wind_bucket="normal",
                avg_temp=5.0,
                actual_kwh=0.10,  # standby-ish
                correlation_data_per_unit=correlation,
                learning_buffer_per_unit=buffers,
                observation_counts=counts,
                balance_point=15.0,
            )

        # Previously guard would have skipped; now cooling bucket is populated.
        assert COOLING_WIND_BUCKET in correlation["ac.unit"]["5"]

    def test_heating_and_cooling_coexist_same_entity_same_temp(self):
        """Same entity in both modes: heating → normal, cooling → cooling."""
        manager = LearningManager()
        correlation: dict = {}
        buffers: dict = {}
        counts: dict = {}

        for _ in range(5):
            _call_process_per_unit_learning(
                manager,
                entity_id="hp.unit",
                unit_mode=MODE_HEATING,
                wind_bucket="normal",
                avg_temp=10.0,
                actual_kwh=1.5,
                correlation_data_per_unit=correlation,
                learning_buffer_per_unit=buffers,
                observation_counts=counts,
                temp_key="10",
            )
        for _ in range(5):
            _call_process_per_unit_learning(
                manager,
                entity_id="hp.unit",
                unit_mode=MODE_COOLING,
                wind_bucket="normal",
                avg_temp=10.0,
                actual_kwh=0.5,
                correlation_data_per_unit=correlation,
                learning_buffer_per_unit=buffers,
                observation_counts=counts,
                temp_key="10",
            )

        assert "normal" in correlation["hp.unit"]["10"]
        assert COOLING_WIND_BUCKET in correlation["hp.unit"]["10"]
        # Values are distinct — heating and cooling learned independently.
        assert correlation["hp.unit"]["10"]["normal"] != correlation["hp.unit"]["10"][COOLING_WIND_BUCKET]


class TestPredictionFallbackGuard:
    """D2a: cooling-bucket prediction must NOT fall back to heating wind buckets."""

    def _stats(self):
        """StatisticsManager with a minimal coordinator stub."""
        coord = MagicMock()
        coord.balance_point = 15.0
        return StatisticsManager(coord)

    def test_empty_cooling_bucket_returns_zero_not_heating(self):
        """Cooling requested at a temp where only heating data exists → 0."""
        stats = self._stats()
        data_map = {
            "25": {"normal": 1.5},  # heating data (physically nonsense but fine for test)
        }
        result = stats._get_prediction_from_model(
            data_map,
            temp_key="25",
            wind_bucket=COOLING_WIND_BUCKET,
            actual_temp=25.0,
            balance_point=15.0,
        )
        # Must not return 1.5 (the heating bucket).  Acceptable: 0 (warm-up).
        assert result != 1.5
        assert result == 0.0

    def test_cooling_bucket_exact_match_returned(self):
        """When cooling data exists at that temp, it's returned directly."""
        stats = self._stats()
        data_map = {
            "25": {"normal": 1.5, COOLING_WIND_BUCKET: 0.80},
        }
        result = stats._get_prediction_from_model(
            data_map,
            temp_key="25",
            wind_bucket=COOLING_WIND_BUCKET,
            actual_temp=25.0,
            balance_point=15.0,
        )
        assert result == 0.80

    def test_cooling_extrapolation_stays_within_cooling_bucket(self):
        """Cooling at T missing; cooling data exists at neighbor temps.

        Extrapolation (thermodynamic scaling from a cooling-bucket neighbor)
        is allowed; the prediction must remain bounded by available cooling
        values, NOT leak into heating data at any nearby temp.
        """
        stats = self._stats()
        data_map = {
            "24": {COOLING_WIND_BUCKET: 0.70},
            "26": {COOLING_WIND_BUCKET: 0.90},
            # "25" is missing entirely — extrapolation scales from a cooling
            # neighbor; must NOT cross over to heating "normal" nearby.
            "23": {"normal": 1.2},  # heating distractor (well above cooling range)
        }
        result = stats._get_prediction_from_model(
            data_map,
            temp_key="25",
            wind_bucket=COOLING_WIND_BUCKET,
            actual_temp=25.0,
            balance_point=15.0,
        )
        # Result is positive (data used), and clearly not the heating
        # distractor 1.2 — stays in the cooling-value range.
        assert 0.0 < result < 1.1

    def test_resolve_bucket_cooling_returns_none_on_miss(self):
        """`_resolve_bucket_for_extrapolation` must not cross over for cooling."""
        stats = self._stats()
        # Only heating buckets present; cooling requested.
        resolved = stats._resolve_bucket_for_extrapolation(
            {"normal": 1.5, "high_wind": 1.8}, COOLING_WIND_BUCKET
        )
        assert resolved is None


class TestRetrainReplay:
    """Retrain paths must honour the same cooling-bucket routing as live learning.

    Without these, retrain from a log containing cooling hours would
    pollute heating buckets (defect 1) and cooling-mode solar NLMS would
    read `expected_unit_base = 0` from the wrong bucket and skip all
    updates (defect 2).
    """

    def _log_entry(
        self,
        *,
        entity_id: str,
        unit_mode: str,
        temp_key: str,
        wind_bucket: str,
        actual_kwh: float,
    ) -> dict:
        """Minimal hourly log entry for replay tests."""
        return {
            "temp_key": temp_key,
            "wind_bucket": wind_bucket,
            "unit_modes": {entity_id: unit_mode},
            "unit_breakdown": {entity_id: actual_kwh},
            "auxiliary_active": False,
            "learning_status": "ok",
            "solar_vector_s": 0.0,
            "solar_vector_e": 0.0,
            "solar_vector_w": 0.0,
            "correction_percent": 100.0,
        }

    def test_replay_per_unit_models_routes_cooling_to_cooling_bucket(self):
        """DirectMeter replay: cooling log hour → cooling bucket."""
        from custom_components.heating_analytics.observation import (
            DirectMeter,
            ModelState,
        )
        manager = LearningManager()
        strategies = {"ac.unit": DirectMeter("ac.unit")}
        model = ModelState(
            correlation_data={},
            correlation_data_per_unit={},
            observation_counts={},
            aux_coefficients={},
            aux_coefficients_per_unit={},
            solar_coefficients_per_unit={},
            learned_u_coefficient=None,
        )
        # 5 cooling hours at 25 °C, logged with wind_bucket="high_wind".
        entries = [
            self._log_entry(
                entity_id="ac.unit",
                unit_mode=MODE_COOLING,
                temp_key="25",
                wind_bucket="high_wind",
                actual_kwh=0.8,
            )
            for _ in range(5)
        ]
        manager.replay_per_unit_models(entries, strategies, model, learning_rate=1.0)

        # Cooling samples land in the "cooling" bucket, not "high_wind".
        assert "ac.unit" in model.correlation_data_per_unit
        assert "25" in model.correlation_data_per_unit["ac.unit"]
        assert COOLING_WIND_BUCKET in model.correlation_data_per_unit["ac.unit"]["25"]
        assert "high_wind" not in model.correlation_data_per_unit["ac.unit"]["25"]

    def test_replay_per_unit_models_preserves_heating_routing(self):
        """DirectMeter replay: heating log hour → actual-wind bucket."""
        from custom_components.heating_analytics.observation import (
            DirectMeter,
            ModelState,
        )
        manager = LearningManager()
        strategies = {"hp.unit": DirectMeter("hp.unit")}
        model = ModelState(
            correlation_data={},
            correlation_data_per_unit={},
            observation_counts={},
            aux_coefficients={},
            aux_coefficients_per_unit={},
            solar_coefficients_per_unit={},
            learned_u_coefficient=None,
        )
        entries = [
            self._log_entry(
                entity_id="hp.unit",
                unit_mode=MODE_HEATING,
                temp_key="-5",
                wind_bucket="high_wind",
                actual_kwh=2.50,
            )
            for _ in range(5)
        ]
        manager.replay_per_unit_models(entries, strategies, model, learning_rate=1.0)

        assert "high_wind" in model.correlation_data_per_unit["hp.unit"]["-5"]
        assert COOLING_WIND_BUCKET not in model.correlation_data_per_unit["hp.unit"]["-5"]

    def _run_legacy_base_learning(
        self,
        *,
        correlation_data: dict,
        unit_modes: dict,
        avg_temp: float,
        total_energy_kwh: float,
        balance_point: float = 15.0,
        temp_key: str = "5",
        wind_bucket: str = "normal",
        base_expected_kwh: float = 2.50,
    ):
        """Drive the global base-EMA path via the legacy kwargs entry point."""
        manager = LearningManager()
        manager.process_learning(
            avg_temp=avg_temp,
            temp_key=temp_key,
            wind_bucket=wind_bucket,
            hourly_sample_count=30,
            total_energy_kwh=total_energy_kwh,
            base_expected_kwh=base_expected_kwh,
            correlation_data=correlation_data,
            correlation_data_per_unit={},
            observation_counts={},
            aux_coefficients={},
            aux_coefficients_per_unit={},
            solar_coefficients_per_unit={},
            learning_buffer_global={},
            learning_buffer_per_unit={},
            learning_buffer_aux_per_unit={},
            learning_buffer_solar_per_unit={},
            hourly_delta_per_unit={sid: 0.0 for sid in unit_modes},
            hourly_expected_per_unit={sid: 0.0 for sid in unit_modes},
            hourly_expected_base_per_unit={sid: base_expected_kwh for sid in unit_modes},
            unit_modes=unit_modes,
            energy_sensors=list(unit_modes.keys()),
            learning_rate=0.05,
            balance_point=balance_point,
            solar_enabled=False,
            learning_enabled=True,
            aux_impact=0.0,
            aux_affected_entities=[],
            is_aux_active=False,
            is_cooldown_active=False,
            has_guest_activity=False,
        )

    def test_bp2_shield_blocks_cold_cooling_regime_global_write(self):
        """Global base-EMA write is skipped when all active units are in
        MODE_COOLING and avg_temp < balance_point - 2.

        Pre-shield: standby compressor consumption (0.10 kWh) would have
        been EMA-written into the cold-temp heating bucket, contaminating
        real heating data until EMA drifted it out.  With the shield, the
        bucket stays untouched.
        """
        correlation = {"5": {"normal": 2.50}}
        self._run_legacy_base_learning(
            correlation_data=correlation,
            unit_modes={"ac.unit": MODE_COOLING},
            avg_temp=5.0,  # BP=15 → BP-2=13 → 5 is well below
            total_energy_kwh=0.10,
        )
        assert correlation["5"]["normal"] == 2.50  # unchanged

    def test_bp2_shield_allows_mixed_mode_cold_hour_global_write(self):
        """Any active heating unit unblocks the shield → global writes normally."""
        correlation = {"5": {"normal": 2.50}}
        self._run_legacy_base_learning(
            correlation_data=correlation,
            unit_modes={"hp.a": MODE_HEATING, "ac.b": MODE_COOLING},
            avg_temp=5.0,
            total_energy_kwh=2.60,
        )
        assert correlation["5"]["normal"] != 2.50
        assert 2.50 < correlation["5"]["normal"] <= 2.60

    def test_bp2_shield_inactive_at_moderate_temp(self):
        """Cooling at BP-1 (above BP-2 threshold) → shield does NOT fire."""
        correlation = {"14": {"normal": 0.50}}
        self._run_legacy_base_learning(
            correlation_data=correlation,
            unit_modes={"ac.unit": MODE_COOLING},
            avg_temp=14.0,  # BP=15 → BP-2=13 → 14 is above the threshold
            total_energy_kwh=0.80,
            temp_key="14",
            base_expected_kwh=0.50,  # EMA starts from the seed
        )
        # Shield did not fire → EMA moved the bucket toward 0.80.
        assert correlation["14"]["normal"] != 0.50
        assert 0.50 < correlation["14"]["normal"] <= 0.80

    def test_replay_solar_nlms_reads_cooling_bucket_for_cooling_mode(self):
        """NLMS replay: cooling-mode entity reads `expected_unit_base` from
        the `"cooling"` wind-bucket.

        Asserted via a lambda that records the ``expected_unit_base``
        argument passed to ``_learn_unit_solar_coefficient``.  Before the
        fix, the lookup went to the heating bucket (0.0) and the NLMS
        threshold gate skipped the update entirely — no learn call.
        After the fix, the lookup hits the populated cooling bucket and
        the learn call receives the expected value.
        """
        from custom_components.heating_analytics.solar import SolarCalculator

        manager = LearningManager()
        # Pre-seed cooling bucket with a learnable value.
        expected_cooling_base = 0.95
        correlation = {
            "ac.unit": {
                "25": {COOLING_WIND_BUCKET: expected_cooling_base},
            },
        }
        # Heating bucket at same temp (distractor — must NOT be read).
        correlation["ac.unit"]["25"]["normal"] = 0.10

        # One qualifying sunny cooling hour.
        entry = {
            "temp_key": "25",
            "wind_bucket": "normal",  # raw wind, should be overridden
            "unit_modes": {"ac.unit": MODE_COOLING},
            "unit_breakdown": {"ac.unit": 0.60},
            "auxiliary_active": False,
            "learning_status": "ok",
            "solar_vector_s": 0.5,
            "solar_vector_e": 0.0,
            "solar_vector_w": 0.0,
            "correction_percent": 100.0,
        }

        captured: list[float] = []

        def _record(*, entity_id, temp_key, expected_unit_base, **_kwargs):
            captured.append(expected_unit_base)

        manager._learn_unit_solar_coefficient = _record

        coord = MagicMock()
        coord.balance_point = 15.0
        solar_calc = SolarCalculator(coord)

        manager.replay_solar_nlms(
            [entry],
            solar_calculator=solar_calc,
            screen_config=None,
            correlation_data_per_unit=correlation,
            solar_coefficients_per_unit={},
            learning_buffer_solar_per_unit={},
            energy_sensors=["ac.unit"],
            learning_rate=0.1,
            balance_point=15.0,
        )

        # The NLMS learn call was reached (not skipped by threshold),
        # and received the cooling-bucket value — not the heating-bucket
        # distractor (0.10) and not zero.
        assert captured, "NLMS replay did not reach _learn_unit_solar_coefficient for cooling entity"
        assert captured[0] == expected_cooling_base
        assert captured[0] != 0.10


class TestObservationCountModeHistory:
    """`_get_unit_observation_count` must read from the bucket the sample
    was written to, not the bucket the unit's CURRENT mode would route to.

    Regression: per #885 live callers want current-mode routing (correct
    for "count in current mode"); historical callers evaluating past log
    entries want the mode AT THE TIME of the entry so they read the bucket
    the sample was actually written to.  Before the ``mode`` parameter fix,
    a mode-switching unit's historical confidence was silently wrong.
    """

    def _make_coord(self, *, current_mode: str, counts: dict):
        """Minimal coordinator exposing just the observation-count helper."""
        from custom_components.heating_analytics.coordinator import (
            HeatingDataCoordinator,
        )
        coord = HeatingDataCoordinator.__new__(HeatingDataCoordinator)
        coord._observation_counts = counts
        coord._unit_modes = {"hp.a": current_mode}
        return coord

    def test_live_caller_uses_current_mode(self):
        """No ``mode`` arg → fall back to coordinator current mode."""
        coord = self._make_coord(
            current_mode="cooling",
            counts={"hp.a": {"5": {"cooling": 7, "normal": 3}}},
        )
        # Live caller passes the hour's physical wind bucket (normal).
        # Current-mode override routes to "cooling" → 7.
        assert coord._get_unit_observation_count("hp.a", "5", "normal") == 7

    def test_historical_caller_passes_log_mode(self):
        """Explicit ``mode`` overrides the current-mode lookup."""
        # Unit is CURRENTLY in heating.  Log entry was from when it was
        # in cooling — samples went to the "cooling" bucket.
        coord = self._make_coord(
            current_mode="heating",
            counts={"hp.a": {"5": {"cooling": 4, "normal": 0}}},
        )
        # Without mode override → current-mode routing returns wrong bucket.
        assert coord._get_unit_observation_count("hp.a", "5", "normal") == 0
        # With mode=cooling (from the log entry) → reads historical bucket.
        assert (
            coord._get_unit_observation_count("hp.a", "5", "normal", mode="cooling")
            == 4
        )

    def test_historical_caller_heating_mode_reads_wind_bucket(self):
        """Log mode=heating → read the passed physical wind bucket as-is."""
        coord = self._make_coord(
            current_mode="cooling",  # different from log
            counts={"hp.a": {"-5": {"high_wind": 5, "cooling": 1}}},
        )
        # Log was in heating at -5 C with high_wind.  With mode=heating,
        # no override → reads the passed "high_wind" bucket directly.
        assert (
            coord._get_unit_observation_count("hp.a", "-5", "high_wind", mode="heating")
            == 5
        )
        # Without the mode override, current-mode=cooling would route to
        # "cooling" (wrong for this historical entry) → returns 1.
        assert coord._get_unit_observation_count("hp.a", "-5", "high_wind") == 1

    def test_calculate_historical_expectations_passes_log_mode(self):
        """Integration: `_calculate_historical_expectations` threads the log's
        `unit_modes[entity]` into the count lookup so a mode-switching
        unit reports the correct historical confidence.
        """
        from custom_components.heating_analytics.statistics import (
            StatisticsManager,
        )

        coord = MagicMock()
        coord._unit_modes = {"hp.a": "heating"}  # CURRENT mode
        # Unit was cooling for these log hours — samples landed in "cooling".
        coord._observation_counts = {
            "hp.a": {"25": {"cooling": 3, "normal": 0}},
        }

        def _count(entity_id, temp_key, wind_bucket, mode=None):
            effective = mode if mode is not None else coord._unit_modes.get(entity_id, "heating")
            if effective == "cooling":
                wind_bucket = "cooling"
            return coord._observation_counts.get(entity_id, {}).get(temp_key, {}).get(wind_bucket, 0)

        coord._get_unit_observation_count = _count
        coord._get_wind_bucket = MagicMock(return_value="normal")

        stats = StatisticsManager(coord)

        processed_logs = [
            {
                "log": {
                    "temp_key": "25",
                    "wind_bucket": "normal",
                    "unit_modes": {"hp.a": "cooling"},  # mode at log time
                },
                "reconstructed_breakdown": {"hp.a": 0.5},
            },
            {
                "log": {
                    "temp_key": "25",
                    "wind_bucket": "normal",
                    "unit_modes": {"hp.a": "cooling"},
                },
                "reconstructed_breakdown": {"hp.a": 0.5},
            },
        ]

        _, obs_count_sum, hours = stats._calculate_historical_expectations(
            "hp.a", processed_logs
        )

        # Each of the 2 log entries reads count=3 from the cooling bucket
        # (because log_mode=cooling overrides current-mode=heating).
        # Pre-fix: would have read "normal" (0) on both → obs_count_sum=0.
        assert hours == 2
        assert obs_count_sum == 6  # 3 per entry × 2 entries
