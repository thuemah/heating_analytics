"""Tests for the cooling-at-cold per-unit base-learning guard (#869 follow-up).

The guard skips per-unit base-EMA updates when a unit is in cooling mode
and the hour's outdoor temperature is below ``balance_point − 2``.
Rationale: a unit left in cooling mode across a full season (seasonal-
automation pattern) produces idle-compressor consumption on cold nights
that contaminates the heating-dominated per-unit base bucket.  Guard
matches the transition-zone boundary in
``diagnose_solar.temperature_stratified``.

Three targeted behaviours:
1. Cooling + temp < BP − 2 → per-unit bucket NOT updated
2. Cooling + temp ≥ BP − 2 → per-unit bucket updated normally
3. Heating + temp < BP − 2 → per-unit bucket updated normally (guard is cooling-only)

Plus a retrain-path check: the guard is live-learning only; replay-
based retrain populates buckets from raw log data without the guard
(so historical data remains available for future mode-stratified work).
"""
from unittest.mock import MagicMock, AsyncMock

import pytest

from custom_components.heating_analytics.learning import LearningManager
from custom_components.heating_analytics.solar import SolarCalculator
from custom_components.heating_analytics.observation import build_strategies
from custom_components.heating_analytics.const import (
    MODE_COOLING,
    MODE_HEATING,
)


def _base_kwargs(*, unit_mode=MODE_HEATING, avg_temp=10.0, balance_point=17.0):
    """Minimal kwargs for LearningManager._process_per_unit_learning.

    Pre-seeds a per-unit bucket at ("10", "normal") with value 1.0 so we
    can observe whether the EMA moves after one iteration.  ``actual``
    delivered by ``hourly_delta_per_unit`` is 0.5 — different enough
    from 1.0 that a real EMA update would shift the bucket noticeably.
    """
    solar = MagicMock()
    solar.calculate_unit_coefficient.return_value = {"s": 0.0, "e": 0.0, "w": 0.0}
    solar.calculate_unit_solar_impact.return_value = 0.0
    solar.normalize_for_learning.side_effect = lambda a, s, m: a  # no-op for this test

    return dict(
        temp_key="10",
        wind_bucket="normal",
        avg_temp=avg_temp,
        avg_solar_vector=(0.0, 0.0, 0.0),  # dark hour, no NLMS
        total_energy_kwh=0.5,
        base_expected_kwh=1.0,
        energy_sensors=["unit_a"],
        hourly_delta_per_unit={"unit_a": 0.5},
        solar_enabled=True,
        learning_rate=1.0,  # aggressive for observable shift
        solar_calculator=solar,
        get_predicted_unit_base_fn=MagicMock(return_value=1.0),
        learning_buffer_per_unit={},
        correlation_data_per_unit={"unit_a": {"10": {"normal": 1.0}}},
        observation_counts={},
        is_aux_active=False,
        aux_coefficients_per_unit={},
        learning_buffer_aux_per_unit={},
        solar_coefficients_per_unit={},
        learning_buffer_solar_per_unit={},
        balance_point=balance_point,
        unit_modes={"unit_a": unit_mode},
        hourly_expected_per_unit={"unit_a": 1.0},
        hourly_expected_base_per_unit={"unit_a": 1.0},
        aux_affected_entities=None,
        is_cooldown_active=False,
        correction_percent=100.0,
        solar_dominant_entities=(),
        solar_factor=0.0,
        battery_filtered_potential=(0.0, 0.0, 0.0),
    )


class TestCoolingColdGuard:
    """Three targeted assertions for the guard matrix (mode × temp)."""

    def test_cooling_below_bp_minus_2_skips_update(self):
        """Mitsubishi case: cooling mode, cold outdoor temp → no learning."""
        lm = LearningManager()
        kwargs = _base_kwargs(
            unit_mode=MODE_COOLING,
            avg_temp=5.0,  # well below balance_point − 2 (= 15.0)
            balance_point=17.0,
        )
        lm._process_per_unit_learning(**kwargs)
        # Bucket must remain at the seeded 1.0 value — no EMA movement.
        assert kwargs["correlation_data_per_unit"]["unit_a"]["10"]["normal"] == pytest.approx(1.0)

    def test_cooling_above_threshold_updates_normally(self):
        """Legitimate cooling (mild/warm temps): learning proceeds."""
        lm = LearningManager()
        kwargs = _base_kwargs(
            unit_mode=MODE_COOLING,
            avg_temp=20.0,  # well above balance_point − 2 (= 15.0)
            balance_point=17.0,
        )
        lm._process_per_unit_learning(**kwargs)
        bucket = kwargs["correlation_data_per_unit"]["unit_a"]["10"]["normal"]
        # With lr=1.0, raw target=0.5 (no-op normalize), EMA moves from 1.0
        # toward 0.5.  Exact value depends on headroom/SNR multipliers;
        # the key claim is the bucket MOVED.
        assert bucket != pytest.approx(1.0), f"expected EMA movement, got {bucket}"

    def test_heating_below_threshold_updates_normally(self):
        """Heating is unaffected by the cooling-mode guard."""
        lm = LearningManager()
        kwargs = _base_kwargs(
            unit_mode=MODE_HEATING,
            avg_temp=5.0,  # cold, but heating mode — no guard applies
            balance_point=17.0,
        )
        lm._process_per_unit_learning(**kwargs)
        bucket = kwargs["correlation_data_per_unit"]["unit_a"]["10"]["normal"]
        assert bucket != pytest.approx(1.0), f"expected EMA movement, got {bucket}"

    def test_cooling_exactly_at_threshold_included(self):
        """Boundary: temp == BP − 2 is strictly NOT below the threshold."""
        lm = LearningManager()
        kwargs = _base_kwargs(
            unit_mode=MODE_COOLING,
            avg_temp=15.0,  # exactly balance_point − 2
            balance_point=17.0,
        )
        lm._process_per_unit_learning(**kwargs)
        bucket = kwargs["correlation_data_per_unit"]["unit_a"]["10"]["normal"]
        assert bucket != pytest.approx(1.0), f"boundary hour must still learn; got {bucket}"

    def test_retrain_path_not_guarded_architecturally(self):
        """Guard lives in _process_per_unit_learning; retrain uses a different path.

        Retrain calls ``replay_per_unit_models`` (via coordinator delegate)
        and ``learn_from_historical_import`` — neither of which invoke
        ``_process_per_unit_learning``.  The guard therefore cannot fire
        during retrain, by construction.  Preserves historical data for
        future mode-stratified work (#869).
        """
        import inspect
        from custom_components.heating_analytics.learning import LearningManager as _LM

        replay_src = inspect.getsource(_LM.replay_per_unit_models)
        hist_import_src = inspect.getsource(_LM.learn_from_historical_import)

        # Neither retrain helper should call _process_per_unit_learning.
        assert "_process_per_unit_learning" not in replay_src
        assert "_process_per_unit_learning" not in hist_import_src
        # Nor should they reference the guard's mode check that lives in
        # the live path (defensive: if someone later refactored guard
        # logic into a shared helper, this test forces them to make an
        # explicit decision about retrain as well).
        assert "balance_point - 2" not in replay_src
        assert "balance_point - 2" not in hist_import_src
