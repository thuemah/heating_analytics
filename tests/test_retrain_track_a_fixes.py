"""Regression tests for Track A retrain_from_history fixes (#847 follow-up).

Two bugs fixed:

1. **Solar normalization missing.** ``learn_from_historical_import`` wrote
   raw ``actual_kwh`` into the base bucket EMA.  Live Track A learning
   writes ``actual + solar_impact`` (dark-equivalent).  Retrain therefore
   converged toward a different surface than live learning — invisible
   but material on installations with any sunny hours.  Fix: pass
   ``solar_normalization_delta`` through and add before EMA.

2. **Per-unit replay missing.** Track A retrain updated
   ``correlation_data`` (global) but never called
   ``_replay_per_unit_models``.  After ``reset_first=True`` the per-unit
   tables stayed empty while global was populated, breaking the
   invariant that per-unit predictions sum to global and silently
   breaking ``isolate_sensor`` subtraction.  Fix: call replay after
   the per-hour loop completes.
"""
from unittest.mock import MagicMock, AsyncMock

import pytest

from custom_components.heating_analytics.learning import LearningManager
from custom_components.heating_analytics.retrain import RetrainEngine


# -----------------------------------------------------------------------------
# Unit-level: learn_from_historical_import now accepts & applies the delta
# -----------------------------------------------------------------------------

class TestLearnFromHistoricalImportSolarNormalization:
    """learn_from_historical_import applies solar_normalization_delta."""

    def _call(self, *, actual_kwh, solar_delta, current_bucket=0.0):
        lm = LearningManager()
        corr = {"10": {"normal": current_bucket}} if current_bucket else {}
        aux = {}
        lm.learn_from_historical_import(
            temp_key="10",
            wind_bucket="normal",
            actual_kwh=actual_kwh,
            is_aux_active=False,
            correlation_data=corr,
            aux_coefficients=aux,
            learning_rate=0.5,  # high rate so one call moves the value visibly
            get_predicted_kwh_fn=lambda *_a, **_k: 0.0,
            actual_temp=10.0,
            solar_normalization_delta=solar_delta,
        )
        return corr["10"]["normal"]

    def test_delta_zero_preserves_pre_fix_behaviour(self):
        """delta=0 → bucket written equals raw actual (first-observation path)."""
        v = self._call(actual_kwh=0.3, solar_delta=0.0)
        assert v == pytest.approx(0.3)

    def test_aux_branch_also_uses_normalized_value(self):
        """is_aux_active=True path computes implied_aux_reduction from
        base_prediction - normalized_actual, so the delta must apply here too.
        """
        lm = LearningManager()
        corr = {}
        aux = {}
        lm.learn_from_historical_import(
            temp_key="10",
            wind_bucket="normal",
            actual_kwh=0.5,          # raw
            is_aux_active=True,
            correlation_data=corr,
            aux_coefficients=aux,
            learning_rate=1.0,       # take the target directly (pre-existing code
                                     # uses implied_aux_reduction directly on first write)
            get_predicted_kwh_fn=lambda *_a, **_k: 1.5,  # base_prediction
            actual_temp=10.0,
            solar_normalization_delta=0.3,
        )
        # normalized_actual = 0.5 + 0.3 = 0.8
        # implied_aux_reduction = 1.5 - 0.8 = 0.7
        assert aux["10"]["normal"] == pytest.approx(0.7)


# -----------------------------------------------------------------------------
# Integration-level: retrain_from_history Track A end-to-end
# -----------------------------------------------------------------------------

def _track_a_coord_with_log(hourly_log):
    """Build a coord mock suitable for calling retrain_from_history as unbound."""
    from custom_components.heating_analytics.solar import SolarCalculator
    coord = MagicMock()
    coord._hourly_log = hourly_log
    coord.daily_learning_mode = False
    coord.learning_rate = 1.0  # first-observation branch writes target directly
    coord.balance_point = 15.0
    coord.energy_sensors = ["sensor.heater1", "sensor.heater2"]
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
    # Use a real SolarCalculator so the NLMS-replay path (which calls
    # _screen_transmittance_vector) works end-to-end.  Legacy tests that
    # didn't set screen_config get a defensive fallback inside the solar
    # layer.
    coord.solar = SolarCalculator(coord)

    # Track per-unit replay calls so tests can assert it ran
    replay_calls = []
    def _replay(entries):
        replay_calls.append(entries)
        # Write per-unit values so tests can verify side effect
        for entry in entries:
            tk = entry.get("temp_key")
            wb = entry.get("wind_bucket")
            if tk is None or wb is None:
                continue
            for sid, kwh in (entry.get("unit_breakdown") or {}).items():
                if kwh <= 0:
                    continue
                coord._correlation_data_per_unit.setdefault(sid, {}).setdefault(tk, {})[wb] = kwh

    coord._replay_per_unit_models = MagicMock(side_effect=_replay)
    coord._replay_calls = replay_calls
    return coord


def _entry(ts, *, actual=0.5, solar_delta=0.0, aux=False, status=None,
           unit_breakdown=None, temp=10.0):
    e = {
        "timestamp": ts,
        "actual_kwh": actual,
        "temp": temp,
        # Real hourly_log entries include a pre-bucketed temp_key from the
        # live coordinator; per-unit replay reads it directly.
        "temp_key": str(int(round(temp))),
        "wind_bucket": "normal",
        "auxiliary_active": aux,
        "solar_normalization_delta": solar_delta,
        "unit_breakdown": unit_breakdown or {},
    }
    if status is not None:
        e["learning_status"] = status
    return e


class TestTrackARetrainSolarNormalization:
    """retrain_from_history Track A passes solar delta to learn_from_historical_import."""

    @pytest.mark.asyncio
    async def test_missing_solar_delta_field_treated_as_zero(self):
        """Legacy log entries without solar_normalization_delta stay backward compat."""
        from custom_components.heating_analytics.coordinator import HeatingDataCoordinator

        entry = _entry("2026-04-10T12:00:00", actual=0.35)
        entry.pop("solar_normalization_delta", None)  # legacy log
        coord = _track_a_coord_with_log([entry])
        await RetrainEngine(coord).retrain_from_history()

        assert coord._correlation_data["10"]["normal"] == pytest.approx(0.35)


class TestTrackARetrainPerUnitReplay:
    """retrain_from_history Track A calls _replay_per_unit_models."""

    @pytest.mark.asyncio
    async def test_per_unit_replay_called_with_processed_entries(self):
        from custom_components.heating_analytics.coordinator import HeatingDataCoordinator

        entries = [
            _entry("2026-04-10T12:00:00", actual=0.3,
                   unit_breakdown={"sensor.heater1": 0.2, "sensor.heater2": 0.1}),
            _entry("2026-04-10T13:00:00", actual=0.4,
                   unit_breakdown={"sensor.heater1": 0.25, "sensor.heater2": 0.15}),
        ]
        coord = _track_a_coord_with_log(entries)
        await RetrainEngine(coord).retrain_from_history()

        # Replay was called once with both entries
        assert coord._replay_per_unit_models.call_count == 1
        replayed = coord._replay_calls[0]
        assert len(replayed) == 2
        # Per-unit side effect is visible
        assert coord._correlation_data_per_unit["sensor.heater1"]["10"]["normal"] == pytest.approx(0.25)
        assert coord._correlation_data_per_unit["sensor.heater2"]["10"]["normal"] == pytest.approx(0.15)

    @pytest.mark.asyncio
    async def test_per_unit_replay_skipped_when_no_processed_entries(self):
        """All entries poisoned → no replay call (avoids empty work)."""
        from custom_components.heating_analytics.coordinator import HeatingDataCoordinator

        coord = _track_a_coord_with_log([
            _entry("2026-04-10T12:00:00", status="skipped_bad_data"),
        ])
        await RetrainEngine(coord).retrain_from_history()

        assert coord._replay_per_unit_models.call_count == 0

    @pytest.mark.asyncio
    async def test_per_unit_replay_excludes_poisoned_hours(self):
        """Poisoned entries are excluded from replay input."""
        from custom_components.heating_analytics.coordinator import HeatingDataCoordinator

        entries = [
            _entry("2026-04-10T12:00:00", actual=0.3,
                   unit_breakdown={"sensor.heater1": 0.3}),
            _entry("2026-04-10T13:00:00", status="skipped_bad_data",
                   unit_breakdown={"sensor.heater1": 999.0}),  # poison value
            _entry("2026-04-10T14:00:00", actual=0.4,
                   unit_breakdown={"sensor.heater1": 0.4}),
        ]
        coord = _track_a_coord_with_log(entries)
        await RetrainEngine(coord).retrain_from_history()

        replayed = coord._replay_calls[0]
        # Poisoned entry NOT in replay input
        assert len(replayed) == 2
        kwhs = [e["unit_breakdown"]["sensor.heater1"] for e in replayed]
        assert 999.0 not in kwhs
