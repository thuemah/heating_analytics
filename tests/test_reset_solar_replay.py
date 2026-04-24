"""Tests for solar-only retrain via reset_solar_learning (#862).

Verifies that ``async_reset_solar_learning_data(replay_from_history=True)``:

1. Clears solar coefficients + buffers (existing reset behaviour).
2. Replays NLMS against the existing base model via
   ``LearningManager.replay_solar_nlms`` — producing non-empty coefficients
   when the log has qualifying sunny hours.
3. Leaves ``correlation_data``, ``correlation_data_per_unit``,
   ``aux_coefficients``, and ``_learned_u_coefficient`` untouched
   (solar-only semantics — the whole point of this path).
4. Respects the ``unit_entity_id`` filter: only the named unit's
   coefficients are affected; other units' coefficients are preserved.
5. Respects ``days_back`` by scoping the replay window.
6. Returns the contract dict (status + diagnostics).

Pure-reset path (no replay) is also asserted to still return the
contract dict so the new SupportsResponse.ONLY contract holds.
"""
from unittest.mock import MagicMock, AsyncMock

import pytest

from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.learning import LearningManager
from custom_components.heating_analytics.solar import SolarCalculator
from custom_components.heating_analytics.observation import build_strategies
from custom_components.heating_analytics.const import MODE_HEATING


def _reset_coord(hourly_log, *, energy_sensors=("sensor.heater1",)):
    """Build a coordinator stub for reset_solar_learning tests.

    Base and per-unit correlation are pre-populated so the NLMS replay
    has a base reference above SOLAR_LEARNING_MIN_BASE (0.15 kWh).
    """
    coord = MagicMock()
    coord._hourly_log = list(hourly_log)
    coord.daily_learning_mode = False
    coord.learning_rate = 1.0
    coord.balance_point = 15.0
    coord.energy_sensors = list(energy_sensors)
    coord.aux_affected_entities = []
    coord.screen_config = (True, True, True)
    coord._correlation_data = {"10": {"normal": 1.5}}
    coord._correlation_data_per_unit = {
        sid: {"10": {"normal": 1.0}} for sid in energy_sensors
    }
    coord._aux_coefficients = {"10": {"normal": 0.3}}
    coord._aux_coefficients_per_unit = {}
    coord._learning_buffer_global = {}
    coord._learning_buffer_per_unit = {}
    coord._learning_buffer_aux_per_unit = {}
    coord._solar_coefficients_per_unit = {}
    coord._learning_buffer_solar_per_unit = {}
    coord._observation_counts = {}
    coord._learned_u_coefficient = 0.15
    coord._daily_history = {}
    coord._async_save_data = AsyncMock()
    coord.learning = LearningManager()
    coord.solar = SolarCalculator(coord)
    coord._unit_strategies = build_strategies(
        energy_sensors=list(energy_sensors),
        track_c_enabled=False,
        mpc_managed_sensor=None,
    )
    return coord


def _sunny_entry(ts, *, sensor_id="sensor.heater1", actual=0.3,
                 solar_s=0.6, solar_e=0.1, solar_w=0.05, temp=10.0):
    """Build a qualifying sunny hour. Defaults are above NLMS thresholds."""
    return {
        "timestamp": ts,
        "hour": int(ts[11:13]),
        "temp": temp,
        "temp_key": str(int(round(temp))),
        "wind_bucket": "normal",
        "actual_kwh": actual,
        "auxiliary_active": False,
        "solar_factor": solar_s,
        "solar_vector_s": solar_s,
        "solar_vector_e": solar_e,
        "solar_vector_w": solar_w,
        "correction_percent": 100.0,
        "unit_modes": {},  # all-heating hour (MODE_HEATING not stored by design)
        "unit_breakdown": {sensor_id: actual},
        "solar_dominant_entities": [],
        "solar_normalization_delta": 0.0,  # irrelevant for NLMS replay path
    }


# -----------------------------------------------------------------------------
# Pure reset (no replay) — existing behaviour plus new return contract
# -----------------------------------------------------------------------------

class TestPureReset:

    @pytest.mark.asyncio
    async def test_all_units_reset_returns_contract_dict(self):
        coord = _reset_coord([])
        coord._solar_coefficients_per_unit = {
            "sensor.heater1": {"s": 0.2, "e": 0.1, "w": 0.05}
        }
        coord._learning_buffer_solar_per_unit = {
            "sensor.heater1": {"10": {"normal": [(1.0, 1.0, (0.5, 0.1, 0.0))]}}
        }
        result = await HeatingDataCoordinator.async_reset_solar_learning_data(coord)
        assert result == {
            "status": "reset",
            "unit_entity_id": None,
            "replay_from_history": False,
            "solar_replay_diagnostics": None,
        }
        assert coord._solar_coefficients_per_unit == {}
        assert coord._learning_buffer_solar_per_unit == {}

    @pytest.mark.asyncio
    async def test_single_unit_reset_preserves_others(self):
        coord = _reset_coord([], energy_sensors=("sensor.heater1", "sensor.heater2"))
        coord._solar_coefficients_per_unit = {
            "sensor.heater1": {"s": 0.2, "e": 0.1, "w": 0.05},
            "sensor.heater2": {"s": 0.3, "e": 0.0, "w": 0.02},
        }
        result = await HeatingDataCoordinator.async_reset_solar_learning_data(
            coord, entity_id="sensor.heater1"
        )
        assert result["unit_entity_id"] == "sensor.heater1"
        assert result["replay_from_history"] is False
        assert "sensor.heater1" not in coord._solar_coefficients_per_unit
        assert coord._solar_coefficients_per_unit["sensor.heater2"] == {
            "s": 0.3, "e": 0.0, "w": 0.02
        }


# -----------------------------------------------------------------------------
# Replay from history — core contract
# -----------------------------------------------------------------------------

class TestReplayFromHistory:

    @pytest.mark.asyncio
    async def test_replay_produces_updates_on_sunny_log(self):
        """Enough sunny hours → NLMS replay triggers updates."""
        entries = [
            _sunny_entry(f"2026-04-{10 + i // 24:02d}T{i % 24:02d}:00:00")
            for i in range(30)
        ]
        coord = _reset_coord(entries)
        result = await HeatingDataCoordinator.async_reset_solar_learning_data(
            coord, replay_from_history=True
        )
        assert result["status"] == "reset"
        assert result["replay_from_history"] is True
        diag = result["solar_replay_diagnostics"]
        assert diag is not None
        assert diag["updates"] > 0
        assert diag["entries_considered"] == 30

    @pytest.mark.asyncio
    async def test_replay_preserves_base_model(self):
        """Solar-only semantics: base/aux/U-coeff must survive replay unchanged."""
        entries = [
            _sunny_entry(f"2026-04-{10 + i // 24:02d}T{i % 24:02d}:00:00")
            for i in range(30)
        ]
        coord = _reset_coord(entries)
        correlation_before = {k: dict(v) for k, v in coord._correlation_data.items()}
        per_unit_before = {
            sid: {tk: dict(wb) for tk, wb in buckets.items()}
            for sid, buckets in coord._correlation_data_per_unit.items()
        }
        aux_before = {k: dict(v) for k, v in coord._aux_coefficients.items()}
        u_before = coord._learned_u_coefficient

        await HeatingDataCoordinator.async_reset_solar_learning_data(
            coord, replay_from_history=True
        )

        assert coord._correlation_data == correlation_before
        assert coord._correlation_data_per_unit == per_unit_before
        assert coord._aux_coefficients == aux_before
        assert coord._learned_u_coefficient == u_before

    @pytest.mark.asyncio
    async def test_replay_populates_coefficients(self):
        """After reset+replay, coefficients exist and are learned (non-default)."""
        entries = [
            _sunny_entry(f"2026-04-{10 + i // 24:02d}T{i % 24:02d}:00:00")
            for i in range(30)
        ]
        coord = _reset_coord(entries)
        coord._solar_coefficients_per_unit = {
            "sensor.heater1": {"s": 99.0, "e": 99.0, "w": 99.0}  # clearly wrong starting point
        }
        await HeatingDataCoordinator.async_reset_solar_learning_data(
            coord, replay_from_history=True
        )
        coeff = coord._solar_coefficients_per_unit.get("sensor.heater1")
        assert coeff is not None, "replay must produce a coefficient"
        # Not the stale 99.0 values
        assert coeff["s"] != 99.0

    @pytest.mark.asyncio
    async def test_replay_no_sunny_hours_returns_zero_updates(self):
        """Replay over dark-only log produces no updates (vector magnitude gate)."""
        entries = []
        for i in range(20):
            e = _sunny_entry(f"2026-04-10T{i:02d}:00:00")
            e["solar_factor"] = 0.0
            e["solar_vector_s"] = 0.0
            e["solar_vector_e"] = 0.0
            e["solar_vector_w"] = 0.0
            entries.append(e)
        coord = _reset_coord(entries)
        result = await HeatingDataCoordinator.async_reset_solar_learning_data(
            coord, replay_from_history=True
        )
        diag = result["solar_replay_diagnostics"]
        assert diag["updates"] == 0
        assert diag["entry_skipped_low_magnitude"] == 20


# -----------------------------------------------------------------------------
# Per-unit filter via energy_sensors list
# -----------------------------------------------------------------------------

class TestPerUnitReplayFilter:

    @pytest.mark.asyncio
    async def test_unit_filter_only_updates_named_unit(self):
        """With entity_id set, other units' coefficients must be untouched."""
        sensors = ("sensor.heater1", "sensor.heater2")
        entries = []
        for i in range(30):
            ts = f"2026-04-{10 + i // 24:02d}T{i % 24:02d}:00:00"
            # Entry sees both units as active with same breakdown
            e = _sunny_entry(ts, sensor_id="sensor.heater1")
            e["unit_breakdown"] = {
                "sensor.heater1": 0.3,
                "sensor.heater2": 0.4,
            }
            entries.append(e)
        coord = _reset_coord(entries, energy_sensors=sensors)
        # Pre-seed heater2 with a known coefficient to verify it's preserved
        coord._solar_coefficients_per_unit = {
            "sensor.heater2": {"s": 0.5, "e": 0.5, "w": 0.5}
        }
        pre_heater2 = dict(coord._solar_coefficients_per_unit["sensor.heater2"])

        await HeatingDataCoordinator.async_reset_solar_learning_data(
            coord,
            entity_id="sensor.heater1",
            replay_from_history=True,
        )

        # heater2 completely preserved
        assert coord._solar_coefficients_per_unit.get("sensor.heater2") == pre_heater2
        # heater1 exists (either learned a coeff or is in cold-start buffer).
        # The key invariant is: replay did NOT touch heater2.


# -----------------------------------------------------------------------------
# days_back window filter
# -----------------------------------------------------------------------------

class TestDaysBackFilter:

    @pytest.mark.asyncio
    async def test_days_back_restricts_replay_window(self, monkeypatch):
        """days_back filters entries before replay — fewer entries considered."""
        import datetime as _dt

        # Pin "now" so days_back cutoff is deterministic
        class _FixedDt:
            @staticmethod
            def now():
                return _dt.datetime(2026, 5, 1, 12, 0, 0)

        from homeassistant.util import dt as dt_util
        monkeypatch.setattr(dt_util, "now", _FixedDt.now)

        old_entries = [
            _sunny_entry(f"2026-04-01T{i:02d}:00:00") for i in range(10)
        ]
        recent_entries = [
            _sunny_entry(f"2026-04-28T{i:02d}:00:00") for i in range(10)
        ]
        coord = _reset_coord(old_entries + recent_entries)

        result = await HeatingDataCoordinator.async_reset_solar_learning_data(
            coord, replay_from_history=True, days_back=7
        )
        diag = result["solar_replay_diagnostics"]
        # 7-day window from 2026-05-01 includes only the 2026-04-28 set
        assert diag["entries_considered"] == 10
