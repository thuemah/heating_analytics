"""Stored degree-days after a balance-point change.

``tdd`` is ``|BP − T| / 24`` at the BP in force when an hour was logged, and
``daily_history`` keeps its sum forever.  After the user changes the BP,
every consumer that reads it back must either recompute it at the current
BP or know it still applies — rebuilding a temperature as ``BP_now ± tdd``
from a tdd computed at another BP moves the whole day by the change.
"""
from __future__ import annotations

from datetime import date
import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

from custom_components.heating_analytics.daily_processor import DailyProcessor
from custom_components.heating_analytics.helpers import (
    daily_tdd_at_balance_point,
    infer_tdd_balance_point,
    recorded_balance_points,
    stored_tdd_matches,
)
from custom_components.heating_analytics.retrain import RetrainEngine
from custom_components.heating_analytics.statistics import StatisticsManager
from custom_components.heating_analytics.storage import (
    StorageManager,
    _migrate_v9_to_v10,
)
from tests.test_csv_import_weather_only import storage_manager  # noqa: F401
from tests.test_reconstruction_logic import MockCoordinator
from tests.test_snr_weighted_learning import _daily_coord, _full_day_entries
from tests.test_storage_migration import _make_coord as _make_restore_coord
from tests.test_storage_migration_v7_v8 import _make_coord

OLD_BP = 15.0
NEW_BP = 18.0
DAY = "2023-01-10"


def _vector_day(temps: list[float], bp: float, *, stamp: bool = True) -> dict:
    """A daily_history day aggregated at ``bp`` with a full hourly vector."""
    tdd_h = [None if t is None else round(abs(bp - t) / 24.0, 3) for t in temps]
    logged = [t for t in temps if t is not None]
    day = {
        "kwh": float(len(logged)),
        "temp": round(sum(logged) / len(logged), 1),
        "tdd": round(sum(v for v in tdd_h if v is not None), 1),
        "wind": 0.0,
        "hourly_vectors": {
            "temp": list(temps),
            "wind": [0.0] * 24,
            "tdd": tdd_h,
            "actual_kwh": [None if t is None else 1.0 for t in temps],
        },
    }
    if stamp:
        day["balance_point"] = bp
    return day


class TestHelpers:
    def test_recorded_balance_points(self):
        assert recorded_balance_points([{"bp_at_log_time": 17.0}] * 3) == {17.0}
        assert recorded_balance_points(
            [{"bp_at_log_time": 17.0}, {"bp_at_log_time": 18.0}]
        ) == {17.0, 18.0}
        # Hours predating the field do not vote.
        assert recorded_balance_points([{"bp_at_log_time": 17.0}, {}]) == {17.0}
        assert recorded_balance_points([]) == set()
        assert recorded_balance_points([{"bp_at_log_time": True}]) == set()

    def test_stored_tdd_matches(self):
        assert stored_tdd_matches({"balance_point": 17.0}, 17.0)
        assert not stored_tdd_matches({"balance_point": 15.0}, 17.0)
        # Recorded as mixed / unknown: applies at no BP.
        assert not stored_tdd_matches({"balance_point": None}, 17.0)
        # Predates the v10 stamp.
        assert stored_tdd_matches({}, 17.0)

    def test_stored_value_while_it_applies(self):
        day = _vector_day([5.0] * 12 + [10.0] * 12, OLD_BP)
        assert daily_tdd_at_balance_point(day, OLD_BP) == day["tdd"]

    def test_recomputed_from_the_hourly_vector(self):
        day = _vector_day([5.0] * 12 + [10.0] * 12, OLD_BP)
        assert day["tdd"] == pytest.approx(7.5)
        assert daily_tdd_at_balance_point(day, NEW_BP) == pytest.approx(10.5)

    def test_mixed_day_is_recomputed(self):
        day = _vector_day([5.0] * 24, OLD_BP)
        day["balance_point"] = None
        assert daily_tdd_at_balance_point(day, OLD_BP) == pytest.approx(10.0)

    def test_without_a_vector_a_changed_bp_falls_back_to_the_mean(self):
        day = {"tdd": 7.5, "temp": 7.5, "balance_point": OLD_BP}
        assert daily_tdd_at_balance_point(day, NEW_BP) == pytest.approx(10.5)

    def test_missing_tdd_is_computed_from_the_mean(self):
        assert daily_tdd_at_balance_point({"temp": 5.0}, OLD_BP) == pytest.approx(10.0)
        assert daily_tdd_at_balance_point({}, OLD_BP) == 0.0

    def test_infer_reads_the_bp_off_the_vectors(self):
        temps = [float(t) for t in range(-5, 19)]  # straddles 17
        assert infer_tdd_balance_point(_vector_day(temps, 17.0)) == (True, 17.0)
        assert infer_tdd_balance_point(_vector_day([5.0] * 24, 16.5)) == (True, 16.5)

    def test_infer_rejects_a_day_logged_under_two_bps(self):
        day = _vector_day([5.0] * 24, OLD_BP)
        day["hourly_vectors"]["tdd"][12:] = [round(13 / 24, 3)] * 12
        assert infer_tdd_balance_point(day) == (True, None)

    def test_infer_tolerates_one_folded_slot(self):
        day = _vector_day([5.0] * 24, OLD_BP)
        day["hourly_vectors"]["tdd"][2] *= 2  # two rows summed into one slot
        assert infer_tdd_balance_point(day) == (True, OLD_BP)

    def test_infer_abstains_without_enough_slots(self):
        assert infer_tdd_balance_point({"tdd": 3.0}) == (False, None)
        day = _vector_day([OLD_BP] * 24, OLD_BP)  # every slot on the BP
        assert infer_tdd_balance_point(day) == (False, None)


@pytest.fixture
def stats():
    coord = MockCoordinator()
    coord.balance_point = NEW_BP
    manager = StatisticsManager(coord)
    # Modelled kWh per hour = the temperature it is evaluated at, so the
    # result exposes exactly which temperatures reached the model.
    manager._get_prediction_from_model = MagicMock(
        side_effect=lambda data_map, temp_key, wind_bucket, actual_temp, bp,
        apply_scaling=True: float(actual_temp)
    )
    return manager, coord


class TestModeledEnergy:
    def test_vector_day_is_evaluated_at_its_own_temperatures(self, stats):
        """The day aggregated at BP 15 must not move 3 °C under BP 18."""
        manager, coord = stats
        temps = [5.0] * 12 + [10.0] * 12
        coord._daily_history = {DAY: _vector_day(temps, OLD_BP)}
        d = date.fromisoformat(DAY)

        kwh, _, avg_temp, _, total_tdd = manager.calculate_modeled_energy(d, d)

        assert kwh == pytest.approx(sum(temps))
        assert avg_temp == pytest.approx(7.5)
        # Degree-days at the current BP, comparable with a day logged today.
        assert total_tdd == pytest.approx(10.5)

    def test_vector_day_unchanged_when_bp_unchanged(self, stats):
        manager, coord = stats
        coord.balance_point = OLD_BP
        temps = [5.0] * 12 + [10.0] * 12
        coord._daily_history = {DAY: _vector_day(temps, OLD_BP)}
        d = date.fromisoformat(DAY)

        kwh, _, avg_temp, _, total_tdd = manager.calculate_modeled_energy(d, d)

        assert kwh == pytest.approx(sum(temps))
        assert avg_temp == pytest.approx(7.5)
        assert total_tdd == pytest.approx(7.5, abs=0.05)

    def test_daily_average_day_keeps_its_reconstruction_at_its_bp(self, stats):
        """Straddling day: T_avg 16, but 3 degree-days at BP 15."""
        manager, coord = stats
        coord.balance_point = OLD_BP
        coord._daily_history = {
            DAY: {"temp": 16.0, "tdd": 3.0, "wind": 0.0, "kwh": 10.0,
                  "balance_point": OLD_BP},
        }
        d = date.fromisoformat(DAY)

        kwh, _, _, _, total_tdd = manager.calculate_modeled_energy(d, d)

        assert kwh == pytest.approx((OLD_BP + 3.0) * 24)
        assert total_tdd == pytest.approx(3.0)

    def test_daily_average_day_is_reconstructed_at_the_bp_it_recorded(self, stats):
        """Old code rebuilt 18 − 7.5 = 10.5 °C (252 kWh) from a BP-15 tdd."""
        manager, coord = stats
        coord._daily_history = {
            DAY: {"temp": 7.5, "tdd": 7.5, "wind": 0.0, "kwh": 10.0,
                  "balance_point": OLD_BP},
        }
        d = date.fromisoformat(DAY)

        kwh, _, avg_temp, _, total_tdd = manager.calculate_modeled_energy(d, d)

        assert kwh == pytest.approx(7.5 * 24)
        assert avg_temp == pytest.approx(7.5)
        assert total_tdd == pytest.approx(10.5)

    def test_daily_average_day_of_mixed_bp_is_not_reconstructed(self, stats):
        manager, coord = stats
        coord._daily_history = {
            DAY: {"temp": 16.0, "tdd": 3.0, "wind": 0.0, "kwh": 10.0,
                  "balance_point": None},
        }
        d = date.fromisoformat(DAY)

        kwh, _, _, _, total_tdd = manager.calculate_modeled_energy(d, d)

        assert kwh == pytest.approx(16.0 * 24)
        assert total_tdd == pytest.approx(2.0)


    def test_partial_day_is_reconstructed_from_its_logged_hours(self, stats):
        """12 h at 5 °C (tdd 5.0 at BP 15) is a 5 °C day, not a 10 °C one."""
        manager, coord = stats
        coord.balance_point = OLD_BP
        day = _vector_day([5.0] * 12 + [None] * 12, OLD_BP)
        coord._daily_history = {DAY: day}
        d = date.fromisoformat(DAY)

        kwh, _, avg_temp, _, _ = manager.calculate_modeled_energy(d, d)

        assert avg_temp == pytest.approx(5.0)
        assert kwh == pytest.approx(5.0 * 24)


class TestEfficiencyStats:
    def test_period_tdd_is_expressed_at_the_current_bp(self, stats):
        manager, coord = stats
        temps = [5.0] * 24
        coord._daily_history = {
            "2023-01-09": _vector_day(temps, OLD_BP),
            DAY: _vector_day(temps, NEW_BP),
        }

        avg_tdd, efficiency = manager._calculate_efficiency_stats(
            ["2023-01-09", DAY]
        )

        assert avg_tdd == pytest.approx(13.0)
        assert efficiency == pytest.approx(48.0 / 26.0, abs=1e-3)

    def test_tdd_yesterday_is_expressed_at_the_current_bp(self, stats):
        from homeassistant.util import dt as dt_util
        from custom_components.heating_analytics.const import ATTR_TDD_YESTERDAY

        manager, coord = stats
        yesterday = (dt_util.now().date() - __import__("datetime").timedelta(days=1)).isoformat()
        coord._daily_history = {yesterday: _vector_day([5.0] * 24, OLD_BP)}

        try:
            manager.calculate_temp_stats()
        except Exception:
            # Other stats in the same pass need a fuller coordinator; the
            # value under test is written before them.
            pass

        assert coord.data[ATTR_TDD_YESTERDAY] == pytest.approx(13.0)


def _log(hour: int, bp: float | None, temp: float = 10.0, day: str = DAY) -> dict:
    entry = {
        "timestamp": f"{day}T{hour:02d}:00:00",
        "hour": hour,
        "temp": temp,
        "effective_wind": 0.0,
        "solar_factor": 0.0,
        "actual_kwh": 1.0,
        "tdd": abs((bp or OLD_BP) - temp) / 24.0,
        "unit_breakdown": {"rad": 1.0},
        "unit_modes": {},
    }
    if bp is not None:
        entry["bp_at_log_time"] = bp
    return entry


def _processor_coord():
    coord = MagicMock()
    coord.solar_enabled = False
    coord.balance_point = OLD_BP
    coord.hourly_solar_impact_kwh = MagicMock(return_value=0.0)
    return coord


class TestDailyAggregation:
    def test_uniform_day_records_its_bp(self):
        result = DailyProcessor(_processor_coord()).aggregate_logs(
            [_log(h, OLD_BP) for h in range(24)]
        )
        assert result["balance_point"] == OLD_BP

    def test_day_spanning_a_bp_change_records_none(self):
        logs = [_log(h, OLD_BP) for h in range(12)]
        logs += [_log(h, NEW_BP) for h in range(12, 24)]
        result = DailyProcessor(_processor_coord()).aggregate_logs(logs)
        assert result["balance_point"] is None

    def test_hours_predating_the_field_do_not_vote(self):
        logs = [_log(h, None) for h in range(6)] + [_log(h, OLD_BP) for h in range(6, 24)]
        result = DailyProcessor(_processor_coord()).aggregate_logs(logs)
        assert result["balance_point"] == OLD_BP

    def test_backfill_keeps_a_stamp_the_logs_cannot_contradict(self):
        coord = _processor_coord()
        coord._hourly_log = [_log(h, None) for h in range(24)]
        coord._daily_history = {DAY: {"kwh": 24.0, "balance_point": NEW_BP}}

        DailyProcessor(coord).backfill_from_hourly()

        assert coord._daily_history[DAY]["balance_point"] == NEW_BP

    def test_backfill_replaces_a_stamp_the_logs_contradict(self):
        coord = _processor_coord()
        coord._hourly_log = [_log(h, OLD_BP) for h in range(24)]
        coord._daily_history = {DAY: {"kwh": 24.0, "balance_point": NEW_BP}}

        DailyProcessor(coord).backfill_from_hourly()

        assert coord._daily_history[DAY]["balance_point"] == OLD_BP

    @pytest.mark.asyncio
    async def test_downtime_day_records_the_current_bp(self):
        coord = _processor_coord()
        coord._hourly_log = []
        coord._daily_history = {}
        coord.data = {}
        coord._accumulated_energy_today = 0.0
        processor = DailyProcessor(coord)
        try:
            await processor.process(date.fromisoformat(DAY))
        except Exception:
            # Later midnight steps need a fuller coordinator; the entry is
            # written before them.
            pass
        assert coord._daily_history[DAY]["balance_point"] == OLD_BP


class TestMigration:
    def test_stamps_days_from_the_log_then_the_vectors_then_the_oldest_bp(self):
        data = {
            "daily_history": {
                # Older than the log, no vectors: BP at the start of the window.
                "2023-01-01": {"tdd": 10.0, "temp": 5.0},
                # Older than the log, vectors logged at 16.5.
                "2023-01-02": _vector_day([5.0] * 24, 16.5, stamp=False),
                # Older than the log, vectors from two BPs.
                "2023-01-03": {
                    **_vector_day([5.0] * 24, OLD_BP, stamp=False),
                },
                DAY: {"tdd": 5.0},
                "2023-01-11": {"tdd": 5.0},
                "2023-01-12": {"tdd": 5.0, "balance_point": 16.0},
            },
            "hourly_log": (
                [_log(h, 14.0) for h in range(24)]
                + [_log(h, 14.0 if h < 12 else OLD_BP, day="2023-01-11") for h in range(24)]
            ),
        }
        mixed = data["daily_history"]["2023-01-03"]["hourly_vectors"]["tdd"]
        mixed[12:] = [round(13 / 24, 3)] * 12

        out = _migrate_v9_to_v10(data, NEW_BP)
        history = out["daily_history"]

        assert history["2023-01-01"]["balance_point"] == 14.0
        assert history["2023-01-02"]["balance_point"] == 16.5
        assert history["2023-01-03"]["balance_point"] is None
        assert history[DAY]["balance_point"] == 14.0
        assert history["2023-01-11"]["balance_point"] is None
        assert history["2023-01-12"]["balance_point"] == 16.0

    def test_without_a_log_the_current_bp_is_used(self):
        out = _migrate_v9_to_v10({"daily_history": {DAY: {"tdd": 5.0}}}, NEW_BP)
        assert out["daily_history"][DAY]["balance_point"] == NEW_BP

    def test_hours_predating_the_field_fall_back_to_the_vectors(self):
        """The in-log branch must agree with what backfill will keep."""
        data = {
            "daily_history": {DAY: _vector_day([5.0] * 24, 16.5, stamp=False)},
            "hourly_log": [_log(h, None) for h in range(24)],
        }
        out = _migrate_v9_to_v10(data, NEW_BP)
        assert out["daily_history"][DAY]["balance_point"] == 16.5

    def test_unusable_bp_falls_back_to_the_default(self):
        out = _migrate_v9_to_v10({"daily_history": {DAY: {}}}, MagicMock())
        assert out["daily_history"][DAY]["balance_point"] == 17.0

    def test_idempotent(self):
        data = {"daily_history": {DAY: {"tdd": 5.0}}, "hourly_log": []}
        once = _migrate_v9_to_v10(data, OLD_BP)
        twice = _migrate_v9_to_v10(once, NEW_BP)
        assert twice == once
        assert "balance_point" not in data["daily_history"][DAY]

    def test_tolerates_missing_or_malformed_history(self):
        assert _migrate_v9_to_v10({}, OLD_BP) == {}
        assert _migrate_v9_to_v10({"daily_history": []}, OLD_BP) == {"daily_history": []}

    @pytest.mark.asyncio
    async def test_full_chain_through_load(self):
        coord = _make_coord()
        coord.balance_point = OLD_BP
        pre = {
            "solar_coefficients_per_unit": {},
            "daily_history": {DAY: {"tdd": 5.0, "kwh": 10.0}},
        }
        with patch("custom_components.heating_analytics.storage.Store") as mock_store_cls:
            mock_store = mock_store_cls.return_value
            sm = StorageManager(coord)
            migrated = await sm._async_migrate(
                old_major_version=9, old_minor_version=0, old_data=pre
            )
            mock_store.async_load = AsyncMock(return_value=migrated)
            await sm.async_load_data()

        assert coord._daily_history[DAY]["balance_point"] == OLD_BP


    def test_migration_then_backfill_keeps_the_stamp(self):
        data = {
            "daily_history": {DAY: {"kwh": 24.0, "tdd": 5.0}},
            "hourly_log": [_log(h, None) for h in range(24)],
        }
        out = _migrate_v9_to_v10(data, NEW_BP)
        coord = _processor_coord()
        coord._hourly_log = out["hourly_log"]
        coord._daily_history = out["daily_history"]

        DailyProcessor(coord).backfill_from_hourly()

        assert coord._daily_history[DAY]["balance_point"] == NEW_BP

    @pytest.mark.asyncio
    async def test_restore_of_a_pre_v10_backup_stamps_its_days(self):
        coord = _make_restore_coord()
        coord.balance_point = OLD_BP
        sm = StorageManager(coord)
        backup = {
            "correlation_data": {},
            "daily_history": {
                DAY: _vector_day([5.0] * 24, 16.5, stamp=False),
                "2023-01-11": {"kwh": 1.0, "tdd": 1.0, "balance_point": None},
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(backup, f)
            path = f.name
        try:
            async def _run_executor(fn, *args, **kwargs):
                return fn(*args, **kwargs)
            coord.hass.async_add_executor_job = _run_executor
            await sm.async_restore_data(path)
        finally:
            os.unlink(path)

        assert coord._daily_history[DAY]["balance_point"] == 16.5
        # A v10 stamp, including None, survives a second pass of the chain.
        assert coord._daily_history["2023-01-11"]["balance_point"] is None


class TestCsvImport:
    @pytest.mark.asyncio
    async def test_weather_only_merge_stamps_hours_still_in_the_log(
        self, storage_manager, mock_coordinator  # noqa: F811
    ):
        mock_coordinator._hourly_log = [
            {"timestamp": "2023-01-01T00:00:00", "hour": 0, "temp": 0.0,
             "tdd": 0.625, "actual_kwh": 1.5, "bp_at_log_time": 12.0},
        ]
        csv_content = (
            "timestamp,temperature,wind_speed,cloud_coverage\n"
            "2023-01-01T00:00:00,5.0,3.0,80.0\n"
        )
        mapping = {
            "timestamp": "timestamp",
            "temperature": "temperature",
            "wind_speed": "wind_speed",
            "cloud_coverage": "cloud_coverage",
        }
        with patch("builtins.open", mock_open(read_data=csv_content)):
            with patch("os.path.exists", return_value=True):
                await storage_manager.import_csv_data("dummy.csv", mapping, update_model=False)

        entry = mock_coordinator._hourly_log[0]
        assert entry["temp"] == 5.0
        assert entry["tdd"] == pytest.approx(10.0 / 24.0, abs=1e-3)
        assert entry["bp_at_log_time"] == 15.0


class TestRetrainTrackB:
    @pytest.mark.asyncio
    async def test_u_uses_degree_days_at_the_current_bp(self):
        """Hours logged at BP 12 (2 TDD/day at 10 °C) replayed under BP 15."""
        entries = _full_day_entries("2026-04-10", tdd_per_hour=2.0 / 24)
        coord = _daily_coord(entries)

        await RetrainEngine(coord).retrain_from_history(reset_first=True)

        # 12 kWh over 5 TDD at BP 15 — the logged tdd would give 12 / 2 = 6.
        assert coord._learned_u_coefficient == pytest.approx(12.0 / 5.0)
