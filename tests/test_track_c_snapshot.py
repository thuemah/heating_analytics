"""Tests for Track C pre-midnight snapshot + S1 track_used tagging (#855 follow-up).

The snapshot mechanism takes three attempts (22:00, 23:00, 23:55) to fetch
the MPC buffer before midnight, so the midnight sync has a fallback when
the live call fails.  Most outages are covered by the 23:55 snapshot; the
Option B skip is reserved for the pathological case (HA restart between
23:55 and 00:01).

S1 tagging writes an explicit ``track_used`` field to every daily_history
entry so diagnose_model, retrain, and future BP-aware consumers can see
the attribution composition directly.
"""
from datetime import date, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.heating_analytics.const import ATTR_TDD, MODE_HEATING
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator


@pytest.fixture
def track_c_coord(hass):
    entry = MagicMock()
    entry.data = {"balance_point": 15.0}
    with patch("custom_components.heating_analytics.storage.Store"):
        coord = HeatingDataCoordinator(hass, entry)
    coord._async_save_data = AsyncMock()
    coord.storage.append_daily_log_csv = AsyncMock()
    coord.forecast = MagicMock()
    coord.forecast.log_accuracy = MagicMock()
    coord.daily_learning_mode = True
    coord.learning_enabled = True
    coord.track_c_enabled = True
    coord.mpc_managed_sensor = "sensor.vp_stue"
    coord.mpc_entry_id = "abc123"
    return coord


# ---------------------------------------------------------------------------
# Snapshot polling (_maybe_snapshot_track_c)
# ---------------------------------------------------------------------------


class TestSnapshotPolling:
    @pytest.mark.asyncio
    async def test_snapshots_at_22_captured(self, track_c_coord):
        """Entering the 22:00 hour triggers a snapshot fetch."""
        records = [{"datetime": "2026-04-20T12:00:00", "q_th": 1.0}]
        cop = {"eta_carnot": 0.45}
        track_c_coord._fetch_mpc_buffer_and_cop = AsyncMock(return_value=(records, cop))

        await track_c_coord._maybe_snapshot_track_c(datetime(2026, 4, 20, 22, 0))

        assert track_c_coord._track_c_snapshot is not None
        assert track_c_coord._track_c_snapshot["slot"] == "2200"
        assert track_c_coord._track_c_snapshot["mpc_records"] == records
        assert track_c_coord._track_c_snapshot["cop_params"] == cop

    @pytest.mark.asyncio
    async def test_snapshot_at_2355_overwrites_22(self, track_c_coord):
        """Later snapshots overwrite earlier ones — freshest data wins."""
        first = ([{"datetime": "2026-04-20T12:00:00"}], {"eta_carnot": 0.40})
        second = ([{"datetime": "2026-04-20T23:00:00"}], {"eta_carnot": 0.45})
        fetch = AsyncMock(side_effect=[first, second])
        track_c_coord._fetch_mpc_buffer_and_cop = fetch

        await track_c_coord._maybe_snapshot_track_c(datetime(2026, 4, 20, 22, 0))
        await track_c_coord._maybe_snapshot_track_c(datetime(2026, 4, 20, 23, 55))

        assert track_c_coord._track_c_snapshot["slot"] == "2355"
        assert track_c_coord._track_c_snapshot["mpc_records"] == second[0]

    @pytest.mark.asyncio
    async def test_snapshot_slot_deduped_within_same_slot(self, track_c_coord):
        """Multiple ticks within the same slot only fetch once."""
        fetch = AsyncMock(return_value=([{"datetime": "x"}], None))
        track_c_coord._fetch_mpc_buffer_and_cop = fetch

        await track_c_coord._maybe_snapshot_track_c(datetime(2026, 4, 20, 22, 0))
        await track_c_coord._maybe_snapshot_track_c(datetime(2026, 4, 20, 22, 30))
        await track_c_coord._maybe_snapshot_track_c(datetime(2026, 4, 20, 22, 55))

        assert fetch.call_count == 1

    @pytest.mark.asyncio
    async def test_snapshot_failure_preserves_previous(self, track_c_coord):
        """If a later snapshot fails, the earlier one is still available."""
        good = ([{"datetime": "2026-04-20T12:00:00"}], None)
        track_c_coord._fetch_mpc_buffer_and_cop = AsyncMock(
            side_effect=[good, None]
        )

        await track_c_coord._maybe_snapshot_track_c(datetime(2026, 4, 20, 22, 0))
        assert track_c_coord._track_c_snapshot is not None
        await track_c_coord._maybe_snapshot_track_c(datetime(2026, 4, 20, 23, 0))

        # Previous snapshot preserved; failed fetch did not clear it.
        assert track_c_coord._track_c_snapshot is not None
        assert track_c_coord._track_c_snapshot["slot"] == "2200"

    @pytest.mark.asyncio
    async def test_no_snapshot_outside_trigger_hours(self, track_c_coord):
        """Ticks outside 22:00/23:00/23:55 slots do nothing."""
        fetch = AsyncMock(return_value=([{"x": 1}], None))
        track_c_coord._fetch_mpc_buffer_and_cop = fetch

        for h in (0, 6, 12, 18, 21):
            await track_c_coord._maybe_snapshot_track_c(datetime(2026, 4, 20, h, 0))

        fetch.assert_not_called()
        assert track_c_coord._track_c_snapshot is None


# ---------------------------------------------------------------------------
# Midnight sync fallback via snapshot
# ---------------------------------------------------------------------------


def _day_log(hour: int, date_str: str = "2026-04-20", temp: float = 10.0) -> dict:
    return {
        "timestamp": f"{date_str}T{hour:02d}:00:00",
        "hour": hour,
        "temp": temp,
        "temp_key": str(int(round(temp))),
        "wind_bucket": "normal",
        "effective_wind": 2.0,
        "solar_factor": 0.0,
        "solar_normalization_delta": 0.0,
        "inertia_temp": temp,
        "actual_kwh": 0.5,  # non-zero so daily_stats["kwh"] > 0 and learning runs
        "tdd": 0.25,  # per-hour tdd; sums to 6.0 > 0.5 guard
        "unit_modes": {},
        "unit_breakdown": {},
    }


class TestMidnightSyncFallback:
    @pytest.mark.asyncio
    async def test_live_success_reports_source_live(self, track_c_coord, monkeypatch):
        """Happy path: live fetch works → source='live'."""
        # Simulate a full day of MPC records.
        records = [
            {"datetime": f"2026-04-20T{h:02d}:00:00+00:00", "q_th": 0.5}
            for h in range(24)
        ]
        track_c_coord._fetch_mpc_buffer_and_cop = AsyncMock(
            return_value=(records, None)
        )

        # Mock the ThermodynamicEngine so we don't depend on physics.
        fake_dist = [{"synthetic_kwh_el": 0.5} for _ in range(24)]
        mock_engine_cls = MagicMock()
        mock_engine = MagicMock()
        mock_engine.calculate_synthetic_baseline = MagicMock(return_value=fake_dist)
        mock_engine_cls.return_value = mock_engine
        # Patch the coordinator's imported symbol — it imports at module
        # load time, so patching the source module has no effect.
        monkeypatch.setattr(
            "custom_components.heating_analytics.coordinator.ThermodynamicEngine",
            mock_engine_cls,
        )

        day_logs = [_day_log(h) for h in range(24)]
        result = await track_c_coord._run_track_c_midnight_sync(day_logs, "2026-04-20")

        assert result is not None
        total, dist, source = result
        assert source == "live"
        assert dist == fake_dist

    @pytest.mark.asyncio
    async def test_live_failure_uses_snapshot(self, track_c_coord, monkeypatch):
        """Live fetch fails but 23:55 snapshot exists → source='snapshot_2355'."""
        # Pre-seed snapshot (as if 23:55 polling succeeded).
        records = [
            {"datetime": f"2026-04-20T{h:02d}:00:00+00:00", "q_th": 0.5}
            for h in range(24)
        ]
        track_c_coord._track_c_snapshot = {
            "date": "2026-04-20",
            "captured_at": "2026-04-20T23:55:00",
            "slot": "2355",
            "mpc_records": records,
            "cop_params": None,
        }
        # Live call fails.
        track_c_coord._fetch_mpc_buffer_and_cop = AsyncMock(return_value=None)

        fake_dist = [{"synthetic_kwh_el": 0.5} for _ in range(24)]
        mock_engine_cls = MagicMock()
        mock_engine = MagicMock()
        mock_engine.calculate_synthetic_baseline = MagicMock(return_value=fake_dist)
        mock_engine_cls.return_value = mock_engine
        # Patch the coordinator's imported symbol — it imports at module
        # load time, so patching the source module has no effect.
        monkeypatch.setattr(
            "custom_components.heating_analytics.coordinator.ThermodynamicEngine",
            mock_engine_cls,
        )

        day_logs = [_day_log(h) for h in range(24)]
        result = await track_c_coord._run_track_c_midnight_sync(day_logs, "2026-04-20")

        assert result is not None
        _, _, source = result
        assert source == "snapshot_2355"
        # Snapshot cleared after successful consumption.
        assert track_c_coord._track_c_snapshot is None

    @pytest.mark.asyncio
    async def test_snapshot_from_wrong_date_ignored(self, track_c_coord):
        """A snapshot from a different date must not be used."""
        track_c_coord._track_c_snapshot = {
            "date": "2026-04-19",  # day before
            "captured_at": "2026-04-19T23:55:00",
            "slot": "2355",
            "mpc_records": [{"datetime": "2026-04-19T12:00:00+00:00"}],
            "cop_params": None,
        }
        track_c_coord._fetch_mpc_buffer_and_cop = AsyncMock(return_value=None)

        day_logs = [_day_log(h) for h in range(24)]
        result = await track_c_coord._run_track_c_midnight_sync(day_logs, "2026-04-20")

        # Snapshot date mismatch → no fallback → Option B skip.
        assert result is None
        # Snapshot kept for potential use if we somehow recover — not consumed.
        assert track_c_coord._track_c_snapshot is not None

    @pytest.mark.asyncio
    async def test_no_live_no_snapshot_triggers_option_b(self, track_c_coord):
        """No live and no snapshot → None → caller applies Option B skip."""
        track_c_coord._fetch_mpc_buffer_and_cop = AsyncMock(return_value=None)
        track_c_coord._track_c_snapshot = None

        day_logs = [_day_log(h) for h in range(24)]
        result = await track_c_coord._run_track_c_midnight_sync(day_logs, "2026-04-20")

        assert result is None


# ---------------------------------------------------------------------------
# S1: daily_history[date]["track_used"] tagging
# ---------------------------------------------------------------------------


def _full_day_logs(date_str: str = "2026-04-20", temp: float = 10.0):
    return [_day_log(h, date_str, temp) for h in range(24)]


class TestTrackUsedTagging:
    @pytest.mark.asyncio
    async def test_track_c_live_tagged_c_live(self, track_c_coord):
        """Successful live Track C → track_used='C_live'."""
        dist = [{"synthetic_kwh_el": 0.5} for _ in range(24)]
        track_c_coord._run_track_c_midnight_sync = AsyncMock(
            return_value=(12.0, dist, "live")
        )
        track_c_coord._apply_strategies_to_global_model = MagicMock(return_value=3)
        from custom_components.heating_analytics.observation import build_strategies
        track_c_coord._unit_strategies = build_strategies(
            energy_sensors=["sensor.vp_stue"],
            track_c_enabled=True,
            mpc_managed_sensor="sensor.vp_stue",
        )
        track_c_coord._hourly_log = _full_day_logs()
        track_c_coord._accumulated_energy_today = 12.0
        track_c_coord.data[ATTR_TDD] = 5.0

        await track_c_coord._process_daily_data(date(2026, 4, 20))

        assert track_c_coord._daily_history["2026-04-20"]["track_used"] == "C_live"

    @pytest.mark.asyncio
    async def test_track_c_snapshot_tagged_c_snapshot(self, track_c_coord):
        """Snapshot-sourced Track C → track_used='C_snapshot_2355'."""
        dist = [{"synthetic_kwh_el": 0.5} for _ in range(24)]
        track_c_coord._run_track_c_midnight_sync = AsyncMock(
            return_value=(12.0, dist, "snapshot_2355")
        )
        track_c_coord._apply_strategies_to_global_model = MagicMock(return_value=3)
        from custom_components.heating_analytics.observation import build_strategies
        track_c_coord._unit_strategies = build_strategies(
            energy_sensors=["sensor.vp_stue"],
            track_c_enabled=True,
            mpc_managed_sensor="sensor.vp_stue",
        )
        track_c_coord._hourly_log = _full_day_logs()
        track_c_coord._accumulated_energy_today = 12.0
        track_c_coord.data[ATTR_TDD] = 5.0

        await track_c_coord._process_daily_data(date(2026, 4, 20))

        assert track_c_coord._daily_history["2026-04-20"]["track_used"] == "C_snapshot_2355"

    @pytest.mark.asyncio
    async def test_track_c_outage_tagged_skipped(self, track_c_coord):
        """MPC outage with no snapshot → track_used='skipped_mpc_outage'."""
        track_c_coord._run_track_c_midnight_sync = AsyncMock(return_value=None)
        track_c_coord._hourly_log = _full_day_logs()
        track_c_coord._accumulated_energy_today = 12.0
        track_c_coord.data[ATTR_TDD] = 5.0

        await track_c_coord._process_daily_data(date(2026, 4, 20))

        assert (
            track_c_coord._daily_history["2026-04-20"]["track_used"]
            == "skipped_mpc_outage"
        )

    @pytest.mark.asyncio
    async def test_track_b_flat_tagged_b_flat(self, hass):
        """Pure Track B install (daily mode, no track_c_enabled) → 'B_flat'."""
        entry = MagicMock()
        entry.data = {"balance_point": 15.0}
        with patch("custom_components.heating_analytics.storage.Store"):
            coord = HeatingDataCoordinator(hass, entry)
        coord._async_save_data = AsyncMock()
        coord.storage.append_daily_log_csv = AsyncMock()
        coord.forecast = MagicMock()
        coord.forecast.log_accuracy = MagicMock()
        coord.daily_learning_mode = True
        coord.learning_enabled = True
        coord.track_c_enabled = False
        coord._try_track_b_cop_smearing = AsyncMock(return_value=0)  # no cop-smear
        coord._hourly_log = _full_day_logs()
        coord._accumulated_energy_today = 12.0
        coord.data[ATTR_TDD] = 5.0

        await coord._process_daily_data(date(2026, 4, 20))

        assert coord._daily_history["2026-04-20"]["track_used"] == "B_flat"

    @pytest.mark.asyncio
    async def test_track_b_cop_smeared_tagged_b_cop(self, hass):
        """Track B with successful cop-smear → 'B_cop'."""
        entry = MagicMock()
        entry.data = {"balance_point": 15.0}
        with patch("custom_components.heating_analytics.storage.Store"):
            coord = HeatingDataCoordinator(hass, entry)
        coord._async_save_data = AsyncMock()
        coord.storage.append_daily_log_csv = AsyncMock()
        coord.forecast = MagicMock()
        coord.forecast.log_accuracy = MagicMock()
        coord.daily_learning_mode = True
        coord.learning_enabled = True
        coord.track_c_enabled = False
        coord._try_track_b_cop_smearing = AsyncMock(return_value=5)  # 5 updates
        coord._hourly_log = _full_day_logs()
        coord._accumulated_energy_today = 12.0
        coord.data[ATTR_TDD] = 5.0

        await coord._process_daily_data(date(2026, 4, 20))

        assert coord._daily_history["2026-04-20"]["track_used"] == "B_cop"

    @pytest.mark.asyncio
    async def test_track_a_install_tagged_a(self, hass):
        """Non-daily-learning install → 'A'."""
        entry = MagicMock()
        entry.data = {"balance_point": 15.0}
        with patch("custom_components.heating_analytics.storage.Store"):
            coord = HeatingDataCoordinator(hass, entry)
        coord._async_save_data = AsyncMock()
        coord.storage.append_daily_log_csv = AsyncMock()
        coord.forecast = MagicMock()
        coord.forecast.log_accuracy = MagicMock()
        coord.daily_learning_mode = False  # Track A
        coord.learning_enabled = True
        coord._hourly_log = _full_day_logs()
        coord._accumulated_energy_today = 12.0
        coord.data[ATTR_TDD] = 5.0

        await coord._process_daily_data(date(2026, 4, 20))

        assert coord._daily_history["2026-04-20"]["track_used"] == "A"
