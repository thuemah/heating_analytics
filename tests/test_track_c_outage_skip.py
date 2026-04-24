"""Tests for #855 Option B: block Track B fallback on Track C installs.

Two protection paths:

1. **Live midnight sync** (`_process_daily_data`): when `track_c_enabled=True`
   and the MPC produces no distribution, the fallback to Track B corrupts the
   correlation_data buckets (mixing MPC-synthetic thermal-per-hour with raw
   electrical). Fix: skip bucket + U updates, increment runtime counter.

2. **Retrain** (`retrain_from_history`): same semantic hazard for historical
   days. A Track C install's retrain must skip days lacking
   `track_c_distribution` in daily_history. The skip covers both pre-Track-C
   history and MPC outages — users who want pre-Track-C days included can
   temporarily disable Track C or narrow `days_back`.

Non-Track-C installs are strictly unaffected.
"""
from datetime import date, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.heating_analytics.const import ATTR_TDD, MODE_HEATING
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.learning import LearningManager
from custom_components.heating_analytics.observation import build_strategies
from custom_components.heating_analytics.solar import SolarCalculator
from custom_components.heating_analytics.retrain import RetrainEngine


# ---------------------------------------------------------------------------
# Live midnight sync
# ---------------------------------------------------------------------------


@pytest.fixture
def base_coord(hass):
    """Coordinator ready for `_process_daily_data` in daily_learning_mode."""
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
    return coord


def _full_day_logs(date_str: str, temp: float = 10.0, kwh_per_hour: float = 0.5):
    """24 hourly entries for a day — enough to pass the 22-hour guard."""
    return [
        {
            "timestamp": f"{date_str}T{h:02d}:00:00",
            "temp": temp,
            "temp_key": str(int(round(temp))),
            "wind_bucket": "normal",
            "effective_wind": 2.0,
            "solar_factor": 0.0,
            "solar_normalization_delta": 0.0,
            "actual_kwh": kwh_per_hour,
            "tdd": abs(15.0 - temp) / 24.0,
            "unit_modes": {},
            "unit_breakdown": {},
        }
        for h in range(24)
    ]


class TestLiveMidnightSyncOutage:
    @pytest.mark.asyncio
    async def test_track_c_disabled_unaffected(self, base_coord):
        """Pure Track B install: Option B code path is a no-op."""
        base_coord.track_c_enabled = False
        base_coord._hourly_log = _full_day_logs("2026-04-20")
        base_coord._accumulated_energy_today = 12.0
        base_coord.data[ATTR_TDD] = 5.0

        await base_coord._process_daily_data(date(2026, 4, 20))

        # Counter untouched on non-Track-C installs.
        assert base_coord._track_c_outage_count_session == 0

    @pytest.mark.asyncio
    async def test_track_c_outage_skips_learning_and_bumps_counter(self, base_coord):
        """Track C enabled + MPC fails → no bucket, no U, counter += 1."""
        base_coord.track_c_enabled = True
        base_coord.mpc_managed_sensor = "sensor.vp_stue"
        base_coord._run_track_c_midnight_sync = AsyncMock(return_value=None)  # outage
        base_coord._hourly_log = _full_day_logs("2026-04-20")
        base_coord._accumulated_energy_today = 12.0
        base_coord.data[ATTR_TDD] = 5.0

        u_before = base_coord._learned_u_coefficient
        buckets_before = dict(base_coord._correlation_data)

        await base_coord._process_daily_data(date(2026, 4, 20))

        assert base_coord._track_c_outage_count_session == 1
        assert base_coord._learned_u_coefficient == u_before
        assert base_coord._correlation_data == buckets_before

    @pytest.mark.asyncio
    async def test_multiple_outages_accumulate(self, base_coord):
        """Counter accumulates across multiple sync calls."""
        base_coord.track_c_enabled = True
        base_coord.mpc_managed_sensor = "sensor.vp_stue"
        base_coord._run_track_c_midnight_sync = AsyncMock(return_value=None)

        base_coord._hourly_log = _full_day_logs("2026-04-20")
        base_coord._accumulated_energy_today = 12.0
        base_coord.data[ATTR_TDD] = 5.0
        await base_coord._process_daily_data(date(2026, 4, 20))

        base_coord._hourly_log = _full_day_logs("2026-04-21")
        base_coord._accumulated_energy_today = 12.0
        base_coord.data[ATTR_TDD] = 5.0
        await base_coord._process_daily_data(date(2026, 4, 21))

        assert base_coord._track_c_outage_count_session == 2

    @pytest.mark.asyncio
    async def test_track_c_success_unaffected(self, base_coord):
        """Track C succeeds → normal path, no outage counted."""
        base_coord.track_c_enabled = True
        base_coord.mpc_managed_sensor = "sensor.vp_stue"
        # Pretend Track C returns (kwh, distribution, source).
        dist = [{"synthetic_kwh_el": 0.5} for _ in range(24)]
        base_coord._run_track_c_midnight_sync = AsyncMock(return_value=(12.0, dist, "live"))
        base_coord._apply_strategies_to_global_model = MagicMock(return_value=3)
        base_coord._hourly_log = _full_day_logs("2026-04-20")
        base_coord._accumulated_energy_today = 12.0
        base_coord.data[ATTR_TDD] = 5.0
        base_coord._unit_strategies = build_strategies(
            energy_sensors=["sensor.vp_stue"],
            track_c_enabled=True,
            mpc_managed_sensor="sensor.vp_stue",
        )

        await base_coord._process_daily_data(date(2026, 4, 20))

        assert base_coord._track_c_outage_count_session == 0
        # U-coefficient should have moved because learning was NOT skipped.
        assert base_coord._learned_u_coefficient is not None


# ---------------------------------------------------------------------------
# Retrain (strategy dispatch path)
# ---------------------------------------------------------------------------


def _retrain_coord(hourly_log, *, track_c_enabled: bool, daily_history: dict | None = None):
    """Coordinator for daily-pass retrain (strategy_dispatch mode)."""
    coord = MagicMock()
    coord._hourly_log = hourly_log
    coord.daily_learning_mode = True  # triggers strategy_dispatch branch
    coord.track_c_enabled = track_c_enabled
    coord.mpc_managed_sensor = "sensor.vp_stue" if track_c_enabled else None
    coord.thermal_mass_kwh_per_degree = 0.0
    coord.learning_enabled = True
    coord.learning_rate = 1.0
    coord.balance_point = 15.0
    coord.energy_sensors = ["sensor.vp_stue", "sensor.panel"]
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
    coord._track_c_outage_count_session = 0
    coord._daily_history = daily_history or {}
    coord.storage.async_save_data = AsyncMock()
    coord.learning = LearningManager()
    coord.solar = SolarCalculator(coord)
    coord._replay_per_unit_models = MagicMock()
    coord._apply_strategies_to_global_model = MagicMock(return_value=1)
    coord._compute_excluded_mode_energy = MagicMock(return_value=0.0)
    coord._get_wind_bucket = MagicMock(return_value="normal")
    coord._try_track_b_cop_smearing = AsyncMock(return_value=0)
    coord._unit_strategies = build_strategies(
        energy_sensors=["sensor.vp_stue", "sensor.panel"],
        track_c_enabled=track_c_enabled,
        mpc_managed_sensor="sensor.vp_stue" if track_c_enabled else None,
    )
    return coord


def _day_entries(date_str: str, temp: float = 10.0):
    return [
        {
            "timestamp": f"{date_str}T{h:02d}:00:00",
            "hour": h,
            "temp": temp,
            "temp_key": str(int(round(temp))),
            "wind_bucket": "normal",
            "effective_wind": 2.0,
            "solar_factor": 0.0,
            "solar_normalization_delta": 0.0,
            "solar_vector_s": 0.0,
            "solar_vector_e": 0.0,
            "solar_vector_w": 0.0,
            "correction_percent": 100.0,
            "actual_kwh": 0.5,
            "tdd": abs(15.0 - temp) / 24.0,
            "auxiliary_active": False,
            "unit_modes": {"sensor.vp_stue": MODE_HEATING, "sensor.panel": MODE_HEATING},
            "unit_breakdown": {"sensor.vp_stue": 0.3, "sensor.panel": 0.2},
            "solar_dominant_entities": [],
        }
        for h in range(24)
    ]


class TestRetrainTrackCOutageSkip:
    @pytest.mark.asyncio
    async def test_non_track_c_install_all_days_processed(self):
        """track_c_enabled=False: behaviour unchanged, skip counter stays 0."""
        entries = _day_entries("2026-04-10") + _day_entries("2026-04-11")
        coord = _retrain_coord(entries, track_c_enabled=False)
        result = await RetrainEngine(coord).retrain_from_history(days_back=60)
        assert result["days_skipped_mpc_unavailable"] == 0
        assert result["days_processed"] >= 1

    @pytest.mark.asyncio
    async def test_track_c_install_skips_day_without_distribution(self):
        """Track C install + day lacks distribution → skipped, counter += 1."""
        entries = _day_entries("2026-04-10")
        daily_history = {"2026-04-10": {"kwh": 12.0}}  # no track_c_distribution
        coord = _retrain_coord(entries, track_c_enabled=True, daily_history=daily_history)
        result = await RetrainEngine(coord).retrain_from_history(days_back=60)
        assert result["days_skipped_mpc_unavailable"] == 1
        assert result["days_processed"] == 0

    @pytest.mark.asyncio
    async def test_track_c_install_processes_day_with_distribution(self):
        """Track C install + day has distribution → normal processing."""
        entries = _day_entries("2026-04-10")
        daily_history = {
            "2026-04-10": {
                "kwh": 12.0,
                "track_c_kwh": 11.5,
                "track_c_distribution": [{"synthetic_kwh_el": 0.5} for _ in range(24)],
            }
        }
        coord = _retrain_coord(entries, track_c_enabled=True, daily_history=daily_history)
        result = await RetrainEngine(coord).retrain_from_history(days_back=60)
        assert result["days_skipped_mpc_unavailable"] == 0
        assert result["days_processed"] == 1

    @pytest.mark.asyncio
    async def test_mixed_window_only_skips_missing_days(self):
        """Track C install with mix of distribution + outage days."""
        entries = _day_entries("2026-04-10") + _day_entries("2026-04-11") + _day_entries("2026-04-12")
        daily_history = {
            "2026-04-10": {
                "kwh": 12.0,
                "track_c_kwh": 11.5,
                "track_c_distribution": [{"synthetic_kwh_el": 0.5} for _ in range(24)],
            },
            # 2026-04-11 missing — outage
            "2026-04-12": {
                "kwh": 12.0,
                "track_c_kwh": 11.5,
                "track_c_distribution": [{"synthetic_kwh_el": 0.5} for _ in range(24)],
            },
        }
        coord = _retrain_coord(entries, track_c_enabled=True, daily_history=daily_history)
        result = await RetrainEngine(coord).retrain_from_history(days_back=60)
        assert result["days_skipped_mpc_unavailable"] == 1
        assert result["days_processed"] == 2

    @pytest.mark.asyncio
    async def test_skipped_day_does_not_update_u_coefficient(self):
        """U-coefficient must remain None if every Track C day is outage."""
        entries = _day_entries("2026-04-10")
        coord = _retrain_coord(entries, track_c_enabled=True, daily_history={})
        await RetrainEngine(coord).retrain_from_history(days_back=60)
        assert coord._learned_u_coefficient is None

    @pytest.mark.asyncio
    async def test_skipped_day_does_not_update_buckets(self):
        """correlation_data must remain empty when every day is outage."""
        entries = _day_entries("2026-04-10")
        coord = _retrain_coord(entries, track_c_enabled=True, daily_history={})
        await RetrainEngine(coord).retrain_from_history(days_back=60)
        assert coord._correlation_data == {}
        coord._apply_strategies_to_global_model.assert_not_called()

    @pytest.mark.asyncio
    async def test_response_shape_contains_skip_counter(self):
        """Retrain response always exposes the new field (even if 0)."""
        entries = _day_entries("2026-04-10")
        coord = _retrain_coord(entries, track_c_enabled=False)
        result = await RetrainEngine(coord).retrain_from_history(days_back=60)
        assert "days_skipped_mpc_unavailable" in result


# ---------------------------------------------------------------------------
# diagnose_model exposes the runtime counter
# ---------------------------------------------------------------------------


class TestDiagnoseModelCounter:
    def test_fresh_coord_reports_zero(self, hass):
        entry = MagicMock()
        entry.data = {"balance_point": 15.0}
        with patch("custom_components.heating_analytics.storage.Store"):
            coord = HeatingDataCoordinator(hass, entry)
        coord._hourly_log = []
        coord._daily_history = {}
        coord._correlation_data = {}
        result = coord.diagnose_model(days_back=30)
        assert result["track_c_outage_session_count"] == 0

    def test_counter_surfaces_after_outage(self, hass):
        entry = MagicMock()
        entry.data = {"balance_point": 15.0}
        with patch("custom_components.heating_analytics.storage.Store"):
            coord = HeatingDataCoordinator(hass, entry)
        coord._hourly_log = []
        coord._daily_history = {}
        coord._correlation_data = {}
        coord._track_c_outage_count_session = 3
        result = coord.diagnose_model(days_back=30)
        assert result["track_c_outage_session_count"] == 3
