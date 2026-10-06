"""Track C days are re-spread at the current settings on retrain (#1111).

The midnight sync spreads a day's MPC thermal energy over its hours with
weights ``|BP − T_inertia| × wind × solar`` and converts it to electrical per
hour.  The stored ``track_c_distribution`` therefore carries the balance
point, inertia axis, wind thresholds and battery decay of that midnight.
Retrain replays the log under the current settings and now re-spreads each
day with them (``resmear_track_c_day``), through the same smearing code, with
the day's totals kept.  The stored distribution stays the record of what that
midnight learned.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.heating_analytics.daily_processor import (
    DailyProcessor,
    resmear_track_c_day,
    track_c_smear_record,
)
from custom_components.heating_analytics.diagnostics import DiagnosticsEngine
from custom_components.heating_analytics.retrain import RetrainEngine
from custom_components.heating_analytics.thermodynamics import ThermodynamicEngine

from tests.test_diagnose_base_model_health import _make_coord as _make_diag_coord
from tests.test_track_c_outage_skip import _retrain_coord

COP = {"eta_carnot": 0.42, "lwt": 35.0, "f_defrost": 0.85}
DAY = "2026-04-15"


class _Settings:
    """The coordinator settings the Track C weights read."""

    def __init__(self, balance_point, *, hourly_log=None):
        self.balance_point = balance_point
        self.wind_threshold = 8.0
        self.extreme_wind_threshold = 10.8
        self.solar_battery_decay = 0.5
        self.inertia_tau = 4.0
        self._hourly_log = hourly_log or []


def _shoulder_day(start=DAY):
    """A shoulder day: inertia temperatures 10–16 °C, a sunny afternoon."""
    base = datetime.fromisoformat(start + "T00:00:00")
    logs = []
    for h in range(24):
        t = 13.0 + 3.0 * (1 if 10 <= h <= 17 else -1) * (1 - abs(h - 14) / 14)
        logs.append({
            "timestamp": (base + timedelta(hours=h)).isoformat(),
            "hour": h,
            "temp": t,
            "inertia_temp": t,
            "effective_wind": 3.0 if h % 5 else 9.0,
            "solar_factor": 0.4 if 10 <= h <= 15 else 0.0,
            "humidity": 70.0,
        })
    return logs


def _mpc_records(logs):
    return [
        {"datetime": e["timestamp"], "kwh_th_sh": 1.0, "kwh_el_sh": 0.3, "mode": "sh"}
        for e in logs
    ]


def _midnight(logs, balance_point, cop_params):
    """The distribution the midnight sync stores, at ``balance_point``."""
    coord = _Settings(balance_point)
    weather = DailyProcessor(coord).track_c_weather([e["timestamp"] for e in logs], logs)
    return ThermodynamicEngine(balance_point).calculate_synthetic_baseline(
        _mpc_records(logs), weather, cop_params=cop_params,
    )


def _resmear(logs, stored, balance_point, cop_params):
    record = {
        "track_c_distribution": stored,
        "track_c_smear": {"cop_params": cop_params},
    }
    return resmear_track_c_day(_Settings(balance_point), logs, record)


def _synth(dist):
    return [d["synthetic_kwh_el"] for d in dist]


# ---------------------------------------------------------------------------
# The re-smear itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cop_params", [COP, None], ids=["per_hour_cop", "daily_avg_cop"])
def test_resmear_at_unchanged_settings_reproduces_the_midnight(cop_params):
    logs = _shoulder_day()
    stored = _midnight(logs, 15.0, cop_params)
    resmeared, info = _resmear(logs, stored, 15.0, cop_params)
    # Up to rounding: the stored hours carry 3 decimals.
    assert _synth(resmeared) == pytest.approx(_synth(stored), abs=3e-3)
    assert info["moved_share"] < 0.005


def test_recovered_cop_reproduces_the_midnight_without_cop_params():
    """A day stored before ``cop_params`` were recorded: the COP recovered
    from its own hours gives back the same shape."""
    logs = _shoulder_day()
    stored = _midnight(logs, 15.0, COP)
    resmeared, info = _resmear(logs, stored, 15.0, None)
    assert info["status"] == "recovered_cop"
    assert _synth(resmeared) == pytest.approx(_synth(stored), abs=3e-3)


def test_balance_point_change_moves_shoulder_energy_and_keeps_totals():
    """At BP 15 the ~14–15 °C hours carry almost no loss weight; at BP 18
    they carry a real share.  The day's totals do not change."""
    logs = _shoulder_day()
    stored = _midnight(logs, 15.0, COP)
    resmeared, info = _resmear(logs, stored, 18.0, COP)
    assert info["status"] == "exact_cop"
    assert sum(_synth(resmeared)) == pytest.approx(sum(_synth(stored)), abs=0.01)
    assert sum(d["smeared_kwh_th"] for d in resmeared) == pytest.approx(
        sum(d["smeared_kwh_th"] for d in stored), abs=0.01
    )
    # The same result the midnight sync would have produced at BP 18.
    assert _synth(resmeared) == pytest.approx(_synth(_midnight(logs, 18.0, COP)), abs=2e-3)
    assert info["moved_share"] > 0.05


def test_recovered_cop_follows_the_exact_one_after_a_bp_change():
    logs = _shoulder_day()
    stored = _midnight(logs, 15.0, COP)
    exact, _ = _resmear(logs, stored, 18.0, COP)
    recovered, _ = _resmear(logs, stored, 18.0, None)
    for e, r in zip(_synth(exact), _synth(recovered)):
        assert r == pytest.approx(e, rel=0.05, abs=0.005)


def test_inertia_axis_change_is_re_spread_too():
    """The weights read the entries' ``inertia_temp``: entries re-keyed on the
    current axis give the midnight shape of the current axis."""
    logs = _shoulder_day()
    stored = _midnight(logs, 15.0, COP)
    shifted = [{**e, "inertia_temp": e["inertia_temp"] - 1.5} for e in logs]
    resmeared, info = _resmear(shifted, stored, 15.0, COP)
    assert _synth(resmeared) == pytest.approx(_synth(_midnight(shifted, 15.0, COP)), abs=2e-3)
    assert info["moved_share"] > 0.02


def test_unreadable_settings_keep_the_stored_distribution():
    logs = _shoulder_day()
    stored = _midnight(logs, 15.0, COP)
    dist, info = resmear_track_c_day(MagicMock(), logs, {"track_c_distribution": stored})
    assert dist is stored
    assert info["status"] == "stored"


def test_no_distribution():
    assert resmear_track_c_day(_Settings(15.0), [], {}) == (None, {"status": "no_distribution"})


def test_smear_record_carries_settings_and_cop_params():
    record = track_c_smear_record(_Settings(16.0), {**COP, "junk": "x"})
    assert record["balance_point"] == 16.0
    assert record["inertia_tau"] == 4.0
    assert record["solar_battery_decay"] == 0.5
    assert record["cop_params"] == COP
    assert track_c_smear_record(_Settings(16.0), None)["cop_params"] is None


# ---------------------------------------------------------------------------
# Wiring: midnight stores the record, retrain and diagnose re-spread
# ---------------------------------------------------------------------------


def _track_c_day_entries(date_str, temp=10.0):
    return [
        {
            "timestamp": f"{date_str}T{h:02d}:00:00",
            "hour": h,
            "temp": temp + (h % 6),
            "inertia_temp": temp + (h % 6),
            "temp_key": str(int(round(temp + (h % 6)))),
            "wind_bucket": "normal",
            "effective_wind": 2.0,
            "solar_factor": 0.0,
            "solar_normalization_delta": 0.0,
            "solar_vector_s": 0.0,
            "solar_vector_e": 0.0,
            "solar_vector_w": 0.0,
            "correction_percent": 100.0,
            "actual_kwh": 0.5,
            "tdd": 0.2,
            "auxiliary_active": False,
            "unit_modes": {},
            "unit_breakdown": {"sensor.vp_stue": 0.3, "sensor.panel": 0.2},
            "solar_dominant_entities": [],
        }
        for h in range(24)
    ]


@pytest.mark.asyncio
async def test_retrain_applies_the_re_spread_distribution():
    entries = _track_c_day_entries(DAY)
    stored = _midnight(entries, 12.0, COP)
    daily_history = {
        DAY: {
            "kwh": 12.0,
            "track_c_kwh": sum(_synth(stored)),
            "track_c_distribution": stored,
            "track_c_smear": {"balance_point": 12.0, "cop_params": COP},
        }
    }
    coord = _retrain_coord(entries, track_c_enabled=True, daily_history=daily_history)
    coord.wind_threshold = 8.0
    coord.extreme_wind_threshold = 10.8
    coord.solar_battery_decay = 0.5  # balance_point is 15.0 in the harness
    coord._apply_strategies_to_global_model = MagicMock(return_value=1)

    result = await RetrainEngine(coord).retrain_from_history(days_back=None)

    (_, applied), _ = coord._apply_strategies_to_global_model.call_args
    # The midnight shape at the current BP and on the current inertia axis.
    expected = _midnight(coord._entries_on_current_axis(entries), 15.0, COP)
    assert _synth(applied) == pytest.approx(_synth(expected), abs=2e-3)
    assert daily_history[DAY]["track_c_distribution"] is stored  # the record is kept
    report = result["track_c_resmear"]
    assert report["days_exact_cop"] == 1
    assert report["moved_share_max"] > 0.0


@pytest.mark.asyncio
async def test_midnight_stores_the_smear_record(hass):
    from datetime import date
    from custom_components.heating_analytics.const import ATTR_TDD
    from tests.test_track_c_snapshot import _full_day_logs
    from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
    from unittest.mock import patch

    entry = MagicMock()
    entry.data = {
        "balance_point": 17.0,
        "energy_sensors": ["sensor.vp_stue"],
        "daily_learning_mode": True,
        "track_c_enabled": True,
        "mpc_managed_sensor": "sensor.vp_stue",
    }
    with patch("custom_components.heating_analytics.storage.Store") as store_cls:
        store_cls.return_value.async_load = AsyncMock(return_value={})
        store_cls.return_value.async_save = AsyncMock()
        coord = HeatingDataCoordinator(hass, entry)
    coord._async_save_data = AsyncMock()
    smear = {"balance_point": 17.0, "cop_params": COP}
    dist = [{"synthetic_kwh_el": 0.5} for _ in range(24)]
    coord._daily_processor.run_track_c_midnight_sync = AsyncMock(
        return_value=(12.0, dist, "live", smear)
    )
    coord._daily_processor.apply_strategies_to_global_model = MagicMock(return_value=3)
    coord._hourly_log = _full_day_logs()
    coord._accumulated_energy_today = 12.0
    coord.data[ATTR_TDD] = 5.0

    await coord._process_daily_data(date(2026, 4, 20))

    assert coord._daily_history["2026-04-20"]["track_c_smear"] == smear


def test_diagnose_model_compares_buckets_with_the_re_spread_day():
    mpc = "sensor.heater_mpc"
    logs = []
    for h in range(24):
        t = 11.0 + h * 0.25
        logs.append({
            "timestamp": f"{DAY}T{h:02d}:00:00",
            "hour": h,
            "temp": t,
            "inertia_temp": t,
            # Two buckets, so a re-spread moves energy between them.
            "temp_key": "10" if h < 12 else "15",
            "wind_bucket": "normal",
            "effective_wind": 2.0,
            "actual_kwh": 0.05,
            "solar_factor": 0.0,
            "auxiliary_active": False,
            "unit_modes": {mpc: "heating"},
            "unit_breakdown": {mpc: 0.05},
            "expected_kwh": 0.0,
        })
    stored = _midnight(logs, 15.0, COP)
    daily_history = {DAY: {"track_c_distribution": stored, "track_c_smear": {"cop_params": COP}}}
    coord = _make_diag_coord({"10": {"normal": 0.5}, "15": {"normal": 0.2}}, logs,
                             daily_history=daily_history,
                             track_c_enabled=True, mpc_managed_sensor=mpc)
    coord.balance_point = 18.0
    coord.wind_threshold = 8.0
    coord.extreme_wind_threshold = 10.8
    coord.solar_battery_decay = 0.5

    result = DiagnosticsEngine(coord).diagnose_model(days_back=30)

    bucket = result["base_model_health"]["buckets"]["10"]["normal"]
    expected, _ = resmear_track_c_day(coord, logs, daily_history[DAY])
    assert bucket["actual_dark_mean_kwh"] == pytest.approx(
        sum(_synth(expected)[:12]) / 12, abs=1e-3
    )
    assert bucket["actual_dark_mean_kwh"] != pytest.approx(
        sum(_synth(stored)[:12]) / 12, abs=1e-3
    )
