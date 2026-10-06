"""Calibration services exclude partial-aux hours, not just aux-dominant ones.

``auxiliary_active`` on ``hourly_log`` entries is the learning-side
dominance flag, set only when aux ran ≥ 80 % of the hour.  A 1–79 % aux
hour carries ``auxiliary_active=False`` with a non-zero ``aux_impact_kwh``
and metered demand already reduced by aux, so it is not a "pure" hour.
``calibrate_wind_thresholds`` and ``calibrate_inertia`` must discard it,
counted under the existing ``auxiliary_active`` reason.
"""
from __future__ import annotations

import math
from datetime import timedelta

import pytest
from homeassistant.util import dt as dt_util

from custom_components.heating_analytics.helpers import hour_had_any_aux
from custom_components.heating_analytics.statistics import StatisticsManager


class TestHourHadAnyAux:
    def test_dominant_flag(self):
        assert hour_had_any_aux({"auxiliary_active": True}) is True

    def test_partial_aux_hour(self):
        assert hour_had_any_aux(
            {"auxiliary_active": False, "aux_impact_kwh": 0.3}
        ) is True

    @pytest.mark.parametrize(
        "entry",
        [
            {},
            {"auxiliary_active": False, "aux_impact_kwh": 0.0},
            {"auxiliary_active": False, "aux_impact_kwh": None},
            {"auxiliary_active": False, "aux_impact_kwh": "bad"},
        ],
    )
    def test_no_aux(self, entry):
        assert hour_had_any_aux(entry) is False


def _wind_log(n_pure: int, n_partial: int) -> list[dict]:
    now = dt_util.now()
    logs = []
    for i in range(n_pure + n_partial):
        partial = i >= n_pure
        logs.append({
            "timestamp": (now - timedelta(hours=i + 1)).isoformat(),
            "temp": 5.0,
            "temp_key": "5",
            "effective_wind": 2.0,
            "actual_kwh": 1.0,
            "solar_impact_kwh": 0.0,
            "auxiliary_active": False,
            "aux_impact_kwh": 0.3 if partial else 0.0,
        })
    return logs


def test_wind_calibration_discards_partial_aux_hours(mock_coordinator):
    mock_coordinator._hourly_log = _wind_log(n_pure=10, n_partial=3)
    mock_coordinator._correlation_data = {"5": {"normal": 1.0}}
    mock_coordinator.wind_threshold = 5.5
    mock_coordinator.extreme_wind_threshold = 10.8
    result = StatisticsManager(mock_coordinator).calibrate_wind_thresholds(days=30)
    assert result["discarded_hours"]["auxiliary_active"] == 3
    assert result["pure_hours_found"] == 10


def test_wind_calibration_with_only_partial_aux_hours_finds_no_pure_data(mock_coordinator):
    mock_coordinator._hourly_log = _wind_log(n_pure=0, n_partial=5)
    mock_coordinator._correlation_data = {"5": {"normal": 1.0}}
    result = StatisticsManager(mock_coordinator).calibrate_wind_thresholds(days=30)
    assert result["pure_hours_found"] == 0
    assert result["discarded_hours"]["auxiliary_active"] == 5


def test_inertia_calibration_discards_partial_aux_hours(mock_coordinator):
    mock_coordinator.balance_point = 17.0
    now = dt_util.now()
    start = now - timedelta(days=20)
    logs = []
    n_partial = 0
    for i in range(20 * 24):
        temp = 5.0 + 5.0 * math.sin(i / 12.0)
        partial = i % 10 == 0
        n_partial += partial
        logs.append({
            "timestamp": (start + timedelta(hours=i)).isoformat(),
            "temp": temp,
            "actual_kwh": max(0.0, 17.0 - temp) / 24.0 * 15.0,
            "solar_impact_kwh": 0.0,
            "auxiliary_active": False,
            "aux_impact_kwh": 0.3 if partial else 0.0,
            "learning_status": "active",
        })
    mock_coordinator._hourly_log = logs

    result = StatisticsManager(mock_coordinator).calibrate_inertia(days=15)

    assert "error" not in result
    in_window = [
        e for e in logs
        if e["timestamp"] >= (now - timedelta(days=15)).isoformat()
    ]
    expected_partial = sum(1 for e in in_window if e["aux_impact_kwh"] > 0)
    assert expected_partial > 0
    assert result["discarded_hours"]["auxiliary_active"] == expected_partial
