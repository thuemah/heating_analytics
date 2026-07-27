"""Per-day thermal-regime split persisted on daily_history (#1051).

``hourly_log`` trims after retention (default 90 days) but ``daily_history``
is unbounded, so a comparison against the same date a year earlier has no
mode information to work from.  Recording the regime energy split on the
daily aggregate starts that clock.

The split is persisted rather than the label so a future change to
``THERMAL_REGIME_DOMINANCE_SHARE`` reclassifies history instead of leaving
stale labels behind.  Absent keys mean "predates recording" and must stay
distinguishable from a recorded 0/0, which is a genuine "idle" day.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from custom_components.heating_analytics.const import (
    MODE_COOLING,
    MODE_DHW,
    MODE_HEATING,
)
from custom_components.heating_analytics.daily_processor import DailyProcessor


def _make_coord():
    coord = MagicMock()
    coord.solar_enabled = False
    coord.balance_point = 15.0
    coord.hourly_solar_impact_kwh = MagicMock(return_value=0.0)
    return coord


def _make_log(hour: int, breakdown: dict, modes: dict | None = None) -> dict:
    return {
        "timestamp": f"2026-05-17T{hour:02d}:00:00",
        "hour": hour,
        "temp": 10.0,
        "effective_wind": 3.0,
        "solar_factor": 0.0,
        "actual_kwh": sum(breakdown.values()),
        "tdd": 0.5,
        "unit_breakdown": breakdown,
        # Mirrors the live logger, which filters MODE_HEATING out to reduce
        # clutter — so the map is sparse by design.
        "unit_modes": {k: v for k, v in (modes or {}).items() if v != MODE_HEATING},
    }


class TestDailyRegimeSplit:
    def test_heating_day_splits_entirely_to_heating(self):
        """Sparse unit_modes must not lose the implicit-heating units."""
        proc = DailyProcessor(_make_coord())
        logs = [_make_log(h, {"rad": 2.0}) for h in range(24)]

        result = proc.aggregate_logs(logs)

        assert result["regime_heating_kwh"] == pytest.approx(48.0)
        assert result["regime_cooling_kwh"] == pytest.approx(0.0)

    def test_cooling_day_splits_entirely_to_cooling(self):
        proc = DailyProcessor(_make_coord())
        logs = [_make_log(h, {"ac": 1.5}, {"ac": MODE_COOLING}) for h in range(24)]

        result = proc.aggregate_logs(logs)

        assert result["regime_heating_kwh"] == pytest.approx(0.0)
        assert result["regime_cooling_kwh"] == pytest.approx(36.0)

    def test_intraday_mode_switch_is_attributed_per_hour(self):
        """Each hour is attributed by its own modes, not the day's last state.

        A shoulder-season day that heats overnight and cools in the afternoon
        must not be stamped wholesale with whichever mode happened to be
        active at midnight.
        """
        proc = DailyProcessor(_make_coord())
        logs = [_make_log(h, {"hp": 2.0}) for h in range(12)]
        logs += [_make_log(h, {"hp": 1.0}, {"hp": MODE_COOLING}) for h in range(12, 24)]

        result = proc.aggregate_logs(logs)

        assert result["regime_heating_kwh"] == pytest.approx(24.0)
        assert result["regime_cooling_kwh"] == pytest.approx(12.0)

    def test_dhw_is_excluded_from_the_daily_split(self):
        proc = DailyProcessor(_make_coord())
        logs = [
            _make_log(h, {"rad": 1.0, "dhw": 3.0}, {"dhw": MODE_DHW})
            for h in range(24)
        ]

        result = proc.aggregate_logs(logs)

        assert result["regime_heating_kwh"] == pytest.approx(24.0)
        assert result["regime_cooling_kwh"] == pytest.approx(0.0)

    def test_split_is_independent_of_the_unit_breakdown_total(self):
        """The split covers only thermal units, so it may be below total kwh."""
        proc = DailyProcessor(_make_coord())
        logs = [_make_log(h, {"rad": 1.0, "dhw": 3.0}, {"dhw": MODE_DHW}) for h in range(24)]

        result = proc.aggregate_logs(logs)

        assert result["kwh"] == pytest.approx(96.0)
        split_total = result["regime_heating_kwh"] + result["regime_cooling_kwh"]
        assert split_total == pytest.approx(24.0)


class TestThermalRegimeForDay:
    """coordinator.thermal_regime_for_day — unrecorded must not read as idle."""

    def _coord(self, history):
        from custom_components.heating_analytics.coordinator import (
            HeatingDataCoordinator,
        )

        coord = MagicMock(spec=HeatingDataCoordinator)
        coord._daily_history = history
        coord.thermal_regime_for_day = (
            HeatingDataCoordinator.thermal_regime_for_day.__get__(coord)
        )
        return coord

    def test_day_predating_the_field_returns_none(self):
        """None, not "idle" — there is no evidence either way.

        Treating it as idle would silently reclassify every pre-upgrade day
        as a building doing nothing.
        """
        coord = self._coord({"2025-07-26": {"kwh": 40.0}})
        assert coord.thermal_regime_for_day("2025-07-26") is None

    def test_recorded_zero_split_is_idle_not_none(self):
        coord = self._coord({
            "2026-07-26": {"regime_heating_kwh": 0.0, "regime_cooling_kwh": 0.0},
        })
        assert coord.thermal_regime_for_day("2026-07-26") == "idle"

    def test_missing_day_returns_none(self):
        coord = self._coord({})
        assert coord.thermal_regime_for_day("2020-01-01") is None

    @pytest.mark.parametrize("heating,cooling,expected", [
        (20.0, 0.0, "heating"),
        (0.0, 20.0, "cooling"),
        (12.0, 8.0, "mixed"),
    ])
    def test_stored_split_classifies_by_the_same_rule(self, heating, cooling, expected):
        coord = self._coord({
            "2026-07-26": {
                "regime_heating_kwh": heating,
                "regime_cooling_kwh": cooling,
            },
        })
        assert coord.thermal_regime_for_day("2026-07-26") == expected

    def test_partial_keys_are_treated_as_unrecorded(self):
        """Half a split is not a split."""
        coord = self._coord({"2026-07-26": {"regime_heating_kwh": 20.0}})
        assert coord.thermal_regime_for_day("2026-07-26") is None
