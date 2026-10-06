"""Raw config scalars are coerced at the coordinator boundary (#1073).

Finishes the sweep ``solar_battery_decay`` started: ``wind_gust_factor``,
``balance_point``, ``learning_rate``, ``wind_threshold`` and
``extreme_wind_threshold`` all arrive from ``entry.data`` and feed bare
arithmetic with no downstream guard.  ``learning_rate`` additionally
arrives from the storage JSON, which overrides the config value on load.

Unlike the silent gamma / tau fallback, a malformed value is logged at
ERROR: ``balance_point`` defines TDD, the cold/mild boundary and the BP-2
shield, so a substitution there must be visible.  Failing hard was
rejected because it would stop the integration from loading at all.
"""
from __future__ import annotations

import logging
import math
from unittest.mock import MagicMock, patch

import pytest

from custom_components.heating_analytics.const import (
    DEFAULT_BALANCE_POINT,
    DEFAULT_EXTREME_WIND_THRESHOLD,
    DEFAULT_WIND_GUST_FACTOR,
    DEFAULT_WIND_THRESHOLD,
)
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.helpers import coerce_config_float


class TestCoerceConfigFloat:
    def test_a_float_passes_through(self):
        assert coerce_config_float(16.5, 17.0, "balance_point") == 16.5

    def test_an_int_becomes_a_float(self):
        out = coerce_config_float(8, 10.0, "wind_threshold")
        assert out == 8.0 and isinstance(out, float)

    @pytest.mark.parametrize(
        "bad", ["16.5", "", None, [], {}, object(), math.nan, math.inf, -math.inf]
    )
    def test_malformed_values_fall_back(self, bad):
        assert coerce_config_float(bad, 17.0, "balance_point") == 17.0

    @pytest.mark.parametrize("huge", [10**400, -(10**400)])
    def test_an_int_beyond_float_range_falls_back_rather_than_raising(self, huge):
        """``math.isfinite(10**400)`` raises OverflowError on the raw int."""
        assert coerce_config_float(huge, 17.0, "balance_point") == 17.0

    @pytest.mark.parametrize("flag", [True, False])
    def test_bool_is_rejected_rather_than_becoming_one_or_zero(self, flag):
        assert coerce_config_float(flag, 0.01, "learning_rate") == 0.01

    def test_fallback_is_logged_at_error_not_silent(self, caplog):
        with caplog.at_level(logging.ERROR):
            coerce_config_float("warm", 17.0, "balance_point")
        assert any(
            r.levelno == logging.ERROR and "balance_point" in r.getMessage()
            for r in caplog.records
        )

    def test_a_valid_value_logs_nothing(self, caplog):
        with caplog.at_level(logging.DEBUG):
            coerce_config_float(17.0, 17.0, "balance_point")
        assert not caplog.records


def _build(hass, data):
    entry = MagicMock()
    entry.data = {"energy_sensors": ["sensor.heater"], **data}
    with patch("custom_components.heating_analytics.storage.Store"), \
         patch("custom_components.heating_analytics.coordinator.SolarCalculator"), \
         patch("custom_components.heating_analytics.coordinator.ForecastManager"), \
         patch("custom_components.heating_analytics.coordinator.StatisticsManager"), \
         patch("custom_components.heating_analytics.coordinator.LearningManager"), \
         patch("custom_components.heating_analytics.coordinator.StorageManager"):
        return HeatingDataCoordinator(hass, entry)


class TestCoordinatorBoundary:
    def test_configured_values_survive(self, hass):
        coord = _build(hass, {
            "balance_point": 15.5,
            "learning_rate": 0.02,
            "wind_gust_factor": 0.4,
            "wind_threshold": 7.0,
            "extreme_wind_threshold": 12.0,
        })
        assert coord.balance_point == 15.5
        assert coord.learning_rate == 0.02
        assert coord.wind_gust_factor == 0.4
        assert coord.wind_threshold == 7.0
        assert coord.extreme_wind_threshold == 12.0

    def test_missing_keys_take_the_defaults(self, hass):
        coord = _build(hass, {})
        assert coord.balance_point == DEFAULT_BALANCE_POINT
        assert coord.learning_rate == 0.01
        assert coord.wind_gust_factor == DEFAULT_WIND_GUST_FACTOR
        assert coord.wind_threshold == DEFAULT_WIND_THRESHOLD
        assert coord.extreme_wind_threshold == DEFAULT_EXTREME_WIND_THRESHOLD

    def test_malformed_values_take_the_defaults(self, hass):
        coord = _build(hass, {
            "balance_point": "seventeen",
            "learning_rate": True,
            "wind_gust_factor": None,
            "wind_threshold": [8],
            "extreme_wind_threshold": math.nan,
        })
        assert coord.balance_point == DEFAULT_BALANCE_POINT
        assert coord.learning_rate == 0.01
        assert coord.wind_gust_factor == DEFAULT_WIND_GUST_FACTOR
        assert coord.wind_threshold == DEFAULT_WIND_THRESHOLD
        assert coord.extreme_wind_threshold == DEFAULT_EXTREME_WIND_THRESHOLD


def test_storage_load_routes_learning_rate_through_the_guard():
    """Both storage load sites coerce the stored learning_rate.

    The stored value overrides the config one on load, so guarding only
    ``coordinator.__init__`` would leave the stronger exposure open.
    """
    from pathlib import Path

    source = (
        Path(__file__).parent.parent
        / "custom_components"
        / "heating_analytics"
        / "storage.py"
    ).read_text()
    assert "self.coordinator.learning_rate = data[\"learning_rate\"]" not in source
    assert source.count("\"learning_rate (stored)\"") == 2
