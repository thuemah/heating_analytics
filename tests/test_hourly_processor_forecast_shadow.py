"""Regression coverage for the hour-boundary forecast-shadow consumption.

``HourlyProcessor.process`` queries a single-point forecast kWh per weather
item via ``_forecast_point_kwh`` (formerly an inline ``_get_f_kwh`` closure).
That consumer destructured the full ``_process_forecast_item`` return into a
fixed-width tuple; when the tuple was widened 10 → 12 (#1037 solar split) the
unpack raised ``ValueError`` on every hourly cycle and the coordinator update
loop crashed in production — yet the whole suite stayed green, because no test
drove this path with a *populated* item (the closure short-circuits on the
``None`` items the unit tests fed).

These tests close that gap: they exercise the consumer with a populated item
and a realistic 12-element ``_process_forecast_item`` return, so any consumer
that re-couples to the tuple arity is caught at test time.
"""
from unittest.mock import MagicMock

import pytest

from custom_components.heating_analytics.hourly_processor import HourlyProcessor


def _make_processor(process_item_return):
    coord = MagicMock()
    coord.forecast._process_forecast_item.return_value = process_item_return
    return HourlyProcessor(coord), coord


# A realistic full-width return: (predicted, solar_kwh, inertia, raw_temp,
# w_speed, w_speed_ms, unit_breakdown, aux_impact, solar_heating_wasted,
# dni_dhi_meta, solar_offset, solar_load) — 12 elements as of #1037.
_TWELVE_TUPLE = (1.23, 0.0, 10.0, 10.0, 5.0, 1.4, {}, 0.0, 0.0, None, 0.1, 0.0)


def test_forecast_point_kwh_returns_predicted_from_populated_item():
    """A populated item yields element [0] without crashing on a 12-tuple."""
    proc, coord = _make_processor(_TWELVE_TUPLE)
    item = {"temperature": 5.0, "wind_speed": 3.0, "datetime": "2026-06-25T12:00:00+02:00"}

    val = proc._forecast_point_kwh(item, [10.0], "m/s", 50.0, ignore_aux=False)

    assert val == pytest.approx(1.23)
    # The consumer must call _process_forecast_item with the item, a *copy*
    # of the inertia seed, the wind unit, the cloud, and ignore_aux.
    args, kwargs = coord.forecast._process_forecast_item.call_args
    assert args[0] is item
    assert args[1] == [10.0]
    assert args[2] == "m/s"
    assert args[3] == 50.0
    assert kwargs["ignore_aux"] is False


def test_forecast_point_kwh_threads_ignore_aux():
    proc, coord = _make_processor(_TWELVE_TUPLE)
    item = {"temperature": 5.0, "wind_speed": 3.0, "datetime": "2026-06-25T12:00:00+02:00"}

    proc._forecast_point_kwh(item, [10.0], "m/s", 50.0, ignore_aux=True)

    _, kwargs = coord.forecast._process_forecast_item.call_args
    assert kwargs["ignore_aux"] is True


def test_forecast_point_kwh_short_circuits_on_falsy_item():
    """None / empty items return None without touching _process_forecast_item
    — this is why the original crash hid: the unit tests only ever hit this
    branch."""
    proc, coord = _make_processor(_TWELVE_TUPLE)

    assert proc._forecast_point_kwh(None, [10.0], "m/s", 50.0) is None
    assert proc._forecast_point_kwh({}, [10.0], "m/s", 50.0) is None
    coord.forecast._process_forecast_item.assert_not_called()


def test_forecast_point_kwh_decoupled_from_tuple_arity():
    """The consumer must not re-couple to a fixed tuple width.  A 13-element
    (hypothetical future widening) return must still yield element [0], not
    raise ValueError as the old fixed-width unpack did."""
    wider = _TWELVE_TUPLE + (0.0,)
    proc, _ = _make_processor(wider)
    item = {"temperature": 5.0, "wind_speed": 3.0, "datetime": "2026-06-25T12:00:00+02:00"}

    assert proc._forecast_point_kwh(item, [10.0], "m/s", 50.0) == pytest.approx(1.23)
