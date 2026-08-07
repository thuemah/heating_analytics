"""The deviation notice is DEBUG, and the level is a decision (#1076).

WARNING means "the operator should look at the component".  Whether a
given deviation is worth acting on depends on season, occupancy and what
this household considers actionable — none of which the component can
see.  It holds the mechanism and publishes it on the sensor; the policy
call belongs to a user automation with the context we lack.

The level had no assertion anywhere in the suite, which is how it stayed
at WARNING long enough to produce 737 lines in 24 h on the reference
install — 63 % of all WARNING/ERROR output.  Same failure shape as the
battery verdict strings: a user-visible decision with nothing pinning it,
so it could drift without breaking anything.

What this file does NOT claim: that the underlying oscillation is fixed.
``_is_deviation_unusual`` is still a bare ``score > threshold``, so
``unusual`` flips in the sensor attribute exactly as it did in the log.
That is tracked separately; these tests pin only where the line goes.
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from custom_components.heating_analytics.statistics import StatisticsManager

ENTITY = "sensor.heater1"
STATS_LOGGER = "custom_components.heating_analytics.statistics"


def _manager_with_unusual_deviation(*, cooldown=False, unusual=True):
    """A StatisticsManager whose next breakdown flags an unusual deviation.

    The helpers around the log line are stubbed rather than driven from a
    synthetic log: the subject here is which channel the line goes to, and
    reconstructing a real deviation would test ``_is_deviation_unusual``
    instead — which has its own coverage.
    """
    coord = MagicMock()
    coord.energy_sensors = [ENTITY]
    coord.daily_individual = {ENTITY: 5.0}
    coord.data = {"effective_wind": 0.0}
    coord._calculate_inertia_temp = MagicMock(return_value=5.0)
    coord._get_wind_bucket = MagicMock(return_value="normal")
    coord.aux_cooldown_active = cooldown
    coord.aux_affected_entities = [ENTITY] if cooldown else []
    coord.hass.states.get = MagicMock(return_value=None)

    mgr = StatisticsManager(coord)
    mgr._get_processed_logs = MagicMock(return_value={})
    mgr._calculate_historical_expectations = MagicMock(return_value=(4.0, 100, 10))
    mgr._calculate_current_hour_expectations = MagicMock(return_value=(0.0, 0, 0))
    # (is_unusual, score, threshold) — the margin that made this noisy.
    mgr._is_deviation_unusual = MagicMock(return_value=(unusual, 0.415, 0.408))
    return mgr


def test_an_unusual_deviation_emits_no_warning(caplog):
    """The regression guard: this line must never be operator-facing."""
    mgr = _manager_with_unusual_deviation()
    with caplog.at_level(logging.DEBUG, logger=STATS_LOGGER):
        breakdown = mgr.calculate_deviation_breakdown()

    assert breakdown[0]["unusual"] is True, "fixture must flag the deviation"
    warnings = [
        r for r in caplog.records
        if r.levelno >= logging.WARNING and "Unusual Deviation" in r.message
    ]
    assert not warnings, (
        "the deviation notice must not reach WARNING — it is user-domain "
        "data, not a call for the operator to inspect the component"
    )


def test_the_notice_is_still_emitted_at_debug(caplog):
    """Moved, not deleted.  It is still useful when debugging this component."""
    mgr = _manager_with_unusual_deviation()
    with caplog.at_level(logging.DEBUG, logger=STATS_LOGGER):
        mgr.calculate_deviation_breakdown()

    debug = [
        r for r in caplog.records
        if r.levelno == logging.DEBUG and "Unusual Deviation" in r.message
    ]
    assert len(debug) == 1


def test_the_measurement_still_reaches_the_sensor(caplog):
    """The channel the changelog points automations at must carry the data.

    Score and threshold in particular: users are told to compare those
    with a margin of their own rather than key on the boolean, so they
    have to be present and numeric.
    """
    mgr = _manager_with_unusual_deviation()
    with caplog.at_level(logging.DEBUG, logger=STATS_LOGGER):
        row = mgr.calculate_deviation_breakdown()[0]

    assert row["unusual"] is True
    assert row["deviation_score"] == pytest.approx(0.415)
    assert row["deviation_threshold"] == pytest.approx(0.408)
    # 100 observations over 10 hours = 10 avg, which is the medium band
    # (>= 5, < 20) — not a claim about maturity, just what the fixture is.
    assert row["confidence"] == "medium"
    assert row["deviation"] == pytest.approx(1.0)


def test_cooldown_suppression_still_suppresses(caplog):
    """Pre-existing behaviour, unchanged by the level move."""
    mgr = _manager_with_unusual_deviation(cooldown=True)
    with caplog.at_level(logging.DEBUG, logger=STATS_LOGGER):
        mgr.calculate_deviation_breakdown()

    assert not [r for r in caplog.records if "Unusual Deviation" in r.message]


def test_nothing_is_logged_when_the_deviation_is_ordinary(caplog):
    mgr = _manager_with_unusual_deviation(unusual=False)
    with caplog.at_level(logging.DEBUG, logger=STATS_LOGGER):
        mgr.calculate_deviation_breakdown()

    assert not [r for r in caplog.records if "Unusual Deviation" in r.message]
