"""Tests for the DNI/DHI ladder source-mix gate.

``diagnose_solar.dni_dhi_source_mix`` is the per-install gate for
``experimental_4d_primary``: 4D is superior wherever a real DNI/DHI
source exists and inferior on the permanent ``kasten_synthetic`` branch,
so the only question is which branch the install resolves through.

Also pins the agreement between ``solar.derive_dni_dhi_source_label``
(availability-based, used by the hour-boundary logger) and
``solar.resolve_dni_dhi`` (value-based, used by the pipeline) — they are
the same ladder expressed over different inputs and must not drift.
"""

from datetime import timedelta
from unittest.mock import MagicMock

import pytest
from homeassistant.util import dt as dt_util

from custom_components.heating_analytics.const import (
    DNI_DHI_REAL_SOURCE_DOMINANCE_MIN,
    DNI_DHI_SOURCE_MIX_MIN_HOURS,
)
from custom_components.heating_analytics.diagnostics import DiagnosticsEngine
from custom_components.heating_analytics.solar import (
    derive_dni_dhi_source_label,
    resolve_dni_dhi,
)


# --------------------------------------------------------------------
# Ladder-mirror agreement
# --------------------------------------------------------------------


@pytest.mark.parametrize(
    "ghi, dni, dhi, cloud, expected",
    [
        (500.0, 800.0, 100.0, 40.0, "erbs_from_ghi"),  # GHI wins over native
        (None, 800.0, 100.0, 40.0, "native"),
        (None, None, None, 40.0, "kasten_synthetic"),
        (None, 800.0, None, 40.0, "kasten_synthetic"),  # native needs BOTH
        (None, None, None, None, "none"),
        # A configured-but-dark GHI sensor must NOT claim the GHI branch:
        # resolve_dni_dhi requires ghi > 0 and falls through, so labelling
        # these hours erbs_from_ghi would report a real DNI/DHI source to
        # the source-mix gate on an install actually running Kasten.
        (0.0, None, None, 40.0, "kasten_synthetic"),
        (0.0, 800.0, 100.0, 40.0, "native"),
        (0.0, None, None, None, "none"),
        (-1.0, None, None, 40.0, "kasten_synthetic"),
    ],
)
def test_label_matches_resolve_dni_dhi_in_daylight(ghi, dni, dhi, cloud, expected):
    """The label agrees with the value ladder above the horizon.

    Both take the same averaged values and must pick the same branch,
    otherwise the logged label misrepresents which path the pipeline
    actually took — and the source-mix gate built on those labels
    inherits the error.
    """
    label = derive_dni_dhi_source_label(ghi, dni, dhi, cloud)
    _, _, resolved = resolve_dni_dhi(dni, dhi, ghi, cloud, 30.0, 180)

    assert label == expected
    assert label == resolved


def test_label_never_returns_no_sun():
    """The label has no sun gate — hence the daylight filter downstream.

    ``resolve_dni_dhi`` short-circuits to ``no_sun`` below the horizon,
    but the logger's label cannot: it has no sun position.  A night hour
    with cloud data is therefore labelled ``kasten_synthetic``.  This
    asymmetry is why the source-mix block must filter on daylight.
    """
    label = derive_dni_dhi_source_label(None, None, None, 40.0)
    _, _, resolved = resolve_dni_dhi(None, None, None, 40.0, -5.0, 180)

    assert label == "kasten_synthetic"
    assert resolved == "no_sun"


# --------------------------------------------------------------------
# Source-mix report
# --------------------------------------------------------------------


def _engine(entries, flag_on=False):
    coordinator = MagicMock()
    coordinator._hourly_log = entries
    coordinator.experimental_4d_primary = flag_on
    return DiagnosticsEngine(coordinator)


def _entry(source, *, solar_factor=0.5, days_ago=1, include_source=True):
    # Anchor on the (stubbed) clock the production cutoff reads, not on
    # the wall clock — otherwise every entry lands after the cutoff and
    # the window filter is never actually exercised.
    ts = dt_util.now() - timedelta(days=days_ago)
    entry = {"timestamp": ts.isoformat(), "solar_factor": solar_factor}
    if include_source:
        entry["dni_dhi_source"] = source
    return entry


def _log(source, n, **kw):
    return [_entry(source, **kw) for _ in range(n)]


def test_native_dominated_install_supports_4d():
    engine = _engine(_log("native", 100))
    result = engine._compute_dni_dhi_source_mix(30)

    assert result["available"] is True
    assert result["daylight_hours_total"] == 100
    assert result["dominant_source"] == "native"
    assert result["real_source_share"] == pytest.approx(1.0)
    assert result["supports_4d_primary"] is True
    assert result["verdict"] == "disabled_but_supported"


def test_kasten_dominated_install_does_not_support_4d():
    engine = _engine(_log("kasten_synthetic", 100))
    result = engine._compute_dni_dhi_source_mix(30)

    assert result["dominant_source"] == "kasten_synthetic"
    assert result["real_source_share"] == pytest.approx(0.0)
    assert result["supports_4d_primary"] is False
    assert result["verdict"] == "disabled_and_unsupported"


def test_flag_on_with_kasten_input_is_flagged_as_unsupported():
    """The footgun case: 4D enabled on an install that cannot feed it."""
    engine = _engine(_log("kasten_synthetic", 100), flag_on=True)
    result = engine._compute_dni_dhi_source_mix(30)

    assert result["experimental_4d_primary_enabled"] is True
    assert result["supports_4d_primary"] is False
    assert result["verdict"] == "enabled_but_unsupported"


def test_ghi_sensor_counts_as_a_real_source():
    engine = _engine(_log("erbs_from_ghi", 80), flag_on=True)
    result = engine._compute_dni_dhi_source_mix(30)

    assert result["supports_4d_primary"] is True
    assert result["verdict"] == "enabled_and_supported"


def test_mixed_provider_below_dominance_bar_does_not_support_4d():
    """An intermittently-dropping provider must not read as supported."""
    entries = _log("native", 70) + _log("kasten_synthetic", 30)
    result = _engine(entries)._compute_dni_dhi_source_mix(30)

    assert result["real_source_share"] == pytest.approx(0.70)
    assert result["real_source_share"] < DNI_DHI_REAL_SOURCE_DOMINANCE_MIN
    assert result["supports_4d_primary"] is False


def test_night_hours_are_excluded_not_counted_as_kasten():
    """The load-bearing filter: darkness must not dilute the mix.

    The logger labels night hours ``kasten_synthetic`` because it has no
    sun gate.  Counting them would make a fully native-sourced install
    look mixed and fail its own gate.
    """
    entries = _log("native", 60) + _log(
        "kasten_synthetic", 200, solar_factor=0.0
    )
    result = _engine(entries)._compute_dni_dhi_source_mix(30)

    assert result["daylight_hours_total"] == 60
    assert result["night_hours_excluded"] == 200
    assert result["dominant_source"] == "native"
    assert result["supports_4d_primary"] is True


def test_unlabelled_legacy_entries_are_reported_not_dropped():
    entries = _log("native", 60) + _log(None, 40, include_source=False)
    result = _engine(entries)._compute_dni_dhi_source_mix(30)

    assert result["daylight_hours_total"] == 60
    assert result["unlabelled_hours"] == 40


def test_thin_window_withholds_a_verdict():
    n = DNI_DHI_SOURCE_MIX_MIN_HOURS - 1
    result = _engine(_log("native", n))._compute_dni_dhi_source_mix(30)

    assert result["available"] is True
    assert result["supports_4d_primary"] is None
    assert result["verdict"] == "insufficient_data"


def test_no_labelled_daylight_hours_is_unavailable():
    result = _engine(_log("kasten_synthetic", 20, solar_factor=0.0))
    result = result._compute_dni_dhi_source_mix(30)

    assert result["available"] is False
    assert result["reason"] == "no_labelled_daylight_hours"
    assert result["night_hours_excluded"] == 20


def test_unsupported_verdict_raises_the_top_level_summary():
    """The footgun must not hide behind ``no_action_needed``.

    A summary that reports all-clear while the gate warns the active 4D
    path is unsupported defeats the point of the gate — dashboards and
    automations read the summary, not the detail blocks.
    """
    engine = _engine(_log("kasten_synthetic", 100), flag_on=True)
    mix = engine._compute_dni_dhi_source_mix(30)

    assert mix["verdict"] == "enabled_but_unsupported"

    any_action = mix.get("verdict") == "enabled_but_unsupported"
    assert any_action is True


def test_supported_but_disabled_is_not_an_input_misconfiguration():
    """Good input with the flag off is an opportunity, not a fault.

    The input half alone never raises the summary on this verdict.  It
    does **not** follow that the summary stays quiet: #1062 added a
    second condition (has 4D actually learned?), and the composite
    ``four_d_readiness`` gate raises ``ready_to_enable`` when both halves
    pass — deliberately reversing the earlier decision, because with the
    learning half added the signal means "input good *and* model
    trained" rather than merely "your provider is fine".  See
    ``tests/test_4d_readiness.py`` for the summary-level behaviour; this
    test pins the input half only.
    """
    engine = _engine(_log("native", 100), flag_on=False)
    mix = engine._compute_dni_dhi_source_mix(30)

    assert mix["verdict"] == "disabled_but_supported"
    assert mix.get("verdict") != "enabled_but_unsupported"


def test_entries_outside_the_window_are_ignored():
    entries = _log("native", 60) + _log("kasten_synthetic", 200, days_ago=90)
    result = _engine(entries)._compute_dni_dhi_source_mix(30)

    assert result["daylight_hours_total"] == 60
    assert result["supports_4d_primary"] is True
