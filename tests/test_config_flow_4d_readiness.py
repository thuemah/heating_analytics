"""The 4D readiness hint in the reconfigure form (#1062).

``experimental_4d_primary`` used to ask the user a question they had no
way to answer: the label said "experimental" (reads as *unfinished*)
while the actual condition is "does your weather provider supply
irradiance data", and nothing at the point of decision told them which
side of the line they were on.  The form now renders the readiness
verdict in its description.

Two properties are load-bearing and tested here: the hint **warns but
never refuses** (a user who just connected a GHI sensor is right to
enable 4D before the 30-day window catches up), and it **never breaks
the form** — a config flow that raises while rendering a hint is
strictly worse than one that renders no hint.

Same MagicMock-config_entries workaround as
``test_config_flow_retention_default`` — see its docstring.
"""
import sys
from unittest.mock import MagicMock

sys.modules.setdefault("homeassistant.data_entry_flow", MagicMock())
sys.modules.setdefault("homeassistant.helpers.selector", MagicMock())


class _FakeConfigFlow:
    """Minimal base class that accepts ``class Foo(Base, domain=X)``."""

    def __init_subclass__(cls, **kwargs):  # swallow domain= and friends
        return None


import homeassistant.config_entries as _ce  # noqa: E402

_ce.ConfigFlow = _FakeConfigFlow

import pytest  # noqa: E402

from custom_components.heating_analytics.config_flow import (  # noqa: E402
    HeatingAnalyticsConfigFlow,
)
from custom_components.heating_analytics.const import DOMAIN  # noqa: E402


def _flow(readiness, *, language="en", coordinator_present=True, raises=False):
    instance = HeatingAnalyticsConfigFlow()
    instance._flow_data = {}
    instance.context = {"entry_id": "entry1"}

    coordinator = MagicMock()
    if raises:
        coordinator.evaluate_4d_readiness.side_effect = RuntimeError("boom")
    else:
        coordinator.evaluate_4d_readiness.return_value = readiness

    hass = MagicMock()
    hass.data = {DOMAIN: {"entry1": coordinator if coordinator_present else None}}
    hass.config.language = language
    instance.hass = hass
    return instance


def _readiness(verdict, *, input_ok=True, learning_ok=True):
    return {
        "verdict": verdict,
        "ready": verdict in ("ready_to_enable", "enabled_and_ready"),
        "input": {"supports_4d_primary": input_ok},
        "learning": {"ready": learning_ok},
    }


@pytest.mark.parametrize(
    "verdict",
    ["ready_to_enable", "enabled_and_ready", "enabled_but_not_ready", "not_ready"],
)
def test_every_actionable_verdict_renders_a_line(verdict):
    flow = _flow(_readiness(verdict, input_ok=False, learning_ok=False))
    text = flow._four_d_readiness_placeholder()

    assert text, f"{verdict} rendered nothing"
    assert text.startswith("\n\n")
    # No unsubstituted placeholder survives into the UI.
    assert "{reason}" not in text


def test_insufficient_data_renders_nothing():
    """A fresh install has nothing useful to say; silence beats noise."""
    assert _flow(_readiness("insufficient_data"))._four_d_readiness_placeholder() == ""


def test_unknown_verdict_renders_nothing():
    assert _flow(_readiness("some_new_verdict"))._four_d_readiness_placeholder() == ""


def test_norwegian_locale_gets_norwegian_text():
    """The hint's whole purpose is removing confusion — half-English defeats it."""
    text = _flow(_readiness("ready_to_enable"), language="nb")._four_d_readiness_placeholder()

    assert "installasjonen din er klar" in text


def test_regional_language_tag_matches_base_language():
    """HA language codes can be regional (``nb-NO``)."""
    text = _flow(
        _readiness("ready_to_enable"), language="nb-NO"
    )._four_d_readiness_placeholder()

    assert "installasjonen din er klar" in text


def test_unknown_language_falls_back_to_english():
    text = _flow(_readiness("ready_to_enable"), language="de")._four_d_readiness_placeholder()

    assert "ready" in text.lower()


@pytest.mark.parametrize(
    "input_ok, learning_ok, expected_fragment",
    [
        (False, True, "does not supply direct/diffuse irradiance"),
        (True, False, "has not learned enough yet"),
        (False, False, "and"),
    ],
)
def test_reason_names_the_half_that_failed(input_ok, learning_ok, expected_fragment):
    """Both halves can fail at once, so the reasons compose."""
    flow = _flow(_readiness("not_ready", input_ok=input_ok, learning_ok=learning_ok))
    text = flow._four_d_readiness_placeholder()

    assert expected_fragment in text


def test_both_halves_failing_names_both():
    flow = _flow(_readiness("not_ready", input_ok=False, learning_ok=False))
    text = flow._four_d_readiness_placeholder()

    assert "does not supply direct/diffuse irradiance" in text
    assert "has not learned enough yet" in text


def test_toggle_is_the_first_field_under_the_verdict():
    """The verdict and the control it describes must render together.

    The readiness line can only live in the step *description* (HA
    substitutes ``description_placeholders`` there), so the toggle has to
    come to it: before this, the verdict sat at the top of the page and
    the toggle 17 fields below, describing a control the user had to
    scroll to find.  Field order is dict insertion order, which makes
    this silently breakable by anyone adding a field above it.
    """
    import voluptuous as vol

    flow = _flow(_readiness("ready_to_enable"))
    vol.Optional.reset_mock()
    flow._schema_advanced(None, {}, include_experimental_4d=True)

    keys = [c[0][0] for c in vol.Optional.call_args_list if c[0]]
    assert keys, "no vol.Optional fields were built"
    assert keys[0] == "experimental_4d_primary", (
        f"4D toggle must be the first field on the page, got {keys[0]!r}. "
        "The readiness verdict renders immediately above it in the step "
        "description and would otherwise refer to a distant control."
    )


def test_toggle_is_absent_from_the_initial_wizard():
    """Reconfigure-only: the setup wizard must not gain the field."""
    import voluptuous as vol

    flow = _flow(_readiness("ready_to_enable"))
    vol.Optional.reset_mock()
    flow._schema_advanced(None, {})

    keys = [c[0][0] for c in vol.Optional.call_args_list if c[0]]
    assert "experimental_4d_primary" not in keys


def test_moving_the_toggle_did_not_drop_other_fields():
    """The insert-first restructure must not lose any field.

    ``schema`` went from a dict literal to ``{}`` + ``.update({...})``; a
    mispaired brace there would silently truncate the page rather than
    raise.
    """
    import voluptuous as vol

    flow = _flow(_readiness("ready_to_enable"))
    vol.Optional.reset_mock()
    flow._schema_advanced(None, {}, include_experimental_4d=True)
    keys = [c[0][0] for c in vol.Optional.call_args_list if c[0]]

    for expected in (
        "daily_learning_mode",
        "track_c_enabled",
        "aux_affected_entities",
        "screen_south",
        "screen_affected_entities",
        "solar_affected_entities",
        "secondary_weather_entity",
        "forecast_crossover_day",
        "solar_hotspot_attenuation_gamma",
        "solar_redistribution_tau_hours",
        "experimental_4d_primary",
        "csv_auto_logging",
        "max_energy_delta",
        "hourly_log_retention_days",
    ):
        assert expected in keys, f"{expected} disappeared from the advanced page"


def test_missing_coordinator_renders_nothing_instead_of_raising():
    """Entry not loaded — the form must still open."""
    assert _flow(None, coordinator_present=False)._four_d_readiness_placeholder() == ""


def test_readiness_failure_never_breaks_the_form():
    """A hint that raises is strictly worse than no hint."""
    assert _flow(None, raises=True)._four_d_readiness_placeholder() == ""
