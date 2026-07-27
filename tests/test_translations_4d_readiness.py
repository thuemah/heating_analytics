"""Translation integrity for the 4D solar setting (#1062).

Two failure modes this pins, both of which reach the user as visible
breakage rather than a test failure:

* A ``description_placeholders`` key that the translation string does not
  contain is silently dropped, and one the string *does* contain but the
  step never supplies renders as a literal ``{four_d_readiness}`` in the
  form.  Both language files must carry the placeholder, and only in the
  step that supplies it.
* The setting is no longer called "experimental" — the key
  ``experimental_4d_primary`` is historical, the condition is about the
  user's weather provider, not maturity.  A well-meaning revert of the
  strings would put the unanswerable question back in front of the user.
"""
import json
from pathlib import Path

import pytest

TRANSLATIONS = (
    Path(__file__).parent.parent
    / "custom_components"
    / "heating_analytics"
    / "translations"
)
LANGUAGES = ("en", "nb")

# The reconfigure step supplies this placeholder; the initial-setup
# ``advanced`` step does not (and never shows the 4D field at all).
PLACEHOLDER = "{four_d_readiness}"


def _steps(lang):
    return json.loads((TRANSLATIONS / f"{lang}.json").read_text())["config"]["step"]


@pytest.mark.parametrize("lang", LANGUAGES)
def test_reconfigure_advanced_carries_the_placeholder(lang):
    assert PLACEHOLDER in _steps(lang)["reconfigure_advanced"]["description"]


@pytest.mark.parametrize("lang", LANGUAGES)
def test_initial_advanced_step_does_not_carry_the_placeholder(lang):
    """It never supplies one, so a placeholder there renders literally."""
    assert PLACEHOLDER not in _steps(lang)["advanced"]["description"]


@pytest.mark.parametrize("lang", LANGUAGES)
def test_setting_is_not_labelled_experimental(lang):
    """The label states the condition, not a maturity judgement."""
    step = _steps(lang)["reconfigure_advanced"]
    label = step["data"]["experimental_4d_primary"]
    description = step["data_description"]["experimental_4d_primary"]

    assert "xperimental" not in label
    assert "ksperiment" not in label  # nb
    assert "xperimental" not in description
    assert "ksperiment" not in description


@pytest.mark.parametrize("lang", LANGUAGES)
def test_description_points_at_the_readiness_diagnostic(lang):
    """The user needs somewhere to go for the full answer."""
    description = _steps(lang)["reconfigure_advanced"]["data_description"][
        "experimental_4d_primary"
    ]

    assert "diagnose_solar" in description
    assert "four_d_readiness" in description


def test_the_step_supplies_the_placeholder_name_the_strings_expect():
    """Pins the *code* side of the placeholder contract.

    The JSON tests above cannot catch a rename on the Python side: change
    the ``description_placeholders`` key to anything else and every test
    in this repo still passes while the reconfigure form renders a
    literal ``{four_d_readiness}`` to every user.  Source inspection is
    the cheap pin — same pattern as
    ``test_experimental_4d_primary_flag.test_explicit_false_is_persisted_on_missing``.
    """
    import inspect
    import sys
    from unittest.mock import MagicMock

    sys.modules.setdefault("homeassistant.data_entry_flow", MagicMock())
    sys.modules.setdefault("homeassistant.helpers.selector", MagicMock())

    class _FakeConfigFlow:
        def __init_subclass__(cls, **kwargs):
            return None

    import homeassistant.config_entries as _ce

    _ce.ConfigFlow = _FakeConfigFlow

    from custom_components.heating_analytics.config_flow import (
        HeatingAnalyticsConfigFlow as _CF,
    )

    src = inspect.getsource(_CF.async_step_reconfigure_advanced)
    key = PLACEHOLDER.strip("{}")

    assert "description_placeholders" in src, (
        "async_step_reconfigure_advanced must supply description_placeholders, "
        f"or the {PLACEHOLDER} in the translated description renders literally."
    )
    assert f'"{key}"' in src, (
        f"the placeholder key must be exactly {key!r} to match "
        f"{PLACEHOLDER} in en.json and nb.json."
    )


@pytest.mark.parametrize("lang", LANGUAGES)
def test_advanced_and_reconfigure_data_keys_stay_in_sync(lang):
    """Documented invariant, minus the reconfigure-only 4D field.

    ``config_flow``'s module docstring requires the two steps to carry
    identical data keys; ``experimental_4d_primary`` is the deliberate
    exception because the initial wizard never renders it.
    """
    steps = _steps(lang)
    advanced = set(steps["advanced"]["data"])
    reconfigure = set(steps["reconfigure_advanced"]["data"])

    assert reconfigure - advanced == {"experimental_4d_primary"}
    assert advanced - reconfigure == set()
