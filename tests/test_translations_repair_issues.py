"""Translation integrity for the repair-issue strings (#1070).

Same class of failure as ``test_translations_4d_readiness`` and the same
reason it needs its own test: every one of these reaches the user as
visible breakage in the repair card, and none of it can fail a test that
only exercises Python.  ``conftest.py`` stubs the whole repairs module,
so no test in the suite renders a single one of these strings.

Three things pinned:

* **Placeholder agreement in both directions.**  A placeholder the string
  references but ``async_check_dni_dhi_outage`` does not supply renders
  literally as ``{real_hours}`` in the card; one supplied but never
  referenced is merely dead.  The first is the damaging direction.
* **Every ``async_abort`` reason has a string.**  A missing one closes
  the dialog with an empty body while the repair stays on screen
  unexplained — the user is told nothing and the issue does not go away.
  This is exactly what shipped: the flow gained ``entry_not_found`` and
  neither language file gained a matching key.
* **Key parity between ``en`` and ``nb``.**  A key missing from ``nb``
  falls back to rendering the translation key itself.
"""
import json
import re
from pathlib import Path

import pytest

COMPONENT = (
    Path(__file__).parent.parent / "custom_components" / "heating_analytics"
)
TRANSLATIONS = COMPONENT / "translations"
LANGUAGES = ("en", "nb")

ISSUE_KEY = "dni_dhi_outage"

# Supplied by ``repairs.async_check_dni_dhi_outage``'s
# ``translation_placeholders``.  Kept as a literal rather than imported
# so a rename in the source has to be made deliberately here too.
SUPPLIED_PLACEHOLDERS = {"name", "hours", "real_hours"}


def _issues(lang):
    return json.loads((TRANSLATIONS / f"{lang}.json").read_text())["issues"]


def _flatten(obj, prefix=""):
    """All leaf key paths, so parity compares structure not just top level."""
    out = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            out |= _flatten(v, f"{prefix}.{k}" if prefix else k)
    else:
        out.add(prefix)
    return out


@pytest.mark.parametrize("lang", LANGUAGES)
def test_issue_block_exists(lang):
    assert ISSUE_KEY in _issues(lang)


@pytest.mark.parametrize("lang", LANGUAGES)
def test_description_references_only_supplied_placeholders(lang):
    """An unsupplied placeholder renders literally in the repair card."""
    description = _issues(lang)[ISSUE_KEY]["description"]
    referenced = set(re.findall(r"\{(\w+)\}", description))
    assert referenced <= SUPPLIED_PLACEHOLDERS, (
        f"{lang}: references placeholders nothing supplies: "
        f"{referenced - SUPPLIED_PLACEHOLDERS}"
    )


@pytest.mark.parametrize("lang", LANGUAGES)
def test_every_abort_reason_in_the_flow_has_a_string(lang):
    """The gap that shipped: ``entry_not_found`` had no translation.

    Reads the reasons out of ``repairs.py`` rather than listing them, so
    a new ``async_abort`` cannot be added without a string.
    """
    source = (COMPONENT / "repairs.py").read_text()
    reasons = set(re.findall(r'async_abort\(\s*reason="(\w+)"', source))
    assert reasons, "no async_abort reasons found — has the flow changed?"

    strings = _issues(lang)[ISSUE_KEY]["fix_flow"].get("abort", {})
    missing = reasons - set(strings)
    assert not missing, f"{lang}: abort reasons with no string: {missing}"


def test_both_languages_carry_the_same_keys():
    """A key missing from nb renders as the raw translation key."""
    assert _flatten(_issues("en")) == _flatten(_issues("nb"))


@pytest.mark.parametrize("lang", LANGUAGES)
def test_strings_are_non_empty(lang):
    for path in _flatten(_issues(lang)):
        node = _issues(lang)
        for part in path.split("."):
            node = node[part]
        assert isinstance(node, str) and node.strip(), f"{lang}: {path} is empty"
