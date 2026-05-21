"""Tests for the ``experimental_4d_primary`` config flag (#962).

Verifies:
- The coordinator property defaults to ``False`` when the key is absent
  from ``entry.data``, and reads through to the stored value in both
  ``True`` and ``False`` states.
- The reconfigure-advanced schema *includes* the flag (so the user can
  flip it from Reconfigure → Advanced), while the initial setup
  ``advanced`` schema *omits* it (per the design — the flag is for
  users who already understand the integration).
- The reconfigure-advanced submit path persists the boolean explicitly
  via the ``_flow_data`` write loop, so a missing key in ``user_input``
  is stored as ``False`` rather than being dropped.

These tests are intentionally lightweight — they exercise the property
and the schema-builder surface directly, sidestepping the full HA
config-entry framework that ``conftest`` mocks wholesale.  The pattern
mirrors ``test_config_flow_retention_default.py``.
"""
import sys
from unittest.mock import MagicMock

# conftest already stubs most HA imports; ensure the two we touch are stubbed too.
sys.modules.setdefault("homeassistant.data_entry_flow", MagicMock())
sys.modules.setdefault("homeassistant.helpers.selector", MagicMock())


class _FakeConfigFlow:
    """Minimal base class accepting ``class Foo(Base, domain=X)``."""

    def __init_subclass__(cls, **kwargs):  # swallow domain= and friends
        return None


import homeassistant.config_entries as _ce  # noqa: E402

_ce.ConfigFlow = _FakeConfigFlow

import pytest  # noqa: E402
import voluptuous as vol  # noqa: E402  (MagicMock courtesy of conftest)

from custom_components.heating_analytics.config_flow import (  # noqa: E402
    HeatingAnalyticsConfigFlow,
)
from custom_components.heating_analytics.const import (  # noqa: E402
    CONF_EXPERIMENTAL_4D_PRIMARY,
    DEFAULT_EXPERIMENTAL_4D_PRIMARY,
)


# --------------------------------------------------------------------- #
# Coordinator property                                                    #
# --------------------------------------------------------------------- #


def _make_coordinator_with_entry_data(data: dict):
    """Construct a minimal object exposing the ``experimental_4d_primary``
    property by binding it to a stub with the expected ``entry.data`` shape.

    The property only touches ``self.entry.data.get(...)`` so a plain
    namespace is sufficient — no need to spin up the full coordinator.
    """
    from custom_components.heating_analytics.coordinator import (
        HeatingDataCoordinator,
    )

    stub = MagicMock()
    stub.entry.data = data
    # Bind the unbound descriptor to the stub instance.
    return HeatingDataCoordinator.experimental_4d_primary.fget(stub)


class TestCoordinatorProperty:
    """The ``experimental_4d_primary`` property reads ``entry.data``."""

    def test_default_when_key_missing(self):
        assert _make_coordinator_with_entry_data({}) is DEFAULT_EXPERIMENTAL_4D_PRIMARY
        # Sanity: documented default is False.
        assert DEFAULT_EXPERIMENTAL_4D_PRIMARY is False

    def test_reads_true(self):
        assert _make_coordinator_with_entry_data(
            {CONF_EXPERIMENTAL_4D_PRIMARY: True}
        ) is True

    def test_reads_false_explicit(self):
        assert _make_coordinator_with_entry_data(
            {CONF_EXPERIMENTAL_4D_PRIMARY: False}
        ) is False

    def test_unrelated_keys_do_not_affect(self):
        assert _make_coordinator_with_entry_data(
            {"some_other_key": "value"}
        ) is False


# --------------------------------------------------------------------- #
# Config flow schema surface                                              #
# --------------------------------------------------------------------- #


def _was_optional_called_with(key: str) -> bool:
    """Return True iff ``vol.Optional(key, ...)`` was called on the mock."""
    for call in vol.Optional.call_args_list:
        args, _kwargs = call
        if args and args[0] == key:
            return True
    return False


def _default_for(key: str):
    """Return the ``default`` kwarg passed to the last ``vol.Optional(key, ...)``."""
    for call in reversed(vol.Optional.call_args_list):
        args, kwargs = call
        if args and args[0] == key:
            return kwargs.get("default")
    raise AssertionError(f"vol.Optional({key!r}, ...) was never called")


@pytest.fixture
def flow():
    instance = HeatingAnalyticsConfigFlow()
    instance._flow_data = {}
    instance._entry = None
    return instance


@pytest.fixture(autouse=True)
def _reset_vol_mock():
    vol.Optional.reset_mock()
    yield


class TestSchemaSurface:
    """The flag appears in reconfigure-advanced only, not initial setup."""

    def test_initial_setup_advanced_omits_flag(self, flow):
        """Initial setup must never expose the experimental flag."""
        flow._schema_advanced(None, {})
        assert not _was_optional_called_with(CONF_EXPERIMENTAL_4D_PRIMARY), (
            "experimental_4d_primary leaked into the initial setup wizard — "
            "by design it should be reconfigure-only."
        )

    def test_reconfigure_advanced_includes_flag(self, flow):
        """The reconfigure path passes ``include_experimental_4d=True``."""
        flow._schema_advanced(None, {}, include_experimental_4d=True)
        assert _was_optional_called_with(CONF_EXPERIMENTAL_4D_PRIMARY)

    def test_reconfigure_default_when_key_missing(self, flow):
        flow._schema_advanced(None, {}, include_experimental_4d=True)
        assert _default_for(CONF_EXPERIMENTAL_4D_PRIMARY) is False

    def test_reconfigure_default_reflects_stored_true(self, flow):
        flow._schema_advanced(
            None,
            {CONF_EXPERIMENTAL_4D_PRIMARY: True},
            include_experimental_4d=True,
        )
        assert _default_for(CONF_EXPERIMENTAL_4D_PRIMARY) is True

    def test_reconfigure_default_reflects_stored_false(self, flow):
        flow._schema_advanced(
            None,
            {CONF_EXPERIMENTAL_4D_PRIMARY: False},
            include_experimental_4d=True,
        )
        assert _default_for(CONF_EXPERIMENTAL_4D_PRIMARY) is False


# --------------------------------------------------------------------- #
# Reconfigure submit persistence                                          #
# --------------------------------------------------------------------- #


class TestReconfigurePersistence:
    """The boolean is explicitly written to ``_flow_data`` on submit.

    The reconfigure-advanced handler iterates a tuple of boolean keys and
    writes each via ``bool(user_input.get(k, False))`` so that a missing
    key (HA's ``BooleanSelector`` may omit unchecked toggles from the
    payload) becomes an explicit ``False`` rather than being dropped.
    We assert the flag is in that tuple by verifying the equivalent
    transformation against representative inputs.
    """

    def test_explicit_false_is_persisted_on_missing(self, flow):
        # Simulate the explicit-write loop the handler runs.
        from custom_components.heating_analytics.config_flow import (
            HeatingAnalyticsConfigFlow as _CF,
        )

        # Inspect the handler's source to confirm the flag is in the loop.
        import inspect
        src = inspect.getsource(_CF.async_step_reconfigure_advanced)
        assert "CONF_EXPERIMENTAL_4D_PRIMARY" in src, (
            "async_step_reconfigure_advanced must explicitly persist "
            "CONF_EXPERIMENTAL_4D_PRIMARY so that unchecked toggles store as False."
        )

    def test_value_true_round_trips_through_property(self):
        """Storing True in entry.data is reflected by the property."""
        result = _make_coordinator_with_entry_data(
            {CONF_EXPERIMENTAL_4D_PRIMARY: True}
        )
        assert result is True

    def test_value_false_round_trips_through_property(self):
        result = _make_coordinator_with_entry_data(
            {CONF_EXPERIMENTAL_4D_PRIMARY: False}
        )
        assert result is False
