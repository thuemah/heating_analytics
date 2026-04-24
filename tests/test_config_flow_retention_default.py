"""Regression test for CONF_HOURLY_LOG_RETENTION_DAYS form persistence.

The retention dropdown uses a SelectSelector whose option values are strings
(``"90"``, ``"180"``, ``"365"``).  ``_build_final_data`` stores the submitted
value as an ``int`` so the coordinator can multiply it by 24.  When the user
reconfigures, ``_schema_advanced`` must cast the stored int back to a string
for the SelectSelector default — otherwise HA cannot match the default to
any option, silently falls back to the first option, and a no-op submit
overwrites the user's choice with 90.

Because conftest mocks ``homeassistant.config_entries`` wholesale, the real
``HeatingAnalyticsConfigFlow`` ends up inheriting from a MagicMock and its
methods are mock attributes.  We sidestep that by patching in a minimal base
class that accepts the ``domain=`` kwarg, then inspecting the recorded
``vol.Optional`` calls.
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
import voluptuous as vol  # noqa: E402  (MagicMock courtesy of conftest)

from custom_components.heating_analytics.config_flow import (  # noqa: E402
    HeatingAnalyticsConfigFlow,
)
from custom_components.heating_analytics.const import (  # noqa: E402
    CONF_HOURLY_LOG_RETENTION_DAYS,
    DEFAULT_HOURLY_LOG_RETENTION_DAYS,
    HOURLY_LOG_RETENTION_OPTIONS,
)


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
    return instance


@pytest.fixture(autouse=True)
def _reset_vol_mock():
    vol.Optional.reset_mock()
    yield


def test_default_is_string_on_initial_config(flow):
    """With no stored value, schema default is DEFAULT_* cast to str."""
    flow._schema_advanced(None, {})
    default = _default_for(CONF_HOURLY_LOG_RETENTION_DAYS)
    assert isinstance(default, str)
    assert default == str(DEFAULT_HOURLY_LOG_RETENTION_DAYS)


@pytest.mark.parametrize("stored", HOURLY_LOG_RETENTION_OPTIONS)
def test_default_is_string_on_reconfigure(flow, stored):
    """Stored int is cast to string so the dropdown preserves the user's choice.

    Before the fix, the int default did not match any of the string option
    values, HA fell back to the first option, and a no-op submit persisted
    ``90`` regardless of what had been stored.
    """
    flow._schema_advanced(None, {CONF_HOURLY_LOG_RETENTION_DAYS: stored})
    default = _default_for(CONF_HOURLY_LOG_RETENTION_DAYS)
    assert isinstance(default, str), (
        f"stored={stored!r} (int) produced default={default!r} "
        f"({type(default).__name__}); SelectSelector option values are "
        "strings, so default must also be str to match."
    )
    assert default == str(stored)


def test_default_matches_a_selector_option_value(flow):
    """The rendered default must equal one of the SelectSelector option values."""
    option_values = [str(v) for v in HOURLY_LOG_RETENTION_OPTIONS]
    for stored in HOURLY_LOG_RETENTION_OPTIONS:
        vol.Optional.reset_mock()
        flow._schema_advanced(None, {CONF_HOURLY_LOG_RETENTION_DAYS: stored})
        default = _default_for(CONF_HOURLY_LOG_RETENTION_DAYS)
        assert default in option_values, (
            f"default={default!r} not in {option_values} — SelectSelector "
            "will silently drop it and the user's choice won't persist."
        )
