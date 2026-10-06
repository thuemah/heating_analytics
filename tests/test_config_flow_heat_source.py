"""The user's heat-source type per unit, set in the config flow.

The advanced page carries one list of energy sensors per type a user can
set.  The lists are UI only: ``_build_final_data`` turns them into
``heat_source_types`` (``{entity: type}``) in the config entry, and the
form reads its defaults back from that map.  ``set_heat_source_type``
writes the same key, so both ways of setting a type land in one place.

Same harness as ``test_config_flow_retention_default``: conftest mocks
``homeassistant.config_entries``, so a minimal base class stands in.
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.modules.setdefault("homeassistant.data_entry_flow", MagicMock())
sys.modules.setdefault("homeassistant.helpers.selector", MagicMock())


class _FakeConfigFlow:
    def __init_subclass__(cls, **kwargs):
        return None


import homeassistant.config_entries as _ce  # noqa: E402

_ce.ConfigFlow = _FakeConfigFlow

import pytest  # noqa: E402
import voluptuous as vol  # noqa: E402  (MagicMock courtesy of conftest)

from custom_components.heating_analytics.config_flow import (  # noqa: E402
    HeatingAnalyticsConfigFlow,
    _heat_source_conflict,
)
from custom_components.heating_analytics.const import (  # noqa: E402
    CONF_HEAT_SOURCE_TYPES,
    HEAT_SOURCE_FORM_FIELDS,
)

HP = "sensor.heat_pump"
PANEL = "sensor.panels"
CABLE = "sensor.cable"
SENSORS = [HP, PANEL, CABLE]

DIRECT = "heat_source_direct_electric_units"
GROUND = "heat_source_ground_source_units"
AIR_WATER = "heat_source_air_to_water_units"
AIR_AIR = "heat_source_air_to_air_units"


@pytest.fixture
def flow():
    instance = HeatingAnalyticsConfigFlow()
    instance._flow_data = {"energy_sensors": list(SENSORS)}
    return instance


@pytest.fixture(autouse=True)
def _reset_vol_mock():
    vol.Optional.reset_mock()
    yield


def _default_for(key: str):
    for call in reversed(vol.Optional.call_args_list):
        args, kwargs = call
        if args and args[0] == key:
            return kwargs.get("default")
    raise AssertionError(f"vol.Optional({key!r}, ...) was never called")


def test_form_fields_are_one_per_device_type():
    assert set(HEAT_SOURCE_FORM_FIELDS) == {DIRECT, GROUND, AIR_WATER, AIR_AIR}


def test_lists_become_one_map_and_are_not_stored(flow):
    data = flow._build_final_data({
        DIRECT: [PANEL, CABLE],
        AIR_AIR: [HP],
        GROUND: [],
        AIR_WATER: ["sensor.not_configured"],
    })

    assert data[CONF_HEAT_SOURCE_TYPES] == {
        PANEL: "direct_electric", CABLE: "direct_electric", HP: "air_to_air",
    }
    for field in HEAT_SOURCE_FORM_FIELDS:
        assert field not in data


def test_empty_lists_clear_every_type(flow):
    flow._flow_data[CONF_HEAT_SOURCE_TYPES] = {HP: "air_to_air"}

    data = flow._build_final_data({field: [] for field in HEAT_SOURCE_FORM_FIELDS})

    assert data[CONF_HEAT_SOURCE_TYPES] == {}


def test_stored_map_is_kept_when_the_lists_are_not_in_the_flow(flow):
    """A flow that never rendered the lists (or a service-written map) keeps
    its types; a sensor no longer configured, or a type no user can set,
    does not."""
    flow._flow_data[CONF_HEAT_SOURCE_TYPES] = {
        HP: "ground_source",
        PANEL: "flat_cop",
        "sensor.removed": "direct_electric",
    }

    data = flow._build_final_data({})

    assert data[CONF_HEAT_SOURCE_TYPES] == {HP: "ground_source"}


def test_new_install_stores_an_empty_map(flow):
    assert flow._build_final_data({})[CONF_HEAT_SOURCE_TYPES] == {}


def test_list_defaults_are_read_back_from_the_stored_map(flow):
    stored = {HP: "air_to_water", PANEL: "direct_electric", CABLE: "direct_electric"}

    flow._schema_advanced(None, {**flow._flow_data, CONF_HEAT_SOURCE_TYPES: stored})

    assert _default_for(AIR_WATER) == [HP]
    assert _default_for(DIRECT) == [PANEL, CABLE]
    assert _default_for(GROUND) == []
    assert _default_for(AIR_AIR) == []


def test_submitted_lists_win_over_the_stored_map_when_the_form_is_shown_again(flow):
    flow._schema_advanced(
        {GROUND: [HP]}, {**flow._flow_data, CONF_HEAT_SOURCE_TYPES: {HP: "air_to_water"}},
    )

    assert _default_for(GROUND) == [HP]


def test_a_sensor_in_two_lists_is_a_conflict():
    assert _heat_source_conflict({DIRECT: [PANEL], AIR_AIR: [PANEL]})
    assert not _heat_source_conflict({DIRECT: [PANEL], AIR_AIR: [HP]})
    assert not _heat_source_conflict({})


@pytest.mark.parametrize("step, kwargs", [
    ("async_step_advanced", {}),
    ("async_step_reconfigure_advanced", {}),
])
async def test_conflicting_lists_show_the_form_again(flow, step, kwargs):
    flow.async_show_form = MagicMock(return_value="form")
    flow._four_d_readiness_placeholder = MagicMock(return_value="")
    flow.async_step_feature_config = MagicMock()
    flow.async_step_reconfigure_feature_config = MagicMock()

    result = await getattr(flow, step)({DIRECT: [PANEL], GROUND: [PANEL]})

    assert result == "form"
    _, show_kwargs = flow.async_show_form.call_args
    assert show_kwargs["errors"] == {"base": "heat_source_conflict"}
    assert DIRECT not in flow._flow_data
    flow.async_step_feature_config.assert_not_called()
    flow.async_step_reconfigure_feature_config.assert_not_called()


@pytest.mark.parametrize("lang", ["en", "nb"])
def test_every_list_is_translated_on_both_steps(lang):
    path = Path(__file__).parents[1] / "custom_components/heating_analytics/translations" / f"{lang}.json"
    config = json.loads(path.read_text(encoding="utf-8"))["config"]

    for step in ("advanced", "reconfigure_advanced"):
        for block in ("data", "data_description"):
            missing = set(HEAT_SOURCE_FORM_FIELDS) - set(config["step"][step][block])
            assert not missing, f"{lang} {step}.{block} lacks {missing}"
    assert config["error"]["heat_source_conflict"]
