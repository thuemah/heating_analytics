"""Tests for the two-condition 4D readiness gate (#1062).

``experimental_4d_primary`` is guarded by two conditions, and either one
alone is misleading:

1. the install's weather input supports 4D (``dni_dhi_source_mix``), and
2. 4D has **actually learned** the regime each in-scope entity is
   currently operating in.

Condition 2 exists because 4D fails differently from 3D when untrained:
an unlearned 3D regime falls back to a seeded default, an unlearned 4D
regime returns the zero-vector.  Routing an untrained install to 4D
predicts zero solar, silently.

The most load-bearing test here is the mirror test: readiness must fire
the *same* predicate the live read path fires, not a copy of it.
"""

from datetime import timedelta
from unittest.mock import MagicMock

import pytest
from homeassistant.util import dt as dt_util

from custom_components.heating_analytics.const import (
    MODE_COOLING,
    MODE_DHW,
    MODE_GUEST_COOLING,
    MODE_GUEST_HEATING,
    MODE_HEATING,
    MODE_OFF,
)
from custom_components.heating_analytics.diagnostics import DiagnosticsEngine
from custom_components.heating_analytics.learning import (
    evaluate_4d_learning_readiness,
)
from custom_components.heating_analytics.solar import (
    SolarCalculator,
    coefficients_4d_are_learned,
)


def _learned(s=0.4, e=0.2, w=0.3, diffuse=0.1):
    return {"s": s, "e": e, "w": w, "diffuse": diffuse, "learned": True}


def _unlearned():
    return {"s": 0.0, "e": 0.0, "w": 0.0, "diffuse": 0.0, "learned": False}


# --------------------------------------------------------------------
# The mirror: readiness and the read path share one predicate
# --------------------------------------------------------------------


@pytest.mark.parametrize(
    "regime_coeff, expected",
    [
        (_learned(), True),
        (_unlearned(), False),
        # learned flag set but every component zero — not a model
        ({"s": 0.0, "e": 0.0, "w": 0.0, "diffuse": 0.0, "learned": True}, False),
        # non-zero components but never marked learned
        ({"s": 0.4, "e": 0.0, "w": 0.0, "diffuse": 0.0, "learned": False}, False),
        # diffuse alone is enough
        ({"s": 0.0, "e": 0.0, "w": 0.0, "diffuse": 0.2, "learned": True}, True),
        ({}, False),
        (None, False),
        ("not a dict", False),
    ],
)
def test_predicate_matches_read_path_fire_condition(regime_coeff, expected):
    """The predicate and ``calculate_unit_coefficient_4d`` agree case for case.

    What this pins is *behavioural* agreement across the table below — a
    re-inlined copy that drifts is caught, a faithful one is not (and
    does no harm).  The structural requirement that there be exactly one
    implementation is stated in CLAUDE.md and enforced by review.

    Note the ``None`` / non-dict / ``{}`` rows exercise the standalone
    function only: three ``isinstance`` guards inside
    ``calculate_unit_coefficient_4d`` short-circuit before the predicate
    is reached, so on the read-path side they assert the guards.
    """
    assert coefficients_4d_are_learned(regime_coeff) is expected

    coordinator = MagicMock()
    coordinator.is_solar_affected.return_value = True
    coordinator.model.solar_coefficients_4d_per_unit = {
        "sensor.unit_a": {"heating": regime_coeff}
    }
    calc = SolarCalculator(coordinator)
    coeffs = calc.calculate_unit_coefficient_4d("sensor.unit_a", "5", MODE_HEATING)

    fires = any(coeffs[k] for k in ("s", "e", "w", "diffuse"))
    assert fires is expected, "read path disagrees with the readiness predicate"


def test_read_path_zero_vector_is_the_cliff_readiness_guards():
    """An unlearned 4D regime predicts zero solar — 3D would not.

    The premise of condition 2.  If this ever stops being true the gate
    can be relaxed; until then it must stay.
    """
    coordinator = MagicMock()
    coordinator.is_solar_affected.return_value = True
    coordinator.model.solar_coefficients_4d_per_unit = {
        "sensor.unit_a": {"heating": _unlearned()}
    }
    calc = SolarCalculator(coordinator)

    assert calc.calculate_unit_coefficient_4d("sensor.unit_a", "5", MODE_HEATING) == {
        "s": 0.0,
        "e": 0.0,
        "w": 0.0,
        "diffuse": 0.0,
    }


# --------------------------------------------------------------------
# Condition 2: evaluate_4d_learning_readiness
# --------------------------------------------------------------------


def test_all_active_regimes_learned_is_ready():
    result = evaluate_4d_learning_readiness(
        {
            "sensor.a": {"heating": _learned(), "cooling": _unlearned()},
            "sensor.b": {"heating": _learned()},
        },
        {"sensor.a", "sensor.b"},
        {},  # sparse — both default to MODE_HEATING
    )

    assert result["ready"] is True
    assert result["reason"] == "all_active_regimes_learned"
    assert result["entities_evaluated"] == 2
    assert result["entities_not_learned"] == []


def test_one_untrained_entity_blocks_readiness():
    result = evaluate_4d_learning_readiness(
        {
            "sensor.a": {"heating": _learned()},
            "sensor.b": {"heating": _unlearned()},
        },
        {"sensor.a", "sensor.b"},
        {},
    )

    assert result["ready"] is False
    assert result["entities_not_learned"] == ["sensor.b"]
    assert result["entities_learned"] == ["sensor.a"]


def test_entity_missing_from_the_4d_dict_is_not_learned():
    """Absent is not learned — it is the untrained case, not abstention."""
    result = evaluate_4d_learning_readiness(
        {"sensor.a": {"heating": _learned()}},
        {"sensor.a", "sensor.b"},
        {},
    )

    assert result["ready"] is False
    assert result["entities_not_learned"] == ["sensor.b"]
    assert result["entities_abstained"] == []


def test_only_the_currently_active_regime_is_checked():
    """A cooling unit is judged on cooling, not on its untrained heating.

    The accepted limitation: gating on all regimes would never pass for a
    unit that only ever cools.
    """
    result = evaluate_4d_learning_readiness(
        {"sensor.ac": {"heating": _unlearned(), "cooling": _learned()}},
        {"sensor.ac"},
        {"sensor.ac": MODE_COOLING},
    )

    assert result["ready"] is True
    assert result["per_entity"]["sensor.ac"]["regime"] == "cooling"


def test_sparse_mode_map_defaults_to_heating():
    """A plain heating install writes no modes at all.

    Same sparse-map trap as ``regime_energy_split``: an entity absent
    from ``_unit_modes`` is heating, not unknown.
    """
    result = evaluate_4d_learning_readiness(
        {"sensor.a": {"heating": _learned(), "cooling": _unlearned()}},
        {"sensor.a"},
        {},
    )

    assert result["per_entity"]["sensor.a"]["mode"] == MODE_HEATING
    assert result["per_entity"]["sensor.a"]["regime"] == "heating"
    assert result["ready"] is True


@pytest.mark.parametrize("mode", [MODE_OFF, MODE_DHW])
def test_off_and_dhw_abstain_rather_than_block(mode):
    """No live 4D read on these units, so they neither qualify nor block."""
    result = evaluate_4d_learning_readiness(
        {
            "sensor.a": {"heating": _learned()},
            "sensor.idle": {"heating": _unlearned()},
        },
        {"sensor.a", "sensor.idle"},
        {"sensor.idle": mode},
    )

    assert result["ready"] is True
    assert result["entities_abstained"] == ["sensor.idle"]
    assert result["entities_evaluated"] == 1
    assert result["per_entity"]["sensor.idle"]["learned_4d"] is None


def test_all_entities_abstaining_yields_none_not_false():
    """No evidence is not negative evidence — ``None`` != ``False``."""
    result = evaluate_4d_learning_readiness(
        {"sensor.a": {"heating": _unlearned()}},
        {"sensor.a"},
        {"sensor.a": MODE_OFF},
    )

    assert result["ready"] is None
    assert result["reason"] == "no_entities_in_active_regime"


def test_empty_solar_scope_yields_none():
    result = evaluate_4d_learning_readiness({}, set(), {})

    assert result["ready"] is None
    assert result["reason"] == "no_entities_in_solar_scope"


@pytest.mark.parametrize(
    "mode, regime",
    [
        (MODE_GUEST_HEATING, "heating"),
        (MODE_GUEST_COOLING, "cooling"),
    ],
)
def test_guest_modes_route_to_their_regime(mode, regime):
    """Guest units predict, so they are judged on the regime they predict in."""
    result = evaluate_4d_learning_readiness(
        {"sensor.guest": {regime: _learned()}},
        {"sensor.guest"},
        {"sensor.guest": mode},
    )

    assert result["per_entity"]["sensor.guest"]["regime"] == regime
    assert result["ready"] is True


def test_tolerates_missing_state_without_raising():
    """Called from the config flow — must never raise on partial state."""
    result = evaluate_4d_learning_readiness(None, None, None)

    assert result["ready"] is None


# --------------------------------------------------------------------
# The composite gate
# --------------------------------------------------------------------


def _engine(*, source="native", n=100, flag_on=False, coeffs=None, modes=None,
            scope=("sensor.a",)):
    entries = []
    for _ in range(n):
        ts = dt_util.now() - timedelta(days=1)
        entries.append(
            {
                "timestamp": ts.isoformat(),
                "solar_factor": 0.5,
                "dni_dhi_source": source,
            }
        )
    coordinator = MagicMock()
    coordinator._hourly_log = entries
    coordinator.experimental_4d_primary = flag_on
    coordinator._solar_coefficients_4d_per_unit = (
        coeffs if coeffs is not None else {"sensor.a": {"heating": _learned()}}
    )
    coordinator._solar_affected_set = frozenset(scope)
    coordinator._unit_modes = modes or {}
    return DiagnosticsEngine(coordinator)


def test_ready_to_enable_needs_both_halves():
    result = _engine()._compute_4d_readiness(30)

    assert result["ready"] is True
    assert result["verdict"] == "ready_to_enable"
    assert result["input"]["supports_4d_primary"] is True
    assert result["learning"]["ready"] is True


def test_good_input_but_untrained_4d_is_not_ready():
    """The case the input half alone cannot see."""
    result = _engine(
        coeffs={"sensor.a": {"heating": _unlearned()}}
    )._compute_4d_readiness(30)

    assert result["ready"] is False
    assert result["verdict"] == "not_ready"
    assert result["input"]["supports_4d_primary"] is True
    assert result["learning"]["ready"] is False


def test_trained_4d_but_kasten_input_is_not_ready():
    result = _engine(source="kasten_synthetic")._compute_4d_readiness(30)

    assert result["ready"] is False
    assert result["verdict"] == "not_ready"
    assert result["input"]["supports_4d_primary"] is False
    assert result["learning"]["ready"] is True


def test_enabled_and_ready_when_flag_on_and_both_conditions_met():
    result = _engine(flag_on=True)._compute_4d_readiness(30)

    assert result["verdict"] == "enabled_and_ready"


def test_enabled_but_untrained_is_a_live_degradation():
    """Flag on, provider fine, 4D untrained — prediction has no solar."""
    result = _engine(
        flag_on=True, coeffs={"sensor.a": {"heating": _unlearned()}}
    )._compute_4d_readiness(30)

    assert result["verdict"] == "enabled_but_not_ready"
    assert result["ready"] is False


def test_thin_source_data_is_insufficient_not_negative():
    result = _engine(n=5)._compute_4d_readiness(30)

    assert result["ready"] is None
    assert result["verdict"] == "insufficient_data"


def test_undeterminable_learning_half_is_insufficient_not_ready():
    result = _engine(modes={"sensor.a": MODE_OFF})._compute_4d_readiness(30)

    assert result["ready"] is None
    assert result["verdict"] == "insufficient_data"


def test_untrained_4d_outranks_an_unknown_input_half():
    """Three-valued AND: a definitive ``False`` beats an unknown.

    The regression this pins is a live zero-solar degradation going
    unreported.  The learning half needs no history window, so it can
    say "untrained" while the input half is still accumulating its 50
    labelled daylight hours — the first 3-9 days of any install, and of
    *every* install upgrading from a log that predates the
    ``dni_dhi_source`` field.  Collapsing that to ``insufficient_data``
    left the flag on, the read path returning the zero-vector, and every
    surface silent.
    """
    result = _engine(
        n=20,  # below DNI_DHI_SOURCE_MIX_MIN_HOURS → input half is None
        flag_on=True,
        coeffs={"sensor.a": {"heating": _unlearned()}},
    )._compute_4d_readiness(30)

    assert result["input"]["supports_4d_primary"] is None
    assert result["learning"]["ready"] is False
    assert result["ready"] is False
    assert result["verdict"] == "enabled_but_not_ready"


def test_unsupported_input_outranks_an_unknown_learning_half():
    """The mirror case — also decisive, also must not collapse."""
    result = _engine(
        source="kasten_synthetic",
        flag_on=True,
        modes={"sensor.a": MODE_OFF},
    )._compute_4d_readiness(30)

    assert result["input"]["supports_4d_primary"] is False
    assert result["learning"]["ready"] is None
    assert result["ready"] is False
    assert result["verdict"] == "enabled_but_not_ready"


def test_both_halves_unknown_is_the_only_insufficient_case():
    result = _engine(n=5, modes={"sensor.a": MODE_OFF})._compute_4d_readiness(30)

    assert result["ready"] is None
    assert result["verdict"] == "insufficient_data"


# --------------------------------------------------------------------
# Summary verdict (reverses the #1061 decision)
# --------------------------------------------------------------------


def _full_engine(*, source="native", n=100, flag_on=False, coeffs=None, modes=None):
    """An engine complete enough to run ``diagnose_solar`` end-to-end.

    Reuses ``test_solar_diagnose._make_coord`` rather than growing a
    second heavyweight coordinator mock — the summary verdict is the
    thing under test here, not the blocks feeding it.
    """
    from tests.test_solar_diagnose import _make_coord, _hour_entry

    entries = []
    for _ in range(n):
        ts = (dt_util.now() - timedelta(days=1)).isoformat()
        entry = _hour_entry(ts)
        entry["dni_dhi_source"] = source
        entries.append(entry)

    coord = _make_coord(entries)
    coord.experimental_4d_primary = flag_on
    coord._solar_coefficients_4d_per_unit = (
        coeffs if coeffs is not None else {"sensor.heater1": {"heating": _learned()}}
    )
    coord._solar_affected_set = frozenset({"sensor.heater1"})
    coord._unit_modes = modes or {}
    # Battery-feedback sweep is unrelated to readiness and cannot run on a
    # MagicMock coordinator; disable it so it does not raise before the
    # summary is assembled.
    coord.battery_thermal_feedback_k = 0.0
    return DiagnosticsEngine(coord)


def test_ready_to_enable_raises_the_summary_verdict():
    """#1062 reverses #1061: with condition 2 added, this is actionable."""
    result = _full_engine().diagnose_solar(30)

    assert result["summary"]["four_d_readiness"]["verdict"] == "ready_to_enable"
    assert result["summary"]["verdict"] == "review_recommended"


def test_enabled_but_not_ready_raises_the_summary_verdict():
    # In-scope entity id: with a different id the entity would be absent
    # from the dict and the assertion would pass without ever reading the
    # unlearned value.
    result = _full_engine(
        flag_on=True, coeffs={"sensor.heater1": {"heating": _unlearned()}}
    ).diagnose_solar(30)

    assert result["summary"]["four_d_readiness"]["verdict"] == "enabled_but_not_ready"
    assert result["summary"]["verdict"] == "review_recommended"


def test_enabled_and_ready_does_not_raise_on_its_own():
    """A correctly-configured 4D install must not nag."""
    result = _full_engine(flag_on=True).diagnose_solar(30)

    assert result["summary"]["four_d_readiness"]["verdict"] == "enabled_and_ready"
    assert result["summary"]["verdict"] == "no_action_needed"


def test_input_misconfiguration_raises_on_both_conditions():
    """``enabled_but_unsupported`` is kept as defence-in-depth.

    Since the combiner became three-valued, a definitive
    ``supports_4d = False`` always reaches ``enabled_but_not_ready``, so
    the composite subsumes the input-half condition rather than merely
    overlapping it.  Both are asserted here deliberately: they are
    computed from different predicates, and a future change to either
    shape must not be able to silence a live misconfiguration alone.
    """
    result = _full_engine(
        source="kasten_synthetic",
        flag_on=True,
        modes={"sensor.heater1": MODE_OFF},
    ).diagnose_solar(30)

    assert result["summary"]["four_d_readiness"]["verdict"] == "enabled_but_not_ready"
    assert result["summary"]["dni_dhi_source"]["verdict"] == "enabled_but_unsupported"
    assert result["summary"]["verdict"] == "review_recommended"


def test_readiness_block_is_exposed_at_top_level():
    result = _full_engine().diagnose_solar(30)

    assert "four_d_readiness" in result
    assert "accepted_limitation" in result["four_d_readiness"]
    assert result["four_d_readiness"]["learning"]["per_entity"]["sensor.heater1"][
        "regime"
    ] == "heating"
