"""Per-(scope, mode) solar aggregate clamp tests (#992 commit 1).

These tests pin the partition-aware solar aggregation in
``StatisticsManager._calculate_total_power_3d``.  The legacy
``max(0, global_net_after_aux + global_solar_effect)`` clamp had two
failure modes:

* Violation 1 (out-of-scope absorption): an out-of-scope entity
  (cellar, DHW circuit, parasitic load excluded from
  ``solar_affected_entities``) saw its base demand silently absorbed
  on sunny hours when in-scope heating units saturated.  Empirical
  signature: ``global_net = 0.0`` but ``unit_sum_net = 0.014``.
* Violation 3 (mixed-mode within scope): when both heating and
  cooling units are in ``solar_affected_entities`` on the same hour,
  heating-side solar reduction (negative) and cooling-side base
  (positive) were summed before clamping at 0 — letting heating
  solar overshoot absorb cooling base.

Design under test (per CLAUDE.md invariants #1, #6):

* Track A (``global_net_after_aux``) remains the magnitude anchor —
  the global aux flow is preserved here (commit 2 fixes aux scope
  leak separately).
* The anchor is allocated across three partitions by per-entity
  ``net_after_aux`` share: heating-in-scope, cooling-in-scope,
  not-in-scope.
* Solar is applied with per-partition semantics: heating-in-scope
  clamps at 0; cooling-in-scope adds (no clamp on the add side);
  not-in-scope passes through.
"""
from unittest.mock import MagicMock

from custom_components.heating_analytics.const import (
    COOLING_WIND_BUCKET,
    MODE_COOLING,
    MODE_DHW,
    MODE_HEATING,
    MODE_OFF,
)
from custom_components.heating_analytics.statistics import StatisticsManager


def _make_coordinator_stub(
    *,
    energy_sensors,
    unit_modes,
    solar_affected_set,
    per_unit_base,        # {entity_id: base_kwh}
    per_unit_solar,       # {entity_id: solar_potential_kwh}
    global_base,
    global_aux_reduction=0.0,
    is_aux_active=False,
    aux_affected_set=None,
):
    """Build a minimal coordinator MagicMock for calculate_total_power.

    The mock pins the per-entity base / solar / aux predictions so the
    per-(scope, mode) partition logic is the only behavior under test.
    """
    coord = MagicMock()
    coord.balance_point = 15.0
    coord.solar_enabled = True
    coord.solar_azimuth = 245.0
    coord.solar_correction_percent = 100.0
    coord.screen_config = (False, False, False)
    coord.energy_sensors = list(energy_sensors)
    coord.aux_affected_entities = list(aux_affected_set or [])
    coord.aux_affected_set = set(aux_affected_set or [])
    coord.auxiliary_heating_active = is_aux_active
    coord.get_unit_mode = lambda eid: unit_modes.get(eid, MODE_HEATING)
    coord.screen_config_for_entity = lambda eid: (False, False, False)
    coord._solar_carryover_state = 0.0
    coord.solar_battery_decay = 0.80
    coord.data = {"solar_factor": 0.7}
    # Scope set — the new partition logic reads this attribute.
    coord._solar_affected_set = frozenset(solar_affected_set or energy_sensors)

    # Build per-unit correlation_data so _get_prediction_from_model
    # finds an exact-temp hit at temp=0.  Cooling-mode entities land
    # in the COOLING_WIND_BUCKET; heating in "normal".
    correlation_per_unit = {}
    for eid in energy_sensors:
        mode = unit_modes.get(eid, MODE_HEATING)
        bucket = COOLING_WIND_BUCKET if mode == MODE_COOLING else "normal"
        correlation_per_unit[eid] = {"0": {bucket: per_unit_base[eid]}}
    coord.model.correlation_data_per_unit = correlation_per_unit
    coord.model.aux_coefficients_per_unit = {}
    # solar_coefficients_per_unit needs a "learned" entry for cooling
    # units so the cold-start guard releases.
    sol_per_unit = {}
    for eid in energy_sensors:
        mode = unit_modes.get(eid, MODE_HEATING)
        regime = "cooling" if mode == MODE_COOLING else "heating"
        sol_per_unit[eid] = {regime: {"s": 0.0, "e": 0.0, "w": 0.0, "learned": True}}
    coord.model.solar_coefficients_per_unit = sol_per_unit

    coord._get_predicted_kwh = MagicMock(return_value=global_base)
    coord.solar.calculate_unit_coefficient = MagicMock(
        return_value={"s": 0.0, "e": 0.0, "w": 0.0}
    )
    # Per-entity solar impact pinned by the mapping.
    def _impact(pot_vec, coeff, entity_id=None):
        # The production call passes (potential_vector, unit_coeff)
        # positionally; we cannot see entity_id directly, so we route
        # via a side_effect closure that tracks call order.
        raise RuntimeError("use side_effect_factory below")
    coord.solar.calculate_unit_solar_impact = MagicMock()

    # Saturation: heating subtracts and clamps at 0; cooling adds;
    # OFF forces net to 0 (matches production solar.calculate_saturation
    # MODE_OFF branch — entity is commanded off, no consumption
    # predicted).
    def _saturation(net_demand, solar_potential, mode):
        if mode == MODE_COOLING:
            return (solar_potential, 0.0, net_demand + solar_potential)
        if mode == MODE_OFF:
            return (0.0, 0.0, 0.0)
        # Heating / fallthrough: solar reduces demand, clamped at 0.
        applied = min(solar_potential, net_demand)
        wasted = max(0.0, solar_potential - net_demand)
        return (applied, wasted, max(0.0, net_demand - solar_potential))
    coord.solar.calculate_saturation = MagicMock(side_effect=_saturation)

    return coord


def _wire_per_entity_impact(coord, per_unit_solar, energy_sensors):
    """Make calculate_unit_solar_impact return per-entity values.

    Iteration order of ``energy_sensors`` in ``calculate_total_power``
    matches the list ordering, so we drive the mock via side_effect.
    """
    # Per-entity sequence following energy_sensors order.
    values = [per_unit_solar.get(eid, 0.0) for eid in energy_sensors]
    coord.solar.calculate_unit_solar_impact.side_effect = values


class TestOutOfScopePreservation:
    """Violation 1: out-of-scope base must not be absorbed by in-scope
    solar saturation.
    """

    def test_cellar_base_preserved_when_in_scope_units_saturate(self):
        """Empirical signature: VP_A + VP_B in scope, cellar out of scope.
        Strong sun saturates both VPs to 0; cellar (out of scope) base
        of 0.014 must flow through to ``total_kwh``.
        """
        entities = ["hp.vp_a", "hp.vp_b", "sensor.cellar"]
        unit_modes = {
            "hp.vp_a": MODE_HEATING,
            "hp.vp_b": MODE_HEATING,
            "sensor.cellar": MODE_HEATING,
        }
        # Per-entity bases: VPs 5 each, cellar 0.014.
        per_unit_base = {"hp.vp_a": 5.0, "hp.vp_b": 5.0, "sensor.cellar": 0.014}
        # Strong solar; each VP saturates at its 5.0 base.  Cellar
        # solar would also be computed but is harmless (=0 here).
        per_unit_solar = {"hp.vp_a": 8.0, "hp.vp_b": 8.0, "sensor.cellar": 0.0}

        coord = _make_coordinator_stub(
            energy_sensors=entities,
            unit_modes=unit_modes,
            solar_affected_set={"hp.vp_a", "hp.vp_b"},
            per_unit_base=per_unit_base,
            per_unit_solar=per_unit_solar,
            # Track A under-predicts vs unit-sum (10.014) — the
            # signature of the field bug.
            global_base=7.0,
        )
        _wire_per_entity_impact(coord, per_unit_solar, entities)

        stats = StatisticsManager(coord)
        result = stats.calculate_total_power(
            temp=0.0,
            effective_wind=2.0,
            solar_impact=0.0,
            is_aux_active=False,
            override_solar_factor=0.7,
            override_solar_vector=(0.5, 0.2, 0.3),
            detailed=True,
        )

        # Pre-fix: max(0, 7 + (0 − 10)) = 0.  Cellar absorbed.
        # Post-fix: heating_in_scope clamps at 0, cellar's anchor share
        # passes through.  Cellar's share of the global anchor is
        # 0.014 / 10.014 ≈ 0.0014 of the 7.0 anchor ≈ 0.0098.
        # The exact value depends on the proportional split; we
        # assert it is strictly > 0 (the bug manifests as total=0).
        assert result["total_kwh"] > 0.0, (
            f"Cellar base must not be absorbed by in-scope solar; "
            f"total_kwh={result['total_kwh']}"
        )
        # The cellar's per-unit net stays at its base (no solar
        # applied to an out-of-scope unit with 0 solar potential).
        cellar_net = result["unit_breakdown"]["sensor.cellar"]["net_kwh"]
        assert cellar_net == 0.014


class TestMixedModeWithinScope:
    """Violation 3: heating-side solar must not cancel cooling-side base
    on mixed-mode hours within ``solar_affected_entities``.
    """

    def test_heating_solar_does_not_absorb_cooling_base(self):
        """One heating unit + one cooling unit, both in
        ``solar_affected_entities``.  Heating unit oversized for the
        solar input (saturates); cooling unit has additive base.
        Pre-fix: ``global_solar_effect = cooling_applied − heating_applied``
        could be negative enough that the cooling base got absorbed by
        the global clamp.  Post-fix: partitions are independent.
        """
        entities = ["hp.heating", "hp.cooling"]
        unit_modes = {"hp.heating": MODE_HEATING, "hp.cooling": MODE_COOLING}
        per_unit_base = {"hp.heating": 3.0, "hp.cooling": 1.5}
        # Strong south sun: heating unit saturates (solar=5 against
        # base 3).  Cooling unit also gets solar (additive, +0.5).
        per_unit_solar = {"hp.heating": 5.0, "hp.cooling": 0.5}

        coord = _make_coordinator_stub(
            energy_sensors=entities,
            unit_modes=unit_modes,
            solar_affected_set={"hp.heating", "hp.cooling"},
            per_unit_base=per_unit_base,
            per_unit_solar=per_unit_solar,
            global_base=4.5,  # matches sum
        )
        _wire_per_entity_impact(coord, per_unit_solar, entities)

        stats = StatisticsManager(coord)
        result = stats.calculate_total_power(
            temp=0.0,
            effective_wind=2.0,
            solar_impact=0.0,
            is_aux_active=False,
            override_solar_factor=0.7,
            override_solar_vector=(0.5, 0.2, 0.3),
            detailed=True,
        )

        # Cooling base + cooling-solar gain must be retained in
        # total_kwh — heating-side solar overshoot doesn't cancel it.
        # Expected partitions:
        #   heating share = 3.0 / 4.5 = 0.667, anchor = 3.0
        #     → heating_in_scope_net = max(0, 3 - 5) = 0
        #   cooling share = 1.5 / 4.5 = 0.333, anchor = 1.5
        #     → cooling_in_scope_net = 1.5 + 0.5 = 2.0
        #   total = 0 + 2.0 + 0 = 2.0
        assert result["total_kwh"] >= 1.5, (
            f"Cooling base + cooling solar must survive heating "
            f"saturation; total_kwh={result['total_kwh']}"
        )
        # Cooling unit breakdown still includes the solar addition.
        cooling_net = result["unit_breakdown"]["hp.cooling"]["net_kwh"]
        assert cooling_net == 2.0


class TestLegacyParity:
    """Legacy single-mode all-in-scope installs must match the per-entity
    clamped-net sum within float tolerance.
    """

    def test_single_mode_all_in_scope_matches_unit_sum(self):
        """All heating, all in scope (= legacy default).  When Track A
        agrees with Track B sum, ``total_kwh ≈ unit_sum_net`` and
        ``unspecified_kwh ≈ 0``.
        """
        entities = ["hp.a", "hp.b", "hp.c"]
        unit_modes = {e: MODE_HEATING for e in entities}
        per_unit_base = {"hp.a": 2.0, "hp.b": 1.5, "hp.c": 1.0}
        per_unit_solar = {"hp.a": 0.5, "hp.b": 0.3, "hp.c": 0.1}

        coord = _make_coordinator_stub(
            energy_sensors=entities,
            unit_modes=unit_modes,
            # solar_affected_set=None → defaults to all energy_sensors.
            solar_affected_set=set(entities),
            per_unit_base=per_unit_base,
            per_unit_solar=per_unit_solar,
            global_base=4.5,  # matches Σ per-entity base
        )
        _wire_per_entity_impact(coord, per_unit_solar, entities)

        stats = StatisticsManager(coord)
        result = stats.calculate_total_power(
            temp=0.0,
            effective_wind=2.0,
            solar_impact=0.0,
            is_aux_active=False,
            override_solar_factor=0.7,
            override_solar_vector=(0.5, 0.2, 0.3),
            detailed=True,
        )

        # Per-entity: net = base - applied_solar
        # hp.a: 2.0 - 0.5 = 1.5; hp.b: 1.2; hp.c: 0.9 → sum = 3.6
        expected_sum = (2.0 - 0.5) + (1.5 - 0.3) + (1.0 - 0.1)
        assert abs(result["total_kwh"] - expected_sum) < 1e-3
        assert abs(result["breakdown"]["unspecified_kwh"]) < 1e-3


class TestAuxScopePreservation:
    """Violation 2 (#992 commit 2): out-of-aux-scope base must not be
    absorbed by Track A's ``max(0, global_base − global_aux_reduction)``
    formula.  Post-c2 ``total_kwh`` comes from the per-(scope, mode)
    partition sum, so an entity outside ``aux_affected_entities`` has
    its base flow through to the total regardless of Track A's global
    aux belief.
    """

    def test_out_of_aux_scope_base_preserved_when_global_aux_overshoots(self):
        """Empirical signature parallel to the solar cellar case:
        h1 + h2 in aux_affected (base=10 each, per-entity aux=2 each).
        sensor.cellar (base=10) NOT in aux_affected.
        Track A global_aux = 10 (over-estimates per-entity sum=4).

        Pre-c2: total_kwh = max(0, 30 − 10) = 20.  Cellar absorbed.
        Post-c2: total_kwh = 8 + 8 + 10 = 26.  Cellar preserved.
        """
        entities = ["hp.h1", "hp.h2", "sensor.cellar"]
        unit_modes = {e: MODE_HEATING for e in entities}
        per_unit_base = {"hp.h1": 10.0, "hp.h2": 10.0, "sensor.cellar": 10.0}
        per_unit_solar = {e: 0.0 for e in entities}
        per_unit_aux = {"hp.h1": 2.0, "hp.h2": 2.0, "sensor.cellar": 0.0}

        coord = _make_coordinator_stub(
            energy_sensors=entities,
            unit_modes=unit_modes,
            solar_affected_set=set(entities),
            per_unit_base=per_unit_base,
            per_unit_solar=per_unit_solar,
            global_base=30.0,
            global_aux_reduction=10.0,
            is_aux_active=True,
            aux_affected_set={"hp.h1", "hp.h2"},
        )
        _wire_per_entity_impact(coord, per_unit_solar, entities)
        # Per-entity aux: only h1, h2 are aux-affected — cellar's aux
        # is 0 even if listed.  Build aux_coefficients_per_unit so the
        # _get_prediction_from_model lookup in pass 2 returns the
        # per-entity values.
        aux_per_unit = {
            eid: {"0": {"normal": per_unit_aux[eid]}} for eid in entities
        }
        coord.model.aux_coefficients_per_unit = aux_per_unit
        # Track A aux bucket — read in calculate_total_power line ~428
        # via ``_get_prediction_from_model(self.coordinator.model.aux_coefficients, ...)``
        # to populate ``global_aux_reduction``.  Set it to 10.0 to
        # match the over-estimate scenario.
        coord.model.aux_coefficients = {"0": {"normal": 10.0}}

        stats = StatisticsManager(coord)
        result = stats.calculate_total_power(
            temp=0.0,
            effective_wind=2.0,
            solar_impact=0.0,
            is_aux_active=True,
            override_solar_factor=0.0,
            override_solar_vector=(0.0, 0.0, 0.0),
            detailed=True,
        )

        # Per-(scope, mode) partition sum:
        #   h1: net_after_aux = 10 - 2 = 8
        #   h2: 8
        #   cellar: 10 (no aux applied — out of aux scope)
        #   sum = 26
        assert result["total_kwh"] == 26.0, (
            f"Out-of-aux-scope base must not be absorbed by Track A's "
            f"global aux clamp; total_kwh={result['total_kwh']}"
        )
        # Cellar's net stays at base.
        assert result["unit_breakdown"]["sensor.cellar"]["net_kwh"] == 10.0
        assert result["unit_breakdown"]["sensor.cellar"]["aux_reduction_kwh"] == 0.0
        # Track A's gap surfaces as orphaned_aux_savings (diagnostic only).
        # global_aux=10 − per-entity sum=4 = 6.
        assert result["breakdown"]["orphaned_aux_savings"] == 6.0


class TestOffModeExclusion:
    """Regression for #992 c5: MODE_OFF entities had their base flow
    into ``base_not_in_scope`` (commit 2 partition logic), but
    ``solar.calculate_saturation`` overrides ``final_net = 0`` for OFF
    entities — the entity is commanded off, so we predict no consumption.
    Including OFF base in the partition inflated ``total_kwh`` past the
    per-entity sum and surfaced as ``unspecified_kwh`` per OFF entity.
    """

    def test_off_entity_excluded_from_partition_and_unit_sum(self):
        """One heating + one OFF.  OFF's base predicted at 1.0 but
        ``calculate_saturation(MODE_OFF)`` returns ``net_final=0``.
        Post-c5: ``total_kwh = heating only`` and ``unspecified_kwh ≈ 0``.
        Pre-c5: ``total_kwh`` included OFF base; gap shown as
        ``unspecified_kwh ≈ 1.0``.
        """
        entities = ["hp.heater", "hp.off_unit"]
        unit_modes = {"hp.heater": MODE_HEATING, "hp.off_unit": MODE_OFF}
        per_unit_base = {"hp.heater": 3.0, "hp.off_unit": 1.0}
        per_unit_solar = {e: 0.0 for e in entities}

        coord = _make_coordinator_stub(
            energy_sensors=entities,
            unit_modes=unit_modes,
            solar_affected_set=set(entities),
            per_unit_base=per_unit_base,
            per_unit_solar=per_unit_solar,
            global_base=4.0,
        )
        _wire_per_entity_impact(coord, per_unit_solar, entities)

        stats = StatisticsManager(coord)
        result = stats.calculate_total_power(
            temp=0.0,
            effective_wind=2.0,
            solar_impact=0.0,
            is_aux_active=False,
            override_solar_factor=0.0,
            override_solar_vector=(0.0, 0.0, 0.0),
            detailed=True,
        )

        # heating net = 3.0 (no solar), off net = 0 (forced by saturation)
        # total = 3.0 (OFF excluded from both partition and unit_sum)
        assert result["total_kwh"] == 3.0
        assert result["unit_breakdown"]["hp.off_unit"]["net_kwh"] == 0.0
        # unspecified_kwh ≈ 0 — partition and per-entity agree.
        assert abs(result["breakdown"]["unspecified_kwh"]) < 1e-3


class TestOffDhwExclusion:
    """OFF / DHW entities listed in ``solar_affected_entities`` go to
    the not-in-scope partition (their mode has no solar semantics).
    """

    def test_dhw_entity_passes_through_untouched(self):
        """DHW unit in solar_affected_set: its mode is MODE_DHW so
        ``calculate_saturation`` falls through (solar_applied=0,
        net_final=base).  The partition routes it to not-in-scope so
        in-scope heating saturation cannot absorb its base.
        """
        entities = ["hp.heating", "hp.dhw"]
        unit_modes = {"hp.heating": MODE_HEATING, "hp.dhw": MODE_DHW}
        per_unit_base = {"hp.heating": 3.0, "hp.dhw": 0.5}
        per_unit_solar = {"hp.heating": 5.0, "hp.dhw": 0.0}

        coord = _make_coordinator_stub(
            energy_sensors=entities,
            unit_modes=unit_modes,
            # Both listed in solar_affected, but mode gates DHW out.
            solar_affected_set={"hp.heating", "hp.dhw"},
            per_unit_base=per_unit_base,
            per_unit_solar=per_unit_solar,
            global_base=3.5,
        )
        _wire_per_entity_impact(coord, per_unit_solar, entities)

        stats = StatisticsManager(coord)
        result = stats.calculate_total_power(
            temp=0.0,
            effective_wind=2.0,
            solar_impact=0.0,
            is_aux_active=False,
            override_solar_factor=0.7,
            override_solar_vector=(0.5, 0.2, 0.3),
            detailed=True,
        )

        # DHW's anchor share = 0.5 / 3.5 ≈ 0.143; anchor = 3.5*0.143 = 0.5
        # heating share = 3.0 / 3.5; anchor = 3.0 → clamps to 0
        # total ≈ 0.5 (DHW preserved)
        assert result["total_kwh"] > 0.4, (
            f"DHW base must pass through partition; "
            f"total_kwh={result['total_kwh']}"
        )
        # DHW unit_breakdown: solar_applied=0, net=base.
        dhw_data = result["unit_breakdown"]["hp.dhw"]
        assert dhw_data["solar_reduction_kwh"] == 0.0
        assert dhw_data["net_kwh"] == 0.5
