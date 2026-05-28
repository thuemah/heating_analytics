"""Tests for ``StatisticsManager._resolve_entity_net`` (#996).

Single source of truth for the per-entity pass-2 logic shared between
``calculate_total_power`` (3D), ``calculate_total_power_4d``, and
``_calculate_fallback_projection``.  Verifies aux clamping, overflow
accounting, and mode-aware solar saturation dispatch.
"""
from unittest.mock import MagicMock

import pytest

from custom_components.heating_analytics.const import (
    MODE_COOLING,
    MODE_DHW,
    MODE_HEATING,
    MODE_OFF,
)
from custom_components.heating_analytics.statistics import StatisticsManager


def _stats():
    """StatisticsManager with a real-shape calculate_saturation mock."""
    coord = MagicMock()

    def _calculate_saturation(net_after_aux, solar, mode):
        if mode == MODE_OFF:
            return 0.0, 0.0, 0.0
        if mode == MODE_HEATING:
            applied = min(solar, net_after_aux)
            wasted = max(0.0, solar - net_after_aux)
            return applied, wasted, net_after_aux - applied
        if mode == MODE_COOLING:
            return solar, 0.0, net_after_aux + solar
        return 0.0, 0.0, net_after_aux

    coord.solar.calculate_saturation.side_effect = _calculate_saturation
    return StatisticsManager(coord)


class TestAuxClamping:
    def test_aux_within_base_passes_through(self):
        applied, overflow, net_after_aux, _, _, _ = _stats()._resolve_entity_net(
            base_kwh=2.0, raw_aux_kwh=0.5, aux_affected=True,
            solar_kwh=0.0, mode=MODE_HEATING,
        )
        assert applied == 0.5
        assert overflow == 0.0
        assert net_after_aux == 1.5

    def test_aux_exceeds_base_clamps_and_overflows(self):
        applied, overflow, net_after_aux, _, _, _ = _stats()._resolve_entity_net(
            base_kwh=1.0, raw_aux_kwh=2.5, aux_affected=True,
            solar_kwh=0.0, mode=MODE_HEATING,
        )
        assert applied == 1.0
        assert overflow == 1.5
        assert net_after_aux == 0.0

    def test_unaffected_entity_ignores_aux(self):
        applied, overflow, _, _, _, _ = _stats()._resolve_entity_net(
            base_kwh=2.0, raw_aux_kwh=0.5, aux_affected=False,
            solar_kwh=0.0, mode=MODE_HEATING,
        )
        assert applied == 0.0
        assert overflow == 0.0


class TestSolarSaturationDispatch:
    def test_heating_solar_clamped_to_net_after_aux(self):
        # base=2, aux=0.5, net_after_aux=1.5, solar=5.0 → applied=1.5, wasted=3.5
        _, _, _, applied, wasted, net_final = _stats()._resolve_entity_net(
            base_kwh=2.0, raw_aux_kwh=0.5, aux_affected=True,
            solar_kwh=5.0, mode=MODE_HEATING,
        )
        assert applied == 1.5
        assert wasted == 3.5
        assert net_final == 0.0

    def test_cooling_solar_adds_to_net(self):
        _, _, _, applied, wasted, net_final = _stats()._resolve_entity_net(
            base_kwh=1.0, raw_aux_kwh=0.0, aux_affected=False,
            solar_kwh=2.0, mode=MODE_COOLING,
        )
        assert applied == 2.0
        assert wasted == 0.0
        assert net_final == 3.0

    def test_off_returns_zero_regardless_of_inputs(self):
        """MODE_OFF override — commanded off, predict no consumption."""
        _, _, _, applied, wasted, net_final = _stats()._resolve_entity_net(
            base_kwh=2.0, raw_aux_kwh=0.0, aux_affected=False,
            solar_kwh=5.0, mode=MODE_OFF,
        )
        assert net_final == 0.0
        assert applied == 0.0
        assert wasted == 0.0

    def test_dhw_falls_through_with_no_solar_effect(self):
        _, _, _, applied, wasted, net_final = _stats()._resolve_entity_net(
            base_kwh=1.5, raw_aux_kwh=0.0, aux_affected=False,
            solar_kwh=3.0, mode=MODE_DHW,
        )
        # Fall-through: solar has no effect, net_final = net_after_aux = base.
        assert applied == 0.0
        assert wasted == 0.0
        assert net_final == 1.5


def test_aux_and_solar_compose_correctly():
    """Aux subtraction happens BEFORE solar saturation."""
    # base=3, aux=1 → net_after_aux=2. Solar=1.5 (heating) → applied=1.5,
    # net_final = 2 - 1.5 = 0.5.
    applied_aux, _, net_after_aux, solar_applied, _, net_final = (
        _stats()._resolve_entity_net(
            base_kwh=3.0, raw_aux_kwh=1.0, aux_affected=True,
            solar_kwh=1.5, mode=MODE_HEATING,
        )
    )
    assert applied_aux == 1.0
    assert net_after_aux == 2.0
    assert solar_applied == 1.5
    assert net_final == pytest.approx(0.5)
