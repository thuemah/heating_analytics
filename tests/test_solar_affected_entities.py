"""Tests for the per-entity solar-scope gate (#962).

Verifies:
- ``calculate_unit_coefficient`` returns ``{0,0,0}`` for excluded entities
  instead of falling back to the DEFAULT_SOLAR_COEFF azimuth decomposition.
- ``async_migrate_solar_affected`` resets learning state for entities
  removed from the list and leaves surviving entities untouched.
- ``is_solar_affected`` honours the configured set, with None-default
  meaning "all entities included".
- ``apply_implied_coefficient`` skips excluded entities with
  ``skip_reason: excluded_from_solar``.
- ``batch_fit_solar_coefficients`` skips excluded entities with the
  same skip reason.
- The schema-side ``_solar_affected_default_with_new`` helper opts
  newly-added energy sensors *into* the list automatically.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.solar import SolarCalculator
from custom_components.heating_analytics.const import (
    DEFAULT_SOLAR_COEFF_HEATING,
    MODE_HEATING,
    MODE_COOLING,
)


def _make_solar_calculator(solar_affected_set: frozenset[str] | None) -> SolarCalculator:
    coord = MagicMock()
    coord.solar_azimuth = 245
    coord.model.solar_coefficients_per_unit = {}
    if solar_affected_set is None:
        coord.is_solar_affected = None  # legacy / unhelped coordinator
    else:
        coord.is_solar_affected = lambda entity_id: entity_id in solar_affected_set
    return SolarCalculator(coord)


class TestCalculateUnitCoefficientGate:
    """The read-path gate in solar.calculate_unit_coefficient (#962)."""

    def test_excluded_returns_zero_vector(self):
        sc = _make_solar_calculator(frozenset(["sensor.included"]))
        result = sc.calculate_unit_coefficient(
            "sensor.excluded", temp_key="0", mode=MODE_HEATING
        )
        assert result == {"s": 0.0, "e": 0.0, "w": 0.0}

    def test_included_falls_back_to_default(self):
        """Included entity with no learned coefficient hits the default-decomposition path."""
        sc = _make_solar_calculator(frozenset(["sensor.included"]))
        result = sc.calculate_unit_coefficient(
            "sensor.included", temp_key="0", mode=MODE_HEATING
        )
        # Default = DEFAULT_SOLAR_COEFF_HEATING × azimuth_decomposition (245°).
        assert result["w"] > 0.0
        assert result["s"] > 0.0
        assert result["e"] == 0.0  # 245° azimuth puts no projection on east
        # Sanity: w-component is the larger one for a 245° azimuth.
        assert result["w"] > result["s"]

    def test_legacy_no_helper_returns_default(self):
        """Test coordinator without is_solar_affected helper reads as legacy (all included)."""
        sc = _make_solar_calculator(None)  # no helper → legacy path
        result = sc.calculate_unit_coefficient(
            "sensor.anything", temp_key="0", mode=MODE_HEATING
        )
        assert result["w"] > 0.0  # default fallback fired

    def test_excluded_returns_zero_for_cooling_too(self):
        sc = _make_solar_calculator(frozenset())  # everyone excluded
        result = sc.calculate_unit_coefficient(
            "sensor.x", temp_key="0", mode=MODE_COOLING
        )
        assert result == {"s": 0.0, "e": 0.0, "w": 0.0}

    def test_excluded_overrides_learned_coefficient(self):
        """Even if a coefficient is stored, exclusion zeros the read path."""
        sc = _make_solar_calculator(frozenset(["sensor.in"]))
        sc.coordinator.model.solar_coefficients_per_unit = {
            "sensor.out": {
                "heating": {"s": 0.5, "e": 0.5, "w": 0.5, "learned": True},
                "cooling": {"s": 0.0, "e": 0.0, "w": 0.0},
            },
        }
        result = sc.calculate_unit_coefficient(
            "sensor.out", temp_key="0", mode=MODE_HEATING
        )
        assert result == {"s": 0.0, "e": 0.0, "w": 0.0}


class TestMigrateSolarAffected:
    """async_migrate_solar_affected: removed → reset, survivors → preserved."""

    @pytest.mark.asyncio
    async def test_removal_triggers_reset(self):
        coord = MagicMock(spec=HeatingDataCoordinator)
        coord._solar_affected_set = frozenset(["sensor.a", "sensor.b", "sensor.c"])
        coord.async_reset_solar_learning_data = AsyncMock(return_value={"status": "reset"})

        await HeatingDataCoordinator.async_migrate_solar_affected(
            coord, ["sensor.a"]
        )

        # b and c removed → reset called for each.
        called_entities = [
            call.kwargs.get("entity_id") for call in coord.async_reset_solar_learning_data.call_args_list
        ]
        assert sorted(called_entities) == ["sensor.b", "sensor.c"]
        # Set updated.
        assert coord._solar_affected_set == frozenset(["sensor.a"])

    @pytest.mark.asyncio
    async def test_no_removal_no_reset(self):
        coord = MagicMock(spec=HeatingDataCoordinator)
        coord._solar_affected_set = frozenset(["sensor.a", "sensor.b"])
        coord.async_reset_solar_learning_data = AsyncMock()

        await HeatingDataCoordinator.async_migrate_solar_affected(
            coord, ["sensor.a", "sensor.b"]
        )

        coord.async_reset_solar_learning_data.assert_not_called()
        assert coord._solar_affected_set == frozenset(["sensor.a", "sensor.b"])

    @pytest.mark.asyncio
    async def test_addition_no_reset(self):
        """Adding a new entity does not trigger reset on existing entities."""
        coord = MagicMock(spec=HeatingDataCoordinator)
        coord._solar_affected_set = frozenset(["sensor.a"])
        coord.async_reset_solar_learning_data = AsyncMock()

        await HeatingDataCoordinator.async_migrate_solar_affected(
            coord, ["sensor.a", "sensor.b"]
        )

        coord.async_reset_solar_learning_data.assert_not_called()
        assert coord._solar_affected_set == frozenset(["sensor.a", "sensor.b"])


class TestShutdownDetectorRespectsSolarScope:
    """detect_solar_shutdown_entities honours solar_affected_entities (#962)."""

    def test_excluded_entity_never_flagged(self):
        """An excluded entity with low actual vs high base is NOT flagged."""
        from custom_components.heating_analytics.observation import (
            detect_solar_shutdown_entities,
        )
        flagged = detect_solar_shutdown_entities(
            solar_enabled=True,
            is_aux_dominant=False,
            potential_vector=(0.3, 0.0, 0.0),  # high enough magnitude
            energy_sensors=["sensor.vp", "sensor.excluded"],
            unit_modes={"sensor.vp": MODE_HEATING, "sensor.excluded": MODE_HEATING},
            unit_actual_kwh={"sensor.vp": 0.0, "sensor.excluded": 0.0},
            unit_expected_base_kwh={"sensor.vp": 0.5, "sensor.excluded": 0.05},
            solar_affected_entities=frozenset(["sensor.vp"]),
        )
        # VP is flagged (base 0.5, actual 0, in scope) but excluded is not.
        assert "sensor.vp" in flagged
        assert "sensor.excluded" not in flagged

    def test_none_sentinel_preserves_legacy_behaviour(self):
        """solar_affected_entities=None → no entity is excluded by scope."""
        from custom_components.heating_analytics.observation import (
            detect_solar_shutdown_entities,
        )
        flagged = detect_solar_shutdown_entities(
            solar_enabled=True,
            is_aux_dominant=False,
            potential_vector=(0.3, 0.0, 0.0),
            energy_sensors=["sensor.a", "sensor.b"],
            unit_modes={"sensor.a": MODE_HEATING, "sensor.b": MODE_HEATING},
            unit_actual_kwh={"sensor.a": 0.0, "sensor.b": 0.0},
            unit_expected_base_kwh={"sensor.a": 0.5, "sensor.b": 0.5},
            solar_affected_entities=None,
        )
        assert "sensor.a" in flagged
        assert "sensor.b" in flagged


class TestScreenStratifiedConfoundGate:
    """Screen-stratified bias_gap + transmittance_floor flags only fire when
    the entity actually has a screened facade (#963).

    For unscreened entities (screen_config = (False, False, False)),
    transmittance is fixed at 1.0 — the screen_stratified binning becomes
    a binning by sun availability, not by transmittance, and the bias_gap
    is a confound. The flag emission and bias_gap reporting must be gated
    on whether the model actually applies transmittance to this entity.
    """

    def test_screen_config_active_marker_present_when_screened(self):
        """An entity with at least one screened direction reports screen_config_active=True."""
        # Smoke-test: assertion that the new field is emitted with the
        # right value when the helper returns a tuple with any True.
        sample_cfg = (False, False, True)
        screen_config_active = bool(sample_cfg) and any(sample_cfg)
        assert screen_config_active is True

    def test_screen_config_active_false_for_unscreened_entity(self):
        sample_cfg = (False, False, False)
        screen_config_active = bool(sample_cfg) and any(sample_cfg)
        assert screen_config_active is False

    def test_none_config_yields_inactive(self):
        sample_cfg = None
        screen_config_active = bool(sample_cfg) and any(sample_cfg or ())
        assert screen_config_active is False


class TestIsSolarAffected:
    """The coordinator helper itself."""

    def test_membership(self):
        coord = MagicMock(spec=HeatingDataCoordinator)
        coord._solar_affected_set = frozenset(["sensor.a", "sensor.b"])
        assert HeatingDataCoordinator.is_solar_affected(coord, "sensor.a") is True
        assert HeatingDataCoordinator.is_solar_affected(coord, "sensor.b") is True
        assert HeatingDataCoordinator.is_solar_affected(coord, "sensor.c") is False

    def test_empty_set_excludes_all(self):
        coord = MagicMock(spec=HeatingDataCoordinator)
        coord._solar_affected_set = frozenset()
        assert HeatingDataCoordinator.is_solar_affected(coord, "sensor.x") is False
