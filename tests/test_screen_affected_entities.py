"""Tests for per-entity screen scope (screen_affected_entities).

A unit in a zone without external screens should not absorb an average
transmittance it never physically experiences.  The `screen_affected_entities`
config list narrows the installation-level `screen_config` to a subset of
entities; others are treated as screen-independent (transmittance=1.0).

Covers:
1. `coordinator.screen_config_for_entity` routing — in-list vs not-in-list.
2. Learning-side routing — an unscreened entity's NLMS update uses the
   un-attenuated effective vector as its potential.
3. Retrain-replay-side routing — same behaviour under `replay_solar_nlms`.
4. Inequality path scope (D1c): unscreened entities skip the inequality
   learner because the coordinator battery is installation-level and
   would under-lift their coefficient if used.
"""
from unittest.mock import MagicMock

import pytest

from custom_components.heating_analytics.learning import LearningManager
from custom_components.heating_analytics.solar import SolarCalculator


class TestScreenConfigForEntity:
    """Coordinator helper returns effective screen_config per entity."""

    def _make_coord(self, screen_config, affected_set):
        """Minimal coordinator stub exposing the helper under test."""
        from custom_components.heating_analytics.coordinator import (
            HeatingDataCoordinator,
        )
        coord = HeatingDataCoordinator.__new__(HeatingDataCoordinator)
        coord.screen_config = screen_config
        coord._screen_affected_set = frozenset(affected_set)
        return coord

    def test_entity_in_set_returns_global_config(self):
        coord = self._make_coord((True, True, True), {"hp.a"})
        assert coord.screen_config_for_entity("hp.a") == (True, True, True)

    def test_entity_not_in_set_returns_all_false(self):
        coord = self._make_coord((True, True, True), {"hp.a"})
        assert coord.screen_config_for_entity("hp.b") == (False, False, False)

    def test_empty_set_excludes_all(self):
        coord = self._make_coord((True, True, True), set())
        assert coord.screen_config_for_entity("hp.a") == (False, False, False)


def _build_mock_solar_calc():
    """Minimal solar-calculator stub for per-unit learning tests."""
    from unittest.mock import MagicMock
    mock = MagicMock()
    mock.calculate_unit_coefficient = MagicMock(return_value={"s": 0.0, "e": 0.0, "w": 0.0})
    mock.calculate_unit_solar_impact = MagicMock(return_value=0.0)
    mock.apply_correction = MagicMock(side_effect=lambda base, impact, mode: base)
    return mock


_mock_solar_calc = _build_mock_solar_calc()


class TestLearningRouting:
    """Per-entity reconstruction during NLMS learning."""

    def test_unscreened_entity_uses_effective_as_potential(self):
        """An entity not in the set receives the effective vector unchanged
        as its potential; its NLMS update reflects the full pre-screen
        irradiance without absorbing the global avg transmittance.

        Asserted indirectly: compare the coefficient learned by two
        otherwise-identical entities at the same hour, one in the set and
        one out.  The unscreened entity must end with a smaller coefficient
        because its "potential" input was larger (same impact target / larger
        potential magnitude = smaller coefficient).
        """
        manager = LearningManager()
        coeffs: dict = {"hp.affected": {}, "hp.unscreened": {}}
        buffers: dict = {"hp.affected": [], "hp.unscreened": []}

        # Drive both through several identical samples.  `avg_solar_vector`
        # here is the effective (post-screen) vector; `correction_percent`=50
        # + screen_south=True means reconstruction doubles the effective for
        # the affected entity.
        for _ in range(20):
            # Affected entity: reconstruction inflates the potential.
            manager._process_per_unit_learning(
                temp_key="5",
                wind_bucket="normal",
                avg_temp=5.0,
                avg_solar_vector=(0.3, 0.0, 0.0),  # effective south
                total_energy_kwh=2.0,
                base_expected_kwh=2.0,
                energy_sensors=["hp.affected", "hp.unscreened"],
                hourly_delta_per_unit={"hp.affected": 1.7, "hp.unscreened": 1.7},
                solar_enabled=True,
                learning_rate=1.0,
                solar_calculator=_mock_solar_calc,
                get_predicted_unit_base_fn=lambda *_a, **_k: 2.0,
                learning_buffer_per_unit={},
                correlation_data_per_unit={},
                observation_counts={},
                is_aux_active=False,
                aux_coefficients_per_unit={},
                learning_buffer_aux_per_unit={},
                solar_coefficients_per_unit=coeffs,
                learning_buffer_solar_per_unit=buffers,
                balance_point=15.0,
                unit_modes={"hp.affected": "heating", "hp.unscreened": "heating"},
                hourly_expected_per_unit={"hp.affected": 2.0, "hp.unscreened": 2.0},
                hourly_expected_base_per_unit={"hp.affected": 2.0, "hp.unscreened": 2.0},
                aux_affected_entities=[],
                is_cooldown_active=False,
                correction_percent=50.0,  # 50 % closed
                screen_config=(True, False, False),  # south has screens
                screen_affected_entities=frozenset({"hp.affected"}),
            )

        # Heating-regime read per #868.
        coeff_affected = coeffs.get("hp.affected", {}).get("heating", {}).get("s", 0.0)
        coeff_unscreened = coeffs.get("hp.unscreened", {}).get("heating", {}).get("s", 0.0)

        # Both should have learned SOMETHING.
        assert coeff_affected > 0.0
        assert coeff_unscreened > 0.0
        # Affected entity reconstructs potential > effective → smaller coeff
        # converges for the same impact.  Unscreened entity uses effective
        # as potential → larger coeff to match same impact.
        assert coeff_unscreened > coeff_affected

    def test_none_affected_set_treats_all_as_affected(self):
        """Legacy compat: `screen_affected_entities=None` means every entity
        uses the installation-level screen_config (prior behaviour)."""
        manager = LearningManager()
        coeffs: dict = {}
        # Persistent buffer across iterations so cold-start can accumulate
        # and solve.  Per-(entity, regime) nested shape since #868.
        buffers: dict = {}
        for _ in range(10):
            manager._process_per_unit_learning(
                temp_key="5",
                wind_bucket="normal",
                avg_temp=5.0,
                avg_solar_vector=(0.3, 0.0, 0.0),
                total_energy_kwh=2.0,
                base_expected_kwh=2.0,
                energy_sensors=["hp.only"],
                hourly_delta_per_unit={"hp.only": 1.7},
                solar_enabled=True,
                learning_rate=1.0,
                solar_calculator=_mock_solar_calc,
                get_predicted_unit_base_fn=lambda *_a, **_k: 2.0,
                learning_buffer_per_unit={},
                correlation_data_per_unit={},
                observation_counts={},
                is_aux_active=False,
                aux_coefficients_per_unit={},
                learning_buffer_aux_per_unit={},
                solar_coefficients_per_unit=coeffs,
                learning_buffer_solar_per_unit=buffers,
                balance_point=15.0,
                unit_modes={"hp.only": "heating"},
                hourly_expected_per_unit={"hp.only": 2.0},
                hourly_expected_base_per_unit={"hp.only": 2.0},
                aux_affected_entities=[],
                is_cooldown_active=False,
                correction_percent=50.0,
                screen_config=(True, False, False),
                screen_affected_entities=None,  # legacy
            )
        # Cold-start completed; heating regime populated (#868).
        assert "hp.only" in coeffs
        assert "s" in coeffs["hp.only"]["heating"]


class TestConfigFlowSentinel:
    """D3: ``_build_final_data`` must distinguish key-absent from empty list.

    A user who explicitly deselects all entities in the form intentionally
    wants zero affected entities (e.g. installation with no external
    screens at all).  Truthiness-based ``if not data.get(...)`` cannot
    tell that apart from "key never seen" and would clobber the empty
    selection with all energy_sensors.
    """

    def _build_flow(self, submitted: dict):
        """Call ``_build_final_data`` on a minimal flow instance."""
        import sys
        from unittest.mock import MagicMock
        sys.modules.setdefault("homeassistant.data_entry_flow", MagicMock())
        sys.modules.setdefault("homeassistant.helpers.selector", MagicMock())

        class _FakeBase:
            def __init_subclass__(cls, **_kwargs):
                return None

        import homeassistant.config_entries as _ce
        _ce.ConfigFlow = _FakeBase

        from custom_components.heating_analytics.config_flow import (
            HeatingAnalyticsConfigFlow,
        )
        flow = HeatingAnalyticsConfigFlow.__new__(HeatingAnalyticsConfigFlow)
        flow._flow_data = {"energy_sensors": ["hp.a", "hp.b"]}
        return flow._build_final_data(submitted)

    def test_screen_affected_empty_list_preserved(self):
        """Explicit empty list (user deselected all) must be preserved."""
        from custom_components.heating_analytics.const import (
            CONF_SCREEN_AFFECTED_ENTITIES,
        )
        data = self._build_flow({CONF_SCREEN_AFFECTED_ENTITIES: []})
        assert data[CONF_SCREEN_AFFECTED_ENTITIES] == []

    def test_screen_affected_absent_defaults_to_all(self):
        """Key absent (user never saw form) defaults to all energy_sensors."""
        from custom_components.heating_analytics.const import (
            CONF_SCREEN_AFFECTED_ENTITIES,
        )
        data = self._build_flow({})
        assert data[CONF_SCREEN_AFFECTED_ENTITIES] == ["hp.a", "hp.b"]

    def test_aux_affected_empty_list_preserved(self):
        """Same sentinel guarantee for aux_affected_entities."""
        from custom_components.heating_analytics.const import (
            CONF_AUX_AFFECTED_ENTITIES,
        )
        data = self._build_flow({CONF_AUX_AFFECTED_ENTITIES: []})
        assert data[CONF_AUX_AFFECTED_ENTITIES] == []


class TestReplayRouting:
    """Per-entity reconstruction during retrain replay."""

    def test_unscreened_entity_in_replay_uses_effective(self):
        """`replay_solar_nlms` must call `_learn_unit_solar_coefficient`
        with the entity's effective (not reconstructed) vector when the
        entity is excluded from screen_affected_entities.
        """
        manager = LearningManager()
        captured: list[tuple] = []

        def _record(*, entity_id, avg_solar_vector, **_kwargs):
            captured.append((entity_id, avg_solar_vector))

        manager._learn_unit_solar_coefficient = _record

        entry = {
            "temp_key": "5",
            "wind_bucket": "normal",
            "unit_modes": {"hp.unscreened": "heating"},
            "unit_breakdown": {"hp.unscreened": 1.5},
            "auxiliary_active": False,
            "learning_status": "ok",
            "solar_vector_s": 0.3,
            "solar_vector_e": 0.0,
            "solar_vector_w": 0.0,
            "correction_percent": 50.0,
        }

        coord = MagicMock()
        coord.balance_point = 15.0
        solar_calc = SolarCalculator(coord)

        correlation = {"hp.unscreened": {"5": {"normal": 2.0}}}

        manager.replay_solar_nlms(
            [entry],
            solar_calculator=solar_calc,
            screen_config=(True, False, False),  # south screened
            correlation_data_per_unit=correlation,
            solar_coefficients_per_unit={},
            learning_buffer_solar_per_unit={},
            energy_sensors=["hp.unscreened"],
            learning_rate=0.1,
            balance_point=15.0,
            screen_affected_entities=frozenset(),  # exclude all
        )

        assert captured, "replay did not reach _learn_unit_solar_coefficient"
        _, vec = captured[0]
        # Unscreened: south component stays at effective 0.3 (no inflation
        # from dividing by south transmittance at 50% slider + screened).
        assert vec[0] == pytest.approx(0.3, abs=0.01)

    def test_inequality_skipped_for_unscreened_entity(self):
        """D1c: unscreened shutdown-flagged entity skips the inequality
        learner in `replay_solar_nlms`.

        The inequality path uses a coordinator-level ``battery_filtered_potential``
        reconstructed with the installation ``screen_config``.  For an
        unscreened entity this battery is inflated by `1/avg_transmittance`,
        which would satisfy the inequality constraint at too-low coeff —
        under-lifting the shutdown signal.  The fix: skip inequality entirely
        for entities not in ``screen_affected_entities``.  Unscreened entities
        already have correct NLMS signal via the live per-entity reconstruction.
        """
        from unittest.mock import MagicMock
        from custom_components.heating_analytics.const import (
            SOLAR_SHUTDOWN_MIN_MAGNITUDE,
        )
        manager = LearningManager()

        inequality_calls: list[str] = []

        def _record_inequality(*, entity_id, **_kwargs):
            inequality_calls.append(entity_id)
            return "updated"

        manager._update_unit_solar_inequality = _record_inequality

        # Entry with shutdown conditions.  `solar_dominant_entities` is
        # the gate that puts an entity into the inequality branch inside
        # `replay_solar_nlms` (line ~1971) — without it, the replay never
        # reaches the inequality code path and the test is theatrical
        # (passes whether or not the screen-scope fix is applied).
        entry = {
            "temp_key": "5",
            "wind_bucket": "normal",
            "unit_modes": {"hp.unscreened": "heating"},
            "unit_breakdown": {"hp.unscreened": 0.05},
            "auxiliary_active": False,
            "learning_status": "ok",
            "solar_vector_s": 0.6,
            "solar_vector_e": 0.0,
            "solar_vector_w": 0.0,
            "correction_percent": 100.0,
            "solar_dominant_entities": ["hp.unscreened"],
        }
        correlation = {"hp.unscreened": {"5": {"normal": 2.5}}}

        coord = MagicMock()
        coord.balance_point = 15.0
        solar_calc = SolarCalculator(coord)

        manager.replay_solar_nlms(
            [entry] * 20,  # many hours to populate battery + trigger shutdown gates
            solar_calculator=solar_calc,
            screen_config=(True, False, False),
            correlation_data_per_unit=correlation,
            solar_coefficients_per_unit={},
            learning_buffer_solar_per_unit={},
            energy_sensors=["hp.unscreened"],
            learning_rate=0.1,
            balance_point=15.0,
            screen_affected_entities=frozenset(),  # exclude all → inequality should skip
        )
        # Inequality learner must not have fired for the unscreened entity.
        assert "hp.unscreened" not in inequality_calls

    def test_inequality_fires_for_screen_affected_entity(self):
        """Positive control for D1c: same shutdown scenario as above, but
        with the entity IN ``screen_affected_entities``, the inequality
        learner MUST fire.  Ensures the gate doesn't accidentally skip
        everything and make the negative test theatrical.
        """
        from custom_components.heating_analytics.const import SOLAR_SHUTDOWN_MIN_BASE
        manager = LearningManager()

        inequality_calls: list[str] = []

        def _record_inequality(*, entity_id, **_kwargs):
            inequality_calls.append(entity_id)
            return "updated"

        manager._update_unit_solar_inequality = _record_inequality

        entry = {
            "temp_key": "5",
            "wind_bucket": "normal",
            "unit_modes": {"hp.screened": "heating"},
            "unit_breakdown": {"hp.screened": 0.05},
            "auxiliary_active": False,
            "learning_status": "ok",
            "solar_vector_s": 0.6,
            "solar_vector_e": 0.0,
            "solar_vector_w": 0.0,
            "correction_percent": 100.0,
            "solar_dominant_entities": ["hp.screened"],
        }
        correlation = {"hp.screened": {"5": {"normal": max(2.5, SOLAR_SHUTDOWN_MIN_BASE * 2)}}}

        coord = MagicMock()
        coord.balance_point = 15.0
        solar_calc = SolarCalculator(coord)

        manager.replay_solar_nlms(
            [entry] * 20,
            solar_calculator=solar_calc,
            screen_config=(True, False, False),
            correlation_data_per_unit=correlation,
            solar_coefficients_per_unit={},
            learning_buffer_solar_per_unit={},
            energy_sensors=["hp.screened"],
            learning_rate=0.1,
            balance_point=15.0,
            screen_affected_entities=frozenset({"hp.screened"}),
        )
        # Inequality learner MUST have fired at least once for the
        # screen-affected entity.
        assert "hp.screened" in inequality_calls

    def test_affected_entity_in_replay_reconstructs_via_screens(self):
        """Counterpoint: affected entity sees inflated potential from the
        reconstruction (divides effective by screen-transmittance)."""
        manager = LearningManager()
        captured: list[tuple] = []

        def _record(*, entity_id, avg_solar_vector, **_kwargs):
            captured.append((entity_id, avg_solar_vector))

        manager._learn_unit_solar_coefficient = _record

        entry = {
            "temp_key": "5",
            "wind_bucket": "normal",
            "unit_modes": {"hp.affected": "heating"},
            "unit_breakdown": {"hp.affected": 1.5},
            "auxiliary_active": False,
            "learning_status": "ok",
            "solar_vector_s": 0.3,
            "solar_vector_e": 0.0,
            "solar_vector_w": 0.0,
            "correction_percent": 50.0,
        }

        coord = MagicMock()
        coord.balance_point = 15.0
        solar_calc = SolarCalculator(coord)

        correlation = {"hp.affected": {"5": {"normal": 2.0}}}

        manager.replay_solar_nlms(
            [entry],
            solar_calculator=solar_calc,
            screen_config=(True, False, False),
            correlation_data_per_unit=correlation,
            solar_coefficients_per_unit={},
            learning_buffer_solar_per_unit={},
            energy_sensors=["hp.affected"],
            learning_rate=0.1,
            balance_point=15.0,
            screen_affected_entities=frozenset({"hp.affected"}),
        )

        assert captured
        _, vec = captured[0]
        # Affected: south component inflated by 1/transmittance.  Exact
        # factor depends on SCREEN_DIRECT_TRANSMITTANCE + slider math in
        # SolarCalculator; the key invariant is vec[0] > 0.3.
        assert vec[0] > 0.3
