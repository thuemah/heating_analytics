"""Tests for solar attributes on HeatingDeviceDailySensor (#862 suite, Nivå 2).

Per-unit sensors now expose the learned solar coefficient, instantaneous
solar impact, saturation percentage, and a learning-status string.  Values
come from live in-memory coordinator state (no log iteration), so the
attribute update is O(1).

Covers:
1. All solar fields present when coefficient has been learned
2. `solar_learning_status` state matrix:
   - "cold_start"  — no coefficient, empty buffer
   - "buffering_N/4" — cold-start buffer accumulating
   - "learning"    — coefficient present, NLMS healthy
   - "dead_zone_warning_N/15" — counter ≥ 10 (SOLAR_DEAD_ZONE_THRESHOLD-5)
3. Saturation % calculation: impact / (predicted + impact)
4. Absence handling: units without learned coefficients get no solar attrs
"""
from unittest.mock import MagicMock

import pytest

from custom_components.heating_analytics.sensor import HeatingDeviceDailySensor
from custom_components.heating_analytics.solar import SolarCalculator


def _build_coord_with_solar(
    entity_id: str = "sensor.vp_stue",
    *,
    coeff: dict | None = None,
    buffer: list | None = None,
    dead_zone_count: int = 0,
    solar_vector: tuple = (0.3, 0.1, 0.05),
    predicted_hourly: float = 0.5,
):
    """Coordinator mock with real SolarCalculator for arithmetic."""
    coord = MagicMock()
    coord.data = {
        "daily_individual": {entity_id: 1.0},
        "effective_wind": 3.0,
        "forecast_today_per_unit": {entity_id: 10.0},
        "solar_vector_s": solar_vector[0],
        "solar_vector_e": solar_vector[1],
        "solar_vector_w": solar_vector[2],
    }
    coord._calculate_inertia_temp.return_value = 10.0
    coord._get_wind_bucket.return_value = "normal"
    coord._get_predicted_kwh_per_unit.return_value = predicted_hourly
    coord._correlation_data_per_unit = {entity_id: {"10": {"normal": predicted_hourly}}}
    coord._hourly_log = [{"timestamp": "2026-05-01T12:00:00"}]
    coord.calculate_unit_rolling_power_watts = MagicMock(return_value=0)

    # Model proxy
    coord.model = MagicMock()
    coord.model.correlation_data_per_unit = coord._correlation_data_per_unit
    coord.model.observation_counts = {}
    coord.model.solar_coefficients_per_unit = {entity_id: coeff} if coeff else {}
    coord.model.hourly_log = coord._hourly_log

    # Solar calculator needs a minimal coordinator pointer
    coord.screen_config = (True, True, True)
    coord.solar = SolarCalculator(coord)

    # Solar-learning state
    coord._learning_buffer_solar_per_unit = {entity_id: buffer} if buffer else {}
    coord.learning = MagicMock()
    coord.learning._dead_zone_counts = {entity_id: dead_zone_count} if dead_zone_count else {}

    return coord


def _make_sensor(coord, entity_id: str = "sensor.vp_stue"):
    entry = MagicMock()
    entry.entry_id = "test_entry"
    state = MagicMock()
    state.name = "VP Stue"
    coord.hass = MagicMock()
    coord.hass.states.get.return_value = state
    sensor = HeatingDeviceDailySensor(coord, entry, entity_id)
    sensor.hass = coord.hass
    sensor.async_write_ha_state = MagicMock()
    return sensor


class TestSolarCoefficientsExposed:

    def test_all_three_components_exposed(self):
        """Learned coefficients appear as separate s/e/w attributes."""
        coord = _build_coord_with_solar(
            coeff={"s": 0.162, "e": 0.075, "w": 0.017},
        )
        sensor = _make_sensor(coord)
        attrs = sensor.extra_state_attributes
        assert attrs["solar_coefficient_s"] == pytest.approx(0.162)
        assert attrs["solar_coefficient_e"] == pytest.approx(0.075)
        assert attrs["solar_coefficient_w"] == pytest.approx(0.017)

    def test_impact_computed_from_current_vector(self):
        """solar_impact_current_kwh = coeff · effective_vector."""
        coord = _build_coord_with_solar(
            coeff={"s": 1.0, "e": 0.0, "w": 0.0},
            solar_vector=(0.4, 0.0, 0.0),
        )
        sensor = _make_sensor(coord)
        attrs = sensor.extra_state_attributes
        # coeff_s × vec_s = 1.0 × 0.4 = 0.4
        assert attrs["solar_impact_current_kwh"] == pytest.approx(0.4)

    def test_saturation_ratio(self):
        """saturation = impact / (predicted + impact)."""
        coord = _build_coord_with_solar(
            coeff={"s": 1.0, "e": 0.0, "w": 0.0},
            solar_vector=(0.2, 0.0, 0.0),  # impact = 0.2
            predicted_hourly=0.8,           # base minus solar
        )
        sensor = _make_sensor(coord)
        attrs = sensor.extra_state_attributes
        # saturation = 0.2 / (0.8 + 0.2) = 20 %
        assert attrs["solar_impact_saturation_pct"] == pytest.approx(20.0)


class TestAbsenceHandling:

    def test_no_coefficient_means_no_solar_attrs(self):
        """Unit without learned coefficient: solar_* keys absent except status."""
        coord = _build_coord_with_solar(coeff=None)
        sensor = _make_sensor(coord)
        attrs = sensor.extra_state_attributes
        assert "solar_coefficient_s" not in attrs
        assert "solar_impact_current_kwh" not in attrs
        # Learning status is always present
        assert "solar_learning_status" in attrs
        assert attrs["solar_learning_status"] == "cold_start"


class TestLearningStatusStateMatrix:

    def test_cold_start_no_coeff_no_buffer(self):
        coord = _build_coord_with_solar(coeff=None, buffer=None)
        sensor = _make_sensor(coord)
        assert sensor.extra_state_attributes["solar_learning_status"] == "cold_start"

    def test_buffering_shows_progress(self):
        """Cold-start buffer accumulating: shows N/4 progress."""
        coord = _build_coord_with_solar(
            coeff=None,
            buffer=[(0.3, 0.0, 0.0, 0.4), (0.4, 0.0, 0.0, 0.5)],  # 2 samples
        )
        sensor = _make_sensor(coord)
        assert sensor.extra_state_attributes["solar_learning_status"] == "buffering_2/4"

    def test_learning_after_convergence(self):
        """Coefficient present, no dead-zone: steady-state NLMS label."""
        coord = _build_coord_with_solar(
            coeff={"s": 0.2, "e": 0.1, "w": 0.05},
        )
        sensor = _make_sensor(coord)
        assert sensor.extra_state_attributes["solar_learning_status"] == "learning"

    def test_dead_zone_warning_at_threshold(self):
        """Dead-zone counter ≥ 10 triggers warning (before 15-reset threshold)."""
        coord = _build_coord_with_solar(
            coeff={"s": 0.05, "e": 0.0, "w": 0.0},
            dead_zone_count=12,
        )
        sensor = _make_sensor(coord)
        status = sensor.extra_state_attributes["solar_learning_status"]
        assert status.startswith("dead_zone_warning_12/")

    def test_dead_zone_below_warning_threshold_stays_learning(self):
        """Counter < 10: learning status, no early warning."""
        coord = _build_coord_with_solar(
            coeff={"s": 0.2, "e": 0.1, "w": 0.05},
            dead_zone_count=5,
        )
        sensor = _make_sensor(coord)
        assert sensor.extra_state_attributes["solar_learning_status"] == "learning"
