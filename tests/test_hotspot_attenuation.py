"""Tests for the experimental hotspot-loss attenuation (#950).

The attenuation gates on:
  1. γ > 0 (configured)
  2. Entity has at least one screened facade (screen_config_for_entity)
  3. Sun elevation > 30°
  4. Screen actively deployed (correction_percent < 80)

When all four hold, per-unit solar_impact is scaled by (1 − γ) before
saturation logic.  When any is false, prediction stays at the legacy
coeff × potential.  γ = 0.0 is a bit-identical no-op.
"""
from unittest.mock import MagicMock

import pytest

from custom_components.heating_analytics.const import DOMAIN, MODE_HEATING
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.solar import SolarCalculator
from custom_components.heating_analytics.statistics import StatisticsManager


class MockHass:
    def __init__(self):
        self.states = MagicMock()
        self.states.get = MagicMock(return_value=None)
        self.data = {DOMAIN: {}}
        self.config_entries = MagicMock()
        self.bus = MagicMock()
        self.is_running = True


@pytest.fixture
def mock_hass():
    return MockHass()


def _build_coord(mock_hass, gamma: float, screen_affected: bool):
    entry = MagicMock()
    entry.data = {
        "energy_sensors": ["sensor.heater_1"],
        "aux_affected_entities": [],
        "screen_affected_entities": (
            ["sensor.heater_1"] if screen_affected else []
        ),
        # Per-direction screen presence — south screened (matches typical
        # high-elevation hotspot scenario from issue #950).
        "screen_south": True,
        "screen_east": False,
        "screen_west": False,
        "outdoor_temp_sensor": "sensor.outdoor_temp",
        "balance_point": 15.0,
        "wind_speed_sensor": "sensor.wind_speed",
        "solar_enabled": True,
        "solar_hotspot_attenuation_gamma": gamma,
    }
    coord = HeatingDataCoordinator(mock_hass, entry)
    coord.statistics = StatisticsManager(coord)
    coord.solar = SolarCalculator(coord)

    # Stub out coefficient + impact so the result is exactly 0.81 before any
    # attenuation.  The post-attenuation expected value is the entire
    # interest of these tests.
    coord.solar.calculate_unit_coefficient = MagicMock(
        return_value={"s": 1.0, "e": 0.0, "w": 0.0}
    )
    coord.solar.calculate_unit_solar_impact = MagicMock(return_value=0.81)

    # Storage hooks
    coord._correlation_data_per_unit = {"sensor.heater_1": {"_id": "unit_base"}}
    coord._aux_coefficients_per_unit = {"sensor.heater_1": {"_id": "unit_aux"}}
    coord._aux_coefficients = {"_id": "global_aux"}
    coord._correlation_data = {"_id": "global_base"}
    coord._hourly_delta_per_unit = {"sensor.heater_1": 0.0}
    coord._collector.aux_breakdown = {}

    # Base predictions: 2.1 kWh base, no aux.  Pick numbers where solar is
    # not clipped by saturation (net_demand 2.1 > 0.81) so the gate's effect
    # surfaces in the breakdown directly.
    def _mock_pred(data_map, temp_key, wind_bucket, temp, bp, apply_scaling=True):
        doc_id = data_map.get("_id") if isinstance(data_map, dict) else None
        if doc_id in ("unit_base", "global_base"):
            return 2.1
        return 0.0

    coord.statistics._get_prediction_from_model = MagicMock(side_effect=_mock_pred)
    coord.get_unit_mode = MagicMock(return_value=MODE_HEATING)
    coord.data["effective_wind"] = 0.0

    return coord


def _run_with_sun(coord, elevation: float, correction_percent: float):
    coord.solar_correction_percent = correction_percent
    coord.solar.get_approx_sun_pos = MagicMock(return_value=(elevation, 180.0))
    return coord.statistics.calculate_total_power(
        temp=5.0,
        effective_wind=0.0,
        solar_impact=0.0,
        is_aux_active=False,
    )


def test_gamma_zero_is_noop(mock_hass):
    """γ = 0.0 produces identical solar reduction vs no-gate code path."""
    coord = _build_coord(mock_hass, gamma=0.0, screen_affected=True)
    result = _run_with_sun(coord, elevation=45.0, correction_percent=20.0)
    bd = result["unit_breakdown"]["sensor.heater_1"]
    # Legacy: solar_impact = 0.81 (unclipped, base 2.1 > 0.81).
    assert bd["raw_solar_kwh"] == pytest.approx(0.81)
    assert bd["solar_reduction_kwh"] == pytest.approx(0.81)


def test_gate_fires_full_attenuation(mock_hass):
    """γ = 0.3 + screened + elev > 30 + correction < 80 → impact × 0.7."""
    coord = _build_coord(mock_hass, gamma=0.3, screen_affected=True)
    result = _run_with_sun(coord, elevation=45.0, correction_percent=20.0)
    bd = result["unit_breakdown"]["sensor.heater_1"]
    assert bd["raw_solar_kwh"] == pytest.approx(0.81 * 0.7)
    assert bd["solar_reduction_kwh"] == pytest.approx(0.81 * 0.7)


def test_unscreened_entity_skips_gate(mock_hass):
    """Entity not in screen_affected_entities → no attenuation."""
    coord = _build_coord(mock_hass, gamma=0.3, screen_affected=False)
    result = _run_with_sun(coord, elevation=45.0, correction_percent=20.0)
    bd = result["unit_breakdown"]["sensor.heater_1"]
    assert bd["raw_solar_kwh"] == pytest.approx(0.81)


def test_low_elevation_skips_gate(mock_hass):
    """Sun elev < 30° → no attenuation even if other conditions hold."""
    coord = _build_coord(mock_hass, gamma=0.3, screen_affected=True)
    result = _run_with_sun(coord, elevation=20.0, correction_percent=20.0)
    bd = result["unit_breakdown"]["sensor.heater_1"]
    assert bd["raw_solar_kwh"] == pytest.approx(0.81)


def test_elevation_boundary_strict(mock_hass):
    """Elev = 30.0 exactly → gate is strict `> 30`, so no attenuation."""
    coord = _build_coord(mock_hass, gamma=0.3, screen_affected=True)
    result = _run_with_sun(coord, elevation=30.0, correction_percent=20.0)
    bd = result["unit_breakdown"]["sensor.heater_1"]
    assert bd["raw_solar_kwh"] == pytest.approx(0.81)


def test_screen_open_skips_gate(mock_hass):
    """correction_percent ≥ 80 → screen effectively open, no attenuation."""
    coord = _build_coord(mock_hass, gamma=0.3, screen_affected=True)
    result = _run_with_sun(coord, elevation=45.0, correction_percent=80.0)
    bd = result["unit_breakdown"]["sensor.heater_1"]
    assert bd["raw_solar_kwh"] == pytest.approx(0.81)


def test_correction_boundary_strict(mock_hass):
    """correction = 80.0 exactly → gate is strict `< 80`, no attenuation."""
    coord = _build_coord(mock_hass, gamma=0.3, screen_affected=True)
    result = _run_with_sun(coord, elevation=45.0, correction_percent=80.0)
    bd = result["unit_breakdown"]["sensor.heater_1"]
    assert bd["raw_solar_kwh"] == pytest.approx(0.81)


def test_just_inside_correction_boundary_fires(mock_hass):
    """correction = 79.999 < 80 → gate fires (sanity for the < boundary)."""
    coord = _build_coord(mock_hass, gamma=0.3, screen_affected=True)
    result = _run_with_sun(coord, elevation=45.0, correction_percent=79.999)
    bd = result["unit_breakdown"]["sensor.heater_1"]
    assert bd["raw_solar_kwh"] == pytest.approx(0.81 * 0.7)
