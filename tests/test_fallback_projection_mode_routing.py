"""Tests for #993: `_calculate_fallback_projection` must route the solar
regime from the entity's ``unit_modes`` value via ``_solar_coeff_regime``,
not from outdoor temperature relative to balance_point.

Pre-fix: ``mode = MODE_HEATING if current_temp < balance_point else MODE_COOLING``
silently routed:
- DHW / OFF / Guest entities to heating- or cooling-regime saturation
- Cooling-mode entities in mild-but-cold shoulder hours to heating
- Heating-mode entities in mild-but-warm hours to cooling

Post-fix: regime comes from ``unit_modes[entity_id]`` via
``_solar_coeff_regime``.  None (OFF/DHW) → no solar contribution; the
fallback returns plain base × minutes/60.
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


def _make_stats(*, unit_mode: str, balance_point: float = 15.0):
    """StatisticsManager with a minimal coordinator stub.

    ``calculate_unit_solar_impact`` returns a strong positive value so
    a heating-regime application produces (base - impact) clamped at 0
    while a cooling-regime application produces (base + impact).  Using
    the sign of the projection lets us assert which regime was applied.
    """
    coord = MagicMock()
    coord.balance_point = balance_point
    coord.solar_enabled = True
    coord.solar_correction_percent = 100.0
    coord.screen_config = (False, False, False)
    coord.screen_config_for_entity = lambda eid: (False, False, False)
    coord.get_unit_mode = lambda eid: unit_mode
    coord.solar_correction_percent = 100.0
    coord.data = {
        "solar_factor": 0.7,
        "solar_vector_s": 0.5,
        "solar_vector_e": 0.0,
        "solar_vector_w": 0.0,
        "effective_wind": 0.0,
    }
    coord.model.correlation_data_per_unit = {
        "hp.unit": {"5": {"normal": 1.0}, "20": {"cooling": 1.0}},
    }
    coord.solar.calculate_unit_coefficient = MagicMock(
        return_value={"s": 0.3, "e": 0.3, "w": 0.3}
    )
    # Strong solar potential — large enough that heating saturates to 0
    # and cooling addition is clearly positive vs. base alone.
    coord.solar.calculate_unit_solar_impact = MagicMock(return_value=5.0)
    # Pass-through apply_correction: subtract on heating, add on cooling,
    # clamp at 0.  Records the mode it was called with.
    apply_call: dict[str, object] = {}

    def _apply_correction(base, impact, mode):
        apply_call["mode"] = mode
        if mode == MODE_HEATING:
            return max(0.0, base - impact)
        return max(0.0, base + impact)

    coord.solar.apply_correction = MagicMock(side_effect=_apply_correction)
    # Cold-start guard returns False everywhere by default.
    stats = StatisticsManager(coord)
    stats._is_cooling_solar_cold_start = MagicMock(return_value=False)
    return stats, coord, apply_call


class TestRegimeRoutingFromUnitMode:
    """The regime must come from ``unit_modes`` via ``_solar_coeff_regime``."""

    def test_cooling_mode_at_cold_outdoor_temp_uses_cooling_regime(self):
        """Cooling unit when outdoor temp < balance_point.

        Pre-fix: temp < BP → routed to MODE_HEATING → solar SUBTRACTED.
        Post-fix: mode=cooling → routed to MODE_COOLING → solar ADDED.
        """
        stats, _coord, apply_call = _make_stats(unit_mode=MODE_COOLING)
        result = stats._calculate_fallback_projection(
            entity_id="hp.unit",
            temp_key="20",
            wind_bucket="cooling",
            current_temp=10.0,  # below BP=15
            minutes_passed=60,
        )
        assert apply_call["mode"] == MODE_COOLING
        # Base=1.0 + impact=5.0 = 6.0 (added, not subtracted to 0).
        assert result == pytest.approx(6.0)

    def test_heating_mode_at_warm_outdoor_temp_uses_heating_regime(self):
        """Heating unit at temp >= BP.

        Pre-fix: temp >= BP → routed to MODE_COOLING → solar ADDED.
        Post-fix: mode=heating → routed to MODE_HEATING → solar SUBTRACTED.
        """
        stats, _coord, apply_call = _make_stats(unit_mode=MODE_HEATING)
        result = stats._calculate_fallback_projection(
            entity_id="hp.unit",
            temp_key="20",
            wind_bucket="normal",
            current_temp=20.0,  # above BP=15
            minutes_passed=60,
        )
        assert apply_call["mode"] == MODE_HEATING
        # Base=0.0 (no heating-bucket data at temp_key=20 "normal") -
        # extrapolation from "5" gives some value; either way saturation
        # clamps the subtraction at 0.
        assert result == pytest.approx(0.0)

    def test_dhw_mode_skips_solar_entirely(self):
        """DHW unit → regime is None → no solar contribution, base only."""
        stats, coord, apply_call = _make_stats(unit_mode=MODE_DHW)
        result = stats._calculate_fallback_projection(
            entity_id="hp.unit",
            temp_key="5",
            wind_bucket="normal",
            current_temp=5.0,
            minutes_passed=60,
        )
        # apply_correction must NOT have been called.
        assert "mode" not in apply_call
        coord.solar.apply_correction.assert_not_called()
        # Result is plain base × 1.0 = 1.0.
        assert result == pytest.approx(1.0)

    def test_off_mode_skips_solar_entirely(self):
        """OFF unit → regime is None → no solar contribution."""
        stats, coord, apply_call = _make_stats(unit_mode=MODE_OFF)
        result = stats._calculate_fallback_projection(
            entity_id="hp.unit",
            temp_key="5",
            wind_bucket="normal",
            current_temp=5.0,
            minutes_passed=60,
        )
        assert "mode" not in apply_call
        coord.solar.apply_correction.assert_not_called()
        assert result == pytest.approx(1.0)
