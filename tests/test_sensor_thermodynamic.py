"""Test the Thermodynamic Projection attributes in Sensor."""
from unittest.mock import MagicMock
from custom_components.heating_analytics.sensor import HeatingDeviationSensor
from custom_components.heating_analytics.const import ATTR_FORECAST_TODAY, ATTR_PREDICTED, ATTR_ENERGY_TODAY, ATTR_EXPECTED_TODAY, ATTR_DEVIATION_BREAKDOWN

def test_sensor_attributes():
    """Test that sensor exposes thermodynamic attributes from coordinator."""
    coordinator = MagicMock()
    coordinator.data = {
        ATTR_FORECAST_TODAY: 100.0,
        ATTR_PREDICTED: 90.0,
        ATTR_ENERGY_TODAY: 50.0,
        ATTR_EXPECTED_TODAY: 45.0,
        ATTR_DEVIATION_BREAKDOWN: [],
        "thermodynamic_projection_kwh": 95.0,
        "thermodynamic_deviation_kwh": -5.0,
        "thermodynamic_deviation_pct": -5.3,
        "savings_aux_hours_today": 0.0,
        "thermodynamic_gross_today_kwh": 50.0,
        "accumulated_solar_impact_kwh": 0.0,
        "accumulated_guest_impact_kwh": 0.0,
        "accumulated_aux_impact_kwh": 0.0,
        "plan_revision_impact": {},
        "weather_adjusted_deviation": {},
    }
    coordinator.auxiliary_heating_active = False
    coordinator._hourly_sample_count = 0

    entry = MagicMock()
    entry.entry_id = "test"

    sensor = HeatingDeviationSensor(coordinator, entry)

    attrs = sensor.extra_state_attributes

    assert attrs["thermodynamic_projection_kwh"] == 95.0
    assert attrs["thermodynamic_deviation_kwh"] == -5.0
    assert attrs["thermodynamic_deviation_pct"] == -5.3
