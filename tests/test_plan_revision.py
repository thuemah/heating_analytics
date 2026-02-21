"""Test the Plan Revision Impact logic."""
from unittest.mock import MagicMock, patch, AsyncMock
import pytest
from datetime import datetime
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator

@pytest.mark.asyncio
async def test_plan_revision_impact_calculation():
    """Test the plan revision impact calculation logic."""
    hass = MagicMock()
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.data = {
        "balance_point": 17.0,
        "outdoor_temp_sensor": "sensor.temp",
        "wind_speed_sensor": "sensor.wind",
        "energy_sensors": ["sensor.unit_1"]
    }

    with patch("custom_components.heating_analytics.storage.Store") as mock_store_cls, \
         patch("custom_components.heating_analytics.coordinator.dt_util.now") as mock_now:

        mock_store = mock_store_cls.return_value
        mock_store.async_load = AsyncMock(return_value={})

        coordinator = HeatingDataCoordinator(hass, entry)

        # --- Mock Data ---

        # Time: 12:30 on a specific day
        current_time = datetime(2023, 11, 1, 12, 30, 0)
        mock_now.return_value = current_time

        # Midnight Forecast (Reference Map)
        coordinator.forecast._cached_reference_map = {
            "2023-11-01": {
                # Hour 10: Forecast was cold (10 kWh)
                10: {"temp": 0.0, "wind": 2.0, "solar_factor": 0.0},
                # Hour 11: Forecast was mild (5 kWh)
                11: {"temp": 10.0, "wind": 1.0, "solar_factor": 0.1},
                # Hour 12 (Current): Forecast was mild (5 kWh)
                12: {"temp": 10.0, "wind": 1.0, "solar_factor": 0.1},
            }
        }

        # Actual Hourly Logs
        coordinator._hourly_log = [
            # Hour 10: Was actually colder, and Aux was used. (Reality: 15 kWh)
            # Plan was 10 kWh. Impact = 15 - 10 = +5
            {"timestamp": "2023-11-01T10:00:00", "hour": 10, "temp": -2.0, "effective_wind": 3.0, "solar_factor": 0.0, "auxiliary_active": True},
            # Hour 11: Was actually warmer, no Aux. (Reality: 3 kWh)
            # Plan was 5 kWh. Impact = 3 - 5 = -2
            {"timestamp": "2023-11-01T11:00:00", "hour": 11, "temp": 12.0, "effective_wind": 1.0, "solar_factor": 0.2, "auxiliary_active": False},
        ]

        # Current Conditions (for partial hour 12)
        # Actual is cold, no aux. (Reality Rate: 10 kWh/h)
        # Plan rate was 5 kWh/h. Impact for 30 mins = (10-5)*0.5 = +2.5
        coordinator._get_float_state = MagicMock(return_value=-1.0) # current_temp
        coordinator.data["effective_wind"] = 2.0
        coordinator.data["solar_factor"] = 0.0
        coordinator.auxiliary_heating_active = False

        # --- Mock the Model ---
        def mock_calculate_total_power(temp, effective_wind, solar_impact, is_aux_active, override_solar_factor, **kwargs):
            # Planned Scenarios (is_aux_active is always False)
            if temp == 0.0: return {"total_kwh": 10.0}  # Cold forecast
            if temp == 10.0: return {"total_kwh": 5.0} # Mild forecast

            # Reality Scenarios
            if temp == -2.0 and is_aux_active: return {"total_kwh": 15.0} # Colder + Aux
            if temp == 12.0 and not is_aux_active: return {"total_kwh": 3.0} # Warmer
            if temp == -1.0 and not is_aux_active: return {"total_kwh": 10.0} # Current Hour Reality (Cold)

            return {"total_kwh": 0.0}

        coordinator.statistics.calculate_total_power = MagicMock(side_effect=mock_calculate_total_power)

        # --- Run Calculation ---
        result = coordinator.forecast.calculate_plan_revision_impact()

        # --- Assertions ---
        # Total Impact = 5 (hour 10) - 2 (hour 11) + 2.5 (hour 12) = 5.5
        assert "estimated_impact_kwh" in result
        assert result["estimated_impact_kwh"] == pytest.approx(5.5)

        # Verify context flags
        assert result["has_aux_usage"] is True
        assert result["weather_driver"] == "colder"
        assert result["hours_analyzed"] == pytest.approx(2.5)
