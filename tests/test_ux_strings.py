"""Test the UX strings in coordinator."""
from unittest.mock import MagicMock, patch, AsyncMock
import pytest
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.const import ATTR_ENERGY_TODAY, ATTR_EXPECTED_TODAY, ATTR_PREDICTED

@pytest.mark.asyncio
async def test_weather_adjusted_explanation_strings():
    """Test the weather adjusted deviation explanation strings."""
    hass = MagicMock()
    entry = MagicMock()
    entry.data = {}

    with patch("custom_components.heating_analytics.storage.Store"), \
         patch("custom_components.heating_analytics.coordinator.dt_util.now"):

        coordinator = HeatingDataCoordinator(hass, entry)

        # Mock dependencies
        coordinator.statistics.calculate_deviation_breakdown = MagicMock(return_value=[])
        coordinator.statistics.calculate_potential_savings = MagicMock()

        # Helper to run the logic
        def run_scenario(actual, expected, weather_impact):
            coordinator.data[ATTR_ENERGY_TODAY] = actual
            coordinator.data[ATTR_EXPECTED_TODAY] = expected
            coordinator.data[ATTR_PREDICTED] = expected # To avoid division by zero in deviation check

            # Mock the return of calculate_plan_revision_impact
            coordinator.forecast.calculate_plan_revision_impact = MagicMock(
                return_value={"estimated_impact_kwh": weather_impact}
            )

            coordinator._update_deviation_stats()
            return coordinator.data.get("weather_adjusted_deviation", {}).get("explanation")

        # Scenario 1: Perfect Match
        expl = run_scenario(10.0, 10.0, 0.0)
        assert "Consumption matches the model perfectly" in expl
        assert "✨" in expl

        # Scenario 2: More Usage
        expl = run_scenario(12.0, 10.0, 0.0)
        assert "You are using +2.0 kWh" in expl
        assert "more than the model expects" in expl
        assert "📈" in expl

        # Scenario 3: Less Usage (Savings)
        # Should use abs() to avoid "-2.0 kWh less"
        expl = run_scenario(8.0, 10.0, 0.0)
        assert "You are using 2.0 kWh" in expl
        assert "-2.0 kWh" not in expl
        assert "less than the model expects" in expl
        assert "📉" in expl
