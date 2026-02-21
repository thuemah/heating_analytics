"""Test the Deviation Breakdown logic."""
from unittest.mock import MagicMock, patch, AsyncMock
import pytest
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.const import ATTR_DEVIATION_BREAKDOWN

@pytest.mark.asyncio
async def test_deviation_breakdown_logic():
    """Test the deviation breakdown calculation."""
    hass = MagicMock()

    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.data = {
        "balance_point": 17.0,
        "outdoor_temp_sensor": "sensor.temp",
        "wind_speed_sensor": "sensor.wind",
        "energy_sensors": ["sensor.unit_1", "sensor.unit_2"]
    }

    with patch("custom_components.heating_analytics.storage.Store") as mock_store_cls, \
         patch("custom_components.heating_analytics.coordinator.dt_util.now") as mock_now:

        mock_store = mock_store_cls.return_value
        mock_store.async_load = AsyncMock(return_value={})
        mock_store.async_save = AsyncMock()

        coordinator = HeatingDataCoordinator(hass, entry)

        mock_now.return_value.hour = 12
        mock_now.return_value.minute = 30
        mock_now.return_value.date.return_value.isoformat.return_value = "2023-10-27"
        mock_now.return_value.isoformat.return_value = "2023-10-27T12:30:00"

        # Models
        # Temp 0 -> 2.0 kWh global
        # Temp 5 -> 1.0 kWh global (Different!)
        coordinator._correlation_data = {
            "0": {"normal": 2.0},
            "5": {"normal": 1.0}
        }
        coordinator._correlation_data_per_unit = {
            "sensor.unit_1": {"0": {"normal": 1.0}, "5": {"normal": 0.5}},
            "sensor.unit_2": {"0": {"normal": 1.0}, "5": {"normal": 0.5}}
        }

        # Logs
        coordinator._hourly_log = [
            # 10:00 - 11:00: NO temp_key (Fallback path). Temp=0.
            # Expected Unit 1: 1.0 kWh
            {"timestamp": "2023-10-27T10:00:00", "temp": 0.0, "wind_bucket": "normal", "solar_impact_kwh": 0.0},

            # 11:00 - 12:00: WITH temp_key="5" (Priority 4 path). Temp=0.
            # Expected Unit 1: Uses key "5" -> 0.5 kWh
            {"timestamp": "2023-10-27T11:00:00", "temp": 0.0, "temp_key": "5", "wind_bucket": "normal", "solar_impact_kwh": 0.0}
        ]

        # 2. Current Partial Hour (12:00 - 12:30)
        coordinator._calculate_inertia_temp = MagicMock(return_value=0.0) # Key "0"
        coordinator.data["effective_wind"] = 0.0
        coordinator.data["solar_impact_kwh"] = 0.0
        # Expected Unit 1: 1.0 * 0.5h = 0.5 kWh

        # TOTAL EXPECTED Unit 1:
        # Hour 1: 1.0
        # Hour 2: 0.5 (Proof of Priority 4 fix)
        # Hour 3: 0.5
        # Total: 2.0 kWh

        # Actual Consumption So Far
        coordinator._daily_individual = {
            "sensor.unit_1": 2.0, # Zero Deviation if logic works
            "sensor.unit_2": 2.0
        }

        hass.states.get.return_value = MagicMock(name="Unit One")

        coordinator._get_float_state = MagicMock(return_value=0.0)
        coordinator._get_speed_in_ms = MagicMock(return_value=0.0)
        coordinator._get_cloud_coverage = MagicMock(return_value=50.0)
        coordinator._calculate_future_forecast_kwh = MagicMock(return_value=(0.0, 0.0))
        coordinator._update_daily_forecast = AsyncMock()

        # Run
        breakdown = coordinator.statistics.calculate_deviation_breakdown()

        # Verify Unit 1
        unit_1 = next(item for item in breakdown if item["entity_id"] == "sensor.unit_1")
        assert unit_1["expected"] == 2.0
        assert unit_1["deviation"] == 0.0

        # If the fallback logic was used for Hour 2, it would use Temp=0 -> 1.0 kWh.
        # Total expected would be 1.0 + 1.0 + 0.5 = 2.5.
        # Deviation would be 2.0 - 2.5 = -0.5.
        # Since deviation is 0.0, the "temp_key" was correctly prioritized.
