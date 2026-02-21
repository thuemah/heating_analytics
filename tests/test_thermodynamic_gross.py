"""Test thermodynamic gross energy calculation logic."""
from unittest.mock import MagicMock, patch, AsyncMock
import pytest
from datetime import datetime
from homeassistant.core import HomeAssistant
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator

@pytest.mark.asyncio
async def test_hourly_thermodynamic_gross(hass: HomeAssistant):
    """Test that hourly processing calculates thermodynamic gross energy correctly."""
    entry = MagicMock()
    entry.data = {
        "balance_point": 17.0,
        "learning_rate": 0.1,
        "energy_sensors": ["sensor.heater"],
        "solar_enabled": True
    }

    with patch("custom_components.heating_analytics.storage.Store") as mock_store_cls:
        coordinator = HeatingDataCoordinator(hass, entry)
        coordinator._async_save_data = AsyncMock()
        coordinator.storage.append_hourly_log_csv = AsyncMock()

        # Mock dependencies
        coordinator.learning.process_learning = MagicMock(return_value={
            "model_base_before": 1.0,
            "model_base_after": 1.0,
            "model_updated": False
        })

        # Setup Solar Mock
        coordinator.solar.calculate_unit_coefficient = MagicMock(return_value=1.0)
        coordinator.solar.calculate_unit_solar_impact = MagicMock(return_value=0.5)

        # 1. Setup Hourly Data
        # Actual Consumption: 5.0 kWh
        # Aux Impact: 2.0 kWh
        # Solar Impact: 0.5 kWh
        # Expected Gross: 5.0 + 2.0 + 0.5 = 7.5 kWh

        coordinator._accumulated_energy_hour = 5.0
        coordinator._accumulated_aux_impact_hour = 2.0

        # Mock Solar Factors to ensure calculation runs
        coordinator._hourly_sample_count = 60
        coordinator._hourly_solar_sum = 60.0 # Avg 1.0
        coordinator._hourly_temp_sum = 0.0
        coordinator._hourly_wind_values = [0.0]

        # Ensure aux impact is passed through
        coordinator._hourly_delta_per_unit = {"sensor.heater": 5.0}

        # Populate Base Model to allow Solar Saturation
        # Solar Potential is 0.5. We need Base >= 0.5 for it to be applied.
        # Temp is 0.0 (default mock). Wind is 0.0.
        coordinator._correlation_data = {
            "0": { "normal": 10.0 }
        }
        coordinator._correlation_data_per_unit = {
            "sensor.heater": { "0": { "normal": 10.0 } }
        }

        current_time = datetime(2023, 10, 27, 13, 0, 0)
        coordinator._hourly_start_time = datetime(2023, 10, 27, 12, 0, 0)

        # Run Processing
        await coordinator._process_hourly_data(current_time)

        # Verify Log
        assert len(coordinator._hourly_log) == 1
        log = coordinator._hourly_log[0]

        assert log["actual_kwh"] == 5.0
        assert log["aux_impact_kwh"] == 2.0
        assert log["solar_impact_kwh"] == 0.5

        # THE NEW FIELD
        # Should be Actual + Aux + Solar
        assert "thermodynamic_gross_kwh" in log
        assert log["thermodynamic_gross_kwh"] == 7.5


def test_daily_thermodynamic_gross():
    """Test aggregation of daily thermodynamic gross energy."""
    # We don't need a full coordinator for this, just the method logic
    # But since it's an instance method, we instantiate carefully

    entry = MagicMock()
    entry.data = {"balance_point": 17.0}
    coordinator = HeatingDataCoordinator(MagicMock(), entry)

    day_logs = [
        {
            "hour": 0, "temp": 0,
            "actual_kwh": 5.0,
            "aux_impact_kwh": 2.0,
            "solar_impact_kwh": 1.0,
            "thermodynamic_gross_kwh": 8.0
        },
        {
            "hour": 1, "temp": 0,
            "actual_kwh": 4.0,
            "aux_impact_kwh": 0.0,
            "solar_impact_kwh": 0.0,
            "thermodynamic_gross_kwh": 4.0
        }
    ]

    result = coordinator._aggregate_daily_logs(day_logs)

    assert result["kwh"] == 9.0 # 5+4
    assert result["aux_impact_kwh"] == 2.0
    assert result["solar_impact_kwh"] == 1.0
    assert result["thermodynamic_gross_kwh"] == 12.0 # 8+4


def test_daily_thermodynamic_gross_fallback():
    """Test fallback calculation for legacy logs missing the field."""
    entry = MagicMock()
    entry.data = {"balance_point": 17.0}
    coordinator = HeatingDataCoordinator(MagicMock(), entry)

    day_logs = [
        {
            "hour": 0, "temp": 0,
            "actual_kwh": 5.0,
            "aux_impact_kwh": 2.0,
            "solar_impact_kwh": 1.0
            # Missing thermodynamic_gross_kwh
        },
        {
            "hour": 1, "temp": 0,
            "actual_kwh": 4.0,
            "aux_impact_kwh": 0.0,
            "solar_impact_kwh": 0.0
            # Missing thermodynamic_gross_kwh
        }
    ]

    result = coordinator._aggregate_daily_logs(day_logs)

    # Logic: Sum(Gross) is 0.
    # Fallback: Total (9) + Aux (2) + Solar (1) = 12.0

    assert result["kwh"] == 9.0
    assert result["thermodynamic_gross_kwh"] == 12.0
