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
        coordinator.solar.calculate_unit_coefficient = MagicMock(return_value={"s": 1.0, "e": 0.0, "w": 0.0})
        coordinator.solar.calculate_unit_solar_impact = MagicMock(return_value=0.5)

        # 1. Setup Hourly Data
        # Actual Consumption: 5.0 kWh
        # Aux Impact: 2.0 kWh
        # Solar Impact: 0.5 kWh
        # Expected Gross: 5.0 + 2.0 + 0.5 = 7.5 kWh

        coordinator._collector.energy_hour = 5.0
        coordinator._collector.aux_impact_hour = 2.0

        # Mock Solar Factors to ensure calculation runs
        coordinator._collector.sample_count = 60
        coordinator._collector.solar_sum = 60.0 # Avg 1.0
        coordinator._collector.temp_sum = 0.0
        coordinator._collector.wind_values = [0.0]

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
        coordinator._collector.start_time = datetime(2023, 10, 27, 12, 0, 0)

        # Run Processing
        await coordinator._process_hourly_data(current_time)

        # Verify Log
        assert len(coordinator._hourly_log) == 1
        log = coordinator._hourly_log[0]

        assert log["actual_kwh"] == 5.0
        assert log["aux_impact_kwh"] == 2.0
        # Solar impact is battery-smoothed (EMA): first hour = raw * (1 - decay)
        assert log["solar_impact_kwh"] == pytest.approx(0.5 * 0.20, abs=0.01)

        # THE NEW FIELD
        # Should be Actual + Aux + Solar (battery-smoothed)
        # 5.0 + 2.0 + 0.125 = 7.125
        assert "thermodynamic_gross_kwh" in log
        assert log["thermodynamic_gross_kwh"] == pytest.approx(5.0 + 2.0 + 0.5 * 0.20, abs=0.01)


class TestThermodynamicGrossModeAware:
    """Mode-aware solar adjustment in ``thermodynamic_gross_kwh`` (#921 follow-up).

    Pre-fix: solar sign was gated on ``avg_temp >= balance_point`` — wrong
    on mixed-mode hours and on cooling-near-BP / heating-above-BP hours.
    Post-fix: per-unit mode share via ``solar_heating_applied_kwh`` /
    ``solar_cooling_applied_kwh`` from ``calculate_total_power``.
    """

    def _build_coordinator(self, hass):
        from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
        entry = MagicMock()
        entry.data = {
            "balance_point": 17.0,
            "learning_rate": 0.1,
            "energy_sensors": ["sensor.heater"],
            "solar_enabled": True,
        }
        with patch("custom_components.heating_analytics.storage.Store"):
            coord = HeatingDataCoordinator(hass, entry)
        coord._async_save_data = AsyncMock()
        coord.storage.append_hourly_log_csv = AsyncMock()
        coord.learning.process_learning = MagicMock(return_value={
            "model_base_before": 1.0, "model_base_after": 1.0, "model_updated": False,
        })
        coord.solar.calculate_unit_coefficient = MagicMock(
            return_value={"s": 1.0, "e": 0.0, "w": 0.0}
        )
        coord.solar.calculate_unit_solar_impact = MagicMock(return_value=0.5)
        coord._collector.energy_hour = 5.0
        coord._collector.aux_impact_hour = 0.0
        coord._collector.sample_count = 60
        coord._collector.solar_sum = 60.0
        coord._collector.temp_sum = 0.0
        coord._collector.wind_values = [0.0]
        coord._hourly_delta_per_unit = {"sensor.heater": 5.0}
        coord._correlation_data = {"0": {"normal": 10.0}}
        coord._correlation_data_per_unit = {"sensor.heater": {"0": {"normal": 10.0}}}
        coord._collector.start_time = datetime(2023, 10, 27, 12, 0, 0)
        return coord

    @pytest.mark.asyncio
    async def test_cooling_unit_solar_subtracts_from_gross(self, hass: HomeAssistant):
        """Single cooling unit: solar ADDS to demand → gross subtracts solar.
        Pre-fix this only worked when temp ≥ BP; post-fix it's mode-driven.
        """
        from custom_components.heating_analytics.const import (
            MODE_COOLING, COOLING_WIND_BUCKET,
        )
        coord = self._build_coordinator(hass)
        # Cooling mode active.  Temp will be 0 (well BELOW BP=17) — under
        # the old ``avg_temp >= BP`` gate this would WRONGLY add solar to
        # gross.  With the per-unit-mode fix, cooling subtracts.
        coord._unit_modes = {"sensor.heater": MODE_COOLING}
        # Release the cooling cold-start gate (#921): cooling base bucket
        # populated AND cooling solar coefficient flagged learned.
        coord._correlation_data_per_unit["sensor.heater"]["0"] = {
            COOLING_WIND_BUCKET: 5.0
        }
        coord._solar_coefficients_per_unit = {
            "sensor.heater": {
                "heating": {"s": 1.0, "e": 0.0, "w": 0.0, "learned": True},
                "cooling": {"s": 1.0, "e": 0.0, "w": 0.0, "learned": True},
            }
        }
        await coord._process_hourly_data(datetime(2023, 10, 27, 13, 0, 0))
        log = coord._hourly_log[-1]
        # Cooling solar was applied additively (solar.calculate_saturation
        # mode-cooling branch returns added=potential).  Battery EMA
        # smooths it: first hour = 0.5 * 0.20 = 0.1.  All units cooling →
        # share = 1 cooling, signed_fraction = -1.  gross = actual + 0 - 0.1.
        assert log["thermodynamic_gross_kwh"] == pytest.approx(5.0 - 0.1, abs=0.01)

    @pytest.mark.asyncio
    async def test_residual_battery_no_active_units_drops_adjustment(self, hass: HomeAssistant):
        """No applied solar this hour AND no active heating/cooling units:
        adjustment = 0 (cannot apportion residual battery).
        """
        from custom_components.heating_analytics.const import MODE_OFF
        coord = self._build_coordinator(hass)
        coord._unit_modes = {"sensor.heater": MODE_OFF}
        # Pre-charge battery so effective_solar_impact > 0.
        coord._solar_battery_state = 0.5
        # Zero solar in collector (night).
        coord._collector.solar_sum = 0.0
        coord.solar.calculate_unit_solar_impact = MagicMock(return_value=0.0)
        await coord._process_hourly_data(datetime(2023, 10, 27, 22, 0, 0))
        log = coord._hourly_log[-1]
        # With all units OFF and no current solar applied, the residual
        # battery cannot be apportioned by mode → adjustment = 0.
        # gross should be actual + aux + 0.
        assert log["thermodynamic_gross_kwh"] == pytest.approx(5.0, abs=0.01)


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
