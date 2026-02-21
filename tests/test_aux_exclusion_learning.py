"""Test the aux exclusion learning bug reproduction."""
from unittest.mock import MagicMock, patch, AsyncMock
import pytest
from datetime import datetime
from homeassistant.core import HomeAssistant
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.const import CONF_AUX_AFFECTED_ENTITIES

@pytest.mark.asyncio
async def test_aux_exclusion_learning_bug(hass: HomeAssistant):
    """Test that excluded units fail to learn when aux is active (Reproduction)."""
    entry = MagicMock()
    entry.data = {
        "balance_point": 17.0,
        "learning_rate": 0.1,
        "energy_sensors": ["sensor.main", "sensor.annex"],
        CONF_AUX_AFFECTED_ENTITIES: ["sensor.main"] # Annex is excluded
    }

    with patch("custom_components.heating_analytics.storage.Store") as mock_store_cls:
        coordinator = HeatingDataCoordinator(hass, entry)
        coordinator._async_save_data = AsyncMock()

        # Initialize Models
        # Base model for both is 1.0
        coordinator._correlation_data_per_unit = {
            "sensor.main": {"0": {"normal": 1.0}},
            "sensor.annex": {"0": {"normal": 1.0}}
        }
        coordinator._correlation_data = {"0": {"normal": 2.0}} # Sum of parts

        # Initialize Aux Coefficients
        coordinator._aux_coefficients_per_unit = {
            "sensor.main": {"0": {"normal": 0.0}},
            "sensor.annex": {"0": {"normal": 0.0}}
        }
        coordinator._aux_coefficients = {"0": {"normal": 0.0}}

        # Setup Hour Data
        coordinator._hourly_sample_count = 60
        coordinator._hourly_temp_sum = 0.0
        coordinator._hourly_wind_values = [0.0] * 60
        coordinator._hourly_bucket_counts = {"normal": 60, "high_wind": 0, "extreme_wind": 0}

        # Enable Aux
        coordinator.auxiliary_heating_active = True
        coordinator._hourly_aux_count = 60 # Dominant Aux

        # Scenario:
        # Main (Affected): Consumes 0.5 (Reduced from 1.0 by Aux)
        # Annex (Excluded): Consumes 1.5 (Increased from 1.0, maybe heater changed)
        # We expect Main to learn Aux Coefficient.
        # We expect Annex to learn Base Model (Update 1.0 -> higher).

        coordinator._accumulated_energy_hour = 2.0 # Total
        coordinator._accumulated_expected_energy_hour = 2.0 # Base expectation

        # Breakdown
        coordinator._hourly_delta_per_unit = {
            "sensor.main": 0.5,
            "sensor.annex": 1.5
        }

        # Important: Expected Base per unit must be set for learning to work
        # In real code this is accumulated, here we mock it to match our '1.0' assumption
        coordinator._hourly_expected_base_per_unit = {
             "sensor.main": 1.0,
             "sensor.annex": 1.0
        }

        current_time = datetime(2023, 10, 27, 13, 0, 0)
        await coordinator._process_hourly_data(current_time)

        # 1. Verify Main (Affected) learned Aux
        # Implied Reduction = 1.0 - 0.5 = 0.5
        # Global Rate = 0.1. Per-Unit Cap = 0.03. Effective Rate = 0.03.
        # New Coeff = 0.0 + 0.03 * (0.5 - 0.0) = 0.015
        # Allow small float diffs
        assert coordinator._aux_coefficients_per_unit["sensor.main"]["0"]["normal"] == pytest.approx(0.015, abs=0.001)

        # 2. Verify Annex (Excluded) learned Base Model
        # Actual = 1.5. Base = 1.0. Diff = 0.5.
        # New Base = 1.0 + 0.03 * (1.5 - 1.0) = 1.015 (Using capped per-unit rate 0.03)

        current_annex_val = coordinator._correlation_data_per_unit["sensor.annex"]["0"]["normal"]

        # This assert should FAIL if the bug exists (it will be 1.0)
        assert current_annex_val == pytest.approx(1.015, abs=0.001), \
            f"Annex (Excluded) should have learned base model (1.015) but stayed frozen at {current_annex_val}"
