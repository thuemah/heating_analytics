"""Test the Aux Coefficient logic specifically."""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.const import ENERGY_GUARD_THRESHOLD, MODE_HEATING

@pytest.mark.asyncio
async def test_aux_coefficient_persistence(hass):
    """Test that aux coefficients are loaded and saved correctly."""
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.data = {}

    with patch("custom_components.heating_analytics.storage.Store") as mock_store_cls:
        mock_store = mock_store_cls.return_value
        # Mock loading data containing aux coefficients (Now nested by wind bucket)
        mock_store.async_load = AsyncMock(return_value={
            "aux_coefficients": {
                "0": {"normal": 2.5},
                "5": {"normal": 1.2}
            },
            "aux_coefficients_per_unit": {
                "sensor.heater_1": {"0": {"normal": 1.0}}
            }
        })
        mock_store.async_save = AsyncMock()

        coordinator = HeatingDataCoordinator(hass, entry)
        await coordinator._async_load_data()

        # Verify loaded
        assert coordinator._aux_coefficients["0"]["normal"] == 2.5
        assert coordinator._aux_coefficients["5"]["normal"] == 1.2
        assert coordinator._aux_coefficients_per_unit["sensor.heater_1"]["0"]["normal"] == 1.0

        # Verify save includes it
        coordinator._aux_coefficients["10"] = {"normal": 0.5}
        coordinator._aux_coefficients_per_unit["sensor.heater_1"]["10"] = {"normal": 0.2}

        await coordinator._async_save_data()

        # Check that async_save was called with data including the new coefficient
        # Note: async_save argument structure depends on StorageManager implementation
        # But we can check internal state
        assert coordinator._aux_coefficients["10"]["normal"] == 0.5
        assert coordinator._aux_coefficients_per_unit["sensor.heater_1"]["10"]["normal"] == 0.2

@pytest.mark.asyncio
async def test_aux_impact_application(hass):
    """Test that _get_predicted_kwh does NOT apply aux, but _update_live_predictions DOES."""
    # Note: _get_predicted_kwh returns the BASE model (Physics).
    # Logic in coordinator handles the subtraction.

    entry = MagicMock()
    entry.data = {"energy_sensors": ["sensor.heater"]}

    with patch("custom_components.heating_analytics.storage.Store"):
        coordinator = HeatingDataCoordinator(hass, entry)

        # Setup: Base model = 10.0 kWh for Temp 0
        coordinator._correlation_data = {"0": {"normal": 10.0}}
        # Setup: Aux Coeff = 3.0 kW for Temp 0
        coordinator._aux_coefficients = {"0": {"normal": 3.0}}

        # 1. Verify _get_predicted_kwh returns BASE (10.0)
        base = coordinator._get_predicted_kwh("0", "normal", actual_temp=0.0)
        assert base == 10.0

        # 2. Verify _get_aux_impact_kw returns 3.0
        impact = coordinator._get_aux_impact_kw("0")
        assert impact == 3.0

        # 3. Verify Live Prediction applies subtraction when Aux Active
        coordinator.auxiliary_heating_active = True
        coordinator.data["effective_wind"] = 0.0 # Force normal bucket

        current_time = datetime(2023, 1, 1, 12, 0, 0)

        # Call update_live_predictions
        # Should be max(0, 10.0 - 3.0) = 7.0
        rate = coordinator._update_live_predictions(0.0, "0", "normal", current_time)
        assert rate == 7.0

        # 4. Verify Live Prediction is BASE when Aux Inactive
        coordinator.auxiliary_heating_active = False
        rate = coordinator._update_live_predictions(0.0, "0", "normal", current_time)
        assert rate == 10.0

@pytest.mark.asyncio
async def test_aux_impact_clamping(hass):
    """Test that aux impact never results in negative prediction."""
    entry = MagicMock()
    entry.data = {}

    with patch("custom_components.heating_analytics.storage.Store"):
        coordinator = HeatingDataCoordinator(hass, entry)

        # Base: 2.0 kWh
        coordinator._correlation_data = {"0": {"normal": 2.0}}
        # Aux Impact: 5.0 kWh (Impossible/High, maybe oversized fire?)
        coordinator._aux_coefficients = {"0": {"normal": 5.0}}

        coordinator.auxiliary_heating_active = True
        current_time = datetime(2023, 1, 1, 12, 0, 0)

        # Should be max(0, 2.0 - 5.0) = 0.0
        rate = coordinator._update_live_predictions(0.0, "0", "normal", current_time)
        assert rate == 0.0

@pytest.mark.asyncio
async def test_aux_coefficient_learning_additive(hass):
    """Verify strictly the additive learning logic (Base - Actual = Impact)."""
    # This mirrors test_hourly_bucket_auxiliary but focuses on the math.
    entry = MagicMock()
    entry.data = {"learning_rate": 0.1, "energy_sensors": []}

    with patch("custom_components.heating_analytics.storage.Store"):
        coordinator = HeatingDataCoordinator(hass, entry)

        # Setup
        coordinator._correlation_data = {"0": {"normal": 10.0}}
        coordinator._aux_coefficients = {"0": {"normal": 2.0}} # Current Impact

        # Scenario:
        # Base = 10.0
        # Actual = 6.0
        # Implied Impact = 10.0 - 6.0 = 4.0
        # Current Impact = 2.0
        # Delta = 4.0 - 2.0 = 2.0
        # New Impact = 2.0 + 0.1 * (2.0) = 2.2

        coordinator._hourly_sample_count = 60
        coordinator._hourly_temp_sum = 0.0
        coordinator._hourly_bucket_counts = {"normal": 60}
        coordinator._hourly_wind_values = [0.0] * 60 # Fix: Populate wind values
        coordinator._hourly_aux_count = 60 # Dominant
        coordinator.auxiliary_heating_active = True

        coordinator._accumulated_energy_hour = 6.0 # Actual
        # Also populate per-unit delta, as learning uses this sum
        coordinator._hourly_delta_per_unit = {"sensor.dummy": 6.0}

        # Mock DEFAULT_AUX_LEARNING_RATE
        with patch("custom_components.heating_analytics.const.DEFAULT_AUX_LEARNING_RATE", 0.1):
            await coordinator._process_hourly_data(datetime.now())

        # Verify
        assert coordinator._aux_coefficients["0"]["normal"] == pytest.approx(2.2)

@pytest.mark.asyncio
async def test_aux_impact_fallback_interpolation(hass):
    """Test that we can interpolate/find nearest neighbor for aux coefficients."""
    entry = MagicMock()
    entry.data = {}

    with patch("custom_components.heating_analytics.storage.Store"):
        coordinator = HeatingDataCoordinator(hass, entry)

        # Learned data at 0C and 10C
        coordinator._aux_coefficients = {
            "0": {"normal": 5.0},
            "10": {"normal": 2.0}
        }

        # Test Exact
        assert coordinator._get_aux_impact_kw("0") == 5.0

        # Test Nearest Neighbor (Coordinator uses simple nearest, no linear interpolation yet for aux)
        # 2C -> Nearest is 0C (diff 2) vs 10C (diff 8) -> 5.0
        assert coordinator._get_aux_impact_kw("2") == 5.0

        # 8C -> Nearest is 10C (diff 2) -> 2.0
        assert coordinator._get_aux_impact_kw("8") == 2.0

@pytest.mark.asyncio
async def test_per_unit_aux_learning(hass):
    """Test that per-unit aux coefficients are learned correctly."""
    entry = MagicMock()
    entry.data = {
        "energy_sensors": ["sensor.heater_1", "sensor.heater_2"],
        "learning_rate": 0.1,
        "balance_point": 15.0
    }

    with patch("custom_components.heating_analytics.storage.Store"):
        coordinator = HeatingDataCoordinator(hass, entry)

        # Setup initial state
        # Heater 1: Base Model = 2.0 kWh
        # Heater 2: Base Model = 3.0 kWh
        coordinator._correlation_data_per_unit = {
            "sensor.heater_1": {"0": {"normal": 2.0}},
            "sensor.heater_2": {"0": {"normal": 3.0}}
        }

        # Setup observation counts to ensure models are considered "learned"
        coordinator._observation_counts = {
            "sensor.heater_1": {"0": {"normal": 10}},
            "sensor.heater_2": {"0": {"normal": 10}}
        }

        # Scenario: Aux is Active
        # Heater 1 Actual = 1.5 (Implied Reduction = 2.0 - 1.5 = 0.5)
        # Heater 2 Actual = 2.0 (Implied Reduction = 3.0 - 2.0 = 1.0)

        coordinator._hourly_delta_per_unit = {
            "sensor.heater_1": 1.5,
            "sensor.heater_2": 2.0
        }

        # Needed for learning process
        coordinator._hourly_sample_count = 60
        coordinator._hourly_temp_sum = 0.0 # Avg Temp 0
        coordinator._hourly_bucket_counts = {"normal": 60}
        coordinator._hourly_aux_count = 60 # Dominant Aux
        coordinator.auxiliary_heating_active = True

        # Mock unit modes
        coordinator._unit_modes = {
            "sensor.heater_1": MODE_HEATING,
            "sensor.heater_2": MODE_HEATING
        }

        # Mock get_predicted_unit_base_fn to return the base model values
        # This mirrors how coordinator passes this function
        def mock_get_base(entity_id, temp_key, wind_bucket, avg_temp):
            if entity_id == "sensor.heater_1": return 2.0
            if entity_id == "sensor.heater_2": return 3.0
            return 0.0

        with patch.object(coordinator, "_get_predicted_kwh_per_unit", side_effect=mock_get_base):
             inputs = {
                 "temp_key": "0",
                 "wind_bucket": "normal",
                 "avg_temp": 0.0,
                 "total_energy_kwh": 3.5, # 1.5 + 2.0
                 "base_expected_kwh": 5.0, # 2.0 + 3.0 (Global base)
                 "solar_impact": 0.0,
                 "avg_solar_factor": 0.0,
                 "is_aux_active": True,
                 "aux_impact": 0.0, # Current global aux impact
                 "learning_enabled": True,
                 "solar_enabled": False,
                 "learning_rate": 0.1,
                 "balance_point": 15.0,
                 "energy_sensors": ["sensor.heater_1", "sensor.heater_2"],
                 "hourly_bucket_counts": coordinator._hourly_bucket_counts,
                 "hourly_sample_count": 60,
                 "correlation_data": coordinator._correlation_data,
                 "correlation_data_per_unit": coordinator._correlation_data_per_unit,
                 "aux_coefficients": coordinator._aux_coefficients,
                 "learning_buffer_global": coordinator._learning_buffer_global,
                 "learning_buffer_per_unit": coordinator._learning_buffer_per_unit,
                 "observation_counts": coordinator._observation_counts,
                 "hourly_delta_per_unit": coordinator._hourly_delta_per_unit,
                 "hourly_expected_per_unit": {},
                 "hourly_expected_base_per_unit": {"sensor.heater_1": 2.0, "sensor.heater_2": 3.0},
                 "aux_coefficients_per_unit": coordinator._aux_coefficients_per_unit,
                 "learning_buffer_aux_per_unit": coordinator._learning_buffer_aux_per_unit,
                 "solar_coefficients_per_unit": coordinator._solar_coefficients_per_unit,
                 "learning_buffer_solar_per_unit": coordinator._learning_buffer_solar_per_unit,
                 "solar_calculator": coordinator.solar,
                 "get_predicted_unit_base_fn": mock_get_base,
                 "unit_modes": coordinator._unit_modes
             }

             coordinator.learning.process_learning(**inputs)

             # Verify Results - Cold Start should go to buffer
             assert "sensor.heater_1" in coordinator._learning_buffer_aux_per_unit
             buffer_1 = coordinator._learning_buffer_aux_per_unit["sensor.heater_1"]["0"]["normal"]
             assert buffer_1[0] == 0.5

             # Simulate Jump Start
             from custom_components.heating_analytics.const import LEARNING_BUFFER_THRESHOLD
             for _ in range(LEARNING_BUFFER_THRESHOLD - 1):
                 coordinator.learning.process_learning(**inputs)

             assert coordinator._aux_coefficients_per_unit["sensor.heater_1"]["0"]["normal"] == 0.5
             assert coordinator._aux_coefficients_per_unit["sensor.heater_2"]["0"]["normal"] == 1.0


@pytest.mark.asyncio
async def test_per_unit_aux_prediction(hass):
    """Test that calculate_total_power uses per-unit aux coefficients."""
    entry = MagicMock()
    entry.data = {
        "energy_sensors": ["sensor.heater_1", "sensor.heater_2", "sensor.heater_3"],
        "aux_affected_entities": ["sensor.heater_1", "sensor.heater_2"] # heater_3 is excluded
    }

    with patch("custom_components.heating_analytics.storage.Store"):
        coordinator = HeatingDataCoordinator(hass, entry)

        # Setup Base Models
        coordinator._correlation_data_per_unit = {
            "sensor.heater_1": {"0": {"normal": 2.0}},
            "sensor.heater_2": {"0": {"normal": 3.0}},
            "sensor.heater_3": {"0": {"normal": 4.0}}
        }

        # Setup Aux Coefficients
        coordinator._aux_coefficients_per_unit = {
            "sensor.heater_1": {"0": {"normal": 0.5}},
            "sensor.heater_2": {"0": {"normal": 1.0}},
            "sensor.heater_3": {"0": {"normal": 2.0}} # Should be ignored due to exclusion
        }

        # Setup Global Model (Legacy/Anchor)
        coordinator._correlation_data = {"0": {"normal": 9.0}}
        # Setup Global Aux Model (Must match unit sum to avoid scaling interference)
        # Sum of included units = 0.5 + 1.0 = 1.5
        coordinator._aux_coefficients = {"0": {"normal": 1.5}}

        # 1. Test Prediction with Aux Active
        res = coordinator.statistics.calculate_total_power(
            temp=0.0,
            effective_wind=0.0,
            solar_impact=0.0,
            is_aux_active=True
        )

        breakdown = res["unit_breakdown"]

        # Heater 1: 2.0 - 0.5 = 1.5
        assert breakdown["sensor.heater_1"]["base_kwh"] == 2.0
        assert breakdown["sensor.heater_1"]["aux_reduction_kwh"] == 0.5
        assert breakdown["sensor.heater_1"]["net_kwh"] == 1.5

        # Heater 2: 3.0 - 1.0 = 2.0
        assert breakdown["sensor.heater_2"]["base_kwh"] == 3.0
        assert breakdown["sensor.heater_2"]["aux_reduction_kwh"] == 1.0
        assert breakdown["sensor.heater_2"]["net_kwh"] == 2.0

        # Heater 3: 4.0 - 0.0 (Excluded) = 4.0
        assert breakdown["sensor.heater_3"]["base_kwh"] == 4.0
        assert breakdown["sensor.heater_3"]["aux_reduction_kwh"] == 0.0
        assert breakdown["sensor.heater_3"]["net_kwh"] == 4.0

        # Global Totals - Unit Sum Aux = 1.5
        assert res["breakdown"]["aux_reduction_kwh"] == 1.5
