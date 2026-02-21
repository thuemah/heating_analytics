"""Golden Master Test Suite for Heating Analytics.

This test ensures that the core logic (Kelvin Protocol) remains consistent
with historical data snapshots. It verifies "Sum of Parts == Whole" and
daily aggregation logic against a known "Golden" dataset.
"""
from unittest.mock import MagicMock, AsyncMock, patch
import pytest
from datetime import date, datetime
from homeassistant.core import HomeAssistant
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.const import (
    MODE_HEATING, MODE_OFF, MODE_GUEST_HEATING
)

# --- Anonymized Golden Data (2026-01-24) ---
# Derived from real production data, anonymized for privacy.
# Scenario: Cold day (-5C), High Wind, Aux Active, Guest Mode Active.

GOLDEN_MAPPING = {
    "sensor.mitsubishi_vp_2_etasje": "sensor.unit_a",
    "sensor.ikea_of_sweden_inspelning_smart_plug_summation_delivered_2": "sensor.unit_b",
    "sensor.socket_garasje_daglig_forbruk": "sensor.unit_c",
    "sensor.toshiba_vp_stue_energi": "sensor.unit_d",
    "sensor.vaskerom_energiforbruk": "sensor.unit_e",
    "sensor.bad_varmekabel_energiforbruk": "sensor.unit_f",
    "sensor.termostat_kjokken_electric_consumption_kwh_4": "sensor.unit_g",
    "sensor.socket_yaser_daglig_forbruk": "sensor.unit_h",
    "sensor.vp_kjeller_energiforbruk": "sensor.unit_i", # Guest Mode Unit
    "sensor.vp_kjokken_energiforbruk": "sensor.unit_j",
    "sensor.bad_kjeller_varmekabel_energiforbruk": "sensor.unit_k"
}

GOLDEN_DAILY_SUMMARY = {
    "date": "2026-01-24",
    "kwh": 33.8,
    "expected_kwh": 30.84,
    "forecasted_kwh": 30.5,
    "aux_impact_kwh": 12.3,
    "solar_impact_kwh": 0.23,
    "guest_impact_kwh": 5.95,
    "tdd": 21.9,
    "temp": -4.9,
    "wind": 5.6,
    "solar_factor": 0.006,
}

# Condensed logs (Focusing on critical transitions)
# We reconstruct the full day list in the test setup.
GOLDEN_HOURLY_LOGS_SOURCE = [
    {
        "timestamp": "2026-01-23T23:00:00+01:00",
        "hour": 23,
        "temp": -4.8,
        "effective_wind": 6.86,
        "actual_kwh": 0.635,
        "unit_breakdown": {"sensor.unit_a": 0.39, "sensor.unit_b": 0.112, "sensor.unit_d": 0.133},
        "auxiliary_active": True,
        "aux_impact_kwh": 0.983,
        "unit_modes": {"sensor.unit_j": "off", "sensor.unit_i": "off", "sensor.unit_k": "off"}
    },
    {
        "timestamp": "2026-01-24T00:00:00+01:00",
        "hour": 0,
        "temp": -4.8,
        "effective_wind": 6.67,
        "actual_kwh": 0.866,
        "auxiliary_active": True,
        "aux_impact_kwh": 0.831,
        "guest_impact_kwh": 0.0,
        "unit_modes": {"sensor.unit_j": "off", "sensor.unit_i": "off", "sensor.unit_k": "off"}
    },
    {
        "timestamp": "2026-01-24T01:00:00+01:00",
        "hour": 1,
        "temp": -4.8,
        "effective_wind": 6.56,
        "actual_kwh": 1.411,
        "auxiliary_active": False,
        "aux_impact_kwh": 0.0,
        "guest_impact_kwh": 0.0,
        "unit_modes": {"sensor.unit_j": "off", "sensor.unit_i": "off", "sensor.unit_k": "off"}
    },
    # ... (Skipping 02-07 for brevity, logic fills gaps if needed, but for aggregation we need all)
    # Adding simplified entries for 02-07 to match totals
    # 02: 1.468, 03: 1.391, 04: 2.366, 05: 1.831, 06: 1.939, 07: 1.643
    {"timestamp": "2026-01-24T02:00:00+01:00", "hour": 2, "temp": -4.8, "effective_wind": 6.22, "actual_kwh": 1.468, "auxiliary_active": False, "aux_impact_kwh": 0.0, "guest_impact_kwh": 0.0},
    {"timestamp": "2026-01-24T03:00:00+01:00", "hour": 3, "temp": -4.9, "effective_wind": 5.80, "actual_kwh": 1.391, "auxiliary_active": False, "aux_impact_kwh": 0.0, "guest_impact_kwh": 0.0},
    {"timestamp": "2026-01-24T04:00:00+01:00", "hour": 4, "temp": -5.1, "effective_wind": 5.61, "actual_kwh": 2.366, "auxiliary_active": False, "aux_impact_kwh": 0.0, "guest_impact_kwh": 0.0},
    {"timestamp": "2026-01-24T05:00:00+01:00", "hour": 5, "temp": -5.4, "effective_wind": 5.66, "actual_kwh": 1.831, "auxiliary_active": False, "aux_impact_kwh": 0.0, "guest_impact_kwh": 0.0},
    {"timestamp": "2026-01-24T06:00:00+01:00", "hour": 6, "temp": -5.4, "effective_wind": 5.66, "actual_kwh": 1.939, "auxiliary_active": False, "aux_impact_kwh": 0.0, "guest_impact_kwh": 0.0},
    {"timestamp": "2026-01-24T07:00:00+01:00", "hour": 7, "temp": -5.5, "effective_wind": 5.42, "actual_kwh": 1.643, "auxiliary_active": False, "aux_impact_kwh": 0.0, "guest_impact_kwh": 0.0},

    # Critical: Aux Active Start
    {
        "timestamp": "2026-01-24T08:00:00+01:00",
        "hour": 8,
        "temp": -5.5,
        "effective_wind": 5.38,
        "actual_kwh": 1.183,
        "auxiliary_active": True,
        "aux_impact_kwh": 0.804,
        "guest_impact_kwh": 0.0,
        "solar_factor": 0.004,
        "solar_impact_kwh": 0.006,
        "unit_modes": {"sensor.unit_j": "off", "sensor.unit_i": "off", "sensor.unit_k": "off"}
    },
    {
        "timestamp": "2026-01-24T09:00:00+01:00",
        "hour": 9,
        "temp": -5.5,
        "effective_wind": 5.0,
        "actual_kwh": 0.862,
        "auxiliary_active": True,
        "aux_impact_kwh": 0.862,
        "guest_impact_kwh": 0.0,
        "solar_factor": 0.033,
        "solar_impact_kwh": 0.054
    },
    {
        "timestamp": "2026-01-24T10:00:00+01:00",
        "hour": 10,
        "temp": -5.4,
        "effective_wind": 5.36,
        "actual_kwh": 1.038,
        "auxiliary_active": True,
        "aux_impact_kwh": 0.887,
        "guest_impact_kwh": 0.0,
        "solar_factor": 0.025,
        "solar_impact_kwh": 0.041
    },
    # 11-12
    {"timestamp": "2026-01-24T11:00:00+01:00", "hour": 11, "temp": -4.9, "effective_wind": 5.99, "actual_kwh": 0.883, "auxiliary_active": True, "aux_impact_kwh": 1.003, "guest_impact_kwh": 0.0},
    {"timestamp": "2026-01-24T12:00:00+01:00", "hour": 12, "temp": -4.4, "effective_wind": 6.16, "actual_kwh": 0.974, "auxiliary_active": True, "aux_impact_kwh": 0.739, "guest_impact_kwh": 0.0},

    # Critical: Guest Mode Start (Unit I)
    {
        "timestamp": "2026-01-24T13:00:00+01:00",
        "hour": 13,
        "temp": -4.4,
        "effective_wind": 6.24,
        "actual_kwh": 1.025,
        "auxiliary_active": True,
        "aux_impact_kwh": 1.008,
        "guest_impact_kwh": 0.33,
        "solar_factor": 0.021,
        "solar_impact_kwh": 0.034,
        "unit_breakdown": {"sensor.unit_i": 0.33},
        "unit_modes": {
            "sensor.unit_j": "off",
            "sensor.unit_i": "guest_heating", # Guest Mode Active
            "sensor.unit_k": "off"
        }
    },
    {
        "timestamp": "2026-01-24T14:00:00+01:00",
        "hour": 14,
        "temp": -4.3,
        "effective_wind": 6.24,
        "actual_kwh": 1.402,
        "auxiliary_active": True,
        "aux_impact_kwh": 0.90,
        "guest_impact_kwh": 0.45,
        "solar_factor": 0.034,
        "solar_impact_kwh": 0.056,
        "unit_breakdown": {"sensor.unit_i": 0.45},
        "unit_modes": {"sensor.unit_i": "guest_heating"}
    },
    # 15-20 (Guest + Aux)
    {"timestamp": "2026-01-24T15:00:00+01:00", "hour": 15, "temp": -4.5, "effective_wind": 5.74, "actual_kwh": 1.244, "auxiliary_active": True, "aux_impact_kwh": 0.879, "guest_impact_kwh": 0.45, "solar_factor": 0.019, "solar_impact_kwh": 0.031, "unit_modes": {"sensor.unit_i": "guest_heating"}},
    {"timestamp": "2026-01-24T16:00:00+01:00", "hour": 16, "temp": -4.7, "effective_wind": 4.91, "actual_kwh": 1.673, "auxiliary_active": True, "aux_impact_kwh": 0.85, "guest_impact_kwh": 0.50, "solar_factor": 0.002, "solar_impact_kwh": 0.003, "unit_modes": {"sensor.unit_i": "guest_heating"}},
    {"timestamp": "2026-01-24T17:00:00+01:00", "hour": 17, "temp": -4.7, "effective_wind": 5.04, "actual_kwh": 1.312, "auxiliary_active": True, "aux_impact_kwh": 0.833, "guest_impact_kwh": 0.60, "unit_modes": {"sensor.unit_i": "guest_heating"}},
    {"timestamp": "2026-01-24T18:00:00+01:00", "hour": 18, "temp": -4.8, "effective_wind": 5.14, "actual_kwh": 1.239, "auxiliary_active": True, "aux_impact_kwh": 0.67, "guest_impact_kwh": 0.50, "unit_modes": {"sensor.unit_i": "guest_heating"}},
    {"timestamp": "2026-01-24T19:00:00+01:00", "hour": 19, "temp": -4.8, "effective_wind": 5.14, "actual_kwh": 1.357, "auxiliary_active": True, "aux_impact_kwh": 0.837, "guest_impact_kwh": 0.76, "unit_modes": {"sensor.unit_i": "guest_heating"}},
    {"timestamp": "2026-01-24T20:00:00+01:00", "hour": 20, "temp": -4.7, "effective_wind": 4.99, "actual_kwh": 1.228, "auxiliary_active": True, "aux_impact_kwh": 0.837, "guest_impact_kwh": 0.65, "unit_modes": {"sensor.unit_i": "guest_heating"}},

    # 21-23 (Guest Only, Aux Off)
    {
        "timestamp": "2026-01-24T21:00:00+01:00",
        "hour": 21,
        "temp": -4.8,
        "effective_wind": 4.88,
        "actual_kwh": 1.526,
        "auxiliary_active": False,
        "aux_impact_kwh": 0.363, # Residual/Partial
        "guest_impact_kwh": 0.57,
        "unit_modes": {"sensor.unit_i": "guest_heating"}
    },
    {
        "timestamp": "2026-01-24T22:00:00+01:00",
        "hour": 22,
        "temp": -4.8,
        "effective_wind": 4.76,
        "actual_kwh": 1.97,
        "auxiliary_active": False,
        "aux_impact_kwh": 0.0,
        "guest_impact_kwh": 0.62,
        "unit_modes": {"sensor.unit_i": "guest_heating"}
    },
    {
        "timestamp": "2026-01-24T23:00:00+01:00",
        "hour": 23,
        "temp": -4.8,
        "effective_wind": 5.19,
        "actual_kwh": 1.966,
        "auxiliary_active": False,
        "aux_impact_kwh": 0.0,
        "guest_impact_kwh": 0.52,
        "unit_modes": {"sensor.unit_i": "guest_heating"}
    },
]

@pytest.fixture
def coordinator(hass: HomeAssistant):
    """Fixture to create a coordinated loaded with Golden Data."""
    entry = MagicMock()
    entry.data = {
        "balance_point": 17.0,
        "learning_rate": 0.03,
        "solar_correction_percent": 100
    }

    with patch("custom_components.heating_analytics.storage.Store") as mock_store:
        mock_store.return_value.async_load = AsyncMock(return_value={})
        mock_store.return_value.async_save = AsyncMock()

        coord = HeatingDataCoordinator(hass, entry)
        # Mock storage to prevent side effects
        coord._async_save_data = AsyncMock()
        coord.storage.async_load_data = AsyncMock()

        # Populate Sensors
        coord.energy_sensors = list(GOLDEN_MAPPING.values())
        coord.aux_affected_entities = list(GOLDEN_MAPPING.values()) # Default to all
        coord._correlation_data_per_unit = {s: {} for s in coord.energy_sensors}

        # Populate Hourly Logs
        # Ensure 'unit_modes' defaults are populated if missing in compact logs
        full_logs = []
        for log in GOLDEN_HOURLY_LOGS_SOURCE:
            enriched = log.copy()
            if "unit_modes" not in enriched:
                enriched["unit_modes"] = {}
            if "solar_factor" not in enriched:
                enriched["solar_factor"] = 0.0
            if "unit_breakdown" not in enriched:
                enriched["unit_breakdown"] = {}
            if "tdd" not in enriched:
                enriched["tdd"] = round(abs(17.0 - log["temp"]) / 24.0, 3)
            if "wind_bucket" not in enriched:
                enriched["wind_bucket"] = coord._get_wind_bucket(log["effective_wind"])

            full_logs.append(enriched)

        coord._hourly_log = full_logs

        return coord

@pytest.mark.asyncio
async def test_daily_aggregation_integrity(coordinator):
    """Verify that daily aggregation matches the golden master summary."""
    # Run aggregation for the target date
    start_date = date.fromisoformat("2026-01-24")
    end_date = start_date # One day

    # We use calculate_modeled_energy to verify aggregation,
    # but we also need to check if the 'calculated' totals (which sum up actuals/aux/guest)
    # match the expectation.

    # Manually trigger the daily aggregation logic (as if day just ended)
    # The coordinator._aggregate_daily_logs logic is what we want to test.
    # Filter logs for the day
    day_logs = [l for l in coordinator._hourly_log if l["timestamp"].startswith("2026-01-24")]

    daily_stats = coordinator._aggregate_daily_logs(day_logs)

    # 1. Verify Total kWh
    # Floating point tolerance required
    assert daily_stats["kwh"] == pytest.approx(GOLDEN_DAILY_SUMMARY["kwh"], abs=0.01)

    # 2. Verify Aux Impact
    assert daily_stats["aux_impact_kwh"] == pytest.approx(GOLDEN_DAILY_SUMMARY["aux_impact_kwh"], abs=0.01)

    # 3. Verify Guest Impact
    assert daily_stats["guest_impact_kwh"] == pytest.approx(GOLDEN_DAILY_SUMMARY["guest_impact_kwh"], abs=0.01)

    # 4. Verify Solar Impact
    # Note: Source has 0.23, lets verify
    assert daily_stats["solar_impact_kwh"] == pytest.approx(GOLDEN_DAILY_SUMMARY["solar_impact_kwh"], abs=0.01)

    # 5. Verify TDD
    assert daily_stats["tdd"] == pytest.approx(GOLDEN_DAILY_SUMMARY["tdd"], abs=0.1)

    # 6. Verify Temp/Wind Averages
    assert daily_stats["temp"] == pytest.approx(GOLDEN_DAILY_SUMMARY["temp"], abs=0.1)
    assert daily_stats["wind"] == pytest.approx(GOLDEN_DAILY_SUMMARY["wind"], abs=0.1)

@pytest.mark.asyncio
async def test_guest_mode_accounting(coordinator):
    """Verify Guest Mode isolation in aggregation."""
    # Focus on the hours where guest mode was active (13:00 onwards)
    guest_logs = [l for l in coordinator._hourly_log if "guest_heating" in l.get("unit_modes", {}).values()]

    # Verify we found the expected number of hours (13 to 23 = 11 hours)
    assert len(guest_logs) == 11

    # Manual sum from golden logs to verify test setup
    expected_guest_sum = sum(l["guest_impact_kwh"] for l in guest_logs)
    assert expected_guest_sum == pytest.approx(5.95, abs=0.01)

    # Verify that the coordinator logic correctly attributes this
    # We'll check the daily stats again specifically for this
    day_logs = [l for l in coordinator._hourly_log if l["timestamp"].startswith("2026-01-24")]
    daily_stats = coordinator._aggregate_daily_logs(day_logs)

    assert daily_stats["guest_impact_kwh"] == pytest.approx(expected_guest_sum, abs=0.01)

@pytest.mark.asyncio
async def test_hourly_distribution_logic(coordinator):
    """Verify Kelvin Protocol distribution logic with Guest Mode active.

    Scenario:
    - High Wind, Aux Active, Solar Active.
    - One unit (Unit I) in Guest Mode.

    Goal:
    - Verify calculate_total_power includes ALL units (including guest) in breakdown
    - Verify Kelvin Protocol correctly scales aux reduction
    - Verify global total matches sum of unit predictions

    Note: Guest Mode accounting happens in _process_hourly_data (guest_impact_kwh),
    not in calculate_total_power. The physics engine calculates what each unit does,
    and the coordinator separates guest consumption from normal operation.
    """
    # 1. Setup specific hour inputs
    temp = -4.3
    wind = 6.24  # high_wind
    aux_active = True
    solar_factor = 0.034

    # Unit modes for 14:00: Unit I is guest_heating, others default (heating)
    unit_modes = {s: MODE_HEATING for s in coordinator.energy_sensors}
    unit_modes["sensor.unit_i"] = MODE_GUEST_HEATING

    # 2. Mock Model Predictions
    # Global Model (Aux-Unaware)
    global_base_prediction = 2.0

    # Global Aux Reduction (Total)
    global_aux_reduction = 0.5

    # Unit Base Predictions (Sum = 2.0)
    # Unit I (Guest) has 0.5 base, remaining 10 units have 1.5 distributed (0.15 each)
    unit_base_predictions = {s: 0.15 for s in coordinator.energy_sensors}
    unit_base_predictions["sensor.unit_i"] = 0.5

    # Unit Aux Reductions (Sum = 0.5)
    # Unit I (Guest) has 0.1 aux, remaining 10 units have 0.4 distributed (0.04 each)
    unit_aux_reductions = {s: 0.04 for s in coordinator.energy_sensors}
    unit_aux_reductions["sensor.unit_i"] = 0.1

    # Populate coordinator data structures with unique identifiers
    # (Required for mock side_effect to distinguish between units)
    coordinator._correlation_data_per_unit = {s: {"id": s, "type": "base"} for s in coordinator.energy_sensors}
    coordinator._aux_coefficients_per_unit = {s: {"id": s, "type": "aux"} for s in coordinator.energy_sensors}
    coordinator._aux_coefficients = {"type": "global_aux"}

    # 3. Mock prediction methods
    def side_effect_get_pred(data_map, temp_key, wind_bucket, actual_temp, balance_point, apply_scaling=True):
        # Check if global aux
        if data_map.get("type") == "global_aux":
            return global_aux_reduction

        # Check unit base predictions
        for eid, map_ in coordinator._correlation_data_per_unit.items():
            if data_map == map_:
                return unit_base_predictions.get(eid, 0.0)

        # Check unit aux reductions
        for eid, map_ in coordinator._aux_coefficients_per_unit.items():
            if data_map == map_:
                return unit_aux_reductions.get(eid, 0.0)

        return 0.0

    with patch.object(coordinator, "_get_predicted_kwh", return_value=global_base_prediction), \
         patch.object(coordinator.statistics, "_get_prediction_from_model", side_effect=side_effect_get_pred):

        # 4. Run Calculation
        result = coordinator.statistics.calculate_total_power(
            temp=temp,
            effective_wind=wind,
            solar_impact=0.0,  # Unused
            is_aux_active=aux_active,
            unit_modes=unit_modes,
            override_solar_factor=solar_factor
        )

        # 5. Assertions

        # A. Verify Guest Mode unit IS in breakdown (physics engine calculates all units)
        breakdown = result["unit_breakdown"]
        assert "sensor.unit_i" in breakdown

        # B. Verify Guest Mode unit has correct base_kwh
        assert breakdown["sensor.unit_i"]["base_kwh"] == 0.5

        # C. Verify Kelvin Protocol scaling
        # Global Aux: 0.5, Sum Unit Aux Raw: 0.1 + (10*0.04) = 0.5
        # Scale = 0.5 / 0.5 = 1.0
        # Unit I Net = 0.5 - 0.1 = 0.4 (before solar correction)
        assert breakdown["sensor.unit_i"]["aux_reduction_kwh"] == pytest.approx(0.1, abs=0.001)

        # D. Verify global totals match
        # Global Base = 2.0, Global Aux = 0.5
        assert result["global_base_kwh"] == 2.0
        assert result["global_aux_reduction_kwh"] == 0.5

        # E. Verify Sum of Parts == Whole (Kelvin Protocol)
        # Total net should be close to global_net (allowing for solar adjustments and rounding)
        total_net_from_units = sum(u["net_kwh"] for u in breakdown.values())

        # DEBUG: Print breakdown for diagnosis
        print(f"\n=== DEBUG: Kelvin Protocol Test ===")
        print(f"Total units in breakdown: {len(breakdown)}")
        print(f"Unit I in breakdown: {'sensor.unit_i' in breakdown}")
        if "sensor.unit_i" in breakdown:
            print(f"Unit I breakdown: {breakdown['sensor.unit_i']}")
        print(f"Sum of unit net_kwh: {total_net_from_units}")
        print(f"Global total_kwh: {result['total_kwh']}")
        print(f"Difference: {abs(total_net_from_units - result['total_kwh'])}")

        # Global net = base - aux + solar = 2.0 - 0.5 + solar_effect
        # With solar_factor=0.034 and all units heating, solar is a reduction
        # We allow tolerance for solar calculations
        assert total_net_from_units == pytest.approx(result["total_kwh"], abs=0.1)
        assert breakdown["sensor.unit_i"]["aux_reduction_kwh"] == 0.1
