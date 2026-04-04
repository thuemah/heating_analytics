"""Test for the ThermodynamicEngine (Track C) implementation."""
import sys
import os

# Add custom_components to sys.path to resolve imports without running HA
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from custom_components.heating_analytics.thermodynamics import ThermodynamicEngine

def test_midnight_sync_smearing():
    """Test that total thermal energy is correctly smeared according to theoretical loss."""
    engine = ThermodynamicEngine()

    # Generate 24h of MPC Data
    # Heat pump only runs at night (e.g., 00:00 to 06:00) due to load shifting
    mpc_data = []
    for h in range(24):
        if h < 6:
            # Heat pump running full blast at night
            kwh_th = 4.0  # Delivered 4.0 kWh thermal
            kwh_el = 1.0  # Consumed 1.0 kWh electrical (COP = 4.0)
            mode = "sh"
        elif h == 14:
            # Heat pump makes hot water at 14:00
            kwh_th = 0.0  # 0.0 thermal delivered to house (DHW)
            kwh_el = 2.0  # 2.0 electrical consumed by DHW (Not Space Heating!)
            mode = "dhw"
        else:
            # Heat pump is OFF the rest of the day
            kwh_th = 0.0
            kwh_el = 0.0
            mode = "off"

        mpc_data.append({
            "datetime": f"2026-03-29T{h:02d}:00:00+01:00",
            "kwh_th_sh": kwh_th,
            "kwh_el_sh": kwh_el,
            "mode": mode
        })

    # Generate 24h of Weather Data
    # Let's say it's colder at night and slightly warmer in the day
    # Delta-T = (Indoor - Outdoor), e.g., 20C - 10C = 10
    weather_data = []
    for h in range(24):
        # Base loss
        delta_t = 15.0 if h < 8 else 10.0  # Colder night, milder day
        wind_factor = 1.0
        solar_factor = 1.0  # No sun initially

        # At 14:00 (when DHW is running), it's milder
        if h == 14:
             delta_t = 8.0

        weather_data.append({
            "datetime": f"2026-03-29T{h:02d}:00:00+01:00",
            "delta_t": delta_t,
            "wind_factor": wind_factor,
            "solar_factor": solar_factor
        })

    # Execute Midnight Sync
    distribution = engine.calculate_synthetic_baseline(mpc_data, weather_data)

    # Assertions
    # 1. Total thermal delivered = 6 hours * 4.0 = 24.0 kWh
    # 2. Total electrical consumed = 6 hours * 1.0 = 6.0 kWh (Excluding DHW)
    # 3. Daily Average COP = 24.0 / 6.0 = 4.0

    # Let's check hour 0 (Cold night, SH running)
    assert distribution[0]["mode"] == "sh"
    # Smeared thermal should be proportional.
    # Total weights = (8 hours * 15.0) + (15 hours * 10.0) + (1 hour * 8.0) = 120 + 150 + 8 = 278
    # Weight for hour 0 is 15.0
    # Expected smeared_th = 24.0 * (15.0 / 278) = 1.2949... -> ~1.295
    assert abs(distribution[0]["smeared_kwh_th"] - 1.295) < 0.01

    # Synthetic electrical = smeared_th / COP
    # COP = 4.0
    # Expected synthetic_el = 1.295 / 4.0 = 0.3237... -> ~0.324
    assert abs(distribution[0]["synthetic_kwh_el"] - 0.324) < 0.01

    # Let's check hour 14 (Mild day, DHW running)
    assert distribution[14]["mode"] == "dhw"
    # Even though DHW was running, house still lost heat based on weather
    # Weight for hour 14 is 8.0
    # Expected smeared_th = 24.0 * (8.0 / 278) = 0.6906... -> ~0.691
    assert abs(distribution[14]["smeared_kwh_th"] - 0.691) < 0.01

    # Synthetic electrical for DHW hour (What it WOULD have cost to heat the house at this hour)
    # Expected synthetic_el = 0.691 / 4.0 = 0.1726... -> ~0.173
    assert abs(distribution[14]["synthetic_kwh_el"] - 0.173) < 0.01

    print("Midnight Sync Smearing Test Passed!")


def test_zero_division_guard():
    """Test that Daily Average COP falls back to 1.0 if electrical is 0."""
    engine = ThermodynamicEngine()

    # Heat pump OFF all day
    mpc_data = []
    weather_data = []
    for h in range(24):
        mpc_data.append({
            "datetime": f"2026-03-29T{h:02d}:00:00+01:00",
            "kwh_th_sh": 0.0,
            "kwh_el_sh": 0.0,
            "mode": "off"
        })
        weather_data.append({
            "datetime": f"2026-03-29T{h:02d}:00:00+01:00",
            "delta_t": 10.0,
            "wind_factor": 1.0,
            "solar_factor": 1.0
        })

    # The code should NOT crash with ZeroDivisionError
    distribution = engine.calculate_synthetic_baseline(mpc_data, weather_data)

    for h in range(24):
         # Smeared thermal is 0.0, Synthetic electrical is 0.0
         assert distribution[h]["smeared_kwh_th"] == 0.0
         assert distribution[h]["synthetic_kwh_el"] == 0.0

    print("Zero Division Guard Test Passed!")

def test_empty_weather_fallback():
    """Test that missing weather data gracefully falls back without IndexError."""
    engine = ThermodynamicEngine()

    mpc_data = [{
        "datetime": "2026-03-29T12:00:00+01:00",
        "kwh_th_sh": 5.0,
        "kwh_el_sh": 1.0,
        "mode": "sh"
    }]
    weather_data = [] # Empty!

    # Should not crash with IndexError
    distribution = engine.calculate_synthetic_baseline(mpc_data, weather_data)

    assert len(distribution) == 1
    assert distribution[0]["smeared_kwh_th"] == 5.0
    assert distribution[0]["synthetic_kwh_el"] == 1.0

    print("Empty Weather Fallback Test Passed!")


if __name__ == "__main__":
    test_midnight_sync_smearing()
    test_zero_division_guard()
