#!/usr/bin/env python3
"""Verify that solar impact minutes are counted correctly."""

# Simulate solar impact calculation

# Constants (from const.py)
DEFAULT_SOLAR_COEFF_HEATING = 0.35

def calculate_solar_impact_kw(solar_factor: float, coeff: float) -> float:
    """Calculate solar impact in kW (same as in solar.py)."""
    # Formula: Impact = Global_Factor * Unit_Coeff
    return solar_factor * coeff

# Test parameters (based on user's setup)
coeff = DEFAULT_SOLAR_COEFF_HEATING  # 0.15
num_minutes = 60

# Simulate user's scenario with 71.7% solar impact
percentage = 71.7  # From user's data
reported_solar_kwh = 0.59  # From user's data (Legacy/Example value, likely high if using old formula)

# Calculate what solar_factor would give this impact
# If percentage = 71.7%, then roughly 71.7% of minutes had solar impact
# Let's assume solar_factor = 0.7 during sunny minutes, 0.0 during cloudy

sunny_minutes = int(num_minutes * (percentage / 100.0))
cloudy_minutes = num_minutes - sunny_minutes

print(f"=== Solar Impact Minute Counting Verification ===")
print(f"\nUser's reported data:")
print(f"  Percentage: {percentage}%")
print(f"  Last hour solar impact kwh: {reported_solar_kwh}")
print(f"\nSimulation with {num_minutes} minutes:")
print(f"  Sunny minutes: {sunny_minutes}")
print(f"  Cloudy minutes: {cloudy_minutes}")

# Method 1: Minute-by-minute accumulation
total_impact_kwh_method1 = 0.0
total_factor_method1 = 0.0

for minute in range(num_minutes):
    if minute < sunny_minutes:
        solar_factor = 0.7  # Sunny
    else:
        solar_factor = 0.0  # Cloudy

    impact_kw = calculate_solar_impact_kw(solar_factor, coeff)
    impact_kwh_this_minute = impact_kw / 60.0  # Convert to kWh for this minute
    total_impact_kwh_method1 += impact_kwh_this_minute
    total_factor_method1 += solar_factor

avg_factor_method1 = total_factor_method1 / num_minutes

print(f"\n=== Method 1: Minute-by-minute accumulation ===")
print(f"  Average solar factor: {avg_factor_method1:.3f}")
print(f"  Total solar impact: {total_impact_kwh_method1:.3f} kWh")

# Method 2: Current code (average factor, then calculate)
avg_solar_factor = total_factor_method1 / num_minutes
impact_kw_avg = calculate_solar_impact_kw(avg_solar_factor, coeff)
total_impact_kwh_method2 = impact_kw_avg * 1.0  # 1 hour

print(f"\n=== Method 2: Current code (avg factor) ===")
print(f"  Average solar factor: {avg_solar_factor:.3f}")
print(f"  Solar impact kW (at avg): {impact_kw_avg:.3f} kW")
print(f"  Total solar impact: {total_impact_kwh_method2:.3f} kWh")

print(f"\n=== Comparison ===")
print(f"  Method 1 (minute-by-minute): {total_impact_kwh_method1:.3f} kWh")
print(f"  Method 2 (current code):     {total_impact_kwh_method2:.3f} kWh")
print(f"  Difference:                  {abs(total_impact_kwh_method1 - total_impact_kwh_method2):.6f} kWh")
print(f"  Methods match: {abs(total_impact_kwh_method1 - total_impact_kwh_method2) < 0.001}")

# Now let's reverse-engineer what coeff would give 0.59 kWh
target_kwh = reported_solar_kwh
# target = avg_factor * coeff * 1.0
required_coeff = target_kwh / avg_solar_factor if avg_solar_factor > 0 else 0

print(f"\n=== Reverse Engineering ===")
print(f"  To get {target_kwh} kWh with avg_factor={avg_solar_factor:.3f}:")
print(f"  Required coefficient: {required_coeff:.3f}")
print(f"  This is {required_coeff / coeff:.1f}x the default ({coeff})")
