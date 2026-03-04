#!/usr/bin/env python3
"""Calculate correct solar coefficient for heat pump setup."""

# Constants (from const.py)
DEFAULT_SOLAR_COEFF_HEATING = 0.35

print("=== Kalkulator: Solar Coefficient for Varmepumpe ===\n")

# User's heat pump COP (estimated from data)
cop_from_data = 3.11

# Alternative: User specifies their actual COP
print("Din varmepumpe COP:")
print(f"  Estimert fra data: {cop_from_data:.2f}")
print("  Typiske verdier:")
print("    - Luft/luft: 2.5 - 3.5")
print("    - Luft/vann: 2.0 - 3.0")
print("    - Bergvarme: 3.0 - 4.5")

# Let's calculate for different COPs
cops = [2.5, 3.0, 3.5, cop_from_data]

print(f"\n=== Anbefalte Solar Coefficients ===")
print(f"Default (uten varmepumpe): 0.35 (gammel default)")
print(f"Default (heat pump friendly): {DEFAULT_SOLAR_COEFF_HEATING}")
print()

# Note: The logic here is tricky.
# DEFAULT_SOLAR_COEFF_HEATING (0.15) IS ALREADY OPTIMIZED FOR COP ~2.5.
# So dividing it by COP *again* would result in double counting if we assume 0.15 is the "COP adjusted" value.
#
# If the user finds 0.15 is still too high, it means their house/windows/shading
# results in less solar gain than the "Average House" model assumes.
# OR their COP is higher.
#
# But for the sake of this script, let's assume we want to adjust based on SPECIFIC COP
# relative to the BASE model (Direct Electric = 1.0).
# Base Coefficient ~ 0.35 (Empirical max for 10m2 windows).
#
# So Target = Base / COP.

BASE_COEFF = 0.35

for cop in cops:
    corrected_coeff = BASE_COEFF / cop
    print(f"COP {cop:.2f}: {corrected_coeff:.4f}")

# Specific recommendation based on data
recommended_coeff = BASE_COEFF / cop_from_data
print(f"\n=== Anbefaling basert på dine data ===")
print(f"COP: {cop_from_data:.2f}")
print(f"Anbefalt coefficient: {recommended_coeff:.4f}")
print(f"(avrundet: {round(recommended_coeff, 4)})")

print(f"\n=== Hvordan justere ===")
print(f"1. Finn din storage fil:")
print(f"   ~/.homeassistant/.storage/heating_analytics_data.json")
print(f"   (eller der Home Assistant lagrer data)")
print(f"")
print(f"2. Stopp Home Assistant")
print(f"")
print(f"3. Rediger filen, finn 'solar_coefficients_per_unit' seksjon:")
print(f'   "solar_coefficients_per_unit": {{')
print(f'     "sensor.din_varmepumpe": {{')
print(f'       "0": {recommended_coeff:.4f},')
print(f'       "-1": {recommended_coeff:.4f},')
print(f'       ... (alle temp_keys du har)')
print(f'     }}')
print(f'   }}')
print(f"")
print(f"4. Start Home Assistant")
print(f"")
print(f"ALTERNATIVT: La modellen lære (~19 dager)")

# What if we want to adjust just the temp_key = 0 (which is likely the current one)?
print(f"\n=== Quick Fix (kun én temperatur) ===")
print(f"Hvis du kun vil justere for gjeldende temp_key (0°C):")
print(f"Sett coefficient['0'] = {recommended_coeff:.4f}")
print(f"\nDette vil gi raskere læring for andre temperaturer,")
print(f"siden de kan lære fra scratch med riktigere startverdier.")
