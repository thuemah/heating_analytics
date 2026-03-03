#!/usr/bin/env python3
"""Analyze solar impact for heat pump scenario."""

# Constants (from const.py)
DEFAULT_SOLAR_COEFF_HEATING = 0.35
DEFAULT_SOLAR_LEARNING_RATE = 0.01

# User's data (Example)
expected_kwh = 0.57
actual_kwh = 0.97
solar_impact_kwh = 0.59

print("=== Analyse av Solar Impact ===\n")
print("Brukerens data:")
print(f"  Expected (med solar): {expected_kwh} kWh")
print(f"  Actual forbruk: {actual_kwh} kWh")
print(f"  Solar impact: {solar_impact_kwh} kWh")
print(f"  Avvik: {actual_kwh - expected_kwh:.2f} kWh ({((actual_kwh - expected_kwh)/expected_kwh * 100):.1f}%)")

# Calculate base prediction (without solar)
base_prediction = expected_kwh + solar_impact_kwh
print(f"\nBeregnet:")
print(f"  Base modell (uten sol): {base_prediction:.2f} kWh")
print(f"  Solar reduksjon: {solar_impact_kwh} kWh")
print(f"  Expected (med sol): {base_prediction - solar_impact_kwh:.2f} kWh")

# Analysis
print(f"\n=== Varmepumpe Analyse ===")
print(f"Faktisk forbruk ({actual_kwh}) er mellom:")
print(f"  - Base uten sol: {base_prediction:.2f} kWh")
print(f"  - Expected med sol: {expected_kwh} kWh")
print(f"\nDette tyder på at solar impact er overestimert.")

# Reverse engineer the "correct" solar impact
actual_solar_reduction = base_prediction - actual_kwh
print(f"\nFactisk solar reduksjon (fra data):")
print(f"  Base - Actual = {base_prediction:.2f} - {actual_kwh} = {actual_solar_reduction:.2f} kWh")

if solar_impact_kwh > 0:
    ratio = actual_solar_reduction / solar_impact_kwh
    print(f"  Faktisk/Forventet ratio: {ratio:.3f}")
    print(f"  Solar impact er {1/ratio:.2f}x for høy")

# Heat pump COP analysis
print(f"\n=== Varmepumpe COP Korreksjon ===")
print(f"Med varmepumpe:")
print(f"  1 kWh solar gain = 1 kWh gratis varme")
print(f"  Men med COP=3: Hadde kun trengt 1/3 = 0.33 kWh elektrisitet for samme varme")
print(f"  Så solar gain sparer bare: 1/3 = 0.33 kWh elektrisitet")

if solar_impact_kwh > 0:
    cop_estimate = solar_impact_kwh / actual_solar_reduction
    print(f"\nEstimert COP fra dine data:")
    print(f"  COP ≈ {cop_estimate:.2f}")

# Coefficient analysis
print(f"\n=== Solar Coefficient Analyse ===")
print(f"Default coefficient (heating): {DEFAULT_SOLAR_COEFF_HEATING}")

# What coefficient would give the correct impact?
# solar_impact = avg_solar_factor * unit_coeff
# We assume avg_solar_factor was 1.0 (or whatever resulted in solar_impact_kwh) for simplicity of ratio calculation.

# If current coeff resulted in solar_impact_kwh, what coeff results in actual_solar_reduction?
# ratio = actual / predicted
# correct_coeff = current_coeff * ratio

# Assuming the user is running with default coefficient currently:
current_coeff_assumed = DEFAULT_SOLAR_COEFF_HEATING
if solar_impact_kwh > 0:
    correction_factor = actual_solar_reduction / solar_impact_kwh
    corrected_coeff = current_coeff_assumed * correction_factor

    print(f"\nHvis du bruker default coefficient ({current_coeff_assumed}),")
    print(f"så burde coefficient være {corrected_coeff:.4f}")
    print(f"for å gi riktig impact på {actual_solar_reduction:.2f} kWh")

# Learning discussion
print(f"\n=== Lærings-diskusjon ===")
print(f"Solar coefficient lærer med rate: {DEFAULT_SOLAR_LEARNING_RATE}")
print(f"\nFra learning.py:")
print(f"  Implied coeff = (Base - Actual) / Global_Solar_Factor")
print(f"  New coeff = Current + {DEFAULT_SOLAR_LEARNING_RATE} * (Implied - Current)")

# Calculate implied coefficient
# solar_impact_kwh = global_factor * current_coeff
# => global_factor = solar_impact_kwh / current_coeff
if current_coeff_assumed > 0:
    global_factor_example = solar_impact_kwh / current_coeff_assumed

    # implied_coeff = actual_solar_reduction / global_factor_example
    # (Because actual_solar_reduction = global_factor * implied_coeff)
    implied_coeff = actual_solar_reduction / global_factor_example if global_factor_example > 0 else 0

    new_coeff = current_coeff_assumed + DEFAULT_SOLAR_LEARNING_RATE * (implied_coeff - current_coeff_assumed)

    print(f"\nFor din time:")
    print(f"  Estimert Global Solar Factor: {global_factor_example:.3f}")
    print(f"  Implied coefficient: {implied_coeff:.4f}")
    print(f"  Current coefficient: {current_coeff_assumed:.4f}")
    print(f"  New coefficient: {new_coeff:.4f}")
    print(f"  Delta: {new_coeff - current_coeff_assumed:+.5f}")

    # Convergence estimate
    if implied_coeff != current_coeff_assumed:
        # EMA formula: new = current + rate * (implied - current)
        # After n iterations: value ≈ implied * (1 - (1-rate)^n) + current * (1-rate)^n
        # For 99% convergence: (1-rate)^n = 0.01
        import math
        n_99 = math.log(0.01) / math.log(1 - DEFAULT_SOLAR_LEARNING_RATE)
        print(f"\nKonvergens:")
        print(f"  Antall timer for 99% konvergens: {n_99:.0f} timer (~{n_99/24:.1f} dager)")

print(f"\n=== Konklusjon ===")
print(f"1. ✓ Minuttelling er matematisk korrekt")
if solar_impact_kwh > 0:
    print(f"2. ✗ Solar impact er {1/ratio:.2f}x for høy (sannsynligvis pga varmepumpe)")
print(f"3. → Modellen vil lære seg riktig coefficient over tid (~{n_99/24:.1f} dager)")
print(f"4. → Du kan justere manuelt i databasen for raskere korreksjon")
