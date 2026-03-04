import csv
import random
import math
from datetime import datetime, timedelta

def generate_csv(filename="test_data_import.csv"):
    start_date = datetime(2024, 1, 1, 0, 0)
    end_date = datetime(2025, 12, 3, 23, 0)

    current = start_date
    rows = []

    # Header
    rows.append(["timestamp", "energy_kwh", "temperature_c", "wind_speed_ms"])

    while current <= end_date:
        # Synthesize data
        # Day of year for seasonality
        day_of_year = current.timetuple().tm_yday
        hour = current.hour

        # Temperature: Cold in winter (Jan/Feb/Dec), Warm in summer (Jun/Jul/Aug)
        # Cosine wave: Peak cold at start/end of year, peak warm in middle
        # Base temp: 5C, Amplitude 15C => Range -10 to 20 approx (daily average)
        # Daily variation: 5C amplitude, peak at 14:00

        seasonal_temp = 5.0 - 15.0 * math.cos(2 * math.pi * (day_of_year) / 365.0)
        daily_variation = 5.0 * math.cos(2 * math.pi * (hour - 14) / 24.0)
        noise = random.uniform(-2, 2)

        temp = seasonal_temp + daily_variation + noise

        # Wind: Random mostly, but let's make it windier in winter
        wind_base = 3.0 + 2.0 * math.cos(2 * math.pi * (day_of_year) / 365.0) # Higher in winter (start of year is cos=1)
        wind = max(0, random.gauss(wind_base, 3.0)) # Normal distribution, non-negative

        # Energy: Inverse to temperature (Heating). Base load + Heating load.
        # Balance point 17C.
        hdd = max(0, 17.0 - temp)

        # Wind factor (cooling effect)
        effective_hdd = hdd * (1 + (wind / 10.0)) # Example factor

        # Base load (always on) = 0.5 kWh
        # Heating load = 0.2 kWh per TDD
        energy = 0.5 + (effective_hdd * 0.2)
        energy += random.uniform(-0.1, 0.1) # Noise
        energy = max(0.1, energy) # Min 0.1

        rows.append([
            current.isoformat(),
            f"{energy:.3f}",
            f"{temp:.1f}",
            f"{wind:.1f}"
        ])

        current += timedelta(hours=1)

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"Generated {len(rows)-1} rows of data to {filename}")

if __name__ == "__main__":
    generate_csv()
