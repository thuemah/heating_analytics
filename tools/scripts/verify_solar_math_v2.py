"""Verification Script for Solar Math (No Astral Dependency)."""
import math

def calculate_solar_factor(elevation: float, azimuth: float, target_azimuth: float, cloud_coverage: float = 0.0) -> tuple[float, float, float]:
    """Calculate solar factor and its components."""
    # 1. Elevation Factor
    if elevation <= 0:
        elev_factor = 0.0
    else:
        elev_rad = math.radians(elevation)
        elev_factor = max(0.0, math.cos(elev_rad))

    # 2. Azimuth Factor (Current Formula)
    az_rad = math.radians(azimuth)
    target_rad = math.radians(target_azimuth)
    az_factor_current = 0.5 + 0.5 * math.cos(az_rad - target_rad)

    # 3. Azimuth Factor (Geometric / Pure)
    # Cosine of incidence angle on vertical surface (ignoring elevation for now)
    # delta = az - target
    # factor = max(0, cos(delta))
    az_factor_geo = max(0.0, math.cos(az_rad - target_rad))

    # Total Factors (assuming cloud=0)
    total_current = elev_factor * az_factor_current
    total_geo = elev_factor * az_factor_geo

    return elev_factor, az_factor_current, total_current, az_factor_geo, total_geo

def simulate_day():
    print("--- Solar Math Verification ---")
    print("Assumptions:")
    print(" - Window Facing: SOUTH (180 degrees)")
    print(" - Location: Oslo (approx 60 deg N)")
    print(" - Date: Summer Solstice (High Sun, Long Day)")
    print("")
    print(f"{'Time':<10} | {'Azimuth':<8} | {'Elev':<6} | {'ElevFact':<8} | {'AzFact(Curr)':<12} | {'Total(Curr)':<12} | {'AzFact(Geo)':<12} | {'Total(Geo)':<12}")
    print("-" * 100)

    # Approximate Sun Path for Oslo Summer Solstice
    # Sunrise ~04:00 at Az ~45 (NE)
    # Noon 13:00 at Az 180 (S), Elev ~53
    # Sunset ~22:00 at Az ~315 (NW)

    # Let's mock some data points
    # Hour, Azimuth, Elevation
    points = [
        (4, 45, 0),   # Sunrise (NE)
        (6, 70, 15),  # Early Morning
        (9, 120, 35), # Morning (SE)
        (12, 165, 52),# Noon-ish
        (13, 180, 53),# Solar Noon (S) - Peak
        (14, 195, 52),# Afternoon
        (17, 240, 35),# Late Afternoon (SW)
        (20, 290, 15),# Evening
        (22, 315, 0), # Sunset (NW)
    ]

    target_az = 180 # South

    for hour, az, elev in points:
        e_fact, a_fact_c, t_c, a_fact_g, t_g = calculate_solar_factor(elev, az, target_az)
        time_str = f"{hour:02d}:00"
        print(f"{time_str:<10} | {az:<8} | {elev:<6} | {e_fact:<8.3f} | {a_fact_c:<12.3f} | {t_c:<12.3f} | {a_fact_g:<12.3f} | {t_g:<12.3f}")

    print("-" * 100)
    print("Note on 90-degree offset (East/West):")
    print("If Azimuth is 90 (East) and Target is 180 (South), Delta is 90.")
    _, a_fact_90, _, a_fact_geo_90, _ = calculate_solar_factor(10, 90, 180) # Elev 10 just to get non-zero
    print(f"Azimuth Factor Current: {a_fact_90:.3f}")
    print(f"Azimuth Factor Geometric: {a_fact_geo_90:.3f}")

if __name__ == "__main__":
    simulate_day()
