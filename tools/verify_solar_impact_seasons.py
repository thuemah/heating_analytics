"""Verification Script for Solar Math across Seasons."""
import math
from datetime import datetime

# Mocking Astral logic for the script to be standalone if needed,
# but ideally we approximate the sun path for Oslo (60N).

def calculate_solar_factor(elevation: float, azimuth: float, target_azimuth: float, cloud_coverage: float = 0.0) -> tuple[float, float, float, float, float]:
    """Calculate solar factor and its components."""
    # 1. Elevation Factor
    if elevation <= 0:
        elev_factor = 0.0
    else:
        elev_rad = math.radians(elevation)
        elev_factor = max(0.0, math.cos(elev_rad))

    # 2. Azimuth Factor (Actual Logic from solar.py)
    delta = abs(azimuth - target_azimuth)
    if delta > 180:
        delta = 360 - delta

    BUFFER_ANGLE = 15.0
    DIFFUSE_FLOOR = 0.1
    BACKSIDE_FLOOR = 0.05

    cutoff = 90.0 - BUFFER_ANGLE  # 75.0
    az_factor_current = 0.0

    if delta <= cutoff:
        # Zone 1: Direct Sun
        normalized_pos = delta / cutoff
        direct_component = math.cos(normalized_pos * (math.pi / 2))
        az_factor_current = direct_component * (1.0 - DIFFUSE_FLOOR) + DIFFUSE_FLOOR
    elif delta <= 90.0:
        # Zone 2: Glancing
        az_factor_current = DIFFUSE_FLOOR
    else:
        # Zone 3: Backside
        az_factor_current = BACKSIDE_FLOOR

    # 3. Azimuth Factor (Geometric / Pure)
    # Cosine of incidence angle on vertical surface
    az_rad = math.radians(azimuth)
    target_rad = math.radians(target_azimuth)
    az_factor_geo = max(0.0, math.cos(az_rad - target_rad))

    # Total Factors (assuming cloud=0)
    total_current = elev_factor * az_factor_current
    total_geo = elev_factor * az_factor_geo

    return elev_factor, az_factor_current, total_current, az_factor_geo, total_geo

def get_oslo_sun_path(season: str):
    """Return list of (hour, azimuth, elevation) for Oslo (60N)."""
    # Rough approximations based on solar charts for 60N

    if season == "SUMMER": # June 21
        # Rise: 03:54 (Az 40), Set: 22:40 (Az 320), Noon: 53.5 deg
        return [
            (4, 45, 2),
            (6, 75, 15),
            (9, 120, 36),
            (12, 165, 52),
            (13, 180, 53.5), # Peak
            (14, 195, 52),
            (17, 240, 36),
            (20, 285, 15),
            (22, 315, 2)
        ]
    elif season == "EQUINOX": # March/Sept 21
        # Rise: 06:00 (Az 90), Set: 18:00 (Az 270), Noon: 30 deg
        return [
            (6, 90, 0),
            (7, 105, 8),
            (9, 135, 20),
            (12, 165, 29),
            (13, 180, 30), # Peak
            (14, 195, 29),
            (17, 240, 15),
            (18, 270, 0)
        ]
    elif season == "WINTER": # Dec 21
        # Rise: 09:18 (Az 140), Set: 15:12 (Az 220), Noon: 6.5 deg
        return [
            (9, 140, 0), # Rise
            (10, 150, 3),
            (11, 165, 6),
            (12, 175, 6.5),
            (13, 180, 6.6), # Peak (very low!)
            (14, 185, 6.5),
            (15, 210, 2),
            (16, 230, -5) # Set
        ]
    return []

def simulate_season(season_name: str, target_az: float):
    print(f"\n=== {season_name} (Oslo 60N) ===")
    print(f"Target Azimuth: {target_az} (South)")
    print(f"{'Time':<8} | {'Az':<5} | {'El':<5} | {'ElevFact':<8} | {'AzFact(Curr)':<12} | {'Total(Curr)':<12} | {'Total(Geo)':<12}")
    print("-" * 80)

    points = get_oslo_sun_path(season_name)
    for hour, az, elev in points:
        e_fact, a_fact_c, t_c, _, t_g = calculate_solar_factor(elev, az, target_az)
        time_str = f"{hour:02d}:00"
        print(f"{time_str:<8} | {az:<5} | {elev:<5} | {e_fact:<8.3f} | {a_fact_c:<12.3f} | {t_c:<12.3f} | {t_g:<12.3f}")

def main():
    target_az = 180 # South
    simulate_season("SUMMER", target_az)
    simulate_season("EQUINOX", target_az)
    simulate_season("WINTER", target_az)

if __name__ == "__main__":
    main()
