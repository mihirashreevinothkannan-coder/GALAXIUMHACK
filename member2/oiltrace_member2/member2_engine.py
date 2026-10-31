# member2_engine.py
import pandas as pd
import numpy as np
import json
from datetime import datetime

# ============================================================
# Input: spill information from Member 1 (simulated here as fixed values)
# In actual integration, Member 3 will pass in Member 1's output
# ============================================================
spill_lat = 25.0
spill_lon = 55.0
spill_time_str = "2025-01-01 12:00:00"  # satellite image time
spill_time = datetime.strptime(spill_time_str, "%Y-%m-%d %H:%M:%S")

# AIS file path
AIS_FILE = "sample_ais.csv"

# Parameters (adjustable)
TIME_WINDOW_HOURS = 3.0      # Search AIS records within 3 hours before satellite time
SPATIAL_RADIUS_KM = 50.0     # Spatial search radius (km)
EARTH_RADIUS_KM = 6371.0

# ============================================================
# Helper function for safe empty output
# ============================================================
def write_empty(reason):
    output = {
        "spill": {"lat": spill_lat, "lon": spill_lon, "time": spill_time_str},
        "estimated_origin": {"lat": None, "lon": None, "method": reason},
        "vessels": []
    }
    with open("output_member2.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote empty output_member2.json ({reason})")
    raise SystemExit(0)

# ============================================================
# 1. Read and preprocess AIS
# ============================================================
df = pd.read_csv(AIS_FILE)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Keep only records within time window: [spill_time - 3h, spill_time]
time_start = spill_time - pd.Timedelta(hours=TIME_WINDOW_HOURS)
df = df[(df["timestamp"] >= time_start) & (df["timestamp"] <= spill_time)].copy()

if df.empty:
    write_empty("no_ais_in_time_window")

# ============================================================
# 2. Calculate Haversine distance for each AIS record to spill point
# ============================================================
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))

df["distance_km"] = haversine(df["lat"], df["lon"], spill_lat, spill_lon)

# Keep only records within spatial radius
df = df[df["distance_km"] <= SPATIAL_RADIUS_KM].copy()

if df.empty:
    write_empty("no_ais_in_spatial_radius")

# ============================================================
# 3. For each vessel (mmsi), calculate score using its closest point to spill
# ============================================================
results = []

for mmsi, group in df.groupby("mmsi"):
    # Find the record closest to spill point
    closest_idx = group["distance_km"].idxmin()
    closest_row = group.loc[closest_idx]

    # Time difference (hours): spill time - vessel closest point time
    time_diff_hours = (spill_time - closest_row["timestamp"]).total_seconds() / 3600.0

    # Distance score: closer is better, full score within 10km, 0 points at 50km
    dist_score = max(0, 1 - closest_row["distance_km"] / SPATIAL_RADIUS_KM) * 100

    # Time score: closer to spill time is better, full score within 0.5h, 0 points at 3h
    time_score = max(0, 1 - time_diff_hours / TIME_WINDOW_HOURS) * 100

    # Total score: distance 60% weight, time 40% weight
    total_score = dist_score * 0.6 + time_score * 0.4

    results.append({
        "mmsi": int(mmsi),
        "closest_lat": round(float(closest_row["lat"]), 5),
        "closest_lon": round(float(closest_row["lon"]), 5),
        "distance_km": round(float(closest_row["distance_km"]), 2),
        "time_diff_hours": round(float(time_diff_hours), 2),
        "sog": round(float(closest_row["sog"]), 1),
        "cog": round(float(closest_row["cog"]), 1),
        "score": round(float(total_score), 1)
    })

# Sort by score descending
results.sort(key=lambda x: x["score"], reverse=True)

# ============================================================
# 4. Simple "drift origin estimation"
# Strategy: Based on the closest point of the highest-scoring vessel,
# offset backward along the direction of movement, estimated as origin
# ============================================================
if results:
    top = results[0]
    # Simple reverse drift: if the vessel was moving toward spill point, backtrack along its course
    # Here only a demonstration offset is made: using time difference and approximate speed
    # Time difference (hours) * speed (knots) * 1.852 = distance (km)
    drift_km = top["time_diff_hours"] * top["sog"] * 1.852
    # Direction: reverse of cog (simplified handling, no precise geodesic calculation done)
    angle_rad = np.radians(top["cog"] + 180)  # reverse direction
    dlat = (drift_km / 111.0) * np.cos(angle_rad)
    dlon = (drift_km / (111.0 * np.cos(np.radians(spill_lat)))) * np.sin(angle_rad)

    origin_lat = round(spill_lat + dlat, 5)
    origin_lon = round(spill_lon + dlon, 5)
else:
    origin_lat, origin_lon = None, None

# ============================================================
# 5. Output JSON (for member 3)
# ============================================================
output = {
    "spill": {
        "lat": spill_lat,
        "lon": spill_lon,
        "time": spill_time_str
    },
    "estimated_origin": {
        "lat": origin_lat,
        "lon": origin_lon,
        "method": "simple_backward_offset_from_top_candidate"
    },
    "vessels": results[:5]  # Only give top 5 to dashboard
}

with open("output_member2.json", "w") as f:
    json.dump(output, f, indent=2)

print("output_member2.json generated")
print(json.dumps(output, indent=2))