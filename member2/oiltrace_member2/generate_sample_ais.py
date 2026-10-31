# generate_sample_ais.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Simulate: satellite image time = 2025-01-01 12:00:00
# Spill center point (simulated as detected by Member 1)
spill_lat, spill_lon = 25.0, 55.0
satellite_time = datetime(2025, 1, 1, 12, 0, 0)

rows = []

# Vessel A: "Suspicious vessel" — was near spill point 90 minutes before satellite time, then moved away
for i in range(5):
    t = satellite_time - timedelta(minutes=90 - i * 15)  # from 90 min to 30 min before
    rows.append({
        "mmsi": 111111111,
        "timestamp": t.strftime("%Y-%m-%d %H:%M:%S"),
        "lat": spill_lat + 0.02 * i,   # slowly moving away
        "lon": spill_lon + 0.02 * i,
        "sog": 8.0,
        "cog": 45.0
    })

# Vessel B: Passed by 30 minutes before, but far
for i in range(3):
    t = satellite_time - timedelta(minutes=60 - i * 20)
    rows.append({
        "mmsi": 222222222,
        "timestamp": t.strftime("%Y-%m-%d %H:%M:%S"),
        "lat": spill_lat + 0.5,
        "lon": spill_lon + 0.5,
        "sog": 12.0,
        "cog": 90.0
    })

# Vessel C: Very close, but 3 hours before (too early)
for i in range(3):
    t = satellite_time - timedelta(hours=3, minutes=-i * 30)
    rows.append({
        "mmsi": 333333333,
        "timestamp": t.strftime("%Y-%m-%d %H:%M:%S"),
        "lat": spill_lat + 0.01,
        "lon": spill_lon + 0.01,
        "sog": 5.0,
        "cog": 180.0
    })

df = pd.DataFrame(rows)
df.to_csv("sample_ais.csv", index=False)
print("sample_ais.csv generated")