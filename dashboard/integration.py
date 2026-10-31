import math
import pandas as pd

EARTH_RADIUS_KM = 6371.0

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
    dlat, dlon = lat2-lat1, lon2-lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))

def analyze_ais(detection, csv_path, radius_km=50.0, time_window_hours=12.0):
    spill = detection.get('spill_location_estimate') or {}
    lat, lon = spill.get('centroid_latitude'), spill.get('centroid_longitude')
    sat = detection.get('satellite_metadata') or {}
    timestamp = sat.get('acquisition_timestamp_utc')
    if lat is None or lon is None or not timestamp:
        return {'available': False, 'reason': 'No geographic spill coordinate or satellite timestamp is available.', 'vessels': []}
    spill_time = pd.to_datetime(timestamp, utc=True)
    df = pd.read_csv(csv_path)
    required = {'mmsi','timestamp','lat','lon','sog','cog'}
    if not required.issubset(df.columns):
        return {'available': False, 'reason': 'AIS CSV is missing required fields.', 'vessels': []}
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    for col in ['lat','lon','sog','cog']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['timestamp','lat','lon','sog','cog'])
    df = df[(df.timestamp >= spill_time-pd.Timedelta(hours=time_window_hours)) & (df.timestamp <= spill_time)].copy()
    if df.empty:
        return {'available': True, 'reason': 'No AIS observations in the selected pre-overpass window.', 'vessels': [], 'spill': {'lat':lat,'lon':lon,'time':timestamp}}
    df['distance_km'] = df.apply(lambda r: haversine_km(r.lat,r.lon,lat,lon), axis=1)
    df = df[df.distance_km <= radius_km]
    results=[]
    for mmsi, group in df.groupby('mmsi'):
        row=group.loc[group.distance_km.idxmin()]
        delta=(spill_time-row.timestamp).total_seconds()/3600
        ds=max(0,1-row.distance_km/radius_km)*100
        ts=max(0,1-delta/time_window_hours)*100
        results.append({'mmsi':int(mmsi),'closest_lat':round(float(row.lat),5),'closest_lon':round(float(row.lon),5),
            'distance_km':round(float(row.distance_km),2),'time_diff_hours':round(float(delta),2),
            'sog':round(float(row.sog),1),'cog':round(float(row.cog),1),'score':round(ds*.6+ts*.4,1)})
    results.sort(key=lambda x:x['score'],reverse=True)
    origin=None
    if results:
        top=results[0]; drift=top['time_diff_hours']*top['sog']*1.852; angle=math.radians(top['cog']+180)
        origin={'lat':round(lat+(drift/111)*math.cos(angle),5),
                'lon':round(lon+(drift/(111*math.cos(math.radians(lat))))*math.sin(angle),5),
                'method':'Simplified backward course offset; illustrative only'}
    return {'available':True,'reason':None,'spill':{'lat':lat,'lon':lon,'time':timestamp},
            'estimated_origin':origin,'vessels':results[:5],
            'parameters':{'radius_km':radius_km,'time_window_hours':time_window_hours}}
