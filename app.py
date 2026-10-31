import json
from pathlib import Path
import streamlit as st
import pandas as pd
import folium
import plotly.graph_objects as go
from streamlit_folium import st_folium
from dashboard.integration import analyze_ais

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='OILTRACE | Global Maritime Intelligence',page_icon='🌊',layout='wide',initial_sidebar_state='expanded')
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--cyan:#48d9e8;--muted:#94adbf;--line:rgba(126,170,196,.18)}
html,body,[class*="css"]{font-family:'Manrope',sans-serif}.stApp{background:radial-gradient(ellipse at 10% 0%,#12394c 0%,#071522 44%,#06111c 100%);color:#eaf5fa}
.block-container{padding-top:1.2rem;max-width:1600px}[data-testid="stSidebar"]{background:#081a29;border-right:1px solid var(--line)}
[data-testid="stMetric"]{background:linear-gradient(145deg,#102a3d,#0b1e2d);border:1px solid var(--line);padding:14px 16px;border-radius:14px}
[data-testid="stMetricLabel"]{color:#9cb6c8!important}[data-testid="stMetricValue"]{color:#f2f8fc!important}
.stTabs [data-baseweb="tab-list"]{gap:8px;border-bottom:1px solid var(--line)}.stTabs [data-baseweb="tab"]{background:#0b2030;border:1px solid var(--line);border-radius:9px 9px 0 0;color:#c0d2dd;padding:9px 14px}
.stTabs [aria-selected="true"]{background:#15364a!important;color:#fff!important;border-bottom:2px solid var(--cyan)!important}
div.stButton>button,div.stDownloadButton>button{border-radius:10px;border:1px solid #2c6078;background:#12364a;color:#eaf8fc}
.hero{background:linear-gradient(110deg,#0b2638,#0d3b4d);border:1px solid rgba(94,204,220,.22);border-radius:19px;padding:22px 25px;margin-bottom:17px}
.eyebrow{font-family:'DM Mono',monospace;letter-spacing:.14em;color:#70dce8;font-size:10px}.hero h1{font-size:31px;font-weight:800;color:#f2fbff;margin:5px 0}.hero p{color:#b2c9d6;margin:0;font-size:13px}
.panel{background:linear-gradient(145deg,#0d2435,#0a1b29);border:1px solid var(--line);border-radius:15px;padding:16px}.panel-title{font-size:11px;color:#a8c2d2;text-transform:uppercase;letter-spacing:.12em;font-weight:700;margin-bottom:9px}
.small{font-size:11px;color:#91aabd}.notice{background:rgba(255,180,94,.09);border:1px solid rgba(255,180,94,.3);padding:12px 14px;border-radius:11px;color:#f5c78e;font-size:12px}
</style>''',unsafe_allow_html=True)

def readj(p):
    with open(p,encoding='utf-8') as f:return json.load(f)
def resolve(p):
    if not p:return None
    q=Path(p.replace('\\','/'))
    return q if q.is_absolute() else ROOT/q

def globe(points,focus=None):
    fig=go.Figure()
    if points:
        fig.add_trace(go.Scattergeo(lon=[p[1] for p in points],lat=[p[0] for p in points],text=[p[2] for p in points],mode='markers+text',textposition='top center',
            marker=dict(size=[14 if p[3] else 10 for p in points],color=['#ff765f' if p[3] else '#48d9e8' for p in points],line=dict(width=1.5,color='#eafaff')),
            hovertemplate='<b>%{text}</b><br>%{lat:.3f}°, %{lon:.3f}°<extra></extra>'))
    geo=dict(projection_type='orthographic',showland=True,landcolor='#24475a',showocean=True,oceancolor='#071b2c',showcoastlines=True,coastlinecolor='#4c7184',showcountries=True,countrycolor='#456477',bgcolor='rgba(0,0,0,0)',showframe=True,framecolor='#31566a')
    if focus:geo['projection_rotation']={'lon':focus[1],'lat':focus[0],'roll':0}
    fig.update_geos(**geo);fig.update_layout(height=465,margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor='rgba(0,0,0,0)',font=dict(color='#d9edf5'))
    return fig

st.markdown('''<div class="hero"><div class="eyebrow">GLOBAL OCEAN INTELLIGENCE · INVESTIGATION NETWORK</div><h1>🌊 OILTRACE</h1><p>Detect candidate oil slicks. Correlate vessel movements. Investigate environmental threats.</p><div style="margin-top:12px"><span style="border:1px solid #367b87;border-radius:20px;padding:4px 9px;color:#78e4ec;font-size:10px">● PROTOTYPE MONITORING MODE</span><span class="small"> &nbsp; Satellite catalogue automation is not yet connected.</span></div></div>''',unsafe_allow_html=True)
meta_files=sorted((ROOT/'outputs/metadata').glob('*.json'))+sorted((ROOT/'outputs/real_eval/metadata').glob('*.json'))
if not meta_files:st.error('No Member 1 metadata found.');st.stop()
default='OILTRACE-S1_CANTARELL_SLICK.json';idx=next((i for i,p in enumerate(meta_files) if p.name==default),0)
with st.sidebar:
    st.markdown('## CONTROL ROOM')
    selected=st.selectbox('Incident workspace',[p.name for p in meta_files],index=idx)
    metadata=next(p for p in meta_files if p.name==selected);det=readj(metadata)
    st.divider();st.markdown('**AIS correlation parameters**')
    radius=st.slider('Search radius · km',5,100,50,5);window=st.slider('Pre-overpass window · hours',1,24,12,1)
    source=st.radio('AIS dataset',['Aligned demo dataset','Original teammate dataset'])
    aisfile=ROOT/'data/ais/sample_ais_cantarell_demo.csv' if source=='Aligned demo dataset' else ROOT/'member2/oiltrace_member2/sample_ais.csv'
    st.divider();st.caption('Prototype data only. Continuous live monitoring is not running.')
geo=det.get('spill_location_estimate') or {};sat=det.get('satellite_metadata') or {};summary=det.get('detection_summary') or {}
geo_ok=bool((det.get('georeferencing') or {}).get('available') and geo.get('centroid_latitude') is not None)
ais=analyze_ais(det,aisfile,radius_km=radius,time_window_hours=window)
c1,c2,c3,c4=st.columns(4)
c1.metric('DETECTION STATE','Candidate anomaly' if det.get('detection_status')=='candidate_slick_detected' else 'No candidate')
c2.metric('CANDIDATE REGIONS',summary.get('num_candidate_regions',0))
c3.metric('ESTIMATED AREA',f"{summary['total_estimated_area_km2']:.2f} km²" if summary.get('total_estimated_area_km2') is not None else 'Unavailable')
c4.metric('VESSELS CORRELATED',len(ais.get('vessels',[])) if ais.get('available') else 'Unavailable')
t1,t2,t3,t4=st.tabs(['🌐 GLOBAL WATCH','🛰️ SATELLITE INTELLIGENCE','🚢 VESSEL INVESTIGATION','📁 INCIDENT DOSSIER'])
with t1:
    left,right=st.columns([1.6,1],gap='large')
    with left:
        st.markdown('<div class="panel-title">EARTH OBSERVATION · INCIDENT GEOGRAPHY</div>',unsafe_allow_html=True)
        if geo_ok:
            lat=float(geo['centroid_latitude']);lon=float(geo['centroid_longitude']);points=[(lat,lon,'CANTARELL · ACTIVE CANDIDATE',True)]
            for p in meta_files:
                if p==metadata:continue
                d=readj(p);g=d.get('spill_location_estimate') or {}
                if (d.get('georeferencing') or {}).get('available') and g.get('centroid_latitude') is not None:points.append((float(g['centroid_latitude']),float(g['centroid_longitude']),d.get('incident_id','OTHER'),False))
            st.plotly_chart(globe(points,(lat,lon)),use_container_width=True,config={'displayModeBar':False})
            st.caption('Orthographic Earth view · Markers represent georeferenced incidents in the loaded datasets, not a live satellite feed.')
        else:
            st.warning('This scene has no geographic reference. Pixel detections cannot be plotted on Earth.')
            st.plotly_chart(globe([]),use_container_width=True,config={'displayModeBar':False})
    with right:
        st.markdown('<div class="panel-title">INCIDENT SIGNAL</div>',unsafe_allow_html=True)
        pos=f"{geo.get('centroid_latitude'):.5f}°, {geo.get('centroid_longitude'):.5f}°" if geo_ok else 'Not georeferenced'
        st.markdown(f'''<div class="panel"><div class="eyebrow">SELECTED CASE</div><h3 style="color:#f2fbff;margin:8px 0">{det.get('incident_id','Incident')}</h3><div class="small">{sat.get('location_name','Location unavailable')}</div><hr style="border-color:#234154"><div class="small">ACQUISITION TIME · UTC</div><div style="color:#eaf7fc;margin:5px 0 14px">{sat.get('acquisition_timestamp_utc','Unavailable')}</div><div class="small">ESTIMATED POSITION</div><div style="color:#eaf7fc;margin:5px 0 14px">{pos}</div><div class="small">DETECTION METHOD</div><div style="color:#eaf7fc;font-size:12px;margin-top:5px">{det.get('detection_method','Unavailable')}</div></div>''',unsafe_allow_html=True)
        st.markdown('<br><div class="panel-title">INVESTIGATION ACTIVITY</div><div class="panel"><div style="color:#69dce7;font-size:11px">01 · SATELLITE SCREENING</div><div class="small">Candidate regions extracted from SAR imagery</div><hr><div style="color:#69dce7;font-size:11px">02 · AIS CORRELATION</div><div class="small">'+(f"{len(ais.get('vessels',[]))} vessels ranked against selected scene" if ais.get('available') else 'Unavailable for this scene')+'</div><hr><div style="color:#ffbd73;font-size:11px">03 · HUMAN REVIEW</div><div class="small">Candidate requires independent verification</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="notice">DEMO DATA NOTICE · Cantarell uses nominal affine georeferencing. The aligned AIS records are synthetic demonstration data, not verified historical vessel tracks.</div>',unsafe_allow_html=True)
with t2:
    st.markdown('### Satellite evidence');st.caption(f"{sat.get('satellite','Sensor unavailable')} · {sat.get('acquisition_timestamp_utc','Timestamp unavailable')} · {det.get('detection_method','')}")
    art=det.get('artifacts') or {};original=resolve(det.get('source_image'));overlay=resolve(art.get('overlay_path'));mask=resolve(art.get('mask_path'))
    a,b=st.columns(2)
    with a:
        st.markdown('**ORIGINAL SAR INPUT**')
        if original and original.exists():st.image(str(original),use_container_width=True)
        else:st.warning('Source image unavailable')
    with b:
        st.markdown('**DETECTION OVERLAY**')
        if overlay and overlay.exists():st.image(str(overlay),use_container_width=True)
        else:st.warning('Overlay unavailable')
    if mask and mask.exists():
        with st.expander('Inspect binary segmentation mask'):st.image(str(mask),use_container_width=True)
    st.markdown('<div class="notice">SCIENTIFIC CAUTION · Dark SAR formations can also result from biogenic films, low-wind calms, grease ice or internal waves. This is a candidate anomaly, not confirmed oil contamination.</div>',unsafe_allow_html=True)
    cand=det.get('candidates') or []
    if cand:
        st.markdown('**Detection characteristics**');st.dataframe(pd.DataFrame([{'Candidate':x.get('candidate_id'),'Class':x.get('classification'),'Uncertainty':x.get('uncertainty_level'),'Pixels':x.get('pixel_count'),'Contrast dB':(x.get('radiometric_stats') or {}).get('contrast_db')} for x in cand[:8]]),use_container_width=True,hide_index=True)
with t3:
    st.markdown('### Vessel correlation & movement evidence')
    if not ais.get('available'):st.warning(ais.get('reason','AIS unavailable'))
    elif ais.get('reason'):st.info(ais['reason'])
    else:
        st.caption(f'Weighted heuristic · 60% proximity + 40% timing · {radius} km · {window} hours before overpass')
        vessels=ais.get('vessels',[])
        if vessels:
            df=pd.DataFrame(vessels);df.insert(0,'Rank',range(1,len(df)+1));df=df.rename(columns={'mmsi':'MMSI','distance_km':'Distance km','time_diff_hours':'Hours before overpass','sog':'Speed kn','cog':'Course °','score':'Candidate score'})
            st.dataframe(df,use_container_width=True,hide_index=True);st.bar_chart(df.set_index('MMSI')[['Candidate score']])
        else:st.info('No vessels matched the selected filters')
        origin=ais.get('estimated_origin')
        if origin and origin.get('lat') is not None:st.markdown('**Illustrative backward offset**');st.code(f"Latitude {origin['lat']} · Longitude {origin['lon']}");st.caption(origin['method'])
        st.markdown('<div class="notice">A high score prioritises investigation; it is not evidence of guilt. The drift offset is not a validated ocean-current model.</div>',unsafe_allow_html=True)
        if geo_ok:
            fmap=folium.Map(location=[geo['centroid_latitude'],geo['centroid_longitude']],zoom_start=9,tiles='OpenStreetMap',control_scale=True)
            folium.Marker([geo['centroid_latitude'],geo['centroid_longitude']],tooltip='SAR candidate anomaly',icon=folium.Icon(color='red',icon='warning-sign')).add_to(fmap)
            bbox=geo.get('bounding_box_geo') or {}
            if all(k in bbox for k in ['min_latitude','min_longitude','max_latitude','max_longitude']):folium.Rectangle([[bbox['min_latitude'],bbox['min_longitude']],[bbox['max_latitude'],bbox['max_longitude']]],color='red',fill=True,fill_opacity=.1).add_to(fmap)
            for v in vessels:folium.CircleMarker([v['closest_lat'],v['closest_lon']],radius=7,color='#087eaa',fill=True,fill_opacity=.9,tooltip=f"MMSI {v['mmsi']} · {v['distance_km']} km · {v['score']}/100").add_to(fmap)
            if origin and origin.get('lat') is not None:
                folium.Marker([origin['lat'],origin['lon']],tooltip='Illustrative drift origin',icon=folium.Icon(color='orange',icon='flag')).add_to(fmap);folium.PolyLine([[origin['lat'],origin['lon']],[geo['centroid_latitude'],geo['centroid_longitude']]],color='orange',dash_array='7,8').add_to(fmap)
            st.markdown('**Regional vessel map · OpenStreetMap (no CARTO API key)**');st_folium(fmap,use_container_width=True,height=460)
with t4:
    st.markdown('### Incident dossier')
    report={'incident_id':det.get('incident_id'),'detection_status':det.get('detection_status'),'detection_method':det.get('detection_method'),'detection_summary':summary,'spill_location_estimate':geo or None,'satellite_metadata':sat or None,'ais_analysis':ais}
    st.json({'incident':det.get('incident_id'),'detection':det.get('detection_status'),'candidate_regions':summary.get('num_candidate_regions'),'estimated_area_km2':summary.get('total_estimated_area_km2'),'georeferenced':geo_ok,'vessels_ranked':len(ais.get('vessels',[])) if ais.get('available') else 0,'top_vessel':(ais.get('vessels') or [{}])[0].get('mmsi')})
    st.download_button('⬇ Download investigation evidence (JSON)',json.dumps(report,indent=2),file_name='oiltrace_incident_dossier.json',mime='application/json')
    st.markdown('<div class="notice">OILTRACE supports environmental investigation. It does not establish confirmed contamination or identify a responsible party.</div>',unsafe_allow_html=True)
st.markdown('<hr><div class="small">OILTRACE · Global maritime intelligence prototype · Live satellite catalogue automation and verified AIS feeds are future integrations.</div>',unsafe_allow_html=True)
