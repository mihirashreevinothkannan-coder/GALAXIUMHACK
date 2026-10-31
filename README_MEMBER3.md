# OILTRACE — Global Maritime Intelligence (Member 3)

## Run on Windows

Open this folder in VS Code. In PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks virtual-environment activation:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

Open the localhost URL printed in the terminal (usually http://localhost:8501).

## What's new

- New Mission Control-style landing experience with an orthographic Earth globe and georeferenced incident pins.
- Four investigation workspaces: Global Watch, Satellite Intelligence, Vessel Investigation, Incident Dossier.
- Folium regional vessel map now uses OpenStreetMap tiles instead of the CARTO tiles that displayed an API-key warning.
- Preserves Member 1's detection source, metadata, masks, overlays, documentation and tests.
- Preserves Member 2's original engine, AIS CSV and output JSON under `member2/oiltrace_member2/`.
- Adds an integration adapter that correlates Member 2 AIS records against Member 1's actual spill coordinate and acquisition timestamp.

## Important data alignment note

Member 1's Cantarell sample is near 19.376°N, 92.033°W at 2020-01-17. Member 2's original sample AIS data is near 25°N, 55°E at 2025-01-01. They are not the same incident. The file `data/ais/sample_ais_cantarell_demo.csv` is synthetic demonstration data aligned to the Cantarell sample so the end-to-end workflow can be shown. It is clearly labelled in the UI. Select the original AIS dataset in the sidebar to inspect the teammate's unchanged data.

## Scientific and operational limitations

- Member 1's active method is classical SAR image processing, not a pretrained deep-learning model.
- The Cantarell fixture uses nominal affine georeferencing, not a calibrated raw Sentinel-1 Level-1 geospatial raster.
- The MKLab sample images have no geographic coordinates; the dashboard disables map plotting and AIS correlation for those scenes.
- Vessel scores are a proximity/time heuristic. The backward drift offset is illustrative, not a validated ocean-current trajectory model.
- The globe plots the georeferenced incidents in the supplied project data. It is not a live satellite feed.
- No hourly satellite polling or real-time AIS stream is currently running. Live catalogue ingestion, credentials, scheduling, and verified AIS access are future integrations.
- Candidate SAR anomalies do not prove mineral oil contamination or vessel responsibility.
