# OILTRACE — Inter-Module Integration Contract
**Author**: Member 1 (AI-Based Oil Spill Detection)  
**Target Consumers**: Member 2 (AIS Vessel Correlation) & Member 3 (Web Dashboard)  
**Version**: 1.0  
**Standard Output Location**: `outputs/metadata/<incident_id>.json`

---

## 1. Quick Integration Overview
Member 1 analyzes Sentinel-1 SAR satellite imagery to identify suspected marine oil slicks, extracts spatial and radiometric features, calculates geographical coordinates (when georeferencing is available), and saves both visual overlays and structured JSON metadata.

```
+-----------------------------------+
|      Member 1: SAR Detection     |
|  - Generates outputs/masks/       |
|  - Generates outputs/overlays/    |
|  - Generates outputs/metadata/    |
+-----------------+-----------------+
                  |
        +---------+---------+
        |                   |
        v                   v
+-------------------+ +-------------------+
|  Member 2 (AIS)   | | Member 3 (UI/Map) |
| - Uses Geo Coords | | - Folium Map Mark |
| - Uses Timestamp  | | - Renders Overlays|
| - Queries Vessels | | - Shows Summary   |
+-------------------+ +-------------------+
```

---

## 2. Directory Structure Conventions

| Directory | Content Type | Typical Format | Consumer |
| :--- | :--- | :--- | :--- |
| `outputs/metadata/` | Complete machine-readable detection report | `.json` | Member 2 & Member 3 |
| `outputs/overlays/` | Alpha-blended SAR overlay with bounding boxes | `.png` | Member 3 (Dashboard image display) |
| `outputs/masks/` | Clean binary candidate slick mask (0/255) | `.png` | Member 3 (Dashboard inspection) |

---

## 3. JSON Output Schema Specification

### Top-Level Fields

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `contract_version` | string | Interface contract version (`"1.0"`). |
| `incident_id` | string | Unique incident identifier (e.g., `"OILTRACE-S1_CANTARELL_SLICK"`). |
| `source_image` | string | Relative path to the original SAR input image. |
| `detection_status` | string | `"candidate_slick_detected"` or `"no_slick_detected"`. |
| `detection_method` | string | `"classical_sar_backscatter_adaptive"` or `"pretrained_unet"`. |
| `scientific_disclaimer` | string | Standard uncertainty disclaimer for dashboard display. |
| `spill_location_estimate` | object / null | Primary geographic position for map markers and AIS queries. `null` if unreferenced. |
| `georeferencing` | object | Status flag and notes explaining coordinate validity. |
| `satellite_metadata` | object / null | Satellite sensor, product ID, and UTC timestamp. |
| `candidates` | array | List of candidate regions with pixel coordinates and shape metrics. |
| `artifacts` | object | Relative paths to generated PNG overlays and masks. |

---

## 4. Guide for Member 2: AIS Vessel Correlation & Drift
Member 2 queries vessel telemetry around the suspected spill. **Only use geographical fields, never pixel coordinates!**

### Fields to Consume for AIS Queries:
1. **Spill Centroid Location**:
   ```json
   "spill_location_estimate": {
     "centroid_latitude": 19.376348,
     "centroid_longitude": -92.033027
   }
   ```
   * *Usage*: Center coordinate for radial AIS proximity search (e.g., query vessels within a 20 km to 50 km radius).

2. **Spill Bounding Box (Geographical Search Window)**:
   ```json
   "bounding_box_geo": {
     "min_longitude": -92.054092,
     "min_latitude": 19.365688,
     "max_longitude": -92.010764,
     "max_latitude": 19.389224,
     "coordinate_reference_system": "EPSG:4326 (WGS84)"
   }
   ```
   * *Usage*: Query bounding box or spatial filter for vessel trajectories.

3. **Satellite Sensing Timestamp**:
   ```json
   "satellite_metadata": {
     "acquisition_timestamp_utc": "2020-01-17T00:15:41Z"
   }
   ```
   * *Usage*: Exact satellite overpass time. Member 2 should query AIS positions within a time window leading up to this timestamp (e.g., $T - 12\text{ hours}$ to $T$).

4. **Handling Unreferenced Images**:
   * If `georeferencing.available == false`, `spill_location_estimate` will be `null`.
   * Member 2 should display: *"AIS correlation cannot proceed: satellite image lacks geographic coordinates."*

---

## 5. Guide for Member 3: Web Dashboard (Streamlit & Folium)

### Map Rendering (Folium):
* Check `result["georeferencing"]["available"]`.
* **If `true`**:
  * Add a Folium map marker at `[centroid_latitude, centroid_longitude]`.
  * Optionally draw a red rectangle for `bounding_box_geo`.
  * Tooltip / Popup content:
    * Incident ID
    * Acquisition Timestamp (UTC)
    * Estimated Area ($\text{km}^2$)
    * Uncertainty Level (e.g., "Suspected Mineral Oil Slick")
* **If `false`**:
  * Display a banner: *"Ungeoreferenced Image: Pixel analysis available, map marker disabled."*

### Image & Overlay Display (Streamlit):
* Display the side-by-side or stacked view:
  * **Original SAR**: `result["source_image"]`
  * **Candidate Overlay**: `result["artifacts"]["overlay_path"]`
  * **Binary Mask**: `result["artifacts"]["mask_path"]`
  * **Comparison with Reference (if present)**: `result["artifacts"]["comparison_overlay_path"]`

### Distinction: Pixel Coordinates vs. Geographical Coordinates
* **DO NOT** use `bounding_box_pixels` `[y_min, x_min, y_max, x_max]` or `centroid_pixel` `[y, x]` as latitude/longitude!
  * `bounding_box_pixels` are image array indices ($0 \le y < \text{height}$, $0 \le x < \text{width}$).
  * `spill_location_estimate` contains true WGS84 coordinates ($19.37^\circ\text{N}, -92.03^\circ\text{W}$).

### Mandatory Uncertainty Disclaimer:
Every dashboard display of Member 1's detection **must** display the disclaimer in `result["scientific_disclaimer"]`:
> *"Dark SAR formations represent areas of dampened capillary surface waves. Low backscatter can be caused by mineral oil spills, but also by natural environmental look-alikes (biogenic films, low-wind calms, grease ice). Detected regions are candidate anomalies requiring multi-sensor or AIS drift verification."*

---

## 6. Full Example Output JSON

```json
{
  "contract_version": "1.0",
  "incident_id": "OILTRACE-S1_CANTARELL_SLICK",
  "source_image": "data/samples/s1_cantarell_slick.png",
  "detection_status": "candidate_slick_detected",
  "detection_method": "classical_sar_backscatter_adaptive",
  "scientific_disclaimer": "SAR dark formations represent areas of dampened capillary surface waves...",
  "image_metadata": {
    "width_pixels": 800,
    "height_pixels": 600,
    "total_pixels": 480000
  },
  "detection_summary": {
    "num_candidate_regions": 1,
    "total_candidate_pixels": 38985,
    "candidate_area_fraction": 0.08122,
    "total_estimated_area_km2": 3.8985
  },
  "spill_location_estimate": {
    "centroid_latitude": 19.376348,
    "centroid_longitude": -92.033027,
    "bounding_box_geo": {
      "min_longitude": -92.054092,
      "min_latitude": 19.365688,
      "max_longitude": -92.010764,
      "max_latitude": 19.389224,
      "coordinate_reference_system": "EPSG:4326 (WGS84)"
    },
    "estimated_area_km2": 3.8985
  },
  "georeferencing": {
    "available": true,
    "type": "affine_grid",
    "notes": "Verified from Garcia-INPE Cantarell dataset table S1A_IW_GRDH_1SDV_20200117T001541"
  },
  "satellite_metadata": {
    "satellite": "SENTINEL-1A",
    "sensor_mode": "IW_GRDH",
    "polarization": "VV",
    "product_id": "S1A_IW_GRDH_1SDV_20200117T001541",
    "acquisition_timestamp_utc": "2020-01-17T00:15:41Z",
    "location_name": "Cantarell Complex, Bay of Campeche, Gulf of Mexico",
    "utm_zone": "15Q"
  },
  "candidates": [
    {
      "candidate_id": 1,
      "classification": "suspected_mineral_oil_slick",
      "uncertainty_level": "moderate",
      "pixel_count": 38985,
      "area_fraction": 0.08122,
      "bounding_box_pixels": [179, 201, 441, 656],
      "centroid_pixel": [322.33, 422.21],
      "radiometric_stats": {
        "mean_intensity": 18.94,
        "contrast_db": 4.08,
        "local_sea_mean": 48.43
      },
      "shape_stats": {
        "compactness": 0.312
      }
    }
  ],
  "artifacts": {
    "mask_path": "outputs/masks/OILTRACE-S1_CANTARELL_SLICK_mask.png",
    "overlay_path": "outputs/overlays/OILTRACE-S1_CANTARELL_SLICK_overlay.png",
    "comparison_overlay_path": "outputs/overlays/OILTRACE-S1_CANTARELL_SLICK_reference_comparison.png",
    "metadata_json_path": "outputs/metadata/OILTRACE-S1_CANTARELL_SLICK.json"
  }
}
```
