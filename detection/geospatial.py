"""
Geospatial Mapping and Metadata Module for OILTRACE (Member 1).

Translates detected SAR candidate slicks into geographical coordinates when
valid georeferencing or scene metadata is provided. Strictly enforces separation
between raw pixel coordinates and geographical coordinates, and never invents
unverified locations or timestamps.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


def parse_sentinel1_timestamp(product_id_or_filename: str) -> Optional[str]:
    """
    Extract standard ISO-8601 UTC timestamp from a Sentinel-1 product ID.
    Example: S1A_IW_GRDH_1SDV_20200117T001541 -> 2020-01-17T00:15:41Z
    """
    match = re.search(r"(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})", product_id_or_filename)
    if match:
        year, month, day, hour, minute, second = match.groups()
        return f"{year}-{month}-{day}T{hour}:{minute}:{second}Z"
    return None


def map_pixel_to_latlon(
    pixel_x: float,
    pixel_y: float,
    image_width: int,
    image_height: int,
    geo_bbox: Dict[str, float]
) -> Tuple[float, float]:
    """
    Map a pixel coordinate (x, y) to (latitude, longitude) using an affine bounding box.
    y=0 is the top edge (max_lat), y=height is the bottom edge (min_lat).
    x=0 is the left edge (min_lon), x=width is the right edge (max_lon).
    """
    min_lon = geo_bbox["min_lon"]
    max_lon = geo_bbox["max_lon"]
    min_lat = geo_bbox["min_lat"]
    max_lat = geo_bbox["max_lat"]

    lon = min_lon + (pixel_x / float(image_width)) * (max_lon - min_lon)
    lat = max_lat - (pixel_y / float(image_height)) * (max_lat - min_lat)

    return round(float(lat), 6), round(float(lon), 6)


def attach_geospatial_metadata(
    candidates: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
    scene_meta: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Enrich detected candidates with geographical coordinates if georeferencing exists.
    If georeferencing is unavailable, explicitly reports that and leaves coordinates as null.

    Args:
        candidates: List of candidate region dicts.
        image_width: Width of the SAR image in pixels.
        image_height: Height of the SAR image in pixels.
        scene_meta: Optional metadata dict from catalog or GeoTIFF tags.

    Returns:
        Dict containing top-level geospatial report and enriched candidates.
    """
    if not scene_meta:
        # No georeferencing available
        for cand in candidates:
            cand["geographic_coordinates"] = None
            cand["bounding_box_geo"] = None

        return {
            "georeferencing_available": False,
            "georeferencing_type": "none",
            "georeferencing_note": (
                "No georeferencing data provided. Candidate pixel coordinates are "
                "retained without geographic projection. Do not treat pixel coordinates as coordinates."
            ),
            "satellite_metadata": None,
            "spill_location_estimate": None
        }

    geo_info = scene_meta.get("georeferencing", {})
    is_available = geo_info.get("available", False)
    geo_bbox = geo_info.get("bounding_box_geo")

    # Extract satellite acquisition metadata
    product_id = scene_meta.get("product_id", "")
    timestamp = scene_meta.get("acquisition_timestamp_utc") or parse_sentinel1_timestamp(product_id)

    sat_metadata = {
        "satellite": scene_meta.get("satellite", "UNKNOWN"),
        "sensor_mode": scene_meta.get("beam_mode", "UNKNOWN"),
        "polarization": scene_meta.get("polarization", "UNKNOWN"),
        "product_id": product_id,
        "acquisition_timestamp_utc": timestamp,
        "location_name": scene_meta.get("location_name", "Unknown Area"),
        "utm_zone": scene_meta.get("utm_zone", "Unknown")
    }

    if not is_available or not geo_bbox:
        for cand in candidates:
            cand["geographic_coordinates"] = None
            cand["bounding_box_geo"] = None

        return {
            "georeferencing_available": False,
            "georeferencing_type": "scene_anchor_only",
            "georeferencing_note": (
                "Scene-level anchor is documented, but detailed pixel-to-coordinate affine "
                "grid is unavailable. Coordinates are withheld to prevent false mapping."
            ),
            "satellite_metadata": sat_metadata,
            "spill_location_estimate": {
                "scene_centroid_latitude": geo_info.get("centroid_latitude"),
                "scene_centroid_longitude": geo_info.get("centroid_longitude"),
                "pixel_level_projection": False
            }
        }

    # Valid georeferencing grid available: map each candidate
    pixel_spacing = geo_info.get("pixel_spacing_meters", [10.0, 10.0])
    pixel_area_m2 = pixel_spacing[0] * pixel_spacing[1]
    pixel_area_km2 = pixel_area_m2 / 1_000_000.0

    primary_candidate_coords = None

    for cand in candidates:
        centroid_y, centroid_x = cand["centroid_pixel"]
        c_lat, c_lon = map_pixel_to_latlon(centroid_x, centroid_y, image_width, image_height, geo_bbox)

        y_min, x_min, y_max, x_max = cand["bounding_box_pixels"]
        lat_top, lon_left = map_pixel_to_latlon(x_min, y_min, image_width, image_height, geo_bbox)
        lat_bot, lon_right = map_pixel_to_latlon(x_max, y_max, image_width, image_height, geo_bbox)

        cand["geographic_coordinates"] = {
            "latitude": c_lat,
            "longitude": c_lon,
            "coordinate_reference_system": "EPSG:4326 (WGS84)"
        }
        cand["bounding_box_geo"] = {
            "min_longitude": min(lon_left, lon_right),
            "min_latitude": min(lat_top, lat_bot),
            "max_longitude": max(lon_left, lon_right),
            "max_latitude": max(lat_top, lat_bot),
            "coordinate_reference_system": "EPSG:4326 (WGS84)"
        }
        cand["estimated_area_km2"] = round(cand["pixel_count"] * pixel_area_km2, 4)

        if primary_candidate_coords is None:
            primary_candidate_coords = {
                "centroid_latitude": c_lat,
                "centroid_longitude": c_lon,
                "bounding_box_geo": cand["bounding_box_geo"],
                "estimated_area_km2": cand["estimated_area_km2"]
            }

    return {
        "georeferencing_available": True,
        "georeferencing_type": geo_info.get("type", "affine_grid"),
        "georeferencing_note": geo_info.get("notes", "Affine transformation WGS84 EPSG:4326"),
        "satellite_metadata": sat_metadata,
        "spill_location_estimate": primary_candidate_coords
    }
