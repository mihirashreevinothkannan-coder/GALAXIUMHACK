"""
Feature Extraction Module for OILTRACE (Member 1).

Extracts geometric and radiometric features from detected candidate slick regions:
- Connected components labeling
- Bounding boxes [min_y, min_x, max_y, max_x]
- Centroids [y, x]
- Pixel counts and area fractions
- Sea-to-slick contrast (dB)
- Morphological compactness and look-alike uncertainty classification
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def _connected_components_pure_numpy(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Connected component labeling with 8-connectivity implemented in pure NumPy/BFS.
    Used when scipy.ndimage is not installed, guaranteeing 100% portability.
    """
    h, w = mask.shape
    labeled = np.zeros((h, w), dtype=np.int32)
    current_label = 0

    # 8-connected neighborhood offsets
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    for y in range(h):
        for x in range(w):
            if mask[y, x] and labeled[y, x] == 0:
                current_label += 1
                queue = [(y, x)]
                labeled[y, x] = current_label

                while queue:
                    cy, cx = queue.pop()
                    for dy, dx in neighbors:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if mask[ny, nx] and labeled[ny, nx] == 0:
                                labeled[ny, nx] = current_label
                                queue.append((ny, nx))

    return labeled, current_label


def label_connected_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Label connected candidate components using 8-connectivity.
    Prefers scipy.ndimage if available; falls back to pure NumPy BFS.
    """
    try:
        from scipy.ndimage import label
        struct8 = np.ones((3, 3), dtype=int)
        labeled, num_features = label(mask, structure=struct8)
        return labeled.astype(np.int32), int(num_features)
    except Exception:
        return _connected_components_pure_numpy(mask)


def extract_candidate_regions(
    binary_mask: np.ndarray,
    sar_intensity: np.ndarray,
    min_pixels: int = 50,
    min_contrast_db: float = 2.0,
    sea_mean: Optional[float] = None,
    bright_pixel_threshold: Optional[float] = None,
    shadow_proximity_px: int = 15,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Identify and profile all candidate dark-patch regions in the segmented mask.

    Args:
        binary_mask: 2D boolean mask from segmentation.
        sar_intensity: 2D float array of SAR intensity.
        min_pixels: Minimum pixel area to discard isolated speckle false alarms.
        min_contrast_db: Minimum contrast below ambient sea (dB) to be considered
            a candidate patch (below 2.0 dB is standard SAR speckle texture).
        sea_mean: Ambient sea mean intensity (computed from mask background if None).
        bright_pixel_threshold: Intensity value above which pixels are considered
            bright features (ships, land). Used to flag shadow suspects. Defaults
            to 90th percentile of the image if None.
        shadow_proximity_px: Bounding-box distance threshold (pixels) within which
            a dark candidate is classified as 'shadow_suspect' (likely radar layover
            or terrain shadow adjacent to bright structures, not open-sea slick).
    """
    labeled, num_features = label_connected_components(binary_mask)
    h, w = binary_mask.shape
    total_image_pixels = h * w

    if sea_mean is None:
        bg_pixels = sar_intensity[~binary_mask]
        sea_mean = float(np.mean(bg_pixels)) if bg_pixels.size > 0 else float(np.mean(sar_intensity))

    # Build a binary map of bright (non-sea) features for shadow proximity detection.
    # A dark candidate whose bounding box is within shadow_proximity_px of a bright
    # feature is classified as a shadow suspect rather than a sea-surface slick.
    if bright_pixel_threshold is None:
        bright_pixel_threshold = max(sea_mean + 2.5 * 15.0, 180.0)
    bright_feature_map = (sar_intensity >= bright_pixel_threshold)

    filtered_mask = np.zeros_like(binary_mask, dtype=bool)
    candidates: List[Dict[str, Any]] = []

    for region_id in range(1, num_features + 1):
        coords = np.argwhere(labeled == region_id)
        pixel_count = int(len(coords))

        if pixel_count < min_pixels:
            continue

        # Radiometric features
        region_intensities = sar_intensity[labeled == region_id]
        fg_mean = float(np.mean(region_intensities))
        fg_min = float(np.min(region_intensities))
        fg_std = float(np.std(region_intensities))

        # Contrast in decibels: 10 * log10(sea_mean / fg_mean)
        ratio = max(sea_mean / max(fg_mean, 1e-3), 1e-3)
        contrast_db = round(float(10.0 * math.log10(ratio)), 2)

        # In SAR physics, contrast < 2.0 dB is standard speckle noise, not a candidate patch
        if contrast_db < min_contrast_db:
            continue

        # Keep this candidate in the filtered output mask
        filtered_mask[labeled == region_id] = True

        y_indices = coords[:, 0]
        x_indices = coords[:, 1]

        # Bounding box [min_y, min_x, max_y, max_x]
        y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))
        x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
        bbox = [y_min, x_min, y_max, x_max]

        # Centroid [centroid_y, centroid_x]
        centroid_y = round(float(np.mean(y_indices)), 2)
        centroid_x = round(float(np.mean(x_indices)), 2)
        centroid = [centroid_y, centroid_x]

        # Boundary perimeter estimation (pixels adjacent to background)
        # Dilate single patch by 1 pixel and subtract patch to get border
        patch_box = (labeled[max(0, y_min - 1) : min(h, y_max + 2), max(0, x_min - 1) : min(w, x_max + 2)] == region_id)
        padded_border = np.pad(patch_box, 1, mode="constant", constant_values=False)
        dilated_patch = np.zeros_like(patch_box, dtype=bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                dilated_patch |= padded_border[1 + dy : 1 + dy + patch_box.shape[0], 1 + dx : 1 + dx + patch_box.shape[1]]
        perimeter_pixels = max(int(np.sum(dilated_patch & ~patch_box)), 4)

        # Compactness = 4 * pi * Area / Perimeter^2 (1.0 = perfect circle, small = elongated)
        compactness = round(float(4.0 * math.pi * pixel_count / (perimeter_pixels ** 2)), 3)

        # Look-alike discrimination & uncertainty classification
        # Environmental look-alikes (low wind calms, biogenic films) often exhibit:
        # - Weak contrast (< 2.5 dB below sea)
        # - High compactness / rounded outlines (> 0.60)
        # Real mineral slicks typically show sharp dampening (> 3.0 dB) and wind-driven elongation.
        #
        # Shadow suspect: dark patch whose bounding box is within shadow_proximity_px
        # of a bright feature (ships, coastline, terrain). In SAR imagery, bright targets
        # cast dark layover/shadow zones that are visually identical to slick patches but
        # are pure radar geometry artifacts. Flagged with high uncertainty.
        expanded_y_min = max(0, y_min - shadow_proximity_px)
        expanded_y_max = min(h - 1, y_max + shadow_proximity_px)
        expanded_x_min = max(0, x_min - shadow_proximity_px)
        expanded_x_max = min(w - 1, x_max + shadow_proximity_px)
        neighbourhood_bright = bright_feature_map[
            expanded_y_min : expanded_y_max + 1,
            expanded_x_min : expanded_x_max + 1
        ]
        near_bright_feature = bool(neighbourhood_bright.any())

        if near_bright_feature and contrast_db > 0:
            classification = "shadow_suspect"
            uncertainty_level = "high"
            description = (
                "Dark patch adjacent to a bright SAR feature (ship, coastline, or terrain). "
                "Likely radar layover shadow or JPEG contrast artifact, not an open-sea oil slick. "
                "Requires independent sensor corroboration before any slick attribution."
            )
        elif contrast_db < 2.5:
            classification = "lookalike_weak_contrast"
            uncertainty_level = "high"
            description = "Faint backscatter dip; likely ambient low-wind calm or biogenic surfactant."
        elif compactness > 0.65:
            classification = "lookalike_compact_blob"
            uncertainty_level = "medium_high"
            description = "Rounded dark formation without characteristic wind-driven slick elongation."
        else:
            classification = "suspected_mineral_oil_slick"
            uncertainty_level = "moderate"
            description = "High backscatter dampening with elongated morphology characteristic of oil."

        candidate_record = {
            "candidate_id": len(candidates) + 1,
            "classification": classification,
            "uncertainty_level": uncertainty_level,
            "scientific_note": description,
            "pixel_count": pixel_count,
            "area_fraction": round(float(pixel_count / total_image_pixels), 5),
            "bounding_box_pixels": bbox,
            "centroid_pixel": centroid,
            "bbox_dimensions": {
                "height_pixels": int(y_max - y_min + 1),
                "width_pixels": int(x_max - x_min + 1)
            },
            "radiometric_stats": {
                "mean_intensity": round(fg_mean, 2),
                "min_intensity": round(fg_min, 2),
                "std_intensity": round(fg_std, 2),
                "contrast_db": contrast_db,
                "local_sea_mean": round(sea_mean, 2)
            },
            "shape_stats": {
                "estimated_perimeter_pixels": perimeter_pixels,
                "compactness": compactness
            }
        }
        candidates.append(candidate_record)

    return candidates, filtered_mask
