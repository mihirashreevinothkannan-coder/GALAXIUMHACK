"""
Visualization Module for OILTRACE (Member 1).

Generates high-contrast visual artifacts:
- Binary candidate masks (PNG)
- Highlighted overlay on SAR background with bounding boxes and centroid markers (PNG)
- Strictly separated ground-truth reference annotations vs. candidate predictions (PNG)
- Confusion / benchmark overlay showing true positives, false alarms, and missed slicks (PNG)
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image, ImageDraw


def save_binary_mask(mask: np.ndarray, output_path: str) -> str:
    """
    Save a 2D boolean or uint8 mask to a standard 8-bit PNG (255 = candidate, 0 = sea).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    uint8_mask = (mask.astype(bool) * 255).astype(np.uint8)
    Image.fromarray(uint8_mask, mode="L").save(output_path)
    return output_path


def create_candidate_overlay(
    sar_intensity: np.ndarray,
    candidate_mask: np.ndarray,
    candidates: List[Dict[str, Any]],
    output_path: str,
    overlay_color: Tuple[int, int, int] = (255, 50, 50),
    alpha: float = 0.55
) -> str:
    """
    Generate highlighted overlay showing candidate detections on the SAR image.
    Candidate slick pixels are tinted in semi-transparent crimson/amber.
    Bounding boxes and centroid crosshairs are drawn on top.

    Returns:
        output_path: Destination path of the saved PNG.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Base grayscale SAR image scaled to 8-bit RGB
    sar_norm = np.clip(sar_intensity, 0.0, 255.0).astype(np.uint8)
    rgb_base = np.stack([sar_norm, sar_norm, sar_norm], axis=-1).astype(np.float32)

    # Blend candidate pixels with overlay color
    mask_bool = candidate_mask.astype(bool)
    if np.any(mask_bool):
        tint_rgb = np.array(overlay_color, dtype=np.float32)
        rgb_base[mask_bool] = (1.0 - alpha) * rgb_base[mask_bool] + alpha * tint_rgb

    overlay_img = Image.fromarray(np.clip(rgb_base, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(overlay_img)

    # Draw bounding boxes and centroids for each candidate
    for cand in candidates:
        y_min, x_min, y_max, x_max = cand["bounding_box_pixels"]
        cy, cx = cand["centroid_pixel"]

        # Bounding box in amber
        draw.rectangle([x_min, y_min, x_max, y_max], outline=(255, 185, 0), width=2)

        # Centroid crosshair in cyan
        cross_rad = 6
        draw.line([cx - cross_rad, cy, cx + cross_rad, cy], fill=(0, 230, 255), width=2)
        draw.line([cx, cy - cross_rad, cx, cy + cross_rad], fill=(0, 230, 255), width=2)

        cand_id = cand["candidate_id"]
        if "mineral" in cand.get("classification", ""):
            class_tag = "Slick"
        elif "shadow" in cand.get("classification", ""):
            class_tag = "Shadow?"
        else:
            class_tag = "Look-alike?"
        draw.text((x_min + 4, max(0, y_min - 14)), f"#{cand_id} {class_tag}", fill=(255, 255, 255))

    overlay_img.save(output_path)
    return output_path


def create_reference_comparison_overlay(
    sar_intensity: np.ndarray,
    candidate_mask: np.ndarray,
    reference_mask: np.ndarray,
    output_path: str
) -> str:
    """
    Generate an honest validation comparison overlay strictly distinguishing
    candidate model/classical detections from human ground-truth reference annotations:
    - Yellow: True Positive (Overlap: Both detected and annotated)
    - Red: False Alarm (Candidate detected, but not in reference)
    - Cyan: Missed Slick (In reference annotation, but missed by candidate detector)
    - Grayscale: Sea background
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    sar_norm = np.clip(sar_intensity, 0.0, 255.0).astype(np.uint8)
    rgb = np.stack([sar_norm, sar_norm, sar_norm], axis=-1).astype(np.float32)

    pred = candidate_mask.astype(bool)
    ref = reference_mask.astype(bool)

    tp = np.logical_and(pred, ref)
    fp = np.logical_and(pred, ~ref)
    fn = np.logical_and(~pred, ref)

    # True Positive: Gold / Yellow
    rgb[tp] = 0.3 * rgb[tp] + 0.7 * np.array([255, 220, 0], dtype=np.float32)
    # False Alarm: Crimson Red
    rgb[fp] = 0.3 * rgb[fp] + 0.7 * np.array([255, 40, 40], dtype=np.float32)
    # Missed ground-truth slick: Cyan Green
    rgb[fn] = 0.3 * rgb[fn] + 0.7 * np.array([0, 220, 255], dtype=np.float32)

    comp_img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(comp_img)

    # Legend at top-left
    draw.rectangle([8, 8, 260, 78], fill=(20, 20, 20), outline=(120, 120, 120), width=1)
    draw.rectangle([16, 16, 26, 26], fill=(255, 220, 0))
    draw.text((34, 14), "Overlap (True Positive)", fill=(240, 240, 240))

    draw.rectangle([16, 36, 26, 46], fill=(255, 40, 40))
    draw.text((34, 34), "Candidate Only (False Alarm)", fill=(240, 240, 240))

    draw.rectangle([16, 56, 26, 66], fill=(0, 220, 255))
    draw.text((34, 54), "Reference Only (Missed)", fill=(240, 240, 240))

    comp_img.save(output_path)
    return output_path
