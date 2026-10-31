"""
Segmentation Module for OILTRACE (Member 1).

Implements:
1. Classical SAR dark-patch detection (adaptive thresholding + morphological speckle cleaning).
2. Optional pretrained deep-learning segmentation loader (with graceful fallback).
3. Scientific evaluation metrics (IoU, Dice, Precision, Recall) comparing predictions
   against reference masks when available.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple
import numpy as np


def _binary_erosion(mask: np.ndarray, size: int = 3) -> np.ndarray:
    """Pure NumPy 2D binary erosion with square structuring element."""
    rad = size // 2
    padded = np.pad(mask, rad, mode="constant", constant_values=False)
    eroded = np.ones_like(mask, dtype=bool)
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            eroded &= padded[rad + dy : rad + dy + mask.shape[0], rad + dx : rad + dx + mask.shape[1]]
    return eroded


def _binary_dilation(mask: np.ndarray, size: int = 3) -> np.ndarray:
    """Pure NumPy 2D binary dilation with square structuring element."""
    rad = size // 2
    padded = np.pad(mask, rad, mode="constant", constant_values=False)
    dilated = np.zeros_like(mask, dtype=bool)
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            dilated |= padded[rad + dy : rad + dy + mask.shape[0], rad + dx : rad + dx + mask.shape[1]]
    return dilated


def morphological_open(mask: np.ndarray, size: int = 3) -> np.ndarray:
    """Morphological opening (erosion followed by dilation) to remove isolated speckle."""
    try:
        from scipy.ndimage import binary_opening
        return binary_opening(mask, structure=np.ones((size, size), dtype=bool))
    except Exception:
        return _binary_dilation(_binary_erosion(mask, size=size), size=size)


def morphological_close(mask: np.ndarray, size: int = 3) -> np.ndarray:
    """Morphological closing (dilation followed by erosion) to consolidate slick boundaries."""
    try:
        from scipy.ndimage import binary_closing
        return binary_closing(mask, structure=np.ones((size, size), dtype=bool))
    except Exception:
        return _binary_erosion(_binary_dilation(mask, size=size), size=size)


def segment_classical_dark_patch(
    sar_intensity: np.ndarray,
    threshold_factor: float = 1.1,
    dark_percentile: float = 15.0,
    morph_kernel_size: int = 3,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Detect candidate oil slicks using classical SAR radar backscatter physics.

    In SAR imagery, oil slicks dampen ocean capillary-gravity waves, causing specular
    reflection away from the sensor and yielding low radar backscatter ("dark patches").
    This method computes sea ambient background statistics and extracts dark patches
    below an adaptive contrast threshold, followed by morphological speckle filtering.

    Args:
        sar_intensity: Preprocessed 2D float array of SAR intensity.
        threshold_factor: Multiplier for standard deviation subtraction (default 1.1).
        dark_percentile: Percentile fallback threshold for uncalibrated tiles (default 15.0).
        morph_kernel_size: Kernel size for morphological open/close (default 3).

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]:
            - binary_mask: 2D boolean array (True = suspected candidate slick pixel).
            - stats: Dictionary with background mean, threshold value, and parameters.
    """
    # -----------------------------------------------------------------------
    # Sea background estimation: trimmed dark-half estimator.
    #
    # Global mean is biased upward in heterogeneous tiles (JPEG display chips)
    # that contain high-backscatter features like ships, coastline, or bright
    # terrain at 200-255 DN alongside true open-sea pixels at 80-120 DN.
    # Using the P75-trimmed dark half isolates the sea background distribution
    # and gives a threshold that tracks the ACTUAL ocean clutter level, rather
    # than an inflated composite of sea + land + ship DN values.
    #
    # This is consistent with operational SAR slick detection practice (e.g.,
    # EMSA CleanSeaNet, which estimates sea clutter from the dark fraction of
    # each sub-scene rather than from the full image histogram).
    # -----------------------------------------------------------------------
    global_mean = float(np.mean(sar_intensity))
    p65 = float(np.percentile(sar_intensity, 65))
    dark_half = sar_intensity[sar_intensity <= p65]
    if dark_half.size < 10:
        # Degenerate case: almost all pixels are very bright (extreme case)
        dark_half = sar_intensity
    sea_mean = float(np.mean(dark_half))
    sea_std = float(np.std(dark_half))

    # Bright-pixel threshold (used by feature_extraction to flag shadow suspects)
    bright_pixel_threshold = max(sea_mean + 2.5 * sea_std, 180.0)

    # In uniform ocean tiles without prominent features, std can be small.
    # Adaptive threshold combines statistical offset and percentile guard.
    stat_threshold = sea_mean - (threshold_factor * sea_std)
    pct_threshold = float(np.percentile(sar_intensity, dark_percentile))

    # A candidate pixel must be darker than the sea mean by at least 15% and fall below threshold
    adaptive_threshold = min(stat_threshold, pct_threshold)
    min_dark_cutoff = sea_mean * 0.85
    effective_threshold = min(adaptive_threshold, min_dark_cutoff)

    # Initial raw dark pixel mask
    raw_dark_mask = (sar_intensity <= effective_threshold)

    # Clean isolated speckle noise (open), then seal internal micro-holes (close)
    opened = morphological_open(raw_dark_mask, size=morph_kernel_size)
    cleaned_mask = morphological_close(opened, size=morph_kernel_size)

    stats = {
        "method": "classical_sar_backscatter_adaptive",
        "global_mean_intensity": round(global_mean, 2),
        "sea_mean_intensity": round(sea_mean, 2),
        "sea_mean_estimator": "p65_trimmed_dark_fraction",
        "sea_std_intensity": round(sea_std, 2),
        "effective_threshold": round(float(effective_threshold), 2),
        "bright_pixel_threshold": round(bright_pixel_threshold, 2),
        "threshold_factor": threshold_factor,
        "dark_percentile": dark_percentile,
        "morph_kernel_size": morph_kernel_size,
    }

    return cleaned_mask, stats


def segment_unet(
    sar_intensity: np.ndarray,
    checkpoint_path: Optional[str] = None
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Attempt pretrained U-Net deep learning segmentation if weights are available.
    If PyTorch or weights are missing, returns (None, explanation) for graceful fallback.
    """
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        return None, {
            "status": "skipped",
            "reason": f"U-Net checkpoint not found at '{checkpoint_path}'. Falling back to classical baseline."
        }

    try:
        import torch
        # Placeholder for PyTorch checkpoint loading
        # If user supplies a verified lightweight model file, inference executes here
        return None, {
            "status": "not_loaded",
            "reason": "PyTorch inference hook available, but specific model architecture was not initialized."
        }
    except ImportError:
        return None, {
            "status": "skipped",
            "reason": "PyTorch is not available in environment. Using classical detection baseline."
        }


def evaluate_segmentation(
    pred_mask: np.ndarray,
    reference_mask: np.ndarray
) -> Dict[str, float]:
    """
    Compute rigorous evaluation metrics comparing predicted candidate mask
    against a human-annotated ground-truth reference mask.

    Returns:
        Dict with IoU (Jaccard), Dice (F1), Precision, Recall, and Accuracy.
    """
    p = pred_mask.astype(bool)
    r = reference_mask.astype(bool)

    true_pos = int(np.logical_and(p, r).sum())
    false_pos = int(np.logical_and(p, ~r).sum())
    false_neg = int(np.logical_and(~p, r).sum())
    true_neg = int(np.logical_and(~p, ~r).sum())

    total = p.size
    intersection = true_pos
    union = true_pos + false_pos + false_neg

    iou = float(intersection / union) if union > 0 else (1.0 if r.sum() == 0 and p.sum() == 0 else 0.0)
    precision = float(true_pos / (true_pos + false_pos)) if (true_pos + false_pos) > 0 else 0.0
    recall = float(true_pos / (true_pos + false_neg)) if (true_pos + false_neg) > 0 else 0.0
    dice = float(2 * true_pos / (2 * true_pos + false_pos + false_neg)) if (2 * true_pos + false_pos + false_neg) > 0 else 0.0
    accuracy = float((true_pos + true_neg) / total)

    return {
        "iou_jaccard": round(iou, 4),
        "dice_f1": round(dice, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "pixel_accuracy": round(accuracy, 4),
        "true_positive_pixels": true_pos,
        "false_positive_pixels": false_pos,
        "false_negative_pixels": false_neg,
        "reference_slick_pixels": int(r.sum())
    }
