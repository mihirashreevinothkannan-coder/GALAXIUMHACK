"""
SAR Preprocessing Module for OILTRACE (Member 1).

Handles input image loading, dimension and channel validation, conversion to
linear radar intensity, and multi-look boxcar speckle filtering.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Tuple, Union
import numpy as np
from PIL import Image


class PreprocessingError(Exception):
    """Raised when image loading or validation fails."""
    pass


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def inspect_image_metadata(image_path: str) -> Dict[str, Any]:
    """
    Inspect image properties without modifying pixel data.
    Identifies color mode, format, and distinguishes display chips from calibrated SAR.
    """
    if not os.path.exists(image_path):
        raise PreprocessingError(f"Image file does not exist: {image_path}")

    ext = os.path.splitext(image_path)[1].lower()
    try:
        with Image.open(image_path) as pil_img:
            mode = pil_img.mode
            fmt = pil_img.format or ext.lstrip(".").upper()
            width, height = pil_img.size
    except Exception as exc:
        raise PreprocessingError(f"Failed to inspect image header at '{image_path}': {exc}") from exc

    is_display_chip = mode in ("RGB", "RGBA", "L") and ext in (".jpg", ".jpeg", ".png")

    return {
        "format": fmt,
        "mode": mode,
        "width": width,
        "height": height,
        "is_display_chip": is_display_chip,
        "data_fidelity_note": (
            "8-bit display chip (non-linear tone curve; not calibrated Level-1 backscatter)."
            if is_display_chip else
            "Scientific raster / single-band SAR format."
        )
    }


def load_sar_image(image_path: str) -> np.ndarray:
    """
    Load a satellite SAR image from disk, validate dimensions, and convert to 2D float32 array.

    Supports PNG, JPEG, TIFF (including 8-bit and 16-bit), and BMP formats.
    Handles RGB/RGBA display images by luminance conversion, and normalizes
    16-bit integer formats to the standard [0.0, 255.0] intensity scale.

    Args:
        image_path: Path to the SAR image (PNG, TIFF, JPG, BMP).

    Returns:
        np.ndarray: 2D array of float32 values scaled to [0.0, 255.0].

    Raises:
        PreprocessingError: If file is missing, empty, unsupported, or invalid.
    """
    if not os.path.exists(image_path):
        raise PreprocessingError(f"Image file does not exist: {image_path}")

    ext = os.path.splitext(image_path)[1].lower()
    if ext and ext not in SUPPORTED_IMAGE_EXTENSIONS:
        raise PreprocessingError(
            f"Unsupported image file format '{ext}'. Supported formats: "
            f"{', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}"
        )

    if os.path.getsize(image_path) == 0:
        raise PreprocessingError(f"Image file is empty (0 bytes): {image_path}")

    try:
        with Image.open(image_path) as pil_img:
            # Handle 16-bit integer TIFF images (e.g. Sentinel-1 GRD uint16)
            if pil_img.mode in ("I;16", "I;16B", "I;16L", "I;16S"):
                raw_arr = np.asarray(pil_img, dtype=np.float32)
                arr = (raw_arr / 65535.0) * 255.0
            elif pil_img.mode in ("F", "I"):
                raw_arr = np.asarray(pil_img, dtype=np.float32)
                vmax = float(raw_arr.max())
                arr = (raw_arr / vmax * 255.0) if vmax > 255.0 else np.clip(raw_arr, 0.0, 255.0)
            elif pil_img.mode != "L":
                gray_img = pil_img.convert("L")
                arr = np.asarray(gray_img, dtype=np.float32)
            else:
                gray_img = pil_img.copy()
                arr = np.asarray(gray_img, dtype=np.float32)
    except PreprocessingError:
        raise
    except Exception as exc:
        raise PreprocessingError(f"Failed to read image at '{image_path}': {exc}") from exc

    if arr.ndim != 2:
        raise PreprocessingError(f"Expected 2D grayscale image, got array with shape {arr.shape}")

    height, width = arr.shape
    if height < 16 or width < 16:
        raise PreprocessingError(f"Image dimensions too small ({width}x{height}). Minimum is 16x16.")

    return arr


def load_reference_mask(mask_path: str, expected_shape: Tuple[int, int]) -> np.ndarray:
    """
    Load and validate a ground-truth reference mask.

    Args:
        mask_path: Path to the binary reference mask file.
        expected_shape: (height, width) tuple that the mask must match.

    Returns:
        np.ndarray: 2D boolean array (True = reference slick pixel).

    Raises:
        PreprocessingError: If mask file is missing, unreadable, or dimensions mismatch.
    """
    if not os.path.exists(mask_path):
        raise PreprocessingError(f"Reference mask file does not exist: {mask_path}")

    try:
        with Image.open(mask_path) as pil_img:
            gray_mask = pil_img.convert("L")
            arr = np.asarray(gray_mask, dtype=np.uint8)
    except Exception as exc:
        raise PreprocessingError(f"Failed to read mask at '{mask_path}': {exc}") from exc

    if arr.shape != expected_shape:
        raise PreprocessingError(
            f"Reference mask dimension mismatch: got {arr.shape}, expected {expected_shape}"
        )

    # Values > 127 are considered positive reference slick pixels
    return (arr > 127)


def boxcar_filter_2d(img: np.ndarray, size: int = 5) -> np.ndarray:
    """
    Perform 2D uniform boxcar filter using an integral image (summed-area table).
    Runs in O(H*W) time in pure NumPy with zero external C-dependencies.
    """
    if size < 2:
        return img.copy()

    h, w = img.shape
    rad = size // 2

    # Pad image reflecting border values
    padded = np.pad(img, pad_width=rad, mode="reflect")
    # Compute 2D integral image (with zero-padded upper-left)
    integral = np.pad(np.cumsum(np.cumsum(padded, axis=0), axis=1), ((1, 0), (1, 0)), mode="constant")

    # Target window slice boundaries in integral coordinates
    y1 = np.arange(h)
    y2 = y1 + size
    x1 = np.arange(w)
    x2 = x1 + size

    # Vectorized boxcar sum from integral table:
    # Sum(D) = I(y2, x2) - I(y1, x2) - I(y2, x1) + I(y1, x1)
    # Using meshgrid indexing
    Y1, X1 = np.meshgrid(y1, x1, indexing="ij")
    Y2, X2 = np.meshgrid(y2, x2, indexing="ij")

    window_sum = (
        integral[Y2, X2]
        - integral[Y1, X2]
        - integral[Y2, X1]
        + integral[Y1, X1]
    )

    return (window_sum / float(size * size)).astype(np.float32)


def multilook_speckle_reduction(sar_intensity: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Multi-look speckle reduction in linear intensity power space.

    In SAR theory, speckle is multiplicative noise. Averaging must happen
    in linear power (intensity = amplitude^2), not in decibels or log space,
    to avoid systematic negative backscatter bias.

    Args:
        sar_intensity: 2D array of SAR intensity values [0.0, 255.0].
        window_size: Odd integer kernel size (default 5).

    Returns:
        np.ndarray: Speckle-reduced SAR intensity array.
    """
    # Clip zero/negative values for stability
    safe_intensity = np.clip(sar_intensity, 0.0, 255.0)
    # Power = intensity^2 (or linear backscatter)
    linear_power = np.square(safe_intensity)

    # Average in linear power space
    filtered_power = boxcar_filter_2d(linear_power, size=window_size)
    # Convert back to amplitude scale
    filtered_intensity = np.sqrt(np.clip(filtered_power, 0.0, None))

    return filtered_intensity.astype(np.float32)
