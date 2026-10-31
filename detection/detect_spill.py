"""
OILTRACE Detection Pipeline CLI & Orchestrator (Member 1).

Main entry point for Sentinel-1 SAR oil spill candidate detection.
Orchestrates preprocessing, segmentation, feature extraction, geospatial projection,
visual artifact generation, and machine-readable metadata export.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

import numpy as np

# Ensure project root is on sys.path when executed directly as a script
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from detection.preprocessing import (
    load_sar_image,
    load_reference_mask,
    inspect_image_metadata,
    multilook_speckle_reduction,
    PreprocessingError,
)
from detection.segmentation import (
    segment_classical_dark_patch,
    segment_unet,
    evaluate_segmentation,
)
from detection.feature_extraction import extract_candidate_regions
from detection.geospatial import attach_geospatial_metadata
from detection.visualization import (
    save_binary_mask,
    create_candidate_overlay,
    create_reference_comparison_overlay,
)


def run_detection_pipeline(
    image_path: str,
    reference_mask_path: Optional[str] = None,
    catalog_path: Optional[str] = None,
    method: str = "classical",
    output_dir: str = "outputs",
    incident_id: Optional[str] = None,
    threshold_factor: float = 1.1,
    min_pixels: int = 50,
    min_contrast_db: float = 2.0,
    save_artifacts: bool = True,
    dataset_source: Optional[str] = None,
    dataset_license: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run end-to-end detection on a SAR image and generate all artifacts.

    Returns:
        Dict[str, Any]: Complete detection result and metadata dictionary.
    """
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    if not incident_id:
        incident_id = f"OILTRACE-{base_name.upper()}"

    # 1. Input Loading & Validation
    img_meta = inspect_image_metadata(image_path)
    raw_sar = load_sar_image(image_path)
    height, width = raw_sar.shape

    # 2. Multi-look Speckle Reduction
    filtered_sar = multilook_speckle_reduction(raw_sar, window_size=5)

    # 3. Detection & Segmentation
    unet_status = None
    if method.lower() == "unet":
        unet_mask, unet_status = segment_unet(filtered_sar)
        if unet_mask is not None:
            raw_mask = unet_mask
            actual_method = "pretrained_unet"
        else:
            # Graceful fallback to classical
            raw_mask, seg_stats = segment_classical_dark_patch(
                filtered_sar, threshold_factor=threshold_factor
            )
            actual_method = "classical_sar_backscatter_fallback"
    else:
        raw_mask, seg_stats = segment_classical_dark_patch(
            filtered_sar, threshold_factor=threshold_factor
        )
        actual_method = "classical_sar_backscatter_adaptive"

    # 4. Feature Extraction & Connected Region Analysis
    bright_thresh_ref = seg_stats.get("bright_pixel_threshold") if seg_stats else None
    candidates, final_mask = extract_candidate_regions(
        binary_mask=raw_mask,
        sar_intensity=filtered_sar,
        min_pixels=min_pixels,
        min_contrast_db=min_contrast_db,
        bright_pixel_threshold=bright_thresh_ref
    )

    total_candidate_pixels = sum(c["pixel_count"] for c in candidates)
    candidate_area_fraction = round(float(total_candidate_pixels / (height * width)), 5)
    detection_status = "candidate_slick_detected" if candidates else "no_slick_detected"

    # 5. Geospatial & Satellite Scene Lookup
    scene_meta = None
    if catalog_path and os.path.exists(catalog_path):
        try:
            with open(catalog_path, "r", encoding="utf-8") as f:
                catalog = json.load(f)
            # Match by exact file name
            file_key = os.path.basename(image_path)
            scene_meta = catalog.get(file_key)
        except Exception as e:
            print(f"[Warning] Failed to read catalog '{catalog_path}': {e}", file=sys.stderr)

    geo_report = attach_geospatial_metadata(
        candidates=candidates,
        image_width=width,
        image_height=height,
        scene_meta=scene_meta
    )

    # 6. Optional Reference Mask Evaluation (Strictly Separated)
    evaluation_metrics = None
    ref_mask_loaded = None
    if reference_mask_path:
        ref_mask_loaded = load_reference_mask(reference_mask_path, expected_shape=(height, width))
        evaluation_metrics = evaluate_segmentation(
            pred_mask=final_mask,
            reference_mask=ref_mask_loaded
        )

    # 7. Generate Visual Artifacts
    artifact_paths: Dict[str, Optional[str]] = {
        "mask_path": None,
        "overlay_path": None,
        "comparison_overlay_path": None,
    }

    if save_artifacts:
        masks_dir = os.path.join(output_dir, "masks")
        overlays_dir = os.path.join(output_dir, "overlays")
        meta_dir = os.path.join(output_dir, "metadata")

        for d in (masks_dir, overlays_dir, meta_dir):
            os.makedirs(d, exist_ok=True)

        mask_file = os.path.join(masks_dir, f"{incident_id}_mask.png")
        overlay_file = os.path.join(overlays_dir, f"{incident_id}_overlay.png")

        save_binary_mask(final_mask, mask_file)
        create_candidate_overlay(
            sar_intensity=filtered_sar,
            candidate_mask=final_mask,
            candidates=candidates,
            output_path=overlay_file
        )

        artifact_paths["mask_path"] = os.path.relpath(mask_file, start=".")
        artifact_paths["overlay_path"] = os.path.relpath(overlay_file, start=".")

        if ref_mask_loaded is not None:
            comp_file = os.path.join(overlays_dir, f"{incident_id}_reference_comparison.png")
            create_reference_comparison_overlay(
                sar_intensity=filtered_sar,
                candidate_mask=final_mask,
                reference_mask=ref_mask_loaded,
                output_path=comp_file
            )
            artifact_paths["comparison_overlay_path"] = os.path.relpath(comp_file, start=".")

    # 8. Assemble Machine-Readable Output Contract
    output_contract = {
        "contract_version": "1.0",
        "incident_id": incident_id,
        "source_image": os.path.relpath(image_path, start="."),
        "detection_status": detection_status,
        "detection_method": actual_method,
        "method_details": {
            "requested_method": method,
            "speckle_filter": "linear_intensity_multilook_5x5",
            "threshold_factor": threshold_factor,
            "min_candidate_pixels": min_pixels,
            "unet_status": unet_status
        },
        "scientific_disclaimer": (
            "SAR dark formations represent areas of dampened capillary surface waves. "
            "Low backscatter can be caused by mineral oil spills, but also by natural "
            "environmental look-alikes such as biogenic films, low-wind calms, grease ice, "
            "or internal waves. Detected regions are candidate anomalies requiring multi-sensor "
            "or AIS drift verification."
        ),
        "image_metadata": {
            "width_pixels": width,
            "height_pixels": height,
            "total_pixels": height * width,
            "original_format": img_meta.get("format"),
            "original_mode": img_meta.get("mode"),
            "is_display_chip": img_meta.get("is_display_chip"),
            "data_fidelity_note": img_meta.get("data_fidelity_note")
        },
        "detection_summary": {
            "num_candidate_regions": len(candidates),
            "total_candidate_pixels": total_candidate_pixels,
            "candidate_area_fraction": candidate_area_fraction,
            "total_estimated_area_km2": (
                round(sum(c.get("estimated_area_km2", 0.0) for c in candidates), 4)
                if geo_report["georeferencing_available"] else None
            )
        },
        "spill_location_estimate": geo_report["spill_location_estimate"],
        "georeferencing": {
            "available": geo_report["georeferencing_available"],
            "type": geo_report["georeferencing_type"],
            "notes": geo_report["georeferencing_note"]
        },
        "satellite_metadata": geo_report["satellite_metadata"],
        "candidates": candidates,
        "reference_ground_truth": {
            "evaluation_performed": (evaluation_metrics is not None),
            "reference_mask_path": os.path.relpath(reference_mask_path, start=".") if reference_mask_path else None,
            "metrics": evaluation_metrics
        },
        "artifacts": artifact_paths,
        "provenance": {
            "author": "OILTRACE Member 1 (AI-Based Oil Spill Detection)",
            "dataset_source": dataset_source,
            "dataset_license": dataset_license,
            "notes": (
                "Provenance fields are null when the source dataset is unspecified. "
                "See data/real_samples/PROVENANCE.md for real sample terms; "
                "see data/generate_samples.py for synthetic fixture parameters."
            ),
            "target_consumers": ["Member 2 (AIS Vessel Correlation)", "Member 3 (Streamlit Dashboard)"]
        }
    }

    # Save JSON metadata artifact
    if save_artifacts:
        meta_file = os.path.join(output_dir, "metadata", f"{incident_id}.json")
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(output_contract, f, indent=2)
        artifact_paths["metadata_json_path"] = os.path.relpath(meta_file, start=".")

    return output_contract


def main():
    parser = argparse.ArgumentParser(
        description="OILTRACE Member 1 — Sentinel-1 SAR Oil Spill Candidate Detection"
    )
    parser.add_argument("--image", "-i", required=True, help="Path to input SAR image (PNG/TIFF)")
    parser.add_argument("--reference-mask", "-r", default=None, help="Optional path to reference ground truth mask")
    parser.add_argument("--catalog", "-c", default="data/samples/scenes_catalog.json", help="Path to scenes catalog JSON")
    parser.add_argument("--method", "-m", choices=["classical", "unet"], default="classical", help="Detection method")
    parser.add_argument("--output-dir", "-o", default="outputs", help="Output directory")
    parser.add_argument("--incident-id", default=None, help="Custom identifier for the incident")
    parser.add_argument("--threshold-factor", type=float, default=1.1, help="Statistical threshold factor (default 1.1)")
    parser.add_argument("--min-pixels", type=int, default=50, help="Minimum candidate slick size in pixels")
    parser.add_argument("--min-contrast-db", type=float, default=2.0, help="Minimum contrast below sea in dB (default 2.0)")
    parser.add_argument("--dataset-source", default=None, help="Human-readable description of the image source dataset for provenance metadata.")
    parser.add_argument("--dataset-license", default=None, help="License or access terms governing the source image.")

    args = parser.parse_args()

    try:
        result = run_detection_pipeline(
            image_path=args.image,
            reference_mask_path=args.reference_mask,
            catalog_path=args.catalog,
            method=args.method,
            output_dir=args.output_dir,
            incident_id=args.incident_id,
            threshold_factor=args.threshold_factor,
            min_pixels=args.min_pixels,
            min_contrast_db=args.min_contrast_db,
            save_artifacts=True,
            dataset_source=args.dataset_source,
            dataset_license=args.dataset_license,
        )

        print("=" * 64)
        print(f" OILTRACE Detection Result: {result['incident_id']}")
        print("=" * 64)
        print(f"Status           : {result['detection_status']}")
        print(f"Method           : {result['detection_method']}")
        print(f"Candidate Regions: {result['detection_summary']['num_candidate_regions']}")
        print(f"Candidate Pixels : {result['detection_summary']['total_candidate_pixels']} ({result['detection_summary']['candidate_area_fraction']*100:.2f}%)")
        
        geo = result['spill_location_estimate']
        if geo and geo.get('centroid_latitude'):
            print(f"Primary Location : Lat {geo['centroid_latitude']:.5f}, Lon {geo['centroid_longitude']:.5f}")
            if geo.get('estimated_area_km2'):
                print(f"Estimated Area   : {geo['estimated_area_km2']:.3f} km²")
        else:
            print("Primary Location : Georeferencing unavailable (pixel coordinates only)")

        if result['reference_ground_truth']['evaluation_performed']:
            m = result['reference_ground_truth']['metrics']
            print("-" * 64)
            print(f"Reference Mask Evaluation: IoU={m['iou_jaccard']:.3f} | Dice={m['dice_f1']:.3f} | Prec={m['precision']:.3f} | Rec={m['recall']:.3f}")

        print(f"Mask Artifact    : {result['artifacts']['mask_path']}")
        print(f"Overlay Artifact : {result['artifacts']['overlay_path']}")
        print(f"Metadata JSON    : {result['artifacts']['metadata_json_path']}")
        print("=" * 64)

    except PreprocessingError as pe:
        print(f"[Error] Preprocessing failed: {pe}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"[Fatal Error] Pipeline execution failed: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
