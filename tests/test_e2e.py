"""
End-to-End Integration Tests for OILTRACE Detection Pipeline.
"""
import json
import os
import unittest

from detection.detect_spill import run_detection_pipeline


class TestEndToEndPipeline(unittest.TestCase):
    def setUp(self):
        self.output_dir = "tests/test_outputs"
        self.catalog_path = "data/samples/scenes_catalog.json"
        self.slick_img = "data/samples/s1_cantarell_slick.png"
        self.ref_mask = "data/samples/s1_cantarell_reference.png"
        self.clean_img = "data/samples/s1_clean_sea.png"

    def tearDown(self):
        # Clean up created artifacts
        if os.path.exists(self.output_dir):
            for root, dirs, files in os.walk(self.output_dir, topdown=False):
                for f in files:
                    os.remove(os.path.join(root, f))
                for d in dirs:
                    os.rmdir(os.path.join(root, d))
            try:
                os.rmdir(self.output_dir)
            except OSError:
                pass

    def test_e2e_slick_detection_with_reference(self):
        result = run_detection_pipeline(
            image_path=self.slick_img,
            reference_mask_path=self.ref_mask,
            catalog_path=self.catalog_path,
            output_dir=self.output_dir,
            incident_id="TEST-SLICK-001"
        )

        # 1. Pipeline status
        self.assertEqual(result["incident_id"], "TEST-SLICK-001")
        self.assertEqual(result["detection_status"], "candidate_slick_detected")
        self.assertGreater(result["detection_summary"]["num_candidate_regions"], 0)
        self.assertGreater(result["detection_summary"]["total_candidate_pixels"], 1000)

        # 2. Geospatial coordinates
        self.assertTrue(result["georeferencing"]["available"])
        geo = result["spill_location_estimate"]
        self.assertIsNotNone(geo["centroid_latitude"])
        self.assertAlmostEqual(geo["centroid_latitude"], 19.378, delta=0.1)
        self.assertAlmostEqual(geo["centroid_longitude"], -92.035, delta=0.1)

        # 3. Reference evaluation
        self.assertTrue(result["reference_ground_truth"]["evaluation_performed"])
        metrics = result["reference_ground_truth"]["metrics"]
        self.assertGreater(metrics["iou_jaccard"], 0.50)
        self.assertGreater(metrics["dice_f1"], 0.65)

        # 4. Artifact existence
        mask_file = os.path.join(self.output_dir, "masks", "TEST-SLICK-001_mask.png")
        overlay_file = os.path.join(self.output_dir, "overlays", "TEST-SLICK-001_overlay.png")
        comp_file = os.path.join(self.output_dir, "overlays", "TEST-SLICK-001_reference_comparison.png")
        json_file = os.path.join(self.output_dir, "metadata", "TEST-SLICK-001.json")

        self.assertTrue(os.path.exists(mask_file), "Candidate mask PNG was not saved")
        self.assertTrue(os.path.exists(overlay_file), "Overlay PNG was not saved")
        self.assertTrue(os.path.exists(comp_file), "Reference comparison PNG was not saved")
        self.assertTrue(os.path.exists(json_file), "Metadata JSON was not saved")

        # Verify JSON is parseable and valid
        with open(json_file, "r", encoding="utf-8") as f:
            saved_json = json.load(f)
        self.assertEqual(saved_json["incident_id"], "TEST-SLICK-001")

    def test_e2e_clean_sea_negative_test(self):
        result = run_detection_pipeline(
            image_path=self.clean_img,
            catalog_path=self.catalog_path,
            output_dir=self.output_dir,
            incident_id="TEST-CLEAN-001",
            threshold_factor=1.5  # higher threshold for clean sea rejection
        )

        self.assertEqual(result["incident_id"], "TEST-CLEAN-001")
        # Should detect no slicks on uniform clean ocean
        self.assertEqual(result["detection_status"], "no_slick_detected")
        self.assertEqual(result["detection_summary"]["num_candidate_regions"], 0)


if __name__ == "__main__":
    unittest.main()
