"""
Unit tests for Feature Extraction module.
"""
import unittest
import numpy as np

from detection.feature_extraction import (
    label_connected_components,
    extract_candidate_regions,
)


class TestFeatureExtraction(unittest.TestCase):
    def test_connected_component_labeling(self):
        mask = np.zeros((50, 50), dtype=bool)
        # Component 1: 10x10 square
        mask[5:15, 5:15] = True
        # Component 2: 5x5 square
        mask[30:35, 30:35] = True

        labeled, num_features = label_connected_components(mask)
        self.assertEqual(num_features, 2)
        self.assertEqual(np.sum(labeled == 1), 100)
        self.assertEqual(np.sum(labeled == 2), 25)

    def test_bounding_box_and_centroid_calculation(self):
        h, w = 100, 100
        mask = np.zeros((h, w), dtype=bool)
        intensity = np.full((h, w), 50.0, dtype=np.float32)

        # Draw precise rectangle: y from 20 to 50 (inclusive: 31 pixels), x from 40 to 60 (inclusive: 21 pixels)
        # Pixel area = 31 * 21 = 651 pixels
        mask[20:51, 40:61] = True
        intensity[20:51, 40:61] = 18.0  # dark slick intensity

        # Also add a tiny 4-pixel speckle dot at (80, 80)
        mask[80:82, 80:82] = True
        intensity[80:82, 80:82] = 10.0

        candidates, filtered_mask = extract_candidate_regions(
            binary_mask=mask,
            sar_intensity=intensity,
            min_pixels=50,
            sea_mean=50.0
        )

        # Tiny 4-pixel speckle should be filtered out by min_pixels=50
        self.assertEqual(len(candidates), 1)
        c = candidates[0]

        # Verify exact pixel count
        self.assertEqual(c["pixel_count"], 651)

        # Verify exact bounding box [y_min, x_min, y_max, x_max]
        self.assertEqual(c["bounding_box_pixels"], [20, 40, 50, 60])

        # Centroid of [20..50] is (20+50)/2 = 35.0, of [40..60] is (40+60)/2 = 50.0
        self.assertAlmostEqual(c["centroid_pixel"][0], 35.0, delta=0.5)
        self.assertAlmostEqual(c["centroid_pixel"][1], 50.0, delta=0.5)

        # Contrast: 10 * log10(50 / 18) = 10 * 0.4437 ~ 4.44 dB
        self.assertGreater(c["radiometric_stats"]["contrast_db"], 4.0)

    def test_zero_candidate_handling(self):
        empty_mask = np.zeros((80, 80), dtype=bool)
        intensity = np.full((80, 80), 45.0, dtype=np.float32)

        candidates, filtered = extract_candidate_regions(empty_mask, intensity)
        self.assertEqual(len(candidates), 0)
    def test_shadow_suspect_classification(self):
        h, w = 120, 120
        intensity = np.full((h, w), 100.0, dtype=np.float32)
        mask = np.zeros((h, w), dtype=bool)

        # Place a bright structure (e.g. ship or cliff) at [20:40, 20:40]
        intensity[20:40, 20:40] = 240.0

        # Dark patch 1: Immediately adjacent to bright structure (within 5 px)
        # Bounding box [20:40, 42:58]
        mask[20:40, 42:58] = True
        intensity[20:40, 42:58] = 40.0

        # Dark patch 2: Far away in open sea (elongated shape)
        # Bounding box [80:88, 60:110]
        mask[80:88, 60:110] = True
        intensity[80:88, 60:110] = 40.0

        candidates, filtered_mask = extract_candidate_regions(
            binary_mask=mask,
            sar_intensity=intensity,
            min_pixels=50,
            sea_mean=100.0,
            bright_pixel_threshold=200.0,
            shadow_proximity_px=15
        )

        self.assertEqual(len(candidates), 2)
        # Find candidate near bright structure (y_min < 50) vs open sea candidate (y_min >= 50)
        near_cand = [c for c in candidates if c["bounding_box_pixels"][0] < 50][0]
        open_cand = [c for c in candidates if c["bounding_box_pixels"][0] >= 50][0]

        self.assertEqual(near_cand["classification"], "shadow_suspect")
        self.assertEqual(near_cand["uncertainty_level"], "high")

        self.assertEqual(open_cand["classification"], "suspected_mineral_oil_slick")
        self.assertEqual(open_cand["uncertainty_level"], "moderate")


if __name__ == "__main__":
    unittest.main()

