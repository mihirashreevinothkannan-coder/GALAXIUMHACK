"""
Unit tests for SAR Segmentation and Evaluation module.
"""
import unittest
import numpy as np

from detection.segmentation import (
    segment_classical_dark_patch,
    evaluate_segmentation,
    morphological_open,
    morphological_close,
)


class TestSegmentation(unittest.TestCase):
    def test_synthetic_dark_slick_detection(self):
        # Create a 200x200 canvas with sea background ~50.0
        canvas = np.full((200, 200), 50.0, dtype=np.float32)
        # Inject an oil slick dark patch ~15.0 in the center (radius 25)
        y, x = np.ogrid[:200, :200]
        slick_mask = ((x - 100)**2 + (y - 100)**2) <= 25**2
        canvas[slick_mask] = 15.0

        mask, stats = segment_classical_dark_patch(canvas, threshold_factor=1.0)

        # Center should be detected as candidate slick
        self.assertTrue(mask[100, 100])
        # Background corner should NOT be detected
        self.assertFalse(mask[10, 10])
        # Detected slick pixels should closely match ground-truth radius 25
        detected_count = np.sum(mask[slick_mask])
        self.assertGreater(detected_count, 0.85 * np.sum(slick_mask))

    def test_clean_sea_zero_or_low_detection(self):
        # Uniform sea with mild variation
        canvas = np.full((100, 100), 45.0, dtype=np.float32) + np.random.normal(0, 1.0, (100, 100)).astype(np.float32)
        mask, stats = segment_classical_dark_patch(canvas, threshold_factor=1.5)

        # Almost no pixels should be flagged in uniform calm sea
        self.assertLess(np.sum(mask), 20)

    def test_evaluation_metrics_perfect_overlap(self):
        mask1 = np.zeros((50, 50), dtype=bool)
        mask1[10:30, 10:30] = True
        mask2 = mask1.copy()

        metrics = evaluate_segmentation(mask1, mask2)
        self.assertEqual(metrics["iou_jaccard"], 1.0)
        self.assertEqual(metrics["dice_f1"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)

    def test_evaluation_metrics_disjoint(self):
        mask1 = np.zeros((50, 50), dtype=bool)
        mask1[0:10, 0:10] = True

        mask2 = np.zeros((50, 50), dtype=bool)
        mask2[20:30, 20:30] = True

        metrics = evaluate_segmentation(mask1, mask2)
        self.assertEqual(metrics["iou_jaccard"], 0.0)
        self.assertEqual(metrics["dice_f1"], 0.0)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)

    def test_morphological_filtering(self):
        # 1 isolated pixel
        raw = np.zeros((20, 20), dtype=bool)
        raw[5, 5] = True
        opened = morphological_open(raw, size=3)
        # Opening removes isolated single pixel
        self.assertFalse(opened[5, 5])

        # Solid 5x5 block with 1 hole in center
        solid = np.zeros((20, 20), dtype=bool)
        solid[5:10, 5:10] = True
        solid[7, 7] = False
        closed = morphological_close(solid, size=3)
        # Closing seals the 1-pixel hole
    def test_trimmed_sea_clutter_estimation_with_bright_features(self):
        # 100x100 canvas: 70% ocean background (100.0), 30% bright land/ice (240.0)
        canvas = np.full((100, 100), 100.0, dtype=np.float32)
        canvas[:, 70:] = 240.0  # bright land on the right

        # Inject a dark slick patch (60.0) in the ocean region
        canvas[40:60, 20:40] = 60.0

        mask, stats = segment_classical_dark_patch(canvas, threshold_factor=1.1)

        # Global mean is inflated by bright land
        self.assertGreater(stats["global_mean_intensity"], 130.0)
        # Trimmed sea mean isolates the ocean background (~100.0)
        self.assertLess(stats["sea_mean_intensity"], 115.0)
        self.assertLess(stats["sea_mean_intensity"], stats["global_mean_intensity"])

        # The dark slick in the ocean region is successfully segmented
        self.assertTrue(mask[50, 30])
        # Bright land is NOT segmented
        self.assertFalse(mask[50, 85])


if __name__ == "__main__":
    unittest.main()

