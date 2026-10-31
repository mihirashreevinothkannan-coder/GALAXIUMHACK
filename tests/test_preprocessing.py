"""
Unit tests for SAR Preprocessing module.
"""
import os
import unittest
import numpy as np
from PIL import Image

from detection.preprocessing import (
    load_sar_image,
    load_reference_mask,
    boxcar_filter_2d,
    multilook_speckle_reduction,
    PreprocessingError,
)


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/temp_fixtures"
        os.makedirs(self.test_dir, exist_ok=True)

        # Create a valid test image (100x80)
        self.valid_img_path = os.path.join(self.test_dir, "valid.png")
        arr = (np.random.rand(80, 100) * 255).astype(np.uint8)
        Image.fromarray(arr).save(self.valid_img_path)

        # Create a tiny invalid image (8x8)
        self.tiny_img_path = os.path.join(self.test_dir, "tiny.png")
        tiny_arr = (np.random.rand(8, 8) * 255).astype(np.uint8)
        Image.fromarray(tiny_arr).save(self.tiny_img_path)

        # Create a reference mask matching valid image (80, 100)
        self.valid_mask_path = os.path.join(self.test_dir, "valid_mask.png")
        mask_arr = np.zeros((80, 100), dtype=np.uint8)
        mask_arr[20:40, 30:60] = 255
        Image.fromarray(mask_arr).save(self.valid_mask_path)

    def tearDown(self):
        for f in [self.valid_img_path, self.tiny_img_path, self.valid_mask_path]:
            if os.path.exists(f):
                os.remove(f)
        if os.path.exists(self.test_dir):
            try:
                os.rmdir(self.test_dir)
            except OSError:
                pass

    def test_load_valid_image(self):
        img = load_sar_image(self.valid_img_path)
        self.assertEqual(img.shape, (80, 100))
        self.assertEqual(img.dtype, np.float32)

    def test_missing_image_raises_error(self):
        with self.assertRaises(PreprocessingError):
            load_sar_image("non_existent_file.png")

    def test_tiny_image_rejected(self):
        with self.assertRaises(PreprocessingError):
            load_sar_image(self.tiny_img_path)

    def test_reference_mask_dimension_mismatch(self):
        # Expecting shape (120, 120), but mask is (80, 100)
        with self.assertRaises(PreprocessingError):
            load_reference_mask(self.valid_mask_path, expected_shape=(120, 120))

    def test_reference_mask_valid_loading(self):
        mask = load_reference_mask(self.valid_mask_path, expected_shape=(80, 100))
        self.assertEqual(mask.shape, (80, 100))
        self.assertEqual(mask.dtype, bool)
        self.assertEqual(mask.sum(), 20 * 30)

    def test_multilook_speckle_reduction_shape_and_variance(self):
        # Adding high synthetic speckle noise
        noisy = np.random.gamma(shape=4.0, scale=10.0, size=(100, 100)).astype(np.float32)
        filtered = multilook_speckle_reduction(noisy, window_size=5)

        self.assertEqual(filtered.shape, (100, 100))
        # Filtering should significantly reduce noise variance
        self.assertLess(float(np.var(filtered)), float(np.var(noisy)))

    def test_unsupported_extension_raises_error(self):
        bad_file = os.path.join(self.test_dir, "document.pdf")
        with open(bad_file, "w") as f:
            f.write("not an image")
        try:
            with self.assertRaises(PreprocessingError):
                load_sar_image(bad_file)
        finally:
            if os.path.exists(bad_file):
                os.remove(bad_file)

    def test_empty_file_raises_error(self):
        empty_file = os.path.join(self.test_dir, "empty.png")
        with open(empty_file, "wb") as f:
            pass  # 0 bytes
        try:
            with self.assertRaises(PreprocessingError):
                load_sar_image(empty_file)
        finally:
            if os.path.exists(empty_file):
                os.remove(empty_file)

    def test_rgb_display_image_conversion(self):
        rgb_file = os.path.join(self.test_dir, "rgb_sample.jpg")
        arr_rgb = (np.random.rand(50, 60, 3) * 255).astype(np.uint8)
        Image.fromarray(arr_rgb).save(rgb_file)
        try:
            img = load_sar_image(rgb_file)
            self.assertEqual(img.shape, (50, 60))
            self.assertEqual(img.ndim, 2)
        finally:
            if os.path.exists(rgb_file):
                os.remove(rgb_file)

    def test_tiff_image_support(self):
        tif_file = os.path.join(self.test_dir, "test_sar.tif")
        arr_tif = (np.random.rand(60, 60) * 255).astype(np.uint8)
        Image.fromarray(arr_tif).save(tif_file)
        try:
            img = load_sar_image(tif_file)
            self.assertEqual(img.shape, (60, 60))
            self.assertEqual(img.dtype, np.float32)
        finally:
            if os.path.exists(tif_file):
                os.remove(tif_file)


if __name__ == "__main__":
    unittest.main()
