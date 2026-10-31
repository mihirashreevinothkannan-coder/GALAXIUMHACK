"""
Unit tests for Geospatial Mapping module.
"""
import unittest

from detection.geospatial import (
    parse_sentinel1_timestamp,
    map_pixel_to_latlon,
    attach_geospatial_metadata,
)


class TestGeospatial(unittest.TestCase):
    def test_parse_sentinel1_timestamp(self):
        prod = "S1A_IW_GRDH_1SDV_20200117T001541_8b.tif"
        ts = parse_sentinel1_timestamp(prod)
        self.assertEqual(ts, "2020-01-17T00:15:41Z")

        # Invalid string returns None
        self.assertIsNone(parse_sentinel1_timestamp("random_image_file.png"))

    def test_affine_pixel_mapping(self):
        bbox = {
            "min_lon": -92.5,
            "max_lon": -91.5,
            "min_lat": 19.0,
            "max_lat": 20.0
        }
        w, h = 1000, 500

        # Top-left corner (x=0, y=0) should map to (max_lat, min_lon) = (20.0, -92.5)
        lat, lon = map_pixel_to_latlon(0, 0, w, h, bbox)
        self.assertEqual(lat, 20.0)
        self.assertEqual(lon, -92.5)

        # Bottom-right corner (x=1000, y=500) should map to (min_lat, max_lon) = (19.0, -91.5)
        lat, lon = map_pixel_to_latlon(1000, 500, w, h, bbox)
        self.assertEqual(lat, 19.0)
        self.assertEqual(lon, -91.5)

        # Center (x=500, y=250) should map to (19.5, -92.0)
        lat, lon = map_pixel_to_latlon(500, 250, w, h, bbox)
        self.assertEqual(lat, 19.5)
        self.assertEqual(lon, -92.0)

    def test_unreferenced_fallback(self):
        candidates = [{
            "candidate_id": 1,
            "pixel_count": 100,
            "bounding_box_pixels": [10, 10, 20, 20],
            "centroid_pixel": [15.0, 15.0]
        }]

        # When scene metadata is None
        res = attach_geospatial_metadata(candidates, 100, 100, scene_meta=None)
        self.assertFalse(res["georeferencing_available"])
        self.assertIsNone(candidates[0]["geographic_coordinates"])
        self.assertIn("No georeferencing", res["georeferencing_note"])


if __name__ == "__main__":
    unittest.main()
