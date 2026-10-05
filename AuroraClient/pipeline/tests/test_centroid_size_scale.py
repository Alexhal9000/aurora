"""Centroid-size scale used as the initial step of rigid align-to-reference."""
import unittest

import numpy as np

from pipeline.rigidAlignment import (
    centroid_size,
    centroid_size_scale_factor,
    post_threshold_mesh_vertices,
    zoom_volume_to_scale,
)


def _ball(shape, radius, center):
    zz, yy, xx = np.ogrid[:shape, :shape, :shape]
    return (
        (zz - center) ** 2 + (yy - center) ** 2 + (xx - center) ** 2
    ) <= radius ** 2


class TestCentroidSizeScale(unittest.TestCase):
    def test_centroid_size_is_rms_distance(self):
        points = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64)
        self.assertAlmostEqual(centroid_size(points), 1.0)

    def test_scale_factor_matches_reference_size(self):
        reference = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        subject = reference * 2.0
        self.assertAlmostEqual(centroid_size_scale_factor(reference, subject), 0.5)
        self.assertAlmostEqual(centroid_size_scale_factor(subject, reference), 2.0)

    def test_empty_or_degenerate_points_do_not_scale(self):
        points = np.array([[1.0, 2.0, 3.0]])
        self.assertEqual(centroid_size_scale_factor(points, points), 1.0)
        self.assertEqual(centroid_size_scale_factor(np.zeros((0, 3)), points), 1.0)

    def test_post_threshold_mesh_scales_with_radius(self):
        small = _ball(48, 8, 24).astype(np.float32)
        large = _ball(48, 16, 24).astype(np.float32)
        small_pts = post_threshold_mesh_vertices(small, 0.5, 1.0, smooth=False)
        large_pts = post_threshold_mesh_vertices(large, 0.5, 1.0, smooth=False)
        self.assertIsNotNone(small_pts)
        self.assertIsNotNone(large_pts)
        ratio = centroid_size(large_pts) / centroid_size(small_pts)
        self.assertGreater(ratio, 1.7)
        self.assertLess(ratio, 2.3)
        scale = centroid_size_scale_factor(large_pts, small_pts)
        self.assertAlmostEqual(scale, ratio)

    def test_zoom_leaves_near_identity_and_spacing_implied_size(self):
        volume = np.ones((4, 4, 4), dtype=np.float32)
        unchanged, applied = zoom_volume_to_scale(volume, 1.00001, 0)
        self.assertIs(unchanged, volume)
        self.assertEqual(applied, 1.0)
        zoomed, applied = zoom_volume_to_scale(volume, 2.0, 0)
        self.assertEqual(applied, 2.0)
        self.assertEqual(zoomed.shape, (8, 8, 8))


if __name__ == "__main__":
    unittest.main()
