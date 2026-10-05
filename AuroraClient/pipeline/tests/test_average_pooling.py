"""Average pooling used when extracting scans at a scaling factor above 1."""

import unittest

import numpy as np

from pipeline.averagePooling import (
    apply_extract_scaling,
    average_pool_volume,
    normalize_scaling_factor,
)


class AveragePoolingTests(unittest.TestCase):
    def test_factor_one_is_identity(self):
        volume = np.arange(8, dtype=np.uint16).reshape(2, 2, 2)
        pooled, spacing, factor = apply_extract_scaling(volume, 0.25, 1)
        self.assertIs(pooled, volume)
        self.assertEqual(spacing, 0.25)
        self.assertEqual(factor, 1)
        self.assertEqual(pooled.dtype, np.uint16)

    def test_invalid_factors_fall_back_to_one(self):
        volume = np.ones((2, 2, 2), dtype=np.uint16)
        for raw in (None, 'nope', 0, 5, -2, 1.5):
            pooled, spacing, factor = apply_extract_scaling(volume, 1.0, raw)
            self.assertEqual(factor, 1)
            self.assertIs(pooled, volume)
            self.assertEqual(spacing, 1.0)
        self.assertEqual(normalize_scaling_factor('3'), 3)
        self.assertEqual(normalize_scaling_factor(2.9), 2)

    def test_factor_two_block_mean(self):
        volume = np.array([
            [[1, 3], [5, 7]],
            [[9, 11], [13, 15]],
        ], dtype=np.uint16)
        pooled = average_pool_volume(volume, 2)
        self.assertEqual(pooled.shape, (1, 1, 1))
        self.assertEqual(int(pooled[0, 0, 0]), 8)
        self.assertEqual(pooled.dtype, np.uint16)

    def test_factors_three_and_four_block_means(self):
        volume3 = np.full((3, 3, 3), 6, dtype=np.uint16)
        pooled3 = average_pool_volume(volume3, 3)
        self.assertEqual(pooled3.shape, (1, 1, 1))
        self.assertEqual(int(pooled3[0, 0, 0]), 6)

        volume4 = np.full((4, 4, 4), 4, dtype=np.uint16)
        pooled4 = average_pool_volume(volume4, 4)
        self.assertEqual(pooled4.shape, (1, 1, 1))
        self.assertEqual(int(pooled4[0, 0, 0]), 4)
        self.assertEqual(pooled4.dtype, np.uint16)

    def test_edge_pad_uses_the_last_plane(self):
        # Length 3 is padded to 4 by repeating the last plane, then pooled by 2.
        volume = np.array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
            [[9, 10], [11, 12]],
        ], dtype=np.uint16)
        pooled, spacing, factor = apply_extract_scaling(volume, 0.116, 2)
        self.assertEqual(factor, 2)
        self.assertEqual(pooled.shape, (2, 1, 1))
        # (1+2+3+4+5+6+7+8) / 8 = 4.5, rounded to even 4.
        self.assertEqual(int(pooled[0, 0, 0]), 4)
        # Repeated last plane: (9+10+11+12) / 4 = 10.5, rounded to even 10.
        self.assertEqual(int(pooled[1, 0, 0]), 10)
        self.assertEqual(pooled.dtype, np.uint16)
        self.assertAlmostEqual(spacing, 0.232)

    def test_float_volumes_keep_the_mean(self):
        volume = np.array([
            [[1, 3], [5, 7]],
            [[9, 11], [13, 15]],
        ], dtype=np.float32)
        pooled = average_pool_volume(volume, 2)
        self.assertEqual(pooled.dtype, np.float32)
        self.assertAlmostEqual(float(pooled[0, 0, 0]), 8.0)


if __name__ == '__main__':
    unittest.main()
