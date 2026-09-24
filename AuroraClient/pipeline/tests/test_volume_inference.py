from django.test import SimpleTestCase

import numpy as np

from pipeline.aiFineTuning.volume_inference import (
    build_full_slice_reference_mask,
    choose_init_slice_from_extent,
    estimate_extent_along_axis,
    estimate_tissue_slice_extent,
    inverse_reorient_for_view,
    reorient_for_view,
)


class VolumeInferenceHelperTests(SimpleTestCase):
    def test_estimate_tissue_slice_extent_finds_central_band(self):
        image = np.zeros((20, 8, 8), dtype=np.uint8)
        image[6:14] = 120
        lo, hi = estimate_tissue_slice_extent(image)
        self.assertLessEqual(lo, 6)
        self.assertGreaterEqual(hi, 13)

    def test_estimate_extent_along_coronal_axis(self):
        image = np.zeros((8, 20, 8), dtype=np.uint8)
        image[:, 6:14, :] = 120
        reoriented = reorient_for_view(image, 1)
        lo, hi = estimate_extent_along_axis(reoriented, axis=0)
        self.assertLessEqual(lo, 6)
        self.assertGreaterEqual(hi, 13)

    def test_reorient_round_trip(self):
        volume = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
        for view in range(3):
            back = inverse_reorient_for_view(reorient_for_view(volume, view), view)
            self.assertTrue(np.array_equal(back, volume))

    def test_choose_init_slice_from_extent_picks_middle(self):
        center = choose_init_slice_from_extent(4, 11, slab_fraction=0.5)
        self.assertGreaterEqual(center, 4)
        self.assertLessEqual(center, 11)

    def test_live_inference_batch_stays_capped(self):
        from pipeline.aiFineTuning.volume_inference import INFERENCE_MAX_BATCH
        self.assertEqual(INFERENCE_MAX_BATCH, 2)

    def test_chunk_indices_keeps_order(self):
        from pipeline.aiFineTuning.volume_inference import _chunk_indices

        chunks = _chunk_indices(list(range(10, 20)), chunk=4)
        self.assertEqual(chunks, [[10, 11, 12, 13], [14, 15, 16, 17], [18, 19]])

    def test_full_slice_prompt_does_not_chain_previous_mask(self):
        from pipeline.aiFineTuning.volume_inference import _initial_prompt

        dense, bbox = _initial_prompt(np.zeros((8, 8), dtype=np.uint8), 8, full_slice=True)
        self.assertTrue(np.allclose(dense, 1.0))
        self.assertEqual(list(bbox), [0.0, 0.0, 7.0, 7.0])
        mask = build_full_slice_reference_mask((10, 6, 7), prompt_z=4)
        self.assertEqual(int(np.sum(mask)), 6 * 7)
        self.assertTrue(np.all(mask[4] == 1))
        self.assertTrue(np.all(mask[3] == 0))
