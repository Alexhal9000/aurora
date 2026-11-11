"""Shell-peak background offset estimator on synthetic phantoms."""

import unittest

import numpy as np

from pipeline.backgroundOffsetEstimation import (
    REGION_OUTSIDE,
    REGION_SURROUNDING_TISSUE,
    choose_sampling_stride,
    estimate_background_peak,
    normalize_region,
    peak_spread,
    range_pile_up,
)


def _phantom(shape=(96, 96, 96), air=3000.0, cavity=None, tissue=20000.0,
             noise=400.0, pad=False, seed=0):
    """Noisy sphere in air, optionally with an enclosed cavity and zero padding."""
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.indices(shape, dtype=np.float32)
    centre = np.array(shape, dtype=np.float32) / 2.0
    r = np.sqrt((zz - centre[0]) ** 2 + (yy - centre[1]) ** 2 + (xx - centre[2]) ** 2)
    volume = np.full(shape, air, dtype=np.float32)
    volume[r <= 30] = tissue
    if cavity is not None:
        volume[r <= 14] = cavity
    volume += rng.normal(0.0, noise, size=shape).astype(np.float32)
    volume = np.clip(volume, 0, 65535).astype(np.uint16)
    if pad:
        # Reorientation-style constant fill in one corner block.
        volume[:20, :20, :] = 0
    return volume


class BackgroundOffsetEstimationTests(unittest.TestCase):
    def test_outside_recovers_air_peak(self):
        volume = _phantom()
        value, info = estimate_background_peak(volume, 10000, region=REGION_OUTSIDE)
        self.assertIsNotNone(value, info)
        self.assertLess(abs(value - 3000), 60)
        self.assertEqual(info['method'], 'distance_guarded_peak')

    def test_outside_ignores_enclosed_cavity(self):
        volume = _phantom(cavity=7000.0)
        value, _ = estimate_background_peak(volume, 10000, region=REGION_OUTSIDE)
        self.assertLess(abs(value - 3000), 60)

    def test_outside_prefers_far_air_over_soft_tissue(self):
        # Bone threshold inside a soft-tissue body: the tissue right next to
        # the foreground is below threshold but is not air.
        rng = np.random.default_rng(1)
        shape = (96, 96, 96)
        zz, yy, xx = np.indices(shape, dtype=np.float32)
        r = np.sqrt((zz - 48) ** 2 + (yy - 48) ** 2 + (xx - 48) ** 2)
        volume = np.full(shape, 1000.0, dtype=np.float32)
        # Body fills most of the field of view, as in whole-animal CT. (A thin
        # tissue layer is covered by the air-exclusion test below.)
        volume[r <= 40] = 8000.0
        volume[r <= 12] = 20000.0
        volume += rng.normal(0.0, 300.0, size=shape).astype(np.float32)
        volume = np.clip(volume, 0, 65535).astype(np.uint16)
        value, _ = estimate_background_peak(volume, 15000, region=REGION_OUTSIDE)
        self.assertLess(abs(value - 1000), 80)
        # The complementary near band finds the soft tissue instead.
        tissue, info = estimate_background_peak(volume, 15000, region=REGION_SURROUNDING_TISSUE)
        self.assertLess(abs(tissue - 8000), 80)
        self.assertEqual(info['region'], REGION_SURROUNDING_TISSUE)

    def test_tissue_found_when_thin_layer_shares_near_band_with_air(self):
        # Thin soft tissue around bone (skulls, limbs): the near band holds
        # more tissue than air, but air is a much narrower peak, so without
        # air exclusion its fullest histogram bin wins (seen on Scanco skull
        # CT, where tissue ~35,000 lost to air ~32,790).
        rng = np.random.default_rng(2)
        shape = (96, 96, 96)
        zz, yy, xx = np.indices(shape, dtype=np.float32)
        r = np.sqrt((zz - 48) ** 2 + (yy - 48) ** 2 + (xx - 48) ** 2)
        volume = np.full(shape, 1000.0, dtype=np.float32)
        noise = np.full(shape, 60.0, dtype=np.float32)
        volume[r <= 30] = 8000.0
        noise[r <= 30] = 900.0
        volume[r <= 10] = 20000.0
        volume += rng.normal(0.0, 1.0, size=shape).astype(np.float32) * noise
        volume = np.clip(volume, 0, 65535).astype(np.uint16)
        tissue, info = estimate_background_peak(volume, 15000, region=REGION_SURROUNDING_TISSUE)
        self.assertLess(abs(tissue - 8000), 250, info)
        self.assertEqual(info['air_exclusion'], 'applied')
        self.assertLess(abs(info['air_peak'] - 1000), 40)
        # Surrounding air is unaffected by the exclusion step.
        air, _ = estimate_background_peak(volume, 15000, region=REGION_OUTSIDE)
        self.assertLess(abs(air - 1000), 40)

    def test_tissue_shell_next_to_specimen_deep_in_large_body(self):
        # Bone deep inside a body that sits in low-density packing (paper,
        # foam) inside a holder, as in SkyScan whole-mouse CT (F20G1F08): the
        # median-distance near band is mostly packing (its volume grows with
        # distance) and packing is not the air peak that air exclusion
        # removes, so the pooled band peaked on packing. The innermost shell
        # next to the specimen holds the tissue.
        rng = np.random.default_rng(3)
        shape = (128, 128, 128)
        zz, yy, xx = np.indices(shape, dtype=np.float32)
        r = np.sqrt((zz - 64) ** 2 + (yy - 64) ** 2 + (xx - 64) ** 2)
        volume = np.full(shape, 1000.0, dtype=np.float32)
        volume[r <= 50] = 2500.0
        volume[r <= 24] = 8000.0
        volume[r <= 6] = 20000.0
        volume += rng.normal(0.0, 300.0, size=shape).astype(np.float32)
        volume = np.clip(volume, 0, 65535).astype(np.uint16)
        tissue, info = estimate_background_peak(volume, 15000, region=REGION_SURROUNDING_TISSUE)
        self.assertLess(abs(tissue - 8000), 150, info)
        self.assertLess(
            info['tissue_shell_outer_sampled_voxels'],
            info['split_distance_sampled_voxels'],
        )

    def test_air_spread_on_clipped_range(self):
        # 8-bit air clipped at 0: the side of the air peak facing 0 is cut
        # off and looks narrow, so the narrower-side rule underestimates the
        # spread; the half-maximum width of the other side does not, even
        # with tissue in that side's tail.
        rng = np.random.default_rng(4)
        air = np.clip(rng.normal(0.8, 1.0, 400_000), 0.0, None)
        tissue = rng.normal(8.0, 1.5, 100_000)
        values = np.concatenate([air, tissue])
        low, high = range_pile_up(values)
        self.assertTrue(low)
        self.assertFalse(high)
        # Spread around the known air mode (the estimator measures it on
        # blurred values, where the pile-up at 0 is smeared out).
        peak = 0.8
        naive, rule = peak_spread(values, peak)
        self.assertEqual(rule, 'narrower_side_rms')
        spread, rule = peak_spread(values, peak, clipped_low=low, clipped_high=high)
        self.assertEqual(rule, 'half_max_upper')
        self.assertLess(abs(spread - 1.0), 0.15)
        self.assertLess(naive, 0.7)
        # Unclipped values keep the narrower-side rule.
        self.assertEqual(range_pile_up(rng.normal(100.0, 5.0, 10_000)), (False, False))

    def test_constant_padding_is_ignored(self):
        volume = _phantom(pad=True)
        value, _ = estimate_background_peak(volume, 10000, region=REGION_OUTSIDE)
        self.assertLess(abs(value - 3000), 60)

    def test_specimen_filling_view_falls_back(self):
        volume = np.full((32, 32, 32), 20000, dtype=np.uint16)
        value, info = estimate_background_peak(volume, 10000)
        self.assertIsNone(value)
        self.assertIn('fallback_reason', info)

    def test_uint8_volume(self):
        volume = _phantom(air=40.0, tissue=200.0, noise=6.0).astype(np.float32)
        volume = np.clip(volume, 0, 255).astype(np.uint8)
        value, _ = estimate_background_peak(volume, 120)
        self.assertLess(abs(value - 40), 2)

    def test_stride_and_region_helpers(self):
        self.assertEqual(choose_sampling_stride((100, 100, 100)), 2)
        self.assertGreaterEqual(choose_sampling_stride((2000, 2000, 2000)), 8)
        self.assertEqual(normalize_region('bogus'), REGION_OUTSIDE)
        self.assertEqual(normalize_region('surrounding_tissue'), REGION_SURROUNDING_TISSUE)


if __name__ == '__main__':
    unittest.main()
