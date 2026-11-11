"""Masking tools must save the voxels they computed, not a rescaled copy (fixed 2026-11-09).

Before the fix, Apply mask, Batch apply mask, Mask out labels and the elastic copies saved
float64 data under the scan's integer header. nibabel then rescaled the values to the full
integer range and wrote scl_slope, and on reload about half the voxels of a uint16 scan
peaking below 65535 came back one grey level lower. maskOperations._masked_voxels_for_save
casts back to the stored dtype first. These tests save through it and reload the file.

They need pipeline.maskOperations (the full Aurora environment) and are skipped otherwise.

Run from AuroraClient:  python manage.py test pipeline.tests.test_masked_save_dtype
"""

import importlib
import os
import tempfile
import unittest

import numpy as np
import nibabel as nib


def _mask_ops():
    try:
        return importlib.import_module('pipeline.maskOperations'), None
    except Exception as exc:  # pragma: no cover - depends on the environment
        return None, f'pipeline.maskOperations not importable here ({type(exc).__name__}: {exc})'


class MaskedSaveDtypeTests(unittest.TestCase):

    def setUp(self):
        self.ops, reason = _mask_ops()
        if self.ops is None:
            self.skipTest(reason)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _source(self, data, slope=None):
        """Save `data` as a source scan (optionally with a real scl_slope) and load it back."""
        header = nib.Nifti1Header()
        header.set_data_dtype(data.dtype)
        img = nib.Nifti1Image(data, np.eye(4), header)
        if slope is not None:
            img.header.set_slope_inter(slope, 0.0)
        path = os.path.join(self.tmp.name, f'source_{len(os.listdir(self.tmp.name))}.nii.gz')
        nib.save(img, path)
        return nib.load(path)

    def _mask_and_save(self, src, keep, use_helper=True):
        """What the masking tools do: float64 load, zero outside `keep`, save with the source header."""
        masked = src.get_fdata() * keep.astype(np.uint8)
        data = self.ops._masked_voxels_for_save(masked, src) if use_helper else masked
        path = os.path.join(self.tmp.name, f'out_{len(os.listdir(self.tmp.name))}.nii.gz')
        nib.save(nib.Nifti1Image(data, src.affine, header=src.header), path)
        return masked, nib.load(path)

    def _scan(self, dtype, top, seed=0):
        rng = np.random.default_rng(seed)
        data = rng.integers(0, top + 1, (24, 20, 16)).astype(dtype)
        data[0, 0, 0] = top  # brightest voxel below the dtype maximum: the case that rescaled
        return data, rng.random(data.shape) > 0.3

    def test_old_route_rescaled_the_voxels(self):
        # Documents the bug: without the helper, the saved uint16 file carries a slope != 1
        # and the usual reload (get_fdata().astype) loses one grey level on many voxels.
        data, keep = self._scan(np.uint16, 9000)
        masked, out = self._mask_and_save(self._source(data), keep, use_helper=False)
        self.assertNotEqual(out.dataobj.slope, 1.0)
        reloaded = out.get_fdata().astype(np.uint16)
        self.assertGreater(np.mean(reloaded != masked.astype(np.uint16)), 0.1)

    def test_integer_scans_save_exactly(self):
        for dtype, top in ((np.uint8, 199), (np.uint8, 255), (np.uint16, 9000), (np.int16, 9000), (np.uint16, 65535)):
            data, keep = self._scan(dtype, top, seed=top)
            masked, out = self._mask_and_save(self._source(data), keep)
            expected = np.where(keep, data, 0).astype(dtype)
            self.assertEqual(out.get_data_dtype(), np.dtype(dtype), f'{np.dtype(dtype)} max {top}')
            self.assertEqual((out.dataobj.slope, out.dataobj.inter), (1.0, 0.0), f'{np.dtype(dtype)} max {top}')
            self.assertTrue(np.array_equal(np.asanyarray(out.dataobj), expected), f'{np.dtype(dtype)} max {top}')
            self.assertTrue(np.array_equal(out.get_fdata().astype(dtype), expected), f'{np.dtype(dtype)} max {top}')

    def test_negative_int16_values_survive(self):
        data, keep = self._scan(np.int16, 3000, seed=4)
        data[1:4] -= 2000  # CT-style negative values
        masked, out = self._mask_and_save(self._source(data), keep)
        self.assertTrue(np.array_equal(np.asanyarray(out.dataobj), np.where(keep, data, 0).astype(np.int16)))

    def test_float_scans_unchanged(self):
        # Float scans were never rescaled: the helper passes them through untouched.
        data = np.random.default_rng(5).normal(100, 30, (12, 10, 8)).astype(np.float32)
        src = self._source(data)
        masked = src.get_fdata() * (data > 90)
        self.assertIs(self.ops._masked_voxels_for_save(masked, src), masked)

    def test_scaled_source_keeps_old_behaviour(self):
        # A source with a real scl_slope has non-integer voxel values; the helper must not
        # truncate them, so it leaves the float data for nibabel to scale, as before.
        data, keep = self._scan(np.uint16, 9000, seed=6)
        src = self._source(data, slope=0.25)
        masked = src.get_fdata() * keep
        self.assertIs(self.ops._masked_voxels_for_save(masked, src), masked)


if __name__ == '__main__':
    unittest.main()
