"""The 2026-11-09 memory changes must not change a single voxel.

Each lower-memory path is compared with a frozen copy of the code it replaced
(pipeline/tests/legacy_memory_reference.py) on small synthetic volumes chosen to reach
every branch: empty and full masks, one and many components, Gaussian blur on and off,
uint8 and uint16 scans, targets at and beyond the dtype edges, scaled NIfTI headers, and so on.
"Same" means same dtype, shape and bytes (np.array_equal is not enough for dtype changes).

- HelperTests and NiftiVoxelsTests need only numpy, scipy and nibabel, and run in about a second.
- RealViewTests call the real methods in preprocessingTools, views, rigidAlignment and
  dinoRegRigid, so they need the full Aurora environment. They are skipped, with the
  reason shown, when those modules cannot be imported.

Run from AuroraClient:  python manage.py test pipeline.tests.test_low_memory_volume_ops
"""

import contextlib
import gzip
import io
import itertools
import os
import tempfile
import unittest

import numpy as np
import nibabel as nib

from pipeline import lowMemoryVolumeOps as low
from pipeline.niftiVoxels import nifti_voxels
from pipeline.tests import legacy_memory_reference as legacy


def _quiet(fn, *args, **kwargs):
    """Call fn with its print() chatter swallowed (both old and new code print a lot)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


class SameArrayMixin:
    def assertSameArray(self, expected, actual, msg=''):
        expected = np.asarray(expected)
        actual = np.asarray(actual)
        self.assertEqual(expected.dtype, actual.dtype, f'dtype {msg}')
        self.assertEqual(expected.shape, actual.shape, f'shape {msg}')
        self.assertTrue(
            expected.tobytes(order='C') == actual.tobytes(order='C'),
            f'values differ {msg}',
        )


def _scan(shape=(40, 36, 32), dtype=np.uint8, seed=0, islands=True, speckle=True, blob=True):
    """Synthetic micro-CT-like scan: noisy background, one large object, optional small
    islands and above-threshold speckles (the JPEG noise case)."""
    rng = np.random.default_rng(seed)
    vol = rng.integers(5, 40, size=shape).astype(np.float64)
    z, y, x = np.indices(shape)
    if blob:
        c = np.array(shape) / 2.0
        r = min(shape) / 3.0
        vol[(z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2 < r ** 2] = 200
    if islands:
        vol[2:5, 2:5, 2:5] = 150
        vol[-6:-3, 3:6, -7:-4] = 160
        vol[3:4, -4:-2, 10:12] = 140
    if speckle:
        idx = rng.integers(0, vol.size, size=max(1, vol.size // 400))
        vol.flat[idx] = 180
    if np.dtype(dtype) == np.uint16:
        vol = vol * 250
    return vol.astype(dtype)


class HelperTests(SameArrayMixin, unittest.TestCase):
    def test_first_true_indices_matches_argwhere_head(self):
        rng = np.random.default_rng(1)
        masks = [
            np.zeros((5, 6, 7), bool),
            np.ones((3, 4, 5), bool),
            rng.random((9, 8, 7)) < 0.002,
            rng.random((9, 8, 7)) < 0.3,
            np.asfortranarray(rng.random((6, 5, 4)) < 0.1),
            rng.random((7, 11)) < 0.05,
        ]
        for m, count in itertools.product(masks, (1, 3, 5, 1000)):
            self.assertSameArray(np.argwhere(m)[:count], low.first_true_indices(m, count), f'{m.shape} {count}')

    def test_axis_extents_matches_argwhere_min_max(self):
        rng = np.random.default_rng(2)
        for shape, p in [((8, 9, 10), 0.01), ((8, 9, 10), 0.5), ((1, 9, 10), 0.2), ((5, 5), 0.1)]:
            m = rng.random(shape) < p
            pts = np.argwhere(m)
            got = low.axis_extents(m)
            if len(pts) == 0:
                self.assertIsNone(got)
                continue
            self.assertSameArray(np.min(pts, axis=0), got[0], f'min {shape}')
            self.assertSameArray(np.max(pts, axis=0), got[1], f'max {shape}')
        self.assertIsNone(low.axis_extents(np.zeros((4, 4, 4), bool)))
        single = np.zeros((6, 7, 8), bool)
        single[5, 0, 7] = True
        self.assertSameArray(np.array([5, 0, 7]), low.axis_extents(single)[0])

    def test_label_sizes_matches_bincount(self):
        from scipy import ndimage
        rng = np.random.default_rng(9)
        saved = low._BINCOUNT_CHUNK
        try:
            for chunk in (saved, 1000, 4097):
                low._BINCOUNT_CHUNK = chunk
                for p in (0.0, 0.01, 0.3, 1.0):
                    labeled, n = ndimage.label(rng.random((23, 19, 17)) < p)
                    self.assertSameArray(np.bincount(labeled.ravel()), low.label_sizes(labeled, n), f'chunk={chunk} p={p}')
            empty = np.zeros((0, 3, 3), np.int32)
            self.assertSameArray(np.bincount(empty.ravel()), low.label_sizes(empty, 0))
        finally:
            low._BINCOUNT_CHUNK = saved

    def test_homogenize_unsigned_matches_int32_path(self):
        legacy_view = legacy.LegacyHomogenizeBackground()
        for dtype in (np.uint8, np.uint16):
            data = _scan(dtype=dtype, seed=3)
            top = int(np.iinfo(dtype).max)
            for target in (0, 1, 17, 17.9, 39, int(data.max()), top - 1, top):
                expected = _quiet(legacy_view._homogenize_background, data.copy(), target)
                before = data.copy()
                got = low.homogenize_unsigned_background(data, target)
                self.assertIsNotNone(got, f'{dtype} {target}')
                self.assertSameArray(expected, got, f'{np.dtype(dtype)} target={target}')
                self.assertSameArray(before, data, 'input must not be modified')
            for target in (-1, -0.5, top + 1, 10 ** 9):
                expected_none = low.homogenize_unsigned_background(data, target)
                if int(target) < 0 or int(target) > top:
                    self.assertIsNone(expected_none, f'{target} keeps the int32 path')
                else:
                    self.assertSameArray(_quiet(legacy_view._homogenize_background, data.copy(), target), expected_none)
        self.assertIsNone(low.homogenize_unsigned_background(np.zeros((2, 2, 2), np.int16), 3))
        self.assertIsNone(low.homogenize_unsigned_background(np.zeros((2, 2, 2), np.float32), 3))

    def test_gaussian_float32_matches_float32_copy(self):
        from scipy.ndimage import gaussian_filter
        for dtype in (np.uint8, np.int8, np.uint16, np.int16, np.float32, np.float64, np.int32):
            info = np.iinfo(dtype) if np.dtype(dtype).kind in 'iu' else None
            rng = np.random.default_rng(4)
            if info is not None:
                data = rng.integers(info.min, info.max, size=(20, 17, 15), dtype=dtype, endpoint=True)
            else:
                data = (rng.standard_normal((20, 17, 15)) * 1000).astype(dtype)
            for sigma in (0.3, 1.0, 2.5, (1.0, 0.5, 2.0)):
                expected = gaussian_filter(np.asarray(data, dtype=np.float32), sigma=sigma)
                self.assertSameArray(expected, low.gaussian_float32(data, sigma), f'{np.dtype(dtype)} sigma={sigma}')
                self.assertSameArray(expected, low.gaussian_float32(np.asfortranarray(data), sigma), f'F-order {np.dtype(dtype)}')

    def test_content_mask_matches_legacy(self):
        # Same composition as the new rigidAlignment._content_mask_from_threshold.
        for dtype, thr in ((np.uint8, 100.0), (np.uint16, 25000.0), (np.float32, 100.0)):
            data = _scan(dtype=dtype if dtype != np.float32 else np.uint8, seed=5).astype(dtype)
            expected = legacy.legacy_content_mask_from_threshold(data, thr, sigma=1.0)
            smoothed = low.gaussian_float32(data, 1.0)
            got = legacy.legacy_largest_voxel_connected_component(smoothed > float(thr))
            self.assertSameArray(expected, got, str(np.dtype(dtype)))

    def test_normalize_input_matches_legacy_dino_normalize(self):
        rng = np.random.default_rng(6)
        cases = [
            _scan(dtype=np.uint8, seed=6),
            _scan(dtype=np.uint16, seed=6),
            np.full((4, 5, 6), 7, np.uint8),                       # constant volume branch
            rng.integers(0, 65535, (9, 8, 7), dtype=np.uint16, endpoint=True),
            (rng.standard_normal((9, 8, 7)) * 50).astype(np.float32),
            rng.integers(-32768, 32767, (9, 8, 7), dtype=np.int16, endpoint=True),  # fallback
            rng.standard_normal((9, 8, 7)),                         # float64 fallback
            np.asfortranarray(_scan(dtype=np.uint8, seed=7)),
        ]
        for vol in cases:
            expected = legacy.legacy_dino_normalize_volume(vol)
            got = legacy.legacy_min_max_normalize(low.normalize_input_without_float_copy(vol), new_min=-1, new_max=1)
            self.assertSameArray(expected, got, f'{vol.dtype} {vol.shape}')
            self.assertEqual(expected.flags.c_contiguous, got.flags.c_contiguous)

    def test_adjusted_background_from_min_max_matches_full_copy(self):
        def adjusted(background_value, image_data, new_min, new_max):  # registrationTools formula
            return (background_value - np.min(image_data)) / (np.max(image_data) - np.min(image_data)) * (new_max - new_min) + new_min
        for vol in (_scan(dtype=np.uint8), _scan(dtype=np.uint16), _scan(dtype=np.uint8).astype(np.float64) / 3.0):
            for bg in (0, 12, 37.5, float(np.max(vol))):
                expected = adjusted(bg, legacy.legacy_adjusted_background_input(vol), -1, 1)
                got = adjusted(bg, np.array([np.min(vol), np.max(vol)], dtype=np.float32), -1, 1)
                self.assertEqual(type(expected), type(got))
                self.assertTrue(expected == got or (np.isnan(expected) and np.isnan(got)), f'{vol.dtype} {bg}')

    def test_n4_in_place_clip_matches(self):
        rng = np.random.default_rng(8)
        corrected = (rng.standard_normal((10, 9, 8)) * 300 + 100).astype(np.float32)
        for dtype in (np.uint8, np.uint16, np.int16):
            expected = legacy.legacy_n4_finish(corrected.copy(), np.dtype(dtype))
            work = corrected.copy()
            np.clip(work, np.iinfo(dtype).min, np.iinfo(dtype).max, out=work)
            self.assertSameArray(expected, work.astype(dtype), str(np.dtype(dtype)))


class NiftiVoxelsTests(SameArrayMixin, unittest.TestCase):
    """nifti_voxels(img, dtype, round_values) == [np.round](img.get_fdata()).astype(dtype)."""

    STORED = [np.uint8, np.int8, np.uint16, np.int16, np.int32, np.uint32, np.int64,
              np.float32, np.float64, '>i2', '>f4', '>u2']
    TARGETS = [None, np.uint8, np.int8, np.int16, np.uint16, np.int32, np.float32, np.float64, bool]

    def _volume(self, rng, dtype):
        dt = np.dtype(dtype).newbyteorder('=')
        shape = (9, 7, 5)
        if dt.kind == 'f':
            a = (rng.standard_normal(shape) * 1000).astype(dt)
            a.flat[:6] = [np.nan, np.inf, -0.0, 2.5, -2.5, 6e4]
            return a
        info = np.iinfo(dt)
        a = rng.integers(info.min, info.max, size=shape, dtype=dt, endpoint=True)
        a.flat[:2] = [info.min, info.max]
        return a

    def test_matches_get_fdata_route(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as tmp:
            for stored, ext, scaled in itertools.product(self.STORED, ('.nii.gz', '.nii'), (False, True)):
                data = self._volume(rng, stored)
                img = nib.Nifti1Image(data, np.eye(4), dtype=np.dtype(stored).newbyteorder('='))
                img.set_data_dtype(stored)
                if scaled:
                    img.header.set_slope_inter(2.0, -3.0)
                path = os.path.join(tmp, f'v_{np.dtype(stored).str.strip("<>|=")}_{int(scaled)}{ext}')
                nib.save(img, path)
                with open(path, 'rb') as handle:
                    raw = handle.read()
                if raw[:2] == b'\x1f\x8b':
                    raw = gzip.decompress(raw)
                loaded = [nib.load(path), nib.load(path, mmap=False), nib.Nifti1Image.from_bytes(raw)]
                for im, target, rnd in itertools.product(loaded, self.TARGETS, (False, True)):
                    with np.errstate(all='ignore'):
                        src = im.get_fdata()
                        expected = (np.round(src) if rnd else src).astype(im.get_data_dtype() if target is None else target)
                        im.uncache()
                        got = nifti_voxels(im, target, round_values=rnd)
                    self.assertSameArray(expected, got, f'{os.path.basename(path)} {target} round={rnd}')
                    self.assertTrue(got.flags.writeable)
                    self.assertEqual(expected.flags.f_contiguous, got.flags.f_contiguous)

    def test_returns_private_copy_and_leaves_no_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'u8.nii.gz')
            nib.save(nib.Nifti1Image(np.arange(60, dtype=np.uint8).reshape(3, 4, 5), np.eye(4)), path)
            img = nib.load(path)
            first = nifti_voxels(img)
            first[:] = 0
            self.assertTrue(nifti_voxels(img).any())
            self.assertFalse(img.in_memory)

    def test_in_memory_image_uses_old_route(self):
        data = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
        img = nib.Nifti1Image(data, np.eye(4))
        got = nifti_voxels(img)
        self.assertSameArray(img.get_fdata().astype(np.uint8), got)
        got[:] = 9
        self.assertEqual(int(data[0, 0, 1]), 1, 'must not share memory with the image array')


def _import_real(module_name):
    try:
        import importlib
        return importlib.import_module(module_name), None
    except Exception as exc:  # pragma: no cover - depends on the environment
        return None, f'{module_name} not importable here ({type(exc).__name__}: {exc})'


class RealViewTests(SameArrayMixin, unittest.TestCase):
    """The real changed methods against the frozen copies (needs the full Aurora environment)."""

    def _need(self, module_name):
        module, reason = _import_real(module_name)
        if module is None:
            self.skipTest(reason)
        return module

    def _clean_mesh_cases(self):
        cases = []
        for dtype, thr in ((np.uint8, 100), (np.uint16, 25000)):
            for blur in (None, 0, 1.0):
                cases.append((f'{np.dtype(dtype)} islands blur={blur}', _scan(dtype=dtype, seed=11), thr, blur, False, 1.0))
            cases.append((f'{np.dtype(dtype)} volume mode 1%', _scan(dtype=dtype, seed=12), thr, 0, True, 1.0))
            cases.append((f'{np.dtype(dtype)} volume mode 0.001%', _scan(dtype=dtype, seed=12), thr, 0, True, 0.001))
            cases.append((f'{np.dtype(dtype)} volume mode blur', _scan(dtype=dtype, seed=13), thr, 1.0, True, 1.0))
            cases.append((f'{np.dtype(dtype)} single object', _scan(dtype=dtype, islands=False, speckle=False), thr, 0, False, 1.0))
            cases.append((f'{np.dtype(dtype)} no foreground', _scan(dtype=dtype, islands=False, speckle=False, blob=False), thr, 0, False, 1.0))
            cases.append((f'{np.dtype(dtype)} speckle only', _scan(dtype=dtype, islands=False, blob=False), thr, 0, False, 1.0))
        return cases

    def test_clean_mesh_matches_legacy(self):
        pre = self._need('pipeline.preprocessingTools')
        new_view = pre.CleanupMeshView()
        old_view = legacy.LegacyCleanupMesh()
        reasons = set()
        for name, scan, thr, blur, vol_mode, pct in self._clean_mesh_cases():
            old_in = scan.copy()
            new_in = scan.copy()
            old = _quiet(old_view.clean_mesh, old_in, thr, scan.dtype, blur,
                         use_island_volume_threshold=vol_mode, min_island_volume_percent=pct)
            new = _quiet(new_view.clean_mesh, new_in, thr, scan.dtype, blur,
                         use_island_volume_threshold=vol_mode, min_island_volume_percent=pct)
            self.assertSameArray(old[0], new[0], f'{name}: cleaned data')
            self.assertEqual(old[1], new[1], f'{name}: cleanup_performed')
            if old[2] is None:
                self.assertIsNone(new[2], f'{name}: removal mask')
            else:
                self.assertSameArray(old[2], new[2], f'{name}: removal mask')
            self.assertEqual(old[3], new[3], f'{name}: outcome')
            self.assertSameArray(old_in, new_in, f'{name}: in-place edit of the input')
            reasons.add((old[3]['reason'], bool(blur)))
        # The cases must reach every branch, or the comparison proves little.
        for expected in [('components_removed', False), ('components_removed', True),
                         ('no_disconnected_components', False), ('no_foreground', False),
                         ('all_components_kept', False)]:
            self.assertIn(expected, reasons)

    def test_homogenize_background_matches_legacy(self):
        pre = self._need('pipeline.preprocessingTools')
        new_view = pre.HomogenizeBackgroundView()
        old_view = legacy.LegacyHomogenizeBackground()
        rng = np.random.default_rng(14)
        signed = rng.integers(-3000, 3000, (12, 11, 10), dtype=np.int16)
        for data in (_scan(dtype=np.uint8), _scan(dtype=np.uint16), signed):
            top = int(np.iinfo(data.dtype).max)
            for target in (0, 21, 21.7, -5, top, top + 10):
                expected = _quiet(old_view._homogenize_background, data.copy(), target)
                got = _quiet(new_view._homogenize_background, data.copy(), target)
                self.assertSameArray(expected, got, f'{data.dtype} target={target}')

    def test_rotation_bbox_matches_legacy(self):
        views = self._need('pipeline.views')
        new_view = views.ApplyRotationView()
        old_view = legacy.LegacyRotation()
        rng = np.random.default_rng(15)
        for data, thr in ((_scan(), 100), (_scan(dtype=np.uint16), 25000), (np.zeros((10, 10, 10), np.uint8), 5),
                          (rng.integers(0, 255, (90, 100, 110), dtype=np.uint8), 254)):
            old = old_view.get_tight_bbox(data, thr)
            new = new_view.get_tight_bbox(data, thr)
            for a, b in zip(old, new):
                if isinstance(a, tuple):
                    self.assertEqual(a, b)
                else:
                    self.assertSameArray(a, b, f'bbox {data.dtype} thr={thr}')

    def test_content_mask_matches_legacy(self):
        rig = self._need('pipeline.rigidAlignment')
        for data, thr in ((_scan(), 100.0), (_scan(dtype=np.uint16), 25000.0), (_scan().astype(np.float64), 100.0)):
            self.assertSameArray(
                legacy.legacy_content_mask_from_threshold(data, thr, sigma=1.0),
                rig._content_mask_from_threshold(data, thr, sigma=1.0),
                str(data.dtype),
            )

    def test_dino_normalize_and_relabel_order_match_legacy(self):
        dino = self._need('pipeline.dinoRegRigid')
        rng = np.random.default_rng(16)
        vols = [_scan(), _scan(dtype=np.uint16), rng.integers(0, 255, (7, 9, 11), dtype=np.uint8)]
        for vol in vols:
            self.assertSameArray(legacy.legacy_dino_normalize_volume(vol), dino._normalize_volume(vol), str(vol.dtype))
        # All 24 proper 90-degree relabels: normalize(relabel(v)) == relabel(normalize(v)).
        rotations = []
        for perm in itertools.permutations(range(3)):
            for signs in itertools.product((1, -1), repeat=3):
                R = np.zeros((3, 3))
                for row, (col, sign) in enumerate(zip(perm, signs)):
                    R[row, col] = sign
                if np.linalg.det(R) > 0:
                    rotations.append(R)
        self.assertEqual(len(rotations), 24)
        for vol in vols:
            for R in rotations:
                old = dino._apply_octahedral_relabel(legacy.legacy_dino_normalize_volume(vol), R)
                new = dino._normalize_volume(dino._apply_octahedral_relabel(vol, R))
                self.assertSameArray(old, new, f'{vol.dtype} R={R.tolist()}')
                self.assertTrue(new.flags.c_contiguous)


if __name__ == '__main__':
    unittest.main()
