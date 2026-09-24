"""Tests for ANTs/GPU-rigid → scipy alignment_calculated_parameters unification."""
import os
import unittest

import numpy as np

from pipeline.rigidAlignment import (
    _write_ants_rigid_mat,
    apply_rigid_rt_on_union_canvas,
    crop_registration_canvas_to_reference,
    forward_scan_voxel_to_reference_voxel_via_ants_rigid,
    has_legacy_ants_affine_metadata,
    kabsch_rotation_matrix,
    paste_volume_on_canvas_clipped,
    project_ants_mat_to_scipy_rigid_params,
    uses_scipy_centroid_embed_rigid,
)


class TestScipyRigidUnification(unittest.TestCase):
    def test_uses_scipy_centroid_embed_rigid_includes_ants_and_gpu(self):
        self.assertTrue(uses_scipy_centroid_embed_rigid('ants'))
        self.assertTrue(uses_scipy_centroid_embed_rigid('gpu-rigid'))
        self.assertTrue(uses_scipy_centroid_embed_rigid('manual-guidepoints'))
        self.assertFalse(uses_scipy_centroid_embed_rigid('alpaca'))

    def test_has_legacy_ants_affine_metadata(self):
        self.assertTrue(has_legacy_ants_affine_metadata({'alignment_ants_affine': 'subj_ants_rigid_affine.mat'}))
        self.assertFalse(has_legacy_ants_affine_metadata({'alignment_rotation_matrix': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}))

    def test_kabsch_matches_guidepoints_convention(self):
        src_c = np.array([10.0, 20.0, 30.0])
        tgt_c = np.array([40.0, 50.0, 60.0])
        rng = np.random.default_rng(1)
        src = rng.normal(size=(64, 3)) + src_c
        angle = np.deg2rad(17.0)
        R_true = np.array([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        tgt = (src - src_c) @ R_true + tgt_c
        R_fit, _, _ = kabsch_rotation_matrix(src, tgt, source_centroid=src_c, target_centroid=tgt_c)
        np.testing.assert_allclose(R_fit, R_true, atol=1e-6)
        pred = (src - src_c) @ R_fit + tgt_c
        np.testing.assert_allclose(pred, tgt, atol=1e-5)

    def test_analytic_ants_to_scipy_roundtrip(self):
        spacing = (1.0, 1.0, 1.0)
        origin = (0.0, 0.0, 0.0)
        direction = np.eye(3)
        mov_paste = np.array([0.0, 0.0, 0.0])
        ref_paste = np.array([0.0, 0.0, 0.0])
        rotation_center = np.array([32.0, 32.0, 32.0])

        angle = np.deg2rad(25.0)
        R_true = np.array([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        t = np.array([2.5, -1.0, 0.5])
        mat_path = _write_ants_rigid_mat(R_true, t)

        scan_centroid = np.array([32.0, 32.0, 32.0])
        ref_centroid = np.array([32.0, 32.0, 32.0])
        rotation, scan_c, ref_c, residual_t = project_ants_mat_to_scipy_rigid_params(
            mat_path,
            scan_centroid,
            ref_centroid,
            spacing,
            scan_name='test',
        )

        for sample in (
            np.array([25.0, 30.0, 35.0]),
            np.array([32.0, 32.0, 32.0]),
            np.array([40.0, 38.0, 28.0]),
        ):
            mapped = forward_scan_voxel_to_reference_voxel_via_ants_rigid(
                sample,
                mat_path=mat_path,
                rotation_center_voxel=rotation_center,
                fixed_spacing=spacing,
                fixed_origin=origin,
                fixed_direction=direction,
                mov_paste=mov_paste,
                ref_paste=ref_paste,
            )
            scipy_pred = (sample - scan_c) @ rotation + scan_c + residual_t
            np.testing.assert_allclose(mapped, scipy_pred, atol=1e-5)

        try:
            os.remove(mat_path)
        except OSError:
            pass

    def test_gpu_canvas_apply_matches_estimation_mat(self):
        spacing = (0.02, 0.02, 0.02)
        origin = (0.0, 0.0, 0.0)
        direction = np.eye(3)
        ref_paste = np.array([20, 30, 40], dtype=np.int64)
        mov_paste = np.array([10, 25, 35], dtype=np.int64)
        canvas_shape = (96, 96, 96)
        reference_shape = (48, 48, 48)

        rng = np.random.default_rng(0)
        ref_vol = np.zeros(reference_shape, dtype=np.float32)
        mov_vol = np.zeros(reference_shape, dtype=np.float32)
        ref_vol[8:40, 8:40, 8:40] = rng.random((32, 32, 32))
        mov_vol[6:38, 10:42, 12:44] = rng.random((32, 32, 32))
        reference_on_canvas = paste_volume_on_canvas_clipped(ref_vol, ref_paste, canvas_shape, 0)
        moving_on_canvas = paste_volume_on_canvas_clipped(mov_vol, mov_paste, canvas_shape, 0)

        angle = np.deg2rad(22.0)
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        t = np.array([0.5, -0.3, 0.2])

        import ants
        from pipeline.rigidAlignment import ants_origin_for_canvas_paste

        canvas_origin = ants_origin_for_canvas_paste(origin, direction, ref_paste, spacing)
        fixed_est = ants.from_numpy(
            reference_on_canvas.astype(np.float32),
            spacing=spacing,
            origin=tuple(float(x) for x in canvas_origin),
            direction=direction,
        )
        moving_est = ants.from_numpy(
            moving_on_canvas.astype(np.float32),
            spacing=spacing,
            origin=tuple(float(x) for x in canvas_origin),
            direction=direction,
        )
        mat_path = _write_ants_rigid_mat(R, t)
        try:
            est_warped = ants.apply_transforms(
                fixed=fixed_est,
                moving=moving_est,
                transformlist=[mat_path],
                interpolator="linear",
                defaultvalue=0,
                verbose=False,
            )
            canvas_warped = apply_rigid_rt_on_union_canvas(
                moving_on_canvas,
                R,
                t,
                spacing=spacing,
                reference_origin=origin,
                direction=direction,
                ref_paste=ref_paste,
                reference_canvas=reference_on_canvas,
                defaultvalue=0,
            )
            cropped = crop_registration_canvas_to_reference(
                canvas_warped, ref_paste, reference_shape
            )
            est_cropped = crop_registration_canvas_to_reference(
                np.asarray(est_warped.numpy()), ref_paste, reference_shape
            )
            np.testing.assert_allclose(cropped, est_cropped, atol=1e-4)
        finally:
            try:
                os.remove(mat_path)
            except OSError:
                pass


if __name__ == '__main__':
    unittest.main()
