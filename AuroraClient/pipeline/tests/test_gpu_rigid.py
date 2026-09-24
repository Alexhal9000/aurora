import unittest

import numpy as np

from pipeline.gpuRigid import (
    normalize_gpu_rigid_options,
    project_homogeneous_affine_to_rigid,
    choose_gpu_rigid_scales,
    fireants_level_shape,
    format_gpu_rigid_resolution_chain,
    adapt_gpu_rigid_opts_to_grid,
    _match_iterations_to_scales,
    _metrics_better_than,
    _metrics_rank,
    DEFAULT_GPU_RIGID_SCALES,
    DEFAULT_GPU_RIGID_TRANSL_MODE,
)
from pipeline.rigidAlignment import is_ants_style_physical_rigid_method


class TestGpuRigidOptions(unittest.TestCase):
    def test_normalize_defaults_four_scales(self):
        opts = normalize_gpu_rigid_options({})
        self.assertEqual(opts["initializer"], "moments")
        self.assertEqual(opts["transl_mode"], DEFAULT_GPU_RIGID_TRANSL_MODE)
        self.assertEqual(opts["loss_type"], "auto")
        self.assertTrue(opts["refine"])
        self.assertTrue(opts["adapt_scales"])
        self.assertTrue(opts["use_registration_mask"])
        self.assertEqual(opts["scales"], DEFAULT_GPU_RIGID_SCALES)
        self.assertEqual(len(opts["scales"]), 4)
        self.assertEqual(len(opts["iterations"]), 4)

    def test_metrics_rank_prefers_recall_then_iou(self):
        better = {"fixed_recall": 0.8, "iou": 0.4, "ncc": -0.1}
        worse = {"fixed_recall": 0.7, "iou": 0.9, "ncc": 0.5}
        self.assertTrue(_metrics_better_than(better, worse))
        self.assertGreater(_metrics_rank(better), _metrics_rank(worse))

    def test_normalize_clips_invalid(self):
        opts = normalize_gpu_rigid_options({
            "gpu_rigid_options": {
                "initializer": "not-valid",
                "scales": [1, 2],
            }
        })
        self.assertEqual(opts["initializer"], "moments")
        self.assertEqual(opts["scales"], DEFAULT_GPU_RIGID_SCALES)

    def test_choose_scales_at_most_four_levels(self):
        est = (285, 269, 281)
        scales = choose_gpu_rigid_scales(est)
        self.assertLessEqual(len(scales), 4)
        levels = [fireants_level_shape(est, s) for s in scales]
        self.assertEqual(len(levels), len(set(levels)))

    def test_kdo70_resolution_chain(self):
        est = (308, 293, 292)
        scales = choose_gpu_rigid_scales(est)
        msg = format_gpu_rigid_resolution_chain(
            reference_shape=(613, 581, 597),
            canvas_shape=(924, 879, 876),
            estimation_stride=3,
            estimation_shape=est,
            scales=scales,
        )
        self.assertIn("stride=3", msg)
        self.assertIn("FireANTs", msg)

    def test_adapt_opts_rewrites_scales_when_enabled(self):
        opts = normalize_gpu_rigid_options({})
        adapted = adapt_gpu_rigid_opts_to_grid(
            opts,
            (308, 293, 292),
            reference_shape=(613, 581, 597),
            canvas_shape=(924, 879, 876),
            estimation_stride=3,
        )
        self.assertEqual(len(adapted["scales"]), len(adapted["iterations"]))

    def test_adapt_opts_respects_lock(self):
        opts = normalize_gpu_rigid_options({"gpu_rigid_options": {"adapt_scales": False}})
        adapted = adapt_gpu_rigid_opts_to_grid(
            opts,
            (308, 293, 292),
            reference_shape=(613, 581, 597),
            canvas_shape=(924, 879, 876),
            estimation_stride=3,
        )
        self.assertEqual(adapted["scales"], opts["scales"])

    def test_split_pyramid_via_normalize_lengths(self):
        opts = normalize_gpu_rigid_options({})
        self.assertEqual(len(opts["scales"]), len(opts["iterations"]))

    def test_match_iterations_to_scales(self):
        iters = _match_iterations_to_scales((8, 4, 2, 1), (80, 100, 150, 180))
        self.assertEqual(len(iters), 4)


class TestMethodHelpers(unittest.TestCase):
    def test_is_ants_style_physical_rigid_method(self):
        self.assertTrue(is_ants_style_physical_rigid_method("ants"))
        self.assertTrue(is_ants_style_physical_rigid_method("gpu-rigid"))
        self.assertFalse(is_ants_style_physical_rigid_method("alpaca"))

    def test_project_affine_to_rigid_strips_shear(self):
        A = np.diag([1.1, 0.9, 1.05])
        mat = np.eye(4)
        mat[:3, :3] = A
        mat[:3, 3] = [2.0, -1.0, 0.5]
        rigid, R, t = project_homogeneous_affine_to_rigid(mat)
        np.testing.assert_allclose(t, mat[:3, 3])
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=5)
        np.testing.assert_allclose(rigid[:3, :3], R)


if __name__ == "__main__":
    unittest.main()
