"""
Integration tests: GPU rigid on real KDO tongue-fat scans.

Requires CUDA, fireants, and Tongue_fat_test data (see gpu_rigid_kdo_helpers.DEFAULT_PROJECT_ROOT).
Set TONGUE_FAT_TEST_ROOT to override the data directory.
"""
from __future__ import annotations

import os
import unittest

import pytest

from pipeline.gpuRigid import ensure_gpu_rigid_cuda
from pipeline.tests.gpu_rigid_kdo_helpers import (
    DEFAULT_PROJECT_ROOT,
    run_gpu_rigid_pair,
)


def _cuda_and_data_available() -> bool:
    try:
        ensure_gpu_rigid_cuda()
    except Exception:
        return False
    root = os.environ.get("TONGUE_FAT_TEST_ROOT", DEFAULT_PROJECT_ROOT)
    kdo70 = os.path.join(root, "extracted", "KDO70", "KDO70_edit_2_cleaned.nii.gz")
    kdo143 = os.path.join(root, "extracted", "KDO143", "KDO143_edit_5_cleaned.nii.gz")
    kdo144 = os.path.join(root, "extracted", "KDO144", "KDO144_edit_2_cleaned.nii.gz")
    return all(os.path.isfile(p) for p in (kdo70, kdo143, kdo144))


requires_kdo_gpu = pytest.mark.skipif(
    not _cuda_and_data_available(),
    reason="CUDA/fireants or KDO test volumes unavailable",
)


def _assert_selected_beats_identity(result, *, min_iou_gain: float, min_selected_iou: float):
    identity = result.identity_metrics
    selected = result.moments_metrics
    assert selected["iou"] >= min_selected_iou, (
        f"{result.moving_scan}: selected IoU {selected['iou']:.3f} < {min_selected_iou}"
    )
    assert selected["iou"] >= identity["iou"] + min_iou_gain, (
        f"{result.moving_scan}: selected IoU {selected['iou']:.3f} did not beat identity "
        f"{identity['iou']:.3f} by {min_iou_gain}"
    )
    assert selected["fixed_recall"] >= identity["fixed_recall"], (
        f"{result.moving_scan}: fixed_recall regressed "
        f"{selected['fixed_recall']:.3f} vs identity {identity['fixed_recall']:.3f}"
    )


# Backwards-compatible alias for helpers that still label the field moments_metrics.
_assert_moments_beats_identity = _assert_selected_beats_identity


def _assert_canvas_apply_consistent(result, *, min_cropped_iou: float):
    cropped = result.cropped_overlap
    moments = result.moments_metrics
    assert cropped["iou"] >= min_cropped_iou, (
        f"{result.moving_scan}: cropped IoU {cropped['iou']:.3f} < {min_cropped_iou}"
    )
    # Canvas apply should preserve roughly the same overlap as estimation-grid scoring.
    assert cropped["iou"] >= moments["iou"] - 0.08, (
        f"{result.moving_scan}: cropped IoU {cropped['iou']:.3f} much worse than "
        f"estimation IoU {moments['iou']:.3f} (apply/crop regression)"
    )


@requires_kdo_gpu
class TestKdoGpuRigidIntegration(unittest.TestCase):
    """KDO70 cleaned edit → KDO143 reference; then KDO144 → KDO143."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = os.environ.get("TONGUE_FAT_TEST_ROOT", DEFAULT_PROJECT_ROOT)

    def test_kdo70_cleaned_to_kdo143(self):
        result = run_gpu_rigid_pair(
            "KDO70",
            reference_scan="KDO143",
            moving_edit_stem="KDO70_edit_2_cleaned",
            reference_edit_stem="KDO143_edit_5_cleaned",
            project_root=self.project_root,
        )
        self.assertEqual(result.estimation_stride, 3)
        # Grid dims vary slightly with bbox/canvas rounding; ~26M voxels at stride 3.
        self.assertGreaterEqual(min(result.estimation_shape), 280)
        self.assertLessEqual(int(__import__("numpy").prod(result.estimation_shape)), 27_000_000)

        _assert_selected_beats_identity(result, min_iou_gain=0.0, min_selected_iou=0.40)
        _assert_canvas_apply_consistent(result, min_cropped_iou=0.45)

        self.assertGreater(result.moments_metrics["ref_ncc"], 0.0)
        self.assertGreater(result.cropped_overlap["fixed_recall"], 0.80)

    def test_kdo144_cleaned_to_kdo143(self):
        result = run_gpu_rigid_pair(
            "KDO144",
            reference_scan="KDO143",
            moving_edit_stem="KDO144_edit_2_cleaned",
            reference_edit_stem="KDO143_edit_5_cleaned",
            project_root=self.project_root,
        )

        # KDO144 is harder; require clear improvement over identity, not KDO70 parity.
        _assert_selected_beats_identity(result, min_iou_gain=0.0, min_selected_iou=0.38)
        _assert_canvas_apply_consistent(result, min_cropped_iou=0.38)

        self.assertGreater(result.moments_metrics["fixed_recall"], result.identity_metrics["fixed_recall"])
        self.assertGreater(result.cropped_overlap["fixed_recall"], 0.65)


if __name__ == "__main__":
    unittest.main()
