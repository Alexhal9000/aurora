"""Shared helpers for KDO GPU rigid integration tests (mirrors production path)."""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Optional

import ants
import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter

from pipeline.gpuRigid import normalize_gpu_rigid_options, run_gpu_rigid_registration
from pipeline.registrationTools import RegistrationTools
from pipeline.rigidAlignment import (
    _ants_binary_mask_image,
    _cap_mask_bbox_to_extent,
    _centroid_of_mask,
    _content_mask_from_threshold,
    _mask_bbox,
    _score_rigid_metrics,
    ants_origin_for_canvas_paste,
    apply_rigid_rt_on_union_canvas,
    build_union_registration_canvas,
    crop_registration_canvas_to_reference,
    paste_volume_on_canvas_clipped,
)

DEFAULT_PROJECT_ROOT = os.environ.get(
    "TONGUE_FAT_TEST_ROOT",
    "/home/alejandro/Documents/Tongue_fat/Tongue_fat_test",
)
DEFAULT_REFERENCE = "KDO143"


@dataclass
class GpuRigidPairResult:
    moving_scan: str
    reference_scan: str
    moving_path: str
    reference_path: str
    canvas_shape: tuple
    estimation_stride: int
    estimation_shape: tuple
    identity_metrics: dict
    moments_metrics: dict
    rotation_deg: float
    cropped_overlap: dict
    R: np.ndarray
    t: np.ndarray


def _project_root() -> str:
    return DEFAULT_PROJECT_ROOT


def _extracted_dir(project_root: Optional[str] = None) -> str:
    root = project_root or _project_root()
    return os.path.join(root, "extracted")


def _load_scan_json(scan_name: str, project_root: Optional[str] = None) -> dict:
    path = os.path.join(_extracted_dir(project_root), scan_name, f"{scan_name}.json")
    with open(path, "r") as handle:
        return json.load(handle)


def resolve_edit_nifti(
    scan_name: str,
    edit_stem: Optional[str] = None,
    project_root: Optional[str] = None,
) -> str:
    """Return path to a specific edit stem or the latest edit NIfTI."""
    scan_dir = os.path.join(_extracted_dir(project_root), scan_name)
    if edit_stem:
        path = os.path.join(scan_dir, f"{edit_stem}.nii.gz")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        return path
    edit_number = 0
    while glob.glob(os.path.join(scan_dir, f"{scan_name}_edit_{edit_number}_*.nii.gz")):
        edit_number += 1
    edit_number -= 1
    if edit_number >= 0:
        return glob.glob(os.path.join(scan_dir, f"{scan_name}_edit_{edit_number}_*.nii.gz"))[0]
    path = os.path.join(scan_dir, f"{scan_name}.nii.gz")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


def _rotation_degrees_from_matrix(R: np.ndarray) -> float:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def score_cropped_volume_overlap(
    reference_volume,
    transformed_volume,
    reference_threshold: float,
    moving_threshold: float,
) -> dict:
    """Foreground overlap on the reference grid after canvas apply + crop."""
    ref = np.asarray(reference_volume, dtype=np.float32)
    warped = np.asarray(transformed_volume, dtype=np.float32)
    ref_fg = ref > float(reference_threshold)
    warped_fg = warped > float(moving_threshold)
    mutual = ref_fg & warped_fg
    if mutual.sum() < 200:
        return {"iou": 0.0, "fixed_recall": 0.0, "ncc": -1.0, "ref_ncc": -1.0}
    ncc = float(np.corrcoef(ref[mutual], warped[mutual])[0, 1])
    if ncc != ncc:
        ncc = -1.0
    ref_ncc = float(np.corrcoef(ref[ref_fg], warped[ref_fg])[0, 1])
    if ref_ncc != ref_ncc:
        ref_ncc = -1.0
    union = ref_fg | warped_fg
    return {
        "iou": float(mutual.sum() / max(union.sum(), 1)),
        "fixed_recall": float(mutual.sum() / max(ref_fg.sum(), 1)),
        "ncc": ncc,
        "ref_ncc": ref_ncc,
    }


def run_gpu_rigid_pair(
    moving_scan: str,
    *,
    reference_scan: str = DEFAULT_REFERENCE,
    moving_edit_stem: Optional[str] = None,
    reference_edit_stem: Optional[str] = None,
    project_root: Optional[str] = None,
    gpu_rigid_options: Optional[dict] = None,
    background_value: int = 0,
) -> GpuRigidPairResult:
    """
    Run the same union-canvas GPU rigid path used in rigidAlignment (moments only).
    """
    project_root = project_root or _project_root()
    ref_meta = _load_scan_json(reference_scan, project_root)
    mov_meta = _load_scan_json(moving_scan, project_root)

    reference_path = resolve_edit_nifti(reference_scan, reference_edit_stem, project_root)
    moving_path = resolve_edit_nifti(moving_scan, moving_edit_stem, project_root)

    reference_threshold = float(ref_meta["threshold"])
    moving_threshold = float(mov_meta["threshold"])

    reference_img = nib.load(reference_path)
    moving_img = nib.load(moving_path)
    reference_data = gaussian_filter(reference_img.get_fdata().astype(reference_img.get_data_dtype()), sigma=1.0)
    scan_data = gaussian_filter(moving_img.get_fdata().astype(moving_img.get_data_dtype()), sigma=1.0)
    reference_shape = reference_data.shape

    reference_content_mask = _content_mask_from_threshold(reference_data, reference_threshold, sigma=1.0)
    scan_content_mask = _content_mask_from_threshold(scan_data, moving_threshold, sigma=1.0)
    reference_centroid = _centroid_of_mask(reference_content_mask)
    scan_centroid = _centroid_of_mask(scan_content_mask)
    ref_content_low, ref_content_high = _mask_bbox(reference_content_mask)
    mov_content_low, mov_content_high = _mask_bbox(scan_content_mask)

    canvas_margin = int(max(32, round(max(reference_shape) * 0.06)))
    ref_extent = np.asarray(ref_content_high - ref_content_low, dtype=np.float64)
    mov_extent = np.asarray(mov_content_high - mov_content_low, dtype=np.float64)
    rotation_margin = int(max(64, round(max(np.linalg.norm(ref_extent), np.linalg.norm(mov_extent)) * 0.15)))
    mov_cap = ref_extent * 1.35 + 2.0 * rotation_margin
    mov_content_low, mov_content_high = _cap_mask_bbox_to_extent(
        mov_content_low,
        mov_content_high,
        scan_centroid,
        mov_cap,
        scan_data.shape,
    )

    opts = normalize_gpu_rigid_options({"gpu_rigid_options": gpu_rigid_options or {}})
    max_est_voxels = int(opts.get("max_estimation_voxels", 40_000_000))
    union_layout = build_union_registration_canvas(
        reference_centroid,
        scan_centroid,
        ref_content_low,
        ref_content_high,
        mov_content_low,
        mov_content_high,
        canvas_margin,
        max_estimation_voxels=max_est_voxels,
        rotation_margin_voxels=rotation_margin,
    )
    canvas_shape = tuple(int(x) for x in union_layout["canvas_shape"])
    ref_paste = union_layout["ref_paste"]
    mov_paste = union_layout["mov_paste"]
    estimation_stride = int(union_layout["estimation_stride"])
    estimation_shape = tuple(int(x) for x in union_layout["estimation_shape"])

    reference_on_canvas = paste_volume_on_canvas_clipped(
        reference_data, ref_paste, canvas_shape, background_value
    )
    moving_on_canvas = paste_volume_on_canvas_clipped(
        scan_data, mov_paste, canvas_shape, background_value
    )

    use_reg_mask = bool(opts.get("use_registration_mask", True))
    reference_mask_on_canvas = None
    moving_mask_on_canvas = None
    fixed_image_mask = None
    moving_image_mask = None
    if use_reg_mask:
        reference_mask_on_canvas = paste_volume_on_canvas_clipped(
            reference_content_mask.astype(np.uint8), ref_paste, canvas_shape, 0
        ).astype(bool)
        moving_mask_on_canvas = paste_volume_on_canvas_clipped(
            scan_content_mask.astype(np.uint8), mov_paste, canvas_shape, 0
        ).astype(bool)

    ref_base = os.path.join(_extracted_dir(project_root), reference_scan, f"{reference_scan}.nii.gz")
    fixed_image_ants = ants.image_read(ref_base)
    fixed_image_spacing = fixed_image_ants.spacing
    fixed_image_origin = fixed_image_ants.origin
    fixed_image_direction = fixed_image_ants.direction
    fixed_image_ants = None

    half_spacing = tuple(float(s) * estimation_stride for s in fixed_image_spacing)
    ants_canvas_origin = ants_origin_for_canvas_paste(
        fixed_image_origin, fixed_image_direction, ref_paste, fixed_image_spacing
    )
    est_slices = tuple(slice(None, None, estimation_stride) for _ in range(3))
    reg_tools = RegistrationTools()

    fixed_image_data = ants.from_numpy(
        reg_tools.min_max_normalize(
            reference_on_canvas[est_slices].astype(np.float32), new_min=-1, new_max=1
        ),
        spacing=half_spacing,
        origin=tuple(float(x) for x in ants_canvas_origin),
        direction=fixed_image_direction,
    )
    moving_image_data = ants.from_numpy(
        reg_tools.min_max_normalize(
            moving_on_canvas[est_slices].astype(np.float32), new_min=-1, new_max=1
        ),
        spacing=half_spacing,
        origin=tuple(float(x) for x in ants_canvas_origin),
        direction=fixed_image_direction,
    )
    if use_reg_mask and reference_mask_on_canvas is not None:
        fixed_image_mask = _ants_binary_mask_image(
            reference_mask_on_canvas[est_slices],
            half_spacing,
            ants_canvas_origin,
            fixed_image_direction,
            dilate_iters=3,
        )
        moving_image_mask = _ants_binary_mask_image(
            moving_mask_on_canvas[est_slices],
            half_spacing,
            ants_canvas_origin,
            fixed_image_direction,
            dilate_iters=3,
        )

    opts_run = dict(opts)
    opts_run["debug_checkpoints"] = False
    gpu_result = run_gpu_rigid_registration(
        fixed_image_data,
        moving_image_data,
        opts_run,
        scan_name=moving_scan,
        fixed_mask=fixed_image_mask,
        moving_mask=moving_image_mask,
        reference_shape=reference_shape,
        canvas_shape=canvas_shape,
        estimation_stride=estimation_stride,
        debug_dir=None,
        reference_name=reference_scan,
    )

    identity_metrics = _score_rigid_metrics(
        fixed_image_data, moving_image_data, np.eye(3), np.zeros(3)
    )
    R = np.asarray(gpu_result["R"], dtype=np.float64)
    t = np.asarray(gpu_result["t"], dtype=np.float64)
    moments_metrics = dict(gpu_result)
    moments_metrics = {
        "ncc": float(gpu_result["ncc"]),
        "iou": float(gpu_result["iou"]),
        "fixed_recall": float(
            _score_rigid_metrics(fixed_image_data, moving_image_data, R, t)["fixed_recall"]
        ),
        "ref_ncc": float(
            _score_rigid_metrics(fixed_image_data, moving_image_data, R, t)["ref_ncc"]
        ),
    }

    warped_canvas = apply_rigid_rt_on_union_canvas(
        moving_on_canvas,
        R,
        t,
        spacing=fixed_image_spacing,
        reference_origin=fixed_image_origin,
        direction=fixed_image_direction,
        ref_paste=ref_paste,
        reference_canvas=reference_on_canvas,
        defaultvalue=background_value,
    )
    transformed = crop_registration_canvas_to_reference(warped_canvas, ref_paste, reference_shape)
    cropped_overlap = score_cropped_volume_overlap(
        reference_data, transformed, reference_threshold, moving_threshold
    )

    mat_path = gpu_result.get("mat_path")
    if mat_path and os.path.isfile(mat_path):
        try:
            os.remove(mat_path)
        except OSError:
            pass

    return GpuRigidPairResult(
        moving_scan=moving_scan,
        reference_scan=reference_scan,
        moving_path=moving_path,
        reference_path=reference_path,
        canvas_shape=canvas_shape,
        estimation_stride=estimation_stride,
        estimation_shape=estimation_shape,
        identity_metrics=identity_metrics,
        moments_metrics=moments_metrics,
        rotation_deg=_rotation_degrees_from_matrix(R),
        cropped_overlap=cropped_overlap,
        R=R,
        t=t,
    )
