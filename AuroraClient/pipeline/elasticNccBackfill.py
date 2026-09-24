"""
TEMPORARY — delete this whole file when legacy elastic_ncc backfill is no longer needed.

Also remove:
  - url route `backfill-elastic-ncc/` in AuroraClient/urls.py
  - Aurora/AuroraClient/ui_src/components/aurora/elasticNccBackfillTemp.js
  - its import + call/dialog wires in filePathExplorer.js
"""

from __future__ import annotations

import glob
import json
import os
import re

import ants
import nibabel as nib
import numpy as np
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .atlas_paths import resolve_atlas_dir
from .coordinateFrames import is_preserved_mesh_metadata
from .meshBasedTools import build_temporary_mesh_reference_volume
from .meshGridTools import strip_ephemeral_scan_metadata
from .registrationHeatmapTools import _latest_voxel_elastic_edit


def _union_mask_ncc(fixed_volume, moving_volume, reference_threshold, moving_threshold):
    """Same formula as views.compute_union_mask_ncc (kept local so this file is deletable)."""
    union_mask = np.logical_or(
        fixed_volume > reference_threshold,
        moving_volume > moving_threshold,
    )
    if int(np.count_nonzero(union_mask)) <= 1:
        return 0.0
    fixed_vals = fixed_volume[union_mask].astype(np.float64, copy=False)
    moving_vals = moving_volume[union_mask].astype(np.float64, copy=False)
    fixed_centered = fixed_vals - fixed_vals.mean()
    moving_centered = moving_vals - moving_vals.mean()
    denom = np.linalg.norm(fixed_centered) * np.linalg.norm(moving_centered)
    if denom <= 0:
        return 0.0
    return float(np.dot(fixed_centered, moving_centered) / denom)


def _project_dir_from_scan_json_path(json_path):
    scan_dir = os.path.dirname(json_path)
    parent = os.path.dirname(scan_dir)
    if os.path.basename(parent) == "extracted":
        return os.path.dirname(parent)
    return parent


def _resolve_fixed_nifti_path(ref_dir, elastic_to):
    """Match ElasticRegistrationView fixed-volume resolution (latest edit, else base)."""
    edit_files = glob.glob(os.path.join(ref_dir, f"{elastic_to}_edit_*.nii.gz"))
    if not edit_files:
        base_path = os.path.join(ref_dir, f"{elastic_to}.nii.gz")
        return base_path if os.path.isfile(base_path) else None

    edit_numbers = []
    for path in edit_files:
        match = re.search(r"_edit_(\d+)_", path)
        if match:
            edit_numbers.append(int(match.group(1)))
    if not edit_numbers:
        base_path = os.path.join(ref_dir, f"{elastic_to}.nii.gz")
        return base_path if os.path.isfile(base_path) else None

    latest_edit = max(edit_numbers)
    matches = glob.glob(os.path.join(ref_dir, f"{elastic_to}_edit_{latest_edit}_*.nii.gz"))
    return matches[0] if matches else None


def _compute_ncc_for_subject(json_path, json_data):
    """Full-res union-mask NCC equivalent to ElasticRegistrationView."""
    elastic_to = json_data.get("elastic_to")
    moving_threshold = json_data.get("threshold")
    if not elastic_to or moving_threshold is None:
        return None
    if json_data.get("is_mesh") is True and json_data.get("voxelized") is False:
        return None

    subject_dir = os.path.dirname(json_path)
    subject_name = json_data.get("name") or os.path.basename(json_path)[:-5]
    project_dir = _project_dir_from_scan_json_path(json_path)

    moving_path = _latest_voxel_elastic_edit(subject_dir, subject_name)
    if moving_path is None or not os.path.isfile(moving_path):
        return None

    if elastic_to == "atlas":
        ref_dir = resolve_atlas_dir(project_dir)
        ref_json_path = os.path.join(ref_dir, "atlas.json")
    else:
        ref_dir = os.path.join(project_dir, "extracted", elastic_to)
        ref_json_path = os.path.join(ref_dir, f"{elastic_to}.json")
    if not os.path.isfile(ref_json_path):
        return None

    with open(ref_json_path, "r") as jf:
        ref_metadata = json.load(jf)

    moving_on_disk = np.asanyarray(nib.load(moving_path).get_fdata())
    if is_preserved_mesh_metadata(ref_metadata):
        # Undo save-time xz swap from _moving_volume_for_mesh_reference_save.
        moving_volume = np.swapaxes(moving_on_disk, 0, 2)
        (
            fixed_volume,
            _spacing,
            _affine,
            reference_threshold,
            _mask,
        ) = build_temporary_mesh_reference_volume(project_dir, elastic_to, ref_metadata)
        fixed_volume = np.asanyarray(fixed_volume)
    else:
        moving_volume = moving_on_disk
        reference_threshold = ref_metadata.get("threshold")
        if reference_threshold is None:
            return None
        fixed_path = _resolve_fixed_nifti_path(ref_dir, elastic_to)
        if fixed_path is None or not os.path.isfile(fixed_path):
            return None
        fixed_volume = ants.image_read(fixed_path).numpy()

    if moving_volume.shape != fixed_volume.shape:
        print(
            f"[elastic_ncc backfill] shape mismatch for {subject_name}: "
            f"moving {moving_volume.shape} vs fixed {fixed_volume.shape}; skipping"
        )
        return None

    return _union_mask_ncc(
        fixed_volume, moving_volume, reference_threshold, moving_threshold
    )


def _needs_backfill(json_data, subject_dir, subject_name):
    if not json_data.get("elastic_to"):
        return False
    if json_data.get("elastic_ncc") is not None:
        return False
    if json_data.get("is_mesh") is True and json_data.get("voxelized") is False:
        return False
    if json_data.get("threshold") is None:
        return False
    return _latest_voxel_elastic_edit(subject_dir, subject_name) is not None


def _list_candidates(directory):
    extracted_dir = os.path.join(directory, "extracted")
    if not os.path.isdir(extracted_dir):
        return []

    candidates = []
    for scan_name in sorted(os.listdir(extracted_dir)):
        subject_dir = os.path.join(extracted_dir, scan_name)
        if not os.path.isdir(subject_dir):
            continue
        json_path = os.path.join(subject_dir, f"{scan_name}.json")
        if not os.path.isfile(json_path):
            continue
        try:
            with open(json_path, "r") as jf:
                metadata = json.load(jf)
        except (OSError, json.JSONDecodeError):
            continue
        strip_ephemeral_scan_metadata(metadata)
        if _needs_backfill(metadata, subject_dir, scan_name):
            candidates.append((json_path, scan_name, metadata))
    return candidates


class BackfillElasticNccView(APIView):
    """TEMPORARY: project-wide elastic_ncc backfill (check_only or compute)."""

    def post(self, request, *args, **kwargs):
        directory = request.data.get("directory")
        check_only = bool(request.data.get("check_only", False))
        if not directory or not isinstance(directory, str):
            return Response({"error": "directory is required"}, status=status.HTTP_400_BAD_REQUEST)

        directory = os.path.abspath(os.path.expanduser(directory))
        if not os.path.isdir(directory):
            return Response(
                {"error": f"Directory not found: {directory}"},
                status=status.HTTP_404_NOT_FOUND,
            )

        candidates = _list_candidates(directory)
        pending_names = [name for _, name, _ in candidates]
        if check_only:
            return Response(
                {"pending_count": len(pending_names), "pending": pending_names},
                status=status.HTTP_200_OK,
            )

        total = len(candidates)
        if total == 0:
            return Response(
                {"updated": [], "failed": [], "count": 0, "pending_count": 0},
                status=status.HTTP_200_OK,
            )

        channel_layer = get_channel_layer()
        updated = []
        failed = []
        for idx, (json_path, scan_name, metadata) in enumerate(candidates):
            if channel_layer is not None:
                async_to_sync(channel_layer.group_send)(
                    "progress_group",
                    {
                        "type": "send_progress",
                        "progress": idx / total,
                        "scan_name": scan_name,
                        "custom_message": (
                            f"Updating registration quality score (NCC) for {scan_name} "
                            f"({idx + 1}/{total})..."
                        ),
                        "total": total,
                        "current": idx + 1,
                    },
                )
            try:
                elastic_ncc = _compute_ncc_for_subject(json_path, metadata)
                if elastic_ncc is None:
                    failed.append({"name": scan_name, "error": "could not compute NCC"})
                    continue
                metadata["elastic_ncc"] = float(elastic_ncc)
                strip_ephemeral_scan_metadata(metadata)
                with open(json_path, "w") as jf:
                    json.dump(metadata, jf, indent=4)
                updated.append({"name": scan_name, "elastic_ncc": float(elastic_ncc)})
                print(f"[elastic_ncc backfill] {scan_name}: {elastic_ncc:.4f}")
            except Exception as exc:
                print(f"[elastic_ncc backfill] failed for {scan_name}: {exc}")
                failed.append({"name": scan_name, "error": str(exc)})

        if channel_layer is not None:
            async_to_sync(channel_layer.group_send)(
                "progress_group",
                {
                    "type": "send_progress",
                    "progress": 1,
                    "scan_name": "all",
                    "custom_message": (
                        f"Updated NCC for {len(updated)} registration"
                        f"{'' if len(updated) == 1 else 's'}."
                    ),
                    "total": total,
                    "current": total,
                },
            )

        return Response(
            {
                "updated": updated,
                "failed": failed,
                "count": len(updated),
                "pending_count": total,
            },
            status=status.HTTP_200_OK,
        )
