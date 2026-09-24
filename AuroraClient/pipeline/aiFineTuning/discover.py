"""Inspect an Aurora project for fine-tuning-eligible subjects and labels.

Discover is a filesystem walk plus NIfTI **headers**. It does not decompress
image or mask voxels — that is what made the old call take minutes on a
full-res project.
"""

from __future__ import annotations

import gzip
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from ..aiModels.registry import list_model_descriptors
from ..coordinateFrames import (
    is_preserved_mesh_metadata,
    list_extracted_scan_names,
    load_extracted_scan_metadata,
    load_project_label_names,
)
from ..meshGridTools import sort_edit_filenames

# Warp fields and conversion leftovers that share the `{name}_edit_{n}*.nii.gz` glob.
_AUXILIARY_VOLUME_SUFFIXES = (
    "_inv.nii.gz",
    "_fwd.nii.gz",
    "_mask_temp.nii.gz",
    "_mask.nii.gz",
    "_removal_mask.nii.gz",
)


def discover_project(directory: str) -> Dict[str, Any]:
    if not directory:
        raise ValueError("directory is required")
    project_dir = os.path.abspath(directory)
    extracted = os.path.join(project_dir, "extracted")
    if not os.path.isdir(extracted):
        raise FileNotFoundError(f"No extracted/ directory in {project_dir}")

    label_names = {
        str(key): str(value)
        for key, value in load_project_label_names(project_dir).items()
        if str(value).strip()
    }
    subjects = []
    for name in list_extracted_scan_names(project_dir):
        subjects.append(inspect_subject(project_dir, name))

    eligible_names = [item["name"] for item in subjects if item.get("eligible")]
    all_ids = sorted({
        int(key) for key in label_names.keys() if str(key).isdigit()
    })
    labels = []
    for label_id in all_ids:
        labels.append({
            "id": label_id,
            "name": label_names.get(str(label_id)) or f"Label {label_id}",
            "subject_count": len(eligible_names),
            "subjects": list(eligible_names),
        })

    return {
        "directory": project_dir,
        "project_name": os.path.basename(os.path.normpath(project_dir)) or project_dir,
        "label_names": label_names,
        "labels": labels,
        "subjects": subjects,
        "eligible_subject_count": len(eligible_names),
        "models": list_model_descriptors(include_capabilities=True),
    }


def inspect_subject(directory: str, scan_name: str) -> Dict[str, Any]:
    scan_dir = os.path.join(directory, "extracted", scan_name)
    issues: List[str] = []
    try:
        metadata = load_extracted_scan_metadata(directory, scan_name) or {}
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        metadata = {}
        issues.append("metadata_missing")

    record: Dict[str, Any] = {
        "name": scan_name,
        "eligible": False,
        "issues": issues,
        "image_filename": None,
        "mask_filename": None,
        "shape": None,
        "voxel_size": metadata.get("voxel_size"),
        "affine_zooms": None,
        "label_ids": [],
        "is_mesh": bool(is_preserved_mesh_metadata(metadata)),
        "faulty": bool(metadata.get("faulty")),
        "alignment_to": metadata.get("alignment_to"),
        "elastic_to": metadata.get("elastic_to"),
        "intensity_value_mapping": metadata.get("intensity_value_mapping"),
        "original_format": metadata.get("original_format"),
    }

    if not os.path.isdir(scan_dir):
        issues.append("subject_directory_missing")
        return record
    if record["is_mesh"]:
        issues.append("mesh_only")
        return record
    if record["faulty"]:
        issues.append("faulty")

    image_filename, mask_filename = find_latest_fullres_with_mask(scan_dir, scan_name)
    record["image_filename"] = image_filename
    record["mask_filename"] = mask_filename
    if not image_filename:
        issues.append("image_missing")
    if not mask_filename:
        issues.append("mask_missing")
        record["eligible"] = False
        return record

    try:
        image_shape, image_zooms = _read_nifti_geometry(os.path.join(scan_dir, image_filename))
        mask_shape, mask_zooms = _read_nifti_geometry(os.path.join(scan_dir, mask_filename))
    except Exception as exc:
        issues.append(f"load_failed:{exc}")
        return record

    record["shape"] = list(image_shape)
    record["affine_zooms"] = [float(z) for z in image_zooms]
    if tuple(image_shape) != tuple(mask_shape):
        issues.append("image_mask_shape_mismatch")
    elif not _zooms_compatible(image_zooms, mask_zooms):
        issues.append("image_mask_geometry_mismatch")

    blocking = {
        "image_missing",
        "mask_missing",
        "image_mask_shape_mismatch",
        "image_mask_geometry_mismatch",
        "mesh_only",
        "faulty",
    }
    record["eligible"] = not any(
        issue in blocking or str(issue).startswith("load_failed")
        for issue in issues
    )
    return record


def find_latest_fullres_with_mask(scan_dir: str, scan_name: str) -> Tuple[Optional[str], Optional[str]]:
    """Latest non-elastic full-res working volume that has a paired mask.

    Elastic registrations (and warp sidecars) are skipped even if they have a
    mask. When there are no non-elastic edits, the raw ``{scan}.nii.gz`` is the
    working volume. If that (or a later non-elastic edit) has no mask, the image
    filename is still returned so the subject can be flagged ``mask_missing``.
    """
    images = _list_fullres_niftis(scan_dir, scan_name)
    if not images:
        return None, None
    paired = []
    for image_name in images:
        mask_name = _mask_name_for_image(image_name)
        if os.path.isfile(os.path.join(scan_dir, mask_name)):
            paired.append((image_name, mask_name))
    if paired:
        return paired[-1]
    return images[-1], None


def _is_elastic_volume(filename: str) -> bool:
    return "elastic" in os.path.basename(filename).lower()


def _is_working_fullres_volume(filename: str, scan_name: str) -> bool:
    if not filename.endswith(".nii.gz"):
        return False
    if "_lossy" in filename:
        return False
    if _is_elastic_volume(filename):
        return False
    if any(filename.endswith(suffix) for suffix in _AUXILIARY_VOLUME_SUFFIXES):
        return False
    if filename == f"{scan_name}.nii.gz":
        return True
    return filename.startswith(f"{scan_name}_edit_")


def _list_fullres_niftis(scan_dir: str, scan_name: str) -> List[str]:
    if not os.path.isdir(scan_dir):
        return []
    names = [
        filename
        for filename in os.listdir(scan_dir)
        if _is_working_fullres_volume(filename, scan_name)
        and os.path.isfile(os.path.join(scan_dir, filename))
    ]
    return sort_edit_filenames(names)


def _mask_name_for_image(image_filename: str) -> str:
    if image_filename.endswith(".nii.gz"):
        return image_filename[: -len(".nii.gz")] + ".nii.mask.gz"
    return image_filename + ".nii.mask.gz"


def _zooms_compatible(left, right, rtol: float = 1e-3) -> bool:
    if left is None or right is None:
        return True
    if len(left) != len(right):
        return False
    return all(abs(float(a) - float(b)) <= rtol * max(abs(float(a)), abs(float(b)), 1.0) for a, b in zip(left, right))


def _read_nifti_geometry(path: str):
    """Read shape and zooms from the NIfTI header only — no voxel decompress."""
    import nibabel as nib
    from nibabel.nifti1 import Nifti1Header

    if path.endswith(".nii.mask.gz") or path.endswith(".nii.mask"):
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rb") as handle:
            header = Nifti1Header.from_fileobj(handle, check=False)
    else:
        header = nib.load(path).header
    shape = tuple(int(v) for v in header.get_data_shape()[:3])
    zooms = tuple(float(v) for v in header.get_zooms()[:3])
    return shape, zooms
