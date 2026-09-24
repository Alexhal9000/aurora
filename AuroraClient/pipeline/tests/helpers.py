"""Tiny synthetic Aurora projects for fine-tuning tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


def write_project(root: Path, *, subjects=None, label_names=None):
    extracted = root / "extracted"
    extracted.mkdir(parents=True, exist_ok=True)
    settings = {
        "selected_reference": None,
        "general_settings": {},
        "label_names": label_names or {"1": "Mandible", "2": "Maxilla"},
    }
    (extracted / "project_settings.json").write_text(json.dumps(settings, indent=2))
    records = subjects or [
        {"name": "SubA", "labels": [1, 2]},
        {"name": "SubB", "labels": [1]},
        {"name": "SubC", "labels": [1, 2]},
        {"name": "SubD", "labels": [2]},
    ]
    for spec in records:
        _write_subject(extracted, spec)
    return root


def _write_subject(extracted: Path, spec):
    import nibabel as nib

    name = spec["name"]
    scan_dir = extracted / name
    scan_dir.mkdir(parents=True, exist_ok=True)
    shape = spec.get("shape", (8, 8, 8))
    image = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    mask = np.zeros(shape, dtype=np.uint8)
    for label in spec.get("labels") or []:
        mask[label:label + 2, label:label + 2, label:label + 2] = int(label)
    affine = np.diag([0.05, 0.05, 0.05, 1.0])
    image_name = spec.get("image_filename", f"{name}.nii.gz")
    mask_name = spec.get("mask_filename", f"{name}.nii.mask.gz")
    nib.save(nib.Nifti1Image(image, affine), str(scan_dir / image_name))
    # Mask sidecar uses a non-standard extension; write as .nii.gz then rename.
    temp_mask = scan_dir / f"{name}_mask_tmp.nii.gz"
    nib.save(nib.Nifti1Image(mask, affine), str(temp_mask))
    os.replace(temp_mask, scan_dir / mask_name)
    metadata = {
        "name": name,
        "voxel_size": 0.05,
        "threshold": 0,
        "original_format": "NIfTI",
        "faulty": bool(spec.get("faulty")),
        "is_mesh": bool(spec.get("is_mesh")),
        "voxelized": False if spec.get("is_mesh") else True,
        "alignment_to": spec.get("alignment_to"),
        "elastic_to": spec.get("elastic_to"),
    }
    (scan_dir / f"{name}.json").write_text(json.dumps(metadata, indent=2))
