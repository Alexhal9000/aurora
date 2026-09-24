"""Compact per-epoch validation previews for the fine-tune Run tab."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .storage import atomic_write_json, run_dir

PREVIEW_MAX_DIM = 192
MANIFEST_NAME = "manifest.json"


def preview_dir(root: Path) -> Path:
    path = Path(root) / "preview"
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_preview_dir(run_id: str) -> Path:
    return preview_dir(run_dir(run_id))


def subject_key(example: Dict[str, Any]) -> str:
    return f"{example.get('subject') or 'subject'}__label{int(example.get('label_id') or 0)}"


def choose_sticky_subject(examples: Sequence[Dict[str, Any]], rng) -> Optional[Dict[str, Any]]:
    if not examples:
        return None
    index = int(rng.integers(0, len(examples))) if rng is not None else 0
    return examples[int(index)]


def lossy_path_for_example(example: Dict[str, Any]) -> Optional[str]:
    image_path = Path(str(example.get("image_path") or ""))
    if not image_path.is_file():
        return None
    name = image_path.name
    if name.endswith(".nii.gz"):
        stem = name[: -len(".nii.gz")]
        candidate = image_path.with_name(f"{stem}_lossy.nii.gz")
        if candidate.is_file():
            return str(candidate)
    return str(image_path)


def preview_target_shape(shape: Sequence[int], max_dim: int = PREVIEW_MAX_DIM) -> Tuple[int, int, int]:
    depth, height, width = [int(v) for v in shape[:3]]
    peak = max(depth, height, width, 1)
    if peak <= int(max_dim):
        return depth, height, width
    scale = float(max_dim) / float(peak)
    return (
        max(1, int(round(depth * scale))),
        max(1, int(round(height * scale))),
        max(1, int(round(width * scale))),
    )


def resize_volume_nearest(volume: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    src = np.asarray(volume)
    target = tuple(int(v) for v in shape[:3])
    if tuple(src.shape[:3]) == target:
        return src
    z = np.clip(np.round(np.linspace(0, src.shape[0] - 1, target[0])).astype(int), 0, src.shape[0] - 1)
    y = np.clip(np.round(np.linspace(0, src.shape[1] - 1, target[1])).astype(int), 0, src.shape[1] - 1)
    x = np.clip(np.round(np.linspace(0, src.shape[2] - 1, target[2])).astype(int), 0, src.shape[2] - 1)
    return src[z[:, None, None], y[None, :, None], x[None, None, :]]


def fill_speckle_zeros(volume: np.ndarray, neighbor_mean_min: float = 40.0) -> np.ndarray:
    """Replace isolated 0s inside tissue with the 6-neighbour mean.

    Full-res scans can contain single-voxel dropouts. Nearest downsample to the
    preview grid turns each of those into a visible black speck. Training still
    uses the untouched volume; this is preview-only.
    """
    src = np.asarray(volume)
    if src.ndim < 3:
        return src
    values = src.astype(np.float32, copy=False)
    acc = np.zeros(src.shape, dtype=np.float32)
    acc[1:] += values[:-1]
    acc[:-1] += values[1:]
    acc[:, 1:] += values[:, :-1]
    acc[:, :-1] += values[:, 1:]
    acc[:, :, 1:] += values[:, :, :-1]
    acc[:, :, :-1] += values[:, :, 1:]
    mean = acc / 6.0
    out = np.array(src, copy=True)
    hole = (src == 0) & (mean >= float(neighbor_mean_min))
    if not np.any(hole):
        return out
    filled = np.clip(np.rint(mean[hole]), 0, 255)
    if np.issubdtype(out.dtype, np.integer):
        out[hole] = filled.astype(out.dtype, copy=False)
    else:
        out[hole] = filled
    return out


def _write_nifti(path: Path, data: np.ndarray) -> None:
    import nibabel as nib

    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(np.ascontiguousarray(data), np.eye(4))
    nib.save(img, str(path))


def _subject_dir(root: Path, key: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in key)
    path = preview_dir(root) / safe
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_subject_image(root: Path, example: Dict[str, Any], image_u8: np.ndarray) -> Dict[str, Any]:
    key = subject_key(example)
    folder = _subject_dir(root, key)
    image_path = folder / "image.nii.gz"
    target = preview_target_shape(image_u8.shape)
    if not image_path.is_file():
        preview = fill_speckle_zeros(resize_volume_nearest(image_u8, target).astype(np.uint8))
        _write_nifti(image_path, preview)
    else:
        preview = None
        target = preview_target_shape(image_u8.shape)
    return {
        "key": key,
        "subject": example.get("subject"),
        "label_id": int(example.get("label_id") or 0),
        "shape": list(target),
        "folder": str(folder.relative_to(preview_dir(root))),
    }


def scatter_clip_slab(orig_shape, view_index, slice_indices, pred_frames) -> np.ndarray:
    """Paint clip-frame preds back into original (D,H,W). Not a 3D ensemble."""
    from .volume_inference import _resize_mask_hw, inverse_reorient_for_view, reorient_for_view

    template = np.zeros(tuple(int(v) for v in orig_shape[:3]), dtype=np.uint8)
    oriented = reorient_for_view(template, int(view_index))
    depth, height, width = [int(v) for v in oriented.shape[:3]]
    filled = np.zeros_like(oriented)
    for z, pred in zip(slice_indices or [], pred_frames or []):
        zi = int(z)
        if zi < 0 or zi >= depth:
            continue
        filled[zi] = _resize_mask_hw(np.asarray(pred) > 0, height, width).astype(np.uint8)
    return inverse_reorient_for_view(filled, int(view_index)).astype(np.uint8)


def assemble_clip_overlay(orig_shape, clip_previews: Sequence[Dict[str, Any]]) -> np.ndarray:
    """Union of 3-view 8-slice clip preds. Empty elsewhere — Fast overlay only."""
    volume = np.zeros(tuple(int(v) for v in orig_shape[:3]), dtype=np.uint8)
    for row in clip_previews or []:
        slab = scatter_clip_slab(
            orig_shape,
            row.get("view_index") or 0,
            row.get("slice_indices") or [],
            row.get("frames") or [],
        )
        volume = np.maximum(volume, slab)
    return volume


def save_epoch_prediction(
    root: Path,
    example: Dict[str, Any],
    image_u8: np.ndarray,
    pred: np.ndarray,
    *,
    epoch: int,
    dice: Optional[float] = None,
) -> Dict[str, Any]:
    info = ensure_subject_image(root, example, image_u8)
    folder = preview_dir(root) / info["folder"]
    target = tuple(int(v) for v in info["shape"])
    mask = resize_volume_nearest((np.asarray(pred) > 0).astype(np.uint8), target)
    _write_nifti(folder / f"pred_epoch_{int(epoch):04d}.nii.gz", mask)
    return {**info, "epoch": int(epoch), "dice": None if dice is None else float(dice)}


def read_manifest(root: Path) -> Dict[str, Any]:
    path = preview_dir(root) / MANIFEST_NAME
    if not path.is_file():
        return {"subjects": [], "sticky_key": None, "epochs": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"subjects": [], "sticky_key": None, "epochs": []}


def update_manifest(
    root: Path,
    *,
    sticky_key: Optional[str],
    subject_info: Dict[str, Any],
    epoch: int,
    preview_kind: Optional[str] = None,
) -> Dict[str, Any]:
    payload = read_manifest(root)
    subjects = list(payload.get("subjects") or [])
    key = subject_info.get("key")
    existing = next((row for row in subjects if row.get("key") == key), None)
    row = {
        "key": key,
        "subject": subject_info.get("subject"),
        "label_id": subject_info.get("label_id"),
        "folder": subject_info.get("folder"),
        "shape": subject_info.get("shape"),
    }
    if existing is None:
        subjects.append(row)
    else:
        existing.update(row)
    epochs = sorted({int(v) for v in (payload.get("epochs") or []) + [int(epoch)]})
    kind = preview_kind or payload.get("preview_kind")
    payload = {
        "subjects": subjects,
        "sticky_key": sticky_key or payload.get("sticky_key") or key,
        "epochs": epochs,
        "preview_kind": kind,
    }
    atomic_write_json(preview_dir(root) / MANIFEST_NAME, payload)
    return payload


def resolve_preview_file(root: Path, *, subject_key_value: str, kind: str, epoch: Optional[int] = None) -> Path:
    manifest = read_manifest(root)
    match = next((row for row in (manifest.get("subjects") or []) if row.get("key") == subject_key_value), None)
    if match is None and (manifest.get("subjects") or []):
        match = next(
            (row for row in manifest["subjects"] if row.get("key") == manifest.get("sticky_key")),
            manifest["subjects"][0],
        )
    if match is None:
        raise FileNotFoundError("No preview subject is available yet")
    folder = preview_dir(root) / str(match.get("folder") or "")
    if kind == "image":
        path = folder / "image.nii.gz"
    elif kind == "pred":
        if epoch is None:
            epochs = manifest.get("epochs") or []
            if not epochs:
                raise FileNotFoundError("No preview epoch is available yet")
            epoch = int(epochs[-1])
        path = folder / f"pred_epoch_{int(epoch):04d}.nii.gz"
    else:
        raise FileNotFoundError(f"Unknown preview kind: {kind}")
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return path
