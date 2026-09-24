"""Build training examples from Aurora working-space image/mask pairs."""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .discover import find_latest_fullres_with_mask, inspect_subject


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def isolate_label(mask: np.ndarray, label_id: int) -> np.ndarray:
    return (np.asarray(mask) == int(label_id)).astype(np.uint8)


def occupied_slice_indices(binary_mask: np.ndarray) -> np.ndarray:
    counts = np.sum(np.asarray(binary_mask) > 0, axis=(1, 2))
    return np.flatnonzero(counts > 0)


def middle_slab_indices(binary_mask: np.ndarray, slab_fraction: float = 0.5) -> np.ndarray:
    """Central portion of the structure's occupied extent along the slice axis."""
    occupied = occupied_slice_indices(binary_mask)
    if occupied.size == 0:
        return np.array([int(binary_mask.shape[0] // 2)], dtype=int)
    lo = int(occupied[0])
    hi = int(occupied[-1])
    span = hi - lo + 1
    slab = max(1, int(round(span * float(slab_fraction))))
    start = lo + max(0, (span - slab) // 2)
    end = start + slab
    in_slab = occupied[(occupied >= start) & (occupied < end)]
    return in_slab if in_slab.size else occupied


def choose_init_slice(binary_mask: np.ndarray, rng=None, slab_fraction: float = 0.5) -> int:
    """Pick a prompt slice from the middle slab of the structure.

    Training (``rng`` provided): a random occupied slice from that middle slab.
    Evaluation (``rng`` is None): the centre occupied slice of the same slab.
    """
    slab = middle_slab_indices(binary_mask, slab_fraction=slab_fraction)
    if rng is None:
        return int(slab[len(slab) // 2])
    return int(rng.choice(slab))


def clip_indices_from_prompt(prompt_idx: int, depth: int, num_frames: int) -> List[int]:
    """Contiguous clip whose first frame is the prompt slice.

    Prefer slices after the prompt (increasing index). If the volume ends,
    walk backwards instead so the prompt remains frame 0.
    """
    if depth <= 0:
        return [0]
    prompt_idx = int(max(0, min(prompt_idx, depth - 1)))
    forward = list(range(prompt_idx, min(depth, prompt_idx + num_frames)))
    if len(forward) >= min(num_frames, depth):
        return forward[:num_frames]
    backward = list(range(prompt_idx, max(-1, prompt_idx - num_frames), -1))
    return backward[:num_frames]


def slice_window(center: int, depth: int, num_frames: int) -> List[int]:
    """Backward-compatible wrapper; prefer ``clip_indices_from_prompt``."""
    return clip_indices_from_prompt(center, depth, num_frames)


from ..aiModels.prompts import is_full_slice_prompt


def bbox_from_mask(mask2d: np.ndarray, padding: int = 2) -> Optional[List[float]]:
    ys, xs = np.where(np.asarray(mask2d) > 0)
    if ys.size == 0:
        return None
    h, w = mask2d.shape[:2]
    x0 = max(0, int(xs.min()) - padding)
    y0 = max(0, int(ys.min()) - padding)
    x1 = min(w - 1, int(xs.max()) + padding)
    y1 = min(h - 1, int(ys.max()) + padding)
    return [float(x0), float(y0), float(x1), float(y1)]


def resize_slice(image2d: np.ndarray, size: int, is_mask: bool) -> np.ndarray:
    from PIL import Image

    array = np.asarray(image2d)
    if array.ndim > 2:
        array = array[..., 0]
    pil = Image.fromarray(array.astype(np.uint8 if is_mask or array.dtype == np.uint8 else np.float32))
    resample = Image.NEAREST if is_mask else Image.BILINEAR
    resized = pil.resize((size, size), resample)
    out = np.array(resized)
    if is_mask:
        return (out > 0).astype(np.uint8)
    return out.astype(np.float32)


def volume_to_uint8(volume: np.ndarray) -> np.ndarray:
    array = np.asarray(volume)
    if array.dtype == np.uint8:
        return array
    vmin = float(np.min(array))
    vmax = float(np.max(array))
    if vmax <= vmin:
        return np.zeros(array.shape, dtype=np.uint8)
    scale = 255.0 / (vmax - vmin)
    return np.clip((array - vmin) * scale, 0, 255).astype(np.uint8)


def build_clip(
    image_3d: np.ndarray,
    binary_mask: np.ndarray,
    *,
    num_frames: int = 8,
    resolution: int = 512,
    rng=None,
    prompt=None,
) -> Optional[Dict[str, Any]]:
    if image_3d.shape != binary_mask.shape:
        raise ValueError(
            f"Image/mask shape mismatch: {tuple(image_3d.shape)} vs {tuple(binary_mask.shape)}"
        )
    init_slice = choose_init_slice(binary_mask, rng=rng)
    indices = clip_indices_from_prompt(init_slice, image_3d.shape[0], num_frames)
    image_u8 = volume_to_uint8(image_3d)

    frames = []
    masks = []
    for index in indices:
        frames.append(resize_slice(image_u8[index], resolution, is_mask=False))
        masks.append(resize_slice(binary_mask[index], resolution, is_mask=True))

    video = np.stack(frames, axis=0)  # T, H, W
    mask_clip = np.stack(masks, axis=0)
    height, width = mask_clip[0].shape
    full_slice = is_full_slice_prompt(prompt)
    if full_slice:
        first_bbox = [0.0, 0.0, float(width - 1), float(height - 1)]
        prompt_mask = np.ones((height, width), dtype=np.float32)
    else:
        first_bbox = bbox_from_mask(mask_clip[0])
        prompt_mask = np.ascontiguousarray(mask_clip[0], dtype=np.float32)
    if first_bbox is None:
        return None

    rgb = np.repeat(video[:, None, :, :], 3, axis=1).astype(np.float32) / 255.0
    rgb = np.ascontiguousarray((rgb - IMAGENET_MEAN) / IMAGENET_STD, dtype=np.float32)
    return {
        "video": rgb,
        "masks": np.ascontiguousarray(mask_clip, dtype=np.float32),
        "prompt_mask": np.ascontiguousarray(prompt_mask, dtype=np.float32),
        "bbox": np.ascontiguousarray(first_bbox, dtype=np.float32),
        "init_slice": int(init_slice),
        "slice_indices": indices,
        "full_slice_prompt": full_slice,
    }


def augment_clip(clip: Dict[str, Any], settings: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
    """Apply synchronized spatial + image-only intensity augmentation."""
    if not settings or not settings.get("enabled"):
        return clip

    video = clip["video"].copy()
    masks = clip["masks"].copy()
    bbox = np.asarray(clip["bbox"], dtype=np.float32).copy()
    _, _, height, width = video.shape

    if settings.get("flip", True) and rng.random() < 0.5:
        video = video[:, :, :, ::-1].copy()
        masks = masks[:, :, ::-1].copy()
        x0, y0, x1, y1 = bbox.tolist()
        bbox = np.asarray([width - 1 - x1, y0, width - 1 - x0, y1], dtype=np.float32)

    degrees = float(settings.get("affine_degrees") or 0)
    if settings.get("affine", True) and degrees > 0:
        angle = float(rng.uniform(-degrees, degrees))
        video, masks, bbox = _rotate_clip(video, masks, bbox, angle)

    if settings.get("brightness", True):
        amount = float(settings.get("brightness_amount") or 0.1)
        delta = float(rng.uniform(-amount, amount))
        video = video + delta
    if settings.get("contrast", True):
        amount = float(settings.get("contrast_amount") or 0.05)
        factor = 1.0 + float(rng.uniform(-amount, amount))
        mean = video.mean(axis=(2, 3), keepdims=True)
        video = (video - mean) * factor + mean

    clip = dict(clip)
    clip["video"] = np.ascontiguousarray(video, dtype=np.float32)
    clip["masks"] = np.ascontiguousarray(masks, dtype=np.float32)
    if clip.get("full_slice_prompt"):
        height, width = masks[0].shape
        clip["bbox"] = np.ascontiguousarray([0.0, 0.0, float(width - 1), float(height - 1)], dtype=np.float32)
        clip["prompt_mask"] = np.ascontiguousarray(np.ones_like(masks[0], dtype=np.float32))
    else:
        clip["bbox"] = np.ascontiguousarray(bbox, dtype=np.float32)
        clip["prompt_mask"] = np.ascontiguousarray(masks[0], dtype=np.float32)
    return clip


def _rotate_clip(video, masks, bbox, angle):
    from scipy.ndimage import rotate

    rotated_frames = []
    for frame in video:
        channels = [
            rotate(frame[c], angle, reshape=False, order=1, mode="nearest")
            for c in range(frame.shape[0])
        ]
        rotated_frames.append(np.stack(channels, axis=0))
    video_r = np.stack(rotated_frames, axis=0)
    mask_r = np.stack(
        [
            (rotate(mask, angle, reshape=False, order=0, mode="nearest") > 0.5).astype(np.float32)
            for mask in masks
        ],
        axis=0,
    )
    new_bbox = bbox_from_mask(mask_r[0])
    if new_bbox is None:
        return video, masks, bbox
    return video_r, mask_r, np.asarray(new_bbox, dtype=np.float32)


def load_nifti(path: str) -> Tuple[np.ndarray, np.ndarray, Tuple[float, ...]]:
    import nibabel as nib
    image = nib.load(path)
    data = np.asanyarray(image.dataobj)
    return data, image.affine.copy(), tuple(float(z) for z in image.header.get_zooms()[:3])


def load_mask(path: str) -> Tuple[np.ndarray, np.ndarray, Tuple[float, ...]]:
    import nibabel as nib
    import shutil

    with tempfile.TemporaryDirectory() as tmp:
        temp_path = os.path.join(tmp, "mask.nii.gz")
        shutil.copy2(path, temp_path)
        image = nib.load(temp_path)
        data = np.round(image.get_fdata()).astype(np.uint8)
        return data, image.affine.copy(), tuple(float(z) for z in image.header.get_zooms()[:3])


def collect_examples(
    directory: str,
    subjects: Sequence[str],
    label_ids: Sequence[int],
    subject_records: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """One binary example per (subject, label) that actually contains the label."""
    examples = []
    for name in subjects:
        record = (subject_records or {}).get(name) or inspect_subject(directory, name)
        scan_dir = os.path.join(directory, "extracted", name)
        image_name = record.get("image_filename")
        mask_name = record.get("mask_filename")
        if not image_name or not mask_name:
            image_name, mask_name = find_latest_fullres_with_mask(scan_dir, name)
        if not image_name or not mask_name:
            continue
        for label_id in label_ids:
            known = [int(v) for v in (record.get("label_ids") or [])]
            if known and int(label_id) not in known:
                continue
            examples.append({
                "subject": name,
                "label_id": int(label_id),
                "image_filename": image_name,
                "mask_filename": mask_name,
                "image_path": os.path.join(scan_dir, image_name),
                "mask_path": os.path.join(scan_dir, mask_name),
                "voxel_size": record.get("voxel_size"),
                "affine_zooms": record.get("affine_zooms"),
                "alignment_to": record.get("alignment_to"),
                "elastic_to": record.get("elastic_to"),
                "intensity_value_mapping": record.get("intensity_value_mapping"),
            })
    return examples


_EXAMPLE_CACHE: Dict[Tuple[str, str, int], Tuple[np.ndarray, np.ndarray]] = {}


def _example_cache_key(example: Dict[str, Any]) -> Tuple[str, str, int]:
    return (
        str(example.get("image_path") or ""),
        str(example.get("mask_path") or ""),
        int(example.get("label_id") or 0),
    )


def clear_example_cache() -> None:
    _EXAMPLE_CACHE.clear()


def load_example_arrays(example: Dict[str, Any], *, use_cache: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Return (uint8 image, binary label mask), cached after the first read."""
    key = _example_cache_key(example)
    if use_cache and key in _EXAMPLE_CACHE:
        return _EXAMPLE_CACHE[key]
    image, _, _ = load_nifti(example["image_path"])
    mask, _, _ = load_mask(example["mask_path"])
    if image.shape != mask.shape:
        raise ValueError(
            f"Image/mask shape mismatch for {example.get('subject')}: "
            f"{tuple(image.shape)} vs {tuple(mask.shape)}"
        )
    image_u8 = volume_to_uint8(image)
    binary = isolate_label(mask, example["label_id"])
    if use_cache:
        _EXAMPLE_CACHE[key] = (image_u8, binary)
    return image_u8, binary


def preload_example_cache(examples: Sequence[Dict[str, Any]]) -> int:
    """Load unique (image, mask, label) volumes into RAM once per fold."""
    loaded = 0
    for example in examples:
        key = _example_cache_key(example)
        if key in _EXAMPLE_CACHE:
            continue
        load_example_arrays(example, use_cache=True)
        loaded += 1
    return loaded
