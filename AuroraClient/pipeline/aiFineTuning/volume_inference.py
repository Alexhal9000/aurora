"""3D MedSAM2 volume propagation shared by training validation and production inference.

Uses the ``SAM2VideoTrainer`` engine with optional 3-view (axial / coronal /
sagittal) majority voting so fine-tuning validation figures and segmentation
inference follow the same path.
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, Optional, Tuple

import numpy as np

MEDSAM2_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "MedSAM2"))
DEFAULT_RESOLUTION = 512
PROPAGATION_CHUNK = 8
ENSEMBLE_VIEWS = 3
ENSEMBLE_VOTES_REQUIRED = 2
INFERENCE_MAX_BATCH = 2
VIEW_NAMES = ("axial", "coronal", "sagittal")


def _log_inference(message: str, *, verbose: bool) -> None:
    if verbose:
        print(message, flush=True)


def _new_progress_tracker(total: int) -> Dict[str, Any]:
    return {
        "total": max(0, int(total)),
        "done": 0,
        "start": time.monotonic(),
        "last_logged_pct": -1,
    }


def _log_chunk_progress(progress: Optional[Dict[str, Any]], *, verbose: bool, label: str = "Trainer inference") -> None:
    if not verbose or not progress:
        return
    total = int(progress.get("total") or 0)
    done = int(progress.get("done") or 0)
    if total <= 0:
        return
    pct = int(100 * done / total)
    last_pct = int(progress.get("last_logged_pct", -1))
    milestone = (pct // 10) * 10
    if done != 0 and done != total and milestone <= last_pct:
        return
    elapsed = time.monotonic() - float(progress.get("start", time.monotonic()))
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else 0.0
    _log_inference(
        f"  {label}: {done}/{total} chunks ({pct}%), "
        f"elapsed {elapsed:.0f}s, ETA {eta:.0f}s",
        verbose=True,
    )
    progress["last_logged_pct"] = 100 if done >= total else milestone


def reorient_for_view(volume: np.ndarray, view_index: int) -> np.ndarray:
    """Map original (D,H,W) to the propagation-first layout for a view."""
    vol = np.asarray(volume)
    if view_index == 0:
        return vol
    if view_index == 1:
        return np.moveaxis(vol, 1, 0)
    if view_index == 2:
        return np.moveaxis(vol, 2, 0)
    raise ValueError(f"Unsupported view_index: {view_index}")


def inverse_reorient_for_view(volume: np.ndarray, view_index: int) -> np.ndarray:
    """Map a propagated mask back to original (D,H,W)."""
    vol = np.asarray(volume)
    if view_index == 0:
        return vol
    if view_index == 1:
        return np.moveaxis(vol, 0, 1)
    if view_index == 2:
        return np.moveaxis(vol, 0, 2)
    raise ValueError(f"Unsupported view_index: {view_index}")


def occupied_indices_along_axis(binary: np.ndarray, axis: int = 0) -> np.ndarray:
    binary = np.asarray(binary) > 0
    reduce_axes = tuple(i for i in range(binary.ndim) if i != axis)
    counts = np.sum(binary, axis=reduce_axes)
    return np.flatnonzero(counts > 0)


def estimate_extent_along_axis(
    image_u8: np.ndarray,
    axis: int = 0,
    relative_threshold: float = 0.15,
) -> Tuple[int, int]:
    """Estimate tissue extent along one axis without a segmentation mask."""
    depth = int(image_u8.shape[axis])
    if depth <= 0:
        return 0, 0
    slabs = np.moveaxis(image_u8, axis, 0) if axis != 0 else image_u8
    flat = slabs.reshape(depth, -1)
    p95 = np.percentile(flat, 95, axis=1)
    p5 = np.percentile(flat, 5, axis=1)
    scores_arr = (p95 - p5).astype(np.float32)
    peak = float(np.max(scores_arr))
    if peak <= 0:
        return 0, depth - 1
    threshold = peak * float(relative_threshold)
    occupied = np.flatnonzero(scores_arr >= threshold)
    if occupied.size == 0:
        return 0, depth - 1
    lo = max(0, int(occupied[0]) - 1)
    hi = min(depth - 1, int(occupied[-1]) + 1)
    return lo, hi


def estimate_tissue_slice_extent(
    image_u8: np.ndarray,
    relative_threshold: float = 0.15,
) -> Tuple[int, int]:
    """Axial (Z) tissue extent — backward compatible helper."""
    return estimate_extent_along_axis(image_u8, axis=0, relative_threshold=relative_threshold)


def choose_init_slice_from_extent(lo: int, hi: int, slab_fraction: float = 0.5) -> int:
    """Middle-slab centre slice, matching eval-time ``choose_init_slice(..., rng=None)``."""
    lo = int(lo)
    hi = int(hi)
    span = hi - lo + 1
    slab = max(1, int(round(span * float(slab_fraction))))
    start = lo + max(0, (span - slab) // 2)
    end = start + slab
    center = start + (end - start) // 2
    return int(max(lo, min(hi, center)))


def _resize_mask_hw(mask2d, height, width):
    from PIL import Image

    arr = (np.asarray(mask2d) > 0.5).astype(np.uint8) * 255
    out = np.array(Image.fromarray(arr).resize((int(width), int(height)), Image.NEAREST))
    return out > 0


def _video_from_z(image_u8, indices, resolution):
    from .dataset import IMAGENET_MEAN, IMAGENET_STD, resize_slice

    frames = [resize_slice(image_u8[int(z)], resolution, is_mask=False) for z in indices]
    video = np.repeat(np.stack(frames, axis=0)[:, None, :, :], 3, axis=1).astype(np.float32) / 255.0
    return np.ascontiguousarray((video - IMAGENET_MEAN) / IMAGENET_STD, dtype=np.float32)


def _infer_prompted_video(trainer, video, bbox, device, dense_mask=None):
    batch = _infer_prompted_video_batch(
        trainer,
        [video],
        [np.asarray(bbox, dtype=np.float32)],
        [dense_mask],
        device,
    )
    return batch[0]


def _infer_prompted_video_batch(trainer, videos, bboxes, dense_masks, device, max_batch=None):
    """Run trainer forwards in groups. Default 2 is for live inference beside other VRAM use."""
    if not videos:
        return []
    out = []
    step = max(1, int(INFERENCE_MAX_BATCH if max_batch is None else max_batch))
    for start in range(0, len(videos), step):
        end = start + step
        out.extend(_infer_prompted_video_batch_once(
            trainer, videos[start:end], bboxes[start:end], dense_masks[start:end], device,
        ))
    return out


def _infer_prompted_video_batch_once(trainer, videos, bboxes, dense_masks, device):
    import torch

    video_t = torch.from_numpy(np.ascontiguousarray(np.stack(videos, axis=0))).to(device).contiguous()
    bbox_t = torch.from_numpy(np.ascontiguousarray(np.stack(bboxes, axis=0))).to(device).contiguous()
    dense_t = None
    if all(mask is not None for mask in dense_masks):
        dense_t = torch.from_numpy(
            np.ascontiguousarray(np.stack(dense_masks, axis=0), dtype=np.float32)
        ).unsqueeze(1).to(device).contiguous()
    amp = getattr(device, "type", None) == "cuda"
    with torch.cuda.amp.autocast(enabled=True) if amp else nullcontext():
        pred_masks, _, _ = trainer(video_t, bbox_t, labels=None, dense_prompt_mask=dense_t)
    batch = []
    for item in range(int(video_t.shape[0])):
        batch.append([
            (mask > 0.5).detach().cpu().numpy()[item, 0]
            for mask in pred_masks
        ])
    del pred_masks, video_t, bbox_t, dense_t
    return batch


def _chunk_indices(indices, chunk=PROPAGATION_CHUNK):
    indices = list(indices)
    return [indices[start:start + chunk] for start in range(0, len(indices), chunk)]


def _initial_prompt(prompt_mask, resolution, full_slice):
    from .dataset import bbox_from_mask, resize_slice

    if full_slice:
        dense = np.ones((resolution, resolution), dtype=np.float32)
        bbox = np.asarray([0.0, 0.0, float(resolution - 1), float(resolution - 1)], dtype=np.float32)
        return dense, bbox
    prompt_res = resize_slice(np.asarray(prompt_mask).astype(np.uint8), resolution, is_mask=True)
    bbox = bbox_from_mask(prompt_res)
    if bbox is None:
        return None, None
    return prompt_res.astype(np.float32), np.asarray(bbox, dtype=np.float32)


def _volume_from_preds(preds, depth, height, width):
    volume = np.zeros((depth, height, width), dtype=bool)
    for z, pred in preds.items():
        volume[int(z)] = _resize_mask_hw(pred, height, width)
    return volume


def propagate_z(
    trainer,
    image_u8,
    indices,
    prompt_mask,
    height,
    width,
    resolution,
    device,
    chunk=PROPAGATION_CHUNK,
    full_slice=False,
):
    """Propagate along one ordered index list.

    Drawn-organ prompts chain each chunk from the previous prediction (video
    tracking). Whole-slice ones prompts reset every chunk to the full-frame
    box + ones mask, matching how no-prompt clips are trained.
    """
    volumes = propagate_directions(
        trainer,
        [{
            "image_u8": image_u8,
            "indices": list(indices),
            "prompt_mask": prompt_mask,
            "height": height,
            "width": width,
            "full_slice": full_slice,
        }],
        resolution,
        device,
        chunk=chunk,
    )
    if not volumes:
        return np.zeros((image_u8.shape[0], height, width), dtype=bool)
    return volumes[0]


def propagate_directions(
    trainer,
    directions,
    resolution,
    device,
    chunk=PROPAGATION_CHUNK,
    max_batch=None,
    *,
    verbose: bool = False,
    progress: Optional[Dict[str, Any]] = None,
):
    """Run several axis-direction streams, batched on GPU.

    Independent no-prompt chunks are all queued up front. Prompted streams
    stay sequential along each ray (left/right need the previous mask) but
    the 6 rays still share one batched forward.
    """
    from .dataset import bbox_from_mask

    streams = []
    for direction in directions:
        indices = list(direction.get("indices") or [])
        full_slice = bool(direction.get("full_slice"))
        dense, bbox = _initial_prompt(
            direction.get("prompt_mask"), resolution, full_slice,
        )
        image_u8 = direction["image_u8"]
        height = int(direction["height"])
        width = int(direction["width"])
        empty = np.zeros((image_u8.shape[0], height, width), dtype=bool)
        if bbox is None or not indices:
            streams.append({"done": True, "volume": empty, "preds": {}})
            continue
        streams.append({
            "done": False,
            "image_u8": image_u8,
            "indices": indices,
            "chunks": _chunk_indices(indices, chunk=chunk),
            "chunk_i": 0,
            "dense": dense,
            "bbox": bbox,
            "dense_first": dense,
            "bbox_first": bbox,
            "full_slice": full_slice,
            "height": height,
            "width": width,
            "preds": {},
        })

    independent = [stream for stream in streams if not stream["done"] and stream["full_slice"]]
    chained = [stream for stream in streams if not stream["done"] and not stream["full_slice"]]
    independent_jobs = sum(len(stream["chunks"]) for stream in independent)
    chained_jobs = sum(len(stream["chunks"]) for stream in chained)
    if verbose:
        _log_inference(
            f"Propagation: {independent_jobs + chained_jobs} chunks "
            f"({independent_jobs} independent, {chained_jobs} chained), "
            f"batch_size={max_batch or INFERENCE_MAX_BATCH}",
            verbose=True,
        )
        _log_chunk_progress(progress, verbose=True)

    if independent:
        _run_independent_chunks(
            trainer, independent, resolution, device,
            max_batch=max_batch, verbose=verbose, progress=progress,
        )
    if chained:
        _run_chained_streams(
            trainer, chained, resolution, device, bbox_from_mask,
            max_batch=max_batch, verbose=verbose, progress=progress,
        )

    volumes = []
    for stream in streams:
        if "volume" in stream and stream.get("done") and not stream.get("preds"):
            volumes.append(stream["volume"])
            continue
        volumes.append(_volume_from_preds(
            stream["preds"],
            stream["image_u8"].shape[0],
            stream["height"],
            stream["width"],
        ))
    return volumes


def _run_independent_chunks(trainer, streams, resolution, device, max_batch=None, *, verbose=False, progress=None):
    jobs = []
    for stream in streams:
        for take in stream["chunks"]:
            jobs.append((stream, take, stream["dense_first"], stream["bbox_first"]))
        stream["done"] = True
    _run_jobs_grouped(
        trainer, jobs, resolution, device,
        max_batch=max_batch, verbose=verbose, progress=progress,
    )


def _run_chained_streams(trainer, streams, resolution, device, bbox_from_mask, max_batch=None, *, verbose=False, progress=None):
    while True:
        jobs = []
        owners = []
        for stream in streams:
            if stream["chunk_i"] >= len(stream["chunks"]):
                stream["done"] = True
                continue
            take = stream["chunks"][stream["chunk_i"]]
            jobs.append((stream, take, stream["dense"], stream["bbox"]))
            owners.append(stream)
        if not jobs:
            break
        _run_jobs_grouped(
            trainer, jobs, resolution, device,
            max_batch=max_batch, verbose=verbose, progress=progress,
        )
        for stream in owners:
            take = stream["chunks"][stream["chunk_i"]]
            previous = stream["preds"].get(int(take[-1]))
            stream["chunk_i"] += 1
            if previous is None:
                stream["done"] = True
                continue
            stream["dense"] = previous.astype(np.float32)
            stream["bbox"] = np.asarray(bbox_from_mask(previous) or stream["bbox"], dtype=np.float32)


def _run_jobs_grouped(trainer, jobs, resolution, device, max_batch=None, *, verbose=False, progress=None):
    grouped = {}
    for stream, take, dense, bbox in jobs:
        grouped.setdefault(len(take), []).append((stream, take, dense, bbox))
    for batch in grouped.values():
        _run_job_batch(
            trainer, batch, resolution, device,
            max_batch=max_batch, verbose=verbose, progress=progress,
        )


def _run_job_batch(trainer, jobs, resolution, device, max_batch=None, *, verbose=False, progress=None):
    videos = []
    bboxes = []
    denses = []
    for stream, take, dense, bbox in jobs:
        videos.append(_video_from_z(stream["image_u8"], take, resolution))
        bboxes.append(np.asarray(bbox, dtype=np.float32))
        denses.append(np.asarray(dense, dtype=np.float32))
    outputs = _infer_prompted_video_batch(
        trainer, videos, bboxes, denses, device, max_batch=max_batch,
    )
    for (stream, take, _dense, _bbox), preds in zip(jobs, outputs):
        for z, pred in zip(take, preds):
            stream["preds"][int(z)] = pred
    if progress is not None:
        progress["done"] = int(progress.get("done", 0)) + len(jobs)
        _log_chunk_progress(progress, verbose=verbose)


def _count_propagation_chunks(directions, chunk=PROPAGATION_CHUNK) -> int:
    total = 0
    for direction in directions:
        indices = list(direction.get("indices") or [])
        if not indices:
            continue
        total += len(_chunk_indices(indices, chunk=chunk))
    return total


def _propagation_window(image_u8, binary_hint, depth: int):
    from .dataset import choose_init_slice, volume_to_uint8

    if binary_hint is not None and np.any(np.asarray(binary_hint) > 0):
        binary = (np.asarray(binary_hint) > 0).astype(np.uint8)
        occupied = occupied_indices_along_axis(binary, axis=0)
        if occupied.size == 0:
            lo, hi = estimate_extent_along_axis(volume_to_uint8(image_u8), axis=0)
            prompt_idx = choose_init_slice_from_extent(lo, hi)
            return prompt_idx, lo, hi
        prompt_idx = choose_init_slice(binary, rng=None)
        lo = max(0, int(occupied[0]) - 1)
        hi = min(depth - 1, int(occupied[-1]) + 1)
        return prompt_idx, lo, hi

    lo, hi = estimate_extent_along_axis(image_u8, axis=0)
    prompt_idx = choose_init_slice_from_extent(lo, hi)
    return prompt_idx, lo, hi


def predict_volume_single_view(
    trainer,
    image,
    device,
    resolution,
    *,
    prompt=None,
    binary_hint=None,
    view_index: int = 0,
):
    """Propagate along one orthogonal view using the trainer engine."""
    import torch
    from .dataset import is_full_slice_prompt, volume_to_uint8

    image_u8 = reorient_for_view(volume_to_uint8(image), view_index)
    hint = None
    if binary_hint is not None:
        hint = reorient_for_view(np.asarray(binary_hint), view_index)

    depth, height, width = [int(v) for v in image_u8.shape[:3]]
    prompt_idx, lo, hi = _propagation_window(image_u8, hint, depth)
    forward = list(range(prompt_idx, hi + 1))
    backward = list(range(prompt_idx, lo - 1, -1))
    full_slice = is_full_slice_prompt(prompt)
    if full_slice:
        prompt_mask = np.ones((height, width), dtype=np.uint8)
    elif hint is not None and np.any(hint > 0):
        prompt_mask = (hint[prompt_idx] > 0).astype(np.uint8)
    else:
        prompt_mask = np.ones((height, width), dtype=np.uint8)

    directions = [{
        "image_u8": image_u8,
        "indices": forward,
        "prompt_mask": prompt_mask,
        "height": height,
        "width": width,
        "full_slice": full_slice,
    }]
    if len(backward) > 1:
        directions.append({
            "image_u8": image_u8,
            "indices": backward,
            "prompt_mask": prompt_mask,
            "height": height,
            "width": width,
            "full_slice": full_slice,
        })

    trainer.model.eval()
    with torch.no_grad():
        volumes = propagate_directions(trainer, directions, resolution, device)
    pred = volumes[0]
    if len(volumes) > 1:
        pred = np.logical_or(pred, volumes[1])
    return inverse_reorient_for_view(pred, view_index)


def predict_volume_ensemble(
    trainer,
    image,
    device,
    resolution=DEFAULT_RESOLUTION,
    *,
    prompt=None,
    binary_hint=None,
    votes_required: int = ENSEMBLE_VOTES_REQUIRED,
    max_batch=None,
    verbose: bool = False,
):
    """3-view majority vote (axial + coronal + sagittal) in original (D,H,W) space.

    All six half-axes (left/right × 3 views) are batched together. No-prompt
    chunks are independent; drawn-organ rays stay sequential along each axis.
    """
    import torch
    from .dataset import is_full_slice_prompt, volume_to_uint8

    shape = tuple(int(v) for v in np.asarray(image).shape[:3])
    full_slice = is_full_slice_prompt(prompt)
    mode = "full_slice_ones" if full_slice else ("mask_prompt" if binary_hint is not None else "tissue_bounded")
    _log_inference(
        f"Trainer ensemble: volume shape={shape}, mode={mode}, "
        f"resolution={resolution}, votes={votes_required}/{ENSEMBLE_VIEWS}",
        verbose=verbose,
    )
    _log_inference("Converting volume to uint8...", verbose=verbose)
    convert_started = time.monotonic()
    image_u8_orig = volume_to_uint8(image)
    _log_inference(
        f"Volume converted in {time.monotonic() - convert_started:.1f}s "
        f"({image_u8_orig.dtype}, {image_u8_orig.nbytes / (1024 ** 3):.2f} GiB)",
        verbose=verbose,
    )
    directions = []
    owners = []
    for view_index in range(ENSEMBLE_VIEWS):
        view_started = time.monotonic()
        view_name = VIEW_NAMES[view_index] if view_index < len(VIEW_NAMES) else f"view_{view_index}"
        _log_inference(
            f"  Planning view {view_index + 1}/{ENSEMBLE_VIEWS} ({view_name})...",
            verbose=verbose,
        )
        image_u8 = reorient_for_view(image_u8_orig, view_index)
        hint = None
        if binary_hint is not None:
            hint = reorient_for_view(np.asarray(binary_hint), view_index)
        depth, height, width = [int(v) for v in image_u8.shape[:3]]
        extent_started = time.monotonic()
        prompt_idx, lo, hi = _propagation_window(image_u8, hint, depth)
        extent_elapsed = time.monotonic() - extent_started
        forward = list(range(prompt_idx, hi + 1))
        backward = list(range(prompt_idx, lo - 1, -1))
        if full_slice:
            prompt_mask = np.ones((height, width), dtype=np.uint8)
        elif hint is not None and np.any(hint > 0):
            prompt_mask = (hint[prompt_idx] > 0).astype(np.uint8)
        else:
            prompt_mask = np.ones((height, width), dtype=np.uint8)
        view_dirs = [{
            "image_u8": image_u8,
            "indices": forward,
            "prompt_mask": prompt_mask,
            "height": height,
            "width": width,
            "full_slice": full_slice,
        }]
        if len(backward) > 1:
            view_dirs.append({
                "image_u8": image_u8,
                "indices": backward,
                "prompt_mask": prompt_mask,
                "height": height,
                "width": width,
                "full_slice": full_slice,
            })
        view_chunks = _count_propagation_chunks(view_dirs)
        _log_inference(
            f"  View {view_index + 1}/{ENSEMBLE_VIEWS} ({view_name}): "
            f"reoriented shape=({depth}, {height}, {width}), "
            f"prompt slice={prompt_idx}, range=[{lo}, {hi}], "
            f"forward={len(forward)} slices, backward={max(0, len(backward) - 1)} slices, "
            f"{view_chunks} chunks "
            f"(extent {extent_elapsed:.1f}s, total {time.monotonic() - view_started:.1f}s)",
            verbose=verbose,
        )
        owners.append((view_index, len(view_dirs)))
        directions.extend(view_dirs)

    progress = _new_progress_tracker(_count_propagation_chunks(directions))
    trainer.model.eval()
    with torch.no_grad():
        volumes = propagate_directions(
            trainer, directions, resolution, device, max_batch=max_batch,
            verbose=verbose, progress=progress,
        )
    _log_inference("Combining views with majority vote...", verbose=verbose)
    vote_array = np.zeros(shape, dtype=np.uint8)
    cursor = 0
    for view_index, count in owners:
        pred = volumes[cursor]
        if count > 1:
            pred = np.logical_or(pred, volumes[cursor + 1])
        cursor += count
        vote_array += inverse_reorient_for_view(pred, view_index).astype(np.uint8)
    result = vote_array >= int(votes_required)
    positive = int(np.sum(result))
    _log_inference(
        f"Trainer ensemble complete: {positive:,} positive voxels "
        f"({100.0 * positive / max(1, int(np.prod(shape))):.2f}% of volume)",
        verbose=verbose,
    )
    return result


def predict_binary_volume(trainer, image, binary, device, resolution, prompt=None, max_batch=None):
    """Full 3-view ensemble prediction when GT is available (validation figures)."""
    return predict_volume_ensemble(
        trainer,
        image,
        device,
        resolution=resolution,
        prompt=prompt,
        binary_hint=binary,
        max_batch=max_batch,
    )


def predict_full_slice_volume(
    trainer,
    image,
    device,
    resolution=DEFAULT_RESOLUTION,
    binary_hint=None,
    *,
    verbose: bool = False,
):
    """No-prompt inference: 3-view ensemble with tissue-bounded propagation."""
    from ..aiModels.prompts import FULL_SLICE_TOKENS

    token = next(iter(FULL_SLICE_TOKENS))
    return predict_volume_ensemble(
        trainer,
        image,
        device,
        resolution=resolution,
        prompt=token,
        binary_hint=binary_hint,
        verbose=verbose,
    )


@contextmanager
def medsam2_trainer_context(checkpoint_path: Optional[str] = None, config_path: Optional[str] = None):
    import torch
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    original_cwd = os.getcwd()
    hydra_cm = None
    try:
        os.chdir(MEDSAM2_DIR)
        if MEDSAM2_DIR not in sys.path:
            sys.path.insert(0, MEDSAM2_DIR)
        GlobalHydra.instance().clear()
        config_dir = os.path.join(MEDSAM2_DIR, "configs")
        hydra_cm = initialize_config_dir(version_base=None, config_dir=config_dir)
        hydra_cm.__enter__()
        from sam2.sam2_video_trainer import SAM2VideoTrainer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config_name = os.path.basename(config_path or "sam2.1_hiera_t512.yaml")
        from ..aiModels.foundation_models import MEDSAM2_ID, require_installed
        foundation_ckpt = str(require_installed(MEDSAM2_ID))
        trainer = SAM2VideoTrainer(config_name, foundation_ckpt, device=device)
        if checkpoint_path and os.path.isfile(checkpoint_path):
            payload = torch.load(checkpoint_path, map_location=device)
            state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
            trainer.model.load_state_dict(state)
        trainer.model.eval()
        yield trainer, device
    finally:
        if hydra_cm is not None:
            hydra_cm.__exit__(None, None, None)
        os.chdir(original_cwd)


def run_trainer_volume_inference(
    image_3d,
    checkpoint_path: str,
    config_path: Optional[str] = None,
    resolution: int = DEFAULT_RESOLUTION,
    *,
    prompt=None,
    binary_hint=None,
    verbose: bool = True,
) -> np.ndarray:
    """Run fine-tuned inference with the trainer 3-view ensemble."""
    from ..aiModels.prompts import is_full_slice_prompt

    shape = tuple(int(v) for v in np.asarray(image_3d).shape[:3])
    prompt_mode = prompt if prompt else ("mask_hint" if binary_hint is not None else "default")
    _log_inference(
        f"Trainer volume inference starting: shape={shape}, prompt={prompt_mode!r}, "
        f"checkpoint={checkpoint_path}",
        verbose=verbose,
    )
    started = time.monotonic()
    with medsam2_trainer_context(checkpoint_path, config_path) as (trainer, device):
        _log_inference(f"Trainer ready on {device}", verbose=verbose)
        if is_full_slice_prompt(prompt):
            pred = predict_full_slice_volume(
                trainer,
                image_3d,
                device,
                resolution=resolution,
                binary_hint=binary_hint,
                verbose=verbose,
            )
        else:
            pred = predict_volume_ensemble(
                trainer,
                image_3d,
                device,
                resolution=resolution,
                prompt=prompt,
                binary_hint=binary_hint,
                verbose=verbose,
            )
    elapsed = time.monotonic() - started
    _log_inference(f"Trainer volume inference finished in {elapsed:.1f}s", verbose=verbose)
    return pred.astype(np.uint8)


def run_trainer_full_slice_inference(
    image_3d,
    checkpoint_path: str,
    config_path: Optional[str] = None,
    resolution: int = DEFAULT_RESOLUTION,
    binary_hint=None,
) -> np.ndarray:
    """Backward-compatible alias for no-prompt fine-tuned inference."""
    return run_trainer_volume_inference(
        image_3d,
        checkpoint_path,
        config_path=config_path,
        resolution=resolution,
        prompt="full_slice_ones",
        binary_hint=binary_hint,
    )


def build_full_slice_reference_mask(image_shape, prompt_z: int) -> np.ndarray:
    """Ones mask on the prompt slice only (foundation MedSAM2Segmenter fallback)."""
    depth, height, width = [int(v) for v in image_shape[:3]]
    prompt_z = int(max(0, min(prompt_z, depth - 1)))
    mask = np.zeros((depth, height, width), dtype=np.uint8)
    mask[prompt_z] = 1
    return mask
