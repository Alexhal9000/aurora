"""GPU batching helpers for MedSAM2 fine-tuning."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def clips_to_host_tensors(clips: Sequence[Dict[str, Any]], pin: bool = False):
    """Stack clips into CPU tensors so H2D copies can overlap compute."""
    import torch

    if not clips:
        raise ValueError("clips_to_host_tensors requires at least one clip")
    videos = np.stack([np.asarray(clip["video"]) for clip in clips], axis=0)
    masks = np.stack([np.asarray(clip["masks"]) for clip in clips], axis=0)
    bboxes = np.stack([np.asarray(clip["bbox"]) for clip in clips], axis=0)
    video_t = torch.from_numpy(np.ascontiguousarray(videos))
    masks_t = torch.from_numpy(np.ascontiguousarray(masks)).unsqueeze(2)
    bbox_t = torch.from_numpy(np.ascontiguousarray(bboxes))
    prompt_masks = [clip.get("prompt_mask") for clip in clips]
    dense_t = None
    if all(item is not None for item in prompt_masks):
        dense = np.stack([np.asarray(item, dtype=np.float32) for item in prompt_masks], axis=0)
        dense_t = torch.from_numpy(np.ascontiguousarray(dense)).unsqueeze(1)
    packed = [video_t, masks_t, bbox_t]
    if dense_t is not None:
        packed.append(dense_t)
    if pin:
        packed = [tensor.pin_memory() for tensor in packed]
    if dense_t is None:
        return packed[0], packed[1], packed[2], None
    return packed[0], packed[1], packed[2], packed[3]


def host_tensors_to_device(host_tensors, device, non_blocking: bool = True):
    video, masks, bbox, dense = host_tensors
    use_async = bool(non_blocking) and getattr(device, "type", None) == "cuda"
    video_t = video.to(device, non_blocking=use_async).contiguous()
    masks_t = masks.to(device, non_blocking=use_async).contiguous()
    bbox_t = bbox.to(device, non_blocking=use_async).contiguous()
    dense_t = None
    if dense is not None:
        dense_t = dense.to(device, non_blocking=use_async).contiguous()
    return video_t, masks_t, bbox_t, dense_t


def clips_to_batch_tensors(clips: Sequence[Dict[str, Any]], device):
    """Stack training clips into batched tensors for ``SAM2VideoTrainer``."""
    pin = getattr(device, "type", None) == "cuda"
    return host_tensors_to_device(clips_to_host_tensors(clips, pin=pin), device)


class GpuBatchPrefetcher:
    """Stage the next batch on pinned CPU memory; copy to GPU only when needed.

    Holding the next batch in VRAM while the current step runs was enough to
    OOM a 16 GB card after auto-batch probing filled most of free memory.
    """

    def __init__(self, device):
        self.device = device
        self._host = None

    def preload(self, clips: Optional[Sequence[Dict[str, Any]]]) -> None:
        if not clips:
            self._host = None
            return
        pin = getattr(self.device, "type", None) == "cuda"
        self._host = clips_to_host_tensors(clips, pin=pin)

    def next(self):
        if self._host is None:
            return None
        batch = host_tensors_to_device(self._host, self.device, non_blocking=True)
        self._host = None
        return batch


def trainer_forward(trainer, batch_tensors):
    video, masks, bbox, dense = batch_tensors
    pred_masks, pred_logits, pred_ious = trainer(video, bbox, labels=masks, dense_prompt_mask=dense)
    return pred_masks, pred_logits, pred_ious


def loss_from_outputs(pred_logits, pred_ious, masks, weights):
    """Clip objective from a trainer forward — same reduction as training."""
    import torch
    import torch.nn.functional as F

    loss_mask = 0.0
    loss_dice = 0.0
    loss_iou = 0.0
    for t, logit in enumerate(pred_logits):
        target = masks[:, t]
        if target.ndim == 3:
            target = target.unsqueeze(1)
        if logit.shape[-2:] != target.shape[-2:]:
            logit = F.interpolate(logit, size=target.shape[-2:], mode="bilinear", align_corners=False)
        loss_mask = loss_mask + F.binary_cross_entropy_with_logits(logit, target)
        pred = torch.sigmoid(logit)
        loss_dice = loss_dice + _dice_loss(pred, target)
        if pred_ious is not None:
            with torch.no_grad():
                intersection = (pred.gt(0.5) & target.gt(0.5)).float().sum(dim=(1, 2, 3))
                union = (pred.gt(0.5) | target.gt(0.5)).float().sum(dim=(1, 2, 3)).clamp_min(1.0)
                actual_iou = intersection / union
            pred_iou = pred_ious[t]
            if pred_iou.ndim > 1:
                pred_iou = pred_iou.reshape(pred_iou.shape[0], -1)[:, 0]
            loss_iou = loss_iou + F.l1_loss(pred_iou.float(), actual_iou.float())

    n = max(len(pred_logits), 1)
    return (
        float(weights.get("loss_mask", 20.0)) * loss_mask / n
        + float(weights.get("loss_dice", 1.0)) * loss_dice / n
        + float(weights.get("loss_iou", 1.0)) * loss_iou / n
    )


def _as_batch_hw(arr) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[None]
    while arr.ndim > 3:
        if arr.shape[1] == 1:
            arr = arr[:, 0]
        else:
            arr = arr.reshape(arr.shape[0], *arr.shape[-2:])
            break
    return arr


def last_frame_binary_metrics(pred_masks, target_masks):
    """Per-clip Dice/IoU on the last frame of a batched trainer output."""
    from .evaluate import binary_metrics

    last = pred_masks[-1]
    target = target_masks[:, -1]
    pred_np = _as_batch_hw((last > 0.5).detach().cpu().numpy())
    if hasattr(target, "detach"):
        tgt_np = _as_batch_hw((target > 0.5).detach().cpu().numpy())
    else:
        tgt_np = _as_batch_hw(np.asarray(target) > 0.5)
    return [
        binary_metrics(pred_np[i], tgt_np[i], include_hd95=False)
        for i in range(int(pred_np.shape[0]))
    ]


def clip_frames_for_index(pred_masks, index: int):
    """All clip frames for one batch item as a list of ``(H, W)`` bool arrays."""
    frames = []
    for mask_t in pred_masks:
        arr = (mask_t[int(index)] > 0.5).detach().cpu().numpy()
        while arr.ndim > 2:
            arr = arr[0]
        frames.append(np.asarray(arr).astype(bool))
    return frames


def first_frame_binary(pred_masks):
    """First-frame predicted masks as ``(B, H, W)`` bool."""
    first = pred_masks[0]
    return _as_batch_hw((first > 0.5).detach().cpu().numpy()).astype(bool)


def forward_loss_from_tensors(trainer, batch_tensors, weights):
    """Run a batched forward pass from GPU/CPU tensors and return the scalar loss."""
    _pred_masks, pred_logits, pred_ious = trainer_forward(trainer, batch_tensors)
    _video, masks, _bbox, _dense = batch_tensors
    return loss_from_outputs(pred_logits, pred_ious, masks, weights)


def forward_loss_from_clips(trainer, clips, device, weights):
    """Run a batched forward pass and return the scalar training loss."""
    return forward_loss_from_tensors(
        trainer, clips_to_batch_tensors(clips, device), weights,
    )


def _dice_loss(pred, target, eps=1.0):
    dims = tuple(range(1, pred.ndim))
    intersection = (pred * target).sum(dim=dims)
    denom = pred.sum(dim=dims) + target.sum(dim=dims)
    dice = (2 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def _cuda_memory_budget(device, safety_fraction: float = 0.75) -> int:
    """Usable VRAM for one training step, leaving headroom for Adam + validation."""
    import torch

    if device.type != "cuda":
        return 0
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    reserve = max(int(2.5 * 1024 ** 3), int(total_bytes * 0.15))
    usable = max(0, int(free_bytes) - reserve)
    return int(usable * float(safety_fraction))


def probe_max_batch_size(
    trainer,
    sample_clip: Dict[str, Any],
    device,
    loss_weights: Dict[str, Any],
    *,
    max_batch: int = 12,
    safety_fraction: float = 0.75,
) -> int:
    """Find the largest batch size that fits in free VRAM without OOM."""
    import torch

    max_batch = max(1, min(int(max_batch), 12))
    if device.type != "cuda" or sample_clip is None:
        return 1

    trainer_was_training = trainer.model.training
    trainer.model.train()
    candidates = [size for size in (1, 2, 3, 4, 6, 8, 10, 12) if size <= max_batch]
    best = 1

    try:
        for batch_size in candidates:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            budget = _cuda_memory_budget(device, safety_fraction=safety_fraction)
            clips = [sample_clip] * batch_size
            try:
                with torch.cuda.amp.autocast(enabled=True):
                    loss = forward_loss_from_clips(trainer, clips, device, loss_weights)
                loss.backward()
                trainer.zero_grad(set_to_none=True)
                peak = int(torch.cuda.max_memory_allocated(device))
                torch.cuda.empty_cache()
                if peak <= budget or batch_size == 1:
                    best = batch_size
                else:
                    break
            except RuntimeError as exc:
                trainer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if "out of memory" in str(exc).lower():
                    break
                raise
    finally:
        trainer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        trainer.model.train(trainer_was_training)

    return best


def resolve_training_batch_size(
    trainer,
    device,
    hyperparameters: Dict[str, Any],
    sample_clip: Optional[Dict[str, Any]],
    loss_weights: Dict[str, Any],
) -> int:
    """Resolve effective batch size (fixed or auto-probed up to the configured cap)."""
    cap = max(1, min(int(hyperparameters.get("batch_size") or 12), 12))
    auto = hyperparameters.get("auto_batch_size")
    if auto is None:
        auto = True
    if not auto:
        return cap
    if device.type != "cuda":
        return 1
    return probe_max_batch_size(
        trainer,
        sample_clip,
        device,
        loss_weights,
        max_batch=cap,
    )
