"""Segmentation metrics for held-out evaluation."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def dice_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    pred_b = np.asarray(pred).astype(bool)
    target_b = np.asarray(target).astype(bool)
    intersection = np.logical_and(pred_b, target_b).sum()
    denom = pred_b.sum() + target_b.sum()
    if denom == 0:
        return 1.0
    return float((2.0 * intersection + eps) / (denom + eps))


def iou_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    pred_b = np.asarray(pred).astype(bool)
    target_b = np.asarray(target).astype(bool)
    intersection = np.logical_and(pred_b, target_b).sum()
    union = np.logical_or(pred_b, target_b).sum()
    if union == 0:
        return 1.0
    return float((intersection + eps) / (union + eps))


def hd95_score(pred: np.ndarray, target: np.ndarray, spacing=None) -> Optional[float]:
    """95th-percentile Hausdorff distance in millimetres, or None if unavailable."""
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        return None

    pred_b = np.asarray(pred).astype(bool)
    target_b = np.asarray(target).astype(bool)
    if pred_b.sum() == 0 or target_b.sum() == 0:
        return None

    voxel_spacing = spacing if spacing is not None else (1.0, 1.0, 1.0)
    dt_pred = distance_transform_edt(~pred_b, sampling=voxel_spacing)
    dt_tgt = distance_transform_edt(~target_b, sampling=voxel_spacing)
    pred_to_tgt = dt_tgt[pred_b]
    tgt_to_pred = dt_pred[target_b]
    if pred_to_tgt.size == 0 or tgt_to_pred.size == 0:
        return None
    return float(max(np.percentile(pred_to_tgt, 95), np.percentile(tgt_to_pred, 95)))


def binary_metrics(pred: np.ndarray, target: np.ndarray, spacing=None, include_hd95: bool = True) -> Dict[str, Optional[float]]:
    result = {
        "dice": dice_score(pred, target),
        "iou": iou_score(pred, target),
    }
    if include_hd95:
        hd95 = hd95_score(pred, target, spacing=spacing)
        if hd95 is not None:
            result["hd95"] = hd95
    return result


def aggregate_metrics(rows: list) -> Dict[str, float]:
    if not rows:
        return {}
    keys = [key for key in rows[0].keys() if isinstance(rows[0][key], (int, float))]
    return {
        key: float(np.mean([row[key] for row in rows if row.get(key) is not None]))
        for key in keys
    }
