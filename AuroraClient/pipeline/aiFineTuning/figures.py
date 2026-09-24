"""Save training-run figures (PNG) under the run directory."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from .dataset import IMAGENET_MEAN, IMAGENET_STD


def figures_dir(run_id: str) -> Path:
    from .storage import run_dir

    path = run_dir(run_id) / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_loss_curves(
    path: Path,
    history: Sequence[Dict[str, Any]],
    *,
    title: str = "Training / validation loss",
) -> Path:
    plt = _pyplot()
    epochs = [row.get("epoch") for row in history]
    train_loss = [row.get("train_loss") for row in history]
    val_loss = [row.get("val_loss") for row in history]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if any(value is not None for value in train_loss):
        ax.plot(epochs, train_loss, color="#1565c0", marker="o", label="Train loss")
    if any(value is not None for value in val_loss):
        ax.plot(epochs, val_loss, color="#ef6c00", marker="s", label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def save_dice_curves(
    path: Path,
    history: Sequence[Dict[str, Any]],
    *,
    title: str = "Training / validation Dice",
) -> Path:
    plt = _pyplot()
    epochs = [row.get("epoch") for row in history]
    train_dice = [row.get("train_dice") for row in history]
    val_dice = [
        row.get("val_dice")
        if row.get("val_dice") is not None
        else ((row.get("val_metrics") or {}).get("dice") if isinstance(row.get("val_metrics"), dict) else None)
        for row in history
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if any(value is not None for value in train_dice):
        ax.plot(epochs, train_dice, color="#2e7d32", marker="o", label="Train Dice (2D clips)")
    if any(value is not None for value in val_dice):
        ax.plot(epochs, val_dice, color="#c62828", marker="s", label="Val Dice (2D clips)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def save_training_curves(
    path: Path,
    history: Sequence[Dict[str, Any]],
    *,
    title: str = "Training / validation",
) -> Path:
    """Deprecated combined plot — prefer ``save_loss_curves`` + ``save_dice_curves``."""
    plt = _pyplot()
    epochs = [row.get("epoch") for row in history]
    train_loss = [row.get("train_loss") for row in history]
    val_dice = [
        row.get("val_dice")
        if row.get("val_dice") is not None
        else ((row.get("val_metrics") or {}).get("dice") if isinstance(row.get("val_metrics"), dict) else None)
        for row in history
    ]

    fig, ax_loss = plt.subplots(figsize=(8, 4.5))
    ax_dice = ax_loss.twinx()
    if any(value is not None for value in train_loss):
        ax_loss.plot(epochs, train_loss, color="#1565c0", marker="o", label="Train loss")
    if any(value is not None for value in val_dice):
        ax_dice.plot(epochs, val_dice, color="#c62828", marker="s", label="Val Dice")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Train loss")
    ax_dice.set_ylabel("Validation Dice")
    ax_dice.set_ylim(0.0, 1.0)
    ax_loss.set_title(title)
    handles_l, labels_l = ax_loss.get_legend_handles_labels()
    handles_d, labels_d = ax_dice.get_legend_handles_labels()
    if handles_l or handles_d:
        ax_loss.legend(handles_l + handles_d, labels_l + labels_d, loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def save_fold_val_summary(path: Path, fold_rows: Sequence[Dict[str, Any]]) -> Path:
    plt = _pyplot()
    labels = [f"Fold {int(row.get('fold', i)) + 1}" for i, row in enumerate(fold_rows)]
    losses = [row.get("best_val_loss") for row in fold_rows]
    use_loss = any(value is not None for value in losses)
    values = [
        float(value) if value is not None else 0.0
        for value in (losses if use_loss else [row.get("best_val_dice") for row in fold_rows])
    ]
    colors = ["#2e7d32" if row.get("selected") else "#90caf9" for row in fold_rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values, color=colors)
    if not use_loss:
        ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Best validation loss" if use_loss else "Best validation Dice")
    ax.set_title(
        "K-fold checkpoint selection (lowest val loss wins; green = registered)"
        if use_loss else "K-fold checkpoint selection (green = registered fold)"
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def save_before_after_bars(path: Path, rows: Sequence[Dict[str, Any]]) -> Path:
    plt = _pyplot()
    names = [
        f"{row.get('subject') or '?'}"
        + (f" L{row['label_id']}" if row.get("label_id") is not None else "")
        for row in rows
    ]
    foundation = [float(row.get("foundation_dice") or 0.0) for row in rows]
    finetuned = [float(row.get("finetuned_dice") or 0.0) for row in rows]
    x = np.arange(len(names))
    width = 0.38
    fig_w = max(8.0, 0.55 * max(len(names), 1))
    fig, ax = plt.subplots(figsize=(fig_w, 4.5))
    ax.bar(x - width / 2, foundation, width, label="Foundation MedSAM2", color="#90a4ae")
    ax.bar(x + width / 2, finetuned, width, label="Fine-tuned", color="#1565c0")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Dice")
    ax.set_title("Validation Dice before and after fine-tuning")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


MAGENTA = np.array([1.0, 0.0, 1.0], dtype=np.float32)
CYAN = np.array([0.0, 1.0, 1.0], dtype=np.float32)
OVERLAP_BLUE = np.array([0.05, 0.35, 1.0], dtype=np.float32)
VIEW_NAMES = ("axial", "coronal", "sagittal")


def even_span_indices(lo: int, hi: int, n: int = 10) -> List[int]:
    """Inclusive, evenly spaced indices through ``[lo, hi]``, up to ``n`` samples."""
    lo = int(lo)
    hi = int(hi)
    if hi < lo:
        lo, hi = hi, lo
    span = hi - lo + 1
    if span <= 0:
        return [lo]
    if n <= 1:
        return [int(round((lo + hi) / 2))]
    if span <= n:
        return list(range(lo, hi + 1))
    return [int(round(lo + i * (hi - lo) / float(n - 1))) for i in range(n)]


def union_bbox(masks: Sequence[np.ndarray]) -> List[tuple]:
    union = None
    for mask in masks:
        if mask is None:
            continue
        bit = np.asarray(mask).astype(bool)
        union = bit if union is None else np.logical_or(union, bit)
    if union is None or not union.any():
        return [(0, 0) for _ in range(3)]
    bounds = []
    for axis, coords in enumerate(np.where(union)):
        bounds.append((int(coords.min()), int(coords.max())))
    return bounds


def axis_indices_for_volume(
    shape,
    masks: Sequence[np.ndarray],
    n: int = 10,
) -> Dict[int, List[int]]:
    """10 (or fewer) slices per axis, evenly through the union bounding box."""
    bounds = union_bbox(masks)
    picked = {}
    for axis, (lo, hi) in enumerate(bounds):
        length = int(shape[axis])
        lo = max(0, min(lo, length - 1))
        hi = max(0, min(hi, length - 1))
        picked[axis] = even_span_indices(lo, hi, n)
    return picked


def take_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        return np.asarray(volume[index])
    if axis == 1:
        return np.asarray(volume[:, index, :])
    return np.asarray(volume[:, :, index])


def _to_gray(slice2d: np.ndarray) -> np.ndarray:
    data = np.asarray(slice2d, dtype=np.float32)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros(data.shape, dtype=np.float32)
    lo, hi = np.percentile(finite, (1.0, 99.0))
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((data - lo) / (hi - lo), 0.0, 1.0)


def overlay_magenta_cyan(
    image2d: np.ndarray,
    magenta_mask: np.ndarray,
    cyan_mask: np.ndarray,
    *,
    alpha: float = 0.55,
) -> np.ndarray:
    """Anatomy in gray; magenta-only, cyan-only, overlap in blue."""
    gray = _to_gray(image2d)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    mag = np.asarray(magenta_mask).astype(bool)
    cyan = np.asarray(cyan_mask).astype(bool)
    if mag.shape != gray.shape:
        mag = np.zeros(gray.shape, dtype=bool)
    if cyan.shape != gray.shape:
        cyan = np.zeros(gray.shape, dtype=bool)
    only_m = mag & ~cyan
    only_c = cyan & ~mag
    both = mag & cyan
    rgb[only_m] = (1.0 - alpha) * rgb[only_m] + alpha * MAGENTA
    rgb[only_c] = (1.0 - alpha) * rgb[only_c] + alpha * CYAN
    rgb[both] = (1.0 - alpha) * rgb[both] + alpha * OVERLAP_BLUE
    return np.clip(rgb, 0.0, 1.0)


def _safe_stem(subject, label_id) -> str:
    raw = f"{subject or 'subject'}_label{label_id}"
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in raw)


def save_validation_multiview(
    path: Path,
    image: np.ndarray,
    foundation_mask: np.ndarray,
    finetuned_mask: np.ndarray,
    *,
    subject: str = "",
    label_id=None,
    gt_mask=None,
    n_slices: int = 10,
    dpi: int = 300,
    min_panel_px: int = 512,
    title: str = "",
) -> Path:
    """High-res 3-view grid: 10 slices × axial / coronal / sagittal.

    Each panel is native resolution (or at least ``min_panel_px``) so zooming
    the PNG does not pixelate the anatomy. Magenta = foundation, cyan =
    fine-tuned, blue = overlap.
    """
    plt = _pyplot()
    image = np.asarray(image)
    foundation_mask = np.asarray(foundation_mask).astype(bool)
    finetuned_mask = np.asarray(finetuned_mask).astype(bool)
    bbox_masks = [foundation_mask, finetuned_mask]
    if gt_mask is not None:
        bbox_masks.append(np.asarray(gt_mask).astype(bool))
    picked = axis_indices_for_volume(image.shape, bbox_masks, n=n_slices)
    max_native = int(max(image.shape))
    panel_px = max(int(min_panel_px), max_native)
    n_cols = max(len(picked[axis]) for axis in range(3))
    fig_w = (n_cols * panel_px) / float(dpi)
    fig_h = (3 * panel_px) / float(dpi) + 1.15
    fig, axes = plt.subplots(3, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    axis_labels = ("z", "y", "x")
    for row, axis in enumerate(range(3)):
        indices = picked[axis]
        for col in range(n_cols):
            ax = axes[row][col]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if col >= len(indices):
                continue
            index = indices[col]
            panel = overlay_magenta_cyan(
                take_slice(image, axis, index),
                take_slice(foundation_mask, axis, index),
                take_slice(finetuned_mask, axis, index),
            )
            ax.imshow(panel, interpolation="nearest", aspect="equal")
            ax.set_title(f"{axis_labels[axis]}={index}", fontsize=8, pad=2)
        axes[row][0].set_ylabel(VIEW_NAMES[axis], fontsize=11)
    heading = title or (
        f"{subject}  label {label_id}  ·  magenta = foundation MedSAM2  ·  "
        "cyan = fine-tuned  ·  blue = overlap"
    )
    fig.suptitle(heading, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return path


def save_validation_subject_grids(
    directory: Path,
    volumes: Sequence[Dict[str, Any]],
    **kwargs,
) -> List[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for row in volumes:
        stem = _safe_stem(row.get("subject"), row.get("label_id"))
        path = directory / f"{stem}_foundation_vs_finetuned.png"
        written.append(save_validation_multiview(
            path,
            row["image"],
            row["foundation_pred"],
            row["finetuned_pred"],
            subject=row.get("subject") or "",
            label_id=row.get("label_id"),
            gt_mask=row.get("gt"),
            **kwargs,
        ))
    return written


def display_frame_from_clip(clip: Dict[str, Any]) -> np.ndarray:
    """Undo ImageNet normalisation on clip frame 0 for overlay figures."""
    frame = np.asarray(clip["video"][0], dtype=np.float32)
    rgb = frame * IMAGENET_STD + IMAGENET_MEAN
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.transpose(rgb, (1, 2, 0))
