"""Checkpoint selection: lower validation loss first, Dice only when loss is not worse."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

CRITERION = "val_loss_gated_dice"
CRITERION_FOLDS = "val_loss_gated_dice_across_folds"
INFERENCE_ARTIFACT_FILENAME = "MedSAM2_finetuned.pt"
INFERENCE_CHECKPOINT_KEY = "model"
INFERENCE_ENGINE = "MedSAM2Segmenter"

# Treat losses this close as the same plateau (absolute, or 0.2% relative).
LOSS_ABS_EPS = 1e-4
LOSS_REL_EPS = 0.002


def _margin(best_loss: float, abs_eps: float = LOSS_ABS_EPS, rel_eps: float = LOSS_REL_EPS) -> float:
    return max(float(abs_eps), float(rel_eps) * abs(float(best_loss)))


def loss_is_better(
    new_loss: Optional[float],
    best_loss: Optional[float],
    *,
    abs_eps: float = LOSS_ABS_EPS,
    rel_eps: float = LOSS_REL_EPS,
) -> bool:
    if new_loss is None:
        return False
    if best_loss is None:
        return True
    return float(new_loss) < float(best_loss) - _margin(best_loss, abs_eps, rel_eps)


def loss_is_tied(
    new_loss: Optional[float],
    best_loss: Optional[float],
    *,
    abs_eps: float = LOSS_ABS_EPS,
    rel_eps: float = LOSS_REL_EPS,
) -> bool:
    if new_loss is None or best_loss is None:
        return False
    return abs(float(new_loss) - float(best_loss)) <= _margin(best_loss, abs_eps, rel_eps)


def should_save_checkpoint(
    val_loss: Optional[float],
    val_dice: Optional[float],
    best_loss: Optional[float],
    best_dice: Optional[float],
) -> bool:
    """True when this epoch should replace the saved weights.

    - A new low in validation loss always wins, even if Dice dropped.
      An earlier high Dice at a worse loss is treated as not-yet-converged.
    - If loss is on the same plateau as the current best, a higher Dice wins.
    - A higher Dice at a worse loss is ignored.
    """
    if val_loss is None:
        return False
    if best_loss is None:
        return True
    if loss_is_better(val_loss, best_loss):
        return True
    if loss_is_tied(val_loss, best_loss):
        if val_dice is None:
            return False
        if best_dice is None:
            return True
        return float(val_dice) > float(best_dice)
    return False


def fold_rank_key(result: Any) -> Tuple[int, float, float]:
    """Lower val loss first, then higher Dice. Missing loss sorts last."""
    loss = None if not isinstance(result, dict) else result.get("best_val_loss")
    dice = None if not isinstance(result, dict) else result.get("best_val_dice")
    if loss is None:
        return (1, 0.0, 0.0)
    return (0, float(loss), -float(dice if dice is not None else 0.0))


def export_inference_checkpoint(src_path: str | Path, dest_path: str | Path) -> Path:
    """Normalize fine-tuned weights for ``MedSAM2Segmenter`` fast inference."""
    import torch

    src = Path(src_path)
    dest = Path(dest_path)
    payload = torch.load(src, map_location="cpu")
    if isinstance(payload, dict) and INFERENCE_CHECKPOINT_KEY in payload:
        state = payload[INFERENCE_CHECKPOINT_KEY]
        extra = {
            key: value
            for key, value in payload.items()
            if key not in (INFERENCE_CHECKPOINT_KEY, "inference_engine")
        }
    else:
        state = payload
        extra = {}
    torch.save(
        {
            INFERENCE_CHECKPOINT_KEY: state,
            "inference_engine": INFERENCE_ENGINE,
            **extra,
        },
        dest,
    )
    return dest
