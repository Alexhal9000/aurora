"""Minimal DINOv3 loader for Aurora's DINO-Reg rigid alignment branch.

This method utilizes DINOv3, developed by Meta AI, licensed under the DINOv3
License Agreement. See ``LICENSE.md`` in this folder.
"""

from .loader import (
    DinoRegError,
    DEFAULT_CHECKPOINT,
    PATCH_SIZE,
    default_checkpoint_path,
    ensure_dino_reg_cuda,
    get_dino_model,
    get_dinov2_model,
)

__all__ = [
    "DinoRegError",
    "DEFAULT_CHECKPOINT",
    "PATCH_SIZE",
    "default_checkpoint_path",
    "ensure_dino_reg_cuda",
    "get_dino_model",
    "get_dinov2_model",
]
