"""Lazy CUDA singleton for the DINOv3 ViT-L/16 encoder used by DINO-Reg.

This method utilizes DINOv3, developed by Meta AI, licensed under the DINOv3
License Agreement. A copy of that license is at ``LICENSE.md`` in this folder
and next to the bundled weights under ``pipeline/MedSAM2/models/``.

Weights are the Hugging Face ``facebook/dinov3-vitl16-pretrain-lvd1689m``
snapshot, stored offline under ``MedSAM2/models/dinov3-vitl16-pretrain-lvd1689m``.
"""

from __future__ import annotations

import os

from ..aiModels.foundation_models import DINOV3_ID, ModelMissingError, resolve_model_path

MEDSAM2_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "MedSAM2")
)
MODELS_DIR = os.path.join(MEDSAM2_DIR, "models")
DEFAULT_MODEL_DIR = os.path.join(MODELS_DIR, "dinov3-vitl16-pretrain-lvd1689m")
DEFAULT_CHECKPOINT = DEFAULT_MODEL_DIR  # directory; kept for call-site compatibility
PATCH_SIZE = 16
HF_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"

_MODEL = None
_DEVICE = None


class DinoRegError(RuntimeError):
    """Raised when DINO-Reg cannot run (no CUDA, missing checkpoint, load failure)."""


def default_checkpoint_path() -> str:
    """Resolved DINOv3 snapshot directory (Documents foundation, then legacy tree)."""
    return str(resolve_model_path(DINOV3_ID))


def ensure_dino_reg_cuda():
    """Fail fast if CUDA is unavailable. Does not load the model."""
    try:
        import torch
    except ImportError as exc:
        raise DinoRegError(
            "DINO-Reg requires PyTorch with CUDA. PyTorch is not installed."
        ) from exc
    if not torch.cuda.is_available():
        raise DinoRegError(
            "DINO-Reg requires a CUDA GPU. No CUDA device is available."
        )


class _Dinov3PatchEncoder:
    """HF DINOv3 backbone with DINOv2-style ``forward_features`` patch tokens."""

    def __init__(self, backbone):
        self.backbone = backbone
        self.patch_size = int(getattr(backbone.config, "patch_size", PATCH_SIZE))

    def eval(self):
        self.backbone.eval()
        return self

    def to(self, device):
        self.backbone.to(device)
        return self

    def parameters(self):
        return self.backbone.parameters()

    def forward_features(self, pixel_values):
        h = int(pixel_values.shape[-2])
        w = int(pixel_values.shape[-1])
        n_patches = (h // self.patch_size) * (w // self.patch_size)
        outputs = self.backbone(pixel_values=pixel_values)
        hidden = outputs.last_hidden_state
        prefix = int(hidden.shape[1]) - n_patches
        if prefix < 0:
            raise RuntimeError(
                f"DINOv3 sequence length {hidden.shape[1]} < expected {n_patches} patches "
                f"for {h}x{w} (patch={self.patch_size})"
            )
        return {"x_norm_patchtokens": hidden[:, prefix:, :]}


def get_dino_model(checkpoint_path: str | None = None):
    """Return ``(model, device)``, loading once per process."""
    global _MODEL, _DEVICE
    if _MODEL is not None:
        return _MODEL, _DEVICE

    ensure_dino_reg_cuda()
    import torch

    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise DinoRegError(
            "DINO-Reg requires the `transformers` package to load DINOv3. "
            "Install with: pip install 'transformers>=4.56.0'"
        ) from exc

    device = torch.device("cuda", 0)
    weights_dir = checkpoint_path or default_checkpoint_path()
    if os.path.isfile(weights_dir):
        weights_dir = os.path.dirname(weights_dir)
    config_path = os.path.join(weights_dir, "config.json")
    weight_files = (
        os.path.join(weights_dir, "model.safetensors"),
        os.path.join(weights_dir, "pytorch_model.bin"),
    )
    if not os.path.isfile(config_path) or not any(os.path.isfile(p) for p in weight_files):
        raise DinoRegError(
            "DINO-Reg requires the DINOv3 snapshot. "
            "Download it from the header menu: AI models. "
            f"Expected config.json and model.safetensors under {weights_dir}."
        ) from ModelMissingError(DINOV3_ID)

    try:
        backbone = AutoModel.from_pretrained(
            weights_dir,
            local_files_only=True,
            trust_remote_code=False,
        )
        patch = int(getattr(backbone.config, "patch_size", PATCH_SIZE))
        if patch != PATCH_SIZE:
            raise DinoRegError(
                f"Expected DINOv3 patch_size={PATCH_SIZE}, got {patch}"
            )
        model = _Dinov3PatchEncoder(backbone)
    except DinoRegError:
        raise
    except Exception as exc:
        raise DinoRegError(
            f"DINO-Reg could not load DINOv3 ({HF_MODEL_ID}) from {weights_dir}: {exc}"
        ) from exc

    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    _MODEL = model
    _DEVICE = device
    print(f"DINOv3 {HF_MODEL_ID} loaded from {weights_dir} on {device}")
    return _MODEL, _DEVICE


def get_dinov2_model(checkpoint_path: str | None = None):
    """Deprecated alias; DINOv3 is the encoder."""
    return get_dino_model(checkpoint_path)
