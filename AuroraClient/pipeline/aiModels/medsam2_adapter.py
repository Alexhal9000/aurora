"""MedSAM2 foundation (and fine-tuned) adapter.

Registered fine-tuned checkpoints use the same fast ``MedSAM2Segmenter`` path as
foundation MedSAM2. Training/validation figures still use the trainer ensemble in
``aiFineTuning.volume_inference`` for score parity during fine-tuning.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Optional

import numpy as np

from .contracts import (
    DIMENSIONALITY_3D,
    FOUNDATION_MEDSAM2,
    PARADIGM_PROMPT_GUIDED,
    PRESET_DECODER_FOCUSED,
    PRESET_ENCODER_DECODER,
    PRESET_FULL,
    medsam2_train_capabilities,
    model_descriptor,
)
from .foundation_models import MEDSAM2_ID, ModelMissingError, require_installed, resolve_model_path
from .prompts import is_full_slice_prompt

MEDSAM2_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "MedSAM2")
)
DEFAULT_CONFIG = "configs/sam2.1_hiera_t512.yaml"


def default_checkpoint() -> str:
    return str(resolve_model_path(MEDSAM2_ID))


# Kept as a name for older imports; resolve at call time via default_checkpoint().
DEFAULT_CHECKPOINT = default_checkpoint


def foundation_descriptor() -> Dict[str, Any]:
    return model_descriptor(
        model_id=FOUNDATION_MEDSAM2,
        display_name=FOUNDATION_MEDSAM2,
        foundation_model=FOUNDATION_MEDSAM2,
        dimensionality=DIMENSIONALITY_3D,
        paradigm=PARADIGM_PROMPT_GUIDED,
        is_foundation=True,
        checkpoint_path=default_checkpoint(),
        config_path=DEFAULT_CONFIG,
        capabilities=medsam2_train_capabilities(),
    )


def _is_finetuned_checkpoint(checkpoint_path: Optional[str]) -> bool:
    if not checkpoint_path:
        return False
    return os.path.normpath(os.path.abspath(checkpoint_path)) != os.path.normpath(
        os.path.abspath(default_checkpoint())
    )


def _segmentation_bounds(
    image_data_3d,
    mask_data_3d,
    target_label,
    *,
    full_slice: bool,
):
    depth, height, width = [int(v) for v in np.asarray(image_data_3d).shape[:3]]
    if full_slice:
        from ..aiFineTuning.dataset import volume_to_uint8
        from ..aiFineTuning.volume_inference import (
            build_full_slice_reference_mask,
            choose_init_slice_from_extent,
            estimate_tissue_slice_extent,
        )

        image_u8 = volume_to_uint8(image_data_3d)
        z_lo, z_hi = estimate_tissue_slice_extent(image_u8)
        prompt_z = choose_init_slice_from_extent(z_lo, z_hi)
        reference_mask = build_full_slice_reference_mask(image_data_3d.shape, prompt_z)
        return (
            (z_lo, 0, 0),
            (z_hi, height - 1, width - 1),
            reference_mask,
            not full_slice,
        )

    label_indices = np.where(mask_data_3d == target_label)
    if len(label_indices[0]) == 0:
        return None
    d_min = int(np.min(label_indices[0]))
    d_max = int(np.max(label_indices[0]))
    h_min = int(np.min(label_indices[1]))
    h_max = int(np.max(label_indices[1]))
    w_min = int(np.min(label_indices[2]))
    w_max = int(np.max(label_indices[2]))
    reference_mask = np.where(mask_data_3d == target_label, 1, 0).astype(np.uint8)
    return (
        (d_min, h_min, w_min),
        (d_max, h_max, w_max),
        reference_mask,
        True,
    )


def run_medsam2_inference(
    image_data_3d,
    mask_data_3d,
    target_label,
    checkpoint_path: Optional[str] = None,
    config_path: Optional[str] = None,
    prompt_initialization: Optional[str] = None,
):
    """Binary prompt-guided MedSAM2 inference on ``target_label``.

    Foundation and registered fine-tuned checkpoints both use
    ``MedSAM2Segmenter`` with 3-view ``propagate_in_video`` ensemble.
    """
    full_slice = is_full_slice_prompt(prompt_initialization)
    if not checkpoint_path:
        require_installed(MEDSAM2_ID)
        resolved_checkpoint = default_checkpoint()
    else:
        resolved_checkpoint = checkpoint_path
        if not os.path.isfile(resolved_checkpoint):
            raise ModelMissingError(
                MEDSAM2_ID,
                f"MedSAM2 checkpoint not found at {resolved_checkpoint}. "
                "Download it from the header menu: AI models.",
            )
        if not _is_finetuned_checkpoint(resolved_checkpoint):
            require_installed(MEDSAM2_ID)
    resolved_config = config_path or DEFAULT_CONFIG
    finetuned = _is_finetuned_checkpoint(resolved_checkpoint)

    if finetuned:
        print(
            "MedSAM2 inference: fine-tuned weights via MedSAM2Segmenter "
            "(3-view ensemble, fast propagate_in_video path)."
        )
    elif full_slice:
        print(
            "MedSAM2 inference: foundation no-prompt mode — 3-view segmenter ensemble, "
            "tissue-bounded Z, ignoring drawn label mask."
        )

    bounds = _segmentation_bounds(
        image_data_3d,
        mask_data_3d,
        target_label,
        full_slice=full_slice,
    )
    if bounds is None:
        return mask_data_3d.copy()
    bbox_min, bbox_max, reference_mask, use_largest_cc = bounds

    from ..MedSAM2.medsam2_standalone import MedSAM2Segmenter

    segmenter = MedSAM2Segmenter(
        checkpoint_path=resolved_checkpoint,
        config_path=resolved_config,
        force_cpu=False,
    )
    segmentation_mask = segmenter.segment(
        image_3d=image_data_3d,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        mask_data_3d=reference_mask,
        use_largest_cc=use_largest_cc,
        existing_mask_as_reference=True,
        ensemble=True,
    )
    result_mask = mask_data_3d.copy()
    result_mask[segmentation_mask == 1] = target_label
    return result_mask


def apply_trainable_preset(model, preset: str) -> Dict[str, bool]:
    """Set ``requires_grad`` from a high-level preset.

    Module-name prefixes come from ``sam2.modeling.sam2_base.SAM2Base``.
    Returns a map of parameter-name → trainable for tests and metadata.
    """
    if preset not in (PRESET_DECODER_FOCUSED, PRESET_ENCODER_DECODER, PRESET_FULL):
        raise ValueError(f"Unsupported trainable preset: {preset}")

    flags: Dict[str, bool] = {}
    for name, parameter in model.named_parameters():
        trainable = _parameter_trainable(name, preset)
        parameter.requires_grad = trainable
        flags[name] = trainable
    return flags


def _parameter_trainable(name: str, preset: str) -> bool:
    if preset == PRESET_FULL:
        return True

    decoder_prefixes = (
        "sam_mask_decoder",
        "sam_prompt_encoder",
        "obj_ptr_proj",
        "obj_ptr_tpos_proj",
    )
    memory_prefixes = ("memory_attention", "memory_encoder")
    neck_prefixes = ("image_encoder.neck",)

    if any(name == prefix or name.startswith(prefix + ".") for prefix in decoder_prefixes):
        return True
    if preset == PRESET_DECODER_FOCUSED:
        return False
    if any(name == prefix or name.startswith(prefix + ".") for prefix in memory_prefixes):
        return True
    if any(name == prefix or name.startswith(prefix + ".") for prefix in neck_prefixes):
        return True
    return False


def summarize_trainable_modules(flags: Dict[str, bool]) -> Dict[str, Any]:
    """Collapse per-parameter flags into module-level summaries."""
    modules = {}
    for name, trainable in flags.items():
        root = name.split(".", 1)[0]
        bucket = modules.setdefault(root, {"trainable": 0, "frozen": 0})
        if trainable:
            bucket["trainable"] += 1
        else:
            bucket["frozen"] += 1
    return modules


def iter_named_trainable(model) -> Iterable[tuple]:
    for name, parameter in model.named_parameters():
        yield name, bool(parameter.requires_grad)
