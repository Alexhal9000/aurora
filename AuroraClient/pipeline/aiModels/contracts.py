"""Minimal AI task contract shared by foundation and fine-tuned models.

Two axes define what a model is allowed to do:

- dimensionality: ``2D`` or ``3D``
- paradigm: ``prompt_guided`` or ``direct``

Do not store redundant booleans that are implied by these axes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


DIMENSIONALITY_2D = "2D"
DIMENSIONALITY_3D = "3D"
PARADIGM_PROMPT_GUIDED = "prompt_guided"
PARADIGM_DIRECT = "direct"
FOUNDATION_MEDSAM2 = "MedSAM2"

SUPPORTED_DIMENSIONALITIES = (DIMENSIONALITY_2D, DIMENSIONALITY_3D)
SUPPORTED_PARADIGMS = (PARADIGM_PROMPT_GUIDED, PARADIGM_DIRECT)

PRESET_DECODER_FOCUSED = "decoder_focused"
PRESET_ENCODER_DECODER = "encoder_decoder"
PRESET_FULL = "full"

DEFAULT_TRAINABLE_PRESETS = (
    {
        "id": PRESET_DECODER_FOCUSED,
        "label": "Decoder-focused",
        "description": "Only the last stage of the model (the part that draws the mask) is updated. Fastest, uses the least memory, and is usually the safest starting point when you have a modest number of scans.",
    },
    {
        "id": PRESET_ENCODER_DECODER,
        "label": "Encoder + decoder",
        "description": "Updates the mask-drawing stage plus the later image-understanding layers, but leaves the earliest image encoder frozen. A middle ground: more flexible than decoder-only, less likely to overwrite the original model than full training.",
    },
    {
        "id": PRESET_FULL,
        "label": "Full fine-tuning",
        "description": "Every part of the model is updated. Most powerful, but slowest, hungriest for GPU memory, and more likely to overfit if you have few subjects.",
    },
)


def medsam2_train_capabilities() -> Dict[str, Any]:
    """Capabilities advertised by the bundled MedSAM2 adapter.

    Values come from the installed inference path and the fine-tune YAML
    defaults, adapted to a single-GPU Aurora workstation (batch size 1).
    """
    return {
        "dimensionality": DIMENSIONALITY_3D,
        "paradigm": PARADIGM_PROMPT_GUIDED,
        "supported_dimensionalities": [DIMENSIONALITY_3D],
        "supported_paradigms": [PARADIGM_PROMPT_GUIDED],
        "supports_direct": False,
        "direct_unavailable_reason": (
            "MedSAM2 in Aurora is a prompt-guided 3D-as-video model. "
            "Empty-prompt / direct segmentation is not supported by the installed architecture."
        ),
        "supports_2d_task": False,
        "two_d_unavailable_reason": (
            "Aurora's MedSAM2 production path is 3D (slice-video with optional 3-view ensemble). "
            "A 2D task contract is not exposed."
        ),
        "supports_cross_validation": True,
        "min_subjects_train_val_test": 3,
        "min_subjects_per_fold": 1,
        "default_split": {"train": 70, "validation": 15, "test": 15},
        "default_seed": 123,
        "trainable_presets": list(DEFAULT_TRAINABLE_PRESETS),
        "prompt_strategies": [
            {
                "id": "middle_slab_random_gt",
                "prompt_type": "mask_and_box",
                "initialization": "middle_slab_random",
                "label": "Random slice from the middle of the structure, then propagate",
                "description": (
                    "For each subject, Aurora finds the block of slices that actually contain "
                    "the chosen structure, keeps the middle half of that block, and picks one "
                    "random slice from it as the starting hint (using your drawn mask and a box "
                    "around it). The model then learns to continue onto neighbouring slices. "
                    "A different slice from that middle block can be chosen on each training "
                    "pass so the model does not always start from the exact same view. "
                    "Quality checks use a fixed slice from the centre of that block so scores stay comparable."
                ),
            },
            {
                "id": "full_slice_ones",
                "prompt_type": "full_slice_ones",
                "initialization": "full_slice_ones",
                "label": "Whole slice as the prompt (no organ-shaped hint)",
                "description": (
                    "True empty-prompt / direct segmentation is not how MedSAM2 is built: it always "
                    "expects a first-frame hint. This mode is the closest Aurora can do. The hint is "
                    "a box around the entire slice plus a mask of all ones — not your drawn organ. "
                    "The model still has to learn which structure to output from the image and from "
                    "your labels. At inference the same whole-slice ones mask is used, so you do not "
                    "need an intersecting-plane sketch of that organ. Best when you fine-tune to a "
                    "single structure. Neighbouring-slice continuation is still learned from your drawings."
                ),
            },
        ],
        "hyperparameters": {
            "learning_rate": {
                "label": "Learning rate",
                "description": "How big a step the model takes when it updates itself. Smaller is slower but usually safer; larger can learn faster but may become unstable.",
                "default": 5.0e-5,
                "min": 1.0e-7,
                "max": 1.0e-2,
                "source": "sam2.1_hiera_tiny_finetune512.yaml scratch.base_lr",
            },
            "vision_learning_rate": {
                "label": "Image-encoder learning rate",
                "description": "A separate, usually smaller step size for the part of the model that reads the image. Keeping this lower helps preserve what the original model already knows about anatomy.",
                "default": 3.0e-5,
                "min": 1.0e-7,
                "max": 1.0e-2,
                "source": "sam2.1_hiera_tiny_finetune512.yaml scratch.vision_lr",
            },
            "batch_size": {
                "label": "Max batch size",
                "description": "Maximum clips processed together on the GPU in one step. With auto batch size enabled, Aurora probes up to this limit based on free VRAM. Turn auto off to use this value exactly.",
                "default": 12,
                "min": 1,
                "max": 12,
                "source": "Aurora single-GPU adaptation of YAML train_video_batch_size=8",
            },
            "auto_batch_size": {
                "label": "Auto batch size",
                "description": "When enabled, Aurora probes your GPU at the start of each fold and picks the largest safe batch up to the max batch size. This improves GPU utilization without manual tuning.",
                "default": True,
                "source": "Aurora default",
            },
            "epochs": {
                "label": "Epochs",
                "description": "How many times the model will see the full training set. More epochs can improve results up to a point, then the model may start memorising your scans instead of generalising.",
                "default": 75,
                "min": 1,
                "max": 500,
                "source": "sam2.1_hiera_tiny_finetune512.yaml scratch.num_epochs",
            },
            "optimizer": {
                "label": "Optimizer",
                "description": "The rule used to update the model weights. AdamW is the method recommended by the MedSAM2 training recipe.",
                "default": "AdamW",
                "options": ["AdamW"],
                "source": "sam2.1_hiera_tiny_finetune512.yaml",
            },
            "weight_decay": {
                "label": "Weight decay",
                "description": "A gentle penalty that keeps weights from growing too large. This reduces overfitting — the model is less likely to memorise your particular scans.",
                "default": 0.1,
                "min": 0.0,
                "max": 1.0,
                "source": "sam2.1_hiera_tiny_finetune512.yaml",
            },
            "validation_frequency": {
                "label": "Validation every N epochs",
                "description": "How often Aurora pauses to score validation. Fast mode scores 8-slice clips (loss and Dice). Intermediate and Slow add 3-view majority-vote volume Dice after that clip loss. 1 means check after every epoch.",
                "default": 1,
                "min": 1,
                "max": 50,
                "source": "Aurora default",
            },
            "validation_mode": {
                "label": "Validation speed",
                "description": "Fast scores 2D clips only. Intermediate adds full-volume Dice on one validation subject after 3-view (axial / coronal / sagittal) majority vote. Slow averages that same 3-view volume Dice over every validation scan. Checkpoint selection still follows val loss first; Dice is only used on a loss plateau.",
                "default": "fast_2d",
                "options": ["fast_2d", "intermediate_one_volume", "slow_all_volumes"],
                "source": "Aurora default",
            },
            "seed": {
                "label": "Random seed",
                "description": "A starting number for all random choices (which subjects go where, which middle-slab slice is used, flips, and so on). The same seed with the same data gives the same experiment.",
                "default": 123,
                "source": "sam2.1_hiera_tiny_finetune512.yaml trainer.seed_value",
            },
            "early_stopping_patience": {
                "label": "Early-stopping patience",
                "description": "Stop if validation loss does not improve for this many checks. A higher Dice at a worse loss does not count as improvement. In Intermediate/Slow the Dice that can break a loss tie is 3-view majority-vote volume Dice. 0 means never stop early.",
                "default": 10,
                "min": 0,
                "max": 100,
                "source": "Aurora default (0 disables)",
            },
            "num_frames": {
                "label": "Slices per clip",
                "description": "How many neighbouring slices are shown together as one training clip, starting from the prompt slice. More slices teach longer-range continuation but need more GPU memory.",
                "default": 8,
                "min": 2,
                "max": 16,
                "source": "sam2.1_hiera_tiny_finetune512.yaml scratch.num_frames",
            },
            "resolution": {
                "label": "Training resolution",
                "description": "Each slice is resized to this square size (pixels) before training. 512 matches the bundled MedSAM2 model.",
                "default": 512,
                "options": [512],
                "source": "sam2.1_hiera_t512.yaml / fine-tune YAML",
            },
            "grad_clip_max_norm": {
                "label": "Gradient clip",
                "description": "Caps how large a single update can be so a noisy batch cannot suddenly wreck the model. Leave at the default unless training becomes unstable.",
                "default": 0.1,
                "min": 0.0,
                "max": 10.0,
                "source": "sam2.1_hiera_tiny_finetune512.yaml",
            },
            "amp": {
                "label": "Mixed precision",
                "description": "Uses lower-precision arithmetic on the GPU when available. This is faster and uses less memory, with little effect on quality for this model.",
                "default": True,
                "source": "sam2.1_hiera_tiny_finetune512.yaml optim.amp",
            },
            "scheduler": {
                "label": "Learning-rate schedule",
                "description": "Cosine slowly reduces the learning rate as training proceeds, which usually helps the model settle. “None” keeps the learning rate fixed.",
                "default": "cosine",
                "options": ["cosine", "none"],
                "source": "sam2.1_hiera_tiny_finetune512.yaml CosineParamScheduler",
            },
            "num_workers": {
                "label": "Data-loading workers",
                "description": "Extra CPU processes that prepare slices while the GPU trains. 0 is safest. Increase only if you see the GPU sitting idle between batches.",
                "default": 0,
                "min": 0,
                "max": 8,
                "source": "Aurora default (YAML uses 15; workstation default is 0)",
            },
        },
        "augmentations": {
            "flip": {
                "id": "flip",
                "label": "Horizontal flip",
                "description": "Randomly mirrors the slice left-to-right. The mask is flipped the same way so it stays aligned. Helps the model ignore which side of the image the structure sits on.",
                "default": True,
                "consistent": True,
                "applies_to": "image_and_mask",
            },
            "affine": {
                "id": "affine",
                "label": "Random rotation",
                "description": "Slightly rotates the slice (and the mask by the same amount). Teaches the model to handle small differences in how a head or specimen was placed in the scanner.",
                "default": True,
                "consistent": True,
                "applies_to": "image_and_mask",
                "degrees": 25,
            },
            "brightness": {
                "id": "brightness",
                "label": "Brightness jitter",
                "description": "Makes the image a little brighter or darker. The mask is not changed. Helps when scans were acquired with different exposure or contrast settings.",
                "default": True,
                "consistent": False,
                "applies_to": "image_only",
                "amount": 0.1,
            },
            "contrast": {
                "id": "contrast",
                "label": "Contrast jitter",
                "description": "Slightly increases or decreases image contrast. The mask is not changed. Useful when some scans look flatter or punchier than others.",
                "default": True,
                "consistent": False,
                "applies_to": "image_only",
                "amount": 0.05,
            },
        },
        "metrics": ["dice", "iou"],
        "optional_metrics": ["hd95"],
        "checkpoint_selection": {
            "default": "val_loss_gated_dice",
            "options": ["val_loss_gated_dice"],
        },
        "loss_weights": {
            "loss_mask": 20.0,
            "loss_dice": 1.0,
            "loss_iou": 1.0,
            "source": "sam2.1_hiera_tiny_finetune512.yaml loss.weight_dict",
        },
    }


def model_descriptor(
    *,
    model_id: str,
    display_name: str,
    foundation_model: str,
    dimensionality: str,
    paradigm: str,
    is_foundation: bool,
    checkpoint_path: Optional[str] = None,
    config_path: Optional[str] = None,
    capabilities: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    label_ids: Optional[List[int]] = None,
    label_names: Optional[List[str]] = None,
    prompt_initialization: Optional[str] = None,
) -> Dict[str, Any]:
    """JSON-serializable model listing used by GET /get-ai-models/."""
    payload = {
        "id": model_id,
        "display_name": display_name,
        "foundation_model": foundation_model,
        "dimensionality": dimensionality,
        "paradigm": paradigm,
        "is_foundation": bool(is_foundation),
    }
    if checkpoint_path:
        payload["checkpoint_path"] = checkpoint_path
    if config_path:
        payload["config_path"] = config_path
    if capabilities is not None:
        payload["capabilities"] = capabilities
    if metadata is not None:
        payload["metadata"] = metadata
        stored_prompt = (metadata.get("prompt") or {}).get("initialization")
        if stored_prompt and not prompt_initialization:
            prompt_initialization = stored_prompt
    if prompt_initialization:
        payload["prompt_initialization"] = prompt_initialization
    if label_ids is not None:
        payload["label_ids"] = list(label_ids)
    if label_names is not None:
        payload["label_names"] = list(label_names)
    return payload
