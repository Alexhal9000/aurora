"""Discover foundation and registered AI segmentation models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .contracts import (
    DIMENSIONALITY_3D,
    FOUNDATION_MEDSAM2,
    PARADIGM_PROMPT_GUIDED,
    medsam2_train_capabilities,
    model_descriptor,
)
from .prompts import resolve_prompt_initialization
from .foundation_models import MEDSAM2_ID, MISSING_HINT, is_installed
from .medsam2_adapter import (
    DEFAULT_CONFIG,
    default_checkpoint,
    foundation_descriptor,
    run_medsam2_inference,
)
from .paths import models_root, registered_model_dir

METADATA_FILENAME = "metadata.json"
CHECKPOINT_CANDIDATES = (
    "MedSAM2_finetuned.pt",
    "checkpoint.pt",
    "model.pt",
)


def _with_availability(descriptor: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(descriptor)
    foundation_ready = is_installed(MEDSAM2_ID)
    if payload.get("is_foundation"):
        payload["available"] = foundation_ready
        payload["unavailable_reason"] = None if foundation_ready else MISSING_HINT
        return payload
    if not foundation_ready:
        payload["available"] = False
        payload["unavailable_reason"] = (
            "This fine-tuned model needs the MedSAM2 foundation weights. " + MISSING_HINT
        )
        return payload
    payload["available"] = True
    payload["unavailable_reason"] = None
    return payload


def list_model_descriptors(include_capabilities: bool = True) -> List[Dict[str, Any]]:
    """Foundation models first, then successfully registered fine-tunes."""
    models = [foundation_descriptor() if include_capabilities else _strip_caps(foundation_descriptor())]
    models.extend(_list_registered_models(include_capabilities=include_capabilities))
    return [_with_availability(item) for item in models]


def get_model(model_id: str) -> Optional[Dict[str, Any]]:
    if not model_id:
        return None
    if str(model_id) == FOUNDATION_MEDSAM2:
        return _with_availability(foundation_descriptor())
    registered = _load_registered_descriptor(str(model_id))
    return _with_availability(registered) if registered is not None else None


def resolve_inference_model(model_id: str) -> Dict[str, Any]:
    """Return descriptor used to run inference, or raise."""
    descriptor = get_model(model_id)
    if descriptor is None:
        raise ValueError(f"Model {model_id} not supported")
    return descriptor


def run_inference(image_data_3d, mask_data_3d, target_label, model_id: str):
    descriptor = resolve_inference_model(model_id)
    foundation = descriptor.get("foundation_model") or descriptor.get("id")
    if foundation != FOUNDATION_MEDSAM2 and descriptor.get("id") != FOUNDATION_MEDSAM2:
        raise ValueError(
            f"Model {model_id} uses unsupported foundation {foundation!r} for inference"
        )
    if descriptor.get("paradigm") == "direct":
        raise ValueError(
            f"Model {model_id} is a direct-segmentation model and cannot be used "
            "in the prompt-guided intersecting-plane workflow"
        )
    checkpoint = descriptor.get("checkpoint_path")
    config = descriptor.get("config_path") or DEFAULT_CONFIG
    if descriptor.get("is_foundation"):
        checkpoint = checkpoint or default_checkpoint()
    prompt_init = resolve_prompt_initialization(descriptor)
    return run_medsam2_inference(
        image_data_3d,
        mask_data_3d,
        target_label,
        checkpoint_path=checkpoint,
        config_path=config,
        prompt_initialization=prompt_init,
    )


def _strip_caps(descriptor: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(descriptor)
    payload.pop("capabilities", None)
    return payload


def _list_registered_models(include_capabilities: bool = True) -> List[Dict[str, Any]]:
    root = models_root()
    if not root.is_dir():
        return []
    found = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name in {"runs", "foundation"}:
            continue
        descriptor = _load_registered_descriptor(child.name, include_capabilities=include_capabilities)
        if descriptor is not None:
            found.append(descriptor)
    return found


def _load_registered_descriptor(
    model_id: str,
    include_capabilities: bool = True,
) -> Optional[Dict[str, Any]]:
    meta_path = registered_model_dir(model_id) / METADATA_FILENAME
    if not meta_path.is_file():
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(metadata, dict):
        return None

    identity = metadata.get("identity") or {}
    contract = metadata.get("task_contract") or {}
    stored_id = identity.get("model_id") or model_id
    if str(stored_id) != str(model_id):
        # Directory name is the stable id; mismatch is corrupt.
        return None

    checkpoint = _resolve_registered_checkpoint(Path(meta_path).parent, metadata)
    if checkpoint is None:
        return None

    dimensionality = contract.get("dimensionality") or DIMENSIONALITY_3D
    paradigm = contract.get("paradigm") or PARADIGM_PROMPT_GUIDED
    foundation = identity.get("foundation_model") or FOUNDATION_MEDSAM2
    display_name = identity.get("display_name") or stored_id

    caps = medsam2_train_capabilities() if include_capabilities and foundation == FOUNDATION_MEDSAM2 else None
    descriptor = model_descriptor(
        model_id=str(stored_id),
        display_name=display_name,
        foundation_model=foundation,
        dimensionality=dimensionality,
        paradigm=paradigm,
        is_foundation=False,
        checkpoint_path=str(checkpoint),
        config_path=identity.get("config_path") or DEFAULT_CONFIG,
        capabilities=caps,
        metadata=metadata if include_capabilities else None,
        label_ids=contract.get("label_ids"),
        label_names=contract.get("label_names"),
    )
    return descriptor


def _resolve_registered_checkpoint(model_dir: Path, metadata: Dict[str, Any]) -> Optional[Path]:
    artifact = (metadata.get("identity") or {}).get("artifact_filename")
    candidates = []
    if artifact:
        candidates.append(model_dir / artifact)
    for name in CHECKPOINT_CANDIDATES:
        candidates.append(model_dir / name)
    for path in candidates:
        if path.is_file():
            return path
    return None
