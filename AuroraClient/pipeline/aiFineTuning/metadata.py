"""Build complete registered-model metadata and default display names."""

from __future__ import annotations

import os
import platform
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from ..aiModels.contracts import FOUNDATION_MEDSAM2
from ..aiFineTuning.checkpoint import INFERENCE_ARTIFACT_FILENAME, INFERENCE_ENGINE
from ..aiModels.medsam2_adapter import DEFAULT_CONFIG, default_checkpoint


def default_display_name(project_name: str, label_names: Sequence[str], foundation: str = FOUNDATION_MEDSAM2) -> str:
    roi = " + ".join(name for name in label_names if name)
    if not roi:
        roi = "custom"
    project = project_name or "Project"
    return f"{foundation} — {project} — {roi}"


def read_aurora_version() -> Optional[str]:
    system = sys.platform
    version_paths = []
    if system == "linux":
        version_paths = ["/opt/aurora-tools/version.txt"]
    elif system == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            version_paths.append(os.path.join(local_app_data, "Aurora", "version.txt"))
        version_paths.append(os.path.join(os.path.expanduser("~"), "AppData", "Local", "Aurora", "version.txt"))
    elif system == "darwin":
        version_paths = ["/Library/Application Support/Aurora/version.txt"]
    for path in version_paths:
        if path and os.path.isfile(path):
            try:
                value = open(path, "r", encoding="utf-8").read().strip()
                if value:
                    return value
            except OSError:
                continue
    return None


def library_versions() -> Dict[str, Any]:
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for module_name in ("torch", "nibabel", "numpy"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", None)
        except Exception:
            versions[module_name] = None
    try:
        import torch
        versions["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            versions["cuda"] = torch.version.cuda
            versions["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        versions["cuda_available"] = False
    return versions


def build_model_metadata(
    *,
    model_id: str,
    display_name: str,
    run_id: str,
    config: Dict[str, Any],
    evaluation: Optional[Dict[str, Any]] = None,
    selected_checkpoint: Optional[Dict[str, Any]] = None,
    architectural_modifications: Optional[Any] = None,
) -> Dict[str, Any]:
    labels = config.get("labels") or []
    label_ids = [int(item["id"]) for item in labels]
    label_names = [item.get("name") or f"Label {item['id']}" for item in labels]
    prompt = config.get("prompt") or {}
    return {
        "identity": {
            "model_id": model_id,
            "display_name": display_name,
            "foundation_model": config.get("foundation_model") or FOUNDATION_MEDSAM2,
            "foundation_model_version": "MedSAM2_latest",
            "base_checkpoint": config.get("base_checkpoint") or default_checkpoint(),
            "config_path": config.get("config_path") or DEFAULT_CONFIG,
            "artifact_filename": INFERENCE_ARTIFACT_FILENAME,
            "artifact_format": "torch_state_dict",
            "artifact_format_version": 1,
        },
        "inference": {
            "engine": INFERENCE_ENGINE,
            "ensemble_views": 3,
            "votes_required": 2,
            "artifact_filename": INFERENCE_ARTIFACT_FILENAME,
            "checkpoint_key": "model",
        },
        "task_contract": {
            "dimensionality": config.get("dimensionality") or "3D",
            "paradigm": config.get("paradigm") or "prompt_guided",
            "single_or_multi_label": "multi" if len(label_ids) > 1 else "single",
            "label_ids": label_ids,
            "label_names": label_names,
        },
        "prompt": {
            "prompt_type": prompt.get("prompt_type") or "mask_and_box",
            "initialization": prompt.get("initialization") or prompt.get("id") or "middle_slab_random",
            "propagation": "teacher_forced_gt_memory",
        } if (config.get("paradigm") or "prompt_guided") == "prompt_guided" else None,
        "training_data": {
            "source_project_path": config.get("directory"),
            "source_project_name": config.get("project_name"),
            "subjects": config.get("subjects"),
            "assignments": config.get("assignments"),
            "labels": labels,
            "examples": config.get("examples"),
        },
        "preprocessing": {
            "working_space": "paired_fullres_non_elastic_or_raw",
            "resolution": config.get("hyperparameters", {}).get("resolution", 512),
            "intensity": "per-volume min-max to uint8, then ImageNet normalize",
            "subject_provenance": config.get("subject_provenance"),
        },
        "training": {
            "run_id": run_id,
            "hyperparameters": config.get("hyperparameters"),
            "augmentation": config.get("augmentation"),
            "trainable_preset": config.get("trainable_preset"),
            "seed": config.get("seed"),
            "loss": config.get("loss_weights"),
        },
        "evaluation": evaluation or {},
        "checkpoint_selection": selected_checkpoint or {},
        "software": {
            "aurora_version": read_aurora_version(),
            "model_implementation": "pipeline.MedSAM2 + pipeline.aiFineTuning",
            "libraries": library_versions(),
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "device": config.get("device"),
        },
        "architectural_modifications": architectural_modifications if architectural_modifications is not None else "none",
    }
