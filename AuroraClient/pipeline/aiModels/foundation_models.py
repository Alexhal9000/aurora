"""Registry, resolver, and availability checks for downloadable foundation weights."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .paths import models_root

MISSING_HINT = "Download it from the header menu: AI models"

LICENSES_DIR = Path(__file__).resolve().parent / "licenses"

MEDSAM2_ID = "medsam2"
DINOV3_ID = "dinov3_vitl16"

MEDSAM2_SPEC: Dict[str, Any] = {
    "id": MEDSAM2_ID,
    "display_name": "MedSAM2",
    "subdir": "medsam2",
    "is_dir": False,
    "files": [
        {
            "name": "MedSAM2_latest.pt",
            "size": 156040129,
            "sha256": "c92743b99f00d078bf32a3afcc38aaa9faf1c1692dffe3eaa7a90938c1991060",
        }
    ],
    "description": (
        "Foundation 3D medical image segmentation model (prompt-guided, "
        "3-view ensemble). Developed by the Bo Wang Lab."
    ),
    "enables": "AI Segmentation (prompt-guided MedSAM2) and fine-tuning of MedSAM2.",
    "article_citation": (
        "Ma, J. et al. MedSAM2: Segment Anything in 3D Medical Images and Videos. "
        "arXiv:2504.03600 (2025)."
    ),
    "article_url": "https://arxiv.org/abs/2504.03600",
    "provider_url": "https://github.com/bowang-lab/MedSAM2",
    "license_file": "medsam2.txt",
}

# TEMP: hidden for release — reinstate when DINO-Reg ships
DINOV3_SPEC: Dict[str, Any] = {
    "id": DINOV3_ID,
    "display_name": "DINOv3 ViT-L/16",
    "subdir": "dinov3_vitl16",
    "is_dir": True,
    "files": [
        {
            "name": "model.safetensors",
            "size": 1212559808,
            "sha256": "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179",
        },
        {
            "name": "config.json",
            "size": 745,
            "sha256": "135ecd23e34a70b6fbed8b083fdecb319b7e3a54e3d849258bbe4ddcf1783bb5",
        },
        {
            "name": "preprocessor_config.json",
            "size": 585,
            "sha256": "960c41d1f3a7778b936365769a2d90550b318a6c0a53a0296957adacfe5e0dd7",
        },
    ],
    "description": (
        "Meta DINOv3 ViT-L/16 vision encoder used by Aurora's DINO-Reg rigid alignment."
    ),
    "enables": "DINO-Reg Rigid alignment (voxel-only, CUDA).",
    "article_citation": (
        "Siméoni, O. et al. DINOv3. arXiv:2508.10104 (2025)."
    ),
    "article_url": "https://arxiv.org/abs/2508.10104",
    "provider_url": "https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m",
    "license_file": "dinov3.md",
}

FOUNDATION_MODELS: List[Dict[str, Any]] = [
    MEDSAM2_SPEC,
    # TEMP: hidden for release — comment back in when DINO-Reg is restored
    # DINOV3_SPEC,
]


class ModelMissingError(FileNotFoundError):
    def __init__(self, model_id: str, message: Optional[str] = None):
        self.model_id = model_id
        super().__init__(message or f"Foundation model {model_id!r} is not installed. {MISSING_HINT}.")


def foundation_dir() -> Path:
    return models_root() / "foundation"


def enabled_models() -> List[Dict[str, Any]]:
    return list(FOUNDATION_MODELS)


def all_known_models() -> List[Dict[str, Any]]:
    """Enabled models plus commented-out specs, so loaders can still resolve them."""
    by_id = {item["id"]: item for item in FOUNDATION_MODELS}
    by_id.setdefault(MEDSAM2_ID, MEDSAM2_SPEC)
    by_id.setdefault(DINOV3_ID, DINOV3_SPEC)
    return list(by_id.values())


def get_spec(model_id: str) -> Optional[Dict[str, Any]]:
    if not model_id:
        return None
    for item in all_known_models():
        if item["id"] == model_id:
            return item
    return None


def _foundation_path(spec: Dict[str, Any]) -> Path:
    return foundation_dir() / spec["subdir"]


def _files_present(directory: Path, spec: Dict[str, Any]) -> bool:
    if spec.get("is_dir"):
        if not directory.is_dir():
            return False
        return all((directory / item["name"]).is_file() and (directory / item["name"]).stat().st_size == item["size"] for item in spec["files"])
    if directory.is_file():
        expected = spec["files"][0]
        return directory.name == expected["name"] and directory.stat().st_size == expected["size"]
    if not directory.is_dir():
        return False
    return all(
        (directory / item["name"]).is_file() and (directory / item["name"]).stat().st_size == item["size"]
        for item in spec["files"]
    )


def resolve_model_dir(model_id: str) -> Path:
    """Directory that contains the model's files under Documents/Aurora AI Models/foundation."""
    spec = get_spec(model_id)
    if spec is None:
        raise KeyError(f"Unknown foundation model {model_id!r}")
    candidate = _foundation_path(spec)
    if _files_present(candidate, spec):
        if candidate.is_file():
            return candidate.parent
        return candidate
    return candidate


def resolve_model_file(model_id: str, filename: Optional[str] = None) -> Path:
    spec = get_spec(model_id)
    if spec is None:
        raise KeyError(f"Unknown foundation model {model_id!r}")
    name = filename or spec["files"][0]["name"]
    directory = resolve_model_dir(model_id)
    return directory / name


def resolve_model_path(model_id: str) -> Path:
    """Primary artifact: file for MedSAM2, directory for DINOv3."""
    spec = get_spec(model_id)
    if spec is None:
        raise KeyError(f"Unknown foundation model {model_id!r}")
    if spec.get("is_dir"):
        return resolve_model_dir(model_id)
    return resolve_model_file(model_id)


def is_installed(model_id: str) -> bool:
    spec = get_spec(model_id)
    if spec is None:
        return False
    try:
        directory = resolve_model_dir(model_id)
    except KeyError:
        return False
    return _files_present(directory, spec)


def require_installed(model_id: str) -> Path:
    if not is_installed(model_id):
        raise ModelMissingError(model_id)
    return resolve_model_path(model_id)


def license_text(model_id: str) -> str:
    spec = get_spec(model_id)
    if spec is None:
        return ""
    path = LICENSES_DIR / spec["license_file"]
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def license_hash(model_id: str) -> str:
    text = license_text(model_id)
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if text else ""


def total_size_bytes(model_ids: Iterable[str]) -> int:
    total = 0
    for model_id in model_ids:
        spec = get_spec(model_id)
        if spec is None:
            continue
        total += sum(int(item["size"]) for item in spec["files"])
    return total


def public_spec(model_id: str) -> Dict[str, Any]:
    spec = get_spec(model_id)
    if spec is None:
        raise KeyError(f"Unknown foundation model {model_id!r}")
    installed = is_installed(model_id)
    return {
        "id": spec["id"],
        "display_name": spec["display_name"],
        "description": spec["description"],
        "enables": spec["enables"],
        "article_citation": spec["article_citation"],
        "article_url": spec["article_url"],
        "provider_url": spec["provider_url"],
        "size_bytes": sum(int(item["size"]) for item in spec["files"]),
        "files": [{"name": item["name"], "size": item["size"]} for item in spec["files"]],
        "installed": installed,
        "license_text": license_text(model_id),
        "license_hash": license_hash(model_id),
        "unavailable_reason": None if installed else MISSING_HINT,
    }
