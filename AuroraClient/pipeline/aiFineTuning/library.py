"""List, inspect, delete, export, and import registered fine-tuned models."""

from __future__ import annotations

import io
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..aiModels.paths import models_root, registered_model_dir
from ..aiModels.registry import get_model, list_model_descriptors
from .checkpoint import INFERENCE_ARTIFACT_FILENAME, export_inference_checkpoint
from .storage import atomic_write_json, read_json

ALLOWED_SUFFIXES = {".json", ".pt", ".pth", ".bin", ".png", ".csv", ".txt", ".md", ".gz", ".nii"}
FIGURE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


class LibraryError(ValueError):
    pass


def list_registered_library() -> List[Dict[str, Any]]:
    rows = []
    for descriptor in list_model_descriptors(include_capabilities=True):
        if descriptor.get("is_foundation"):
            continue
        rows.append(_summary(descriptor))
    return rows


def model_detail(model_id: str) -> Dict[str, Any]:
    descriptor = get_model(model_id)
    if descriptor is None or descriptor.get("is_foundation"):
        raise LibraryError(f"Registered model {model_id} was not found")
    model_dir = registered_model_dir(model_id)
    metadata = descriptor.get("metadata") or read_json(model_dir / "metadata.json") or {}
    evaluation = metadata.get("evaluation") or {}
    figures = _list_figures(model_dir)
    return {
        **_summary(descriptor),
        "path": str(model_dir),
        "metadata": metadata,
        "evaluation": evaluation,
        "figures": figures,
        "preview": _preview_manifest(model_dir),
        "files": sorted(
            str(path.relative_to(model_dir))
            for path in model_dir.rglob("*")
            if path.is_file()
        ),
    }


def delete_registered_model(model_id: str) -> Dict[str, Any]:
    descriptor = get_model(model_id)
    if descriptor is None:
        raise LibraryError(f"Registered model {model_id} was not found")
    if descriptor.get("is_foundation"):
        raise LibraryError("The bundled MedSAM2 foundation model cannot be deleted")
    model_dir = registered_model_dir(model_id)
    if not model_dir.is_dir() or model_dir.resolve() == models_root().resolve():
        raise LibraryError("Refusing to delete this path")
    if model_dir.parent.resolve() != models_root().resolve():
        raise LibraryError("Refusing to delete a model outside the Aurora AI Models folder")
    shutil.rmtree(model_dir)
    return {"ok": True, "model_id": model_id, "deleted": True}


def export_registered_zip(model_id: str) -> Tuple[io.BytesIO, str]:
    detail = model_detail(model_id)
    model_dir = Path(detail["path"])
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in model_dir.rglob("*"):
            if not path.is_file():
                continue
            archive.write(path, arcname=str(path.relative_to(model_dir)))
    buffer.seek(0)
    display = detail.get("display_name") or model_id
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in display)[:80]
    filename = f"{safe}_{model_id[:8]}.zip"
    return buffer, filename


def import_registered_zip(upload) -> Dict[str, Any]:
    raw = upload.read() if hasattr(upload, "read") else upload
    if not raw:
        raise LibraryError("The zip file is empty")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                _safe_extract(archive, extract_dir)
        except zipfile.BadZipFile as exc:
            raise LibraryError("That file is not a valid zip") from exc
        source = _find_metadata_dir(extract_dir)
        if source is None:
            raise LibraryError("The zip does not contain metadata.json")
        metadata = read_json(source / "metadata.json") or {}
        old_id = str((metadata.get("identity") or {}).get("model_id") or "")
        model_id = old_id if old_id and not registered_model_dir(old_id).exists() else str(uuid.uuid4())
        dest = registered_model_dir(model_id)
        if dest.exists():
            raise LibraryError(f"A model already exists at {dest}")
        shutil.copytree(source, dest)
        identity = dict(metadata.get("identity") or {})
        identity["model_id"] = model_id
        metadata["identity"] = identity
        atomic_write_json(dest / "metadata.json", metadata)
        artifact = dest / INFERENCE_ARTIFACT_FILENAME
        if artifact.is_file():
            export_inference_checkpoint(artifact, artifact)
    loaded = get_model(model_id)
    if loaded is None:
        shutil.rmtree(dest, ignore_errors=True)
        raise LibraryError("Imported zip is missing a usable checkpoint")
    return {"ok": True, "model_id": model_id, "display_name": loaded.get("display_name"), "path": str(dest)}


def figure_path(model_id: str, relative: str) -> Path:
    model_dir = registered_model_dir(model_id).resolve()
    if get_model(model_id) is None:
        raise LibraryError(f"Registered model {model_id} was not found")
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise LibraryError("Invalid figure path")
    path = (model_dir / rel).resolve()
    if not str(path).startswith(str(model_dir)) or not path.is_file():
        raise LibraryError("Figure was not found")
    if path.suffix.lower() not in FIGURE_SUFFIXES | {".json"}:
        raise LibraryError("That file type cannot be previewed")
    return path


def _summary(descriptor: Dict[str, Any]) -> Dict[str, Any]:
    metadata = descriptor.get("metadata") or {}
    evaluation = metadata.get("evaluation") or {}
    validation = evaluation.get("validation") or {}
    test = evaluation.get("test") or {}
    identity = metadata.get("identity") or {}
    model_id = descriptor.get("id")
    model_dir = registered_model_dir(model_id)
    return {
        "id": model_id,
        "display_name": descriptor.get("display_name"),
        "foundation_model": descriptor.get("foundation_model"),
        "label_ids": descriptor.get("label_ids") or (metadata.get("task_contract") or {}).get("label_ids"),
        "label_names": descriptor.get("label_names") or (metadata.get("task_contract") or {}).get("label_names"),
        "prompt_initialization": descriptor.get("prompt_initialization") or (metadata.get("prompt") or {}).get("initialization"),
        "trained_at": (metadata.get("software") or {}).get("trained_at"),
        "run_id": (metadata.get("training") or {}).get("run_id"),
        "val_dice": validation.get("dice") if isinstance(validation, dict) else None,
        "test_dice": test.get("dice") if isinstance(test, dict) else None,
        "foundation_dice": ((evaluation.get("foundation_vs_finetuned") or {}).get("foundation") or {}).get("dice"),
        "finetuned_dice": ((evaluation.get("foundation_vs_finetuned") or {}).get("finetuned") or {}).get("dice"),
        "artifact_filename": identity.get("artifact_filename"),
        "has_figures": (model_dir / "figures").is_dir(),
        "has_preview": (model_dir / "preview" / "manifest.json").is_file(),
        "path": str(model_dir),
    }


def _preview_manifest(model_dir: Path) -> Dict[str, Any]:
    from .preview_store import read_manifest

    return read_manifest(model_dir)


def _list_figures(model_dir: Path) -> List[Dict[str, str]]:
    figures_root = model_dir / "figures"
    if not figures_root.is_dir():
        return []
    rows = []
    for path in sorted(figures_root.rglob("*")):
        if path.suffix.lower() not in FIGURE_SUFFIXES:
            continue
        rows.append({
            "name": path.name,
            "relative": str(path.relative_to(model_dir)).replace("\\", "/"),
        })
    return rows


def _safe_extract(archive: zipfile.ZipFile, dest: Path) -> None:
    dest = dest.resolve()
    for info in archive.infolist():
        name = Path(info.filename)
        if info.is_dir() or str(info.filename).endswith("/"):
            continue
        if name.is_absolute() or ".." in name.parts:
            raise LibraryError("Zip contains an unsafe path")
        if name.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        target = (dest / name).resolve()
        if not str(target).startswith(str(dest)):
            raise LibraryError("Zip contains an unsafe path")
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as source, open(target, "wb") as handle:
            shutil.copyfileobj(source, handle)


def _find_metadata_dir(root: Path) -> Optional[Path]:
    direct = root / "metadata.json"
    if direct.is_file():
        return root
    matches = list(root.rglob("metadata.json"))
    if len(matches) == 1:
        return matches[0].parent
    for child in root.iterdir():
        if child.is_dir() and (child / "metadata.json").is_file():
            return child
    return None
