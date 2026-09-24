"""Background download of foundation weights from the lab server."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import requests

from .foundation_models import (
    foundation_dir,
    get_spec,
    is_installed,
    license_hash,
    public_spec,
    total_size_bytes,
)
from .paths import ensure_models_root

logger = logging.getLogger(__name__)

ALLOWED_LAB_ORIGINS = frozenset({
    "https://hallgrimssonlab.ca",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "http://127.0.0.1:8001",
    "http://localhost:8001",
    "http://127.0.0.1:8002",
    "http://localhost:8002",
    "http://127.0.0.1:8003",
    "http://localhost:8003",
    "http://127.0.0.1:8004",
    "http://localhost:8004",
})

_LOCK = threading.Lock()
_STATE: Dict[str, Any] = {
    "active": False,
    "cancel": False,
    "model_id": None,
    "filename": None,
    "bytes_done": 0,
    "bytes_total": 0,
    "error": None,
    "thread": None,
}


class DownloadError(Exception):
    pass


def normalize_lab_origin(origin: str) -> str:
    parsed = urlparse((origin or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise DownloadError("Invalid lab origin.")
    host = parsed.hostname or ""
    if host == "0.0.0.0":
        host = "127.0.0.1"
    port = parsed.port
    if port:
        normalized = f"{parsed.scheme}://{host}:{port}"
    else:
        normalized = f"{parsed.scheme}://{host}"
    if normalized not in ALLOWED_LAB_ORIGINS:
        raise DownloadError("Lab origin is not allowed.")
    return normalized


def _disk_free(path: Path) -> int:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free


def _progress_snapshot() -> Dict[str, Any]:
    if not _STATE["active"] and not _STATE["model_id"]:
        return None
    return {
        "active": bool(_STATE["active"]),
        "model_id": _STATE["model_id"],
        "filename": _STATE["filename"],
        "bytes_done": int(_STATE["bytes_done"] or 0),
        "bytes_total": int(_STATE["bytes_total"] or 0),
        "error": _STATE["error"],
    }


def status_payload() -> Dict[str, Any]:
    from .foundation_models import enabled_models

    with _LOCK:
        progress = _progress_snapshot()
    models = []
    for spec in enabled_models():
        item = public_spec(spec["id"])
        if progress and progress.get("model_id") == spec["id"]:
            item["download"] = progress
        else:
            item["download"] = None
        models.append(item)
    return {"models": models, "count": len(models), "progress": progress}


def record_accepted_terms(model_ids: List[str], user_name: str = "") -> None:
    ensure_models_root()
    dest = foundation_dir()
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "accepted_terms.json"
    existing: List[Dict[str, Any]] = []
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8")).get("entries") or []
        except (OSError, json.JSONDecodeError):
            existing = []
    now = datetime.now(timezone.utc).isoformat()
    for model_id in model_ids:
        existing.append({
            "model_id": model_id,
            "license_hash": license_hash(model_id),
            "accepted_at": now,
            "user_name": user_name or "",
        })
    path.write_text(json.dumps({"entries": existing}, indent=2) + "\n", encoding="utf-8")


def start_download(model_ids: List[str], token: str, lab_origin: str, user_name: str = "") -> None:
    if not token:
        raise DownloadError("Login token is required.")
    origin = normalize_lab_origin(lab_origin)
    wanted = []
    for model_id in model_ids:
        spec = get_spec(model_id)
        if spec is None:
            raise DownloadError(f"Unknown model {model_id!r}.")
        from .foundation_models import enabled_models
        if spec["id"] not in {item["id"] for item in enabled_models()}:
            raise DownloadError(f"Model {model_id!r} is not offered in this release.")
        if not is_installed(model_id):
            wanted.append(model_id)
    if not wanted:
        return
    needed = total_size_bytes(wanted)
    dest = foundation_dir()
    dest.mkdir(parents=True, exist_ok=True)
    free = _disk_free(dest)
    if free < needed + (50 * 1024 * 1024):
        raise DownloadError(
            f"Not enough free disk space. Need about {needed} bytes, have {free}."
        )
    with _LOCK:
        if _STATE["active"]:
            raise DownloadError("A download is already in progress.")
        _STATE.update({
            "active": True,
            "cancel": False,
            "model_id": wanted[0],
            "filename": None,
            "bytes_done": 0,
            "bytes_total": needed,
            "error": None,
        })
        thread = threading.Thread(
            target=_run_download,
            args=(wanted, token, origin, user_name),
            daemon=True,
        )
        _STATE["thread"] = thread
        thread.start()


def cancel_download() -> None:
    with _LOCK:
        _STATE["cancel"] = True


def _run_download(model_ids: List[str], token: str, origin: str, user_name: str) -> None:
    try:
        for model_id in model_ids:
            _download_one(model_id, token, origin)
        record_accepted_terms(model_ids, user_name=user_name)
        with _LOCK:
            _STATE["error"] = None
    except Exception as exc:
        logger.exception("Foundation model download failed")
        with _LOCK:
            _STATE["error"] = str(exc)
    finally:
        with _LOCK:
            _STATE["active"] = False
            _STATE["thread"] = None


def _download_one(model_id: str, token: str, origin: str) -> None:
    spec = get_spec(model_id)
    if spec is None:
        raise DownloadError(f"Unknown model {model_id!r}")
    dest_dir = foundation_dir() / spec["subdir"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        _STATE["model_id"] = model_id
    for file_spec in spec["files"]:
        _download_file(model_id, file_spec, token, origin, dest_dir)


def _download_file(
    model_id: str,
    file_spec: Dict[str, Any],
    token: str,
    origin: str,
    dest_dir: Path,
) -> None:
    filename = file_spec["name"]
    expected_size = int(file_spec["size"])
    expected_hash = file_spec["sha256"]
    url = urljoin(origin.rstrip("/") + "/", f"auroraModelFile/{model_id}/{filename}")
    part_path = dest_dir / f"{filename}.part"
    final_path = dest_dir / filename
    with _LOCK:
        if _STATE["cancel"]:
            raise DownloadError("Download cancelled.")
        _STATE["filename"] = filename
    digest = hashlib.sha256()
    downloaded = 0
    headers = {"Authorization": f"Token {token}"}
    with requests.get(url, headers=headers, stream=True, timeout=60) as response:
        if response.status_code != 200:
            raise DownloadError(
                f"Lab server refused {filename} (HTTP {response.status_code})."
            )
        with part_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                with _LOCK:
                    if _STATE["cancel"]:
                        handle.close()
                        try:
                            part_path.unlink()
                        except OSError:
                            pass
                        raise DownloadError("Download cancelled.")
                handle.write(chunk)
                digest.update(chunk)
                downloaded += len(chunk)
                with _LOCK:
                    _STATE["bytes_done"] = int(_STATE["bytes_done"] or 0) + len(chunk)
    if downloaded != expected_size:
        try:
            part_path.unlink()
        except OSError:
            pass
        raise DownloadError(
            f"{filename} size mismatch: got {downloaded}, expected {expected_size}."
        )
    if digest.hexdigest() != expected_hash:
        try:
            part_path.unlink()
        except OSError:
            pass
        raise DownloadError(f"{filename} failed integrity check.")
    part_path.replace(final_path)


def wait_until_idle(timeout: float = 0.0) -> None:
    deadline = time.time() + timeout if timeout else None
    while True:
        with _LOCK:
            thread = _STATE.get("thread")
            active = _STATE["active"]
        if not active and thread is None:
            return
        if deadline is not None and time.time() >= deadline:
            return
        time.sleep(0.2)
