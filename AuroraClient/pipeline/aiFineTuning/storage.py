"""Filesystem helpers for training runs and registered models."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from ..aiModels.paths import (
    ensure_models_root,
    models_root,
    registered_model_dir,
    run_dir,
    runs_root,
)

STATUS_RUNNING = "running"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
TERMINAL_STATUSES = (STATUS_SUCCEEDED, STATUS_FAILED, STATUS_CANCELLED)

CONFIG_FILENAME = "config.json"
STATUS_FILENAME = "status.json"
CANCEL_FILENAME = "cancel.flag"
EVAL_FILENAME = "evaluation.json"
PID_FILENAME = "pid.json"


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.replace(tmp, path)


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def write_run_status(run_id: str, payload: Dict[str, Any]) -> Path:
    path = run_dir(run_id) / STATUS_FILENAME
    atomic_write_json(path, payload)
    return path


def read_run_status(run_id: str) -> Optional[Dict[str, Any]]:
    return read_json(run_dir(run_id) / STATUS_FILENAME)


def read_run_config(run_id: str) -> Optional[Dict[str, Any]]:
    return read_json(run_dir(run_id) / CONFIG_FILENAME)


def write_run_config(run_id: str, config: Dict[str, Any]) -> Path:
    """Write the immutable training-run configuration. Refuses to overwrite."""
    path = run_dir(run_id) / CONFIG_FILENAME
    if path.exists():
        raise FileExistsError(f"Training-run config already exists: {path}")
    atomic_write_json(path, config)
    return path


def request_cancel(run_id: str) -> Path:
    path = run_dir(run_id) / CANCEL_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("cancel\n", encoding="utf-8")
    return path


def cancel_requested(run_id: str) -> bool:
    return (run_dir(run_id) / CANCEL_FILENAME).is_file()


def write_pid(run_id: str, pid: int) -> None:
    atomic_write_json(run_dir(run_id) / PID_FILENAME, {"pid": int(pid)})


def read_pid(run_id: str) -> Optional[int]:
    payload = read_json(run_dir(run_id) / PID_FILENAME)
    if not payload:
        return None
    try:
        return int(payload.get("pid"))
    except (TypeError, ValueError):
        return None


def checkpoints_dir(run_id: str) -> Path:
    path = run_dir(run_id) / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path


def logs_dir(run_id: str) -> Path:
    path = run_dir(run_id) / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def figures_dir(run_id: str) -> Path:
    path = run_dir(run_id) / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


# Re-export path helpers used by APIs and tests.
ensure_models_root
models_root
registered_model_dir
run_dir
runs_root
