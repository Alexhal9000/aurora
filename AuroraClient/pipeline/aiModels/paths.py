"""OS-aware location of Aurora AI model artifacts."""

from __future__ import annotations

import os
from pathlib import Path


MODELS_DIR_NAME = "Aurora AI Models"
ENV_MODELS_ROOT = "AURORA_AI_MODELS_DIR"


_SERVICE_HOME = "/var/lib/aurora-tools"


def _service_shared_folder():
    """Under the Linux background service (Aurora 1.9.4 network sharing) the account's own
    home is empty: use the first shared folder (by default the owner's Documents), so the
    owner's existing "Aurora AI Models" are found instead of downloaded again."""
    if os.environ.get("AURORA_SERVICE") != "1" and str(Path.home()) != _SERVICE_HOME:
        return None
    try:
        from AuroraClient import engineConfig
        raw = engineConfig._read_raw().get("shared_folders")
    except Exception:  # noqa: BLE001 - fall back to the account's own Documents
        return None
    folders = [f for f in raw if isinstance(f, str) and f.strip()] if isinstance(raw, list) else []
    for folder in folders:
        path = Path(os.path.expanduser(folder))
        if (path / MODELS_DIR_NAME).is_dir():
            return path
    return Path(os.path.expanduser(folders[0])) if folders else None


def documents_dir() -> Path:
    """User Documents directory, matching QuickAccessPathsView."""
    return _service_shared_folder() or (Path.home() / "Documents")


def models_root() -> Path:
    """``<Documents>/Aurora AI Models/``, overridable for tests."""
    override = os.environ.get(ENV_MODELS_ROOT)
    if override:
        return Path(override)
    return documents_dir() / MODELS_DIR_NAME


def runs_root() -> Path:
    return models_root() / "runs"


def run_dir(run_id: str) -> Path:
    return runs_root() / str(run_id)


def registered_model_dir(model_id: str) -> Path:
    return models_root() / str(model_id)


def ensure_models_root() -> Path:
    root = models_root()
    root.mkdir(parents=True, exist_ok=True)
    runs_root().mkdir(parents=True, exist_ok=True)
    return root
