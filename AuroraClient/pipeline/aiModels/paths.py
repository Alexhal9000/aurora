"""OS-aware location of Aurora AI model artifacts."""

from __future__ import annotations

import os
from pathlib import Path


MODELS_DIR_NAME = "Aurora AI Models"
ENV_MODELS_ROOT = "AURORA_AI_MODELS_DIR"


def documents_dir() -> Path:
    """User Documents directory, matching QuickAccessPathsView."""
    return Path.home() / "Documents"


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
