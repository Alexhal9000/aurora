"""Prompt-mode helpers shared by fine-tuning and inference."""

from __future__ import annotations

from typing import Any, Dict, Optional

FULL_SLICE_TOKENS = frozenset({"full_slice_ones", "full_slice"})


def is_full_slice_prompt(prompt=None) -> bool:
    """True when the model expects a whole-slice ones mask + full-frame box."""
    if not prompt:
        return False
    if isinstance(prompt, str):
        token = prompt
    elif isinstance(prompt, dict):
        token = (
            prompt.get("initialization")
            or prompt.get("id")
            or prompt.get("prompt_type")
            or ""
        )
    else:
        return False
    return str(token) in FULL_SLICE_TOKENS


def resolve_prompt_initialization(descriptor: Optional[Dict[str, Any]]) -> Optional[str]:
    """Read prompt mode from a model descriptor or its registered metadata."""
    if not descriptor:
        return None
    init = descriptor.get("prompt_initialization")
    if init:
        return str(init)
    prompt = (descriptor.get("metadata") or {}).get("prompt") or {}
    if not isinstance(prompt, dict):
        return None
    for key in ("initialization", "id", "prompt_type"):
        value = prompt.get(key)
        if value:
            return str(value)
    return None
