"""Epoch validation speed: 2D clips always, optional 3-view volume Dice."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

MODE_FAST = "fast_2d"
MODE_INTERMEDIATE = "intermediate_one_volume"
MODE_SLOW = "slow_all_volumes"

MODES = (MODE_FAST, MODE_INTERMEDIATE, MODE_SLOW)

DICE_SOURCE_CLIP = "clip_2d"
DICE_SOURCE_ONE_VOLUME = "volume_3d_one"
DICE_SOURCE_ALL_VOLUMES = "volume_3d_all"


def resolve_validation_mode(value: Any) -> str:
    raw = str(value or MODE_FAST).strip()
    if raw in MODES:
        return raw
    return MODE_FAST


def uses_volume_dice(mode: Any) -> bool:
    return resolve_validation_mode(mode) in (MODE_INTERMEDIATE, MODE_SLOW)


def dice_source_for_mode(mode: Any) -> str:
    resolved = resolve_validation_mode(mode)
    if resolved == MODE_INTERMEDIATE:
        return DICE_SOURCE_ONE_VOLUME
    if resolved == MODE_SLOW:
        return DICE_SOURCE_ALL_VOLUMES
    return DICE_SOURCE_CLIP


def volume_val_examples(
    mode: Any,
    val_examples: Optional[Sequence[Dict[str, Any]]],
    sticky: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Subjects that get full 3-view majority-vote Dice this epoch."""
    examples = list(val_examples or [])
    resolved = resolve_validation_mode(mode)
    if resolved == MODE_FAST or not examples:
        return []
    if resolved == MODE_INTERMEDIATE:
        chosen = sticky or examples[0]
        return [chosen]
    return examples
