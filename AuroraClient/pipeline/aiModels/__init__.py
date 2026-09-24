"""Aurora AI model adapters: task contracts, registry, and inference dispatch."""

from .contracts import (
    DIMENSIONALITY_2D,
    DIMENSIONALITY_3D,
    PARADIGM_DIRECT,
    PARADIGM_PROMPT_GUIDED,
    FOUNDATION_MEDSAM2,
)
from .registry import (
    get_model,
    list_model_descriptors,
    resolve_inference_model,
)

__all__ = [
    "DIMENSIONALITY_2D",
    "DIMENSIONALITY_3D",
    "PARADIGM_DIRECT",
    "PARADIGM_PROMPT_GUIDED",
    "FOUNDATION_MEDSAM2",
    "get_model",
    "list_model_descriptors",
    "resolve_inference_model",
]
