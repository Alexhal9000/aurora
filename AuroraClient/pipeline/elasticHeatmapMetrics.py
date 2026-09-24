"""
Registry of per-vertex elastic heatmap metrics.

Contract (frontend + backend):
  - Values are float32 per display-mesh vertex.
  - Higher value = worse (hot / red on the shared jet colormap).
  - Response fields: dense_distances, dense_distances_encoding, dense_distances_metric.

To add a new intensity-based metric:
  1. Implement a compute_fn(fixed, moving, vertices_xyz, spacing_per_voxel) -> float32[N]
  2. Register it in HEATMAP_METRICS below with supports_voxel/supports_mesh.
  3. Mirror display/scale metadata in the frontend elasticHeatmapMetrics.js registry.
Surface-distance stays the structural fallback (computed in MarchingCubesView).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .elasticLocalNccHeatmap import (
    compute_per_vertex_local_ncc_dissimilarity,
)

# Fixed voxel half-width for local NCC (species-agnostic). Full cube is (2R+1)³.
# Start at 5 so you can judge smoothness; bump to 6+ if it looks too noisy.
LOCAL_NCC_RADIUS_VOXELS = 5


SURFACE_DISTANCE = 'surface_distance'
NCC = 'ncc'
DEFAULT_HEATMAP_METRIC = SURFACE_DISTANCE
FALLBACK_HEATMAP_METRIC = SURFACE_DISTANCE

IntensityComputeFn = Callable[
    [np.ndarray, np.ndarray, np.ndarray, float],
    np.ndarray,
]


@dataclass(frozen=True)
class HeatmapMetricSpec:
    id: str
    supports_voxel: bool
    supports_mesh: bool
    # 'surface' = Open3D / mesh distance path in MarchingCubesView
    # 'intensity' = fixed vs moving intensity window at each vertex
    compute_kind: str
    intensity_compute: Optional[IntensityComputeFn] = None


def _compute_ncc_dissimilarity(fixed, moving, vertices_xyz, spacing_per_voxel):
    return compute_per_vertex_local_ncc_dissimilarity(
        fixed,
        moving,
        vertices_xyz,
        spacing_per_voxel,
        LOCAL_NCC_RADIUS_VOXELS,
    )


HEATMAP_METRICS = {
    SURFACE_DISTANCE: HeatmapMetricSpec(
        id=SURFACE_DISTANCE,
        supports_voxel=True,
        supports_mesh=True,
        compute_kind='surface',
    ),
    NCC: HeatmapMetricSpec(
        id=NCC,
        supports_voxel=True,
        supports_mesh=False,
        compute_kind='intensity',
        intensity_compute=_compute_ncc_dissimilarity,
    ),
}


def normalize_heatmap_metric(raw) -> str:
    """Map request / settings strings onto a known metric id."""
    if raw is None:
        return DEFAULT_HEATMAP_METRIC
    key = str(raw).strip().lower()
    if key in HEATMAP_METRICS:
        return key
    # Legacy / list metric names that still mean "surface heatmap"
    if key in ('dice', 'surface', 'surface_distance'):
        return SURFACE_DISTANCE
    return DEFAULT_HEATMAP_METRIC


def resolve_heatmap_metric(requested, *, is_mesh: bool = False) -> str:
    """
    Choose a computable metric for this subject type.
    Unsupported combos fall back to surface distance.
    """
    metric_id = normalize_heatmap_metric(requested)
    spec = HEATMAP_METRICS.get(metric_id)
    if spec is None:
        return FALLBACK_HEATMAP_METRIC
    if is_mesh and not spec.supports_mesh:
        return FALLBACK_HEATMAP_METRIC
    if (not is_mesh) and not spec.supports_voxel:
        return FALLBACK_HEATMAP_METRIC
    return metric_id


def get_heatmap_metric_spec(metric_id: str) -> HeatmapMetricSpec:
    return HEATMAP_METRICS.get(
        normalize_heatmap_metric(metric_id),
        HEATMAP_METRICS[FALLBACK_HEATMAP_METRIC],
    )


def try_compute_intensity_heatmap(
    metric_id: str,
    fixed_volume: np.ndarray,
    moving_volume: np.ndarray,
    vertices_xyz: np.ndarray,
    spacing_per_voxel: float,
) -> Optional[np.ndarray]:
    """
    Run a registered intensity metric. Returns None for surface or unknown metrics.
    Raises on compute failures so callers can fall back.
    """
    spec = get_heatmap_metric_spec(metric_id)
    if spec.compute_kind != 'intensity' or spec.intensity_compute is None:
        return None
    if fixed_volume.shape != moving_volume.shape:
        raise ValueError(
            f"{spec.id} heatmap requires matching grids "
            f"(fixed {fixed_volume.shape} vs moving {moving_volume.shape})"
        )
    return spec.intensity_compute(
        fixed_volume,
        moving_volume,
        vertices_xyz,
        spacing_per_voxel,
    )
