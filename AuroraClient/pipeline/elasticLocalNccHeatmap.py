"""
Per-vertex local intensity NCC for elastic registration heatmaps.

Computes zero-mean NCC in a cubic window around each mesh vertex, then returns
dissimilarity in [0, 1] so existing jet coloring (high = bad) matches surface
distance heatmaps:

  dissimilarity = 1 - (ncc + 1) / 2
  → ncc = +1 → 0 (best), ncc = 0 → 0.5, ncc = -1 → 1 (worst)
"""

from __future__ import annotations

import numpy as np


def _integral_image_3d(volume: np.ndarray) -> np.ndarray:
    """Prefix sums with a leading zero plane so box sums are O(1)."""
    z, y, x = volume.shape
    integ = np.zeros((z + 1, y + 1, x + 1), dtype=np.float64)
    integ[1:, 1:, 1:] = volume.astype(np.float64, copy=False)
    np.cumsum(integ, axis=0, out=integ)
    np.cumsum(integ, axis=1, out=integ)
    np.cumsum(integ, axis=2, out=integ)
    return integ


def _box_sum(integ: np.ndarray, z0: int, z1: int, y0: int, y1: int, x0: int, x1: int) -> float:
    """Sum of volume[z0:z1, y0:y1, x0:x1] (half-open) via integral image."""
    return float(
        integ[z1, y1, x1]
        - integ[z0, y1, x1]
        - integ[z1, y0, x1]
        - integ[z1, y1, x0]
        + integ[z0, y0, x1]
        + integ[z0, y1, x0]
        + integ[z1, y0, x0]
        - integ[z0, y0, x0]
    )


def radius_voxels_from_mm(voxel_size_mm: float, radius_mm: float = 2.5) -> int:
    """Optional mm→voxel helper. Heatmap NCC uses a fixed voxel radius instead."""
    vs = float(voxel_size_mm) if voxel_size_mm and float(voxel_size_mm) > 0 else 1.0
    return max(2, int(round(float(radius_mm) / vs)))


def compute_per_vertex_local_ncc_dissimilarity(
    fixed_volume: np.ndarray,
    moving_volume: np.ndarray,
    vertices_xyz: np.ndarray,
    spacing_per_voxel: float,
    radius_voxels: int,
) -> np.ndarray:
    """
    Local zero-mean NCC at each vertex, returned as dissimilarity in [0, 1].

    Volumes and vertex axis 0..2 must share the same layout as marching cubes
    (vertices_mm ≈ voxel_index * spacing_per_voxel).
    """
    if vertices_xyz is None or len(vertices_xyz) == 0:
        return np.zeros(0, dtype=np.float32)

    fixed = np.asarray(fixed_volume)
    moving = np.asarray(moving_volume)
    if fixed.shape != moving.shape:
        raise ValueError("fixed and moving volumes must have the same shape")

    z_dim, y_dim, x_dim = fixed.shape
    r = int(max(1, radius_voxels))
    spacing = float(spacing_per_voxel) if spacing_per_voxel and spacing_per_voxel > 0 else 1.0

    coords = np.round(np.asarray(vertices_xyz, dtype=np.float64) / spacing).astype(np.int64)
    coords[:, 0] = np.clip(coords[:, 0], 0, z_dim - 1)
    coords[:, 1] = np.clip(coords[:, 1], 0, y_dim - 1)
    coords[:, 2] = np.clip(coords[:, 2], 0, x_dim - 1)

    fixed_f = fixed.astype(np.float64, copy=False)
    moving_f = moving.astype(np.float64, copy=False)
    sum_f = _integral_image_3d(fixed_f)
    sum_m = _integral_image_3d(moving_f)
    sum_ff = _integral_image_3d(fixed_f * fixed_f)
    sum_mm = _integral_image_3d(moving_f * moving_f)
    sum_fm = _integral_image_3d(fixed_f * moving_f)

    n_verts = coords.shape[0]
    out = np.empty(n_verts, dtype=np.float32)

    for i in range(n_verts):
        z, y, x = int(coords[i, 0]), int(coords[i, 1]), int(coords[i, 2])
        z0, z1 = max(0, z - r), min(z_dim, z + r + 1)
        y0, y1 = max(0, y - r), min(y_dim, y + r + 1)
        x0, x1 = max(0, x - r), min(x_dim, x + r + 1)
        n = float((z1 - z0) * (y1 - y0) * (x1 - x0))
        if n <= 1.0:
            out[i] = 0.5
            continue

        sf = _box_sum(sum_f, z0, z1, y0, y1, x0, x1)
        sm = _box_sum(sum_m, z0, z1, y0, y1, x0, x1)
        sff = _box_sum(sum_ff, z0, z1, y0, y1, x0, x1)
        smm = _box_sum(sum_mm, z0, z1, y0, y1, x0, x1)
        sfm = _box_sum(sum_fm, z0, z1, y0, y1, x0, x1)

        # Zero-mean NCC from local moments
        num = sfm - (sf * sm) / n
        var_f = sff - (sf * sf) / n
        var_m = smm - (sm * sm) / n
        if var_f <= 1e-12 or var_m <= 1e-12:
            ncc = 0.0
        else:
            ncc = float(num / np.sqrt(var_f * var_m))
            ncc = max(-1.0, min(1.0, ncc))

        out[i] = np.float32(1.0 - (ncc + 1.0) * 0.5)

    return out


def compute_dense_local_ncc_dissimilarity(
    fixed_volume: np.ndarray,
    moving_volume: np.ndarray,
    radius_voxels: int,
) -> np.ndarray:
    """
    Dense local zero-mean NCC dissimilarity for every voxel (float32 in [0, 1]).

    Same formula as the per-vertex path; uses a fixed cubic window via
    scipy.ndimage.uniform_filter for a smooth full-volume map suitable for
    slice overlays.
    """
    from scipy.ndimage import uniform_filter

    fixed = np.asarray(fixed_volume)
    moving = np.asarray(moving_volume)
    if fixed.shape != moving.shape:
        raise ValueError("fixed and moving volumes must have the same shape")

    r = int(max(1, radius_voxels))
    size = 2 * r + 1
    f = fixed.astype(np.float64, copy=False)
    m = moving.astype(np.float64, copy=False)

    mean_f = uniform_filter(f, size=size, mode='nearest')
    mean_m = uniform_filter(m, size=size, mode='nearest')
    mean_ff = uniform_filter(f * f, size=size, mode='nearest')
    mean_mm = uniform_filter(m * m, size=size, mode='nearest')
    mean_fm = uniform_filter(f * m, size=size, mode='nearest')

    cov = mean_fm - mean_f * mean_m
    var_f = mean_ff - mean_f * mean_f
    var_m = mean_mm - mean_m * mean_m

    denom = np.sqrt(np.maximum(var_f, 0.0) * np.maximum(var_m, 0.0))
    ncc = np.zeros(fixed.shape, dtype=np.float64)
    valid = denom > 1e-12
    np.divide(cov, denom, out=ncc, where=valid)
    np.clip(ncc, -1.0, 1.0, out=ncc)

    dissimilarity = (1.0 - (ncc + 1.0) * 0.5).astype(np.float32)
    return dissimilarity
