"""Average-pool an extracted volume by a discrete scaling factor.

Factor 1 leaves the array and voxel size unchanged. Factors 2, 3, and 4
mean-pool with the same edge-pad used by the lossy writer, then multiply
voxel size by that factor. Integer volumes are rounded back to their dtype.
"""

import math

import numpy as np
from skimage.measure import block_reduce


VALID_SCALING_FACTORS = (1, 2, 3, 4)


def normalize_scaling_factor(value):
    """Return 1, 2, 3, or 4. Anything else becomes 1."""
    try:
        factor = int(value)
    except (TypeError, ValueError):
        return 1
    if factor not in VALID_SCALING_FACTORS:
        return 1
    return factor


def average_pool_volume(image_data, factor):
    """Mean-pool a 3D volume. Factor 1 returns the same array."""
    factor = normalize_scaling_factor(factor)
    image_data = np.asarray(image_data)
    if factor <= 1 or image_data.ndim != 3 or any(size == 0 for size in image_data.shape):
        return image_data

    padded_shape = tuple(math.ceil(size / factor) * factor for size in image_data.shape)
    pad_widths = [(0, padded - size) for size, padded in zip(image_data.shape, padded_shape)]
    padded = np.pad(image_data, pad_widths, mode='edge')
    pooled = block_reduce(padded, block_size=(factor, factor, factor), func=np.mean)

    if np.issubdtype(image_data.dtype, np.integer):
        info = np.iinfo(image_data.dtype)
        pooled = np.clip(np.rint(pooled), info.min, info.max).astype(image_data.dtype)
    return pooled


def apply_extract_scaling(image_data, voxel_size, factor):
    """Pool the volume and scale its voxel size.

    Returns (pooled_volume, scaled_voxel_size, applied_factor).
    """
    factor = normalize_scaling_factor(factor)
    pooled = average_pool_volume(image_data, factor)
    try:
        spacing = float(voxel_size)
    except (TypeError, ValueError):
        spacing = 1.0
    return pooled, spacing * factor, factor
