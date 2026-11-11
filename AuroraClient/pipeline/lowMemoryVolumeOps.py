"""Lower-memory versions of a few volume steps, each bit-for-bit identical to the code it replaced.

Large uint8 micro-CT stacks (SkyScan JPEG folders, ~8 billion voxels for 2000^3) ran out of
memory in Clean mesh and DINO-Reg on a 64 GB machine (2026-11-09 audit). The peaks came from
full-volume temporaries that do not change any result: an int32 copy where a boolean mask
was enough, an int32 working copy for a subtraction the stored dtype can do exactly, a
float32 copy of a scan only to feed a Gaussian whose output is float32 anyway, a list of
every foreground coordinate only to read its minimum and maximum.

Each helper below documents the old expression it reproduces and why the result cannot
differ. pipeline/tests/test_low_memory_volume_ops.py checks every one of them against a
frozen copy of the old code. Keep the two in step: a helper that is not provably identical
belongs somewhere else.

Only numpy and scipy are imported, so the tests run without Django, ANTs or torch.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

# dtypes whose every value is exactly representable in float32, so that converting to
# float32 before a float32-output filter changes nothing (float64 and 32/64-bit integers
# are not: they keep the old explicit conversion).
_FLOAT32_EXACT_DTYPES = (
    np.dtype(np.uint8), np.dtype(np.int8),
    np.dtype(np.uint16), np.dtype(np.int16),
    np.dtype(np.float32),
)


def first_true_indices(mask, count):
    """`np.argwhere(mask)[:count]` without building the full coordinate list.

    np.argwhere returns every True voxel as int64 coordinates (24 bytes per voxel in 3-D);
    callers that only print the first few paid that for the whole mask. This walks the
    first axis one plane at a time and stops after `count` voxels. np.argwhere lists
    coordinates in C (row-major) index order whatever the memory layout, and so does this,
    so the rows are the same rows, with the same dtype (intp) and shape (k, ndim).
    """
    mask = np.asarray(mask)
    ndim = mask.ndim
    if ndim == 0 or count <= 0:
        return np.argwhere(mask)[:max(count, 0)]
    found = []
    for i in range(mask.shape[0]):
        plane = mask[i]
        if not plane.any():
            continue
        rows = np.argwhere(plane)[:count - len(found)]
        found.extend((i,) + tuple(int(v) for v in row) for row in rows)
        if len(found) >= count:
            break
    return np.array(found, dtype=np.intp).reshape(-1, ndim)


def axis_extents(mask):
    """Per-axis (min, max) index of the True voxels, as `np.argwhere(mask)` would give them.

    Returns (mins, maxs) as int64 arrays equal to `np.min(points, axis=0)` and
    `np.max(points, axis=0)` with `points = np.argwhere(mask)`, or None when the mask is
    empty. The minimum of a coordinate over all True voxels is the first index whose
    projection along the other axes has any True voxel (likewise the maximum), so an
    any() projection per axis gives the same numbers without the 24-bytes-per-voxel list.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return None
    mins = []
    maxs = []
    for axis in range(mask.ndim):
        others = tuple(a for a in range(mask.ndim) if a != axis)
        hit = np.flatnonzero(mask.any(axis=others) if others else mask)
        mins.append(hit[0])
        maxs.append(hit[-1])
    return np.array(mins, dtype=np.intp), np.array(maxs, dtype=np.intp)


# Voxels per bincount chunk: np.bincount converts its input to int64 first, so one call on
# a whole int32 label volume costs 8 bytes per voxel; 16 M voxels per chunk caps that at 128 MB.
_BINCOUNT_CHUNK = 16 * 1024 * 1024


def label_sizes(labeled, num_features):
    """`np.bincount(labeled.ravel())` for a label image from ndimage.label, in chunks.

    np.bincount casts its whole input to int64 (8 bytes per voxel, twice the int32 labels)
    before counting. Counting chunk by chunk with minlength = num_features + 1 and adding
    the int64 counts gives the same array: same length (labels run 0..num_features, so
    the full bincount has num_features + 1 bins), same dtype (intp) and the same counts.
    """
    flat = np.asarray(labeled).reshape(-1)
    length = int(num_features) + 1
    if flat.size == 0:
        return np.bincount(flat)  # empty in, empty out, as before
    if flat.size <= _BINCOUNT_CHUNK:
        return np.bincount(flat, minlength=length)
    sizes = np.zeros(length, dtype=np.intp)
    for start in range(0, flat.size, _BINCOUNT_CHUNK):
        sizes += np.bincount(flat[start:start + _BINCOUNT_CHUNK], minlength=length)
    return sizes


def homogenize_unsigned_background(data, target_background):
    """Background subtraction in the stored dtype for unsigned scans, or None.

    Reproduces HomogenizeBackgroundView._homogenize_background:

        out = clip(data.astype(int32) - int(target_background), dtype_min, dtype_max)

    For an unsigned dtype and 0 <= t <= dtype_max (t = int(target_background)), the
    difference never exceeds dtype_max and is clipped at 0 from below, so it equals
    max(data, t) - t, which the stored dtype computes exactly (max(data, t) >= t, so the
    subtraction cannot wrap). Peak: one output array instead of three int32 copies.
    The input is not modified (the old code returned a new array too).

    Returns None (caller keeps the int32 path) for signed or non-integer dtypes and for
    targets outside [0, dtype_max].
    """
    data = np.asarray(data)
    if data.dtype.kind != 'u':
        return None
    t = int(target_background)
    if t < 0 or t > np.iinfo(data.dtype).max:
        return None
    t_value = data.dtype.type(t)
    out = np.maximum(data, t_value)
    out -= t_value
    return out


def gaussian_float32(volume, sigma):
    """`gaussian_filter(np.asarray(volume, dtype=np.float32), sigma)` without the float32 input copy.

    scipy.ndimage filters each axis in a double-precision line buffer and stores each pass
    in the output array, which feeds the next pass. With output=np.float32, the first pass
    reads the original voxels into that double buffer. For dtypes that float32 holds
    exactly, those are the same doubles the float32 copy would have given, and every later
    pass reads the same float32 output either way. The result is identical. Other dtypes
    (float64, 32/64-bit integers) keep the explicit conversion.
    """
    arr = np.asarray(volume)
    if arr.dtype in _FLOAT32_EXACT_DTYPES:
        return gaussian_filter(arr, sigma=sigma, output=np.float32)
    return gaussian_filter(np.asarray(volume, dtype=np.float32), sigma=sigma)


def normalize_input_without_float_copy(volume):
    """Input for RegistrationTools.min_max_normalize that skips a redundant float32 copy.

    min_max_normalize(image) makes its own float32 copy (image.astype(np.float32)) and then
    subtracts the minimum and divides by (max - min) in place. Callers passed it
    np.asarray(volume, dtype=np.float32), a second full copy. For uint8 and uint16, the
    minimum and maximum are integers below 2**24, so their float32 values and the
    difference (max - min, which cannot wrap for unsigned max >= min) are exact. The
    in-place float32 arithmetic therefore sees the same numbers, and the result is the
    same. float32 input is passed through, as np.asarray already did. Other dtypes
    (signed, where max - min can overflow the dtype, and float64) keep the conversion.
    """
    arr = np.asarray(volume)
    if arr.dtype in (np.dtype(np.uint8), np.dtype(np.uint16), np.dtype(np.float32)):
        return arr
    return np.asarray(volume, dtype=np.float32)
