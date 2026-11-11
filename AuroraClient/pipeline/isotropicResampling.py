"""Physical-space resampling of anisotropic voxels onto an isotropic grid.

Aurora stores every extracted volume on an isotropic grid (save_as_nifti writes one
voxel size for all three axes). Volumes whose voxels are thicker along exactly one
axis, with the other two equal, go through the registration-based interslice
interpolation in resampleAnisotropicVoxels.py. Every other anisotropic shape lands
here:

  * one axis finer than the other two, e.g. (0.5, 0.5, 0.25) mm;
  * all three spacings different, e.g. (0.3, 0.4, 0.6) mm;
  * small anisotropy (up to 10 %) that is not worth an interslice registration;
  * a registration-based interpolation that failed (the caller falls back here).

Target spacing
    'finest'   (default) the smallest input spacing; every other axis is upsampled.
               This is the same target the registration-based path produces.
    'coarsest' the largest input spacing; every other axis is downsampled.

Interpolation, chosen per axis (the operation is separable):
    upsampled axis, intensity volume  cubic B-spline interpolation (scipy.ndimage
                                      convention, order 3, 'reflect' boundary).
    downsampled axis                  exact area-weighted average: every output
                                      voxel is the mean of the input voxels it
                                      overlaps, each weighted by the overlapping
                                      length, so non-integer ratios need no
                                      special case.
    upsampled axis, binary volume     nearest neighbour (labels are never blended).
    downsampled axis, binary volume   area-weighted average, then each voxel takes
                                      the label that covers at least half of it.

Geometry
    Voxels are treated as cells: input voxel i along an axis of spacing s covers
    [i*s, (i+1)*s]. The output has exactly the target spacing t and
    m = round(n*s/t) voxels, and its field of view is centred on the input field of
    view, so any rounding difference (at most t/2 in total) is split evenly between
    the two ends and neither end of the volume is favoured. Samples that fall
    within that half voxel outside the input use the 'reflect' boundary (spline)
    or the edge voxel (area, nearest).

Values
    Interpolation never adds information: the output is finer, not sharper.
    Cubic B-splines can overshoot near sharp edges, so the result is clipped to
    the input's own [min, max] and integer volumes are rounded back to their dtype.
    Intermediates are float32 (exact for integers up to 2**24, which covers the
    uint8/uint16 volumes extraction produces); each axis pass accumulates in
    float64 inside a bounded slab, so peak memory is the input plus one float32
    copy at each intermediate size plus one slab.

All functions are pure numpy/scipy and do not touch the file system.
"""

import math

import numpy as np
from scipy import ndimage, sparse


# Spacings whose largest/smallest ratio is within this are treated as isotropic.
# 0.1 % keeps rounding noise in scanner headers (0.0100 vs 0.01001 mm) from
# triggering a resample; real anisotropy, even the 1-10 % the old code ignored,
# is resampled.
ISOTROPIC_TOLERANCE = 1e-3

# The registration-based path (resampleAnisotropicVoxels.py) is used only when one
# axis is at least this much thicker than the other two...
REGISTRATION_MIN_RATIO = 1.1
# ...and the other two match within this relative tolerance (the same 1 % the
# registration interpolator itself checks).
REGISTRATION_EQUAL_TOLERANCE = 0.01

TARGET_CHOICES = ('finest', 'coarsest')
DEFAULT_TARGET = 'finest'

# Cubic B-spline weights below this fraction of the largest weight are dropped from
# the sparse interpolation matrices. The prefilter's impulse response decays as
# (sqrt(3) - 2)**k, so this keeps about 21 neighbours per side; the dropped mass is
# far below float32 resolution.
_SPLINE_WEIGHT_CUTOFF = 1e-12

# Each axis pass works on slabs of at most this many input elements (float64
# temporaries of ~256 MB plus the slab's output).
_MAX_SLAB_ELEMENTS = 32 * 1024 * 1024


def normalize_target_choice(value):
    """Return 'finest' or 'coarsest'. Anything else becomes the default ('finest')."""
    value = str(value or '').strip().lower()
    return value if value in TARGET_CHOICES else DEFAULT_TARGET


def anisotropy_ratio(voxel_sizes):
    """Largest spacing divided by the smallest."""
    sizes = [float(v) for v in voxel_sizes]
    return max(sizes) / min(sizes)


def is_isotropic(voxel_sizes, tolerance=ISOTROPIC_TOLERANCE):
    """True when every spacing equals the others within ``tolerance`` (relative)."""
    return anisotropy_ratio(voxel_sizes) <= 1.0 + tolerance


def suits_registration_interpolation(voxel_sizes):
    """
    True for the shape the registration-based interslice interpolation is built for:
    exactly one axis thicker than the others by more than REGISTRATION_MIN_RATIO,
    and the other two equal within REGISTRATION_EQUAL_TOLERANCE.
    """
    sizes = [float(v) for v in voxel_sizes]
    thick_axis = int(np.argmax(sizes))
    others = [sizes[i] for i in range(3) if i != thick_axis]
    if abs(others[0] - others[1]) / min(others) > REGISTRATION_EQUAL_TOLERANCE:
        return False
    return sizes[thick_axis] / max(others) > REGISTRATION_MIN_RATIO


def classify_voxel_sizes(voxel_sizes):
    """
    Which extraction path a volume takes:
      'isotropic'    nothing to do;
      'registration' one thick axis (resampleAnisotropicVoxels.py);
      'resample'     any other anisotropy (resample_to_isotropic below).
    """
    if is_isotropic(voxel_sizes):
        return 'isotropic'
    if suits_registration_interpolation(voxel_sizes):
        return 'registration'
    return 'resample'


def target_spacing(voxel_sizes, target='finest'):
    """Isotropic output spacing for the chosen target."""
    sizes = [float(v) for v in voxel_sizes]
    return max(sizes) if normalize_target_choice(target) == 'coarsest' else min(sizes)


def resampled_shape(shape, voxel_sizes, spacing):
    """Output shape: round(n * s / t) voxels per axis, at least one."""
    return tuple(
        max(1, int(round(int(n) * float(s) / float(spacing))))
        for n, s in zip(shape, voxel_sizes)
    )


def _output_centres_in_input_index(n, s, m, t):
    """
    Centres of the m output voxels (spacing t) in input index units (input voxel i
    has its centre at index i), with the output field of view centred on the input
    one.
    """
    offset = (n * s - m * t) / 2.0
    physical_centres = offset + (np.arange(m, dtype=np.float64) + 0.5) * t
    return physical_centres / s - 0.5


def _spline_matrix(n, positions):
    """
    Sparse (m x n) matrix of cubic B-spline interpolation along one axis: row j holds
    the weights that give the value at input-index ``positions[j]`` from the n input
    samples. It equals scipy.ndimage.map_coordinates(..., order=3, mode='reflect')
    applied to unit impulses: the prefilter of the identity gives the B-spline
    coefficients of every impulse, and the cubic basis evaluated at each position
    (with the same half-sample-symmetric index reflection) combines them.
    """
    positions = np.asarray(positions, dtype=np.float64)
    m = positions.size
    if n == 1:
        return sparse.csr_matrix(np.ones((m, 1), dtype=np.float64))

    # Coefficients: column i holds the B-spline coefficients of unit impulse e_i.
    coefficients = ndimage.spline_filter1d(
        np.eye(n, dtype=np.float64), order=3, axis=0, mode='reflect')
    coefficients[np.abs(coefficients) < _SPLINE_WEIGHT_CUTOFF * np.abs(coefficients).max()] = 0.0
    coefficients = sparse.csr_matrix(coefficients)

    # Basis weights: each output position touches the 4 coefficients around it.
    base = np.floor(positions).astype(np.int64) - 1
    rows, cols, vals = [], [], []
    for k in range(4):
        idx = base + k
        distance = np.abs(positions - idx)
        weight = np.where(
            distance < 1.0,
            (4.0 - 6.0 * distance ** 2 + 3.0 * distance ** 3) / 6.0,
            np.where(distance < 2.0, (2.0 - distance) ** 3 / 6.0, 0.0),
        )
        # 'reflect' (half-sample symmetric, d c b a | a b c d | d c b a); applied
        # repeatedly so an axis shorter than the stencil still folds back inside.
        reflected = idx.copy()
        for _ in range(4):
            reflected = np.where(reflected < 0, -reflected - 1, reflected)
            reflected = np.where(reflected > n - 1, 2 * n - 1 - reflected, reflected)
        rows.append(np.arange(m))
        cols.append(reflected)
        vals.append(weight)
    basis = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(m, n))
    matrix = (basis @ coefficients).tocsr()
    matrix.data[np.abs(matrix.data) < _SPLINE_WEIGHT_CUTOFF] = 0.0
    matrix.eliminate_zeros()
    return matrix


def _area_matrix(n, s, m, t):
    """
    Sparse (m x n) matrix of the exact area-weighted average along one axis: output
    voxel j covers [offset + j*t, offset + (j+1)*t] and averages the input voxels
    [i*s, (i+1)*s] it overlaps, weighted by the overlapping length. The part of an
    edge output voxel that lies outside the input field of view (at most t/2) is
    left out of its average, which is the same as repeating the edge voxel.
    """
    offset = (n * s - m * t) / 2.0
    extent = n * s
    rows, cols, vals = [], [], []
    for j in range(m):
        start = max(offset + j * t, 0.0)
        stop = min(offset + (j + 1) * t, extent)
        if stop <= start:
            # Output voxel fully outside the input (only possible when m*t exceeds
            # n*s by a whole voxel, which rounding never produces); use the edge voxel.
            rows.append(j)
            cols.append(0 if start <= 0.0 else n - 1)
            vals.append(1.0)
            continue
        first = int(math.floor(start / s))
        last = min(int(math.ceil(stop / s)) - 1, n - 1)
        weights = []
        for i in range(first, last + 1):
            overlap = min(stop, (i + 1) * s) - max(start, i * s)
            if overlap > 0.0:
                weights.append((i, overlap))
        total = sum(w for _, w in weights)
        for i, w in weights:
            rows.append(j)
            cols.append(i)
            vals.append(w / total)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(m, n))


def _nearest_matrix(n, positions):
    """Sparse (m x n) nearest-neighbour selection: the input voxel containing each centre."""
    positions = np.asarray(positions, dtype=np.float64)
    source = np.clip(np.floor(positions + 0.5).astype(np.int64), 0, n - 1)
    m = positions.size
    return sparse.csr_matrix((np.ones(m), (np.arange(m), source)), shape=(m, n))


def _axis_matrix(n, s, t, is_binary):
    """Interpolation matrix and its name for one axis, or (None, 'unchanged')."""
    m = max(1, int(round(n * s / t)))
    if m == n and abs(s - t) <= 1e-9 * t:
        return None, 'unchanged'
    if s > t:  # input voxels coarser than the target: upsample
        positions = _output_centres_in_input_index(n, s, m, t)
        if is_binary:
            return _nearest_matrix(n, positions), 'nearest neighbour'
        return _spline_matrix(n, positions), 'cubic B-spline'
    return _area_matrix(n, s, m, t), 'area-weighted average'


def _apply_axis(volume, axis, matrix):
    """
    Apply an (m x n) matrix along ``axis`` of a 3D volume, slab by slab along the
    largest other axis, and return a new float32 volume.
    """
    m, n = matrix.shape
    out_shape = list(volume.shape)
    out_shape[axis] = m
    output = np.empty(out_shape, dtype=np.float32)

    other_axes = [a for a in range(3) if a != axis]
    slab_axis = max(other_axes, key=lambda a: volume.shape[a])
    per_index = max(1, volume.size // max(1, volume.shape[slab_axis]))
    step = max(1, _MAX_SLAB_ELEMENTS // per_index)

    for start in range(0, volume.shape[slab_axis], step):
        stop = min(start + step, volume.shape[slab_axis])
        selector = [slice(None)] * 3
        selector[slab_axis] = slice(start, stop)
        slab = np.moveaxis(volume[tuple(selector)], axis, 0)
        rest_shape = slab.shape[1:]
        flat = np.asarray(slab, dtype=np.float64).reshape(n, -1)
        result = matrix @ flat
        output[tuple(selector)] = np.moveaxis(result.reshape((m,) + rest_shape), 0, axis)
    return output


def resample_to_isotropic(image_data, voxel_sizes, target='finest', is_binary=False):
    """
    Resample a 3D volume with spacings ``voxel_sizes`` (one per array axis, in mm)
    onto an isotropic grid.

    Returns (resampled_volume, isotropic_spacing, details). The volume keeps the
    input dtype; ``details`` describes what was done, for the scan's metadata.
    """
    data = np.asarray(image_data)
    if data.ndim != 3:
        raise ValueError(f"Isotropic resampling needs a 3D volume, got shape {data.shape}")
    sizes = [float(v) for v in voxel_sizes]
    if len(sizes) != 3 or any(not math.isfinite(v) or v <= 0.0 for v in sizes):
        raise ValueError(f"Voxel sizes must be three positive numbers, got {voxel_sizes}")
    if data.size == 0:
        raise ValueError("Cannot resample an empty volume")

    target = normalize_target_choice(target)
    spacing = target_spacing(sizes, target)
    source_min = data.min()
    source_max = data.max()
    binary_values = None
    if is_binary:
        binary_values = (float(source_min), float(source_max))

    # Shrinking axes first and growing axes last keeps the intermediates small.
    order = sorted(range(3), key=lambda a: sizes[a] / spacing)
    working = data
    per_axis = [None, None, None]
    for axis in order:
        matrix, method = _axis_matrix(working.shape[axis], sizes[axis], spacing, is_binary)
        per_axis[axis] = method
        if matrix is not None:
            working = _apply_axis(working, axis, matrix)

    if working is data:
        result = data  # every axis already at the target spacing
    elif is_binary:
        low, high = binary_values
        result = np.where(working >= (low + high) / 2.0, high, low).astype(data.dtype)
    else:
        np.clip(working, float(source_min), float(source_max), out=working)
        if np.issubdtype(data.dtype, np.integer):
            np.rint(working, out=working)
        result = working.astype(data.dtype, copy=False)
    working = None

    details = {
        'method': 'physical_space_resampling',
        'target': target,
        'interpolation_per_axis': per_axis,
        'field_of_view': 'centred on the input field of view',
        'values': ('binary labels kept (nearest neighbour / majority by area)'
                   if is_binary else 'clipped to the input range'),
        'original_shape': [int(v) for v in data.shape],
        'final_shape': [int(v) for v in result.shape],
    }
    return result, spacing, details
