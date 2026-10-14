"""Full-resolution intensity value counts for the histogram bar detail view.

The preview histogram in the viewer is built from the 8-bit lossy volume, so
one preview bar spans only two or three distinct preview values. When the
user asks for detail on a bar, the viewer requests the value distribution of
the full-resolution NIfTI (stored dtype, every voxel) once per edit and then
derives sub-bin counts, mean and median for any bar range on the client.

Integer volumes are summarised exactly: ``counts[i]`` is the number of voxels
with value ``offset + i`` (a dense ``np.bincount``), so any range statistic
computed from it equals the statistic over the voxels themselves. Float
volumes are summarised with ``FLOAT_BINS`` equal-width bins over
``[min, max]``; statistics are then exact to one bin width.

Results are cached per file path and modification time, so repeated requests
(neighbouring bars, reopening the zoom) do not reload the volume. The cache
holds only these compact counts, never the volume itself.
"""

import os
import threading
from collections import OrderedDict

import nibabel as nib
import numpy as np


# Equal-width bins for floating-point volumes.
FLOAT_BINS = 65536
# Largest integer span summarised densely (covers uint8, uint16, int16).
MAX_DENSE_INTEGER_SPAN = 1 << 17
# Number of recent files whose summaries are kept.
CACHE_SIZE = 8
# Voxels per slab when counting, to bound temporary memory.
SLAB_VOXELS = 32_000_000

_cache = OrderedDict()
_cache_lock = threading.Lock()


def _slabs(data):
    """Yield contiguous slabs along axis 0 holding about SLAB_VOXELS voxels."""
    per_plane = max(1, int(np.prod(data.shape[1:])))
    step = max(1, SLAB_VOXELS // per_plane)
    for start in range(0, data.shape[0], step):
        yield np.asarray(data[start:start + step]).ravel()


def summarise_values(data):
    """Return a JSON-ready value distribution of ``data`` (see module docstring)."""
    data = np.asanyarray(data)
    n_voxels = int(data.size)
    if n_voxels == 0:
        return {'kind': 'empty', 'voxels': 0}
    lo = data.min()
    hi = data.max()

    if np.issubdtype(data.dtype, np.integer) and int(hi) - int(lo) < MAX_DENSE_INTEGER_SPAN:
        offset = int(lo)
        span = int(hi) - offset + 1
        counts = np.zeros(span, dtype=np.int64)
        for slab in _slabs(data):
            counts += np.bincount((slab.astype(np.int64) - offset), minlength=span)
        return {
            'kind': 'integer',
            'dtype': str(data.dtype),
            'voxels': n_voxels,
            'offset': offset,
            'counts': counts.tolist(),
        }

    lo_f, hi_f = float(lo), float(hi)
    if not np.isfinite(lo_f) or not np.isfinite(hi_f) or hi_f <= lo_f:
        hi_f = lo_f + 1.0
    width = (hi_f - lo_f) / FLOAT_BINS
    counts = np.zeros(FLOAT_BINS, dtype=np.int64)
    for slab in _slabs(data):
        finite = slab[np.isfinite(slab)]
        idx = np.clip(((finite - lo_f) / width).astype(np.int64), 0, FLOAT_BINS - 1)
        counts += np.bincount(idx, minlength=FLOAT_BINS)
    return {
        'kind': 'binned',
        'dtype': str(data.dtype),
        'voxels': n_voxels,
        'start': lo_f,
        'width': width,
        'counts': counts.tolist(),
    }


def value_histogram_for_file(path):
    """Cached ``summarise_values`` for a NIfTI file (keyed by path and mtime)."""
    key = (os.path.abspath(path), os.path.getmtime(path))
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    img = nib.load(path)
    # dataobj keeps the stored dtype; get_fdata() would allocate float64.
    data = np.asanyarray(img.dataobj)
    summary = summarise_values(data)
    summary['filename'] = os.path.basename(path)
    del data
    with _cache_lock:
        _cache[key] = summary
        _cache.move_to_end(key)
        while len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)
    return summary
