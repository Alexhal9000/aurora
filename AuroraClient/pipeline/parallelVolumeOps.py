"""Multi-threaded, exact replacements for two slow 3D steps (Aurora 1.9.4+).

1. ``binary_closing_ball(mask, radius)`` is used by edge detection (edgeTools.py). It
   gives the same result as ``skimage.morphology.binary_closing(mask, ball(radius))``.
   skimage dilates and then erodes with the ball footprint itself, so every voxel visits
   every footprint voxel. That costs O(N * r^3) on one core: on a 200^3 volume r=3 took
   0.6 s, r=6 took 4.2 s and r=10 took 21 s. Here both steps come from Euclidean
   distance transforms instead, whose cost does not grow with the radius:

       dilation(A)(x) = 1  <=>  some voxel of A lies within distance r of x
                            <=>  EDT(x to nearest A voxel)       <= r
       erosion(D)(x)  = 1  <=>  no voxel outside D lies within distance r of x
                            <=>  EDT(x to nearest non-D voxel)   >  r

   ``ball(r)`` is the set of integer offsets with dx^2 + dy^2 + dz^2 <= r^2, and both
   transforms measure exact distances between voxel centres, so the two definitions
   pick the same voxels. A squared distance between voxel centres is an integer, and
   the square root of a perfect square is exact in floating point, so the ``<= r``
   tests are exact at the ball surface too. The borders also follow skimage: dilation
   sees nothing beyond the volume, while erosion counts the outside as foreground
   (skimage ``mode='ignore'``; an EDT only finds background inside the array).

   The volume is cut into slabs along the first axis. Each slab is processed with a
   halo of 2r voxels on each cut side: a closed voxel depends on the dilation within r
   of it, and that depends on the mask within r of those voxels. The slabs are then
   independent and run in threads, since SciPy's EDT releases the GIL. Memory is
   bounded per slab (the EDT needs about 20 bytes per voxel of slab plus halo), not
   per volume.

2. ``equalize_adapthist(image, kernel_size, clip_limit, nbins)`` is used by restoration
   (restoreTools.py, CLAHE). It is skimage 0.25's ``equalize_adapthist`` (grayscale
   path) with the same arithmetic in the same order, so results are bit-identical.
   Only the scheduling changes:
     - The per-tile histograms come from one ``np.bincount`` over (tile, bin) pairs
       instead of a Python ``apply_along_axis`` loop over tiles, which was very slow
       for small kernels (many tiles).
     - The multilinear interpolation, which takes most of the time, runs over chunks
       of tiles in threads (NumPy's fancy indexing and arithmetic release the GIL).
       skimage built full-volume int64 and float64 temporaries for this step (more
       than 20 bytes per voxel); here they exist for one chunk at a time.
   ``clip_histogram`` and ``map_histogram`` are skimage's own functions. RGB-like inputs
   (last axis of length 3 or 4, which skimage converts through HSV) go straight to
   skimage.

Threads: one per logical core, capped by free RAM. AURORA_VOLUME_THREADS=N forces N
threads (1 = serial; the results are the same either way). AURORA_FAST_VOLUME_OPS=0
calls skimage directly, as before.
"""

from __future__ import annotations

import math
import numbers
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import ndimage as ndi

# Never more threads than this: the steps are memory-bound and gain little beyond it.
_MAX_THREADS = 16
# Rough working memory per voxel of one slab or chunk (EDT: int32 feature indices for
# 3 axes + float64 distances; CLAHE: int64 mapped values + float64 products), used only
# to keep the thread count down when free RAM is short.
_BYTES_PER_WORK_VOXEL = 24


def fast_ops_enabled():
    """False when AURORA_FAST_VOLUME_OPS=0 asks for the plain skimage calls."""
    return os.environ.get("AURORA_FAST_VOLUME_OPS", "1").strip() != "0"


def plan_threads(n_tasks, work_voxels_per_task=0):
    """Number of threads for n_tasks independent pieces on this machine.

    One per logical core, capped by _MAX_THREADS, by n_tasks and by free RAM (each running
    piece needs about work_voxels_per_task * _BYTES_PER_WORK_VOXEL bytes). Overridden by
    AURORA_VOLUME_THREADS.
    """
    forced = os.environ.get("AURORA_VOLUME_THREADS", "").strip()
    if forced:
        try:
            threads = max(1, int(forced))
        except ValueError:
            threads = 1
    else:
        threads = min(_MAX_THREADS, os.cpu_count() or 1)
        if work_voxels_per_task > 0:
            try:
                import psutil
                free = psutil.virtual_memory().available
                threads = min(threads, max(1, int(free * 0.7 // (work_voxels_per_task * _BYTES_PER_WORK_VOXEL))))
            except Exception:
                pass
    return max(1, min(threads, n_tasks))


def _run(fn, items, threads):
    """fn over items, in order; in threads when threads > 1."""
    if threads <= 1 or len(items) <= 1:
        return [fn(it) for it in items]
    with ThreadPoolExecutor(max_workers=threads) as ex:
        return list(ex.map(fn, items))


# ---------------------------------------------------------------------------
# Binary closing with a ball footprint (edge detection)
# ---------------------------------------------------------------------------

def _close_block(block, radius):
    """Closing of one block, exact at the block's own borders (skimage border rules)."""
    if not block.any():
        # Nothing to dilate: the closing is empty (erosion of an empty set).
        return np.zeros(block.shape, dtype=bool)
    # Distance from each voxel to the nearest mask voxel (EDT of the complement).
    dilated = ndi.distance_transform_edt(~block) <= radius
    if dilated.all():
        # No background inside the block, and the outside counts as foreground.
        return dilated
    # Distance from each voxel to the nearest background voxel of the dilation.
    return ndi.distance_transform_edt(dilated) > radius


def binary_closing_ball(mask, radius):
    """Same result as skimage.morphology.binary_closing(mask, footprint=ball(radius)).

    mask: 3D array (any dtype; nonzero = foreground). radius: int >= 0. Returns bool.
    See the module docstring for why this is exact and how it is split into threads.
    """
    from skimage import morphology

    radius = int(radius)
    mask = np.asarray(mask)
    if not fast_ops_enabled() or mask.ndim != 3 or radius < 1:
        return morphology.binary_closing(mask, footprint=morphology.ball(radius))
    mask = mask.astype(bool, copy=False)
    n0 = mask.shape[0]
    halo = 2 * radius
    plane = mask.shape[1] * mask.shape[2]

    # Slab thickness: aim for several slabs per thread, but keep the halo (which is
    # computed twice) small next to the slab itself.
    threads = plan_threads(n0, plane * (max(1, n0 // 8) + 2 * halo))
    if threads <= 1:
        return _close_block(mask, radius)
    thickness = max(4 * halo, int(math.ceil(n0 / (threads * 2))))
    if thickness >= n0:
        return _close_block(mask, radius)
    starts = list(range(0, n0, thickness))
    threads = plan_threads(len(starts), plane * (thickness + 2 * halo))

    out = np.empty(mask.shape, dtype=bool)

    def work(s):
        e = min(n0, s + thickness)
        lo, hi = max(0, s - halo), min(n0, e + halo)
        closed = _close_block(mask[lo:hi], radius)
        out[s:e] = closed[s - lo:s - lo + (e - s)]

    _run(work, starts, threads)
    return out


# ---------------------------------------------------------------------------
# CLAHE (restoration)
# ---------------------------------------------------------------------------

def equalize_adapthist(image, kernel_size=None, clip_limit=0.01, nbins=256):
    """skimage.exposure.equalize_adapthist, bit-identical, multi-threaded and lighter.

    Same arguments and return value (float image rescaled to [0, 1]).
    """
    from skimage import exposure

    image = np.asarray(image)
    rgb_like = image.ndim == 3 and image.shape[-1] in (3, 4)
    if not fast_ops_enabled() or rgb_like or image.ndim < 2:
        return exposure.equalize_adapthist(image, kernel_size=kernel_size,
                                           clip_limit=clip_limit, nbins=nbins)

    # --- skimage.exposure._adapthist.equalize_adapthist, grayscale path, unchanged ---
    # These are skimage internals (checked against 0.25.0); if a future skimage moves
    # them, fall back to the plain skimage call rather than fail the restoration.
    try:
        from skimage._shared.utils import _supported_float_type
        from skimage.util import img_as_uint
        from skimage.exposure import rescale_intensity
        from skimage.exposure._adapthist import NR_OF_GRAY, clip_histogram, map_histogram  # noqa: F401
    except ImportError:
        return exposure.equalize_adapthist(image, kernel_size=kernel_size,
                                           clip_limit=clip_limit, nbins=nbins)

    float_dtype = _supported_float_type(image.dtype)
    image = img_as_uint(image)
    image = np.round(rescale_intensity(image, out_range=(0, NR_OF_GRAY - 1))).astype(
        np.min_scalar_type(NR_OF_GRAY)
    )

    if kernel_size is None:
        kernel_size = tuple([max(s // 8, 1) for s in image.shape])
    elif isinstance(kernel_size, numbers.Number):
        kernel_size = (kernel_size,) * image.ndim
    elif len(kernel_size) != image.ndim:
        raise ValueError(f'Incorrect value of `kernel_size`: {kernel_size}')

    kernel_size = [int(k) for k in kernel_size]

    image = _clahe(image, kernel_size, clip_limit, nbins)
    image = image.astype(float_dtype, copy=False)
    return rescale_intensity(image)


def _clahe(image, kernel_size, clip_limit, nbins):
    """skimage.exposure._adapthist._clahe with batched histograms and threaded,
    chunked interpolation. Comments marked "skimage:" follow the original's steps."""
    from skimage.exposure._adapthist import NR_OF_GRAY, clip_histogram, map_histogram

    ndim = image.ndim
    dtype = image.dtype

    # skimage: pad so each dimension is a multiple of the kernel size and starts with
    # half a kernel.
    pad_start_per_dim = [k // 2 for k in kernel_size]
    pad_end_per_dim = [
        (k - s % k) % k + int(np.ceil(k / 2.0))
        for k, s in zip(kernel_size, image.shape)
    ]
    image = np.pad(
        image,
        [[p_i, p_f] for p_i, p_f in zip(pad_start_per_dim, pad_end_per_dim)],
        mode='reflect',
    )

    # skimage: gray value bins.
    bin_size = 1 + NR_OF_GRAY // nbins
    lut = np.arange(NR_OF_GRAY, dtype=np.min_scalar_type(NR_OF_GRAY))
    lut //= bin_size
    image = lut[image]

    # skimage: one contextual region (tile) per row.
    ns_hist = [int(s / k) - 1 for s, k in zip(image.shape, kernel_size)]
    hist_blocks_shape = np.array([ns_hist, kernel_size]).T.flatten()
    hist_blocks_axis_order = np.array(
        [np.arange(0, ndim * 2, 2), np.arange(1, ndim * 2, 2)]
    ).flatten()
    hist_slices = [slice(k // 2, k // 2 + n * k) for k, n in zip(kernel_size, ns_hist)]
    hist_blocks = image[tuple(hist_slices)].reshape(hist_blocks_shape)
    hist_blocks = np.transpose(hist_blocks, axes=hist_blocks_axis_order)
    hist_block_assembled_shape = hist_blocks.shape
    n_tiles = math.prod(ns_hist)
    tile_elements = math.prod(kernel_size)

    kernel_elements = math.prod(kernel_size)
    if clip_limit > 0.0:
        clim = int(np.clip(clip_limit * kernel_elements, 1, None))
    else:
        clim = kernel_elements

    # Histograms of all tiles, a chunk of tiles at a time: bincount of tile * nbins + bin.
    # Equal to skimage's per-tile np.bincount(minlength=nbins), since every bin index is
    # below nbins. (skimage makes the same (tiles, tile_elements) copy with this reshape.)
    hist_blocks = hist_blocks.reshape((n_tiles, -1))
    hist = np.empty((n_tiles, nbins), dtype=np.int64)
    tiles_per_chunk = max(1, (1 << 22) // max(1, tile_elements))
    hist_starts = list(range(0, n_tiles, tiles_per_chunk))

    def count(a):
        b = min(n_tiles, a + tiles_per_chunk)
        ids = hist_blocks[a:b].astype(np.int64)
        ids += (np.arange(a, b, dtype=np.int64) * nbins - a * nbins)[:, None]
        hist[a:b] = np.bincount(ids.reshape(-1), minlength=(b - a) * nbins).reshape(b - a, nbins)

    _run(count, hist_starts, plan_threads(len(hist_starts), tiles_per_chunk * tile_elements))
    del hist_blocks
    # skimage: clip and redistribute each tile's histogram (same function, same order).
    for t in range(n_tiles):
        clip_histogram(hist[t], clip_limit=clim)
    hist = map_histogram(hist, 0, NR_OF_GRAY - 1, kernel_elements)
    hist = hist.reshape(hist_block_assembled_shape[:ndim] + (-1,))

    # skimage: duplicate leading mappings in each dimension.
    map_array = np.pad(hist, [[1, 1] for _ in range(ndim)] + [[0, 0]], mode='edge')

    # skimage: rearrange the image into blocks (one per row) for the interpolation.
    ns_proc = [int(s / k) for s, k in zip(image.shape, kernel_size)]
    blocks_shape = np.array([ns_proc, kernel_size]).T.flatten()
    blocks_axis_order = np.array(
        [np.arange(0, ndim * 2, 2), np.arange(1, ndim * 2, 2)]
    ).flatten()
    blocks = image.reshape(blocks_shape)
    blocks = np.transpose(blocks, axes=blocks_axis_order)
    blocks_flattened_shape = blocks.shape
    blocks = np.reshape(blocks, (math.prod(ns_proc), math.prod(blocks.shape[ndim:])))
    del image

    # skimage: interpolation coefficients.
    coeffs = np.meshgrid(
        *tuple([np.arange(k) / k for k in kernel_size[::-1]]), indexing='ij'
    )
    coeffs = [np.transpose(c).flatten() for c in coeffs]
    inv_coeffs = [1 - c for dim, c in enumerate(coeffs)]
    edges = list(np.ndindex(*([2] * ndim)))
    edge_maps_all = []
    edge_coeffs_all = []
    for edge in edges:
        edge_maps = map_array[tuple([slice(e, e + n) for e, n in zip(edge, ns_proc)])]
        edge_maps_all.append(edge_maps.reshape((math.prod(ns_proc), -1)))
        edge_coeffs_all.append(np.prod(
            [[inv_coeffs, coeffs][e][d] for d, e in enumerate(edge[::-1])], 0
        ))

    # skimage: sum the contributions of the neighbouring regions, in the same edge order
    # and dtypes, so each block's result is identical. Rows (blocks) are independent and
    # are done in chunks, in threads.
    n_blocks, block_elements = blocks.shape
    result = np.empty(blocks.shape, dtype=dtype)
    # About 4M voxels per chunk keeps the temporaries small; at least a few chunks per thread.
    rows_per_chunk = max(1, min(n_blocks, (1 << 22) // max(1, block_elements)))
    threads = plan_threads(max(1, n_blocks // rows_per_chunk), rows_per_chunk * block_elements)
    rows_per_chunk = max(1, min(rows_per_chunk, int(math.ceil(n_blocks / (threads * 4)))))
    starts = list(range(0, n_blocks, rows_per_chunk))

    def work(s):
        e = min(n_blocks, s + rows_per_chunk)
        b = blocks[s:e]
        acc = np.zeros(b.shape, dtype=np.float32)
        for edge_maps, edge_coeffs in zip(edge_maps_all, edge_coeffs_all):
            edge_mapped = np.take_along_axis(edge_maps[s:e], b, axis=-1)
            acc += (edge_mapped * edge_coeffs).astype(acc.dtype)
        result[s:e] = acc.astype(dtype)

    _run(work, starts, threads)
    del blocks

    # skimage: rebuild the image from blocks and undo the padding.
    result = result.reshape(blocks_flattened_shape)
    blocks_axis_rebuild_order = np.array(
        [np.arange(0, ndim), np.arange(ndim, ndim * 2)]
    ).T.flatten()
    result = np.transpose(result, axes=blocks_axis_rebuild_order)
    padded_shape = [n * k for n, k in zip(ns_proc, kernel_size)]
    result = result.reshape(padded_shape)
    unpad_slices = tuple(
        [
            slice(p_i, s - p_f)
            for p_i, p_f, s in zip(pad_start_per_dim, pad_end_per_dim, padded_shape)
        ]
    )
    return result[unpad_slices]
