"""Load NIfTI voxels in their stored dtype without the float64 detour.

Aurora loads scans and masks almost everywhere as

    data = img.get_fdata().astype(img.get_data_dtype())          # scans
    data = np.round(img.get_fdata()).astype(mask_dtype)          # masks

`get_fdata()` first builds a float64 copy of the whole volume (8 bytes per voxel, 8x a
uint8 scan) and nibabel keeps that copy cached on the image object for as long as the
object lives. The cast back then makes a second copy. For a large uint8 micro-CT volume
that load alone is ~9x the volume size, and the cached float64 copy stays in memory
through the whole tool (see the 2026-11-09 memory audit).

`nifti_voxels(img, dtype)` returns exactly what `img.get_fdata().astype(dtype)` returns
(same values, dtype, shape, memory order, a private writable ndarray) but reads the stored
array directly when that is provably identical:

- the file is read through nibabel's array proxy (a file on disk, not an in-memory image),
- the header applies no scaling (scl_slope 1 / scl_inter 0, which is what nibabel reports
  for the NaN/0 "no scaling" header fields every Aurora writer leaves), and
- casting the stored values straight to `dtype` gives the same result as going through
  float64 first: same dtype; bool; a float target from any float or <=32-bit integer
  (float64 holds those exactly, so there is only one rounding step either way); or an
  integer target that can hold every value of the stored integer type.

Anything else (scaled files, narrowing casts such as uint16 -> uint8 whose out-of-range
behaviour differs between the integer and the float route, float -> integer, and 64-bit
integer files, whose values above 2**53 the float64 route rounds) falls back to the old
expression itself, only without leaving nibabel's float64 cache behind
(`caching='unchanged'`). Either way the numbers are the old numbers.

`round_values=True` reproduces `np.round(img.get_fdata()).astype(dtype)`; rounding is a
no-op on integer storage, so the fast path is only taken for integer files.
"""

import numpy as np
import nibabel as nib

# 64-bit integers are not exact in float64 (above 2**53), so the old route through float64
# can change them; those files always take the old route to stay identical.
_WIDE_INTEGERS = (np.dtype(np.int64), np.dtype(np.uint64))


def _direct_cast_matches_float64_route(stored, target):
    """True when stored.astype(target) == stored.astype(float64).astype(target) for every
    value the stored dtype can hold."""
    if stored in _WIDE_INTEGERS:
        return False
    if target == stored:
        return True
    if target == np.dtype(bool):
        # Non-zero test either way (NaN is non-zero in both).
        return True
    if target.kind == 'f':
        if stored.kind == 'f':
            return True
        return stored.kind in 'iub'
    if target.kind in 'iu' and stored.kind in 'iub':
        return bool(np.can_cast(stored, target, 'safe'))
    return False


def nifti_voxels(img, dtype=None, round_values=False):
    """Voxel array of `img`, identical to `img.get_fdata().astype(dtype)`.

    Args:
        img: nibabel image (from nib.load or crashSafety.load_nifti_readonly).
        dtype: target dtype; None means the stored dtype (img.get_data_dtype()).
        round_values: reproduce `np.round(img.get_fdata()).astype(dtype)` (mask loads).

    Returns:
        numpy.ndarray owned by the caller (writable, not a memory map, not shared with
        the image), so in-place edits behave exactly as they did on the old copy.
    """
    stored = np.dtype(img.get_data_dtype())
    target = stored if dtype is None else np.dtype(dtype)
    dataobj = img.dataobj
    direct = (
        nib.is_proxy(dataobj)
        and getattr(dataobj, 'slope', None) == 1.0
        and getattr(dataobj, 'inter', None) == 0.0
        and np.dtype(getattr(dataobj, 'dtype', stored)) == stored
        and _direct_cast_matches_float64_route(stored, target)
        and (not round_values or stored.kind in 'iub')
    )
    if direct:
        data = np.asanyarray(dataobj)
        if type(data) is not np.ndarray or not data.flags.writeable:
            # Uncompressed .nii files come back memory-mapped; the old path always
            # returned a private in-memory array.
            data = np.array(data)
        if data.dtype != target:
            data = data.astype(target)
        return data
    data = img.get_fdata(caching='unchanged')
    if round_values:
        data = np.round(data)
    return data.astype(target)
