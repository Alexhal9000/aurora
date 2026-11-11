"""Multichannel volumes at extraction: how the channels become Aurora subjects.

Aurora subjects are single-channel 3D volumes, but a loader can hand back a 4D array:
RGB/RGBA multi-page TIFFs and TIFF folders, multichannel ImageJ/OME stacks, 4D NIfTI or
MINC, colour DICOM. Until 2026-11-10 extraction always kept channel 1 (index 0) of the
axis with the smallest extent (ExtractScansView._reduce_extra_dims_to_3d). That axis rule
and that default are unchanged; the extract preflight now reports each scan's channel
count, and the Extraction Options dialog lets the person choose, per group of scans with
the same channel count, one of these modes:

  first      Channel 1 only. The default, and the behaviour before this module.
  channel    One chosen channel (``channel`` is 1-based, as shown in the dialog).
  mean       Per-voxel arithmetic mean of all channels, rounded for integer data.
  max        Per-voxel maximum over the channels (keeps the brightest signal of any channel).
  luminance  ITU-R BT.601 luma, 0.299 R + 0.587 G + 0.114 B, the usual RGB -> grey
             conversion (3 or 4 channels; a 4th channel is taken to be alpha and ignored).
  sum        Per-voxel sum, clipped to the data type range (a saturating add: 200 + 100
             in uint8 gives 255). Unsigned integer data only.
  multiply   Per-voxel product normalised to the data type range,
             c1 * c2 * ... * cn / max^(n-1) with max the largest value of the type
             (a "multiply blend": full scale x full scale = full scale, anything x 0 = 0).
             Unsigned integer data only.
  split      Every channel becomes its own subject, linked together (beta, see below).

Every result keeps the source data type, so the usual dtype standardisation and intensity
handling (_prepare_scan_image_data) run on it exactly as on a single-channel scan. Each
extracted JSON records what was done under ``channel_extraction`` (scientific provenance).

Sum and multiply are offered for unsigned integer data only: their result is defined by the
type's fixed range (0..255, 0..65535). Float and 32-bit+ data have no fixed range; Aurora
asks for their value window in the same dialog, from preflight extrema taken over all
channels, and every other mode stays inside that window (a channel, the mean, the maximum
and the luma all lie between the smallest and the largest channel value).

Split (beta testers only, like Link Scans): the chosen channel keeps the scan's own name,
so the scan still shows as extracted next to its source file, and every other channel k is
written as the subject ``<scan>_ch<k>``. All of them get ``linked_to = <scan>``, the same
metadata Link Scans writes (pipeline/views.py LinkScansView: the main points at itself,
the others at the main), so edits on the main replay on every channel. The channels share
the scan's grid, so the equal-dimensions rule of Link Scans holds by construction.

The channel axis is the axis with the smallest extent (unchanged rule). Arrays with more
than 4 dimensions are first brought to 4D with the old rule (index 0 along the smallest
axis), so e.g. a single-timepoint T x Z x C x Y x X stack keeps its channels.

Large volumes are combined in slabs along the first spatial axis, so the float64 working
copy is never bigger than CHUNK_VOXELS voxels per channel.
"""

import numpy as np

MODES = ('first', 'channel', 'mean', 'max', 'luminance', 'sum', 'multiply', 'split')
# Modes whose result is defined by the fixed range of an unsigned integer type.
UNSIGNED_ONLY_MODES = ('sum', 'multiply')
# ITU-R BT.601 luma weights (R, G, B), the standard RGB -> greyscale conversion.
LUMA_WEIGHTS = (0.299, 0.587, 0.114)
# Voxels per channel combined at once (float64 working slab of ~8 bytes x this x channels).
CHUNK_VOXELS = 16_000_000


def reduce_to_4d(arr):
    """Bring a 5D+ array down to 4D with the old rule (index 0 along the smallest axis)."""
    out = arr
    while out.ndim > 4:
        out = np.take(out, 0, axis=int(np.argmin(out.shape)))
    return out


def channel_axis(shape):
    """The channel axis of a 4D shape (smallest extent), or None for 3D and below."""
    if len(shape) < 4:
        return None
    return int(np.argmin(shape))


def describe_channels(arr):
    """Channel facts for the extract preflight: count, axis and the shape as loaded."""
    arr4 = reduce_to_4d(np.asanyarray(arr))
    axis = channel_axis(arr4.shape)
    return {
        'channel_count': int(arr4.shape[axis]) if axis is not None else 1,
        'channel_axis': axis,
        'source_shape': [int(v) for v in np.shape(arr)],
    }


def is_unsigned_integer(dtype):
    return np.issubdtype(np.dtype(dtype), np.unsignedinteger)


def mode_allowed(mode, dtype, channel_count):
    """Whether a mode can be applied to data of this dtype and channel count."""
    if mode not in MODES:
        return False
    if mode in UNSIGNED_ONLY_MODES:
        return is_unsigned_integer(dtype)
    if mode == 'luminance':
        return channel_count in (3, 4)
    return True


def choices_by_subject(channel_options):
    """
    {scan name: {'mode', 'channel', 'channel_count'}} from the dialog's
    channel_extraction_options = {'groups': [{subjects, channel_count, mode, channel}]}.
    Scans not listed fall back to the default ('first').
    """
    choices = {}
    for group in (channel_options or {}).get('groups') or []:
        if not isinstance(group, dict):
            continue
        mode = group.get('mode') or 'first'
        if mode not in MODES:
            raise ValueError(f"Unknown channel mode: {mode}")
        for subject in group.get('subjects') or []:
            choices[subject] = {
                'mode': mode,
                'channel': int(group.get('channel') or 1),
                'channel_count': int(group.get('channel_count') or 0),
            }
    return choices


def split_subject_names(scan_name, choice):
    """Names of the extra subjects a split writes (every channel but the chosen one)."""
    if not choice or choice.get('mode') != 'split':
        return []
    keep = int(choice.get('channel') or 1)
    return [f"{scan_name}_ch{k}" for k in range(1, int(choice.get('channel_count') or 0) + 1) if k != keep]


def _to_dtype(values, dtype):
    """float64 slab -> the source dtype (rounded and clipped to the range for integers)."""
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.rint(values), info.min, info.max).astype(dtype)
    return values.astype(dtype)


def _combine_in_slabs(stack, combine):
    """Apply combine(float64 slab (C, z, ...)) -> float64 (z, ...) slab by slab."""
    out = np.empty(stack.shape[1:], dtype=stack.dtype)
    voxels_per_plane = int(np.prod(stack.shape[2:])) or 1
    step = max(1, CHUNK_VOXELS // voxels_per_plane)
    for start in range(0, stack.shape[1], step):
        slab = stack[:, start:start + step].astype(np.float64)
        out[start:start + step] = _to_dtype(combine(slab), stack.dtype)
    return out


def _combine(stack, mode, channel):
    """One 3D volume (source dtype) from the (C, ...) channel stack."""
    n = stack.shape[0]
    if mode in ('first', 'channel', 'split'):
        if not 1 <= channel <= n:
            raise ValueError(f"Channel {channel} does not exist (the scan has {n} channels).")
        return stack[channel - 1]
    if mode == 'max':
        return stack.max(axis=0)
    if mode == 'mean':
        return _combine_in_slabs(stack, lambda s: s.mean(axis=0))
    if mode == 'luminance':
        weights = np.asarray(LUMA_WEIGHTS).reshape((3,) + (1,) * (stack.ndim - 1))
        return _combine_in_slabs(stack[:3], lambda s: (s * weights).sum(axis=0))
    if mode == 'sum':
        # Clipping to the type range happens in _to_dtype (saturating add).
        return _combine_in_slabs(stack, lambda s: s.sum(axis=0))
    if mode == 'multiply':
        full_scale = float(np.iinfo(stack.dtype).max)

        def multiply(s):
            # c1 * c2 / max * c3 / max ...: the same value as prod / max^(n-1), but the
            # running product stays inside the type range (no float64 overflow for many channels).
            acc = s[0].copy()
            for c in s[1:]:
                acc *= c / full_scale
            return acc
        return _combine_in_slabs(stack, multiply)
    raise ValueError(f"Unknown channel mode: {mode}")


def channel_outputs(image_data, scan_name, choice=None):
    """
    Yield (subject name, 3D volume, JSON fields) for one loaded scan.

    3D input: yields the volume unchanged once, with no JSON fields (nothing to record).
    4D+ input: one volume combined as chosen, or one per channel for 'split'. The JSON
    fields hold 'channel_extraction' (and 'linked_to' for split) for the subject's JSON.
    A generator, so a split keeps only the source stack and one channel in memory.
    """
    if image_data is None or np.ndim(image_data) <= 3:
        yield scan_name, image_data, None
        return

    source_shape = [int(v) for v in np.shape(image_data)]
    arr = reduce_to_4d(np.asanyarray(image_data))
    axis = channel_axis(arr.shape)
    n = int(arr.shape[axis])
    stack = np.moveaxis(arr, axis, 0)  # a view: channels first, no copy

    choice = choice or {}
    mode = choice.get('mode') or 'first'
    channel = 1 if mode == 'first' else int(choice.get('channel') or 1)
    if not mode_allowed(mode, stack.dtype, n):
        raise ValueError(
            f"{scan_name}: channel mode '{mode}' cannot be used for {n} channels of {stack.dtype} data.")
    expected = int(choice.get('channel_count') or n)
    if mode == 'split' and expected != n:
        raise ValueError(
            f"{scan_name}: the scan now has {n} channels, not {expected} as analysed; extract again.")

    def record(channel_number=None, combined_mode=mode):
        fields = {
            'source_shape': source_shape,
            'channel_axis': axis,
            'channel_count': n,
            'mode': combined_mode,
        }
        if channel_number is not None:
            fields['channel'] = channel_number
        return fields

    if mode != 'split':
        print(f"{scan_name}: {n} channels (shape {source_shape}, channel axis {axis}), mode '{mode}'"
              + (f", channel {channel}" if mode in ('first', 'channel') else ''))
        uses_channel = mode in ('first', 'channel')
        yield scan_name, _combine(stack, mode, channel), {
            'channel_extraction': record(channel if uses_channel else None)}
        return

    # Split: the chosen channel keeps the scan's name (the link main), then the others.
    print(f"{scan_name}: splitting {n} channels into linked subjects; channel {channel} keeps the name")
    order = [channel] + [k for k in range(1, n + 1) if k != channel]
    for k in order:
        name = scan_name if k == channel else f"{scan_name}_ch{k}"
        fields = record(k)
        fields['source_scan'] = scan_name
        yield name, _combine(stack, 'split', k), {'channel_extraction': fields, 'linked_to': scan_name}
