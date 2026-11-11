"""
Nikon NIS-Elements ``.nd2`` files (confocal and widefield stacks) as one scan ("Nikon ND2").

ND2 is the native format of Nikon NIS-Elements (A1/AX/C2 confocals, spinning disk,
widefield, Ti2 and others share it). It is read with the ``nd2`` package (Talley Lambert,
BSD-3-Clause, pure Python; the same reader napari and Bio-Formats users rely on). Only
``nd2.ND2File`` is used, which needs numpy alone: its OME/dask extras are never imported
here. Legacy (pre-2015, JPEG2000) files additionally need ``imagecodecs``, which Aurora
already ships.

What becomes the Aurora scan
----------------------------
An ND2 file is a set of experiment loops (time ``T``, XY positions ``P``, Z stack ``Z``,
custom ``U``) around frames of ``Y x X`` pixels with ``C`` channels (and ``S`` = 3 RGB
components for colour cameras). The scan is the Z stack of the first time point and the
first XY position; any other loop index is dropped and reported as a warning, never
merged silently.

The array is returned as (z, y, x) for one channel, or (z, y, x, channels) for several,
the same layout as a multichannel multi-page TIFF. Extraction then lets the person pick,
combine or split the channels (channelCombination, Extraction Options > Channels), and
the channel names and wavelengths are kept in the provenance so the choice can be made
knowingly. RGB components count as channels (luminance is offered for 3 of them).

channelCombination takes the axis with the smallest extent as the channel axis. A stack
with no more slices than channels (a single plane, or 3 slices of 4 channels) would make
it take the slices for channels, so such files are refused with a clear message instead
of being cut along the wrong axis.

Voxel size (scientific rigour: no guessed millimetres, same policy as SkyScan)
------------------------------------------------------------------------------
* x/y: the pixel calibration NIS-Elements stores (``volume.axesCalibration``, microns,
  includes the objective, zoom and binning). An uncalibrated file is refused: a default
  size would silently give wrong millimetres.
* z: the Z stack loop step (``stepUm``, also what ``nd2`` reports as the z calibration).
  A stack without a usable step is refused. A single plane has no z extent and is
  refused above anyway.
* Spacings are returned as (x, y, z) in mm, like ``_extract_3d_voxel_sizes`` and the other
  slice-stack importers; views maps them to array axis order (z, y, x) before the
  anisotropic-voxel step (isotropicResampling). Confocal z steps are usually coarser than
  the pixel size, so most ND2 stacks go through that resampler.
* Not corrected (recorded so a reader of the JSON knows): the axial scale error from a
  refractive-index mismatch between immersion medium and sample (focal shift). Aurora
  uses the nominal stage step, as NIS-Elements, Fiji and Bio-Formats do.

Slice order follows the file (frame order of the Z loop), as in Fiji/Bio-Formats; the
loop's ``bottomToTop`` flag is recorded in the provenance, nothing is flipped.

Everything used is returned in ``scanner_metadata['nd2']`` (provenance, warnings).
"""

from __future__ import annotations

import numpy as np

SUBTYPE = "Nikon ND2"
EXTENSIONS = (".nd2",)

# Loop axes that the scan keeps. Every other loop (T, P, U...) is reduced to index 0.
KEPT_LOOP_AXES = ("Z",)


class ND2ImportError(ValueError):
    """The file cannot be imported without guessing (no calibration, missing frames...)."""


def _nd2_module():
    """Import nd2 lazily, so a missing package only affects ND2 scans, not engine start."""
    try:
        import nd2  # noqa: PLC0415 (lazy on purpose)
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ND2ImportError(
            "Reading .nd2 files needs the 'nd2' Python package, which is not installed in this "
            "Aurora environment. Update Aurora (or run: pip install nd2)."
        ) from exc
    return nd2


def _safe(value):
    """JSON-friendly scalar (numpy numbers and enums become plain Python values)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _channel_provenance(f) -> list:
    """Name, wavelengths and optics of each channel, as NIS-Elements recorded them."""
    channels = []
    try:
        for ch in (f.metadata.channels or []):
            meta, scope = ch.channel, ch.microscope
            channels.append({
                'index': _safe(meta.index),
                'name': _safe(meta.name),
                'emission_nm': _safe(meta.emissionLambdaNm),
                'excitation_nm': _safe(meta.excitationLambdaNm),
                'objective': _safe(getattr(scope, 'objectiveName', None)),
                'objective_magnification': _safe(getattr(scope, 'objectiveMagnification', None)),
                'numerical_aperture': _safe(getattr(scope, 'objectiveNumericalAperture', None)),
                'immersion_refractive_index': _safe(getattr(scope, 'immersionRefractiveIndex', None)),
                'zoom': _safe(getattr(scope, 'zoomMagnification', None)),
                'pinhole_um': _safe(getattr(scope, 'pinholeDiameterUm', None)),
                'modality': [_safe(m) for m in (getattr(scope, 'modalityFlags', None) or [])],
            })
    except Exception:  # metadata is informative only; never block the import on it
        pass
    return channels


def _z_loop(f):
    """The Z stack loop of the experiment, or None."""
    for loop in (f.experiment or []):
        if getattr(loop, 'type', None) == 'ZStackLoop':
            return loop
    return None


def load_nd2(path: str) -> tuple:
    """Load the first time point / XY position of an ND2 file as one Aurora volume.

    Returns (image_data, voxel_sizes_mm (x, y, z), scanner_metadata):
      image_data is (z, y, x) for one channel, (z, y, x, channels) for several, in the
      file's own dtype (12-bit cameras arrive as uint16).
    Raises ND2ImportError when the file cannot be imported safely.
    """
    nd2 = _nd2_module()
    warnings = []

    with nd2.ND2File(path) as f:
        attrs = f.attributes
        sizes = dict(f.sizes)
        height = int(attrs.heightPx)
        n_channels = int(attrs.channelCount or 1)
        n_components = int(attrs.componentCount or n_channels) // n_channels  # 3 for RGB
        channels_total = n_channels * n_components

        # ---- Which frames make up the scan: Z of T=0, P=0, U=0 ----
        loop_indices = f.loop_indices
        n_stored = int(getattr(attrs, 'sequenceCount', None) or len(loop_indices))
        dropped = {axis: count for axis, count in sizes.items()
                   if axis not in KEPT_LOOP_AXES + ('C', 'S', 'Y', 'X') and count > 1}
        if dropped:
            labels = {'T': 'time points', 'P': 'XY positions', 'U': 'custom loop steps'}
            described = ', '.join(f"{count} {labels.get(axis, axis + ' steps')}" for axis, count in dropped.items())
            warnings.append(
                f"The file has {described}; Aurora imported only the first of each "
                "(time point 1, XY position 1).")

        frames = []  # (z index, frame index)
        for frame_index, coords in enumerate(loop_indices):
            if any(value != 0 for axis, value in coords.items() if axis not in KEPT_LOOP_AXES):
                continue
            frames.append((int(coords.get('Z', 0)), frame_index))
        frames.sort()
        n_slices = len(frames)
        if n_slices == 0:
            raise ND2ImportError("The file has no image frames.")
        missing = [z for z, frame_index in frames if frame_index >= n_stored]
        if missing:
            raise ND2ImportError(
                f"The Z stack is incomplete: {len(missing)} of {n_slices} slices were never recorded "
                "(acquisition stopped early?). A partial stack would have the wrong extent.")

        if n_slices <= channels_total:
            raise ND2ImportError(
                f"The stack has {n_slices} slice(s) and {channels_total} channel(s). Aurora needs more slices "
                "than channels to tell them apart (and a single plane has no z size); this is not a 3D scan "
                "Aurora can use.")

        # ---- Voxel size: refuse rather than guess ----
        try:
            calibrated = tuple(bool(v) for v in f.metadata.channels[0].volume.axesCalibrated)
        except Exception:
            calibrated = (False, False, False)
        vs = f.voxel_size()  # microns (x, y, z)
        x_um, y_um = float(vs.x), float(vs.y)
        if not (calibrated[0] and calibrated[1]) or x_um <= 0 or y_um <= 0:
            raise ND2ImportError(
                "The file has no pixel calibration (NIS-Elements 'Uncalibrated'), so its voxel size is "
                "unknown. Calibrate the image in NIS-Elements (or export with the objective set) and try again.")

        z_loop = _z_loop(f)
        z_step = abs(float(z_loop.parameters.stepUm)) if z_loop is not None else 0.0
        if z_step <= 0:
            raise ND2ImportError(
                "The Z stack has no step size, so the spacing between slices is unknown.")
        if abs(float(vs.z) - z_step) > 1e-6 * max(z_step, 1.0) and float(vs.z) not in (0.0, 1.0):
            warnings.append(
                f"The z calibration ({float(vs.z):g} um) differs from the Z loop step ({z_step:g} um); "
                "the Z loop step is used.")

        # ---- Read the frames: (C, Y, X, S) each -> (z, y, x, channels) ----
        first = np.asarray(f.read_frame(frames[0][1]))
        dtype = first.dtype
        width = first.size // (height * n_channels * n_components)
        frame_shape = (n_channels, height, width, n_components)
        volume = np.empty((n_slices, height, width, channels_total), dtype=dtype)

        def place(k, frame):
            frame = np.asarray(frame).reshape(frame_shape)
            # (C, Y, X, S) -> (Y, X, C, S) -> (Y, X, C*S): channel-major, components inside.
            volume[k] = frame.transpose(1, 2, 0, 3).reshape(height, width, channels_total)

        place(0, first)
        for k, (_, frame_index) in enumerate(frames[1:], start=1):
            place(k, f.read_frame(frame_index))

        provenance = {
            'reader': f"nd2 {getattr(nd2, '__version__', '?')}",
            'legacy_format': bool(f.is_legacy),
            'sizes': {k: int(v) for k, v in sizes.items()},
            'imported': {'Z': n_slices, 'channels': channels_total,
                         'time_point': 0, 'xy_position': 0},
            'dropped_loops': {k: int(v) for k, v in dropped.items()},
            'rgb_components': n_components if n_components > 1 else None,
            'pixel_size_um': [x_um, y_um],
            'z_step_um': z_step,
            'z_source': 'ZStackLoop stepUm',
            'z_bottom_to_top': _safe(getattr(z_loop.parameters, 'bottomToTop', None)),
            'z_not_corrected_for': 'refractive-index mismatch (axial focal shift); nominal stage step used',
            'bits_per_component_significant': _safe(getattr(attrs, 'bitsPerComponentSignificant', None)),
            'channels': _channel_provenance(f),
            'warnings': warnings,
        }

    if channels_total == 1:
        volume = volume[..., 0]

    voxel_sizes = (x_um / 1000.0, y_um / 1000.0, z_step / 1000.0)
    scanner_metadata = {
        'file_info': {'shape': list(volume.shape), 'dtype': str(volume.dtype), 'format': SUBTYPE},
        'geometry': {
            'voxel_size_x_mm': voxel_sizes[0],
            'voxel_size_y_mm': voxel_sizes[1],
            'voxel_size_z_mm': voxel_sizes[2],
        },
        'nd2': provenance,
    }
    print(f"ND2 loaded: shape={volume.shape} {volume.dtype}, voxel sizes (x, y, z) {voxel_sizes} mm")
    return volume, voxel_sizes, scanner_metadata
