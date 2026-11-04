"""
Bruker SkyScan / NRecon reconstruction folders as one scan ("SkyScan Reconstruction").

A reconstruction folder holds one image per section, named ``<prefix>_rec<number>.<ext>``
(JPG, BMP, PNG or TIF, as chosen in NRecon), the reconstruction log (``<prefix>_rec.log``)
and extras such as the ``<prefix>_spr.bmp`` sinogram preview, which is not a slice.

Why a dedicated importer and not "any folder of JPEGs" (decided 2026-11-04 with
Alejandro): a plain image folder has no trustworthy voxel size, so guessing one would
silently give wrong millimetres. Here the voxel size comes only from the NRecon log, and
the import refuses when it cannot be read:

* Pixel size: ``Pixel Size (um)`` in the log's ``[Reconstruction]`` section. This is the
  reconstructed voxel and already accounts for binning/resizing in NRecon. The
  acquisition value (``Image Pixel Size (um)`` in ``[Acquisition]``) is deliberately NOT
  used as a fallback: after a resized reconstruction it is the wrong size.
* Section spacing: ``Section to Section Step`` sections of one pixel each (assumed 1, and
  recorded as assumed, when the log has no such line).
* Slices: one prefix, one file type, evenly numbered, all the same size. Gaps, mixed sets
  or colour images are refused instead of being stacked into a misleading volume.

Everything used is returned as provenance (stored in the scan JSON under
``scanner_metadata.skyscan``), including warnings: JPEG slices are lossy and 8-bit, and a
section count that differs from the log (an interrupted copy, for example) is reported.

Array layout matches the other slice-folder importers (Zeiss TIFF folder, DICOM): slices
stacked on axis 0; voxel sizes are returned as (x, y, z) like ``_extract_3d_voxel_sizes``.
"""

from __future__ import annotations

import concurrent.futures
import os
import re

import numpy as np
from PIL import Image

SUBTYPE = "SkyScan Reconstruction"

# <prefix>_rec<digits>.<ext>; the sinogram preview (<prefix>_spr.*) does not match.
SLICE_RE = re.compile(r"^(?P<prefix>.+)_rec(?P<index>\d+)\.(?P<ext>jpe?g|bmp|png|tiff?)$", re.IGNORECASE)
LOSSY_EXTENSIONS = {"jpg", "jpeg"}

# Log keys (compared case-insensitively, inside [Reconstruction]).
KEY_PIXEL_SIZE = "pixel size (um)"
KEY_STEP = "section to section step"
KEY_SECTIONS = "sections count"
PROVENANCE_KEYS = {
    "reconstruction program": "reconstruction_program",
    "program version": "program_version",
    "result file type": "result_file_type",
    "first section": "first_section",
    "last section": "last_section",
}
SYSTEM_KEYS = {"scanner": "scanner", "instrument s/n": "instrument_serial"}


class SkyScanImportError(ValueError):
    """The folder cannot be imported safely; the message says why (shown to the person)."""


def parse_log(path: str) -> dict:
    """Read an NRecon/SkyScan log into {section: {key: value}} with lower-case names."""
    sections: dict = {}
    current = ""
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith("[") and line.endswith("]"):
                current = line[1:-1].strip().lower()
                sections.setdefault(current, {})
            elif "=" in line:
                key, value = line.split("=", 1)
                # First value wins: later duplicates (rare) never override what was read.
                sections.setdefault(current, {}).setdefault(key.strip().lower(), value.strip())
    return sections


def _positive_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 and np.isfinite(number) else None


def _reconstruction_log(scan_folder: str, prefix: str) -> tuple:
    """The one usable reconstruction log: (filename, parsed). Several logs are fine only
    when they agree on pixel size and step; otherwise the import is refused."""
    candidates = []
    for name in sorted(os.listdir(scan_folder)):
        path = os.path.join(scan_folder, name)
        if not (name.lower().endswith(".log") and os.path.isfile(path)):
            continue
        parsed = parse_log(path)
        recon = parsed.get("reconstruction", {})
        if _positive_float(recon.get(KEY_PIXEL_SIZE)) is not None:
            candidates.append((name, parsed))
    if not candidates:
        raise SkyScanImportError(
            "No NRecon reconstruction log with 'Pixel Size (um)' in its [Reconstruction] section was found in "
            "this folder, so the voxel size is unknown. Copy the reconstruction .log (usually "
            f"{prefix}_rec.log) into the folder next to the slices and try again.")
    # Prefer the log named after the slices when there is one.
    preferred = [c for c in candidates if c[0].lower() in (f"{prefix}_rec.log".lower(), f"{prefix}.log".lower())]
    chosen = preferred[0] if preferred else candidates[0]
    # Compared as numbers, so "10.00007" and "10.000070" agree.
    signature = lambda c: (_positive_float(c[1]["reconstruction"].get(KEY_PIXEL_SIZE)),
                           _positive_float(c[1]["reconstruction"].get(KEY_STEP, "1")))
    disagreeing = [c[0] for c in candidates if signature(c) != signature(chosen)]
    if disagreeing:
        raise SkyScanImportError(
            f"The logs in this folder disagree on the voxel size ({chosen[0]} vs {', '.join(disagreeing)}). "
            "Keep only the log of this reconstruction and try again.")
    return chosen


def find_slices(scan_folder: str) -> tuple:
    """(prefix, ext, [(index, filename), ...] sorted) for the folder's reconstructed slices."""
    groups: dict = {}
    for name in os.listdir(scan_folder):
        match = SLICE_RE.match(name)
        if match and os.path.isfile(os.path.join(scan_folder, name)):
            key = (match.group("prefix"), match.group("ext").lower().replace("jpeg", "jpg").replace("tiff", "tif"))
            groups.setdefault(key, []).append((int(match.group("index")), name))
    if not groups:
        raise SkyScanImportError("No reconstructed slices (<name>_rec0000.jpg, .bmp, .png or .tif) in this folder.")
    if len(groups) > 1:
        sets = ", ".join(f"{p}_rec*.{e} ({len(v)})" for (p, e), v in sorted(groups.items()))
        raise SkyScanImportError(
            f"This folder holds more than one set of slices ({sets}). Keep one reconstruction per folder.")
    (prefix, ext), slices = next(iter(groups.items()))
    slices.sort()
    return prefix, ext, slices


def _read_slice(path: str) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode in ("RGB", "RGBA"):
            array = np.asarray(image.convert("RGB"))
            # NRecon writes greyscale; an RGB file is only accepted when it is grey.
            if not (np.array_equal(array[..., 0], array[..., 1]) and np.array_equal(array[..., 0], array[..., 2])):
                raise SkyScanImportError(f"{os.path.basename(path)} is a colour image, not a reconstructed slice.")
            return array[..., 0]
        if image.mode == "P":
            return np.asarray(image.convert("L"))
        return np.asarray(image)


def load_reconstruction(scan_folder: str) -> tuple:
    """Load a SkyScan reconstruction folder.

    Returns (image_data (slices, rows, cols), voxel_sizes_mm (x, y, z), scanner_metadata).
    Raises SkyScanImportError when the folder cannot be imported safely.
    """
    prefix, ext, slices = find_slices(scan_folder)
    log_name, log = _reconstruction_log(scan_folder, prefix)
    recon = log.get("reconstruction", {})
    warnings = []

    pixel_um = _positive_float(recon.get(KEY_PIXEL_SIZE))
    step_raw = recon.get(KEY_STEP)
    if step_raw is None:
        step, step_source = 1, "assumed 1 (no 'Section to Section Step' in the log)"
    else:
        try:
            step = int(float(step_raw))
        except ValueError:
            step = 0
        if step < 1:
            raise SkyScanImportError(f"'Section to Section Step={step_raw}' in {log_name} is not a whole number of sections.")
        step_source = f"{log_name} [Reconstruction] Section to Section Step"

    indices = [index for index, _ in slices]
    numbering_step = 1
    if len(indices) > 1:
        gaps = {b - a for a, b in zip(indices, indices[1:])}
        numbering_step = min(gaps)
        if len(gaps) != 1:
            missing = [a for a, b in zip(indices, indices[1:]) if b - a != min(gaps)][:5]
            raise SkyScanImportError(
                f"The slice numbers have gaps (after {', '.join(map(str, missing))}); slices are missing, so the "
                "spacing between them would be wrong. Copy the complete reconstruction and try again.")
        if numbering_step > 1 and step_raw is None:
            # Every n-th section saved but the log does not say the step: the z size is unknown.
            raise SkyScanImportError(
                f"The slices are numbered every {numbering_step} sections, but the log has no 'Section to Section "
                "Step', so the spacing between slices is unknown. Use the log written with this reconstruction.")
        if numbering_step not in (1, step):
            warnings.append(
                f"Slice numbers advance by {numbering_step} but the log's Section to Section Step is {step}; "
                f"the z size uses the log ({step} x pixel size).")

    expected = _positive_float(recon.get(KEY_SECTIONS))
    if expected is not None and int(expected) != len(slices):
        warnings.append(
            f"The log reports {int(expected)} sections but the folder has {len(slices)} slices "
            "(was the copy interrupted, or only part of the reconstruction saved?).")
    if ext in LOSSY_EXTENSIONS:
        warnings.append(
            "Slices are JPEG: lossy and 8-bit, so fine grey-level detail was lost when NRecon saved them. "
            "Prefer TIF/BMP/PNG output from NRecon for new reconstructions.")

    paths = [os.path.join(scan_folder, name) for _, name in slices]
    first = _read_slice(paths[0])
    if first.ndim != 2:
        raise SkyScanImportError(f"{slices[0][1]} is not a 2D greyscale slice.")
    volume = np.empty((len(paths),) + first.shape, dtype=first.dtype)
    volume[0] = first

    def fill(k):
        array = _read_slice(paths[k])
        if array.shape != first.shape or array.dtype != first.dtype:
            raise SkyScanImportError(
                f"{slices[k][1]} is {array.shape[1]}x{array.shape[0]} {array.dtype}, unlike the first slice "
                f"({first.shape[1]}x{first.shape[0]} {first.dtype}).")
        volume[k] = array

    # PIL releases the GIL while decoding, so threads decode in parallel into one array.
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
        for _ in pool.map(fill, range(1, len(paths))):
            pass

    pixel_mm = pixel_um / 1000.0
    voxel_sizes = (pixel_mm, pixel_mm, pixel_mm * step)

    system = log.get("system", {})
    provenance = {
        "log_file": log_name,
        "pixel_size_um": pixel_um,
        "voxel_size_mm": list(voxel_sizes),
        "voxel_size_source": f"{log_name} [Reconstruction] Pixel Size (um)",
        "section_step": step,
        "section_step_source": step_source,
        "slice_numbering_step": numbering_step,
        "slice_prefix": prefix,
        "slice_format": ext,
        "slice_count": len(slices),
        "first_index": indices[0],
        "last_index": indices[-1],
        "dtype": str(volume.dtype),
        "lossy_source": ext in LOSSY_EXTENSIONS,
        "warnings": warnings,
    }
    for key, name in PROVENANCE_KEYS.items():
        if key in recon:
            provenance[name] = recon[key]
    for key, name in SYSTEM_KEYS.items():
        if key in system:
            provenance[name] = system[key]

    for warning in warnings:
        print(f"\033[93mSkyScan import warning ({prefix}): {warning}\033[0m")
    print(f"SkyScan import {prefix}: {len(slices)} {ext.upper()} slices, voxel {pixel_um} um "
          f"(step {step}) from {log_name}")
    return volume, voxel_sizes, {"skyscan": provenance}
