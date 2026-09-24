import os
import sys
import json
import time
import base64
import shutil
import glob
import io
import re
import concurrent.futures
import tempfile
import threading
from datetime import datetime

import pandas as pd

import psutil
import numpy as np
import nibabel as nib
import imageio
import aim2numpy
import trimesh
import open3d as o3d
import gc  # Add garbage collection

from django.shortcuts import render
from django.http import FileResponse
from rest_framework import viewsets, status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser

from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

from .batch_flag_filter import load_flagged_subject_names, apply_flag_filter, normalize_flag_filter_value
from .linkedScans import filter_out_linked_children, get_link_group, get_same_shape_siblings

from scipy import ndimage, signal, spatial
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import convolve

from skimage import exposure, measure, transform
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu

from PIL import Image, ImageDraw, ImageFont
from pydicom import dcmread

from .ALPACA import ALPACA
from .registrationTools import RegistrationTools
from .meshGridTools import PreservedMeshEditWriter, strip_ephemeral_scan_metadata
from .coordinateFrames import (
    mesh_centroid_local_to_reference_prism_mm,
    mesh_local_to_shared_mm,
    mesh_reference_prism_corner_mm,
    resolve_mesh_reference_prism,
    swap_voxel_volume_xz,
)
from .smartLandmarking import SmartLandmarkingView
from .preprocessingTools import *

import matplotlib
matplotlib.use('Agg') # Use non-interactive backend suitable for web servers
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Required for 3D plotting

# Set the number of threads for ITK to utilize
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(psutil.cpu_count(logical=True))

alpaca = ALPACA()


def has_elastic_registration(scan_name, directory):
    """
    Check if a scan has completed elastic registration (latest edit is elastic).
    
    Args:
        scan_name (str): Name of the scan to check
        directory (str): Root project directory containing 'extracted' folder (atlas is at top level)
    
    Returns:
        bool: True if scan has elastic registration as latest edit, False otherwise
    """
    try:
        # Check if the scan is in the atlas (special case - atlas is at top level, not in extracted)
        if scan_name == "atlas":
            # Assume atlas as always elastic
            return True
        
        # For all other scans, they're in the extracted folder
        # Load .json metadata to check if the scan is marked as the reference scan, if so, check if there are any elastic edits on any of the subjects on the project, if yes, this means that the reference scan has been used in elastic registration and thus must be locked in by returning True.
        json_path = os.path.join(directory, "extracted", scan_name, f"{scan_name}.json")
        with open(json_path, 'r') as jf:
            metadata = json.load(jf)
            if metadata.get('reference', False):
                # Check if there are any elastic_to in the json metadata on any of the subjects on the project matching the scan_name
                for subject in os.listdir(os.path.join(directory, "extracted")):
                    if subject == "project_settings.json" or not os.path.isdir(os.path.join(directory, "extracted", subject)):
                        continue
                    json_path = os.path.join(directory, "extracted", subject, f"{subject}.json")
                    if os.path.isfile(json_path):  # Safety check
                        with open(json_path, 'r') as jf:
                            metadata = json.load(jf)
                            if metadata.get('elastic_to', "none") == scan_name:
                                return True
        
        scan_path = os.path.join(directory, "extracted", scan_name)
        edit_pattern = os.path.join(scan_path, f"{scan_name}_edit_*_*.nii.gz")
        
        # Return False if scan directory doesn't exist
        if not os.path.isdir(scan_path):
            return False
        
        # Get all edit files
        edit_files = glob.glob(edit_pattern)
        
        # If no edit files exist, there's no elastic registration
        if not edit_files:
            return False
        
        # Extract edit numbers from all files
        edit_numbers = []
        for file in edit_files:
            try:
                num_str = file.split('_edit_')[1].split('_')[0]
                edit_numbers.append(int(num_str))
            except (IndexError, ValueError):
                continue
        
        # If no valid edit numbers found, no elastic registration
        if not edit_numbers:
            return False
        
        # Get the latest edit number
        latest_edit_num = max(edit_numbers)
        
        # Check if the latest edit is elastic
        latest_edit_pattern = os.path.join(scan_path, f"{scan_name}_edit_{latest_edit_num}_*elastic.nii.gz")
        
        # Return True only if elastic file exists for latest edit
        return len(glob.glob(latest_edit_pattern)) > 0
        
    except Exception as e:
        print(f"Error checking elastic registration for {scan_name}: {e}")
        return False


def _extracted_scan_marked_faulty(directory, scan_name):
    """True if extracted/<scan>/<scan>.json exists and marks the scan as faulty."""
    json_path = os.path.join(directory, "extracted", scan_name, f"{scan_name}.json")
    if not os.path.isfile(json_path):
        return False
    try:
        with open(json_path, "r") as jf:
            metadata = json.load(jf)
        return bool(metadata.get("faulty", False))
    except (json.JSONDecodeError, OSError):
        return False


def first_aligned_edit_number(scan_dir, scan_name):
    """Smallest rigid ``_aligned`` edit number, or None if none exist."""
    aligned_nums = []
    for pattern in (
        f'{scan_name}_edit_*_aligned.nii.gz',
        f'{scan_name}_edit_*_aligned.ply',
        f'{scan_name}_lossy_edit_*_aligned.nii.gz',
    ):
        for path in glob.glob(os.path.join(scan_dir, pattern)):
            m = re.search(r'_edit_(\d+)_aligned\.', os.path.basename(path))
            if m:
                aligned_nums.append(int(m.group(1)))
    return min(aligned_nums) if aligned_nums else None


def voxel_size_for_edit_stem(metadata, edit_stem, scan_name, scan_dir=None):
    """
    Voxel size (mm) that applies to a specific edit.

    Alignment overwrites the global ``voxel_size`` to the reference spacing but
    stores the subject's native spacing in ``alignment_previous_voxel_size``.
    Pre-alignment volumes (raw and edits before the first ``_aligned``) must
    still be interpreted at that native spacing — otherwise meshes inflate and
    landmarks appear shrunken.
    """
    try:
        current = float(metadata.get('voxel_size') or 1.0)
    except (TypeError, ValueError):
        current = 1.0
    if not np.isfinite(current) or current <= 0:
        current = 1.0

    previous = metadata.get('alignment_previous_voxel_size')
    try:
        previous = float(previous) if previous is not None else None
    except (TypeError, ValueError):
        previous = None
    if previous is None or not np.isfinite(previous) or previous <= 0:
        return current

    stem = scan_name if not edit_stem else normalize_threshold_edit_stem(edit_stem, scan_name)
    first_aligned = first_aligned_edit_number(scan_dir, scan_name) if scan_dir else None

    if first_aligned is None:
        if stem == scan_name:
            return previous
        if '_aligned' in stem or '_elastic' in stem:
            return current
        return previous

    if stem == scan_name:
        return previous
    m = re.search(r'_edit_(\d+)_', stem)
    if not m:
        return previous
    if int(m.group(1)) < first_aligned:
        return previous
    return current


def resolve_pre_alignment_landmark_stem(scan_dir, scan_name):
    """
    Landmark stem for the volume before the first rigid ``_aligned`` edit.

    Edit 0 aligned → raw / ``{scan_name}_landmarks.json``.
    Edit N>0 aligned → stem of the ``edit_{N-1}_*`` volume (lossy stripped).
    No aligned edit (e.g. reference) → raw slot ``{scan_name}``.
    """
    first_aligned = first_aligned_edit_number(scan_dir, scan_name)
    if first_aligned is None:
        return scan_name
    if first_aligned == 0:
        return scan_name

    prev = first_aligned - 1
    candidates = []
    for pattern in (
        f'{scan_name}_edit_{prev}_*.nii.gz',
        f'{scan_name}_edit_{prev}_*.ply',
        f'{scan_name}_lossy_edit_{prev}_*.nii.gz',
    ):
        for path in glob.glob(os.path.join(scan_dir, pattern)):
            base = os.path.basename(path)
            if 'backup' in base or 'mask' in base:
                continue
            stem = (
                base.replace('_lossy', '')
                .replace('.nii.gz', '')
                .replace('.ply', '')
            )
            if stem and stem not in candidates:
                candidates.append(stem)

    return candidates[0] if candidates else scan_name


def normalize_threshold_edit_stem(edit, scan_name):
    """Normalize an edit filename/stem to the landmark/old_thresholds key."""
    if edit in (None, '', 'raw', 'null', 'undefined'):
        return scan_name
    stem = os.path.basename(str(edit)).replace('_lossy', '').replace('.nii.gz', '').replace('.ply', '')
    return stem if stem else scan_name


def list_subject_threshold_edit_stems(scan_dir, scan_name):
    """
    All edit/raw stems that should have an ``old_thresholds`` entry.
    Always includes raw ``scan_name``, plus every ``_edit_N_*`` volume/mesh stem.
    """
    stems = {scan_name}
    for pattern in (
        f'{scan_name}_edit_*_*.nii.gz',
        f'{scan_name}_edit_*_*.ply',
        f'{scan_name}_lossy_edit_*_*.nii.gz',
    ):
        for path in glob.glob(os.path.join(scan_dir, pattern)):
            base = os.path.basename(path)
            if 'backup' in base or 'mask' in base:
                continue
            stem = normalize_threshold_edit_stem(base, scan_name)
            if stem and stem != scan_name:
                stems.add(stem)
    return sorted(stems)


def upsert_old_threshold(metadata, edit_stem, threshold):
    """Upsert ``{edit, threshold}`` into metadata['old_thresholds']."""
    try:
        thr = int(round(float(threshold)))
    except (TypeError, ValueError):
        return metadata
    entries = metadata.get('old_thresholds')
    if not isinstance(entries, list):
        entries = []
    updated = False
    new_entries = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get('edit') == edit_stem:
            new_entries.append({'edit': edit_stem, 'threshold': thr})
            updated = True
        else:
            new_entries.append({
                'edit': entry.get('edit'),
                'threshold': int(round(float(entry.get('threshold', thr)))),
            })
    if not updated:
        new_entries.append({'edit': edit_stem, 'threshold': thr})
    metadata['old_thresholds'] = new_entries
    return metadata


def remove_old_threshold(metadata, edit_stem):
    """Remove any ``old_thresholds`` entry for ``edit_stem``."""
    entries = metadata.get('old_thresholds')
    if not isinstance(entries, list):
        metadata['old_thresholds'] = []
        return metadata
    metadata['old_thresholds'] = [
        entry for entry in entries
        if isinstance(entry, dict) and entry.get('edit') != edit_stem
    ]
    return metadata


def rename_old_threshold_edit(metadata, old_stem, new_stem):
    """Rename an ``old_thresholds`` edit key (e.g. elastic un-bump)."""
    if not old_stem or not new_stem or old_stem == new_stem:
        return metadata
    entries = metadata.get('old_thresholds')
    if not isinstance(entries, list):
        metadata['old_thresholds'] = []
        return metadata
    renamed = []
    saw_new = False
    moved_thr = None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        stem = entry.get('edit')
        if stem == old_stem:
            moved_thr = entry.get('threshold')
            continue
        if stem == new_stem:
            saw_new = True
        renamed.append(entry)
    if moved_thr is not None:
        if saw_new:
            for entry in renamed:
                if entry.get('edit') == new_stem:
                    try:
                        entry['threshold'] = int(round(float(moved_thr)))
                    except (TypeError, ValueError):
                        pass
                    break
        else:
            try:
                renamed.append({
                    'edit': new_stem,
                    'threshold': int(round(float(moved_thr))),
                })
            except (TypeError, ValueError):
                pass
    metadata['old_thresholds'] = renamed
    return metadata


def find_old_threshold(metadata, edit_stem):
    """Return the stored int threshold for ``edit_stem``, or None if absent."""
    entries = metadata.get('old_thresholds')
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if isinstance(entry, dict) and entry.get('edit') == edit_stem:
            try:
                return int(round(float(entry.get('threshold'))))
            except (TypeError, ValueError):
                return None
    return None


def get_old_threshold(metadata, edit_stem, default=None):
    """Return stored threshold for edit_stem, or default (global threshold if None)."""
    if default is None:
        default = metadata.get('threshold')
    found = find_old_threshold(metadata, edit_stem)
    return found if found is not None else default


def snapshot_previous_edit_threshold(metadata, previous_stem, threshold=None):
    """
    Persist the pre-mutation threshold onto ``previous_stem`` in ``old_thresholds``.

    Used when creating threshold-changing edits so delete can restore cleanly.
    """
    if threshold is None:
        threshold = metadata.get('threshold')
    return upsert_old_threshold(metadata, previous_stem, threshold)


def append_background_offset_history(metadata, edit_stem, mode, shift):
    """Append or update per-edit background offset provenance for reporting."""
    try:
        shift_int = int(shift)
    except (TypeError, ValueError):
        return metadata
    mode_str = str(mode or '').strip() or 'unknown'
    stem = str(edit_stem or '').strip()
    if not stem:
        return metadata
    history = metadata.get('background_offset_history')
    if not isinstance(history, list):
        history = []
    kept = [
        entry
        for entry in history
        if not (isinstance(entry, dict) and entry.get('edit') == stem)
    ]
    kept.append({'edit': stem, 'mode': mode_str, 'shift': shift_int})
    metadata['background_offset_history'] = kept
    return metadata


def restore_threshold_after_deleted_edit(metadata, previous_stem, legacy_restore=None):
    """
    Restore global ``threshold`` after deleting an edit.

    Prefers ``old_thresholds[previous_stem]``; falls back to ``legacy_restore(metadata)``
    for projects that only have shift / histogram_matched_old_threshold fields.
    """
    stored = find_old_threshold(metadata, previous_stem)
    if stored is not None:
        metadata['threshold'] = stored
        return True
    if callable(legacy_restore):
        legacy_restore(metadata)
    return False


def ensure_old_thresholds(metadata, scan_dir, scan_name):
    """
    Normalize ``old_thresholds`` to a list for load-time lookups.

    Missing per-edit entries are intentionally left absent so callers fall back
    to the global ``threshold`` (legacy projects). Entries are created by
    pipeline snapshots and by user saves on non-latest edits.
    """
    if not isinstance(metadata.get('old_thresholds'), list):
        metadata['old_thresholds'] = []
    return metadata


def latest_edit_threshold_stem(scan_dir, scan_name, *, exclude_elastic=False):
    """Landmark stem for the latest edit volume, or scan_name when none exist."""
    best = None
    best_num = -1
    for pattern in (
        f'{scan_name}_edit_*_*.nii.gz',
        f'{scan_name}_edit_*_*.ply',
        f'{scan_name}_lossy_edit_*_*.nii.gz',
    ):
        for path in glob.glob(os.path.join(scan_dir, pattern)):
            base = os.path.basename(path)
            if 'backup' in base or 'mask' in base:
                continue
            if exclude_elastic and 'elastic' in base:
                continue
            m = re.search(r'_edit_(\d+)_', base)
            if not m:
                continue
            num = int(m.group(1))
            stem = normalize_threshold_edit_stem(base, scan_name)
            if num > best_num:
                best_num = num
                best = stem
    return best if best is not None else scan_name


def latest_non_elastic_edit_threshold_stem(scan_dir, scan_name):
    """Latest edit stem excluding elastic registration edits (else raw)."""
    return latest_edit_threshold_stem(scan_dir, scan_name, exclude_elastic=True)


def resolve_edit_nifti_path(scan_dir, scan_name, edit_stem):
    """Resolve a NIfTI path for distance/mesh work on an edit stem."""
    candidates = [
        os.path.join(scan_dir, f'{edit_stem}.nii.gz'),
    ]
    if edit_stem == scan_name:
        candidates.append(os.path.join(scan_dir, f'{scan_name}_lossy.nii.gz'))
    elif edit_stem.startswith(f'{scan_name}_edit_'):
        lossy = edit_stem.replace(f'{scan_name}_edit_', f'{scan_name}_lossy_edit_', 1)
        candidates.append(os.path.join(scan_dir, f'{lossy}.nii.gz'))
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def unpack_landmark_entry(landmark):
    """Return ``(xyz ndarray, landmark_type)`` from array or ``{position, landmark_type}``."""
    landmark_type = 'main'
    if isinstance(landmark, dict):
        landmark_type = landmark.get('landmark_type', 'main') or 'main'
        position = landmark.get('position', landmark)
        if isinstance(position, dict):
            if all(axis in position for axis in ('x', 'y', 'z')):
                position = [position['x'], position['y'], position['z']]
            else:
                raise ValueError(f'Landmark position dict missing x/y/z: {position}')
        return np.asarray(position, dtype=np.float64), landmark_type
    return np.asarray(landmark, dtype=np.float64), landmark_type


def landmarks_to_positions_array(landmarks):
    """Normalize landmark JSON / arrays to an (N, 3) float64 positions array."""
    if landmarks is None:
        return None
    if isinstance(landmarks, np.ndarray):
        return np.asarray(landmarks, dtype=np.float64)
    positions = []
    for lm in landmarks:
        xyz, _ = unpack_landmark_entry(lm)
        positions.append(xyz)
    return np.asarray(positions, dtype=np.float64)


def integer_canvas_embed_shift(source_centroid, reference_centroid):
    """
    Voxel copy used when embedding a rotated volume onto the reference canvas.

    The volume paste is ``canvas[i] = source[i - (int(ref) - int(src))]``,
    not a continuous ``- src + ref`` snap. Landmarks must use the same
    integer shift or they drift by the fractional-centroid residual
    (up to ~1 voxel per axis, which looks like buried vs floating points).
    """
    source_centroid = np.asarray(source_centroid, dtype=np.float64).reshape(3)
    reference_centroid = np.asarray(reference_centroid, dtype=np.float64).reshape(3)
    return np.array([
        int(reference_centroid[0]) - int(source_centroid[0]),
        int(reference_centroid[1]) - int(source_centroid[1]),
        int(reference_centroid[2]) - int(source_centroid[2]),
    ], dtype=np.float64)


def canvas_embed_shift_from_crop_or_centroids(params, source_centroid=None, reference_centroid=None):
    """Prefer the stored in/out crop window; fall back to integer centroid shift."""
    crop = (params or {}).get('alignment_crop_indices') or {}
    in_idx = crop.get('in')
    out_idx = crop.get('out')
    if isinstance(in_idx, dict) and isinstance(out_idx, dict):
        try:
            return np.array([
                int(in_idx['x'][0]) - int(out_idx['x'][0]),
                int(in_idx['y'][0]) - int(out_idx['y'][0]),
                int(in_idx['z'][0]) - int(out_idx['z'][0]),
            ], dtype=np.float64)
        except (KeyError, TypeError, ValueError, IndexError):
            pass
    if source_centroid is None:
        source_centroid = (params or {}).get('alignment_rotation_center_voxel')
    if reference_centroid is None:
        reference_centroid = (params or {}).get('alignment_canvas_centroid_voxel')
    return integer_canvas_embed_shift(source_centroid, reference_centroid)


def _largest_voxel_connected_component(mask):
    """Keep only the largest 3-D connected component of a binary voxel mask."""
    labeled, num_features = ndimage.label(np.asarray(mask, dtype=bool))
    if num_features == 0:
        return np.asarray(mask, dtype=bool)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return labeled == int(sizes.argmax())


def _content_mask_from_threshold(volume, threshold, sigma=1.0):
    """Geometry-only mask: Gaussian smooth, threshold, largest connected component."""
    smoothed = gaussian_filter(np.asarray(volume, dtype=np.float32), sigma=float(sigma))
    binary = smoothed > float(threshold)
    return _largest_voxel_connected_component(binary)


def _cap_mask_bbox_to_extent(low, high, centroid, max_extent, volume_shape):
    """Clip an oversized mask bbox (e.g. scanner artifact) to a reference-scaled extent."""
    low = np.asarray(low, dtype=np.int64).reshape(3)
    high = np.asarray(high, dtype=np.int64).reshape(3)
    centroid = np.asarray(centroid, dtype=np.float64).reshape(3)
    extent = (high - low).astype(np.float64)
    max_extent = np.asarray(max_extent, dtype=np.float64).reshape(3)
    if np.all(extent <= max_extent):
        return low, high
    half = np.minimum(extent, max_extent) / 2.0
    new_low = np.floor(centroid - half).astype(np.int64)
    new_high = np.ceil(centroid + half).astype(np.int64)
    shape = np.asarray(volume_shape, dtype=np.int64).reshape(3)
    return np.maximum(new_low, 0), np.minimum(new_high, shape)


def _centroid_of_mask(mask):
    coords = np.argwhere(np.asarray(mask, dtype=bool))
    if coords.size == 0:
        return np.asarray(mask.shape, dtype=np.float64) / 2.0
    return coords.mean(axis=0).astype(np.float64)


def _mask_bbox(mask):
    coords = np.argwhere(np.asarray(mask, dtype=bool))
    if coords.size == 0:
        shape = np.asarray(mask.shape, dtype=np.int64)
        return np.zeros(3, dtype=np.int64), shape.copy()
    low = coords.min(axis=0).astype(np.int64)
    high = (coords.max(axis=0) + 1).astype(np.int64)
    return low, high


def _round_shape_even(shape):
    return tuple(int(((int(s) + 1) // 2) * 2) for s in shape)


def _round_shape_multiple(shape, multiple):
    multiple = max(1, int(multiple))
    return tuple(int(((int(s) + multiple - 1) // multiple) * multiple) for s in shape)


def choose_estimation_stride(canvas_shape, max_estimation_voxels=40_000_000):
    """
    Pick stride in {2, 3, 4} so the downsampled estimation grid stays under
    *max_estimation_voxels*. Returns (stride, rounded_canvas_shape, estimation_shape).
    """
    raw_shape = tuple(int(s) for s in canvas_shape)
    max_voxels = int(max(1, max_estimation_voxels))
    for stride in (2, 3, 4):
        rounded = _round_shape_multiple(raw_shape, stride)
        est_shape = tuple(rounded[i] // stride for i in range(3))
        if int(np.prod(est_shape)) <= max_voxels:
            return stride, rounded, est_shape
    rounded = _round_shape_multiple(raw_shape, 4)
    est_shape = tuple(rounded[i] // 4 for i in range(3))
    return 4, rounded, est_shape


def build_union_registration_canvas(
    reference_centroid,
    moving_centroid,
    ref_content_low,
    ref_content_high,
    mov_content_low,
    mov_content_high,
    margin_voxels,
    max_estimation_voxels=40_000_000,
    rotation_margin_voxels=0,
):
    """
    Reversible union canvas sized from content-bbox overlap plus margin (not full volumes).
    Reference at ref_paste, moving centroid-aligned at mov_paste. Canvas axes are rounded to
    a multiple of the chosen estimation stride so [::stride] downsampling is exact.
    """
    margin = int(max(0, margin_voxels))
    rotation_margin = int(max(0, rotation_margin_voxels))
    ref_c = np.asarray(reference_centroid, dtype=np.float64).reshape(3)
    mov_c = np.asarray(moving_centroid, dtype=np.float64).reshape(3)
    ref_content_low = np.asarray(ref_content_low, dtype=np.int64).reshape(3)
    ref_content_high = np.asarray(ref_content_high, dtype=np.int64).reshape(3)
    mov_content_low = np.asarray(mov_content_low, dtype=np.int64).reshape(3)
    mov_content_high = np.asarray(mov_content_high, dtype=np.int64).reshape(3)

    mov_paste = np.round(ref_c - mov_c).astype(np.int64)
    ref_paste = np.zeros(3, dtype=np.int64)

    ref_content_lo = ref_paste + ref_content_low
    ref_content_hi = ref_paste + ref_content_high
    mov_content_lo = mov_paste + mov_content_low
    mov_content_hi = mov_paste + mov_content_high

    min_corner = np.minimum(ref_content_lo, mov_content_lo) - margin - rotation_margin
    max_corner = np.maximum(ref_content_hi, mov_content_hi) + margin + rotation_margin

    canvas_offset = -min_corner
    ref_paste = ref_paste + canvas_offset
    mov_paste = mov_paste + canvas_offset
    raw_shape = tuple(int(max_corner[i] - min_corner[i]) for i in range(3))
    estimation_stride, canvas_shape, estimation_shape = choose_estimation_stride(
        raw_shape, max_estimation_voxels=max_estimation_voxels
    )

    return {
        "canvas_shape": canvas_shape,
        "ref_paste": ref_paste,
        "mov_paste": mov_paste,
        "centroid_delta": np.round(ref_c - mov_c).astype(np.int64),
        "estimation_stride": estimation_stride,
        "estimation_shape": estimation_shape,
    }


def paste_volume_on_canvas(volume, paste_at, canvas_shape, background_value):
    """Paste *volume* at *paste_at* on a background canvas — no cropping."""
    volume = np.asarray(volume)
    canvas = np.full(canvas_shape, background_value, dtype=volume.dtype)
    paste_at = np.asarray(paste_at, dtype=np.int64).reshape(3)
    px, py, pz = (int(paste_at[i]) for i in range(3))
    sx, sy, sz = volume.shape
    canvas[px:px + sx, py:py + sy, pz:pz + sz] = volume
    return canvas


def paste_volume_on_canvas_clipped(volume, paste_at, canvas_shape, background_value):
    """Paste *volume* at *paste_at*, clipping to *canvas_shape* bounds."""
    volume = np.asarray(volume)
    canvas = np.full(canvas_shape, background_value, dtype=volume.dtype)
    paste_at = np.asarray(paste_at, dtype=np.int64).reshape(3)
    vol_shape = np.array(volume.shape, dtype=np.int64)
    canvas_shape_arr = np.array(canvas_shape, dtype=np.int64)

    src_lo = np.maximum(0, -paste_at)
    src_hi = np.minimum(vol_shape, canvas_shape_arr - paste_at)
    dst_lo = np.maximum(paste_at, 0)
    dst_hi = np.minimum(paste_at + vol_shape, canvas_shape_arr)

    slices_vol = tuple(slice(int(src_lo[i]), int(src_hi[i])) for i in range(3))
    slices_canvas = tuple(slice(int(dst_lo[i]), int(dst_hi[i])) for i in range(3))
    if all(s.start < s.stop for s in slices_vol):
        canvas[slices_canvas] = volume[slices_vol]
    return canvas


def crop_registration_canvas_to_reference(canvas, ref_paste, reference_shape):
    """
    Extract the reference-sized subvolume from the padded registration canvas.

    Post-rotation anatomy outside the reference box is discarded here (acceptable
    when the reference was not padded); pre-rotation cropping is avoided upstream.
    """
    canvas = np.asarray(canvas)
    ref_paste = np.asarray(ref_paste, dtype=np.int64).reshape(3)
    rs = tuple(int(s) for s in reference_shape)
    px, py, pz = (int(ref_paste[i]) for i in range(3))
    return canvas[px:px + rs[0], py:py + rs[1], pz:pz + rs[2]].copy()


def apply_rigid_rt_on_union_canvas(
    moving_canvas,
    R,
    t,
    *,
    spacing,
    reference_origin,
    direction,
    ref_paste,
    reference_canvas=None,
    defaultvalue=0,
    interpolator="linear",
):
    """
    Warp moving canvas into fixed canvas using y = R x + t in physical space.

    Uses the same origin for fixed and moving (reference pasted at ref_paste on
    the union canvas), matching GPU/ANTs estimation and debug checkpoints.
    """
    spacing_tuple = tuple(float(s) for s in spacing)
    canvas_origin = ants_origin_for_canvas_paste(
        reference_origin, direction, ref_paste, spacing_tuple
    )
    moving_np = np.asarray(moving_canvas, dtype=np.float32)
    if reference_canvas is None:
        fixed_np = np.zeros_like(moving_np, dtype=np.float32)
    else:
        fixed_np = np.asarray(reference_canvas, dtype=np.float32)

    moving_ants = ants.from_numpy(
        moving_np,
        spacing=spacing_tuple,
        origin=tuple(float(x) for x in canvas_origin),
        direction=direction,
    )
    fixed_ants = ants.from_numpy(
        fixed_np,
        spacing=spacing_tuple,
        origin=tuple(float(x) for x in canvas_origin),
        direction=direction,
    )
    tx_path = _write_ants_rigid_mat(R, t)
    try:
        warped = ants.apply_transforms(
            fixed=fixed_ants,
            moving=moving_ants,
            transformlist=[tx_path],
            interpolator=interpolator,
            defaultvalue=float(defaultvalue),
            verbose=False,
        )
        return np.asarray(warped.numpy())
    finally:
        try:
            os.remove(tx_path)
        except OSError:
            pass


def ants_origin_for_canvas_paste(reference_origin, reference_direction, paste_at, spacing):
    """
    Physical origin for ``ants.from_numpy`` when reference index ``(0, 0, 0)`` is
    pasted at *paste_at* on the union canvas (shared by fixed and moving images).
    """
    paste_at = np.asarray(paste_at, dtype=np.float64).reshape(3)
    spacing = np.asarray(spacing, dtype=np.float64).reshape(3)
    origin = np.asarray(reference_origin, dtype=np.float64).reshape(3)
    direction = np.asarray(reference_direction, dtype=np.float64).reshape(3, 3)
    return origin - direction @ (paste_at * spacing)


def compute_and_save_landmark_distances(
    scan_dir,
    scan_name,
    edit_stem,
    landmarks,
    snap_distance=None,
    *,
    update_landmarks_metadata=True,
    metadata=None,
    json_path=None,
    distances_basename=None,
):
    """
    Marching-cubes at the edit-specific threshold and write
    ``{distances_basename}_landmark_distances.json``.

    Threshold comes from ``old_thresholds`` for the normalized ``edit_stem``
    when present, otherwise the global ``threshold``. Does not change the
    global threshold.

    ``distances_basename`` defaults to ``edit_stem`` but may keep a ``_lossy``
    token so the distances sidecar matches the landmark file naming used by
    transfer (which often stores lossy edit basenames).
    """
    if json_path is None:
        json_path = os.path.join(scan_dir, f'{scan_name}.json')
    if metadata is None:
        with open(json_path, 'r') as jf:
            metadata = json.load(jf)

    ensure_old_thresholds(metadata, scan_dir, scan_name)
    threshold_stem = normalize_threshold_edit_stem(edit_stem, scan_name)
    if distances_basename is None:
        distances_basename = threshold_stem
    else:
        distances_basename = (
            os.path.basename(str(distances_basename))
            .replace('.nii.gz', '')
            .replace('.ply', '')
        )
        if not distances_basename:
            distances_basename = threshold_stem

    threshold = get_old_threshold(metadata, threshold_stem, metadata.get('threshold'))
    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        threshold = 0.0

    positions = landmarks_to_positions_array(landmarks)
    if positions is None or len(positions) == 0:
        raise ValueError(f'No landmarks provided for distance analysis on {threshold_stem}')

    nifti_path = None
    direct = os.path.join(scan_dir, f'{distances_basename}.nii.gz')
    if os.path.isfile(direct):
        nifti_path = direct
    if nifti_path is None:
        nifti_path = resolve_edit_nifti_path(scan_dir, scan_name, threshold_stem)
    if nifti_path is None:
        raise FileNotFoundError(
            f'No NIfTI found for edit {distances_basename!r} / stem {threshold_stem!r} in {scan_dir}'
        )

    nifti_img = nib.load(nifti_path)
    original_dtype = nifti_img.get_data_dtype()
    nifti_data = nifti_img.get_fdata().astype(original_dtype)
    nifti_data = gaussian_filter(nifti_data, sigma=0.3)

    if np.any(np.isnan(nifti_data)) or np.any(np.isinf(nifti_data)):
        print(f'ERROR: Invalid values in nifti data for {scan_name} / {distances_basename}')
        nifti_data = np.nan_to_num(nifti_data, nan=0.0, posinf=0.0, neginf=0.0)

    data_min, data_max = float(np.min(nifti_data)), float(np.max(nifti_data))
    if threshold < data_min or threshold > data_max:
        print(
            f'WARNING: Threshold {threshold} outside data range '
            f'[{data_min}, {data_max}] for {distances_basename}'
        )
        threshold = float(np.clip(threshold, data_min, data_max))

    voxel_size = voxel_size_for_edit_stem(
        metadata, distances_basename or threshold_stem, scan_name, scan_dir=scan_dir
    )
    vertices, faces, _, _ = measure.marching_cubes(
        nifti_data,
        level=threshold,
        spacing=(voxel_size, voxel_size, voxel_size),
    )

    distances = alpaca.calculate_landmark_distances(positions, vertices, faces)
    mean_distance = float(np.mean(distances))
    std_distance = float(np.std(distances))
    outliers = int(np.sum(distances > voxel_size))
    outliers_six = int(np.sum(distances > voxel_size * 6))

    if snap_distance is not None:
        snap_arr = np.asarray(snap_distance, dtype=np.float64)
        snapped_outliers = int(np.sum(snap_arr > voxel_size))
        snapped_outliers_six = int(np.sum(snap_arr > voxel_size * 6))
        snapped_mean_distance = float(np.mean(snap_arr))
        snapped_std_distance = float(np.std(snap_arr))
        snap_list = snap_arr.tolist()
    else:
        snapped_outliers = None
        snapped_outliers_six = None
        snapped_mean_distance = None
        snapped_std_distance = None
        snap_list = None

    distances_dict = {
        'distances': distances.tolist(),
        'snap_distance': snap_list,
    }
    distances_path = os.path.join(scan_dir, f'{distances_basename}_landmark_distances.json')
    with open(distances_path, 'w') as jf:
        json.dump(distances_dict, jf, indent=4)

    if update_landmarks_metadata:
        metadata['landmarks'] = {
            'num_landmarks': int(len(positions)),
            'mean_distance': mean_distance,
            'std_distance': std_distance,
            'outliers': outliers,
            'outliers_six': outliers_six,
            'snapped_outliers': snapped_outliers,
            'snapped_outliers_six': snapped_outliers_six,
            'snapped_mean_distance': snapped_mean_distance,
            'snapped_std_distance': snapped_std_distance,
            'edit_name': distances_basename,
        }

    # Persist any old_thresholds backfill from ensure_old_thresholds.
    with open(json_path, 'w') as jf:
        json.dump(metadata, jf, indent=4)

    del nifti_data, nifti_img, vertices, faces
    cleanup_memory()

    print(
        f'[LandmarkDistances] {scan_name}/{distances_basename}: '
        f'n={len(positions)} mean={mean_distance:.4f} thr={threshold} → '
        f'{os.path.basename(distances_path)}'
    )
    return {
        'distances_path': os.path.basename(distances_path),
        'edit_stem': threshold_stem,
        'threshold': threshold,
        'num_landmarks': int(len(positions)),
        'mean_distance': mean_distance,
    }


def _is_reference_subject(directory, scan_name):
    """
    True if scan is the project reference.
    Primary signal is extracted/project_settings.json:selected_reference.
    Fallbacks are kept for backward compatibility.
    """
    project_settings_path = os.path.join(directory, "extracted", "project_settings.json")
    if os.path.isfile(project_settings_path):
        try:
            with open(project_settings_path, "r") as jf:
                settings = json.load(jf)
            selected_reference = settings.get("selected_reference", "")
            if selected_reference:
                return selected_reference == scan_name
        except (json.JSONDecodeError, OSError):
            pass

    json_path = os.path.join(directory, "extracted", scan_name, f"{scan_name}.json")
    if os.path.isfile(json_path):
        try:
            with open(json_path, "r") as jf:
                metadata = json.load(jf)
            if metadata.get("reference", False):
                return True
        except (json.JSONDecodeError, OSError):
            pass

    extracted_dir = os.path.join(directory, "extracted")
    if not os.path.isdir(extracted_dir):
        return False

    for subject in os.listdir(extracted_dir):
        subject_dir = os.path.join(extracted_dir, subject)
        if subject == "project_settings.json" or not os.path.isdir(subject_dir):
            continue
        subject_json = os.path.join(subject_dir, f"{subject}.json")
        if not os.path.isfile(subject_json):
            continue
        try:
            with open(subject_json, "r") as jf:
                subject_metadata = json.load(jf)
            if subject_metadata.get("elastic_to", "none") == scan_name:
                return True
        except (json.JSONDecodeError, OSError):
            continue
    return False


def _ants_linear_matrix_from_parameters(parameters):
    """3×3 linear part of an ANTs GenericAffine (column vectors in ITK layout)."""
    return np.asarray(parameters[:9], dtype=np.float64).reshape(3, 3).T


def verify_ants_rigid_transform_mat(transform_path, tol=0.05):
    """
    Return True when the saved GenericAffine is a proper rigid map (orthogonal, det ≈ +1).
    ANTs Rigid registration should satisfy this; log if not.
    """
    tx = ants.read_transform(transform_path)
    linear = _ants_linear_matrix_from_parameters(tx.parameters)
    ortho_err = float(np.max(np.abs(linear.T @ linear - np.eye(3))))
    det = float(np.linalg.det(linear))
    is_rigid = ortho_err <= tol and abs(det - 1.0) <= tol
    return is_rigid, ortho_err, det


def _ants_rigid_log(scan_name, message):
    """Timestamped, flushed log line for long ANTs rigid jobs (visible in Django terminal)."""
    stamp = datetime.now().strftime("%H:%M:%S")
    subject = scan_name or "?"
    print(f"[ANTs rigid {subject} {stamp}] {message}", flush=True)


def _ants_run_with_heartbeat(scan_name, stage_label, fn, interval_sec=30.0):
    """Run a blocking ANTs call; emit heartbeat lines so the terminal shows liveness."""
    done = threading.Event()

    def _heartbeat():
        while not done.wait(interval_sec):
            _ants_rigid_log(scan_name, f"{stage_label} still running…")

    heartbeat = threading.Thread(target=_heartbeat, daemon=True)
    heartbeat.start()
    t0 = time.time()
    try:
        return fn()
    finally:
        done.set()
        heartbeat.join(timeout=1.0)
        _ants_rigid_log(scan_name, f"{stage_label} finished in {time.time() - t0:.1f}s")


def _format_ants_rigid_opts_for_log(opts):
    if not isinstance(opts, dict):
        return str(opts)
    shrink = opts.get("aff_shrink_factors")
    iters = opts.get("aff_iterations")
    sigmas = opts.get("aff_smoothing_sigmas")
    parts = [
        f"metric={opts.get('aff_metric', 'mattes')}",
        f"shrink={list(shrink) if shrink is not None else shrink}",
        f"smooth={list(sigmas) if sigmas is not None else sigmas}",
        f"iters={list(iters) if iters is not None else iters}",
        f"sampling={opts.get('aff_sampling')}",
        f"random_rate={opts.get('aff_random_sampling_rate')}",
        f"grad_step={opts.get('grad_step')}",
        f"histogram_matching={opts.get('use_histogram_matching')}",
        f"singleprecision={opts.get('singleprecision')}",
    ]
    if opts.get("use_affine_initializer", True):
        parts.append(
            "initializer=affine_initializer "
            f"(search={opts.get('initializer_search_factor')}, "
            f"radian_fraction={opts.get('initializer_radian_fraction')}, "
            f"use_principal_axis={opts.get('initializer_use_principal_axis')}, "
            f"local_iters={opts.get('initializer_local_search_iterations')})"
        )
    elif opts.get("use_voxel_patch_init", False):
        pyramid = opts.get("patch_pyramid_factors") or (4, 2, 1)
        parts.append(
            "initializer=voxel-patch "
            f"(pyramid={list(pyramid)}, n={opts.get('patch_n_points', 180)}, "
            f"radius={opts.get('patch_radius', 6)})"
        )
    else:
        parts.append("initializer=Identity")
    if opts.get("use_registration_mask", False):
        parts.append("mask=threshold-foreground")
    else:
        parts.append("mask=none (texture/Mattes MI on full grid)")
    return "; ".join(str(p) for p in parts)


def _ants_index_to_physical(index, spacing, origin, direction):
    index = np.asarray(index, dtype=np.float64).reshape(3)
    spacing = np.asarray(spacing, dtype=np.float64).reshape(3)
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    direction = np.asarray(direction, dtype=np.float64).reshape(3, 3)
    return origin + direction @ (index * spacing)


def _ants_physical_to_numpy_index(physical, spacing, origin, direction):
    """Inverse of ``_ants_index_to_physical`` (continuous voxel indices)."""
    physical = np.asarray(physical, dtype=np.float64).reshape(3)
    spacing = np.asarray(spacing, dtype=np.float64).reshape(3)
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    direction = np.asarray(direction, dtype=np.float64).reshape(3, 3)
    return (np.linalg.inv(direction) @ (physical - origin)) / spacing


def _read_ants_rigid_rt(transform_path):
    """Return (R, t) from a saved ANTs GenericAffine .mat (moving → fixed)."""
    tx = ants.read_transform(transform_path)
    params = np.asarray(tx.parameters, dtype=np.float64)
    return _ants_linear_matrix_from_parameters(params), params[9:12]


def _nearest_rotation_from_linear(linear):
    """Proper rotation nearest to a 3×3 linear map (polar/SVD factor)."""
    linear = np.asarray(linear, dtype=np.float64).reshape(3, 3)
    u, _, vt = np.linalg.svd(linear)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u = u.copy()
        u[:, -1] *= -1
        r = u @ vt
    return r


def _project_ants_affine_to_rigid_mat(affine_mat_path, out_path=None):
    """Nearest rigid .mat from a similarity/affine initializer (keep translation)."""
    r_raw, t = _read_ants_rigid_rt(affine_mat_path)
    r_rigid = _nearest_rotation_from_linear(r_raw)
    return _write_ants_rigid_mat(r_rigid, t, out_path=out_path)


def project_affine_mat_path_to_rigid_transform(reg_result, scan_name=None):
    """Take ants.registration(Affine) output; persist only rotation+translation."""
    mat_path = None
    for candidate in list(reg_result.get("fwdtransforms") or []):
        if isinstance(candidate, str) and candidate.endswith(".mat") and os.path.isfile(candidate):
            mat_path = candidate
            break
    if not mat_path:
        return None, None
    rigid_path = _project_ants_affine_to_rigid_mat(mat_path)
    _ants_rigid_log(
        scan_name,
        f"Affine registration projected to rigid GenericAffine -> {rigid_path}",
    )
    return {"fwdtransforms": [rigid_path]}, rigid_path


def _pick_affine_initializer_for_rigid(
    init_mat,
    fixed_image,
    moving_image,
    scan_name=None,
):
    """
    After affine_initializer, always feed Rigid the best available start (raw or
    SVD-projected). Rigid refine is meant to clean up imperfect inits — only fall
    back to Identity when both candidates fail to apply on the estimation grid.
    """
    is_rigid, ortho_err, det = verify_ants_rigid_transform_mat(init_mat)
    if is_rigid:
        return init_mat, "affine_initializer"

    projected_path = _project_ants_affine_to_rigid_mat(init_mat)
    id_ncc, id_iou = _score_rigid_on_overlap(fixed_image, moving_image, np.eye(3), np.zeros(3))
    raw_ncc, raw_iou = _score_ants_rigid_transform(fixed_image, moving_image, init_mat)
    proj_ncc, proj_iou = _score_ants_rigid_transform(fixed_image, moving_image, projected_path)
    _ants_rigid_log(
        scan_name,
        f"affine_initializer non-rigid (ortho_err={ortho_err:.2e}, det={det:.4f}); "
        f"raw ncc={raw_ncc:.3f} iou={raw_iou:.3f}, "
        f"projected ncc={proj_ncc:.3f} iou={proj_iou:.3f}, "
        f"identity ncc={id_ncc:.3f} iou={id_iou:.3f}",
    )

    candidates = [
        (init_mat, raw_ncc, raw_iou, "affine_initializer"),
        (projected_path, proj_ncc, proj_iou, "affine_initializer_projected"),
    ]
    best_path, best_ncc, best_iou, best_path_name = max(candidates, key=lambda item: (item[2], item[1]))

    scoring_failed = best_iou < 0 and best_ncc < 0
    if scoring_failed:
        _ants_rigid_log(
            scan_name,
            "initializer transforms could not be scored on estimation grid; starting Rigid from Identity",
        )
        return ["Identity"], "identity_after_weak_affine_init"

    _ants_rigid_log(
        scan_name,
        f"using {best_path_name} for Rigid start (ncc={best_ncc:.3f} iou={best_iou:.3f}; "
        f"identity ncc={id_ncc:.3f} iou={id_iou:.3f})",
    )
    return best_path, best_path_name


def _initial_transform_from_initializer(initial_transform):
    """True when Rigid will start from a saved initializer .mat (not Identity)."""
    if isinstance(initial_transform, str) and initial_transform.endswith(".mat"):
        return os.path.isfile(initial_transform)
    if isinstance(initial_transform, (list, tuple)):
        return any(
            isinstance(x, str) and x.endswith(".mat") and os.path.isfile(x)
            for x in initial_transform
        )
    return False


def _padding_for_ants_physical_rigid_warp(
    volume_shape,
    rotation_center_voxel,
    R,
    t,
    spacing,
    origin,
    direction,
):
    """
    Pad a volume so an ANTs rigid map (physical R, t about rotation_center) does not
    clip corners. Same corner-expansion idea as ``_padding_for_rigid_volume_warp`` but
    warps through the ants.from_numpy index ↔ physical frame.
    """
    shape = np.asarray(volume_shape, dtype=np.float64).reshape(3)
    center = np.asarray(rotation_center_voxel, dtype=np.float64).reshape(3)
    spacing = tuple(float(s) for s in spacing)
    origin = tuple(float(o) for o in origin)
    direction = np.asarray(direction, dtype=np.float64).reshape(3, 3)
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)

    corners = np.array([
        [0, 0, 0],
        [shape[0], 0, 0],
        [0, shape[1], 0],
        [0, 0, shape[2]],
        [shape[0], shape[1], 0],
        [shape[0], 0, shape[2]],
        [0, shape[1], shape[2]],
        [shape[0], shape[1], shape[2]],
    ], dtype=np.float64)

    center_phys = _ants_index_to_physical(center, spacing, origin, direction)
    corners_phys = np.array(
        [_ants_index_to_physical(c, spacing, origin, direction) for c in corners],
        dtype=np.float64,
    )
    warped_phys = (R @ (corners_phys - center_phys).T).T + center_phys + t
    warped_idx = np.array(
        [_ants_physical_to_numpy_index(p, spacing, origin, direction) for p in warped_phys],
        dtype=np.float64,
    )

    min_coords = np.floor(np.min(warped_idx, axis=0)).astype(int)
    max_coords = np.ceil(np.max(warped_idx, axis=0)).astype(int)
    padding_low = np.maximum(0, -min_coords)
    padding_high = np.maximum(0, max_coords - np.ceil(shape).astype(int))
    return padding_low.astype(np.int64), padding_high.astype(np.int64)


def _estimate_padded_rigid_apply_bytes(
    volume_shape,
    rotation_center_voxel,
    transformlist,
    spacing,
    origin,
    direction,
):
    """Rough peak RAM for padded ants.apply_transforms (fixed pad + warped + scratch)."""
    mat_path = None
    for candidate in list(transformlist or []):
        if isinstance(candidate, str) and candidate.endswith(".mat") and os.path.isfile(candidate):
            mat_path = candidate
            break
    if mat_path is None:
        return 0, None, None, None
    R, t = _read_ants_rigid_rt(mat_path)
    padding_low, padding_high = _padding_for_ants_physical_rigid_warp(
        volume_shape,
        rotation_center_voxel,
        R,
        t,
        spacing,
        origin,
        direction,
    )
    shape = np.asarray(volume_shape, dtype=np.int64).reshape(3)
    padded_shape = tuple(int(shape[i] + padding_low[i] + padding_high[i]) for i in range(3))
    padded_voxels = int(padded_shape[0]) * int(padded_shape[1]) * int(padded_shape[2])
    est_bytes = padded_voxels * 4 * 3
    return est_bytes, padded_shape, padding_low, padding_high


def _decimate_volume_axis(vol: np.ndarray, stride: int) -> np.ndarray:
    """Simple strided crop/decimate along i,j,k (used for low-memory rigid apply)."""
    s = int(stride)
    return np.asarray(vol, dtype=np.float32)[::s, ::s, ::s].copy()


def _ants_padded_origin(origin, direction, padding_low, spacing):
    """Physical origin for a volume padded on the low-index sides only."""
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    direction = np.asarray(direction, dtype=np.float64).reshape(3, 3)
    padding_low = np.asarray(padding_low, dtype=np.float64).reshape(3)
    spacing = np.asarray(spacing, dtype=np.float64).reshape(3)
    return origin - direction @ (padding_low * spacing)


def apply_ants_rigid_transform_padded(
    fixed_image,
    moving_image,
    transformlist,
    rotation_center_voxel,
    *,
    defaultvalue=-1,
    singleprecision=True,
    verbose=False,
    scan_name=None,
    interpolator="linear",
):
    """
    Apply a rigid ANTs transform on padded fixed/moving images, then crop to the
    original fixed grid. Avoids diagonal clipping when rotating on a tight canvas.
    """
    mat_path = None
    for candidate in list(transformlist or []):
        if isinstance(candidate, str) and candidate.endswith(".mat") and os.path.isfile(candidate):
            mat_path = candidate
            break
    if mat_path is None:
        raise FileNotFoundError("No .mat transform in transformlist for padded apply")

    R, t = _read_ants_rigid_rt(mat_path)
    spacing = fixed_image.spacing
    direction = fixed_image.direction
    fixed_origin = np.asarray(fixed_image.origin, dtype=np.float64)
    fixed_np = np.asarray(fixed_image.numpy())

    padding_low, padding_high = _padding_for_ants_physical_rigid_warp(
        fixed_np.shape,
        rotation_center_voxel,
        R,
        t,
        spacing,
        fixed_origin,
        direction,
    )
    _ants_rigid_log(
        scan_name,
        f"rotation padding before apply_transforms: low={padding_low.tolist()} "
        f"high={padding_high.tolist()}",
    )

    pad_spec = (
        (int(padding_low[0]), int(padding_high[0])),
        (int(padding_low[1]), int(padding_high[1])),
        (int(padding_low[2]), int(padding_high[2])),
    )
    # Only the output (fixed) grid needs room to rotate. The moving image is
    # sampled in physical space; padding it with the fixed origin used to wipe
    # the union-canvas paste shift and reframe the subject into a corner.
    origin_fixed_padded = _ants_padded_origin(fixed_origin, direction, padding_low, spacing)
    fixed_padded_ants = ants.from_numpy(
        np.pad(fixed_np, pad_spec, mode="constant", constant_values=defaultvalue).astype(np.float32),
        spacing=spacing,
        origin=tuple(float(x) for x in origin_fixed_padded),
        direction=direction,
    )

    warped = ants.apply_transforms(
        fixed=fixed_padded_ants,
        moving=moving_image,
        transformlist=transformlist,
        defaultvalue=defaultvalue,
        singleprecision=singleprecision,
        interpolator=interpolator,
        verbose=verbose,
    )
    warped_np = np.asarray(warped.numpy())
    del warped, fixed_padded_ants
    sx, sy, sz = (int(padding_low[i]) for i in range(3))
    ox, oy, oz = fixed_np.shape
    cropped = warped_np[sx : sx + ox, sy : sy + oy, sz : sz + oz].copy()
    del warped_np
    return cropped, padding_low, padding_high


def apply_ants_rigid_to_reference_fov(
    moving_volume,
    reference_volume,
    transformlist,
    rotation_center_voxel,
    fixed_spacing,
    fixed_origin,
    fixed_direction,
    ref_paste,
    mov_paste,
    *,
    defaultvalue=-1,
    normalize=False,
    min_max_normalize=None,
    restore_range=None,
    intensity_range=None,
    adjusted_background_value=None,
    output_dtype=None,
    interpolator="linear",
    singleprecision=True,
    scan_name=None,
    verbose=False,
):
    """
    Apply a saved rigid ANTs transform onto the native reference FOV.

    Fixed image uses the reference NIfTI origin/spacing/direction. The moving image uses
    the native scan with an origin shift so the physical frame matches estimation on the
    union canvas (paste offsets, no translation rescaling on the .mat).
    """
    spacing_tuple = tuple(float(s) for s in fixed_spacing)
    rotation_center = np.asarray(rotation_center_voxel, dtype=np.float64).reshape(3)
    ref_paste_arr = np.asarray(ref_paste, dtype=np.float64)
    mov_paste_arr = np.asarray(mov_paste, dtype=np.float64)

    if normalize:
        if min_max_normalize is None or restore_range is None or intensity_range is None:
            raise ValueError("normalize=True requires min_max_normalize, restore_range, intensity_range")
        fixed_np = min_max_normalize(
            np.asarray(reference_volume, dtype=np.float32), new_min=-1, new_max=1
        )
        moving_np = min_max_normalize(
            np.asarray(moving_volume, dtype=np.float32), new_min=-1, new_max=1
        )
    else:
        fixed_np = np.asarray(reference_volume, dtype=np.float32)
        moving_np = np.asarray(moving_volume, dtype=np.float32)

    full_shape = fixed_np.shape
    apply_stride = 1
    est_bytes, padded_shape, _, _ = _estimate_padded_rigid_apply_bytes(
        full_shape,
        rotation_center,
        transformlist,
        spacing_tuple,
        fixed_origin,
        fixed_direction,
    )
    avail = psutil.virtual_memory().available
    if est_bytes > max(avail * 0.35, 5.5e9):
        apply_stride = 2
        _ants_rigid_log(
            scan_name,
            f"padded apply est ~{est_bytes / 1e9:.1f} GB (shape {padded_shape}), "
            f"avail {avail / 1e9:.1f} GB — stride-{apply_stride} apply",
        )
    elif padded_shape is not None:
        _ants_rigid_log(
            scan_name,
            f"padded apply est ~{est_bytes / 1e9:.1f} GB (shape {padded_shape}), "
            f"avail {avail / 1e9:.1f} GB",
        )

    if apply_stride > 1:
        fixed_np = _decimate_volume_axis(fixed_np, apply_stride)
        moving_np = _decimate_volume_axis(moving_np, apply_stride)
        spacing_tuple = tuple(float(s) * apply_stride for s in spacing_tuple)
        rotation_center = rotation_center / float(apply_stride)
        paste_shift = (ref_paste_arr - mov_paste_arr) / float(apply_stride)
    else:
        paste_shift = ref_paste_arr - mov_paste_arr

    moving_origin = ants_origin_for_canvas_paste(
        fixed_origin,
        fixed_direction,
        paste_shift,
        spacing_tuple,
    )

    fixed_ants = ants.from_numpy(
        fixed_np,
        spacing=spacing_tuple,
        origin=tuple(float(x) for x in fixed_origin),
        direction=fixed_direction,
    )
    moving_ants = ants.from_numpy(
        moving_np,
        spacing=spacing_tuple,
        origin=tuple(float(x) for x in moving_origin),
        direction=fixed_direction,
    )
    del fixed_np, moving_np

    warped, _, _ = apply_ants_rigid_transform_padded(
        fixed_ants,
        moving_ants,
        transformlist,
        rotation_center,
        defaultvalue=defaultvalue,
        singleprecision=singleprecision,
        verbose=verbose,
        scan_name=scan_name,
        interpolator=interpolator,
    )
    del fixed_ants, moving_ants

    if apply_stride > 1:
        from scipy.ndimage import zoom

        zoom_factors = tuple(
            full_shape[i] / max(warped.shape[i], 1) for i in range(3)
        )
        warped = zoom(warped, zoom_factors, order=1, mode="nearest")

    if normalize:
        warped = np.where(
            warped == defaultvalue,
            adjusted_background_value,
            warped,
        )
        warped = restore_range(
            warped,
            intensity_range[0],
            intensity_range[1],
        )
    if output_dtype is not None:
        warped = np.asarray(warped, dtype=output_dtype)
    return warped


def _voxel_patch_foreground_mask(volume, fg_floor=-0.45):
    """Interior-ish foreground on ANTs-normalized volumes (background ≈ -1)."""
    vol = np.asarray(volume, dtype=np.float32)
    return vol > float(fg_floor)


def _voxel_patch_interest_points(volume, mask, n_points=180, min_separation=4):
    """
    Pick interior voxels with high local intensity variance (texture), not surface blobs.
    Mask should already be eroded so fur/hair at the outer shell is excluded.
    """
    from scipy.ndimage import binary_erosion, gaussian_filter, maximum_filter, uniform_filter

    vol = np.asarray(volume, dtype=np.float32)
    work_mask = np.asarray(mask, dtype=bool)
    if work_mask.sum() < 64:
        return np.empty((0, 3), dtype=np.int64)

    # Stay inside the blob so descriptors see internal texture, not variable fur.
    eroded = binary_erosion(work_mask, iterations=2)
    if eroded.sum() < 64:
        eroded = work_mask

    mean = uniform_filter(vol, size=5, mode="nearest")
    mean_sq = uniform_filter(vol * vol, size=5, mode="nearest")
    variance = np.clip(mean_sq - mean * mean, 0.0, None)
    variance = gaussian_filter(variance, sigma=0.6, mode="nearest")
    variance = np.where(eroded, variance, 0.0)

    peaks = (variance == maximum_filter(variance, size=int(min_separation))) & eroded & (variance > 0)
    coords = np.argwhere(peaks)
    if coords.size == 0:
        coords = np.argwhere(eroded)
        if coords.size == 0:
            return np.empty((0, 3), dtype=np.int64)
        rng = np.random.default_rng(0)
        idx = rng.choice(len(coords), size=min(int(n_points), len(coords)), replace=False)
        return coords[idx].astype(np.int64)

    scores = variance[tuple(coords.T)]
    order = np.argsort(scores)[::-1]
    coords = coords[order]

    selected = []
    min_sep_sq = int(min_separation) ** 2
    for pt in coords:
        if len(selected) >= int(n_points):
            break
        if selected:
            delta = np.asarray(selected, dtype=np.int64) - pt
            if np.any(np.sum(delta * delta, axis=1) < min_sep_sq):
                continue
        selected.append(pt)
    if len(selected) < 8:
        selected = coords[: max(int(n_points), 8)]
    return np.asarray(selected, dtype=np.int64)


def _voxel_patch_radial_descriptor(volume, points, radius=6, n_shells=8):
    """
    Rotation-invariant descriptor: mean/std intensity in concentric spherical shells.
    Survives large unknown rotations because bins are radial, not oriented.
    """
    vol = np.asarray(volume, dtype=np.float32)
    pts = np.asarray(points, dtype=np.int64)
    if pts.size == 0:
        return np.empty((0, n_shells * 2), dtype=np.float32)

    radius = max(2, int(radius))
    n_shells = max(3, int(n_shells))
    shape = np.array(vol.shape, dtype=np.int64)
    descriptors = np.zeros((len(pts), n_shells * 2), dtype=np.float32)
    shell_edges = np.linspace(0.0, float(radius) + 0.5, n_shells + 1)

    offsets = np.mgrid[-radius : radius + 1, -radius : radius + 1, -radius : radius + 1]
    offsets = np.stack(offsets, axis=-1).reshape(-1, 3)
    dist = np.linalg.norm(offsets.astype(np.float32), axis=1)
    in_sphere = dist <= (radius + 0.5)
    offsets = offsets[in_sphere]
    dist = dist[in_sphere]
    shell_idx = np.clip(np.digitize(dist, shell_edges) - 1, 0, n_shells - 1)

    for i, pt in enumerate(pts):
        loc = pt + offsets
        valid = np.all((loc >= 0) & (loc < shape), axis=1)
        if not np.any(valid):
            continue
        vals = vol[loc[valid, 0], loc[valid, 1], loc[valid, 2]]
        shells = shell_idx[valid]
        for s in range(n_shells):
            bin_vals = vals[shells == s]
            if bin_vals.size == 0:
                continue
            descriptors[i, s] = float(bin_vals.mean())
            descriptors[i, n_shells + s] = float(bin_vals.std())

        norm = np.linalg.norm(descriptors[i])
        if norm > 1e-6:
            descriptors[i] /= norm
    return descriptors


def _voxel_patch_mutual_matches(moving_desc, fixed_desc, min_similarity=0.55):
    """Ranked mutual nearest-neighbour matches by cosine similarity."""
    if moving_desc.size == 0 or fixed_desc.size == 0:
        return []
    # Cosine sim because descriptors are L2-normalized.
    sim = moving_desc @ fixed_desc.T
    mov_best = np.argmax(sim, axis=1)
    fix_best = np.argmax(sim, axis=0)
    matches = []
    for mi, fi in enumerate(mov_best):
        if fix_best[fi] != mi:
            continue
        score = float(sim[mi, fi])
        if score < float(min_similarity):
            continue
        matches.append((score, int(mi), int(fi)))
    matches.sort(reverse=True, key=lambda t: t[0])
    return matches


def _rigid_rotation_angle_deg(R):
    """Proper rotation angle (degrees) extracted from a 3×3 rotation matrix."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    u, _, vt = np.linalg.svd(R)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))))


def _rigid_rotation_distance_deg(R1, R2):
    """Geodesic rotation distance (degrees) between two proper rotations."""
    R1 = np.asarray(R1, dtype=np.float64).reshape(3, 3)
    R2 = np.asarray(R2, dtype=np.float64).reshape(3, 3)
    return _rigid_rotation_angle_deg(R1 @ R2.T)


def _min_pairwise_sep_mm(points):
    """Minimum Euclidean separation (mm) among a set of 3D points."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return 0.0
    seps = [
        float(np.linalg.norm(points[i] - points[j]))
        for i in range(len(points))
        for j in range(i + 1, len(points))
    ]
    return min(seps) if seps else 0.0


def _refine_voxel_patch_ransac_inliers(
    moving_phys,
    fixed_phys,
    match_scores,
    R,
    t,
    inlier_residual_mm,
):
    """Re-fit Kabsch on geometric inliers for one RANSAC hypothesis."""
    moving_phys = np.asarray(moving_phys, dtype=np.float64)
    fixed_phys = np.asarray(fixed_phys, dtype=np.float64)
    match_scores = np.asarray(match_scores, dtype=np.float64)
    pred = (np.asarray(R) @ moving_phys.T).T + np.asarray(t).reshape(3)
    residuals = np.linalg.norm(pred - fixed_phys, axis=1)
    inlier_mask = residuals <= float(inlier_residual_mm)
    if int(inlier_mask.sum()) < 3:
        keep = min(len(residuals), 3)
        order = np.argsort(residuals)[:keep]
        inlier_mask = np.zeros(len(residuals), dtype=bool)
        inlier_mask[order] = True
    mov_in = moving_phys[inlier_mask]
    fix_in = fixed_phys[inlier_mask]
    sim_in = match_scores[inlier_mask]
    R2, t2 = _kabsch_rigid(mov_in, fix_in)
    if R2 is None:
        return None
    return {
        "R": R2,
        "t": t2,
        "inliers": int(inlier_mask.sum()),
        "mean_match": float(np.mean(sim_in)),
        "angle_deg": _rigid_rotation_angle_deg(R2),
        "match_score": float(np.sum(sim_in)) * np.sqrt(int(inlier_mask.sum())),
    }


def _voxel_patch_ransac_rigid(
    moving_phys,
    fixed_phys,
    match_scores,
    *,
    n_iterations=400,
    min_inliers=8,
    min_spread_mm=2.5,
    inlier_residual_mm=8.0,
    top_candidates=12,
    scan_name=None,
    stage_label="",
):
    """
    Similarity-weighted RANSAC for voxel-patch correspondences.

    Descriptor mutual matches are often ambiguous (many patches look alike). RANSAC
    repeatedly samples minimal rigid sets (3 points, spatially separated), fits
    Kabsch, and scores hypotheses by geometric inlier support weighted by match
    similarity — the same philosophy as mesh RANSAC, but on interior voxel patches.
    """
    moving_phys = np.asarray(moving_phys, dtype=np.float64)
    fixed_phys = np.asarray(fixed_phys, dtype=np.float64)
    match_scores = np.asarray(match_scores, dtype=np.float64)
    n_corr = len(moving_phys)
    if n_corr < 3:
        return None

    probs = np.clip(match_scores, 1e-6, None) ** 2
    probs = probs / probs.sum()
    rng = np.random.default_rng(0)
    n_iters = max(int(n_iterations), n_corr * 8)

    candidates = []
    for _ in range(n_iters):
        sample_n = min(3, n_corr)
        idx = rng.choice(n_corr, size=sample_n, replace=False, p=probs)
        if _min_pairwise_sep_mm(moving_phys[idx]) < float(min_spread_mm):
            continue
        R, t = _kabsch_rigid(moving_phys[idx], fixed_phys[idx])
        if R is None:
            continue
        pred = (R @ moving_phys.T).T + t
        residuals = np.linalg.norm(pred - fixed_phys, axis=1)
        inlier_mask = residuals <= float(inlier_residual_mm)
        n_in = int(inlier_mask.sum())
        if n_in < 3:
            continue
        # Descriptor agreement × geometric support (same scoring as DINO-Reg RANSAC).
        hyp_score = float(np.sum(match_scores[inlier_mask])) * np.sqrt(n_in)
        candidates.append((hyp_score, R, t))

    if not candidates:
        if scan_name:
            _ants_rigid_log(
                scan_name,
                f"voxel-patch RANSAC{stage_label}: no hypotheses "
                f"({n_corr} correspondences)",
            )
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    if scan_name:
        _ants_rigid_log(
            scan_name,
            f"voxel-patch RANSAC{stage_label}: {len(candidates)} hypotheses, "
            f"top score={candidates[0][0]:.2f}",
        )

    seen_rotations = []
    best = None
    for hyp_score, R, t in candidates[: max(int(top_candidates), 1)]:
        dup = False
        for R_seen in seen_rotations:
            if _rigid_rotation_distance_deg(R, R_seen) < 8.0:
                dup = True
                break
        if dup:
            continue
        seen_rotations.append(R.copy())

        refined = _refine_voxel_patch_ransac_inliers(
            moving_phys,
            fixed_phys,
            match_scores,
            R,
            t,
            inlier_residual_mm,
        )
        if refined is None:
            continue
        if int(refined["inliers"]) < int(min_inliers):
            continue
        pick_score = (refined["inliers"], refined["match_score"])
        if best is None or pick_score > best[0]:
            best = (pick_score, refined)

    if best is None:
        if scan_name:
            _ants_rigid_log(
                scan_name,
                f"voxel-patch RANSAC{stage_label}: no hypothesis reached "
                f"min_inliers={min_inliers}",
            )
        return None
    return best[1]


def _accept_voxel_patch_candidate(
    ncc,
    iou,
    identity_ncc,
    identity_iou,
    *,
    angle_deg=0.0,
    ransac_inliers=0,
    min_ransac_inliers=8,
):
    """
    Reject initializer transforms that do not clearly beat the identity baseline.

    Large rotations require enough RANSAC inlier support and absolute NCC/IoU quality
    so a false-patch rotation cannot slip through on a tiny NCC gain.
    """
    if iou < identity_iou - 0.005:
        return False
    if int(ransac_inliers) < int(min_ransac_inliers):
        return False

    ncc_gain = ncc - identity_ncc
    iou_gain = iou - identity_iou
    abs_ncc_floor = max(0.10, identity_ncc + 0.05)

    if float(angle_deg) > 30.0:
        if ncc < abs_ncc_floor or iou < identity_iou + 0.01:
            return False
        if ncc_gain < 0.08 and iou_gain < 0.03:
            return False
        return True

    if ncc >= abs_ncc_floor and iou >= identity_iou:
        return True
    if ncc_gain >= 0.05 and iou_gain >= 0.01:
        return True
    if ncc_gain >= 0.03 and iou_gain >= 0.03:
        return True
    return False


def _kabsch_inlier_residual_mm(moving_phys, fixed_phys, R, t, inlier_frac=0.75):
    """Median reprojection residual (mm) on the best inlier fraction of correspondences."""
    pred = (R @ moving_phys.T).T + t
    residuals = np.linalg.norm(pred - fixed_phys, axis=1)
    keep = max(3, int(np.ceil(len(residuals) * float(inlier_frac))))
    return float(np.median(np.sort(residuals)[:keep]))


def _ants_binary_mask_image(mask_array, spacing, origin, direction, dilate_iters=3):
    """Build an ANTs mask image from a boolean numpy volume (optionally dilated)."""
    from scipy.ndimage import binary_dilation

    mask = np.asarray(mask_array, dtype=bool)
    if dilate_iters > 0:
        mask = binary_dilation(mask, iterations=int(dilate_iters))
    return ants.from_numpy(
        mask.astype(np.uint8),
        spacing=tuple(float(s) for s in spacing),
        origin=tuple(float(o) for o in origin),
        direction=np.asarray(direction, dtype=np.float64),
    )


def _kabsch_rigid(moving_pts, fixed_pts):
    """Proper rotation + translation mapping moving points onto fixed (no scale)."""
    moving_pts = np.asarray(moving_pts, dtype=np.float64)
    fixed_pts = np.asarray(fixed_pts, dtype=np.float64)
    if moving_pts.shape[0] < 3:
        return None, None
    c_m = moving_pts.mean(axis=0)
    c_f = fixed_pts.mean(axis=0)
    H = (moving_pts - c_m).T @ (fixed_pts - c_f)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = c_f - R @ c_m
    return R, t


def is_point_pair_rigid_method(method):
    """True for Kabsch-from-correspondences methods (manual guidepoints only)."""
    return str(method or "").strip().lower() == "manual-guidepoints"


def is_ants_style_physical_rigid_method(method):
    """ANTs / GPU-rigid estimators (pose found via GenericAffine; applied via scipy)."""
    return str(method or "").strip().lower() in ("ants", "gpu-rigid")


def uses_scipy_centroid_embed_rigid(method):
    """
    Rigid methods that share the guidepoints scipy pipeline:
    pad → affine_transform(R, offset) → residual shift → integer canvas embed.
    """
    m = str(method or "").strip().lower()
    return m in ("manual-guidepoints", "dino-reg", "ants", "gpu-rigid")


def kabsch_rotation_matrix(source_pts, target_pts, source_centroid=None, target_centroid=None):
    """
    Return (R, source_centroid, target_centroid) for row-vector points.

    Guidepoints / scipy rigid use ``target = (source - source_centroid) @ R + target_centroid``.
    ``ndimage.affine_transform`` maps output→input so the matrix argument is ``R`` (with
    centroid offset); landmarks and meshes apply the same ``R`` on row vectors.
    """
    source_pts = np.asarray(source_pts, dtype=np.float64)
    target_pts = np.asarray(target_pts, dtype=np.float64)
    if source_pts.shape != target_pts.shape or source_pts.shape[0] < 3:
        raise ValueError("kabsch_rotation_matrix requires matching Nx3 arrays with N>=3")
    if source_centroid is None:
        source_centroid = np.mean(source_pts, axis=0)
    else:
        source_centroid = np.asarray(source_centroid, dtype=np.float64).reshape(3)
    if target_centroid is None:
        target_centroid = np.mean(target_pts, axis=0)
    else:
        target_centroid = np.asarray(target_centroid, dtype=np.float64).reshape(3)
    src_centered = source_pts - source_centroid
    tgt_centered = target_pts - target_centroid
    H = src_centered.T @ tgt_centered
    U, _, Vt = np.linalg.svd(H)
    rotation = U @ Vt
    if np.linalg.det(rotation) < 0:
        U = U.copy()
        U[:, -1] *= -1
        rotation = U @ Vt
    return rotation, source_centroid, target_centroid


def forward_scan_voxel_to_reference_voxel_via_ants_rigid(
    scan_voxel,
    *,
    mat_path,
    rotation_center_voxel,
    fixed_spacing,
    fixed_origin,
    fixed_direction,
    mov_paste,
    ref_paste,
):
    """
    Map scan-native voxel indices (reference spacing grid) to reference canvas
    indices using the same physical frame as ``apply_ants_rigid_to_reference_fov``.
    """
    scan_voxel = np.asarray(scan_voxel, dtype=np.float64).reshape(3)
    spacing = tuple(float(s) for s in fixed_spacing)
    fixed_origin_arr = np.asarray(fixed_origin, dtype=np.float64).reshape(3)
    direction = np.asarray(fixed_direction, dtype=np.float64).reshape(3, 3)
    paste_shift = (
        np.asarray(ref_paste, dtype=np.float64).reshape(3)
        - np.asarray(mov_paste, dtype=np.float64).reshape(3)
    )
    moving_origin = ants_origin_for_canvas_paste(
        fixed_origin_arr, direction, paste_shift, spacing
    )
    physical_moving = _ants_index_to_physical(scan_voxel, spacing, moving_origin, direction)
    center_phys = _ants_index_to_physical(
        np.asarray(rotation_center_voxel, dtype=np.float64).reshape(3),
        spacing,
        fixed_origin_arr,
        direction,
    )
    R, t = _read_ants_rigid_rt(mat_path)
    physical_fixed = R @ (physical_moving - center_phys) + center_phys + t
    return _ants_physical_to_numpy_index(
        physical_fixed, spacing, fixed_origin_arr, direction
    )


def _as_isotropic_spacing_xyz(spacing, *, label="spacing"):
    """Scalar or length-3 spacing → (3,) float64 array."""
    s = np.asarray(spacing, dtype=np.float64).reshape(-1)
    if s.size == 1:
        return np.full(3, float(s[0]), dtype=np.float64)
    if s.size == 3:
        return s.astype(np.float64, copy=False)
    raise ValueError(f"Expected scalar or length-3 {label}, got shape {s.shape}")


def project_physical_rigid_rt_to_scipy_rigid_params(
    R_col,
    t_phys,
    scan_centroid_voxel,
    reference_centroid_voxel,
    fixed_spacing,
    scan_name="",
):
    """
    Convert a physical rigid ``p_fixed = R_col @ p_moving + t_phys`` (mm) into
    scipy/guidepoints parameters:

        pad → affine_transform(rotation, offset) → residual shift → canvas embed

    Residual is chosen so that after rotate-about-``scan_c`` and canvas paste
    anchoring ``scan_c → ref_c``, the volume matches ``R_col @ p + t``.

    Requires isotropic spacing (the Aurora reference grid after resample).
    """
    spacing = _as_isotropic_spacing_xyz(fixed_spacing, label="fixed_spacing")
    if not np.allclose(spacing, spacing[0]):
        raise ValueError(
            f"Physical→scipy rigid projection requires isotropic spacing for {scan_name}"
        )
    t_vox = np.asarray(t_phys, dtype=np.float64).reshape(3) / spacing[0]
    scan_c = np.asarray(scan_centroid_voxel, dtype=np.float64).reshape(3)
    ref_c = np.asarray(reference_centroid_voxel, dtype=np.float64).reshape(3)
    R_col = np.asarray(R_col, dtype=np.float64).reshape(3, 3)

    # Guidepoints store ``rotation`` for row-vector volumes: (v - c) @ rotation.
    # Physical rigid uses column action R_col @ p + t; on row vectors that is rotation = R_col.T.
    rotation = R_col.T
    # Rotate about scan_c, then shift so canvas snap scan_c→ref_c yields R@p + t.
    # p_canvas = R@(p - scan_c) + residual + ref_c  ⇒  residual = R@scan_c + t_vox - ref_c.
    residual_translation = R_col @ scan_c + t_vox - ref_c
    return rotation, scan_c, ref_c, residual_translation


def project_ants_mat_to_scipy_rigid_params(
    mat_path,
    scan_centroid_voxel,
    reference_centroid_voxel,
    fixed_spacing,
    scan_name="",
):
    """
    Convert an ANTs GenericAffine (rotate about reference centroid in physical
    space) into scipy/guidepoints parameters.

    Requires isotropic spacing (the Aurora reference grid after resample).
    """
    R_col, t_ants = _read_ants_rigid_rt(mat_path)
    spacing = _as_isotropic_spacing_xyz(fixed_spacing, label="fixed_spacing")
    ref_c = np.asarray(reference_centroid_voxel, dtype=np.float64).reshape(3)
    # ANTs: p' = R @ (p - ref_c) + ref_c + t_ants  ⇒  p' = R @ p + t_full
    # with t_full = t_ants + (I - R) @ ref_c_mm.
    ref_c_mm = ref_c * float(spacing[0])
    t_full = np.asarray(t_ants, dtype=np.float64).reshape(3) + ref_c_mm - R_col @ ref_c_mm
    rotation, scan_c, ref_c, residual_translation = project_physical_rigid_rt_to_scipy_rigid_params(
        R_col,
        t_full,
        scan_centroid_voxel,
        reference_centroid_voxel,
        fixed_spacing,
        scan_name=scan_name,
    )

    is_rigid, ortho_err, det = verify_ants_rigid_transform_mat(mat_path)
    if is_rigid:
        _ants_rigid_log(
            scan_name,
            f"ANTs→scipy analytic projection (ortho_err={ortho_err:.2e}, det={det:.4f}, "
            f"residual translation={residual_translation.round(3).tolist()})",
        )
    else:
        _ants_rigid_log(
            scan_name,
            f"WARNING: ANTs→scipy analytic projection on non-rigid .mat "
            f"(ortho_err={ortho_err:.2e}, det={det:.4f})",
        )
    return rotation, scan_c, ref_c, residual_translation


def has_legacy_ants_affine_metadata(params):
    """True when metadata still references a persisted ANTs sidecar (pre-unification projects)."""
    if not isinstance(params, dict):
        return False
    affine = params.get('alignment_ants_affine')
    return bool(affine and str(affine).endswith('.mat'))


def _write_ants_rigid_mat(R, t, out_path=None):
    """Persist y = R x + t as an ANTs AffineTransform .mat (rotation+translation only)."""
    if out_path is None:
        fd, out_path = tempfile.mkstemp(suffix=".mat")
        os.close(fd)
    tx = ants.create_ants_transform(
        transform_type="AffineTransform",
        precision="float",
        dimension=3,
        matrix=np.asarray(R, dtype=np.float64),
        translation=np.asarray(t, dtype=np.float64).tolist(),
        center=(0.0, 0.0, 0.0),
    )
    ants.write_transform(tx, out_path)
    return out_path


def _score_rigid_metrics(fixed_image, moving_image, R, t):
    """
    NCC and overlap metrics after applying y = R x + t on the estimation grid.

    iou: mutual foreground / union (penalizes extra moving junk outside reference).
    fixed_recall: mutual / fixed foreground (reference coverage; robust when moving has artifacts).
    """
    tx_path = _write_ants_rigid_mat(R, t)
    try:
        warped = ants.apply_transforms(
            fixed=fixed_image,
            moving=moving_image,
            transformlist=[tx_path],
            interpolator="linear",
            defaultvalue=-1,
            verbose=False,
        )
        fixed = np.asarray(fixed_image.numpy(), dtype=np.float32)
        warped_np = np.asarray(warped.numpy(), dtype=np.float32)
        fixed_fg = fixed > -0.5
        warped_fg = warped_np > -0.5
        mutual = fixed_fg & warped_fg
        if mutual.sum() < 200:
            return {"ncc": -1.0, "iou": 0.0, "fixed_recall": 0.0, "ref_ncc": -1.0}
        ncc = float(np.corrcoef(fixed[mutual], warped_np[mutual])[0, 1])
        if ncc != ncc:
            ncc = -1.0
        ref_ncc = float(np.corrcoef(fixed[fixed_fg], warped_np[fixed_fg])[0, 1])
        if ref_ncc != ref_ncc:
            ref_ncc = -1.0
        union = fixed_fg | warped_fg
        iou = float(mutual.sum() / max(union.sum(), 1))
        fixed_recall = float(mutual.sum() / max(fixed_fg.sum(), 1))
        return {"ncc": ncc, "iou": iou, "fixed_recall": fixed_recall, "ref_ncc": ref_ncc}
    finally:
        try:
            os.remove(tx_path)
        except OSError:
            pass


def _score_rigid_on_overlap(fixed_image, moving_image, R, t):
    """Cheap NCC/overlap after applying y = R x + t with ANTs (same convention as refine)."""
    metrics = _score_rigid_metrics(fixed_image, moving_image, R, t)
    return metrics["ncc"], metrics["iou"]


def _score_ants_rigid_transform(fixed_image, moving_image, transform_path):
    """NCC/IoU after applying a saved ANTs rigid .mat to the registration grid."""
    if not transform_path or not os.path.isfile(transform_path):
        return -1.0, 0.0
    try:
        warped = ants.apply_transforms(
            fixed=fixed_image,
            moving=moving_image,
            transformlist=[transform_path],
            interpolator="linear",
            defaultvalue=-1,
            verbose=False,
        )
        fixed = np.asarray(fixed_image.numpy(), dtype=np.float32)
        warped_np = np.asarray(warped.numpy(), dtype=np.float32)
        fg = (fixed > -0.5) & (warped_np > -0.5)
        if fg.sum() < 200:
            return -1.0, 0.0
        ncc = float(np.corrcoef(fixed[fg], warped_np[fg])[0, 1])
        if ncc != ncc:
            ncc = -1.0
        iou = float(fg.sum() / max(((fixed > -0.5) | (warped_np > -0.5)).sum(), 1))
        return ncc, iou
    except Exception:
        return -1.0, 0.0


def _ants_rigid_mat_path_from_initial(initial_transform):
    if isinstance(initial_transform, str) and initial_transform.endswith(".mat") and os.path.isfile(initial_transform):
        return initial_transform
    return None


def _voxel_patch_ransac_at_scale(
    fixed_image,
    moving_image,
    fixed_vol,
    moving_vol,
    spacing,
    origin,
    direction,
    opts,
    downsample_factor,
    scan_name=None,
):
    """
    Match interior patches on a downsampled pyramid level, then estimate rigid via RANSAC.

    Correspondences are found at *downsample_factor* (coarser = larger physical patches,
    more unique descriptors). The RANSAC rigid map is always scored on the ANTs
    registration grid (fixed_image / moving_image).
    """
    factor = max(1, int(downsample_factor))
    if factor > 1:
        fixed_ds = fixed_vol[::factor, ::factor, ::factor]
        moving_ds = moving_vol[::factor, ::factor, ::factor]
    else:
        fixed_ds = fixed_vol
        moving_ds = moving_vol

    base_radius = int(opts.get("patch_radius", 6) or 6)
    base_sep = int(opts.get("patch_min_separation", 4) or 4)
    # Keep physical patch footprint roughly constant across pyramid levels.
    radius = max(3, int(round(base_radius / factor)))
    min_sep = max(2, int(round(base_sep / factor)))
    n_points = int(opts.get("patch_n_points", 180) or 180)
    n_shells = int(opts.get("patch_n_shells", 8) or 8)
    min_sim = float(opts.get("patch_min_similarity", 0.60) or 0.60)
    # Coarser scales tolerate slightly lower descriptor similarity (more averaging).
    if factor >= 4:
        min_sim = max(0.45, min_sim - 0.08)
    elif factor >= 2:
        min_sim = max(0.50, min_sim - 0.04)

    spacing_ds = tuple(float(s) * factor for s in spacing)

    fixed_mask = _voxel_patch_foreground_mask(fixed_ds)
    moving_mask = _voxel_patch_foreground_mask(moving_ds)
    fixed_pts = _voxel_patch_interest_points(fixed_ds, fixed_mask, n_points=n_points, min_separation=min_sep)
    moving_pts = _voxel_patch_interest_points(moving_ds, moving_mask, n_points=n_points, min_separation=min_sep)
    if len(fixed_pts) < 3 or len(moving_pts) < 3:
        return None

    moving_desc = _voxel_patch_radial_descriptor(moving_ds, moving_pts, radius=radius, n_shells=n_shells)
    fixed_desc = _voxel_patch_radial_descriptor(fixed_ds, fixed_pts, radius=radius, n_shells=n_shells)
    matches = _voxel_patch_mutual_matches(moving_desc, fixed_desc, min_similarity=min_sim)
    if len(matches) < 3:
        return None

    match_scores = np.array([m[0] for m in matches], dtype=np.float64)
    moving_phys = np.array(
        [_ants_index_to_physical(moving_pts[mi], spacing_ds, origin, direction) for _, mi, _ in matches],
        dtype=np.float64,
    )
    fixed_phys = np.array(
        [_ants_index_to_physical(fixed_pts[fi], spacing_ds, origin, direction) for _, _, fi in matches],
        dtype=np.float64,
    )

    stage_label = f" scale ÷{factor}"
    ransac = _voxel_patch_ransac_rigid(
        moving_phys,
        fixed_phys,
        match_scores,
        n_iterations=int(opts.get("patch_ransac_iterations", 400) or 400),
        min_inliers=int(opts.get("patch_ransac_min_inliers", 8) or 8),
        min_spread_mm=float(opts.get("patch_ransac_min_spread_mm", 2.5) or 2.5),
        inlier_residual_mm=float(opts.get("patch_max_kabsch_residual_mm", 12.0) or 12.0),
        top_candidates=int(opts.get("patch_ransac_top_candidates", 12) or 12),
        scan_name=scan_name,
        stage_label=stage_label,
    )
    if ransac is None:
        return None

    ncc, iou = _score_rigid_on_overlap(fixed_image, moving_image, ransac["R"], ransac["t"])
    return {
        "downsample_factor": factor,
        "matches": len(matches),
        "inliers": ransac["inliers"],
        "angle_deg": ransac["angle_deg"],
        "R": ransac["R"],
        "t": ransac["t"],
        "ncc": ncc,
        "iou": iou,
        "mean_match": ransac["mean_match"],
    }


def compute_voxel_patch_initial_transform(fixed_image, moving_image, opts, scan_name=None):
    """
    Multi-scale voxel initializer with similarity-weighted RANSAC on patch matches.

      1. Match rotation-invariant radial descriptors at coarse → fine pyramid levels.
      2. Sample minimal rigid sets (3 points) weighted by descriptor similarity; score
         hypotheses by geometric inliers × match agreement; re-fit Kabsch on inliers.
      3. Score on the ANTs registration grid (interior NCC).
      4. Accept the scale with the strongest inlier support that beats identity.

    No mesh or guidepoints; same RANSAC philosophy as ALPACA/DINO-Reg, on voxel patches.
    """
    pyramid_factors = opts.get("patch_pyramid_factors") or (4, 2, 1)
    if isinstance(pyramid_factors, (list, tuple)):
        pyramid_factors = tuple(int(f) for f in pyramid_factors if int(f) >= 1)
    else:
        pyramid_factors = (4, 2, 1)
    if not pyramid_factors:
        pyramid_factors = (1,)

    fixed_vol = np.asarray(fixed_image.numpy(), dtype=np.float32)
    moving_vol = np.asarray(moving_image.numpy(), dtype=np.float32)
    spacing = tuple(float(s) for s in fixed_image.spacing)
    origin = tuple(float(o) for o in fixed_image.origin)
    direction = np.asarray(fixed_image.direction, dtype=np.float64)

    _ants_rigid_log(
        scan_name,
        f"voxel-patch init: pyramid={list(pyramid_factors)}, "
        f"n_points={opts.get('patch_n_points', 180)}, base_radius={opts.get('patch_radius', 6)}",
    )

    identity_ncc, identity_iou = _score_rigid_on_overlap(
        fixed_image, moving_image, np.eye(3), np.zeros(3)
    )
    _ants_rigid_log(scan_name, f"voxel-patch identity baseline: ncc={identity_ncc:.3f} iou={identity_iou:.3f}")

    min_ransac_inliers = int(opts.get("patch_ransac_min_inliers", 8) or 8)
    best = None
    for factor in pyramid_factors:
        scale_result = _voxel_patch_ransac_at_scale(
            fixed_image,
            moving_image,
            fixed_vol,
            moving_vol,
            spacing,
            origin,
            direction,
            opts,
            factor,
            scan_name=scan_name,
        )
        if scale_result is None:
            _ants_rigid_log(scan_name, f"voxel-patch scale ÷{factor}: no RANSAC solution")
            continue

        inliers = scale_result["inliers"]
        ncc = scale_result["ncc"]
        iou = scale_result["iou"]
        angle = scale_result["angle_deg"]
        _ants_rigid_log(
            scan_name,
            f"voxel-patch scale ÷{factor}: matches={scale_result['matches']} "
            f"ransac_inliers={inliers} "
            f"ncc={ncc:.3f} iou={iou:.3f} rotation≈{angle:.1f}° "
            f"mean_match={scale_result['mean_match']:.3f}",
        )
        if not _accept_voxel_patch_candidate(
            ncc,
            iou,
            identity_ncc,
            identity_iou,
            angle_deg=angle,
            ransac_inliers=inliers,
            min_ransac_inliers=min_ransac_inliers,
        ):
            continue

        # Prefer stronger RANSAC inlier support, then overlap/NCC quality.
        score = (inliers, ncc + 0.20 * iou)
        if best is None or score > best[0]:
            best = (score, scale_result)

    if best is None:
        _ants_rigid_log(
            scan_name,
            "voxel-patch init rejected: no pyramid scale produced a RANSAC rigid "
            "that beat identity on overlap/NCC",
        )
        return None

    _, scale_result = best
    R, t = scale_result["R"], scale_result["t"]
    tx_path = _write_ants_rigid_mat(R, t)
    _ants_rigid_log(
        scan_name,
        f"voxel-patch init accepted (scale ÷{scale_result['downsample_factor']}, "
        f"ransac_inliers={scale_result['inliers']}): ncc={scale_result['ncc']:.3f} "
        f"iou={scale_result['iou']:.3f} rotation≈{scale_result['angle_deg']:.1f}° -> {tx_path}",
    )
    return tx_path


def _log_ants_rigid_transform_summary(scan_name, transform_path):
    if not transform_path or not os.path.isfile(transform_path):
        return
    try:
        tx = ants.read_transform(transform_path)
        params = np.asarray(tx.parameters, dtype=np.float64)
        linear = _ants_linear_matrix_from_parameters(params)
        u, _, vt = np.linalg.svd(linear)
        r = u @ vt
        if np.linalg.det(r) < 0:
            u[:, -1] *= -1
            r = u @ vt
        angle_deg = float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))))
        trans = params[9:12]
        _ants_rigid_log(
            scan_name,
            f"transform summary: rotation≈{angle_deg:.2f}°, "
            f"translation=({trans[0]:.4f}, {trans[1]:.4f}, {trans[2]:.4f}), det={np.linalg.det(linear):.4f}",
        )
    except Exception as exc:
        _ants_rigid_log(scan_name, f"transform summary unavailable: {exc}")


def _ants_affine_initializer_provenance(opts, path="affine_initializer"):
    """Recorded initializer path + parameters for Scientific Report provenance."""
    return {
        "path": str(path),
        "search_factor": int(opts["initializer_search_factor"]),
        "radian_fraction": float(opts["initializer_radian_fraction"]),
        "use_principal_axis": bool(opts["initializer_use_principal_axis"]),
        "local_search_iterations": int(opts["initializer_local_search_iterations"]),
    }


def compute_ants_rigid_initial_transform(fixed_image, moving_image, opts, scan_name=None):
    """
    Rigid initializer before ``ants.registration(type_of_transform='Rigid')``.

    Primary: ``ants.affine_initializer`` on the registration embed grid (dense overlap).
    Optional legacy: voxel-patch RANSAC when ``use_voxel_patch_init`` is True.
    Fallback: Identity.

    Returns ``(initial_transform, initializer_record)`` persisted as
    ``alignment_rigid_provenance.ants_initializer``.
    """
    if opts.get("use_affine_initializer", True):
        try:
            _ants_rigid_log(
                scan_name,
                "affine_initializer starting "
                f"(search={opts['initializer_search_factor']}, "
                f"radian_fraction={opts['initializer_radian_fraction']}, "
                f"use_principal_axis={opts['initializer_use_principal_axis']}, "
                f"local_iters={opts['initializer_local_search_iterations']})",
            )

            def _run_initializer():
                return ants.affine_initializer(
                    fixed_image,
                    moving_image,
                    search_factor=opts["initializer_search_factor"],
                    radian_fraction=opts["initializer_radian_fraction"],
                    use_principal_axis=opts["initializer_use_principal_axis"],
                    local_search_iterations=opts["initializer_local_search_iterations"],
                )

            txfn = _ants_run_with_heartbeat(scan_name, "affine_initializer", _run_initializer)
            _ants_rigid_log(scan_name, f"affine_initializer done -> {txfn}")
            _log_ants_rigid_transform_summary(scan_name, txfn)
            init_mat = _ants_rigid_mat_path_from_initial(txfn)
            if init_mat:
                return init_mat, _ants_affine_initializer_provenance(opts, path="affine_initializer")
            return txfn, _ants_affine_initializer_provenance(opts, path="affine_initializer")
        except Exception as exc:
            _ants_rigid_log(scan_name, f"affine_initializer failed ({exc}); trying fallback")

    if opts.get("use_voxel_patch_init", False):
        identity_ncc, identity_iou = _score_rigid_on_overlap(
            fixed_image, moving_image, np.eye(3), np.zeros(3)
        )
        try:
            patch_tx = compute_voxel_patch_initial_transform(
                fixed_image, moving_image, opts, scan_name=scan_name
            )
            if patch_tx:
                init_ncc, init_iou = _score_ants_rigid_transform(fixed_image, moving_image, patch_tx)
                init_mat = _ants_rigid_mat_path_from_initial(patch_tx)
                if init_mat and init_iou >= identity_iou - 0.01:
                    _log_ants_rigid_transform_summary(scan_name, patch_tx)
                    return patch_tx, {"path": "voxel_patch"}
                _ants_rigid_log(
                    scan_name,
                    f"voxel-patch init file rejected before registration "
                    f"(ncc={init_ncc:.3f} iou={init_iou:.3f} vs identity "
                    f"ncc={identity_ncc:.3f} iou={identity_iou:.3f})",
                )
        except Exception as exc:
            _ants_rigid_log(scan_name, f"voxel-patch init failed ({exc}); trying fallback")

    _ants_rigid_log(scan_name, "starting Affine from Identity")
    return ["Identity"], {"path": "identity"}


def build_alignment_rigid_provenance(
    method,
    scaling,
    outer_surface,
    interpolation,
    ants_rigid_opts=None,
    ants_initializer=None,
    gpu_rigid_opts=None,
    gpu_initializer=None,
    dino_reg_result=None,
):
    """
    Serializable rigid-alignment provenance for Scientific Report Methods and audit trails.
    All methods produce rotation + translation only (no deformable warp in this step).
    """
    method_norm = str(method or "").strip()
    provenance = {
        "method": method_norm,
        "transform_kind": "rigid",
        "scaling_enabled": bool(scaling),
        "outer_surface": bool(outer_surface),
        "interpolation": str(interpolation or "linear"),
    }
    if method_norm.lower() == "ants" and isinstance(ants_rigid_opts, dict):
        serializable = {}
        for key, val in ants_rigid_opts.items():
            if isinstance(val, tuple):
                serializable[key] = list(val)
            else:
                serializable[key] = val
        provenance["ants_rigid_options"] = serializable
    if method_norm.lower() == "ants" and isinstance(ants_initializer, dict):
        provenance["ants_initializer"] = ants_initializer
    if method_norm.lower() == "gpu-rigid" and isinstance(gpu_rigid_opts, dict):
        serializable = {}
        for key, val in gpu_rigid_opts.items():
            if isinstance(val, tuple):
                serializable[key] = list(val)
            else:
                serializable[key] = val
        provenance["gpu_rigid_options"] = serializable
    if method_norm.lower() == "gpu-rigid" and isinstance(gpu_initializer, dict):
        provenance["gpu_initializer"] = gpu_initializer
    if method_norm.lower() == "dino-reg" and isinstance(dino_reg_result, dict):
        # Compact pose diagnostics for Scientific Report (no arrays / debug paths).
        bag = {}
        for key in (
            "source",
            "probe_name",
            "angle_deg",
            "refined_cosine",
            "label",
        ):
            if key not in dino_reg_result or dino_reg_result[key] is None:
                continue
            val = dino_reg_result[key]
            if isinstance(val, (np.floating, float)):
                bag[key] = float(val)
            elif isinstance(val, (np.integer, int)):
                bag[key] = int(val)
            else:
                bag[key] = str(val)
        if bag:
            provenance["dino_reg_result"] = bag
    return provenance


def normalize_ants_rigid_options(request_data):
    """
    Parse and clip ANTs rigid registration options from the JSON request body.
    Invalid or missing values fall back to the previous hard-coded defaults.
    """
    raw = None
    if isinstance(request_data, dict):
        raw = request_data.get("ants_rigid_options")
    if not isinstance(raw, dict):
        raw = {}

    def _clip_int(v, lo, hi, default):
        try:
            return max(lo, min(hi, int(v)))
        except (TypeError, ValueError):
            return default

    def _clip_float(v, lo, hi, default):
        try:
            fv = float(v)
            if fv != fv:  # NaN
                return default
            return max(lo, min(hi, fv))
        except (TypeError, ValueError):
            return default

    def _clip_int_tuple(seq, lo, hi, defaults):
        if not isinstance(seq, (list, tuple)) or len(seq) != len(defaults):
            return tuple(defaults)
        return tuple(_clip_int(x, lo, hi, d) for x, d in zip(seq, defaults))

    allowed_metrics = {"mattes", "GC", "meansquares"}
    metric = raw.get("aff_metric", "mattes")
    if metric not in allowed_metrics:
        metric = "mattes"

    # Coarse-to-fine pyramid tuned for large pose gaps; initializer handles global rotation.
    default_iters = (2100, 1500, 1000, 200)
    default_shrink = (12, 8, 4, 2)
    default_sigmas = (6, 4, 2, 1)

    return {
        "aff_metric": metric,
        "aff_iterations": _clip_int_tuple(raw.get("aff_iterations"), 0, 5000, default_iters),
        "aff_shrink_factors": _clip_int_tuple(raw.get("aff_shrink_factors"), 1, 16, default_shrink),
        "aff_smoothing_sigmas": _clip_int_tuple(raw.get("aff_smoothing_sigmas"), 0, 8, default_sigmas),
        "aff_sampling": _clip_int(raw.get("aff_sampling"), 8, 256, 32),
        "aff_random_sampling_rate": _clip_float(raw.get("aff_random_sampling_rate"), 0.01, 1.0, 0.25),
        "grad_step": _clip_float(raw.get("grad_step"), 0.001, 1.0, 0.10),
        "use_histogram_matching": bool(raw.get("use_histogram_matching", True)),
        "singleprecision": bool(raw.get("singleprecision", True)),
        "use_registration_mask": bool(raw.get("use_registration_mask", True)),
        "max_estimation_voxels": _clip_int(raw.get("max_estimation_voxels"), 1_000_000, 200_000_000, 40_000_000),
        "use_voxel_patch_init": bool(raw.get("use_voxel_patch_init", False)),
        "patch_n_points": _clip_int(raw.get("patch_n_points"), 32, 800, 180),
        "patch_radius": _clip_int(raw.get("patch_radius"), 3, 16, 6),
        "patch_n_shells": _clip_int(raw.get("patch_n_shells"), 4, 16, 8),
        "patch_min_similarity": _clip_float(raw.get("patch_min_similarity"), 0.2, 0.95, 0.60),
        "patch_min_separation": _clip_int(raw.get("patch_min_separation"), 2, 16, 4),
        "patch_max_kabsch_residual_mm": _clip_float(raw.get("patch_max_kabsch_residual_mm"), 2.0, 40.0, 12.0),
        "patch_pyramid_factors": _clip_int_tuple(raw.get("patch_pyramid_factors"), 1, 16, (4, 2, 1)),
        "patch_ransac_iterations": _clip_int(
            raw.get("patch_ransac_iterations", raw.get("patch_triplet_samples")), 32, 2000, 400
        ),
        "patch_ransac_min_inliers": _clip_int(
            raw.get("patch_ransac_min_inliers", raw.get("patch_min_consensus_votes")), 3, 200, 8
        ),
        "patch_ransac_min_spread_mm": _clip_float(
            raw.get("patch_ransac_min_spread_mm", raw.get("patch_triplet_min_sep_mm")), 0.5, 20.0, 2.5
        ),
        "patch_ransac_top_candidates": _clip_int(raw.get("patch_ransac_top_candidates"), 1, 50, 12),
        "use_affine_initializer": bool(raw.get("use_affine_initializer", True)),
        "initializer_search_factor": _clip_int(raw.get("initializer_search_factor"), 5, 60, 15),
        "initializer_radian_fraction": _clip_float(raw.get("initializer_radian_fraction"), 0.01, 1.0, 1.0),
        "initializer_use_principal_axis": bool(raw.get("initializer_use_principal_axis", True)),
        "initializer_local_search_iterations": _clip_int(
            raw.get("initializer_local_search_iterations"), 0, 100, 10
        ),
    }


class AlignToReferenceView(APIView):

    registration_tools = RegistrationTools()    

    def get_reference_base_path(self, directory, reference):
        """
        Get the base path for a reference scan.
        Atlas is at the top level, other scans are in extracted folder.
        """
        if reference == "atlas":
            return os.path.join(directory, "atlas")
        else:
            return os.path.join(directory, "extracted", reference)

    # ------------------------------------------------------------------
    # Preserved mesh helpers
    # ------------------------------------------------------------------
    # These helpers keep PLY-only subjects on a clearly separate path from the
    # voxel/NIfTI alignment code. Voxel subjects still use marching cubes and
    # NIfTI saves; preserved meshes are loaded directly from their PLY and saved
    # back as PLY edits.
    def _load_subject_metadata(self, directory, scan_name):
        base_path = self.get_reference_base_path(directory, scan_name)
        with open(os.path.join(base_path, f"{scan_name}.json"), "r") as jf:
            return json.load(jf)

    def _is_preserved_mesh_metadata(self, metadata):
        return metadata.get("is_mesh") is True and metadata.get("voxelized") is False

    def _latest_any_edit_number(self, directory, scan_name):
        base_path = self.get_reference_base_path(directory, scan_name)
        edit_number = 0
        while (
            glob.glob(os.path.join(base_path, f"{scan_name}_edit_{edit_number}_*.nii.gz")) or
            glob.glob(os.path.join(base_path, f"{scan_name}_edit_{edit_number}_*.ply"))
        ):
            edit_number += 1
        return edit_number - 1

    def _latest_any_edit_stem(self, directory, scan_name):
        """
        Return (edit_number, stem) for the latest PLY/NIfTI edit.

        edit_number is -1 when the scan has no edits yet (same contract as
        `_latest_any_edit_number`). Callers use `edit_number >= 0` and
        `edit_number + 1` for the next slot, so returning None here used to make
        linked-child ALPACA propagation crash before writing any sibling edit.
        """
        latest_edit = self._latest_any_edit_number(directory, scan_name)
        if latest_edit < 0:
            return -1, None
        base_path = self.get_reference_base_path(directory, scan_name)
        matches = (
            glob.glob(os.path.join(base_path, f"{scan_name}_edit_{latest_edit}_*.ply")) +
            glob.glob(os.path.join(base_path, f"{scan_name}_edit_{latest_edit}_*.nii.gz"))
        )
        if not matches:
            return latest_edit, None
        basename = os.path.basename(matches[0])
        for suffix in (".nii.gz", ".ply"):
            if basename.endswith(suffix):
                basename = basename[:-len(suffix)]
                break
        return latest_edit, basename

    def _preserved_mesh_path(self, directory, scan_name, metadata, edit_stem=None):
        base_path = self.get_reference_base_path(directory, scan_name)
        if edit_stem:
            mesh_filename = os.path.basename(edit_stem)
            if mesh_filename.endswith((".json", ".mat", ".png", ".nii.gz")):
                raise ValueError(f"Preserved mesh edit stem must reference a PLY mesh, got: {mesh_filename}")
            if not mesh_filename.endswith(".ply"):
                mesh_filename = f"{mesh_filename}.ply"
        else:
            mesh_filename = metadata.get("mesh_file") or f"{scan_name}.ply"
        mesh_path = os.path.join(base_path, mesh_filename)
        if not os.path.isfile(mesh_path):
            raise FileNotFoundError(f"Preserved mesh file not found: {mesh_path}")
        return mesh_path

    def _load_preserved_mesh_arrays(self, directory, scan_name, metadata, edit_stem=None):
        mesh_path = self._preserved_mesh_path(directory, scan_name, metadata, edit_stem)
        mesh_obj = trimesh.load(mesh_path, process=False)
        if isinstance(mesh_obj, trimesh.Scene):
            geometries = [
                geom for geom in mesh_obj.geometry.values()
                if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) > 0
            ]
            if not geometries:
                raise ValueError(f"No mesh geometry found in {mesh_path}")
            mesh_obj = trimesh.util.concatenate(geometries)
        if not isinstance(mesh_obj, trimesh.Trimesh) or len(mesh_obj.vertices) == 0 or len(mesh_obj.faces) == 0:
            raise ValueError(f"Preserved mesh has no usable vertices/faces: {mesh_path}")
        return (
            np.asarray(mesh_obj.vertices, dtype=np.float64),
            np.asarray(mesh_obj.faces, dtype=np.int64),
            mesh_obj,
            mesh_path,
        )

    def _swap_xz_for_voxel_display_basis(self, points):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Expected an Nx3 coordinate array.")
        return points[:, [2, 1, 0]]

    def _volume_for_mesh_reference_save(self, volume, reference_is_preserved_mesh):
        """Map an aligned canvas from display basis back to on-disk NIfTI storage layout."""
        if reference_is_preserved_mesh:
            return swap_voxel_volume_xz(volume)
        return volume

    def _padding_for_rigid_volume_warp(self, scan_shape, rotation_center, rotation, translation=None):
        """
        Pad a volume so rotation (and optional translation) around rotation_center
        does not clip foreground voxels. Same corner-expansion strategy as manual
        guidepoints alignment.
        """
        scan_shape = np.asarray(scan_shape, dtype=np.float64)
        rotation_center = np.asarray(rotation_center, dtype=np.float64)
        rotation = np.asarray(rotation, dtype=np.float64)

        corners = np.array([
            [0, 0, 0],
            [scan_shape[0], 0, 0],
            [0, scan_shape[1], 0],
            [0, 0, scan_shape[2]],
            [scan_shape[0], scan_shape[1], 0],
            [scan_shape[0], 0, scan_shape[2]],
            [0, scan_shape[1], scan_shape[2]],
            [scan_shape[0], scan_shape[1], scan_shape[2]],
        ], dtype=np.float64)

        corners_centered = corners - rotation_center
        # Same row-vector convention as guidepoints / kabsch: (v - c) @ rotation + c.
        warped_corners = corners_centered @ rotation + rotation_center
        if translation is not None:
            warped_corners = warped_corners + np.asarray(translation, dtype=np.float64)

        min_coords = np.floor(np.min(warped_corners, axis=0)).astype(int)
        max_coords = np.ceil(np.max(warped_corners, axis=0)).astype(int)
        padding_low = np.maximum(0, -min_coords)
        padding_high = np.maximum(0, max_coords - np.ceil(scan_shape).astype(int))
        return padding_low, padding_high

    def _apply_scipy_rigid_volume_warp(
        self,
        scan_data,
        reference_data,
        rotation,
        scan_centroid_voxel,
        reference_centroid_voxel,
        reference_shape,
        background_value,
        interpolation_order,
        border_width,
        original_voxel_size,
        reference_voxel_size,
        scale_match=1.0,
        residual_translation=None,
        scan_name="",
    ):
        """
        Universal scipy rigid warp: optional scale → pad → rotate about centroid →
        optional residual shift → integer canvas embed onto the reference FOV.

        Shared by manual guidepoints, DINO-Reg, and ANTs/GPU-rigid (after analytic pose projection).
        """
        rotation = np.asarray(rotation, dtype=np.float64)
        scan_centroid_voxel = np.asarray(scan_centroid_voxel, dtype=np.float64).reshape(3)
        reference_centroid_voxel = np.asarray(reference_centroid_voxel, dtype=np.float64).reshape(3)
        reference_shape = np.asarray(reference_shape, dtype=np.int64).reshape(3)
        residual_translation = np.zeros(3, dtype=np.float64) if residual_translation is None else np.asarray(
            residual_translation, dtype=np.float64
        ).reshape(3)

        working_data = scan_data
        if float(scale_match) != 1.0:
            working_data = ndimage.zoom(
                scan_data,
                float(scale_match),
                order=interpolation_order,
                mode='constant',
                cval=background_value,
            )
            scan_centroid_voxel = scan_centroid_voxel * float(scale_match)

        padding_low, padding_high = self._padding_for_rigid_volume_warp(
            working_data.shape,
            scan_centroid_voxel,
            rotation,
            translation=residual_translation if np.any(residual_translation) else None,
        )

        padded_scan_data = np.pad(
            working_data,
            ((padding_low[0], padding_high[0]),
             (padding_low[1], padding_high[1]),
             (padding_low[2], padding_high[2])),
            mode='constant',
            constant_values=background_value,
        )

        padded_scan_centroid = scan_centroid_voxel + padding_low
        # Forward: (v - c) @ rotation + c  ⇒  affine_transform matrix = rotation
        # (output→input: x_in = rotation @ (x_out - c) + c).
        offset = padded_scan_centroid - np.dot(rotation, padded_scan_centroid)

        transformed_data = ndimage.affine_transform(
            padded_scan_data,
            rotation,
            offset=offset,
            output_shape=padded_scan_data.shape,
            order=interpolation_order,
            mode='constant',
            cval=background_value,
        )

        if np.any(residual_translation):
            transformed_data = ndimage.shift(
                transformed_data,
                shift=residual_translation,
                order=interpolation_order,
                mode='constant',
                cval=background_value,
            )

        # Residual moves content in the padded array; canvas paste must anchor on the
        # pre-residual rotation centre so residual is not cancelled by the snap.
        embed_centroid_voxel = padded_scan_centroid + residual_translation
        paste_anchor_voxel = padded_scan_centroid
        transformed_data_canvas = np.full_like(reference_data, background_value)

        in_start_x = max(0, int(reference_centroid_voxel[0]) - int(paste_anchor_voxel[0]))
        in_start_y = max(0, int(reference_centroid_voxel[1]) - int(paste_anchor_voxel[1]))
        in_start_z = max(0, int(reference_centroid_voxel[2]) - int(paste_anchor_voxel[2]))

        in_end_x = min(
            reference_shape[0],
            int(reference_centroid_voxel[0]) - int(paste_anchor_voxel[0]) + transformed_data.shape[0],
        )
        in_end_y = min(
            reference_shape[1],
            int(reference_centroid_voxel[1]) - int(paste_anchor_voxel[1]) + transformed_data.shape[1],
        )
        in_end_z = min(
            reference_shape[2],
            int(reference_centroid_voxel[2]) - int(paste_anchor_voxel[2]) + transformed_data.shape[2],
        )

        out_start_x = max(0, int(paste_anchor_voxel[0]) - int(reference_centroid_voxel[0]))
        out_start_y = max(0, int(paste_anchor_voxel[1]) - int(reference_centroid_voxel[1]))
        out_start_z = max(0, int(paste_anchor_voxel[2]) - int(reference_centroid_voxel[2]))

        out_end_x = min(
            transformed_data.shape[0],
            int(paste_anchor_voxel[0]) - int(reference_centroid_voxel[0]) + reference_shape[0],
        )
        out_end_y = min(
            transformed_data.shape[1],
            int(paste_anchor_voxel[1]) - int(reference_centroid_voxel[1]) + reference_shape[1],
        )
        out_end_z = min(
            transformed_data.shape[2],
            int(paste_anchor_voxel[2]) - int(reference_centroid_voxel[2]) + reference_shape[2],
        )

        if (
            in_end_x - in_start_x != out_end_x - out_start_x
            or in_end_y - in_start_y != out_end_y - out_start_y
            or in_end_z - in_start_z != out_end_z - out_start_z
        ):
            print(f"Index error will occur, skipping {scan_name}")
            print("x in width: ", in_end_x - in_start_x, "out width: ", out_end_x - out_start_x)
            print("y in width: ", in_end_y - in_start_y, "out width: ", out_end_y - out_start_y)
            print("z in width: ", in_end_z - in_start_z, "out width: ", out_end_z - out_start_z)
            return None

        transformed_data_canvas[
            in_start_x:in_end_x,
            in_start_y:in_end_y,
            in_start_z:in_end_z,
        ] = transformed_data[
            out_start_x:out_end_x,
            out_start_y:out_end_y,
            out_start_z:out_end_z,
        ]
        transformed_data = transformed_data_canvas

        if transformed_data.shape[0] > 2 * border_width:
            transformed_data[:border_width, :, :] = background_value
            transformed_data[-border_width:, :, :] = background_value
        if transformed_data.shape[1] > 2 * border_width:
            transformed_data[:, :border_width, :] = background_value
            transformed_data[:, -border_width:, :] = background_value
        if transformed_data.shape[2] > 2 * border_width:
            transformed_data[:, :, :border_width] = background_value
            transformed_data[:, :, -border_width:] = background_value

        embed = {
            'padding_low': padding_low,
            'padding_high': padding_high,
            'rotation': rotation,
            'offset': offset,
            'centroid_translation': residual_translation,
            'padded_scan_centroid': padded_scan_centroid,
            'embed_centroid_voxel': embed_centroid_voxel,
            'reference_centroid_voxel': reference_centroid_voxel,
            'crop_indices': {
                'x': (in_start_x, in_end_x),
                'y': (in_start_y, in_end_y),
                'z': (in_start_z, in_end_z),
            },
            'reference_shape': reference_shape,
            'reference_voxel_size': reference_voxel_size,
            'original_voxel_size': original_voxel_size,
            'scale_match': scale_match,
        }
        return {
            'transformed_data': transformed_data,
            'embed': embed,
            'rotation': rotation,
            'offset': offset,
            'padding_low': padding_low,
            'padding_high': padding_high,
            'centroid_translation': residual_translation,
            'padded_scan_centroid': padded_scan_centroid,
            'embed_centroid_voxel': embed_centroid_voxel,
            'reference_centroid_voxel': reference_centroid_voxel,
            'in_start_x': in_start_x, 'in_end_x': in_end_x,
            'in_start_y': in_start_y, 'in_end_y': in_end_y,
            'in_start_z': in_start_z, 'in_end_z': in_end_z,
            'out_start_x': out_start_x, 'out_end_x': out_end_x,
            'out_start_y': out_start_y, 'out_end_y': out_end_y,
            'out_start_z': out_start_z, 'out_end_z': out_end_z,
        }

    def _apply_scipy_rigid_from_ants_mat(
        self,
        scan_data,
        reference_data,
        mat_path,
        *,
        scan_centroid_voxel,
        reference_centroid_voxel,
        rotation_center_voxel,
        fixed_spacing,
        fixed_origin,
        fixed_direction,
        mov_paste,
        ref_paste,
        reference_shape,
        background_value,
        interpolation_order,
        border_width,
        original_voxel_size,
        reference_voxel_size,
        scale_match=1.0,
        foreground_threshold,
        scan_name="",
    ):
        """Estimate ANTs/GPU pose, project to scipy parameters, apply unified warp."""
        rotation, scan_centroid, reference_centroid, residual_translation = project_ants_mat_to_scipy_rigid_params(
            mat_path,
            scan_centroid_voxel,
            reference_centroid_voxel,
            fixed_spacing,
            scan_name=scan_name,
        )
        is_rigid, ortho_err, det = verify_ants_rigid_transform_mat(mat_path)
        if not is_rigid:
            _ants_rigid_log(
                scan_name,
                f"WARNING: estimation GenericAffine may not be proper rigid "
                f"(ortho_err={ortho_err:.2e}, det={det:.4f}); scipy projection still applied",
            )
        return self._apply_scipy_rigid_volume_warp(
            scan_data,
            reference_data,
            rotation,
            scan_centroid,
            reference_centroid,
            reference_shape,
            background_value,
            interpolation_order,
            border_width,
            original_voxel_size,
            reference_voxel_size,
            scale_match=scale_match,
            residual_translation=residual_translation,
            scan_name=scan_name,
        )

    def _apply_scipy_rigid_from_physical_rt(
        self,
        scan_data,
        reference_data,
        R_col,
        t_phys,
        scan_centroid_voxel,
        reference_centroid_voxel,
        reference_shape,
        background_value,
        interpolation_order,
        border_width,
        original_voxel_size,
        reference_voxel_size,
        scale_match=1.0,
        scan_name="",
    ):
        """Apply y = R x + t (physical mm) by projecting into the shared scipy warp."""
        rotation, scan_centroid, reference_centroid, residual_translation = (
            project_physical_rigid_rt_to_scipy_rigid_params(
                R_col,
                t_phys,
                scan_centroid_voxel,
                reference_centroid_voxel,
                reference_voxel_size,
                scan_name=scan_name,
            )
        )
        return self._apply_scipy_rigid_volume_warp(
            scan_data,
            reference_data,
            rotation,
            scan_centroid,
            reference_centroid,
            reference_shape,
            background_value,
            interpolation_order,
            border_width,
            original_voxel_size,
            reference_voxel_size,
            scale_match=scale_match,
            residual_translation=residual_translation,
            scan_name=scan_name,
        )

    def _apply_rigid_warp_from_point_pairs(
        self,
        scan_data,
        reference_data,
        subject_pts_mm,
        reference_pts_mm,
        voxel_size,
        reference_voxel_size,
        reference_shape,
        background_value,
        interpolation_order,
        border_width,
        original_voxel_size,
        scale_match=1,
        scan_name="",
    ):
        """
        Kabsch rigid warp from homologous point pairs, then pad / rotate / embed
        onto the reference canvas. Shared by manual guidepoints.
        Returns a result dict, or None if the canvas embed would be invalid.
        """
        subject_pts_mm = np.asarray(subject_pts_mm, dtype=np.float64)
        reference_pts_mm = np.asarray(reference_pts_mm, dtype=np.float64)
        if subject_pts_mm.shape != reference_pts_mm.shape or subject_pts_mm.shape[0] < 3:
            print(f"Point-pair rigid warp skipped for {scan_name}: need matching Nx3 arrays with N>=3")
            return None

        scan_centroid_physical = np.mean(subject_pts_mm, axis=0)
        reference_centroid_physical = np.mean(reference_pts_mm, axis=0)
        scan_centroid_voxel = scan_centroid_physical / float(voxel_size)
        reference_centroid_voxel = reference_centroid_physical / float(reference_voxel_size)

        subject_voxel = subject_pts_mm / float(voxel_size)
        reference_voxel = reference_pts_mm / float(reference_voxel_size)
        rotation, _, _ = kabsch_rotation_matrix(
            subject_voxel,
            reference_voxel,
            source_centroid=scan_centroid_voxel,
            target_centroid=reference_centroid_voxel,
        )
        print(f"Rotation assuming both scans have their centroid as the origin: {rotation}")

        return self._apply_scipy_rigid_volume_warp(
            scan_data,
            reference_data,
            rotation,
            scan_centroid_voxel,
            reference_centroid_voxel,
            reference_shape,
            background_value,
            interpolation_order,
            border_width,
            original_voxel_size,
            reference_voxel_size,
            scale_match=scale_match,
            residual_translation=np.zeros(3, dtype=np.float64),
            scan_name=scan_name,
        )

    def _pre_align_nifti_path(self, directory, scan_name):
        """Latest full-res NIfTI that is not an aligned/elastic result (source for catch-up)."""
        scan_dir = os.path.join(directory, "extracted", scan_name)
        latest = self._latest_any_edit_number(directory, scan_name)
        for edit_number in range(latest, -1, -1):
            matches = [
                path for path in glob.glob(os.path.join(scan_dir, f"{scan_name}_edit_{edit_number}_*.nii.gz"))
                if '_lossy' not in os.path.basename(path)
            ]
            if not matches:
                continue
            matches.sort()
            path = matches[-1]
            name = os.path.basename(path).lower()
            if 'aligned' in name or 'elastic' in name:
                continue
            return path
        original = os.path.join(scan_dir, f"{scan_name}.nii.gz")
        return original if os.path.isfile(original) else None

    def _reconstruct_alpaca_propagation_geometry(self, directory, scan_name, scan_metadata, reference_shape):
        """
        Rebuild the ALPACA pad/rotate/translate/crop knobs from saved metadata so an
        already-aligned main can still push the same transform onto unaligned children.
        """
        acp = scan_metadata.get('alignment_calculated_parameters') or {}
        required = (
            'alignment_rotation_matrix',
            'alignment_rotation_center_voxel',
            'alignment_canvas_centroid_voxel',
            'alignment_padding_low',
            'alignment_translation_voxel',
        )
        missing = [key for key in required if key not in acp]
        if missing:
            raise ValueError(f"alignment_calculated_parameters missing {missing} for {scan_name}")

        rotation = np.asarray(acp['alignment_rotation_matrix'], dtype=np.float64)
        padding_bottom_val = np.asarray(acp['alignment_padding_low'], dtype=np.int64)
        translation = np.asarray(acp['alignment_translation_voxel'], dtype=np.float64)
        padded_target_centroid = np.asarray(acp['alignment_rotation_center_voxel'], dtype=np.float64)
        reference_centroid_voxel = np.asarray(acp['alignment_canvas_centroid_voxel'], dtype=np.float64)
        scale_match = float(scan_metadata.get('alignment_scale') or 1.0)
        offset = padded_target_centroid - np.dot(rotation.T, padded_target_centroid)

        if acp.get('alignment_padding_high') is not None:
            padding_top_val = np.asarray(acp['alignment_padding_high'], dtype=np.int64)
        else:
            pre_align_path = self._pre_align_nifti_path(directory, scan_name)
            if not pre_align_path:
                raise ValueError(f"Cannot recompute ALPACA padding_top; no pre-align volume for {scan_name}")
            pre_shape = nib.load(pre_align_path).shape[:3]
            target_centroid = padded_target_centroid - padding_bottom_val
            _recomputed_bottom, padding_top_val = self._padding_for_rigid_volume_warp(
                pre_shape, target_centroid, rotation
            )

        crop = acp.get('alignment_crop_indices')
        if crop and crop.get('in') and crop.get('out'):
            in_start_x, in_end_x = crop['in']['x']
            in_start_y, in_end_y = crop['in']['y']
            in_start_z, in_end_z = crop['in']['z']
            out_start_x, out_end_x = crop['out']['x']
            out_start_y, out_end_y = crop['out']['y']
            out_start_z, out_end_z = crop['out']['z']
        else:
            pre_align_path = self._pre_align_nifti_path(directory, scan_name)
            if not pre_align_path:
                raise ValueError(f"Cannot recompute ALPACA crop indices; no pre-align volume for {scan_name}")
            pre_shape = np.asarray(nib.load(pre_align_path).shape[:3], dtype=np.int64)
            transformed_shape = pre_shape + padding_bottom_val + padding_top_val
            transformed_data_centroid = padded_target_centroid
            in_start_x = max(0, int(reference_centroid_voxel[0]) - int(transformed_data_centroid[0]))
            in_start_y = max(0, int(reference_centroid_voxel[1]) - int(transformed_data_centroid[1]))
            in_start_z = max(0, int(reference_centroid_voxel[2]) - int(transformed_data_centroid[2]))
            in_end_x = min(reference_shape[0], int(reference_centroid_voxel[0]) - int(transformed_data_centroid[0]) + transformed_shape[0])
            in_end_y = min(reference_shape[1], int(reference_centroid_voxel[1]) - int(transformed_data_centroid[1]) + transformed_shape[1])
            in_end_z = min(reference_shape[2], int(reference_centroid_voxel[2]) - int(transformed_data_centroid[2]) + transformed_shape[2])
            out_start_x = max(0, int(transformed_data_centroid[0]) - int(reference_centroid_voxel[0]))
            out_start_y = max(0, int(transformed_data_centroid[1]) - int(reference_centroid_voxel[1]))
            out_start_z = max(0, int(transformed_data_centroid[2]) - int(reference_centroid_voxel[2]))
            out_end_x = min(transformed_shape[0], int(transformed_data_centroid[0]) - int(reference_centroid_voxel[0]) + reference_shape[0])
            out_end_y = min(transformed_shape[1], int(transformed_data_centroid[1]) - int(reference_centroid_voxel[1]) + reference_shape[1])
            out_end_z = min(transformed_shape[2], int(transformed_data_centroid[2]) - int(reference_centroid_voxel[2]) + reference_shape[2])

        return {
            'rotation': rotation,
            'padding_bottom_val': padding_bottom_val,
            'padding_top_val': padding_top_val,
            'translation': translation,
            'offset': offset,
            'scale_match': scale_match,
            'in_start_x': int(in_start_x), 'in_end_x': int(in_end_x),
            'in_start_y': int(in_start_y), 'in_end_y': int(in_end_y),
            'in_start_z': int(in_start_z), 'in_end_z': int(in_end_z),
            'out_start_x': int(out_start_x), 'out_end_x': int(out_end_x),
            'out_start_y': int(out_start_y), 'out_end_y': int(out_end_y),
            'out_start_z': int(out_start_z), 'out_end_z': int(out_end_z),
        }

    def _prepare_mesh_reference_alignment_context(
        self,
        directory,
        reference_metadata,
        reference_vertices_local,
        reference_faces,
    ):
        """
        Build the imaginary voxel reference canvas used when a preserved mesh is the
        rigid-alignment reference. Vertices are expressed in shared prism corner-origin mm.

        Spacing follows ``resolve_mesh_reference_prism``: persisted
        ``ply_reference_voxel_size`` in project_settings.json when set, otherwise
        the median native voxel size across voxel subjects (saved on first use).
        """
        mesh_metadata = reference_metadata.get("mesh_metadata")
        prism = resolve_mesh_reference_prism(
            reference_vertices_local,
            mesh_metadata,
            directory,
        )
        reference_voxel_size = float(prism["voxel_size"])
        reference_vertices_shared = mesh_local_to_shared_mm(
            reference_vertices_local,
            mesh_metadata,
            prism_vertices=reference_vertices_local,
            voxel_size=reference_voxel_size,
        )
        reference_shape = tuple(int(axis) for axis in prism["shape_voxels"])
        reference_threshold = int(reference_metadata.get("threshold", 1) or 1)
        background_value = 0
        reference_data = np.full(reference_shape, background_value, dtype=np.float32)
        reference_affine_data = np.eye(4, dtype=np.float64)
        reference_affine_data[0, 0] = reference_voxel_size
        reference_affine_data[1, 1] = reference_voxel_size
        reference_affine_data[2, 2] = reference_voxel_size
        return {
            "reference_vertices": reference_vertices_shared,
            "reference_faces": reference_faces,
            "reference_data": reference_data,
            "reference_shape": reference_shape,
            "reference_voxel_size": reference_voxel_size,
            "reference_affine_data": reference_affine_data,
            "reference_threshold": reference_threshold,
            "reference_vertices_local": reference_vertices_local,
        }

    def _save_preserved_aligned_mesh(
        self,
        directory,
        scan_name,
        scan_metadata,
        transformed_vertices,
        faces,
        reference,
        method,
        source_mesh_path,
        transform_summary,
        transformed_guidepoints=None,
        guidepoints_origin=None,
        alignment_rigid_provenance=None,
    ):
        base_path = self.get_reference_base_path(directory, scan_name)
        new_edit_number = self._latest_any_edit_number(directory, scan_name) + 1
        edit_filename = f"{scan_name}_edit_{new_edit_number}_aligned.ply"
        output_path = os.path.join(base_path, edit_filename)
        aligned_mesh = trimesh.Trimesh(
            vertices=np.asarray(transformed_vertices, dtype=np.float64),
            faces=np.asarray(faces, dtype=np.int64),
            process=False,
        )
        PreservedMeshEditWriter.save(aligned_mesh, output_path)

        scan_metadata["alignment_to"] = reference
        scan_metadata["alignment_shift"] = 0
        scan_metadata["alignment_method"] = method
        if isinstance(alignment_rigid_provenance, dict):
            scan_metadata["alignment_rigid_provenance"] = alignment_rigid_provenance
        existing_mesh_metadata = scan_metadata.get("mesh_metadata")
        if not isinstance(existing_mesh_metadata, dict):
            existing_mesh_metadata = {}
        scan_metadata["mesh_metadata"] = {
            **existing_mesh_metadata,
            "vertex_count": int(len(aligned_mesh.vertices)),
            "face_count": int(len(aligned_mesh.faces)),
            "bounds": np.asarray(aligned_mesh.bounds, dtype=float).tolist(),
            "extents": np.asarray(aligned_mesh.extents, dtype=float).tolist(),
            "is_watertight": bool(aligned_mesh.is_watertight),
        }
        strip_ephemeral_scan_metadata(scan_metadata)
        with open(os.path.join(base_path, f"{scan_name}.json"), "w") as jf:
            json.dump(scan_metadata, jf, indent=4)

        edit_stem = f"{scan_name}_edit_{new_edit_number}_aligned"
        if transformed_guidepoints is not None:
            guidepoints_path = os.path.join(base_path, f"{edit_stem}_guidepoints.json")
            with open(guidepoints_path, "w") as jf:
                json.dump(np.asarray(transformed_guidepoints, dtype=np.float64).tolist(), jf, indent=4)
            print(f"Saved aligned preserved-mesh guidepoints: {guidepoints_path}")
        return new_edit_number, output_path

    def _load_guidepoints_for_stem(self, directory, scan_name, edit_stem=None):
        base_path = self.get_reference_base_path(directory, scan_name)
        if edit_stem:
            guidepoint_path = os.path.join(base_path, f"{edit_stem}_guidepoints.json")
        else:
            guidepoint_path = os.path.join(base_path, f"{scan_name}_guidepoints.json")
        if not os.path.isfile(guidepoint_path):
            return None, None
        with open(guidepoint_path, "r") as jf:
            return guidepoint_path, np.asarray(json.load(jf), dtype=np.float64)

    def clean_mesh(self, nifti_data, threshold, original_dtype, background_value):
        """
        This function is used to clean up the mesh by removing small disconnected components and trying to separate connected objects
        """

        # To identify and remove small disconnected components, use threshold to identify the islands, keep the largest island
        islands = (nifti_data > threshold).astype(bool)
        
        # Label connected components
        labeled_islands, num_features = ndimage.label(islands)
        
        # Find the sizes of each component
        sizes = np.bincount(labeled_islands.ravel())
               
        # Exclude the background (size of 0)
        sizes = sizes[1:]  # Skip the first element which corresponds to the background
        
        if sizes.size == 0:
            print("No islands found.")
            cleaned_data = np.zeros_like(nifti_data, dtype=original_dtype)  # No islands, return empty data
            largest_island_mask = np.zeros_like(nifti_data, dtype=bool)
        else:
            # print("Sizes: ", sizes)
            largest_island = sizes.argmax() + 1  # +1 to account for background
            print("Largest island label: ", largest_island)

            # Create an array of labels excluding the largest island and the background
            labels_without_largest = np.where((labeled_islands != largest_island) & (labeled_islands > 0), labeled_islands, 0)

            # Apply a dilation of one voxel to the mask to ensure the mask is connected
            islands_mask = ndimage.binary_dilation(labels_without_largest, iterations=1)

            # Replace the corresponding islands_mask pixels from nifti_data with background_value
            nifti_data[islands_mask] = background_value

            cleaned_data = nifti_data

        return cleaned_data.astype(original_dtype)

    def store_alignment_embedding(self, method, transformation_data):
        """
        Store embedding information needed for landmark transformation.
        Similar to _rotation_embed in ApplyRotationView.
        """
        if method.lower() == 'alpaca':
            self._alignment_embed = {
                'method': 'alpaca',
                'padding_bottom': transformation_data.get('padding_bottom_val'),
                'padding_top': transformation_data.get('padding_top_val'),
                'rotation': transformation_data.get('rotation'),
                'offset': transformation_data.get('offset'),
                'translation': transformation_data.get('translation'),
                'source_centroid': transformation_data.get('source_centroid'),
                'target_centroid': transformation_data.get('target_centroid'),
                'padded_target_centroid': transformation_data.get('padded_target_centroid'),
                'crop_indices': transformation_data.get('crop_indices'),
                'reference_shape': transformation_data.get('reference_shape'),
                'reference_voxel_size': transformation_data.get('reference_voxel_size'),
                'original_voxel_size': transformation_data.get('original_voxel_size'),
                'scale_match': transformation_data.get('scale_match'),
            }
        elif uses_scipy_centroid_embed_rigid(method):
            self._alignment_embed = {
                'method': method.lower(),
                'padding_low': transformation_data.get('padding_low'),
                'padding_high': transformation_data.get('padding_high'),
                'rotation': transformation_data.get('rotation'),
                'offset': transformation_data.get('offset'),
                'centroid_translation': transformation_data.get('centroid_translation'),
                'padded_scan_centroid': transformation_data.get('padded_scan_centroid'),
                'reference_centroid_voxel': transformation_data.get('reference_centroid_voxel'),
                'crop_indices': transformation_data.get('crop_indices'),
                'reference_shape': transformation_data.get('reference_shape'),
                'reference_voxel_size': transformation_data.get('reference_voxel_size'),
                'original_voxel_size': transformation_data.get('original_voxel_size'),
                'scale_match': transformation_data.get('scale_match'),
            }


    def transform_landmarks_after_alignment(self, directory, scan_name, latest_edit, new_edit_number,
                                            voxel_size, scan_metadata, method):
        """
        Transform landmarks using the stored alignment embedding information.
        Properly accounts for resampling and scaling transformations.
        """
        try:
            # Determine source landmark base from the latest edit
            source_landmark_base = f"{scan_name}_edit_{latest_edit}_*" if latest_edit >= 0 else scan_name
            source_landmarks_paths = glob.glob(
                os.path.join(directory, "extracted", scan_name, f"{source_landmark_base}_landmarks.json")
            )
            
            if not source_landmarks_paths:
                print(f"No landmarks found for {scan_name}, skipping landmark transformation")
                return
            
            source_landmarks_path = source_landmarks_paths[0]
            
            print(f"Found landmarks for {scan_name}: {source_landmarks_path}")
            with open(source_landmarks_path, 'r') as jf:
                landmarks_list = json.load(jf)
            
            if not hasattr(self, '_alignment_embed'):
                print("No alignment embedding stored, cannot transform landmarks")
                return
            
            embed = self._alignment_embed
            
            # Transform landmarks based on alignment method
            if embed['method'] == 'alpaca':
                transformed_landmarks = self._transform_landmarks_alpaca(
                    landmarks_list, embed, voxel_size
                )
                
            elif uses_scipy_centroid_embed_rigid(embed.get('method')):
                transformed_landmarks = self._transform_landmarks_guidepoints(
                    landmarks_list, embed, voxel_size
                )
            else:
                transformed_landmarks = None
            
            if transformed_landmarks:
                # Save transformed landmarks
                new_landmark_base = f"{scan_name}_edit_{new_edit_number}_aligned"
                dest_landmarks_path = os.path.join(
                    directory, "extracted", scan_name, f"{new_landmark_base}_landmarks.json"
                )
                with open(dest_landmarks_path, 'w') as jf:
                    json.dump(transformed_landmarks, jf, indent=4)
                print(f"Saved aligned landmarks: {dest_landmarks_path}")

                scan_dir = os.path.join(directory, "extracted", scan_name)
                json_path = os.path.join(scan_dir, f"{scan_name}.json")
                try:
                    compute_and_save_landmark_distances(
                        scan_dir,
                        scan_name,
                        new_landmark_base,
                        transformed_landmarks,
                        metadata=scan_metadata,
                        json_path=json_path,
                    )
                except Exception as dist_error:
                    print(f"Error recomputing aligned landmark distances for {scan_name}: {dist_error}")
                    source_landmark_distances_paths = glob.glob(
                        os.path.join(directory, "extracted", scan_name, f"{source_landmark_base}_landmark_distances.json")
                    )
                    if source_landmark_distances_paths:
                        dest_landmark_distances_path = os.path.join(
                            directory, "extracted", scan_name, f"{new_landmark_base}_landmark_distances.json"
                        )
                        shutil.copy(source_landmark_distances_paths[0], dest_landmark_distances_path)
                        print(f"Copied landmark distances: {dest_landmark_distances_path}")
            else:
                print(f"No landmarks remained for {scan_name} after alignment")
        
        except Exception as e:
            print(f"Error transforming landmarks for {scan_name}: {e}")
    
    def _transform_landmarks_alpaca(self, landmarks_list, embed, current_voxel_size):
        """
        Transform landmarks through ALPACA alignment pipeline.
        Accounts for: resampling -> scaling -> padding -> rotation -> translation -> cropping
        """
        transformed_landmarks = []
        
        # Get transformation parameters
        original_voxel_size = embed.get('original_voxel_size', current_voxel_size)
        reference_voxel_size = embed['reference_voxel_size']
        scale_match = embed.get('scale_match', 1.0)
        
        # Get geometric parameters
        rotation = embed['rotation']
        translation = embed['translation']
        padding_bottom_val = embed['padding_bottom']
        padded_target_centroid = embed['padded_target_centroid']
        source_centroid = embed['source_centroid']
        
        # Get crop indices
        final_bbox_min = np.array([
            embed['crop_indices']['x'][0],
            embed['crop_indices']['y'][0],
            embed['crop_indices']['z'][0]
        ])
        final_bbox_max = np.array([
            embed['crop_indices']['x'][1],
            embed['crop_indices']['y'][1],
            embed['crop_indices']['z'][1]
        ])
        reference_shape = np.array(embed['reference_shape'])
        
        print(f"ALPACA landmark transformation parameters:")
        print(f"  Original voxel size: {original_voxel_size}")
        print(f"  Reference voxel size: {reference_voxel_size}")
        print(f"  Scale match: {scale_match}")
        print(f"  Padding bottom: {padding_bottom_val}")
        print(f"  Crop bounds: [{final_bbox_min} to {final_bbox_max}]")
        
        for idx, landmark_data in enumerate(landmarks_list):
            try:
                landmark_mm, landmark_type = unpack_landmark_entry(landmark_data)

                # Step 1: Convert from physical space (mm) to original voxel space
                landmark_voxel = np.asarray(landmark_mm, dtype=np.float64) / float(original_voxel_size)
                
                # Step 2: Apply resampling (zoom_factor adjusts for voxel size change)
                zoom_factor = original_voxel_size / reference_voxel_size
                landmark_voxel = landmark_voxel * zoom_factor
                
                # Step 3: Apply scale matching (same scale applied to scan data)
                landmark_voxel = landmark_voxel * scale_match
                
                # Step 4: Apply padding offset (same padding applied to scan data)
                landmark_voxel = landmark_voxel + padding_bottom_val
                
                # Step 5: Apply rotation around padded_target_centroid
                # Rotate the point relative to the centroid, then translate back
                centered_landmark = landmark_voxel - padded_target_centroid
                rotated_landmark = np.dot(rotation, centered_landmark) + padded_target_centroid
                
                # Step 6: Apply translation
                translated_landmark = rotated_landmark + translation
                
                # Step 7: Map to reference canvas space
                # The canvas is created with the reference shape and filled at source_centroid position
                reference_centroid_voxel = source_centroid
                transformed_data_centroid = padded_target_centroid
                
                # Calculate where the transformed data will be placed in reference canvas.
                # Match the volume paste: integer in/out crop, not a continuous centroid snap.
                canvas_shift = integer_canvas_embed_shift(
                    transformed_data_centroid, reference_centroid_voxel
                )
                canvas_landmark = translated_landmark + canvas_shift
                
                # Check if within reference canvas bounds (before any cropping)
                if (0 <= canvas_landmark[0] < reference_shape[0] and
                    0 <= canvas_landmark[1] < reference_shape[1] and
                    0 <= canvas_landmark[2] < reference_shape[2]):
                    
                    # Convert back to physical space (mm) using reference voxel size
                    landmark_transformed_mm = canvas_landmark * float(reference_voxel_size)
                    
                    transformed_landmarks.append({
                        'position': landmark_transformed_mm.tolist(),
                        'landmark_type': landmark_type,
                    })
                    print(f"Landmark {idx}: {np.asarray(landmark_mm).tolist()} -> {landmark_transformed_mm.tolist()}")
                else:
                    print(f"Landmark {idx} at canvas position {canvas_landmark} is outside reference bounds {reference_shape}, excluding")
                    
            except Exception as e:
                print(f"Error transforming landmark {idx}: {e}")
        
        return transformed_landmarks
    
    def _transform_landmarks_guidepoints(self, landmarks_list, embed, current_voxel_size):
        """
        Transform landmarks through manual guidepoints alignment pipeline.
        Accounts for: resampling -> scaling -> padding -> rotation -> integer canvas paste.

        The volume is embedded with ``int(reference_centroid) - int(padded_scan_centroid)``
        (pre-residual anchor) so residual translation is preserved on the canvas.
        """
        transformed_landmarks = []
        
        # Get transformation parameters
        original_voxel_size = embed.get('original_voxel_size', current_voxel_size)
        reference_voxel_size = embed['reference_voxel_size']
        scale_match = embed.get('scale_match', 1.0)
        
        # Get geometric parameters
        rotation = embed['rotation']
        padding_low = embed['padding_low']
        padded_scan_centroid = embed['padded_scan_centroid']
        reference_centroid_voxel = embed['reference_centroid_voxel']
        reference_shape = np.array(embed['reference_shape'])
        translation = np.asarray(embed.get('centroid_translation') or [0, 0, 0], dtype=np.float64).reshape(3)
        embed_centroid_voxel = np.asarray(
            embed.get('embed_centroid_voxel', padded_scan_centroid + translation),
            dtype=np.float64,
        ).reshape(3)
        
        print(f"Guidepoints landmark transformation parameters:")
        print(f"  Original voxel size: {original_voxel_size}")
        print(f"  Reference voxel size: {reference_voxel_size}")
        print(f"  Scale match: {scale_match}")
        print(f"  Padding low: {padding_low}")
        print(f"  Rotation center (padded): {padded_scan_centroid}")
        print(f"  Embed centroid (post-residual): {embed_centroid_voxel}")
        print(f"  Canvas centroid: {reference_centroid_voxel}")
        canvas_shift = integer_canvas_embed_shift(
            padded_scan_centroid, reference_centroid_voxel
        )
        print(f"  Canvas integer embed shift: {canvas_shift.tolist()}")
        
        for idx, landmark_data in enumerate(landmarks_list):
            try:
                landmark_mm, landmark_type = unpack_landmark_entry(landmark_data)

                # Step 1: Convert from physical space (mm) to original voxel space
                landmark_voxel = np.asarray(landmark_mm, dtype=np.float64) / float(original_voxel_size)

                # Step 2: Apply resampling to the reference voxel grid (same zoom as the volume)
                zoom_factor = float(original_voxel_size) / float(reference_voxel_size)
                if zoom_factor != 1:
                    landmark_voxel = landmark_voxel * zoom_factor
                
                # Step 3: Apply scaling (if scaling was applied to scan data)
                if scale_match != 1:
                    landmark_voxel = landmark_voxel * scale_match
                
                # Step 4: Apply padding offset (same padding applied to scan data)
                landmark_voxel = landmark_voxel + padding_low
                
                # Step 5: Apply rotation around padded_scan_centroid (row-vector kabsch)
                centered_landmark = landmark_voxel - padded_scan_centroid
                rotated_landmark = centered_landmark @ rotation + padded_scan_centroid

                translated_landmark = rotated_landmark + translation
                
                # Step 6: Map to reference canvas with the same integer paste as the volume
                canvas_landmark = translated_landmark + canvas_shift
                
                # Check if within reference canvas bounds
                if (0 <= canvas_landmark[0] < reference_shape[0] and
                    0 <= canvas_landmark[1] < reference_shape[1] and
                    0 <= canvas_landmark[2] < reference_shape[2]):
                    
                    # Convert back to physical space (mm) using reference voxel size
                    landmark_transformed_mm = canvas_landmark * float(reference_voxel_size)
                    
                    transformed_landmarks.append({
                        'position': landmark_transformed_mm.tolist(),
                        'landmark_type': landmark_type,
                    })
                    print(f"Landmark {idx}: {np.asarray(landmark_mm).tolist()} -> {landmark_transformed_mm.tolist()}")
                else:
                    print(f"Landmark {idx} at canvas position {canvas_landmark} is outside reference bounds {reference_shape}, excluding")
                    
            except Exception as e:
                print(f"Error transforming landmark {idx}: {e}")
        
        return transformed_landmarks

    @staticmethod
    def _ants_numpy_index_to_physical(index, spacing, origin, direction):
        """Map continuous numpy voxel indices into the ants.from_numpy physical frame."""
        index = np.asarray(index, dtype=np.float64).reshape(3)
        spacing = np.asarray(spacing, dtype=np.float64).reshape(3)
        origin = np.asarray(origin, dtype=np.float64).reshape(3)
        direction = np.asarray(direction, dtype=np.float64).reshape(3, 3)
        return origin + direction @ (index * spacing)

    @staticmethod
    def _ants_physical_to_numpy_index(physical, spacing, origin, direction):
        """Inverse of ``_ants_numpy_index_to_physical``."""
        physical = np.asarray(physical, dtype=np.float64).reshape(3)
        spacing = np.asarray(spacing, dtype=np.float64).reshape(3)
        origin = np.asarray(origin, dtype=np.float64).reshape(3)
        direction = np.asarray(direction, dtype=np.float64).reshape(3, 3)
        return (np.linalg.inv(direction) @ (physical - origin)) / spacing

    def ants_transform_to_numpy(self, ants_transform, lps_to_ras=True):
        # Extract components
        params = ants_transform.parameters
        center = ants_transform.fixed_parameters
        
        # Reshape and transpose matrix for column-major format
        linear_matrix = np.array(params[:9]).reshape(3,3).T
        
        # Get adjusted translation (already accounts for center)
        translation = params[9:12]
        
        # Build 4x4 matrix
        affine_np = np.eye(4)
        affine_np[:3,:3] = linear_matrix
        affine_np[:3,3] = translation
        
        # Coordinate system conversion (LPS → RAS)
        if lps_to_ras:
            flip = np.diag([-1, -1, 1, 1])  # Z-axis flip for ANTs → NiBabel compatibility
            affine_np = flip @ affine_np @ flip
        
        return affine_np

    def get_background_value(self, scan_data, threshold):
        # Get values below threshold
        background_values = scan_data[scan_data < threshold]
        
        # Get the minimum value
        min_value = np.min(scan_data)
        
        # Count occurrences of minimum value
        min_count = np.sum(background_values == min_value)
        
        # Create bins for other background values (excluding min_value)
        other_background = background_values[background_values > min_value]
        
        if len(other_background) > 0:
            bins = np.arange(np.min(other_background), threshold, 100)
            # Ensure we have at least 2 bins
            if len(bins) < 2:
                bins = np.linspace(np.min(other_background), threshold, 2)
            
            # Calculate histogram of non-minimum background values
            hist, bin_edges = np.histogram(other_background, bins=bins)
            
            # If count of min_value is greater than any other bin, return min_value
            if min_count > np.max(hist):
                return min_value
            
            # Otherwise return the mode
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            return bin_centers[np.argmax(hist)]
        else:
            # If all background values are at minimum, return minimum
            return min_value

    def post(self, request):
        print("Aligning to reference scan")
        directory = request.data['directory']
        reference = request.data['reference']
        # Add method parameter with default value 'alpaca'
        method = request.data.get('method', 'alpaca')
        scaling = request.data.get('scaling', False)
        outer_surface = request.data.get('outer_surface', False)
        scale_match = 1
        # Linking is the opt-in: children skipped by the batch inherit the main's transform.
        propagate_linked = bool(request.data.get('propagate_linked', True))

        interpolation = request.data.get('interpolation', 'linear')
        if interpolation == 'nearest':
            interpolation_order = 0  # Nearest neighbor
        else:  # default to linear
            interpolation_order = 1  # Linear interpolation

        border_width = 20 # how many voxels to replace as background around the data

        flag_filter = normalize_flag_filter_value(
            request.data.get('flagFilter', 'off') if request.data else 'off',
            only_current_scan=False,
        )

        ants_rigid_opts = normalize_ants_rigid_options(request.data if request.data else {})
        from .gpuRigid import (
            GpuRigidError,
            ensure_gpu_rigid_cuda,
            normalize_gpu_rigid_options,
            run_gpu_rigid_registration,
        )
        from .gpuRigidDebug import resolve_gpu_rigid_debug_dir
        gpu_rigid_opts = normalize_gpu_rigid_options(request.data if request.data else {})
        
        print(f"Using alignment method: {method}")
        print(f"Flag filter: {flag_filter}")
        if method.lower() == "ants":
            print(f"ANTs rigid options: {ants_rigid_opts}")
        if method.lower() == "gpu-rigid":
            print(f"GPU Rigid options: {gpu_rigid_opts}")

        # Load the reference landmarks
        edit_number = 0
        reference_base_path = self.get_reference_base_path(directory, reference)
        while glob.glob(os.path.join(reference_base_path, f"{reference}_edit_{edit_number}_*.nii.gz")):
            edit_number += 1

        # Make a list of faulty files
        faulty_files = []
        for file in os.listdir(os.path.join(directory, "extracted")):
            # Skip project_settings.json file
            if file == "project_settings.json" or not os.path.isdir(os.path.join(directory, "extracted", file)):
                continue
            if file != reference:
                json_path = os.path.join(directory, "extracted", file, f"{file}.json")
                with open(json_path, 'r') as jf:
                    metadata = json.load(jf)
                    if metadata.get('faulty', False):
                        faulty_files.append(file)
        
        # Find all unique scan names by listing subdirectories in extracted folder
        scan_names = [d for d in os.listdir(os.path.join(directory, "extracted")) 
                     if os.path.isdir(os.path.join(directory, "extracted", d)) and d not in faulty_files]

        scan_names.sort()
        
        # Remove reference from the list of scans to process
        if reference in scan_names:
            scan_names.remove(reference)
        print(scan_names)

        # Iterate over every scan and if the latest edit is already aligned, remove it from scan_names
        # Exception: an already-aligned link main whose children still lack an aligned edit must stay
        # in the batch so we can catch up linked propagation without re-running ALPACA on the main.
        scans_to_remove = []
        propagate_only_scans = set()
        for scan_name in scan_names:

            # Remove from scan_names if any geometry edit is already aligned
            scan_dir = os.path.join(directory, "extracted", scan_name)
            edit_number = 0
            already_aligned = False
            while (
                glob.glob(os.path.join(scan_dir, f"{scan_name}_edit_{edit_number}_*.ply"))
                or glob.glob(os.path.join(scan_dir, f"{scan_name}_edit_{edit_number}_*.nii.gz"))
                or glob.glob(os.path.join(scan_dir, f"{scan_name}_lossy_edit_{edit_number}_*.nii.gz"))
            ):
                aligned_paths = [
                    os.path.join(scan_dir, f"{scan_name}_edit_{edit_number}_aligned.ply"),
                    os.path.join(scan_dir, f"{scan_name}_edit_{edit_number}_aligned.nii.gz"),
                    os.path.join(scan_dir, f"{scan_name}_lossy_edit_{edit_number}_aligned.nii.gz"),
                ]
                if any(os.path.isfile(path) for path in aligned_paths):
                    already_aligned = True
                    break
                edit_number += 1

            if already_aligned:
                keep_for_linked_catchup = False
                if propagate_linked:
                    siblings, _ = get_same_shape_siblings(directory, scan_name)
                    for sibling_name in siblings or []:
                        sib_latest, _ = self._latest_any_edit_stem(directory, sibling_name)
                        if sib_latest < 0:
                            keep_for_linked_catchup = True
                            break
                        sib_files = glob.glob(
                            os.path.join(directory, "extracted", sibling_name, f"{sibling_name}_edit_{sib_latest}_*.nii.gz")
                        )
                        if not sib_files or 'aligned' not in os.path.basename(sib_files[0]):
                            keep_for_linked_catchup = True
                            break
                if keep_for_linked_catchup:
                    print(
                        f"Keeping already-aligned {scan_name} for linked-child catch-up "
                        f"(children still need the shared rigid transform)"
                    )
                    propagate_only_scans.add(scan_name)
                else:
                    print(f"Skipping {scan_name} as it's already aligned")
                    scans_to_remove.append(scan_name)

            # Remove from scan_names if it has elastic registration
            if has_elastic_registration(scan_name, directory):
                scans_to_remove.append(scan_name)

        
        # Remove scans that are already aligned
        for scan_name in scans_to_remove:
            if scan_name in scan_names:
                scan_names.remove(scan_name)
            else:
                print(f"Scan {scan_name} not found in scan_names")

        flagged_set = load_flagged_subject_names(directory)
        scan_names = apply_flag_filter(scan_names, flagged_set, flag_filter)
        scan_names = filter_out_linked_children(directory, scan_names)
        reference_metadata = self._load_subject_metadata(directory, reference)
        reference_is_preserved_mesh = self._is_preserved_mesh_metadata(reference_metadata)
        metadata_by_scan = {}
        preserved_moving_scans = []
        voxel_moving_scans = []
        for scan_name in list(scan_names):
            try:
                scan_metadata_for_filter = self._load_subject_metadata(directory, scan_name)
            except (FileNotFoundError, json.JSONDecodeError) as exc:
                print(f"Skipping {scan_name}: could not read metadata ({exc})")
                scan_names.remove(scan_name)
                continue
            metadata_by_scan[scan_name] = scan_metadata_for_filter
            if self._is_preserved_mesh_metadata(scan_metadata_for_filter):
                preserved_moving_scans.append(scan_name)
            else:
                voxel_moving_scans.append(scan_name)

        method_key = method.lower()
        if method_key in ("ants", "dino-reg", "gpu-rigid"):
            voxel_only_labels = {
                "ants": "ANTs",
                "dino-reg": "DINO-Reg",
                "gpu-rigid": "GPU Rigid",
            }
            voxel_only_label = voxel_only_labels[method_key]
            if reference_is_preserved_mesh:
                return Response(
                    {
                        "error": f"{voxel_only_label} rigid alignment only works with voxel-based volumes. Select a voxel-based reference or use ALPACA/manual guidepoints for preserved meshes."
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )
            if preserved_moving_scans:
                print(f"Skipping preserved mesh subjects for {voxel_only_label} alignment: {preserved_moving_scans}")
                scan_names = [scan_name for scan_name in scan_names if scan_name not in preserved_moving_scans]
            if not scan_names:
                return Response(
                    {"error": f"{voxel_only_label} rigid alignment only works with voxel-based volumes. No voxel-based moving scans are available."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            if method_key == "dino-reg":
                from .dinoRegRigid import DinoRegError, ensure_dino_reg_cuda, get_dino_model
                try:
                    ensure_dino_reg_cuda()
                    channel_layer = get_channel_layer()
                    if channel_layer is not None:
                        async_to_sync(channel_layer.group_send)(
                            'progress_group',
                            {
                                'type': 'send_progress',
                                'progress': 0,
                                'scan_name': scan_names[0] if scan_names else reference,
                                'custom_message': 'Loading DINOv3 model for DINO-Reg...',
                                'total': max(len(scan_names), 1),
                                'current': 0,
                            }
                        )
                    get_dino_model()
                except DinoRegError as exc:
                    return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
            if method_key == "gpu-rigid":
                try:
                    ensure_gpu_rigid_cuda()
                    channel_layer = get_channel_layer()
                    if channel_layer is not None:
                        async_to_sync(channel_layer.group_send)(
                            'progress_group',
                            {
                                'type': 'send_progress',
                                'progress': 0,
                                'scan_name': scan_names[0] if scan_names else reference,
                                'custom_message': 'Checking CUDA for GPU Rigid (FireANTs)...',
                                'total': max(len(scan_names), 1),
                                'current': 0,
                            }
                        )
                except GpuRigidError as exc:
                    return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        elif method_key in ("alpaca", "manual-guidepoints"):
            pass
        else:
            return Response({'error': f'Unknown alignment method: {method}'}, status=status.HTTP_400_BAD_REQUEST)

        if not scan_names:
            return Response({'error': 'No eligible moving scans are available for alignment.'}, status=status.HTTP_400_BAD_REQUEST)

        # Initialize WebSocket progress tracking
        total_scans = len(scan_names)
        channel_layer = get_channel_layer()
        if channel_layer is not None and total_scans > 0:
            print("Sending progress update")
            async_to_sync(channel_layer.group_send)(
                'progress_group',
                {
                    'type': 'send_progress',
                    'progress': 0,
                    'scan_name': scan_names[0],
                    'custom_message': f'Analyzing the general properties of the reference {reference}...',
                    'total': total_scans,
                    'current': 0,
                }
            )
        
        reference_base_path = self.get_reference_base_path(directory, reference)
        reference_edit_number, reference_edit_stem = self._latest_any_edit_stem(directory, reference)

        reference_data = None
        reference_affine_data = None
        reference_shape = None
        reference_voxel_size = float(reference_metadata.get('voxel_size', 1.0) or 1.0)
        reference_vertices = None
        reference_faces = None

        reference_vertices_local = None

        if reference_is_preserved_mesh:
            reference_vertices_local, reference_faces, _, reference_path = self._load_preserved_mesh_arrays(
                directory, reference, reference_metadata, reference_edit_stem
            )
            print(f"Loaded preserved PLY reference mesh: {reference_path}")
            mesh_ref_ctx = self._prepare_mesh_reference_alignment_context(
                directory,
                reference_metadata,
                reference_vertices_local,
                reference_faces,
            )
            reference_vertices = mesh_ref_ctx["reference_vertices"]
            reference_faces = mesh_ref_ctx["reference_faces"]
            reference_data = mesh_ref_ctx["reference_data"]
            reference_shape = mesh_ref_ctx["reference_shape"]
            reference_voxel_size = mesh_ref_ctx["reference_voxel_size"]
            reference_affine_data = mesh_ref_ctx["reference_affine_data"]
            reference_threshold = mesh_ref_ctx["reference_threshold"]
        else:
            # Find the latest NIfTI edit of the reference scan for voxel alignment.
            reference_nifti_edit_number = 0
            while glob.glob(os.path.join(reference_base_path, f"{reference}_edit_{reference_nifti_edit_number}_*.nii.gz")):
                reference_nifti_edit_number += 1
            reference_nifti_edit_number -= 1

            if reference_nifti_edit_number >= 0:
                reference_path = glob.glob(os.path.join(reference_base_path, f"{reference}_edit_{reference_nifti_edit_number}_*.nii.gz"))[0]
            else:
                reference_path = os.path.join(reference_base_path, f"{reference}.nii.gz")

            reference_threshold = reference_metadata['threshold']
            reference_img = nib.load(reference_path)
            reference_affine_data = reference_img.affine
            reference_original_dtype = reference_img.get_data_dtype()
            reference_data = reference_img.get_fdata().astype(reference_original_dtype)
            reference_data = gaussian_filter(reference_data, sigma=1.0)
            reference_shape = reference_data.shape
            reference_voxel_size = reference_metadata['voxel_size']

            if method.lower() == 'alpaca':
                # Get the reference mesh from the voxel volume via marching cubes.
                reference_vertices, reference_faces, _, _ = measure.marching_cubes(reference_data, level=reference_threshold, spacing=(reference_voxel_size, reference_voxel_size, reference_voxel_size))
                reference_faces = reference_faces[:, ::-1]

                # Decimate if the number of vertices is too high and outer_surface is False
                if len(reference_vertices) > 15000000 and not outer_surface:
                    decimation_factor = max(min(1, 1-(15000000/len(reference_vertices))), 0)
                    print("vertices before decimation: ", reference_vertices.shape)
                    reference_vertices, reference_faces = fast_simplification.simplify(reference_vertices, reference_faces, decimation_factor)
                    print("vertices after decimation: ", reference_vertices.shape)
                    cleanup_memory()
                else:
                    print("not simplifying reference mesh")

        if method.lower() == 'manual-guidepoints':
            # Load reference guidepoints (from the latest edit or original if no edits)
            if reference_edit_number >= 0:
                # Look for guidepoints from the latest edit
                reference_guidepoints_path = glob.glob(os.path.join(reference_base_path, f"{reference}_edit_{reference_edit_number}_*_guidepoints.json"))
            else:
                # No edits exist, look for original guidepoints
                reference_guidepoints_path = glob.glob(os.path.join(reference_base_path, f"{reference}_guidepoints.json"))
            
            if len(reference_guidepoints_path) > 0:
                reference_guidepoints_path = reference_guidepoints_path[0]
            else:
                print(f"No guidepoints found for reference scan {reference}")   
                return Response({"error": "No guidepoints found for reference scan"}, status=status.HTTP_400_BAD_REQUEST)                        
            
            # Load reference guidepoints
            with open(reference_guidepoints_path, 'r') as f:
                reference_guidepoints = np.array(json.load(f))

            if reference_is_preserved_mesh:
                reference_guidepoints = mesh_local_to_shared_mm(
                    reference_guidepoints,
                    reference_metadata.get("mesh_metadata"),
                    prism_vertices=reference_vertices_local,
                    directory=directory,
                    voxel_size=reference_voxel_size,
                )


        # Track linked-scan propagation results
        linked_propagation_results = []

        # -------------------------------------- Process each scan --------------------------------------
        for idx, scan_name in enumerate(scan_names):
            try:
                ants_initializer_record = None
                gpu_initializer_record = None
                dino_reg_result = None
                print(f"Processing scan {idx + 1}/{total_scans}: {scan_name}")

                # Find the latest edit for this scan. For voxel subjects the
                # existing NIfTI path below still uses latest NIfTI edits; for
                # preserved meshes this stem points to the latest PLY edit.
                latest_edit, latest_edit_stem = self._latest_any_edit_stem(directory, scan_name)


                # Send progress update 
                channel_layer = get_channel_layer()
                if channel_layer is not None:
                    progress = ((idx+(0/9)) / total_scans)
                    async_to_sync(channel_layer.group_send)(
                        'progress_group',
                        {
                            'type': 'send_progress',
                            'progress': progress,
                            'scan_name': scan_name,
                            'custom_message': f'Extracting properties of {scan_name}...',
                            'total': total_scans,
                            'current': idx + 1,
                        }
                    )
                    print("Extracting properties of ", scan_name)

                # Check available memory
                remaining_memory = psutil.virtual_memory().available / 1024 / 1024 / 1024
                print(f"Available memory: {remaining_memory:.2f} GB")

                # Load the scan's metadata
                scan_metadata = metadata_by_scan.get(scan_name) or self._load_subject_metadata(directory, scan_name)

                # Already-aligned link main kept only so unaligned children can inherit
                # the stored rigid transform — do not re-run ALPACA/ANTs on the main.
                if scan_name in propagate_only_scans:
                    print(f"Catch-up linked propagation for already-aligned {scan_name}")
                    if method.lower() != 'alpaca':
                        print(
                            f"  Linked catch-up currently supports ALPACA only "
                            f"(got {method}); skipping {scan_name}"
                        )
                        continue
                    if self._is_preserved_mesh_metadata(scan_metadata):
                        print(f"  Preserved-mesh catch-up not implemented for {scan_name}")
                        continue
                    try:
                        geom = self._reconstruct_alpaca_propagation_geometry(
                            directory, scan_name, scan_metadata, reference_shape
                        )
                        rotation = geom['rotation']
                        padding_bottom_val = geom['padding_bottom_val']
                        padding_top_val = geom['padding_top_val']
                        translation = geom['translation']
                        offset = geom['offset']
                        scale_match = geom['scale_match']
                        in_start_x, in_end_x = geom['in_start_x'], geom['in_end_x']
                        in_start_y, in_end_y = geom['in_start_y'], geom['in_end_y']
                        in_start_z, in_end_z = geom['in_start_z'], geom['in_end_z']
                        out_start_x, out_end_x = geom['out_start_x'], geom['out_end_x']
                        out_start_y, out_end_y = geom['out_start_y'], geom['out_end_y']
                        out_start_z, out_end_z = geom['out_start_z'], geom['out_end_z']
                    except Exception as catchup_err:
                        print(f"  Could not reconstruct ALPACA geometry for {scan_name}: {catchup_err}")
                        linked_propagation_results.append({
                            'scan_name': scan_name,
                            'status': 'failed',
                            'reason': f'catch-up geometry: {catchup_err}',
                        })
                        continue

                    if propagate_linked:
                        try:
                            siblings, sib_err = get_same_shape_siblings(directory, scan_name)
                            if sib_err:
                                print(f"Could not resolve siblings for {scan_name}: {sib_err}")
                            elif siblings:
                                print(f"Propagating stored alignment from {scan_name} to {len(siblings)} siblings: {siblings}")
                                for sibling_name in siblings:
                                    try:
                                        sib_latest_edit, _ = self._latest_any_edit_stem(directory, sibling_name)
                                        if sib_latest_edit >= 0:
                                            sib_edit_files = glob.glob(os.path.join(directory, "extracted", sibling_name, f"{sibling_name}_edit_{sib_latest_edit}_*.nii.gz"))
                                            if sib_edit_files and 'aligned' in os.path.basename(sib_edit_files[0]):
                                                linked_propagation_results.append({'scan_name': sibling_name, 'status': 'skipped', 'reason': 'already aligned'})
                                                continue
                                        if has_elastic_registration(sibling_name, directory):
                                            linked_propagation_results.append({'scan_name': sibling_name, 'status': 'skipped', 'reason': 'elastic registration present'})
                                            continue
                                        sib_metadata = self._load_subject_metadata(directory, sibling_name)
                                        if self._is_preserved_mesh_metadata(sib_metadata):
                                            linked_propagation_results.append({'scan_name': sibling_name, 'status': 'skipped', 'reason': 'preserved mesh'})
                                            continue
                                        if sib_latest_edit >= 0:
                                            sib_edit_files = glob.glob(os.path.join(directory, "extracted", sibling_name, f"{sibling_name}_edit_{sib_latest_edit}_*.nii.gz"))
                                            sib_nifti_path = sib_edit_files[0] if sib_edit_files else os.path.join(directory, "extracted", sibling_name, f"{sibling_name}.nii.gz")
                                        else:
                                            sib_nifti_path = os.path.join(directory, "extracted", sibling_name, f"{sibling_name}.nii.gz")
                                        sib_img = nib.load(sib_nifti_path)
                                        sib_dtype = sib_img.get_data_dtype()
                                        sib_data = sib_img.get_fdata().astype(sib_dtype)
                                        if scale_match != 1:
                                            sib_data = ndimage.zoom(sib_data, scale_match, order=interpolation_order, mode='constant', cval=0)
                                        sib_padded = np.pad(
                                            sib_data.astype(sib_dtype),
                                            ((padding_bottom_val[0], padding_top_val[0]),
                                             (padding_bottom_val[1], padding_top_val[1]),
                                             (padding_bottom_val[2], padding_top_val[2])),
                                            mode='constant',
                                            constant_values=0
                                        ).astype(sib_dtype)
                                        sib_rotated = np.zeros_like(sib_padded, dtype=sib_dtype)
                                        ndimage.affine_transform(
                                            sib_padded, rotation.T, offset=offset, output=sib_rotated,
                                            output_shape=sib_padded.shape, order=interpolation_order,
                                            mode='constant', prefilter=False, cval=0
                                        )
                                        sib_translated = np.zeros_like(sib_rotated, dtype=sib_dtype)
                                        ndimage.shift(
                                            sib_rotated, shift=translation, output=sib_translated,
                                            order=interpolation_order, mode='constant', cval=0
                                        )
                                        sib_transformed = np.zeros_like(reference_data, dtype=sib_dtype)
                                        sib_transformed[in_start_x:in_end_x, in_start_y:in_end_y, in_start_z:in_end_z] = (
                                            sib_translated[out_start_x:out_end_x, out_start_y:out_end_y, out_start_z:out_end_z]
                                        )
                                        sib_background_value = np.array(0, dtype=sib_transformed.dtype)
                                        if sib_transformed.shape[0] > 2 * border_width:
                                            sib_transformed[:border_width, :, :] = sib_background_value
                                            sib_transformed[-border_width:, :, :] = sib_background_value
                                        if sib_transformed.shape[1] > 2 * border_width:
                                            sib_transformed[:, :border_width, :] = sib_background_value
                                            sib_transformed[:, -border_width:, :] = sib_background_value
                                        if sib_transformed.shape[2] > 2 * border_width:
                                            sib_transformed[:, :, :border_width] = sib_background_value
                                            sib_transformed[:, :, -border_width:] = sib_background_value
                                        sib_new_edit = sib_latest_edit + 1
                                        sib_output_path = os.path.join(directory, "extracted", sibling_name, f"{sibling_name}_edit_{sib_new_edit}_aligned.nii.gz")
                                        sib_volume_to_save = self._volume_for_mesh_reference_save(sib_transformed, reference_is_preserved_mesh)
                                        nib.save(nib.Nifti1Image(sib_volume_to_save, reference_affine_data), sib_output_path)
                                        sib_metadata['alignment_to'] = reference
                                        sib_metadata['alignment_shift'] = 0
                                        sib_metadata['alignment_method'] = method
                                        sib_metadata['voxel_size'] = reference_voxel_size
                                        if 'alignment_calculated_parameters' in scan_metadata:
                                            sib_metadata['alignment_calculated_parameters'] = scan_metadata['alignment_calculated_parameters']
                                        if 'alignment_rigid_provenance' in scan_metadata:
                                            sib_metadata['alignment_rigid_provenance'] = scan_metadata['alignment_rigid_provenance']
                                        strip_ephemeral_scan_metadata(sib_metadata)
                                        sib_json_path = os.path.join(directory, "extracted", sibling_name, f"{sibling_name}.json")
                                        with open(sib_json_path, 'w') as jf:
                                            json.dump(sib_metadata, jf, indent=4)
                                        if 'lossy_compression' in sib_metadata:
                                            self.registration_tools.save_as_lossy_nifti(
                                                image_data=sib_volume_to_save,
                                                voxel_size=reference_affine_data[0, 0],
                                                json_file=sib_json_path,
                                                output_file=sib_output_path.replace("_edit", "_lossy_edit"),
                                            )
                                        linked_propagation_results.append({
                                            'scan_name': sibling_name,
                                            'status': 'success',
                                            'edit': sib_new_edit,
                                        })
                                        print(f"  Catch-up propagated alignment to {sibling_name} (edit {sib_new_edit})")
                                        cleanup_memory()
                                    except Exception as sib_e:
                                        linked_propagation_results.append({
                                            'scan_name': sibling_name,
                                            'status': 'failed',
                                            'reason': str(sib_e),
                                        })
                                        print(f"  Error in catch-up propagation to {sibling_name}: {sib_e}")
                        except Exception as prop_e:
                            print(f"Error in linked catch-up for {scan_name}: {prop_e}")
                    continue

                # --------------------------------------------------------------
                # Preserved PLY moving subject path
                # --------------------------------------------------------------
                # Mesh-only subjects have no voxel data to resample. For ALPACA
                # and manual guidepoints we align their PLY vertices directly and
                # save a new PLY edit, then skip all NIfTI-specific code below.
                if self._is_preserved_mesh_metadata(scan_metadata):
                    if method.lower() not in ("alpaca", "manual-guidepoints"):
                        print(f"Skipping preserved mesh subject {scan_name} for voxel-only method {method}")
                        continue

                    moving_vertices, moving_faces, _, moving_mesh_path = self._load_preserved_mesh_arrays(
                        directory, scan_name, scan_metadata, latest_edit_stem
                    )

                    ref_prism_corner = None
                    if reference_is_preserved_mesh:
                        ref_prism_corner = mesh_reference_prism_corner_mm(
                            reference_vertices_local,
                            reference_metadata.get("mesh_metadata"),
                            directory=directory,
                            voxel_size=reference_voxel_size,
                        )
                        moving_vertices_for_align = mesh_centroid_local_to_reference_prism_mm(
                            moving_vertices,
                            scan_metadata.get("mesh_metadata"),
                            mesh_vertices_for_prism=moving_vertices,
                            reference_prism_corner_mm=ref_prism_corner,
                        )
                    else:
                        moving_vertices_for_align = moving_vertices

                    guidepoints_origin = None

                    if method.lower() == "alpaca":
                        if reference_vertices is None or reference_faces is None:
                            raise ValueError("ALPACA preserved-mesh alignment requires a reference mesh.")
                        transformation_data = alpaca.align_landmarks_to_mesh(
                            reference_vertices,
                            reference_faces,
                            moving_vertices_for_align,
                            moving_faces,
                            os.path.join(directory, "extracted", scan_name, f"{scan_name}.json"),
                            scaling_mode=scaling,
                            outer_surface=outer_surface,
                        )
                        # align_landmarks_to_mesh's 'transformation_matrix' field intentionally
                        # omits the source/target centroid terms (voxel-volume callers add those
                        # back themselves via explicit centroid-snapping when embedding into a
                        # reference canvas). Preserved-mesh point clouds have no such separate
                        # embedding step, so compose the full raw-to-raw transform here directly
                        # from the same ingredients that pathway already returns.
                        transformation_matrix = alpaca.compose_source_to_target_transform(
                            np.linalg.inv(transformation_data["chosen_transform"]),
                            transformation_data["target_centroid"],
                            transformation_data["source_centroid"],
                            transformation_data["normalization_scale_factor"],
                        )
                        transformed_vertices = alpaca.apply_rigid_transform(moving_vertices_for_align, transformation_matrix)
                        _, source_guidepoints = self._load_guidepoints_for_stem(directory, scan_name, latest_edit_stem)
                        transformed_guidepoints = (
                            alpaca.apply_rigid_transform(source_guidepoints, transformation_matrix)
                            if source_guidepoints is not None else None
                        )
                        # Manual homologous guidepoints only (project-wide). Per-pair auto
                        # correspondences are not stored — they are not homologous across scans.
                        if transformed_guidepoints is not None:
                            guidepoints_origin = "manual"
                        transform_summary = {
                            "matrix": transformation_matrix.tolist(),
                            "scale": float(transformation_data.get("scale", 1.0)),
                        }
                    else:
                        _, subject_guidepoints = self._load_guidepoints_for_stem(directory, scan_name, latest_edit_stem)
                        if subject_guidepoints is None:
                            print(f"No guidepoints found for preserved mesh subject {scan_name}, skipping")
                            continue
                        if reference_is_preserved_mesh:
                            subject_guidepoints = mesh_centroid_local_to_reference_prism_mm(
                                subject_guidepoints,
                                scan_metadata.get("mesh_metadata"),
                                mesh_vertices_for_prism=moving_vertices,
                                reference_prism_corner_mm=ref_prism_corner,
                                directory=directory,
                                voxel_size=reference_voxel_size,
                            )
                        if reference_guidepoints.shape != subject_guidepoints.shape:
                            print(f"Guidepoint arrays for reference and subject {scan_name} have different shapes")
                            continue

                        scale_match = 1.0
                        working_vertices = moving_vertices_for_align
                        working_guidepoints = subject_guidepoints
                        if scaling:
                            reference_gp_centroid = np.mean(reference_guidepoints, axis=0)
                            subject_gp_centroid = np.mean(subject_guidepoints, axis=0)
                            reference_mean_distance = np.mean(np.linalg.norm(reference_guidepoints - reference_gp_centroid, axis=1))
                            subject_mean_distance = np.mean(np.linalg.norm(subject_guidepoints - subject_gp_centroid, axis=1))
                            scale_match = float(reference_mean_distance / subject_mean_distance) if subject_mean_distance > 0 else 1.0
                            working_vertices = moving_vertices_for_align * scale_match
                            working_guidepoints = subject_guidepoints * scale_match

                        subject_centroid = np.mean(working_guidepoints, axis=0)
                        reference_centroid = np.mean(reference_guidepoints, axis=0)
                        rotation, _, _ = kabsch_rotation_matrix(
                            working_guidepoints,
                            reference_guidepoints,
                            source_centroid=subject_centroid,
                            target_centroid=reference_centroid,
                        )
                        # Preserved meshes are transformed as raw point clouds.
                        # Do not use the voxel-image affine convention here:
                        # volume warp uses affine_transform(matrix=R) for the inverse
                        # of row-vector (v-c)@R; raw PLY vertices and guidepoints apply
                        # that same R directly on row vectors.
                        transformed_vertices = (working_vertices - subject_centroid) @ rotation + reference_centroid
                        transformed_guidepoints = (working_guidepoints - subject_centroid) @ rotation + reference_centroid
                        guidepoints_origin = "manual"
                        transform_summary = {
                            "rotation": rotation.tolist(),
                            "source_centroid": subject_centroid.tolist(),
                            "target_centroid": reference_centroid.tolist(),
                            "scale": scale_match,
                        }

                    # Vertices are stored in the reference shared corner-origin mm frame.
                    output_vertices = transformed_vertices
                    output_faces = moving_faces
                    output_guidepoints = transformed_guidepoints
                    if not reference_is_preserved_mesh:
                        # Voxel-derived meshes (NPZ/GLB) are always generated from a
                        # volume with axes 0 and 2 swapped (see MarchingCubesView and
                        # _load_voxel_subject_display_mesh) to match the BabylonJS
                        # display convention. The ALPACA/guidepoint reference mesh used
                        # above for alignment is built from the raw (unswapped) NIfTI
                        # data, so the aligned output must be swapped back into that
                        # same convention to land in the voxel reference's real shared
                        # coordinate space. This is a deterministic geometric transform,
                        # not a stored flag: it is re-derived every time purely from
                        # whether the reference is voxel-based.
                        output_vertices = self._swap_xz_for_voxel_display_basis(transformed_vertices)
                        output_faces = np.asarray(moving_faces, dtype=np.int64)[:, ::-1]
                        output_guidepoints = (
                            self._swap_xz_for_voxel_display_basis(transformed_guidepoints)
                            if transformed_guidepoints is not None else None
                        )

                    alignment_rigid_provenance = build_alignment_rigid_provenance(
                        method, scaling, outer_surface, interpolation, ants_rigid_opts, ants_initializer_record
                    )

                    new_edit_number, output_path = self._save_preserved_aligned_mesh(
                        directory,
                        scan_name,
                        scan_metadata,
                        output_vertices,
                        output_faces,
                        reference,
                        method,
                        moving_mesh_path,
                        transform_summary,
                        transformed_guidepoints=output_guidepoints,
                        guidepoints_origin=guidepoints_origin,
                        alignment_rigid_provenance=alignment_rigid_provenance,
                    )
                    print(f"Saved preserved mesh aligned edit for {scan_name}: {output_path}")

                    # Linked children must inherit the same rigid transform before we
                    # `continue` past the voxel sibling-propagation block below.
                    if propagate_linked:
                        try:
                            link_group = get_link_group(directory, scan_name)
                            mesh_siblings = [
                                name for name in (link_group or {}).get("children", [])
                                if name != scan_name
                            ]
                            if mesh_siblings:
                                print(
                                    f"Propagating preserved-mesh alignment from {scan_name} "
                                    f"to {len(mesh_siblings)} linked members: {mesh_siblings}"
                                )
                            for sibling_name in mesh_siblings:
                                try:
                                    sib_metadata = self._load_subject_metadata(directory, sibling_name)
                                    if not self._is_preserved_mesh_metadata(sib_metadata):
                                        linked_propagation_results.append({
                                            'scan_name': sibling_name,
                                            'status': 'skipped',
                                            'reason': 'voxel sibling of mesh main (no shared NIfTI canvas)',
                                        })
                                        continue
                                    sib_latest_edit, sib_latest_stem = self._latest_any_edit_stem(
                                        directory, sibling_name
                                    )
                                    if sib_latest_edit >= 0 and sib_latest_stem and 'aligned' in sib_latest_stem:
                                        linked_propagation_results.append({
                                            'scan_name': sibling_name,
                                            'status': 'skipped',
                                            'reason': 'already aligned',
                                        })
                                        continue
                                    if has_elastic_registration(sibling_name, directory):
                                        linked_propagation_results.append({
                                            'scan_name': sibling_name,
                                            'status': 'skipped',
                                            'reason': 'elastic registration present',
                                        })
                                        continue

                                    sib_vertices, sib_faces, _, sib_mesh_path = self._load_preserved_mesh_arrays(
                                        directory, sibling_name, sib_metadata, sib_latest_stem
                                    )
                                    if reference_is_preserved_mesh:
                                        sib_vertices_for_align = mesh_centroid_local_to_reference_prism_mm(
                                            sib_vertices,
                                            sib_metadata.get("mesh_metadata"),
                                            mesh_vertices_for_prism=sib_vertices,
                                            reference_prism_corner_mm=ref_prism_corner,
                                            directory=directory,
                                            voxel_size=reference_voxel_size,
                                        )
                                    else:
                                        sib_vertices_for_align = sib_vertices

                                    if method.lower() == "alpaca":
                                        sib_transformed = alpaca.apply_rigid_transform(
                                            sib_vertices_for_align, transformation_matrix
                                        )
                                        _, sib_gps = self._load_guidepoints_for_stem(
                                            directory, sibling_name, sib_latest_stem
                                        )
                                        sib_transformed_gps = (
                                            alpaca.apply_rigid_transform(sib_gps, transformation_matrix)
                                            if sib_gps is not None else None
                                        )
                                    else:
                                        working_sib = sib_vertices_for_align * scale_match
                                        sib_transformed = (
                                            (working_sib - subject_centroid) @ rotation + reference_centroid
                                        )
                                        _, sib_gps = self._load_guidepoints_for_stem(
                                            directory, sibling_name, sib_latest_stem
                                        )
                                        if sib_gps is not None:
                                            if reference_is_preserved_mesh:
                                                sib_gps = mesh_centroid_local_to_reference_prism_mm(
                                                    sib_gps,
                                                    sib_metadata.get("mesh_metadata"),
                                                    mesh_vertices_for_prism=sib_vertices,
                                                    reference_prism_corner_mm=ref_prism_corner,
                                                    directory=directory,
                                                    voxel_size=reference_voxel_size,
                                                )
                                            working_sib_gps = sib_gps * scale_match
                                            sib_transformed_gps = (
                                                (working_sib_gps - subject_centroid) @ rotation
                                                + reference_centroid
                                            )
                                        else:
                                            sib_transformed_gps = None

                                    sib_output_vertices = sib_transformed
                                    sib_output_faces = sib_faces
                                    sib_output_gps = sib_transformed_gps
                                    if not reference_is_preserved_mesh:
                                        sib_output_vertices = self._swap_xz_for_voxel_display_basis(sib_transformed)
                                        sib_output_faces = np.asarray(sib_faces, dtype=np.int64)[:, ::-1]
                                        sib_output_gps = (
                                            self._swap_xz_for_voxel_display_basis(sib_transformed_gps)
                                            if sib_transformed_gps is not None else None
                                        )

                                    sib_edit, sib_path = self._save_preserved_aligned_mesh(
                                        directory,
                                        sibling_name,
                                        sib_metadata,
                                        sib_output_vertices,
                                        sib_output_faces,
                                        reference,
                                        method,
                                        sib_mesh_path,
                                        transform_summary,
                                        transformed_guidepoints=sib_output_gps,
                                        guidepoints_origin=guidepoints_origin,
                                        alignment_rigid_provenance=alignment_rigid_provenance,
                                    )
                                    linked_propagation_results.append({
                                        'scan_name': sibling_name,
                                        'status': 'success',
                                        'edit': sib_edit,
                                    })
                                    print(f"  Propagated preserved-mesh alignment to {sibling_name}: {sib_path}")
                                except Exception as sib_e:
                                    linked_propagation_results.append({
                                        'scan_name': sibling_name,
                                        'status': 'failed',
                                        'reason': str(sib_e),
                                    })
                                    print(f"  Error propagating preserved-mesh alignment to {sibling_name}: {sib_e}")
                        except Exception as prop_e:
                            print(f"Error in preserved-mesh linked propagation for {scan_name}: {prop_e}")

                    continue

                # Load voxel scan data for the existing NIfTI alignment path.
                latest_nifti_edit = 0
                while glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{latest_nifti_edit}_*.nii.gz")):
                    latest_nifti_edit += 1
                latest_nifti_edit -= 1
                latest_edit = latest_nifti_edit
                if latest_nifti_edit >= 0:
                    scan_path = glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{latest_nifti_edit}_*.nii.gz"))[0]
                else:
                    scan_path = os.path.join(directory, "extracted", scan_name, f"{scan_name}.nii.gz")

                # Load the scan data
                nifti_img = nib.load(scan_path)
                nifti_affine_data = nifti_img.affine
                original_dtype = nifti_img.get_data_dtype()
                scan_data = nifti_img.get_fdata().astype(original_dtype)

                # Get threshold from scan metadata
                threshold = scan_metadata['threshold']   

                # Get background value as we have done it before using the histogram of the Gaussian smoothed nifti data
                background_value = self.get_background_value(scan_data, threshold)
                print(f"Background value obtained: {background_value}")

                # Get voxel size from metadata and save original for landmark transformation
                voxel_size = scan_metadata['voxel_size']
                original_voxel_size = voxel_size

                # Resample scan data to match reference voxel size
                if round(voxel_size, 4) != round(reference_voxel_size, 4):
                    zoom_factor = voxel_size / reference_voxel_size
                    print(f"voxels size of subject {scan_name} is {round(voxel_size, 4)} vs reference {round(reference_voxel_size, 4)}")
                    print("Resampling scan data by a factor of ", zoom_factor)
                    prog_message = f'Resampling {scan_name} by a factor of {zoom_factor}...'
                else:
                    zoom_factor = 1
                    prog_message = f'No resampling needed for {scan_name}...'

                # Send progress update 
                channel_layer = get_channel_layer()
                if channel_layer is not None:
                    progress = ((idx+(1/9)) / total_scans)
                    async_to_sync(channel_layer.group_send)(
                        'progress_group',
                        {
                            'type': 'send_progress',
                            'progress': progress,
                            'scan_name': scan_name,
                            'custom_message': prog_message,
                            'total': total_scans,
                            'current': idx + 1,
                        }
                )

                if zoom_factor != 1:
                    scan_data = zoom(scan_data, zoom_factor, order=1, mode='constant', cval=background_value) # linear interpolation
                    voxel_size = reference_voxel_size
                    print("Resampling done, the data type is ", scan_data.dtype)

                nifti_img = None
                cleanup_memory()                            

                # Now branch based on the alignment method
                padded_scan_centroid = None

                # -------------------------------------------------------------------------------------------------------
                # -------------------------------------------------- ALPACA Method --------------------------------------
                # -------------------------------------------------------------------------------------------------------

                if method.lower() == 'alpaca':
                    # === ALPACA Method (existing code) ===

                    # Send progress update 
                    channel_layer = get_channel_layer()
                    if channel_layer is not None:
                        progress = ((idx+(1/9)) / total_scans)
                        async_to_sync(channel_layer.group_send)(
                            'progress_group',
                            {
                                'type': 'send_progress',
                                'progress': progress,
                                'scan_name': scan_name,
                                'custom_message': f'Generating 3D mesh and landmarking {scan_name}...',
                                'total': total_scans,
                                'current': idx + 1,
                            }
                        )
                        print("Generating 3D mesh and landmarking ", scan_name)
                    
                    # Check available memory
                    remaining_memory = psutil.virtual_memory().available / 1024 / 1024 / 1024
                    print(f"Available memory: {remaining_memory:.2f} GB")

                    # Print signal to noise ratio
                    print("Signal to noise ratio: ", np.mean(scan_data) / np.std(scan_data))

                    # Marching cubes to get the surface mesh
                    volume = gaussian_filter(scan_data, sigma=1.0)         

                    # Print signal to noise ratio after gaussian filtering
                    print("Signal to noise ratio after gaussian filtering: ", np.mean(volume) / np.std(volume))

                    # Apply the Marching Cubes algorithm to extract the mesh's vertices and faces from the volume data
                    vertices, faces, _, _ = measure.marching_cubes(volume, level=threshold, spacing=(reference_voxel_size, reference_voxel_size, reference_voxel_size))

                    # Invert the faces
                    faces = faces[:, ::-1]

                    # Remove unused variables and collect garbage
                    volume = None                    
                    cleanup_memory()

                    # Send progress update 
                    channel_layer = get_channel_layer()
                    if channel_layer is not None:
                        progress = ((idx+(2/9)) / total_scans)
                        async_to_sync(channel_layer.group_send)(
                            'progress_group',
                            {
                                'type': 'send_progress',
                                'progress': progress,
                                'scan_name': scan_name,
                                'custom_message': f'Calculating rough mesh alignment from {scan_name} to the reference scan...',
                                'total': total_scans,
                                'current': idx + 1,
                            }
                        )
                        print("Calculating rough mesh alignment from ", scan_name, " to the reference scan")
                    # Check available memory
                    remaining_memory = psutil.virtual_memory().available / 1024 / 1024 / 1024
                    print(f"Available memory: {remaining_memory:.2f} GB")

                    # Decimate if the number of vertices is too high and outer_surface is False
                    if len(vertices) > 15000000 and not outer_surface:
                        decimation_factor = max(min(1, 1-(15000000/len(vertices))), 0)
                        print("vertices before decimation: ", vertices.shape)
                        # Simplify the mesh by decimation
                        vertices, faces = fast_simplification.simplify(vertices, faces, decimation_factor)
                        print("vertices after decimation: ", vertices.shape)
                        cleanup_memory()
                    else:
                        print("not simplifying mesh")
                    
                    # Get the transformation matrix using ALPACA
                    transformation_data = alpaca.align_landmarks_to_mesh(reference_vertices, reference_faces, vertices, faces, os.path.join(directory, "extracted", scan_name, f"{scan_name}.json"), scaling_mode=scaling, outer_surface=outer_surface)
                    with open(os.path.join(directory, "extracted", scan_name, f"{scan_name}.json"), 'r') as jf:
                        scan_metadata = json.load(jf)
                    transformation_matrix = transformation_data['transformation_matrix']
                    target_centroid = transformation_data['target_centroid']
                    source_centroid = transformation_data['source_centroid']    
                    scale_match = transformation_data['scale']                

                    # Convert centroids from physical space to voxel space
                    target_centroid = target_centroid / voxel_size
                    source_centroid = source_centroid / voxel_size

                    # Scale scan data to match the scaling calculated by ALPACA (True size of current subject will be lost)
                    if scale_match != 1:
                        print("Scaling the scan data by a factor of ", scale_match)
                        scan_data = ndimage.zoom(scan_data, scale_match, order=1, mode='constant', cval=background_value)
                    
                    # Get rotation and translation
                    rotation = transformation_matrix[:3, :3]
                    # Convert translation from physical space to voxel space
                    translation = transformation_matrix[:3, 3] / voxel_size

                    # Remove unused variables and collect garbage
                    transformation_matrix = None
                    transformation_data = None
                    vertices = None
                    faces = None
                    cleanup_memory()

                    # Send progress update for applying alignment
                    channel_layer = get_channel_layer()
                    if channel_layer is not None:
                        progress = ((idx+(3/9)) / total_scans)
                        async_to_sync(channel_layer.group_send)(
                            'progress_group',
                            {
                                'type': 'send_progress',
                                'progress': progress,
                                'scan_name': scan_name,
                                'custom_message': f'Applying the calculated rough alignment to the voxel data from {scan_name}...',
                                'total': total_scans,
                                'current': idx + 1,
                            }
                        )
                        print("Applying the calculated rough alignment to the voxel data from ", scan_name)
                    
                    # Pad before rotation so corners are not clipped (same corner-expansion
                    # strategy as manual guidepoints).
                    padding_bottom_val, padding_top_val = self._padding_for_rigid_volume_warp(
                        scan_data.shape,
                        target_centroid,
                        rotation,
                    )
                    print(
                        f"ALPACA warp padding low={padding_bottom_val.tolist()} "
                        f"high={padding_top_val.tolist()}"
                    )

                    # Pad the scan data
                    padded_scan_data = np.pad(
                        scan_data.astype(original_dtype), 
                        ((padding_bottom_val[0], padding_top_val[0]), 
                         (padding_bottom_val[1], padding_top_val[1]), 
                         (padding_bottom_val[2], padding_top_val[2])),
                        mode='constant',
                        constant_values=background_value
                    ).astype(original_dtype)

                    scan_data = None
                    cleanup_memory()
                    
                    # Update the centroid coordinates and apply rotation
                    padded_target_centroid = target_centroid + padding_bottom_val
                    offset = padded_target_centroid - np.dot(rotation.T, padded_target_centroid)

                    output_array_rotated = np.zeros_like(padded_scan_data, dtype=original_dtype)
                    
                    # Apply rotation
                    ndimage.affine_transform(
                        padded_scan_data,
                        rotation.T,
                        offset=offset,
                        output=output_array_rotated,
                        output_shape=padded_scan_data.shape,
                        order=interpolation_order,
                        mode='constant',
                        prefilter=False,
                        cval=background_value
                    )

                    padded_scan_data = None
                    cleanup_memory()
                    output_array_translated = np.zeros_like(output_array_rotated, dtype=original_dtype)

                    # Apply translation
                    ndimage.shift(
                        output_array_rotated,
                        shift=translation,
                        output=output_array_translated,
                        order=0,
                        mode='constant',
                        cval=background_value
                    )
                                        

                    # --- Create a canvas to store the transformed data ---
                    transformed_data_canvas = np.full_like(reference_data, background_value)

                    # Reassign variables
                    reference_centroid_voxel = source_centroid
                    transformed_data_centroid = padded_target_centroid
                    transformed_data = output_array_translated

                    # Calculate crop indices
                    in_start_x = max(0, int(reference_centroid_voxel[0]) - int(transformed_data_centroid[0]))
                    in_start_y = max(0, int(reference_centroid_voxel[1]) - int(transformed_data_centroid[1]))
                    in_start_z = max(0, int(reference_centroid_voxel[2]) - int(transformed_data_centroid[2]))

                    in_end_x = min(reference_shape[0], int(reference_centroid_voxel[0]) - int(transformed_data_centroid[0]) + transformed_data.shape[0])
                    in_end_y = min(reference_shape[1], int(reference_centroid_voxel[1]) - int(transformed_data_centroid[1]) + transformed_data.shape[1])
                    in_end_z = min(reference_shape[2], int(reference_centroid_voxel[2]) - int(transformed_data_centroid[2]) + transformed_data.shape[2])        

                    out_start_x = max(0, int(transformed_data_centroid[0]) - int(reference_centroid_voxel[0]))
                    out_start_y = max(0, int(transformed_data_centroid[1]) - int(reference_centroid_voxel[1]))
                    out_start_z = max(0, int(transformed_data_centroid[2]) - int(reference_centroid_voxel[2]))

                    out_end_x = min(transformed_data.shape[0], int(transformed_data_centroid[0]) - int(reference_centroid_voxel[0]) + reference_shape[0])
                    out_end_y = min(transformed_data.shape[1], int(transformed_data_centroid[1]) - int(reference_centroid_voxel[1]) + reference_shape[1])
                    out_end_z = min(transformed_data.shape[2], int(transformed_data_centroid[2]) - int(reference_centroid_voxel[2]) + reference_shape[2])


                    # Verify that this will be a match and not cause an index error
                    if in_end_x - in_start_x != out_end_x - out_start_x or in_end_y - in_start_y != out_end_y - out_start_y or in_end_z - in_start_z != out_end_z - out_start_z:
                        print(f"Index error will occur, skipping {scan_name}")
                        print("x in width: ", in_end_x - in_start_x, "out width: ", out_end_x - out_start_x)
                        print("y in width: ", in_end_y - in_start_y, "out width: ", out_end_y - out_start_y)
                        print("z in width: ", in_end_z - in_start_z, "out width: ", out_end_z - out_start_z)
                        continue

                    # Copy the valid portion of the data
                    transformed_data_canvas[
                        in_start_x:in_end_x,                     
                        in_start_y:in_end_y, 
                        in_start_z:in_end_z] = transformed_data[
                                                    out_start_x:out_end_x,
                                                    out_start_y:out_end_y,
                                                    out_start_z:out_end_z
                                                ]  

                    transformed_data = transformed_data_canvas

                    # Create an x-voxel gap around the data by setting border voxels to background_value.
                    # This helps prevent edge artifacts in subsequent elastic registration steps.
                    
                    if transformed_data.shape[0] > 2 * border_width:
                        transformed_data[:border_width, :, :] = background_value
                        transformed_data[-border_width:, :, :] = background_value
                    if transformed_data.shape[1] > 2 * border_width:
                        transformed_data[:, :border_width, :] = background_value
                        transformed_data[:, -border_width:, :] = background_value
                    if transformed_data.shape[2] > 2 * border_width:
                        transformed_data[:, :, :border_width] = background_value
                        transformed_data[:, :, -border_width:] = background_value

                    # Print shapes to assert that the data is correctly aligned
                    print("reference_data shape: ", reference_data.shape)
                    print("transformed_data shape: ", transformed_data.shape)

                    # Clean up memory
                    output_array_translated = None
                    output_array_rotated = None
                    cleanup_memory()

                    # Store alignment embedding for landmark transformation
                    self.store_alignment_embedding('alpaca', {
                        'padding_bottom_val': padding_bottom_val,
                        'padding_top_val': padding_top_val,
                        'rotation': rotation,
                        'offset': offset,
                        'translation': translation,
                        'source_centroid': source_centroid,
                        'target_centroid': target_centroid,
                        'padded_target_centroid': padded_target_centroid,
                        'crop_indices': {
                            'x': (in_start_x, in_end_x),
                            'y': (in_start_y, in_end_y),
                            'z': (in_start_z, in_end_z),
                        },
                        'reference_shape': reference_shape,
                        'reference_voxel_size': reference_voxel_size,
                        'original_voxel_size': original_voxel_size,
                        'scale_match': scale_match,
                    })

                    # Match variable name used by shared metadata so alignment_rotation_center_voxel is saved correctly
                    padded_scan_centroid = padded_target_centroid

                # -------------------------------------------------------------------------------------------------------
                # -------------------------- ANTs / GPU Rigid Method (voxel intensity rigid) ----------------------------
                # -------------------------------------------------------------------------------------------------------

                elif method.lower() in ('ants', 'gpu-rigid'):
                    gpu_rigid_R = None
                    gpu_rigid_t = None
                    # === ANTs Method (new code based on commented section) ===
                    
                    # Send progress update for ANTs alignment
                    channel_layer = get_channel_layer()
                    if channel_layer is not None:
                        progress = ((idx+(3/9)) / total_scans)
                        async_to_sync(channel_layer.group_send)(
                            'progress_group',
                            {
                                'type': 'send_progress',
                                'progress': progress,
                                'scan_name': scan_name,
                                'custom_message': f'Preparing ANTs-based alignment for {scan_name}...',
                                'total': total_scans,
                                'current': idx + 1,
                            }
                        )     

                    # === Scale matching ===
                    if scaling:
                        reference_vertices, _, _, _ = measure.marching_cubes(reference_data[::2, ::2, ::2], level=reference_threshold, spacing=(reference_voxel_size*2, reference_voxel_size*2, reference_voxel_size*2))
                        scan_vertices, _, _, _ = measure.marching_cubes(scan_data[::2, ::2, ::2], level=threshold, spacing=(reference_voxel_size*2, reference_voxel_size*2, reference_voxel_size*2))
                        scale_match = np.mean(np.linalg.norm(reference_vertices, axis=1)) / np.mean(np.linalg.norm(scan_vertices, axis=1))
                        print("Scale match: ", scale_match)

                        scan_data = ndimage.zoom(scan_data, scale_match, order=1, mode='constant', cval=background_value)

                        reference_vertices = None
                        scan_vertices = None
                        cleanup_memory()

                    # === Union canvas (content-mask centroids, paste not shift) ===
                    _ants_rigid_log(scan_name, "union canvas layout starting")

                    reference_content_mask = _content_mask_from_threshold(
                        reference_data, reference_threshold, sigma=1.0
                    )
                    scan_content_mask = _content_mask_from_threshold(
                        scan_data, threshold, sigma=1.0
                    )

                    reference_centroid = _centroid_of_mask(reference_content_mask)
                    scan_centroid = _centroid_of_mask(scan_content_mask)
                    ref_content_low, ref_content_high = _mask_bbox(reference_content_mask)
                    mov_content_low, mov_content_high = _mask_bbox(scan_content_mask)

                    print("Reference centroid (content mask):", reference_centroid)
                    print("Scan centroid (content mask):", scan_centroid)

                    canvas_margin = int(max(32, round(max(reference_shape) * 0.06)))
                    ref_extent = np.asarray(ref_content_high - ref_content_low, dtype=np.float64)
                    mov_extent = np.asarray(mov_content_high - mov_content_low, dtype=np.float64)
                    rotation_margin = int(
                        max(64, round(max(np.linalg.norm(ref_extent), np.linalg.norm(mov_extent)) * 0.15))
                    )
                    # Shrink oversized moving bbox (e.g. scanner disk artifact) vs reference.
                    # Paired .nii.mask.gz files are sub-anatomical segmentations, not head geometry.
                    mov_cap = ref_extent * 1.35 + 2.0 * rotation_margin
                    raw_mov_extent = mov_content_high - mov_content_low
                    mov_content_low, mov_content_high = _cap_mask_bbox_to_extent(
                        mov_content_low,
                        mov_content_high,
                        scan_centroid,
                        mov_cap,
                        scan_data.shape,
                    )
                    mov_extent = np.asarray(mov_content_high - mov_content_low, dtype=np.float64)
                    if np.any((mov_content_high - mov_content_low) < raw_mov_extent):
                        _ants_rigid_log(
                            scan_name,
                            f"capped moving content bbox to ref-scaled extent "
                            f"(max {mov_cap.astype(int).tolist()} vox)",
                        )
                    max_est_voxels = int(
                        (ants_rigid_opts if method.lower() == 'ants' else gpu_rigid_opts).get(
                            "max_estimation_voxels", 40_000_000
                        )
                    )
                    union_layout = build_union_registration_canvas(
                        reference_centroid,
                        scan_centroid,
                        ref_content_low,
                        ref_content_high,
                        mov_content_low,
                        mov_content_high,
                        canvas_margin,
                        max_estimation_voxels=max_est_voxels,
                        rotation_margin_voxels=rotation_margin,
                    )
                    canvas_shape = union_layout["canvas_shape"]
                    ref_paste = union_layout["ref_paste"]
                    mov_paste = union_layout["mov_paste"]
                    centroid_delta = union_layout["centroid_delta"]
                    estimation_stride = int(union_layout["estimation_stride"])
                    estimation_shape = union_layout["estimation_shape"]

                    reference_on_canvas = paste_volume_on_canvas_clipped(
                        reference_data,
                        ref_paste,
                        canvas_shape,
                        background_value,
                    )
                    moving_on_canvas = paste_volume_on_canvas_clipped(
                        scan_data,
                        mov_paste,
                        canvas_shape,
                        background_value,
                    )

                    use_reg_mask = bool(
                        (ants_rigid_opts if method.lower() == 'ants' else gpu_rigid_opts).get(
                            "use_registration_mask", True
                        )
                    )
                    reference_mask_on_canvas = None
                    moving_mask_on_canvas = None
                    if use_reg_mask:
                        reference_mask_on_canvas = paste_volume_on_canvas_clipped(
                            reference_content_mask.astype(np.uint8),
                            ref_paste,
                            canvas_shape,
                            0,
                        ).astype(bool)
                        moving_mask_on_canvas = paste_volume_on_canvas_clipped(
                            scan_content_mask.astype(np.uint8),
                            mov_paste,
                            canvas_shape,
                            0,
                        ).astype(bool)

                    _ants_rigid_log(
                        scan_name,
                        f"estimation canvas: shape {canvas_shape} stride={estimation_stride} "
                        f"est_grid={estimation_shape} rotation_margin={rotation_margin} "
                        f"(ref paste {ref_paste.tolist()}, mov paste {mov_paste.tolist()}, "
                        f"delta {centroid_delta.tolist()})",
                    )
                    if estimation_stride >= 4:
                        _ants_rigid_log(
                            scan_name,
                            f"estimation stride {estimation_stride} forced by canvas size "
                            f"({int(np.prod(estimation_shape)):,} voxels); tighten canvas "
                            f"(crop artifact) or raise max_estimation_voxels for stride 2–3",
                        )

                    moving_image_range = [float(np.min(scan_data)), float(np.max(scan_data))]
                    
                    method_label = "GPU Rigid" if method.lower() == "gpu-rigid" else "ANTs"
                    if channel_layer is not None:
                        progress = ((idx+(4/9)) / total_scans)
                        async_to_sync(channel_layer.group_send)(
                            'progress_group',
                            {
                                'type': 'send_progress',
                                'progress': progress,
                                'scan_name': scan_name,
                                'custom_message': f'Performing {method_label} registration for {scan_name}...',
                                'total': total_scans,
                                'current': idx + 1,
                            }
                        )
                    
                    # Store original data type and range
                    reference_range = [float(np.min(reference_data)), float(np.max(reference_data))]
                    
                    print(f"Original dtype: {original_dtype}")
                    print(f"Moving image range: {moving_image_range}")
                    print(f"Reference range: {reference_range}")                

                    reference_base_path = self.get_reference_base_path(directory, reference)
                    fixed_image_ants = ants.image_read(os.path.join(reference_base_path, f"{reference}.nii.gz"))
                    fixed_image_spacing = fixed_image_ants.spacing
                    fixed_image_origin = fixed_image_ants.origin
                    fixed_image_direction = fixed_image_ants.direction

                    fixed_image_ants = None
                    cleanup_memory()

                    half_spacing = tuple(float(s) * estimation_stride for s in fixed_image_spacing)
                    ants_canvas_origin = ants_origin_for_canvas_paste(
                        fixed_image_origin,
                        fixed_image_direction,
                        ref_paste,
                        fixed_image_spacing,
                    )
                    ants_rotation_center_voxel = np.asarray(reference_centroid, dtype=np.float64)
                    est_slices = tuple(slice(None, None, estimation_stride) for _ in range(3))

                    adjusted_background_value = self.registration_tools.get_adjusted_background_value(
                        background_value, scan_data.astype(np.float32), new_min=-1, new_max=1
                    )
                    print(f"Adjusted background value: {adjusted_background_value} from {background_value}")
                    
                    print(
                        f"Creating normalized ANTs images for estimation "
                        f"(stride={estimation_stride}, grid={estimation_shape})"
                    )

                    fixed_image_data = ants.from_numpy(
                        self.registration_tools.min_max_normalize(
                            reference_on_canvas[est_slices].astype(np.float32), new_min=-1, new_max=1
                        ),
                        spacing=half_spacing,
                        origin=tuple(float(x) for x in ants_canvas_origin),
                        direction=fixed_image_direction,
                    )

                    moving_image_data = ants.from_numpy(
                        self.registration_tools.min_max_normalize(
                            moving_on_canvas[est_slices].astype(np.float32), new_min=-1, new_max=1
                        ),
                        spacing=half_spacing,
                        origin=tuple(float(x) for x in ants_canvas_origin),
                        direction=fixed_image_direction,
                    )

                    fixed_image_mask = None
                    moving_image_mask = None
                    if use_reg_mask and reference_mask_on_canvas is not None:
                        fixed_image_mask = _ants_binary_mask_image(
                            reference_mask_on_canvas[est_slices],
                            half_spacing,
                            ants_canvas_origin,
                            fixed_image_direction,
                            dilate_iters=3,
                        )
                        moving_image_mask = _ants_binary_mask_image(
                            moving_mask_on_canvas[est_slices],
                            half_spacing,
                            ants_canvas_origin,
                            fixed_image_direction,
                            dilate_iters=3,
                        )
                        _ants_rigid_log(scan_name, "content masks enabled for Rigid registration")
                    elif use_reg_mask:
                        _ants_rigid_log(scan_name, "content masks requested but unavailable")
                    else:
                        _ants_rigid_log(
                            scan_name,
                            "masks off — Mattes MI uses full voxel grid (texture-driven)",
                        )
                
                    cleanup_memory()

                    if method.lower() == 'gpu-rigid':
                        _ants_rigid_log(scan_name, f"=== GPU Rigid (FireANTs): {scan_name} -> {reference} ===")
                        _ants_rigid_log(scan_name, f"gpu_rigid_options: {gpu_rigid_opts}")
                        _ants_rigid_log(
                            scan_name,
                            f"estimation grid shape fixed={fixed_image_data.shape} moving={moving_image_data.shape}, "
                            f"spacing={fixed_image_data.spacing}, stride={estimation_stride}",
                        )
                        try:
                            gpu_debug_dir = None
                            if gpu_rigid_opts.get("debug_checkpoints", True):
                                gpu_debug_dir = resolve_gpu_rigid_debug_dir(directory, scan_name)
                                _ants_rigid_log(scan_name, f"GPU Rigid debug checkpoints → {gpu_debug_dir}")
                            gpu_result = run_gpu_rigid_registration(
                                fixed_image_data,
                                moving_image_data,
                                gpu_rigid_opts,
                                scan_name=scan_name,
                                fixed_mask=fixed_image_mask,
                                moving_mask=moving_image_mask,
                                reference_shape=reference_shape,
                                canvas_shape=canvas_shape,
                                estimation_stride=estimation_stride,
                                debug_dir=gpu_debug_dir,
                                reference_name=reference,
                            )
                        except GpuRigidError as exc:
                            print(f"GPU Rigid failed for {scan_name}: {exc}")
                            if gpu_rigid_opts.get("debug_checkpoints", True):
                                print(
                                    f"GPU Rigid debug checkpoints saved under "
                                    f"extracted/{scan_name}/gpu-rigid-debug/"
                                )
                            continue
                        gpu_initializer_record = gpu_result.get("initializer_record")
                        mat_path = gpu_result["mat_path"]
                        gpu_rigid_R = gpu_result.get("R")
                        gpu_rigid_t = gpu_result.get("t")
                        rigid_transform = {"fwdtransforms": [mat_path]}
                        _log_ants_rigid_transform_summary(scan_name, mat_path)
                    else:
                        _ants_rigid_log(scan_name, f"=== ANTs rigid: {scan_name} -> {reference} ===")
                        _ants_rigid_log(scan_name, _format_ants_rigid_opts_for_log(ants_rigid_opts))
                        _ants_rigid_log(
                            scan_name,
                            f"estimation grid shape fixed={fixed_image_data.shape} moving={moving_image_data.shape}, "
                            f"spacing={fixed_image_data.spacing}, stride={estimation_stride}",
                        )

                        # affine_initializer → Affine registration → project to rigid (R,t only).
                        initial_transform, ants_initializer_record = compute_ants_rigid_initial_transform(
                            fixed_image_data, moving_image_data, ants_rigid_opts, scan_name=scan_name
                        )
                        if isinstance(initial_transform, str):
                            _ants_rigid_log(scan_name, f"Affine registration initial_transform file: {initial_transform}")
                        else:
                            _ants_rigid_log(scan_name, f"Affine registration initial_transform: {initial_transform}")

                        reg_identity_ncc, reg_identity_iou = _score_rigid_on_overlap(
                            fixed_image_data, moving_image_data, np.eye(3), np.zeros(3)
                        )
                        _ants_rigid_log(
                            scan_name,
                            f"registration-grid identity baseline: ncc={reg_identity_ncc:.3f} "
                            f"iou={reg_identity_iou:.3f}",
                        )

                        def _run_affine_registration(initial_tx, use_masks=True):
                            reg_kwargs = {
                                "fixed": fixed_image_data,
                                "moving": moving_image_data,
                                "type_of_transform": "Affine",
                                "aff_metric": ants_rigid_opts["aff_metric"],
                                "aff_iterations": ants_rigid_opts["aff_iterations"],
                                "aff_shrink_factors": ants_rigid_opts["aff_shrink_factors"],
                                "aff_smoothing_sigmas": ants_rigid_opts["aff_smoothing_sigmas"],
                                "aff_sampling": ants_rigid_opts["aff_sampling"],
                                "aff_random_sampling_rate": ants_rigid_opts["aff_random_sampling_rate"],
                                "grad_step": ants_rigid_opts["grad_step"],
                                "use_histogram_matching": ants_rigid_opts["use_histogram_matching"],
                                "initial_transform": initial_tx,
                                "singleprecision": ants_rigid_opts["singleprecision"],
                                "verbose": True,
                            }
                            if use_masks and fixed_image_mask is not None:
                                reg_kwargs["mask"] = fixed_image_mask
                            if use_masks and moving_image_mask is not None:
                                reg_kwargs["moving_mask"] = moving_image_mask
                            return ants.registration(**reg_kwargs)

                        def _registration_quality_ok_from_mat(mat_path):
                            if not mat_path:
                                return False, None, -1.0, 0.0
                            ncc, iou = _score_ants_rigid_transform(
                                fixed_image_data, moving_image_data, mat_path
                            )
                            improved = (
                                iou >= reg_identity_iou - 0.01
                                and (ncc >= reg_identity_ncc + 0.01 or iou >= reg_identity_iou + 0.02)
                            )
                            return improved, mat_path, ncc, iou

                        started_from_initializer = _initial_transform_from_initializer(initial_transform)

                        _ants_rigid_log(
                            scan_name,
                            "ants.registration (Affine) starting — ITK iteration log follows below",
                        )
                        affine_reg = _ants_run_with_heartbeat(
                            scan_name,
                            "ants.registration (Affine)",
                            lambda: _run_affine_registration(initial_transform, use_masks=True),
                            interval_sec=45.0,
                        )
                        rigid_transform, mat_path = project_affine_mat_path_to_rigid_transform(
                            affine_reg, scan_name=scan_name
                        )
                        ok, mat_path, reg_ncc, reg_iou = _registration_quality_ok_from_mat(mat_path)
                        if not ok and fixed_image_mask is not None:
                            _ants_rigid_log(
                                scan_name,
                                f"ants.registration weak with masks (ncc={reg_ncc:.3f} iou={reg_iou:.3f}); "
                                "retrying same initial transform without masks",
                            )
                            affine_reg = _ants_run_with_heartbeat(
                                scan_name,
                                "ants.registration (Affine, no masks)",
                                lambda: _run_affine_registration(initial_transform, use_masks=False),
                                interval_sec=45.0,
                            )
                            rigid_transform, mat_path = project_affine_mat_path_to_rigid_transform(
                                affine_reg, scan_name=scan_name
                            )
                            ok, mat_path, reg_ncc, reg_iou = _registration_quality_ok_from_mat(mat_path)
                        if not ok and not started_from_initializer:
                            _ants_rigid_log(
                                scan_name,
                                f"ants.registration result weak (ncc={reg_ncc:.3f} iou={reg_iou:.3f} "
                                f"vs identity ncc={reg_identity_ncc:.3f} iou={reg_identity_iou:.3f}); "
                                "retrying from Identity",
                            )
                            affine_reg = _ants_run_with_heartbeat(
                                scan_name,
                                "ants.registration (Affine retry Identity)",
                                lambda: _run_affine_registration(["Identity"], use_masks=False),
                                interval_sec=45.0,
                            )
                            rigid_transform, mat_path = project_affine_mat_path_to_rigid_transform(
                                affine_reg, scan_name=scan_name
                            )
                            ok, mat_path, reg_ncc, reg_iou = _registration_quality_ok_from_mat(mat_path)
                            _ants_rigid_log(
                                scan_name,
                                f"Identity retry quality: ncc={reg_ncc:.3f} iou={reg_iou:.3f}",
                            )
                        elif not ok and started_from_initializer:
                            _ants_rigid_log(
                                scan_name,
                                f"keeping initializer-based Affine→rigid result despite weak score vs identity "
                                f"(ncc={reg_ncc:.3f} iou={reg_iou:.3f} vs identity ncc={reg_identity_ncc:.3f} "
                                f"iou={reg_identity_iou:.3f})",
                            )
                        else:
                            _ants_rigid_log(
                                scan_name,
                                f"registration quality ok: ncc={reg_ncc:.3f} iou={reg_iou:.3f}",
                            )

                        if mat_path:
                            _log_ants_rigid_transform_summary(scan_name, mat_path)

                    moving_image_data = None
                    fixed_image_data = None
                    fixed_image_mask = None
                    moving_image_mask = None
                    reference_mask_on_canvas = None
                    moving_mask_on_canvas = None
                    if method.lower() != "gpu-rigid":
                        reference_on_canvas = None
                        moving_on_canvas = None
                    if method.lower() == "gpu-rigid":
                        try:
                            import torch

                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    cleanup_memory()

                    print("Rigid transformation complete")

                    if method.lower() == "gpu-rigid":
                        if gpu_rigid_R is None or gpu_rigid_t is None:
                            print(f"GPU Rigid missing R/t for {scan_name}, skipping")
                            continue
                        _ants_rigid_log(
                            scan_name,
                            "Applying GPU pose on union canvas (same frame as estimation/debug)",
                        )
                        warped_canvas = apply_rigid_rt_on_union_canvas(
                            moving_on_canvas,
                            gpu_rigid_R,
                            gpu_rigid_t,
                            spacing=fixed_image_spacing,
                            reference_origin=fixed_image_origin,
                            direction=fixed_image_direction,
                            ref_paste=ref_paste,
                            reference_canvas=reference_on_canvas,
                            defaultvalue=background_value,
                        )
                        reference_on_canvas = None
                        moving_on_canvas = None
                        cleanup_memory()
                        transformed_data = crop_registration_canvas_to_reference(
                            warped_canvas, ref_paste, reference_shape
                        )
                        del warped_canvas
                        rotation = np.asarray(gpu_rigid_R, dtype=np.float64).T
                        padding_low = np.zeros(3, dtype=np.int64)
                        padding_high = np.zeros(3, dtype=np.int64)
                        offset = np.zeros(3, dtype=np.float64)
                        centroid_translation = np.zeros(3, dtype=np.float64)
                        padded_scan_centroid = np.asarray(scan_centroid, dtype=np.float64)
                        reference_centroid_voxel = np.asarray(reference_centroid, dtype=np.float64)
                        rs = tuple(int(s) for s in reference_shape)
                        in_start_x, in_end_x = 0, rs[0]
                        in_start_y, in_end_y = 0, rs[1]
                        in_start_z, in_end_z = 0, rs[2]
                        out_start_x, out_end_x = 0, rs[0]
                        out_start_y, out_end_y = 0, rs[1]
                        out_start_z, out_end_z = 0, rs[2]
                        padding_bottom_val = padding_low
                        padding_top_val = padding_high
                        translation = centroid_translation
                        print(
                            f"After GPU canvas apply: {np.min(transformed_data)} to {np.max(transformed_data)}"
                        )
                        self.store_alignment_embedding(
                            method,
                            {
                                "padding_low": padding_low,
                                "padding_high": padding_high,
                                "rotation": rotation,
                                "offset": offset,
                                "centroid_translation": centroid_translation,
                                "padded_scan_centroid": padded_scan_centroid,
                                "reference_centroid_voxel": reference_centroid_voxel,
                                "crop_indices": {
                                    "x": (in_start_x, in_end_x),
                                    "y": (in_start_y, in_end_y),
                                    "z": (in_start_z, in_end_z),
                                },
                                "reference_shape": reference_shape,
                                "reference_voxel_size": reference_voxel_size,
                                "original_voxel_size": original_voxel_size,
                                "scale_match": 1.0,
                            },
                        )
                        scan_data = None
                        cleanup_memory()
                    else:
                        print("Projecting ANTs pose into scipy rigid and applying on reference FOV")

                        ants_mat_path = mat_path
                        if not ants_mat_path or not os.path.isfile(ants_mat_path):
                            for candidate in (
                                list(rigid_transform.get('fwdtransforms') or [])
                                + list(rigid_transform.get('invtransforms') or [])
                            ):
                                if isinstance(candidate, str) and candidate.endswith('.mat') and os.path.isfile(candidate):
                                    ants_mat_path = candidate
                                    break
                        if not ants_mat_path:
                            print(f"No GenericAffine .mat from registration for {scan_name}, skipping")
                            continue

                        _ants_rigid_log(
                            scan_name,
                            f"scipy rigid apply starting — reference_shape={reference_shape}",
                        )
                        # scan_data is already scale-matched in-place when ``scaling`` is enabled.
                        warp = self._apply_scipy_rigid_from_ants_mat(
                            scan_data,
                            reference_data,
                            ants_mat_path,
                            scan_centroid_voxel=scan_centroid,
                            reference_centroid_voxel=reference_centroid,
                            rotation_center_voxel=ants_rotation_center_voxel,
                            fixed_spacing=fixed_image_spacing,
                            fixed_origin=fixed_image_origin,
                            fixed_direction=fixed_image_direction,
                            mov_paste=mov_paste,
                            ref_paste=ref_paste,
                            reference_shape=reference_shape,
                            background_value=background_value,
                            interpolation_order=interpolation_order,
                            border_width=border_width,
                            original_voxel_size=original_voxel_size,
                            reference_voxel_size=reference_voxel_size,
                            scale_match=1.0,
                            foreground_threshold=threshold,
                            scan_name=scan_name,
                        )
                        if warp is None:
                            print(f"Scipy rigid warp failed for {scan_name}, skipping")
                            continue

                        transformed_data = warp['transformed_data']
                        rotation = warp['rotation']
                        offset = warp['offset']
                        padding_low = warp['padding_low']
                        padding_high = warp['padding_high']
                        centroid_translation = warp['centroid_translation']
                        padded_scan_centroid = warp['padded_scan_centroid']
                        reference_centroid_voxel = warp['reference_centroid_voxel']
                        in_start_x, in_end_x = warp['in_start_x'], warp['in_end_x']
                        in_start_y, in_end_y = warp['in_start_y'], warp['in_end_y']
                        in_start_z, in_end_z = warp['in_start_z'], warp['in_end_z']
                        out_start_x, out_end_x = warp['out_start_x'], warp['out_end_x']
                        out_start_y, out_end_y = warp['out_start_y'], warp['out_end_y']
                        out_start_z, out_end_z = warp['out_start_z'], warp['out_end_z']
                        padding_bottom_val = padding_low
                        padding_top_val = padding_high
                        translation = centroid_translation

                        scan_data = None
                        cleanup_memory()

                        print(f"After scipy rigid warp: {np.min(transformed_data)} to {np.max(transformed_data)}")

                        # Same scipy embedding metadata as manual guidepoints / DINO-Reg.
                        self.store_alignment_embedding(method, warp['embed'])

                # -------------------------------------------------------------------------------------------------------
                # -------------------------------------- Manual Guidepoints Method --------------------------------------
                # -------------------------------------------------------------------------------------------------------
                
                elif method.lower() == 'manual-guidepoints':
                    
                    # Calculate scaling factor if scaling is enabled
                    scale_match = 1
                    alternative = 1
                    if scaling and alternative == 0:
                        # Send progress update for scaling calculation
                        channel_layer = get_channel_layer()
                        if channel_layer is not None:
                            progress = ((idx+(3/9)) / total_scans)
                            async_to_sync(channel_layer.group_send)(
                                'progress_group',
                                {
                                    'type': 'send_progress',
                                    'progress': progress,
                                    'scan_name': scan_name,
                                    'custom_message': f'Calculating scaling factor for {scan_name}...',
                                    'total': total_scans,
                                    'current': idx + 1,
                                }
                            )
                        
                        # Generate meshes with marching cubes
                        reference_vertices, reference_faces, _, _ = measure.marching_cubes(
                            reference_data, level=reference_threshold, 
                            spacing=(reference_voxel_size, reference_voxel_size, reference_voxel_size)
                        )
                        reference_faces = reference_faces[:, ::-1]  # Invert faces for consistent orientation
                        
                        scan_vertices, scan_faces, _, _ = measure.marching_cubes(
                            scan_data, level=threshold, 
                            spacing=(voxel_size, voxel_size, voxel_size)
                        )
                        scan_faces = scan_faces[:, ::-1]  # Invert faces for consistent orientation
                        
                        # Use the existing get_outer_mesh function from ALPACA class
                        reference_outer_vertices, reference_outer_faces = alpaca.get_outer_mesh(reference_vertices, reference_faces)
                        scan_outer_vertices, scan_outer_faces = alpaca.get_outer_mesh(scan_vertices, scan_faces)
                        
                        # Calculate centroids of outer shell vertices
                        reference_outer_centroid = np.mean(reference_outer_vertices, axis=0)
                        scan_outer_centroid = np.mean(scan_outer_vertices, axis=0)
                        
                        # Center the shells at origin
                        reference_outer_centered = reference_outer_vertices - reference_outer_centroid
                        scan_outer_centered = scan_outer_vertices - scan_outer_centroid
                        
                        # Calculate mean absolute distances from centroid
                        reference_mean_distance = np.mean(np.linalg.norm(reference_outer_centered, axis=1))
                        scan_mean_distance = np.mean(np.linalg.norm(scan_outer_centered, axis=1))
                        
                        # Calculate scaling factor
                        scale_match = reference_mean_distance / scan_mean_distance
                        print(f"Calculated scaling factor: {scale_match}")
                        
                        # Clean up
                        reference_vertices = reference_faces = scan_vertices = scan_faces = None
                        reference_outer_vertices = reference_outer_faces = None
                        scan_outer_vertices = scan_outer_faces = None
                        reference_outer_centered = scan_outer_centered = None
                        cleanup_memory()
                         
                    else:
                        # Send progress update for guidepoint-based alignment
                        channel_layer = get_channel_layer()
                        if channel_layer is not None:
                            progress = ((idx+(3/9)) / total_scans)
                            async_to_sync(channel_layer.group_send)(
                                'progress_group',
                                {
                                    'type': 'send_progress',
                                    'progress': progress,
                                    'scan_name': scan_name,
                                    'custom_message': f'Preparing guidepoint-based alignment for {scan_name}...',
                                    'total': total_scans,
                                    'current': idx + 1,
                                }
                            )
                    
                   
                    
                    # Load subject guidepoints (from the latest edit or original if no edits)
                    if latest_edit >= 0:
                        # Look for guidepoints from the latest edit
                        subject_guidepoints_path = glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{latest_edit}_*_guidepoints.json"))
                    else:
                        # No edits exist, look for original guidepoints
                        subject_guidepoints_path = glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_guidepoints.json"))
                    
                    if len(subject_guidepoints_path) > 0:
                        subject_guidepoints_path = subject_guidepoints_path[0]
                    else:
                        print(f"No guidepoints found for subject {scan_name}, skipping")
                        continue
                    
                    # Load subject guidepoints
                    with open(subject_guidepoints_path, 'r') as f:
                        subject_guidepoints = np.array(json.load(f))
                    
                    # Verify shape consistency
                    if reference_guidepoints.shape != subject_guidepoints.shape:
                        print(f"Guidepoint arrays for reference and subject {scan_name} have different shapes")
                        print(f"Reference: {reference_guidepoints.shape}, Subject: {subject_guidepoints.shape}")
                        continue
                    
                    # Send progress update for alignment calculation
                    channel_layer = get_channel_layer()
                    if channel_layer is not None:
                        progress = ((idx+(4/9)) / total_scans)
                        async_to_sync(channel_layer.group_send)(
                            'progress_group',
                            {
                                'type': 'send_progress',
                                'progress': progress,
                                'scan_name': scan_name,
                                'custom_message': f'Calculating guidepoint-based transformation for {scan_name}...',
                                'total': total_scans,
                                'current': idx + 1,
                            }
                        )

                    if scaling and alternative == 1:
                        # Scale the subject, reference and their guidepoints by the scale_match calculated by mean distance to guidepoint centroid
                        guidepoints_reference_centroid = np.mean(reference_guidepoints, axis=0)
                        guidepoints_scan_centroid = np.mean(subject_guidepoints, axis=0)

                        # Mean distance to guidepoint centroid
                        reference_mean_distance = np.mean(np.linalg.norm(reference_guidepoints - guidepoints_reference_centroid, axis=1))
                        scan_mean_distance = np.mean(np.linalg.norm(subject_guidepoints - guidepoints_scan_centroid, axis=1))

                        # Calculate scaling factor
                        scale_match = reference_mean_distance / scan_mean_distance
                        print(f"Calculated scaling factor: {scale_match}")

                        # Scale the guidepoints
                        subject_guidepoints = subject_guidepoints * scale_match

                        # Scale image data by the same scaling factor
                        scan_data = ndimage.zoom(scan_data, scale_match, order=1, mode='constant', cval=background_value)
                        

                    warp = self._apply_rigid_warp_from_point_pairs(
                        scan_data,
                        reference_data,
                        subject_guidepoints,
                        reference_guidepoints,
                        voxel_size,
                        reference_voxel_size,
                        reference_shape,
                        background_value,
                        interpolation_order,
                        border_width,
                        original_voxel_size,
                        scale_match=scale_match,
                        scan_name=scan_name,
                    )
                    if warp is None:
                        continue

                    channel_layer = get_channel_layer()
                    if channel_layer is not None:
                        progress = ((idx+(5/9)) / total_scans)
                        async_to_sync(channel_layer.group_send)(
                            'progress_group',
                            {
                                'type': 'send_progress',
                                'progress': progress,
                                'scan_name': scan_name,
                                'custom_message': f'Applying guidepoint-based transformation to {scan_name}...',
                                'total': total_scans,
                                'current': idx + 1,
                            }
                        )

                    transformed_data = warp['transformed_data']
                    rotation = warp['rotation']
                    offset = warp['offset']
                    padding_low = warp['padding_low']
                    padding_high = warp['padding_high']
                    centroid_translation = warp['centroid_translation']
                    padded_scan_centroid = warp['padded_scan_centroid']
                    reference_centroid_voxel = warp['reference_centroid_voxel']
                    in_start_x, in_end_x = warp['in_start_x'], warp['in_end_x']
                    in_start_y, in_end_y = warp['in_start_y'], warp['in_end_y']
                    in_start_z, in_end_z = warp['in_start_z'], warp['in_end_z']
                    out_start_x, out_end_x = warp['out_start_x'], warp['out_end_x']
                    out_start_y, out_end_y = warp['out_start_y'], warp['out_end_y']
                    out_start_z, out_end_z = warp['out_start_z'], warp['out_end_z']

                    scan_data = None
                    cleanup_memory()

                    self.store_alignment_embedding('manual-guidepoints', warp['embed'])

                elif method.lower() == 'dino-reg':
                    from .dinoRegRigid import DinoRegError, find_dino_reg_rigid_pose

                    scale_match = 1
                    channel_layer = get_channel_layer()
                    if channel_layer is not None:
                        progress = ((idx+(3/9)) / total_scans)
                        async_to_sync(channel_layer.group_send)(
                            'progress_group',
                            {
                                'type': 'send_progress',
                                'progress': progress,
                                'scan_name': scan_name,
                                'custom_message': f'Encoding DINOv3 features for {scan_name}...',
                                'total': total_scans,
                                'current': idx + 1,
                            }
                        )

                    def _dino_progress(message):
                        progress_layer = get_channel_layer()
                        if progress_layer is not None:
                            async_to_sync(progress_layer.group_send)(
                                'progress_group',
                                {
                                    'type': 'send_progress',
                                    'progress': ((idx+(4/9)) / total_scans),
                                    'scan_name': scan_name,
                                    'custom_message': message,
                                    'total': total_scans,
                                    'current': idx + 1,
                                }
                            )

                    try:
                        from .dinoRegDebug import resolve_dino_reg_debug_dir
                        dino_debug_dir = resolve_dino_reg_debug_dir(directory, scan_name)
                        pose = find_dino_reg_rigid_pose(
                            reference_data,
                            scan_data,
                            reference_voxel_size,
                            voxel_size,
                            scan_name=scan_name,
                            reference_name=reference,
                            moving_threshold=threshold,
                            fixed_threshold=reference_threshold,
                            debug_dir=dino_debug_dir,
                            progress_callback=_dino_progress,
                        )
                    except DinoRegError as exc:
                        return Response({'error': str(exc)}, status=status.HTTP_400_BAD_REQUEST)

                    if pose is None:
                        print(
                            f"DINO-Reg could not support a rigid for {scan_name} "
                            "(semantic landmarks / refine / pyramid); skipping"
                        )
                        continue

                    # Pose extras → alignment_rigid_provenance.dino_reg_result for Scientific Report.
                    dino_reg_result = {
                        key: pose[key]
                        for key in (
                            "source",
                            "probe_name",
                            "angle_deg",
                            "refined_cosine",
                            "label",
                        )
                        if key in pose and pose[key] is not None
                    }

                    _dino_progress(f'Applying DINO-Reg rigid warp to {scan_name}...')
                    # Content-mask centroids on the resampled reference grid (not DINO FG centroids).
                    # Physical p_fixed = R @ p_moving + t → scipy rotate-about-scan_c + residual + canvas snap.
                    scan_content_mask = _content_mask_from_threshold(
                        scan_data, threshold, sigma=1.0
                    )
                    reference_content_mask = _content_mask_from_threshold(
                        reference_data, reference_threshold, sigma=1.0
                    )
                    scan_centroid_voxel = _centroid_of_mask(scan_content_mask)
                    reference_centroid_voxel = _centroid_of_mask(reference_content_mask)
                    scipy_rotation, scipy_scan_c, scipy_ref_c, residual_translation = (
                        project_physical_rigid_rt_to_scipy_rigid_params(
                            pose["R"],
                            pose["t"],
                            scan_centroid_voxel,
                            reference_centroid_voxel,
                            reference_voxel_size,
                            scan_name=scan_name,
                        )
                    )
                    print(
                        f"DINO-Reg {scan_name} scipy apply: "
                        f"content centroids scan={scipy_scan_c.round(1).tolist()} "
                        f"ref={scipy_ref_c.round(1).tolist()} "
                        f"residual_vox={residual_translation.round(2).tolist()} "
                        f"angle≈{float(pose.get('angle_deg', 0)):.1f}°"
                    )
                    warp = self._apply_scipy_rigid_volume_warp(
                        scan_data,
                        reference_data,
                        scipy_rotation,
                        scipy_scan_c,
                        scipy_ref_c,
                        reference_shape,
                        background_value,
                        interpolation_order,
                        border_width,
                        original_voxel_size,
                        reference_voxel_size,
                        scale_match=scale_match,
                        residual_translation=residual_translation,
                        scan_name=scan_name,
                    )
                    if warp is None:
                        continue

                    if dino_debug_dir:
                        try:
                            from .dinoRegDebug import append_dino_reg_apply_forensics
                            append_dino_reg_apply_forensics(
                                dino_debug_dir,
                                pose=pose,
                                reference_data=reference_data,
                                transformed_data=warp["transformed_data"],
                                scan_content_mask=scan_content_mask,
                                reference_content_mask=reference_content_mask,
                                scan_centroid_voxel=scipy_scan_c,
                                reference_centroid_voxel=scipy_ref_c,
                                scipy_rotation=scipy_rotation,
                                residual_translation_voxel=residual_translation,
                                warp=warp,
                                reference_voxel_size=float(reference_voxel_size),
                                scan_name=scan_name,
                            )
                        except Exception as debug_exc:
                            print(f"DINO-Reg apply debug forensics skipped for {scan_name}: {debug_exc}")

                    transformed_data = warp['transformed_data']
                    rotation = warp['rotation']
                    offset = warp['offset']
                    padding_low = warp['padding_low']
                    padding_high = warp['padding_high']
                    centroid_translation = warp['centroid_translation']
                    padded_scan_centroid = warp['padded_scan_centroid']
                    reference_centroid_voxel = warp['reference_centroid_voxel']
                    in_start_x, in_end_x = warp['in_start_x'], warp['in_end_x']
                    in_start_y, in_end_y = warp['in_start_y'], warp['in_end_y']
                    in_start_z, in_end_z = warp['in_start_z'], warp['in_end_z']
                    out_start_x, out_end_x = warp['out_start_x'], warp['out_end_x']
                    out_start_y, out_end_y = warp['out_start_y'], warp['out_end_y']
                    out_start_z, out_end_z = warp['out_start_z'], warp['out_end_z']

                    scan_data = None
                    cleanup_memory()
                    self.store_alignment_embedding('dino-reg', warp['embed'])

                else:
                    # Unknown method
                    return Response({'error': f'Unknown alignment method: {method}'}, status=status.HTTP_400_BAD_REQUEST)

                # ------- Alignment complete, now some general cleanup -------
                

                # Send progress update for cleaning aligned data
                channel_layer = get_channel_layer()
                if channel_layer is not None:
                    progress = ((idx+(4/9)) / total_scans)
                    async_to_sync(channel_layer.group_send)(
                        'progress_group',
                        {
                            'type': 'send_progress',
                            'progress': progress,
                            'scan_name': scan_name,
                            'custom_message': f'Cleaning the aligned voxel data from {scan_name}...',
                            'total': total_scans,
                            'current': idx + 1,
                        }
                    )
                    print("Cleaning the aligned voxel data from ", scan_name)

                # Check available memory
                remaining_memory = psutil.virtual_memory().available / 1024 / 1024 / 1024
                print(f"Available memory: {remaining_memory:.2f} GB")

                # Clean the resulting mesh using the class CleanupMeshView and save it in the variable transformed_data
                # transformed_data = self.clean_mesh(transformed_data, threshold, original_dtype, background_value)
                gc.collect()

                
                # Send progress update 
                channel_layer = get_channel_layer()
                if channel_layer is not None:
                    progress = ((idx+(5/9)) / total_scans)
                    async_to_sync(channel_layer.group_send)(
                        'progress_group',
                        {
                            'type': 'send_progress',
                            'progress': progress,
                            'scan_name': scan_name,
                            'custom_message': f'Performing adjustments to {scan_name}...',
                            'total': total_scans,
                            'current': idx + 1,
                        }
                    )
                    print("Performing adjustments to ", scan_name)

                # Check available memory
                remaining_memory = psutil.virtual_memory().available / 1024 / 1024 / 1024
                print(f"Available memory: {remaining_memory:.2f} GB")
                

                """
                # Send progress update 
                channel_layer = get_channel_layer()
                if channel_layer is not None:
                    progress = ((idx+(6/9)) / total_scans)
                    async_to_sync(channel_layer.group_send)(
                        'progress_group',
                        {
                            'type': 'send_progress',
                            'progress': progress,
                            'scan_name': scan_name,
                            'custom_message': f'Linearly shifting the histogram of {scan_name} to match the reference scan...',
                            'total': total_scans,
                            'current': idx + 1,
                        }
                    )
                    print("Linearly shifting the histogram of ", scan_name)

                # Check available memory
                remaining_memory = psutil.virtual_memory().available / 1024 / 1024 / 1024
                print(f"Available memory: {remaining_memory:.2f} GB")

                moving_image_range = [float(np.min(transformed_data)), float(np.max(transformed_data))]
                reference_range = [float(np.min(reference_data)), float(np.max(reference_data))]

                # Normalize both images to [0,1] for histogram matching
                registered_normalized = self.registration_tools.min_max_normalize(
                    transformed_data[::2, ::2, ::2].astype(np.float32), 
                    new_min=0, 
                    new_max=1
                )
                reference_normalized = self.registration_tools.min_max_normalize(
                    reference_data[::2, ::2, ::2].astype(np.float32), 
                    new_min=0, 
                    new_max=1
                )

                # Match histograms and restore to reference scan range
                transformed_data_matched = exposure.match_histograms(
                    registered_normalized, 
                    reference_normalized, 
                    channel_axis=None
                )

                print(f"histograms matched")

                reference_normalized = None
                registered_normalized = None
                cleanup_memory()

                transformed_data_matched = self.registration_tools.restore_range(
                    transformed_data_matched, 
                    reference_range[0], 
                    reference_range[1]
                ).astype(original_dtype)

                # Calculate and apply alignment shift
                alignment_shift = np.mean(gaussian_filter(transformed_data_matched, sigma=2)) - np.mean(gaussian_filter(transformed_data, sigma=2)).astype(original_dtype)
                print(f"Alignment shift: {alignment_shift}")

                transformed_data_matched = None
                cleanup_memory()

                # Determine low and upper bounds allowed by original dtype
                low_bound = np.iinfo(original_dtype).min
                high_bound = np.iinfo(original_dtype).max

                # Apply shift while preserving original dtype
                transformed_data = np.clip(
                    transformed_data.astype(np.int32) + alignment_shift.astype(np.int32), 
                    low_bound, 
                    high_bound
                ).astype(original_dtype)

                print(f"min and max from transformed data: {np.min(transformed_data)} {np.max(transformed_data)}")
                print(f"min and max from reference scan data: {np.min(reference_data)} {np.max(reference_data)}")
                """
                alignment_shift = 0


                # Send progress update 
                channel_layer = get_channel_layer()
                if channel_layer is not None:
                    progress = ((idx+(7/9)) / total_scans)
                    async_to_sync(channel_layer.group_send)(
                        'progress_group',
                        {
                            'type': 'send_progress',
                            'progress': progress,
                            'scan_name': scan_name,
                            'custom_message': f'Saving and compressing the aligned version of {scan_name}...',
                            'total': total_scans,
                            'current': idx + 1,
                        }
                    )
                    print("Saving and compressing the aligned version of ", scan_name)

                # Check available memory
                remaining_memory = psutil.virtual_memory().available / 1024 / 1024 / 1024
                print(f"Available memory: {remaining_memory:.2f} GB")

                # Update metadata with alignment info
                previous_threshold = scan_metadata.get('threshold')
                previous_stem = (
                    normalize_threshold_edit_stem(os.path.basename(scan_path), scan_name)
                    if latest_edit >= 0
                    else scan_name
                )
                snapshot_previous_edit_threshold(scan_metadata, previous_stem, previous_threshold)
                scan_metadata['threshold'] = int(scan_metadata['threshold']) + int(alignment_shift)
                scan_metadata['alignment_to'] = reference
                scan_metadata['alignment_shift'] = float(alignment_shift)
                scan_metadata['alignment_method'] = method  # Add method information to metadata
                scan_metadata['alignment_rigid_provenance'] = build_alignment_rigid_provenance(
                    method,
                    scaling,
                    outer_surface,
                    interpolation,
                    ants_rigid_opts,
                    ants_initializer_record,
                    gpu_rigid_opts if method.lower() == 'gpu-rigid' else None,
                    gpu_initializer_record if method.lower() == 'gpu-rigid' else None,
                    dino_reg_result if method.lower() == 'dino-reg' else None,
                )
                # Use original_voxel_size: voxel_size may have been overwritten when resampling to reference spacing
                scan_metadata['alignment_previous_voxel_size'] = original_voxel_size
                scan_metadata['voxel_size'] = reference_voxel_size
                if scale_match != 1:
                    scan_metadata['alignment_scale'] = scale_match
                
                # Store calculated alignment parameters for future landmark inversion
                # Determine padding variable based on method
                alignment_padding = None
                # ALPACA: residual translation after rotate-about-centroid (matrix t with
                # centroid terms intentionally omitted by ALPACA). Guidepoints: no residual
                # in-array shift — centroid snap is canvas_centroid vs rotation_center only.
                if method.lower() == 'alpaca':
                    alignment_padding = padding_bottom_val.tolist() if isinstance(padding_bottom_val, np.ndarray) else padding_bottom_val
                    alignment_translation = translation
                elif uses_scipy_centroid_embed_rigid(method):
                    alignment_padding = padding_low.tolist() if isinstance(padding_low, np.ndarray) else padding_low
                    alignment_translation = centroid_translation
                else:
                    alignment_translation = np.zeros(3)

                if isinstance(alignment_translation, np.ndarray):
                    alignment_translation = alignment_translation.tolist()

                scan_metadata['alignment_calculated_parameters'] = {
                    'alignment_rotation_matrix': rotation.tolist() if isinstance(rotation, np.ndarray) else rotation,
                    'alignment_rotation_center_voxel': padded_scan_centroid.tolist() if isinstance(padded_scan_centroid, np.ndarray) else padded_scan_centroid,
                    'alignment_canvas_centroid_voxel': reference_centroid_voxel.tolist() if isinstance(reference_centroid_voxel, np.ndarray) else reference_centroid_voxel,
                    'alignment_padding_low': alignment_padding,
                    'alignment_padding_high': (
                        padding_top_val.tolist() if method.lower() == 'alpaca' and isinstance(padding_top_val, np.ndarray)
                        else (padding_high.tolist() if uses_scipy_centroid_embed_rigid(method) and isinstance(padding_high, np.ndarray)
                              else (padding_top_val if method.lower() == 'alpaca' else None))
                    ),
                    'alignment_translation_voxel': alignment_translation,
                    'alignment_crop_indices': {
                        'in': {
                            'x': [int(in_start_x), int(in_end_x)],
                            'y': [int(in_start_y), int(in_end_y)],
                            'z': [int(in_start_z), int(in_end_z)],
                        },
                        'out': {
                            'x': [int(out_start_x), int(out_end_x)],
                            'y': [int(out_start_y), int(out_end_y)],
                            'z': [int(out_start_z), int(out_end_z)],
                        },
                    } if method.lower() == 'alpaca' or uses_scipy_centroid_embed_rigid(method) else None,
                }
                
                with open(os.path.join(directory, "extracted", scan_name, f"{scan_name}.json"), 'w') as jf:
                    json.dump(scan_metadata, jf, indent=4)

                # Save the transformed scan
                new_edit_number = latest_edit + 1
                output_path = os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{new_edit_number}_aligned.nii.gz")
                volume_to_save = self._volume_for_mesh_reference_save(
                    transformed_data, reference_is_preserved_mesh
                )
                new_nifti = nib.Nifti1Image(volume_to_save, reference_affine_data)
                nib.save(new_nifti, output_path)

                # Transform landmarks if they exist (before creating lossy version)
                self.transform_landmarks_after_alignment(
                    directory, scan_name, latest_edit, new_edit_number,
                    voxel_size, scan_metadata, method
                )


                # Create lossy version using the proper method from registrationTools
                if 'lossy_compression' in scan_metadata:
                    lossy_path = output_path.replace("_edit", "_lossy_edit")
                    json_file = os.path.join(directory, "extracted", scan_name, f"{scan_name}.json")
                    
                    # Use the existing save_as_lossy_nifti method which handles the new list format
                    self.registration_tools.save_as_lossy_nifti(
                        image_data=volume_to_save,
                        voxel_size=reference_affine_data[0,0],  # Extract voxel size from affine matrix
                        json_file=json_file,
                        output_file=lossy_path
                    )

                transformed_data = None
                cleanup_memory()

                # ---------------------------------------------------------------
                # Paired mask: apply same transformations (padding, rotation, translation, cropping)
                # using nearest neighbor interpolation for binary mask data
                # ---------------------------------------------------------------
                try:
                    # Determine source mask base from the latest edit
                    if latest_edit >= 0:
                        latest_files = glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{latest_edit}_*.nii.gz"))
                        if latest_files:
                            source_mask_edit_base = os.path.basename(latest_files[0]).replace('.nii.gz', '').replace('_lossy', '')
                        else:
                            source_mask_edit_base = scan_name
                    else:
                        source_mask_edit_base = scan_name

                    mask_src_path = os.path.join(directory, "extracted", scan_name, f"{source_mask_edit_base}.nii.mask.gz")
                    
                    if os.path.isfile(mask_src_path):
                        print(f"Found paired mask for alignment: {mask_src_path}")
                        
                        # Load mask data
                        mask_src_temp = os.path.join(directory, "extracted", scan_name, f"{source_mask_edit_base}_mask.nii.gz")
                        os.replace(mask_src_path, mask_src_temp)
                        try:
                            mask_img = nib.load(mask_src_temp)
                            mask_dtype = mask_img.get_data_dtype()
                            # Round before astype to avoid truncating floats
                            mask_data = np.round(mask_img.get_fdata()).astype(mask_dtype)

                            # Apply the same transformations as the scan data, but with nearest neighbor
                            if uses_scipy_centroid_embed_rigid(method):
                                # scipy centroid-embed methods: scaling -> padding -> rotation -> residual -> canvas
                                
                                # 1. Apply scaling if it was applied to scan data
                                if scaling and scale_match != 1:
                                    mask_data = ndimage.zoom(mask_data, scale_match, order=0, mode='constant', cval=0)
                                
                                # 2. Apply same padding as scan data
                                mask_padded = np.pad(
                                    mask_data,
                                    ((padding_low[0], padding_high[0]), 
                                    (padding_low[1], padding_high[1]), 
                                    (padding_low[2], padding_high[2])),
                                    mode='constant',
                                    constant_values=0  # Use 0 for mask background
                                )
                                
                                # 3. Apply rotation with nearest neighbor (same matrix as volume)
                                mask_rotated = ndimage.affine_transform(
                                    mask_padded,
                                    rotation,
                                    offset=offset,  # Same offset as scan data
                                    output_shape=mask_padded.shape,
                                    order=0,  # Nearest neighbor for binary mask
                                    mode='constant',
                                    cval=0
                                )

                                # 4. Residual translation (ANTs / DINO physical pose)
                                residual = np.asarray(
                                    centroid_translation if centroid_translation is not None else [0, 0, 0],
                                    dtype=np.float64,
                                ).reshape(3)
                                if np.any(residual):
                                    mask_rotated = ndimage.shift(
                                        mask_rotated,
                                        shift=residual,
                                        order=0,
                                        mode='constant',
                                        cval=0,
                                    )
                                
                                # 5. Create canvas and crop same as scan data
                                mask_transformed_canvas = np.zeros_like(reference_data, dtype=mask_dtype)
                                
                                # Use same crop indices as calculated for scan data
                                mask_transformed_canvas[
                                    in_start_x:in_end_x,                     
                                    in_start_y:in_end_y, 
                                    in_start_z:in_end_z] = mask_rotated[
                                                                out_start_x:out_end_x,
                                                                out_start_y:out_end_y,
                                                                out_start_z:out_end_z
                                                            ]
                                
                                mask_transformed = mask_transformed_canvas

                            elif method.lower() == 'alpaca':
                                # For ALPACA method: scaling -> padding -> rotation -> translation -> cropping
                                
                                # 1. Apply scaling if it was applied to scan data
                                if scale_match != 1:
                                    mask_data = ndimage.zoom(mask_data, scale_match, order=0, mode='constant', cval=0)
                                
                                # 2. Apply same padding as scan data (using same padding values)
                                mask_padded = np.pad(
                                    mask_data.astype(mask_dtype), 
                                    ((padding_bottom_val[0], padding_top_val[0]), 
                                    (padding_bottom_val[1], padding_top_val[1]), 
                                    (padding_bottom_val[2], padding_top_val[2])),
                                    mode='constant',
                                    constant_values=0
                                ).astype(mask_dtype)
                                
                                # 3. Apply rotation with nearest neighbor (same as ALPACA scan data)
                                mask_rotated = np.zeros_like(mask_padded, dtype=mask_dtype)
                                ndimage.affine_transform(
                                    mask_padded,
                                    rotation.T,
                                    offset=offset,  # Same offset as calculated for scan data
                                    output=mask_rotated,
                                    output_shape=mask_padded.shape,
                                    order=0,  # Nearest neighbor for binary mask
                                    mode='constant',
                                    prefilter=False,
                                    cval=0
                                )
                                
                                # 4. Apply translation (same as ALPACA scan data)
                                mask_translated = np.zeros_like(mask_rotated, dtype=mask_dtype)
                                ndimage.shift(
                                    mask_rotated,
                                    shift=translation,  # Same translation as scan data
                                    output=mask_translated,
                                    order=0,  # Nearest neighbor
                                    mode='constant',
                                    cval=0
                                )
                                
                                # 5. Create canvas and crop same as scan data
                                mask_transformed_canvas = np.zeros_like(reference_data, dtype=mask_dtype)
                                
                                # Use same crop indices as calculated for scan data (ALPACA uses different variable names)
                                mask_transformed_canvas[
                                    in_start_x:in_end_x,                     
                                    in_start_y:in_end_y, 
                                    in_start_z:in_end_z] = mask_translated[
                                                                out_start_x:out_end_x,
                                                                out_start_y:out_end_y,
                                                                out_start_z:out_end_z
                                                            ]
                                
                                mask_transformed = mask_transformed_canvas

                            else:
                                print(f"Mask transformation not implemented for method: {method}")
                                mask_transformed = None
                                                        
                            if mask_transformed is not None:
                                # Save full-res aligned mask
                                new_mask_edit_base = f"{scan_name}_edit_{new_edit_number}_aligned"
                                mask_dest_path = os.path.join(directory, "extracted", scan_name, f"{new_mask_edit_base}.nii.mask.gz")
                                mask_dest_temp = os.path.join(directory, "extracted", scan_name, f"{new_mask_edit_base}_mask.nii.gz")
                                mask_volume_to_save = self._volume_for_mesh_reference_save(
                                    mask_transformed, reference_is_preserved_mesh
                                )
                                nib.save(
                                    nib.Nifti1Image(mask_volume_to_save.astype(mask_dtype), reference_affine_data),
                                    mask_dest_temp,
                                )
                                os.replace(mask_dest_temp, mask_dest_path)
                                print(f"Saved aligned mask: {mask_dest_path}")

                                # Create lossy mask by downsampling if lossy compression is enabled
                                if 'lossy_compression' in scan_metadata:
                                    resolution_factor = 2
                                    lc = scan_metadata.get('lossy_compression', None)
                                    if isinstance(lc, list) and len(lc) > 0:
                                        try:
                                            resolution_factor = int(lc[-1].get('resolution_factor', resolution_factor))
                                        except Exception:
                                            pass
                                    elif isinstance(lc, dict) and lc:
                                        try:
                                            resolution_factor = int(lc.get('resolution_factor', resolution_factor))
                                        except Exception:
                                            pass

                                    slices = [slice(None, None, resolution_factor) for _ in range(3)]
                                    lossy_mask = mask_volume_to_save[slices[0], slices[1], slices[2]].astype(
                                        mask_dtype, copy=False
                                    )

                                    lossy_mask_edit_base = f"{scan_name}_lossy_edit_{new_edit_number}_aligned"
                                    lossy_mask_dest_path = os.path.join(directory, "extracted", scan_name, f"{lossy_mask_edit_base}.nii.mask.gz")
                                    lossy_mask_temp = os.path.join(directory, "extracted", scan_name, f"{lossy_mask_edit_base}_mask.nii.gz")
                                    nib.save(nib.Nifti1Image(lossy_mask, reference_affine_data), lossy_mask_temp)
                                    os.replace(lossy_mask_temp, lossy_mask_dest_path)
                                    print(f"Saved lossy aligned mask: {lossy_mask_dest_path}")
                                
                                # Clean up mask data
                                mask_transformed = None
                                cleanup_memory()
                                
                        finally:
                            # Restore original mask filename
                            os.replace(mask_src_temp, mask_src_path)
                    else:
                        print("No paired mask found for alignment; skipping mask propagation.")
                        
                except Exception as me:
                    print(f"Error handling paired mask during alignment: {me}")

                # ------------------------------------------------------------------
                # Propagate alignment to same-shape linked siblings
                # ------------------------------------------------------------------
                if propagate_linked:
                    try:
                        siblings, sib_err = get_same_shape_siblings(directory, scan_name)
                        if sib_err:
                            print(f"Could not resolve siblings for {scan_name}: {sib_err}")
                        elif siblings:
                            print(f"Propagating alignment from {scan_name} to {len(siblings)} siblings: {siblings}")
                            for sibling_name in siblings:
                                try:
                                    # Skip if sibling is already aligned
                                    sib_latest_edit, sib_latest_stem = self._latest_any_edit_stem(directory, sibling_name)
                                    if sib_latest_edit >= 0:
                                        sib_edit_files = glob.glob(os.path.join(directory, "extracted", sibling_name, f"{sibling_name}_edit_{sib_latest_edit}_*.nii.gz"))
                                        if sib_edit_files and 'aligned' in os.path.basename(sib_edit_files[0]):
                                            linked_propagation_results.append({'scan_name': sibling_name, 'status': 'skipped', 'reason': 'already aligned'})
                                            print(f"  Sibling {sibling_name} already aligned, skipping")
                                            continue
                                    
                                    # Skip if sibling has elastic registration
                                    if has_elastic_registration(sibling_name, directory):
                                        linked_propagation_results.append({'scan_name': sibling_name, 'status': 'skipped', 'reason': 'elastic registration present'})
                                        print(f"  Sibling {sibling_name} has elastic registration, skipping")
                                        continue
                                    
                                    # Skip if sibling is preserved mesh
                                    sib_metadata = self._load_subject_metadata(directory, sibling_name)
                                    if self._is_preserved_mesh_metadata(sib_metadata):
                                        linked_propagation_results.append({'scan_name': sibling_name, 'status': 'skipped', 'reason': 'preserved mesh'})
                                        print(f"  Sibling {sibling_name} is preserved mesh, skipping")
                                        continue
                                    
                                    # Load sibling's latest volume
                                    if sib_latest_edit >= 0:
                                        sib_edit_files = glob.glob(os.path.join(directory, "extracted", sibling_name, f"{sibling_name}_edit_{sib_latest_edit}_*.nii.gz"))
                                        if sib_edit_files:
                                            sib_nifti_path = sib_edit_files[0]
                                        else:
                                            sib_nifti_path = os.path.join(directory, "extracted", sibling_name, f"{sibling_name}.nii.gz")
                                    else:
                                        sib_nifti_path = os.path.join(directory, "extracted", sibling_name, f"{sibling_name}.nii.gz")
                                    
                                    sib_img = nib.load(sib_nifti_path)
                                    sib_dtype = sib_img.get_data_dtype()
                                    sib_data = sib_img.get_fdata().astype(sib_dtype)
                                    
                                    # Apply the same transform based on method
                                    if method.lower() == 'alpaca':
                                        # For ALPACA: reuse geometric parameters
                                        # Apply scaling
                                        if scale_match != 1:
                                            sib_data = ndimage.zoom(sib_data, scale_match, order=interpolation_order, mode='constant', cval=0)
                                        
                                        # Pad
                                        sib_padded = np.pad(
                                            sib_data.astype(sib_dtype),
                                            ((padding_bottom_val[0], padding_top_val[0]),
                                             (padding_bottom_val[1], padding_top_val[1]),
                                             (padding_bottom_val[2], padding_top_val[2])),
                                            mode='constant',
                                            constant_values=0
                                        ).astype(sib_dtype)
                                        
                                        # Rotate
                                        sib_rotated = np.zeros_like(sib_padded, dtype=sib_dtype)
                                        ndimage.affine_transform(
                                            sib_padded,
                                            rotation.T,
                                            offset=offset,
                                            output=sib_rotated,
                                            output_shape=sib_padded.shape,
                                            order=interpolation_order,
                                            mode='constant',
                                            prefilter=False,
                                            cval=0
                                        )
                                        
                                        # Translate
                                        sib_translated = np.zeros_like(sib_rotated, dtype=sib_dtype)
                                        ndimage.shift(
                                            sib_rotated,
                                            shift=translation,
                                            output=sib_translated,
                                            order=interpolation_order,
                                            mode='constant',
                                            cval=0
                                        )
                                        
                                        # Create canvas and crop
                                        sib_canvas = np.zeros_like(reference_data, dtype=sib_dtype)
                                        sib_canvas[
                                            in_start_x:in_end_x,
                                            in_start_y:in_end_y,
                                            in_start_z:in_end_z
                                        ] = sib_translated[
                                            out_start_x:out_end_x,
                                            out_start_y:out_end_y,
                                            out_start_z:out_end_z
                                        ]
                                        sib_transformed = sib_canvas
                                        
                                    elif uses_scipy_centroid_embed_rigid(method):
                                        # Reuse main geometric parameters (rotation + residual + canvas).
                                        if scaling and scale_match != 1:
                                            sib_data = ndimage.zoom(sib_data, scale_match, order=interpolation_order, mode='constant', cval=0)
                                        
                                        # Pad
                                        sib_padded = np.pad(
                                            sib_data,
                                            ((padding_low[0], padding_high[0]),
                                             (padding_low[1], padding_high[1]),
                                             (padding_low[2], padding_high[2])),
                                            mode='constant',
                                            constant_values=0
                                        )
                                        
                                        # Rotate (same matrix convention as main volume warp)
                                        sib_rotated = ndimage.affine_transform(
                                            sib_padded,
                                            rotation,
                                            offset=offset,
                                            output_shape=sib_padded.shape,
                                            order=interpolation_order,
                                            mode='constant',
                                            cval=0
                                        )

                                        residual = np.asarray(
                                            centroid_translation if centroid_translation is not None else [0, 0, 0],
                                            dtype=np.float64,
                                        ).reshape(3)
                                        if np.any(residual):
                                            sib_rotated = ndimage.shift(
                                                sib_rotated,
                                                shift=residual,
                                                order=interpolation_order,
                                                mode='constant',
                                                cval=0,
                                            )
                                        
                                        # Create canvas and crop
                                        sib_canvas = np.zeros_like(reference_data, dtype=sib_dtype)
                                        sib_canvas[
                                            in_start_x:in_end_x,
                                            in_start_y:in_end_y,
                                            in_start_z:in_end_z
                                        ] = sib_rotated[
                                            out_start_x:out_end_x,
                                            out_start_y:out_end_y,
                                            out_start_z:out_end_z
                                        ]
                                        sib_transformed = sib_canvas
                                    else:
                                        linked_propagation_results.append({'scan_name': sibling_name, 'status': 'failed', 'reason': f'unknown method {method}'})
                                        continue
                                    
                                    # Same border gap the main gets after warping, so the group
                                    # keeps identical voxel coverage going into elastic steps.
                                    sib_background_value = np.array(0, dtype=sib_transformed.dtype)
                                    if sib_transformed.shape[0] > 2 * border_width:
                                        sib_transformed[:border_width, :, :] = sib_background_value
                                        sib_transformed[-border_width:, :, :] = sib_background_value
                                    if sib_transformed.shape[1] > 2 * border_width:
                                        sib_transformed[:, :border_width, :] = sib_background_value
                                        sib_transformed[:, -border_width:, :] = sib_background_value
                                    if sib_transformed.shape[2] > 2 * border_width:
                                        sib_transformed[:, :, :border_width] = sib_background_value
                                        sib_transformed[:, :, -border_width:] = sib_background_value

                                    # Save sibling aligned volume
                                    sib_new_edit = sib_latest_edit + 1
                                    sib_output_path = os.path.join(directory, "extracted", sibling_name, f"{sibling_name}_edit_{sib_new_edit}_aligned.nii.gz")
                                    sib_volume_to_save = self._volume_for_mesh_reference_save(sib_transformed, reference_is_preserved_mesh)
                                    sib_nifti = nib.Nifti1Image(sib_volume_to_save, reference_affine_data)
                                    nib.save(sib_nifti, sib_output_path)
                                    
                                    # Update sibling metadata: copy alignment fields from main
                                    sib_metadata['alignment_to'] = reference
                                    sib_metadata['alignment_shift'] = 0
                                    sib_metadata['alignment_method'] = method
                                    sib_metadata['voxel_size'] = reference_voxel_size
                                    if 'alignment_calculated_parameters' in scan_metadata:
                                        sib_metadata['alignment_calculated_parameters'] = scan_metadata['alignment_calculated_parameters']
                                    if 'alignment_rigid_provenance' in scan_metadata:
                                        sib_metadata['alignment_rigid_provenance'] = scan_metadata['alignment_rigid_provenance']
                                    strip_ephemeral_scan_metadata(sib_metadata)
                                    sib_json_path = os.path.join(directory, "extracted", sibling_name, f"{sibling_name}.json")
                                    with open(sib_json_path, 'w') as jf:
                                        json.dump(sib_metadata, jf, indent=4)
                                    
                                    # Create lossy version if needed
                                    if 'lossy_compression' in sib_metadata:
                                        sib_lossy_path = sib_output_path.replace("_edit", "_lossy_edit")
                                        self.registration_tools.save_as_lossy_nifti(
                                            image_data=sib_volume_to_save,
                                            voxel_size=reference_affine_data[0, 0],
                                            json_file=sib_json_path,
                                            output_file=sib_lossy_path
                                        )
                                    
                                    # Propagate paired mask if present
                                    if sib_latest_edit >= 0:
                                        sib_edit_base_files = glob.glob(os.path.join(directory, "extracted", sibling_name, f"{sibling_name}_edit_{sib_latest_edit}_*.nii.gz"))
                                        if sib_edit_base_files:
                                            sib_edit_base = os.path.basename(sib_edit_base_files[0]).replace('.nii.gz', '').replace('_lossy', '')
                                        else:
                                            sib_edit_base = sibling_name
                                    else:
                                        sib_edit_base = sibling_name
                                    
                                    sib_mask_src = os.path.join(directory, "extracted", sibling_name, f"{sib_edit_base}.nii.mask.gz")
                                    if os.path.isfile(sib_mask_src):
                                        sib_mask_temp = sib_mask_src.replace('.nii.mask.gz', '_mask.nii.gz')
                                        os.replace(sib_mask_src, sib_mask_temp)
                                        try:
                                            sib_mask_img = nib.load(sib_mask_temp)
                                            sib_mask_dtype = sib_mask_img.get_data_dtype()
                                            sib_mask_data = np.round(sib_mask_img.get_fdata()).astype(sib_mask_dtype)
                                            
                                            # Apply same transform to mask with nearest neighbor
                                            if method.lower() == 'alpaca':
                                                if scale_match != 1:
                                                    sib_mask_data = ndimage.zoom(sib_mask_data, scale_match, order=0, mode='constant', cval=0)
                                                sib_mask_padded = np.pad(
                                                    sib_mask_data.astype(sib_mask_dtype),
                                                    ((padding_bottom_val[0], padding_top_val[0]),
                                                     (padding_bottom_val[1], padding_top_val[1]),
                                                     (padding_bottom_val[2], padding_top_val[2])),
                                                    mode='constant',
                                                    constant_values=0
                                                ).astype(sib_mask_dtype)
                                                sib_mask_rotated = np.zeros_like(sib_mask_padded, dtype=sib_mask_dtype)
                                                ndimage.affine_transform(
                                                    sib_mask_padded,
                                                    rotation.T,
                                                    offset=offset,
                                                    output=sib_mask_rotated,
                                                    output_shape=sib_mask_padded.shape,
                                                    order=0,
                                                    mode='constant',
                                                    prefilter=False,
                                                    cval=0
                                                )
                                                sib_mask_translated = np.zeros_like(sib_mask_rotated, dtype=sib_mask_dtype)
                                                ndimage.shift(
                                                    sib_mask_rotated,
                                                    shift=translation,
                                                    output=sib_mask_translated,
                                                    order=0,
                                                    mode='constant',
                                                    cval=0
                                                )
                                                sib_mask_canvas = np.zeros_like(reference_data, dtype=sib_mask_dtype)
                                                sib_mask_canvas[
                                                    in_start_x:in_end_x,
                                                    in_start_y:in_end_y,
                                                    in_start_z:in_end_z
                                                ] = sib_mask_translated[
                                                    out_start_x:out_end_x,
                                                    out_start_y:out_end_y,
                                                    out_start_z:out_end_z
                                                ]
                                                sib_mask_transformed = sib_mask_canvas
                                            elif uses_scipy_centroid_embed_rigid(method):
                                                if scaling and scale_match != 1:
                                                    sib_mask_data = ndimage.zoom(sib_mask_data, scale_match, order=0, mode='constant', cval=0)
                                                sib_mask_padded = np.pad(
                                                    sib_mask_data,
                                                    ((padding_low[0], padding_high[0]),
                                                     (padding_low[1], padding_high[1]),
                                                     (padding_low[2], padding_high[2])),
                                                    mode='constant',
                                                    constant_values=0
                                                )
                                                sib_mask_rotated = ndimage.affine_transform(
                                                    sib_mask_padded,
                                                    rotation,
                                                    offset=offset,
                                                    output_shape=sib_mask_padded.shape,
                                                    order=0,
                                                    mode='constant',
                                                    cval=0
                                                )
                                                residual = np.asarray(
                                                    centroid_translation if centroid_translation is not None else [0, 0, 0],
                                                    dtype=np.float64,
                                                ).reshape(3)
                                                if np.any(residual):
                                                    sib_mask_rotated = ndimage.shift(
                                                        sib_mask_rotated,
                                                        shift=residual,
                                                        order=0,
                                                        mode='constant',
                                                        cval=0,
                                                    )
                                                sib_mask_canvas = np.zeros_like(reference_data, dtype=sib_mask_dtype)
                                                sib_mask_canvas[
                                                    in_start_x:in_end_x,
                                                    in_start_y:in_end_y,
                                                    in_start_z:in_end_z
                                                ] = sib_mask_rotated[
                                                    out_start_x:out_end_x,
                                                    out_start_y:out_end_y,
                                                    out_start_z:out_end_z
                                                ]
                                                sib_mask_transformed = sib_mask_canvas
                                            
                                            # Save sibling aligned mask
                                            sib_mask_edit_base = f"{sibling_name}_edit_{sib_new_edit}_aligned"
                                            sib_mask_dest_path = os.path.join(directory, "extracted", sibling_name, f"{sib_mask_edit_base}.nii.mask.gz")
                                            sib_mask_dest_temp = os.path.join(directory, "extracted", sibling_name, f"{sib_mask_edit_base}_mask.nii.gz")
                                            sib_mask_vol_to_save = self._volume_for_mesh_reference_save(sib_mask_transformed, reference_is_preserved_mesh)
                                            nib.save(
                                                nib.Nifti1Image(sib_mask_vol_to_save.astype(sib_mask_dtype), reference_affine_data),
                                                sib_mask_dest_temp
                                            )
                                            os.replace(sib_mask_dest_temp, sib_mask_dest_path)
                                            
                                            # Create lossy mask if needed
                                            if 'lossy_compression' in sib_metadata:
                                                resolution_factor = 2
                                                lc = sib_metadata.get('lossy_compression', None)
                                                if isinstance(lc, list) and len(lc) > 0:
                                                    try:
                                                        resolution_factor = int(lc[-1].get('resolution_factor', resolution_factor))
                                                    except Exception:
                                                        pass
                                                elif isinstance(lc, dict) and lc:
                                                    try:
                                                        resolution_factor = int(lc.get('resolution_factor', resolution_factor))
                                                    except Exception:
                                                        pass
                                                slices = [slice(None, None, resolution_factor) for _ in range(3)]
                                                sib_lossy_mask = sib_mask_vol_to_save[slices[0], slices[1], slices[2]].astype(sib_mask_dtype, copy=False)
                                                sib_lossy_mask_edit_base = f"{sibling_name}_lossy_edit_{sib_new_edit}_aligned"
                                                sib_lossy_mask_dest = os.path.join(directory, "extracted", sibling_name, f"{sib_lossy_mask_edit_base}.nii.mask.gz")
                                                sib_lossy_mask_temp = os.path.join(directory, "extracted", sibling_name, f"{sib_lossy_mask_edit_base}_mask.nii.gz")
                                                nib.save(nib.Nifti1Image(sib_lossy_mask, reference_affine_data), sib_lossy_mask_temp)
                                                os.replace(sib_lossy_mask_temp, sib_lossy_mask_dest)
                                        finally:
                                            os.replace(sib_mask_temp, sib_mask_src)
                                    
                                    linked_propagation_results.append({
                                        'scan_name': sibling_name,
                                        'status': 'success',
                                        'edit': sib_new_edit
                                    })
                                    print(f"  Successfully propagated alignment to {sibling_name} (edit {sib_new_edit})")
                                    cleanup_memory()
                                    
                                except Exception as sib_e:
                                    linked_propagation_results.append({
                                        'scan_name': sibling_name,
                                        'status': 'failed',
                                        'reason': str(sib_e)
                                    })
                                    print(f"  Error propagating to sibling {sibling_name}: {sib_e}")
                    except Exception as prop_e:
                        print(f"Error in linked propagation for {scan_name}: {prop_e}")

                # Send progress update after each scan is processed
                channel_layer = get_channel_layer()
                if channel_layer is not None:
                    progress = ((idx+(7/9)) / total_scans)
                    async_to_sync(channel_layer.group_send)(
                        'progress_group',
                        {
                            'type': 'send_progress',
                            'progress': progress,
                            'scan_name': scan_name,
                            'custom_message': f'Alignment complete for {scan_name}.',
                            'total': total_scans,
                            'current': idx + 1,
                        }
                    )
                    print("Alignment complete for ", scan_name)
                
                # Check available memory
                remaining_memory = psutil.virtual_memory().available / 1024 / 1024 / 1024
                print(f"Available memory: {remaining_memory:.2f} GB")

            except Exception as e:
                print(f"Error processing scan {scan_name}: {str(e)}")
                continue
        
        # Send progress update after each scan is processed
        channel_layer = get_channel_layer()
        if channel_layer is not None:
            progress = 1
            async_to_sync(channel_layer.group_send)(
                'progress_group',
                {
                    'type': 'send_progress',
                    'progress': progress,
                    'scan_name': '',
                    'custom_message': f'Alignment complete for all scans.',
                    'total': total_scans,
                    'current': total_scans,
                }
            )

        return Response({
            'message': f'Alignment complete using {method} method',
            'linked_propagation': linked_propagation_results if propagate_linked else None
        }, status=status.HTTP_200_OK)


class InvertAlignmentLandmarksView(APIView):
    """
    Inverts landmarks placed on rigidly-aligned scans back to the
    pre-alignment coordinate space (physical mm at reference voxel spacing).

    Writes inverted landmarks onto the pre-alignment edit/raw landmark slot
    (``{scan}_landmarks.json`` when rigid is edit 0, otherwise the edit before
    ``_aligned``) so they can be viewed in the UI and exported as CSV.

    All alignment methods (ALPACA, guidepoints) resample to reference voxel
    spacing before rotation, so the inverted output is uniformly in
    reference-spacing mm — suitable for cross-subject morphometric comparison.

    POST body:
        directory      (str)        – project root
        scan_names     (list, opt)  – subset of subject names; all if omitted
        flagFilter     (str, opt)   – off | exclude | only (metadata ``flag``)
        overwriteExistingLandmarks (bool, opt) – if False, skip subjects whose
                                      pre-alignment edit/raw already has landmarks.

    Subjects with faulty=true in metadata are skipped (explicit scan_names receive status skipped).
    """

    def post(self, request):
        directory     = request.data['directory']
        scan_names    = request.data.get('scan_names', None)
        flag_filter = normalize_flag_filter_value(
            request.data.get('flagFilter', 'off') if request.data else 'off',
        )
        overwrite_existing = request.data.get('overwriteExistingLandmarks', True)
        if isinstance(overwrite_existing, str):
            overwrite_existing = overwrite_existing.lower() in ('1', 'true', 'yes')

        extracted_path = os.path.join(directory, 'extracted')
        reference_subject = None

        results = {}
        if scan_names:
            subjects = []
            for s in scan_names:
                if _extracted_scan_marked_faulty(directory, s):
                    results[s] = {'status': 'skipped', 'reason': 'marked as faulty in metadata'}
                    print(f"[InvertAlignment] {s}: skipped (marked as faulty in metadata)")
                else:
                    subjects.append(s)
                if _is_reference_subject(directory, s):
                    reference_subject = s
        else:
            subjects = []
            for d in os.listdir(extracted_path):
                if not os.path.isdir(os.path.join(extracted_path, d)) or d == 'project_settings.json':
                    continue
                if _is_reference_subject(directory, d):
                    reference_subject = d
                if _extracted_scan_marked_faulty(directory, d):
                    if d == reference_subject:
                        print(f"[InvertAlignment] {d}: reference subject kept despite faulty flag")
                    else:
                        print(f"[InvertAlignment] {d}: skipped (marked as faulty in metadata)")
                        continue
                subjects.append(d)

        if reference_subject and reference_subject not in subjects:
            subjects.append(reference_subject)
            print(f"[InvertAlignment] {reference_subject}: added as reference subject")

        flagged_set = load_flagged_subject_names(directory)
        subjects = apply_flag_filter(subjects, flagged_set, flag_filter)
        subjects = filter_out_linked_children(directory, subjects)
        print(f"[InvertAlignment] flag filter: {flag_filter}, remaining subjects: {len(subjects)}")

        total_scans = len(subjects)
        channel_layer = get_channel_layer()
        if channel_layer is not None and total_scans > 0:
            async_to_sync(channel_layer.group_send)(
                'progress_group',
                {
                    'type': 'send_progress',
                    'progress': 0,
                    'scan_name': subjects[0],
                    'custom_message': 'Preparing to invert landmarks to pre-alignment space...',
                    'total': total_scans,
                    'current': 0,
                }
            )

        for idx, scan_name in enumerate(subjects):
            if channel_layer is not None and total_scans > 0:
                progress = idx / total_scans
                async_to_sync(channel_layer.group_send)(
                    'progress_group',
                    {
                        'type': 'send_progress',
                        'progress': progress,
                        'scan_name': scan_name,
                        'custom_message': (
                            f'Inverting landmarks to pre-alignment for {scan_name} '
                            f'({idx + 1}/{total_scans})...'
                        ),
                        'total': total_scans,
                        'current': idx + 1,
                    }
                )
            try:
                result = self._process_subject(
                    directory, extracted_path, scan_name,
                    overwrite_existing=overwrite_existing,
                )
                results[scan_name] = result
            except Exception as e:
                results[scan_name] = {'error': str(e)}
                print(f"Error processing {scan_name}: {e}")

        if channel_layer is not None and total_scans > 0:
            async_to_sync(channel_layer.group_send)(
                'progress_group',
                {
                    'type': 'send_progress',
                    'progress': 1,
                    'scan_name': subjects[-1] if subjects else '',
                    'custom_message': 'Pre-alignment landmark transfer complete.',
                    'total': total_scans,
                    'current': total_scans,
                }
            )

        return Response({'results': results}, status=status.HTTP_200_OK)

    def _write_pre_alignment_landmarks(
        self, scan_dir, scan_name, landmarks_mm, overwrite_existing,
    ):
        """
        Write inverted landmarks onto the pre-alignment edit/raw slot.
        Ensures per-edit thresholds exist, then writes landmark distances for
        that pre-alignment stem using its stored threshold.
        Returns status metadata for the response payload.
        """
        stem = resolve_pre_alignment_landmark_stem(scan_dir, scan_name)
        attach_path = os.path.join(scan_dir, f'{stem}_landmarks.json')
        if os.path.exists(attach_path) and not overwrite_existing:
            print(
                f"[InvertAlignment] {scan_name}: pre-alignment landmarks already exist "
                f"({os.path.basename(attach_path)}), skipping (overwrite disabled)"
            )
            return {
                'status': 'skipped',
                'reason': 'pre-alignment landmarks already exist',
                'attach_target': os.path.basename(attach_path),
                'attached': False,
                'attach_skipped': True,
            }

        with open(attach_path, 'w') as f:
            json.dump(landmarks_mm, f, indent=4)
        print(
            f"[InvertAlignment] {scan_name}: wrote {len(landmarks_mm)} landmarks "
            f"→ {os.path.basename(attach_path)}"
        )

        result = {
            'attached': True,
            'attach_skipped': False,
            'attach_target': os.path.basename(attach_path),
            'output': os.path.basename(attach_path),
            'pre_alignment_stem': stem,
        }

        json_path = os.path.join(scan_dir, f'{scan_name}.json')
        try:
            with open(json_path, 'r') as jf:
                metadata = json.load(jf)
            ensure_old_thresholds(metadata, scan_dir, scan_name)
            with open(json_path, 'w') as jf:
                json.dump(metadata, jf, indent=4)

            distance_info = compute_and_save_landmark_distances(
                scan_dir,
                scan_name,
                stem,
                landmarks_mm,
                snap_distance=None,
                update_landmarks_metadata=False,
                metadata=metadata,
                json_path=json_path,
            )
            result['landmark_distances'] = distance_info
        except Exception as e:
            print(
                f"[InvertAlignment] {scan_name}: landmark distance analysis failed "
                f"for {stem}: {e}"
            )
            result['landmark_distances_error'] = str(e)

        return result

    def _process_subject(
        self, directory, extracted_path, scan_name,
        overwrite_existing=True,
    ):
        """Process a single subject: find latest pre-elastic landmarks and invert them."""
        def _skip(reason):
            print(f"[InvertAlignment] {scan_name}: skipped ({reason})")
            return {'status': 'skipped', 'reason': reason}

        scan_dir  = os.path.join(extracted_path, scan_name)
        json_path = os.path.join(scan_dir, f'{scan_name}.json')

        if not os.path.isfile(json_path):
            return _skip('no JSON metadata')

        # Respect overwrite guard before doing expensive inversion work.
        pre_stem = resolve_pre_alignment_landmark_stem(scan_dir, scan_name)
        pre_landmarks_path = os.path.join(scan_dir, f'{pre_stem}_landmarks.json')
        if os.path.exists(pre_landmarks_path) and not overwrite_existing:
            return _skip('pre-alignment landmarks already exist')

        with open(json_path, 'r') as f:
            metadata = json.load(f)

        if _is_reference_subject(directory, scan_name):
            all_landmark_files = [
                lf for lf in glob.glob(
                    os.path.join(scan_dir, f'{scan_name}_edit_*_landmarks.json')
                )
                if '_landmark_distances' not in os.path.basename(lf)
            ]
            valid_candidates = []
            for lf in all_landmark_files:
                m = re.search(r'_edit_(\d+)_[^.]+_landmarks\.json$', os.path.basename(lf))
                if not m:
                    continue
                valid_candidates.append((int(m.group(1)), lf))

            if not valid_candidates:
                return _skip('reference scan has no landmark files from edits')

            valid_candidates.sort(key=lambda x: x[0])
            latest_edit_num, latest_landmark_path = valid_candidates[-1]

            with open(latest_landmark_path, 'r') as f:
                landmarks_mm = json.load(f)
            if not landmarks_mm:
                return _skip('reference landmark file is empty')

            write_info = self._write_pre_alignment_landmarks(
                scan_dir, scan_name, landmarks_mm, overwrite_existing,
            )
            if write_info.get('status') == 'skipped':
                return write_info

            print(f"[InvertAlignment] {scan_name}: reference scan copied {len(landmarks_mm)} landmarks "
                  f"from edit {latest_edit_num} → {write_info.get('output')}")
            return {
                'status': 'ok',
                'source_landmark_file': os.path.basename(latest_landmark_path),
                'source_edit': latest_edit_num,
                'alignment_edit': None,
                'elastic_boundary_edit': None,
                'n_landmarks': len(landmarks_mm),
                'reference_copied': True,
                **write_info,
            }

        params = metadata.get('alignment_calculated_parameters')
        if not params:
            return _skip('no alignment_calculated_parameters')

        # --- Find all landmark files (excluding distance sidecars) ---
        all_landmark_files = [
            lf for lf in glob.glob(
                os.path.join(scan_dir, f'{scan_name}_edit_*_landmarks.json')
            )
            if '_landmark_distances' not in os.path.basename(lf)
        ]

        if not all_landmark_files:
            return _skip('no landmark files found')

        # --- Detect earliest elastic edit number, if present ---
        # Check both nifti and landmarks variants to stay robust after cleanup.
        elastic_edit_nums = set()
        elastic_niftis = glob.glob(
            os.path.join(scan_dir, f'{scan_name}_edit_*_elastic.nii.gz')
        )
        for ef in elastic_niftis:
            m = re.search(r'_edit_(\d+)_elastic\.nii\.gz$', os.path.basename(ef))
            if m:
                elastic_edit_nums.add(int(m.group(1)))

        for lf in all_landmark_files:
            m = re.search(r'_edit_(\d+)_elastic_landmarks\.json$', os.path.basename(lf))
            if m:
                elastic_edit_nums.add(int(m.group(1)))

        min_elastic_edit = min(elastic_edit_nums) if elastic_edit_nums else float('inf')

        # --- Pick latest landmark file before elastic edits ---
        valid_candidates = []
        for lf in all_landmark_files:
            m = re.search(r'_edit_(\d+)_[^.]+_landmarks\.json$', os.path.basename(lf))
            if not m:
                continue
            edit_num = int(m.group(1))
            if edit_num < min_elastic_edit:
                valid_candidates.append((edit_num, lf))

        if not valid_candidates:
            return _skip('no valid landmark files found before elastic edits')

        valid_candidates.sort(key=lambda x: x[0])
        latest_edit_num, latest_landmark_path = valid_candidates[-1]

        with open(latest_landmark_path, 'r') as f:
            landmarks_mm = json.load(f)

        if not landmarks_mm:
            return _skip('landmark file is empty')

        # --- Apply inverse transform ---
        inverted_mm = self._invert_landmarks(
            landmarks_mm, params, metadata,
            scan_dir=scan_dir,
            scan_name=scan_name,
            alignment_method=metadata.get('alignment_method'),
        )

        write_info = self._write_pre_alignment_landmarks(
            scan_dir, scan_name, inverted_mm, overwrite_existing,
        )
        if write_info.get('status') == 'skipped':
            return write_info

        print(f"[InvertAlignment] {scan_name}: {len(inverted_mm)} landmarks "
              f"inverted from edit {latest_edit_num} → {write_info.get('output')}")

        return {
            'status': 'ok',
            'source_landmark_file': os.path.basename(latest_landmark_path),
            'source_edit': latest_edit_num,
            'alignment_edit': None,
            'elastic_boundary_edit': min_elastic_edit if min_elastic_edit != float('inf') else None,
            'n_landmarks': len(inverted_mm),
            **write_info,
        }

    def _resolve_pre_alignment_volume_path(self, scan_dir, scan_name):
        """Prefer full-resolution pre-alignment nifti over lossy."""
        for name in (f'{scan_name}.nii.gz', f'{scan_name}_lossy.nii.gz'):
            path = os.path.join(scan_dir, name)
            if os.path.isfile(path):
                return path
        return None

    def _resolve_aligned_volume_path(self, scan_dir, scan_name):
        """Earliest rigid ``_aligned`` edit, preferring non-lossy."""
        candidates = []
        for path in glob.glob(os.path.join(scan_dir, f'{scan_name}_edit_*_aligned.nii.gz')):
            base = os.path.basename(path)
            m = re.search(r'_edit_(\d+)_aligned\.nii\.gz$', base.replace('_lossy', ''))
            if not m:
                continue
            is_lossy = '_lossy_' in base
            candidates.append((int(m.group(1)), is_lossy, path))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates[0][2]

    def _recover_alignment_translation_voxel(self, scan_dir, scan_name, params, metadata):
        """
        Recover ALPACA residual translation when it was not persisted.

        Replays the known inverse mapping for many aligned-canvas FG samples and
        maximises intensity correlation with the pre-alignment volume:
            padded = R.T @ (canvas - rotation_center - integer_shift - T) + rotation_center
            unpadded = padded - padding_low
        """
        from scipy.optimize import minimize
        from scipy.ndimage import map_coordinates, zoom as nd_zoom

        pre_path = self._resolve_pre_alignment_volume_path(scan_dir, scan_name)
        aln_path = self._resolve_aligned_volume_path(scan_dir, scan_name)
        if not pre_path or not aln_path:
            print(
                f"[InvertAlignment] {scan_name}: cannot recover translation "
                f"(pre={bool(pre_path)}, aligned={bool(aln_path)})"
            )
            return np.zeros(3, dtype=float)

        R = np.array(params['alignment_rotation_matrix'], dtype=float)
        rotation_center = np.array(params['alignment_rotation_center_voxel'], dtype=float)
        canvas_centroid = np.array(params['alignment_canvas_centroid_voxel'], dtype=float)
        padding_low = np.array(params.get('alignment_padding_low') or [0, 0, 0], dtype=float)
        scale_match = float(metadata.get('alignment_scale') or 1.0)

        print(
            f"[InvertAlignment] {scan_name}: recovering alignment_translation_voxel "
            f"from {os.path.basename(pre_path)} ↔ {os.path.basename(aln_path)}"
        )

        pre = nib.load(pre_path).get_fdata(dtype=np.float32)
        aln = nib.load(aln_path).get_fdata(dtype=np.float32)

        # Lossy pre-alignment is often 2× downsampled vs stored voxel params.
        pre_zoom = float(nib.load(pre_path).header.get_zooms()[0])
        ref_vs = float(metadata.get('voxel_size') or pre_zoom)
        grid_scale = pre_zoom / ref_vs if ref_vs > 0 else 1.0
        if abs(grid_scale - 1.0) > 1e-3:
            # Map full-res voxel indices into the loaded grid.
            index_scale = 1.0 / grid_scale
        else:
            index_scale = 1.0

        if scale_match != 1.0:
            pre = nd_zoom(pre, scale_match, order=1, mode='constant', cval=float(np.percentile(pre, 5)))

        thr_a = np.percentile(aln, 90)
        fg_coords = np.argwhere(aln > thr_a)
        if len(fg_coords) < 500:
            thr_a = np.percentile(aln, 80)
            fg_coords = np.argwhere(aln > thr_a)
        if len(fg_coords) < 100:
            print(f"[InvertAlignment] {scan_name}: too few aligned FG voxels to recover translation")
            return np.zeros(3, dtype=float)

        rng = np.random.default_rng(0)
        sample_n = min(12000, len(fg_coords))
        coords = fg_coords[rng.choice(len(fg_coords), size=sample_n, replace=False)].astype(np.float64)
        vals = aln[
            coords[:, 0].astype(int),
            coords[:, 1].astype(int),
            coords[:, 2].astype(int),
        ].astype(np.float64)
        canvas_shift = canvas_embed_shift_from_crop_or_centroids(
            params, rotation_center, canvas_centroid
        )
        bg = float(np.percentile(pre, 5))

        def corr_for_T(translation):
            translation = np.asarray(translation, dtype=float).reshape(3)
            padded = (
                R.T @ (coords - rotation_center - canvas_shift - translation).T
            ).T + rotation_center
            unpadded = (padded - padding_low) * index_scale
            pred = map_coordinates(pre, unpadded.T, order=1, mode='constant', cval=bg)
            if pred.std() < 1e-6 or vals.std() < 1e-6:
                return -1.0
            return float(np.corrcoef(pred, vals)[0, 1])

        guesses = [
            np.zeros(3, dtype=float),
            np.asarray(canvas_centroid - rotation_center, dtype=float),
        ]
        best_corr = -1.0
        best_T = np.zeros(3, dtype=float)
        for guess in guesses:
            # Coarse local grid around each guess
            for dx in range(-24, 25, 8):
                for dy in range(-24, 25, 8):
                    for dz in range(-24, 25, 8):
                        T = guess + np.array([dx, dy, dz], dtype=float)
                        c = corr_for_T(T)
                        if c > best_corr:
                            best_corr = c
                            best_T = T.copy()

        # Continuous refinement
        result = minimize(
            lambda T: -corr_for_T(T),
            x0=best_T,
            method='Nelder-Mead',
            options={'xatol': 0.05, 'fatol': 1e-5, 'maxiter': 200},
        )
        if result.success or np.isfinite(result.fun):
            refined = np.asarray(result.x, dtype=float).reshape(3)
            refined_corr = corr_for_T(refined)
            if refined_corr >= best_corr:
                best_T = refined
                best_corr = refined_corr

        print(
            f"[InvertAlignment] {scan_name}: recovered translation={best_T.tolist()} "
            f"(corr={best_corr:.4f})"
        )
        if best_corr < 0.2:
            print(
                f"[InvertAlignment] {scan_name}: low recovery correlation — "
                f"translation estimate may be unreliable"
            )
        return best_T

    def _invert_landmarks(
        self, landmarks_mm, params, metadata,
        scan_dir=None, scan_name=None, alignment_method=None,
    ):
        """
        Invert landmarks from aligned-canvas mm back to pre-alignment mm.

        Forward (ALPACA / guidepoints after resample), in voxel units::

            padded = unpadded * scale_match + padding_low
            rotated = R @ (padded - rotation_center) + rotation_center + translation
            canvas  = rotated + integer_shift

        ``integer_shift`` is ``int(canvas_centroid) - int(rotation_center)``,
        matching the volume's integer in/out crop — not the continuous
        ``- rotation_center + canvas_centroid`` snap.

        So the inverse is::

            canvas_voxel  = aligned_mm / ref_vs
            rotated_voxel = canvas_voxel - integer_shift
            padded_voxel  = R.T @ (rotated_voxel - rotation_center - translation) + rotation_center
            unpadded_voxel = padded_voxel - padding_low
            pre_scale_voxel = unpadded_voxel / scale_match
            output_mm     = pre_scale_voxel * ref_vs

        ``translation`` is ALPACA's residual shift after rotate-about-centroid.
        Guidepoints encode the centroid snap in the integer canvas paste,
        so translation is zero for that method.

        ANTs/GPU-rigid projects aligned before scipy unification may still store
        ``alignment_ants_affine``; see ``_invert_landmarks_ants``.

        When ``alignment_translation_voxel`` is missing (older projects), it is
        recovered by intensity-matching the aligned volume against the
        pre-alignment volume under the equations above, then persisted.
        """
        method = (alignment_method or metadata.get('alignment_method') or '').lower()
        if has_legacy_ants_affine_metadata(params):
            return self._invert_landmarks_ants(
                landmarks_mm, params, metadata, scan_dir=scan_dir, scan_name=scan_name
            )

        R               = np.array(params['alignment_rotation_matrix'], dtype=float)
        rotation_center = np.array(params['alignment_rotation_center_voxel'], dtype=float)
        canvas_centroid = np.array(params['alignment_canvas_centroid_voxel'], dtype=float)
        padding_low     = np.array(params['alignment_padding_low'], dtype=float)

        reference_vs    = float(metadata['voxel_size'])
        scale_match     = float(metadata.get('alignment_scale') or 1.0)

        translation = None
        recovered = False
        if 'alignment_translation_voxel' in params and params['alignment_translation_voxel'] is not None:
            translation = np.array(params['alignment_translation_voxel'], dtype=float)
        elif method == 'alpaca' and scan_dir and scan_name:
            # Missing on older ALPACA projects — recover from before/after volumes.
            translation = self._recover_alignment_translation_voxel(
                scan_dir, scan_name, params, metadata,
            )
            recovered = True
        else:
            translation = np.zeros(3, dtype=float)

        if recovered and scan_dir and scan_name:
            # Persist so later inversions / exports reuse the same T.
            try:
                json_path = os.path.join(scan_dir, f'{scan_name}.json')
                with open(json_path, 'r') as jf:
                    meta_out = json.load(jf)
                acp = dict(meta_out.get('alignment_calculated_parameters') or params)
                acp['alignment_translation_voxel'] = translation.tolist()
                acp['alignment_translation_recovered'] = True
                meta_out['alignment_calculated_parameters'] = acp
                with open(json_path, 'w') as jf:
                    json.dump(meta_out, jf, indent=4)
                print(
                    f"[InvertAlignment] {scan_name}: persisted recovered "
                    f"alignment_translation_voxel to metadata"
                )
            except Exception as exc:
                print(f"[InvertAlignment] {scan_name}: could not persist translation: {exc}")

        inverted = []
        for lm_mm in landmarks_mm:
            landmark_type = 'main'
            if isinstance(lm_mm, dict):
                landmark_type = lm_mm.get('landmark_type', 'main')
                lm = np.array(lm_mm['position'], dtype=float)
            else:
                lm = np.array(lm_mm, dtype=float)

            canvas_voxel   = lm / reference_vs
            canvas_shift   = canvas_embed_shift_from_crop_or_centroids(
                params, rotation_center, canvas_centroid
            )
            padded_voxel   = (
                R.T @ (canvas_voxel - rotation_center - translation - canvas_shift)
                + rotation_center
            )
            unpadded_voxel = padded_voxel - padding_low

            if scale_match != 1.0:
                unpadded_voxel = unpadded_voxel / scale_match

            # Post-resample indices are mm/reference_vs, so this restores native mm.
            output_mm = unpadded_voxel * reference_vs
            inverted.append({
                'position': output_mm.tolist(),
                'landmark_type': landmark_type
            })

        return inverted

    def _invert_landmarks_ants(
        self, landmarks_mm, params, metadata, scan_dir=None, scan_name=None,
    ):
        """
        Inverse of ``_transform_landmarks_ants``:

            canvas_mm → ants physical → GenericAffine (fixed→moving) →
            undo mov_paste (or legacy centroid shift) → undo scale → native mm.
        """
        reference_vs = float(metadata['voxel_size'])
        scale_match = float(metadata.get('alignment_scale') or 1.0)

        mov_paste_raw = params.get('alignment_ants_mov_paste')
        if mov_paste_raw is not None:
            mov_paste = np.asarray(mov_paste_raw, dtype=np.float64).reshape(3)
        else:
            centroid_shift = params.get('alignment_centroid_shift_voxel')
            if centroid_shift is None:
                centroid_shift = params.get('alignment_translation_voxel') or [0, 0, 0]
            mov_paste = np.array([int(v) for v in centroid_shift], dtype=np.float64)

        spacing = params.get('alignment_ants_spacing') or (reference_vs, reference_vs, reference_vs)
        origin = params.get('alignment_ants_origin') or (0.0, 0.0, 0.0)
        direction = params.get('alignment_ants_direction') or np.eye(3).tolist()
        union_ref_paste = np.asarray(
            params.get('alignment_ants_union_ref_paste') or [0, 0, 0],
            dtype=np.float64,
        ).reshape(3)

        affine_name = params.get('alignment_ants_affine') or (
            f"{scan_name}_ants_rigid_affine.mat" if scan_name else None
        )
        if not scan_dir or not affine_name:
            raise ValueError("ANTs invert requires scan_dir and alignment_ants_affine")
        affine_path = os.path.join(scan_dir, os.path.basename(affine_name))
        fixed_to_moving = ants.read_transform(affine_path)

        print(
            f"[InvertAlignment] {scan_name}: ANTs invert via {os.path.basename(affine_path)} "
            f"(mov_paste={mov_paste.tolist()}, scale={scale_match})"
        )

        inverted = []
        for lm_mm in landmarks_mm:
            landmark_mm, landmark_type = unpack_landmark_entry(lm_mm)
            canvas_voxel = np.asarray(landmark_mm, dtype=np.float64) / reference_vs
            canvas_voxel_union = canvas_voxel + union_ref_paste
            physical_fixed = AlignToReferenceView._ants_numpy_index_to_physical(
                canvas_voxel_union, spacing, origin, direction
            )
            physical_moving = np.asarray(
                fixed_to_moving.apply_to_point(physical_fixed.tolist()),
                dtype=np.float64,
            )
            moved_voxel = AlignToReferenceView._ants_physical_to_numpy_index(
                physical_moving, spacing, origin, direction
            )
            unpasted = moved_voxel - mov_paste
            if scale_match != 1.0:
                unpasted = unpasted / scale_match
            output_mm = unpasted * reference_vs
            inverted.append({
                'position': output_mm.tolist(),
                'landmark_type': landmark_type,
            })
        return inverted


class ExportInvertedLandmarksView(APIView):
    """
    Exports pre-alignment landmarks for all subjects as a CSV.

    Reads landmarks from the edit/raw slot that came before rigid alignment
    (see ``resolve_pre_alignment_landmark_stem``), not from a sidecar file.

    POST body:
        directory      (str)       – project root
    """

    def post(self, request):
        try:
            directory = request.data.get('directory')

            if not directory:
                return Response(
                    {'error': 'No directory provided'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            extracted_dir = os.path.join(directory, 'extracted')
            if not os.path.exists(extracted_dir):
                return Response(
                    {'error': 'No extracted directory found'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            scans = sorted(
                [d for d in os.listdir(extracted_dir)
                 if os.path.isdir(os.path.join(extracted_dir, d))],
                key=lambda name: name.lower()
            )

            print(f"Exporting pre-alignment landmarks from {len(scans)} scans")

            all_data = []
            for scan_name in scans:
                sub_path      = os.path.join(extracted_dir, scan_name)
                metadata_path = os.path.join(sub_path, f'{scan_name}.json')

                if not os.path.exists(metadata_path):
                    print(f"Skipping {scan_name}: no metadata found")
                    continue

                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    if metadata.get('faulty', False):
                        print(f"Skipping {scan_name}: marked as faulty")
                        continue
                except (json.JSONDecodeError, FileNotFoundError):
                    print(f"Skipping {scan_name}: could not read metadata")
                    continue

                stem = resolve_pre_alignment_landmark_stem(sub_path, scan_name)
                landmarks_path = os.path.join(sub_path, f'{stem}_landmarks.json')
                if not os.path.exists(landmarks_path):
                    print(
                        f"Skipping {scan_name}: no pre-alignment landmarks "
                        f"({os.path.basename(landmarks_path)})"
                    )
                    continue

                try:
                    with open(landmarks_path, 'r') as f:
                        landmarks = json.load(f)

                    landmarks_dict = {}
                    for i, lm in enumerate(landmarks):
                        if isinstance(lm, dict) and 'position' in lm:
                            prefix = 'semi_' if lm.get('landmark_type') == 'semi' else ''
                            landmarks_dict[f'{prefix}landmark_{i}'] = lm['position']
                        else:
                            landmarks_dict[f'landmark_{i}'] = lm

                    if landmarks_dict:
                        row = {'Subject': scan_name}
                        for lm_name, coord in landmarks_dict.items():
                            if isinstance(coord, (list, tuple)) and len(coord) >= 3:
                                row[f'{lm_name}_x'] = coord[0]
                                row[f'{lm_name}_y'] = coord[1]
                                row[f'{lm_name}_z'] = coord[2]
                        all_data.append(row)
                        print(
                            f"Loaded pre-alignment landmarks from {scan_name} "
                            f"({os.path.basename(landmarks_path)}): "
                            f"{len(landmarks_dict)} landmarks"
                        )

                except (json.JSONDecodeError, FileNotFoundError):
                    print(f"Could not read landmarks file: {landmarks_path}")
                    continue

            print(f"Collected pre-alignment landmarks from {len(all_data)} scans")

            if not all_data:
                return Response(
                    {'error': 'No pre-alignment landmarks found in any scans'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            df = pd.DataFrame(all_data)
            if 'Subject' in df.columns:
                df = df.sort_values(by='Subject', key=lambda col: col.str.lower()).reset_index(drop=True)
            df = df.fillna('')

            print(f"DataFrame shape: {df.shape}")

            temp_dir = os.path.join(directory, '.temp')
            os.makedirs(temp_dir, exist_ok=True)

            csv_filename = f"pre_alignment_landmarks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            csv_path = os.path.join(temp_dir, csv_filename)

            print(f"Writing CSV to temporary file: {csv_path}")
            df.to_csv(csv_path, index=False)

            print(f"Sending CSV file to client")
            response = FileResponse(
                open(csv_path, 'rb'),
                as_attachment=True,
                filename=csv_filename,
                content_type='text/csv'
            )
            response['Content-Length'] = os.path.getsize(csv_path)

            return response

        except Exception as e:
            error_message = f"Error exporting inverted landmarks: {str(e)}"
            print(error_message)
            import traceback
            print(traceback.format_exc())
            return Response(
                {'error': error_message},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        