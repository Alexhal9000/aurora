from django.shortcuts import render
from rest_framework import viewsets
from rest_framework.views import APIView
from rest_framework.response import Response
from django.http import FileResponse
from rest_framework.parsers import MultiPartParser, FormParser
import json
import os
from rest_framework import status
import shutil
import glob
from urllib.parse import unquote
import re
import numpy as np
import nibabel as nib
import imageio
import aim2numpy
from scipy.ndimage import gaussian_filter, zoom
from PIL import Image
import concurrent.futures
import io
import time
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
import trimesh
import base64
import tempfile
import skimage.measure as measure
import skimage.transform as transform
import pygltflib
from scipy import ndimage
from scipy import signal
import sys
import psutil
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import spatial
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu, butterworth, median, threshold_li
import skimage.morphology
from scipy.signal import convolve
from .ALPACA import ALPACA
from .batch_flag_filter import load_flagged_subject_names, apply_flag_filter, normalize_flag_filter_value
from .linkedScans import filter_out_linked_children, get_same_shape_siblings, full_res_nifti_path
from skimage import exposure
# Set the number of threads for ITK to utilize
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(psutil.cpu_count(logical=True))
import ants
from scipy.interpolate import RegularGridInterpolator
import open3d as o3d
import fast_simplification
import gc  # Add garbage collection
from pydicom import dcmread
from scipy.stats import entropy
from skimage.restoration import (
    denoise_nl_means, denoise_tv_chambolle, denoise_tv_bregman,
    denoise_wavelet, estimate_sigma
)
from skimage.exposure import match_histograms 
import multiprocessing
from multiprocessing import shared_memory
import math
import threading
from multiprocessing import Manager
from .meshGridTools import patch_subject_json
from .registrationTools import RegistrationTools
from .coordinateFrames import (
    is_preserved_mesh_metadata,
    load_extracted_scan_metadata,
    partition_voxel_and_preserved_mesh_scans,
    voxel_only_all_meshes_message,
    voxel_only_no_eligible_targets_message,
)
from .meshBasedTools import ApplyMeshCleanupView

# Summary of classes:
# CleanupMeshView: Used to clean up the mesh by removing small disconnected components and trying to separate connected objects
# HomogenizeBackgroundView: Used to homogenize the background of all scans to a single value
# ApplyThresholdView: Applies threshold-based processing to scans using different methods (same-threshold, Otsu's method, histogram matching, percentile matching)
# RemoveBackgroundView: Removes background from scans by setting all values below the threshold to the minimum value
# MatchHistogramView: Matches histogram/intensity distributions between scans using various methods (full histogram matching, peak alignment, tissue-aware normalization, percentile normalization)
# BatchCleanupMeshView: Applies mesh cleanup to multiple scans in batch mode with progress tracking
# N4BiasCorrectionView: N4 bias field correction (ANTs n4_bias_field_correction) for intensity inhomogeneity
# SaveBackgroundValuesView: Saves background correction values and applies background homogenization to individual scans
# SaveThresholdValuesView: Saves threshold values to scan metadata files


def cleanup_memory():
    # --- memory cleanup ---
    def get_var_size(var):
        if isinstance(var, np.ndarray):
            return var.nbytes
        return sys.getsizeof(var)

    def force_memory_cleanup():
        # Force multiple garbage collection passes
        for _ in range(3):
            gc.collect()
        
        # Try to release memory back to the system
        if hasattr(gc, 'collect'):
            gc.collect()
        if hasattr(os, 'sync'):
            os.sync()
        
        # On Linux systems, try to release memory more aggressively
        if sys.platform.startswith('linux'):
            try:
                with open('/proc/sys/vm/drop_caches', 'w') as f:
                    f.write('1')
            except:
                pass

        # Sleep briefly to allow OS to reclaim memory
        time.sleep(3)

    # Log memory usage before cleanup
    total_memory_before = psutil.virtual_memory().used
    print(f"Total memory usage before cleanup: {total_memory_before / (1024 ** 3):.2f} GB")

    

    # Force aggressive cleanup
    force_memory_cleanup()

    # Total memory usage after cleanup
    total_memory_after = psutil.virtual_memory().used
    print(f"Total memory usage after cleanup: {total_memory_after / (1024 ** 3):.2f} GB")
    # --- end of memory cleanup ---


def has_elastic_registration(scan_name, directory):
    """
    Check if a scan has completed elastic registration (latest edit is elastic).
    
    Args:
        scan_name (str): Name of the scan to check
        directory (str): Root project directory containing 'extracted' folder
    
    Returns:
        bool: True if scan has elastic registration as latest edit, False otherwise
    """
    try:

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
                    with open(json_path, 'r') as jf:
                        metadata = json.load(jf)
                        if metadata.get('elastic_to', "none") == scan_name:
                            return True
                

        # Check if the scan is in the atlas (special case)
        if scan_name == "atlas":
            # Assume atlas as always elastic
            return True
        else:
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


def has_bias_correction_output(directory, scan_name):
    """
    True if subject already has a bias-corrected volume saved (current n4corrected filenames
    or legacy n3corrected filenames from older builds).
    """
    base = os.path.join(directory, "extracted", scan_name)
    if glob.glob(os.path.join(base, f"{scan_name}_edit_*_n4corrected.nii.gz")):
        return True
    if glob.glob(os.path.join(base, f"{scan_name}_edit_*_n3corrected.nii.gz")):
        return True
    return False


def _clamp_island_volume_percent(value, default=1.0):
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = default
    return max(0.1, min(30.0, v))


def build_mesh_cleanup_settings(
    *,
    use_island_volume_threshold=False,
    min_island_volume_percent=None,
    gaussian_blur=None,
    geometry='voxel',
    cleanup_performed=None,
    reason=None,
    source_edit=None,
    components_found=None,
):
    """
    Compact forensic provenance for mesh/island cleanup on subject JSON.

    geometry: ``voxel`` (NIfTI island cleanup) or ``mesh`` (preserved PLY faces).

    When cleanup finds nothing to remove, still write this block with
    ``cleanup_performed=False`` so scientific reports can distinguish
    "cleanup omitted from the workflow" from "cleanup attempted, already clean".
    """
    use_threshold = bool(use_island_volume_threshold)
    if geometry == 'mesh':
        mode = 'face_percent_threshold' if use_threshold else 'largest_component'
    else:
        mode = 'island_volume_threshold' if use_threshold else 'largest_component'
    settings = {
        'mode': mode,
        'geometry': geometry,
        'use_island_volume_threshold': use_threshold,
        'min_island_volume_percent': (
            float(min_island_volume_percent) if use_threshold and min_island_volume_percent is not None else None
        ),
    }
    if geometry == 'voxel':
        try:
            gb = float(gaussian_blur) if gaussian_blur is not None else 0.0
        except (TypeError, ValueError):
            gb = 0.0
        settings['gaussian_blur'] = gb if gb else 0.0
    if cleanup_performed is not None:
        settings['cleanup_performed'] = bool(cleanup_performed)
    if reason:
        settings['reason'] = str(reason)
    if source_edit:
        settings['source_edit'] = os.path.basename(str(source_edit))
    if components_found is not None:
        try:
            settings['components_found'] = int(components_found)
        except (TypeError, ValueError):
            pass
    return settings


def write_subject_json_settings(json_path, key, value, metadata=None):
    """Merge a settings object onto subject JSON and write it back. Returns updated dict."""
    # Reload-and-patch so a stale in-memory metadata dict cannot clobber
    # lossy_compression rows appended by save_as_lossy_nifti.
    return patch_subject_json(json_path, {key: value}, metadata=metadata)


N4_BIAS_DEFAULTS = {
    'shrink_factor': 4,
    'convergence_iters': 50,
    'convergence_tol_exp': 7,
    'spline_param': 200,
}


def build_n4_bias_settings(params=None):
    settings = dict(N4_BIAS_DEFAULTS)
    if isinstance(params, dict):
        for key, value in params.items():
            if value is not None:
                settings[key] = value
    return settings


DENOISE_PARAMETER_DEFAULTS = {
    'Gaussian': {'sigma': 1.0, 'truncate': 4.0},
    'Median': {},
    'NonLocalMeans': {
        'patch_size': 7,
        'patch_distance': 11,
        'h': 0.01,
        'fast_mode': True,
        'mode': '2.5D',
    },
    'TotalVariationChambolle': {'weight': 0.1, 'eps': 0.0002, 'max_num_iter': 200},
    'TotalVariationBregman': {'weight': 5.0, 'eps': 0.001, 'max_num_iter': 100},
    'Wavelet': {
        'wavelet': 'db1',
        'level': 1,
        'sigma': 0.1,
        'mode': 'soft',
        'method': 'BayesShrink',
    },
    'Butterworth': {
        'cutoff': 0.005,
        'order': 2,
        'high_pass': False,
        'squared_butterworth': True,
    },
}


def build_denoise_settings(algorithm, parameters=None):
    defaults = dict(DENOISE_PARAMETER_DEFAULTS.get(algorithm, {}))
    if isinstance(parameters, dict):
        for key, value in parameters.items():
            if value is not None:
                defaults[key] = value
    return {
        'algorithm': algorithm,
        'parameters': defaults,
    }


CLAHE_RESTORE_DEFAULTS = {
    'kernel_size': 0,
    'clip_limit': 0.01,
    'nbins': 256,
}


def build_restore_settings(algorithm, parameters=None, prefer_mask=False, mask_label=1):
    defaults = dict(CLAHE_RESTORE_DEFAULTS) if algorithm == 'CLAHE' else {}
    if isinstance(parameters, dict):
        for key, value in parameters.items():
            if value is not None:
                defaults[key] = value
    return {
        'algorithm': algorithm,
        'parameters': defaults,
        'prefer_mask': bool(prefer_mask),
        'mask_label': int(mask_label) if prefer_mask else None,
    }


class CleanupMeshView(APIView):
    def _get_background_value_for_cleanup(self, data, current_threshold):
        """
        Calculate background value as the mode of values strictly below the threshold.
        Uses the same robust approach as HomogenizeBackgroundView._get_background_value.
        Returns: background value as int
        """
        # Subsample for speed
        vals = data[::2, ::2, ::2]
        thr = float(current_threshold)

        # uint8: discrete mode between 2 and threshold
        if np.issubdtype(vals.dtype, np.integer) and vals.dtype == np.uint8:
            low = 2
            thri = int(np.clip(thr, 0, 255))
            if thri <= low:
                return 0
            mask = (vals >= low) & (vals <= thri)
            if not np.any(mask):
                return 0
            sel = vals[mask].astype(np.int16, copy=False)
            hist = np.bincount(sel, minlength=256)
            hist[:low] = 0
            if thri + 1 < hist.size:
                hist[thri + 1:] = 0
            return int(np.argmax(hist))

        # uint16: mode bin between 1000 and threshold with 500-step bins
        if np.issubdtype(vals.dtype, np.integer) and vals.dtype == np.uint16:
            low = 1000
            thri = int(np.clip(thr, 0, np.iinfo(np.uint16).max))
            if thri <= low:
                mask = (vals > 0) & (vals < thri)
                return int(round(float(np.mean(vals[mask])))) if np.any(mask) else 0

            edges = np.arange(low, thri + 500, 500, dtype=np.int64)
            if edges.size < 2:
                edges = np.array([low, thri], dtype=np.int64)

            hist, _ = np.histogram(vals, bins=edges)
            if hist.size == 0 or hist.max() == 0:
                mask = (vals > 0) & (vals < thri)
                return int(round(float(np.mean(vals[mask])))) if np.any(mask) else 0

            k = int(np.argmax(hist))
            lo, hi = int(edges[k]), int(edges[min(k + 1, edges.size - 1)])
            in_bin = vals[(vals >= lo) & (vals < hi)]
            if in_bin.size == 0:
                return lo
            return int(round(float(np.mean(in_bin))))

        # Fallback for other dtypes: mode in (0, threshold)
        mask = (vals > 0) & (vals < thr)
        return int(round(float(np.mean(vals[mask])))) if np.any(mask) else 0

    def clean_mesh(
        self,
        nifti_data,
        threshold,
        original_dtype,
        gaussian_blur,
        use_island_volume_threshold=False,
        min_island_volume_percent=1.0,
    ):
        """
        Clean up the mesh by removing small disconnected components.

        Returns: (cleaned_data, cleanup_performed, removal_mask, outcome)
        where removal_mask is boolean/uint8 True where voxels were set to background
        (or None if no cleanup), and outcome describes components found / skip reason.
        """
        # Apply Gaussian smoothing to the nifti data
        nifti_data_gaussian = nifti_data if gaussian_blur is None or gaussian_blur == 0 else gaussian_filter(nifti_data, sigma=gaussian_blur)

        # Calculate the background value as the mode of values below the threshold
        background_value = np.array(self._get_background_value_for_cleanup(nifti_data, threshold), dtype=original_dtype)
        print("background_value: ", background_value)

        # To identify and remove small disconnected components, use threshold to identify the islands, keep the largest island
        islands = (nifti_data_gaussian > threshold).astype(bool)
        
        # Label connected components
        labeled_islands, num_features = ndimage.label(islands)
        
        # Find the sizes of each component
        sizes = np.bincount(labeled_islands.ravel())
               
        # Exclude the background (size of 0)
        sizes = sizes[1:]  # Skip the first element which corresponds to the background
        
        cleanup_performed = False
        sample_inds = np.array([], dtype=np.intp).reshape(0, 3)  # empty, for debug use when cleanup_performed
        islands_mask = None
        n_to_remove = 0
        cleaned_data = nifti_data.astype(original_dtype)
        removal_mask = None  # Will be set to boolean mask of removed voxels if cleanup occurs
        reason = 'no_disconnected_components'
        components_found = int(sizes.size)

        if sizes.size == 0:
            print("No islands found.")
            cleaned_data = np.zeros_like(nifti_data, dtype=original_dtype)  # No islands, return empty data
            reason = 'no_foreground'
            components_found = 0
        elif sizes.size == 1:
            print("Only one connected component found - no cleanup needed.")
            cleaned_data = nifti_data.astype(original_dtype)
            reason = 'no_disconnected_components'
        else:
            print(f"Found {sizes.size} connected components.")
            if use_island_volume_threshold:
                total_foreground = int(islands.sum())
                min_voxels = max(1, int(total_foreground * min_island_volume_percent / 100.0))
                keep_labels = {label for label, size in enumerate(sizes, start=1) if size >= min_voxels}
                if not keep_labels:
                    keep_labels = {int(sizes.argmax()) + 1}
                all_labels = set(range(1, len(sizes) + 1))
                remove_labels = all_labels - keep_labels
                if not remove_labels:
                    print("All connected components above volume threshold - no cleanup needed.")
                    reason = 'all_components_kept'
                else:
                    print(
                        f"Volume threshold mode: keeping {len(keep_labels)} island(s), "
                        f"removing {len(remove_labels)} (min {min_voxels} voxels, {min_island_volume_percent}%)."
                    )
                    islands_mask = np.isin(labeled_islands, list(remove_labels))
                    cleanup_performed = True
                    reason = 'components_removed'
            else:
                print("Largest-island mode - cleanup needed.")
                cleanup_performed = True
                reason = 'components_removed'
                largest_island = sizes.argmax() + 1  # +1 to account for background
                print("Largest island label: ", largest_island)
                labels_without_largest = np.where(
                    (labeled_islands != largest_island) & (labeled_islands > 0),
                    labeled_islands,
                    0,
                )
                islands_mask = labels_without_largest > 0

            if cleanup_performed and islands_mask is not None:
                n_to_remove = int(islands_mask.sum())
                print(f"[clean_mesh] islands_mask.sum() = {n_to_remove} (voxels to set to background)")
                sample_inds = np.argwhere(islands_mask)
                if len(sample_inds) > 0:
                    for idx in sample_inds[:3]:
                        i, j, k = tuple(idx)
                        print(f"[clean_mesh] before: nifti_data[{i},{j},{k}] = {nifti_data[i, j, k]}")

                print(f"[DEBUG] BEFORE clear: above_threshold_total = {int((nifti_data > threshold).sum())}")
                print(f"[DEBUG] BEFORE clear: mask & above_threshold = {int((islands_mask & (nifti_data > threshold)).sum())}")

                nifti_data[islands_mask] = background_value

                print(f"[DEBUG] AFTER clear:  above_threshold_total = {int((nifti_data > threshold).sum())}")

                if len(sample_inds) > 0:
                    for idx in sample_inds[:3]:
                        i, j, k = tuple(idx)
                        print(f"[clean_mesh] after:  nifti_data[{i},{j},{k}] = {nifti_data[i, j, k]} (expect {background_value})")
                print(f"[clean_mesh] nifti_data.flags.writeable = {nifti_data.flags.writeable}")

                cleaned_data = nifti_data
                # Initialize removal_mask from islands_mask
                removal_mask = islands_mask.copy()

        #  If gaussian blur is not none, then we need to find the noise and remove it too (the noise is cleaned_data > threshold that doesn't show up in nifti_data_gaussian > threshold, while ignoring the largest island dilated 1 voxel)
        if gaussian_blur is not None and gaussian_blur != 0 and cleanup_performed:
            noise_mask = (cleaned_data > threshold) & (nifti_data_gaussian <= threshold) & (islands_mask == False)
            cleaned_data[noise_mask] = background_value
            # Add noise_mask to removal_mask
            if removal_mask is not None:
                removal_mask = removal_mask | noise_mask
            else:
                removal_mask = noise_mask.copy()

        if cleanup_performed:
            above_in_mask = int(((islands_mask) & (nifti_data > threshold)).sum())
            print(f"[clean_mesh] mask voxels that are ABOVE threshold: {above_in_mask} out of {n_to_remove}")

        returned = cleaned_data.astype(original_dtype)
        print(f"[clean_mesh] return: id(cleaned_data)==id(nifti_data)? {cleaned_data is nifti_data}, id(returned)==id(nifti_data)? {returned is nifti_data}")
        if cleanup_performed and len(sample_inds) > 0:
            i, j, k = tuple(sample_inds[0])
            print(f"[clean_mesh] returned[{i},{j},{k}] = {returned[i, j, k]}")
        outcome = {
            'cleanup_performed': bool(cleanup_performed),
            'reason': reason,
            'components_found': components_found,
        }
        return returned, cleanup_performed, removal_mask, outcome

    # This function is used to clean up the mesh by removing small disconnected components and trying to separate connected objects
    def post(self, request):
        print("Cleaning up mesh")
        directory = request.data['directory']
        filename = request.data['filename']
        edit = request.data['edit']
        gaussian_blur = request.data['gaussian_blur']
        use_island_volume_threshold = bool(request.data.get('use_island_volume_threshold', False))
        min_island_volume_percent = _clamp_island_volume_percent(
            request.data.get('min_island_volume_percent', 1.0)
        )

        # Check if the latest edit is elastic - skip if it is
        if edit and 'elastic' in edit.lower():
            return Response({
                'message': 'error: Cannot cleanup mesh after elastic registration.',
                'edit': edit
            }, status=status.HTTP_400_BAD_REQUEST)

        # Load the metadata JSON to get the voxel size
        if "atlas" in filename:
            json_path = os.path.join(directory, "atlas", filename+".json")
        else:
            json_path = os.path.join(directory, "extracted", filename, filename+".json")
        with open(json_path, 'r') as jf:
            json_metadata = json.load(jf)

        # Get threshold from json metadata
        threshold = json_metadata['threshold']

        print("directory: ", directory)
        print("filename: ", filename)
        print("edit: ", edit)
        print("threshold: ", threshold)
        print("gaussian_blur: ", gaussian_blur)
        print("use_island_volume_threshold: ", use_island_volume_threshold)
        print("min_island_volume_percent: ", min_island_volume_percent)
        # Get the 3D nifti file and load it into a numpy array
        sub_path = "atlas" if "atlas" in filename else "extracted/"+filename
        if edit is None:
            nifti_path = os.path.join(directory, sub_path, filename + ".nii.gz")
        else:
            nifti_path = os.path.join(directory, sub_path, edit.replace("_lossy", ""))

        print("nifti_path: "+nifti_path)
        nifti_img = nib.load(nifti_path)
        original_dtype = nifti_img.get_data_dtype()
        nifti_data = nifti_img.get_fdata().astype(original_dtype)

        # Clean the resulting mesh using the class CleanMesh and save it in the variable cleaned_data
        cleaned_data, cleanup_performed, removal_mask, cleanup_outcome = self.clean_mesh(
            nifti_data,
            threshold,
            original_dtype,
            gaussian_blur,
            use_island_volume_threshold=use_island_volume_threshold,
            min_island_volume_percent=min_island_volume_percent,
        )

        above = int((cleaned_data > threshold).sum())
        print(f"[post] cleaned_data: voxels above threshold = {above}")

        source_edit_name = None
        if edit:
            source_edit_name = os.path.basename(str(edit).replace("_lossy", ""))
        elif edit is None:
            source_edit_name = f"{filename}.nii.gz"

        # Skip writing a new edit when nothing was removed, but still record that
        # cleanup was attempted so scientific reports can phrase this deterministically.
        if not cleanup_performed:
            try:
                cleanup_settings = build_mesh_cleanup_settings(
                    use_island_volume_threshold=use_island_volume_threshold,
                    min_island_volume_percent=min_island_volume_percent,
                    gaussian_blur=gaussian_blur,
                    geometry='voxel',
                    cleanup_performed=False,
                    reason=(cleanup_outcome or {}).get('reason') or 'no_disconnected_components',
                    source_edit=source_edit_name,
                    components_found=(cleanup_outcome or {}).get('components_found'),
                )
                write_subject_json_settings(json_path, 'mesh_cleanup_settings', cleanup_settings, metadata=json_metadata)
            except Exception as meta_err:
                print(f"Warning: could not save mesh_cleanup_settings (no-op): {meta_err}")
            return Response({
                'message': 'No cleanup performed - no islands removed',
                'edit': edit,
                'cleanup_performed': False,
                'reason': (cleanup_outcome or {}).get('reason'),
                'components_found': (cleanup_outcome or {}).get('components_found'),
            }, status=status.HTTP_200_OK)

        # Create new nifti image with cleaned data
        new_nifti = nib.Nifti1Image(cleaned_data, nifti_img.affine)

        # Find the last edit number
        edit_number = 0
        while glob.glob(os.path.join(directory, sub_path, f"{filename}_edit_{edit_number}_*.nii.gz")):
            edit_number += 1

        # Save the cleaned image
        output_path = os.path.join(directory, sub_path, f"{filename}_edit_{edit_number}_cleaned.nii.gz")
        nib.save(new_nifti, output_path)

        reloaded = nib.load(output_path).get_fdata()
        above_reload = int((reloaded > threshold).sum())
        print(f"[post] after save: reloaded voxels above threshold = {above_reload}, path = {output_path}")

        # Save removal mask if cleanup was performed
        removal_mask_path = None
        if removal_mask is not None:
            # Save as uint8 NIfTI with .nii.removal_mask.gz extension
            # Use temp rename pattern like other masks
            removal_mask_uint8 = removal_mask.astype(np.uint8)
            removal_mask_nifti = nib.Nifti1Image(removal_mask_uint8, nifti_img.affine)
            removal_mask_temp = os.path.join(directory, sub_path, f"{filename}_edit_{edit_number}_cleaned_removal_mask.nii.gz")
            removal_mask_path = os.path.join(directory, sub_path, f"{filename}_edit_{edit_number}_cleaned.nii.removal_mask.gz")
            nib.save(removal_mask_nifti, removal_mask_temp)
            os.replace(removal_mask_temp, removal_mask_path)
            print(f"Saved removal mask to: {removal_mask_path}")

        # Generate the lossy version using the existing save_as_lossy_nifti function
        RegistrationTools().save_as_lossy_nifti(
            cleaned_data, 
            json_metadata['voxel_size'], 
            os.path.join(directory, sub_path, filename + ".json"),
            os.path.join(directory, sub_path, f"{filename}_lossy_edit_{edit_number}_cleaned.nii.gz")
        )

        # Persist cleanup provenance on subject JSON (forensic scientific report)
        try:
            cleanup_settings = build_mesh_cleanup_settings(
                use_island_volume_threshold=use_island_volume_threshold,
                min_island_volume_percent=min_island_volume_percent,
                gaussian_blur=gaussian_blur,
                geometry='voxel',
                cleanup_performed=True,
                reason=(cleanup_outcome or {}).get('reason') or 'components_removed',
                source_edit=source_edit_name,
                components_found=(cleanup_outcome or {}).get('components_found'),
            )
            write_subject_json_settings(json_path, 'mesh_cleanup_settings', cleanup_settings, metadata=json_metadata)
        except Exception as meta_err:
            print(f"Warning: could not save mesh_cleanup_settings: {meta_err}")

        # Copy paired mask (non-geometry change) and generate lossy by downsampling from copied full-res
        try:
            # Determine source base - if edit is provided, use it; otherwise use filename
            if edit is None:
                source_base = filename
            else:
                source_base = edit.replace("_lossy", "").replace(".nii.gz", "")
            
            src_mask_full = os.path.join(directory, sub_path, f"{source_base}.nii.mask.gz")
            dst_mask_full = os.path.join(directory, sub_path, f"{filename}_edit_{edit_number}_cleaned.nii.mask.gz")
            dst_mask_lossy = os.path.join(directory, sub_path, f"{filename}_lossy_edit_{edit_number}_cleaned.nii.mask.gz")
            dst_nifti_lossy = os.path.join(directory, sub_path, f"{filename}_lossy_edit_{edit_number}_cleaned.nii.gz")

            if os.path.isfile(src_mask_full):
                shutil.copyfile(src_mask_full, dst_mask_full)
                print(f"Copied full-res mask to: {dst_mask_full}")

                # Downsample copied full-res mask to create lossy mask
                resolution_factor = 2
                lc = json_metadata.get('lossy_compression', None)
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

                # load the lossy destination nifti image to get the affine
                lossy_nifti_img = nib.load(dst_nifti_lossy)
                lossy_affine = lossy_nifti_img.affine

                mask_temp = dst_mask_full.replace('.nii.mask.gz', '_mask.nii.gz')
                lossy_temp = dst_mask_lossy.replace('.nii.mask.gz', '_mask.nii.gz')
                os.replace(dst_mask_full, mask_temp)
                try:
                    mask_img = nib.load(mask_temp)
                    mask_dtype = mask_img.get_data_dtype()
                    # Round before astype to avoid truncating floats
                    mask_arr = np.round(mask_img.get_fdata()).astype(mask_dtype)
                    slices = [slice(None, None, resolution_factor) for _ in range(3)]
                    lossy_mask = mask_arr[slices[0], slices[1], slices[2]].astype(mask_dtype, copy=False)
                    nib.save(nib.Nifti1Image(lossy_mask, lossy_affine), lossy_temp)
                    os.replace(lossy_temp, dst_mask_lossy)
                    print(f"Generated lossy mask: {dst_mask_lossy}")
                finally:
                    os.replace(mask_temp, dst_mask_full)
            else:
                print("No source full-res mask found; skipping.")
        except Exception as me:
            print(f"Error handling paired mask (cleaned): {me}")

        
        # Copy paired landmark files (non-geometry change) if they exist
        try:
            if edit is None:
                source_landmark_base = filename
            else:
                source_landmark_base = edit.replace("_lossy", "").replace(".nii.gz", "")
            
            source_landmarks_path = os.path.join(directory, sub_path, f"{source_landmark_base}_landmarks.json")
            source_distances_path = os.path.join(directory, sub_path, f"{source_landmark_base}_landmark_distances.json")
            
            dest_landmark_base = f"{filename}_edit_{edit_number}_cleaned"
            dest_landmarks_path = os.path.join(directory, sub_path, f"{dest_landmark_base}_landmarks.json")
            dest_distances_path = os.path.join(directory, sub_path, f"{dest_landmark_base}_landmark_distances.json")
            
            if os.path.isfile(source_landmarks_path):
                shutil.copyfile(source_landmarks_path, dest_landmarks_path)
                print(f"Copied landmarks to: {dest_landmarks_path}")
            
            if os.path.isfile(source_distances_path):
                shutil.copyfile(source_distances_path, dest_distances_path)
                print(f"Copied landmark distances to: {dest_distances_path}")
        except Exception as le:
            print(f"Error handling paired landmark files (cleaned): {le}")
        

        return Response({
            'message': 'Cleanup applied successfully',
            'edit': f"{filename}_lossy_edit_{edit_number}_cleaned.nii.gz",
            'edit_number': edit_number,
            'removal_mask': os.path.basename(removal_mask_path) if removal_mask_path else None,
        }, status=status.HTTP_200_OK)


class ApplyCleanupRemovalMaskView(APIView):
    """
    Apply a removal mask to target scans (typically same-shape linked siblings).

    Used by any edit that removes voxels (island cleanup, slicing): the operated scan
    records exactly which voxels it cleared and the siblings replay that mask verbatim,
    so the group stays voxel-for-voxel identical instead of each scan redetecting what
    to remove from its own intensities.

    POST /apply-cleanup-removal-mask/
    Body:
        - directory: str
        - source_filename: str (scan that has the removal mask)
        - source_edit: int (edit number of cleaned edit with removal mask)
        OR
        - removal_mask: str (path/basename of removal mask file)
        - targets: list[str] (scan names to apply mask to)
        OR
        - propagate_linked: bool (auto-resolve same-shape siblings of source)
        - edit_suffix: str (edit name suffix for the written edits, default 'cleaned')
    """
    def post(self, request):
        directory = request.data.get('directory')
        source_filename = request.data.get('source_filename')
        source_edit = request.data.get('source_edit')
        removal_mask_param = request.data.get('removal_mask')
        targets = request.data.get('targets', [])
        propagate_linked = bool(request.data.get('propagate_linked', False))
        edit_suffix = str(request.data.get('edit_suffix') or 'cleaned').strip('_') or 'cleaned'
        
        if not directory:
            return Response({'error': 'directory is required'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Resolve removal mask path
        removal_mask_path = None
        if removal_mask_param:
            # Provided as path or basename
            if os.path.isabs(removal_mask_param) and os.path.isfile(removal_mask_param):
                removal_mask_path = removal_mask_param
            else:
                # Try as basename in extracted/source_filename if source_filename provided
                if source_filename:
                    removal_mask_path = os.path.join(directory, "extracted", source_filename, removal_mask_param)
                    if not os.path.isfile(removal_mask_path):
                        return Response({'error': f'Removal mask not found: {removal_mask_path}'}, status=status.HTTP_404_NOT_FOUND)
                else:
                    return Response({'error': 'removal_mask provided but source_filename missing'}, status=status.HTTP_400_BAD_REQUEST)
        elif source_filename and source_edit is not None:
            # Resolve from source_filename and edit number
            # source_edit can be an int or a string like "foo_edit_3_cleaned.nii.gz"
            if isinstance(source_edit, str):
                # Extract edit number from string using regex
                match = re.search(r'_edit_(\d+)_', source_edit)
                if match:
                    edit_num = int(match.group(1))
                else:
                    return Response({'error': f'Could not extract edit number from source_edit: {source_edit}'}, status=status.HTTP_400_BAD_REQUEST)
            else:
                edit_num = int(source_edit)
            
            removal_mask_path = os.path.join(
                directory, "extracted", source_filename,
                f"{source_filename}_edit_{edit_num}_cleaned.nii.removal_mask.gz"
            )
            if not os.path.isfile(removal_mask_path):
                return Response({'error': f'Removal mask not found: {removal_mask_path}'}, status=status.HTTP_404_NOT_FOUND)
        else:
            return Response({'error': 'Must provide (source_filename + source_edit) OR removal_mask'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Resolve targets
        if propagate_linked and source_filename:
            siblings, err = get_same_shape_siblings(directory, source_filename)
            if err:
                return Response({'error': f'Could not resolve same-shape siblings: {err}'}, status=status.HTTP_400_BAD_REQUEST)
            targets = siblings
        
        if not targets:
            return Response({'error': 'No targets specified or resolved'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Load removal mask
        removal_mask_temp = removal_mask_path.replace('.nii.removal_mask.gz', '_removal_mask.nii.gz')
        os.replace(removal_mask_path, removal_mask_temp)
        try:
            mask_img = nib.load(removal_mask_temp)
            removal_mask = np.round(mask_img.get_fdata()).astype(bool)
            mask_affine = mask_img.affine
        finally:
            os.replace(removal_mask_temp, removal_mask_path)

        source_cleanup_settings = None
        if source_filename:
            try:
                source_meta = load_extracted_scan_metadata(directory, source_filename)
                if isinstance(source_meta.get('mesh_cleanup_settings'), dict):
                    source_cleanup_settings = source_meta['mesh_cleanup_settings']
            except Exception:
                pass
        
        # Apply to each target
        successful = []
        failed = []
        for target_name in targets:
            try:
                result = self._apply_to_target(
                    directory,
                    target_name,
                    removal_mask,
                    mask_affine,
                    edit_suffix,
                    cleanup_settings=source_cleanup_settings,
                )
                if result['status'] == 'success':
                    successful.append(result)
                else:
                    failed.append(result)
            except Exception as e:
                failed.append({'name': target_name, 'reason': str(e)})
        
        message = f"Applied removal mask to {len(successful)}/{len(targets)} targets"
        return Response({
            'message': message,
            'successful': successful,
            'failed': failed,
        }, status=status.HTTP_200_OK)
    
    def _apply_to_target(
        self,
        directory,
        target_name,
        removal_mask,
        mask_affine,
        edit_suffix='cleaned',
        cleanup_settings=None,
    ):
        """Apply removal mask to a single target scan."""
        # Skip preserved meshes
        try:
            metadata = load_extracted_scan_metadata(directory, target_name)
            if is_preserved_mesh_metadata(metadata):
                return {'name': target_name, 'status': 'skipped', 'reason': 'preserved mesh'}
        except Exception:
            pass
        
        # Skip if has elastic registration
        if has_elastic_registration(target_name, directory):
            return {'name': target_name, 'status': 'skipped', 'reason': 'elastic registration present'}
        
        # Find latest non-elastic edit
        latest_edit = 0
        while glob.glob(os.path.join(directory, "extracted", target_name, f"{target_name}_edit_{latest_edit}_*.nii.gz")):
            latest_edit += 1
        latest_edit -= 1
        
        if latest_edit >= 0:
            edit_files = glob.glob(os.path.join(directory, "extracted", target_name, f"{target_name}_edit_{latest_edit}_*.nii.gz"))
            if edit_files:
                edit_file = os.path.basename(edit_files[0])
                if 'elastic' in edit_file.lower():
                    return {'name': target_name, 'status': 'skipped', 'reason': 'latest edit is elastic'}
                nifti_path = os.path.join(directory, "extracted", target_name, edit_file)
            else:
                nifti_path = os.path.join(directory, "extracted", target_name, f"{target_name}.nii.gz")
        else:
            nifti_path = os.path.join(directory, "extracted", target_name, f"{target_name}.nii.gz")
        
        # Load target nifti
        target_img = nib.load(nifti_path)
        target_dtype = target_img.get_data_dtype()
        target_data = target_img.get_fdata().astype(target_dtype)
        
        # Check shape match
        if target_data.shape != removal_mask.shape:
            return {'name': target_name, 'status': 'skipped', 'reason': f'shape mismatch: {target_data.shape} vs {removal_mask.shape}'}
        
        # Load target metadata
        json_path = os.path.join(directory, "extracted", target_name, f"{target_name}.json")
        with open(json_path, 'r') as jf:
            target_metadata = json.load(jf)
        target_threshold = target_metadata.get('threshold', 1)
        
        # Compute background value
        vals = target_data[::2, ::2, ::2]
        thr = float(target_threshold)
        if np.issubdtype(target_dtype, np.integer) and target_dtype == np.uint8:
            low = 2
            thri = int(np.clip(thr, 0, 255))
            if thri <= low:
                background_value = 0
            else:
                mask_bg = (vals >= low) & (vals <= thri)
                if not np.any(mask_bg):
                    background_value = 0
                else:
                    sel = vals[mask_bg].astype(np.int16, copy=False)
                    hist = np.bincount(sel, minlength=256)
                    hist[:low] = 0
                    if thri + 1 < hist.size:
                        hist[thri + 1:] = 0
                    background_value = int(np.argmax(hist))
        elif np.issubdtype(target_dtype, np.integer) and target_dtype == np.uint16:
            low = 1000
            thri = int(np.clip(thr, 0, np.iinfo(np.uint16).max))
            if thri <= low:
                mask_bg = (vals > 0) & (vals < thri)
                background_value = int(round(float(np.mean(vals[mask_bg])))) if np.any(mask_bg) else 0
            else:
                edges = np.arange(low, thri + 500, 500, dtype=np.int64)
                if edges.size < 2:
                    edges = np.array([low, thri], dtype=np.int64)
                hist, _ = np.histogram(vals, bins=edges)
                if hist.size == 0 or hist.max() == 0:
                    mask_bg = (vals > 0) & (vals < thri)
                    background_value = int(round(float(np.mean(vals[mask_bg])))) if np.any(mask_bg) else 0
                else:
                    k = int(np.argmax(hist))
                    lo, hi = int(edges[k]), int(edges[min(k + 1, edges.size - 1)])
                    in_bin = vals[(vals >= lo) & (vals < hi)]
                    background_value = int(round(float(np.mean(in_bin)))) if in_bin.size > 0 else lo
        else:
            mask_bg = (vals > 0) & (vals < thr)
            background_value = int(round(float(np.mean(vals[mask_bg])))) if np.any(mask_bg) else 0
        
        background_value = np.array(background_value, dtype=target_dtype)
        
        # Apply removal mask
        target_data[removal_mask] = background_value
        
        # Find next edit number
        new_edit_number = 0
        while glob.glob(os.path.join(directory, "extracted", target_name, f"{target_name}_edit_{new_edit_number}_*.nii.gz")):
            new_edit_number += 1
        
        # Save cleaned target
        new_nifti = nib.Nifti1Image(target_data, target_img.affine)
        output_path = os.path.join(directory, "extracted", target_name, f"{target_name}_edit_{new_edit_number}_{edit_suffix}.nii.gz")
        nib.save(new_nifti, output_path)
        
        # Save lossy version
        RegistrationTools().save_as_lossy_nifti(
            target_data,
            target_metadata['voxel_size'],
            json_path,
            os.path.join(directory, "extracted", target_name, f"{target_name}_lossy_edit_{new_edit_number}_{edit_suffix}.nii.gz")
        )

        if isinstance(cleanup_settings, dict) and edit_suffix == 'cleaned':
            try:
                write_subject_json_settings(
                    json_path, 'mesh_cleanup_settings', cleanup_settings, metadata=target_metadata
                )
            except Exception as meta_err:
                print(f"  Warning: could not save mesh_cleanup_settings for {target_name}: {meta_err}")
        
        # Copy paired mask if exists
        try:
            if latest_edit >= 0:
                source_base = os.path.basename(nifti_path).replace('.nii.gz', '')
            else:
                source_base = target_name
            src_mask_full = os.path.join(directory, "extracted", target_name, f"{source_base}.nii.mask.gz")
            if os.path.isfile(src_mask_full):
                dst_mask_full = os.path.join(directory, "extracted", target_name, f"{target_name}_edit_{new_edit_number}_{edit_suffix}.nii.mask.gz")
                shutil.copyfile(src_mask_full, dst_mask_full)
        except Exception as me:
            print(f"  Error copying paired mask for {target_name}: {me}")
        
        # Copy paired landmarks if exist
        try:
            if latest_edit >= 0:
                source_landmark_base = os.path.basename(nifti_path).replace('.nii.gz', '')
            else:
                source_landmark_base = target_name
            source_landmarks_path = os.path.join(directory, "extracted", target_name, f"{source_landmark_base}_landmarks.json")
            if os.path.isfile(source_landmarks_path):
                dest_landmarks_path = os.path.join(directory, "extracted", target_name, f"{target_name}_edit_{new_edit_number}_{edit_suffix}_landmarks.json")
                shutil.copyfile(source_landmarks_path, dest_landmarks_path)
        except Exception as le:
            print(f"  Error copying landmarks for {target_name}: {le}")
        
        return {
            'name': target_name,
            'status': 'success',
            'edit': f"{target_name}_lossy_edit_{new_edit_number}_{edit_suffix}.nii.gz"
        }


class HomogenizeBackgroundView(APIView):
    def _load_scan_metadata(self, directory, scan_name):
        json_path = os.path.join(directory, "extracted", scan_name, f"{scan_name}.json")
        with open(json_path, "r") as jf:
            return json.load(jf)

    def _is_preserved_mesh_metadata(self, metadata):
        return metadata.get("is_mesh") is True and metadata.get("voxelized") is False

    def post(self, request):
        print("Homogenizing background for all scans")
        directory = request.data['directory']
        background_value = request.data['background_value']
        flag_filter = normalize_flag_filter_value(
            request.data.get('flagFilter', 'off') if request.data else 'off',
            only_current_scan=False,
        )

        print("background_value is ", background_value)
        print(f"Flag filter: {flag_filter}")

        if background_value is not None:
            background_value = float(background_value)

        extracted_dir = os.path.join(directory, "extracted")
        scan_names = []
        skipped_preserved_meshes = []
        for scan_name in os.listdir(extracted_dir):
            if scan_name == "project_settings.json" or not os.path.isdir(os.path.join(extracted_dir, scan_name)):
                continue
            try:
                metadata = self._load_scan_metadata(directory, scan_name)
            except (FileNotFoundError, json.JSONDecodeError) as exc:
                print(f"Skipping {scan_name}: could not read metadata ({exc})")
                continue
            if metadata.get('faulty', False):
                continue
            if self._is_preserved_mesh_metadata(metadata):
                skipped_preserved_meshes.append(scan_name)
                continue
            scan_names.append(scan_name)

        if not scan_names:
            return Response(
                {
                    "status": "error",
                    "message": "Set Background Value only works with voxel-based volumes. This project currently contains only preserved PLY meshes.",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Exclude scans that are already backfixed or have elastic registration in any edit
        scan_names_to_remove = []

        for scan_name in scan_names:
            # Find all edits for this scan
            edit_files = glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_*_*.nii.gz"))
            # If any edit file is backfixed, mark for removal
            if any("_backfixed" in os.path.basename(f) for f in edit_files):
                scan_names_to_remove.append(scan_name)
                continue  # Already marked, do not double-count elastics

            # Remove if it has elastic registration in any edit
            if has_elastic_registration(scan_name, directory):
                scan_names_to_remove.append(scan_name)

        # Remove marked scan names from scan_names
        for scan_name in scan_names_to_remove:
            if scan_name in scan_names:
                scan_names.remove(scan_name)
            else:
                print(f"Scan {scan_name} not found in scan_names")

        # Order scan names alphabetically
        scan_names.sort()

        flagged_set = load_flagged_subject_names(directory)
        scan_names = apply_flag_filter(scan_names, flagged_set, flag_filter)
        scan_names = filter_out_linked_children(directory, scan_names)
        if not scan_names:
            return Response(
                {
                    "status": "error",
                    "message": "No eligible voxel-based scans were found for Set Background Value. Preserved PLY meshes are skipped.",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )
        if skipped_preserved_meshes:
            print(f"Skipping preserved mesh subjects for background homogenization: {skipped_preserved_meshes}")

        background_values = []
        optimal_background = None
        last_idx = 0  

        scan_count = len(scan_names)

        if background_value is None:
            # Each auto phase is its own 0→1 progress epoch. Phase 1 (mode-finding)
            # is much cheaper than phase 2 (load/shift/save), so a shared 2n bar
            # races to ~50% then wraps current back to 1 and overshoots the canvas.
            channel_layer = get_channel_layer()
            if channel_layer is not None and scan_count > 0:
                print("Sending progress update")
                async_to_sync(channel_layer.group_send)(
                    'progress_group',
                    {
                        'type': 'send_progress',
                        'progress': 0,
                        'scan_name': scan_names[0],
                        'custom_message': f'Phase 1/2: finding background values for {scan_count} scan(s)...',
                        'total': scan_count,
                        'current': 0,
                    }
                )
            
            # First pass: analyze all scans to determine background properties                        
            for last_idx, scan_name in enumerate(scan_names):
                try:
                    # Find the latest edit for this scan
                    latest_edit = 0
                    while glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{latest_edit}_*.nii.gz")):
                        latest_edit += 1
                    latest_edit -= 1
                    
                    # Send progress update
                    if channel_layer is not None:
                        progress = last_idx / scan_count
                        async_to_sync(channel_layer.group_send)(
                            'progress_group',
                            {
                                'type': 'send_progress',
                                'progress': progress,
                                'scan_name': scan_name,
                                'custom_message': f'Phase 1/2: finding background value for {scan_name}...',
                                'total': scan_count,
                                'current': last_idx + 1,
                            }
                        )
                    
                    # Load the scan data
                    if latest_edit >= 0:
                        scan_path = glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{latest_edit}_*.nii.gz"))[0]
                    else:
                        scan_path = os.path.join(directory, "extracted", scan_name, f"{scan_name}.nii.gz")
                    
                    # Load the scan's metadata
                    with open(os.path.join(directory, "extracted", scan_name, f"{scan_name}.json"), 'r') as jf:
                        scan_metadata = json.load(jf)
                    
                    # Load the scan data
                    time_start = time.time()
                    nifti_img = nib.load(scan_path)
                    original_dtype = nifti_img.get_data_dtype()  # Get the original data type
                    scan_data = nifti_img.get_fdata().astype(original_dtype)  # Keep original dtype
                    time_end = time.time()
                    print(f"Time taken to load {scan_name} scan data: {time_end - time_start} seconds")
                    print(f"Original data type: {original_dtype}")
                    
                    # Analyze background
                    time_start = time.time()
                    background_value = self._get_background_value(scan_data, np.array(scan_metadata['threshold'], dtype=original_dtype))
                    background_values.append(background_value)
                    time_end = time.time()
                    print(f"Time taken to analyze background: {time_end - time_start} seconds")
                    
                    print(f"Analyzed {scan_name}, background value: {background_value}")
                    
                except Exception as e:
                    print(f"Error processing {scan_name}: {str(e)}")
            
        else:
            optimal_background = np.array(background_value, dtype=int)

        # Restart the bar for the apply pass so current/total and 0→1 progress stay aligned.
        if optimal_background is None and scan_count > 0:
            channel_layer = get_channel_layer()
            if channel_layer is not None:
                async_to_sync(channel_layer.group_send)(
                    'progress_group',
                    {
                        'type': 'send_progress',
                        'progress': 0,
                        'scan_name': scan_names[0],
                        'custom_message': f'Phase 2/2: applying background values to {scan_count} scan(s)...',
                        'total': scan_count,
                        'current': 0,
                    }
                )
        
        # Second pass: apply homogenization to all scans
        for idx, scan_name in enumerate(scan_names):
            try:
                # Send progress update
                channel_layer = get_channel_layer()
                if channel_layer is not None:
                    progress = idx / scan_count
                    if optimal_background is not None:
                        progress_message = f'Setting background of {scan_name} to value {optimal_background}...'
                    else:
                        progress_message = f'Phase 2/2: applying background value {background_values[idx]} to {scan_name}...'
                    async_to_sync(channel_layer.group_send)(
                        'progress_group',
                        {
                            'type': 'send_progress',
                            'progress': progress,
                            'scan_name': scan_name,
                            'custom_message': progress_message,
                            'total': scan_count,
                            'current': idx + 1,
                        }
                    )

                # find the latest edit number
                latest_edit = 0
                while glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{latest_edit}_*.nii.gz")):
                    latest_edit += 1
                latest_edit -= 1

                # Load the scan data path
                if latest_edit >= 0:
                    scan_path = glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{latest_edit}_*.nii.gz"))[0]
                else:
                    scan_path = os.path.join(directory, "extracted", scan_name, f"{scan_name}.nii.gz")

                # Load the scan data
                time_start = time.time()
                nifti_img = nib.load(scan_path)
                original_dtype = nifti_img.get_data_dtype()  # Get the original data type
                scan_data = nifti_img.get_fdata().astype(original_dtype)  # keep original dtype
                time_end = time.time()
                print(f"Time taken to load {scan_name} scan data: {time_end - time_start} seconds")
                print(f"Original data type: {original_dtype}")
                
                # Apply homogenization
                time_start = time.time()
                if optimal_background is not None:
                    scan_data = self._homogenize_background(scan_data, optimal_background)
                else:
                    scan_data = self._homogenize_background(scan_data, background_values[idx])
                time_end = time.time()
                print(f"Time taken to homogenize {scan_name} background: {time_end - time_start} seconds")

                cleanup_memory()
                
                # Save the homogenized data
                new_edit_number = latest_edit + 1
                output_path = os.path.join(directory, "extracted", scan_name, 
                                          f"{scan_name}_edit_{new_edit_number}_backfixed.nii.gz")
                output_path_lossy = os.path.join(directory, "extracted", scan_name, 
                                          f"{scan_name}_lossy_edit_{new_edit_number}_backfixed.nii.gz")
                
                                # Save the homogenized data with the original data type
                new_img = nib.Nifti1Image(scan_data.astype(original_dtype), nifti_img.affine, nifti_img.header)
                nib.save(new_img, output_path)
                
                print(f"Saved homogenized version of {scan_name} with original dtype: {original_dtype}")

                # Update threshold in metadata 
                with open(os.path.join(directory, "extracted", scan_name, f"{scan_name}.json"), 'r') as jf:
                    scan_metadata = json.load(jf)
                from .rigidAlignment import (
                    normalize_threshold_edit_stem,
                    snapshot_previous_edit_threshold,
                    append_background_offset_history,
                )
                previous_stem = (
                    normalize_threshold_edit_stem(os.path.basename(scan_path), scan_name)
                    if latest_edit >= 0
                    else scan_name
                )
                snapshot_previous_edit_threshold(scan_metadata, previous_stem)
                if optimal_background is not None:
                    applied_shift = int(optimal_background)
                    offset_mode = 'fixed'
                else:
                    applied_shift = int(background_values[idx])
                    offset_mode = 'auto_per_scan'
                scan_metadata['threshold'] = scan_metadata['threshold'] - applied_shift
                scan_metadata['backfixed_shift'] = applied_shift
                scan_metadata['background_offset_settings'] = {
                    'mode': offset_mode,
                    'shift': applied_shift,
                }
                append_background_offset_history(
                    scan_metadata,
                    f"{scan_name}_edit_{new_edit_number}_backfixed",
                    offset_mode,
                    applied_shift,
                )

                with open(os.path.join(directory, "extracted", scan_name, f"{scan_name}.json"), 'w') as jf:
                    json.dump(scan_metadata, jf, indent=4)


                # Save compressed version
                RegistrationTools().save_as_lossy_nifti(scan_data, scan_metadata['voxel_size'], os.path.join(directory, "extracted", scan_name, f"{scan_name}.json"), output_path_lossy)
                print(f"Saved compressed version of {scan_name}")

                # Copy paired mask (non-geometry change) and generate lossy by downsampling from copied full-res
                try:
                    source_base = os.path.basename(scan_path).replace('.nii.gz', '') if latest_edit >= 0 else scan_name
                    src_mask_full = os.path.join(directory, "extracted", scan_name, f"{source_base}.nii.mask.gz")

                    dst_mask_full = os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{new_edit_number}_backfixed.nii.mask.gz")
                    dst_mask_lossy = os.path.join(directory, "extracted", scan_name, f"{scan_name}_lossy_edit_{new_edit_number}_backfixed.nii.mask.gz")

                    if os.path.isfile(src_mask_full):
                        shutil.copyfile(src_mask_full, dst_mask_full)
                        print(f"Copied full-res mask to: {dst_mask_full}")

                        # Downsample copied full-res mask to create lossy mask
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

                        # load the lossy destination nifti image to get the affine
                        lossy_nifti_img = nib.load(output_path_lossy)
                        lossy_affine = lossy_nifti_img.affine

                        mask_temp = dst_mask_full.replace('.nii.mask.gz', '_mask.nii.gz')
                        lossy_temp = dst_mask_lossy.replace('.nii.mask.gz', '_mask.nii.gz')
                        os.replace(dst_mask_full, mask_temp)
                        try:
                            mask_img = nib.load(mask_temp)
                            mask_dtype = mask_img.get_data_dtype()
                            # Round before astype to avoid truncating floats
                            mask_arr = np.round(mask_img.get_fdata()).astype(mask_dtype)
                            slices = [slice(None, None, resolution_factor) for _ in range(3)]
                            lossy_mask = mask_arr[slices[0], slices[1], slices[2]].astype(mask_dtype, copy=False)
                            nib.save(nib.Nifti1Image(lossy_mask, lossy_affine), lossy_temp)
                            os.replace(lossy_temp, dst_mask_lossy)
                            print(f"Generated lossy mask: {dst_mask_lossy}")
                        finally:
                            os.replace(mask_temp, dst_mask_full)
                    else:
                        print("No source full-res mask found; skipping.")
                except Exception as me:
                    print(f"Error handling paired mask (backfixed): {me}")

                # Copy paired landmark files (non-geometry change) if they exist
                try:
                    if latest_edit >= 0:
                        source_landmark_base = os.path.basename(scan_path).replace('.nii.gz', '')
                    else:
                        source_landmark_base = scan_name
                    
                    source_landmarks_path = os.path.join(directory, "extracted", scan_name, f"{source_landmark_base}_landmarks.json")
                    source_distances_path = os.path.join(directory, "extracted", scan_name, f"{source_landmark_base}_landmark_distances.json")
                    
                    dest_landmark_base = f"{scan_name}_edit_{new_edit_number}_backfixed"
                    dest_landmarks_path = os.path.join(directory, "extracted", scan_name, f"{dest_landmark_base}_landmarks.json")
                    dest_distances_path = os.path.join(directory, "extracted", scan_name, f"{dest_landmark_base}_landmark_distances.json")
                    
                    if os.path.isfile(source_landmarks_path):
                        shutil.copyfile(source_landmarks_path, dest_landmarks_path)
                        print(f"  Copied landmarks to: {dest_landmarks_path}")
                    
                    if os.path.isfile(source_distances_path):
                        shutil.copyfile(source_distances_path, dest_distances_path)
                        print(f"  Copied landmark distances to: {dest_distances_path}")
                except Exception as le:
                    print(f"  Error handling paired landmark files (backfixed): {le}")

                cleanup_memory()

                
            except Exception as e:
                print(f"Error homogenizing {scan_name}: {str(e)}")

        # Send progress update
        channel_layer = get_channel_layer()
        if channel_layer is not None:
            progress = 1  
            async_to_sync(channel_layer.group_send)(
                'progress_group',
                {
                    'type': 'send_progress',
                    'progress': progress,
                    'scan_name': "All scans",
                    'custom_message': f'Finished background homogenization...',
                    'total': scan_count,
                    'current': scan_count,
                }
            )
        
        return Response({"status": "success", "message": "Background homogenization completed"})
    
    def _get_background_value(self, data, current_threshold):
        # Subsample for speed, like before
        vals = data[::2, ::2, ::2]
        thr = float(current_threshold)

        # uint8: discrete mode between 2 and threshold
        if np.issubdtype(vals.dtype, np.integer) and vals.dtype == np.uint8:
            low = 2
            thri = int(np.clip(thr, 0, 255))
            if thri <= low:
                return 0
            mask = (vals >= low) & (vals <= thri)
            if not np.any(mask):
                return 0
            sel = vals[mask].astype(np.int16, copy=False)
            hist = np.bincount(sel, minlength=256)
            # Guard ranges explicitly
            hist[:low] = 0
            if thri + 1 < hist.size:
                hist[thri + 1:] = 0
            return int(np.argmax(hist))

        # uint16: mode bin between 1000 and threshold with 500-step bins
        if np.issubdtype(vals.dtype, np.integer) and vals.dtype == np.uint16:
            low = 1000
            thri = int(np.clip(thr, 0, np.iinfo(np.uint16).max))
            # Fallback if threshold too low
            if thri <= low:
                mask = (vals > 0) & (vals < thri)
                return int(round(float(np.mean(vals[mask])))) if np.any(mask) else 0

            # Build bin edges [1000, 1500, 2000, ..., >= threshold)
            edges = np.arange(low, thri + 500, 500, dtype=np.int64)
            # Ensure at least 1 bin
            if edges.size < 2:
                edges = np.array([low, thri], dtype=np.int64)

            hist, _ = np.histogram(vals, bins=edges)
            if hist.size == 0 or hist.max() == 0:
                # Fallback to mean if no counts
                mask = (vals > 0) & (vals < thri)
                return int(round(float(np.mean(vals[mask])))) if np.any(mask) else 0

            k = int(np.argmax(hist))
            lo, hi = int(edges[k]), int(edges[min(k + 1, edges.size - 1)])
            in_bin = vals[(vals >= lo) & (vals < hi)]
            if in_bin.size == 0:
                return lo  # conservative fallback
            return int(round(float(np.mean(in_bin))))

        # Fallback for other dtypes: original mean in (0, threshold)
        mask = (vals > 0) & (vals < thr)
        return int(round(float(np.mean(vals[mask])))) if np.any(mask) else 0
    
    def _homogenize_background(self, data, target_background):
        # The target background which will be now 0 and everything below this 0 will be set to 0
        # The current background is the peak background value of the current scan

        # Work with original dtype
        original_dtype = data.dtype
        low_bound = np.iinfo(original_dtype).min
        high_bound = np.iinfo(original_dtype).max
        
        # Subtract background value more efficiently
        data = data.astype(np.int32) - int(target_background)
        data = np.clip(data, low_bound, high_bound).astype(np.int32)

        print("done with background subtraction")
        
        # Clip and convert back to original dtype
        return np.clip(data, low_bound, high_bound).astype(original_dtype)





class ApplyThresholdView(APIView):
    def _load_scan_metadata(self, directory, scan_name):
        json_path = os.path.join(directory, "extracted", scan_name, f"{scan_name}.json")
        with open(json_path, "r") as jf:
            return json.load(jf)

    def _is_preserved_mesh_metadata(self, metadata):
        return metadata.get("is_mesh") is True and metadata.get("voxelized") is False

    def _is_voxel_threshold_candidate(self, metadata):
        return not self._is_preserved_mesh_metadata(metadata)

    def post(self, request):
        print("Applying threshold based on reference scan")
        directory = request.data['directory']
        reference = request.data.get('reference')
        method = request.data.get('method', 'same-threshold')  # Default to same-threshold if not specified
        onlyCurrentScan = request.data.get('onlyCurrentScan', False)
        selectedScan = request.data.get('selectedScan', None)
        
        print(f"Using threshold method: {method}")

        ref_required_methods = {"histogram-matching", "same-threshold", "percentile-matching"}
        ref_required = method in ref_required_methods

        extracted_dir = os.path.join(directory, "extracted")
        all_scan_names = [
            d for d in os.listdir(extracted_dir)
            if d != "project_settings.json" and os.path.isdir(os.path.join(extracted_dir, d))
        ]
        all_scan_names.sort()

        metadata_by_scan = {}
        preserved_mesh_scans = []
        voxel_scan_names = []
        faulty_files = []
        for scan_name in all_scan_names:
            try:
                metadata = self._load_scan_metadata(directory, scan_name)
            except (FileNotFoundError, json.JSONDecodeError) as exc:
                print(f"Skipping {scan_name}: could not read metadata ({exc})")
                continue

            metadata_by_scan[scan_name] = metadata
            if metadata.get("faulty", False):
                faulty_files.append(scan_name)
                continue
            if self._is_preserved_mesh_metadata(metadata):
                preserved_mesh_scans.append(scan_name)
                continue
            voxel_scan_names.append(scan_name)

        if not voxel_scan_names:
            return Response(
                {
                    "status": "error",
                    "message": "Align Thresholds only works with voxel-based volumes. This project currently contains only preserved PLY meshes.",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        if ref_required:
            if not reference:
                return Response({"status": "error", "message": "No reference selected."}, status=status.HTTP_400_BAD_REQUEST)
            reference_metadata = metadata_by_scan.get(reference)
            if reference_metadata is None:
                try:
                    reference_metadata = self._load_scan_metadata(directory, reference)
                    metadata_by_scan[reference] = reference_metadata
                except (FileNotFoundError, json.JSONDecodeError):
                    return Response({"status": "error", "message": f"Reference scan '{reference}' was not found."}, status=status.HTTP_400_BAD_REQUEST)
            if self._is_preserved_mesh_metadata(reference_metadata):
                return Response(
                    {
                        "status": "error",
                        "message": "This threshold method requires a voxel-based reference. The selected reference is a preserved PLY mesh.",
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

        scan_names = [
            scan_name for scan_name in voxel_scan_names
            if not has_elastic_registration(scan_name, directory)
        ]

        if reference in scan_names:
            scan_names.remove(reference)

        # Filter to only current scan if requested
        if onlyCurrentScan and selectedScan:
            selected_metadata = metadata_by_scan.get(selectedScan)
            if selected_metadata is not None and self._is_preserved_mesh_metadata(selected_metadata):
                return Response(
                    {
                        "status": "error",
                        "message": f"Selected scan '{selectedScan}' is a preserved PLY mesh. Align Thresholds only works with voxel-based volumes.",
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )
            if selectedScan in scan_names:
                scan_names = [selectedScan]
                print(f"Filtering to apply threshold only to current scan: {selectedScan}")
            else:
                return Response(
                    {"status": "error", "message": f"Selected scan '{selectedScan}' is not an eligible voxel-based scan for thresholding"},
                    status=status.HTTP_400_BAD_REQUEST
                )
        elif onlyCurrentScan and not selectedScan:
            return Response(
                {"status": "error", "message": "onlyCurrentScan flag set but no selectedScan provided"},
                status=status.HTTP_400_BAD_REQUEST
            )

        flag_filter = normalize_flag_filter_value(
            request.data.get('flagFilter', 'off') if request.data else 'off',
            only_current_scan=onlyCurrentScan,
        )
        flagged_set = load_flagged_subject_names(directory)
        scan_names = apply_flag_filter(scan_names, flagged_set, flag_filter)
        if not (onlyCurrentScan and selectedScan):
            scan_names = filter_out_linked_children(directory, scan_names)
        print(f"Apply threshold flag filter: {flag_filter}, remaining scans: {len(scan_names)}")

        if not scan_names:
            return Response(
                {
                    "status": "error",
                    "message": "No eligible voxel-based target scans were found. Preserved PLY meshes are skipped.",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        reference_metadata = metadata_by_scan.get(reference) if reference else None
        reference_threshold = reference_metadata.get('threshold') if reference_metadata else None
        if ref_required:
            print(f"Reference threshold: {reference_threshold}")

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
                    'scan_name': reference if ref_required else scan_names[0],
                    'custom_message': (
                        f'Analyzing reference scan {reference} with a threshold of {reference_threshold} for thresholding...'
                        if ref_required else
                        f'Preparing per-scan thresholding for {len(scan_names)} voxel-based scan(s)...'
                    ),
                    'total': total_scans,
                    'current': 0,
                }
            )
        
        # First, analyze the reference scan to determine threshold parameters
        try:
            reference_data = None
            if ref_required:
                # Find the latest edit of the reference scan
                reference_edit_number = 0
                while glob.glob(os.path.join(directory, "extracted", reference, f"{reference}_edit_{reference_edit_number}_*.nii.gz")):
                    reference_edit_number += 1
                reference_edit_number -= 1  # Go back to the last valid edit number
                
                # Load the reference scan data
                if reference_edit_number >= 0:
                    reference_path = glob.glob(os.path.join(directory, "extracted", reference, f"{reference}_edit_{reference_edit_number}_*.nii.gz"))[0]
                else:
                    reference_path = os.path.join(directory, "extracted", reference, f"{reference}.nii.gz")
                
                reference_img = nib.load(reference_path)
                reference_original_dtype = reference_img.get_data_dtype()
                reference_data = reference_img.get_fdata().astype(reference_original_dtype)
                reference_data = gaussian_filter(reference_data, sigma=1.0)
                
                # Get reference threshold from metadata, falling back to Otsu if absent.
                if reference_metadata and 'threshold' in reference_metadata:
                    reference_threshold = reference_metadata['threshold']
                    print(f"Using existing threshold from reference metadata: {reference_threshold}")
                else:
                    reference_threshold = threshold_otsu(reference_data)
                    print(f"No threshold in metadata, calculated with Otsu: {reference_threshold}")
                
                reference_binary = reference_data > reference_threshold
                reference_volume = np.sum(reference_binary)
                reference_variance = np.var(reference_data[reference_binary])
                
                print(f"Reference threshold: {reference_threshold}")
                print(f"Reference volume: {reference_volume}")
                print(f"Reference variance: {reference_variance}")
            
            # Process each scan
            for idx, scan_name in enumerate(scan_names):
                try:
                    # Find the latest edit for this scan
                    latest_edit = 0
                    while glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{latest_edit}_*.nii.gz")):
                        latest_edit += 1
                    latest_edit -= 1
                    
                    # Send progress update
                    if channel_layer is not None:
                        progress = ((idx) / total_scans)
                        async_to_sync(channel_layer.group_send)(
                            'progress_group',
                            {
                                'type': 'send_progress',
                                'progress': progress,
                                'scan_name': scan_name,
                                'custom_message': f'Applying threshold to {scan_name}...',
                                'total': total_scans,
                                'current': idx + 1,
                            }
                        )
                    
                    # Load the scan data
                    if latest_edit >= 0:
                        scan_path = glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{latest_edit}_*.nii.gz"))[0]
                    else:
                        scan_path = os.path.join(directory, "extracted", scan_name, f"{scan_name}.nii.gz")
                    
                    # Load the scan's metadata
                    with open(os.path.join(directory, "extracted", scan_name, f"{scan_name}.json"), 'r') as jf:
                        scan_metadata = json.load(jf)
                    
                    if method != 'same-threshold':
                        # Load the scan data
                        nifti_img = nib.load(scan_path)
                        scan_original_dtype = nifti_img.get_data_dtype()
                        scan_data = nifti_img.get_fdata().astype(scan_original_dtype)
                        scan_data = gaussian_filter(scan_data, sigma=1.0)
                    

                    
                    # Apply preprocessing based on method
                    if method == 'histogram-matching':
                        # Calculate memory requirements and available memory
                        threshold_value = self._find_histogram_matching_threshold(scan_data, reference_data, reference_threshold)
                    elif method == 'otsus-method':
                        # Calculate threshold using Otsu's method for each scan individually
                        threshold_value = threshold_otsu(scan_data)
                    elif method == 'li-method':
                        # Calculate threshold using Li's method for each scan individually
                        threshold_value = self._find_li_threshold(scan_data)
                    elif method == 'kapur-entropy':
                        # Calculate threshold using Kapur's entropy method for each scan individually
                        threshold_value = self._find_kapur_threshold(scan_data)
                    elif method == 'same-threshold':
                        threshold_value = reference_threshold    
                    elif method == 'percentile-matching':
                        # Find the percentile matching threshold
                        threshold_value = self._find_percentile_matching_threshold(scan_data, reference_data, reference_threshold)
                    else:
                        raise ValueError(f"Invalid threshold method: {method}")
                                        
                    # Update metadata with the new threshold
                    if method != "same-threshold":
                        if threshold_value < np.min(scan_data):
                            threshold_value = np.min(scan_data)
                    scan_metadata['threshold'] = float(threshold_value)
                    print(f"New threshold for {scan_name}: {threshold_value}")
                    with open(os.path.join(directory, "extracted", scan_name, f"{scan_name}.json"), 'w') as jf:
                        json.dump(scan_metadata, jf, indent=4)
                    
                    print(f"Applied threshold of {threshold_value} to {scan_name}")
                    
                except Exception as e:
                    print(f"Error processing {scan_name}: {str(e)}")
            
            return Response({"status": "success", "message": "Threshold application completed"})
            
        except Exception as e:
            print(f"Error analyzing reference scan: {str(e)}")
            return Response({"status": "error", "message": f"Error analyzing reference scan: {str(e)}"}, 
                          status=status.HTTP_500_INTERNAL_SERVER_ERROR)


    def _find_li_threshold(self, data):
        try:            
            # The objective function of Li's method is not always convex.
            # Using a 95th percentile initial guess ensures the algorithm 
            # doesn't get stuck in a local minimum.
            initial_guess = np.percentile(data, 95)
            return int(threshold_li(data, initial_guess=initial_guess))
        except Exception as e:
            print(f"Li thresholding failed: {e}.")    
            return "Li thresholding failed" + str(e)        

    def _find_percentile_matching_threshold(self, scan_data, reference_data, reference_threshold):
        # Find the percentile rank of the reference threshold in the reference data
        # This gives us what percentile the threshold represents in the reference scan
        percentile_rank = (np.sum(reference_data <= reference_threshold) / reference_data.size) * 100
        
        # Now find the threshold value that corresponds to the same percentile in the scan data
        matching_threshold = np.percentile(scan_data, percentile_rank)
        print(f"Reference threshold {reference_threshold} is at {percentile_rank:.2f} percentile")
        print(f"Matching threshold: {matching_threshold}")
        return matching_threshold
    
    

    def _find_histogram_matching_threshold(self, scan_data, reference_data, reference_threshold):
        # Histogram matching only needs memory for the arrays themselves plus small histogram buffers
        required_memory = (scan_data.size + reference_data.size) * 8 / (1024**3)  # Memory in GB
        available_memory = psutil.virtual_memory().available / (1024**3)  # Available memory in GB
        print(f"Required memory: {required_memory:.2f} GB, Available: {available_memory:.2f} GB")
        
        scan_hist, scan_bins = np.histogram(scan_data[::2, ::2, ::2], bins=64)
        ref_hist, ref_bins = np.histogram(reference_data[::2, ::2, ::2], bins=64)

        # Ignore all idxs below otsu threshold
        ref_otsu_threshold = threshold_otsu(reference_data)
        ref_hist[0:np.argmin(np.abs(ref_bins - ref_otsu_threshold))] = np.min(reference_data)
        scan_otsu_threshold = threshold_otsu(scan_data)
        scan_hist[0:np.argmin(np.abs(scan_bins - scan_otsu_threshold))] = np.min(scan_data)

        # Find highest frequency peak in each histogram
        scan_peak_idx = np.argmax(scan_hist)
        ref_peak_idx = np.argmax(ref_hist)

        # Calculate bin centers
        scan_bin_centers = (scan_bins[:-1] + scan_bins[1:]) / 2
        ref_bin_centers = (ref_bins[:-1] + ref_bins[1:]) / 2
        
        # Calculate intensity values at peak locations
        scan_peak_value = scan_bin_centers[scan_peak_idx]
        ref_peak_value = ref_bin_centers[ref_peak_idx]            
        
        # Calculate shift between peaks
        intensity_shift = ref_peak_value - scan_peak_value
        print(f"Scan peak: {scan_peak_value}, Ref peak: {ref_peak_value}, Shift: {intensity_shift}")
        
        # Apply shift to reference threshold
        threshold_value = reference_threshold - intensity_shift
        if threshold_value < np.min(scan_data):
            threshold_value = np.min(scan_data)

        print(f"Adjusted threshold: {threshold_value} (original: {reference_threshold})")

        """
        # Find the two highest peaks in each histogram
        # Use signal.find_peaks for more robust peak detection
        scan_peaks, _ = signal.find_peaks(scan_hist, distance=7)
        ref_peaks, _ = signal.find_peaks(ref_hist, distance=7)
        
        if len(scan_peaks) == 0 or len(ref_peaks) == 0:
            print("Warning: Could not find peaks in histograms, using original threshold")
            threshold_value = reference_threshold
        else:
            # Find the first two highest peaks in each histogram.
            scan_peaks = scan_peaks[np.argsort(scan_hist[scan_peaks])[0:2]]
            ref_peaks = ref_peaks[np.argsort(ref_hist[ref_peaks])[0:2]]

            print(scan_hist[scan_peaks])
            print(ref_hist[ref_peaks])

            print(scan_peaks)
            print(ref_peaks)

            # The highest intensity value (not frequency) should be our reference of signal to threshold gap, even if it's not the second highest peak       
            scan_peak_idx = scan_peaks[0] if scan_peaks[0] > scan_peaks[1] else scan_peaks[1]
            ref_peak_idx = ref_peaks[0] if ref_peaks[0] > ref_peaks[1] else ref_peaks[1]
            
            # Calculate bin centers
            scan_bin_centers = (scan_bins[:-1] + scan_bins[1:]) / 2
            ref_bin_centers = (ref_bins[:-1] + ref_bins[1:]) / 2
            
            # Calculate intensity values at peak locations
            scan_peak_value = scan_bin_centers[scan_peak_idx]
            ref_peak_value = ref_bin_centers[ref_peak_idx]            
            
            # Calculate shift between peaks
            intensity_shift = ref_peak_value - scan_peak_value
            print(f"Scan peak: {scan_peak_value}, Ref peak: {ref_peak_value}, Shift: {intensity_shift}")
            
            # Apply shift to reference threshold
            threshold_value = reference_threshold - intensity_shift
            print(f"Adjusted threshold: {threshold_value} (original: {reference_threshold})")
        """
        
        # Clean up large arrays
        del scan_hist, ref_hist
        
        # Force garbage collection
        cleanup_memory()

        return int(threshold_value)

    def _find_kapur_threshold(self, scan_data):
        """
        Kapur entropy-based thresholding (Kapur, Sahoo & Wong, 1985).
        Finds the threshold that maximises the sum of Shannon entropies
        of the background and foreground distributions.
        Fully vectorised — O(bins) after the histogram is built.
        """
        # Subsample to save memory on large 3-D volumes (every other voxel)
        hist, bin_edges = np.histogram(scan_data[::2, ::2, ::2].ravel(), bins=256)

        # Normalised probability distribution
        prob = hist.astype(np.float64)
        prob /= prob.sum()

        # Cumulative probability (background weight at each threshold)
        c_prob = prob.cumsum()
        c_prob_fg = 1.0 - c_prob

        # Guard: avoid log(0) and division by zero
        eps = 1e-10
        c_prob    = np.clip(c_prob,    eps, 1.0)
        c_prob_fg = np.clip(c_prob_fg, eps, 1.0)

        # Cumulative raw entropy terms: cumsum of  -p * log(p)
        # (using 0 * log(0) → 0 convention)
        raw_ent = np.where(prob > eps, -prob * np.log(prob), 0.0)
        c_ent = raw_ent.cumsum()
        c_ent_fg = c_ent[-1] - c_ent

        # Per-threshold entropy of each partition
        h_bg = c_ent    / c_prob    + np.log(c_prob)
        h_fg = c_ent_fg / c_prob_fg + np.log(c_prob_fg)

        best_t = np.argmax(h_bg + h_fg)

        # Map bin index back to actual intensity value
        threshold_value = float(bin_edges[best_t + 1])
        print(f"Kapur entropy threshold: {threshold_value} (bin {best_t})")
        return threshold_value

class RemoveBackgroundView(APIView):
    def _apply_backremove_mask_to_sibling(self, directory, sibling_name, removal_mask_path, mask_affine, threshold, min_value):
        """Apply a removal mask from background removal to a same-shape linked sibling scan."""
        # Skip if has elastic registration
        if has_elastic_registration(sibling_name, directory):
            print(f"      Skipping {sibling_name}: elastic registration present")
            return {'scan_name': sibling_name, 'status': 'skipped', 'reason': 'elastic registration present'}
        
        # Find latest non-elastic edit
        latest_edit = 0
        while glob.glob(os.path.join(directory, "extracted", sibling_name, f"{sibling_name}_edit_{latest_edit}_*.nii.gz")):
            latest_edit += 1
        latest_edit -= 1
        
        if latest_edit >= 0:
            edit_files = glob.glob(os.path.join(directory, "extracted", sibling_name, f"{sibling_name}_edit_{latest_edit}_*.nii.gz"))
            if edit_files:
                edit_file = os.path.basename(edit_files[0])
                if 'elastic' in edit_file.lower():
                    print(f"      Skipping {sibling_name}: latest edit is elastic")
                    return {'scan_name': sibling_name, 'status': 'skipped', 'reason': 'latest edit is elastic'}
                nifti_path = os.path.join(directory, "extracted", sibling_name, edit_file)
            else:
                nifti_path = os.path.join(directory, "extracted", sibling_name, f"{sibling_name}.nii.gz")
        else:
            nifti_path = os.path.join(directory, "extracted", sibling_name, f"{sibling_name}.nii.gz")
        
        # Load removal mask
        removal_mask_temp = removal_mask_path.replace('.nii.removal_mask.gz', '_removal_mask.nii.gz')
        os.replace(removal_mask_path, removal_mask_temp)
        try:
            mask_img = nib.load(removal_mask_temp)
            removal_mask = np.round(mask_img.get_fdata()).astype(bool)
        finally:
            os.replace(removal_mask_temp, removal_mask_path)
        
        # Load sibling nifti
        sibling_img = nib.load(nifti_path)
        sibling_dtype = sibling_img.get_data_dtype()
        sibling_data = sibling_img.get_fdata().astype(sibling_dtype)
        
        # Check shape match
        if sibling_data.shape != removal_mask.shape:
            print(f"      Skipping {sibling_name}: shape mismatch: {sibling_data.shape} vs {removal_mask.shape}")
            return {
                'scan_name': sibling_name,
                'status': 'skipped',
                'reason': f'shape mismatch: {sibling_data.shape} vs {removal_mask.shape}',
            }
        
        # Load sibling metadata
        json_path = os.path.join(directory, "extracted", sibling_name, f"{sibling_name}.json")
        with open(json_path, 'r') as jf:
            sibling_metadata = json.load(jf)
        
        # Get minimum value for sibling
        sibling_min_value = np.min(sibling_data)
        
        # Apply removal mask to sibling
        sibling_data[removal_mask] = sibling_min_value
        
        # Find next edit number for sibling
        new_edit_number = 0
        while glob.glob(os.path.join(directory, "extracted", sibling_name, f"{sibling_name}_edit_{new_edit_number}_*.nii.gz")):
            new_edit_number += 1
        
        # Save backremoved sibling
        new_nifti = nib.Nifti1Image(sibling_data, sibling_img.affine)
        output_path = os.path.join(directory, "extracted", sibling_name, f"{sibling_name}_edit_{new_edit_number}_backremoved.nii.gz")
        nib.save(new_nifti, output_path)
        
        # Save lossy version
        RegistrationTools().save_as_lossy_nifti(
            sibling_data,
            sibling_metadata['voxel_size'],
            json_path,
            os.path.join(directory, "extracted", sibling_name, f"{sibling_name}_lossy_edit_{new_edit_number}_backremoved.nii.gz")
        )
        
        # Copy paired mask if exists
        try:
            if latest_edit >= 0:
                source_base = os.path.basename(nifti_path).replace('.nii.gz', '')
            else:
                source_base = sibling_name
            src_mask_full = os.path.join(directory, "extracted", sibling_name, f"{source_base}.nii.mask.gz")
            if os.path.isfile(src_mask_full):
                dst_mask_full = os.path.join(directory, "extracted", sibling_name, f"{sibling_name}_edit_{new_edit_number}_backremoved.nii.mask.gz")
                shutil.copyfile(src_mask_full, dst_mask_full)
        except Exception as me:
            print(f"      Error copying paired mask for {sibling_name}: {me}")
        
        # Copy paired landmarks if exist
        try:
            if latest_edit >= 0:
                source_landmark_base = os.path.basename(nifti_path).replace('.nii.gz', '')
            else:
                source_landmark_base = sibling_name
            source_landmarks_path = os.path.join(directory, "extracted", sibling_name, f"{source_landmark_base}_landmarks.json")
            if os.path.isfile(source_landmarks_path):
                dest_landmarks_path = os.path.join(directory, "extracted", sibling_name, f"{sibling_name}_edit_{new_edit_number}_backremoved_landmarks.json")
                shutil.copyfile(source_landmarks_path, dest_landmarks_path)
        except Exception as le:
            print(f"      Error copying landmarks for {sibling_name}: {le}")

        return {
            'scan_name': sibling_name,
            'status': 'success',
            'edit': f"{sibling_name}_lossy_edit_{new_edit_number}_backremoved.nii.gz",
        }
    
    def remove_background(self, directory, reference, flag_filter='off', propagate_linked=False):
        print(f"Removing background for all scans in directory: {directory}")
        linked_propagation_results = []

        # Make a list of faulty files
        faulty_files = []
        for file in os.listdir(os.path.join(directory, "extracted")):
            # Skip project_settings.json file
            if file == "project_settings.json" or not os.path.isdir(os.path.join(directory, "extracted", file)):
                continue
            json_path = os.path.join(directory, "extracted", file, f"{file}.json")
            with open(json_path, 'r') as jf:
                metadata = json.load(jf)
                if metadata.get('faulty', False):
                    faulty_files.append(file)
        
        # Find all unique scan names by listing subdirectories in extracted folder
        scan_names = [d for d in os.listdir(os.path.join(directory, "extracted")) 
                     if os.path.isdir(os.path.join(directory, "extracted", d)) and d not in faulty_files]

        scan_names, skipped_preserved_meshes = partition_voxel_and_preserved_mesh_scans(directory, scan_names)
        if not scan_names:
            raise ValueError(voxel_only_all_meshes_message('Remove Background'))

        scan_names.sort()

        print(f"Scan names: {scan_names}")
        print(f"Reference: {reference}")

        # Remove reference from scan_names
        if reference in scan_names:
            scan_names.remove(reference)
            print(f"Reference removed from scan_names: {reference}")
                
        # Iterate over every scan and if the latest edit is already background removed, remove it from scan_names
        scans_to_remove = []
        for scan_name in scan_names:
            # Find the latest edit for this scan
            latest_edit = 0
            while len(glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{latest_edit}_*"))) > 0:
                latest_edit += 1
            latest_edit -= 1
            
            # Remove from scan_names if the latest edit is already background removed
            if latest_edit >= 0:
                for edit in range(0, latest_edit+1):
                    latest_edit_files = glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{edit}_*"))
                    if any("backremoved" in file for file in latest_edit_files):
                        print(f"Skipping {scan_name} as it's already background removed")
                        scans_to_remove.append(scan_name)
                        break
            
            # Remove from scan_names if it has elastic registration
            if has_elastic_registration(scan_name, directory):
                scans_to_remove.append(scan_name)
        
        # Remove scans that are already aligned
        for scan_name in scans_to_remove:
            if scan_name in scan_names:
                scan_names.remove(scan_name)
            else:
                print(f"Scan {scan_name} not found in scan_names")
        
        print(f"After filtering, {len(scan_names)} scans need background removal")

        flagged_set = load_flagged_subject_names(directory)
        scan_names = apply_flag_filter(scan_names, flagged_set, flag_filter)
        scan_names = filter_out_linked_children(directory, scan_names)
        print(f"Remove background flag filter: {flag_filter}, remaining scans: {len(scan_names)}")

        if not scan_names:
            raise ValueError(voxel_only_no_eligible_targets_message('Remove Background'))
        if skipped_preserved_meshes:
            print(f"Skipping preserved mesh subjects for background removal: {skipped_preserved_meshes}")

        # Initialize WebSocket progress tracking
        total_scans = len(scan_names)
        channel_layer = get_channel_layer()
        if channel_layer is not None and total_scans > 0:
            print("Sending initial progress update")
            async_to_sync(channel_layer.group_send)(
                'progress_group',
                {
                    'type': 'send_progress',
                    'progress': 0,
                    'scan_name': 'All scans',
                    'custom_message': f'Loading scans for background removal...',
                    'total': total_scans,
                    'current': 0,
                }
            )

        for idx, scan_name in enumerate(scan_names):
            try:
                # Update progress
                if channel_layer is not None:
                    progress = idx / total_scans
                    async_to_sync(channel_layer.group_send)(
                        'progress_group',
                        {
                            'type': 'send_progress',
                            'progress': progress,
                            'scan_name': scan_name,
                            'custom_message': f'Removing background from {scan_name}...',
                            'total': total_scans,
                            'current': idx + 1,
                        }
                    )
                
                # Find the latest edit for this scan
                latest_edit = 0
                while glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{latest_edit}_*.nii.gz")):
                    latest_edit += 1
                latest_edit -= 1
                
                # Determine the file to process
                if latest_edit >= 0:
                    edit_files = glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{latest_edit}_*.nii.gz"))
                    if edit_files:
                        edit_file = os.path.basename(edit_files[0])
                        nifti_path = os.path.join(directory, "extracted", scan_name, edit_file)
                    else:
                        nifti_path = os.path.join(directory, "extracted", scan_name, f"{scan_name}.nii.gz")
                else:
                    nifti_path = os.path.join(directory, "extracted", scan_name, f"{scan_name}.nii.gz")
                
                # Load metadata to get threshold
                json_path = os.path.join(directory, "extracted", scan_name, f"{scan_name}.json")
                with open(json_path, 'r') as jf:
                    scan_metadata = json.load(jf)
                
                threshold = scan_metadata['threshold']
                print(f"Processing {scan_name} with threshold {threshold}")
                
                # Load the scan data
                nifti_img = nib.load(nifti_path)
                original_dtype = nifti_img.get_data_dtype()
                nifti_data = nifti_img.get_fdata().astype(original_dtype)
                
                # Compute removal mask BEFORE modifying nifti_data (True where background will be removed)
                removal_mask = (nifti_data < threshold)
                
                # Get the minimum value for the current scan
                min_value = np.min(nifti_data)
                
                # Set background (values below threshold) to minimum value
                nifti_data[removal_mask] = min_value
                
                # Create new nifti image
                new_nifti = nib.Nifti1Image(nifti_data, nifti_img.affine)
                
                # Save the processed image with new edit number
                new_edit = latest_edit + 1
                output_path = os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{new_edit}_backremoved.nii.gz")
                nib.save(new_nifti, output_path)
                
                # Save the lossy version
                lossy_path = os.path.join(directory, "extracted", scan_name, f"{scan_name}_lossy_edit_{new_edit}_backremoved.nii.gz")
                RegistrationTools().save_as_lossy_nifti(
                    nifti_data,
                    scan_metadata['voxel_size'],
                    json_path,
                    lossy_path
                )
                
                # Save removal mask
                removal_mask_path = None
                if removal_mask is not None:
                    removal_mask_uint8 = removal_mask.astype(np.uint8)
                    removal_mask_nifti = nib.Nifti1Image(removal_mask_uint8, nifti_img.affine)
                    removal_mask_temp = os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{new_edit}_backremoved_removal_mask.nii.gz")
                    removal_mask_path = os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{new_edit}_backremoved.nii.removal_mask.gz")
                    nib.save(removal_mask_nifti, removal_mask_temp)
                    os.replace(removal_mask_temp, removal_mask_path)
                    print(f"  Saved removal mask to: {removal_mask_path}")
                
                # Copy paired mask (non-geometry change) and generate lossy by downsampling from copied full-res
                try:
                    source_base = os.path.basename(nifti_path).replace('.nii.gz', '') if latest_edit >= 0 else scan_name
                    src_mask_full = os.path.join(directory, "extracted", scan_name, f"{source_base}.nii.mask.gz")
                    dst_mask_full = os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{new_edit}_backremoved.nii.mask.gz")
                    dst_mask_lossy = os.path.join(directory, "extracted", scan_name, f"{scan_name}_lossy_edit_{new_edit}_backremoved.nii.mask.gz")

                    if os.path.isfile(src_mask_full):
                        shutil.copyfile(src_mask_full, dst_mask_full)
                        print(f"  Copied full-res mask to: {dst_mask_full}")

                        # Downsample copied full-res mask to create lossy mask
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

                        # load the lossy destination nifti image to get the affine
                        lossy_nifti_img = nib.load(lossy_path)
                        lossy_affine = lossy_nifti_img.affine

                        mask_temp = dst_mask_full.replace('.nii.mask.gz', '_mask.nii.gz')
                        lossy_temp = dst_mask_lossy.replace('.nii.mask.gz', '_mask.nii.gz')
                        os.replace(dst_mask_full, mask_temp)
                        try:
                            mask_img = nib.load(mask_temp)
                            mask_dtype = mask_img.get_data_dtype()
                            # Round before astype to avoid truncating floats
                            mask_arr = np.round(mask_img.get_fdata()).astype(mask_dtype)
                            slices = [slice(None, None, resolution_factor) for _ in range(3)]
                            lossy_mask = mask_arr[slices[0], slices[1], slices[2]].astype(mask_dtype, copy=False)
                            nib.save(nib.Nifti1Image(lossy_mask, lossy_affine), lossy_temp)
                            os.replace(lossy_temp, dst_mask_lossy)
                            print(f"  Generated lossy mask: {dst_mask_lossy}")
                        finally:
                            os.replace(mask_temp, dst_mask_full)
                    else:
                        print("  No source full-res mask found; skipping.")
                except Exception as me:
                    print(f"  Error handling paired mask (backremoved): {me}")
                
                # Copy paired landmark files (non-geometry change) if they exist
                try:
                    if latest_edit >= 0:
                        source_landmark_base = os.path.basename(nifti_path).replace('.nii.gz', '')
                    else:
                        source_landmark_base = scan_name
                    
                    source_landmarks_path = os.path.join(directory, "extracted", scan_name, f"{source_landmark_base}_landmarks.json")
                    source_distances_path = os.path.join(directory, "extracted", scan_name, f"{source_landmark_base}_landmark_distances.json")
                    
                    dest_landmark_base = f"{scan_name}_edit_{new_edit}_backremoved"
                    dest_landmarks_path = os.path.join(directory, "extracted", scan_name, f"{dest_landmark_base}_landmarks.json")
                    dest_distances_path = os.path.join(directory, "extracted", scan_name, f"{dest_landmark_base}_landmark_distances.json")
                    
                    if os.path.isfile(source_landmarks_path):
                        shutil.copyfile(source_landmarks_path, dest_landmarks_path)
                        print(f"  Copied landmarks to: {dest_landmarks_path}")
                    
                    if os.path.isfile(source_distances_path):
                        shutil.copyfile(source_distances_path, dest_distances_path)
                        print(f"  Copied landmark distances to: {dest_distances_path}")
                except Exception as le:
                    print(f"  Error handling paired landmark files (backremoved): {le}")
                
                # Propagate removal mask to same-shape linked children if requested
                if propagate_linked and removal_mask_path and os.path.isfile(removal_mask_path):
                    try:
                        siblings, err = get_same_shape_siblings(directory, scan_name)
                        if err:
                            print(f"  Could not get same-shape siblings for {scan_name}: {err}")
                        elif siblings:
                            print(f"  Propagating background removal from {scan_name} to {len(siblings)} same-shape linked children")
                            for sibling_name in siblings:
                                try:
                                    sib_result = self._apply_backremove_mask_to_sibling(
                                        directory, sibling_name, removal_mask_path, nifti_img.affine, threshold, min_value
                                    )
                                    if isinstance(sib_result, dict):
                                        linked_propagation_results.append(sib_result)
                                    else:
                                        linked_propagation_results.append({
                                            'scan_name': sibling_name,
                                            'status': 'success',
                                        })
                                    print(f"    Propagated to {sibling_name}")
                                except Exception as sib_err:
                                    linked_propagation_results.append({
                                        'scan_name': sibling_name,
                                        'status': 'failed',
                                        'reason': str(sib_err),
                                    })
                                    print(f"    Failed to propagate to {sibling_name}: {sib_err}")
                        else:
                            print(f"  No same-shape linked children found for {scan_name}")
                    except Exception as prop_err:
                        print(f"  Error during linked propagation for {scan_name}: {prop_err}")
                
                print(f"Successfully removed background from {scan_name}")
                
                # Clean up memory after each scan
                cleanup_memory()
                                
            except Exception as e:
                print(f"Error processing {scan_name}: {str(e)}")
        
        # Final progress update
        if channel_layer is not None:
            async_to_sync(channel_layer.group_send)(
                'progress_group',
                {
                    'type': 'send_progress',
                    'progress': 1.0,
                    'scan_name': 'All scans',
                    'custom_message': 'Background removal completed for all scans',
                    'total': total_scans,
                    'current': total_scans,
                }
            )
        return linked_propagation_results if propagate_linked else None
    
    def post(self, request, format=None):
        directory = request.data.get('directory')
        reference = request.data.get('reference')
        flag_filter = normalize_flag_filter_value(
            request.data.get('flagFilter', 'off') if request.data else 'off',
            only_current_scan=False,
        )
        # Linking is the opt-in: children skipped by the batch inherit the main's result.
        propagate_linked = bool(request.data.get('propagate_linked', True))
        try:
            linked_propagation = self.remove_background(directory, reference, flag_filter, propagate_linked)
        except ValueError as e:
            return Response({"status": "error", "message": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        return Response({
            "status": "success",
            "message": "Background removal completed",
            "linked_propagation": linked_propagation,
        })

class MatchHistogramView(APIView):
    def match_histogram_full(self, directory, reference, scan_names, reference_data, reference_threshold, reference_metadata, total_scans, channel_layer):
        """Full histogram matching (original approach)"""
        # Create reference foreground mask
        reference_mask = reference_data > reference_threshold

        for idx, scan_name in enumerate(scan_names):
            try:
                # Update progress
                if channel_layer is not None:
                    progress = idx / total_scans
                    async_to_sync(channel_layer.group_send)(
                        'progress_group',
                        {
                            'type': 'send_progress',
                            'progress': progress,
                            'scan_name': scan_name,
                            'custom_message': f'Matching histogram of {scan_name} to {reference}...',
                            'total': total_scans,
                            'current': idx + 1,
                        }
                    )
                
                print(f"Processing {scan_name} ({idx+1}/{total_scans})")
                
                # Load scan
                latest_edit = 0
                while glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{latest_edit}_*.nii.gz")):
                    latest_edit += 1
                latest_edit -= 1
                scan_path = glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{latest_edit}_*.nii.gz"))[0]
                print(f"Using scan file: {os.path.basename(scan_path)}")
                
                scan_img = nib.load(scan_path)
                scan_affine = scan_img.affine
                scan_original_dtype = scan_img.get_data_dtype()
                scan_data = scan_img.get_fdata().astype(scan_original_dtype)

                # Get the scan threshold from metadata
                with open(os.path.join(directory, "extracted", scan_name, f"{scan_name}.json"), 'r') as jf:
                    scan_metadata = json.load(jf)
                scan_threshold = scan_metadata['threshold']
                
                # Create scan foreground mask
                scan_mask = scan_data > scan_threshold
                
                # Create a copy of the scan data to modify
                matched_data = scan_data.copy()
                
                # Apply histogram matching only to foreground voxels
                print(f"Matching histogram of {scan_name} to {reference} (foreground only)")
                ref_foreground = reference_data[reference_mask]
                scan_foreground = scan_data[scan_mask]
                
                if scan_foreground.size > 0 and ref_foreground.size > 0:
                    # Match histograms of foreground regions only
                    matched_foreground = match_histograms(scan_foreground, ref_foreground, channel_axis=None)
                    
                    # Replace foreground voxels in the output with the matched values
                    matched_data[scan_mask] = matched_foreground
                else:
                    print(f"Warning: Empty foreground in either reference or scan {scan_name}")

                print("Completed histogram matching, saving...")

                # Save results using consistent helper
                self._save_processed_scan(directory, scan_name, matched_data, scan_affine, latest_edit + 1, 
                                        "histmatched", reference_metadata, scan_metadata)
                
                # Clean up memory after each scan
                del scan_data, scan_img, matched_data
                cleanup_memory()
                
            except Exception as e:
                print(f"Error processing {scan_name}: {str(e)}")

    def match_histogram_align_peaks(self, directory, reference, scan_names, reference_data, reference_threshold, reference_metadata, total_scans, channel_layer):
        """Align intensity values by matching thresholds"""
        print(f"Reference threshold for alignment: {reference_threshold}")
        
        for idx, scan_name in enumerate(scan_names):
            try:
                # Update progress
                if channel_layer is not None:
                    progress = idx / total_scans
                    async_to_sync(channel_layer.group_send)(
                        'progress_group',
                        {
                            'type': 'send_progress',
                            'progress': progress,
                            'scan_name': scan_name,
                            'custom_message': f'Aligning threshold of {scan_name} to {reference}...',
                            'total': total_scans,
                            'current': idx + 1,
                        }
                    )
                
                print(f"Processing {scan_name} ({idx+1}/{total_scans})")
                
                # Load scan
                latest_edit = 0
                while glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{latest_edit}_*.nii.gz")):
                    latest_edit += 1
                latest_edit -= 1
                scan_path = glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{latest_edit}_*.nii.gz"))[0]
                
                scan_img = nib.load(scan_path)
                scan_affine = scan_img.affine
                scan_original_dtype = scan_img.get_data_dtype()
                scan_data = scan_img.get_fdata().astype(scan_original_dtype)
                
                # Get scan threshold
                with open(os.path.join(directory, "extracted", scan_name, f"{scan_name}.json"), 'r') as jf:
                    scan_metadata = json.load(jf)
                scan_threshold = scan_metadata['threshold']
                
                # Calculate shift needed to align thresholds
                shift = reference_threshold - scan_threshold
                print(f"Scan {scan_name} threshold: {scan_threshold}, reference threshold: {reference_threshold}, shift: {shift}")
                
                # Apply linear shift to ALL values
                aligned_data = scan_data.astype(np.float32) + shift
                
                # Determine valid range based on data type
                if np.issubdtype(scan_original_dtype, np.integer):
                    min_val = 0
                    max_val = np.iinfo(scan_original_dtype).max
                else:
                    min_val = 0.0
                    max_val = np.finfo(scan_original_dtype).max
                
                # Clip values to valid range
                aligned_data = np.clip(aligned_data, min_val, max_val)
                aligned_data = aligned_data.astype(scan_original_dtype)
                
                print(f"Values clipped to range [{min_val}, {max_val}]")
                
                # Save results using consistent helper
                self._save_processed_scan(directory, scan_name, aligned_data, scan_affine, latest_edit + 1, 
                                        "histmatched", reference_metadata, scan_metadata)
                
                # Clean up
                del scan_data, scan_img, aligned_data
                cleanup_memory()
                
            except Exception as e:
                print(f"Error processing {scan_name}: {str(e)}")
    
    def match_histogram_tissue_aware(self, directory, reference, scan_names, reference_data, reference_threshold, reference_metadata, total_scans, channel_layer):
        """Normalize intensity ranges with tissue-aware scaling"""
        # Calculate reference statistics focused on tissue distribution
        reference_foreground = reference_data[reference_data > reference_threshold]
        if reference_foreground.size == 0:
            print("Warning: No foreground values in reference")
            return
        
        # Find the soft tissue peak (mode of lower intensity values)
        ref_hist, ref_bins = np.histogram(reference_foreground, bins=256)
        ref_peak_idx = np.argmax(ref_hist)
        ref_peak = ref_bins[ref_peak_idx]
        
        # Define tissue-specific ranges based on peak
        # Soft tissue range: peak ± some window (where most registration features are)
        soft_tissue_window = ref_peak * 0.5  # Adjustable parameter
        ref_soft_min = max(reference_threshold, ref_peak - soft_tissue_window)
        ref_soft_max = ref_peak + soft_tissue_window
        
        # Bone range: everything above soft tissue
        ref_bone_min = ref_soft_max
        ref_bone_max = np.percentile(reference_foreground, 99)
        
        print(f"Reference tissue ranges - Peak: {ref_peak:.1f}, Soft: {ref_soft_min:.1f}-{ref_soft_max:.1f}, Bone: {ref_bone_min:.1f}-{ref_bone_max:.1f}")
        
        for idx, scan_name in enumerate(scan_names):
            try:
                # Update progress
                if channel_layer is not None:
                    progress = idx / total_scans
                    async_to_sync(channel_layer.group_send)(
                        'progress_group',
                        {
                            'type': 'send_progress',
                            'progress': progress,
                            'scan_name': scan_name,
                            'custom_message': f'Tissue-aware normalizing {scan_name} to match {reference}...',
                            'total': total_scans,
                            'current': idx + 1,
                        }
                    )
                
                print(f"Processing {scan_name} ({idx+1}/{total_scans})")
                
                # Load scan
                latest_edit = 0
                while glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{latest_edit}_*.nii.gz")):
                    latest_edit += 1
                latest_edit -= 1
                scan_path = glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{latest_edit}_*.nii.gz"))[0]
                
                scan_img = nib.load(scan_path)
                scan_affine = scan_img.affine
                scan_original_dtype = scan_img.get_data_dtype()
                scan_data = scan_img.get_fdata().astype(scan_original_dtype)
                
                # Get scan threshold
                with open(os.path.join(directory, "extracted", scan_name, f"{scan_name}.json"), 'r') as jf:
                    scan_metadata = json.load(jf)
                scan_threshold = scan_metadata['threshold']
                
                # Find scan tissue ranges
                scan_foreground = scan_data[scan_data > scan_threshold]
                if scan_foreground.size == 0:
                    print(f"Warning: No foreground values in scan {scan_name}")
                    continue
                
                scan_hist, scan_bins = np.histogram(scan_foreground, bins=256)
                scan_peak_idx = np.argmax(scan_hist)
                scan_peak = scan_bins[scan_peak_idx]
                
                # Define corresponding tissue ranges for scan
                scan_soft_window = scan_peak * 0.5
                scan_soft_min = max(scan_threshold, scan_peak - scan_soft_window)
                scan_soft_max = scan_peak + scan_soft_window
                scan_bone_min = scan_soft_max
                scan_bone_max = np.percentile(scan_foreground, 99)
                
                print(f"Scan {scan_name} tissue ranges - Peak: {scan_peak:.1f}, Soft: {scan_soft_min:.1f}-{scan_soft_max:.1f}, Bone: {scan_bone_min:.1f}-{scan_bone_max:.1f}")
                
                # Create normalized data
                normalized_data = scan_data.copy().astype(np.float32)
                foreground_mask = scan_data > scan_threshold
                
                # Apply tissue-specific normalization
                # Soft tissue range (most important for registration)
                soft_tissue_mask = foreground_mask & (scan_data >= scan_soft_min) & (scan_data <= scan_soft_max)
                if np.any(soft_tissue_mask) and (scan_soft_max - scan_soft_min) > 0:
                    normalized_data[soft_tissue_mask] = (
                        (scan_data[soft_tissue_mask].astype(np.float32) - scan_soft_min) / 
                        (scan_soft_max - scan_soft_min) * (ref_soft_max - ref_soft_min) + ref_soft_min
                    )
                
                # Bone range (preserve relative spacing but don't let it dominate)
                bone_mask = foreground_mask & (scan_data > scan_soft_max)
                if np.any(bone_mask) and (scan_bone_max - scan_bone_min) > 0:
                    normalized_data[bone_mask] = (
                        (scan_data[bone_mask].astype(np.float32) - scan_bone_min) / 
                        (scan_bone_max - scan_bone_min) * (ref_bone_max - ref_bone_min) + ref_bone_min
                    )
                
                # Ensure values stay within valid range
                normalized_data = np.clip(normalized_data, 0, np.iinfo(scan_original_dtype).max if np.issubdtype(scan_original_dtype, np.integer) else normalized_data.max())
                normalized_data = normalized_data.astype(scan_original_dtype)
                
                # Save results using consistent helper
                self._save_processed_scan(directory, scan_name, normalized_data, scan_affine, latest_edit + 1, 
                                        "histmatched", reference_metadata, scan_metadata)
                
                # Clean up
                del scan_data, scan_img, normalized_data
                cleanup_memory()
                
            except Exception as e:
                print(f"Error processing {scan_name}: {str(e)}")
    
    def match_histogram_percentile_normalize(self, directory, reference, scan_names, reference_data, reference_threshold, reference_metadata, total_scans, channel_layer, percentile_low=2, percentile_high=98):
        """Clip to [percentile_low, percentile_high] (foreground) and linearly map to the full data type range."""
        for idx, scan_name in enumerate(scan_names):
            try:
                if channel_layer is not None:
                    progress = idx / total_scans
                    async_to_sync(channel_layer.group_send)(
                        'progress_group',
                        {
                            'type': 'send_progress',
                            'progress': progress,
                            'scan_name': scan_name,
                            'custom_message': f'Percentile normalizing {scan_name} ({percentile_low}–{percentile_high}%)...',
                            'total': total_scans,
                            'current': idx + 1,
                        }
                    )

                # Find the latest edit file or use raw file
                latest_edit = 0
                while glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{latest_edit}_*.nii.gz")):
                    latest_edit += 1
                latest_edit -= 1
                
                # Choose the appropriate file path
                if latest_edit >= 0:
                    scan_path = glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{latest_edit}_*.nii.gz"))[0]
                else:
                    scan_path = os.path.join(directory, "extracted", scan_name, f"{scan_name}.nii.gz")

                print(f"Processing {scan_name} ({idx+1}/{total_scans}) using file: {os.path.basename(scan_path)}")

                scan_img = nib.load(scan_path)
                scan_affine = scan_img.affine
                scan_original_dtype = scan_img.get_data_dtype()
                scan_data = scan_img.get_fdata().astype(scan_original_dtype)

                with open(os.path.join(directory, "extracted", scan_name, f"{scan_name}.json"), 'r') as jf:
                    scan_metadata = json.load(jf)
                scan_threshold = scan_metadata['threshold']

                # FIXED: Calculate percentiles on ENTIRE image data, not just foreground
                # This matches what the frontend shows
                all_voxels = scan_data.reshape(-1)
                p_low = np.percentile(all_voxels, percentile_low)
                p_high = np.percentile(all_voxels, percentile_high)

                print(f"  Percentiles for {scan_name}: {percentile_low}% = {p_low:.1f}, {percentile_high}% = {p_high:.1f}")

                if not np.isfinite(p_low) or not np.isfinite(p_high) or p_high <= p_low:
                    print(f"Invalid percentiles in {scan_name} (p{percentile_low}={p_low}, p{percentile_high}={p_high}); skipping")
                    continue

                clipped = np.clip(scan_data.astype(np.float32), p_low, p_high)
                # Determine output range based on dtype
                if np.issubdtype(scan_original_dtype, np.integer):
                    out_min = 0
                    out_max = np.iinfo(scan_original_dtype).max
                else:
                    out_min = 0.0
                    out_max = 1.0

                scaled = (clipped - p_low) / (p_high - p_low)
                normalized_data = (scaled * (out_max - out_min) + out_min)

                # Safety clip and cast back
                if np.issubdtype(scan_original_dtype, np.integer):
                    normalized_data = np.clip(normalized_data, out_min, out_max)
                else:
                    normalized_data = np.clip(normalized_data, out_min, out_max)
                normalized_data = normalized_data.astype(scan_original_dtype)

                # Calculate the new threshold using the linear mapping
                # Map the original threshold through the same transformation
                if scan_threshold <= p_low:
                    new_threshold = out_min
                elif scan_threshold >= p_high:
                    new_threshold = out_max
                else:
                    # Linear interpolation
                    new_threshold = ((scan_threshold - p_low) / (p_high - p_low)) * (out_max - out_min) + out_min

                # Update reference_metadata with the calculated threshold before saving
                reference_metadata_copy = reference_metadata.copy()
                reference_metadata_copy['threshold'] = float(new_threshold)
                reference_metadata_copy['filename'] = f"Percentile normalized ({percentile_low}%-{percentile_high}%)"

                # Determine next edit number
                next_edit = latest_edit + 1

                self._save_processed_scan(directory, scan_name, normalized_data, scan_affine, next_edit,
                                          "histmatched", reference_metadata_copy, scan_metadata)

                del scan_data, scan_img, normalized_data
                cleanup_memory()

            except Exception as e:
                print(f"Error processing {scan_name}: {str(e)}")
    
    def _save_processed_scan(self, directory, scan_name, processed_data, affine, edit_number, method_suffix, reference_metadata, scan_metadata):
        """Helper method to save processed scan data and metadata"""
        # Determine source base before saving (previous latest edit or base scan)
        try:
            prev_latest = 0
            while len(glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{prev_latest}_*"))) > 0:
                prev_latest += 1
            prev_latest -= 1
            if prev_latest >= 0:
                prev_source_path = glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{prev_latest}_*.nii.gz"))[0]
                source_base_for_mask = os.path.basename(prev_source_path).replace('.nii.gz', '')
            else:
                source_base_for_mask = scan_name
        except Exception as _:
            source_base_for_mask = scan_name

        # Save the processed data
        output_path = os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{edit_number}_{method_suffix}.nii.gz")
        scan_img = nib.Nifti1Image(processed_data, affine)
        nib.save(scan_img, output_path)
        print(f"  Saved processed data to: {output_path}")
        
        # Update metadata
        json_path = os.path.join(directory, "extracted", scan_name, f"{scan_name}.json")
        reference_file = reference_metadata.get('filename', 'unknown')
        from .rigidAlignment import (
            normalize_threshold_edit_stem,
            snapshot_previous_edit_threshold,
        )
        previous_stem = normalize_threshold_edit_stem(source_base_for_mask, scan_name)
        previous_threshold = scan_metadata.get('threshold')
        snapshot_previous_edit_threshold(scan_metadata, previous_stem, previous_threshold)
        scan_metadata['histogram_matched_to'] = reference_file
        scan_metadata['histogram_matched_old_threshold'] = scan_metadata['threshold']
        scan_metadata['threshold'] = float(reference_metadata['threshold'])
        intensity_norm = reference_metadata.get('intensity_norm_settings')
        if isinstance(intensity_norm, dict):
            scan_metadata['intensity_norm_settings'] = intensity_norm
        
        with open(json_path, 'w') as jf:
            json.dump(scan_metadata, jf, indent=4)
        print(f"  Updated metadata with threshold: {reference_metadata['threshold']}")
        
        # Save lossy version
        lossy_path = os.path.join(directory, "extracted", scan_name, f"{scan_name}_lossy_edit_{edit_number}_{method_suffix}.nii.gz")
        registration_tools = RegistrationTools()
        registration_tools.save_as_lossy_nifti(
            processed_data, 
            reference_metadata['voxel_size'],
            json_path,
            lossy_path
        )
        print(f"  Saved lossy version to: {lossy_path}")

        # Copy paired mask (non-geometry change) and generate lossy by downsampling from copied full-res
        try:
            src_mask_full = os.path.join(directory, "extracted", scan_name, f"{source_base_for_mask}.nii.mask.gz")
            dst_mask_full = os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{edit_number}_{method_suffix}.nii.mask.gz")
            dst_mask_lossy = os.path.join(directory, "extracted", scan_name, f"{scan_name}_lossy_edit_{edit_number}_{method_suffix}.nii.mask.gz")

            if os.path.isfile(src_mask_full):
                shutil.copyfile(src_mask_full, dst_mask_full)
                print(f"  Copied full-res mask to: {dst_mask_full}")

                # Downsample copied full-res mask to create lossy mask
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

                # load the lossy destination nifti image to get the affine
                lossy_nifti_img = nib.load(lossy_path)
                lossy_affine = lossy_nifti_img.affine

                mask_temp = dst_mask_full.replace('.nii.mask.gz', '_mask.nii.gz')
                lossy_temp = dst_mask_lossy.replace('.nii.mask.gz', '_mask.nii.gz')
                os.replace(dst_mask_full, mask_temp)
                try:
                    mask_img = nib.load(mask_temp)
                    mask_dtype = mask_img.get_data_dtype()
                    # Round before astype to avoid truncating floats
                    mask_arr = np.round(mask_img.get_fdata()).astype(mask_dtype)
                    slices = [slice(None, None, resolution_factor) for _ in range(3)]
                    lossy_mask = mask_arr[slices[0], slices[1], slices[2]].astype(mask_dtype, copy=False)
                    nib.save(nib.Nifti1Image(lossy_mask, lossy_affine), lossy_temp)
                    os.replace(lossy_temp, dst_mask_lossy)
                    print(f"  Generated lossy mask: {dst_mask_lossy}")
                finally:
                    os.replace(mask_temp, dst_mask_full)
            else:
                print("  No source full-res mask found; skipping.")
        except Exception as me:
            print(f"  Error handling paired mask (histmatched): {me}")
        
        # Copy paired landmark files (non-geometry change) if they exist
        try:
            source_landmarks_path = os.path.join(directory, "extracted", scan_name, f"{source_base_for_mask}_landmarks.json")
            source_distances_path = os.path.join(directory, "extracted", scan_name, f"{source_base_for_mask}_landmark_distances.json")
            
            dest_landmark_base = f"{scan_name}_edit_{edit_number}_{method_suffix}"
            dest_landmarks_path = os.path.join(directory, "extracted", scan_name, f"{dest_landmark_base}_landmarks.json")
            dest_distances_path = os.path.join(directory, "extracted", scan_name, f"{dest_landmark_base}_landmark_distances.json")
            
            if os.path.isfile(source_landmarks_path):
                shutil.copyfile(source_landmarks_path, dest_landmarks_path)
                print(f"  Copied landmarks to: {dest_landmarks_path}")
            
            if os.path.isfile(source_distances_path):
                shutil.copyfile(source_distances_path, dest_distances_path)
                print(f"  Copied landmark distances to: {dest_distances_path}")
        except Exception as le:
            print(f"  Error handling paired landmark files (histmatched): {le}")


    def match_histogram(self, directory, reference, method="histogram_match", percentile_low=2, percentile_high=98, onlyCurrentScan=False, selectedScan=None, flag_filter='off'):
        print(f"Starting histogram matching with reference: {reference}, method: {method}")
        
        # Load the reference and find scan names (existing logic)
        edit_number = 0
        while glob.glob(os.path.join(directory, "extracted", reference, f"{reference}_edit_{edit_number}_*.nii.gz")):
            edit_number += 1
        edit_number -= 1

        try:
            reference_metadata = load_extracted_scan_metadata(directory, reference)
        except (FileNotFoundError, json.JSONDecodeError):
            raise ValueError(f"Reference scan '{reference}' was not found.")
        if is_preserved_mesh_metadata(reference_metadata):
            raise ValueError(
                "Intensity Normalization requires a voxel-based reference. "
                "The selected reference is a preserved PLY mesh."
            )

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

        scan_names, skipped_preserved_meshes = partition_voxel_and_preserved_mesh_scans(directory, scan_names)
        if not scan_names:
            raise ValueError(voxel_only_all_meshes_message('Intensity Normalization'))

        scan_names.sort()
        
        # For percentile normalization, include the reference scan; for others, exclude it
        if method == "percentile_normalize":
            print(f"Percentile normalization: including reference {reference} in processing list")
        else:
            # Remove reference from the list of scans to process for other methods
            if reference in scan_names:
                scan_names.remove(reference)
        print(f"Found {len(scan_names)} scans to process: {scan_names}")

        # Filter to only current scan if requested
        if onlyCurrentScan and selectedScan:
            try:
                selected_metadata = load_extracted_scan_metadata(directory, selectedScan)
            except (FileNotFoundError, json.JSONDecodeError):
                raise ValueError(f"Selected scan '{selectedScan}' not found in scan list")
            if is_preserved_mesh_metadata(selected_metadata):
                raise ValueError(
                    f"Selected scan '{selectedScan}' is a preserved PLY mesh. "
                    "Intensity Normalization only works with voxel-based volumes."
                )
            if selectedScan in scan_names:
                scan_names = [selectedScan]
                print(f"Filtering to process only current scan: {selectedScan}")
            else:
                raise ValueError(f"Selected scan '{selectedScan}' not found in scan list")
        elif onlyCurrentScan and not selectedScan:
            raise ValueError("onlyCurrentScan flag set but no selectedScan provided")

        # Filter out already processed scans
        scans_to_remove = []
        for scan_name in scan_names:
            latest_edit = 0
            while len(glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{latest_edit}_*"))) > 0:
                latest_edit += 1
            latest_edit -= 1
            
            if latest_edit >= 0:
                for edit in range(0, latest_edit+1):
                    latest_edit_files = glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{edit}_*"))
                    if any("histmatched" in file for file in latest_edit_files):
                        print(f"Skipping {scan_name} as it's already histogram matched")
                        scans_to_remove.append(scan_name)
                        break

            # Remove from scan_names if it has elastic registration
            if has_elastic_registration(scan_name, directory):
                scans_to_remove.append(scan_name)
        
        for scan_name in scans_to_remove:
            if scan_name in scan_names:
                scan_names.remove(scan_name)
            else:
                print(f"Scan {scan_name} not found in scan_names")
        
        print(f"After filtering, {len(scan_names)} scans need histogram matching")

        flagged_set = load_flagged_subject_names(directory)
        scan_names = apply_flag_filter(scan_names, flagged_set, flag_filter)
        if not (onlyCurrentScan and selectedScan):
            scan_names = filter_out_linked_children(directory, scan_names)
        print(f"Match histogram flag filter: {flag_filter}, remaining scans: {len(scan_names)}")

        if not scan_names:
            raise ValueError(voxel_only_no_eligible_targets_message('Intensity Normalization'))
        if skipped_preserved_meshes:
            print(f"Skipping preserved mesh subjects for intensity normalization: {skipped_preserved_meshes}")

        # Initialize WebSocket progress tracking
        total_scans = len(scan_names)
        channel_layer = get_channel_layer()
        if channel_layer is not None and total_scans > 0:
            async_to_sync(channel_layer.group_send)(
                'progress_group',
                {
                    'type': 'send_progress',
                    'progress': 0,
                    'scan_name': reference,
                    'custom_message': f'Loading reference scan {reference} for histogram matching...',
                    'total': total_scans,
                    'current': 0,
                }
            )

        # Load the reference (only needed for non-percentile methods that use reference data)
        if method != "percentile_normalize":
            print(f"Loading reference scan from: {directory}/extracted/{reference}")
            if edit_number >= 0:
                reference_path = glob.glob(os.path.join(directory, "extracted", reference, f"{reference}_edit_{edit_number}_*.nii.gz"))[0]
            else:
                reference_path = os.path.join(directory, "extracted", reference, f"{reference}.nii.gz")
            reference_file = os.path.basename(reference_path)
            print(f"Using reference file: {reference_file}")
            reference_img = nib.load(reference_path)
            reference_original_dtype = reference_img.get_data_dtype()
            reference_data = reference_img.get_fdata().astype(reference_original_dtype)
            
            reference_metadata['filename'] = reference_file
            reference_threshold = reference_metadata['threshold']
            print(f"Reference threshold: {reference_threshold}")
        else:
            # For percentile normalization, we don't need reference data, but we need reference metadata for saving
            reference_metadata['filename'] = f"Normalized (no reference)"  # Placeholder filename
            reference_data = None
            reference_img = None
            reference_threshold = None

        reference_metadata['intensity_norm_settings'] = {
            "method": method,
            "percentile_low": percentile_low,
            "percentile_high": percentile_high,
        }
        
        # Select and execute method
        if method == "histogram_match":
            self.match_histogram_full(directory, reference, scan_names, reference_data, reference_threshold, reference_metadata, total_scans, channel_layer)
        elif method == "peak_align":
            self.match_histogram_align_peaks(directory, reference, scan_names, reference_data, reference_threshold, reference_metadata, total_scans, channel_layer)
        elif method == "tissue_aware":
            self.match_histogram_tissue_aware(directory, reference, scan_names, reference_data, reference_threshold, reference_metadata, total_scans, channel_layer)
        elif method == "percentile_normalize":
            self.match_histogram_percentile_normalize(directory, reference, scan_names, reference_data, reference_threshold, reference_metadata, total_scans, channel_layer, percentile_low, percentile_high)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Final progress update
        if channel_layer is not None:
            async_to_sync(channel_layer.group_send)(
                'progress_group',
                {
                    'type': 'send_progress',
                    'progress': 1.0,
                    'scan_name': 'All scans',
                    'custom_message': f'Histogram matching completed for all scans using {method}',
                    'total': total_scans,
                    'current': total_scans,
                }
            )
        
        # Clean up reference data
        del reference_data, reference_img
        cleanup_memory()
        print("Histogram matching completed")

    def post(self, request):
        directory = request.data.get('directory')
        reference = request.data.get('reference')
        method = request.data.get('method', 'histogram_match')
        percentile_low = request.data.get('percentile_low', 2)
        percentile_high = request.data.get('percentile_high', 98)
        onlyCurrentScan = request.data.get('onlyCurrentScan', False)
        selectedScan = request.data.get('selectedScan', None)
        flag_filter = normalize_flag_filter_value(
            request.data.get('flagFilter', 'off') if request.data else 'off',
            only_current_scan=onlyCurrentScan,
        )
        print(f"Received histogram matching request. Directory: {directory}, Reference: {reference}, Method: {method}")
        if method == "percentile_normalize":
            print(f"Percentiles: {percentile_low}% - {percentile_high}%")
        if onlyCurrentScan and not selectedScan:
            return Response({"status": "error", "message": "onlyCurrentScan flag set but no selectedScan provided"}, status=400)
        try:
            self.match_histogram(directory, reference, method, percentile_low, percentile_high, onlyCurrentScan, selectedScan, flag_filter)
            return Response({"status": "success", "message": f"Histogram matching completed using {method}"})
        except ValueError as e:
            return Response({"status": "error", "message": str(e)}, status=400)

# For a batch of scans inside a directory, do cleanupMesh on the latest edit, and save the result in original and lossy formats
class BatchCleanupMeshView(APIView):
    def _apply_removal_mask_to_sibling(
        self,
        directory,
        sibling_name,
        removal_mask_path,
        mask_affine,
        threshold,
        cleanup_settings=None,
    ):
        """
        Apply a removal mask to a same-shape linked sibling scan.
        Returns dict with scan_name, status, and edit or reason.
        """
        # Skip if already has elastic registration
        if has_elastic_registration(sibling_name, directory):
            return {'scan_name': sibling_name, 'status': 'skipped', 'reason': 'elastic registration present'}
        
        # Find latest non-elastic edit
        latest_edit = 0
        while glob.glob(os.path.join(directory, "extracted", sibling_name, f"{sibling_name}_edit_{latest_edit}_*.nii.gz")):
            latest_edit += 1
        latest_edit -= 1
        
        if latest_edit >= 0:
            edit_files = glob.glob(os.path.join(directory, "extracted", sibling_name, f"{sibling_name}_edit_{latest_edit}_*.nii.gz"))
            if edit_files:
                edit_file = os.path.basename(edit_files[0])
                if 'elastic' in edit_file.lower():
                    return {'scan_name': sibling_name, 'status': 'skipped', 'reason': 'latest edit is elastic'}
                nifti_path = os.path.join(directory, "extracted", sibling_name, edit_file)
            else:
                nifti_path = os.path.join(directory, "extracted", sibling_name, f"{sibling_name}.nii.gz")
        else:
            nifti_path = os.path.join(directory, "extracted", sibling_name, f"{sibling_name}.nii.gz")
        
        # Load removal mask
        removal_mask_temp = removal_mask_path.replace('.nii.removal_mask.gz', '_removal_mask.nii.gz')
        os.replace(removal_mask_path, removal_mask_temp)
        try:
            mask_img = nib.load(removal_mask_temp)
            removal_mask = np.round(mask_img.get_fdata()).astype(bool)
        finally:
            os.replace(removal_mask_temp, removal_mask_path)
        
        # Load sibling nifti
        sibling_img = nib.load(nifti_path)
        sibling_dtype = sibling_img.get_data_dtype()
        sibling_data = sibling_img.get_fdata().astype(sibling_dtype)
        
        # Check shape match
        if sibling_data.shape != removal_mask.shape:
            return {'scan_name': sibling_name, 'status': 'skipped', 'reason': f'shape mismatch: {sibling_data.shape} vs {removal_mask.shape}'}
        
        # Load sibling metadata to get threshold
        json_path = os.path.join(directory, "extracted", sibling_name, f"{sibling_name}.json")
        with open(json_path, 'r') as jf:
            sibling_metadata = json.load(jf)
        sibling_threshold = sibling_metadata.get('threshold', threshold)
        
        # Compute background value as mode below threshold (same logic as CleanupMeshView)
        vals = sibling_data[::2, ::2, ::2]
        thr = float(sibling_threshold)
        if np.issubdtype(sibling_dtype, np.integer) and sibling_dtype == np.uint8:
            low = 2
            thri = int(np.clip(thr, 0, 255))
            if thri <= low:
                background_value = 0
            else:
                mask_bg = (vals >= low) & (vals <= thri)
                if not np.any(mask_bg):
                    background_value = 0
                else:
                    sel = vals[mask_bg].astype(np.int16, copy=False)
                    hist = np.bincount(sel, minlength=256)
                    hist[:low] = 0
                    if thri + 1 < hist.size:
                        hist[thri + 1:] = 0
                    background_value = int(np.argmax(hist))
        elif np.issubdtype(sibling_dtype, np.integer) and sibling_dtype == np.uint16:
            low = 1000
            thri = int(np.clip(thr, 0, np.iinfo(np.uint16).max))
            if thri <= low:
                mask_bg = (vals > 0) & (vals < thri)
                background_value = int(round(float(np.mean(vals[mask_bg])))) if np.any(mask_bg) else 0
            else:
                edges = np.arange(low, thri + 500, 500, dtype=np.int64)
                if edges.size < 2:
                    edges = np.array([low, thri], dtype=np.int64)
                hist, _ = np.histogram(vals, bins=edges)
                if hist.size == 0 or hist.max() == 0:
                    mask_bg = (vals > 0) & (vals < thri)
                    background_value = int(round(float(np.mean(vals[mask_bg])))) if np.any(mask_bg) else 0
                else:
                    k = int(np.argmax(hist))
                    lo, hi = int(edges[k]), int(edges[min(k + 1, edges.size - 1)])
                    in_bin = vals[(vals >= lo) & (vals < hi)]
                    background_value = int(round(float(np.mean(in_bin)))) if in_bin.size > 0 else lo
        else:
            mask_bg = (vals > 0) & (vals < thr)
            background_value = int(round(float(np.mean(vals[mask_bg])))) if np.any(mask_bg) else 0
        
        background_value = np.array(background_value, dtype=sibling_dtype)
        
        # Apply removal mask to sibling
        sibling_data[removal_mask] = background_value
        
        # Find next edit number for sibling
        new_edit_number = 0
        while glob.glob(os.path.join(directory, "extracted", sibling_name, f"{sibling_name}_edit_{new_edit_number}_*.nii.gz")):
            new_edit_number += 1
        
        # Save cleaned sibling
        new_nifti = nib.Nifti1Image(sibling_data, sibling_img.affine)
        output_path = os.path.join(directory, "extracted", sibling_name, f"{sibling_name}_edit_{new_edit_number}_cleaned.nii.gz")
        nib.save(new_nifti, output_path)
        
        # Save lossy version
        RegistrationTools().save_as_lossy_nifti(
            sibling_data,
            sibling_metadata['voxel_size'],
            json_path,
            os.path.join(directory, "extracted", sibling_name, f"{sibling_name}_lossy_edit_{new_edit_number}_cleaned.nii.gz")
        )

        if isinstance(cleanup_settings, dict):
            try:
                write_subject_json_settings(
                    json_path, 'mesh_cleanup_settings', cleanup_settings, metadata=sibling_metadata
                )
            except Exception as meta_err:
                print(f"  Warning: could not save mesh_cleanup_settings for {sibling_name}: {meta_err}")
        
        # Copy paired mask if exists
        try:
            if latest_edit >= 0:
                source_base = os.path.basename(nifti_path).replace('.nii.gz', '')
            else:
                source_base = sibling_name
            src_mask_full = os.path.join(directory, "extracted", sibling_name, f"{source_base}.nii.mask.gz")
            if os.path.isfile(src_mask_full):
                dst_mask_full = os.path.join(directory, "extracted", sibling_name, f"{sibling_name}_edit_{new_edit_number}_cleaned.nii.mask.gz")
                shutil.copyfile(src_mask_full, dst_mask_full)
        except Exception as me:
            print(f"  Error copying paired mask for {sibling_name}: {me}")
        
        # Copy paired landmarks if exist
        try:
            if latest_edit >= 0:
                source_landmark_base = os.path.basename(nifti_path).replace('.nii.gz', '')
            else:
                source_landmark_base = sibling_name
            source_landmarks_path = os.path.join(directory, "extracted", sibling_name, f"{source_landmark_base}_landmarks.json")
            if os.path.isfile(source_landmarks_path):
                dest_landmarks_path = os.path.join(directory, "extracted", sibling_name, f"{sibling_name}_edit_{new_edit_number}_cleaned_landmarks.json")
                shutil.copyfile(source_landmarks_path, dest_landmarks_path)
        except Exception as le:
            print(f"  Error copying landmarks for {sibling_name}: {le}")
        
        return {
            'scan_name': sibling_name,
            'status': 'success',
            'edit': f"{sibling_name}_lossy_edit_{new_edit_number}_cleaned.nii.gz"
        }
    
    def post(self, request):
        directory = request.data.get('directory')
        gaussian_blur = request.data.get('gaussian_blur', None)
        use_island_volume_threshold = bool(request.data.get('use_island_volume_threshold', False))
        min_island_volume_percent = _clamp_island_volume_percent(
            request.data.get('min_island_volume_percent', 1.0)
        )
        flag_filter = normalize_flag_filter_value(
            request.data.get('flagFilter', 'off') if request.data else 'off',
            only_current_scan=False,
        )
        # Linking is the opt-in: children skipped by the batch inherit the main's result.
        propagate_linked = bool(request.data.get('propagate_linked', True))

        # Make a list of faulty files
        faulty_files = []
        for file in os.listdir(os.path.join(directory, "extracted")):
            # Skip project_settings.json file
            if file == "project_settings.json" or not os.path.isdir(os.path.join(directory, "extracted", file)):
                continue
            json_path = os.path.join(directory, "extracted", file, f"{file}.json")
            with open(json_path, 'r') as jf:
                metadata = json.load(jf)
                if metadata.get('faulty', False):
                    faulty_files.append(file)

        # Find all unique scan names by listing subdirectories in extracted folder
        scan_names = [d for d in os.listdir(os.path.join(directory, "extracted")) 
                     if os.path.isdir(os.path.join(directory, "extracted", d)) and d not in faulty_files]
        
        # Remove voxel-based scans that have elastic registration; preserved meshes use PLY edits.
        elastic_scan_names = []
        for scan_name in scan_names:
            try:
                metadata = load_extracted_scan_metadata(directory, scan_name)
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            if is_preserved_mesh_metadata(metadata):
                continue
            if has_elastic_registration(scan_name, directory):
                elastic_scan_names.append(scan_name)
        for scan_name in elastic_scan_names:
            if scan_name in scan_names:
                scan_names.remove(scan_name)

        # Order scan names alphabetically
        scan_names.sort()

        flagged_set = load_flagged_subject_names(directory)
        scan_names = apply_flag_filter(scan_names, flagged_set, flag_filter)
        scan_names = filter_out_linked_children(directory, scan_names)
        print(f"Batch cleanup mesh flag filter: {flag_filter}, remaining scans: {len(scan_names)}")
        
        if not scan_names:
            return Response({
                'message': 'No eligible scans found for batch cleanup',
                'results': [],
            }, status=status.HTTP_400_BAD_REQUEST)
        
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
                    'scan_name': scan_names[0] if scan_names else "",
                    'custom_message': 'Starting batch cleanup of all scans...',
                    'total': total_scans,
                    'current': 0,
                }
            )
        
        cleanup_mesh_view = CleanupMeshView()
        preserved_mesh_cleanup_view = ApplyMeshCleanupView()
        voxel_cleanup_settings = build_mesh_cleanup_settings(
            use_island_volume_threshold=use_island_volume_threshold,
            min_island_volume_percent=min_island_volume_percent,
            gaussian_blur=gaussian_blur,
            geometry='voxel',
        )
        
        # Process each scan
        results = []
        for idx, scan_name in enumerate(scan_names):
            try:
                json_path = os.path.join(directory, "extracted", scan_name, f"{scan_name}.json")
                with open(json_path, 'r') as jf:
                    json_metadata = json.load(jf)
                is_preserved_mesh = is_preserved_mesh_metadata(json_metadata)

                # Send progress update
                if channel_layer is not None:
                    progress = ((idx) / total_scans)
                    cleanup_label = "preserved mesh" if is_preserved_mesh else "voxel mesh"
                    async_to_sync(channel_layer.group_send)(
                        'progress_group',
                        {
                            'type': 'send_progress',
                            'progress': progress,
                            'scan_name': scan_name,
                            'custom_message': f'Cleaning up {cleanup_label} for {scan_name}...',
                            'total': total_scans,
                            'current': idx + 1,
                        }
                    )

                if is_preserved_mesh:
                    mesh_result = preserved_mesh_cleanup_view.cleanup_preserved_mesh(
                        directory,
                        scan_name,
                        edit="latest",
                        use_island_volume_threshold=use_island_volume_threshold,
                        min_island_volume_percent=min_island_volume_percent,
                    )
                    results.append({
                        'scan_name': scan_name,
                        'mode': 'preserved_mesh',
                        **mesh_result,
                    })
                    cleanup_memory()
                    continue

                # Find the latest edit for this voxel-based scan
                latest_edit = 0
                while glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{latest_edit}_*.nii.gz")):
                    latest_edit += 1
                latest_edit -= 1
                
                # Determine the file to process
                if latest_edit >= 0:
                    edit_files = glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{latest_edit}_*.nii.gz"))
                    if edit_files:
                        edit_file = os.path.basename(edit_files[0])
                    else:
                        edit_file = None
                else:
                    edit_file = None

                # If edit file is elastic, skip
                if edit_file and 'elastic' in edit_file.lower():
                    print(f"Skipping {scan_name} - elastic registration detected")
                    results.append({
                        'scan_name': scan_name,
                        'status': 'skipped',
                        'mode': 'voxel',
                        'message': 'Elastic registration detected, skipping cleanup'
                    })
                    continue

                # If latest edit is already cleaned, skip
                if edit_file and 'cleaned' in edit_file.lower():
                    print(f"Skipping {scan_name} - latest edit is already cleaned")
                    results.append({
                        'scan_name': scan_name,
                        'status': 'skipped',
                        'mode': 'voxel',
                        'message': 'Latest edit is already cleaned'
                    })
                    continue
                
                # Get threshold from json metadata
                threshold = json_metadata.get('threshold')
                
                # Get the 3D nifti file and load it into a numpy array
                if edit_file is None:
                    nifti_path = os.path.join(directory, "extracted", scan_name, f"{scan_name}.nii.gz")
                else:
                    nifti_path = os.path.join(directory, "extracted", scan_name, edit_file)
                
                print(f"Processing voxel scan {scan_name} with file {nifti_path}")
                
                # Load the nifti file
                nifti_img = nib.load(nifti_path)
                original_dtype = nifti_img.get_data_dtype()
                nifti_data = nifti_img.get_fdata().astype(original_dtype)
                
                # Clean the mesh and check if cleanup was actually performed
                cleaned_data, cleanup_performed, removal_mask, cleanup_outcome = cleanup_mesh_view.clean_mesh(
                    nifti_data,
                    threshold,
                    original_dtype,
                    gaussian_blur,
                    use_island_volume_threshold=use_island_volume_threshold,
                    min_island_volume_percent=min_island_volume_percent,
                )

                source_edit_name = edit_file if edit_file else f"{scan_name}.nii.gz"
                
                if not cleanup_performed:
                    print(f"Skipping {scan_name} - no islands removed, no cleanup needed")
                    try:
                        noop_settings = build_mesh_cleanup_settings(
                            use_island_volume_threshold=use_island_volume_threshold,
                            min_island_volume_percent=min_island_volume_percent,
                            gaussian_blur=gaussian_blur,
                            geometry='voxel',
                            cleanup_performed=False,
                            reason=(cleanup_outcome or {}).get('reason') or 'no_disconnected_components',
                            source_edit=source_edit_name,
                            components_found=(cleanup_outcome or {}).get('components_found'),
                        )
                        write_subject_json_settings(
                            json_path, 'mesh_cleanup_settings', noop_settings, metadata=json_metadata
                        )
                    except Exception as meta_err:
                        print(f"Warning: could not save mesh_cleanup_settings (no-op) for {scan_name}: {meta_err}")
                    results.append({
                        'scan_name': scan_name,
                        'status': 'skipped',
                        'mode': 'voxel',
                        'message': 'No islands removed, no cleanup needed',
                        'cleanup_performed': False,
                        'reason': (cleanup_outcome or {}).get('reason'),
                        'components_found': (cleanup_outcome or {}).get('components_found'),
                        'source_edit': source_edit_name,
                    })
                    
                    # Clean up memory after each scan
                    cleanup_memory()
                    continue
                
                # Update progress message for actual file saving
                if channel_layer is not None:
                    progress = ((idx) / total_scans)
                    async_to_sync(channel_layer.group_send)(
                        'progress_group',
                        {
                            'type': 'send_progress',
                            'progress': progress,
                            'scan_name': scan_name,
                            'custom_message': f'Saving cleaned mesh for {scan_name}...',
                            'total': total_scans,
                            'current': idx + 1,
                        }
                    )
                
                # Create new nifti image with cleaned data
                new_nifti = nib.Nifti1Image(cleaned_data, nifti_img.affine)
                
                # Find the next edit number
                edit_number = 0
                while glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{edit_number}_*.nii.gz")):
                    edit_number += 1
                
                # Save the cleaned image
                output_path = os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{edit_number}_cleaned.nii.gz")
                nib.save(new_nifti, output_path)
                
                # Generate the lossy version
                RegistrationTools().save_as_lossy_nifti(
                    cleaned_data, 
                    json_metadata['voxel_size'], 
                    os.path.join(directory, "extracted", scan_name, f"{scan_name}.json"),
                    os.path.join(directory, "extracted", scan_name, f"{scan_name}_lossy_edit_{edit_number}_cleaned.nii.gz")
                )

                try:
                    performed_settings = {
                        **voxel_cleanup_settings,
                        'cleanup_performed': True,
                        'reason': (cleanup_outcome or {}).get('reason') or 'components_removed',
                        'source_edit': os.path.basename(str(source_edit_name)),
                    }
                    if (cleanup_outcome or {}).get('components_found') is not None:
                        performed_settings['components_found'] = int(cleanup_outcome['components_found'])
                    write_subject_json_settings(
                        json_path, 'mesh_cleanup_settings', performed_settings, metadata=json_metadata
                    )
                except Exception as meta_err:
                    print(f"Warning: could not save mesh_cleanup_settings for {scan_name}: {meta_err}")

                # Save removal mask if cleanup was performed
                removal_mask_path = None
                if removal_mask is not None:
                    removal_mask_uint8 = removal_mask.astype(np.uint8)
                    removal_mask_nifti = nib.Nifti1Image(removal_mask_uint8, nifti_img.affine)
                    removal_mask_temp = os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{edit_number}_cleaned_removal_mask.nii.gz")
                    removal_mask_path = os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{edit_number}_cleaned.nii.removal_mask.gz")
                    nib.save(removal_mask_nifti, removal_mask_temp)
                    os.replace(removal_mask_temp, removal_mask_path)
                    print(f"Saved removal mask to: {removal_mask_path}")

                # Copy paired mask (non-geometry change) and generate lossy by downsampling from copied full-res
                try:
                    if latest_edit >= 0:
                        source_base = os.path.basename(nifti_path).replace('.nii.gz', '')
                    else:
                        source_base = scan_name
                    
                    src_mask_full = os.path.join(directory, "extracted", scan_name, f"{source_base}.nii.mask.gz")
                    dst_mask_full = os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{edit_number}_cleaned.nii.mask.gz")
                    dst_mask_lossy = os.path.join(directory, "extracted", scan_name, f"{scan_name}_lossy_edit_{edit_number}_cleaned.nii.mask.gz")
                    dst_nifti_lossy = os.path.join(directory, "extracted", scan_name, f"{scan_name}_lossy_edit_{edit_number}_cleaned.nii.gz")

                    if os.path.isfile(src_mask_full):
                        shutil.copyfile(src_mask_full, dst_mask_full)
                        print(f"Copied full-res mask to: {dst_mask_full}")

                        # Downsample copied full-res mask to create lossy mask
                        resolution_factor = 2
                        lc = json_metadata.get('lossy_compression', None)
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

                        # load the lossy destination nifti image to get the affine
                        lossy_nifti_img = nib.load(dst_nifti_lossy)
                        lossy_affine = lossy_nifti_img.affine

                        mask_temp = dst_mask_full.replace('.nii.mask.gz', '_mask.nii.gz')
                        lossy_temp = dst_mask_lossy.replace('.nii.mask.gz', '_mask.nii.gz')
                        os.replace(dst_mask_full, mask_temp)
                        try:
                            mask_img = nib.load(mask_temp)
                            mask_dtype = mask_img.get_data_dtype()
                            # Round before astype to avoid truncating floats
                            mask_arr = np.round(mask_img.get_fdata()).astype(mask_dtype)
                            slices = [slice(None, None, resolution_factor) for _ in range(3)]
                            lossy_mask = mask_arr[slices[0], slices[1], slices[2]].astype(mask_dtype, copy=False)
                            nib.save(nib.Nifti1Image(lossy_mask, lossy_affine), lossy_temp)
                            os.replace(lossy_temp, dst_mask_lossy)
                            print(f"Generated lossy mask: {dst_mask_lossy}")
                        finally:
                            os.replace(mask_temp, dst_mask_full)
                    else:
                        print("No source full-res mask found; skipping.")
                except Exception as me:
                    print(f"Error handling paired mask (batch cleaned): {me}")
                
                # Copy paired landmark files (non-geometry change) if they exist
                try:
                    if latest_edit >= 0:
                        source_landmark_base = os.path.basename(nifti_path).replace('.nii.gz', '')
                    else:
                        source_landmark_base = scan_name
                    
                    source_landmarks_path = os.path.join(directory, "extracted", scan_name, f"{source_landmark_base}_landmarks.json")
                    source_distances_path = os.path.join(directory, "extracted", scan_name, f"{source_landmark_base}_landmark_distances.json")
                    
                    dest_landmark_base = f"{scan_name}_edit_{edit_number}_cleaned"
                    dest_landmarks_path = os.path.join(directory, "extracted", scan_name, f"{dest_landmark_base}_landmarks.json")
                    dest_distances_path = os.path.join(directory, "extracted", scan_name, f"{dest_landmark_base}_landmark_distances.json")
                    
                    if os.path.isfile(source_landmarks_path):
                        shutil.copyfile(source_landmarks_path, dest_landmarks_path)
                        print(f"Copied landmarks to: {dest_landmarks_path}")
                    
                    if os.path.isfile(source_distances_path):
                        shutil.copyfile(source_distances_path, dest_distances_path)
                        print(f"Copied landmark distances to: {dest_distances_path}")
                except Exception as le:
                    print(f"Error handling paired landmark files (batch cleaned): {le}")
                
                # Propagate cleanup removal mask to same-shape linked children if requested
                linked_propagation = []
                if propagate_linked and removal_mask_path and os.path.isfile(removal_mask_path):
                    try:
                        siblings, err = get_same_shape_siblings(directory, scan_name)
                        if err:
                            print(f"Could not get same-shape siblings for {scan_name}: {err}")
                        elif siblings:
                            print(f"Propagating cleanup removal mask from {scan_name} to {len(siblings)} same-shape linked children")
                            for sibling_name in siblings:
                                try:
                                    sibling_result = self._apply_removal_mask_to_sibling(
                                        directory,
                                        sibling_name,
                                        removal_mask_path,
                                        nifti_img.affine,
                                        threshold,
                                        cleanup_settings=voxel_cleanup_settings,
                                    )
                                    linked_propagation.append(sibling_result)
                                    print(f"  Propagated cleanup to {sibling_name}: {sibling_result['status']}")
                                except Exception as sib_err:
                                    print(f"  Failed to propagate cleanup to {sibling_name}: {sib_err}")
                                    linked_propagation.append({'scan_name': sibling_name, 'status': 'error', 'reason': str(sib_err)})
                        else:
                            print(f"No same-shape linked children found for {scan_name}")
                    except Exception as prop_err:
                        print(f"Error during linked propagation for {scan_name}: {prop_err}")
                
                result_entry = {
                    'scan_name': scan_name,
                    'status': 'success',
                    'mode': 'voxel',
                    'edit': f"{scan_name}_lossy_edit_{edit_number}_cleaned.nii.gz"
                }
                if linked_propagation:
                    result_entry['linked_propagation'] = linked_propagation
                results.append(result_entry)
                
                print(f"Successfully cleaned {scan_name}")
                
                # Clean up memory after each scan
                cleanup_memory()
                
            except Exception as e:
                print(f"Error processing {scan_name}: {str(e)}")
                results.append({
                    'scan_name': scan_name,
                    'status': 'error',
                    'message': str(e)
                })
        
        # Send final progress update
        if channel_layer is not None:
            async_to_sync(channel_layer.group_send)(
                'progress_group',
                {
                    'type': 'send_progress',
                    'progress': 1.0,
                    'scan_name': "All scans",
                    'custom_message': 'Batch cleanup completed',
                    'total': total_scans,
                    'current': total_scans,
                }
            )
        
        return Response({
            'message': 'Batch cleanup completed',
            'results': results
        }, status=status.HTTP_200_OK)


class N4BiasCorrectionView(APIView):
    def post(self, request):
        directory = request.data.get('directory')
        filename = request.data.get('filename')
        flag_filter = normalize_flag_filter_value(
            request.data.get('flagFilter', 'off') if request.data else 'off',
            only_current_scan=(filename is not None),
        )
        parameters = request.data.get('parameters', {})
        try:
            self.n4_bias_correction(directory, filename, flag_filter=flag_filter, params=parameters)
        except ValueError as e:
            return Response({"status": "error", "message": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        return Response({"status": "success", "message": "N4 bias correction completed"})

    def n4_bias_correction(self, directory, filename=None, flag_filter='off', params=None):
        """
        Apply N4 bias correction to the latest edit of every subject in the directory. If a filename is provided, only apply to that scan.
        
        Args:
            directory (str): Path to the main directory containing extracted scans
            flag_filter (str): For directory batch only: ``off`` | ``exclude`` (skip flagged) | ``only`` (flagged only). Ignored when ``filename`` is set.
        """

        if filename is not None:
            try:
                selected_metadata = load_extracted_scan_metadata(directory, filename)
            except (FileNotFoundError, json.JSONDecodeError):
                raise ValueError(f"Selected scan '{filename}' was not found.")
            if is_preserved_mesh_metadata(selected_metadata):
                raise ValueError(
                    f"Selected scan '{filename}' is a preserved PLY mesh. "
                    "N4 bias correction only works with voxel-based volumes."
                )
            scan_names = [filename]
        else:
            # Make a list of faulty files
            faulty_files = []
            for file in os.listdir(os.path.join(directory, "extracted")):
                # Skip project_settings.json file
                if file == "project_settings.json" or not os.path.isdir(os.path.join(directory, "extracted", file)):
                    continue
                json_path = os.path.join(directory, "extracted", file, f"{file}.json")
                with open(json_path, 'r') as jf:
                    metadata = json.load(jf)
                    if metadata.get('faulty', False):
                        faulty_files.append(file)

            # Find all unique scan names by listing subdirectories in extracted folder
            scan_names = [d for d in os.listdir(os.path.join(directory, "extracted")) 
                        if os.path.isdir(os.path.join(directory, "extracted", d)) and d not in faulty_files]
            
            # Order scan names alphabetically
            scan_names.sort()

            scan_names, skipped_preserved_meshes = partition_voxel_and_preserved_mesh_scans(directory, scan_names)
            if not scan_names:
                raise ValueError(voxel_only_all_meshes_message('N4 bias correction'))

            # Remove any scans that are already corrected (n4 or legacy n3 output filenames)
            scan_names = [scan_name for scan_name in scan_names if not has_bias_correction_output(directory, scan_name)]

            # Completed = latest edit is elastic (has_elastic_registration);
            scan_names = [s for s in scan_names if not has_elastic_registration(s, directory)]

            flagged_set = load_flagged_subject_names(directory)
            scan_names = apply_flag_filter(scan_names, flagged_set, flag_filter)
            scan_names = filter_out_linked_children(directory, scan_names)
            print(f"N4 bias correction flag filter: {flag_filter}, remaining scans: {len(scan_names)}")

            if not scan_names:
                raise ValueError(voxel_only_no_eligible_targets_message('N4 bias correction'))
            if skipped_preserved_meshes:
                print(f"Skipping preserved mesh subjects for N4 bias correction: {skipped_preserved_meshes}")

        # Initialize WebSocket progress tracking
        if filename is None:
            total_scans = len(scan_names)
            channel_layer = get_channel_layer()
            if channel_layer is not None and total_scans > 0:
                print("Sending progress update")
                async_to_sync(channel_layer.group_send)(
                    'progress_group',
                    {
                        'type': 'send_progress',
                        'progress': 0,
                        'scan_name': scan_names[0] if scan_names else "",
                        'custom_message': 'Starting N4 bias correction for all scans...',
                        'total': total_scans,
                        'current': 0,
                    }
                )
        else:
            channel_layer = None
        
        processed_scans = []
        # Process each scan
        for idx, scan_name in enumerate(scan_names):
            try:
                # Find the latest edit for this scan
                latest_edit = 0
                while glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{latest_edit}_*.nii.gz")):
                    latest_edit += 1
                latest_edit -= 1
                
                # Send progress update
                if channel_layer is not None and filename is None:
                    progress = ((idx) / total_scans)
                    async_to_sync(channel_layer.group_send)(
                        'progress_group',
                        {
                            'type': 'send_progress',
                            'progress': progress,
                            'scan_name': scan_name,
                            'custom_message': f'Applying N4 bias correction to {scan_name}...',
                            'total': total_scans,
                            'current': idx + 1,
                        }
                    )
                
                # Determine the file to process
                if latest_edit >= 0:
                    edit_files = glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{latest_edit}_*.nii.gz"))
                    if edit_files:
                        edit_file = os.path.basename(edit_files[0])
                        nifti_path = os.path.join(directory, "extracted", scan_name, edit_file)
                    else:
                        nifti_path = os.path.join(directory, "extracted", scan_name, f"{scan_name}.nii.gz")
                else:
                    nifti_path = os.path.join(directory, "extracted", scan_name, f"{scan_name}.nii.gz")
                
                print(f"Processing {scan_name} with file {nifti_path}")
                
                # Load the nifti file
                nifti_img = nib.load(nifti_path)
                original_dtype = nifti_img.get_data_dtype()
                nifti_data = nifti_img.get_fdata().astype(np.float32)  # Convert to float32 for ANTs
                
                # Apply N4 bias correction
                corrected_data = self.correct(nifti_data, params)
                print(f"Corrected data shape: {corrected_data.shape}")
                
                
                # Convert back to original data type
                corrected_data = np.clip(corrected_data, np.iinfo(original_dtype).min, np.iinfo(original_dtype).max).astype(original_dtype)
                
                # Create new nifti image with corrected data
                new_nifti = nib.Nifti1Image(corrected_data, nifti_img.affine)
                
                # Find the next edit number
                edit_number = 0
                while glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{edit_number}_*.nii.gz")):
                    edit_number += 1
                
                                # Save the corrected image
                output_path = os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{edit_number}_n4corrected.nii.gz")
                nib.save(new_nifti, output_path)
                
                # Load the metadata JSON
                json_path = os.path.join(directory, "extracted", scan_name, f"{scan_name}.json")
                with open(json_path, 'r') as jf:
                    json_metadata = json.load(jf)
                json_metadata['n4_bias_settings'] = build_n4_bias_settings(params)
                with open(json_path, 'w') as jf:
                    json.dump(json_metadata, jf, indent=4)
                
                # Generate the lossy version
                RegistrationTools().save_as_lossy_nifti(
                    corrected_data, 
                    json_metadata['voxel_size'], 
                    json_path,
                    os.path.join(directory, "extracted", scan_name, f"{scan_name}_lossy_edit_{edit_number}_n4corrected.nii.gz")
                )
                
                print(f"Successfully applied N4 bias correction to {scan_name}")
                processed_scans.append(scan_name)

                # Copy paired mask (non-geometry change) and generate lossy by downsampling from copied full-res
                try:
                    # Determine source base used as input
                    if latest_edit >= 0:
                        source_path = glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{latest_edit}_*.nii.gz"))[0]
                        source_base = os.path.basename(source_path).replace('.nii.gz', '')
                    else:
                        source_base = scan_name

                    src_mask_full = os.path.join(directory, "extracted", scan_name, f"{source_base}.nii.mask.gz")

                    dst_mask_full = os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{edit_number}_n4corrected.nii.mask.gz")
                    dst_mask_lossy = os.path.join(directory, "extracted", scan_name, f"{scan_name}_lossy_edit_{edit_number}_n4corrected.nii.mask.gz")
                    dst_nifti_lossy = os.path.join(directory, "extracted", scan_name, f"{scan_name}_lossy_edit_{edit_number}_n4corrected.nii.gz")

                    if os.path.isfile(src_mask_full):
                        shutil.copyfile(src_mask_full, dst_mask_full)
                        print(f"Copied full-res mask to: {dst_mask_full}")

                        # Downsample copied full-res mask to create lossy mask
                        resolution_factor = 2
                        lc = json_metadata.get('lossy_compression', None)
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

                        # load the lossy destination nifti image to get the affine
                        lossy_nifti_img = nib.load(dst_nifti_lossy)
                        lossy_affine = lossy_nifti_img.affine

                        mask_temp = dst_mask_full.replace('.nii.mask.gz', '_mask.nii.gz')
                        lossy_temp = dst_mask_lossy.replace('.nii.mask.gz', '_mask.nii.gz')
                        os.replace(dst_mask_full, mask_temp)
                        try:
                            mask_img = nib.load(mask_temp)
                            mask_dtype = mask_img.get_data_dtype()
                            # Round before astype to avoid truncating floats
                            mask_arr = np.round(mask_img.get_fdata()).astype(mask_dtype)
                            slices = [slice(None, None, resolution_factor) for _ in range(3)]
                            lossy_mask = mask_arr[slices[0], slices[1], slices[2]].astype(mask_dtype, copy=False)
                            nib.save(nib.Nifti1Image(lossy_mask, lossy_affine), lossy_temp)
                            os.replace(lossy_temp, dst_mask_lossy)
                            print(f"Generated lossy mask: {dst_mask_lossy}")
                        finally:
                            os.replace(mask_temp, dst_mask_full)
                    else:
                        print("No source full-res mask found; skipping.")
                except Exception as me:
                    print(f"Error handling paired mask (N4): {me}")
                
                # Copy paired landmark files (non-geometry change) if they exist
                try:
                    if latest_edit >= 0:
                        source_landmark_base = os.path.basename(nifti_path).replace('.nii.gz', '')
                    else:
                        source_landmark_base = scan_name
                    
                    source_landmarks_path = os.path.join(directory, "extracted", scan_name, f"{source_landmark_base}_landmarks.json")
                    source_distances_path = os.path.join(directory, "extracted", scan_name, f"{source_landmark_base}_landmark_distances.json")
                    
                    dest_landmark_base = f"{scan_name}_edit_{edit_number}_n4corrected"
                    dest_landmarks_path = os.path.join(directory, "extracted", scan_name, f"{dest_landmark_base}_landmarks.json")
                    dest_distances_path = os.path.join(directory, "extracted", scan_name, f"{dest_landmark_base}_landmark_distances.json")
                    
                    if os.path.isfile(source_landmarks_path):
                        shutil.copyfile(source_landmarks_path, dest_landmarks_path)
                        print(f"Copied landmarks to: {dest_landmarks_path}")
                    
                    if os.path.isfile(source_distances_path):
                        shutil.copyfile(source_distances_path, dest_distances_path)
                        print(f"Copied landmark distances to: {dest_distances_path}")
                except Exception as le:
                    print(f"Error handling paired landmark files (N4): {le}")

            except Exception as e:
                print(f"Error processing {scan_name}: {str(e)}")
        
        # Send final progress update
        if channel_layer is not None and filename is None:
            async_to_sync(channel_layer.group_send)(
                'progress_group',
                {
                    'type': 'send_progress',
                    'progress': 1.0,
                    'scan_name': "All scans",
                    'custom_message': 'N4 bias correction completed',
                    'total': total_scans,
                    'current': total_scans,
                }
            )

    def correct(self, image_array, params=None):
        if params is None:
            params = {}
            
        shrink_factor = int(params.get('shrink_factor', 4))
        convergence_iters = int(params.get('convergence_iters', 50))
        convergence_tol_exp = int(params.get('convergence_tol_exp', 7))
        spline_param = int(params.get('spline_param', 200))

        convergence_tol = 10 ** (-convergence_tol_exp)

        image_ants = ants.from_numpy(image_array)
        corrected_ants = ants.n4_bias_field_correction(
            image_ants,
            shrink_factor=shrink_factor,
            convergence={'iters': [convergence_iters] * 4, 'tol': convergence_tol},
            spline_param=spline_param,
            verbose=True           
        )
        corrected_array = corrected_ants.numpy()
        return corrected_array


# Backward compatibility (older code / bookmarks may still reference N3)
N3BiasCorrectionView = N4BiasCorrectionView


class SaveBackgroundValuesView(APIView):
    def post(self, request):
        print("Saving background values and applying background correction")
        directory = request.data.get('directory')
        values = request.data.get('values', {})
        
        if not directory or not values:
            return Response({
                'status': 'error', 
                'message': 'Directory and values are required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Filter out scans that don't have background values to update
        scans_to_process = {k: v for k, v in values.items() if 'background' in v}

        if not scans_to_process:
            return Response({
                'status': 'error', 
                'message': 'No background values provided'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        updated_scans = []
        errors = []
    
        
        for idx, (scan_name, scan_values) in enumerate(scans_to_process.items()):            
            try:
                # If the background value is 0, then skip
                if scan_values['background'] == 0:
                    continue

                # Find the latest edit for this scan
                latest_edit = 0
                while glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{latest_edit}_*.nii.gz")):
                    latest_edit += 1
                latest_edit -= 1
                
                # Load the scan data path
                if latest_edit >= 0:
                    scan_path = glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{latest_edit}_*.nii.gz"))[0]
                else:
                    scan_path = os.path.join(directory, "extracted", scan_name, f"{scan_name}.nii.gz")
                
                # Load the scan data
                nifti_img = nib.load(scan_path)
                original_dtype = nifti_img.get_data_dtype()
                scan_data = nifti_img.get_fdata().astype(original_dtype)
                
                # Apply background correction using the HomogenizeBackgroundView method
                background_value = float(scan_values['background'])
                homogenizer = HomogenizeBackgroundView()
                corrected_data = homogenizer._homogenize_background(scan_data, background_value)
                
                # Save the corrected data
                new_edit_number = latest_edit + 1
                output_path = os.path.join(directory, "extracted", scan_name, 
                                          f"{scan_name}_edit_{new_edit_number}_backfixed.nii.gz")
                output_path_lossy = os.path.join(directory, "extracted", scan_name, 
                                          f"{scan_name}_lossy_edit_{new_edit_number}_backfixed.nii.gz")
                
                # Save the corrected data with the original data type
                new_img = nib.Nifti1Image(corrected_data.astype(original_dtype), nifti_img.affine, nifti_img.header)
                nib.save(new_img, output_path)
                
                # Update threshold in metadata (subtract the background shift)
                json_path = os.path.join(directory, "extracted", scan_name, f"{scan_name}.json")
                with open(json_path, 'r') as jf:
                    scan_metadata = json.load(jf)

                from .rigidAlignment import (
                    normalize_threshold_edit_stem,
                    snapshot_previous_edit_threshold,
                    append_background_offset_history,
                )
                previous_stem = (
                    normalize_threshold_edit_stem(os.path.basename(scan_path), scan_name)
                    if latest_edit >= 0
                    else scan_name
                )
                snapshot_previous_edit_threshold(scan_metadata, previous_stem)
                applied_shift = int(background_value)
                scan_metadata['threshold'] = scan_metadata['threshold'] - applied_shift
                scan_metadata['backfixed_shift'] = applied_shift
                scan_metadata['background_offset_settings'] = {
                    'mode': 'manual_per_scan',
                    'shift': applied_shift,
                }
                append_background_offset_history(
                    scan_metadata,
                    f"{scan_name}_edit_{new_edit_number}_backfixed",
                    'manual_per_scan',
                    applied_shift,
                )
                
                with open(json_path, 'w') as jf:
                    json.dump(scan_metadata, jf, indent=4)
                
                # Save compressed version
                RegistrationTools().save_as_lossy_nifti(
                    corrected_data, 
                    scan_metadata['voxel_size'], 
                    json_path,
                    output_path_lossy
                )

                # Copy paired mask (non-geometry change) and generate lossy by downsampling from copied full-res
                try:
                    source_base = os.path.basename(scan_path).replace('.nii.gz', '') if latest_edit >= 0 else scan_name
                    src_mask_full = os.path.join(directory, "extracted", scan_name, f"{source_base}.nii.mask.gz")

                    dst_mask_full = os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{new_edit_number}_backfixed.nii.mask.gz")
                    dst_mask_lossy = os.path.join(directory, "extracted", scan_name, f"{scan_name}_lossy_edit_{new_edit_number}_backfixed.nii.mask.gz")

                    if os.path.isfile(src_mask_full):
                        shutil.copyfile(src_mask_full, dst_mask_full)
                        print(f"Copied full-res mask to: {dst_mask_full}")

                        # Downsample copied full-res mask to create lossy mask
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

                        # load the lossy destination nifti image to get the affine
                        lossy_nifti_img = nib.load(output_path_lossy)
                        lossy_affine = lossy_nifti_img.affine

                        mask_temp = dst_mask_full.replace('.nii.mask.gz', '_mask.nii.gz')
                        lossy_temp = dst_mask_lossy.replace('.nii.mask.gz', '_mask.nii.gz')
                        os.replace(dst_mask_full, mask_temp)
                        try:
                            mask_img = nib.load(mask_temp)
                            mask_dtype = mask_img.get_data_dtype()
                            # Round before astype to avoid truncating floats
                            mask_arr = np.round(mask_img.get_fdata()).astype(mask_dtype)
                            slices = [slice(None, None, resolution_factor) for _ in range(3)]
                            lossy_mask = mask_arr[slices[0], slices[1], slices[2]].astype(mask_dtype, copy=False)
                            nib.save(nib.Nifti1Image(lossy_mask, lossy_affine), lossy_temp)
                            os.replace(lossy_temp, dst_mask_lossy)
                            print(f"Generated lossy mask: {dst_mask_lossy}")
                        finally:
                            os.replace(mask_temp, dst_mask_full)
                    else:
                        print("No source full-res mask found; skipping.")
                except Exception as me:
                    print(f"Error handling paired mask (backfixed values): {me}")

                # Copy paired landmark files (non-geometry change) if they exist
                try:
                    if latest_edit >= 0:
                        source_landmark_base = os.path.basename(scan_path).replace('.nii.gz', '')
                    else:
                        source_landmark_base = scan_name
                    
                    source_landmarks_path = os.path.join(directory, "extracted", scan_name, f"{source_landmark_base}_landmarks.json")
                    source_distances_path = os.path.join(directory, "extracted", scan_name, f"{source_landmark_base}_landmark_distances.json")
                    
                    dest_landmark_base = f"{scan_name}_edit_{new_edit_number}_backfixed"
                    dest_landmarks_path = os.path.join(directory, "extracted", scan_name, f"{dest_landmark_base}_landmarks.json")
                    dest_distances_path = os.path.join(directory, "extracted", scan_name, f"{dest_landmark_base}_landmark_distances.json")
                    
                    if os.path.isfile(source_landmarks_path):
                        shutil.copyfile(source_landmarks_path, dest_landmarks_path)
                        print(f"Copied landmarks to: {dest_landmarks_path}")
                    
                    if os.path.isfile(source_distances_path):
                        shutil.copyfile(source_distances_path, dest_distances_path)
                        print(f"Copied landmark distances to: {dest_distances_path}")
                except Exception as le:
                    print(f"Error handling paired landmark files (backfixed values): {le}")
                
                updated_scans.append(scan_name)
                print(f"Applied background correction to {scan_name} with value {background_value}")
                
                # Clean up memory
                cleanup_memory()
                
            except Exception as e:
                error_msg = f"Error processing {scan_name}: {str(e)}"
                errors.append(error_msg)
                print(error_msg)
        
        
        response_data = {
            'status': 'success',
            'message': f'Applied background correction to {len(updated_scans)} scans',
            'updated_scans': updated_scans
        }
        
        if errors:
            response_data['errors'] = errors
            response_data['message'] += f' with {len(errors)} errors'
        
        return Response(response_data, status=status.HTTP_200_OK)


class SaveThresholdValuesView(APIView):
    def post(self, request):
        print("Saving threshold values")
        directory = request.data.get('directory')
        values = request.data.get('values', {})
        
        if not directory or not values:
            return Response({
                'status': 'error', 
                'message': 'Directory and values are required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        updated_scans = []
        errors = []
        
        for scan_name, scan_values in values.items():
            try:
                # Path to the scan's metadata JSON file
                json_path = os.path.join(directory, "extracted", scan_name, f"{scan_name}.json")
                
                if not os.path.exists(json_path):
                    errors.append(f"Metadata file not found for {scan_name}")
                    continue
                
                # Load existing metadata
                with open(json_path, 'r') as jf:
                    metadata = json.load(jf)
                
                # Update threshold value if provided
                if 'threshold' in scan_values:
                    metadata['threshold'] = float(scan_values['threshold'])
                    print(f"Updated threshold value for {scan_name}: {scan_values['threshold']}")
                
                # Save updated metadata
                with open(json_path, 'w') as jf:
                    json.dump(metadata, jf, indent=4)
                
                updated_scans.append(scan_name)
                
            except Exception as e:
                error_msg = f"Error updating {scan_name}: {str(e)}"
                errors.append(error_msg)
                print(error_msg)
        
        response_data = {
            'status': 'success',
            'message': f'Updated threshold values for {len(updated_scans)} scans',
            'updated_scans': updated_scans
        }
        
        if errors:
            response_data['errors'] = errors
            response_data['message'] += f' with {len(errors)} errors'
        
        return Response(response_data, status=status.HTTP_200_OK)

class GetRotationMatrixView(APIView):
    """
    View to retrieve the rotation matrix from scan metadata.
    Returns the rotation matrix if it exists in the 'rotated_by_matrix' field,
    otherwise returns a 3x3 identity matrix (representing no rotation).
    """
    def post(self, request):
        """
        POST endpoint to get rotation matrix for a scan.
        
        Expected request data:
        {
            'directory': str,           # Path to the scans directory
            'scan_name': str            # Name of the scan
        }
        
        Returns:
        {
            'status': 'success' or 'error',
            'rotation_matrix': list of lists (3x3 matrix),
            'message': str (description of result)
        }
        """
        directory = request.data.get('directory')
        scan_name = request.data.get('scan_name')
        
        if not directory or not scan_name:
            return Response({
                'status': 'error',
                'message': 'Directory and scan_name are required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Identity matrix (no rotation)
        identity_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        
        try:
            # Path to the scan's metadata JSON file
            json_path = os.path.join(directory, "extracted", scan_name, f"{scan_name}.json")
            
            if not os.path.exists(json_path):
                # Return identity matrix if metadata file doesn't exist
                return Response({
                    'status': 'success',
                    'rotation_matrix': identity_matrix,
                    'message': 'Metadata file not found, returning identity matrix'
                }, status=status.HTTP_200_OK)
            
            # Load metadata
            with open(json_path, 'r') as jf:
                metadata = json.load(jf)
            
            # Extract rotation matrix if it exists
            if 'rotated_by_matrix' in metadata and 'rotation_matrix' in metadata['rotated_by_matrix']:
                rotation_matrix = metadata['rotated_by_matrix']['rotation_matrix']
                return Response({
                    'status': 'success',
                    'rotation_matrix': rotation_matrix,
                    'message': 'Rotation matrix retrieved successfully'
                }, status=status.HTTP_200_OK)
            else:
                # Return identity matrix if rotation matrix doesn't exist in metadata
                return Response({
                    'status': 'success',
                    'rotation_matrix': identity_matrix,
                    'message': 'Rotation matrix not found in metadata, returning identity matrix'
                }, status=status.HTTP_200_OK)
        
        except Exception as e:
            error_msg = f"Error retrieving rotation matrix for {scan_name}: {str(e)}"
            print(error_msg)
            return Response({
                'status': 'error',
                'message': error_msg,
                'rotation_matrix': identity_matrix
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)