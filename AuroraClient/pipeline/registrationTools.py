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
import math
import numpy as np
import nibabel as nib
from nibabel.filebasedimages import ImageFileError
import imageio
import aim2numpy
from scipy.ndimage import gaussian_filter
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
from skimage.filters import threshold_otsu
import skimage.morphology
from scipy.signal import convolve
from .ALPACA import ALPACA
from skimage import exposure
import os
# Set the number of threads for ITK to utilize
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(psutil.cpu_count(logical=True))
import ants
import shutil
from scipy.interpolate import RegularGridInterpolator
import open3d as o3d
import fast_simplification
import gc  # Add garbage collection
from pydicom import dcmread
from scipy import stats


def _isometric_preview_rotation_matrix():
    """Rotation matching previewScan isometric projection (ix, iy, iz array indices)."""
    cos30 = math.cos(math.pi / 6)
    sin30 = math.sin(math.pi / 6)
    screen_x = np.array([-cos30, 0.0, cos30], dtype=np.float64)
    screen_x /= np.linalg.norm(screen_x)
    screen_y = np.array([sin30, -1.0, sin30], dtype=np.float64)
    screen_y /= np.linalg.norm(screen_y)
    view_depth = np.cross(screen_x, screen_y)
    view_depth /= np.linalg.norm(view_depth)
    return np.column_stack([screen_x, screen_y, view_depth])


PROJECTION_DEPTH_DECAY = 2.0
PROJECTION_VTK_WINDOW_MAX = 768

# Tunable defaults — run Aurora/tune_projection_render.py to adjust interactively.
PROJECTION_VTK_VIEW_SIGN = 1
PROJECTION_VTK_UP_SIGN = 1
PROJECTION_VTK_FLIP_VERTICAL = True
PROJECTION_VTK_GREY_LEVEL = 255
PROJECTION_VTK_EXPOSURE = 1.0
PROJECTION_VTK_LIFT = 0.28
PROJECTION_VTK_ALPHA_CAP = 255
PROJECTION_VTK_ALPHA_PERCENTILE = 94
PROJECTION_VTK_SHADE = False
PROJECTION_VTK_AMBIENT = 0.999
PROJECTION_VTK_DIFFUSE = 0.999
PROJECTION_VTK_OPACITY_LOW = 0.25
PROJECTION_VTK_OPACITY_HIGH = 0.75
PROJECTION_VTK_INVERT = True
# Pillow PNG compress_level: 0 = fast/large, 9 = slow/smallest (lossless pixels either way).
PROJECTION_PNG_COMPRESS_LEVEL = 7


def save_projection_png(rgba, projection_file):
    """Write a projection RGBA array to disk with shared PNG compression settings."""
    Image.fromarray(rgba, 'RGBA').save(
        projection_file,
        compress_level=PROJECTION_PNG_COMPRESS_LEVEL,
        optimize=True,
    )


def _ensure_vtk_headless():
    """Prefer OSMesa only on headless Linux.

    Windows/macOS do not use the X11 DISPLAY variable. Forcing
    vtkOSOpenGLRenderWindow there looks for osmesa.dll / OSMesa and can
    access-violate (Windows exit 0xC0000005) instead of raising a Python
    exception — which kills Daphne before the numpy fallback can run.
    """
    if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
        os.environ.setdefault('VTK_DEFAULT_OPENGL_WINDOW', 'vtkOSOpenGLRenderWindow')
        return

    # Clear stale OSMesa preference left by older Aurora builds that always
    # forced vtkOSOpenGLRenderWindow whenever DISPLAY was unset.
    if (
        not sys.platform.startswith('linux')
        and os.environ.get('VTK_DEFAULT_OPENGL_WINDOW') == 'vtkOSOpenGLRenderWindow'
    ):
        os.environ.pop('VTK_DEFAULT_OPENGL_WINDOW', None)


def _prefer_vtk_projection():
    """Whether to attempt VTK/PyVista projection rendering."""
    return os.environ.get('AURORA_PROJECTION_FALLBACK', '').strip().lower() not in ('1', 'true', 'yes')


def _window_volume_for_projection(image_data_8bit):
    proj_min = float(np.percentile(image_data_8bit, 2))
    proj_max = float(np.percentile(image_data_8bit, 99))
    if proj_max > proj_min:
        working_volume = np.clip(image_data_8bit.astype(np.float32), proj_min, proj_max) - proj_min
        working_volume /= proj_max - proj_min
    else:
        max_val = float(np.max(image_data_8bit))
        working_volume = image_data_8bit.astype(np.float32)
        if max_val > 0:
            working_volume /= max_val
    return working_volume


def _rotate_volume_to_isometric_view(working_volume):
    shape = np.array(working_volume.shape, dtype=np.float64)
    center = (shape - 1.0) / 2.0
    rotation_matrix = _isometric_preview_rotation_matrix()
    origin_offset = center - rotation_matrix.T @ center
    return ndimage.affine_transform(
        working_volume,
        rotation_matrix.T,
        offset=origin_offset,
        output_shape=working_volume.shape,
        order=1,
        mode='constant',
        cval=0.0,
        prefilter=False,
    )


def _compose_projection_rgba_from_render(
    rendered,
    *,
    flip_vertical=PROJECTION_VTK_FLIP_VERTICAL,
    grey_level=PROJECTION_VTK_GREY_LEVEL,
    exposure=PROJECTION_VTK_EXPOSURE,
    lift=PROJECTION_VTK_LIFT,
    alpha_cap=PROJECTION_VTK_ALPHA_CAP,
    alpha_percentile=PROJECTION_VTK_ALPHA_PERCENTILE,
    invert=PROJECTION_VTK_INVERT,
):
    """Map VTK greyscale raycast to grey RGBA with optional inversion and brightness lift."""
    image = np.flipud(rendered) if flip_vertical else rendered.copy()
    luminance = image[..., :3].astype(np.float32).mean(axis=2) / 255.0
    alpha = image[..., 3].astype(np.float32)
    foreground = alpha > 0

    if np.any(foreground):
        lum_ref = float(np.percentile(luminance[foreground], alpha_percentile))
        if lum_ref <= 0:
            lum_ref = float(np.max(luminance))
        lum_norm = np.clip(luminance / lum_ref, 0.0, 1.0)
        if invert:
            lum_norm = 1.0 - lum_norm
        # lift compresses [0,1] → [lift,1] so dense tissue never goes pure black
        lum_lifted = lift + lum_norm * (1.0 - lift)
        value = np.clip(lum_lifted * exposure * grey_level, 0, 255).astype(np.uint8)

        alpha_ref = float(np.percentile(alpha[foreground], alpha_percentile))
        if alpha_ref <= 0:
            alpha_ref = float(np.max(alpha))
        alpha = np.clip(alpha / alpha_ref * alpha_cap, 0, 255).astype(np.uint8)
    else:
        value = np.zeros(image.shape[:2], dtype=np.uint8)
        alpha = np.zeros(image.shape[:2], dtype=np.uint8)

    rgba = np.zeros(image.shape, dtype=np.uint8)
    rgba[..., 0] = value
    rgba[..., 1] = value
    rgba[..., 2] = value
    rgba[..., 3] = alpha
    return rgba


def _projection_back_file(projection_file):
    base, ext = os.path.splitext(projection_file)
    return f"{base}_back{ext}"


def _generate_lossy_isometric_projection_png_vtk(
    working_volume,
    projection_file,
    rgb=(255, 255, 255),
    view_sign=PROJECTION_VTK_VIEW_SIGN,
):
    _ensure_vtk_headless()
    import pyvista as pv

    pv.OFF_SCREEN = True
    shape = working_volume.shape
    grid = pv.ImageData(dimensions=shape, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))
    grid.point_data['scalars'] = working_volume.flatten(order='F')

    rotation_matrix = _isometric_preview_rotation_matrix()
    center = np.array([(dim - 1.0) / 2.0 for dim in shape], dtype=np.float64)
    view_dir = rotation_matrix[:, 2]
    screen_up = rotation_matrix[:, 1]
    camera_distance = float(max(shape) * 3.0)
    window_size = int(min(PROJECTION_VTK_WINDOW_MAX, max(max(shape) * 8, 256)))

    opacity = [
        0.0,
        0.0,
        PROJECTION_VTK_OPACITY_LOW,
        PROJECTION_VTK_OPACITY_LOW * 3.0,
        PROJECTION_VTK_OPACITY_HIGH * 0.65,
        PROJECTION_VTK_OPACITY_HIGH,
    ]

    plotter = pv.Plotter(off_screen=True, window_size=[window_size, window_size])
    plotter.set_background([0.0, 0.0, 0.0, 0.0])
    plotter.add_volume(
        grid,
        scalars='scalars',
        clim=[0.0, 1.0],
        opacity=opacity,
        cmap='Greys',
        shade=PROJECTION_VTK_SHADE,
        show_scalar_bar=False,
        ambient=PROJECTION_VTK_AMBIENT,
        diffuse=PROJECTION_VTK_DIFFUSE,
        specular=0.1 if PROJECTION_VTK_SHADE else 0.0,
    )
    plotter.enable_parallel_projection()
    plotter.camera.position = center + view_dir * camera_distance * view_sign
    plotter.camera.focal_point = center
    plotter.camera.up = screen_up * PROJECTION_VTK_UP_SIGN
    plotter.reset_camera()
    plotter.hide_axes()

    rendered = plotter.screenshot(return_img=True, transparent_background=True)
    plotter.close()

    rgba = _compose_projection_rgba_from_render(rendered)
    save_projection_png(rgba, projection_file)


def _generate_lossy_isometric_projection_png_fallback(
    working_volume,
    projection_file,
    rgb=(255, 255, 255),
    back_view=False,
):
    rotated_volume = _rotate_volume_to_isometric_view(working_volume)
    if back_view:
        rotated_volume = rotated_volume[:, :, ::-1]

    depth = rotated_volume.shape[2]
    depth_weights = np.exp(-PROJECTION_DEPTH_DECAY * np.linspace(0.0, 1.0, depth, dtype=np.float32))
    projection = np.sum(rotated_volume * depth_weights, axis=2)
    projection = np.flipud(projection)

    foreground = projection > 0
    if np.any(foreground):
        projection_ref = float(np.percentile(projection[foreground], 99))
        if projection_ref <= 0:
            projection_ref = float(np.max(projection))
        alpha = np.clip((projection / projection_ref) * 255.0, 0, 255).astype(np.uint8)
    else:
        alpha = np.zeros(projection.shape, dtype=np.uint8)

    rgba = np.zeros(projection.shape + (4,), dtype=np.uint8)
    rgba[..., 0] = rgb[0]
    rgba[..., 1] = rgb[1]
    rgba[..., 2] = rgb[2]
    rgba[..., 3] = alpha
    save_projection_png(rgba, projection_file)


def generate_lossy_isometric_projection_png(image_data_8bit, projection_file, rgb=(255, 255, 255)):
    """Build a transparent PNG preview via VTK volume raycasting (isometric orthographic view)."""
    working_volume = _window_volume_for_projection(image_data_8bit)
    back_projection_file = _projection_back_file(projection_file)
    if not _prefer_vtk_projection():
        print("Using weighted-sum projection fallback (VTK projection disabled for this platform/env).")
        _generate_lossy_isometric_projection_png_fallback(working_volume, projection_file, rgb=rgb)
        _generate_lossy_isometric_projection_png_fallback(
            working_volume,
            back_projection_file,
            rgb=rgb,
            back_view=True,
        )
        return
    try:
        _generate_lossy_isometric_projection_png_vtk(working_volume, projection_file, rgb=rgb)
        _generate_lossy_isometric_projection_png_vtk(
            working_volume,
            back_projection_file,
            rgb=rgb,
            view_sign=-PROJECTION_VTK_VIEW_SIGN,
        )
    except Exception as exc:
        print(f"VTK projection failed ({exc}); falling back to weighted sum projection.")
        _generate_lossy_isometric_projection_png_fallback(working_volume, projection_file, rgb=rgb)
        _generate_lossy_isometric_projection_png_fallback(
            working_volume,
            back_projection_file,
            rgb=rgb,
            back_view=True,
        )


class RegistrationTools():

    def get_background_value(self, image_data, mode="peak-no-zero", threshold=None):
        # Get the original data type
        original_dtype = image_data.dtype        
        
        if mode == "peak-no-zero":
            # Calculate the histogram of the Gaussian smoothed nifti data
            nifti_data_gaussian = gaussian_filter(image_data, sigma=3)
            hist, bin_edges = np.histogram(nifti_data_gaussian.flatten(), bins=256)
            # Find all peaks
            peaks, peak_properties = signal.find_peaks(hist, distance=5)
            
            if len(peaks) == 0:
                # If no peaks found, use minimum value
                background_value = np.min(nifti_data_gaussian).astype(original_dtype)
            else:
                # Get the peak heights (frequencies)
                peak_heights = hist[peaks]
                
                # Sort peaks by height (frequency) in descending order and get indices
                sorted_indices = np.argsort(peak_heights)[::-1]
                
                # Get the two highest frequency peaks (or all if less than 2)
                top_peaks = peaks[sorted_indices[:min(2, len(peaks))]]
                
                # From these peaks, choose the one with lowest intensity value
                peak_intensities = bin_edges[top_peaks]
                background_peak_idx = top_peaks[np.argmin(peak_intensities)]
                
                # Calculate the average value in the bin range of the selected peak
                peak_bin_start = bin_edges[background_peak_idx]
                peak_bin_end = bin_edges[background_peak_idx + 1]
                peak_mask = (nifti_data_gaussian >= peak_bin_start) & (nifti_data_gaussian < peak_bin_end)
                background_value = int(np.mean(nifti_data_gaussian[peak_mask]))
        elif mode == "border":

            # assign the image data to the gaussian variable, small smoothing
            nifti_data_gaussian = gaussian_filter(image_data, sigma=1.0)

            # Get the background value as the mean of the largest peak in the histogram comprised of only the 2px border pixels
            border_mask = np.zeros_like(nifti_data_gaussian, dtype=bool)
            border_mask[0:2, :, :] = True
            border_mask[-2:, :, :] = True
            border_mask[:, 0:2, :] = True
            border_mask[:, -2:, :] = True
            border_mask[:, :, 0:2] = True
            border_mask[:, :, -2:] = True
            if threshold is not None:
                border_mask[nifti_data_gaussian > threshold] = False
            # get histogram of border pixels
            border_pixels = nifti_data_gaussian[border_mask]
            print("mean of border pixels: ", np.mean(border_pixels).astype(original_dtype))            
            #background_value = int(stats.mode(border_pixels, axis=None, keepdims=False).mode)
            if np.std(border_pixels) > 1:
                hist, bin_edges = np.histogram(border_pixels.flatten(), bins=256)
                # get lower edge of the highest peak
                background_value = bin_edges[np.argmax(hist)].astype(original_dtype)
            else:
                background_value = np.floor(np.mean(border_pixels)).astype(original_dtype)
            print("mode of border pixels: ", background_value)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        return background_value


    def get_adjusted_background_value(self, background_value, image_data, new_min, new_max):
        # find the corresponding value in the new range
        return (background_value - np.min(image_data)) / (np.max(image_data) - np.min(image_data)) * (new_max - new_min) + new_min

    def load_nifti_mask_data(self, mask_path):
        """Load mask voxel data from disk.

        NiBabel's type sniffer often fails on ``*.nii.mask.gz`` (non-standard double suffix).
        Same workaround as elsewhere in this project: copy to a ``*_temp.nii.gz`` path.
        """
        try:
            return nib.load(mask_path).get_fdata()
        except ImageFileError:
            fd, tmp_path = tempfile.mkstemp(suffix='_mask_temp.nii.gz')
            os.close(fd)
            try:
                shutil.copyfile(mask_path, tmp_path)
                return nib.load(tmp_path).get_fdata()
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def min_max_normalize(self, image, new_min=0, new_max=1):
        """
        Normalizes image values to a new range while preserving relative differences.
        Memory-efficient implementation that minimizes array copies.
        
        Args:
            image: Input image array
            new_min: Target minimum value (default: 0)
            new_max: Target maximum value (default: 1)
        
        Returns:
            Normalized image array in the new range
        """
        # Get the current range before modifying the array
        img_min = np.min(image)
        img_max = np.max(image)
        
        # Avoid division by zero
        if img_max == img_min:
            return np.full_like(image, new_min, dtype=np.float32)
        
        # Create a single output array (only one copy)
        result = image.astype(np.float32, copy=True)
        
        # Perform all operations in-place
        # Normalize to [0,1] range
        result -= img_min
        result /= (img_max - img_min)
        
        # Scale to new range
        result *= (new_max - new_min)
        result += new_min
        
        # Clip values in-place
        np.clip(result, new_min, new_max, out=result)
        
        return result

    def restore_range(self, image, original_min, original_max):
        """
        Restores the original range of values while preserving relative differences.
        Memory-efficient implementation that minimizes array copies.
        
        Args:
            image: Input image array
            original_min: Original minimum value to restore to
            original_max: Original maximum value to restore to
        
        Returns:
            Image array with restored range
        """
        # Get current min/max before modifying the array
        img_min = np.min(image)
        img_max = np.max(image)
        
        # Avoid division by zero
        if img_max == img_min:
            return np.full_like(image, original_min, dtype=np.float32)
        
        # Create a single output array
        result = image.astype(np.float32, copy=True)
        
        # Perform all operations in-place
        # Normalize to [0,1] range
        result -= img_min
        result /= (img_max - img_min)
        
        # Scale to original range
        result *= (original_max - original_min)
        result += original_min
        
        # Clip values in-place
        np.clip(result, original_min, original_max, out=result)
        
        return result

    def save_as_lossy_nifti(self, image_data, voxel_size, json_file, output_file):
        # Load existing metadata
        with open(json_file, 'r') as jf:
            json_metadata = json.load(jf)

        # Resolve resolution_factor (default 2)
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

        # Mean-pool NxNxN voxels (anti-aliases without stride sampling). Pad each axis to
        # ceil(s/sf)*sf so the output shape matches stride [::sf] (ceil(s/sf) per axis),
        # same as the rest of the pipeline (e.g. lossy masks in views.py).
        # Old method:
        # image_data = gaussian_filter(image_data, sigma=1.0)
        # slices = [slice(None, None, resolution_factor) for _ in range(3)]
        # downscaled_volume = image_data[slices[0], slices[1], slices[2]]
        sf = resolution_factor
        sh = image_data.shape
        padded_shape = tuple(math.ceil(s / sf) * sf for s in sh)
        pad_widths = [(0, p - s) for s, p in zip(sh, padded_shape)]
        padded = np.pad(image_data, pad_widths, mode='edge')
        downscaled_volume = measure.block_reduce(
            padded, block_size=(sf, sf, sf), func=np.mean
        )
        #downscaled_volume = gaussian_filter(downscaled_volume, sigma=0.3)  # optional post-pool blur for JPEG size; comment out to disable
        image_data = None
        gc.collect()

        # Compute per-call mapping based on min/max
        downscaled_volume = downscaled_volume.astype(np.float32, copy=False)
        data_min = float(np.min(downscaled_volume))
        data_max = float(np.max(downscaled_volume))

        if data_max > data_min:
            scale_factor = (data_max - data_min) / 255.0
            zero_point_shift = data_min
            normalized = (downscaled_volume - zero_point_shift) / scale_factor
        else:
            scale_factor = 1.0
            zero_point_shift = data_min
            normalized = np.zeros_like(downscaled_volume, dtype=np.float32)

        image_data_8bit = np.clip(normalized, 0, 255).astype(np.uint8)

        downscaled_volume = None
        normalized = None
        gc.collect()

        # Generate projection PNG
        try:
            projection_file = output_file.replace('.nii.gz', '_projection.png')
            generate_lossy_isometric_projection_png(image_data_8bit, projection_file)
        except Exception as e:
            print(f"Warning: could not generate projection PNG: {e}")

        # JPEG compress per slice
        compressed_slices = []

        # Compress each slice using JPEG
        for i in range(image_data_8bit.shape[2]):
            slice_data = image_data_8bit[:, :, i]
            
            # Compress the slice using JPEG and store it in a temporary buffer
            zero_mask = (slice_data == 0)
            buffer = imageio.imwrite('<bytes>', slice_data, format='JPEG', quality=70)
            
            # Read the compressed slice back into a numpy array
            compressed_slice = imageio.imread(buffer, format='JPEG')
            
            # Append the compressed slice to the list
            compressed_slice[zero_mask] = 0  # enforce exact zeros
            compressed_slices.append(compressed_slice)
            
            # After compressing each slice, clear buffers
            buffer = None
            gc.collect()  # Optional: only do this every few slices if there are many

        # Clear 8-bit data once we're done with it
        image_data_8bit = None
        gc.collect()

        # Stack the compressed slices back into a 3D numpy array
        compressed_image_data = np.stack(compressed_slices, axis=-1)
        compressed_slices = None
        affine = np.diag([voxel_size * resolution_factor] * 3 + [1])
        
        # Create and save the NIfTI image using the provided output_file
        nifti_img = nib.Nifti1Image(compressed_image_data, affine)
        nib.save(nifti_img, output_file)
        
        # After saving, free the last large array
        compressed_image_data = None
        gc.collect()

        # Prepare entry and append to lossy_compression array
        new_entry = {
            "filename": os.path.basename(output_file),
            "scale_factor": float(scale_factor),
            "zero_point_shift": float(zero_point_shift),
            "resolution_factor": int(resolution_factor),
            "compression_info": "JPEG compression with quality 70, 8-bit per pixel",
        }

        # Ensure array shape under 'lossy_compression'
        if isinstance(lc, list):
            lossy_list = lc
        elif isinstance(lc, dict) and lc:
            # migrate legacy single dict into array once
            legacy = dict(lc)
            if "filename" not in legacy:
                legacy["filename"] = ""  # legacy unknown
            lossy_list = [legacy]
        else:
            lossy_list = []

        lossy_list.append(new_entry)
        json_metadata["lossy_compression"] = lossy_list

        with open(json_file, 'w') as jf:
            json.dump(json_metadata, jf, indent=4)
        # Callers that later dump an older in-memory metadata dict must reload
        # or merge this list; returning it makes that merge explicit.
        return json_metadata

    def calculate_affine_transform_as_field(self, fixed_image_data, moving_image_data):
        """
        Calculate an affine transformation between fixed and moving images and convert it to a displacement field.

        Steps:
          1. Find the best fitting transformation matrix and shift vector that
             aligns the moving image to the fixed image.
          2. Build a 3D grid of voxel coordinates in the fixed image.
          3. Convert those grid coordinates from voxel units to real-world
             spatial coordinates.
          4. Apply the matrix and shift vector to each coordinate to get its
             new position.
          5. Subtract the original coordinates from the moved coordinates to
             get a 3D displacement vector at each point.
          6. Save this 3D array of vectors as a displacement field file, which
             can then warp any same-sized image to align with the fixed image.
        
        Args:
            fixed_image_data: Fixed ANTs image
            moving_image_data: Moving ANTs image
            
        Returns:
            Path to displacement field .nii.gz file and the affine matrix

        Goal:
            When either of them is applied to any image of the same resolution, both should result in the same image.
            
        """
        # Perform Affine registration with Identity initial transform
        
        print("\nPerforming Affine registration with Identity initial_transform...")
        affine_transform = ants.registration(
            fixed=fixed_image_data,
            moving=moving_image_data,
            type_of_transform='Affine',
            aff_iterations=(2100, 1200, 250, 250),
            grad_step=0.3,
            aff_metric='GC',
            initial_transform=["Identity"],
            verbose=True
        )

        # Get path to the resulting .mat affine transformation file
        affine_file_path = affine_transform['fwdtransforms'][0]
        print(f"Affine registration successful. Transform file: {affine_file_path}")
        # Note: On Windows, ANTs will use %TEMP% directory automatically
        # On Unix systems, this will be in /tmp/ managed by tempfile

        # Get grid dimensions and spacing from the fixed image
        volume_dimensions_fixed = fixed_image_data.shape
        voxel_spacing_fixed = fixed_image_data.spacing
        origin_fixed = fixed_image_data.origin
        direction_fixed = fixed_image_data.direction
        
        # print(f"DEBUG - Fixed image shape: {volume_dimensions_fixed}")
        # print(f"DEBUG - Fixed image spacing: {voxel_spacing_fixed}")
        # print(f"DEBUG - Fixed image origin: {origin_fixed}")
        # print(f"DEBUG - Fixed image direction: {direction_fixed}")

        # read the complete transform -----------------------------
        tx   = ants.read_transform(affine_file_path, precision="float")
        A    = np.asarray(tx.parameters[:9],  dtype=np.float32).reshape(3,3)
        b    = np.asarray(tx.parameters[9:12],dtype=np.float32)
        c    = np.asarray(tx.fixed_parameters, dtype=np.float32)
        t    = b + c - A @ c     # correct ITK offset
        # ----------------------------------------------------------
        
        # print(f"DEBUG - Affine matrix A:\n{A}")
        # print(f"DEBUG - Translation vector t: {t}")
        
        # Clean up
        del tx, b, c

        # Unpack image metadata – ITK (X, Y, Z) order
        X, Y, Z           = volume_dimensions_fixed
        dx, dy, dz        = voxel_spacing_fixed            # voxel sizes (mm)
        origin            = np.asarray(origin_fixed, dtype=np.float32)  # (3,)
        D                 = np.asarray(direction_fixed, dtype=np.float32).reshape(3, 3)  # orientation

        # ---------------------------------------------------------------
        # 0. Compute the 4×4 field‐matrix (forward or inverse)
        # ---------------------------------------------------------------
        # If ANTsApplyTransforms expects a backward (inverse) field, set use_inverse=True
        use_inverse = False
        M = np.eye(4, dtype=np.float32)
        M[:3,:3] = A
        M[:3,3]  = t
        if use_inverse:
            M = np.linalg.inv(M).astype(np.float32)
        A_field = M[:3,:3]
        t_field = M[:3,3]
        # print(f"DEBUG - Using {'inverse' if use_inverse else 'forward'} field: A_field=\n{A_field}\nt_field={t_field}")
        # ---------------------------------------------------------------

        # ---------------------------------------------------------------------------
        # 1. Build a grid of voxel indices (i,j,k)
        # ---------------------------------------------------------------------------
        ix, iy, iz = np.meshgrid(
            np.arange(X, dtype=np.float32),
            np.arange(Y, dtype=np.float32),
            np.arange(Z, dtype=np.float32),
            indexing='ij'
        )                                     # each ix/iy/iz has shape (X,Y,Z)

        # Stack → shape (X,Y,Z,3)
        indices = np.stack([ix, iy, iz], axis=-1)           # integer voxel indices

        # ---------------------------------------------------------------------------
        # 2. Convert indices → physical (mm) before the affine
        #    p = D @ (indices * spacing) + origin
        # ---------------------------------------------------------------------------
        spacing = np.array([dx, dy, dz], dtype=np.float32)  # (3,)
        index_mm = indices * spacing                        # broadcast multiply
        world_coords_physical = np.tensordot(index_mm, D.T, axes=1) + origin
        # shape: (X, Y, Z, 3)

        # ---------------------------------------------------------------------------
        # 3. Apply the "field" affine:  p′ = A_field @ p + t_field
        # ---------------------------------------------------------------------------
        transformed_coords = np.tensordot(world_coords_physical, A_field.T, axes=1) + t_field
        # shape: (X, Y, Z, 3)

        # ---------------------------------------------------------------------------
        # 4. Displacement field in mm
        # ---------------------------------------------------------------------------
        displacement_field = transformed_coords - world_coords_physical
        
        # Check displacement field ranges
        # print(f"DEBUG - Displacement field range X: {displacement_field[...,0].min()} to {displacement_field[...,0].max()}")
        # print(f"DEBUG - Displacement field range Y: {displacement_field[...,1].min()} to {displacement_field[...,1].max()}")
        # print(f"DEBUG - Displacement field range Z: {displacement_field[...,2].min()} to {displacement_field[...,2].max()}")
        
        # Reshape to (X, Y, Z, 1, 3) for ANTs
        # print(f"DEBUG - Coordinates before reshape: {displacement_field[120, 120, 120, :]}")
        displacement_field_from_affine = displacement_field.reshape(X, Y, Z, 1, 3)
        # print(f"DEBUG - Coordinates after reshape: {displacement_field_from_affine[120, 120, 120, 0, :]}")
        # Debug override: uniform 2 mm shift to the right (X direction)
        #displacement_field_from_affine = np.zeros((X, Y, Z, 1, 3), dtype=np.float32)
        #displacement_field_from_affine[:, :, :, 0, 0] = 1.0
        
        
        # Run a quick SyNOnly registration to get a properly formatted displacement field file
        quick_syn = ants.registration(
            fixed=fixed_image_data,
            moving=moving_image_data,
            type_of_transform='SyNOnly',
            reg_iterations=[1, 0, 0, 0],  # Minimal iterations
            initial_transform=["Identity"],
            syn_metric='CC',
            syn_sampling=2,
            grad_step=0.3,
            flow_sigma=3,
            total_sigma=0,
            singleprecision=True,
            verbose=True
        )
        
        # Get the displacement field file path
        displacement_field_path = quick_syn['fwdtransforms'][0]
        print(f"Template displacement field created at: {displacement_field_path}")
        # Note: ANTs returns paths in the system temp directory which is cross-platform
        # (uses %TEMP% on Windows, /tmp on Unix)
        
        # Load the template field file to preserve header information
        displacement_field_img = nib.load(displacement_field_path)
        
        # Check template field shape and make sure our calculated field matches
        # print(f"DEBUG - Template field shape: {displacement_field_img.shape}")
        # print(f"DEBUG - Our calculated field shape: {displacement_field_from_affine.shape}")
        
        # Create new image with our calculated displacement field but keep original header
        warp_img = nib.Nifti1Image(displacement_field_from_affine.astype(np.float32),
                                  displacement_field_img.affine, displacement_field_img.header)
        
        # Save back to the same file
        nib.save(warp_img, displacement_field_path)
        
        return affine_file_path, displacement_field_path

    @staticmethod
    def physical_extent_mm(shape_xyz, spacing_xyz):
        """Axis-aligned physical FOV (mm) for an ANTs/NIfTI volume grid."""
        shape = np.asarray(shape_xyz[:3], dtype=np.float64)
        spacing = np.abs(np.asarray(spacing_xyz[:3], dtype=np.float64))
        return shape * spacing

    def affine_for_field_matching_reference_fov(self, reference_ants_image, field_shape_xyz):
        """
        Build a RAS-style diag affine so a field on `field_shape_xyz` spans the
        same physical FOV as `reference_ants_image`.

        Shrink-path elastic warps live on a coarser grid; ants.apply_transforms
        samples them in physical space, so FOV must match the full-res volumes
        even when the voxel grid does not.
        """
        field_shape = np.asarray(field_shape_xyz[:3], dtype=np.float64)
        if np.any(field_shape <= 0):
            raise ValueError(f"Invalid field shape: {field_shape_xyz}")

        ref_extent = self.physical_extent_mm(
            reference_ants_image.shape[:3], reference_ants_image.spacing
        )
        field_spacing = ref_extent / field_shape
        # Aurora volumes use origin-0 positive-diag affines; keep the same
        # convention so ants.image_read remaps direction consistently.
        affine = np.diag([
            float(field_spacing[0]),
            float(field_spacing[1]),
            float(field_spacing[2]),
            1.0,
        ])
        return affine, tuple(float(s) for s in field_spacing)

    def save_displacement_field_nifti(self, path, field_data, reference_ants_image):
        """
        Save a displacement field with FOV locked to `reference_ants_image`.

        Accepts (X,Y,Z,3) or (X,Y,Z,1,3). Builds a fresh NIfTI header from the
        FOV-matched affine only — reusing a stale ANTs header alongside a new
        affine can leave pixdim out of sync and break later apply_transforms.
        """
        data = np.asarray(field_data, dtype=np.float32)
        if data.ndim == 4 and data.shape[-1] == 3:
            data = data[:, :, :, np.newaxis, :]
        if data.ndim != 5 or data.shape[3] != 1 or data.shape[4] != 3:
            raise ValueError(f"Expected displacement field (X,Y,Z,1,3), got {data.shape}")

        affine, field_spacing = self.affine_for_field_matching_reference_fov(
            reference_ants_image, data.shape[:3]
        )
        img = nib.Nifti1Image(data, affine)
        img.header.set_intent('vector')
        img.header.set_xyzt_units('mm')
        nib.save(img, path)
        print(
            f"Saved displacement field {os.path.basename(path)}: "
            f"grid={tuple(int(x) for x in data.shape[:3])} spacing={field_spacing} "
            f"(FOV matched to reference {tuple(int(x) for x in reference_ants_image.shape[:3])} "
            f"@ {tuple(float(s) for s in reference_ants_image.spacing)})"
        )
        return path

    def ensure_displacement_field_fov(
        self,
        field_path,
        reference_ants_image,
        rtol=0.02,
        atol_mm=0.05,
        rewrite_in_place=False,
    ):
        """
        If a displacement field's physical extent disagrees with the reference
        image, rewrite its affine so FOV matches (vector data unchanged).

        Returns the path callers should pass to ants.apply_transforms (original,
        in-place rewrite, or a temp file).
        """
        if not field_path or not os.path.isfile(field_path):
            return field_path

        field_nib = nib.load(field_path)
        field_extent = self.physical_extent_mm(
            field_nib.shape[:3], field_nib.header.get_zooms()[:3]
        )
        ref_extent = self.physical_extent_mm(
            reference_ants_image.shape[:3], reference_ants_image.spacing
        )

        if np.allclose(field_extent, ref_extent, rtol=rtol, atol=atol_mm):
            return field_path

        ratio = field_extent / np.maximum(ref_extent, 1e-12)
        print(
            f"Displacement field FOV mismatch for {os.path.basename(field_path)}: "
            f"field_extent={field_extent} ref_extent={ref_extent} ratio={ratio}. "
            f"Rewriting affine to match reference FOV "
            f"({'in-place' if rewrite_in_place else 'temp file'})."
        )

        data = np.asanyarray(field_nib.dataobj)
        if rewrite_in_place:
            out_path = field_path
        else:
            fd, out_path = tempfile.mkstemp(
                prefix=os.path.splitext(os.path.basename(field_path))[0] + '_',
                suffix='_fov_fixed.nii.gz',
            )
            os.close(fd)

        self.save_displacement_field_nifti(out_path, data, reference_ants_image)
        return out_path


    def transform_landmarks_by_rotation(self, landmarks_list, voxel_size, rotation_matrix, center_box, origin_offset, final_bbox_min, final_bbox_max, embedding_offset):
        """
        Transform landmarks through the same rotation and cropping as the image.
        
        Args:
            landmarks_list: List of [x,y,z] physical coordinates (mm) or objects with 'position' and 'landmark_type'
            voxel_size: Voxel size in mm
            rotation_matrix: 3x3 rotation matrix applied to image
            center_box: Center point in voxel space used for rotation
            origin_offset: Offset for rotation transformation
            final_bbox_min: Minimum coordinates of final crop in voxel space
            final_bbox_max: Maximum coordinates of final crop in voxel space
        
        Returns:
            Transformed landmarks list of objects with 'position' and 'landmark_type' fields
        """
        transformed_landmarks = []
        
        for idx, landmark_data in enumerate(landmarks_list):
            try:
                # Extract position and landmark_type from either format (array or object)
                if isinstance(landmark_data, dict):
                    landmark_mm = landmark_data.get('position', landmark_data)
                    landmark_type = landmark_data.get('landmark_type', 'main')
                else:
                    # Array format - default to 'main' type
                    landmark_mm = landmark_data
                    landmark_type = 'main'

                # Convert from physical space (mm) to voxel space
                landmark_voxel = np.array(landmark_mm) / float(voxel_size)

                # Convert to target_box space
                landmark_voxel_target = landmark_voxel - embedding_offset

                # Then use landmark_voxel_target instead of landmark_voxel in the rotation:
                rotated_point = np.dot(rotation_matrix, landmark_voxel_target - origin_offset)
                                                
                # Apply the same final cropping as the image
                # Check if point is within crop bounds
                if (final_bbox_min[0] <= rotated_point[0] < final_bbox_max[0] and
                    final_bbox_min[1] <= rotated_point[1] < final_bbox_max[1] and
                    final_bbox_min[2] <= rotated_point[2] < final_bbox_max[2]):
                    
                    # Adjust for cropping offset
                    cropped_point = rotated_point - final_bbox_min
                    
                    # Convert back to physical space (mm)
                    landmark_transformed_mm = cropped_point * float(voxel_size)
                    
                    # Always append as object with position and landmark_type
                    transformed_landmarks.append({
                        'position': landmark_transformed_mm.tolist(),
                        'landmark_type': landmark_type
                    })
                    print(f"Landmark {idx}: {landmark_mm} -> {landmark_transformed_mm.tolist()}")
                else:
                    print(f"Landmark {idx} at voxel {landmark_voxel} -> rotated {rotated_point} is outside crop bounds [{final_bbox_min} to {final_bbox_max}], excluding")
                    
            except Exception as e:
                print(f"Error transforming landmark {idx}: {e}")
        
        return transformed_landmarks

    def transform_landmarks_by_crop(self, landmarks_list, voxel_size, z_min, z_max, y_min, y_max, x_min, x_max, padding):
        """
        Transform landmarks through the same cropping and padding as the image.
        
        Args:
            landmarks_list: List of [x,y,z] physical coordinates (mm) or objects with 'position' and 'landmark_type'
            voxel_size: Voxel size in mm
            z_min, z_max, y_min, y_max, x_min, x_max: Crop boundaries in voxel space
            padding: Padding added after cropping in voxels
        
        Returns:
            Transformed landmarks list of objects with 'position' and 'landmark_type' fields
        """
        transformed_landmarks = []
        
        for idx, landmark_data in enumerate(landmarks_list):
            try:
                # Extract position and landmark_type from either format (array or object)
                if isinstance(landmark_data, dict):
                    landmark_mm = landmark_data.get('position', landmark_data)
                    landmark_type = landmark_data.get('landmark_type', 'main')
                else:
                    # Array format - default to 'main' type
                    landmark_mm = landmark_data
                    landmark_type = 'main'

                # Convert from physical space (mm) to voxel space
                landmark_voxel = np.array(landmark_mm) / float(voxel_size)
                
                # Check if landmark is within crop region
                if (z_min <= landmark_voxel[0] < z_max and
                    y_min <= landmark_voxel[1] < y_max and
                    x_min <= landmark_voxel[2] < x_max):
                    
                    # Apply crop
                    cropped_voxel = np.array([
                        landmark_voxel[0] - z_min,
                        landmark_voxel[1] - y_min,
                        landmark_voxel[2] - x_min
                    ])
                    
                    # Apply padding
                    padded_voxel = cropped_voxel + padding
                    
                    # Convert back to physical space (mm)
                    landmark_transformed_mm = padded_voxel * float(voxel_size)
                    
                    # Always append as object with position and landmark_type
                    transformed_landmarks.append({
                        'position': landmark_transformed_mm.tolist(),
                        'landmark_type': landmark_type
                    })
                    print(f"Landmark {idx}: {landmark_mm} -> {landmark_transformed_mm.tolist()}")
                else:
                    print(f"Landmark {idx} at voxel {landmark_voxel} is outside crop region, excluding")
                    
            except Exception as e:
                print(f"Error transforming landmark {idx}: {e}")
        
        return transformed_landmarks