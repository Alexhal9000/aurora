"""Regular Grid View previews, generated in parallel worker processes (Aurora 1.9.4+).

A regular grid tile is a 2x2 JPEG built from the scan's lossy (half-resolution, uint8)
volume: z-sum projection, a 3D marching-cubes view, and XZ / YZ slices through the
centroid (create_preview_image). It is cached next to the scan as <scan>_preview.jpg plus
<scan>_preview.json and only rebuilt when the latest edit or the threshold changes.

Building one takes seconds (loading the volume, histogram, centroid, marching cubes,
matplotlib render), and the engine used to build them one at a time: the UI asked for one
scan per request, and Daphne runs the synchronous DRF views on a single thread, so a page
of new scans kept one core busy while the rest of the machine sat idle.

This module holds the preview code itself, moved unchanged in substance from
pipeline/views.py (GetGridView keeps thin wrappers, so the single-scan /get-grid-view/
endpoint behaves exactly as before), plus a process pool that GetGridViewBatch uses to
build the previews of one page in parallel:

    from pipeline import gridPreviewWorker as gpw
    outcomes = gpw.generate_previews([(scan_name, directory, ignore_elastic), ...])
    outcomes[i]["image_bytes"] or outcomes[i]["error"]

Pool sizing ("as much of the machine as is safe"):
  * workers = logical cores - 1 (one core stays free for the engine and the browser),
    capped by _MAX_WORKERS and by what free RAM can hold.
  * Every task is admitted against a RAM budget: free RAM minus a safety reserve
    (max(_RAM_RESERVE_MIN_GB, _RAM_RESERVE_FRACTION of total RAM)) minus the workers'
    own baseline. A task's cost is estimated from its lossy volume's voxel count
    (header only, see estimate_task_bytes), so a page of large scans runs fewer at a time
    instead of running out of memory. One task always runs, even if it exceeds the budget
    (that is the old serial behaviour).
  * The pool uses ``spawn`` (never ``fork``: the engine has threads and OpenMP runtimes),
    and is kept alive between pages so workers import NumPy/SciPy/scikit-image/matplotlib
    only once; it shuts down after _POOL_IDLE_SECONDS without grid work.
  * AURORA_GRID_WORKERS=N forces N workers; 1 restores the previous in-engine serial build.
  * If the pool cannot start or a worker dies (e.g. out of memory), the remaining
    previews are built in the engine and the next request starts a fresh pool.

Previews are identical to the serial build: each one is produced by the same function
from the same files, only in another process.
"""

from __future__ import annotations

import gc
import glob
import io
import json
import math
import os
import threading
import time

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (workers have no display)
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter
from skimage import exposure, measure

from .atlas_paths import resolve_atlas_dir
# Crash-safe protocol (atomic writes): see crashSafety.py.
from .crashSafety import atomic_write_json, atomic_write_bytes

# Render size of one preview quadrant and JPEG quality. The values are defined in
# meshGridTools.py (shared with the preserved-mesh previews); the engine passes them in
# through configure() and the pool hands them to every worker, so this module does not
# have to import meshGridTools (and its VTK/registration stack) in each worker.
QUADRANT_RENDER_SIZE = 512
GRID_PREVIEW_JPEG_QUALITY = 80


def configure(quadrant_render_size, jpeg_quality):
    """Set the render constants (called by views.py at import, and in every worker)."""
    global QUADRANT_RENDER_SIZE, GRID_PREVIEW_JPEG_QUALITY
    QUADRANT_RENDER_SIZE = int(quadrant_render_size)
    GRID_PREVIEW_JPEG_QUALITY = int(jpeg_quality)


# --- Grid preview fast paths (exact) -------------------------------------------------
# The regular grid preview is built from the lossy (half-resolution, uint8) volume. It
# used to be loaded with get_fdata(), i.e. a float64 copy at 8x the memory, and every
# step below then worked on floats. These helpers keep the stored uint8 data and give
# results identical to the float64 path (same histogram counts, same centroid, same
# z-sum / block-reduce values), so the preview pixels do not change, only the time.
# Any file that is not plain unscaled uint8 falls back to exactly the old float64 path.

# Voxels per chunk when counting the uint8 histogram (bounds the bincount temporary).
_GRID_PREVIEW_HIST_CHUNK = 16 * 1024 * 1024


def _grid_preview_volume(nii):
    """Return the lossy volume for the grid preview: its stored uint8 array when the
    file holds unscaled uint8 voxels (every lossy file Aurora writes), otherwise
    get_fdata() exactly as before."""
    dataobj = nii.dataobj
    slope = getattr(dataobj, 'slope', 1.0)
    inter = getattr(dataobj, 'inter', 0.0)
    if nii.get_data_dtype() == np.uint8 and slope == 1.0 and inter == 0.0:
        return np.asanyarray(dataobj)
    return nii.get_fdata()


def _grid_preview_histogram(data, bins=65, value_range=(0, 255)):
    """np.histogram(data, bins, range) without converting a uint8 volume to float.

    For uint8 data the 256 possible values are counted once (bincount, in chunks), and
    numpy's own np.histogram then bins those 256 values weighted by their counts, so the
    bin edges and edge handling are numpy's and the counts are identical."""
    if data.dtype != np.uint8:
        return np.histogram(data, bins=bins, range=value_range)
    flat = data.ravel(order='K')
    value_counts = np.zeros(256, dtype=np.int64)
    for start in range(0, flat.size, _GRID_PREVIEW_HIST_CHUNK):
        value_counts += np.bincount(flat[start:start + _GRID_PREVIEW_HIST_CHUNK], minlength=256)
    hist, bin_edges = np.histogram(np.arange(256), bins=bins, range=value_range, weights=value_counts)
    return hist.astype(np.int64), bin_edges


def _binary_center_of_mass(mask):
    """ndimage.center_of_mass for a boolean mask, from per-axis voxel-count profiles.

    The centroid along each axis is sum(index * count) / total; profiles and the
    numerator are exact integers, so this matches center_of_mass to float rounding at
    a fraction of the cost (no full-volume coordinate weighting). Caller ensures the
    mask is not empty."""
    total = int(np.count_nonzero(mask))
    centroid = []
    for axis in range(mask.ndim):
        other_axes = tuple(a for a in range(mask.ndim) if a != axis)
        profile = mask.sum(axis=other_axes, dtype=np.int64)
        weighted = int(np.dot(profile, np.arange(mask.shape[axis], dtype=np.int64)))
        centroid.append(weighted / total)
    return tuple(centroid)


# ---------------------------------------------------------------------------
# Preview rendering (moved from GetGridView in views.py)
# ---------------------------------------------------------------------------

# Face budget for the 3D quadrant (see create_preview_image). matplotlib draws ~17 us per
# face (mouse skulls at step 2: 250-375k faces took 4-7 s), so 175k faces is ~3 s, which
# keeps a cold page of 50 scans at about 10 s with the worker pool (2-3 rounds). Most
# scans stay at the step chosen from the active voxel count; only the heaviest meshes go
# one step coarser. AURORA_GRID_3D_MAX_FACES overrides; 0 disables the cap (previous
# behaviour: the step chosen from the active voxel count only).
_PREVIEW_3D_MAX_FACES = 175_000


def _preview_3d_max_faces():
    try:
        return max(0, int(os.environ.get("AURORA_GRID_3D_MAX_FACES", _PREVIEW_3D_MAX_FACES)))
    except ValueError:
        return _PREVIEW_3D_MAX_FACES


def _preview_mesh_at_step(volume, threshold, spacing_tuple, step):
    """Marching-cubes mesh of the preview volume at a given downsampling step
    (code unchanged from create_preview_image, factored out so the face budget can
    rebuild it at a coarser step). Returns (verts, faces)."""
    # Use Max Pooling (block_reduce) to downsample strategically.
    # Max pooling ensures that thin, high-intensity structures (like edges or small muscles/bones)
    # are preserved and not "eaten away" or blurred below the threshold.
    if step > 1:
        reduced_image_data = measure.block_reduce(
            volume, 
            block_size=(step, step, step), 
            func=np.max
        ).astype(np.float64)  # max-pool on uint8, then float for the blur (as before)

        # Apply a mild Gaussian blur to the downsampled volume to smooth blocky edges
        smoothed_image_data = gaussian_filter(reduced_image_data, sigma=1.0)

        # Since we actually shrunk the array, we tell marching cubes to use step_size=1
        # and compensate by multiplying the physical spacing.
        effective_spacing = (spacing_tuple[0] * step, 
                             spacing_tuple[1] * step, 
                             spacing_tuple[2] * step)
        effective_step = 1
    else:
        # Marching cubes gets the same float64 volume as before.
        smoothed_image_data = volume.astype(np.float64, copy=False)
        effective_spacing = spacing_tuple
        effective_step = 1

    # Run marching cubes
    verts, faces, _, _ = measure.marching_cubes(smoothed_image_data, level=threshold, spacing=effective_spacing, step_size=effective_step)
    return verts, faces


# Helper function to resize images while preserving aspect ratio
def resize_with_aspect_ratio(img, target_size, bg_color="black"):
    """
    Resizes a PIL Image to fit within a target size, preserving aspect ratio.
    This handles both scaling down and scaling up smaller images.
    Pads with a background color if necessary.
    """
    # Calculate aspect ratios
    img_aspect = img.width / img.height
    target_aspect = target_size[0] / target_size[1]

    if img_aspect > target_aspect:
        # Image is wider than target area: fit to target width
        new_width = target_size[0]
        new_height = int(new_width / img_aspect)
    else:
        # Image is taller than target area: fit to target height
        new_height = target_size[1]
        new_width = int(new_height * img_aspect)

    # Resize image (this scales up or down as needed)
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    new_img = Image.new("RGB", target_size, bg_color)
    # Calculate position to paste the scaled image (centered)
    paste_x = (target_size[0] - new_width) // 2
    paste_y = (target_size[1] - new_height) // 2
    new_img.paste(img_resized, (paste_x, paste_y))
    return new_img

# CRASH-SAFE PROTOCOL (see pipeline/crashSafety.py): every file this writes goes through
# an atomic_* helper, so a crash never leaves a half-written file. Subject data written
# here is covered by the transaction that the calling view or batch loop opens. Adding a
# new derivative file? Use an atomic_* helper and keep it inside that transaction.
def create_preview_image(scan_name, directory, ignoreElastic=False, step_times=None):
    """Build (and cache next to the scan) the 2x2 regular grid preview JPEG of one scan.
    Returns the JPEG bytes. step_times, if given, is filled with per-step timings as they
    complete, so a caller can report how far a failed build got."""
    start_time = time.time() # Start timing
    # Per-step timings are collected here and printed as ONE summary line at the end
    # (instead of ~15 lines per scan, which flooded the terminal for large grids).
    step_times = {} if step_times is None else step_times
    preview_3d_detail = ''

    # 1. Load Data
    latest_edit = 0
    # Find the latest edit number for lossy files specifically
    if scan_name == "atlas":
        lossy_edit_files = glob.glob(os.path.join(resolve_atlas_dir(directory), "atlas_lossy_edit_*_*.nii.gz"))
        lossy_original_file = os.path.join(resolve_atlas_dir(directory), "atlas_lossy.nii.gz")
    else:
        lossy_edit_files = glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_lossy_edit_*_*.nii.gz"))
        lossy_original_file = os.path.join(directory, "extracted", scan_name, f"{scan_name}_lossy.nii.gz")

    latest_lossy_edit_file_path = lossy_original_file # Default to original lossy file
    latest_edit_file_identifier = os.path.basename(lossy_original_file).replace("_lossy", "") # Identifier for preview.json (points to non-lossy version)

    if lossy_edit_files:
        # Helper function to check if a file is an elastic edit
        def is_elastic_file(filename):
            return 'elastic' in filename.lower()

        edit_numbers = []
        for f in lossy_edit_files:
            try:
                # Skip elastic files if ignoreElastic is True
                if ignoreElastic and is_elastic_file(f):
                    continue
                num_str = f.split('_lossy_edit_')[1].split('_')[0]
                edit_numbers.append((int(num_str), f))
            except (IndexError, ValueError):
                continue

        if edit_numbers:
            # Find the latest edit number from tuples
            latest_edit = max(edit_numbers, key=lambda x: x[0])[0]
            # Find the specific lossy file corresponding to the latest edit number
            if scan_name == "atlas":
                latest_lossy_edit_file_pattern = os.path.join(resolve_atlas_dir(directory), f"atlas_lossy_edit_{latest_edit}_*.nii.gz")
            else:
                latest_lossy_edit_file_pattern = os.path.join(directory, "extracted", scan_name, f"{scan_name}_lossy_edit_{latest_edit}_*.nii.gz")

            matching_files = glob.glob(latest_lossy_edit_file_pattern)
            if matching_files:
                # Filter out displacement fields (inv and fwd) to get the actual image
                image_files = [f for f in matching_files if not (f.endswith('_lossy_inv.nii.gz') or f.endswith('_lossy_fwd.nii.gz'))]
                if image_files:
                    latest_lossy_edit_file_path = sorted(image_files)[-1]
                else:
                    # Fallback to original logic if no filtered files found
                    latest_lossy_edit_file_path = sorted(matching_files)[-1]
                # Get the corresponding non-lossy filename for the preview.json identifier
                latest_edit_file_identifier = os.path.basename(latest_lossy_edit_file_path).replace("_lossy", "")

    load_start_time = time.time()
    # Load the latest lossy NIfTI file
    try:
        latest_edit_image_nii = nib.load(latest_lossy_edit_file_path)
        # Stored uint8 data instead of a float64 copy (see _grid_preview_volume).
        latest_edit_image_data = _grid_preview_volume(latest_edit_image_nii)
    except FileNotFoundError:
         print(f"Error: Could not find NIfTI file: {latest_lossy_edit_file_path}")
         # Return placeholder/error data
         img = Image.new('RGB', (QUADRANT_RENDER_SIZE * 2, QUADRANT_RENDER_SIZE * 2), color = 'black')
         draw = ImageDraw.Draw(img)
         draw.text((10, 10), f"Error: NIfTI file not found\n{os.path.basename(latest_lossy_edit_file_path)}", fill="red")
         buf = io.BytesIO()
         img.save(buf, format='JPEG')
         buf.seek(0)
         return buf.read()

    step_times['load'] = time.time() - load_start_time

    metadata_start_time = time.time()
    # Load metadata
    if scan_name == "atlas":
        metadata_json_path = os.path.join(resolve_atlas_dir(directory), "atlas.json")
    else:
        metadata_json_path = os.path.join(directory, "extracted", scan_name, f"{scan_name}.json")

    try:
        with open(metadata_json_path, 'r') as json_file:
            metadata_data = json.load(json_file)
    except (FileNotFoundError, json.JSONDecodeError):
         print(f"Error: Could not load metadata JSON for {scan_name}")
         # Return placeholder/error data (similar to above)
         img = Image.new('RGB', (QUADRANT_RENDER_SIZE * 2, QUADRANT_RENDER_SIZE * 2), color = 'black')
         draw = ImageDraw.Draw(img)
         draw.text((10, 10), f"Error: Metadata JSON missing/corrupt\n{scan_name}.json", fill="red")
         buf = io.BytesIO()
         img.save(buf, format='JPEG')
         buf.seek(0)
         return buf.read()

    step_times['metadata'] = time.time() - metadata_start_time

    threshold_orig = metadata_data.get('threshold')

    # Get lossy parameters
    lossy_params = metadata_data.get('lossy_compression', [])
    # If lossy_params is a dict then turn into a list (recover legacy format)
    if isinstance(lossy_params, dict):
        lossy_params = [lossy_params]
        # if the dict has no filename, add it
        if 'filename' not in lossy_params[0]:
            lossy_params[0]['filename'] = os.path.basename(latest_lossy_edit_file_path)
        # save the updated metadata
        atomic_write_json(metadata_json_path, metadata_data, indent=4)
    # Get the lossy parameters for the current lossy file
    lossy_filename = os.path.basename(latest_lossy_edit_file_path)
    matching_lossy_params = [p for p in lossy_params if p['filename'] == lossy_filename]
    if not matching_lossy_params: # if no lossy parameters found, use the last one
        print(f"Warning: No lossy parameters found for {scan_name}. Using default values.")
        lossy_params = lossy_params[-1] if lossy_params else {}
    else:
        lossy_params = matching_lossy_params[0]
    # Get the scale factor, zero point shift, voxel size, and resolution factor
    scale_factor = lossy_params.get('scale_factor', 1.0)
    zero_point_shift = lossy_params.get('zero_point_shift', 0.0)
    voxel_size = metadata_data.get('voxel_size', 1.0)
    resolution_factor = lossy_params.get('resolution_factor', 1.0)

    # Generate a histogram of the lossy image
    # Same counts as np.histogram(data, bins=65, range=(0, 255)), much faster on uint8.
    hist, bin_edges = _grid_preview_histogram(latest_edit_image_data, bins=65, value_range=(0, 255))
    histogram_data = {
        'bins': bin_edges.tolist(),
        'frequency': hist.tolist()
    }
    # Update preview JSON file *before* generating image
    if scan_name == "atlas":
        preview_json_path = os.path.join(resolve_atlas_dir(directory), "atlas_preview.json")
    else:
        preview_json_path = os.path.join(directory, "extracted", scan_name, f"{scan_name}_preview.json")

    # Get the timestamp of the latest edit file for tracking changes
    try:
        latest_edit_timestamp = os.path.getctime(latest_lossy_edit_file_path.replace("_lossy", ""))
    except OSError:
        print(f"Warning: Could not get timestamp for {latest_lossy_edit_file_path.replace('_lossy', '')}")
        latest_edit_timestamp = time.time()  # Use current time as fallback

    atomic_write_json(preview_json_path, {
            'last_edit': latest_edit_file_identifier, 
            'threshold': threshold_orig, 
            'last_edit_timestamp': latest_edit_timestamp, # Add timestamp tracking
            'histogram_data': histogram_data,
        }, indent=None)

    threshold_adjust_start_time = time.time()
    # 2. Adjust Threshold for lossy image
    if scale_factor != 0:
        adjusted_threshold = (threshold_orig - zero_point_shift) / scale_factor
    else:
        adjusted_threshold = threshold_orig if zero_point_shift == 0 else (threshold_orig - zero_point_shift)
        print(f"Warning: scale_factor is zero or missing for {scan_name}. Using threshold {adjusted_threshold}")

    # Ensure threshold is within the range of the lossy data (often 0-255 for uint8)
    # float(): with uint8 data, np.clip against uint8 bounds would cast the threshold to uint8.
    data_min, data_max = float(np.min(latest_edit_image_data)), float(np.max(latest_edit_image_data))
    adjusted_threshold = np.clip(adjusted_threshold, data_min, data_max)
    step_times['threshold'] = time.time() - threshold_adjust_start_time

    centroid_start_time = time.time()
    # 3. Calculate Centroid (using center_of_mass on thresholded data)
    binary_image = latest_edit_image_data > adjusted_threshold
    if np.any(binary_image):
         # Use scipy.ndimage.center_of_mass on the binary mask for geometric center
         # Same result as ndimage.center_of_mass(binary_image), from axis profiles.
         centroid_voxel = _binary_center_of_mass(binary_image)
         # Ensure centroid coordinates are integers for slicing
         centroid_voxel = tuple(int(round(c)) for c in centroid_voxel)
         # Clamp centroid coordinates to be within image bounds
         centroid_voxel = tuple(np.clip(c, 0, s - 1) for c, s in zip(centroid_voxel, latest_edit_image_data.shape))
    else:
         # Fallback if no voxels are above threshold: use image center
         centroid_voxel = tuple(s // 2 for s in latest_edit_image_data.shape)
         print(f"Warning: No voxels above threshold {adjusted_threshold} for {scan_name}. Using image center {centroid_voxel} for centroid.")

    # Assuming Nifti shape is (X, Y, Z)
    centroid_x, centroid_y, centroid_z = centroid_voxel
    step_times['centroid'] = time.time() - centroid_start_time

    quad1_start_time = time.time()
    # 4. Generate Quadrant 1 (Z-Sum Projection)
    # float64 accumulator: same values as summing the old float64 volume.
    z_sum_projection = np.sum(latest_edit_image_data, axis=2, dtype=np.float64)
    # Normalize for display
    z_sum_norm = exposure.rescale_intensity(z_sum_projection, out_range=(0, 255)).astype(np.uint8)
    step_times['z-sum'] = time.time() - quad1_start_time

    quad2_start_time = time.time()
    # 5. Generate Quadrant 2 (Marching Cubes 3D View)
    quadrant2_img_array = None # Initialize
    try:
        # Calculate spacing for the lossy image
        lossy_spacing = float(voxel_size) * float(resolution_factor) # Ensure float
        spacing_tuple = (lossy_spacing, lossy_spacing, lossy_spacing)

        # Calculate dynamic step size based on active voxel count to keep mesh size reasonable
        # Marching cubes complexity scales with N / (step_size^3).
        # These cutoffs aim for roughly 50k-80k "effective" voxels processed to keep time consistent.
        active_voxels = np.count_nonzero(binary_image)
        if active_voxels > 50_000_000:
            dynamic_step = 9
        elif active_voxels > 30_000_000:
            dynamic_step = 8
        elif active_voxels > 20_000_000:
            dynamic_step = 7
        elif active_voxels > 12_000_000:
            dynamic_step = 6
        elif active_voxels > 7_000_000:
            dynamic_step = 5
        elif active_voxels > 3_500_000:
            dynamic_step = 4
        elif active_voxels > 1_500_000:
            dynamic_step = 3
        elif active_voxels > 500_000:
            dynamic_step = 2
        else:
            dynamic_step = 1

        preview_3d_detail = f"; 3D {active_voxels:,} voxels, step {dynamic_step}"

        verts, faces = _preview_mesh_at_step(latest_edit_image_data, adjusted_threshold, spacing_tuple, dynamic_step)

        # Face budget: matplotlib's plot_trisurf below draws triangle by triangle in Python
        # (~18 us per face, 4-7 s for a step-2 mouse skull), far more than marching cubes
        # itself. Above the budget, rebuild once at the coarser step that brings the face
        # count under it (faces scale with 1/step^2).
        max_faces = _preview_3d_max_faces()
        if max_faces and len(faces) > max_faces:
            capped_step = max(dynamic_step + 1,
                              int(math.ceil(dynamic_step * math.sqrt(len(faces) / max_faces))))
            uncapped_faces = len(faces)
            verts, faces = _preview_mesh_at_step(latest_edit_image_data, adjusted_threshold, spacing_tuple, capped_step)
            preview_3d_detail += f" → {capped_step} (face cap: {uncapped_faces:,} → {len(faces):,} faces)"
        else:
            preview_3d_detail += f", {len(faces):,} faces"

        if len(faces) > 0:
            faces = faces[:, ::-1]

        # Create a 3D plot using matplotlib
        fig_3d = plt.figure(figsize=(8, 8), frameon=False) # Control figure size
        fig_3d.subplots_adjust(0, 0, 1, 1)  # Remove all padding
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        if len(verts) > 0 and len(faces) > 0:
             # Plot the mesh
             ax_3d.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], color='lightgray', shade=True, edgecolor=None, alpha=1.0) 
        else:
             # Handle empty mesh case
             ax_3d.text(0.5, 0.5, 0.5, "No Mesh", horizontalalignment='center', verticalalignment='center', transform=ax_3d.transAxes, color='black')

        shape_mm = np.array(latest_edit_image_data.shape) * lossy_spacing
        ax_3d.set_xlim(0, shape_mm[0])
        ax_3d.set_ylim(0, shape_mm[1])
        ax_3d.set_zlim(0, shape_mm[2])

        # Set equal aspect ratio to prevent distortion
        ax_3d.set_box_aspect([shape_mm[0], shape_mm[1], shape_mm[2]])

        # Improve appearance
        ax_3d.xaxis.pane.fill = False # Remove planes
        ax_3d.yaxis.pane.fill = False
        ax_3d.zaxis.pane.fill = False
        ax_3d.grid(False) # No grid
        ax_3d.set_xticks([]) # No ticks
        ax_3d.set_yticks([])
        ax_3d.set_zticks([])
        # Make the axes invisible to remove boundary box
        ax_3d.set_axis_off()

        # Set background color
        ax_3d.set_facecolor('black')
        fig_3d.patch.set_facecolor('black')

        # Remove top title to maximize 3D space
        ax_3d.set_title("", pad=0)
        ax_3d.view_init(elev=-30, azim=25, roll=90) # Adjust viewing angle, elev is left-right (negative is right), azim is up-down (negative is down), roll is roll

        # Remove margin padding completely
        plt.margins(0, 0, 0)
        plt.tight_layout(pad=0, h_pad=0, w_pad=0)

        # Render the 3D plot to a buffer
        buf_3d = io.BytesIO()
        fig_3d.savefig(buf_3d, format='png', facecolor="black", bbox_inches='tight', pad_inches=0, dpi=200)
        buf_3d.seek(0)
        img_3d = Image.open(buf_3d)
        plt.close(fig_3d) # Close the figure to free memory
        quadrant2_img_array = np.array(img_3d)
        buf_3d.close()

    except Exception as e:
        print(f"Error generating 3D preview for {scan_name}: {e}")
        # Create a placeholder image indicating error
        img_placeholder = Image.new('RGB', (QUADRANT_RENDER_SIZE, QUADRANT_RENDER_SIZE), color = 'darkred')
        draw = ImageDraw.Draw(img_placeholder)
        draw.text((10, 10), "3D Preview Error", fill="white")
        quadrant2_img_array = np.array(img_placeholder)

    step_times['3D'] = time.time() - quad2_start_time

    quad3_start_time = time.time()
    # 6. Generate Quadrant 3 (X, Z Slice at Centroid Y)
    # Slice along Y axis (axis=1) at centroid_y
    xz_slice = latest_edit_image_data[:, centroid_y, :]
    # Normalize and ensure correct orientation (transpose if needed)
    xz_slice_norm = np.rot90(xz_slice).astype(np.uint8) # Rotate for better view
    step_times['xz'] = time.time() - quad3_start_time

    quad4_start_time = time.time()
    # 7. Generate Quadrant 4 (Y, Z Slice at Centroid X)
    # Slice along X axis (axis=0) at centroid_x
    yz_slice = latest_edit_image_data[centroid_x, :, :]
    # Normalize and ensure correct orientation
    yz_slice_norm = np.rot90(yz_slice).astype(np.uint8) # Rotate for better view
    step_times['yz'] = time.time() - quad4_start_time

    combine_start_time = time.time()
    # 8. Combine Quadrants
    # Ensure all images are RGB for consistent stacking
    if len(quadrant2_img_array.shape) == 3 and quadrant2_img_array.shape[2] == 4: # RGBA from PNG
        # Blend with black background if alpha exists
        alpha = quadrant2_img_array[:, :, 3] / 255.0
        bg = Image.new('RGB', (QUADRANT_RENDER_SIZE, QUADRANT_RENDER_SIZE), color='black') # PIL size is (width, height)
        fg = Image.fromarray(quadrant2_img_array[:, :, :3])
        # Resize fg before pasting onto bg to handle potential size mismatch from fig_3d
        fg_resized = resize_with_aspect_ratio(fg, (QUADRANT_RENDER_SIZE, QUADRANT_RENDER_SIZE), bg_color='black')
        # Don't use the original mask with the resized image - they won't match dimensions
        bg.paste(fg_resized, (0,0))
        quadrant2_img_array = np.array(bg)
    elif len(quadrant2_img_array.shape) == 2: # Grayscale
        quadrant2_img_array = np.stack([quadrant2_img_array]*3, axis=-1)
    elif quadrant2_img_array is None:
        # Create a placeholder if it failed earlier and wasn't caught
        img_placeholder = Image.new('RGB', (QUADRANT_RENDER_SIZE, QUADRANT_RENDER_SIZE), color = 'darkred')
        draw = ImageDraw.Draw(img_placeholder)
        draw.text((10, 10), "3D Preview Error", fill="white")
        quadrant2_img_array = np.array(img_placeholder)

    # Convert grayscale slices to RGB
    z_sum_rgb = np.stack([z_sum_norm]*3, axis=-1)
    xz_slice_rgb = np.stack([xz_slice_norm]*3, axis=-1)
    yz_slice_rgb = np.stack([yz_slice_norm]*3, axis=-1)

    # Resize images to a common size (e.g., 512x512)
    target_size = (QUADRANT_RENDER_SIZE, QUADRANT_RENDER_SIZE)

    # Convert all arrays to PIL images safely
    try:
        img_q1_pil = Image.fromarray(z_sum_rgb.astype(np.uint8))
        img_q2_pil = Image.fromarray(quadrant2_img_array.astype(np.uint8))
        img_q3_pil = Image.fromarray(xz_slice_rgb.astype(np.uint8))
        img_q4_pil = Image.fromarray(yz_slice_rgb.astype(np.uint8))
    except Exception as e:
        print(f"Error converting arrays to PIL images: {e}")
        # Create fallback images if conversion fails
        img_q1_pil = Image.new('RGB', target_size, color='black')
        img_q2_pil = Image.new('RGB', target_size, color='black')
        img_q3_pil = Image.new('RGB', target_size, color='black')
        img_q4_pil = Image.new('RGB', target_size, color='black')

    # Safely resize images with error handling
    try:
        img_q1 = resize_with_aspect_ratio(img_q1_pil, target_size)
        img_q2 = resize_with_aspect_ratio(img_q2_pil, target_size)
        img_q3 = resize_with_aspect_ratio(img_q3_pil, target_size)
        img_q4 = resize_with_aspect_ratio(img_q4_pil, target_size)
    except Exception as e:
        print(f"Error resizing images: {e}")
        img_q1 = Image.new('RGB', target_size, color='black')
        img_q2 = Image.new('RGB', target_size, color='black')
        img_q3 = Image.new('RGB', target_size, color='black')
        img_q4 = Image.new('RGB', target_size, color='black')

    # Create the grid image
    grid_img = Image.new('RGB', (target_size[0]*2, target_size[1]*2))

    # Paste images with error handling
    try:
        grid_img.paste(img_q1, (0, 0))
        grid_img.paste(img_q2, (target_size[0], 0))
        grid_img.paste(img_q3, (0, target_size[1]))
        grid_img.paste(img_q4, (target_size[0], target_size[1]))
    except Exception as e:
        print(f"Error pasting images onto grid: {e}")
        # If pasting fails, create a simple error image
        grid_img = Image.new('RGB', (target_size[0]*2, target_size[1]*2), color='black')
        draw = ImageDraw.Draw(grid_img)
        draw.text((10, 10), f"Error creating preview: {e}", fill="white")

    # Add titles
    draw = ImageDraw.Draw(grid_img)
    try:
        # Try loading a simple font
        font = ImageFont.truetype("DejaVuSans.ttf", 20) # Use a common sans-serif font if available
    except IOError:
        font = ImageFont.load_default() # Fallback

    # Add titles with background box for better visibility
    def draw_text_with_bg(draw, pos, text, font, text_color="white", bg_color="black"):
         bbox = draw.textbbox(pos, text, font=font)
         # Add small padding
         bg_bbox = (bbox[0]-2, bbox[1]-1, bbox[2]+2, bbox[3]+1)
         try: # Handle potential RGBA color for background
             draw.rectangle(bg_bbox, fill=bg_color)
         except TypeError: # Fallback to RGB if RGBA fails
             draw.rectangle(bg_bbox, fill=bg_color[:3])
         draw.text(pos, text, fill=text_color, font=font)

    draw_text_with_bg(draw, (5, 5), "Z Sum Projection", font=font)
    draw_text_with_bg(draw, (target_size[0] + 5, 5), "3D Marching Cubes", font=font)
    draw_text_with_bg(draw, (5, target_size[1] + 5), f"XZ Slice (Y={centroid_y})", font=font)
    draw_text_with_bg(draw, (target_size[0] + 5, target_size[1] + 5), f"YZ Slice (X={centroid_x})", font=font)

    step_times['compose'] = time.time() - combine_start_time

    save_start_time = time.time()
    # 9. Save/Return Image
    buf_final = io.BytesIO()
    grid_img.save(buf_final, format='JPEG', quality=GRID_PREVIEW_JPEG_QUALITY)
    buf_final.seek(0)
    image_data_bytes = buf_final.read()
    buf_final.close()

    # Save the image to disk as well
    if scan_name == "atlas":
        preview_path = os.path.join(resolve_atlas_dir(directory), "atlas_preview.jpg")
    else:
        preview_path = os.path.join(directory, "extracted", scan_name, f"{scan_name}_preview.jpg")

    try:
        atomic_write_bytes(preview_path, image_data_bytes)
    except IOError as e:
        print(f"Error saving preview image {preview_path}: {e}")

    step_times['save'] = time.time() - save_start_time

    # Clean up large data structures
    cleanup_start_time = time.time()
    latest_edit_image_data = None
    binary_image = None
    verts = faces = None
    gc.collect()
    step_times['cleanup'] = time.time() - cleanup_start_time

    total_time = time.time() - start_time
    step_summary = ' · '.join(f"{step} {seconds:.2f}" for step, seconds in step_times.items())
    print(
        f"[GridView] {scan_name}: preview generated in {total_time:.2f} s "
        f"(thr {threshold_orig} → {float(adjusted_threshold):.1f} lossy, range [{data_min}, {data_max}]"
        f"{preview_3d_detail}; {step_summary})"
    )
    return image_data_bytes # Return bytes for the response


# ---------------------------------------------------------------------------
# One task (runs in a worker process, or in the engine as the fallback)
# ---------------------------------------------------------------------------

def run_preview_task(scan_name, directory, ignore_elastic=False, render_constants=None):
    """Build one scan's preview and return an outcome dict instead of raising:
    {"scan_name", "image_bytes"} on success, {"scan_name", "error", "steps", "traceback"}
    on failure (the engine prints those the way GetGridView.post does)."""
    if render_constants is not None:
        configure(*render_constants)
    step_times = {}
    try:
        image_bytes = create_preview_image(scan_name, directory, ignore_elastic, step_times=step_times)
        return {"scan_name": scan_name, "image_bytes": image_bytes}
    except Exception as e:  # noqa: BLE001 - reported per scan, the page goes on
        import traceback
        return {
            "scan_name": scan_name,
            "error": str(e),
            "steps": ', '.join(step_times.keys()) or 'none',
            "traceback": traceback.format_exc(),
        }


def _worker_task(args):
    return run_preview_task(*args)


# ---------------------------------------------------------------------------
# Pool sizing
# ---------------------------------------------------------------------------

# Hard cap on worker processes: beyond this the disk (reading .nii.gz) and RAM are the
# limit, not the cores, and every idle worker keeps its baseline memory. 24 lets a
# 32-thread workstation build half a page (GRID_PAGE_SIZE = 50) per round; free RAM
# still lowers it (plan_workers).
_MAX_WORKERS = 24
# Resident memory of one idle worker (Python + NumPy + SciPy + scikit-image + matplotlib).
_WORKER_BASE_GB = 0.35
# Peak bytes per lossy voxel while one preview is built: the uint8 volume (1), the
# thresholded boolean mask (1), nibabel's decompression buffer while loading (~1) and
# block_reduce padding / comparison temporaries (~1). Measured on a 62M-voxel synthetic
# volume: ~220 MB above the worker's baseline (~3.5 bytes/voxel); this errs high on purpose.
_PEAK_BYTES_PER_VOXEL = 4
# Fixed part of a task: matplotlib figure + 3D mesh render + PIL compositing.
_TASK_FIXED_GB = 0.15
# Worker baseline measured at ~0.1 GB on Linux; _WORKER_BASE_GB above keeps headroom for
# other platforms.
# Free RAM that previews never use, so the OS, the browser and the engine stay responsive.
_RAM_RESERVE_MIN_GB = 2.0
_RAM_RESERVE_FRACTION = 0.15
# Keep the pool alive this long after the last grid request (page changes reuse it).
_POOL_IDLE_SECONDS = 180

_GB = 1024 ** 3


def _memory():
    """(available_bytes, total_bytes), or (None, None) if psutil is unavailable."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return vm.available, vm.total
    except Exception:  # noqa: BLE001
        return None, None


def _ram_reserve_bytes(total):
    return max(_RAM_RESERVE_MIN_GB * _GB, _RAM_RESERVE_FRACTION * total)


def plan_workers():
    """Number of preview workers for this machine (1 = build in the engine, serially).

    Logical cores - 1, capped by _MAX_WORKERS and by how many workers (baseline plus one
    typical task each) fit in free RAM above the reserve. AURORA_GRID_WORKERS overrides.
    """
    forced = os.environ.get("AURORA_GRID_WORKERS", "").strip()
    if forced:
        try:
            return max(1, int(forced))
        except ValueError:
            return 1
    workers = min(_MAX_WORKERS, max(1, (os.cpu_count() or 1) - 1))
    available, total = _memory()
    if available is not None:
        usable = available - _ram_reserve_bytes(total)
        per_worker = (_WORKER_BASE_GB + _TASK_FIXED_GB + 0.5) * _GB
        workers = min(workers, max(1, int(usable // per_worker)))
    return workers


def estimate_task_bytes(scan_name, directory):
    """Rough peak memory of building this scan's preview, from the largest lossy volume of
    the scan (NIfTI headers only, no voxel data is read). Crop edits make later volumes
    smaller, so the largest one errs on the safe side."""
    if scan_name == "atlas":
        pattern = os.path.join(resolve_atlas_dir(directory), "atlas_lossy*.nii.gz")
    else:
        pattern = os.path.join(directory, "extracted", scan_name, f"{scan_name}_lossy*.nii.gz")
    voxels = 0
    for path in glob.glob(pattern):
        # Displacement fields are not previewed.
        if path.endswith('_lossy_inv.nii.gz') or path.endswith('_lossy_fwd.nii.gz'):
            continue
        try:
            voxels = max(voxels, int(np.prod(nib.load(path).shape[:3])))
        except Exception:  # noqa: BLE001 - unreadable header: the task reports the error
            continue
    return int(voxels * _PEAK_BYTES_PER_VOXEL + _TASK_FIXED_GB * _GB)


# ---------------------------------------------------------------------------
# Persistent pool
# ---------------------------------------------------------------------------

_pool_lock = threading.Lock()
_pool = None
_pool_workers = 0
_pool_active = 0
_pool_idle_timer = None


def _acquire_pool(workers):
    """The shared spawn pool with `workers` processes (recreated if the size changed).
    Returns None when workers <= 1 or the pool cannot be created."""
    global _pool, _pool_workers, _pool_active
    if workers <= 1:
        return None
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    with _pool_lock:
        if _pool_idle_timer is not None:
            _pool_idle_timer.cancel()
        if _pool is not None and _pool_workers != workers and _pool_active == 0:
            _pool.shutdown(wait=False, cancel_futures=True)
            _pool = None
        if _pool is None:
            try:
                _pool = ProcessPoolExecutor(
                    max_workers=workers,
                    mp_context=multiprocessing.get_context("spawn"),
                )
                _pool_workers = workers
            except (OSError, RuntimeError, ValueError) as e:
                print(f"[GridView] could not start the preview worker pool ({type(e).__name__}: {e}); "
                      f"building previews in the engine.", flush=True)
                _pool = None
                return None
        _pool_active += 1
        return _pool


def _release_pool(broken=False):
    """Done with the pool for this request. A broken pool is dropped (the next request
    creates a new one); otherwise it shuts down after _POOL_IDLE_SECONDS of no use."""
    global _pool, _pool_active, _pool_idle_timer
    with _pool_lock:
        _pool_active = max(0, _pool_active - 1)
        if broken and _pool is not None:
            _pool.shutdown(wait=False, cancel_futures=True)
            _pool = None
        if _pool is not None and _pool_active == 0:
            _pool_idle_timer = threading.Timer(_POOL_IDLE_SECONDS, _shutdown_idle_pool)
            _pool_idle_timer.daemon = True
            _pool_idle_timer.start()


def _shutdown_idle_pool():
    global _pool
    with _pool_lock:
        if _pool is not None and _pool_active == 0:
            _pool.shutdown(wait=False, cancel_futures=True)
            _pool = None


# ---------------------------------------------------------------------------
# Entry point used by GetGridViewBatch
# ---------------------------------------------------------------------------

def generate_previews(tasks, workers=None):
    """Build the previews for tasks [(scan_name, directory, ignore_elastic), ...].

    Returns run_preview_task outcomes in task order. Tasks are started in order (the UI
    sends them in page order, so the first tiles come first), at most `workers` at a
    time and only while their estimated memory fits the RAM budget.
    """
    from concurrent.futures import FIRST_COMPLETED, wait
    from concurrent.futures.process import BrokenProcessPool

    render_constants = (QUADRANT_RENDER_SIZE, GRID_PREVIEW_JPEG_QUALITY)
    outcomes = [None] * len(tasks)
    workers = min(workers or plan_workers(), len(tasks))
    pool = _acquire_pool(workers) if workers > 1 else None

    if pool is not None:
        t0 = time.perf_counter()
        available, total = _memory()
        if available is None:
            budget = float('inf')
        else:
            budget = available - _ram_reserve_bytes(total) - workers * _WORKER_BASE_GB * _GB
        estimates = [estimate_task_bytes(scan_name, directory) for scan_name, directory, _ in tasks]
        in_flight = {}
        in_flight_bytes = 0
        next_i = 0
        broken = False
        try:
            while next_i < len(tasks) or in_flight:
                # Admit tasks in order while a worker is free and the RAM budget allows
                # (one task always runs, so a single huge scan still gets built).
                while (next_i < len(tasks) and len(in_flight) < workers
                       and (not in_flight or in_flight_bytes + estimates[next_i] <= budget)):
                    scan_name, directory, ignore_elastic = tasks[next_i]
                    future = pool.submit(_worker_task, (scan_name, directory, ignore_elastic, render_constants))
                    in_flight[future] = next_i
                    in_flight_bytes += estimates[next_i]
                    next_i += 1
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    i = in_flight.pop(future)
                    in_flight_bytes -= estimates[i]
                    outcomes[i] = future.result()
            print(f"[GridView] {len(tasks)} preview(s) built in {time.perf_counter() - t0:.2f} s "
                  f"on {workers} worker processes ({os.cpu_count()} logical CPUs)", flush=True)
        except (BrokenProcessPool, OSError, RuntimeError) as e:
            broken = True
            done_count = sum(outcome is not None for outcome in outcomes)
            print(f"[GridView] preview worker pool failed after {done_count}/{len(tasks)} previews "
                  f"({type(e).__name__}: {e}); building the rest in the engine.", flush=True)
        finally:
            _release_pool(broken=broken)

    # Serial path (1 worker, no pool, or what a broken pool left over).
    for i, (scan_name, directory, ignore_elastic) in enumerate(tasks):
        if outcomes[i] is None:
            outcomes[i] = run_preview_task(scan_name, directory, ignore_elastic, render_constants)
    return outcomes
