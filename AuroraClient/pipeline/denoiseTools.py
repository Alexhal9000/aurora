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
from skimage.filters import threshold_otsu, butterworth, median
import skimage.morphology
from scipy.signal import convolve
from .ALPACA import ALPACA
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
import torch
from .GPUTorchNLMDenoise import nlm3d

from .registrationTools import RegistrationTools
from .preprocessingTools import cleanup_memory, build_denoise_settings
from .batch_flag_filter import load_flagged_subject_names, apply_flag_filter, normalize_flag_filter_value
from .linkedScans import filter_out_linked_children
from .coordinateFrames import (
    is_preserved_mesh_metadata,
    load_extracted_scan_metadata,
    partition_voxel_and_preserved_mesh_scans,
    voxel_only_all_meshes_message,
    voxel_only_no_eligible_targets_message,
)




# ============================================================================
# MODULE-LEVEL WORKER FUNCTIONS WITH SHARED MEMORY AND BATCHING
# ============================================================================


def _denoise_slab_batch_worker_shared(worker_id, batch_indices, shm_name, array_shape, array_dtype, h, template_size, search_size, progress_dict, lock):
    """
    Worker function that processes a batch of slabs using shared memory.
    Uses full dict replacement for proper Manager().dict() synchronization.
    """
    try:
        # Attach to existing shared memory
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        shared_array = np.ndarray(array_shape, dtype=array_dtype, buffer=existing_shm.buf)
        
        results = []
        total_slabs = len(batch_indices)
        start_time = time.time()
        
        for idx, slab_idx in enumerate(batch_indices):
            # Update current slab - use full dict replacement for sync
            progress_dict[worker_id] = {
                'completed': idx,
                'total': total_slabs,
                'current_slab': slab_idx,
                'start_time': start_time
            }
            
            # Get slab from shared memory (no copying)
            slab_data = shared_array[slab_idx]
            
            # Denoise it
            denoised_slab = denoise_nl_means(
                slab_data,
                h=h,
                fast_mode=True,
                patch_size=int(template_size),
                patch_distance=int(search_size)
            )
            results.append((slab_idx, denoised_slab.copy()))
            
            # Update progress - full dict replacement for sync
            progress_dict[worker_id] = {
                'completed': idx + 1,
                'total': total_slabs,
                'current_slab': slab_idx,
                'start_time': start_time
            }
        
        existing_shm.close()
        return results
        
    except Exception as e:
        print(f"Error in worker {worker_id}: {e}")
        raise


def _print_worker_status(progress_dict, lock, num_workers, total_items, start_time, item_type="slabs"):
    """Print a status table of all workers - overwrites previous output"""
    
    current_time = time.time()
    elapsed = current_time - start_time
    
    # Calculate totals
    total_completed = sum(w.get('completed', 0) for w in progress_dict.values())
    overall_progress = (total_completed / total_items) * 100 if total_items > 0 else 0
    
    # Determine which field to use for current item
    item_field = 'current_slab' if item_type == "slabs" else 'current_cube'
    
    # Build the status output
    output_lines = []
    output_lines.append("=" * 90)
    output_lines.append(f"⏱️  ELAPSED: {elapsed:.1f}s | PROGRESS: {total_completed}/{total_items} {item_type} ({overall_progress:.1f}%)")
    output_lines.append("=" * 90)
    output_lines.append(f"{'Worker':<8} {'Completed':<14} {'Progress':<15} {'Current Item':<15} {'Rate':<12}")
    output_lines.append("-" * 90)
    
    for worker_id in sorted(progress_dict.keys()):
        info = progress_dict[worker_id]
        completed = info.get('completed', 0)
        total = info.get('total', 0)
        current = info.get(item_field, -1)
        worker_elapsed = current_time - info.get('start_time', current_time)
        rate = completed / worker_elapsed if worker_elapsed > 0 else 0
        progress_pct = (completed / total * 100) if total > 0 else 0
        
        # Progress bar
        bar_width = 10
        filled = int(bar_width * progress_pct / 100)
        bar = '█' * filled + '░' * (bar_width - filled)
        
        current_str = f"#{current}" if current >= 0 else "starting"
        rate_str = f"{rate:.2f}/s"
        
        output_lines.append(f"W{worker_id:<7} {completed}/{total:<12} {bar} {progress_pct:5.1f}%  {current_str:<15} {rate_str:<12}")
    
    output_lines.append("=" * 90)
    
    # Print with carriage return to overwrite previous output
    output = "\n".join(output_lines)
    # Clear the screen and move cursor to top (optional, for cleaner output)
    sys.stdout.write("\033[2J\033[H")  # Clear screen on some terminals
    sys.stdout.write(output)
    sys.stdout.flush()

def _denoise_cube_batch_worker_shared(worker_id, cube_batch_coords, shm_name, array_shape, array_dtype, progress_dict, lock):
    """Worker function that processes cubes from shared memory"""

    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2" 
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
    os.environ["NUMEXPR_NUM_THREADS"] = "2"
    
    results = []
    total_cubes = len(cube_batch_coords)
    start_time = time.time()
    
    # Attach to shared memory
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    shared_array = np.ndarray(array_shape, dtype=array_dtype, buffer=existing_shm.buf)
    
    try:
        for idx, cube_coord in enumerate(cube_batch_coords):
            cube_idx, x0, x1, y0, y1, z0, z1, h, template_size, search_size = cube_coord
            
            # Update progress
            progress_dict[worker_id] = {
                'completed': idx,
                'total': total_cubes,
                'current_cube': cube_idx,
                'start_time': start_time
            }
            
            # Extract cube from shared memory
            cube_array = shared_array[x0:x1, y0:y1, z0:z1].copy()
            
            # Denoise
            cube_denoised = denoise_nl_means(
                cube_array,
                h=h, fast_mode=True,
                patch_size=int(template_size),
                patch_distance=int(search_size)
            )
            
            results.append((cube_idx, x0, x1, y0, y1, z0, z1, cube_denoised.copy()))

            del cube_array
            gc.collect()
            
            # Update progress
            progress_dict[worker_id] = {
                'completed': idx + 1,
                'total': total_cubes,
                'current_cube': cube_idx,
                'start_time': start_time
            }
        
        existing_shm.close()
        return results
        
    except Exception as e:
        print(f"Error in cube worker {worker_id}: {e}")
        existing_shm.close()
        raise


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




class DenoiseAllScansView(APIView):
    """
    View for applying denoising algorithms to all scans in a directory.
    Supports multiple denoising methods with configurable parameters.
    """
    
    def denoise_gaussian(self, scan_data, sigma, truncate):
        """Apply Gaussian blur denoising"""
        return gaussian_filter(scan_data, sigma=sigma, truncate=truncate)
    
    def denoise_median(self, scan_data):
        """Apply Median filter denoising"""
        return median(scan_data)


    def denoise_nl_means_module(self, scan_data, h, template_size, search_size, split=False, mode="3D_CPU"):
        """Apply Non-Local Means denoising with optimized parallel processing"""
        
        start_time = time.time()
        print(f"Starting Non-Local Means denoising with split={split}")

        # Normalize
        data_min, data_max = scan_data.min(), scan_data.max()
        data_normalized = ((scan_data.astype(np.float32) - data_min) / (data_max - data_min + 1e-8)).astype(np.float32)
        if mode == "3D_CPU" or mode == "3D_GPU":
            max_voxels_allowed = 1000000 #1M
        elif mode == "2.5D":
            max_voxels_allowed = 0
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # Always use at most 8 CPU threads on NLM denoising for parallel workers
        MAX_NLM_THREADS = 8

        if split:
            print("Using split mode for large dataset")

            # Identify longest axis
            longest_axis = np.argmax(scan_data.shape)
            if longest_axis == 0:
                slice_area = scan_data.shape[1] * scan_data.shape[2]
            elif longest_axis == 1:
                slice_area = scan_data.shape[0] * scan_data.shape[2]
            else:
                slice_area = scan_data.shape[0] * scan_data.shape[1]

            print(f"Longest axis: {longest_axis}, slice area: {slice_area}")

            if mode == "2.5D":
                print("Using 2.5D  denoising")
                print("Optimizations: Shared Memory + Balanced Batching + Parallel Time Estimation")

                def process_along_axis_parallel_optimized(data, axis):
                    """Process slabs in parallel with LIVE worker progress monitoring"""

                    data = np.moveaxis(data, axis, 0)
                    num_slabs = data.shape[0]
                    
                    # Determine number of workers: limit to 8 maximum
                    max_workers = min(MAX_NLM_THREADS, max(1, multiprocessing.cpu_count() - 1))
                    
                    print(f"Processing {num_slabs} slabs using {max_workers} workers")
                    
                    result = np.zeros_like(data)
                    
                    # Create shared memory
                    shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
                    shared_array = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
                    shared_array[:] = data
                    
                    print(f"Created shared memory: {shm.name} ({data.nbytes / 1024**2:.1f} MB)")
                    
                    # Create shared progress tracking
                    manager = Manager()
                    progress_dict = manager.dict()
                    lock = manager.Lock()
                    
                    try:
                        # ✅ INTERLEAVED BATCHING: Distribute slabs in round-robin fashion
                        batches = [[] for _ in range(max_workers)]
                        for slab_idx in range(num_slabs):
                            worker_idx = slab_idx % max_workers
                            batches[worker_idx].append(slab_idx)
                        
                        # Remove empty batches
                        batches = [b for b in batches if b]
                        num_batches = len(batches)
                        
                        # Calculate batch statistics
                        batch_sizes = [len(b) for b in batches]
                        min_batch = min(batch_sizes)
                        max_batch = max(batch_sizes)
                        avg_batch = sum(batch_sizes) / len(batch_sizes)
                        
                        print(f"Interleaved batching: {num_batches} batches")
                        print(f"  Batch sizes: min={min_batch}, max={max_batch}, avg={avg_batch:.1f}")
                        print(f"  Each worker gets slabs from across the entire volume (load-balanced)")
                        
                        all_workers_start_time = time.time()
                        
                        # ✅ PRE-INITIALIZE all worker entries BEFORE launching workers
                        for batch_idx in range(num_batches):
                            progress_dict[batch_idx] = {
                                'completed': 0,
                                'total': len(batches[batch_idx]),
                                'current_slab': -1,
                                'start_time': all_workers_start_time
                            }
                        
                        # Start progress monitoring thread
                        stop_monitoring = threading.Event()
                        
                        def monitor_progress():
                            """Background thread that prints progress every 3 seconds"""
                            while not stop_monitoring.is_set():
                                time.sleep(3)
                                if not stop_monitoring.is_set():
                                    _print_worker_status(progress_dict, lock, num_batches, num_slabs, all_workers_start_time, item_type="slabs")
                        
                        monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
                        monitor_thread.start()
                        
                        print(f"\n✅ All {num_batches} workers starting... (live updates every 3s)\n")
                        
                        completed_batches = 0
                        batch_completion_times = []
                        
                        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                            # Submit all batches with progress tracking
                            future_to_batch = {
                                executor.submit(
                                    _denoise_slab_batch_worker_shared,
                                    batch_idx,  # worker_id
                                    batch_indices,
                                    shm.name,
                                    data.shape,
                                    data.dtype,
                                    h,
                                    template_size,
                                    search_size,
                                    progress_dict,
                                    lock
                                ): (batch_idx, batch_indices)
                                for batch_idx, batch_indices in enumerate(batches)
                            }
                            
                            # Process results as they complete
                            for future in concurrent.futures.as_completed(future_to_batch):
                                batch_idx, batch_indices = future_to_batch[future]
                                completion_time = time.time()
                                
                                try:
                                    batch_results = future.result()
                                    
                                    # Store results
                                    for slab_idx, denoised_slab in batch_results:
                                        result[slab_idx] = denoised_slab
                                    
                                    completed_batches += 1
                                    time_since_start = completion_time - all_workers_start_time
                                    batch_completion_times.append(time_since_start)
                                    
                                    print(f"✓ Worker {batch_idx} finished ({len(batch_indices)} slabs) at {time_since_start:.1f}s")
                                    
                                except Exception as e:
                                    print(f"Error processing batch {batch_idx}: {e}")
                                    raise
                        
                        # Stop monitoring thread
                        stop_monitoring.set()
                        monitor_thread.join(timeout=1)
                        
                        total_time = time.time() - all_workers_start_time
                        throughput = num_slabs / total_time
                        
                        # Final summary
                        print("\n" + "="*90)
                        print("✅ ALL WORKERS COMPLETE - FINAL SUMMARY")
                        print("="*90)
                        print(f"Total time: {total_time:.1f}s ({throughput:.1f} slabs/sec)")
                        print(f"Fastest worker: {min(batch_completion_times):.1f}s")
                        print(f"Slowest worker: {max(batch_completion_times):.1f}s")
                        print(f"Average worker: {np.mean(batch_completion_times):.1f}s")
                        print(f"Worker time variance: {np.std(batch_completion_times):.1f}s")
                        efficiency = (min(batch_completion_times) / max(batch_completion_times)) * 100
                        print(f"Efficiency: {efficiency:.1f}% (100% = perfect load balance)")
                        print("="*90 + "\n")
                        
                    finally:
                        shm.close()
                        shm.unlink()
                        print(f"Released shared memory: {shm.name}")
                    
                    return np.moveaxis(result, 0, axis)
                # Run along 3 axes and average
                print("\n" + "="*60)
                print("Processing along axis 0...")
                print("="*60)
                den_xy = process_along_axis_parallel_optimized(data_normalized, 0)
                

                denoised = den_xy
                print("\n" + "="*60)
                print("Completed 2.5D 3-axis denoising")
                print("="*60)

            elif mode == "3D_CPU":
                print("Using cubic chunks with parallel processing and batching")

                # Compute cube side length
                cube_side = int(round(max_voxels_allowed ** (1/3)))
                print(f"Cube side length: {cube_side}")

                x_size, y_size, z_size = data_normalized.shape
                print(f"Data shape: {data_normalized.shape}")

                # Build list of cube coordinate ranges
                x_ranges = [(x, min(x + cube_side, x_size)) for x in range(0, x_size, cube_side)]
                y_ranges = [(y, min(y + cube_side, y_size)) for y in range(0, y_size, cube_side)]
                z_ranges = [(z, min(z + cube_side, z_size)) for z in range(0, z_size, cube_side)]

                total_cubes = len(x_ranges) * len(y_ranges) * len(z_ranges)
                print(f"Total cubes: {total_cubes}")

                denoised = np.zeros_like(data_normalized)
                weight = np.zeros_like(data_normalized)

                                # Create all cube tasks - ONLY store coordinates, not data
                cube_tasks = []
                cube_idx = 0
                for xi, (x0, x1) in enumerate(x_ranges):
                    for yi, (y0, y1) in enumerate(y_ranges):
                        for zi, (z0, z1) in enumerate(z_ranges):
                            cube_idx += 1
                            cube_tasks.append((cube_idx, x0, x1, y0, y1, z0, z1, h, template_size, search_size))

                # Determine number of workers and create INTERLEAVED batches
                max_workers = min(MAX_NLM_THREADS, max(1, multiprocessing.cpu_count() - 1))
                
                # INTERLEAVED BATCHING: Distribute cubes in round-robin fashion
                batches = [[] for _ in range(max_workers)]
                for task_idx, task in enumerate(cube_tasks):
                    worker_idx = task_idx % max_workers
                    batches[worker_idx].append(task)
                
                batches = [b for b in batches if b]
                num_batches = len(batches)
                
                batch_sizes = [len(b) for b in batches]
                print(f"Interleaved batching: {num_batches} batches")
                print(f"  Batch sizes: min={min(batch_sizes)}, max={max(batch_sizes)}, avg={sum(batch_sizes)/len(batch_sizes):.1f}")
                print(f"  Each worker gets cubes from throughout the volume (load-balanced)")
                
                # Create shared progress tracking
                manager = Manager()
                progress_dict = manager.dict()
                lock = manager.Lock()
                
                overall_start = time.time()
                
                # PRE-INITIALIZE all worker entries BEFORE launching workers
                for batch_idx in range(num_batches):
                    progress_dict[batch_idx] = {
                        'completed': 0,
                        'total': len(batches[batch_idx]),
                        'current_cube': -1,
                        'start_time': overall_start
                    }
                
                # Start progress monitoring thread
                stop_monitoring = threading.Event()
                
                def monitor_cube_progress():
                    """Background thread that prints progress every 3 seconds"""
                    while not stop_monitoring.is_set():
                        time.sleep(3)
                        if not stop_monitoring.is_set():
                            _print_worker_status(progress_dict, lock, num_batches, total_cubes, overall_start, item_type="cubes")
                
                monitor_thread = threading.Thread(target=monitor_cube_progress, daemon=True)
                monitor_thread.start()
                
                print(f"\n✅ All {num_batches} workers starting... (live updates every 3s)\n")
                
                completed_batches = 0
                batch_completion_times = []
                
                # Create shared memory for the entire normalized data
                shm_data = shared_memory.SharedMemory(create=True, size=data_normalized.nbytes)
                shared_array = np.ndarray(data_normalized.shape, dtype=data_normalized.dtype, buffer=shm_data.buf)
                shared_array[:] = data_normalized
                print(f"Created shared memory for cubes: {shm_data.name} ({data_normalized.nbytes / 1024**2:.1f} MB)")

                try:
                    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                        future_to_batch = {
                            executor.submit(
                                _denoise_cube_batch_worker_shared,
                                batch_idx,
                                batch,
                                shm_data.name,
                                data_normalized.shape,
                                data_normalized.dtype,
                                progress_dict,
                                lock
                            ): (batch_idx, batch)
                            for batch_idx, batch in enumerate(batches)
                        }
                        
                        for future in concurrent.futures.as_completed(future_to_batch):
                            batch_idx, batch = future_to_batch[future]
                            completion_time = time.time()
                            
                            try:
                                batch_results = future.result()
                                
                                # Store results
                                for cube_idx, x0, x1, y0, y1, z0, z1, cube_denoised in batch_results:
                                    denoised[x0:x1, y0:y1, z0:z1] += cube_denoised
                                    weight[x0:x1, y0:y1, z0:z1] += 1
                                
                                completed_batches += 1
                                time_since_start = completion_time - overall_start
                                batch_completion_times.append(time_since_start)
                                
                                print(f"✓ Worker {batch_idx} finished ({len(batch)} cubes) at {time_since_start:.1f}s")
                                
                            except Exception as e:
                                print(f"Error processing cube batch {batch_idx}: {e}")
                                raise
                
                finally:
                    # Clean up shared memory
                    shm_data.close()
                    shm_data.unlink()
                    print(f"Released shared memory: {shm_data.name}")
                
                # Stop monitoring thread
                stop_monitoring.set()
                monitor_thread.join(timeout=1)
                
                total_time = time.time() - overall_start
                throughput = total_cubes / total_time
                
                # Final summary
                print("\n" + "="*90)
                print("✅ ALL CUBE WORKERS COMPLETE - FINAL SUMMARY")
                print("="*90)
                print(f"Total time: {total_time:.1f}s ({throughput:.1f} cubes/sec)")
                print(f"Fastest worker: {min(batch_completion_times):.1f}s")
                print(f"Slowest worker: {max(batch_completion_times):.1f}s")
                print(f"Average worker: {np.mean(batch_completion_times):.1f}s")
                print(f"Worker time variance: {np.std(batch_completion_times):.1f}s")
                efficiency = (min(batch_completion_times) / max(batch_completion_times)) * 100
                print(f"Efficiency: {efficiency:.1f}% (100% = perfect load balance)")
                print("="*90 + "\n")
                
                # Normalize by counts
                denoised = denoised / (weight + 1e-8)
                print(f"Completed cube processing in {total_time:.1f}s")
            elif mode == "3D_GPU":
                print("Using 3D GPU denoising with torch_nlm (with automatic OOM fallback to tiling)")
                
                # Check GPU availability
                try:                    
                    if torch.cuda.is_available():
                        device = torch.device("cuda")
                        print(f"Using GPU: {device}")
                        os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
                        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
                    else:
                        raise RuntimeError("No CUDA-capable GPU detected.")
                except Exception as e:
                    print(f"GPU check failed: {e}")
                    raise RuntimeError("GPU not available for 3D_GPU mode")
                
                # Move normalized data to GPU
                print("Moving data to GPU...")
                image_torch = torch.as_tensor(data_normalized.astype(np.float32), dtype=torch.float32).to("cuda")

                print(f"Image torch shape: {image_torch.shape}")
                print(f"Image torch dtype: {image_torch.dtype}")
                print(f"Image torch device: {image_torch.device}")
                print(f"Image torch size in GB: {image_torch.numel() * image_torch.element_size() / 1024**3:.2f}")
                print("Processing... (will auto-fallback to tiling if GPU runs out of memory)")

                # Process with automatic OOM fallback
                search_radius = int(search_size)
                kernel_size = 2 * search_radius + 1
                denoised = nlm3d(
                    image_torch,
                    kernel_size=kernel_size,
                    std=float(h),
                    kernel_size_mean=int(template_size),
                    sub_filter_size=1,
                    debug=False
                ).cpu().numpy()
                                
                print("GPU denoising complete")

        else:
            print("Processing entire dataset without splitting")
            
            if mode == "3D_GPU":
                print("Using 3D GPU denoising with torch_nlm (without splitting)")
                try:
                    if torch.cuda.is_available():
                        device = torch.device("cuda")
                        print(f"Using GPU: {device}")
                        os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
                        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
                    else:
                        raise RuntimeError("No CUDA-capable GPU detected.")
                except Exception as e:
                    print(f"GPU check failed: {e}. Falling back to CPU denoising.")
                    denoised = denoise_nl_means(data_normalized, h=h, fast_mode=True,
                                                patch_size=int(template_size), patch_distance=int(search_size))
                else:
                    print("Moving data to GPU...")
                    image_torch = torch.as_tensor(data_normalized.astype(np.float32), dtype=torch.float32).to("cuda")
                    print(f"Image torch shape: {image_torch.shape}")
                    print(f"Image torch dtype: {image_torch.dtype}")
                    print(f"Image torch device: {image_torch.device}")
                    print(f"Image torch size in GB: {image_torch.numel() * image_torch.element_size() / 1024**3:.2f}")
                    
                    denoised = nlm3d(image_torch,
                                    kernel_size=int(search_size),
                                    std=float(h),
                                    kernel_size_mean=int(template_size),
                                    sub_filter_size=1,
                                    debug=False
                                    ).cpu().numpy()
                    
                    print("GPU denoising complete")
            elif mode == "3D_CPU":
                # For 2.5D and 3D_CPU modes without splitting, use standard CPU denoising
                print(f"Using {mode} denoising without splitting")
                denoised = denoise_nl_means(data_normalized, h=h, fast_mode=True,
                                            patch_size=int(template_size), patch_distance=int(search_size))

            elif mode == "2.5D":
                print(f"Using 2.5D denoising without splitting (sequential slab processing)")
                
                # Identify longest axis (same logic as split mode)
                longest_axis = np.argmax(data_normalized.shape)
                data_moved = np.moveaxis(data_normalized, longest_axis, 0)
                num_slabs = data_moved.shape[0]
                
                print(f"Longest axis: {longest_axis}, processing {num_slabs} slabs sequentially")
                
                result = np.zeros_like(data_moved)
                slab_start_time = time.time()
                
                for slab_idx in range(num_slabs):
                    slab = data_moved[slab_idx]
                    # Apply NLM to each 2D slab
                    denoised_slab = denoise_nl_means(slab, h=h, fast_mode=True,
                                                    patch_size=int(template_size), 
                                                    patch_distance=int(search_size))
                    result[slab_idx] = denoised_slab
                    
                    # Print progress every 10% or on first slab
                    if (slab_idx + 1) % max(1, num_slabs // 10) == 0 or slab_idx == 0:
                        elapsed = time.time() - slab_start_time
                        print(f"  Processed {slab_idx + 1}/{num_slabs} slabs ({elapsed:.1f}s)")
                
                print(f"Completed 2.5D sequential slab processing in {time.time() - slab_start_time:.1f}s")
                denoised = np.moveaxis(result, 0, longest_axis)

        # Denormalize and return
        result = (denoised * (data_max - data_min) + data_min).astype(scan_data.dtype)
        print(f"Total NLM time: {time.time() - start_time:.2f}s")
        return result

    
    def denoise_tv_chambolle(self, scan_data, weight, eps, max_num_iter):
        """Apply Total Variation Chambolle denoising"""
        # Normalize data to 0-1 range
        data_min, data_max = scan_data.min(), scan_data.max()
        data_normalized = (scan_data.astype(np.float32) - data_min) / (data_max - data_min + 1e-8)
        denoised = denoise_tv_chambolle(data_normalized, weight=weight, eps=eps, 
                                       max_num_iter=int(max_num_iter))
        # Denormalize back
        return (denoised * (data_max - data_min) + data_min).astype(scan_data.dtype)
    
    def denoise_tv_bregman(self, scan_data, weight, eps, max_num_iter):
        """Apply Total Variation Bregman denoising"""
        # Normalize data to 0-1 range
        data_min, data_max = scan_data.min(), scan_data.max()
        data_normalized = (scan_data.astype(np.float32) - data_min) / (data_max - data_min + 1e-8)
        denoised = denoise_tv_bregman(data_normalized, weight=weight, eps=eps, 
                                      max_num_iter=int(max_num_iter))
        # Denormalize back
        return (denoised * (data_max - data_min) + data_min).astype(scan_data.dtype)
    
    def denoise_wavelet(self, scan_data, wavelet, level, sigma):
        """Apply Wavelet denoising"""
        # Normalize data to 0-1 range
        data_min, data_max = scan_data.min(), scan_data.max()
        data_normalized = (scan_data.astype(np.float32) - data_min) / (data_max - data_min + 1e-8)
        
        # Handle None values for auto parameters
        wavelet_levels = None if level is None else int(level)
        sigma_param = None if sigma is None else sigma
        
        denoised = denoise_wavelet(data_normalized, method='BayesShrink', mode='soft',
                                  wavelet=wavelet, sigma=sigma_param, wavelet_levels=wavelet_levels)
        # Denormalize back
        return (denoised * (data_max - data_min) + data_min).astype(scan_data.dtype)
    
    def denoise_butterworth(self, scan_data, cutoff, order, high_pass, squared_butterworth):
        """Apply Butterworth filter denoising"""
        # Normalize data to 0-1 range
        data_min, data_max = scan_data.min(), scan_data.max()
        data_normalized = (scan_data.astype(np.float32) - data_min) / (data_max - data_min + 1e-8)
        denoised = butterworth(data_normalized, cutoff_frequency_ratio=cutoff, order=order, high_pass=high_pass, squared_butterworth=squared_butterworth)
        # Denormalize back
        return (denoised * (data_max - data_min) + data_min).astype(scan_data.dtype)

    
    
    def post(self, request):
        print("Denoising all scans")
        directory = request.data.get('directory')
        algorithm = request.data.get('algorithm', 'Gaussian')
        parameters = request.data.get('parameters', {})
        denoiseAutoMode = request.data.get('denoiseAutoMode', {})
        onlyCurrentScan = request.data.get('onlyCurrentScan', False)
        selectedScan = request.data.get('selectedScan', None)
        flag_filter = normalize_flag_filter_value(
            request.data.get('flagFilter', 'off') if request.data else 'off',
            only_current_scan=onlyCurrentScan,
        )
        
        print(f"Algorithm: {algorithm}")
        print(f"Parameters: {parameters}")
        print(f"Auto Mode: {denoiseAutoMode}")
        print(f"Only Current Scan: {onlyCurrentScan}")
        print(f"Selected Scan: {selectedScan}")
        print(f"Flag filter: {flag_filter}")
        
        # Make a list of faulty files
        faulty_files = []
        for file in os.listdir(os.path.join(directory, "extracted")):
            if file == "project_settings.json" or not os.path.isdir(os.path.join(directory, "extracted", file)):
                continue
            json_path = os.path.join(directory, "extracted", file, f"{file}.json")
            with open(json_path, 'r') as jf:
                metadata = json.load(jf)
                if metadata.get('faulty', False):
                    faulty_files.append(file)
        
        # Find all unique scan names
        scan_names = [d for d in os.listdir(os.path.join(directory, "extracted")) 
                     if os.path.isdir(os.path.join(directory, "extracted", d)) and d not in faulty_files]

        if onlyCurrentScan and selectedScan:
            try:
                selected_metadata = load_extracted_scan_metadata(directory, selectedScan)
            except (FileNotFoundError, json.JSONDecodeError):
                return Response({
                    'status': 'error',
                    'message': f'Selected scan "{selectedScan}" not found in directory'
                }, status=status.HTTP_400_BAD_REQUEST)
            if is_preserved_mesh_metadata(selected_metadata):
                return Response({
                    'status': 'error',
                    'message': (
                        f'Selected scan "{selectedScan}" is a preserved PLY mesh. '
                        'Denoise only works with voxel-based volumes.'
                    ),
                }, status=status.HTTP_400_BAD_REQUEST)

        scan_names, skipped_preserved_meshes = partition_voxel_and_preserved_mesh_scans(directory, scan_names)
        if not scan_names:
            return Response({
                'status': 'error',
                'message': voxel_only_all_meshes_message('Denoise'),
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # If only denoising current scan, filter to that specific scan
        if onlyCurrentScan and selectedScan:
            if selectedScan in scan_names:
                scan_names = [selectedScan]
            else:
                return Response({
                    'status': 'error',
                    'message': f'Selected scan "{selectedScan}" not found in directory'
                }, status=status.HTTP_400_BAD_REQUEST)
        
        # Skip if the latest scan edit already has denoising applied
        scan_names_to_remove = []
        for scan_name in scan_names:
            latest_edit = 0
            while glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{latest_edit}_*.nii.gz")):
                latest_edit += 1
            latest_edit -= 1
            if latest_edit >= 0:
                edit_files = glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{latest_edit}_*.nii.gz"))
                if edit_files and "_denoised" in edit_files[0]:
                    scan_names_to_remove.append(scan_name)

        # Skip if it has elastic registration
        for scan_name in scan_names:
            if has_elastic_registration(scan_name, directory):
                scan_names_to_remove.append(scan_name)
        
        for scan_name in scan_names_to_remove:
            scan_names.remove(scan_name)
        
        scan_names.sort()

        flagged_set = load_flagged_subject_names(directory)
        scan_names = apply_flag_filter(scan_names, flagged_set, flag_filter)
        # Keep explicit only-current child selections; batch mode skips linked children.
        if not (onlyCurrentScan and selectedScan):
            scan_names = filter_out_linked_children(directory, scan_names)
        if not scan_names:
            return Response({
                'status': 'error',
                'message': voxel_only_no_eligible_targets_message('Denoise'),
            }, status=status.HTTP_400_BAD_REQUEST)
        if skipped_preserved_meshes:
            print(f"Skipping preserved mesh subjects for denoising: {skipped_preserved_meshes}")
        
        total_scans = len(scan_names)
        channel_layer = get_channel_layer()
        
        if channel_layer is not None and total_scans > 0:
            async_to_sync(channel_layer.group_send)(
                'progress_group',
                {
                    'type': 'send_progress',
                    'progress': 0,
                    'scan_name': scan_names[0] if scan_names else 'N/A',
                    'custom_message': f'Denoising all scans using {algorithm}...',
                    'total': total_scans,
                    'current': 0,
                }
            )
        
        updated_scans = []
        errors = []
        
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
                            'custom_message': f'Denoising {scan_name} using {algorithm}...',
                            'total': total_scans,
                            'current': idx + 1,
                        }
                    )
                
                print(f"Processing {scan_name} ({idx+1}/{total_scans})")
                
                # Find the latest edit for this scan
                latest_edit = 0
                while glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{latest_edit}_*.nii.gz")):
                    latest_edit += 1
                latest_edit -= 1
                
                # Load the scan data
                if latest_edit >= 0:
                    scan_path = glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{latest_edit}_*.nii.gz"))[0]
                else:
                    scan_path = os.path.join(directory, "extracted", scan_name, f"{scan_name}.nii.gz")
                
                # Load metadata
                with open(os.path.join(directory, "extracted", scan_name, f"{scan_name}.json"), 'r') as jf:
                    scan_metadata = json.load(jf)
                
                # Load the scan
                time_start = time.time()
                nifti_img = nib.load(scan_path)
                original_dtype = nifti_img.get_data_dtype()
                scan_data = nifti_img.get_fdata().astype(original_dtype)
                scan_affine = nifti_img.affine
                time_end = time.time()
                print(f"Time taken to load {scan_name}: {time_end - time_start} seconds")
                
                # Apply denoising algorithm
                time_start = time.time()
                if algorithm == 'Gaussian':
                    denoised_data = self.denoise_gaussian(scan_data, parameters.get('sigma', 1.0), parameters.get('truncate', 4.0))
                elif algorithm == 'Median':
                    denoised_data = self.denoise_median(scan_data)

                elif algorithm == 'NonLocalMeans':
                    denoised_data = self.denoise_nl_means_module(scan_data,
                                                         parameters.get('h', 0.1),
                                                         parameters.get('template_size', 7),
                                                         parameters.get('search_size', 21),
                                                         split=True,
                                                         mode=parameters.get('mode', "2.5D"))
                elif algorithm == 'TotalVariationChambolle':
                    denoised_data = self.denoise_tv_chambolle(scan_data,
                                                             parameters.get('weight', 0.1),
                                                             parameters.get('eps', 0.002),
                                                             parameters.get('max_num_iter', 200))
                elif algorithm == 'TotalVariationBregman':
                    denoised_data = self.denoise_tv_bregman(scan_data,
                                                           parameters.get('weight', 0.1),
                                                           parameters.get('eps', 0.002),
                                                           parameters.get('max_num_iter', 200))
                elif algorithm == 'Wavelet':
                    denoised_data = self.denoise_wavelet(scan_data,
                                                        parameters.get('wavelet', 'db1'),
                                                        parameters.get('level', 1),
                                                        parameters.get('sigma', 0.1))
                elif algorithm == 'Butterworth':
                    denoised_data = self.denoise_butterworth(scan_data,
                                                            parameters.get('cutoff', 0.05),
                                                            parameters.get('order', 2),
                                                            parameters.get('high_pass', True),
                                                            parameters.get('squared_butterworth', False))
                else:
                    raise ValueError(f"Unknown algorithm: {algorithm}")
                
                time_end = time.time()
                print(f"Time taken to denoise {scan_name}: {time_end - time_start} seconds")
                
                # Determine the next edit number
                new_edit_number = latest_edit + 1
                
                # Save the denoised scan
                output_path = os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{new_edit_number}_denoised.nii.gz")
                output_img = nib.Nifti1Image(denoised_data.astype(original_dtype), scan_affine)
                nib.save(output_img, output_path)
                print(f"Saved denoised scan to: {output_path}")
                
                # Save lossy version using proper compression method
                lossy_path = os.path.join(directory, "extracted", scan_name, f"{scan_name}_lossy_edit_{new_edit_number}_denoised.nii.gz")
                registration_tools = RegistrationTools()
                json_path = os.path.join(directory, "extracted", scan_name, f"{scan_name}.json")
                registration_tools.save_as_lossy_nifti(
                    denoised_data.astype(original_dtype),
                    scan_metadata['voxel_size'],
                    json_path,
                    lossy_path
                )
                print(f"Saved lossy version to: {lossy_path}")

                # Persist denoise provenance on subject JSON
                try:
                    with open(json_path, 'r') as jf:
                        _meta = json.load(jf)
                    _meta['denoise_settings'] = build_denoise_settings(algorithm, parameters)
                    with open(json_path, 'w') as jf:
                        json.dump(_meta, jf, indent=4)
                except Exception as meta_err:
                    print(f"  Warning: could not save denoise_settings: {meta_err}")
                
                # Copy mask files if they exist
                src_mask_full = glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{latest_edit}_*.nii.mask.gz"))
                src_mask_lossy = glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_lossy_edit_{latest_edit}_*.nii.mask.gz"))
                if src_mask_full:
                    dst_mask_full = os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{new_edit_number}_denoised.nii.mask.gz")
                    shutil.copy(src_mask_full[0], dst_mask_full)

                if src_mask_lossy:
                    dst_mask_lossy = os.path.join(directory, "extracted", scan_name, f"{scan_name}_lossy_edit_{new_edit_number}_denoised.nii.mask.gz")
                    shutil.copy(src_mask_lossy[0], dst_mask_lossy)
                    

                
                # Copy paired landmark files (non-geometry change) if they exist
                try:
                    if latest_edit >= 0:
                        source_landmark_base = glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{latest_edit}_*.nii.gz"))[0].replace('.nii.gz', '')
                    else:
                        source_landmark_base = scan_name
                    
                    source_landmarks_path = os.path.join(directory, "extracted", scan_name, f"{source_landmark_base}_landmarks.json")
                    source_distances_path = os.path.join(directory, "extracted", scan_name, f"{source_landmark_base}_landmark_distances.json")
                    
                    dest_landmark_base = f"{scan_name}_edit_{new_edit_number}_denoised"
                    dest_landmarks_path = os.path.join(directory, "extracted", scan_name, f"{dest_landmark_base}_landmarks.json")
                    dest_distances_path = os.path.join(directory, "extracted", scan_name, f"{dest_landmark_base}_landmark_distances.json")
                    
                    if os.path.isfile(source_landmarks_path):
                        shutil.copyfile(source_landmarks_path, dest_landmarks_path)
                        print(f"  Copied landmarks to: {dest_landmarks_path}")
                    
                    if os.path.isfile(source_distances_path):
                        shutil.copyfile(source_distances_path, dest_distances_path)
                        print(f"  Copied landmark distances to: {dest_distances_path}")
                except Exception as le:
                    print(f"  Error handling paired landmark files (denoised): {le}")

                updated_scans.append(scan_name)
                
                # Clean up
                del scan_data, denoised_data, nifti_img
                gc.collect()
                
            except Exception as e:
                error_msg = f"Error processing {scan_name}: {str(e)}"
                errors.append(error_msg)
                print(error_msg)
                import traceback
                traceback.print_exc()
        
        response_data = {
            'status': 'success',
            'message': f'Denoised {len(updated_scans)} scans using {algorithm}',
            'updated_scans': updated_scans
        }
        
        if errors:
            response_data['errors'] = errors
            response_data['message'] += f' with {len(errors)} errors'

        return Response(response_data, status=status.HTTP_200_OK)


# Global cache for raw patches ONLY
DENOISE_PREVIEW_CACHE = {}
# Structure: {cache_key: {'raw_patch': ndarray}}
MAX_CACHE_SIZE = 1

class PreviewDenoiseView(APIView):
    """
    Preview denoising on a 100x100x100 patch around given coordinates.
    Caches raw patches to avoid re-extracting if coordinates haven't changed.
    Recalculates denoising each time to reflect current parameters.
    """
    
    def _get_cache_key(self, scan_name, x, y, z):
        """Generate cache key from scan name and coordinates"""
        return f"{scan_name}_{int(x)}_{int(y)}_{int(z)}"
    
    def _prune_cache(self):
        """Remove oldest entry if cache exceeds MAX_CACHE_SIZE"""
        global DENOISE_PREVIEW_CACHE
        if len(DENOISE_PREVIEW_CACHE) >= MAX_CACHE_SIZE:
            oldest_key = next(iter(DENOISE_PREVIEW_CACHE))
            del DENOISE_PREVIEW_CACHE[oldest_key]
            print(f"Cache pruned: removed {oldest_key}")
    
    def _extract_patch(self, scan_data, x, y, z, patch_size=100):
        """
        Extract a cubic patch centered at (x, y, z)
        Pads with zeros if near boundaries
        """

        print(f"Extracting patch at ({x}, {y}, {z})")
        print(f"Scan data shape: {scan_data.shape}")

        half = patch_size // 2
        
        # Calculate boundaries with clipping
        x_start = max(0, int(x) - half)
        x_end = min(scan_data.shape[0], int(x) + half)
        y_start = max(0, int(y) - half)
        y_end = min(scan_data.shape[1], int(y) + half)
        z_start = max(0, int(z) - half)
        z_end = min(scan_data.shape[2], int(z) + half)
        
        patch = scan_data[x_start:x_end, y_start:y_end, z_start:z_end].copy()
        
        # Pad if necessary to maintain 100x100x100
        if patch.shape != (patch_size, patch_size, patch_size):
            padded = np.zeros((patch_size, patch_size, patch_size), dtype=patch.dtype)
            # Calculate where the center should be in the padded array
            center_x = half
            center_y = half  
            center_z = half
            # Calculate where the extracted patch should start in the padded array
            # to keep the center at the right position
            pad_x = center_x - (int(x) - x_start)
            pad_y = center_y - (int(y) - y_start)
            pad_z = center_z - (int(z) - z_start)
            padded[pad_x:pad_x+patch.shape[0], 
                pad_y:pad_y+patch.shape[1], 
                pad_z:pad_z+patch.shape[2]] = patch
            patch = padded
        
        return patch
    
    def _get_center_slice(self, volume_3d):
        """Extract center 2D slice from 3D volume (along Z axis)"""
        center_z = volume_3d.shape[2] // 2
        return volume_3d[:, :, center_z]
    
    def _load_and_extract_patch(self, directory, scan_name, x, y, z):
        """Load scan from file and extract patch"""
        # Find latest edit or original scan
        scan_path = glob.glob(
            os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_*_*.nii.gz")
        )
        if scan_path:
            scan_path = sorted(scan_path)[-1]  # Latest edit
        else:
            scan_path = os.path.join(directory, "extracted", scan_name, f"{scan_name}.nii.gz")
        
        print(f"Loading scan from: {scan_path}")
        nifti_img = nib.load(scan_path)
        scan_data = nifti_img.get_fdata()
        
        patch = self._extract_patch(scan_data, x, y, z)
        del nifti_img, scan_data
        gc.collect()
        
        return patch
    
    def _apply_denoise_algorithm(self, patch, algorithm, parameters):
        """Apply denoising algorithm to patch"""
        denoiser = DenoiseAllScansView()
        
        if algorithm == 'Gaussian':
            return denoiser.denoise_gaussian(
                patch, 
                parameters.get('sigma', 1.0), 
                parameters.get('truncate', 4.0)
            )
        elif algorithm == 'Median':
            return denoiser.denoise_median(patch)
        elif algorithm == 'NonLocalMeans':
            return denoiser.denoise_nl_means_module(
                patch,
                parameters.get('h', 0.1),
                parameters.get('patch_size', 7),
                parameters.get('patch_distance', 11), 
                mode=parameters.get('mode', "2.5D"),
                split=False
            )
        elif algorithm == 'TotalVariationChambolle':
            return denoiser.denoise_tv_chambolle(
                patch,
                parameters.get('weight', 0.1),
                parameters.get('eps', 0.0002),
                parameters.get('max_num_iter', 200)
            )
        elif algorithm == 'TotalVariationBregman':
            return denoiser.denoise_tv_bregman(
                patch,
                parameters.get('weight', 5.0),
                parameters.get('eps', 0.001),
                parameters.get('max_num_iter', 100)
            )
        elif algorithm == 'Wavelet':
            return denoiser.denoise_wavelet(
                patch,
                parameters.get('wavelet', 'db1'),
                parameters.get('level', 1),
                parameters.get('sigma', 0.1)
            )
        elif algorithm == 'Butterworth':
            return denoiser.denoise_butterworth(
                patch,
                parameters.get('cutoff', 0.005),
                parameters.get('order', 2),
                parameters.get('high_pass', False),
                parameters.get('squared_butterworth', True)
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def post(self, request):
        global DENOISE_PREVIEW_CACHE
        
        try:
            directory = request.data.get('directory')
            scan_name = request.data.get('scan_name')
            algorithm = request.data.get('algorithm', 'Gaussian')
            parameters = request.data.get('parameters', {})
            denoiseAutoMode = request.data.get('denoiseAutoMode', {})
            x = request.data.get('x', 0)
            y = request.data.get('y', 0)
            z = request.data.get('z', 0)
            
            print(f"\n=== PREVIEW DENOISE REQUEST ===")
            print(f"Scan: {scan_name} at ({x}, {y}, {z})")
            print(f"Algorithm: {algorithm}")
            print(f"Parameters: {parameters}")
            
            cache_key = self._get_cache_key(scan_name, x, y, z)
            
            # Check if raw patch is cached at these coordinates
            if cache_key in DENOISE_PREVIEW_CACHE:
                print(f"✓ Cache HIT for key: {cache_key}")
                raw_patch = DENOISE_PREVIEW_CACHE[cache_key]['raw_patch']
                from_cache = True
            else:
                print(f"✗ Cache MISS for key: {cache_key}, extracting from file...")
                time_start = time.time()
                raw_patch = self._load_and_extract_patch(directory, scan_name, x, y, z)
                time_end = time.time()
                print(f"  Extraction took: {time_end - time_start:.2f}s")
                
                # Cache the raw patch
                self._prune_cache()
                DENOISE_PREVIEW_CACHE[cache_key] = {
                    'raw_patch': raw_patch
                }
                print(f"  Cached raw patch, cache size now: {len(DENOISE_PREVIEW_CACHE)}")
                from_cache = False
            
            # ALWAYS apply denoise to raw patch (fresh calculation every time)
            print(f"Applying {algorithm} denoising to patch...")
            time_start = time.time()
            denoised_patch = self._apply_denoise_algorithm(
                raw_patch, 
                algorithm, 
                parameters
            )
            time_end = time.time()
            print(f"  Denoise took: {time_end - time_start:.2f}s")
            
            # Extract center 2D slice
            center_slice = self._get_center_slice(denoised_patch)
            
            # Normalize to 0-255 for display
            slice_min = center_slice.min()
            slice_max = center_slice.max()
            if slice_max > slice_min:
                center_slice_normalized = ((center_slice - slice_min) / (slice_max - slice_min) * 255).astype(np.uint8)
            else:
                center_slice_normalized = center_slice.astype(np.uint8)

            # Rotate the image 90 degrees counter-clockwise
            center_slice_normalized = np.rot90(center_slice_normalized, 1)

            # center slice raw
            center_slice_raw = self._get_center_slice(raw_patch)
            slice_min_raw = center_slice_raw.min()
            slice_max_raw = center_slice_raw.max()
            if slice_max_raw > slice_min_raw:
                center_slice_raw_normalized = ((center_slice_raw - slice_min_raw) / (slice_max_raw - slice_min_raw) * 255).astype(np.uint8)
            else:
                center_slice_raw_normalized = center_slice_raw.astype(np.uint8)

            # Rotate the image 90 degrees counter-clockwise
            center_slice_raw_normalized = np.rot90(center_slice_raw_normalized, 1)
            
            print(f"Center slice shape: {center_slice.shape}, range: [{slice_min}, {slice_max}]")
            print(f"Center slice raw shape: {center_slice_raw.shape}, range: [{slice_min_raw}, {slice_max_raw}]")
            print(f"From cache: {from_cache}\n")
            
            return Response({
                'status': 'success',
                'message': 'Preview generated',
                'center_slice_raw': center_slice_raw_normalized.flatten().tolist(),
                'center_slice': center_slice_normalized.flatten().tolist(),
                'from_cache': from_cache,
                'slice_shape': list(center_slice.shape)
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            error_msg = f"Error in preview denoise: {str(e)}"
            print(f"✗ {error_msg}")
            import traceback
            traceback.print_exc()
            
            return Response({
                'status': 'error',
                'message': error_msg
            }, status=status.HTTP_400_BAD_REQUEST)