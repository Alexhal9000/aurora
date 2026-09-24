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


import psutil
import numpy as np
import nibabel as nib
import open3d as o3d
import gc  # Add garbage collection
import platform
from scipy.ndimage import zoom, binary_dilation

from django.shortcuts import render
from django.http import FileResponse, HttpResponse
from rest_framework import viewsets, status
from rest_framework.views import APIView
from rest_framework.response import Response

from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

import matplotlib
matplotlib.use('Agg') # Use non-interactive backend suitable for web servers
import matplotlib.pyplot as plt

from .registrationTools import RegistrationTools
from .batch_flag_filter import load_flagged_subject_names, apply_flag_filter, normalize_flag_filter_value
from .linkedScans import filter_out_linked_children, get_same_shape_siblings, latest_edit_stem
from .coordinateFrames import (
    is_preserved_mesh_metadata,
    partition_voxel_and_preserved_mesh_scans,
    voxel_only_all_meshes_message,
    voxel_only_no_eligible_targets_message,
)


def _filter_voxel_subject_dirs(directory, subject_dirs):
    """Keep atlas and voxel-based extracted subjects; skip preserved PLY meshes."""
    voxel_dirs = []
    skipped_meshes = []
    for subject_name in subject_dirs:
        if subject_name == "atlas":
            voxel_dirs.append(subject_name)
            continue
        json_path = os.path.join(directory, "extracted", subject_name, f"{subject_name}.json")
        if not os.path.isfile(json_path):
            voxel_dirs.append(subject_name)
            continue
        try:
            with open(json_path, "r") as jf:
                metadata = json.load(jf)
        except (json.JSONDecodeError, OSError):
            voxel_dirs.append(subject_name)
            continue
        if is_preserved_mesh_metadata(metadata):
            skipped_meshes.append(subject_name)
        else:
            voxel_dirs.append(subject_name)
    return voxel_dirs, skipped_meshes


def _scan_marked_faulty(directory, subject_name):
    """
    True if subject JSON marks the scan as faulty.
    For 'atlas', reads atlas/atlas.json; otherwise extracted/<name>/<name>.json.
    Missing or unreadable metadata is treated as not faulty (existing logic handles missing files).
    """
    if subject_name == "atlas":
        json_path = os.path.join(directory, "atlas", "atlas.json")
    else:
        json_path = os.path.join(directory, "extracted", subject_name, f"{subject_name}.json")
    if not os.path.isfile(json_path):
        return False
    try:
        with open(json_path, "r") as jf:
            metadata = json.load(jf)
        return bool(metadata.get("faulty", False))
    except (json.JSONDecodeError, OSError):
        return False


def _parse_mask_dilation(request_data):
    """Batch mask dilation in voxels (0–100, default 0)."""
    raw = request_data.get("dilation", 0)
    try:
        v = int(raw)
    except (TypeError, ValueError):
        v = 0
    return max(0, min(100, v))


def _dilate_binary_mask(mask_bool, iterations):
    """Expand a boolean 3D mask by `iterations` morphological steps (0 = no-op)."""
    if iterations <= 0:
        return mask_bool.astype(bool)
    m = np.asarray(mask_bool, dtype=bool)
    if not np.any(m):
        return m
    return binary_dilation(m, iterations=int(iterations))


class SaveMaskView(APIView):
    def post(self, request):
        directory = request.data.get('directory')
        filename = request.data.get('filename')
        edit = request.data.get('edit')
        if edit is not None:
            edit = edit.replace(".nii.gz", "")

        # --- Resolve fullResolutionState (works for JSON bool or multipart string) ---
        raw_full_res = request.data.get('fullResolutionState', False)
        if isinstance(raw_full_res, str):
            full_res = raw_full_res.strip().lower() in ('true', '1', 'yes', 'on')
        else:
            full_res = bool(raw_full_res)

        # --- Resolve mask payload: prefer binary upload, fall back to JSON list ---
        mask_file = request.FILES.get('maskData')
        if mask_file is not None:
            # New path: raw uint8 bytes posted as multipart/form-data
            try:
                mask_bytes = mask_file.read()
            finally:
                try:
                    mask_file.close()
                except Exception:
                    pass
            mask_flat = np.frombuffer(mask_bytes, dtype=np.uint8)
            # np.frombuffer returns a read-only view; copy so we can index freely
            mask_flat = np.array(mask_flat, dtype=np.uint8, copy=True)
            payload_type = 'binary'
            payload_size_mb = len(mask_bytes) / (1024 * 1024)
        else:
            # Legacy path: JSON array of ints
            mask_data = request.data.get('maskData')
            if mask_data is None:
                return Response(
                    {"error": "Missing 'maskData' in request."},
                    status=status.HTTP_400_BAD_REQUEST
                )
            mask_flat = np.asarray(mask_data, dtype=np.uint8)
            payload_type = 'json'
            payload_size_mb = len(mask_flat) / (1024 * 1024)
        
        print(f"[SaveMaskView] Received mask via {payload_type} | "
              f"Size: {payload_size_mb:.2f} MB | "
              f"Voxels: {len(mask_flat)} | "
              f"FullRes: {full_res} | "
              f"Edit: {edit} | "
              f"Filename: {filename}")
    
        try:
            # Determine base paths
            if filename == "atlas":
                base_path = os.path.join(directory, "atlas")
                metadata_path = os.path.join(directory, "atlas", "atlas.json")
            else:
                base_path = os.path.join(directory, "extracted", filename)
                metadata_path = os.path.join(base_path, f"{filename}.json")
            
            # Ensure directory exists
            os.makedirs(base_path, exist_ok=True)
            
            # Load metadata to get resolution factor
            resolution_factor = 2  # Default fallback
            
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        
                        # Get lossy parameters
                        lossy_params = metadata.get('lossy_compression', [])
                        # If lossy_params is a dict then turn into a list (recover legacy format)
                        if isinstance(lossy_params, dict):
                            lossy_params = [lossy_params]
                            # Handle case where edit is None
                            if edit is not None:
                                filename_for_metadata = edit + '.nii.gz'
                            else:
                                filename_for_metadata = f"{filename}_lossy.nii.gz"
                            if 'filename' not in lossy_params[0]:
                                lossy_params[0]['filename'] = filename_for_metadata
                            # save the updated metadata
                            with open(metadata_path, 'w') as json_file:
                                json.dump(metadata, json_file, indent=4)
                        
                        # Handle case where edit is None
                        if edit is not None:
                            lossy_filename = edit + '.nii.gz'
                        else:
                            lossy_filename = f"{filename}_lossy.nii.gz"
                        matching_lossy_params = [p for p in lossy_params if p['filename'] == lossy_filename]
                        if not matching_lossy_params: # if no lossy parameters found, use the last one
                            print(f"Warning: No lossy parameters found for {filename}. Using default values.")
                            lossy_params = lossy_params[-1] if lossy_params else {}
                        else:
                            lossy_params = matching_lossy_params[0]
                        
                        # Get the resolution factor
                        resolution_factor = lossy_params.get('resolution_factor', 2)
                        print(f"Found resolution factor: {resolution_factor}")
                        
                except Exception as e:
                    print(f"Warning: Could not read metadata, using default resolution factor: {str(e)}")
            
            # Handle case where edit is None
            if edit is None:
                lossy_edit = f"{filename}_lossy"
                full_edit = filename
            else:
                # Determine whether edit name is already in lossy or full-res form
                # by checking if it contains '_lossy' in the naming pattern
                if '_lossy' in edit:
                    # Edit is in lossy form (e.g., 'scan_lossy' or 'scan_lossy_edit_001')
                    lossy_edit = edit
                    full_edit = edit.replace('_lossy', '')
                else:
                    # Edit is in full-res form (e.g., 'scan' or 'scan_edit_001')
                    # Need to derive the lossy name by adding _lossy
                    full_edit = edit
                    if '_edit_' in edit:
                        lossy_edit = edit.replace('_edit_', '_lossy_edit_')
                    else:
                        lossy_edit = f"{edit}_lossy"
            
            # Determine which image we're working with
            if full_res:
                # Working with full resolution
                source_image_path = os.path.join(base_path, full_edit + ".nii.gz")
                source_edit = full_edit
                target_image_path = os.path.join(base_path, lossy_edit + ".nii.gz")
                target_edit = lossy_edit
            else:
                # Working with lossy resolution
                source_image_path = os.path.join(base_path, lossy_edit + ".nii.gz")
                source_edit = lossy_edit
                target_image_path = os.path.join(base_path, full_edit + ".nii.gz")
                target_edit = full_edit
            
            # Load the source image (the one we're working with)
            source_image = nib.load(source_image_path)
            source_dims = source_image.header.get_data_shape()
            source_width, source_height, source_depth = source_dims[0], source_dims[1], source_dims[2]
            
            # mask_flat is already constructed above from either binary or JSON payload
            # Fast vectorized reshaping matching the frontend's coordinate system
            expected_len = source_width * source_height * source_depth
            if len(mask_flat) < expected_len:
                mask_flat = np.pad(mask_flat, (0, expected_len - len(mask_flat)))
            elif len(mask_flat) > expected_len:
                mask_flat = mask_flat[:expected_len]

            source_mask_array = mask_flat.reshape((source_depth, source_height, source_width), order='C')
            source_mask_array = source_mask_array.transpose(2, 1, 0)[:, ::-1, ::-1].astype(np.uint8)
            
            # Save the mask at the source resolution
            source_mask_filename = f"{source_edit}.nii.mask.gz"
            source_mask_temp_filename = f"{source_edit}_mask.nii.gz"
            source_save_path = os.path.join(base_path, source_mask_filename)
            source_temp_path = os.path.join(base_path, source_mask_temp_filename)
            
            source_mask_img = nib.Nifti1Image(source_mask_array, source_image.affine, header=source_image.header)
            nib.save(source_mask_img, source_temp_path)
            os.replace(source_temp_path, source_save_path)
            
            # Now create the complementary resolution mask
            try:
                if os.path.exists(target_image_path):
                    target_image = nib.load(target_image_path)
                    target_dims = target_image.header.get_data_shape()
                    target_shape = (target_dims[0], target_dims[1], target_dims[2])
                    
                    # Scale the mask to target resolution
                    scaled_mask = self.safe_zoom(source_mask_array, target_shape)
                    
                    # Save target resolution mask
                    target_mask_filename = f"{target_edit}.nii.mask.gz"
                    target_mask_temp_filename = f"{target_edit}_mask.nii.gz"
                    target_save_path = os.path.join(base_path, target_mask_filename)
                    target_temp_path = os.path.join(base_path, target_mask_temp_filename)
                    
                    target_mask_img = nib.Nifti1Image(scaled_mask, target_image.affine, header=target_image.header)
                    nib.save(target_mask_img, target_temp_path)
                    os.replace(target_temp_path, target_save_path)
                    
                    resolution_type = "full" if not full_res else "lossy"
                    print(f"Created complementary {resolution_type} resolution mask with resolution factor: {resolution_factor}")
                        
            except Exception as e:
                print(f"Warning: Could not create complementary resolution mask: {str(e)}")

            try:
                from .pipeline_log import merge_mask_workflow_metadata
                from datetime import datetime, timezone
                merge_mask_workflow_metadata(directory, filename, {
                    "creation": {
                        "on": filename,
                        "edit": source_edit,
                        "method": "manual_paint",
                        "ts": datetime.now(timezone.utc).isoformat(),
                    },
                })
            except Exception as _mw_err:
                print(f"Warning: mask_workflow creation metadata update failed: {_mw_err}")
            
            return Response(
                {"message": f"Mask saved successfully as {source_mask_filename}", "maskName": source_mask_filename},
                status=status.HTTP_200_OK
            )
            
        except Exception as e:
            import traceback
            print(f"[SaveMaskView ERROR] Failed to save mask: {str(e)}")
            print(f"[SaveMaskView ERROR] Details: {traceback.format_exc()}")
            print(f"[SaveMaskView ERROR] Mask info: shape={mask_flat.shape}, dtype={mask_flat.dtype}, "
                  f"voxels={len(mask_flat)}, size_mb={len(mask_flat) / (1024 * 1024):.2f}")
            return Response(
                {"error": f"Failed to save mask: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def safe_zoom(self, arr, target_shape):
        """Scale array to target shape using the same approach as ElasticRegistrationView"""
        from scipy.ndimage import zoom
        
        # Calculate zoom factors for each dimension
        zoom_factors = [target_shape[i] / arr.shape[i] for i in range(len(target_shape))]
        
        # Apply zoom with nearest neighbor interpolation for masks
        scaled_arr = zoom(arr, zoom_factors, order=0)  # order=0 for nearest neighbor
        
        # Ensure the output has exactly the target shape
        if scaled_arr.shape != target_shape:
            # Crop or pad if necessary
            result = np.zeros(target_shape, dtype=arr.dtype)
            slices = tuple(slice(0, min(scaled_arr.shape[i], target_shape[i])) for i in range(len(target_shape)))
            result[slices] = scaled_arr[slices]
            return result
        
        return scaled_arr

class LoadMaskView(APIView):
    def post(self, request):
        directory = request.data.get('directory')
        filename = request.data.get('filename')
        edit = request.data.get('edit')
        if edit is not None:
            edit = edit.replace(".nii.gz", "")
        full_res = request.data.get('fullResolutionState', False)
        
        if not directory or not filename:
            return Response(
                {"error": "Both 'directory' and 'filename' fields are required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            if edit is None:
                mask_edit = filename if full_res else f"{filename}_lossy"
            else:
                # Strip any existing _lossy to get a clean base string
                base_edit = edit.replace('_lossy', '')
                
                # Reconstruct the correct name for the requested resolution
                if full_res:
                    mask_edit = base_edit
                else:
                    if "_edit_" in base_edit:
                        mask_edit = base_edit.replace("_edit_", "_lossy_edit_")
                    else:
                        mask_edit = f"{base_edit}_lossy"
            
            # Create mask filename and temp filename
            mask_filename = f"{mask_edit}.nii.mask.gz"
            mask_temp_filename = f"{mask_edit}_mask.nii.gz"
            print("Loading mask: ", mask_filename)
            
            # Determine file paths
            if filename == "atlas":
                file_path = os.path.join(directory, "atlas", mask_filename)
                temp_path = os.path.join(directory, "atlas", mask_temp_filename)
            else:
                file_path = os.path.join(directory, "extracted", filename, mask_filename)
                temp_path = os.path.join(directory, "extracted", filename, mask_temp_filename)

            
            if not os.path.isfile(file_path):
                print(f"Mask file not found: {file_path}")
                print(f"Directory exists: {os.path.exists(os.path.dirname(file_path))}")
                print(f"Directory contents: {os.listdir(os.path.dirname(file_path)) if os.path.exists(os.path.dirname(file_path)) else 'Directory does not exist'}")
                print(f"Looking for file: {os.path.basename(file_path)}")
                return Response(
                    {"error": f"Mask file not found: {mask_filename}"},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Temporarily rename to standard extension for nibabel
            os.replace(file_path, temp_path)
            print(f"Mask file renamed to: {temp_path}")
            assert os.path.isfile(temp_path)

            try:
                # Load the mask using temp file
                mask_img = nib.load(temp_path)
                dims = mask_img.header.get_data_shape()
                mask_data = mask_img.get_fdata()
                width = dims[0]   # dims[1] in frontend
                height = dims[1]  # dims[2] in frontend  
                depth = dims[2]   # dims[3] in frontend
                
                # Convert 3D mask back to flat array using reverse of save indexing
                # Z and Y axes are reversed in the frontend flat array, and Z is the slowest dimension (depth, height, width).
                # Round before astype to avoid truncating floats (e.g. 0.999 to 0) caused by NIfTI scl_slope headers
                mask_flat = np.round(mask_data[:, ::-1, ::-1].transpose(2, 1, 0).flatten(order='C')).astype(np.uint8)
                flat_size = mask_flat.size
                
                # Convert to list for JSON serialization (OLD - commented out)
                # brush_data = mask_flat.tolist()

            finally:
                # Always rename back to original custom extension
                os.replace(temp_path, file_path)

            # Return binary data instead of JSON for better performance with large masks
            response = HttpResponse(mask_flat.tobytes(), content_type='application/octet-stream')
            response['X-Mask-Name'] = mask_filename
            response['X-Mask-Dimensions'] = f"{width},{height},{depth}"
            return response

            # OLD JSON response (commented out for reference)
            # return Response(
            #     {"maskData": brush_data, "maskName": mask_filename},
            #     status=status.HTTP_200_OK
            # )
            
        except Exception as e:
            print(f"Error loading mask: {str(e)}")
            return Response(
                {"error": f"Failed to load mask: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class UploadMaskView(APIView):
    def post(self, request):
        directory = request.data.get('directory')
        filename = request.data.get('filename')
        edit = request.data.get('edit')
        uploaded_file = request.FILES.get('maskFile')
        
        if not directory or not filename or not edit or not uploaded_file:
            return Response(
                {"error": "Directory, filename, edit, and file are required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Validate file extension
        if not (uploaded_file.name.endswith('.nii') or uploaded_file.name.endswith('.nii.gz')):
            return Response(
                {"error": "Only .nii and .nii.gz files are supported."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # Handle case where edit is None (raw extracted scan, no edits yet)
            if not edit or edit == "" or edit == None or edit == "null":
                edit = filename + "_lossy.nii.gz"

            # Process edit parameter - frontend passes lossy name
            edit_clean = edit.replace(".nii.gz", "")
            lossy_edit = edit_clean
            full_edit = edit_clean.replace('_lossy', '')

            #debug prints
            print("filename: ", filename)
            print("edit: ", edit)
            print("edit_clean: ", edit_clean)
            print("lossy_edit: ", lossy_edit)
            print("full_edit: ", full_edit)
            
            # Determine base paths
            if filename == "atlas":
                base_path = os.path.join(directory, "atlas")
                metadata_path = os.path.join(directory, "atlas", "atlas.json")
            else:
                base_path = os.path.join(directory, "extracted", filename)
                metadata_path = os.path.join(base_path, f"{filename}.json")
            
            # Ensure directory exists
            os.makedirs(base_path, exist_ok=True)
            
            # Load metadata to get resolution factor
            resolution_factor = 2  # Default fallback
            
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        
                        # Get lossy parameters
                        lossy_params = metadata.get('lossy_compression', [])
                        # If lossy_params is a dict then turn into a list (recover legacy format)
                        if isinstance(lossy_params, dict):
                            lossy_params = [lossy_params]
                            # if the dict has no filename, add it
                            if 'filename' not in lossy_params[0]:
                                lossy_params[0]['filename'] = edit
                            # save the updated metadata
                            with open(metadata_path, 'w') as json_file:
                                json.dump(metadata, json_file, indent=4)
                        
                        # Get the lossy parameters for the current lossy file
                        lossy_filename = edit
                        matching_lossy_params = [p for p in lossy_params if p['filename'] == lossy_filename]
                        if not matching_lossy_params: # if no lossy parameters found, use the last one
                            print(f"Warning: No lossy parameters found for {filename}. Using default values.")
                            lossy_params = lossy_params[-1] if lossy_params else {}
                        else:
                            lossy_params = matching_lossy_params[0]
                        
                        # Get the resolution factor
                        resolution_factor = lossy_params.get('resolution_factor', 2)
                        print(f"Found resolution factor: {resolution_factor}")
                        
                except Exception as e:
                    print(f"Warning: Could not read metadata, using default resolution factor: {str(e)}")
            
            # Always save uploaded mask as full resolution
            full_mask_filename = f"{full_edit}.nii.mask.gz"
            full_mask_temp_filename = f"{full_edit}_mask.nii.gz"
            full_save_path = os.path.join(base_path, full_mask_filename)
            full_temp_path = os.path.join(base_path, full_mask_temp_filename)
            
            # Save uploaded file temporarily with proper extension
            file_extension = '.nii.gz' if uploaded_file.name.endswith('.nii.gz') else '.nii'
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_upload:
                for chunk in uploaded_file.chunks():
                    temp_upload.write(chunk)
                temp_upload_path = temp_upload.name
            
            try:
                # Load and re-save as temp file with standard extension
                mask_img = nib.load(temp_upload_path)
                nib.save(mask_img, full_temp_path)
                
                # Rename to custom extension
                os.replace(full_temp_path, full_save_path)
                
                # Now create the lossy version
                try:
                    # Load the corresponding lossy image to get dimensions
                    lossy_image_path = os.path.join(base_path, lossy_edit + ".nii.gz")
                    if os.path.exists(lossy_image_path):
                        lossy_image = nib.load(lossy_image_path)
                        lossy_dims = lossy_image.header.get_data_shape()
                        target_shape = (lossy_dims[0], lossy_dims[1], lossy_dims[2])
                        
                        # Get the uploaded mask data
                        mask_data = mask_img.get_fdata()
                        
                        # Scale down the mask
                        scaled_mask = self.safe_zoom(mask_data, target_shape)
                        
                        # Save lossy mask
                        lossy_mask_filename = f"{lossy_edit}.nii.mask.gz"
                        lossy_mask_temp_filename = f"{lossy_edit}_mask.nii.gz"
                        lossy_save_path = os.path.join(base_path, lossy_mask_filename)
                        lossy_temp_path = os.path.join(base_path, lossy_mask_temp_filename)
                        
                        lossy_mask_img = nib.Nifti1Image(scaled_mask, lossy_image.affine, header=lossy_image.header)
                        nib.save(lossy_mask_img, lossy_temp_path)
                        os.replace(lossy_temp_path, lossy_save_path)
                        print(f"Created complementary lossy mask with resolution factor: {resolution_factor}")
                        
                except Exception as e:
                    print(f"Warning: Could not create complementary lossy mask: {str(e)}")
                
            finally:
                # Clean up upload temp file
                os.unlink(temp_upload_path)
            
            return Response(
                {"message": f"Mask uploaded successfully as {full_mask_filename}", "maskName": full_mask_filename},
                status=status.HTTP_200_OK
            )
            
        except Exception as e:
            print(f"Error uploading mask: {str(e)}")
            return Response(
                {"error": f"Failed to upload mask: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def safe_zoom(self, arr, target_shape):
        """Scale array to target shape using the same approach as SaveMaskView"""
        from scipy.ndimage import zoom
        
        # Calculate zoom factors for each dimension
        zoom_factors = [target_shape[i] / arr.shape[i] for i in range(len(target_shape))]
        
        # Apply zoom with nearest neighbor interpolation for masks
        scaled_arr = zoom(arr, zoom_factors, order=0)  # order=0 for nearest neighbor
        
        # Ensure the output has exactly the target shape
        if scaled_arr.shape != target_shape:
            # Crop or pad if necessary
            result = np.zeros(target_shape, dtype=arr.dtype)
            slices = tuple(slice(0, min(scaled_arr.shape[i], target_shape[i])) for i in range(len(target_shape)))
            result[slices] = scaled_arr[slices]
            return result
        
        return scaled_arr

class DownloadMaskView(APIView):
    def post(self, request):
        directory = request.data.get('directory')
        filename = request.data.get('filename')
        edit = request.data.get('edit')
        
        if not directory or not filename or not edit:
            return Response(
                {"error": "Directory, filename, and edit fields are required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:

            # Handle case where edit is None (raw extracted scan, no edits yet)
            if not edit or edit == "" or edit == None or edit == "null":
                edit = filename + ".nii.gz"

            # Process edit parameter - always download full resolution
            edit_clean = edit.replace(".nii.gz", "")
            full_edit = edit_clean.replace('_lossy', '')  # Remove _lossy to get full resolution
            
            # Create mask filename and temp filename for full resolution
            mask_filename = f"{full_edit}.nii.mask.gz"
            mask_temp_filename = f"{full_edit}_mask.nii.gz"
            
            # Determine file paths
            if filename == "atlas":
                file_path = os.path.join(directory, "atlas", mask_filename)
                temp_path = os.path.join(directory, "atlas", mask_temp_filename)
            else:
                file_path = os.path.join(directory, "extracted", filename, mask_filename)
                temp_path = os.path.join(directory, "extracted", filename, mask_temp_filename)
            
            if not os.path.isfile(file_path):
                return Response(
                    {"error": f"Mask file not found: {mask_filename}"},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Temporarily rename to standard extension for serving
            os.replace(file_path, temp_path)
            
            try:
                # Return the file for download
                response = FileResponse(
                    open(temp_path, 'rb'), 
                    content_type='application/octet-stream'
                )
                response['Content-Disposition'] = f'attachment; filename="{mask_filename}"'
                return response
            finally:
                # Always rename back to original custom extension
                if os.path.isfile(temp_path):
                    os.replace(temp_path, file_path)
            
        except FileNotFoundError:
            return Response(
                {"error": f"Mask file not found: {mask_filename}"},
                status=status.HTTP_404_NOT_FOUND
            )
        except PermissionError:
            return Response(
                {"error": f"Permission denied: {mask_filename}"},
                status=status.HTTP_403_FORBIDDEN
            )
        except Exception as e:
            print(f"Error downloading mask: {str(e)}")
            return Response(
                {"error": f"Failed to download mask: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class DeleteMaskView(APIView):
    def post(self, request):
        directory = request.data.get('directory')
        filename = request.data.get('filename')
        edit = request.data.get('edit')
        
        if not directory or not filename or not edit:
            return Response(
                {"error": "Directory, filename, and edit fields are required."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # Handle case where edit is None (raw extracted scan, no edits yet)
            if not edit or edit == "" or edit == None or edit == "null":
                edit = filename + ".nii.gz"

            # Process edit parameter - delete both full and lossy versions
            edit_clean = edit.replace(".nii.gz", "")
            lossy_edit = edit_clean
            full_edit = edit_clean.replace('_lossy', '')
            
            # Create mask filenames for both resolutions
            full_mask_filename = f"{full_edit}.nii.mask.gz"
            lossy_mask_filename = f"{lossy_edit}.nii.mask.gz"
            
            # Determine file paths
            if filename == "atlas":
                full_file_path = os.path.join(directory, "atlas", full_mask_filename)
                lossy_file_path = os.path.join(directory, "atlas", lossy_mask_filename)
            else:
                full_file_path = os.path.join(directory, "extracted", filename, full_mask_filename)
                lossy_file_path = os.path.join(directory, "extracted", filename, lossy_mask_filename)
            
            deleted_files = []
            
            # Delete full resolution mask if it exists
            if os.path.isfile(full_file_path):
                os.remove(full_file_path)
                deleted_files.append(full_mask_filename)
                print(f"Deleted full resolution mask: {full_mask_filename}")
            
            # Delete lossy mask if it exists
            if os.path.isfile(lossy_file_path):
                os.remove(lossy_file_path)
                deleted_files.append(lossy_mask_filename)
                print(f"Deleted lossy mask: {lossy_mask_filename}")
            
            if not deleted_files:
                return Response(
                    {"error": f"No mask files found to delete"},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            return Response(
                {"message": f"Mask(s) deleted successfully: {', '.join(deleted_files)}"},
                status=status.HTTP_200_OK
            )
            
        except PermissionError:
            return Response(
                {"error": f"Permission denied: Cannot delete mask files"},
                status=status.HTTP_403_FORBIDDEN
            )
        except Exception as e:
            print(f"Error deleting mask: {str(e)}")
            return Response(
                {"error": f"Failed to delete mask: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )



class CopyMaskToReferenceView(APIView):
    def post(self, request):
        directory = request.data.get('directory')
        selected_scan = request.data.get('selected_scan')
        selected_edit_mask = request.data.get('selected_edit_mask')
        reference_scan = request.data.get('reference_scan')
        reference_edit_mask = request.data.get('reference_edit_mask')
        
        # Validate required parameters
        if not all([directory, selected_scan, selected_edit_mask, reference_scan, reference_edit_mask]):
            return Response(
                {"error": "Missing required parameters: directory, selected_scan, selected_edit_mask, reference_scan, reference_edit_mask"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Validate that selected edit is different from reference
        if selected_scan == reference_scan:
            return Response(
                {"error": "Cannot copy mask to the same scan"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:                        
            # Find available source mask files (both full and lossy versions) the ones passed from the frontend include the word lossy
            source_masks = []
            
            # Look for full resolution version
            full_mask_name = f"{selected_edit_mask.replace('_lossy', '')}"
            full_mask_path = os.path.join(directory, "extracted", selected_scan, full_mask_name)
            if os.path.isfile(full_mask_path):
                source_masks.append(('full', full_mask_name, full_mask_path))
            
            # Look for lossy version
            lossy_mask_name = f"{selected_edit_mask}"
            lossy_mask_path = os.path.join(directory, "extracted", selected_scan, lossy_mask_name)
            if os.path.isfile(lossy_mask_path):
                source_masks.append(('lossy', lossy_mask_name, lossy_mask_path))
            
            if not source_masks:
                return Response(
                    {"error": f"No mask files found for selected scan '{selected_scan}' with edit '{selected_edit_mask}'"},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Check for existing reference masks
            existing_reference_masks = []
            copied_masks = []
            overwritten_masks = []
            
            for mask_type, mask_name, source_path in source_masks:
                if mask_type == 'full':
                    # Copy full resolution to reference full resolution
                    dest_name = f"{reference_edit_mask.replace('_lossy', '')}"
                else:
                    # Copy lossy to reference lossy
                    dest_name = f"{reference_edit_mask}"
                    
                dest_path = os.path.join(directory, "extracted", reference_scan, dest_name)
                
                # Check if destination already exists
                if os.path.exists(dest_path):
                    existing_reference_masks.append(dest_name)
                
                # Perform the copy operation
                try:
                    # Copy with temporary file to ensure atomic operation
                    temp_dest_path = dest_path + '.tmp'
                    
                    # Copy the file
                    shutil.copy2(source_path, temp_dest_path)
                    
                    # Verify the copy was successful
                    if not os.path.exists(temp_dest_path) or os.path.getsize(temp_dest_path) != os.path.getsize(source_path):
                        raise Exception(f"Copy verification failed for {dest_name}")
                    
                    # Atomic rename to final destination
                    if os.path.exists(dest_path):
                        os.remove(dest_path)  # Remove existing file
                        overwritten_masks.append(dest_name)
                    else:
                        copied_masks.append(dest_name)
                        
                    os.replace(temp_dest_path, dest_path)
                    
                    print(f"Successfully copied mask: {mask_name} -> {dest_name}")
                    
                except Exception as e:
                    # Clean up temp file if it exists
                    if os.path.exists(temp_dest_path):
                        try:
                            os.remove(temp_dest_path)
                        except:
                            pass
                    
                    raise Exception(f"Failed to copy {mask_name} to {dest_name}: {str(e)}")
            
            # Prepare response message
            if existing_reference_masks:
                message = f"Mask copied successfully. Overwritten existing masks."
            else:
                message = f"Mask copied successfully."
            
            return Response(
                {
                    "message": message,
                    "source_scan": selected_scan,
                    "reference_scan": reference_scan,
                    "copied_masks": copied_masks,
                    "overwritten_masks": overwritten_masks,
                    "total_masks_copied": len(copied_masks) + len(overwritten_masks)
                },
                status=status.HTTP_200_OK
            )
            
        except Exception as e:
            print(f"Error copying mask to reference: {str(e)}")
            return Response(
                {"error": f"Failed to copy mask: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ListMaskFoldersView(APIView):
    """
    API endpoint to list existing mask folders in the main directory.
    
    POST Parameters:
        - directory (str): Path to the main data directory
    
    Response:
        {
            "mask_folders": ["folder1", "folder2", ...],
            "count": int
        }
    """
    
    def post(self, request):
        directory = request.data.get('directory')
        
        if not directory:
            return Response(
                {"error": "Directory parameter is required"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            masks_dir = os.path.join(directory, "masks")
            
            # Check if masks directory exists
            if not os.path.exists(masks_dir):
                # Return empty list if masks directory doesn't exist yet
                return Response(
                    {
                        "mask_folders": [],
                        "count": 0,
                        "message": "No masks directory found"
                    },
                    status=status.HTTP_200_OK
                )
            
            # List all subdirectories in masks folder
            mask_folders = [
                name for name in os.listdir(masks_dir)
                if os.path.isdir(os.path.join(masks_dir, name))
            ]
            
            # Sort alphabetically for better UX
            mask_folders.sort()
            
            return Response(
                {
                    "mask_folders": mask_folders,
                    "count": len(mask_folders)
                },
                status=status.HTTP_200_OK
            )
            
        except Exception as e:
            print(f"Error listing mask folders: {str(e)}")
            return Response(
                {"error": f"Failed to list mask folders: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
def _resolve_mask_source(directory, own_base_path, own_edit_stem, mask_source):
    """
    Where to read the segmentation mask from.

    Defaults to the scan's own directory and edit. When `mask_source` names another scan
    (a linked-group replay), the mask is read from that scan instead so every member of the
    group has the exact same voxels removed rather than each applying its own segmentation.

    Returns:
        tuple: (base_path, edit_stem)
    """
    if not isinstance(mask_source, dict):
        return own_base_path, own_edit_stem

    source_filename = mask_source.get('filename')
    if not source_filename:
        return own_base_path, own_edit_stem

    if source_filename == "atlas":
        source_base_path = os.path.join(directory, "atlas")
    else:
        source_base_path = os.path.join(directory, "extracted", source_filename)

    source_edit = mask_source.get('edit')
    if not source_edit:
        source_edit_stem = source_filename
    else:
        source_edit_stem = str(source_edit).replace('.nii.gz', '').replace('_lossy', '')

    return source_base_path, source_edit_stem


class ApplyMaskToImageView(APIView):
    """
    API endpoint to apply a mask to an image and save it to a specified folder.
    
    This function:
    1. Loads the mask (from .nii.mask.gz)
    2. Loads the original image
    3. Applies the mask (zeros out non-masked voxels)
    4. Creates proper directory structure: masks/folder_name/extracted/subject_name/
    5. Copies project_settings.json if it exists
    6. Copies elastic transformations if the next edit is elastic
    7. Creates empty raw file in main folder: masks/folder_name/
    8. Copies JSON metadata (without lossy_compression items)
    9. Saves both full resolution and lossy versions
    
    POST Parameters:
        - directory (str): Path to the main data directory
        - filename (str): Subject/scan filename
        - edit (str): The specific edit being masked (e.g., "scan_lossy.nii.gz")
        - mask_folder (str): Target mask folder name
        - create_new (bool): Whether to create a new folder
    
    Response:
        {
            "message": "Success message",
            "mask_folder": str,
            "saved_files": [list of saved filenames],
            "output_path": str
        }
    """
    
    def post(self, request):
        directory = request.data.get('directory')
        filename = request.data.get('filename')
        edit = request.data.get('edit')
        mask_folder = request.data.get('mask_folder')
        create_new = request.data.get('create_new', False)
        # Linked siblings must keep the exact voxels the operated scan kept, so they borrow
        # its mask rather than using their own segmentation of the same anatomy.
        mask_source = request.data.get('mask_source')
        
        # Validate required parameters
        if not all([directory, filename, edit, mask_folder]):
            return Response(
                {
                    "error": "Missing required parameters: directory, filename, edit, mask_folder"
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # Handle case where edit is None (raw extracted scan, no edits yet)
            if not edit or edit == "" or edit == None or edit == "null":
                edit = filename + "_lossy.nii.gz"

            # Process edit parameter - remove .nii.gz extension if present
            edit_clean = edit.replace(".nii.gz", "")
            lossy_edit = edit_clean
            full_edit = edit_clean.replace('_lossy', '')
            
            # Determine base path for the scan
            if filename == "atlas":
                base_path = os.path.join(directory, "atlas")
                metadata_source_path = os.path.join(directory, "atlas", "atlas.json")
            else:
                base_path = os.path.join(directory, "extracted", filename)
                metadata_source_path = os.path.join(base_path, f"{filename}.json")
            
            if not os.path.exists(base_path):
                return Response(
                    {"error": f"Subject directory not found: {base_path}"},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Create proper directory structure: masks/folder_name/extracted/subject_name/
            masks_base_dir = os.path.join(directory, "masks", mask_folder)
            
            if filename == "atlas":
                target_subject_dir = os.path.join(masks_base_dir, "atlas")
            else:
                target_subject_dir = os.path.join(masks_base_dir, "extracted", filename)
            
            # Create both the base masks directory and the subject directory
            os.makedirs(target_subject_dir, exist_ok=True)
            print(f"Created mask folder structure: {target_subject_dir}")
            
            if create_new:
                print(f"Initialized new mask folder: {masks_base_dir}")

            # ===== COPY PROJECT SETTINGS IF EXISTS =====
            if filename != "atlas":
                project_settings_source = os.path.join(directory, "extracted", "project_settings.json")
                project_settings_target_dir = os.path.join(masks_base_dir, "extracted")
                project_settings_target = os.path.join(project_settings_target_dir, "project_settings.json")
                
                if os.path.exists(project_settings_source) and not os.path.exists(project_settings_target):
                    os.makedirs(project_settings_target_dir, exist_ok=True)
                    shutil.copy2(project_settings_source, project_settings_target)
                    print(f"Copied project_settings.json to: {project_settings_target}")

            # ===== CHECK FOR ELASTIC TRANSFORMATIONS =====
            # Extract edit number from full_edit to find elastic transformations
            edit_number = None
            if '_edit_' in full_edit:
                try:
                    edit_number = int(full_edit.split('_edit_')[-1].split('_')[0])
                except (ValueError, IndexError):
                    edit_number = None

            if edit_number is not None and filename != "atlas":
                next_edit_number = edit_number + 1
                
                # Look for elastic transformations from the original edit
                # These should be renamed to edit_1 (since masked data is now edit_0)
                elastic_fwd_src = os.path.join(base_path, f"{filename}_edit_{next_edit_number}_elastic_fwd.nii.gz")
                elastic_inv_src = os.path.join(base_path, f"{filename}_edit_{next_edit_number}_elastic_inv.nii.gz")
                
                print(f"Looking for: {elastic_fwd_src}")
                print(f"Looking for: {elastic_inv_src}")
                
                # If either elastic transformation exists, copy both and rename to edit_1
                if os.path.exists(elastic_fwd_src) or os.path.exists(elastic_inv_src):
                    print(f"Found elastic transformations for edit {next_edit_number}, copying and renaming to edit_1...")
                    
                    # Rename to edit_1 (since masked data is edit_0)
                    # Elastic files don't include the suffix
                    elastic_fwd_dst = os.path.join(target_subject_dir, f"{filename}_edit_1_elastic_fwd.nii.gz")
                    elastic_inv_dst = os.path.join(target_subject_dir, f"{filename}_edit_1_elastic_inv.nii.gz")
                    
                    if os.path.exists(elastic_fwd_src):
                        shutil.copy2(elastic_fwd_src, elastic_fwd_dst)
                        print(f"Copied elastic forward transformation to: {elastic_fwd_dst}")
                    else:
                        print(f"Warning: Elastic forward transformation not found at: {elastic_fwd_src}")
                    
                    if os.path.exists(elastic_inv_src):
                        shutil.copy2(elastic_inv_src, elastic_inv_dst)
                        print(f"Copied elastic inverse transformation to: {elastic_inv_dst}")
                    else:
                        print(f"Warning: Elastic inverse transformation not found at: {elastic_inv_src}")
                else:
                    print(f"No elastic transformations found for edit {next_edit_number}")
            else:
                if edit_number is None:
                    print(f"Could not extract edit number from: {full_edit}")
            # ===== CREATE EMPTY RAW FILE IN MAIN FOLDER =====
            # Create an empty .nii.gz file in the main mask folder (beside extracted)
            empty_raw_filename = f"{filename}.nii.gz"
            empty_raw_path = os.path.join(masks_base_dir, empty_raw_filename)

            # Create an empty NIfTI file (minimal header, no data)
            # This follows the platform convention of having a raw file at the top level
            empty_header = nib.Nifti1Header()
            empty_img = nib.Nifti1Image(np.zeros((1, 1, 1), dtype=np.uint8), np.eye(4), header=empty_header)
            nib.save(empty_img, empty_raw_path)

            print(f"Created empty raw file: {empty_raw_path}")
            saved_files = [empty_raw_filename]

            # ===== COPY AND MODIFY JSON METADATA =====
            if os.path.exists(metadata_source_path):
                with open(metadata_source_path, 'r') as f:
                    metadata = json.load(f)
                
                # Remove lossy_compression field to reset it for the masked images
                metadata_copy = metadata.copy()
                if 'lossy_compression' in metadata_copy:
                    del metadata_copy['lossy_compression']
                
                # Save metadata to target directory
                if filename == "atlas":
                    target_metadata_path = os.path.join(target_subject_dir, "atlas.json")
                else:
                    target_metadata_path = os.path.join(target_subject_dir, f"{filename}.json")
                
                with open(target_metadata_path, 'w') as f:
                    json.dump(metadata_copy, f, indent=4)
                
                print(f"Copied metadata to: {target_metadata_path}")
            else:
                print(f"Warning: Source metadata not found at {metadata_source_path}")
            
            # ===== LOAD FULL RESOLUTION MASK =====
            mask_base_path, mask_edit_stem = _resolve_mask_source(
                directory, base_path, full_edit, mask_source
            )
            full_mask_filename = f"{mask_edit_stem}.nii.mask.gz"
            full_mask_temp_filename = f"{mask_edit_stem}_mask.nii.gz"
            full_mask_path = os.path.join(mask_base_path, full_mask_filename)
            full_mask_temp_path = os.path.join(mask_base_path, full_mask_temp_filename)
            
            if not os.path.exists(full_mask_path):
                return Response(
                    {"error": f"Mask file not found: {full_mask_filename}"},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Load mask with temporary rename
            os.replace(full_mask_path, full_mask_temp_path)
            
            try:
                mask_img = nib.load(full_mask_temp_path)
                # Round before astype to avoid truncating floats (e.g. 0.999 to 0) caused by NIfTI scl_slope headers
                mask_data_3d = np.round(mask_img.get_fdata()).astype(np.uint8)
            finally:
                # Always rename back
                os.replace(full_mask_temp_path, full_mask_path)
            
            # ===== LOAD AND APPLY MASK TO FULL RESOLUTION IMAGE =====
            full_image_filename = f"{full_edit}.nii.gz"
            full_image_path = os.path.join(base_path, full_image_filename)
            
            if not os.path.exists(full_image_path):
                return Response(
                    {"error": f"Full resolution image not found: {full_image_filename}"},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            full_image_img = nib.load(full_image_path)
            full_image_data = full_image_img.get_fdata()

            if mask_data_3d.shape != full_image_data.shape:
                return Response(
                    {
                        "error": (
                            f"Mask shape {mask_data_3d.shape} does not match image shape "
                            f"{full_image_data.shape} for {filename}."
                        )
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Apply mask: zero out voxels where mask is 0
            # Create binary mask (any non-zero value in mask = keep the voxel)
            binary_mask = (mask_data_3d > 0).astype(np.float32)
            masked_full_data = full_image_data * binary_mask
            
            # Extract suffix from original edit name to preserve it
            # Pattern: {filename}_edit_{number}_{suffix}
            suffix = ""
            suffix_pattern = f"{filename}_edit_\\d+(.*)$"
            import re
            match = re.match(suffix_pattern, full_edit)
            if match:
                suffix = match.group(1)

            # Save full resolution masked image with edit number changed to 0
            full_output_filename = f"{filename}_edit_0{suffix}.nii.gz"
            full_output_path = os.path.join(target_subject_dir, full_output_filename)

            full_masked_img = nib.Nifti1Image(
                masked_full_data, 
                full_image_img.affine, 
                header=full_image_img.header
            )
            nib.save(full_masked_img, full_output_path)

            print(f"Saved full resolution masked image: {full_output_path}")
            saved_files.append(full_output_filename)

            # Create empty placeholder for {filename}.nii.gz for frontend compatibility
            placeholder_full_filename = f"{filename}.nii.gz"
            placeholder_full_path = os.path.join(target_subject_dir, placeholder_full_filename)
            empty_header = nib.Nifti1Header()
            empty_img = nib.Nifti1Image(np.zeros((1, 1, 1), dtype=np.uint8), np.eye(4), header=empty_header)
            nib.save(empty_img, placeholder_full_path)
            print(f"Created empty placeholder: {placeholder_full_path}")
            saved_files.append(placeholder_full_filename)

            # ===== CREATE AND SAVE LOSSY VERSION WITH METADATA UPDATE =====
            # Create lossy version with _lossy prefix and edit 0
            lossy_output_filename = f"{filename}_lossy_edit_0{suffix}.nii.gz"
            lossy_output_path = os.path.join(target_subject_dir, lossy_output_filename)

            try:
                # Get voxel size from metadata
                voxel_size = metadata_copy.get('voxel_size', 1.0)
                
                # Use RegistrationTools to save lossy version with automatic metadata update
                registration_tools = RegistrationTools()
                registration_tools.save_as_lossy_nifti(
                    masked_full_data,
                    voxel_size,
                    target_metadata_path,  # This will be updated with lossy_compression entry
                    lossy_output_path
                )
                
                print(f"Saved lossy resolution masked image with metadata update: {lossy_output_path}")
                saved_files.append(lossy_output_filename)
                
                # Create empty placeholder for {filename}_lossy.nii.gz for frontend compatibility
                placeholder_lossy_filename = f"{filename}_lossy.nii.gz"
                placeholder_lossy_path = os.path.join(target_subject_dir, placeholder_lossy_filename)
                nib.save(empty_img, placeholder_lossy_path)
                print(f"Created empty placeholder: {placeholder_lossy_path}")
                saved_files.append(placeholder_lossy_filename)
                
            except Exception as e:
                print(f"Warning: Could not create lossy version with compression: {str(e)}")
                # Fallback: Create simple downscaled version without compression
                try:
                    lossy_image_path = os.path.join(base_path, f"{lossy_edit}.nii.gz")
                    if os.path.exists(lossy_image_path):
                        lossy_image = nib.load(lossy_image_path)
                        lossy_shape = lossy_image.header.get_data_shape()
                        target_lossy_shape = (lossy_shape[0], lossy_shape[1], lossy_shape[2])
                        
                        # Scale down the masked image to lossy resolution
                        zoom_factors = [
                            target_lossy_shape[i] / masked_full_data.shape[i] 
                            for i in range(len(target_lossy_shape))
                        ]
                        scaled_lossy_masked = zoom(masked_full_data, zoom_factors, order=1)
                        
                        # Ensure exact shape
                        if scaled_lossy_masked.shape != target_lossy_shape:
                            result = np.zeros(target_lossy_shape, dtype=scaled_lossy_masked.dtype)
                            slices = tuple(
                                slice(0, min(scaled_lossy_masked.shape[i], target_lossy_shape[i])) 
                                for i in range(len(target_lossy_shape))
                            )
                            result[slices] = scaled_lossy_masked[slices]
                            scaled_lossy_masked = result
                        
                        lossy_masked_img = nib.Nifti1Image(
                            scaled_lossy_masked,
                            lossy_image.affine,
                            header=lossy_image.header
                        )
                        nib.save(lossy_masked_img, lossy_output_path)
                        print(f"Saved lossy resolution masked image (uncompressed fallback): {lossy_output_path}")
                        saved_files.append(lossy_output_filename)
                except Exception as fallback_error:
                    print(f"Warning: Could not create lossy version even with fallback: {str(fallback_error)}")
            
            return Response(
                {
                    "message": f"Mask applied successfully and saved to {mask_folder}",
                    "mask_folder": mask_folder,
                    "saved_files": saved_files,
                    "voxels_masked": int(np.sum(binary_mask > 0)),
                    "output_path": target_subject_dir
                },
                status=status.HTTP_200_OK
            )
            
        except Exception as e:
            print(f"Error applying mask to image: {str(e)}")
            import traceback
            traceback.print_exc()
            return Response(
                {"error": f"Failed to apply mask to image: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class BatchApplyMaskToScansView(APIView):
    """
    API endpoint to apply masks with a specific label to all subjects in a directory.
    
    This function:
    1. Scans the extracted directory for all subjects
    2. For each subject, checks if they have saved masks
    3. For subjects with masks containing the specified label:
       - Loads the mask
       - Filters to only include voxels with the specified label value
       - Applies this filtered mask to the image
       - Saves to the specified mask folder
    4. Skips subjects without masks or without the specified label
    
    POST Parameters:
        - directory (str): Path to the main data directory
        - label (int): The label value to filter (0-255)
        - mask_folder (str): Target mask folder name
        - create_new (bool): Whether to create a new folder
        - dilation (int, optional): Morphological dilation of the label mask in voxels (0–100, default 0)
        - propagate_linked (bool, optional): If true, apply same mask to same-shape linked children
    
    Response:
        {
            "message": "Success message",
            "mask_folder": str,
            "label": int,
            "processed_count": int,
            "skipped_count": int,
            "processed_subjects": [list of processed subject names],
            "skipped_subjects": [list of skipped subject names with reasons]
        }
    """
    
    def post(self, request):
        directory = request.data.get('directory')
        label = request.data.get('label')
        mask_folder = request.data.get('mask_folder')
        create_new = request.data.get('create_new', False)
        dilation = _parse_mask_dilation(request.data)
        flag_filter = normalize_flag_filter_value(request.data.get('flagFilter', 'off'))
        # Linking is the opt-in: children skipped by the batch inherit the main's mask.
        propagate_linked = bool(request.data.get('propagate_linked', True))
        
        # Validate required parameters
        if not all([directory is not None, label is not None, mask_folder]):
            return Response(
                {
                    "error": "Missing required parameters: directory, label, mask_folder"
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Validate label is in valid range
        try:
            label = int(label)
            if label < 0 or label > 255:
                return Response(
                    {"error": "Label must be between 0 and 255"},
                    status=status.HTTP_400_BAD_REQUEST
                )
        except (ValueError, TypeError):
            return Response(
                {"error": "Label must be a valid integer"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            extracted_dir = os.path.join(directory, "extracted")
            
            if not os.path.exists(extracted_dir):
                return Response(
                    {"error": f"Extracted directory not found: {extracted_dir}"},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Get all subject directories
            subject_dirs = [
                d for d in os.listdir(extracted_dir)
                if os.path.isdir(os.path.join(extracted_dir, d)) and d != "project_settings.json"
            ]

            # Check if atlas directory exists alongside extracted
            atlas_dir = os.path.join(directory, "atlas")
            if os.path.exists(atlas_dir) and os.path.isdir(atlas_dir):
                subject_dirs.append("atlas")

            subject_dirs = [d for d in subject_dirs if not _scan_marked_faulty(directory, d)]

            subject_dirs, skipped_preserved_meshes = _filter_voxel_subject_dirs(directory, subject_dirs)
            if not subject_dirs:
                return Response(
                    {"error": voxel_only_all_meshes_message('Apply masks')},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            flagged_set = load_flagged_subject_names(directory)
            subject_dirs = apply_flag_filter(subject_dirs, flagged_set, flag_filter)
            subject_dirs = filter_out_linked_children(directory, subject_dirs)
            if not subject_dirs:
                return Response(
                    {"error": voxel_only_no_eligible_targets_message('Apply masks')},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            if skipped_preserved_meshes:
                print(f"Skipping preserved mesh subjects for batch apply mask: {skipped_preserved_meshes}")
            
            print(f"Found {len(subject_dirs)} subjects to process")
            
            processed_subjects = []
            skipped_subjects = []
            linked_propagation_results = []
            
            # Send initial progress update
            channel_layer = get_channel_layer()
            if channel_layer is not None:
                async_to_sync(channel_layer.group_send)(
                    'progress_group',
                    {
                        'type': 'send_progress',
                        'progress': 0,
                        'scan_name': '',
                        'total': len(subject_dirs),
                        'custom_message': f'Preparing to apply mask with label {label} to all subjects...',
                        'current': 0,
                    }
                )
            
            # Create mask folder structure once
            masks_base_dir = os.path.join(directory, "masks", mask_folder)
            if create_new:
                os.makedirs(masks_base_dir, exist_ok=True)
                print(f"Created new mask folder: {masks_base_dir}")
            
            # Copy project_settings.json if it exists
            project_settings_source = os.path.join(extracted_dir, "project_settings.json")
            project_settings_target_dir = os.path.join(masks_base_dir, "extracted")
            project_settings_target = os.path.join(project_settings_target_dir, "project_settings.json")
            
            if os.path.exists(project_settings_source) and not os.path.exists(project_settings_target):
                os.makedirs(project_settings_target_dir, exist_ok=True)
                shutil.copy2(project_settings_source, project_settings_target)
                print(f"Copied project_settings.json to mask folder")
            
            # Process each subject
            for idx, subject_name in enumerate(subject_dirs):
                try:
                    # Send progress update
                    progress = int((idx / len(subject_dirs)))
                    if channel_layer is not None:
                        async_to_sync(channel_layer.group_send)(
                            'progress_group',
                            {
                                'type': 'send_progress',
                                'progress': progress,
                                'scan_name': subject_name,
                                'total': len(subject_dirs),
                                'custom_message': f'Processing {subject_name}...',
                                'current': idx + 1,
                            }
                        )
                    subject_path = os.path.join(extracted_dir, subject_name)
                    
                    # Find the latest edit with a mask
                    # Look for files matching pattern: {subject_name}_edit_*.nii.mask.gz (full resolution only)
                    # Filter out lossy masks to ensure we work with full resolution masks
                    mask_files = [
                        f for f in os.listdir(subject_path)
                        if f.endswith('.nii.mask.gz') and f.startswith(subject_name) and '_lossy' not in f
                    ]
                    
                    if not mask_files:
                        skipped_subjects.append({
                            "subject": subject_name,
                            "reason": "No mask found"
                        })
                        print(f"Skipping {subject_name}: No mask found")
                        continue
                    
                    # Sort to get the latest edit (assumes edit numbers are sequential)
                    mask_files.sort()
                    latest_mask_file = mask_files[-1]
                    
                    # Extract edit name from mask file
                    # Pattern: {subject_name}_edit_X_suffix.nii.mask.gz -> {subject_name}_edit_X_suffix
                    edit_name = latest_mask_file.replace('.nii.mask.gz', '')
                    
                    # Load mask temporarily
                    mask_path = os.path.join(subject_path, latest_mask_file)
                    temp_mask_path = mask_path.replace('.nii.mask.gz', '_mask.nii.gz')
                    
                    os.replace(mask_path, temp_mask_path)
                    
                    try:
                        mask_img = nib.load(temp_mask_path)
                        # Round before astype to avoid truncating floats (e.g. 0.999 to 0) caused by NIfTI scl_slope headers
                        mask_data = np.round(mask_img.get_fdata()).astype(np.uint8)
                    finally:
                        os.replace(temp_mask_path, mask_path)
                    
                    # Check if the mask contains the specified label
                    unique_labels = np.unique(mask_data)
                    if label not in unique_labels:
                        skipped_subjects.append({
                            "subject": subject_name,
                            "reason": f"Label {label} not found in mask (available: {unique_labels.tolist()})"
                        })
                        print(f"Skipping {subject_name}: Label {label} not found")
                        continue
                    
                    # Create binary mask for only this label, then optional dilation
                    label_mask_bool = mask_data == label
                    if dilation > 0:
                        label_mask_bool = _dilate_binary_mask(label_mask_bool, dilation)
                    label_mask = label_mask_bool.astype(np.uint8)
                    
                    # Load the corresponding image
                    image_file = f"{edit_name}.nii.gz"
                    image_path = os.path.join(subject_path, image_file)
                    
                    if not os.path.exists(image_path):
                        skipped_subjects.append({
                            "subject": subject_name,
                            "reason": f"Image file not found: {image_file}"
                        })
                        print(f"Skipping {subject_name}: Image file not found")
                        continue
                    
                    image_img = nib.load(image_path)
                    image_data = image_img.get_fdata()
                    
                    # Apply label-specific mask
                    masked_image_data = image_data * label_mask
                    
                    # Create output directory for this subject
                    target_subject_dir = os.path.join(masks_base_dir, "extracted", subject_name)
                    os.makedirs(target_subject_dir, exist_ok=True)
                    
                    # Extract suffix from edit name
                    suffix = ""
                    suffix_pattern = f"{subject_name}_edit_\\d+(.*)$"
                    match = re.match(suffix_pattern, edit_name)
                    if match:
                        suffix = match.group(1)
                    
                    # Save as edit_0
                    output_filename = f"{subject_name}_edit_0{suffix}.nii.gz"
                    output_path = os.path.join(target_subject_dir, output_filename)
                    
                    masked_img = nib.Nifti1Image(
                        masked_image_data,
                        image_img.affine,
                        header=image_img.header
                    )
                    nib.save(masked_img, output_path)
                    
                    # Copy metadata (without lossy_compression)
                    metadata_source = os.path.join(subject_path, f"{subject_name}.json")
                    metadata_copy = {}
                    if os.path.exists(metadata_source):
                        with open(metadata_source, 'r') as f:
                            metadata = json.load(f)
                        
                        metadata_copy = metadata.copy()
                        if 'lossy_compression' in metadata_copy:
                            del metadata_copy['lossy_compression']
                        
                        metadata_target = os.path.join(target_subject_dir, f"{subject_name}.json")
                        with open(metadata_target, 'w') as f:
                            json.dump(metadata_copy, f, indent=4)
                    
                    # Create empty placeholder files
                    empty_header = nib.Nifti1Header()
                    empty_img = nib.Nifti1Image(np.zeros((1, 1, 1), dtype=np.uint8), np.eye(4), header=empty_header)
                    
                    placeholder_path = os.path.join(target_subject_dir, f"{subject_name}.nii.gz")
                    nib.save(empty_img, placeholder_path)
                    
                    # Try to create lossy version with compression
                    try:
                        lossy_filename = f"{subject_name}_lossy_edit_0{suffix}.nii.gz"
                        lossy_path = os.path.join(target_subject_dir, lossy_filename)
                        
                        voxel_size = metadata_copy.get('voxel_size', 1.0)
                        registration_tools = RegistrationTools()
                        registration_tools.save_as_lossy_nifti(
                            masked_image_data,
                            voxel_size,
                            metadata_target,
                            lossy_path
                        )
                        
                        placeholder_lossy_path = os.path.join(target_subject_dir, f"{subject_name}_lossy.nii.gz")
                        nib.save(empty_img, placeholder_lossy_path)
                    except Exception as e:
                        print(f"Warning: Could not create lossy version for {subject_name}: {str(e)}")
                    
                    # Copy elastic transformations if they exist
                    edit_number = None
                    if '_edit_' in edit_name:
                        try:
                            edit_number = int(edit_name.split('_edit_')[-1].split('_')[0])
                        except (ValueError, IndexError):
                            pass
                    
                    if edit_number is not None:
                        next_edit = edit_number + 1
                        elastic_fwd_src = os.path.join(subject_path, f"{subject_name}_edit_{next_edit}_elastic_fwd.nii.gz")
                        elastic_inv_src = os.path.join(subject_path, f"{subject_name}_edit_{next_edit}_elastic_inv.nii.gz")
                        
                        if os.path.exists(elastic_fwd_src) or os.path.exists(elastic_inv_src):
                            elastic_fwd_dst = os.path.join(target_subject_dir, f"{subject_name}_edit_1_elastic_fwd.nii.gz")
                            elastic_inv_dst = os.path.join(target_subject_dir, f"{subject_name}_edit_1_elastic_inv.nii.gz")
                            
                            if os.path.exists(elastic_fwd_src):
                                shutil.copy2(elastic_fwd_src, elastic_fwd_dst)
                            if os.path.exists(elastic_inv_src):
                                shutil.copy2(elastic_inv_src, elastic_inv_dst)

                            # The elastic edit is already in reference space, so save a
                            # masked copy directly (no transform) into the mask folder.
                            _e_img_src  = os.path.join(subject_path, f"{subject_name}_edit_{next_edit}_elastic.nii.gz")
                            _e_mask_src = os.path.join(subject_path, f"{subject_name}_edit_{next_edit}_elastic.nii.mask.gz")
                            if os.path.exists(_e_img_src):
                                try:
                                    _ei = nib.load(_e_img_src)
                                    _edata = _ei.get_fdata().copy()
                                    if os.path.exists(_e_mask_src):
                                        _e_mask_temp = _e_mask_src.replace('.nii.mask.gz', '_mask.nii.gz')
                                        os.replace(_e_mask_src, _e_mask_temp)
                                        try:
                                            _em = nib.load(_e_mask_temp)
                                            # Round before astype to avoid truncating floats
                                            _emdata = np.round(_em.get_fdata()).astype(np.uint8)
                                            if label in np.unique(_emdata):
                                                _e_lmask = (_emdata == label).astype(np.uint8)
                                                if dilation > 0:
                                                    _e_lmask = _dilate_binary_mask(
                                                        _e_lmask.astype(bool), dilation
                                                    ).astype(np.uint8)
                                                _edata = _edata * _e_lmask
                                        finally:
                                            os.replace(_e_mask_temp, _e_mask_src)
                                    _e_img_dst = os.path.join(target_subject_dir, f"{subject_name}_edit_1_elastic.nii.gz")
                                    nib.save(nib.Nifti1Image(_edata, _ei.affine, header=_ei.header), _e_img_dst)
                                    print(f"Saved masked elastic copy to mask folder: {subject_name}")
                                except Exception as _ce:
                                    print(f"Warning: could not save elastic copy for {subject_name}: {_ce}")

                    processed_subjects.append(subject_name)
                    print(f"Successfully processed {subject_name} with label {label}")
                    
                    # Propagate to same-shape linked siblings
                    if propagate_linked:
                        try:
                            siblings, sib_err = get_same_shape_siblings(directory, subject_name)
                            if sib_err:
                                print(f"Could not resolve siblings for {subject_name}: {sib_err}")
                            elif siblings:
                                print(f"Propagating mask from {subject_name} to {len(siblings)} siblings: {siblings}")
                                for sibling_name in siblings:
                                    try:
                                        # Always reuse the main's final (already dilated) label mask.
                                        # Segmenting the same label in the sibling's own mask would
                                        # keep a different set of voxels per channel.
                                        sib_label_mask = label_mask

                                        sib_edit_name = latest_edit_stem(directory, sibling_name)
                                        if sib_edit_name is None:
                                            linked_propagation_results.append({
                                                'scan_name': sibling_name,
                                                'status': 'skipped',
                                                'reason': 'latest edit is elastic'
                                            })
                                            continue
                                        
                                        # Load sibling image
                                        sib_image_file = f"{sib_edit_name}.nii.gz"
                                        sib_image_path = os.path.join(extracted_dir, sibling_name, sib_image_file)
                                        if not os.path.exists(sib_image_path):
                                            linked_propagation_results.append({
                                                'scan_name': sibling_name,
                                                'status': 'skipped',
                                                'reason': f'image file not found: {sib_image_file}'
                                            })
                                            print(f"  Sibling {sibling_name} image not found, skipping")
                                            continue
                                        
                                        sib_image_img = nib.load(sib_image_path)
                                        sib_image_data = sib_image_img.get_fdata()

                                        if sib_image_data.shape != sib_label_mask.shape:
                                            linked_propagation_results.append({
                                                'scan_name': sibling_name,
                                                'status': 'skipped',
                                                'reason': f'shape mismatch: {sib_image_data.shape} vs {sib_label_mask.shape}'
                                            })
                                            continue
                                        
                                        # Apply label mask
                                        sib_masked_data = sib_image_data * sib_label_mask
                                        
                                        # Create output directory
                                        sib_target_dir = os.path.join(masks_base_dir, "extracted", sibling_name)
                                        os.makedirs(sib_target_dir, exist_ok=True)
                                        
                                        # Extract suffix from edit name
                                        sib_suffix = ""
                                        sib_suffix_pattern = f"{sibling_name}_edit_\\d+(.*)$"
                                        sib_match = re.match(sib_suffix_pattern, sib_edit_name)
                                        if sib_match:
                                            sib_suffix = sib_match.group(1)
                                        
                                        # Save as edit_0
                                        sib_output_filename = f"{sibling_name}_edit_0{sib_suffix}.nii.gz"
                                        sib_output_path = os.path.join(sib_target_dir, sib_output_filename)
                                        sib_masked_img = nib.Nifti1Image(
                                            sib_masked_data,
                                            sib_image_img.affine,
                                            header=sib_image_img.header
                                        )
                                        nib.save(sib_masked_img, sib_output_path)
                                        
                                        # Copy metadata
                                        sib_metadata_source = os.path.join(extracted_dir, sibling_name, f"{sibling_name}.json")
                                        sib_metadata_copy = {}
                                        if os.path.exists(sib_metadata_source):
                                            with open(sib_metadata_source, 'r') as f:
                                                sib_metadata = json.load(f)
                                            sib_metadata_copy = sib_metadata.copy()
                                            if 'lossy_compression' in sib_metadata_copy:
                                                del sib_metadata_copy['lossy_compression']
                                            sib_metadata_target = os.path.join(sib_target_dir, f"{sibling_name}.json")
                                            with open(sib_metadata_target, 'w') as f:
                                                json.dump(sib_metadata_copy, f, indent=4)
                                        
                                        # Create empty placeholder files
                                        empty_header = nib.Nifti1Header()
                                        empty_img = nib.Nifti1Image(np.zeros((1, 1, 1), dtype=np.uint8), np.eye(4), header=empty_header)
                                        sib_placeholder_path = os.path.join(sib_target_dir, f"{sibling_name}.nii.gz")
                                        nib.save(empty_img, sib_placeholder_path)
                                        
                                        # Try to create lossy version
                                        try:
                                            sib_lossy_filename = f"{sibling_name}_lossy_edit_0{sib_suffix}.nii.gz"
                                            sib_lossy_path = os.path.join(sib_target_dir, sib_lossy_filename)
                                            sib_voxel_size = sib_metadata_copy.get('voxel_size', 1.0)
                                            registration_tools = RegistrationTools()
                                            registration_tools.save_as_lossy_nifti(
                                                sib_masked_data,
                                                sib_voxel_size,
                                                sib_metadata_target,
                                                sib_lossy_path
                                            )
                                            sib_placeholder_lossy = os.path.join(sib_target_dir, f"{sibling_name}_lossy.nii.gz")
                                            nib.save(empty_img, sib_placeholder_lossy)
                                        except Exception as e:
                                            print(f"  Warning: Could not create lossy version for sibling {sibling_name}: {str(e)}")
                                        
                                        linked_propagation_results.append({
                                            'scan_name': sibling_name,
                                            'status': 'success'
                                        })
                                        print(f"  Successfully propagated mask to sibling {sibling_name}")
                                        
                                    except Exception as sib_e:
                                        linked_propagation_results.append({
                                            'scan_name': sibling_name,
                                            'status': 'failed',
                                            'reason': str(sib_e)
                                        })
                                        print(f"  Error propagating to sibling {sibling_name}: {sib_e}")
                        except Exception as prop_e:
                            print(f"Error in linked propagation for {subject_name}: {prop_e}")
                    
                except Exception as e:
                    skipped_subjects.append({
                        "subject": subject_name,
                        "reason": f"Error: {str(e)}"
                    })
                    print(f"Error processing {subject_name}: {str(e)}")
                    continue
            
            # Create empty raw files in main folder for each processed subject
            for subject_name in processed_subjects:
                empty_raw_path = os.path.join(masks_base_dir, f"{subject_name}.nii.gz")
                if not os.path.exists(empty_raw_path):
                    empty_header = nib.Nifti1Header()
                    empty_img = nib.Nifti1Image(np.zeros((1, 1, 1), dtype=np.uint8), np.eye(4), header=empty_header)
                    nib.save(empty_img, empty_raw_path)
            
            # Send final progress update
            if channel_layer is not None:
                async_to_sync(channel_layer.group_send)(
                    'progress_group',
                    {
                        'type': 'send_progress',
                        'progress': 1.0,
                        'scan_name': '',
                        'total': len(subject_dirs),
                        'custom_message': f'Batch mask application completed! Applied to {len(processed_subjects)} subjects.',
                        'current': len(subject_dirs),
                    }
                )
            
            return Response(
                {
                    "message": f"Batch mask application completed for label {label}",
                    "mask_folder": mask_folder,
                    "label": label,
                    "processed_count": len(processed_subjects),
                    "skipped_count": len(skipped_subjects),
                    "processed_subjects": processed_subjects,
                    "skipped_subjects": skipped_subjects,
                    "output_path": masks_base_dir,
                    "linked_propagation": linked_propagation_results if propagate_linked else None
                },
                status=status.HTTP_200_OK
            )
            
        except Exception as e:
            print(f"Error in batch mask application: {str(e)}")
            import traceback
            traceback.print_exc()
            return Response(
                {"error": f"Failed to apply masks: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class MaskOutLabelsView(APIView):
    """
    API endpoint to mask out (delete/set to 0) voxels in the image based on specified mask labels.
    
    This function:
    1. Loads the current image edit
    2. Loads the corresponding mask
    3. For each label in the labels array, sets those voxels to 0 in the image (or inverse if invert=True)
    4. Finds the next edit number
    5. Saves as "{filename}_edit_{N}_masked.nii.gz"
    6. Creates lossy version with metadata update
    7. Copies mask to new edit with downsampling
    8. Copies landmarks (if present) without transformation
    9. Copies landmark distances (if present)
    
    POST Parameters:
        - directory (str): Path to the main data directory
        - filename (str): Subject/scan filename
        - edit (str): Current edit name (e.g., "scan_edit_2_elastic.nii.gz")
        - labels (list[int]): Array of label values to mask out (e.g., [1] or [1,2,3])
        - fullResolutionState (bool): Whether working with full or lossy resolution
        - invert (bool): If True, keep only labeled voxels; if False, delete labeled voxels (default: False)
    
    Response:
        {
            "message": str,
            "new_edit": str (full resolution name),
            "new_edit_lossy": str (lossy name),
            "labels_masked": list[int],
            "voxels_affected": int,
            "landmarks_copied": bool,
            "operation": str ("masked_out" or "isolated")
        }
    """
    
    def post(self, request):
        directory = request.data.get('directory')
        filename = request.data.get('filename')
        edit = request.data.get('edit')
        labels = request.data.get('labels', [])
        full_res = request.data.get('fullResolutionState', False)
        invert = request.data.get('invert', False)
        # Linked siblings must lose the exact voxels the operated scan lost, so they borrow
        # its label mask rather than resolving the same label in their own segmentation.
        mask_source = request.data.get('mask_source')
        
        # Default edit to filename if not provided (for raw scans)
        if not edit:
            edit = filename
        
        # Validate parameters
        if not all([directory, filename, edit]):
            return Response(
                {"error": "Missing required parameters: directory, filename, edit"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if not labels or not isinstance(labels, list):
            return Response(
                {"error": "labels must be a non-empty array"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Validate all labels are positive integers
        try:
            labels = [int(l) for l in labels]
            if any(l <= 0 for l in labels):
                return Response(
                    {"error": "All labels must be positive integers (> 0)"},
                    status=status.HTTP_400_BAD_REQUEST
                )
        except (ValueError, TypeError):
            return Response(
                {"error": "labels must contain valid integers"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # Determine paths
            if filename == "atlas":
                base_path = os.path.join(directory, "atlas")
                metadata_path = os.path.join(base_path, "atlas.json")
            else:
                base_path = os.path.join(directory, "extracted", filename)
                metadata_path = os.path.join(base_path, f"{filename}.json")
            
            # Load metadata
            if not os.path.exists(metadata_path):
                return Response(
                    {"error": f"Metadata not found: {metadata_path}"},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Process edit name
            edit_clean = edit.replace(".nii.gz", "")
            lossy_edit = edit_clean
            full_edit = edit_clean.replace('_lossy', '')
            
            # ===== ALWAYS LOAD FULL RESOLUTION MASK =====
            mask_base_path, mask_edit_stem = _resolve_mask_source(
                directory, base_path, full_edit, mask_source
            )
            full_mask_filename = f"{mask_edit_stem}.nii.mask.gz"
            full_mask_temp_filename = f"{mask_edit_stem}_mask.nii.gz"
            full_mask_path = os.path.join(mask_base_path, full_mask_filename)
            full_mask_temp_path = os.path.join(mask_base_path, full_mask_temp_filename)
            
            if not os.path.exists(full_mask_path):
                return Response(
                    {"error": f"Mask not found: {full_mask_filename}"},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Load mask with temp rename
            os.replace(full_mask_path, full_mask_temp_path)
            try:
                mask_img = nib.load(full_mask_temp_path)
                # Round before astype to avoid truncating floats
                mask_data = np.round(mask_img.get_fdata()).astype(np.uint8)
            finally:
                os.replace(full_mask_temp_path, full_mask_path)
            
            # ===== ALWAYS LOAD FULL RESOLUTION IMAGE =====
            full_image_filename = f"{full_edit}.nii.gz"
            full_image_path = os.path.join(base_path, full_image_filename)
            
            if not os.path.exists(full_image_path):
                return Response(
                    {"error": f"Full resolution image not found: {full_image_filename}"},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            full_image_img = nib.load(full_image_path)
            full_image_data = full_image_img.get_fdata()
            
            if mask_data.shape != full_image_data.shape:
                return Response(
                    {
                        "error": (
                            f"Mask shape {mask_data.shape} does not match image shape "
                            f"{full_image_data.shape} for {filename}."
                        )
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Create combined mask for all labels
            combined_mask = np.zeros_like(mask_data, dtype=bool)
            for label in labels:
                combined_mask |= (mask_data == label)
            
            voxels_affected = int(np.sum(combined_mask))
            
            if voxels_affected == 0:
                return Response(
                    {"error": f"No voxels found with labels {labels}"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Apply mask: set to 0 based on invert flag
            masked_full_data = full_image_data.copy()
            if invert:
                masked_full_data[~combined_mask] = 0
            else:
                masked_full_data[combined_mask] = 0
            
            # Find next edit number
            edit_number = 0
            while glob.glob(os.path.join(base_path, f"{filename}_edit_{edit_number}_*.nii.gz")):
                edit_number += 1
            
            # Save full resolution version
            full_output_name = f"{filename}_edit_{edit_number}_masked"
            full_output_path = os.path.join(base_path, f"{full_output_name}.nii.gz")
            
            full_masked_img = nib.Nifti1Image(
                masked_full_data,
                full_image_img.affine,
                header=full_image_img.header
            )
            nib.save(full_masked_img, full_output_path)
            print(f"Saved full resolution masked image: {full_output_path}")
            
            # Create lossy version from full resolution masked data
            lossy_output_name = f"{filename}_lossy_edit_{edit_number}_masked"
            lossy_output_path = os.path.join(base_path, f"{lossy_output_name}.nii.gz")
            
            try:
                voxel_size = metadata.get('voxel_size', 1.0)
                registration_tools = RegistrationTools()
                registration_tools.save_as_lossy_nifti(
                    masked_full_data,
                    voxel_size,
                    metadata_path,
                    lossy_output_path
                )
                print(f"Saved lossy masked image: {lossy_output_path}")
            except Exception as e:
                print(f"Warning: Could not create lossy version: {str(e)}")
            
            # Copy and downsample mask to new edit
            try:
                # Copy full resolution mask
                src_mask = os.path.join(base_path, f"{full_edit}.nii.mask.gz")
                dst_mask_full = os.path.join(base_path, f"{full_output_name}.nii.mask.gz")
                dst_mask_lossy = os.path.join(base_path, f"{lossy_output_name}.nii.mask.gz")
                
                if os.path.exists(src_mask):
                    shutil.copy2(src_mask, dst_mask_full)
                    print(f"Copied mask to: {dst_mask_full}")
                    
                    # Downsample mask for lossy version
                    resolution_factor = 2
                    lc = metadata.get('lossy_compression', None)
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
                    
                    # Load lossy image to get affine
                    lossy_img = nib.load(lossy_output_path)
                    lossy_affine = lossy_img.affine
                    
                    # Downsample mask
                    mask_temp = dst_mask_full.replace('.nii.mask.gz', '_mask.nii.gz')
                    lossy_mask_temp = dst_mask_lossy.replace('.nii.mask.gz', '_mask.nii.gz')
                    os.replace(dst_mask_full, mask_temp)
                    try:
                        mask_img = nib.load(mask_temp)
                        mask_dtype = mask_img.get_data_dtype()
                        # Round before astype to avoid truncating floats
                        mask_arr = np.round(mask_img.get_fdata()).astype(mask_dtype)
                        slices = [slice(None, None, resolution_factor) for _ in range(3)]
                        lossy_mask_arr = mask_arr[slices[0], slices[1], slices[2]].astype(mask_dtype)
                        nib.save(nib.Nifti1Image(lossy_mask_arr, lossy_affine), lossy_mask_temp)
                        os.replace(lossy_mask_temp, dst_mask_lossy)
                        print(f"Created lossy mask: {dst_mask_lossy}")
                    finally:
                        os.replace(mask_temp, dst_mask_full)
            except Exception as e:
                print(f"Warning: Could not copy/downsample mask: {str(e)}")
            
            # === Copy landmarks if they exist (WITHOUT TRANSFORMATION) ===
            landmarks_copied = False
            try:
                # Determine source base for landmark files
                source_landmark_base = full_edit
                
                # Check for landmarks file
                source_landmarks_path = os.path.join(base_path, f"{source_landmark_base}_landmarks.json")
                if os.path.isfile(source_landmarks_path):
                    # Copy landmarks directly without transformation (geometry not changed)
                    dest_landmarks_path = os.path.join(base_path, f"{full_output_name}_landmarks.json")
                    shutil.copy2(source_landmarks_path, dest_landmarks_path)
                    print(f"Copied landmarks to: {dest_landmarks_path}")
                    landmarks_copied = True
                else:
                    print(f"No landmarks found: {source_landmarks_path}")
                
                # Check for landmark distances file
                source_landmark_distances_path = os.path.join(base_path, f"{source_landmark_base}_landmark_distances.json")
                if os.path.isfile(source_landmark_distances_path):
                    dest_landmark_distances_path = os.path.join(base_path, f"{full_output_name}_landmark_distances.json")
                    shutil.copy2(source_landmark_distances_path, dest_landmark_distances_path)
                    print(f"Copied landmark distances to: {dest_landmark_distances_path}")
                else:
                    print(f"No landmark distances found: {source_landmark_distances_path}")
                    
            except Exception as lme:
                print(f"Warning: Could not copy landmarks: {str(lme)}")
                # Don't fail operation if landmarks can't be copied

            try:
                from .pipeline_log import merge_mask_workflow_metadata
                from datetime import datetime, timezone
                merge_mask_workflow_metadata(directory, filename, {
                    "application": {
                        "operation": "isolated" if invert else "masked_out",
                        "label": labels[0] if labels else None,
                        "labels": list(labels),
                        "inserted_before_elastic": False,
                        "dilation": 0,
                        "ts": datetime.now(timezone.utc).isoformat(),
                    },
                })
            except Exception as _mw_err:
                print(f"Warning: mask_workflow metadata update failed for {filename}: {_mw_err}")
            
            return Response(
                {
                    "message": f"Successfully masked out labels {labels}",
                    "new_edit": f"{full_output_name}.nii.gz",
                    "new_edit_lossy": f"{lossy_output_name}.nii.gz",
                    "labels_masked": labels,
                    "voxels_affected": voxels_affected,
                    "landmarks_copied": landmarks_copied,
                    "operation": "isolated" if invert else "masked_out"
                },
                status=status.HTTP_200_OK
            )
            
        except Exception as e:
            print(f"Error masking out labels: {str(e)}")
            import traceback
            traceback.print_exc()
            return Response(
                {"error": f"Failed to mask out labels: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class BatchMaskOutScansView(APIView):
    """
    API endpoint to mask out (delete/set to 0) voxels in all scans based on specified mask labels.
    Creates new edits in-place (no sub-folders).
    
    This function:
    1. Scans the extracted directory for all subjects
    2. For each subject, checks if they have saved masks
    3. For subjects with masks containing the specified label:
       - Loads the full resolution mask and image
       - Creates combined mask for all specified labels
       - Either deletes labeled voxels (invert=False) or keeps only labeled voxels (invert=True)
       - Saves as new edit "{subject}_edit_{N}_masked.nii.gz" in the same subject directory
       - Creates lossy version
       - Copies mask with downsampling
       - Copies landmarks if present
    4. Skips subjects without masks or without the specified label
    
    POST Parameters:
        - directory (str): Path to the main data directory
        - label (int): The label value to mask (0-255)
        - invert (bool): If True, keep only labeled voxels; if False, delete labeled voxels (default: False)
        - dilation (int, optional): Morphological dilation of the label mask in voxels (0–100, default 0)
        - overwrite (bool): If True, overwrite an existing masked-before-elastic insertion using the
          edit prior to the masked slot as source; if False, skip subjects where the insertion already
          exists (default: False)
    
    Response:
        {
            "message": "Success message",
            "label": int,
            "operation": str ("masked_out" or "isolated"),
            "processed_count": int,
            "skipped_count": int,
            "processed_subjects": [list of processed subject names],
            "skipped_subjects": [list of skipped subject names with reasons]
        }
    """
    
    def post(self, request):
        directory = request.data.get('directory')
        label = request.data.get('label')
        invert = request.data.get('invert', False)
        overwrite = request.data.get('overwrite', False)
        dilation = _parse_mask_dilation(request.data)
        flag_filter = normalize_flag_filter_value(request.data.get('flagFilter', 'off'))
        
        # Validate required parameters
        if not all([directory is not None, label is not None]):
            return Response(
                {"error": "Missing required parameters: directory, label"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Validate label is in valid range
        try:
            label = int(label)
            if label < 0 or label > 255:
                return Response(
                    {"error": "Label must be between 0 and 255"},
                    status=status.HTTP_400_BAD_REQUEST
                )
        except (ValueError, TypeError):
            return Response(
                {"error": "Label must be a valid integer"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            extracted_dir = os.path.join(directory, "extracted")
            
            if not os.path.exists(extracted_dir):
                return Response(
                    {"error": f"Extracted directory not found: {extracted_dir}"},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Get all subject directories
            subject_dirs = [
                d for d in os.listdir(extracted_dir)
                if os.path.isdir(os.path.join(extracted_dir, d)) and d != "project_settings.json"
            ]

            # Check if atlas directory exists alongside extracted
            atlas_dir = os.path.join(directory, "atlas")
            if os.path.exists(atlas_dir) and os.path.isdir(atlas_dir):
                subject_dirs.append("atlas")

            subject_dirs = [d for d in subject_dirs if not _scan_marked_faulty(directory, d)]

            subject_dirs, skipped_preserved_meshes = _filter_voxel_subject_dirs(directory, subject_dirs)
            if not subject_dirs:
                return Response(
                    {"error": voxel_only_all_meshes_message('Apply masks')},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            flagged_set = load_flagged_subject_names(directory)
            subject_dirs = apply_flag_filter(subject_dirs, flagged_set, flag_filter)
            subject_dirs = filter_out_linked_children(directory, subject_dirs)
            if not subject_dirs:
                return Response(
                    {"error": voxel_only_no_eligible_targets_message('Apply masks')},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            if skipped_preserved_meshes:
                print(f"Skipping preserved mesh subjects for batch mask out: {skipped_preserved_meshes}")
            
            print(f"Found {len(subject_dirs)} subjects to process")
            
            processed_subjects = []
            skipped_subjects = []
            
            # Send initial progress update
            channel_layer = get_channel_layer()
            if channel_layer is not None:
                async_to_sync(channel_layer.group_send)(
                    'progress_group',
                    {
                        'type': 'send_progress',
                        'progress': 0,
                        'scan_name': '',
                        'total': len(subject_dirs),
                        'custom_message': f'Preparing to mask out label {label} in all subjects...',
                        'current': 0,
                    }
                )
            
            # Process each subject
            for idx, subject_name in enumerate(subject_dirs):
                try:
                    # Send progress update
                    progress = idx / len(subject_dirs)
                    if channel_layer is not None:
                        async_to_sync(channel_layer.group_send)(
                            'progress_group',
                            {
                                'type': 'send_progress',
                                'progress': progress,
                                'scan_name': subject_name,
                                'total': len(subject_dirs),
                                'custom_message': f'Processing {subject_name}...',
                                'current': idx + 1,
                            }
                        )
                    
                    # Determine paths
                    if subject_name == "atlas":
                        subject_path = os.path.join(directory, "atlas")
                        metadata_path = os.path.join(subject_path, "atlas.json")
                    else:
                        subject_path = os.path.join(extracted_dir, subject_name)
                        metadata_path = os.path.join(subject_path, f"{subject_name}.json")
                    
                    # Load metadata
                    if not os.path.exists(metadata_path):
                        skipped_subjects.append({
                            "subject": subject_name,
                            "reason": "Metadata not found"
                        })
                        print(f"Skipping {subject_name}: Metadata not found")
                        continue
                    
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # --- Step 1: Determine next free edit slot ---
                    edit_number = 0
                    while glob.glob(os.path.join(subject_path, f"{subject_name}_edit_{edit_number}_*.nii.gz")):
                        edit_number += 1

                    # --- Step 2: Detect if latest edit is elastic ---
                    elastic_edit_num = edit_number - 1
                    elastic_files_to_bump = (
                        glob.glob(os.path.join(subject_path, f"{subject_name}_edit_{elastic_edit_num}_elastic*")) +
                        glob.glob(os.path.join(subject_path, f"{subject_name}_lossy_edit_{elastic_edit_num}_elastic*"))
                    ) if elastic_edit_num >= 0 else []

                    # --- Step 3: Detect if a masked insertion already exists before the elastic ---
                    # After a previous run the layout is: ..._edit_{E-1}_*, _edit_{E}_masked, _edit_{E+1}_elastic
                    # so pre_masked_edit_num == E and source_for_overwrite == E-1.
                    already_inserted = False
                    pre_masked_edit_num = elastic_edit_num - 1
                    if elastic_files_to_bump and pre_masked_edit_num >= 0:
                        already_inserted = bool(glob.glob(
                            os.path.join(subject_path, f"{subject_name}_edit_{pre_masked_edit_num}_*masked*.nii.gz")
                        ))

                    # Also detect when the last edit is already masked with no elastic following it.
                    last_edit_is_masked = (
                        not elastic_files_to_bump and edit_number > 0 and
                        bool(glob.glob(
                            os.path.join(subject_path, f"{subject_name}_edit_{edit_number - 1}_*masked*.nii.gz")
                        ))
                    )

                    # --- Step 4: Skip or redirect based on insertion state ---
                    if not overwrite and (already_inserted or last_edit_is_masked):
                        reason = (
                            "Masked edit already inserted before elastic (pass overwrite=True to replace)"
                            if already_inserted
                            else "Last edit is already masked (pass overwrite=True to replace)"
                        )
                        skipped_subjects.append({"subject": subject_name, "reason": reason})
                        print(f"Skipping {subject_name}: {reason}")
                        continue

                    # --- Step 5: Determine source mask file ---
                    def _source_mask_candidates(source_edit_num):
                        if source_edit_num < 0:
                            raw_mask = f"{subject_name}.nii.mask.gz"
                            return [raw_mask] if os.path.exists(os.path.join(subject_path, raw_mask)) else []
                        return sorted([
                            f for f in os.listdir(subject_path)
                            if f.endswith('.nii.mask.gz') and f.startswith(subject_name)
                            and '_lossy' not in f and f'_edit_{source_edit_num}_' in f
                        ])

                    if already_inserted:
                        # Overwrite path: source is the edit immediately before the existing masked slot
                        source_edit_num = pre_masked_edit_num - 1
                        source_mask_candidates = _source_mask_candidates(source_edit_num)
                        if not source_mask_candidates:
                            skipped_subjects.append({
                                "subject": subject_name,
                                "reason": f"Source mask not found at edit {source_edit_num} for overwrite"
                            })
                            print(f"Skipping {subject_name}: source mask not found at edit {source_edit_num}")
                            continue
                        latest_mask_file = source_mask_candidates[-1]
                        masked_edit_number = pre_masked_edit_num
                    elif elastic_files_to_bump:
                        # First insertion before elastic: source must be the edit immediately
                        # before the elastic, not the current elastic mask that sorts latest.
                        source_edit_num = elastic_edit_num - 1
                        source_mask_candidates = _source_mask_candidates(source_edit_num)
                        if not source_mask_candidates:
                            skipped_subjects.append({
                                "subject": subject_name,
                                "reason": f"Source mask not found at edit {source_edit_num} before elastic"
                            })
                            print(f"Skipping {subject_name}: source mask not found at edit {source_edit_num} before elastic")
                            continue
                        latest_mask_file = source_mask_candidates[-1]
                        masked_edit_number = None  # resolved after bumping below
                    else:
                        # Normal path: use the latest mask in the directory
                        mask_files = [
                            f for f in os.listdir(subject_path)
                            if f.endswith('.nii.mask.gz') and f.startswith(subject_name) and '_lossy' not in f
                        ]
                        if not mask_files:
                            skipped_subjects.append({
                                "subject": subject_name,
                                "reason": "No mask found"
                            })
                            print(f"Skipping {subject_name}: No mask found")
                            continue
                        mask_files.sort()
                        latest_mask_file = mask_files[-1]
                        masked_edit_number = None  # resolved after bumping below

                    # Extract source edit name from the chosen mask file
                    edit_name = latest_mask_file.replace('.nii.mask.gz', '')

                    # --- Step 6: Load mask temporarily ---
                    mask_path = os.path.join(subject_path, latest_mask_file)
                    temp_mask_path = mask_path.replace('.nii.mask.gz', '_mask.nii.gz')

                    os.replace(mask_path, temp_mask_path)
                    try:
                        mask_img = nib.load(temp_mask_path)
                        # Round before astype to avoid truncating floats (e.g. 0.999 to 0) caused by NIfTI scl_slope headers
                        mask_data = np.round(mask_img.get_fdata()).astype(np.uint8)
                    finally:
                        os.replace(temp_mask_path, mask_path)

                    # Check if the mask contains the specified label
                    unique_labels = np.unique(mask_data)
                    if label not in unique_labels:
                        skipped_subjects.append({
                            "subject": subject_name,
                            "reason": f"Label {label} not found in mask"
                        })
                        print(f"Skipping {subject_name}: Label {label} not found")
                        continue

                    # Create combined mask for this label, then optional dilation
                    combined_mask = (mask_data == label).astype(bool)
                    if dilation > 0:
                        combined_mask = _dilate_binary_mask(combined_mask, dilation)
                    voxels_affected = int(np.sum(combined_mask))

                    # --- Step 7: Load source image (full resolution) ---
                    image_file = f"{edit_name}.nii.gz"
                    image_path = os.path.join(subject_path, image_file)

                    if not os.path.exists(image_path):
                        skipped_subjects.append({
                            "subject": subject_name,
                            "reason": f"Image file not found: {image_file}"
                        })
                        print(f"Skipping {subject_name}: Image file not found")
                        continue

                    image_img = nib.load(image_path)
                    image_data = image_img.get_fdata()

                    # Apply mask based on invert flag
                    masked_image_data = image_data.copy()
                    if invert:
                        masked_image_data[~combined_mask] = 0
                    else:
                        masked_image_data[combined_mask] = 0

                    # --- Step 8: Resolve output slot and prepare filesystem ---
                    if already_inserted:
                        # Overwrite: strip stale lossy_compression entries for the old masked
                        # files before removing them (mirrors DeleteEditView behaviour).
                        try:
                            with open(metadata_path, 'r') as f:
                                _meta = json.load(f)
                            _lc = _meta.get('lossy_compression', [])
                            if isinstance(_lc, dict):
                                _lc = [_lc]
                            _lc = [e for e in _lc
                                   if f'_edit_{masked_edit_number}_masked' not in e.get('filename', '')]
                            _meta['lossy_compression'] = _lc
                            with open(metadata_path, 'w') as f:
                                json.dump(_meta, f, indent=4)
                        except Exception as _e:
                            print(f"Warning: could not clean lossy_compression before overwrite for {subject_name}: {_e}")

                        # Remove all existing masked files at the insertion slot
                        for old_f in (
                            glob.glob(os.path.join(subject_path, f"{subject_name}_edit_{masked_edit_number}_masked*")) +
                            glob.glob(os.path.join(subject_path, f"{subject_name}_lossy_edit_{masked_edit_number}_masked*"))
                        ):
                            try:
                                os.remove(old_f)
                            except OSError as rm_err:
                                print(f"Warning: could not remove {old_f}: {rm_err}")
                        print(f"Overwriting masked edit at slot {masked_edit_number} for {subject_name}")
                    elif elastic_files_to_bump:
                        # First insertion: bump elastic files from E to E+1, occupy slot E
                        for src in elastic_files_to_bump:
                            dst = src.replace(
                                f"_edit_{elastic_edit_num}_elastic",
                                f"_edit_{elastic_edit_num + 1}_elastic",
                            )
                            os.rename(src, dst)
                        masked_edit_number = elastic_edit_num
                        print(f"Inserted masked edit before elastic for {subject_name}: "
                              f"elastic bumped {elastic_edit_num} → {elastic_edit_num + 1}")

                        # Backup the bumped elastic images (full-res and lossy) and apply the
                        # same mask operation in-place using the elastic's own mask, so the
                        # elastic edit stays consistent with the masked pre-elastic edit.
                        new_elastic_num = elastic_edit_num + 1
                        for _e_stem in [
                            f"{subject_name}_edit_{new_elastic_num}_elastic",
                            f"{subject_name}_lossy_edit_{new_elastic_num}_elastic",
                        ]:
                            _e_img_path    = os.path.join(subject_path, f"{_e_stem}.nii.gz")
                            _e_backup_path = os.path.join(subject_path, f"{_e_stem}.nii.backup.gz")
                            _e_mask_path   = os.path.join(subject_path, f"{_e_stem}.nii.mask.gz")
                            if not os.path.exists(_e_img_path):
                                continue
                            try:
                                shutil.copy2(_e_img_path, _e_backup_path)
                                # Load the elastic's own mask and check for the label
                                _e_combined = None
                                if os.path.exists(_e_mask_path):
                                    _e_mask_temp = _e_mask_path.replace('.nii.mask.gz', '_mask.nii.gz')
                                    os.replace(_e_mask_path, _e_mask_temp)
                                    try:
                                        _em = nib.load(_e_mask_temp)
                                        # Round before astype to avoid truncating floats
                                        _ed = np.round(_em.get_fdata()).astype(np.uint8)
                                        if label in np.unique(_ed):
                                            _e_combined = (_ed == label).astype(bool)
                                            if dilation > 0:
                                                _e_combined = _dilate_binary_mask(_e_combined, dilation)
                                    finally:
                                        os.replace(_e_mask_temp, _e_mask_path)
                                if _e_combined is not None:
                                    _ei = nib.load(_e_img_path)
                                    _edata = _ei.get_fdata().copy()
                                    if invert:
                                        _edata[~_e_combined] = 0
                                    else:
                                        _edata[_e_combined] = 0
                                    nib.save(
                                        nib.Nifti1Image(_edata, _ei.affine, header=_ei.header),
                                        _e_img_path,
                                    )
                                    print(f"Backed up and masked elastic ({_e_stem}) for {subject_name}")
                                else:
                                    print(f"Backed up elastic ({_e_stem}); label not in elastic mask — image unchanged")
                            except Exception as _be:
                                print(f"Warning: could not backup/mask elastic ({_e_stem}) for {subject_name}: {_be}")
                    else:
                        masked_edit_number = edit_number

                    # Save full resolution version
                    full_output_name = f"{subject_name}_edit_{masked_edit_number}_masked"
                    full_output_path = os.path.join(subject_path, f"{full_output_name}.nii.gz")
                    
                    full_masked_img = nib.Nifti1Image(
                        masked_image_data,
                        image_img.affine,
                        header=image_img.header
                    )
                    nib.save(full_masked_img, full_output_path)
                    print(f"Saved masked image for {subject_name}: {full_output_path}")
                    
                    # Create lossy version from full resolution masked data
                    lossy_output_name = f"{subject_name}_lossy_edit_{masked_edit_number}_masked"
                    lossy_output_path = os.path.join(subject_path, f"{lossy_output_name}.nii.gz")
                    
                    try:
                        voxel_size = metadata.get('voxel_size', 1.0)
                        registration_tools = RegistrationTools()
                        registration_tools.save_as_lossy_nifti(
                            masked_image_data,
                            voxel_size,
                            metadata_path,
                            lossy_output_path
                        )
                        print(f"Saved lossy masked image for {subject_name}")
                    except Exception as e:
                        print(f"Warning: Could not create lossy version for {subject_name}: {str(e)}")

                    # After save_as_lossy_nifti has appended the masked entry, fix the elastic
                    # lossy_compression entry (stale name from the bump) and re-sort by edit
                    # number so the list order matches the physical file sequence.
                    if elastic_files_to_bump and not already_inserted:
                        try:
                            with open(metadata_path, 'r') as f:
                                _meta = json.load(f)
                            _lc = _meta.get('lossy_compression', [])
                            if isinstance(_lc, dict):
                                _lc = [_lc]
                            for _entry in _lc:
                                _fn = _entry.get('filename', '')
                                if f'_edit_{elastic_edit_num}_elastic' in _fn:
                                    _entry['filename'] = _fn.replace(
                                        f'_edit_{elastic_edit_num}_elastic',
                                        f'_edit_{elastic_edit_num + 1}_elastic',
                                    )
                            def _lc_sort_key(e):
                                m = re.search(r'_edit_(\d+)_', e.get('filename', ''))
                                return int(m.group(1)) if m else 0
                            _lc.sort(key=_lc_sort_key)
                            _meta['lossy_compression'] = _lc
                            with open(metadata_path, 'w') as f:
                                json.dump(_meta, f, indent=4)
                        except Exception as _e:
                            print(f"Warning: could not fix lossy_compression after elastic bump for {subject_name}: {_e}")

                    # Copy and downsample mask to new edit
                    try:
                        src_mask = os.path.join(subject_path, f"{edit_name}.nii.mask.gz")
                        dst_mask_full = os.path.join(subject_path, f"{full_output_name}.nii.mask.gz")
                        dst_mask_lossy = os.path.join(subject_path, f"{lossy_output_name}.nii.mask.gz")
                        
                        if os.path.exists(src_mask):
                            shutil.copy2(src_mask, dst_mask_full)
                            
                            # Downsample mask for lossy version
                            resolution_factor = 2
                            lc = metadata.get('lossy_compression', None)
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
                            
                            # Load lossy image to get affine
                            lossy_img = nib.load(lossy_output_path)
                            lossy_affine = lossy_img.affine
                            
                            # Downsample mask
                            mask_temp = dst_mask_full.replace('.nii.mask.gz', '_mask.nii.gz')
                            lossy_mask_temp = dst_mask_lossy.replace('.nii.mask.gz', '_mask.nii.gz')
                            os.replace(dst_mask_full, mask_temp)
                            try:
                                mask_img = nib.load(mask_temp)
                                mask_dtype = mask_img.get_data_dtype()
                                # Round before astype to avoid truncating floats
                                mask_arr = np.round(mask_img.get_fdata()).astype(mask_dtype)
                                slices = [slice(None, None, resolution_factor) for _ in range(3)]
                                lossy_mask_arr = mask_arr[slices[0], slices[1], slices[2]].astype(mask_dtype)
                                nib.save(nib.Nifti1Image(lossy_mask_arr, lossy_affine), lossy_mask_temp)
                                os.replace(lossy_mask_temp, dst_mask_lossy)
                            finally:
                                os.replace(mask_temp, dst_mask_full)
                    except Exception as e:
                        print(f"Warning: Could not copy/downsample mask for {subject_name}: {str(e)}")
                    
                    # Copy landmarks if they exist
                    try:
                        source_landmark_base = edit_name
                        source_landmarks_path = os.path.join(subject_path, f"{source_landmark_base}_landmarks.json")
                        if os.path.isfile(source_landmarks_path):
                            dest_landmarks_path = os.path.join(subject_path, f"{full_output_name}_landmarks.json")
                            shutil.copy2(source_landmarks_path, dest_landmarks_path)
                            print(f"Copied landmarks for {subject_name}")
                        
                        source_landmark_distances_path = os.path.join(subject_path, f"{source_landmark_base}_landmark_distances.json")
                        if os.path.isfile(source_landmark_distances_path):
                            dest_landmark_distances_path = os.path.join(subject_path, f"{full_output_name}_landmark_distances.json")
                            shutil.copy2(source_landmark_distances_path, dest_landmark_distances_path)
                    except Exception as e:
                        print(f"Warning: Could not copy landmarks for {subject_name}: {str(e)}")
                    
                    processed_subjects.append(subject_name)
                    print(f"Successfully masked {subject_name} with label {label}")

                    try:
                        from .pipeline_log import merge_mask_workflow_metadata
                        from datetime import datetime, timezone
                        inserted = bool(elastic_files_to_bump) or bool(already_inserted)
                        merge_mask_workflow_metadata(directory, subject_name, {
                            "application": {
                                "operation": "isolated" if invert else "masked_out",
                                "label": label,
                                "labels": [label],
                                "inserted_before_elastic": inserted,
                                "dilation": int(dilation) if dilation else 0,
                                "ts": datetime.now(timezone.utc).isoformat(),
                            },
                        })
                    except Exception as _mw_err:
                        print(f"Warning: mask_workflow metadata update failed for {subject_name}: {_mw_err}")
                    
                except Exception as e:
                    skipped_subjects.append({
                        "subject": subject_name,
                        "reason": f"Error: {str(e)}"
                    })
                    print(f"Error processing {subject_name}: {str(e)}")
                    continue
            
            # Send final progress update
            if channel_layer is not None:
                async_to_sync(channel_layer.group_send)(
                    'progress_group',
                    {
                        'type': 'send_progress',
                        'progress': 1.0,
                        'scan_name': '',
                        'total': len(subject_dirs),
                        'custom_message': f'Completed masking {len(processed_subjects)} subjects',
                        'current': len(subject_dirs),
                    }
                )

            return Response(
                {
                    "message": f"Batch mask out completed",
                    "label": label,
                    "operation": "isolated" if invert else "masked_out",
                    "processed_count": len(processed_subjects),
                    "skipped_count": len(skipped_subjects),
                    "processed_subjects": processed_subjects,
                    "skipped_subjects": skipped_subjects
                },
                status=status.HTTP_200_OK
            )
            
        except Exception as e:
            print(f"Error in batch mask out: {str(e)}")
            import traceback
            traceback.print_exc()
            return Response(
                {"error": f"Failed to mask out scans: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
