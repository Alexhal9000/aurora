from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import json
import os
import glob
import numpy as np
import nibabel as nib
import time
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
import gc
from skimage import exposure
import shutil

from .registrationTools import RegistrationTools
from .preprocessingTools import has_elastic_registration, build_restore_settings
from .batch_flag_filter import load_flagged_subject_names, apply_flag_filter, normalize_flag_filter_value
from .linkedScans import filter_out_linked_children
from .coordinateFrames import (
    is_preserved_mesh_metadata,
    load_extracted_scan_metadata,
    partition_voxel_and_preserved_mesh_scans,
    voxel_only_all_meshes_message,
    voxel_only_no_eligible_targets_message,
)


class RestoreAllScansView(APIView):
    """
    View for applying restoration algorithms to all scans in a directory.
    Currently supports CLAHE (Contrast Limited Adaptive Histogram Equalization).
    """
    
    def clahe_equalize(self, scan_data, kernel_size, clip_limit, nbins):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        # Normalize data to 0-1 range
        data_min, data_max = scan_data.min(), scan_data.max()
        data_normalized = (scan_data.astype(np.float32) - data_min) / (data_max - data_min + 1e-8)
        
        # kernel_size=None means auto (1/8 of each dimension)
        kernel_size_param = None if kernel_size == 0 else int(kernel_size)
        
        # Apply CLAHE
        restored = exposure.equalize_adapthist(
            data_normalized,
            kernel_size=kernel_size_param,
            clip_limit=clip_limit,
            nbins=int(nbins)
        )
        
        # Scale back to original intensity range
        return (restored * (data_max - data_min) + data_min).astype(scan_data.dtype)
    
    def post(self, request):
        print("Restoring all scans")
        directory = request.data.get('directory')
        algorithm = request.data.get('algorithm', 'CLAHE')
        parameters = request.data.get('parameters', {})
        onlyCurrentScan = request.data.get('onlyCurrentScan', False)
        selectedScan = request.data.get('selectedScan', None)
        prefer_mask = request.data.get('prefer_mask', False)
        mask_label = request.data.get('mask_label', 1)
        flag_filter = normalize_flag_filter_value(
            request.data.get('flagFilter', 'off') if request.data else 'off',
            only_current_scan=onlyCurrentScan,
        )
        
        print(f"Algorithm: {algorithm}")
        print(f"Parameters: {parameters}")
        print(f"Only Current Scan: {onlyCurrentScan}")
        print(f"Selected Scan: {selectedScan}")
        print(f"Prefer Mask: {prefer_mask}, Label: {mask_label}")
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
                        'Restore only works with voxel-based volumes.'
                    ),
                }, status=status.HTTP_400_BAD_REQUEST)

        scan_names, skipped_preserved_meshes = partition_voxel_and_preserved_mesh_scans(directory, scan_names)
        if not scan_names:
            return Response({
                'status': 'error',
                'message': voxel_only_all_meshes_message('Restore'),
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # If only restoring current scan, filter to that specific scan
        if onlyCurrentScan and selectedScan:
            if selectedScan in scan_names:
                scan_names = [selectedScan]
            else:
                return Response({
                    'status': 'error',
                    'message': f'Selected scan "{selectedScan}" not found in directory'
                }, status=status.HTTP_400_BAD_REQUEST)
        
        # Skip if the latest scan edit already has restoration applied
        scan_names_to_remove = []
        for scan_name in scan_names:
            latest_edit = 0
            while glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{latest_edit}_*.nii.gz")):
                latest_edit += 1
            latest_edit -= 1
            if latest_edit >= 0:
                edit_files = glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{latest_edit}_*.nii.gz"))
                if edit_files and "_restored" in edit_files[0]:
                    scan_names_to_remove.append(scan_name)
        
        for scan_name in scan_names_to_remove:
            scan_names.remove(scan_name)
        
        scan_names.sort()

        flagged_set = load_flagged_subject_names(directory)
        scan_names = apply_flag_filter(scan_names, flagged_set, flag_filter)
        if not (onlyCurrentScan and selectedScan):
            scan_names = filter_out_linked_children(directory, scan_names)
        if not scan_names:
            return Response({
                'status': 'error',
                'message': voxel_only_no_eligible_targets_message('Restore'),
            }, status=status.HTTP_400_BAD_REQUEST)
        if skipped_preserved_meshes:
            print(f"Skipping preserved mesh subjects for restoration: {skipped_preserved_meshes}")
        
        total_scans = len(scan_names)
        channel_layer = get_channel_layer()
        
        if channel_layer is not None and total_scans > 0:
            async_to_sync(channel_layer.group_send)(
                'progress_group',
                {
                    'type': 'send_progress',
                    'progress': 0,
                    'scan_name': scan_names[0] if scan_names else 'N/A',
                    'custom_message': f'Restoring all scans using {algorithm}...',
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
                            'custom_message': f'Restoring {scan_name} using {algorithm}...',
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
                
                # Apply restoration algorithm
                time_start = time.time()
                if algorithm == 'CLAHE':
                    restored_data = self.clahe_equalize(
                        scan_data,
                        parameters.get('kernel_size', 0),
                        parameters.get('clip_limit', 0.01),
                        parameters.get('nbins', 256)
                    )
                    
                    if prefer_mask:
                        # Try to find exactly the mask file corresponding to the latest edit we loaded
                        # If latest_edit is e.g. 4, scan_path is likely ..._edit_4_something.nii.gz
                        # We want the mask ..._edit_4_something.nii.mask.gz
                        base_scan_name = os.path.basename(scan_path).replace('.nii.gz', '')
                        specific_mask_path = os.path.join(directory, "extracted", scan_name, f"{base_scan_name}.nii.mask.gz")
                        
                        if os.path.exists(specific_mask_path):
                            src_mask_full = [specific_mask_path]
                        else:
                            src_mask_full = glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{latest_edit}_*.nii.mask.gz"))
                            # Filter out lossy masks just in case
                            src_mask_full = [f for f in src_mask_full if "lossy" not in os.path.basename(f)]
                            
                            if not src_mask_full:
                                src_mask_full = glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}.nii.mask.gz"))
                                src_mask_full = [f for f in src_mask_full if "lossy" not in os.path.basename(f)]
                        
                        if src_mask_full:
                            try:
                                mask_path = src_mask_full[0]
                                temp_mask_path = mask_path.replace('.nii.mask.gz', '_mask_temp.nii.gz')
                                
                                # Rename temporarily to load with nibabel
                                os.replace(mask_path, temp_mask_path)
                                try:
                                    mask_img = nib.load(temp_mask_path)
                                    mask_data = mask_img.get_fdata()
                                    binary_mask = (mask_data == mask_label)
                                    restored_data = np.where(binary_mask, restored_data, scan_data)
                                finally:
                                    # Always restore the original name
                                    os.replace(temp_mask_path, mask_path)
                            except Exception as e:
                                print(f"Warning: Could not load mask for {scan_name}, proceeding without mask. Error: {e}")
                else:
                    raise ValueError(f"Unknown algorithm: {algorithm}")
                
                time_end = time.time()
                print(f"Time taken to restore {scan_name}: {time_end - time_start} seconds")
                
                # Determine the next edit number
                new_edit_number = latest_edit + 1
                
                # Save the restored scan
                output_path = os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{new_edit_number}_restored.nii.gz")
                output_img = nib.Nifti1Image(restored_data.astype(original_dtype), scan_affine)
                nib.save(output_img, output_path)
                print(f"Saved restored scan to: {output_path}")
                
                # Save lossy version using proper compression method
                lossy_path = os.path.join(directory, "extracted", scan_name, f"{scan_name}_lossy_edit_{new_edit_number}_restored.nii.gz")
                registration_tools = RegistrationTools()
                json_path = os.path.join(directory, "extracted", scan_name, f"{scan_name}.json")
                registration_tools.save_as_lossy_nifti(
                    restored_data.astype(original_dtype),
                    scan_metadata['voxel_size'],
                    json_path,
                    lossy_path
                )
                print(f"Saved lossy version to: {lossy_path}")

                # Persist restore provenance on subject JSON (forensic scientific report)
                try:
                    with open(json_path, 'r') as jf:
                        _meta = json.load(jf)
                    _meta['restore_settings'] = build_restore_settings(
                        algorithm,
                        parameters,
                        prefer_mask=prefer_mask,
                        mask_label=mask_label,
                    )
                    with open(json_path, 'w') as jf:
                        json.dump(_meta, jf, indent=4)
                except Exception as meta_err:
                    print(f"  Warning: could not save restore_settings: {meta_err}")
                
                # Copy mask files if they exist
                src_mask_full = glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{latest_edit}_*.nii.mask.gz"))
                src_mask_lossy = glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_lossy_edit_{latest_edit}_*.nii.mask.gz"))
                if src_mask_full:
                    dst_mask_full = os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{new_edit_number}_restored.nii.mask.gz")
                    shutil.copy(src_mask_full[0], dst_mask_full)

                if src_mask_lossy:
                    dst_mask_lossy = os.path.join(directory, "extracted", scan_name, f"{scan_name}_lossy_edit_{new_edit_number}_restored.nii.mask.gz")
                    shutil.copy(src_mask_lossy[0], dst_mask_lossy)
                
                # Copy paired landmark files (non-geometry change) if they exist
                try:
                    if latest_edit >= 0:
                        source_landmark_base = glob.glob(os.path.join(directory, "extracted", scan_name, f"{scan_name}_edit_{latest_edit}_*.nii.gz"))[0].replace('.nii.gz', '')
                    else:
                        source_landmark_base = scan_name
                    
                    source_landmarks_path = os.path.join(directory, "extracted", scan_name, f"{source_landmark_base}_landmarks.json")
                    source_distances_path = os.path.join(directory, "extracted", scan_name, f"{source_landmark_base}_landmark_distances.json")
                    
                    dest_landmark_base = f"{scan_name}_edit_{new_edit_number}_restored"
                    dest_landmarks_path = os.path.join(directory, "extracted", scan_name, f"{dest_landmark_base}_landmarks.json")
                    dest_distances_path = os.path.join(directory, "extracted", scan_name, f"{dest_landmark_base}_landmark_distances.json")
                    
                    if os.path.isfile(source_landmarks_path):
                        shutil.copyfile(source_landmarks_path, dest_landmarks_path)
                        print(f"  Copied landmarks to: {dest_landmarks_path}")
                    
                    if os.path.isfile(source_distances_path):
                        shutil.copyfile(source_distances_path, dest_distances_path)
                        print(f"  Copied landmark distances to: {dest_distances_path}")
                except Exception as le:
                    print(f"  Error handling paired landmark files (restored): {le}")

                updated_scans.append(scan_name)
                
                # Clean up
                del scan_data, restored_data, nifti_img
                gc.collect()
                
            except Exception as e:
                error_msg = f"Error processing {scan_name}: {str(e)}"
                errors.append(error_msg)
                print(error_msg)
                import traceback
                traceback.print_exc()
        
        response_data = {
            'status': 'success',
            'message': f'Restored {len(updated_scans)} scans using {algorithm}',
            'updated_scans': updated_scans
        }
        
        if errors:
            response_data['errors'] = errors
            response_data['message'] += f' with {len(errors)} errors'
        
        return Response(response_data, status=status.HTTP_200_OK)
