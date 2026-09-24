import os
import json
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response




# ==================== AI Segmentation Function ====================

def ai_segmentation(image_data_3d, mask_data_3d, target_label, model):
    """
    Placeholder function for AI segmentation.

    This is where the actual AI model will be integrated later.

    Args:
        image_data_3d (np.ndarray): 3D numpy array of the medical image to be segmented
                                    Shape: (width, height, depth)
                                    dtype: typically uint8 or float32

        mask_data_3d (np.ndarray): 3D numpy array of the current mask (already loaded from .nii.mask.gz)
                                   Shape: (width, height, depth)
                                   dtype: uint8 (contains label numbers, 0 = background)

        target_label (int): The label number that the user is requesting to segment
                           Range: 1-255 (0 is reserved for background)

        model (str): The name of the AI model to use

    Returns:
        np.ndarray: 3D numpy array of the same shape as input arrays containing predicted segmentation
                    The values at target_label positions will contain the label number
                    Other positions can be 0 (will only affect the target_label voxels in merge)

    Raises:
        Exception: Any error during AI processing should be caught and re-raised with context

    Example Usage:
        result_mask = ai_segmentation(image_3d, current_mask_3d, label_num, model)
    """
    print(f"ai_segmentation() called with:")
    print(f"  image_data_3d shape: {image_data_3d.shape}, dtype: {image_data_3d.dtype}")
    print(f"  mask_data_3d shape: {mask_data_3d.shape}, dtype: {mask_data_3d.dtype}")
    print(f"  target_label: {target_label}")
    print(f"  model: {model}")

    try:
        from .aiModels.registry import run_inference
        print(f"Dispatching AI segmentation through model registry: {model}")
        result_mask = run_inference(image_data_3d, mask_data_3d, target_label, model)

        print(f"AI segmentation completed successfully using {model}")
        print(f"Result mask shape: {result_mask.shape}")

        return result_mask

    except Exception as e:
        from .aiModels.foundation_models import ModelMissingError
        print(f"Error in AI segmentation: {str(e)}")
        if isinstance(e, ModelMissingError):
            raise
        raise Exception(f"AI segmentation failed: {str(e)}")

def medsam2_segmentation(image_data_3d, mask_data_3d, target_label):
    """
    Run MedSAM2 segmentation on the given image and mask.

    Args:
        image_data_3d (np.ndarray): 3D numpy array of the medical image to be segmented
                                    Shape: (D, H, W) = (depth, height, width)
        mask_data_3d (np.ndarray): 3D numpy array of the current mask
                                   Shape: (D, H, W)
        target_label (int): The label number that the user is requesting to segment

    Returns:
        np.ndarray: 3D segmentation mask with the target_label
    """
    try:
        print(f"Running MedSAM2 segmentation")
        print(f"Image shape: {image_data_3d.shape}, Mask shape: {mask_data_3d.shape}")
        print(f"Target label: {target_label}")
        from .aiModels.medsam2_adapter import run_medsam2_inference
        return run_medsam2_inference(image_data_3d, mask_data_3d, target_label)
    except Exception as e:
        print(f"Error in MedSAM2 segmentation: {str(e)}")
        import traceback
        traceback.print_exc()
        raise Exception(f"MedSAM2 segmentation failed: {str(e)}")


# ==================== API Views ====================

class GetAIModelsView(APIView):
    """
    API endpoint to retrieve available AI segmentation models.
    
    GET Parameters: None
    
    Response:
        {
            "models": ["model_1_name", "model_2_name", ...],
            "count": 2
        }
    """
    
    def get(self, request):
        try:
            from .aiModels.registry import list_model_descriptors
            available_models = list_model_descriptors(include_capabilities=True)
            return Response(
                {
                    "models": available_models,
                    "count": len(available_models)
                },
                status=status.HTTP_200_OK
            )
            
        except Exception as e:
            print(f"Error retrieving AI models: {str(e)}")
            return Response(
                {"error": f"Failed to retrieve models: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class RunAISegmentationView(APIView):
    """
    API endpoint to run AI segmentation on the current mask.
    
    POST Parameters:
        - directory (str): Path to the data directory
        - subject (str): Subject/scan filename
        - edit (str): The specific edit being segmented (e.g., "scan_lossy.nii.gz")
        - model (str): Name of the AI model to use
        - current_label (int): The label number to assign to predicted regions
    
    Response:
        {
            "message": "Success message",
            "segmentation_label": int,
            "voxels_segmented": int,
            "model_used": str
        }
    """
    
    def post(self, request):
        print("=" * 60)
        print("RunAISegmentationView.post() called")
        print(f"Request data: {request.data}")
        print("=" * 60)

        # Extract parameters with debug logging
        directory = request.data.get('directory')
        subject = request.data.get('subject')
        edit = request.data.get('edit')
        model = request.data.get('model')
        current_label = request.data.get('current_label')

        print(f"Extracted parameters:")
        print(f"  directory: {directory} (type: {type(directory)})")
        print(f"  subject: {subject} (type: {type(subject)})")
        print(f"  edit: {edit} (type: {type(edit)})")
        print(f"  model: {model} (type: {type(model)})")
        print(f"  current_label: {current_label} (type: {type(current_label)})")

        # Validate required parameters - FIX: check each parameter explicitly
        missing_params = []
        if not directory:
            missing_params.append("directory")
        if not subject:
            missing_params.append("subject")
        # edit can be None (meaning use raw image), so don't validate it
        if not model:
            missing_params.append("model")
        if current_label is None:
            missing_params.append("current_label")

        if missing_params:
            error_msg = f"Missing required parameters: {', '.join(missing_params)}"
            print(f"VALIDATION ERROR: {error_msg}")
            return Response(
                {"error": error_msg},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Convert current_label to int with better error handling
            try:
                current_label = int(current_label)
                print(f"Converted current_label to int: {current_label}")
            except (ValueError, TypeError) as e:
                error_msg = f"Invalid current_label format: {current_label} - {str(e)}"
                print(f"VALIDATION ERROR: {error_msg}")
                return Response(
                    {"error": error_msg},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Validate label range
            if current_label < 1 or current_label > 255:
                error_msg = f"Label must be between 1 and 255, got: {current_label}"
                print(f"VALIDATION ERROR: {error_msg}")
                return Response(
                    {"error": error_msg},
                    status=status.HTTP_400_BAD_REQUEST
                )

            print(f"Label validation passed: {current_label}")

            # Process edit parameter - handle None case (use raw image)
            if edit is None:
                # When edit is None, use the subject as the base filename
                lossy_edit = f"{subject}_lossy"
                full_edit = subject
            else:
                # Process edit parameter - remove .nii.gz extension if present
                edit_clean = edit.replace(".nii.gz", "")
                lossy_edit = edit_clean
                full_edit = edit_clean.replace('_lossy', '')

            print(f"Edit name processing:")
            print(f"  Original edit: {edit}")
            if edit is None:
                print(f"  Using raw image (edit was None)")
            else:
                print(f"  edit_clean: {edit_clean}")
            print(f"  lossy_edit: {lossy_edit}")
            print(f"  full_edit: {full_edit}")

            # Determine base path
            if subject == "atlas":
                base_path = os.path.join(directory, "atlas")
            else:
                base_path = os.path.join(directory, "extracted", subject)

            print(f"Base path: {base_path}")
            print(f"Base path exists: {os.path.exists(base_path)}")

            if not os.path.exists(base_path):
                error_msg = f"Subject directory not found: {base_path}"
                print(f"PATH ERROR: {error_msg}")
                return Response(
                    {"error": error_msg},
                    status=status.HTTP_404_NOT_FOUND
                )

            # Load the image (working with full resolution)
            image_filename = f"{full_edit}.nii.gz"
            image_path = os.path.join(base_path, image_filename)

            print(f"Image file check:")
            print(f"  Expected filename: {image_filename}")
            print(f"  Expected path: {image_path}")
            print(f"  File exists: {os.path.exists(image_path)}")

            if not os.path.exists(image_path):
                # List available files for debugging
                try:
                    available_files = os.listdir(base_path)
                    print(f"Available files in {base_path}: {available_files}")
                except Exception as list_e:
                    print(f"Could not list directory: {str(list_e)}")

                error_msg = f"Image file not found: {image_filename} at {image_path}"
                print(f"FILE NOT FOUND ERROR: {error_msg}")
                return Response(
                    {"error": error_msg},
                    status=status.HTTP_404_NOT_FOUND
                )

            # Load the current mask
            mask_filename = f"{full_edit}.nii.mask.gz"
            mask_temp_filename = f"{full_edit}_mask.nii.gz"
            mask_path = os.path.join(base_path, mask_filename)
            mask_temp_path = os.path.join(base_path, mask_temp_filename)

            print(f"Mask file check:")
            print(f"  Expected mask filename: {mask_filename}")
            print(f"  Mask path: {mask_path}")
            print(f"  Mask exists: {os.path.exists(mask_path)}")

            # Load image data
            try:
                print(f"Loading image from: {image_path}")
                image_img = nib.load(image_path)
                image_data_3d = image_img.get_fdata()
                print(f"Image loaded successfully. Shape: {image_data_3d.shape}, dtype: {image_data_3d.dtype}")
            except Exception as img_e:
                error_msg = f"Failed to load image: {str(img_e)}"
                print(f"IMAGE LOAD ERROR: {error_msg}")
                raise

            # Load mask data (following the pattern from LoadMaskView)
            mask_data_3d = None

            try:
                if os.path.exists(mask_path):
                    print(f"Existing mask found at: {mask_path}")
                    # Temporarily rename to standard extension for nibabel
                    os.replace(mask_path, mask_temp_path)
                    print(f"Renamed mask to temporary name: {mask_temp_path}")

                    try:
                        print(f"Loading mask from temporary path")
                        mask_img = nib.load(mask_temp_path)
                        # Round before astype to avoid truncating floats
                        mask_data_3d = np.round(mask_img.get_fdata()).astype(np.uint8)
                        print(f"Mask loaded successfully. Shape: {mask_data_3d.shape}, dtype: {mask_data_3d.dtype}")

                    finally:
                        # Always rename back to original custom extension
                        os.replace(mask_temp_path, mask_path)
                        print(f"Renamed mask back to original name: {mask_path}")
                else:
                    # No existing mask, create empty one with same shape as image
                    print(f"No existing mask found, creating empty mask")
                    mask_data_3d = np.zeros_like(image_data_3d)
                    print(f"Empty mask created. Shape: {mask_data_3d.shape}, dtype: {mask_data_3d.dtype}")
            except Exception as mask_e:
                error_msg = f"Failed to load/create mask: {str(mask_e)}"
                print(f"MASK LOAD ERROR: {error_msg}")
                # Try to recover if rename failed
                if os.path.exists(mask_temp_path) and not os.path.exists(mask_path):
                    os.replace(mask_temp_path, mask_path)
                    print(f"Recovered: renamed temporary mask back to original")
                raise

            # Run AI segmentation
            try:
                print(f"Starting AI segmentation with model: {model}")
                segmented_mask_3d = ai_segmentation(
                    image_data_3d,
                    mask_data_3d,
                    current_label,
                    model
                )
                print(f"AI segmentation completed. Result shape: {segmented_mask_3d.shape}")
            except Exception as seg_e:
                error_msg = f"AI segmentation failed: {str(seg_e)}"
                print(f"SEGMENTATION ERROR: {error_msg}")
                raise

            # Count segmented voxels
            voxels_with_label = np.sum(segmented_mask_3d)
            print(f"Total voxels with target label: {voxels_with_label}")

            # Save the updated mask back (following the pattern from SaveMaskView)
            try:
                print(f"Starting segmentation result save")
                self._save_segmentation_result(
                    segmented_mask_3d,
                    image_img,
                    mask_path,
                    mask_temp_path,
                    lossy_edit,
                    full_edit,
                    base_path,
                    current_label
                )
                print(f"Segmentation result saved successfully")
            except Exception as save_e:
                error_msg = f"Failed to save segmentation: {str(save_e)}"
                print(f"SAVE ERROR: {error_msg}")
                raise

            return Response(
                {
                    "message": f"AI segmentation completed successfully using {model}",
                    "segmentation_label": current_label,
                    "voxels_segmented": int(voxels_with_label),
                    "model_used": model
                },
                status=status.HTTP_200_OK
            )

        except Exception as e:
            from .aiModels.foundation_models import ModelMissingError
            if isinstance(e, ModelMissingError):
                return Response(
                    {"code": "model_missing", "model": e.model_id, "error": str(e)},
                    status=status.HTTP_409_CONFLICT,
                )
            error_msg = f"Failed to run AI segmentation: {str(e)}"
            print(f"FATAL ERROR: {error_msg}")
            print("=" * 60)
            import traceback
            traceback.print_exc()
            print("=" * 60)
            return Response(
                {"error": error_msg},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def _save_segmentation_result(self, segmented_mask_3d, image_img, mask_path, 
                                mask_temp_path, lossy_edit, full_edit, base_path, label):
        """
        Save the segmentation result back to disk.
        
        IMPORTANT: This saves BOTH full and lossy resolution masks, following SaveMaskView pattern.
        The AI result overwrites the existing mask at the selected label location.
        Only voxels where the AI predicted the target label are updated; all other labels are preserved.
        
        Args:
            segmented_mask_3d (np.ndarray): The segmented 3D mask array from AI model
            image_img (nibabel.Nifti1Image): The original full resolution image
            mask_path (str): Path to save the mask with custom extension (.nii.mask.gz)
            mask_temp_path (str): Temporary path with standard extension (_mask.nii.gz)
            lossy_edit (str): Lossy edit name
            full_edit (str): Full resolution edit name
            base_path (str): Base directory path
            label (int): The label that was segmented
        """
        try:
            # ===== SAVE FULL RESOLUTION FIRST =====
            full_image_path = os.path.join(base_path, f"{full_edit}.nii.gz")
            
            if os.path.exists(full_image_path):
                full_image = nib.load(full_image_path)
                full_shape = full_image.header.get_data_shape()
                target_shape = (full_shape[0], full_shape[1], full_shape[2])
                
                # Load existing full resolution mask to preserve other labels
                full_mask_filename = f"{full_edit}.nii.mask.gz"
                full_mask_temp_filename = f"{full_edit}_mask.nii.gz"
                full_save_path = os.path.join(base_path, full_mask_filename)
                full_temp_path = os.path.join(base_path, full_mask_temp_filename)
                
                if os.path.exists(full_save_path):
                    os.replace(full_save_path, full_temp_path)
                    try:
                        existing_full_mask_img = nib.load(full_temp_path)
                        # Round before astype to avoid truncating floats
                        existing_full_mask_data = np.round(existing_full_mask_img.get_fdata()).astype(np.uint8)
                    finally:
                        os.replace(full_temp_path, full_save_path)
                else:
                    existing_full_mask_data = np.zeros(target_shape, dtype=np.uint8)
                
                # Ensure segmented_mask_3d matches target shape
                if segmented_mask_3d.shape != target_shape:
                    zoom_factors = [target_shape[i] / segmented_mask_3d.shape[i] 
                                for i in range(len(target_shape))]
                    scaled_seg_mask = zoom(segmented_mask_3d, zoom_factors, order=0)
                    
                    # Ensure exact shape
                    if scaled_seg_mask.shape != target_shape:
                        result = np.zeros(target_shape, dtype=segmented_mask_3d.dtype)
                        slices = tuple(slice(0, min(scaled_seg_mask.shape[i], target_shape[i])) 
                                    for i in range(len(target_shape)))
                        result[slices] = scaled_seg_mask[slices]
                        scaled_seg_mask = result
                else:
                    scaled_seg_mask = segmented_mask_3d
                
                
                # Save full resolution mask
                full_mask_img = nib.Nifti1Image(scaled_seg_mask.astype(np.uint8), full_image.affine, 
                                            header=full_image.header)
                nib.save(full_mask_img, full_temp_path)
                os.replace(full_temp_path, full_save_path)
                
                print(f"Saved AI segmentation (full resolution) to: {full_save_path}")
            
            # ===== SAVE LOSSY RESOLUTION (scaled down from full) =====
            lossy_image_path = os.path.join(base_path, f"{lossy_edit}.nii.gz")

            print(f"Saving lossy resolution mask from image: {lossy_image_path}")

            if os.path.exists(lossy_image_path):
                lossy_image = nib.load(lossy_image_path)
                lossy_shape = lossy_image.header.get_data_shape()
                target_lossy_shape = (lossy_shape[0], lossy_shape[1], lossy_shape[2])
                
                # Scale the already-merged full resolution mask down to lossy resolution
                zoom_factors = [target_lossy_shape[i] / scaled_seg_mask.shape[i] 
                            for i in range(len(target_lossy_shape))]
                scaled_lossy_mask = zoom(scaled_seg_mask, zoom_factors, order=0)
                
                # Ensure exact shape
                if scaled_lossy_mask.shape != target_lossy_shape:
                    result = np.zeros(target_lossy_shape, dtype=scaled_seg_mask.dtype)
                    slices = tuple(slice(0, min(scaled_lossy_mask.shape[i], target_lossy_shape[i])) 
                                for i in range(len(target_lossy_shape)))
                    result[slices] = scaled_lossy_mask[slices]
                    scaled_lossy_mask = result
                
                # Save lossy resolution mask directly (no merging needed)
                lossy_mask_filename = f"{lossy_edit}.nii.mask.gz"
                lossy_mask_temp_filename = f"{lossy_edit}_mask.nii.gz"
                lossy_save_path = os.path.join(base_path, lossy_mask_filename)
                lossy_temp_path = os.path.join(base_path, lossy_mask_temp_filename)
                
                lossy_mask_img = nib.Nifti1Image(scaled_lossy_mask.astype(np.uint8), lossy_image.affine, 
                                                header=lossy_image.header)
                nib.save(lossy_mask_img, lossy_temp_path)
                os.replace(lossy_temp_path, lossy_save_path)
                
                print(f"Saved AI segmentation (lossy resolution) to: {lossy_save_path}")
                
        except Exception as e:
            print(f"Error saving segmentation result: {str(e)}")
            raise Exception(f"Failed to save segmentation result: {str(e)}")


