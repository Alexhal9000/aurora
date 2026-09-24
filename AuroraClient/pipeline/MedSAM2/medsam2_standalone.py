"""
Standalone MedSAM2 Segmentation Class
Handles 3D medical image segmentation with bounding box prompts
"""

import os
import sys
import numpy as np
import torch
from PIL import Image
from contextlib import nullcontext
from skimage import measure
from typing import Tuple, Optional

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")


class MedSAM2Segmenter:
    """
    Standalone MedSAM2 segmentation class for 3D medical images.
    
    This class handles:
    - Loading and preprocessing 3D images of any size
    - Converting bounding boxes to proper coordinates
    - Running MedSAM2 segmentation
    - Returning masks in the original image coordinate space
    
    Attributes:
        checkpoint_path (str): Path to the MedSAM2 model checkpoint
        config_path (str): Path to the config file for the model
        device (torch.device): Device to use for inference (cuda or cpu)
        predictor: Loaded MedSAM2 predictor
    """
    
    def __init__(
        self,
        checkpoint_path: str = None,
        config_path: str = "configs/sam2.1_hiera_t512.yaml",
        device: Optional[str] = None,
        work_dir: Optional[str] = None,
        force_cpu: bool = False
    ):
        """
        Initialize MedSAM2Segmenter.
        
        Args:
            checkpoint_path: Path to model checkpoint (default: models/MedSAM2_latest.pt relative to script)
            config_path: Path to config file (relative paths are resolved from work_dir)
            device: Device to use ('cuda' or 'cpu'). If None, auto-detects.
            work_dir: Working directory for config resolution. Defaults to script directory.
            force_cpu: If True, force CPU mode regardless of CUDA availability
        """
        # Default checkpoint: Documents/Aurora AI Models/foundation/medsam2 only.
        if checkpoint_path is None:
            from pipeline.aiModels.foundation_models import MEDSAM2_ID, resolve_model_path
            checkpoint_path = str(resolve_model_path(MEDSAM2_ID))
        
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.work_dir = work_dir or SCRIPT_DIR
        self.force_cpu = force_cpu
        
        # Determine device
        if force_cpu:
            self.device = torch.device("cpu")
            print("CPU mode forced")
        elif device is None:
            # Auto-detect with fallback to CPU on error
            if torch.cuda.is_available():
                try:
                    # Test CUDA availability
                    torch.tensor([1.0], device="cuda")
                    self.device = torch.device("cuda")
                    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
                except RuntimeError as e:
                    print(f"CUDA device test failed: {e}")
                    print("Falling back to CPU")
                    self.device = torch.device("cpu")
            else:
                self.device = torch.device("cpu")
                print("CUDA not available, using CPU")
        else:
            self.device = torch.device(device)
        
        self.predictor = None
        self._original_cwd = None
        
        # Load predictor
        self._load_predictor()
    
    def _load_predictor(self):
        """Load the MedSAM2 predictor."""
        try:
            # Convert work_dir to absolute path
            abs_work_dir = os.path.abspath(self.work_dir)
            
            # Add work directory to Python path for module resolution
            # This must be done BEFORE any imports to ensure Hydra can find modules
            if abs_work_dir not in sys.path:
                sys.path.insert(0, abs_work_dir)
            
            # Change to work directory for config resolution
            self._original_cwd = os.getcwd()
            os.chdir(abs_work_dir)
            
            # Pre-import sam2 and efficient_track_anything modules to ensure they're discoverable by Hydra
            import sam2
            import sam2.modeling
            import sam2.modeling.backbones
            import sam2.modeling.backbones.image_encoder
            import sam2.modeling.backbones.hieradet
            import sam2.modeling.backbones.vitdet
            import sam2.modeling.memory_attention
            import sam2.modeling.memory_encoder
            import sam2.modeling.position_encoding
            import sam2.modeling.sam
            import sam2.modeling.sam.mask_decoder
            import sam2.modeling.sam.prompt_encoder
            import sam2.modeling.sam.transformer
            import efficient_track_anything
            import efficient_track_anything.modeling
            import efficient_track_anything.modeling.efficienttam_base
            import efficient_track_anything.modeling.backbones.image_encoder as eta_image_encoder
            import efficient_track_anything.modeling.backbones.vitdet as eta_vitdet
            import efficient_track_anything.modeling.position_encoding as eta_position_encoding
            import efficient_track_anything.modeling.memory_attention as eta_memory_attention
            
            # Import after path is set and modules are preloaded
            from sam2.build_sam import build_sam2_video_predictor_npz
            
            # Configure Hydra to search for configs in the correct location
            from hydra import initialize_config_dir, compose
            from hydra.core.global_hydra import GlobalHydra
            
            # Clear any existing Hydra instances to avoid conflicts
            GlobalHydra.instance().clear()
            
            # Initialize Hydra with absolute config directory path
            config_dir = os.path.join(abs_work_dir, "configs")
            
            # Extract just the filename from config path since Hydra will search in config_dir
            # If config_path is 'configs/sam2.1_hiera_t512.yaml', extract 'sam2.1_hiera_t512.yaml'
            config_path = self.config_path
            if config_path.startswith("configs/"):
                config_path = config_path.replace("configs/", "", 1)
            # If it's a nested path like 'sam2.1/sam2.1_hiera_t512.yaml', keep as is
            # Hydra will search in config_dir and find it there
            
            # Convert checkpoint path to absolute if relative
            checkpoint_path = self.checkpoint_path
            if not os.path.isabs(checkpoint_path):
                checkpoint_path = os.path.join(abs_work_dir, checkpoint_path)
            
            # Initialize Hydra with the correct config directory
            with initialize_config_dir(version_base=None, config_dir=config_dir):
                self.predictor = build_sam2_video_predictor_npz(
                    config_path,
                    checkpoint_path,
                    device=str(self.device)
                )
            print(f"MedSAM2 predictor loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading predictor: {e}")
            raise
        finally:
            # Restore original working directory
            if self._original_cwd:
                os.chdir(self._original_cwd)
    
    def _resize_grayscale_to_rgb_and_resize(
        self,
        array: np.ndarray,
        image_size: int = 512
    ) -> np.ndarray:
        """
        Resize a 3D grayscale NumPy array to RGB format.
        
        Directly resizes to target size without padding. The predictor internally
        handles any coordinate transformations needed.
        
        Args:
            array: Input array of shape (d, h, w) with values in [0, 255]
            image_size: Target size (512)
        
        Returns:
            Resized array of shape (d, 3, image_size, image_size)
        """
        d, h, w = array.shape
        resized_array = np.zeros((d, 3, image_size, image_size), dtype=np.uint8)
        
        for i in range(d):
            # Convert to PIL Image and to RGB
            img_pil = Image.fromarray(array[i].astype(np.uint8))
            img_rgb = img_pil.convert("RGB")
            
            # Resize directly to (image_size, image_size)
            # This stretches the image but predictor handles the distortion
            img_resized = img_rgb.resize((image_size, image_size), Image.LANCZOS)
            img_array = np.array(img_resized).transpose(2, 0, 1)  # (3, image_size, image_size)
            
            resized_array[i] = img_array
        
        return resized_array
    
    def _get_largest_cc(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Get the largest connected component from a binary segmentation.
        
        Args:
            segmentation: Binary segmentation mask
        
        Returns:
            Binary mask containing only the largest connected component
        """
        labels = measure.label(segmentation)
        if labels.max() == 0:
            return segmentation
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        return largestCC
    
    def _preprocess_image(self, image_3d: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Preprocess a 3D image for MedSAM2 inference.
        
        Args:
            image_3d: 3D numpy array of shape (d, h, w)
        
        Returns:
            Tuple of (preprocessed_image, min_val, max_val) for later restoration
        """
        # Normalize to [0, 255]
        min_val = np.min(image_3d)
        max_val = np.max(image_3d)
        
        if max_val == min_val:
            preprocessed = np.zeros_like(image_3d, dtype=np.uint8)
        else:
            preprocessed = ((image_3d - min_val) / (max_val - min_val)) * 255.0
            preprocessed = np.uint8(preprocessed)
        
        return preprocessed, min_val, max_val
    
    def _normalize_for_model(self, img_resized: torch.Tensor) -> torch.Tensor:
        """
        Apply ImageNet normalization to preprocessed image.
        
        Args:
            img_resized: Image tensor of shape (d, 3, h, w) with values in [0, 1]
        
        Returns:
            Normalized image tensor
        """
        img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[
            :, None, None
        ].to(self.device)
        img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[
            :, None, None
        ].to(self.device)
        
        img_resized -= img_mean
        img_resized /= img_std
        
        return img_resized
    
    def _ensure_mask_binary_and_valid(self, mask: np.ndarray) -> np.ndarray:
        """
        Ensure mask is valid binary format.
        
        Args:
            mask: 2D or 3D numpy array
        
        Returns:
            Binary mask (0 and 1 values only)
        """
        # Ensure binary - any non-zero becomes 1
        return (mask > 0).astype(np.uint8)
    
    def _validate_mask_for_frame(
        self,
        mask_data_3d: np.ndarray,
        frame_idx: int,
        expected_h: int,
        expected_w: int,
        bbox_min: Tuple[int, int, int] = None,
        bbox_max: Tuple[int, int, int] = None
    ) -> Optional[np.ndarray]:
        """
        Extract, validate, and prepare mask for a specific frame.
        
        Validates that mask aligns with image coordinates and has content within bbox.
        
        Args:
            mask_data_3d: 3D mask array (D, H, W)
            frame_idx: Frame index to extract
            expected_h: Expected height of frame
            expected_w: Expected width of frame
            bbox_min: Optional bbox for validation (z_min, y_min, x_min)
            bbox_max: Optional bbox for validation (z_max, y_max, x_max)
        
        Returns:
            Binary 2D mask (H, W) in original coordinates, or None if invalid
        """
        if mask_data_3d is None or mask_data_3d.ndim != 3:
            return None
        
        try:
            # Validate index
            if not (0 <= frame_idx < mask_data_3d.shape[0]):
                return None
            
            # Extract 2D mask for this frame
            frame_mask = mask_data_3d[frame_idx]
            
            # Validate shape
            if frame_mask.shape != (expected_h, expected_w):
                print(f"    Warning: Mask shape {frame_mask.shape} doesn't match "
                      f"expected ({expected_h}, {expected_w})")
                return None
            
            # Ensure binary
            frame_mask_binary = self._ensure_mask_binary_and_valid(frame_mask)
            
            # Check for content
            voxel_count = np.sum(frame_mask_binary)
            if voxel_count == 0:
                return None
            
            # Optional: validate content is within bbox
            if bbox_min is not None and bbox_max is not None:
                _, y_min, x_min = bbox_min
                _, y_max, x_max = bbox_max
                bbox_region_count = np.sum(
                    frame_mask_binary[y_min:y_max+1, x_min:x_max+1]
                )
                if bbox_region_count == 0:
                    print(f"    Warning: No mask content within bbox region for frame {frame_idx}")
                    return None
            
            return frame_mask_binary
            
        except Exception as e:
            print(f"    Error validating mask for frame {frame_idx}: {e}")
            return None
    
    def segment(
        self,
        image_3d: np.ndarray,
        bbox_min: Tuple[int, int, int],
        bbox_max: Tuple[int, int, int],
        mask_data_3d: np.ndarray = None,
        use_largest_cc: bool = True,
        existing_mask_as_reference: bool = False,
        ensemble: bool = False
    ) -> np.ndarray:
        """
        Segment a 3D medical image with optional mask-guided refinement.
        
        Args:
            image_3d: 3D numpy array of shape (D, H, W)
            bbox_min: Tuple of (z_min, y_min, x_min)
            bbox_max: Tuple of (z_max, y_max, x_max)
            mask_data_3d: Optional binary mask in same coordinate space as image_3d.
                         Shape (D, H, W), values 0 or non-zero.
            use_largest_cc: Whether to extract largest connected component
            existing_mask_as_reference: If True, use mask_data_3d to:
                                       1) Find center slice with max voxels
                                       2) Provide mask as prompt to guide predictions
            ensemble: If True, use 3-view ensemble
        
        Returns:
            Binary segmentation mask of shape (D, H, W)
        """
        try:
            # Validate inputs
            if image_3d.ndim != 3:
                raise ValueError(f"Expected 3D image, got shape {image_3d.shape}")
            
            d_orig, h_orig, w_orig = image_3d.shape
            z_min, y_min, x_min = bbox_min
            z_max, y_max, x_max = bbox_max
            
            # Validate bbox coordinates
            if not (0 <= z_min < z_max <= d_orig):
                raise ValueError(f"Invalid z range: {z_min} to {z_max}, image depth: {d_orig}")
            if not (0 <= y_min < y_max <= h_orig):
                raise ValueError(f"Invalid y range: {y_min} to {y_max}, image height: {h_orig}")
            if not (0 <= x_min < x_max <= w_orig):
                raise ValueError(f"Invalid x range: {x_min} to {x_max}, image width: {w_orig}")
            
            # Ensemble mode
            if ensemble:
                print("Ensemble mode enabled: using 3 orthogonal views")
                return self._segment_ensemble(
                    image_3d,
                    bbox_min,
                    bbox_max,
                    mask_data_3d,
                    use_largest_cc,
                    existing_mask_as_reference
                )
            
            # Determine center slice
            if existing_mask_as_reference and mask_data_3d is not None:
                # Find slice with maximum mask content within bbox
                max_count = -1
                center_slice_orig = z_min
                
                for z in range(z_min, z_max + 1):
                    count = np.sum(mask_data_3d[z, y_min:y_max+1, x_min:x_max+1] > 0)
                    if count > max_count:
                        max_count = count
                        center_slice_orig = z
                
                if max_count > 0:
                    print(f"Center slice (max mask): {center_slice_orig} with {max_count} voxels")
                else:
                    center_slice_orig = (z_min + z_max) // 2
                    print(f"No mask content found, using geometric center: {center_slice_orig}")
            else:
                center_slice_orig = (z_min + z_max) // 2
            
            # Preprocess the full image
            img_preprocessed, _, _ = self._preprocess_image(image_3d)
            
            # Resize to 512x512 for model
            img_resized = self._resize_grayscale_to_rgb_and_resize(img_preprocessed, 512)
            img_resized = img_resized / 255.0
            img_resized = torch.from_numpy(img_resized).to(self.device)
            
            # Normalize for ImageNet
            img_resized = self._normalize_for_model(img_resized)
            
            # Initialize segmentation array
            segs_3d = np.zeros(image_3d.shape, dtype=np.uint8)
            
            # Bbox in original image coordinates (NOT scaled)
            # The predictor's init_state(img, h_orig, w_orig) internally handles all scaling
            bbox = np.array([x_min, y_min, x_max, y_max])
            
            print(f"\n=== MEDSAM2 SEGMENTATION ===")
            print(f"Image shape: {image_3d.shape}")
            print(f"BBox: {bbox}")
            print(f"Center slice: {center_slice_orig}")
            print(f"Using mask reference: {existing_mask_as_reference}")
            print(f"============================\n")
            
            device_str = "cuda" if self.device.type == "cuda" else "cpu"
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device_str == "cuda"
                else nullcontext()
            )
            
            with torch.inference_mode(), autocast_context:
                # ========== FORWARD PROPAGATION ==========
                inference_state = self.predictor.init_state(
                    img_resized,
                    h_orig,
                    w_orig
                )
                
                # Prepare mask prompt for center slice if available
                center_mask_prompt = None
                if existing_mask_as_reference and mask_data_3d is not None:
                    center_mask_prompt = self._validate_mask_for_frame(
                        mask_data_3d,
                        center_slice_orig,
                        h_orig,
                        w_orig,
                        bbox_min,
                        bbox_max
                    )
                
                # Add prompts to predictor
                if center_mask_prompt is not None:
                    # Add mask as prior for center slice
                    print(f"Adding center slice mask as prior")
                    try:
                        # Convert numpy mask to torch tensor
                        mask_tensor = torch.from_numpy(center_mask_prompt).bool()
                        self.predictor.add_new_mask(
                            inference_state=inference_state,
                            frame_idx=center_slice_orig,
                            obj_id=1,
                            mask=mask_tensor,
                        )
                    except Exception as e:
                        print(f"Warning: Error adding mask prompt: {e}")
                        print("Falling back to box-only prompting")
                        _, _, _ = self.predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=center_slice_orig,
                            obj_id=1,
                            box=bbox,
                        )
                else:
                    # No mask, use box prompting
                    _, _, _ = self.predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=center_slice_orig,
                        obj_id=1,
                        box=bbox,
                    )
                
                # Propagate forward (constrained to bbox Z range)
                # Forward: from center_slice to z_max
                max_forward_frames = z_max - center_slice_orig + 1
                print(f"Forward propagation: frames {center_slice_orig} to {z_max} (max_frames={max_forward_frames})")
                
                for out_frame_idx, _, out_mask_logits in self.predictor.propagate_in_video(
                    inference_state,
                    start_frame_idx=center_slice_orig,
                    max_frame_num_to_track=max_forward_frames,
                    reverse=False
                ):
                    mask = (out_mask_logits[0] > 0.0).cpu().numpy()
                    if mask.ndim == 3:
                        mask = mask[0]
                    segs_3d[out_frame_idx, mask.astype(bool)] = 1
                
                self.predictor.reset_state(inference_state)
                
                # ========== BACKWARD PROPAGATION ==========
                inference_state = self.predictor.init_state(
                    img_resized,
                    h_orig,
                    w_orig
                )
                
                # Re-add prompts for backward pass
                if center_mask_prompt is not None:
                    try:
                        # Convert numpy mask to torch tensor
                        mask_tensor = torch.from_numpy(center_mask_prompt).bool()
                        self.predictor.add_new_mask(
                            inference_state=inference_state,
                            frame_idx=center_slice_orig,
                            obj_id=1,
                            mask=mask_tensor,
                        )
                    except Exception as e:
                        print(f"Warning: Error adding mask prompt in backward pass: {e}")
                        print("Falling back to box-only prompting")
                        _, _, _ = self.predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=center_slice_orig,
                            obj_id=1,
                            box=bbox,
                        )
                else:
                    _, _, _ = self.predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=center_slice_orig,
                        obj_id=1,
                        box=bbox,
                    )
                
                # Propagate backward (constrained to bbox Z range)
                # Backward: from center_slice to z_min
                max_backward_frames = center_slice_orig - z_min + 1
                print(f"Backward propagation: frames {center_slice_orig} to {z_min} (max_frames={max_backward_frames})")
                
                for out_frame_idx, _, out_mask_logits in self.predictor.propagate_in_video(
                    inference_state,
                    start_frame_idx=center_slice_orig,
                    max_frame_num_to_track=max_backward_frames,
                    reverse=True
                ):
                    mask = (out_mask_logits[0] > 0.0).cpu().numpy()
                    if mask.ndim == 3:
                        mask = mask[0]
                    segs_3d[out_frame_idx, mask.astype(bool)] = 1
                
                self.predictor.reset_state(inference_state)
            
            # Clip segmentation to bbox boundaries (Z, Y, X)
            print(f"Clipping segmentation to bbox boundaries:")
            print(f"  Z: {z_min} to {z_max}")
            print(f"  Y: {y_min} to {y_max}")
            print(f"  X: {x_min} to {x_max}")
            
            # Clip Z (depth)
            segs_3d[:z_min, :, :] = 0
            segs_3d[z_max+1:, :, :] = 0
            
            # Clip Y (height)
            segs_3d[:, :y_min, :] = 0
            segs_3d[:, y_max+1:, :] = 0
            
            # Clip X (width)
            segs_3d[:, :, :x_min] = 0
            segs_3d[:, :, x_max+1:] = 0
            
            # Post-process
            if np.max(segs_3d) > 0 and use_largest_cc:
                segs_3d = self._get_largest_cc(segs_3d)
                segs_3d = np.uint8(segs_3d)
            
            return segs_3d
        
        except Exception as e:
            print(f"Error during segmentation: {e}")
            raise
    
    def _segment_ensemble(
        self,
        image_3d: np.ndarray,
        bbox_min: Tuple[int, int, int],
        bbox_max: Tuple[int, int, int],
        mask_data_3d: np.ndarray = None,
        use_largest_cc: bool = True,
        existing_mask_as_reference: bool = False
    ) -> np.ndarray:
        """
        Segment using 3 orthogonal views and combine with majority voting.
        
        Three views are used:
        - View 1: XY plane (looking along Z-axis) - center slice along Z
        - View 2: XZ plane (looking along Y-axis) - center slice along Y
        - View 3: YZ plane (looking along X-axis) - center slice along X
        
        Final segmentation keeps voxels that appear in at least 2 out of 3 views.
        
        Args:
            image_3d: 3D numpy array of shape (D, H, W)
            bbox_min: Tuple of (z_min, y_min, x_min)
            bbox_max: Tuple of (z_max, y_max, x_max)
            mask_data_3d: Optional binary mask for reference segmentation
            use_largest_cc: Whether to extract largest connected component
            existing_mask_as_reference: If True, use mask_data_3d to find center slices
        
        Returns:
            Binary segmentation mask with majority voting
        """
        d_orig, h_orig, w_orig = image_3d.shape
        z_min, y_min, x_min = bbox_min
        z_max, y_max, x_max = bbox_max
        
        # Vote array: accumulates votes from each view
        vote_array = np.zeros(image_3d.shape, dtype=np.uint8)
        
        # View 1: XY plane (original orientation, D-H-W)
        print("  View 1/3: XY plane (center along Z-axis)...")
        seg1 = self._segment_single_orientation(
            image_3d,
            bbox_min,
            bbox_max,
            mask_data_3d,
            center_axis=0,  # Z-axis
            spatial_h=h_orig,
            spatial_w=w_orig,
            existing_mask_as_reference=existing_mask_as_reference
        )
        vote_array += seg1
        
        # View 2: XZ plane (reoriented to H-D-W, center along H)
        print("  View 2/3: XZ plane (center along Y-axis)...")
        image_reorient2 = np.moveaxis(image_3d, 1, 0)  # (D, H, W) -> (H, D, W)
        mask_reorient2 = np.moveaxis(mask_data_3d, 1, 0) if mask_data_3d is not None else None
        bbox_min_2 = (y_min, z_min, x_min)
        bbox_max_2 = (y_max, z_max, x_max)
        seg2 = self._segment_single_orientation(
            image_reorient2,
            bbox_min_2,
            bbox_max_2,
            mask_reorient2,
            center_axis=0,  # H-axis (originally Y)
            spatial_h=h_orig,
            spatial_w=w_orig,
            existing_mask_as_reference=existing_mask_as_reference
        )
        # Move back to original orientation
        seg2 = np.moveaxis(seg2, 0, 1)  # (H, D, W) -> (D, H, W)
        vote_array += seg2
        
        # View 3: YZ plane (reoriented to W-D-H, center along W)
        print("  View 3/3: YZ plane (center along X-axis)...")
        image_reorient3 = np.moveaxis(image_3d, 2, 0)  # (D, H, W) -> (W, D, H)
        mask_reorient3 = np.moveaxis(mask_data_3d, 2, 0) if mask_data_3d is not None else None
        bbox_min_3 = (x_min, z_min, y_min)
        bbox_max_3 = (x_max, z_max, y_max)
        seg3 = self._segment_single_orientation(
            image_reorient3,
            bbox_min_3,
            bbox_max_3,
            mask_reorient3,
            center_axis=0,  # W-axis (originally X)
            spatial_h=h_orig,
            spatial_w=w_orig,
            existing_mask_as_reference=existing_mask_as_reference
        )
        # Move back to original orientation
        seg3 = np.moveaxis(seg3, 0, 2)  # (W, D, H) -> (D, H, W)
        vote_array += seg3
        
        # Majority voting: keep voxels with votes >= 2 (at least 2 out of 3)
        print("  Combining views with majority voting (2/3 agreement)...")
        ensemble_mask = (vote_array >= 2).astype(np.uint8)
        
        # Post-process if requested
        if np.max(ensemble_mask) > 0 and use_largest_cc:
            ensemble_mask = self._get_largest_cc(ensemble_mask)
            ensemble_mask = np.uint8(ensemble_mask)
        
        return ensemble_mask
    
    def _segment_single_orientation(
        self,
        image_3d: np.ndarray,
        bbox_min: Tuple[int, int, int],
        bbox_max: Tuple[int, int, int],
        mask_data_3d: np.ndarray = None,
        center_axis: int = 0,
        spatial_h: Optional[int] = None,
        spatial_w: Optional[int] = None,
        existing_mask_as_reference: bool = False
    ) -> np.ndarray:
        """
        Run segmentation on a single orientation.
        
        Args:
            image_3d: 3D numpy array of shape (D, H, W) - may be reoriented for ensemble views
            bbox_min: Tuple of (d_min, h_min, w_min)
            bbox_max: Tuple of (d_max, h_max, w_max)
            mask_data_3d: Optional 3D binary mask for reference segmentation
            center_axis: Which axis to use for center frame (0, 1, or 2)
            spatial_h: Original spatial height dimension (for reoriented images). If None, uses image shape.
            spatial_w: Original spatial width dimension (for reoriented images). If None, uses image shape.
            existing_mask_as_reference: If True, use mask_data_3d to find center slice
        
        Returns:
            Binary segmentation mask of shape (D, H, W)
        """
        d_ori, h_ori, w_ori = image_3d.shape
        d_min, h_min, w_min = bbox_min
        d_max, h_max, w_max = bbox_max
        
        # Use provided spatial dimensions or fall back to current image shape
        # This allows correct scaling even when image is reoriented
        if spatial_h is None:
            spatial_h = h_ori
        if spatial_w is None:
            spatial_w = w_ori
        
        # Determine center slice
        if existing_mask_as_reference and mask_data_3d is not None:
            # Find slice with maximum mask content within bbox along the center_axis
            max_count = -1
            center_slice = d_min if center_axis == 0 else (h_min if center_axis == 1 else w_min)
            
            if center_axis == 0:
                for d in range(d_min, d_max + 1):
                    count = np.sum(mask_data_3d[d, h_min:h_max+1, w_min:w_max+1] > 0)
                    if count > max_count:
                        max_count = count
                        center_slice = d
            elif center_axis == 1:
                for h in range(h_min, h_max + 1):
                    count = np.sum(mask_data_3d[d_min:d_max+1, h, w_min:w_max+1] > 0)
                    if count > max_count:
                        max_count = count
                        center_slice = h
            else:
                for w in range(w_min, w_max + 1):
                    count = np.sum(mask_data_3d[d_min:d_max+1, h_min:h_max+1, w] > 0)
                    if count > max_count:
                        max_count = count
                        center_slice = w
            
            print(f"  Using existing mask. Center slice along axis {center_axis}: {center_slice} with {max_count} voxels")
        else:
            # Calculate center slice
            if center_axis == 0:
                center_slice = (d_min + d_max) // 2
            elif center_axis == 1:
                center_slice = (h_min + h_max) // 2
            else:
                center_slice = (w_min + w_max) // 2
        
        # Preprocess the image
        img_preprocessed, _, _ = self._preprocess_image(image_3d)
        
        # Resize to 512x512 for model
        img_resized = self._resize_grayscale_to_rgb_and_resize(img_preprocessed, 512)
        img_resized = img_resized / 255.0
        img_resized = torch.from_numpy(img_resized).to(self.device)
        
        # Normalize for ImageNet
        img_resized = self._normalize_for_model(img_resized)
        
        # Initialize segmentation array
        segs_3d = np.zeros(image_3d.shape, dtype=np.uint8)
        
        # Bbox in original image coordinates (NOT scaled)
        # The predictor's init_state(img, h_ori, w_ori) internally handles all scaling
        bbox = np.array([w_min, h_min, w_max, h_max])
        
        print(f"\n=== MEDSAM2 ENSEMBLE VIEW DEBUG ===")
        print(f"Current image shape (D,H,W): {image_3d.shape}")
        print(f"Spatial dimensions (H,W): {spatial_h}, {spatial_w}")
        print(f"BBox (original coords): {bbox} [w_min, h_min, w_max, h_max]")
        print(f"Center slice: {center_slice}")
        print(f"Using existing mask: {existing_mask_as_reference}")
        print(f"===================================\n")
        
        # Run inference
        device_str = "cuda" if self.device.type == "cuda" else "cpu"
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device_str == "cuda"
            else nullcontext()
        )
        
        with torch.inference_mode(), autocast_context:
            # ========== FORWARD PROPAGATION ==========
            inference_state = self.predictor.init_state(
                img_resized,
                h_ori,
                w_ori
            )
            
            # Prepare mask prompt for center slice if available
            center_mask_prompt = None
            if existing_mask_as_reference and mask_data_3d is not None:
                center_mask_prompt = self._validate_mask_for_frame(
                    mask_data_3d,
                    center_slice,
                    h_ori,
                    w_ori,
                    bbox_min,
                    bbox_max
                )
            
            # Add prompts to predictor
            if center_mask_prompt is not None:
                # Add mask as prior for center slice
                print(f"  Adding center slice mask as prior")
                try:
                    # Convert numpy mask to torch tensor
                    mask_tensor = torch.from_numpy(center_mask_prompt).bool()
                    self.predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=center_slice,
                        obj_id=1,
                        mask=mask_tensor,
                    )
                except Exception as e:
                    print(f"  Warning: Error adding mask prompt: {e}")
                    print("  Falling back to box-only prompting")
                    _, _, _ = self.predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=center_slice,
                        obj_id=1,
                        box=bbox,
                    )
            else:
                # No mask, use box prompting
                _, _, _ = self.predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=center_slice,
                    obj_id=1,
                    box=bbox,
                )
            
            # Propagate forward (constrained to bbox range along center_axis)
            # Calculate max frames based on center_axis
            if center_axis == 0:
                max_forward_frames = d_max - center_slice + 1
                range_str = f"{center_slice} to {d_max}"
            elif center_axis == 1:
                max_forward_frames = h_max - center_slice + 1
                range_str = f"{center_slice} to {h_max}"
            else:
                max_forward_frames = w_max - center_slice + 1
                range_str = f"{center_slice} to {w_max}"
            
            print(f"  Forward propagation (axis {center_axis}): {range_str} (max_frames={max_forward_frames})")
            
            for out_frame_idx, _, out_mask_logits in self.predictor.propagate_in_video(
                inference_state,
                start_frame_idx=center_slice,
                max_frame_num_to_track=max_forward_frames,
                reverse=False
            ):
                mask = (out_mask_logits[0] > 0.0).cpu().numpy()
                if mask.ndim == 3:
                    mask = mask[0]
                segs_3d[out_frame_idx, mask.astype(bool)] = 1
            
            self.predictor.reset_state(inference_state)
            
            # ========== BACKWARD PROPAGATION ==========
            inference_state = self.predictor.init_state(
                img_resized,
                h_ori,
                w_ori
            )
            
            # Re-add prompts for backward pass
            if center_mask_prompt is not None:
                try:
                    # Convert numpy mask to torch tensor
                    mask_tensor = torch.from_numpy(center_mask_prompt).bool()
                    self.predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=center_slice,
                        obj_id=1,
                        mask=mask_tensor,
                    )
                except Exception as e:
                    print(f"  Warning: Error adding mask prompt in backward pass: {e}")
                    print("  Falling back to box-only prompting")
                    _, _, _ = self.predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=center_slice,
                        obj_id=1,
                        box=bbox,
                    )
            else:
                _, _, _ = self.predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=center_slice,
                    obj_id=1,
                    box=bbox,
                )
            
            # Propagate backward (constrained to bbox range along center_axis)
            # Calculate max frames based on center_axis
            if center_axis == 0:
                max_backward_frames = center_slice - d_min + 1
                range_str = f"{center_slice} to {d_min}"
            elif center_axis == 1:
                max_backward_frames = center_slice - h_min + 1
                range_str = f"{center_slice} to {h_min}"
            else:
                max_backward_frames = center_slice - w_min + 1
                range_str = f"{center_slice} to {w_min}"
            
            print(f"  Backward propagation (axis {center_axis}): {range_str} (max_frames={max_backward_frames})")
            
            for out_frame_idx, _, out_mask_logits in self.predictor.propagate_in_video(
                inference_state,
                start_frame_idx=center_slice,
                max_frame_num_to_track=max_backward_frames,
                reverse=True
            ):
                mask = (out_mask_logits[0] > 0.0).cpu().numpy()
                if mask.ndim == 3:
                    mask = mask[0]
                segs_3d[out_frame_idx, mask.astype(bool)] = 1
            
            self.predictor.reset_state(inference_state)
        
        return segs_3d
    
    def __del__(self):
        """Cleanup on deletion."""
        if self._original_cwd:
            try:
                os.chdir(self._original_cwd)
            except:
                pass


# Toy example
if __name__ == "__main__":
    import nibabel as nib
    
    print("=" * 80)
    print("MedSAM2 Standalone Segmentation - Toy Example")
    print("=" * 80)
    
    # Load the mouse embryo test image
    nifti_path = "/home/alejandro/Documents/medsam2-gradio/MedSAM2/Mouse_embryo_test/Z_Ctr_1_Eby_1_reoriented_lossy_edit_3_cropped.nii.gz"
    
    print(f"\n1. Loading NIfTI image from: {nifti_path}")
    nib_img = nib.load(nifti_path)
    image_3d = nib_img.get_fdata()
    print(f"   Image shape: {image_3d.shape}")
    print(f"   Image dtype: {image_3d.dtype}")
    print(f"   Image intensity range: [{np.min(image_3d):.2f}, {np.max(image_3d):.2f}]")
    
    # Define a random bounding box
    d, h, w = image_3d.shape
    
    # Set bbox in original coordinates - approximately in the center region
    z_min = d // 4
    z_max = 3 * d // 4
    y_min = h // 4
    y_max = 3 * h // 4
    x_min = w // 4
    x_max = 3 * w // 4
    
    bbox_min = (z_min, y_min, x_min)
    bbox_max = (z_max, y_max, x_max)
    
    print(f"\n2. Defining bounding box:")
    print(f"   Z range (slices): {z_min} to {z_max}")
    print(f"   Y range (height): {y_min} to {y_max}")
    print(f"   X range (width): {x_min} to {x_max}")
    print(f"   Bbox min: {bbox_min}")
    print(f"   Bbox max: {bbox_max}")
    
    # Initialize segmenter
    print(f"\n3. Initializing MedSAM2 Segmenter...")
    segmenter = MedSAM2Segmenter(
        checkpoint_path="/home/alejandro/Documents/medsam2-gradio/MedSAM2/checkpoints/MedSAM2_latest.pt",
        config_path="configs/sam2.1_hiera_t512.yaml"
    )
    
    # Run single-view segmentation
    print(f"\n4a. Running single-view segmentation (center slice along Z-axis)...")
    segmentation_mask = segmenter.segment(
        image_3d=image_3d,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        use_largest_cc=True,
        ensemble=False
    )
    
    print(f"   Single-view segmentation complete!")
    print(f"   Output shape: {segmentation_mask.shape}")
    print(f"   Output dtype: {segmentation_mask.dtype}")
    print(f"   Segmented voxels: {np.sum(segmentation_mask)}")
    print(f"   Segmentation volume: {np.sum(segmentation_mask)} voxels")
    
    # Save single-view result
    output_path_single = "/home/alejandro/Documents/medsam2-gradio/MedSAM2/segmentation_standalone_single.nii.gz"
    print(f"\n4b. Saving single-view result to: {output_path_single}")
    result_nib = nib.Nifti1Image(segmentation_mask.astype(np.uint8), nib_img.affine)
    nib.save(result_nib, output_path_single)
    print(f"   Saved successfully!")
    
    # Run ensemble segmentation
    print(f"\n5. Running ensemble-view segmentation (3 orthogonal views with majority voting)...")
    segmentation_mask_ensemble = segmenter.segment(
        image_3d=image_3d,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        use_largest_cc=True,
        ensemble=True
    )
    
    print(f"   Ensemble segmentation complete!")
    print(f"   Output shape: {segmentation_mask_ensemble.shape}")
    print(f"   Output dtype: {segmentation_mask_ensemble.dtype}")
    print(f"   Segmented voxels: {np.sum(segmentation_mask_ensemble)}")
    print(f"   Segmentation volume: {np.sum(segmentation_mask_ensemble)} voxels")
    
    # Verify output is in same coordinate space
    print(f"\n6. Verification:")
    print(f"   Input and single-view output shapes match: {image_3d.shape == segmentation_mask.shape}")
    print(f"   Input and ensemble output shapes match: {image_3d.shape == segmentation_mask_ensemble.shape}")
    print(f"   Single-view output is binary: {np.all((segmentation_mask == 0) | (segmentation_mask == 1))}")
    print(f"   Ensemble output is binary: {np.all((segmentation_mask_ensemble == 0) | (segmentation_mask_ensemble == 1))}")
    print(f"   Single-view segmentation found: {np.any(segmentation_mask)}")
    print(f"   Ensemble segmentation found: {np.any(segmentation_mask_ensemble)}")
    
    # Compare results
    intersection = np.sum((segmentation_mask > 0) & (segmentation_mask_ensemble > 0))
    union = np.sum((segmentation_mask > 0) | (segmentation_mask_ensemble > 0))
    jaccard_index = intersection / union if union > 0 else 0
    dice_score = (2 * intersection) / (np.sum(segmentation_mask) + np.sum(segmentation_mask_ensemble)) if (np.sum(segmentation_mask) + np.sum(segmentation_mask_ensemble)) > 0 else 0
    
    print(f"\n7. Comparison between single-view and ensemble:")
    print(f"   Single-view segmented voxels: {np.sum(segmentation_mask)}")
    print(f"   Ensemble segmented voxels: {np.sum(segmentation_mask_ensemble)}")
    print(f"   Overlap voxels: {intersection}")
    print(f"   Jaccard Index: {jaccard_index:.4f}")
    print(f"   Dice Score: {dice_score:.4f}")
    
    # Save ensemble result
    output_path_ensemble = "/home/alejandro/Documents/medsam2-gradio/MedSAM2/segmentation_standalone_ensemble.nii.gz"
    print(f"\n8. Saving ensemble result to: {output_path_ensemble}")
    result_nib_ensemble = nib.Nifti1Image(segmentation_mask_ensemble.astype(np.uint8), nib_img.affine)
    nib.save(result_nib_ensemble, output_path_ensemble)
    print(f"   Saved successfully!")
    
    print("\n" + "=" * 80)
    print("Toy example completed successfully!")
    print("Results saved:")
    print(f"  - Single-view: {output_path_single}")
    print(f"  - Ensemble:    {output_path_ensemble}")
    print("=" * 80)

