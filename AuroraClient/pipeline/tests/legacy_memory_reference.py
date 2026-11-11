"""Frozen copies of the code that the 2026-11-09 memory changes replaced.

These are the reference ("golden") implementations for test_low_memory_volume_ops: every
new lower-memory path must return exactly what these return. They are copied verbatim
from the versions before the change (backups in BHTools/.claude_backups/
memory_identical_20261109/), apart from being wrapped in small classes so they can be
called without Django. Do not edit them to match new behaviour; if a test fails, the new
code is what changed.
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter


class LegacyCleanupMesh:
    """preprocessingTools.CleanupMeshView.clean_mesh and its background helper, before 2026-11-09."""

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


class LegacyHomogenizeBackground:
    """preprocessingTools.HomogenizeBackgroundView._homogenize_background, before 2026-11-09."""

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


class LegacyRotation:
    """views.ApplyRotationView.get_tight_bbox, before 2026-11-09."""

    def get_tight_bbox(self, data, threshold):
        # Find points above threshold
        points = np.argwhere(data > threshold)
        if len(points) == 0:
            return (0, 0, 0), data.shape
        
        # Get min and max coordinates
        padding = 40
        bbox_min = np.min(points, axis=0) - padding
        bbox_max = np.max(points, axis=0) + padding
        
        # Ensure bbox is within image bounds
        bbox_min = np.maximum(bbox_min, 0)
        bbox_max = np.minimum(bbox_max, data.shape)

        # Clear memory from points
        points = None

        return np.floor(bbox_min).astype(np.int16), np.ceil(bbox_max).astype(np.int16)


def legacy_largest_voxel_connected_component(mask):
    """rigidAlignment._largest_voxel_connected_component (unchanged; used by the legacy mask)."""
    labeled, num_features = ndimage.label(np.asarray(mask, dtype=bool))
    if num_features == 0:
        return np.asarray(mask, dtype=bool)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return labeled == int(sizes.argmax())


def legacy_content_mask_from_threshold(volume, threshold, sigma=1.0):
    """rigidAlignment._content_mask_from_threshold, before 2026-11-09."""
    smoothed = gaussian_filter(np.asarray(volume, dtype=np.float32), sigma=float(sigma))
    binary = smoothed > float(threshold)
    return legacy_largest_voxel_connected_component(binary)


def legacy_min_max_normalize(image, new_min=0, new_max=1):
    """registrationTools.RegistrationTools.min_max_normalize (unchanged), as the legacy callers used it."""
    img_min = np.min(image)
    img_max = np.max(image)
    if img_max == img_min:
        return np.full_like(image, new_min, dtype=np.float32)
    result = image.astype(np.float32, copy=True)
    result -= img_min
    result /= (img_max - img_min)
    result *= (new_max - new_min)
    result += new_min
    np.clip(result, new_min, new_max, out=result)
    return result


def legacy_dino_normalize_volume(volume):
    """dinoRegRigid._normalize_volume, before 2026-11-09."""
    return legacy_min_max_normalize(
        np.asarray(volume, dtype=np.float32), new_min=-1, new_max=1
    )


def legacy_adjusted_background_input(scan_data):
    """rigidAlignment ANTs/GPU-rigid path, before 2026-11-09: a full float32 copy of the scan."""
    return scan_data.astype(np.float32)


def legacy_n4_finish(corrected_data, original_dtype):
    """N4BiasCorrectionView: final clip and cast, before 2026-11-09."""
    return np.clip(corrected_data, np.iinfo(original_dtype).min, np.iinfo(original_dtype).max).astype(original_dtype)
