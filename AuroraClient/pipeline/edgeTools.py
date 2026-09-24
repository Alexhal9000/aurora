"""
Edge detection pipeline for 3D volumes: hysteresis thresholding, morphological
closing, optional hole filling and largest-component selection, then either
preview mesh (marching cubes) or save as segmentation mask.
"""
import os
import glob
import json
import base64
import tempfile

import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

from skimage import filters, morphology, measure
from skimage.measure import marching_cubes

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

import pygltflib

from .preprocessingTools import cleanup_memory


def _load_volume_and_meta(directory, filename, edit, full_resolution):
    """
    Load volume and metadata using the same path logic as MarchingCubesView.
    Returns (volume_float32, nii_img, json_metadata, spacing_per_voxel, scan_folder).
    Volume is in NIfTI orientation (no axis swap).
    """
    if 'atlas' in filename:
        sub_path = "atlas"
        scan_folder = os.path.join(directory, "atlas")
        json_path = os.path.join(directory, "atlas", f"{filename}.json")
    else:
        sub_path = os.path.join("extracted", filename)
        scan_folder = os.path.join(directory, "extracted", filename)
        json_path = os.path.join(scan_folder, f"{filename}.json")

    with open(json_path, 'r') as jf:
        meta = json.load(jf)

    if full_resolution:
        if edit:
            nifti_file = os.path.join(directory, sub_path, edit.replace("_lossy", ""))
        else:
            nifti_file = os.path.join(directory, sub_path, f"{filename}.nii.gz")
    else:
        if edit:
            nifti_file = os.path.join(directory, sub_path, edit)
        else:
            nifti_file = os.path.join(directory, sub_path, f"{filename}_lossy.nii.gz")

    nii_img = nib.load(nifti_file)
    volume = nii_img.get_fdata().astype(np.float32)

    lossy_params = meta.get('lossy_compression', [])
    if isinstance(lossy_params, dict):
        lossy_params = [lossy_params]
    if full_resolution:
        resolution_factor = 1
    else:
        lossy_filename = os.path.basename(nifti_file)
        matching = [p for p in lossy_params if isinstance(p, dict) and p.get('filename') == lossy_filename]
        if not matching and lossy_params:
            matching = [lossy_params[-1] if isinstance(lossy_params[-1], dict) else {}]
        resolution_factor = (matching[0].get('resolution_factor', 2)) if matching else 2

    spacing = meta['voxel_size'] * resolution_factor
    return volume, nii_img, meta, spacing, scan_folder


def _run_pipeline(volume, low, high, closing_radius, fill_holes,
                  fill_area_threshold, largest_component_mode, min_size):
    """
    Run edge detection and morphology. Returns (binary_mask_bool, error_message_or_None).
    Mask is in same orientation as volume (NIfTI).
    largest_component_mode: "off" | "largest" | "above_min"
      - off: no component filtering
      - largest: keep only the single largest connected component, then remove tiny debris
      - above_min: keep all connected components with volume >= min_size (remove only smaller ones)
    """
    v_min, v_max = volume.min(), volume.max()
    if v_max <= v_min:
        return None, "Volume is empty or has uniform intensity."
    vol_norm = (volume - v_min) / (v_max - v_min)

    edges = filters.apply_hysteresis_threshold(vol_norm, low=low, high=high)
    closed = morphology.binary_closing(edges, footprint=morphology.ball(closing_radius))

    # Fill small holes (remove_small_holes) — disabled for now; closing radius usually suffices
    # if fill_holes:
    #     filled = morphology.remove_small_holes(closed, area_threshold=fill_area_threshold)
    # else:
    filled = closed

    if largest_component_mode == "largest":
        labels = measure.label(filled)
        if labels.max() == 0:
            return None, "No objects found after edge detection."
        props = measure.regionprops(labels)
        best = max(props, key=lambda r: r.area)
        filled = (labels == best.label)
        filled = morphology.remove_small_objects(filled.astype(bool), min_size=min_size)
    elif largest_component_mode == "above_min":
        filled = morphology.remove_small_objects(filled.astype(bool), min_size=min_size)

    return filled, None


# Slice preview cache (same pattern as PreviewDenoiseView)
EDGE_PREVIEW_CACHE = {}
MAX_EDGE_CACHE_SIZE = 3


def _extract_patch(volume, x, y, z, patch_size=100):
    """Extract a cubic patch centred at (x, y, z), zero-padded at boundaries."""
    half = patch_size // 2
    x_start = max(0, int(x) - half)
    x_end = min(volume.shape[0], int(x) + half)
    y_start = max(0, int(y) - half)
    y_end = min(volume.shape[1], int(y) + half)
    z_start = max(0, int(z) - half)
    z_end = min(volume.shape[2], int(z) + half)
    patch = volume[x_start:x_end, y_start:y_end, z_start:z_end].copy()
    if patch.shape != (patch_size, patch_size, patch_size):
        padded = np.zeros((patch_size, patch_size, patch_size), dtype=patch.dtype)
        px = half - (int(x) - x_start)
        py = half - (int(y) - y_start)
        pz = half - (int(z) - z_start)
        padded[px:px + patch.shape[0], py:py + patch.shape[1], pz:pz + patch.shape[2]] = patch
        patch = padded
    return patch


def _patch_to_center_slice_flat(patch_3d):
    """Return the centre-Z 2D slice as a 0-255 uint8 flat list (rot90 for display)."""
    sl = patch_3d[:, :, patch_3d.shape[2] // 2]
    s_min, s_max = sl.min(), sl.max()
    if s_max > s_min:
        sl = ((sl - s_min) / (s_max - s_min) * 255).astype(np.uint8)
    else:
        sl = sl.astype(np.uint8)
    sl = np.rot90(sl, 1)
    return sl.flatten().tolist()


class EdgeDetectionSlicePreviewView(APIView):
    """
    Fast 2D slice preview of hysteresis edge detection on a 100³ patch around (x,y,z).
    Uses the same global volume min/max for normalization as the full mesh pipeline,
    so preview and generated mesh match. Raw patch and v_min/v_max are cached per
    (scan, edit, x, y, z, resolution) to speed up repeated slider adjustments.
    """

    def post(self, request):
        global EDGE_PREVIEW_CACHE

        directory = request.data.get('directory')
        filename = request.data.get('filename')
        edit = request.data.get('edit')
        full_resolution = bool(request.data.get('full_resolution', False))
        x = float(request.data.get('x', 0))
        y = float(request.data.get('y', 0))
        z = float(request.data.get('z', 0))
        low = float(request.data.get('low', 0.10))
        high = float(request.data.get('high', 0.30))
        closing_radius = int(request.data.get('closing_radius', 3))
        # fill_holes = bool(request.data.get('fill_holes', True))
        # fill_area_threshold = int(request.data.get('fill_area_threshold', 500))

        if not directory or not filename:
            return Response({'error': 'Missing required parameters.'}, status=status.HTTP_400_BAD_REQUEST)

        cache_key = f"{filename}_{edit or 'none'}_{int(x)}_{int(y)}_{int(z)}_{'fr' if full_resolution else 'lo'}"

        if cache_key in EDGE_PREVIEW_CACHE:
            entry = EDGE_PREVIEW_CACHE[cache_key]
            raw_patch = entry['patch']
            v_min, v_max = entry['v_min'], entry['v_max']
        else:
            try:
                volume, _, _, _, _ = _load_volume_and_meta(directory, filename, edit, full_resolution)
            except Exception as e:
                return Response({'error': f'Could not load volume: {e}'}, status=status.HTTP_400_BAD_REQUEST)
            v_min, v_max = float(volume.min()), float(volume.max())
            raw_patch = _extract_patch(volume, x, y, z)
            if len(EDGE_PREVIEW_CACHE) >= MAX_EDGE_CACHE_SIZE:
                oldest = next(iter(EDGE_PREVIEW_CACHE))
                del EDGE_PREVIEW_CACHE[oldest]
            EDGE_PREVIEW_CACHE[cache_key] = {'patch': raw_patch, 'v_min': v_min, 'v_max': v_max}

        if v_max <= v_min:
            return Response({'error': 'Volume has uniform intensity.'}, status=status.HTTP_400_BAD_REQUEST)
        patch_norm = (raw_patch - v_min) / (v_max - v_min)

        edges = filters.apply_hysteresis_threshold(patch_norm, low=low, high=high)
        closed = morphology.binary_closing(edges, footprint=morphology.ball(closing_radius))
        # if fill_holes:
        #     filled = morphology.remove_small_holes(closed, area_threshold=fill_area_threshold)
        # else:
        filled = closed

        raw_slice_flat = _patch_to_center_slice_flat(raw_patch.astype(np.float32))
        edge_slice_flat = _patch_to_center_slice_flat((edges * 255).astype(np.float32))
        closed_slice_flat = _patch_to_center_slice_flat((closed.astype(np.float32)) * 255)
        filled_slice_flat = _patch_to_center_slice_flat((filled.astype(np.float32)) * 255)

        return Response({
            'raw_slice': raw_slice_flat,
            'edge_slice': edge_slice_flat,
            'closed_slice': closed_slice_flat,
            'filled_slice': filled_slice_flat,
            'patch_norm_min': round(float(patch_norm.min()), 4),
            'patch_norm_max': round(float(patch_norm.max()), 4),
            'volume_min': round(float(v_min), 4),
            'volume_max': round(float(v_max), 4),
        }, status=status.HTTP_200_OK)


def _vertices_faces_to_glb_base64(vertices, faces):
    """Encode mesh as GLB and return base64 string (same pattern as MarchingCubesView)."""
    vertex_data = vertices.astype('float32').tobytes()
    face_data = faces.astype('uint32').tobytes()
    buffer_data = vertex_data + face_data

    gltf = pygltflib.GLTF2()
    gltf.buffers.append(pygltflib.Buffer(byteLength=len(buffer_data), uri=None))
    gltf.set_binary_blob(buffer_data)
    gltf.bufferViews.extend([
        pygltflib.BufferView(buffer=0, byteOffset=0, byteLength=len(vertex_data), target=pygltflib.ARRAY_BUFFER),
        pygltflib.BufferView(buffer=0, byteOffset=len(vertex_data), byteLength=len(face_data), target=pygltflib.ELEMENT_ARRAY_BUFFER)
    ])
    gltf.accessors.extend([
        pygltflib.Accessor(bufferView=0, componentType=pygltflib.FLOAT, count=len(vertices), type=pygltflib.VEC3,
                           max=vertices.max(axis=0).tolist(), min=vertices.min(axis=0).tolist()),
        pygltflib.Accessor(bufferView=1, componentType=pygltflib.UNSIGNED_INT, count=len(faces.flatten()), type=pygltflib.SCALAR)
    ])
    primitive = pygltflib.Primitive(attributes=pygltflib.Attributes(POSITION=0), indices=1)
    gltf.meshes.append(pygltflib.Mesh(primitives=[primitive]))
    gltf.nodes.append(pygltflib.Node(mesh=0))
    gltf.scenes.append(pygltflib.Scene(nodes=[0]))
    gltf.scene = 0

    with tempfile.NamedTemporaryFile(suffix='.glb', delete=False) as tmp:
        gltf.save_binary(tmp.name)
        tmp_path = tmp.name
    try:
        with open(tmp_path, 'rb') as f:
            glb_data = base64.b64encode(f.read()).decode('utf-8')
    finally:
        try:
            os.unlink(tmp_path)
        except (OSError, PermissionError):
            pass
    return glb_data


class EdgeDetectionView(APIView):

    def post(self, request):
        directory = request.data.get('directory')
        filename = request.data.get('filename')
        edit = request.data.get('edit')
        
        was_full_res = bool(request.data.get('full_resolution', False))
        preview_only = bool(request.data.get('preview_only', True))
        force_new_label = bool(request.data.get('force_new_label', False))
        save_as_mask = bool(request.data.get('save_as_mask', False))

        # Always run on full resolution if we are applying the mask permanently
        full_resolution = True if (save_as_mask and not preview_only) else was_full_res

        low = float(request.data.get('low', 0.10))
        high = float(request.data.get('high', 0.30))
        closing_radius = int(request.data.get('closing_radius', 3))
        # fill_holes = bool(request.data.get('fill_holes', True))
        # fill_area_threshold = int(request.data.get('fill_area_threshold', 500))
        fill_holes = False
        fill_area_threshold = 500
        largest_component_mode = request.data.get('largest_component_mode', 'largest')
        if largest_component_mode not in ('off', 'largest', 'above_min'):
            largest_component_mode = 'largest'
        min_size = int(request.data.get('min_size', 1000))
        fill_mesh = bool(request.data.get('fill_mesh', True))

        if not directory or not filename:
            return Response({'error': 'Missing required parameters.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            volume, nii_img, meta, spacing, scan_folder = _load_volume_and_meta(
                directory, filename, edit, full_resolution
            )
        except Exception as e:
            return Response({'error': f'Could not load volume: {e}'}, status=status.HTTP_400_BAD_REQUEST)

        # If the user tuned the morphological parameters on the lossy preview but we are applying
        # them to the full resolution, we must scale them up accordingly.
        if save_as_mask and not preview_only and not was_full_res:
            lossy_params = meta.get('lossy_compression', [])
            if isinstance(lossy_params, dict):
                lossy_params = [lossy_params]
            
            res_factor = 2 # fallback
            if edit:
                lossy_filename = os.path.basename(edit)
                matching = [p for p in lossy_params if isinstance(p, dict) and p.get('filename') == lossy_filename]
                if not matching and lossy_params:
                    matching = [lossy_params[-1] if isinstance(lossy_params[-1], dict) else {}]
                res_factor = matching[0].get('resolution_factor', 2) if matching else 2

            closing_radius = int(closing_radius * res_factor)
            # fill_area_threshold = int(fill_area_threshold * (res_factor ** 3))
            min_size = int(min_size * (res_factor ** 3))

        filled, err = _run_pipeline(
            volume, low, high, closing_radius, fill_holes,
            fill_area_threshold, largest_component_mode, min_size
        )
        if err:
            return Response({'error': err}, status=status.HTTP_400_BAD_REQUEST)

        if save_as_mask and not preview_only:
            return self._save_mask(
                filled, nii_img, meta, scan_folder, filename,
                edit, full_resolution, force_new_label
            )

        # Build input for marching cubes (Babylon.js orientation = swap axes 0 and 2)
        if fill_mesh:
            mc_input = filled.astype(np.float32)
        else:
            v_min, v_max = volume.min(), volume.max()
            vol_norm = (volume - v_min) / (v_max - v_min) if v_max > v_min else volume
            mc_input = filters.apply_hysteresis_threshold(vol_norm, low=low, high=high).astype(np.float32)
        mc_input = np.swapaxes(mc_input, 0, 2)

        try:
            verts, faces, _, _ = marching_cubes(
                mc_input, level=0.5,
                spacing=(spacing, spacing, spacing)
            )
        except Exception as e:
            return Response({'error': f'Marching cubes failed: {e}'}, status=status.HTTP_400_BAD_REQUEST)

        cleanup_memory()

        faces = faces[:, ::-1]
        center = verts.mean(axis=0).tolist()
        verts = verts - np.array(center)
        scale_factor = float(np.max(np.abs(verts))) or 1.0
        verts = verts / scale_factor

        vertex_cap = 250000
        if len(verts) > vertex_cap:
            import fast_simplification
            decimation_factor = max(min(1, 1 - (vertex_cap / len(verts))), 0)
            verts, faces = fast_simplification.simplify(verts, faces, decimation_factor)

        glb_data = _vertices_faces_to_glb_base64(verts, faces)

        return Response({
            'gltf': glb_data,
            'scale_factor': scale_factor,
            'center': center,
        })

    def _save_mask(self, filled, nii_img, meta, scan_folder, filename,
                   edit, full_resolution, force_new_label):
        """Save filled binary mask as segmentation (new label or new file)."""
        if edit:
            clean_edit = edit.replace('.nii.gz', '').replace('_lossy', '')
            if full_resolution:
                mask_base = clean_edit
            else:
                if "_edit_" in clean_edit:
                    mask_base = clean_edit.replace("_edit_", "_lossy_edit_")
                else:
                    mask_base = f"{clean_edit}_lossy"
        else:
            mask_base = filename if full_resolution else f"{filename}_lossy"

        mask_path = os.path.join(scan_folder, f"{mask_base}.nii.mask.gz")
        mask_temp_path = mask_path.replace('.nii.mask.gz', '_mask_temp.nii.gz')

        mask_exists = os.path.isfile(mask_path)

        if mask_exists and not force_new_label:
            return Response({'mask_already_exists': True}, status=status.HTTP_200_OK)

        if mask_exists:
            os.replace(mask_path, mask_temp_path)
            try:
                existing_img = nib.load(mask_temp_path)
                # Round before astype to avoid truncating floats
                existing_array = np.round(existing_img.get_fdata()).astype(np.uint8)
                new_label = int(existing_array.max()) + 1
                new_array = existing_array.copy()
                new_array[filled] = new_label
                save_img = nib.Nifti1Image(new_array, existing_img.affine, existing_img.header)
            except Exception:
                if os.path.isfile(mask_temp_path):
                    try:
                        os.replace(mask_temp_path, mask_path)
                    except OSError:
                        pass
                raise
        else:
            new_label = 1
            new_array = filled.astype(np.uint8)
            save_img = nib.Nifti1Image(new_array, nii_img.affine, nii_img.header)

        nib.save(save_img, mask_temp_path)
        os.replace(mask_temp_path, mask_path)

        self._write_companion_mask(
            new_array, nii_img, meta, scan_folder, filename, edit,
            full_resolution, mask_base
        )

        return Response({'message': 'Mask saved.', 'label': new_label}, status=status.HTTP_200_OK)

    def _write_companion_mask(self, mask_array, nii_img, meta, scan_folder, filename, edit,
                              full_resolution, mask_base):
        """Write the mask at the other resolution (full <-> lossy) for consistency."""
        try:
            if full_resolution:
                other_base = (filename + "_lossy" + mask_base[len(filename):]) if mask_base.startswith(filename) else (mask_base.replace("_edit_", "_lossy_edit_", 1) if "_edit_" in mask_base else f"{filename}_lossy")
            else:
                other_base = mask_base.replace("_lossy_edit_", "_edit_") if "_lossy_edit_" in mask_base else mask_base.replace("_lossy", "")

            other_nii_name = other_base + ".nii.gz"
            other_path = os.path.join(scan_folder, other_nii_name)
            if not os.path.isfile(other_path):
                return
            other_img = nib.load(other_path)
            target_shape = other_img.get_fdata().shape
            zoom_factors = [t / s for t, s in zip(target_shape, mask_array.shape)]
            other_mask = zoom(mask_array.astype(np.float32), zoom_factors, order=0).astype(np.uint8)
            other_mask_path = os.path.join(scan_folder, f"{other_base}.nii.mask.gz")
            other_temp = other_mask_path.replace('.nii.mask.gz', '_mask_temp.nii.gz')
            nib.save(nib.Nifti1Image(other_mask, other_img.affine, other_img.header), other_temp)
            os.replace(other_temp, other_mask_path)
        except Exception as e:
            print(f"Warning: could not write companion mask: {e}")
