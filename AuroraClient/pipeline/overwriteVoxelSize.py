"""
Override declared isotropic voxel size (mm) for an extracted subject.

Corrects spacing metadata and NIfTI affines while keeping the voxel grid
unchanged. Landmarks, guidepoints, distances, and other physical-mm assets are
left untouched — they are already stored in mm and must not be rescaled when
only the declared spacing changes.
"""

from __future__ import annotations

import glob
import json
import os
from datetime import datetime, timezone

import nibabel as nib
import numpy as np
import trimesh
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .atlas_paths import resolve_atlas_dir
from .coordinateFrames import is_preserved_mesh_metadata


DISTANCE_ARRAY_KEYS = (
    'distances',
    'snap_distance',
    'dense_distances',
)


def _resolve_scan_paths(directory, filename):
    if filename == 'atlas':
        scan_dir = resolve_atlas_dir(directory)
        json_path = os.path.join(scan_dir, 'atlas.json')
    else:
        scan_dir = os.path.join(directory, 'extracted', filename)
        json_path = os.path.join(scan_dir, f'{filename}.json')
    return scan_dir, json_path


def _scale_xyz_point(point, scale):
    if not isinstance(point, (list, tuple)) or len(point) < 3:
        return point
    try:
        return [float(point[0]) * scale, float(point[1]) * scale, float(point[2]) * scale, *list(point[3:])]
    except (TypeError, ValueError):
        return point


def _scale_landmarks_payload(payload, scale):
    if not isinstance(payload, list):
        return payload, 0

    scaled = []
    count = 0
    for item in payload:
        if isinstance(item, dict):
            entry = dict(item)
            if 'position' in entry:
                entry['position'] = _scale_xyz_point(entry['position'], scale)
                count += 1
            scaled.append(entry)
        elif isinstance(item, (list, tuple)):
            scaled.append(_scale_xyz_point(item, scale))
            count += 1
        else:
            scaled.append(item)
    return scaled, count


def _scale_guidepoints_payload(payload, scale):
    if not isinstance(payload, list):
        return payload, 0
    scaled = []
    count = 0
    for item in payload:
        if isinstance(item, (list, tuple)):
            scaled.append(_scale_xyz_point(item, scale))
            count += 1
        elif isinstance(item, dict) and 'position' in item:
            entry = dict(item)
            entry['position'] = _scale_xyz_point(entry['position'], scale)
            scaled.append(entry)
            count += 1
        else:
            scaled.append(item)
    return scaled, count


def _scale_numeric_list(values, scale):
    if not isinstance(values, list):
        return values
    out = []
    for value in values:
        try:
            out.append(float(value) * scale)
        except (TypeError, ValueError):
            out.append(value)
    return out


def _recompute_outlier_counts(distances, voxel_size):
    if distances is None or voxel_size is None:
        return None, None
    arr = np.asarray(distances, dtype=float)
    if arr.size == 0:
        return 0, 0
    return int(np.sum(arr > voxel_size)), int(np.sum(arr > voxel_size * 6))


def _refresh_distance_outliers(path, new_voxel_size):
    """Recompute outlier counts for a new voxel_size without changing distance mm values."""
    with open(path, 'r') as jf:
        data = json.load(jf)

    if isinstance(data, list):
        outliers, outliers_six = _recompute_outlier_counts(data, new_voxel_size)
        return {
            'file': os.path.basename(path),
            'outliers': outliers,
            'outliers_six': outliers_six,
            'snapped_outliers': None,
            'snapped_outliers_six': None,
            'mean_distance': float(np.mean(data)) if data else None,
            'std_distance': float(np.std(data)) if data else None,
            'snapped_mean_distance': None,
            'snapped_std_distance': None,
        }

    if not isinstance(data, dict):
        return None

    distances = data.get('distances')
    snap_distance = data.get('snap_distance')
    outliers, outliers_six = _recompute_outlier_counts(distances, new_voxel_size)
    snapped_outliers, snapped_outliers_six = _recompute_outlier_counts(snap_distance, new_voxel_size)

    return {
        'file': os.path.basename(path),
        'outliers': outliers,
        'outliers_six': outliers_six,
        'snapped_outliers': snapped_outliers,
        'snapped_outliers_six': snapped_outliers_six,
        'mean_distance': data.get('mean_distance'),
        'std_distance': data.get('std_distance'),
        'snapped_mean_distance': data.get('snapped_mean_distance'),
        'snapped_std_distance': data.get('snapped_std_distance'),
    }


def _refresh_landmarks_metadata(json_metadata, distance_summaries):
    landmarks_meta = json_metadata.get('landmarks')
    if not isinstance(landmarks_meta, dict):
        return False

    preferred = None
    edit_name = landmarks_meta.get('edit_name')
    if edit_name:
        for summary in distance_summaries:
            if summary and summary['file'].startswith(f'{edit_name}_'):
                preferred = summary
                break
    if preferred is None:
        preferred = next((s for s in distance_summaries if s), None)
    if preferred is None:
        return False

    changed = False
    for key in (
        'outliers',
        'outliers_six',
        'snapped_outliers',
        'snapped_outliers_six',
    ):
        if preferred.get(key) is not None:
            landmarks_meta[key] = preferred[key]
            changed = True

    if changed:
        json_metadata['landmarks'] = landmarks_meta
    return changed


def _is_displacement_field_path(path):
    name = os.path.basename(path).lower()
    return (
        name.endswith('_fwd.nii.gz')
        or name.endswith('_inv.nii.gz')
        or '_fwd.' in name
        or '_inv.' in name
    )


def _scale_mesh_metadata_extents(json_metadata, scale):
    mesh_metadata = json_metadata.get('mesh_metadata')
    if not isinstance(mesh_metadata, dict):
        return False
    changed = False
    for key in ('extents', 'bounds_min', 'bounds_max', 'centroid'):
        value = mesh_metadata.get(key)
        if isinstance(value, list) and value:
            try:
                mesh_metadata[key] = [float(v) * scale for v in value]
                changed = True
            except (TypeError, ValueError):
                pass
    bounds = mesh_metadata.get('bounds')
    if isinstance(bounds, list) and len(bounds) == 2:
        try:
            mesh_metadata['bounds'] = [
                [float(v) * scale for v in bounds[0]],
                [float(v) * scale for v in bounds[1]],
            ]
            changed = True
        except (TypeError, ValueError):
            pass
    if changed:
        json_metadata['mesh_metadata'] = mesh_metadata
    return changed


def _scale_quick_mesh_properties(json_metadata, scale):
    props = json_metadata.get('quick_mesh_properties')
    if not isinstance(props, dict):
        return False
    changed = False
    center = props.get('center')
    if isinstance(center, list) and center:
        try:
            props['center'] = [float(v) * scale for v in center]
            changed = True
        except (TypeError, ValueError):
            pass
    scale_factor = props.get('scale_factor')
    if isinstance(scale_factor, (int, float)):
        props['scale_factor'] = float(scale_factor) * scale
        changed = True
    if changed:
        json_metadata['quick_mesh_properties'] = props
    return changed


def _update_nifti_affine(path, scale, scale_vector_data=False):
    img = nib.load(path)
    affine = np.array(img.affine, dtype=np.float64, copy=True)
    affine[:3, :] *= scale
    header = img.header.copy()
    if scale_vector_data:
        data = np.asanyarray(img.dataobj)
        data = np.asarray(data, dtype=np.float64) * scale
        out = nib.Nifti1Image(data.astype(img.get_data_dtype(), copy=False), affine, header)
    else:
        out = nib.Nifti1Image(img.dataobj, affine, header)
    out.set_data_dtype(img.get_data_dtype())
    nib.save(out, path)


def _scale_vertices_npz(path, scale):
    with np.load(path, allow_pickle=False) as npz:
        keys = list(npz.files)
        payload = {key: npz[key] for key in keys}
    if 'vertices' not in payload:
        return False
    payload['vertices'] = np.asarray(payload['vertices'], dtype=np.float64) * scale
    np.savez(path, **payload)
    return True


def _scale_ply(path, scale):
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        geometries = [
            geom for geom in mesh.geometry.values()
            if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) > 0
        ]
        if not geometries:
            return False
        mesh = trimesh.util.concatenate(geometries)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
        return False
    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) * scale
    mesh.export(path)
    return True


def _scale_distance_file_arrays(path, scale):
    with open(path, 'r') as jf:
        data = json.load(jf)

    if isinstance(data, list):
        with open(path, 'w') as jf:
            json.dump(_scale_numeric_list(data, scale), jf, indent=4)
        return True

    if not isinstance(data, dict):
        return False

    changed = False
    for key in DISTANCE_ARRAY_KEYS:
        if key in data and isinstance(data[key], list):
            data[key] = _scale_numeric_list(data[key], scale)
            changed = True
    for key in ('mean_distance', 'std_distance', 'snapped_mean_distance', 'snapped_std_distance'):
        if key in data and isinstance(data[key], (int, float)):
            data[key] = float(data[key]) * scale
            changed = True

    if changed:
        with open(path, 'w') as jf:
            json.dump(data, jf, indent=4)
    return changed


def _scale_mm_assets(scan_dir, json_path, json_metadata, scale):
    """
    Scale physical-mm sidecar assets only (not voxel_size / NIfTI affines).

    Used to undo a previous mistaken rescale of landmarks and related files.
    """
    updated = {
        'landmark_files': [],
        'guidepoint_files': [],
        'distance_files': [],
        'ply_files': [],
        'vertices_npz': [],
        'displacement_fields': [],
        'metadata_fields': [],
    }

    if _scale_mesh_metadata_extents(json_metadata, scale):
        updated['metadata_fields'].append('mesh_metadata')
    if _scale_quick_mesh_properties(json_metadata, scale):
        updated['metadata_fields'].append('quick_mesh_properties')

    landmarks_meta = json_metadata.get('landmarks')
    if isinstance(landmarks_meta, dict):
        meta_changed = False
        for key in ('mean_distance', 'std_distance', 'snapped_mean_distance', 'snapped_std_distance'):
            if key in landmarks_meta and isinstance(landmarks_meta[key], (int, float)):
                landmarks_meta[key] = float(landmarks_meta[key]) * scale
                meta_changed = True
        if meta_changed:
            json_metadata['landmarks'] = landmarks_meta
            updated['metadata_fields'].append('landmarks')

    for path in sorted(glob.glob(os.path.join(scan_dir, '*.json'))):
        basename = os.path.basename(path)
        if basename == os.path.basename(json_path):
            continue
        if basename.endswith('_landmarks.json'):
            with open(path, 'r') as jf:
                payload = json.load(jf)
            scaled, count = _scale_landmarks_payload(payload, scale)
            with open(path, 'w') as jf:
                json.dump(scaled, jf, indent=4)
            updated['landmark_files'].append({'file': basename, 'count': count})
        elif 'guidepoints' in basename and basename.endswith('.json'):
            with open(path, 'r') as jf:
                payload = json.load(jf)
            scaled, count = _scale_guidepoints_payload(payload, scale)
            with open(path, 'w') as jf:
                json.dump(scaled, jf, indent=4)
            updated['guidepoint_files'].append({'file': basename, 'count': count})
        elif basename.endswith('_landmark_distances.json'):
            if _scale_distance_file_arrays(path, scale):
                updated['distance_files'].append(basename)

    for path in sorted(glob.glob(os.path.join(scan_dir, '*.ply'))):
        basename = os.path.basename(path)
        try:
            if _scale_ply(path, scale):
                updated['ply_files'].append(basename)
        except Exception as exc:
            print(f'Warning: could not scale PLY {basename}: {exc}')

    for path in sorted(glob.glob(os.path.join(scan_dir, '*_vertices.npz'))):
        basename = os.path.basename(path)
        try:
            if _scale_vertices_npz(path, scale):
                updated['vertices_npz'].append(basename)
        except Exception as exc:
            print(f'Warning: could not scale vertices cache {basename}: {exc}')

    for path in sorted(glob.glob(os.path.join(scan_dir, '*.nii.gz'))):
        if not _is_displacement_field_path(path):
            continue
        basename = os.path.basename(path)
        try:
            # Revert displacement *vectors* only; affine stays with current voxel_size.
            img = nib.load(path)
            data = np.asarray(np.asanyarray(img.dataobj), dtype=np.float64) * scale
            out = nib.Nifti1Image(data.astype(img.get_data_dtype(), copy=False), img.affine, img.header.copy())
            out.set_data_dtype(img.get_data_dtype())
            nib.save(out, path)
            updated['displacement_fields'].append(basename)
        except Exception as exc:
            print(f'Warning: could not scale displacement field {basename}: {exc}')

    return updated


def _entry_scaled_mm_assets(entry):
    """Old overrides scaled mm assets; new ones set scaled_mm_assets=False."""
    if not isinstance(entry, dict):
        return False
    if entry.get('mm_assets_reverted'):
        return False
    if entry.get('scaled_mm_assets') is False:
        return False
    # Missing key => legacy behavior that scaled mm assets.
    return True


def _revert_legacy_mm_rescales(scan_dir, json_path, json_metadata):
    history = json_metadata.get('voxel_size_overrides')
    if not isinstance(history, list) or not history:
        return 1.0, {}

    inverse = 1.0
    any_reverted = False
    for entry in history:
        if not _entry_scaled_mm_assets(entry):
            continue
        try:
            scale_factor = float(entry.get('scale_factor', 1.0))
        except (TypeError, ValueError):
            entry['mm_assets_reverted'] = True
            continue
        if abs(scale_factor - 1.0) >= 1e-15:
            inverse /= scale_factor
            any_reverted = True
        entry['mm_assets_reverted'] = True

    if not any_reverted or abs(inverse - 1.0) < 1e-15:
        json_metadata['voxel_size_overrides'] = history
        return 1.0, {}

    repaired = _scale_mm_assets(scan_dir, json_path, json_metadata, inverse)
    json_metadata['voxel_size_overrides'] = history
    return inverse, repaired


def apply_voxel_size_override(directory, filename, new_voxel_size):
    scan_dir, json_path = _resolve_scan_paths(directory, filename)
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f'Metadata not found: {json_path}')
    if not os.path.isdir(scan_dir):
        raise FileNotFoundError(f'Scan directory not found: {scan_dir}')

    with open(json_path, 'r') as jf:
        json_metadata = json.load(jf)

    if is_preserved_mesh_metadata(json_metadata):
        raise ValueError('Preserved mesh subjects do not use voxel spacing; edit mesh scale instead.')

    try:
        old_voxel_size = float(json_metadata.get('voxel_size'))
    except (TypeError, ValueError):
        raise ValueError('Existing voxel_size is missing or invalid.')

    if not np.isfinite(old_voxel_size) or old_voxel_size <= 0:
        raise ValueError('Existing voxel_size must be a positive finite number.')
    if not np.isfinite(new_voxel_size) or new_voxel_size <= 0:
        raise ValueError('New voxel_size must be a positive finite number.')

    # Undo any prior mistaken landmark/guidepoint/mm-asset rescale from older builds.
    repair_scale, repaired = _revert_legacy_mm_rescales(scan_dir, json_path, json_metadata)

    if abs(new_voxel_size - old_voxel_size) < 1e-15:
        if repaired:
            # Refresh outlier thresholds against the (unchanged) voxel size.
            distance_summaries = []
            for path in sorted(glob.glob(os.path.join(scan_dir, '*_landmark_distances.json'))):
                summary = _refresh_distance_outliers(path, old_voxel_size)
                if summary:
                    distance_summaries.append(summary)
            _refresh_landmarks_metadata(json_metadata, distance_summaries)
            with open(json_path, 'w') as jf:
                json.dump(json_metadata, jf, indent=4)
        return {
            'voxel_size': old_voxel_size,
            'previous_voxel_size': old_voxel_size,
            'scale_factor': 1.0,
            'unchanged': not bool(repaired),
            'repaired_mm_assets_scale': repair_scale if repaired else 1.0,
            'repaired': repaired,
            'updated': {'metadata_fields': ['voxel_size_overrides'] if repaired else []},
            'landmarks': json_metadata.get('landmarks'),
        }

    scale = float(new_voxel_size) / float(old_voxel_size)
    updated = {
        'nifti_affines': [],
        'metadata_fields': [],
        'distance_outlier_files': [],
    }

    json_metadata['voxel_size'] = float(new_voxel_size)
    updated['metadata_fields'].append('voxel_size')

    # Keep acquisition history and alignment_previous_voxel_size as-is.
    # Only the active declared spacing and volume affines change.

    history = json_metadata.get('voxel_size_overrides')
    if not isinstance(history, list):
        history = []
    history.append({
        'previous_voxel_size': old_voxel_size,
        'voxel_size': float(new_voxel_size),
        'scale_factor': scale,
        'scaled_mm_assets': False,
        'preserved_mm_assets': True,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    })
    json_metadata['voxel_size_overrides'] = history
    updated['metadata_fields'].append('voxel_size_overrides')

    # --- NIfTI affines only (do not rescale displacement vector magnitudes) ---
    for nifti_path in sorted(glob.glob(os.path.join(scan_dir, '*.nii.gz'))):
        basename = os.path.basename(nifti_path)
        try:
            _update_nifti_affine(nifti_path, scale, scale_vector_data=False)
            updated['nifti_affines'].append(basename)
        except Exception as exc:
            print(f'Warning: could not update NIfTI affine for {basename}: {exc}')

    # Landmarks / guidepoints / meshes stay in their existing mm coordinates.
    # Only refresh outlier counts that depend on the voxel_size threshold.
    distance_summaries = []
    for path in sorted(glob.glob(os.path.join(scan_dir, '*_landmark_distances.json'))):
        summary = _refresh_distance_outliers(path, new_voxel_size)
        if summary:
            distance_summaries.append(summary)
            updated['distance_outlier_files'].append(summary['file'])
    if _refresh_landmarks_metadata(json_metadata, distance_summaries):
        updated['metadata_fields'].append('landmarks')

    with open(json_path, 'w') as jf:
        json.dump(json_metadata, jf, indent=4)

    return {
        'voxel_size': float(new_voxel_size),
        'previous_voxel_size': old_voxel_size,
        'scale_factor': scale,
        'unchanged': False,
        'filename': filename,
        'preserved_mm_assets': True,
        'repaired_mm_assets_scale': repair_scale if repaired else 1.0,
        'repaired': repaired,
        'updated': updated,
        'lossy_compression': json_metadata.get('lossy_compression', []),
        'landmarks': json_metadata.get('landmarks'),
    }


class OverwriteVoxelSizeView(APIView):
    """
    POST /overwrite-voxel-size/

    Body: { directory, filename, voxel_size }

    Overrides the subject's isotropic voxel_size (mm) and updates NIfTI affines.
    Landmarks, guidepoints, and other mm-space sidecars are preserved as-is.
    """

    def post(self, request, *args, **kwargs):
        directory = request.data.get('directory')
        filename = request.data.get('filename')
        voxel_size_raw = request.data.get('voxel_size')

        if not directory or not filename:
            return Response(
                {'error': 'directory and filename are required'},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            new_voxel_size = float(voxel_size_raw)
        except (TypeError, ValueError):
            return Response(
                {'error': 'voxel_size must be a number'},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            result = apply_voxel_size_override(directory, filename, new_voxel_size)
        except FileNotFoundError as exc:
            return Response({'error': str(exc)}, status=status.HTTP_404_NOT_FOUND)
        except ValueError as exc:
            return Response({'error': str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as exc:
            return Response({'error': str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response(result, status=status.HTTP_200_OK)
