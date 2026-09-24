import base64
import scipy.sparse as sp
import glob
import json
import os
import shutil
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import fast_simplification
import nibabel as nib
import numpy as np
import open3d as o3d
import pygltflib
import trimesh
from tps import ThinPlateSpline
from scipy.ndimage import binary_dilation
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from skimage import measure
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

from .ALPACA import ALPACA
from .atlas_paths import resolve_atlas_dir
from .batch_flag_filter import apply_flag_filter, load_flagged_subject_names, normalize_flag_filter_value
from .linkedScans import filter_out_linked_children
from .coordinateFrames import (
    build_mesh_reference_prism,
    derive_mesh_reference_voxel_size,
    is_preserved_mesh_metadata,
    mesh_local_to_shared_mm,
    mesh_reference_prism_corner_mm,
    partition_voxel_and_preserved_mesh_scans,
    mesh_reorigin_offset_after_mm_crop,
    reorigin_mesh_coords_after_mm_crop,
    resolve_mesh_reference_prism,
    resolve_mesh_reference_voxel_size,
)
from .builtinTextureTools import (
    invalidate_display_mesh_cache,
    load_uv_sidecar,
    propagate_uvs_after_edit,
    remap_face_uvs_for_decimated_mesh,
)
from .meshGridTools import (
    PreservedMeshEditWriter,
    patch_subject_json,
    proximity_weld_trimesh,
    resolve_latest_preserved_mesh_edit,
    sanitize_trimesh_faces,
    strip_ephemeral_scan_metadata,
)
from .registrationTools import RegistrationTools


alpaca = ALPACA()

_SLICE_WORKER_VERTICES = None
_SLICE_WORKER_FACES = None
_SLICE_WORKER_PLANE_SETS = None


def _rasterize_mesh_to_volume_mask(vertices, faces, shape, spacing, fill_watertight=False):
    """
    Place mesh geometry onto a voxel grid in corner-origin physical mm.
    Vertices must already match the NIfTI array axis order (Z, Y, X).
    """
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    raster_vertex_cap = 300000
    if len(mesh.vertices) > raster_vertex_cap:
        decimation_factor = max(min(1, 1 - (raster_vertex_cap / len(mesh.vertices))), 0)
        simplified_vertices, simplified_faces = fast_simplification.simplify(
            np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.faces, dtype=np.uint32),
            decimation_factor,
        )
        mesh = trimesh.Trimesh(
            vertices=simplified_vertices,
            faces=simplified_faces,
            process=False,
        )

    voxel_grid = mesh.voxelized(pitch=float(spacing))
    if fill_watertight and mesh.is_watertight:
        try:
            voxel_grid = voxel_grid.fill()
        except Exception:
            pass

    points = np.asarray(voxel_grid.points, dtype=np.float64)
    if len(points) == 0:
        raise ValueError("Mesh voxelization produced no occupied voxels")

    indices = np.floor(points / float(spacing) + 1e-6).astype(np.int64)
    shape_arr = np.asarray(shape, dtype=np.int64)
    valid = np.all((indices >= 0) & (indices < shape_arr), axis=1)
    indices = indices[valid]
    mask = np.zeros(shape, dtype=bool)
    if len(indices) > 0:
        mask[indices[:, 0], indices[:, 1], indices[:, 2]] = True
    if not np.any(mask):
        raise ValueError("Rasterized mesh mask is empty in the voxel grid")
    return mask


def build_temporary_mesh_reference_volume(directory, reference, reference_metadata=None):
    """
    Build an in-memory voxel grid for a preserved-mesh reference without persisting
    it as a NIfTI edit. Uses the same virtual prism as rigid alignment so voxel
    subjects aligned to the mesh reference share shape and spacing.
    """
    if reference_metadata is None:
        json_path = os.path.join(directory, "extracted", reference, f"{reference}.json")
        with open(json_path, "r") as jf:
            reference_metadata = json.load(jf)

    if not is_preserved_mesh_metadata(reference_metadata):
        raise ValueError(
            f"build_temporary_mesh_reference_volume requires a preserved mesh reference, got {reference}"
        )

    scan_dir = os.path.join(directory, "extracted", reference)
    _, ply_path = resolve_latest_preserved_mesh_edit(scan_dir, reference)
    if not ply_path or not os.path.isfile(ply_path):
        mesh_file = reference_metadata.get("mesh_file") or f"{reference}.ply"
        ply_path = os.path.join(scan_dir, mesh_file)
    if not os.path.isfile(ply_path):
        raise FileNotFoundError(f"Preserved mesh PLY not found for reference {reference}: {ply_path}")

    mesh_obj = trimesh.load(ply_path, process=False)
    if isinstance(mesh_obj, trimesh.Scene):
        geometries = [
            geom for geom in mesh_obj.geometry.values()
            if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) > 0
        ]
        if not geometries:
            raise ValueError(f"No mesh geometry found in reference PLY for {reference}")
        mesh_obj = trimesh.util.concatenate(geometries)

    vertices_local = np.asarray(mesh_obj.vertices, dtype=np.float64)
    faces = np.asarray(mesh_obj.faces, dtype=np.int64)
    mesh_metadata = reference_metadata.get("mesh_metadata")

    prism = resolve_mesh_reference_prism(vertices_local, mesh_metadata, directory)
    shape = tuple(int(axis) for axis in prism["shape_voxels"])
    spacing = float(prism["voxel_size"])
    vertices_shared = mesh_local_to_shared_mm(
        vertices_local,
        mesh_metadata,
        prism_vertices=vertices_local,
        voxel_size=spacing,
    )

    mask = _rasterize_mesh_to_volume_mask(
        vertices_shared,
        faces,
        shape,
        spacing,
        fill_watertight=bool(mesh_obj.is_watertight),
    )

    threshold = int(reference_metadata.get("threshold", 1) or 1)
    volume = np.zeros(shape, dtype=np.float32)
    volume[mask] = float(threshold)

    affine = np.eye(4, dtype=np.float64)
    affine[0, 0] = spacing
    affine[1, 1] = spacing
    affine[2, 2] = spacing

    return volume, (spacing, spacing, spacing), affine, threshold, mask


def _parse_bool_request_flag(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "on")
    return bool(value)


def _slice_worker_init(vertices, faces, plane_sets):
    global _SLICE_WORKER_VERTICES, _SLICE_WORKER_FACES, _SLICE_WORKER_PLANE_SETS
    _SLICE_WORKER_VERTICES = vertices
    _SLICE_WORKER_FACES = faces
    _SLICE_WORKER_PLANE_SETS = plane_sets


def _slice_clip_polygon_to_halfspace(polygon, plane_origin, plane_normal, epsilon=1e-9):
    if not polygon:
        return []

    clipped = []
    previous = polygon[-1]
    previous_distance = float(np.dot(previous - plane_origin, plane_normal))
    previous_inside = previous_distance >= -epsilon

    for current in polygon:
        current_distance = float(np.dot(current - plane_origin, plane_normal))
        current_inside = current_distance >= -epsilon

        if current_inside != previous_inside:
            denominator = previous_distance - current_distance
            if abs(denominator) > epsilon:
                t = previous_distance / denominator
                clipped.append(previous + t * (current - previous))

        if current_inside:
            clipped.append(current)

        previous = current
        previous_distance = current_distance
        previous_inside = current_inside

    deduped = []
    for point in clipped:
        if not deduped or not np.allclose(point, deduped[-1], atol=epsilon):
            deduped.append(point)
    if len(deduped) > 1 and np.allclose(deduped[0], deduped[-1], atol=epsilon):
        deduped.pop()
    return deduped


def _slice_subtract_planes_from_polygon(polygon, planes):
    remaining_inside = [polygon]
    outside_pieces = []

    for plane_origin, plane_normal in planes:
        next_inside = []
        for candidate in remaining_inside:
            inside_piece = _slice_clip_polygon_to_halfspace(candidate, plane_origin, plane_normal)
            outside_piece = _slice_clip_polygon_to_halfspace(candidate, plane_origin, -plane_normal)
            if len(outside_piece) >= 3:
                outside_pieces.append(outside_piece)
            if len(inside_piece) >= 3:
                next_inside.append(inside_piece)
        remaining_inside = next_inside
        if not remaining_inside:
            break

    return outside_pieces


def _slice_cull_face_chunk(vertices, face_chunk, planes, epsilon=1e-8):
    origins = np.array([p[0] for p in planes], dtype=np.float64)
    normals = np.array([p[1] for p in planes], dtype=np.float64)
    triangles = vertices[face_chunk]

    dists = np.sum(
        (triangles[np.newaxis, :, :, :] - origins[:, np.newaxis, np.newaxis, :])
        * normals[:, np.newaxis, np.newaxis, :],
        axis=3,
    )
    is_inside = dists >= -epsilon

    face_all_outside_plane = ~np.any(is_inside, axis=2)
    face_entirely_outside_wedge = np.any(face_all_outside_plane, axis=0)

    face_all_inside_plane = np.all(is_inside, axis=2)
    face_entirely_inside_wedge = np.all(face_all_inside_plane, axis=0)

    needs_clipping = ~(face_entirely_outside_wedge | face_entirely_inside_wedge)
    return face_entirely_outside_wedge, face_entirely_inside_wedge, needs_clipping


def _slice_subtract_worker(face_range):
    start, end = face_range
    vertices = _SLICE_WORKER_VERTICES
    faces = _SLICE_WORKER_FACES
    plane_sets = _SLICE_WORKER_PLANE_SETS
    face_chunk = faces[start:end]

    chunk_count = len(face_chunk)
    face_outside_all = np.ones(chunk_count, dtype=bool)
    face_inside_any = np.zeros(chunk_count, dtype=bool)
    wedge_needs_clipping = []

    for planes in plane_sets:
        out_w, in_w, clip_w = _slice_cull_face_chunk(vertices, face_chunk, planes)
        face_outside_all &= out_w
        face_inside_any |= in_w
        wedge_needs_clipping.append(clip_w)

    untouched = face_outside_all
    dropped = face_inside_any
    needs_processing = ~(untouched | dropped)

    kept_polygons = []
    if np.any(untouched):
        kept_polygons.extend(vertices[face_chunk[untouched]])

    for local_face_idx in np.nonzero(needs_processing)[0]:
        current_pieces = [[vertices[idx] for idx in face_chunk[local_face_idx]]]
        for wedge_idx, planes in enumerate(plane_sets):
            if not wedge_needs_clipping[wedge_idx][local_face_idx]:
                continue

            next_pieces = []
            for piece in current_pieces:
                next_pieces.extend(_slice_subtract_planes_from_polygon(piece, planes))
            current_pieces = next_pieces
            if not current_pieces:
                break

        kept_polygons.extend(current_pieces)

    return kept_polygons


class MeshEditMixin:
    def _subject_dir(self, directory, filename):
        return os.path.join(directory, "extracted", filename)

    def _metadata_path(self, directory, filename):
        return os.path.join(self._subject_dir(directory, filename), f"{filename}.json")

    def _load_metadata(self, directory, filename):
        json_path = self._metadata_path(directory, filename)
        with open(json_path, "r") as jf:
            return json_path, json.load(jf)

    def _mesh_edit_files(self, subject_dir, filename):
        pattern = os.path.join(subject_dir, f"{filename}_edit_*_*.ply")
        return glob.glob(pattern)

    def _next_edit_number(self, subject_dir, filename):
        edit_number = 0
        while glob.glob(os.path.join(subject_dir, f"{filename}_edit_{edit_number}_*.ply")):
            edit_number += 1
        return edit_number

    def _next_nifti_edit_number(self, subject_dir, filename):
        edit_number = 0
        while glob.glob(os.path.join(subject_dir, f"{filename}_edit_{edit_number}_*.nii.gz")):
            edit_number += 1
        return edit_number

    def _source_nifti_path(self, directory, filename, edit):
        subject_dir = self._subject_dir(directory, filename)
        if edit:
            full_res_path = os.path.join(subject_dir, os.path.basename(edit).replace("_lossy", ""))
            if os.path.isfile(full_res_path):
                return full_res_path
            edit_path = os.path.join(subject_dir, os.path.basename(edit))
            if os.path.isfile(edit_path):
                return edit_path.replace("_lossy", "")
        return os.path.join(subject_dir, f"{filename}.nii.gz")

    def _nifti_stem(self, nifti_path):
        basename = os.path.basename(nifti_path)
        if basename.endswith(".nii.gz"):
            return basename[:-7]
        return os.path.splitext(basename)[0]

    def _latest_edit_file(self, subject_dir, filename):
        edit_files = self._mesh_edit_files(subject_dir, filename)
        if not edit_files:
            return None

        def edit_number(path):
            basename = os.path.basename(path)
            try:
                return int(basename.split("_edit_")[1].split("_")[0])
            except (IndexError, ValueError):
                return -1

        return max(edit_files, key=edit_number)

    def _source_mesh_path(self, directory, filename, edit, metadata):
        subject_dir = self._subject_dir(directory, filename)
        if edit == "latest":
            latest = self._latest_edit_file(subject_dir, filename)
            if latest:
                return latest
        if edit:
            edit_path = os.path.join(subject_dir, os.path.basename(edit))
            if os.path.isfile(edit_path):
                return edit_path
        mesh_file = metadata.get("mesh_file") or f"{filename}.ply"
        return os.path.join(subject_dir, mesh_file)

    def _edit_stem(self, filename, edit):
        if edit:
            edit = os.path.basename(edit)
            for suffix in (".nii.gz", ".ply"):
                if edit.endswith(suffix):
                    return edit[:-len(suffix)]
            return edit
        return filename

    def _mesh_stem(self, mesh_path):
        basename = os.path.basename(mesh_path)
        return basename[:-4] if basename.endswith(".ply") else os.path.splitext(basename)[0]

    def _save_mesh_edit(
        self,
        mesh,
        output_path,
        source_mesh=None,
        source_ply_path=None,
        uv_lookup_vertex_offset=None,
    ):
        saved = PreservedMeshEditWriter.save(mesh, output_path)
        if source_mesh is not None and source_ply_path:
            propagate_uvs_after_edit(
                source_mesh,
                source_ply_path,
                mesh,
                output_path,
                target_vertex_offset=uv_lookup_vertex_offset,
            )
        invalidate_display_mesh_cache(output_path)
        return saved

    def _load_mesh(self, path):
        loaded = trimesh.load(path, process=False)
        if isinstance(loaded, trimesh.Scene):
            geometries = [
                geom for geom in loaded.geometry.values()
                if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) > 0
            ]
            if not geometries:
                raise ValueError("No mesh geometry found.")
            loaded = trimesh.util.concatenate(geometries)
        if not isinstance(loaded, trimesh.Trimesh):
            raise ValueError("Unsupported mesh content.")
        if loaded.vertices is None or len(loaded.vertices) == 0 or loaded.faces is None or len(loaded.faces) == 0:
            raise ValueError("Mesh has no vertices or faces.")
        return loaded

    def _append_edit_metadata(self, json_path, metadata, edit_filename, operation=None, details=None):
        """
        Persist scan JSON after a new PLY/volume edit is written.

        Edit files on disk are the source of truth; ``edits`` is not stored in JSON.
        ``operation`` and other ``details`` are accepted for call-site compatibility.

        Reloads from disk before writing so a volume path that just called
        ``save_as_lossy_nifti`` does not lose the appended lossy_compression row.
        """
        details = details if isinstance(details, dict) else {}
        updates = {}
        source = metadata if isinstance(metadata, dict) else {}
        on_disk = {}
        if os.path.isfile(json_path):
            try:
                with open(json_path, 'r') as jf:
                    loaded = json.load(jf)
                if isinstance(loaded, dict):
                    on_disk = loaded
            except (OSError, json.JSONDecodeError):
                on_disk = {}
        mesh_summary = details.get("mesh_summary")
        if isinstance(mesh_summary, dict):
            existing_mesh_metadata = on_disk.get("mesh_metadata")
            if not isinstance(existing_mesh_metadata, dict):
                existing_mesh_metadata = source.get("mesh_metadata")
            if not isinstance(existing_mesh_metadata, dict):
                existing_mesh_metadata = {}
            updates["mesh_metadata"] = {
                **existing_mesh_metadata,
                **mesh_summary,
            }
        mesh_elastic_settings = details.get("mesh_elastic_settings")
        if isinstance(mesh_elastic_settings, dict):
            updates["mesh_elastic_settings"] = mesh_elastic_settings
            # Keep report helpers that look for elastic_settings consistent on meshes.
            if not isinstance(source.get("elastic_settings"), dict):
                updates["elastic_settings"] = {
                    "registration_method": "mesh_nricp",
                    **{k: mesh_elastic_settings.get(k) for k in ("mode", "steps", "gamma", "eps") if k in mesh_elastic_settings},
                }
        mesh_cleanup_settings = details.get("mesh_cleanup_settings")
        if isinstance(mesh_cleanup_settings, dict):
            updates["mesh_cleanup_settings"] = mesh_cleanup_settings
        mesh_weld_settings = details.get("mesh_weld_settings")
        if isinstance(mesh_weld_settings, dict):
            updates["mesh_weld_settings"] = mesh_weld_settings
        mesh_scale_calibration = details.get("mesh_scale_calibration")
        if isinstance(mesh_scale_calibration, dict):
            updates["mesh_scale_calibration"] = mesh_scale_calibration
        mesh_scale = details.get("mesh_scale")
        if isinstance(mesh_scale, dict):
            updates["mesh_scale"] = mesh_scale
        if updates:
            patch_subject_json(json_path, updates, metadata=metadata if isinstance(metadata, dict) else None)

    def _mesh_summary(self, mesh):
        return {
            "vertex_count": int(len(mesh.vertices)),
            "face_count": int(len(mesh.faces)),
            "bounds": np.asarray(mesh.bounds, dtype=float).tolist(),
            "extents": np.asarray(mesh.extents, dtype=float).tolist(),
            "is_watertight": bool(mesh.is_watertight),
        }

    def _load_landmark_rows_for_stem(self, subject_dir, stem):
        landmarks_path = os.path.join(subject_dir, f"{stem}_landmarks.json")
        if not os.path.exists(landmarks_path):
            return None, None

        with open(landmarks_path, "r") as jf:
            landmarks_data = json.load(jf)

        if isinstance(landmarks_data, list):
            rows = []
            for item in landmarks_data:
                if isinstance(item, dict) and "position" in item:
                    rows.append(dict(item))
                else:
                    rows.append({"position": list(item), "landmark_type": "main"})
            return landmarks_path, rows

        if isinstance(landmarks_data, dict):
            rows = []
            for _, item in sorted(landmarks_data.items(), key=lambda pair: (0, int(pair[0])) if str(pair[0]).isdigit() else (1, str(pair[0]))):
                if isinstance(item, dict):
                    rows.append({
                        "position": item.get("position"),
                        "landmark_type": item.get("landmark_type", "main"),
                    })
                else:
                    rows.append({"position": item, "landmark_type": "main"})
            return landmarks_path, rows

        return landmarks_path, []

    def _write_landmarks_for_edit(self, subject_dir, edit_stem, landmarks, mesh):
        landmarks_path = os.path.join(subject_dir, f"{edit_stem}_landmarks.json")
        with open(landmarks_path, "w") as jf:
            json.dump(landmarks, jf, indent=4)

        if not landmarks:
            return landmarks_path, None

        positions = np.asarray([row["position"] for row in landmarks], dtype=np.float64)
        distances = alpaca.calculate_landmark_distances(
            positions,
            np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.faces, dtype=np.int64),
        )
        distances_path = os.path.join(subject_dir, f"{edit_stem}_landmark_distances.json")
        with open(distances_path, "w") as jf:
            json.dump({
                "distances": distances.tolist(),
                "snap_distance": None,
            }, jf, indent=4)

        return landmarks_path, distances_path

    def _write_landmark_rows_for_edit(self, subject_dir, edit_stem, landmarks):
        landmarks_path = os.path.join(subject_dir, f"{edit_stem}_landmarks.json")
        with open(landmarks_path, "w") as jf:
            json.dump(landmarks, jf, indent=4)
        return landmarks_path

    def _copy_landmark_distances_for_edit(self, subject_dir, source_stem, edit_stem, kept_indices=None):
        source_path = os.path.join(subject_dir, f"{source_stem}_landmark_distances.json")
        if not os.path.isfile(source_path):
            return None

        destination_path = os.path.join(subject_dir, f"{edit_stem}_landmark_distances.json")
        if kept_indices is None:
            shutil.copy(source_path, destination_path)
            return destination_path

        with open(source_path, "r") as jf:
            distances_data = json.load(jf)

        if isinstance(distances_data, dict):
            filtered_distances = {}
            for key, value in distances_data.items():
                if isinstance(value, list):
                    filtered_distances[key] = [
                        value[idx] for idx in kept_indices
                        if idx < len(value)
                    ]
                else:
                    filtered_distances[key] = value
        elif isinstance(distances_data, list):
            filtered_distances = [
                distances_data[idx] for idx in kept_indices
                if idx < len(distances_data)
            ]
        else:
            filtered_distances = distances_data

        with open(destination_path, "w") as jf:
            json.dump(filtered_distances, jf, indent=4)
        return destination_path

    def _mesh_cleanup_skip_reason(self, subject_dir, filename):
        latest = self._latest_edit_file(subject_dir, filename)
        if not latest:
            return None
        basename = os.path.basename(latest).lower()
        if "elastic" in basename:
            return "Elastic registration detected, skipping cleanup"
        if "cleaned" in basename:
            return "Latest edit is already cleaned"
        return None

    def cleanup_preserved_mesh(
        self,
        directory,
        filename,
        edit="latest",
        use_island_volume_threshold=False,
        min_island_volume_percent=1.0,
    ):
        """
        Remove disconnected mesh components from a preserved PLY subject.
        Returns a dict with status ('success' | 'skipped' | 'error'), message, and optional edit.
        """
        min_island_volume_percent = max(0.1, min(30.0, float(min_island_volume_percent or 1.0)))
        subject_dir = self._subject_dir(directory, filename)
        json_path, metadata = self._load_metadata(directory, filename)

        skip_reason = self._mesh_cleanup_skip_reason(subject_dir, filename)
        if skip_reason:
            return {"status": "skipped", "message": skip_reason}

        source_path = self._source_mesh_path(directory, filename, edit, metadata)
        mesh = self._load_mesh(source_path)

        edges = mesh.edges_unique
        vertex_count = len(mesh.vertices)
        if len(edges) == 0 or vertex_count == 0:
            return {"status": "skipped", "message": "Empty mesh"}

        data = np.ones(len(edges), dtype=bool)
        adj = sp.coo_matrix((data, (edges[:, 0], edges[:, 1])), shape=(vertex_count, vertex_count))
        adj = adj + adj.T

        n_components, vertex_labels = sp.csgraph.connected_components(adj, directed=False)
        face_labels = vertex_labels[mesh.faces[:, 0]]
        face_areas = mesh.area_faces
        component_stats = []

        for i in range(n_components):
            mask = face_labels == i
            face_count = int(np.sum(mask))
            if face_count == 0:
                continue
            component_stats.append({
                "index": i,
                "vertex_count": int(np.sum(vertex_labels == i)),
                "face_count": face_count,
                "surface_area": float(np.sum(face_areas[mask])),
            })

        if len(component_stats) <= 1:
            try:
                self._append_edit_metadata(json_path, metadata, None, "cleaned", {
                    "mesh_cleanup_settings": {
                        "mode": "face_percent_threshold" if use_island_volume_threshold else "largest_component",
                        "geometry": "mesh",
                        "use_island_volume_threshold": bool(use_island_volume_threshold),
                        "min_island_volume_percent": (
                            float(min_island_volume_percent) if use_island_volume_threshold else None
                        ),
                        "cleanup_performed": False,
                        "reason": "no_disconnected_components",
                        "source_edit": os.path.basename(source_path),
                        "components_found": len(component_stats),
                    },
                })
            except Exception as meta_err:
                print(f"Warning: could not save mesh_cleanup_settings (no-op): {meta_err}")
            return {
                "status": "skipped",
                "message": "No disconnected mesh components removed",
                "component_count": len(component_stats),
                "cleanup_performed": False,
                "reason": "no_disconnected_components",
            }

        if use_island_volume_threshold:
            total_area = sum(stat["surface_area"] for stat in component_stats)
            min_area = total_area * (min_island_volume_percent / 100.0) if total_area > 0 else 0
            keep_indices = [
                stat["index"] for stat in component_stats
                if stat["surface_area"] >= min_area
            ]
            if not keep_indices:
                keep_indices = [max(component_stats, key=lambda stat: stat["surface_area"])["index"]]
        else:
            keep_indices = [max(component_stats, key=lambda stat: stat["surface_area"])["index"]]

        if len(keep_indices) == len(component_stats):
            try:
                self._append_edit_metadata(json_path, metadata, None, "cleaned", {
                    "mesh_cleanup_settings": {
                        "mode": "face_percent_threshold" if use_island_volume_threshold else "largest_component",
                        "geometry": "mesh",
                        "use_island_volume_threshold": bool(use_island_volume_threshold),
                        "min_island_volume_percent": (
                            float(min_island_volume_percent) if use_island_volume_threshold else None
                        ),
                        "cleanup_performed": False,
                        "reason": "all_components_kept",
                        "source_edit": os.path.basename(source_path),
                        "components_found": len(component_stats),
                    },
                })
            except Exception as meta_err:
                print(f"Warning: could not save mesh_cleanup_settings (no-op): {meta_err}")
            return {
                "status": "skipped",
                "message": "All mesh components were kept",
                "component_count": len(component_stats),
                "cleanup_performed": False,
                "reason": "all_components_kept",
            }

        keep_mask = np.isin(face_labels, keep_indices)
        kept_faces = mesh.faces[keep_mask]
        visual = mesh.visual.copy() if hasattr(mesh, "visual") and mesh.visual is not None else None
        cleaned_mesh = trimesh.Trimesh(
            vertices=mesh.vertices.copy(),
            faces=kept_faces,
            visual=visual,
            process=False,
        )
        cleaned_mesh.remove_unreferenced_vertices()

        edit_number = self._next_edit_number(subject_dir, filename)
        edit_filename = f"{filename}_edit_{edit_number}_cleaned.ply"
        output_path = os.path.join(subject_dir, edit_filename)
        self._save_mesh_edit(
            cleaned_mesh,
            output_path,
            source_mesh=mesh,
            source_ply_path=source_path,
        )
        edit_stem = self._mesh_stem(edit_filename)
        source_stem = self._mesh_stem(source_path)

        removed_components = len(component_stats) - len(keep_indices)
        landmark_details = {}
        source_landmarks_path, source_landmarks = self._load_landmark_rows_for_stem(subject_dir, source_stem)
        if source_landmarks is not None:
            kept_landmarks = []
            kept_landmark_indices = []
            face_centers = np.asarray(mesh.triangles_center, dtype=np.float64)
            face_tree = cKDTree(face_centers) if len(face_centers) > 0 else None

            for landmark_index, landmark in enumerate(source_landmarks):
                try:
                    position = np.asarray(landmark.get("position"), dtype=np.float64)
                except (TypeError, ValueError):
                    continue
                if position.shape != (3,) or not np.all(np.isfinite(position)) or face_tree is None:
                    continue
                _, nearest_face_index = face_tree.query(position)
                if keep_mask[int(nearest_face_index)]:
                    kept_landmark_indices.append(landmark_index)
                    kept_landmarks.append({
                        **landmark,
                        "position": position.tolist(),
                    })

            landmarks_path = self._write_landmark_rows_for_edit(subject_dir, edit_stem, kept_landmarks)
            distances_path = self._copy_landmark_distances_for_edit(
                subject_dir,
                source_stem,
                edit_stem,
                kept_indices=kept_landmark_indices,
            )
            landmark_details = {
                "source_landmarks": os.path.basename(source_landmarks_path),
                "landmarks_file": os.path.basename(landmarks_path),
                "landmark_distances_file": os.path.basename(distances_path) if distances_path else None,
                "landmark_count": len(kept_landmarks),
            }

        self._append_edit_metadata(json_path, metadata, edit_filename, "cleaned", {
            "source": os.path.basename(source_path),
            "cleanup_mode": "face_percent_threshold" if use_island_volume_threshold else "largest_component",
            "min_component_face_percent": min_island_volume_percent if use_island_volume_threshold else None,
            "min_component_faces": None,
            "mesh_cleanup_settings": {
                "mode": "face_percent_threshold" if use_island_volume_threshold else "largest_component",
                "geometry": "mesh",
                "use_island_volume_threshold": bool(use_island_volume_threshold),
                "min_island_volume_percent": (
                    float(min_island_volume_percent) if use_island_volume_threshold else None
                ),
                "cleanup_performed": True,
                "reason": "components_removed",
                "source_edit": os.path.basename(source_path),
                "components_found": len(component_stats),
            },
            "component_summary": {
                "initial_components": len(component_stats),
                "kept_components": len(keep_indices),
                "removed_components": removed_components,
                "component_stats": component_stats,
            },
            **landmark_details,
            "mesh_summary": self._mesh_summary(cleaned_mesh),
        })

        return {
            "status": "success",
            "message": "Mesh cleanup applied successfully",
            "edit": edit_filename,
            "mode": "preserved_mesh",
            "component_summary": {
                "initial_components": len(component_stats),
                "kept_components": len(keep_indices),
                "removed_components": removed_components,
                "component_stats": component_stats,
            },
            "mesh_summary": self._mesh_summary(cleaned_mesh),
        }

    def _mesh_weld_skip_reason(self, subject_dir, filename):
        latest = self._latest_edit_file(subject_dir, filename)
        if not latest:
            return None
        basename = os.path.basename(latest).lower()
        if "welded" in basename:
            return (
                "Latest edit is already welded — no new file was written. "
                "Delete the *_welded.ply edit (or the whole extracted subject folder) "
                "if you need to re-weld from the original mesh."
            )
        return None

    @staticmethod
    def _format_permission_error(exc):
        text = str(exc)
        lowered = text.lower()
        if (
            isinstance(exc, PermissionError)
            or getattr(exc, "winerror", None) == 5
            or "access is denied" in lowered
            or "permission denied" in lowered
        ):
            return (
                f"Could not write mesh files (access denied): {text}. "
                "Close Explorer/preview windows for that folder, check the drive is writable, "
                "and try copying the project off USB/cloud sync if it persists."
            )
        return text

    def weld_preserved_mesh(self, directory, filename, edit="latest", distance_factor=1.0):
        """
        Proximity-weld near-touching vertices on a preserved PLY subject.
        Returns a dict with status ('success' | 'skipped' | 'error'), message, and optional edit.
        """
        distance_factor = max(0.25, min(5.0, float(distance_factor or 1.0)))
        subject_dir = self._subject_dir(directory, filename)
        try:
            dir_listing = sorted(os.listdir(subject_dir)) if os.path.isdir(subject_dir) else []
        except OSError as listing_err:
            dir_listing = [f"<listdir failed: {listing_err}>"]
        edit_files = [n for n in dir_listing if "_edit_" in n and n.lower().endswith(".ply")]
        print(
            f"Weld [{filename}]: start "
            f"name_repr={filename!r} dir={subject_dir!r} "
            f"edit_arg={edit!r} distance_factor={distance_factor:g} "
            f"dir_exists={os.path.isdir(subject_dir)} "
            f"edit_plys={edit_files or []} "
            f"has_welded={any('welded' in n.lower() for n in edit_files)} "
            f"trimesh={getattr(trimesh, '__version__', '?')}",
            flush=True,
        )
        try:
            json_path, metadata = self._load_metadata(directory, filename)
        except Exception as exc:
            message = self._format_permission_error(exc)
            print(f"Weld [{filename}]: failed to load metadata — {message}", flush=True)
            return {"status": "error", "message": message}

        if not is_preserved_mesh_metadata(metadata):
            message = "Not a preserved mesh subject"
            print(f"Weld [{filename}]: skipped — {message}", flush=True)
            return {"status": "skipped", "message": message}

        skip_reason = self._mesh_weld_skip_reason(subject_dir, filename)
        if skip_reason:
            print(f"Weld [{filename}]: skipped — {skip_reason}", flush=True)
            return {"status": "skipped", "message": skip_reason}

        source_path = self._source_mesh_path(directory, filename, edit, metadata)
        print(
            f"Weld [{filename}]: loading source={source_path!r} "
            f"exists={os.path.isfile(source_path)} "
            f"(distance_factor={distance_factor:g})",
            flush=True,
        )
        try:
            mesh = self._load_mesh(source_path)
        except Exception as exc:
            message = (
                f"Failed to load mesh {os.path.basename(source_path)}: "
                f"{self._format_permission_error(exc)}"
            )
            print(f"Weld [{filename}]: error — {message}", flush=True)
            return {"status": "error", "message": message}

        print(
            f"Weld [{filename}]: loaded {len(mesh.vertices)} vertices / {len(mesh.faces)} faces",
            flush=True,
        )

        try:
            welded_mesh, weld_stats = proximity_weld_trimesh(mesh, distance_factor=distance_factor)
        except Exception as exc:
            print(f"Weld [{filename}]: proximity weld failed — {exc}", flush=True)
            return {"status": "error", "message": str(exc)}

        weld_settings = {
            "mode": "proximity",
            "geometry": "mesh",
            "distance_factor": float(distance_factor),
            "median_edge_length": weld_stats.get("median_edge_length"),
            "epsilon": weld_stats.get("epsilon"),
            "weld_performed": bool(weld_stats.get("weld_performed")),
            "reason": weld_stats.get("reason"),
            "source_edit": os.path.basename(source_path),
            "components_before": weld_stats.get("components_before"),
            "components_after": weld_stats.get("components_after"),
            "vertices_before": weld_stats.get("vertices_before"),
            "vertices_after": weld_stats.get("vertices_after"),
            "faces_before": weld_stats.get("faces_before"),
            "faces_after": weld_stats.get("faces_after"),
            "pairs_merged": weld_stats.get("pairs_merged"),
        }

        if not weld_stats.get("weld_performed"):
            reason = weld_stats.get("reason") or "no_topology_change"
            print(
                f"Weld [{filename}]: no-op "
                f"(reason={reason}, components={weld_stats.get('components_before')}, "
                f"eps={weld_stats.get('epsilon')})",
                flush=True,
            )
            try:
                self._append_edit_metadata(json_path, metadata, None, "welded", {
                    "mesh_weld_settings": weld_settings,
                })
            except Exception as meta_err:
                print(f"Warning: could not save mesh_weld_settings (no-op): {meta_err}", flush=True)
                return {
                    "status": "skipped",
                    "message": (
                        f"No near-touching vertices found to weld ({reason}). "
                        f"Also could not update subject JSON: {self._format_permission_error(meta_err)}. "
                        "Try increasing the distance factor."
                    ),
                    "weld_performed": False,
                    "reason": reason,
                    "mesh_weld_settings": weld_settings,
                }
            return {
                "status": "skipped",
                "message": (
                    f"No near-touching vertices found to weld ({reason}). "
                    "No new edit was written — try increasing the distance factor."
                ),
                "weld_performed": False,
                "reason": reason,
                "mesh_weld_settings": weld_settings,
            }

        edit_number = self._next_edit_number(subject_dir, filename)
        edit_filename = f"{filename}_edit_{edit_number}_welded.ply"
        output_path = os.path.join(subject_dir, edit_filename)
        print(f"Weld [{filename}]: saving {edit_filename}", flush=True)
        try:
            self._save_mesh_edit(
                welded_mesh,
                output_path,
                source_mesh=mesh,
                source_ply_path=source_path,
            )
        except Exception as exc:
            message = (
                f"Weld computed but failed to save {edit_filename}: "
                f"{self._format_permission_error(exc)}"
            )
            print(f"Weld [{filename}]: error — {message}", flush=True)
            return {"status": "error", "message": message}
        edit_stem = self._mesh_stem(edit_filename)
        source_stem = self._mesh_stem(source_path)

        landmark_details = {}
        source_landmarks_path, source_landmarks = self._load_landmark_rows_for_stem(subject_dir, source_stem)
        if source_landmarks is not None:
            try:
                landmarks_path = self._write_landmark_rows_for_edit(subject_dir, edit_stem, source_landmarks)
                distances_path = self._copy_landmark_distances_for_edit(subject_dir, source_stem, edit_stem)
                landmark_details = {
                    "source_landmarks": os.path.basename(source_landmarks_path),
                    "landmarks_file": os.path.basename(landmarks_path),
                    "landmark_distances_file": os.path.basename(distances_path) if distances_path else None,
                    "landmark_count": len(source_landmarks),
                }
            except Exception as landmark_err:
                print(f"Warning: could not copy landmarks after weld for {filename}: {landmark_err}", flush=True)

        try:
            self._append_edit_metadata(json_path, metadata, edit_filename, "welded", {
                "source": os.path.basename(source_path),
                "mesh_weld_settings": weld_settings,
                **landmark_details,
                "mesh_summary": self._mesh_summary(welded_mesh),
            })
        except Exception as meta_err:
            print(f"Warning: welded PLY saved but metadata update failed for {filename}: {meta_err}", flush=True)
            return {
                "status": "success",
                "message": (
                    f"Mesh welded to {edit_filename} "
                    f"({weld_stats.get('components_before')} → {weld_stats.get('components_after')} components), "
                    f"but subject JSON could not be updated: {self._format_permission_error(meta_err)}"
                ),
                "edit": edit_filename,
                "mode": "preserved_mesh",
                "weld_performed": True,
                "warning": self._format_permission_error(meta_err),
                "mesh_weld_settings": weld_settings,
                "mesh_summary": self._mesh_summary(welded_mesh),
            }

        print(
            f"Weld [{filename}]: success — wrote {edit_filename} "
            f"({weld_stats.get('components_before')} → {weld_stats.get('components_after')} components)",
            flush=True,
        )
        return {
            "status": "success",
            "message": (
                f"Mesh welded successfully "
                f"({weld_stats.get('components_before')} → {weld_stats.get('components_after')} components)"
            ),
            "edit": edit_filename,
            "mode": "preserved_mesh",
            "weld_performed": True,
            "mesh_weld_settings": weld_settings,
            "mesh_summary": self._mesh_summary(welded_mesh),
        }

    def _clip_polygon_to_halfspace(self, polygon, plane_origin, plane_normal, epsilon=1e-9):
        if not polygon:
            return []

        clipped = []
        previous = polygon[-1]
        previous_distance = float(np.dot(previous - plane_origin, plane_normal))
        previous_inside = previous_distance >= -epsilon

        for current in polygon:
            current_distance = float(np.dot(current - plane_origin, plane_normal))
            current_inside = current_distance >= -epsilon

            if current_inside != previous_inside:
                denominator = previous_distance - current_distance
                if abs(denominator) > epsilon:
                    t = previous_distance / denominator
                    clipped.append(previous + t * (current - previous))

            if current_inside:
                clipped.append(current)

            previous = current
            previous_distance = current_distance
            previous_inside = current_inside

        deduped = []
        for point in clipped:
            if not deduped or not np.allclose(point, deduped[-1], atol=epsilon):
                deduped.append(point)
        if len(deduped) > 1 and np.allclose(deduped[0], deduped[-1], atol=epsilon):
            deduped.pop()
        return deduped

    def _clip_mesh_to_bounds(self, mesh, mins, maxs):
        mins = np.asarray(mins, dtype=np.float64)
        maxs = np.asarray(maxs, dtype=np.float64)
        source_vertices = np.asarray(mesh.vertices, dtype=np.float64)
        source_faces = np.asarray(mesh.faces, dtype=np.int64)
        if source_faces.size == 0:
            raise ValueError("Crop removed all faces; no mesh edit was saved.")

        # Classify faces in bulk so we only run the expensive polygon clipper on the
        # relatively few triangles that actually cross a crop plane.
        face_verts = source_vertices[source_faces]
        below_min = (face_verts < mins).all(axis=1)
        above_max = (face_verts > maxs).all(axis=1)
        fully_outside = (below_min | above_max).any(axis=1)
        fully_inside = (
            (face_verts >= mins).all(axis=(1, 2)) &
            (face_verts <= maxs).all(axis=(1, 2))
        )
        needs_clip = ~fully_outside & ~fully_inside

        inside_faces = source_faces[fully_inside]
        clip_faces = source_faces[needs_clip]
        print(
            f"clip_mesh_to_bounds: {len(source_faces)} faces "
            f"({fully_inside.sum()} inside, {fully_outside.sum()} outside, {needs_clip.sum()} boundary)"
        )

        if inside_faces.size == 0 and clip_faces.size == 0:
            raise ValueError("Crop removed all faces; no mesh edit was saved.")

        result_vertices_parts = []
        result_faces_parts = []
        vertex_offset = 0

        if inside_faces.size:
            used_vertex_indices = np.unique(inside_faces.ravel())
            index_map = np.full(len(source_vertices), -1, dtype=np.int64)
            index_map[used_vertex_indices] = np.arange(len(used_vertex_indices))
            result_vertices_parts.append(source_vertices[used_vertex_indices])
            result_faces_parts.append(index_map[inside_faces])
            vertex_offset = len(used_vertex_indices)

        if clip_faces.size:
            crop_planes = (
                (np.array([mins[0], 0.0, 0.0], dtype=np.float64), np.array([1.0, 0.0, 0.0], dtype=np.float64)),
                (np.array([maxs[0], 0.0, 0.0], dtype=np.float64), np.array([-1.0, 0.0, 0.0], dtype=np.float64)),
                (np.array([0.0, mins[1], 0.0], dtype=np.float64), np.array([0.0, 1.0, 0.0], dtype=np.float64)),
                (np.array([0.0, maxs[1], 0.0], dtype=np.float64), np.array([0.0, -1.0, 0.0], dtype=np.float64)),
                (np.array([0.0, 0.0, mins[2]], dtype=np.float64), np.array([0.0, 0.0, 1.0], dtype=np.float64)),
                (np.array([0.0, 0.0, maxs[2]], dtype=np.float64), np.array([0.0, 0.0, -1.0], dtype=np.float64)),
            )
            clipped_vertices = []
            clipped_faces = []

            for face in clip_faces:
                polygon = [source_vertices[index].copy() for index in face]
                for plane_origin, plane_normal in crop_planes:
                    polygon = self._clip_polygon_to_halfspace(polygon, plane_origin, plane_normal)
                    if len(polygon) < 3:
                        break

                if len(polygon) < 3:
                    continue

                base_index = len(clipped_vertices)
                clipped_vertices.extend(polygon)
                for polygon_index in range(1, len(polygon) - 1):
                    clipped_faces.append([
                        base_index,
                        base_index + polygon_index,
                        base_index + polygon_index + 1,
                    ])

            if clipped_faces:
                clipped_vertices_arr = np.asarray(clipped_vertices, dtype=np.float64)
                clipped_faces_arr = np.asarray(clipped_faces, dtype=np.int64) + vertex_offset
                result_vertices_parts.append(clipped_vertices_arr)
                result_faces_parts.append(clipped_faces_arr)

        if not result_faces_parts:
            raise ValueError("Crop removed all faces; no mesh edit was saved.")

        clipped_mesh = trimesh.Trimesh(
            vertices=np.vstack(result_vertices_parts),
            faces=np.vstack(result_faces_parts) if len(result_faces_parts) > 1 else result_faces_parts[0],
            process=False,
        )
        clipped_mesh.merge_vertices()
        clipped_mesh.remove_unreferenced_vertices()
        return clipped_mesh

    def _mesh_from_polygons(self, polygons, empty_error="Slice removed all faces; no mesh edit was saved."):
        clipped_vertices = []
        clipped_faces = []
        for polygon in polygons:
            if len(polygon) < 3:
                continue
            base_index = len(clipped_vertices)
            clipped_vertices.extend(polygon)
            for polygon_index in range(1, len(polygon) - 1):
                clipped_faces.append([
                    base_index,
                    base_index + polygon_index,
                    base_index + polygon_index + 1,
                ])

        if not clipped_faces:
            raise ValueError(empty_error)

        clipped_mesh = trimesh.Trimesh(
            vertices=np.asarray(clipped_vertices, dtype=np.float64),
            faces=np.asarray(clipped_faces, dtype=np.int64),
            process=False,
        )
        clipped_mesh.merge_vertices()
        clipped_mesh.remove_unreferenced_vertices()
        return clipped_mesh

    def _volume_planes_from_rays(self, volume):
        rays = volume.get("rays") if isinstance(volume, dict) else None
        if not isinstance(rays, list) or len(rays) < 3:
            raise ValueError("Each slicer volume requires at least three projected rays.")

        origins = []
        directions = []
        for ray in rays:
            origin = np.asarray([
                ray.get("origin", {}).get("x"),
                ray.get("origin", {}).get("y"),
                ray.get("origin", {}).get("z"),
            ], dtype=np.float64)
            direction = np.asarray([
                ray.get("direction", {}).get("x"),
                ray.get("direction", {}).get("y"),
                ray.get("direction", {}).get("z"),
            ], dtype=np.float64)
            if origin.shape != (3,) or direction.shape != (3,) or not np.all(np.isfinite(origin)) or not np.all(np.isfinite(direction)):
                raise ValueError("Invalid slicer ray coordinates.")
            direction_norm = np.linalg.norm(direction)
            if direction_norm < 1e-12:
                raise ValueError("Invalid zero-length slicer ray direction.")
            origins.append(origin)
            directions.append(direction / direction_norm)

        center_origin = np.mean(origins, axis=0)
        center_direction = np.mean(directions, axis=0)
        center_direction_norm = np.linalg.norm(center_direction)
        if center_direction_norm < 1e-12:
            raise ValueError("Slicer projection rays are degenerate.")
        center_direction = center_direction / center_direction_norm
        center_probe = center_origin + center_direction

        planes = []
        for index in range(len(directions)):
            d1 = directions[index]
            d2 = directions[(index + 1) % len(directions)]
            origin = origins[index]
            normal = np.cross(d1, d2)
            normal_norm = np.linalg.norm(normal)
            if normal_norm < 1e-12:
                continue
            normal = normal / normal_norm
            if float(np.dot(center_probe - origin, normal)) < 0:
                normal = -normal
            planes.append((origin, normal))

        if len(planes) < 3:
            raise ValueError("Slicer projection is too narrow or degenerate.")
        return planes

    def _polygon_area_2d(self, points):
        area = 0.0
        for index, point in enumerate(points):
            next_point = points[(index + 1) % len(points)]
            area += (point[0] * next_point[1]) - (next_point[0] * point[1])
        return area / 2.0

    def _point_in_triangle_2d(self, point, a, b, c, epsilon=1e-9):
        def signed_area(p1, p2, p3):
            return (
                p1[0] * (p2[1] - p3[1]) +
                p2[0] * (p3[1] - p1[1]) +
                p3[0] * (p1[1] - p2[1])
            )

        total = signed_area(a, b, c)
        if abs(total) < epsilon:
            return False
        w1 = signed_area(point, b, c) / total
        w2 = signed_area(a, point, c) / total
        w3 = signed_area(a, b, point) / total
        return w1 > epsilon and w2 > epsilon and w3 > epsilon

    def _triangulate_slicer_outline_indices(self, points):
        if not isinstance(points, list) or len(points) < 3:
            raise ValueError("Slicer outline requires at least three points.")

        normalized_points = []
        for point in points:
            if isinstance(point, dict):
                normalized_points.append((float(point.get("x")), float(point.get("y"))))
            else:
                normalized_points.append((float(point[0]), float(point[1])))

        indices = list(range(len(normalized_points)))
        if self._polygon_area_2d(normalized_points) < 0:
            indices.reverse()

        triangles = []
        guard = 0
        while len(indices) > 3 and guard < len(normalized_points) * len(normalized_points):
            guard += 1
            clipped = False
            for cursor in range(len(indices)):
                prev_index = indices[(cursor - 1) % len(indices)]
                current_index = indices[cursor]
                next_index = indices[(cursor + 1) % len(indices)]
                a = normalized_points[prev_index]
                b = normalized_points[current_index]
                c = normalized_points[next_index]
                cross = ((b[0] - a[0]) * (c[1] - b[1])) - ((b[1] - a[1]) * (c[0] - b[0]))
                if cross <= 1e-9:
                    continue
                has_point_inside = any(
                    idx not in (prev_index, current_index, next_index)
                    and self._point_in_triangle_2d(normalized_points[idx], a, b, c)
                    for idx in indices
                )
                if has_point_inside:
                    continue
                triangles.append([prev_index, current_index, next_index])
                indices.pop(cursor)
                clipped = True
                break
            if not clipped:
                break

        if len(indices) == 3:
            triangles.append([indices[0], indices[1], indices[2]])

        if not triangles:
            raise ValueError("Could not triangulate slicer outline.")
        return triangles

    def _operation_plane_sets(self, operation):
        volumes = operation.get("volumes")
        if isinstance(volumes, list) and volumes:
            return [self._volume_planes_from_rays(volume) for volume in volumes]

        outline_points = operation.get("outline_points")
        outline_rays = operation.get("outline_rays")
        if not isinstance(outline_rays, list) or not isinstance(outline_points, list) or len(outline_rays) != len(outline_points):
            raise ValueError("Slicer operation requires either volumes or compact outline rays.")

        triangles = self._triangulate_slicer_outline_indices(outline_points)
        plane_sets = []
        for triangle in triangles:
            plane_sets.append(self._volume_planes_from_rays({
                "rays": [outline_rays[index] for index in triangle],
            }))
        return plane_sets

    def _cull_faces_for_planes(self, vertices, faces, planes, epsilon=1e-8):
        origins = np.array([p[0] for p in planes], dtype=np.float64)
        normals = np.array([p[1] for p in planes], dtype=np.float64)
        
        diff = vertices[np.newaxis, :, :] - origins[:, np.newaxis, :]
        dists = np.sum(diff * normals[:, np.newaxis, :], axis=2)
        
        is_inside = dists >= -epsilon
        face_inside = is_inside[:, faces]
        
        face_all_outside_plane = ~np.any(face_inside, axis=2)
        face_entirely_outside_wedge = np.any(face_all_outside_plane, axis=0)
        
        face_all_inside_plane = np.all(face_inside, axis=2)
        face_entirely_inside_wedge = np.all(face_all_inside_plane, axis=0)
        
        needs_clipping = ~(face_entirely_outside_wedge | face_entirely_inside_wedge)
        
        return face_entirely_outside_wedge, face_entirely_inside_wedge, needs_clipping

    def _clip_mesh_to_planes(self, mesh, planes, empty_error="Slicer isolate removed all faces."):
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        
        face_outside, face_inside, needs_clipping = self._cull_faces_for_planes(vertices, faces, planes)
        clipped_polygons = []
        
        if np.any(face_inside):
            clipped_polygons.extend(vertices[faces[face_inside]])
            
        for face in faces[needs_clipping]:
            polygon = [vertices[idx] for idx in face]
            for plane_origin, plane_normal in planes:
                polygon = self._clip_polygon_to_halfspace(polygon, plane_origin, plane_normal)
                if len(polygon) < 3:
                    break
            if len(polygon) >= 3:
                clipped_polygons.append(polygon)
                
        return self._mesh_from_polygons(clipped_polygons, empty_error=empty_error)

    def _subtract_planes_from_polygon(self, polygon, planes):
        remaining_inside = [polygon]
        outside_pieces = []

        for plane_origin, plane_normal in planes:
            next_inside = []
            for candidate in remaining_inside:
                inside_piece = self._clip_polygon_to_halfspace(candidate, plane_origin, plane_normal)
                outside_piece = self._clip_polygon_to_halfspace(candidate, plane_origin, -plane_normal)
                if len(outside_piece) >= 3:
                    outside_pieces.append(outside_piece)
                if len(inside_piece) >= 3:
                    next_inside.append(inside_piece)
            remaining_inside = next_inside
            if not remaining_inside:
                break

        return outside_pieces

    def _subtract_multiple_convex_volumes(self, mesh, list_of_plane_sets):
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)

        num_faces = len(faces)
        worker_count = max(1, (os.cpu_count() or 2) - 1)
        if worker_count > 1:
            chunk_size = int(np.ceil(num_faces / worker_count))
            face_ranges = [
                (start, min(start + chunk_size, num_faces))
                for start in range(0, num_faces, chunk_size)
            ]
            print(
                "[apply-mesh-slice] "
                f"parallel delete: faces={num_faces} wedges={len(list_of_plane_sets)} workers={worker_count}",
                flush=True,
            )
            kept_polygons = []
            with ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_slice_worker_init,
                initargs=(vertices, faces, list_of_plane_sets),
            ) as executor:
                futures = [executor.submit(_slice_subtract_worker, face_range) for face_range in face_ranges]
                for completed, future in enumerate(as_completed(futures), start=1):
                    kept_polygons.extend(future.result())
                    print(
                        "[apply-mesh-slice] "
                        f"parallel delete chunk {completed}/{len(futures)} complete",
                        flush=True,
                    )

            return self._mesh_from_polygons(
                kept_polygons,
                empty_error="Slicer delete removed all faces; no mesh edit was saved.",
            )

        face_outside_all = np.ones(num_faces, dtype=bool)
        face_inside_any = np.zeros(num_faces, dtype=bool)
        wedge_needs_clipping = []
        
        for planes in list_of_plane_sets:
            out_w, in_w, clip_w = self._cull_faces_for_planes(vertices, faces, planes)
            face_outside_all &= out_w
            face_inside_any |= in_w
            wedge_needs_clipping.append(clip_w)
            
        untouched = face_outside_all
        dropped = face_inside_any
        needs_processing = ~(untouched | dropped)
        
        kept_polygons = []
        if np.any(untouched):
            kept_polygons.extend(vertices[faces[untouched]])
            
        for face_idx in np.nonzero(needs_processing)[0]:
            current_pieces = [[vertices[idx] for idx in faces[face_idx]]]
            for w_idx, planes in enumerate(list_of_plane_sets):
                if not wedge_needs_clipping[w_idx][face_idx]:
                    continue
                    
                next_pieces = []
                for piece in current_pieces:
                    next_pieces.extend(self._subtract_planes_from_polygon(piece, planes))
                current_pieces = next_pieces
                if not current_pieces:
                    break
                    
            kept_polygons.extend(current_pieces)
            
        return self._mesh_from_polygons(
            kept_polygons,
            empty_error="Slicer delete removed all faces; no mesh edit was saved.",
        )

    def _apply_slice_operations_to_mesh(self, mesh, operations):
        working_mesh = mesh.copy()
        for operation in operations:
            if not isinstance(operation, dict):
                raise ValueError("Each slicer operation must be an object.")
            action = operation.get("action")
            if action not in ("isolate", "delete"):
                raise ValueError("Slicer operations require action isolate/delete.")
            plane_sets = self._operation_plane_sets(operation)

            if action == "isolate":
                isolated_meshes = []
                for planes in plane_sets:
                    try:
                        isolated_meshes.append(self._clip_mesh_to_planes(working_mesh, planes))
                    except ValueError:
                        continue
                if not isolated_meshes:
                    raise ValueError("Slicer isolate removed all faces; no mesh edit was saved.")
                working_mesh = isolated_meshes[0] if len(isolated_meshes) == 1 else trimesh.util.concatenate(isolated_meshes)
                working_mesh.merge_vertices()
                working_mesh.remove_unreferenced_vertices()
            else:
                working_mesh = self._subtract_multiple_convex_volumes(working_mesh, plane_sets)

        return working_mesh

    def _point_inside_volume_planes(self, point, planes, epsilon=1e-7):
        point = np.asarray(point, dtype=np.float64)
        return all(float(np.dot(point - plane_origin, plane_normal)) >= -epsilon for plane_origin, plane_normal in planes)

    def _apply_slice_operations_to_landmarks(self, landmarks, operations):
        if landmarks is None:
            return None, None

        kept_landmarks = list(landmarks)
        kept_indices = list(range(len(landmarks)))
        for operation in operations:
            action = operation.get("action")
            plane_sets = self._operation_plane_sets(operation)
            next_landmarks = []
            next_indices = []
            for landmark, original_index in zip(kept_landmarks, kept_indices):
                position = np.asarray(landmark.get("position"), dtype=np.float64)
                inside_any = position.shape == (3,) and any(
                    self._point_inside_volume_planes(position, planes) for planes in plane_sets
                )
                if (action == "isolate" and inside_any) or (action == "delete" and not inside_any):
                    next_landmarks.append(landmark)
                    next_indices.append(original_index)
            kept_landmarks = next_landmarks
            kept_indices = next_indices

        return kept_landmarks, kept_indices

    def _physical_mm_from_zyx_indices(self, coords_zyx, voxel_size):
        vx = float(voxel_size)
        xx = coords_zyx[:, 2].astype(np.float64) * vx
        yy = coords_zyx[:, 1].astype(np.float64) * vx
        zz = coords_zyx[:, 0].astype(np.float64) * vx
        return xx, yy, zz

    def _coords_inside_any_wedge(self, coords_zyx, voxel_size, plane_sets, epsilon=1e-7):
        if coords_zyx.size == 0:
            return np.zeros(0, dtype=bool)
        xx, yy, zz = self._physical_mm_from_zyx_indices(coords_zyx, voxel_size)
        inside_any = np.zeros(len(coords_zyx), dtype=bool)
        for planes in plane_sets:
            inside_wedge = np.ones(len(coords_zyx), dtype=bool)
            for plane_origin, plane_normal in planes:
                origin = np.asarray(plane_origin, dtype=np.float64)
                normal = np.asarray(plane_normal, dtype=np.float64)
                dist = (
                    (xx - origin[0]) * normal[0]
                    + (yy - origin[1]) * normal[1]
                    + (zz - origin[2]) * normal[2]
                )
                inside_wedge &= dist >= -epsilon
            inside_any |= inside_wedge
        return inside_any

    def _volume_has_foreground(self, volume, background_value, threshold=None):
        if threshold is not None:
            return bool(np.any(volume >= threshold))
        return bool(np.any(volume != background_value))

    def _apply_slice_mask_to_coords(self, data, coords_zyx, inside, action, clear_value):
        if coords_zyx.size == 0:
            return
        if action == "isolate":
            remove = ~inside
        else:
            remove = inside
        if not np.any(remove):
            return
        removed = coords_zyx[remove]
        data[removed[:, 0], removed[:, 1], removed[:, 2]] = clear_value

    def _apply_slice_operations_to_volume(self, volume, voxel_size, operations, background_value, threshold=None):
        data = np.array(volume, copy=True)
        for operation in operations:
            if not isinstance(operation, dict):
                raise ValueError("Each slicer operation must be an object.")
            action = operation.get("action")
            if action not in ("isolate", "delete"):
                raise ValueError("Slicer operations require action isolate/delete.")
            plane_sets = self._operation_plane_sets(operation)
            if threshold is not None:
                candidate_coords = np.argwhere(data >= threshold)
            else:
                candidate_coords = np.argwhere(data != background_value)
            if candidate_coords.size == 0:
                if action == "isolate":
                    raise ValueError("Slicer isolate removed all voxels; no volume edit was saved.")
                continue
            inside = self._coords_inside_any_wedge(candidate_coords, voxel_size, plane_sets)
            self._apply_slice_mask_to_coords(data, candidate_coords, inside, action, background_value)

        if not self._volume_has_foreground(data, background_value, threshold=threshold):
            raise ValueError("Slicer isolate removed all voxels; no volume edit was saved.")
        return data

    def _apply_slice_operations_to_mask(self, mask_volume, voxel_size, operations):
        data = np.array(mask_volume, copy=True)
        for operation in operations:
            action = operation.get("action")
            plane_sets = self._operation_plane_sets(operation)
            candidate_coords = np.argwhere(data > 0)
            inside = self._coords_inside_any_wedge(candidate_coords, voxel_size, plane_sets)
            self._apply_slice_mask_to_coords(data, candidate_coords, inside, action, 0)
        return data

    def _volume_summary(self, volume):
        volume = np.asarray(volume)
        return {
            "shape": [int(dim) for dim in volume.shape],
            "dtype": str(volume.dtype),
            "min": float(np.min(volume)),
            "max": float(np.max(volume)),
        }

    def _propagate_sliced_mask(self, subject_dir, source_mask_edit_base, edit_number, filename, operations, voxel_size, nifti_affine):
        mask_src_path = os.path.join(subject_dir, f"{source_mask_edit_base}.nii.mask.gz")
        if not os.path.isfile(mask_src_path):
            return None

        mask_src_temp = os.path.join(subject_dir, f"{source_mask_edit_base}_mask.nii.gz")
        os.replace(mask_src_path, mask_src_temp)
        try:
            mask_img = nib.load(mask_src_temp)
            mask_dtype = mask_img.get_data_dtype()
            mask_data = np.round(mask_img.get_fdata()).astype(mask_dtype)
            sliced_mask = self._apply_slice_operations_to_mask(mask_data, voxel_size, operations)
        finally:
            if os.path.exists(mask_src_temp):
                os.replace(mask_src_temp, mask_src_path)

        new_mask_edit_base = f"{filename}_edit_{edit_number}_sliced"
        mask_dest_path = os.path.join(subject_dir, f"{new_mask_edit_base}.nii.mask.gz")
        mask_dest_temp = os.path.join(subject_dir, f"{new_mask_edit_base}_mask.nii.gz")
        nib.save(nib.Nifti1Image(sliced_mask, nifti_affine), mask_dest_temp)
        os.replace(mask_dest_temp, mask_dest_path)

        lc = None
        json_path = os.path.join(subject_dir, f"{filename}.json")
        if os.path.isfile(json_path):
            with open(json_path, "r") as jf:
                metadata = json.load(jf)
            lc = metadata.get("lossy_compression")

        resolution_factor = 2
        if isinstance(lc, list) and lc:
            try:
                resolution_factor = int(lc[-1].get("resolution_factor", resolution_factor))
            except Exception:
                pass
        elif isinstance(lc, dict) and lc:
            try:
                resolution_factor = int(lc.get("resolution_factor", resolution_factor))
            except Exception:
                pass

        slices = [slice(None, None, resolution_factor) for _ in range(3)]
        lossy_mask = sliced_mask[slices[0], slices[1], slices[2]].astype(mask_dtype, copy=False)
        lossy_mask_edit_base = f"{filename}_lossy_edit_{edit_number}_sliced"
        lossy_mask_dest_path = os.path.join(subject_dir, f"{lossy_mask_edit_base}.nii.mask.gz")
        lossy_mask_temp = os.path.join(subject_dir, f"{lossy_mask_edit_base}_mask.nii.gz")
        nib.save(nib.Nifti1Image(lossy_mask, nifti_affine), lossy_mask_temp)
        os.replace(lossy_mask_temp, lossy_mask_dest_path)
        return os.path.basename(mask_dest_path)

    def _lossy_mask_resolution_factor(self, json_path):
        resolution_factor = 2
        if not os.path.isfile(json_path):
            return resolution_factor
        with open(json_path, "r") as jf:
            metadata = json.load(jf)
        lc = metadata.get("lossy_compression")
        if isinstance(lc, list) and lc:
            try:
                resolution_factor = int(lc[-1].get("resolution_factor", resolution_factor))
            except Exception:
                pass
        elif isinstance(lc, dict) and lc:
            try:
                resolution_factor = int(lc.get("resolution_factor", resolution_factor))
            except Exception:
                pass
        return resolution_factor

    def _propagate_shell_mask(self, subject_dir, source_mask_edit_base, edit_number, filename, shell_mask, nifti_affine, json_path):
        mask_src_path = os.path.join(subject_dir, f"{source_mask_edit_base}.nii.mask.gz")
        if not os.path.isfile(mask_src_path):
            return None

        mask_src_temp = os.path.join(subject_dir, f"{source_mask_edit_base}_mask.nii.gz")
        os.replace(mask_src_path, mask_src_temp)
        try:
            mask_img = nib.load(mask_src_temp)
            mask_dtype = mask_img.get_data_dtype()
            mask_data = np.round(mask_img.get_fdata()).astype(mask_dtype)
            shell_mask = np.asarray(shell_mask, dtype=bool)
            if shell_mask.shape != mask_data.shape:
                raise ValueError(
                    f"Shell mask shape {shell_mask.shape} does not match segmentation mask shape {mask_data.shape}"
                )
            shelled_mask = np.where(shell_mask, mask_data, 0).astype(mask_dtype)
        finally:
            if os.path.exists(mask_src_temp):
                os.replace(mask_src_temp, mask_src_path)

        new_mask_edit_base = f"{filename}_edit_{edit_number}_shell"
        mask_dest_path = os.path.join(subject_dir, f"{new_mask_edit_base}.nii.mask.gz")
        mask_dest_temp = os.path.join(subject_dir, f"{new_mask_edit_base}_mask.nii.gz")
        nib.save(nib.Nifti1Image(shelled_mask, nifti_affine), mask_dest_temp)
        os.replace(mask_dest_temp, mask_dest_path)

        resolution_factor = self._lossy_mask_resolution_factor(json_path)
        slices = [slice(None, None, resolution_factor) for _ in range(3)]
        lossy_mask = shelled_mask[slices[0], slices[1], slices[2]].astype(mask_dtype, copy=False)
        lossy_mask_edit_base = f"{filename}_lossy_edit_{edit_number}_shell"
        lossy_mask_dest_path = os.path.join(subject_dir, f"{lossy_mask_edit_base}.nii.mask.gz")
        lossy_mask_temp = os.path.join(subject_dir, f"{lossy_mask_edit_base}_mask.nii.gz")
        nib.save(nib.Nifti1Image(lossy_mask, nifti_affine), lossy_mask_temp)
        os.replace(lossy_mask_temp, lossy_mask_dest_path)
        return os.path.basename(mask_dest_path)

    def _resolve_fullres_volume_threshold(self, metadata, nifti_data, request_threshold=None):
        threshold = request_threshold if request_threshold is not None else metadata.get("threshold")
        if threshold is None:
            raise ValueError("threshold is required in metadata or request.")
        threshold = float(threshold)
        return float(np.clip(threshold, np.min(nifti_data), np.max(nifti_data)))

    def _parse_optional_gaussian_blur(self, value):
        if value is None or value == "":
            return None
        blur = float(value)
        return blur if blur > 0 else None

    def _build_glb_response(self, vertices, faces):
        vertices = np.asarray(vertices, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.uint32)
        gltf = pygltflib.GLTF2()
        vertex_data = vertices.tobytes()
        face_data = faces.tobytes()
        buffer_data = vertex_data + face_data

        gltf.buffers.append(pygltflib.Buffer(byteLength=len(buffer_data), uri=None))
        gltf.set_binary_blob(buffer_data)
        gltf.bufferViews.extend([
            pygltflib.BufferView(buffer=0, byteOffset=0, byteLength=len(vertex_data), target=pygltflib.ARRAY_BUFFER),
            pygltflib.BufferView(buffer=0, byteOffset=len(vertex_data), byteLength=len(face_data), target=pygltflib.ELEMENT_ARRAY_BUFFER),
        ])
        gltf.accessors.extend([
            pygltflib.Accessor(
                bufferView=0,
                componentType=pygltflib.FLOAT,
                count=len(vertices),
                type=pygltflib.VEC3,
                max=vertices.max(axis=0).tolist(),
                min=vertices.min(axis=0).tolist(),
            ),
            pygltflib.Accessor(
                bufferView=1,
                componentType=pygltflib.UNSIGNED_INT,
                count=len(faces.flatten()),
                type=pygltflib.SCALAR,
            ),
        ])
        gltf.meshes.append(pygltflib.Mesh(
            primitives=[pygltflib.Primitive(attributes=pygltflib.Attributes(POSITION=0), indices=1)]
        ))
        gltf.nodes.append(pygltflib.Node(mesh=0))
        gltf.scenes.append(pygltflib.Scene(nodes=[0]))
        gltf.scene = 0

        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
            gltf.save_binary(tmp.name)
            tmp_path = tmp.name
        try:
            with open(tmp_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        finally:
            try:
                os.unlink(tmp_path)
            except (OSError, PermissionError):
                pass

    def _mesh_preview_payload(self, mesh):
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.uint32)
        center = vertices.mean(axis=0)
        display_vertices = vertices - center
        scale_factor = float(np.max(np.abs(display_vertices)))
        if scale_factor <= 0:
            scale_factor = 1.0
        display_vertices = display_vertices / scale_factor
        return {
            "gltf": self._build_glb_response(display_vertices, faces),
            "center": center.tolist(),
            "scale_factor": scale_factor,
            "mesh_summary": self._mesh_summary(mesh),
        }


def _clip_float_value(value, lo, hi, default):
    try:
        parsed = float(value)
        if parsed != parsed:  # NaN
            return default
        return max(lo, min(hi, parsed))
    except (TypeError, ValueError):
        return default


def _mesh_elastic_log(message):
    """Timestamped stdout line for long-running NR-ICP jobs (flush so logs appear immediately)."""
    stamp = time.strftime("%H:%M:%S")
    print(f"[mesh-elastic {stamp}] {message}", flush=True)


def _mesh_bbox_diagonal(mesh):
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    return float(np.linalg.norm(bounds[1] - bounds[0]))


def _mesh_elastic_mesh_summary(mesh, label="mesh"):
    vertex_count = len(mesh.vertices)
    face_count = len(mesh.faces)
    diagonal = _mesh_bbox_diagonal(mesh)
    return f"{label}: {vertex_count:,} vertices, {face_count:,} faces, bbox diagonal {diagonal:.3f} mm"


# NR-ICP is far more memory-hungry than its "just an ICP variant" name suggests: each iteration
# solves a sparse linear system with 4 unknowns per vertex (trimesh.registration.nricp_amberg's
# _solve_system does a direct sparse LU factorization via scipy's spsolve), and — when the target
# is a Trimesh — every correspondence query goes through trimesh.proximity.closest_point, which
# builds one giant *unchunked* array of (query point, candidate triangle) pairs with no size limit.
# Empirically (icosphere benchmark, this repo, 2026-07-02): ~41k vertices/faces already needs
# ~2.3 GB RAM per stage even with the safer vertex-based correspondence search below, and ~82k
# vertices reliably blows past several GB and crashes the sparse LU factorization outright — which
# manifests as the whole machine freezing/swapping to death, not a clean Python exception. 250,000
# vertices (the previous cap) is roughly 3-6x past the point of guaranteed failure.
#
# Batch registration processes one subject at a time, so peak memory is per-subject. We therefore
# run close to the ~41k safe point: more registration vertices produce a finer, less faceted
# deformation field and more TPS control points, which is the main lever against the wrinkly /
# bumpy full-resolution PLYs seen with coarser (20k) warps. Headroom below 41k remains for
# meshes with worse-than-icosphere connectivity and for the dense TPS fit (O(n²) in control count).
MESH_ELASTIC_NRICP_VERTEX_CAP = 35000

# When the registration mesh was decimated, its elastic deformation is propagated onto the full-
# resolution mesh via a single global Thin Plate Spline fit using every vertex from the registration
# mesh (source = pre-elastic positions, target = post-elastic positions), rather than copying each
# full-res vertex's nearest decimated-mesh neighbour's displacement. All ~MESH_ELASTIC_NRICP_VERTEX_CAP
# registration vertices are used as control points — TPS transform is chunked for memory safety.
#
# alpha=0 exactly interpolates every NR-ICP control displacement and tends to introduce
# high-frequency wrinkles between control points on the dense PLY. A mild positive alpha (as a
# fraction of median nearest-neighbour control spacing; for 3D control points the TPS kernel is
# G(r)=r, so alpha is in the same units as that spacing) trades a tiny amount of control-point
# fidelity for a much smoother warp. Transform evaluation is already chunked, so larger control
# counts only cost fit memory/time, not a single giant transform allocation.
MESH_ELASTIC_TPS_ALPHA_SPACING_FRACTION = 0.25

MESH_ELASTIC_REFERENCE_PLY_SUFFIX = "_mesh_elastic_reference.ply"


def mesh_elastic_reference_ply_basename(reference):
    return f"{reference}{MESH_ELASTIC_REFERENCE_PLY_SUFFIX}"


def mesh_elastic_display_mesh_npz_paths(elastic_ply_path):
    """
    Cached decimated display mesh geometry for this edit (same idea as voxel
    *_vertices.npz/_faces.npz): a single, forever-overwritten pair per edit rather than one
    pair per vertex_cap/decimation outcome.
    """
    stem, _ext = os.path.splitext(elastic_ply_path)
    base = f"{stem}_display_mesh"
    return f"{base}_vertices.npz", f"{base}_faces.npz"


def try_load_cached_preserved_mesh_display_geometry(ply_path, vertex_cap):
    """
    Fast-path cache peek: returns (vertices, faces, original_vertex_count, face_uv) on a hit,
    or None. ``face_uv`` may be None when the subject has no built-in texture cache.
    """
    vertices_npz, faces_npz = mesh_elastic_display_mesh_npz_paths(ply_path)
    if not (os.path.isfile(vertices_npz) and os.path.isfile(faces_npz)):
        return None
    try:
        from .builtinTextureTools import uv_sidecar_path

        uv_sidecar = uv_sidecar_path(ply_path)
        if os.path.isfile(uv_sidecar) and os.path.getmtime(uv_sidecar) > os.path.getmtime(vertices_npz):
            return None
    except Exception:
        pass
    try:
        cached_v = np.load(vertices_npz)
        cached_vertices = np.asarray(cached_v['vertices'], dtype=np.float64)
        cached_cap = int(cached_v['vertex_cap']) if 'vertex_cap' in cached_v else None
        original_vertex_count = (
            int(cached_v['original_vertex_count']) if 'original_vertex_count' in cached_v else None
        )
        cached_face_uv = None
        if 'face_uv' in cached_v:
            cached_face_uv = np.asarray(cached_v['face_uv'], dtype=np.float32)
        cached_faces = np.asarray(np.load(faces_npz)['faces'], dtype=np.uint32)
        if 'face_uv' in cached_v:
            sidecar_face_uv = load_uv_sidecar(ply_path)
            if sidecar_face_uv is None or sidecar_face_uv.shape[0] != len(cached_faces):
                return None
        if (
            cached_cap == int(vertex_cap)
            and len(cached_vertices) > 100
            and len(cached_faces) > 0
            and original_vertex_count is not None
        ):
            return cached_vertices, cached_faces, original_vertex_count, cached_face_uv
    except Exception as exc:
        print(f"Warning: could not read preserved mesh display cache at {vertices_npz}: {exc}")
    return None


def build_and_cache_preserved_mesh_display_geometry(
    ply_path,
    full_vertices,
    full_faces,
    vertex_cap,
    full_face_uv=None,
):
    """
    Slow path: decimate the full-resolution mesh down to `vertex_cap` and persist the result.
    When ``full_face_uv`` is provided, a decimated face-corner UV field is cached too.
    """
    full_vertices = np.asarray(full_vertices, dtype=np.float64)
    full_faces = np.asarray(full_faces, dtype=np.uint32)
    vertices, faces = full_vertices, full_faces
    display_face_uv = None
    if len(vertices) > vertex_cap:
        decimation_factor = max(min(1.0, 1.0 - (vertex_cap / len(vertices))), 0.0)
        vertices, faces = fast_simplification.simplify(full_vertices, full_faces, decimation_factor)
        if full_face_uv is not None:
            display_face_uv = remap_face_uvs_for_decimated_mesh(
                full_vertices,
                full_faces,
                full_face_uv,
                vertices,
                faces,
            )
    elif full_face_uv is not None:
        display_face_uv = np.asarray(full_face_uv, dtype=np.float32)
    vertices_npz, faces_npz = mesh_elastic_display_mesh_npz_paths(ply_path)
    try:
        cache_payload = {
            'vertices': np.asarray(vertices, dtype=np.float64),
            'vertex_cap': int(vertex_cap),
            'original_vertex_count': int(len(full_vertices)),
        }
        if display_face_uv is not None:
            cache_payload['face_uv'] = np.asarray(display_face_uv, dtype=np.float32)
        np.savez(vertices_npz, **cache_payload)
        np.savez(faces_npz, faces=np.asarray(faces, dtype=np.uint32))
    except OSError as exc:
        print(f"Warning: could not save preserved mesh display cache to {vertices_npz}: {exc}")
    return vertices, faces, display_face_uv


def _heatmap_reference_cache_paths(cache_stem, vertex_cap):
    """
    Decimated reference surface used only for individual heatmap distances.
    The full mesh_elastic_reference.ply is left untouched for landmark propagation.

    ``shared`` marks caches built in prism corner-origin mm (aligned-subject frame).
    """
    base = f"{cache_stem}_heatmap_shared_{int(vertex_cap)}"
    return f"{base}_vertices.npz", f"{base}_faces.npz"


def preserved_mesh_reference_in_shared_mm(mesh, ref_metadata, directory=None):
    """
    Map a centroid-local preserved-mesh PLY into prism corner-origin shared mm.

    Rigid alignment stores subject edits in this frame; NR-ICP and heatmaps must
    load the preserved-mesh reference in the same frame or every correspondence
    falls outside ``distance_threshold`` and nricp_amberg fails with a singular
    factor (all data weights zero).
    """
    shared_vertices = mesh_local_to_shared_mm(
        mesh.vertices,
        ref_metadata.get("mesh_metadata") if isinstance(ref_metadata, dict) else None,
        prism_vertices=mesh.vertices,
        directory=directory,
    )
    return trimesh.Trimesh(
        vertices=shared_vertices,
        faces=np.asarray(mesh.faces, dtype=np.int64),
        process=False,
    )


def mesh_elastic_reference_in_registration_frame(mesh, ref_metadata, directory=None):
    """Return ``mesh`` in the frame used by aligned preserved-mesh subjects."""
    if is_preserved_mesh_metadata(ref_metadata):
        return preserved_mesh_reference_in_shared_mm(mesh, ref_metadata, directory)
    return mesh


def _decimate_trimesh_to_vertex_cap(mesh, vertex_cap):
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    if len(vertices) <= int(vertex_cap):
        return vertices, faces
    decimation_factor = max(min(1.0, 1.0 - (int(vertex_cap) / len(vertices))), 0.0)
    return fast_simplification.simplify(vertices, faces, decimation_factor)


def prepare_heatmap_reference_mesh(directory, reference, vertex_cap):
    """
    Reference surface for subject→reference heatmap distances, capped like the display mesh.

    The on-disk mesh_elastic_reference.ply (full-res, for 1:1 landmark propagation) is never
    modified. For distance queries we use a display-cap decimation, cached beside the source
    PLY so subsequent elastic views avoid re-parsing multi-million-vertex references.
    Voxel normal-quality heatmaps are fast because their reference is a lossy MC surface of
    similar density; this brings the mesh-subject path in line with that cost.
    """
    vertex_cap = int(vertex_cap)
    ref_dir = _mesh_elastic_reference_dir(directory, reference)
    json_name = "atlas.json" if reference == "atlas" else f"{reference}.json"
    json_path = os.path.join(ref_dir, json_name)
    with open(json_path, "r") as jf:
        ref_metadata = json.load(jf)

    source_ply_path = None
    ref_ply_name = ref_metadata.get("mesh_elastic_reference_ply")
    if ref_ply_name:
        candidate = os.path.join(ref_dir, os.path.basename(ref_ply_name))
        if os.path.isfile(candidate):
            source_ply_path = candidate
    if source_ply_path is None and is_preserved_mesh_metadata(ref_metadata):
        mesh_file = ref_metadata.get("mesh_file") or f"{reference}.ply"
        candidate = os.path.join(ref_dir, mesh_file)
        if os.path.isfile(candidate):
            source_ply_path = candidate
    if source_ply_path is None:
        candidate = os.path.join(ref_dir, mesh_elastic_reference_ply_basename(reference))
        if os.path.isfile(candidate):
            source_ply_path = candidate

    if source_ply_path is not None:
        cache_stem, _ext = os.path.splitext(source_ply_path)
        source_mtime = os.path.getmtime(source_ply_path)
    else:
        cache_stem = os.path.join(ref_dir, f"{reference}_heatmap_mc")
        source_mtime = None

    vertices_npz, faces_npz = _heatmap_reference_cache_paths(cache_stem, vertex_cap)
    if os.path.isfile(vertices_npz) and os.path.isfile(faces_npz):
        try:
            cached_v = np.load(vertices_npz)
            cached_cap = int(cached_v["vertex_cap"]) if "vertex_cap" in cached_v else None
            cached_mtime = float(cached_v["source_mtime"]) if "source_mtime" in cached_v else None
            cached_vertices = np.asarray(cached_v["vertices"], dtype=np.float64)
            cached_faces = np.asarray(np.load(faces_npz)["faces"], dtype=np.uint32)
            mtime_ok = source_mtime is None or (
                cached_mtime is not None and abs(cached_mtime - source_mtime) < 1e-6
            )
            if (
                cached_cap == vertex_cap
                and mtime_ok
                and len(cached_vertices) > 100
                and len(cached_faces) > 0
            ):
                _mesh_elastic_log(
                    f"heatmap reference cache hit for {reference}: "
                    f"{len(cached_vertices):,} vertices (cap={vertex_cap})"
                )
                return trimesh.Trimesh(
                    vertices=cached_vertices, faces=cached_faces, process=False
                )
        except Exception as exc:
            print(f"Warning: could not read heatmap reference cache at {vertices_npz}: {exc}")

    if source_ply_path is not None:
        full_mesh = _load_trimesh_from_ply(source_ply_path)
        full_mesh = mesh_elastic_reference_in_registration_frame(full_mesh, ref_metadata, directory)
    else:
        full_mesh = _extract_voxel_reference_surface_mm(directory, reference, ref_metadata)

    full_count = len(full_mesh.vertices)
    vertices, faces = _decimate_trimesh_to_vertex_cap(full_mesh, vertex_cap)
    _mesh_elastic_log(
        f"heatmap reference for {reference}: {full_count:,} → {len(vertices):,} vertices "
        f"(cap={vertex_cap}); full PLY preserved for landmark propagation"
    )

    try:
        save_kwargs = {
            "vertices": np.asarray(vertices, dtype=np.float64),
            "vertex_cap": vertex_cap,
            "original_vertex_count": int(full_count),
        }
        if source_mtime is not None:
            save_kwargs["source_mtime"] = float(source_mtime)
        np.savez(vertices_npz, **save_kwargs)
        np.savez(faces_npz, faces=np.asarray(faces, dtype=np.uint32))
    except OSError as exc:
        print(f"Warning: could not save heatmap reference cache to {vertices_npz}: {exc}")

    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def compute_mesh_to_reference_surface_distances(query_vertices, reference_mesh, label=None, chunk_size=100_000):
    """
    Signed-free surface distance from each query point to the reference triangle mesh
    (same Open3D RaycastingScene approach as MarchingCubesView dense_distances).
    """
    ref_vertices = np.asarray(reference_mesh.vertices, dtype=np.float64)
    ref_faces = np.asarray(reference_mesh.faces, dtype=np.int32)
    ref_mesh = o3d.geometry.TriangleMesh()
    ref_mesh.vertices = o3d.utility.Vector3dVector(ref_vertices)
    ref_mesh.triangles = o3d.utility.Vector3iVector(ref_faces)
    scene = o3d.t.geometry.RaycastingScene()
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(ref_mesh)
    _ = scene.add_triangles(mesh_t)

    query_vertices = np.asarray(query_vertices, dtype=np.float64)
    distances = np.empty(len(query_vertices), dtype=np.float64)
    if label:
        _mesh_elastic_log(
            f"{label}: computing surface distances for {len(query_vertices):,} vertices "
            f"against reference mesh ({len(ref_vertices):,} vertices)..."
        )
    t0 = time.perf_counter()
    for start in range(0, len(query_vertices), chunk_size):
        chunk = query_vertices[start:start + chunk_size]
        points = o3d.core.Tensor(chunk.astype(np.float32))
        result = scene.compute_closest_points(points)
        closest_points = result["points"].numpy()
        distances[start:start + len(chunk)] = np.linalg.norm(chunk - closest_points, axis=1)
    if label:
        _mesh_elastic_log(f"{label}: surface distance computation finished in {time.perf_counter() - t0:.1f}s")
    return distances


def compute_mesh_to_reference_parallel_surface_distances(
    query_vertices,
    reference_mesh,
    label=None,
    chunk_size=100_000,
    tangential_tol_mm=None,
):
    """
    Parallel-to-surface pad distances for template masking.

    Euclidean ``‖q−c‖`` grows a rounded tubular bulb past open rims. Padding
    *parallel to the surface* instead measures the offset in the local
    normal/tangent frame and keeps the separable slab

        max(|n·(q−c)|, ‖(q−c)_tangent‖) ≤ pad

    so the footprint expands smoothly by ``pad`` along the surface (smooth
    silhouette offset) while the normal seating thickness stays ``pad`` — without
    the diagonal corner of the old isotropic tube.
    """
    ref_vertices = np.asarray(reference_mesh.vertices, dtype=np.float64)
    ref_faces = np.asarray(reference_mesh.faces, dtype=np.int32)
    if len(ref_vertices) == 0 or len(ref_faces) == 0:
        raise ValueError("parallel-surface distances require a non-empty reference mesh")

    ref_mesh = o3d.geometry.TriangleMesh()
    ref_mesh.vertices = o3d.utility.Vector3dVector(ref_vertices)
    ref_mesh.triangles = o3d.utility.Vector3iVector(ref_faces)
    scene = o3d.t.geometry.RaycastingScene()
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(ref_mesh)
    _ = scene.add_triangles(mesh_t)

    query_vertices = np.asarray(query_vertices, dtype=np.float64)
    distances = np.empty(len(query_vertices), dtype=np.float64)

    if tangential_tol_mm is None:
        # Tiny floor only — faceting noise, not a second pad budget.
        edges = np.asarray(reference_mesh.edges_unique, dtype=np.int64)
        if len(edges) == 0:
            tangential_tol_mm = 1e-4
        else:
            edge_lengths = np.linalg.norm(
                ref_vertices[edges[:, 0]] - ref_vertices[edges[:, 1]], axis=1
            )
            tangential_tol_mm = max(1e-4, 0.05 * float(np.median(edge_lengths)))
    tol = float(tangential_tol_mm)

    if label:
        _mesh_elastic_log(
            f"{label}: computing parallel-to-surface distances for {len(query_vertices):,} "
            f"vertices against reference mesh ({len(ref_vertices):,} vertices; "
            f"tangential_tol={tol:.4f} mm)..."
        )
    t0 = time.perf_counter()
    for start in range(0, len(query_vertices), chunk_size):
        chunk = query_vertices[start:start + chunk_size]
        points = o3d.core.Tensor(chunk.astype(np.float32))
        result = scene.compute_closest_points(points)
        closest = result["points"].numpy().astype(np.float64)
        normals = result["primitive_normals"].numpy().astype(np.float64)
        nlen = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(nlen, 1e-12)
        offset = chunk - closest
        normal_comp = np.sum(offset * normals, axis=1, keepdims=True)
        normal_offset = np.abs(normal_comp[:, 0])
        tangential = np.linalg.norm(offset - normals * normal_comp, axis=1)
        # Separable pad: smooth lateral offset by pad past the rim, same pad along n.
        distances[start:start + len(chunk)] = np.maximum(
            normal_offset, np.maximum(0.0, tangential - tol)
        )
    if label:
        _mesh_elastic_log(
            f"{label}: parallel-to-surface distance computation finished in {time.perf_counter() - t0:.1f}s"
        )
    return distances


def mesh_elastic_surface_metric(distances, error_threshold_mm):
    """
    Mirror ElasticRegistrationView's surface metric (Dice ignored): return
    (fraction_above_threshold, elastic_surface_distance_score, mean_distance_mm).
    """
    distances = np.asarray(distances, dtype=np.float64)
    if len(distances) == 0:
        return 0.0, 0.0, 0.0
    above_fraction = float(np.sum(distances > error_threshold_mm) / len(distances))
    return above_fraction, float(1.0 - above_fraction), float(np.mean(distances))


def _load_trimesh_from_ply(path):
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = [
            geom for geom in loaded.geometry.values()
            if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) > 0
        ]
        if not geometries:
            raise ValueError(f"No mesh geometry found in {path}")
        loaded = trimesh.util.concatenate(geometries)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"Unsupported mesh payload in {path}")
    return loaded


def _mesh_elastic_reference_dir(directory, reference):
    if reference == "atlas":
        return resolve_atlas_dir(directory)
    return os.path.join(directory, "extracted", reference)


def _latest_voxel_reference_nifti_path(ref_dir, reference, ref_metadata):
    """Latest non-lossy edit NIfTI for a voxel reference, else the base volume."""
    latest_edit_number = -1
    latest_path = None
    for path in glob.glob(os.path.join(ref_dir, f"{reference}_edit_*.nii.gz")):
        basename = os.path.basename(path)
        if "_lossy" in basename or "_inv" in basename or "_fwd" in basename:
            continue
        try:
            edit_number = int(basename.split("_edit_")[1].split("_")[0])
        except (IndexError, ValueError):
            continue
        if edit_number > latest_edit_number:
            latest_edit_number = edit_number
            latest_path = path
    if latest_path is None:
        latest_path = os.path.join(ref_dir, f"{reference}.nii.gz")
    voxel_size = ref_metadata.get("voxel_size") or 1.0
    return latest_path, float(voxel_size)


def _extract_voxel_reference_surface_mm(directory, reference, ref_metadata):
    """
    Marching-cubes fallback for voxel references when mesh_elastic_reference.ply is absent.
    Matches mesh-elastic registration conventions: display-axis swap, light Gaussian, face flip.
    """
    ref_dir = _mesh_elastic_reference_dir(directory, reference)
    nifti_path, voxel_size = _latest_voxel_reference_nifti_path(ref_dir, reference, ref_metadata)
    if not os.path.isfile(nifti_path):
        raise FileNotFoundError(
            f"Voxel reference volume not found for {reference}: {nifti_path}"
        )
    volume = nib.load(nifti_path).get_fdata()
    volume = np.swapaxes(volume, 0, 2)
    volume = gaussian_filter(volume, sigma=1.0)
    threshold = ref_metadata.get("threshold")
    if threshold is None:
        threshold = float(np.mean(volume))
    threshold = float(np.clip(threshold, np.min(volume), np.max(volume)))
    verts, faces, _, _ = measure.marching_cubes(
        volume, level=threshold, spacing=(voxel_size,) * 3
    )
    faces = faces[:, ::-1]
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def load_mesh_elastic_reference_surface(directory, reference):
    """
    Load the reference surface for mesh-elastic metrics / individual heatmaps.

    Preference order:
      1. mesh_elastic_reference_ply recorded at registration (mesh or voxel-derived PLY)
      2. Preserved-mesh reference: mesh_file PLY from metadata
      3. Voxel reference: dedicated mesh_elastic_reference.ply basename
      4. Voxel reference fallback: marching cubes from the latest full-res NIfTI

    This keeps mesh and voxel references interchangeable for subject→reference distances.
    """
    ref_dir = _mesh_elastic_reference_dir(directory, reference)
    json_name = "atlas.json" if reference == "atlas" else f"{reference}.json"
    json_path = os.path.join(ref_dir, json_name)
    with open(json_path, "r") as jf:
        ref_metadata = json.load(jf)

    ref_ply_name = ref_metadata.get("mesh_elastic_reference_ply")
    if ref_ply_name:
        ref_ply_path = os.path.join(ref_dir, os.path.basename(ref_ply_name))
        if os.path.isfile(ref_ply_path):
            mesh = _load_trimesh_from_ply(ref_ply_path)
            return mesh_elastic_reference_in_registration_frame(mesh, ref_metadata, directory)

    if is_preserved_mesh_metadata(ref_metadata):
        mesh_file = ref_metadata.get("mesh_file") or f"{reference}.ply"
        ref_mesh_path = os.path.join(ref_dir, mesh_file)
        if not os.path.isfile(ref_mesh_path):
            raise FileNotFoundError(
                f"Mesh elastic reference surface not found for {reference}: {ref_mesh_path}"
            )
        mesh = _load_trimesh_from_ply(ref_mesh_path)
        return mesh_elastic_reference_in_registration_frame(mesh, ref_metadata, directory)

    ref_mesh_path = os.path.join(ref_dir, mesh_elastic_reference_ply_basename(reference))
    if os.path.isfile(ref_mesh_path):
        return _load_trimesh_from_ply(ref_mesh_path)

    return _extract_voxel_reference_surface_mm(directory, reference, ref_metadata)


def _decimate_mesh_for_nricp(mesh, vertex_cap, label):
    if len(mesh.vertices) <= vertex_cap:
        return mesh.copy(), False

    decimation_factor = max(min(1.0, 1.0 - (vertex_cap / len(mesh.vertices))), 0.0)
    _mesh_elastic_log(
        f"{label}: decimating from {len(mesh.vertices):,} to ~{vertex_cap:,} vertices "
        f"(decimation_factor={decimation_factor:.4f}) before NR-ICP..."
    )
    t0 = time.perf_counter()
    simplified_vertices, simplified_faces = fast_simplification.simplify(
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.uint32),
        decimation_factor,
    )
    simplified = trimesh.Trimesh(
        vertices=np.asarray(simplified_vertices, dtype=np.float64),
        faces=np.asarray(simplified_faces, dtype=np.int64),
        process=False,
    )
    _mesh_elastic_log(
        f"{label}: decimated to {len(simplified.vertices):,} vertices in {time.perf_counter() - t0:.1f}s"
    )
    return simplified, True


def _sanitize_mesh_for_nricp(mesh, label):
    """
    Quadric-collapse decimation (fast_simplification) can leave behind duplicate/near-coincident
    vertices, zero-area faces, and — critically — small disconnected shards (marching-cubes noise,
    debris broken off a thin structure like a septum or tooth root). nricp_amberg's smoothness
    term is a graph Laplacian built from mesh edges: an isolated shard whose vertices all fall
    outside distance_threshold (zero data weight) contributes an all-zero block to the global
    matrix, which makes scipy's sparse LU factorization fail with "Factor is exactly singular"
    (reproduced directly with a single stray triangle in testing, 2026-07-02). Keeping only the
    largest connected component after cleanup eliminates this failure mode structurally.
    """
    before_vertices, before_faces = len(mesh.vertices), len(mesh.faces)
    cleaned = mesh.copy()
    cleaned.merge_vertices()
    sanitize_trimesh_faces(cleaned)
    cleaned.remove_unreferenced_vertices()

    components = cleaned.split(only_watertight=False)
    if len(components) > 1:
        components = sorted(components, key=lambda c: len(c.faces), reverse=True)
        dropped_faces = sum(len(c.faces) for c in components[1:])
        _mesh_elastic_log(
            f"{label}: dropped {len(components) - 1} disconnected component(s) "
            f"({dropped_faces:,} faces) to keep the largest ({len(components[0].faces):,} faces)"
        )
        cleaned = components[0]

    if len(cleaned.vertices) != before_vertices or len(cleaned.faces) != before_faces:
        _mesh_elastic_log(
            f"{label}: sanitized {before_vertices:,} -> {len(cleaned.vertices):,} vertices, "
            f"{before_faces:,} -> {len(cleaned.faces):,} faces"
        )
    return trimesh.Trimesh(vertices=cleaned.vertices, faces=cleaned.faces, process=False)


def _tps_transform_chunked(tps, points, label, target_elements=50_000_000):
    """
    tps.transform() builds a dense (n_points, n_control_points) RBF distance matrix plus a couple
    of same-shaped temporaries (power, log) in one shot. For millions of full-resolution vertices
    against tens of thousands of control points that's tens of GB per array if done in one shot —
    e.g. 2M points x 35,000 control points x 8 bytes = 560GB just for the first temporary. Each
    output point only depends on itself and the (fixed, already-fitted) control points, so this
    is trivially chunkable with zero accuracy loss.
    """
    n_control = len(tps.control_points)
    chunk_size = max(1, target_elements // max(n_control, 1))
    if len(points) <= chunk_size:
        return tps.transform(points)

    _mesh_elastic_log(
        f"{label}: TPS transform of {len(points):,} points vs {n_control:,} control points "
        f"exceeds the {target_elements:,}-element chunk budget — processing in chunks of {chunk_size:,}"
    )
    t0 = time.perf_counter()
    chunks = []
    for start in range(0, len(points), chunk_size):
        chunks.append(tps.transform(points[start:start + chunk_size]))
    result = np.vstack(chunks)
    _mesh_elastic_log(
        f"{label}: chunked TPS transform ({len(points):,} points, "
        f"{-(-len(points) // chunk_size)} chunk(s)) finished in {time.perf_counter() - t0:.1f}s"
    )
    return result


def _tps_regularization_alpha(source_control, spacing_fraction=None):
    """
    Mild bending-energy regularization scaled to control-point spacing.

    For 3D sources the default polyharmonic kernel is G(r)=r, so alpha shares units with that
    spacing. Using a fraction of the median nearest-neighbour distance keeps the same smoothness
    behaviour across differently sized / differently decimated meshes.
    """
    fraction = (
        MESH_ELASTIC_TPS_ALPHA_SPACING_FRACTION
        if spacing_fraction is None
        else float(spacing_fraction)
    )
    source_control = np.asarray(source_control, dtype=np.float64)
    if len(source_control) < 2:
        return 0.0
    nn_dists, _ = cKDTree(source_control).query(source_control, k=2)
    median_spacing = float(np.median(nn_dists[:, 1]))
    if not np.isfinite(median_spacing) or median_spacing <= 0:
        return 0.0
    return fraction * median_spacing


def _voxel_display_basis_xyz(points):
    """Map physical mm from raw NIfTI/MC axes into Aurora display [X,Y,Z] (swap axes 0↔2)."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 1:
        return points[[2, 1, 0]]
    return points[:, [2, 1, 0]]


def _meshes_share_topology(mesh_a, mesh_b):
    """True when two meshes have identical vertex count, face count, and face index order."""
    faces_a = np.asarray(mesh_a.faces, dtype=np.int64)
    faces_b = np.asarray(mesh_b.faces, dtype=np.int64)
    return (
        len(mesh_a.vertices) == len(mesh_b.vertices)
        and len(faces_a) == len(faces_b)
        and np.array_equal(faces_a, faces_b)
    )


def _assert_elastic_preserves_source_topology(source_mesh, elastic_mesh, label):
    """
    Elastic edits must preserve pre-elastic vertex count/order and face connectivity so
    landmark transfer can copy coordinates by shared vertex index.
    """
    if not _meshes_share_topology(source_mesh, elastic_mesh):
        raise ValueError(
            f"{label}: elastic output topology mismatch — "
            f"source {len(source_mesh.vertices):,} vertices / {len(source_mesh.faces):,} faces, "
            f"elastic {len(elastic_mesh.vertices):,} vertices / {len(elastic_mesh.faces):,} faces. "
            f"Registration source and elastic edit must share identical connectivity."
        )


def _outer_shell_trimesh(mesh, label="mesh"):
    """
    Reduce a mesh to its outward-visible surface sheet (marching-cubes inner sheets removed).

    Part of the **complex mesh volume** pathway: MC iso-surfaces often contain nested sheets at
    nearly the same location; NR-ICP nearest-vertex matching against those is ambiguous. This
    keeps only triangles visible from outside (via ``ALPACA.get_outer_mesh``).
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0:
        return mesh.copy()
    shell_vertices, shell_faces = alpaca.get_outer_mesh(vertices, faces)
    if len(shell_vertices) == 0 or len(shell_faces) == 0:
        _mesh_elastic_log(f"{label}: outer shell extraction empty; using input mesh")
        return mesh.copy()
    shell = trimesh.Trimesh(
        vertices=np.asarray(shell_vertices, dtype=np.float64),
        faces=np.asarray(shell_faces, dtype=np.int64),
        process=False,
    )
    _mesh_elastic_log(
        f"{label}: outer shell — {len(vertices):,} → {len(shell.vertices):,} vertices, "
        f"{len(faces):,} → {len(shell.faces):,} faces"
    )
    return shell


def _filter_landmarks_to_shell_mesh(landmarks, shell_mesh):
    """Keep landmarks that lie on the shell surface within a small tolerance."""
    if landmarks is None:
        return None, None

    vertices = np.asarray(shell_mesh.vertices, dtype=np.float64)
    faces = np.asarray(shell_mesh.faces, dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0:
        return [], []

    bounds = np.asarray(shell_mesh.bounds, dtype=np.float64)
    diagonal = float(np.linalg.norm(bounds[1] - bounds[0])) if bounds.shape == (2, 3) else 1.0
    tolerance = max(diagonal * 0.005, 1e-6)

    valid_rows = []
    positions = []
    for landmark_index, landmark in enumerate(landmarks):
        try:
            position = np.asarray(landmark.get("position"), dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            continue
        valid_rows.append((landmark_index, landmark, position))
        positions.append(position)

    if not positions:
        return [], []

    distances = alpaca.calculate_landmark_distances(
        np.asarray(positions, dtype=np.float64),
        vertices,
        faces,
    )

    kept_landmarks = []
    kept_indices = []
    for row_index, (landmark_index, landmark, position) in enumerate(valid_rows):
        if float(distances[row_index]) <= tolerance:
            kept_indices.append(landmark_index)
            kept_landmarks.append({
                **landmark,
                "position": position.tolist(),
            })

    return kept_landmarks, kept_indices


def _tps_propagate_to_full_mesh(
    full_pre, reg_pre, reg_elastic, rigid_matrix, label, tps_alpha_spacing_fraction=None,
):
    """
    Propagate the decimated registration mesh's elastic deformation onto the full-resolution
    subject mesh via a single global Thin Plate Spline warp, instead of copying each full-res
    vertex's nearest decimated-mesh neighbour's displacement. Every registration-mesh vertex is
    used as a TPS control point, paired by index between its pre- and post-elastic positions —
    both share identical topology since only NR-ICP's solve changed vertex coordinates, not
    connectivity.

    `rigid_matrix` is the rigid pre-alignment applied to the decimated registration mesh before
    NR-ICP (identity if that step was skipped/disabled): the full-resolution mesh is transformed
    into that same frame so the TPS control points and the points it warps are consistent, then
    the resulting displacement (not absolute position) is added back onto the untouched original
    full-resolution coordinates — this mirrors how landmark/atlas displacement fields are applied
    elsewhere in this codebase and keeps the saved elastic edit in the same frame as its source.
    """
    control_count = len(reg_pre.vertices)
    source_control = np.asarray(reg_pre.vertices, dtype=np.float64)
    target_control = np.asarray(reg_elastic.vertices, dtype=np.float64)
    alpha = _tps_regularization_alpha(source_control, spacing_fraction=tps_alpha_spacing_fraction)
    alpha_fraction = (
        MESH_ELASTIC_TPS_ALPHA_SPACING_FRACTION
        if tps_alpha_spacing_fraction is None
        else float(tps_alpha_spacing_fraction)
    )
    _mesh_elastic_log(
        f"{label}: fitting TPS from all {control_count:,} registration-mesh control points "
        f"(alpha={alpha:.4f}, spacing_fraction={alpha_fraction:.2f}) to warp "
        f"{len(full_pre.vertices):,} full-resolution vertices..."
    )
    t0 = time.perf_counter()

    tps = ThinPlateSpline(alpha=alpha)
    tps.fit(source_control, target_control)

    full_pre_rigid = trimesh.transform_points(
        np.asarray(full_pre.vertices, dtype=np.float64), rigid_matrix
    )
    full_elastic_rigid = _tps_transform_chunked(tps, full_pre_rigid, label)
    displacements = np.asarray(full_elastic_rigid, dtype=np.float64) - full_pre_rigid
    full_elastic_vertices = np.asarray(full_pre.vertices, dtype=np.float64) + displacements
    _mesh_elastic_log(f"{label}: TPS propagation to full resolution finished in {time.perf_counter() - t0:.1f}s")
    return trimesh.Trimesh(vertices=full_elastic_vertices, faces=full_pre.faces, process=False)


def _nn_displacement_propagate_to_full_mesh(full_pre, reg_pre, reg_elastic, rigid_matrix, label, k_neighbors=4):
    """
    Propagate decimated NR-ICP displacements onto a higher-resolution surface mesh by
    inverse-distance-weighted k-nearest-neighbour copying.

    Used for outer-shell-only registration instead of global TPS: a 3D TPS fit over ~35k
    control points on a closed skull shell tends to produce long-range oscillations and
    needle-like spikes far from the control hull, even with moderate alpha. Neighbour
    copying keeps displacements local to the surface sheet.
    """
    reg_pre_v = np.asarray(reg_pre.vertices, dtype=np.float64)
    reg_elastic_v = np.asarray(reg_elastic.vertices, dtype=np.float64)
    if len(reg_pre_v) == 0:
        raise ValueError(f"{label}: cannot propagate displacements from empty registration mesh")

    control_displacements = reg_elastic_v - reg_pre_v
    full_vertices = np.asarray(full_pre.vertices, dtype=np.float64)
    full_pre_rigid = trimesh.transform_points(full_vertices, rigid_matrix)

    k = max(1, min(int(k_neighbors), len(reg_pre_v)))
    tree = cKDTree(reg_pre_v)
    dists, indices = tree.query(full_pre_rigid, k=k)
    indices = np.asarray(indices, dtype=np.int64)
    if k == 1:
        displacements = control_displacements[indices.reshape(-1)]
    else:
        dists = np.asarray(dists, dtype=np.float64)
        weights = 1.0 / np.maximum(dists, 1e-10)
        weights /= np.sum(weights, axis=1, keepdims=True)
        gathered = control_displacements[indices]
        displacements = np.sum(weights[:, :, None] * gathered, axis=1)

    full_elastic_vertices = full_vertices + np.asarray(displacements, dtype=np.float64)
    _mesh_elastic_log(
        f"{label}: k-NN displacement propagation (k={k}) from {len(reg_pre_v):,} "
        f"registration vertices onto {len(full_vertices):,} full-resolution vertices"
    )
    return trimesh.Trimesh(vertices=full_elastic_vertices, faces=full_pre.faces, process=False)


class MeshElasticRegistrationMixin(MeshEditMixin):
    """
    Non-Rigid ICP (Amberg et al. 2007, via trimesh.registration.nricp_amberg) elastic
    registration for preserved PLY subjects. Mirrors ElasticRegistrationView's voxel/ANTs
    SyN pathway but keeps vertex count/order fixed, so no forward/inverse deformation
    fields are stored: any vertex's displacement is later derivable as
    ``elastic_edit.vertices[i] - pre_elastic_edit.vertices[i]``.
    """

    # Domain presets exposed in the UI (Facial, Complex mesh volume). Custom mode uses
    # mesh_elastic_custom_options from the request. All presets assume prior rigid
    # alignment (ALPACA or manual); NR-ICP never runs an extra ICP pre-alignment pass.
    MESH_ELASTIC_CUSTOM_DEFAULT_STEPS = [
        [0.01, 0.5, 10],
        [0.02, 0.5, 10],
        [0.03, 0.5, 10],
        [0.01, 0.0, 10],
    ]

    MESH_ELASTIC_PRESETS = {
        # Clean single-sheet surfaces (e.g. face scans). Stiff→soft schedule with normals.
        # Optional mask_subject_with_template crops subject geometry outside a warped-reference ROI.
        "facial": {
            "steps": [
                [0.045, 0.5, 10],
                [0.03, 0.5, 12],
                [0.018, 0.5, 15],
                [0.012, 0.5, 15],
                [0.009, 0.0, 12],
            ],
            "distance_threshold": 0.08,
            "use_vertex_normals": True,
        },
        # -------------------------------------------------------------------------
        # Complex mesh volume pathway — see module-level notes on _outer_shell_trimesh,
        # _nn_displacement_propagate_to_full_mesh, and _run_mesh_nricp outer_shell_only.
        #
        # Problem: marching-cubes iso-surfaces contain nested interior sheets that make
        # nearest-vertex NR-ICP correspondences ambiguous. Full MC clouds must not be
        # registered or upsampled directly if landmark transfer needs faithful geometry.
        #
        # Flow:
        #   1. Subject: user provides an outer-shell PLY, or auto_extract saves _shell.ply.
        #   2. Reference: outer shell extracted on the fly (voxel MC or mesh).
        #   3. NR-ICP on decimated shells (~35k cap); elastic output keeps the registration
        #      input topology (the edit immediately before elastic) index-for-index.
        #   4. When solver topology differs (shell/decimated), upsample via k-NN displacement
        #      copy (not global TPS, which produces long-range spikes on closed skull-like surfaces).
        # -------------------------------------------------------------------------
        "complex_mesh_volume": {
            "steps": [
                [0.04, 0.5, 12],
                [0.028, 0.5, 14],
                [0.018, 0.5, 16],
                [0.012, 0.5, 16],
                [0.008, 0.3, 12],
            ],
            "distance_threshold": 0.12,
            "use_vertex_normals": True,
            "outer_shell_only": True,
            "use_outer_shell": True,
        },
    }

    # Template→subject schedule for the ROI mask only. Must seat the warped template
    # well inside the parallel-to-surface crop pad (default 4% of reference centroid size);
    # a coarse fit leaves valid subject vertices outside the pad and punches holes.
    # Stiff→soft with a long, normal-free final stage for close apposition.
    # nricp_amberg distance_threshold is absolute mm (not a fraction); the mask pass
    # sets it to the pad so only in-pad correspondences drive the solve.
    MESH_ELASTIC_TEMPLATE_MASK_STEPS = [
        [0.05, 0.5, 15],
        [0.035, 0.5, 15],
        [0.025, 0.5, 18],
        [0.015, 0.5, 20],
        [0.01, 0.5, 20],
        [0.006, 0.0, 25],
        [0.004, 0.0, 25],
    ]
    MESH_ELASTIC_TEMPLATE_MASK_PAD_CENTROID_FRACTION = 0.04

    def _normalize_mesh_elastic_custom_options(self, request_data):
        raw = request_data.get("mesh_elastic_custom_options") if isinstance(request_data, dict) else None
        if not isinstance(raw, dict):
            raw = {}

        default_steps = self.MESH_ELASTIC_CUSTOM_DEFAULT_STEPS
        steps = []
        steps_raw = raw.get("steps")
        if isinstance(steps_raw, (list, tuple)) and steps_raw:
            for i, row in enumerate(steps_raw[:6]):
                default_row = default_steps[i] if i < len(default_steps) else default_steps[-1]
                if isinstance(row, (list, tuple)) and len(row) == 3:
                    steps.append([
                        _clip_float_value(row[0], 0.0001, 1.0, default_row[0]),
                        _clip_float_value(row[1], 0.0, 3.0, default_row[1]),
                        int(_clip_float_value(row[2], 1, 100, default_row[2])),
                    ])
        if not steps:
            steps = [list(row) for row in default_steps]

        return {
            "steps": steps,
            "gamma": _clip_float_value(raw.get("gamma"), 0.01, 10.0, 1.0),
            "eps": _clip_float_value(raw.get("eps"), 1e-6, 1.0, 1e-4),
        }

    def _resolve_mesh_elastic_settings(self, request_data):
        """Build full NR-ICP settings (steps as [ws, wl, wn, max_iter] rows, distance_threshold, etc.)."""
        mode = request_data.get("mode") or "custom"
        # Soft landmark/guidepoint terms are not used: landmarks are often the transfer
        # target, and guidepoints (when present) only initialize a Procrustes of the
        # floating mesh. wl is always 0. Final elastic is always subject→reference.
        # Rigid pose is always assumed from ALPACA / manual alignment — no extra ICP pass.
        rigid_prealignment = False
        # Reference surfaces are used as extracted (marching cubes or PLY) without Laplacian
        # smoothing so landmark transfer retains full geometric fidelity.
        reference_smoothing_iterations = 0
        if mode == "custom":
            custom = self._normalize_mesh_elastic_custom_options(request_data)
            steps3 = custom["steps"]
            gamma = custom["gamma"]
            eps = custom["eps"]
            distance_threshold = _clip_float_value(request_data.get("distance_threshold"), 0.01, 0.5, 0.10)
            use_vertex_normals = _parse_bool_request_flag(request_data.get("use_vertex_normals"), default=True)
            use_outer_shell = _parse_bool_request_flag(request_data.get("use_outer_shell"), default=False)
            outer_shell_only = _parse_bool_request_flag(request_data.get("outer_shell_only"), default=False)
            auto_extract_subject_outer_shell = _parse_bool_request_flag(
                request_data.get("auto_extract_subject_outer_shell"), default=False,
            )
            tps_alpha_spacing_fraction = _clip_float_value(
                request_data.get("tps_alpha_spacing_fraction"),
                0.0,
                2.0,
                MESH_ELASTIC_TPS_ALPHA_SPACING_FRACTION,
            )
        else:
            preset = self.MESH_ELASTIC_PRESETS.get(mode)
            if preset is None:
                raise ValueError(
                    f"Unknown mesh elastic registration mode: {mode!r}. "
                    f"Expected one of: custom, {', '.join(sorted(self.MESH_ELASTIC_PRESETS))}."
                )
            steps3 = preset["steps"]
            gamma = 1.0
            eps = 1e-4
            distance_threshold = preset["distance_threshold"]
            if "use_vertex_normals" in preset:
                use_vertex_normals = bool(preset["use_vertex_normals"])
            else:
                use_vertex_normals = _parse_bool_request_flag(
                    request_data.get("use_vertex_normals"), default=True
                )
            if "use_outer_shell" in preset:
                use_outer_shell = bool(preset["use_outer_shell"])
            else:
                use_outer_shell = _parse_bool_request_flag(
                    request_data.get("use_outer_shell"), default=False
                )
            if "outer_shell_only" in preset:
                outer_shell_only = bool(preset["outer_shell_only"])
            else:
                outer_shell_only = _parse_bool_request_flag(
                    request_data.get("outer_shell_only"), default=False
                )
            auto_extract_subject_outer_shell = _parse_bool_request_flag(
                request_data.get("auto_extract_subject_outer_shell"), default=False,
            )
            if "tps_alpha_spacing_fraction" in preset:
                tps_alpha_spacing_fraction = float(preset["tps_alpha_spacing_fraction"])
            else:
                tps_alpha_spacing_fraction = _clip_float_value(
                    request_data.get("tps_alpha_spacing_fraction"),
                    0.0,
                    2.0,
                    MESH_ELASTIC_TPS_ALPHA_SPACING_FRACTION,
                )

        return {
            "mode": mode,
            "steps": [[ws, 0.0, wn, max_iter] for ws, wn, max_iter in steps3],
            "gamma": gamma,
            "eps": eps,
            "distance_threshold": distance_threshold,
            "use_vertex_normals": use_vertex_normals,
            "rigid_prealignment": rigid_prealignment,
            "use_outer_shell": use_outer_shell,
            "outer_shell_only": outer_shell_only,
            "auto_extract_subject_outer_shell": auto_extract_subject_outer_shell,
            "tps_alpha_spacing_fraction": tps_alpha_spacing_fraction,
            # Internal only: template→subject warp used solely to crop the subject ROI.
            "homology_mode": False,
            "mask_subject_with_template": _parse_bool_request_flag(
                request_data.get("mask_subject_with_template"), default=False,
            ),
            "mask_pad_centroid_fraction": _clip_float_value(
                request_data.get("mask_pad_centroid_fraction"),
                0.005,
                0.20,
                self.MESH_ELASTIC_TEMPLATE_MASK_PAD_CENTROID_FRACTION,
            ),
            "reference_smoothing_iterations": reference_smoothing_iterations,
        }

    def _latest_reference_nifti_for_mesh_elastic(self, directory, reference, ref_metadata):
        ref_dir = self._subject_dir(directory, reference)
        latest_edit_number = -1
        latest_path = None
        for path in glob.glob(os.path.join(ref_dir, f"{reference}_edit_*.nii.gz")):
            basename = os.path.basename(path)
            try:
                edit_number = int(basename.split("_edit_")[1].split("_")[0])
            except (IndexError, ValueError):
                continue
            if edit_number > latest_edit_number:
                latest_edit_number = edit_number
                latest_path = path
        if latest_path is None:
            latest_path = os.path.join(ref_dir, f"{reference}.nii.gz")
        voxel_size = ref_metadata.get("voxel_size") or 1.0
        return latest_path, float(voxel_size)

    def _load_reference_surface_for_mesh_elastic(
        self, directory, reference, smoothing_iterations, use_outer_shell=False,
    ):
        """
        Resolve the reference geometry as a Trimesh in the registration frame used by
        aligned preserved-mesh subjects (prism corner-origin shared mm).

        Preserved-mesh PLYs on disk are centroid-local; they are mapped into shared mm
        here. Voxel references are surfaced with marching cubes using the same convention
        (spacing = voxel size, Gaussian pre-smoothing, reversed face winding, NIfTI axes
        0↔2 swapped) already used to build registration/display meshes from voxel volumes
        elsewhere (see _load_voxel_subject_display_mesh in views.py).

        Reference surfaces are **not** Laplacian-smoothed: smoothing would blur detail needed
        for accurate landmark transfer. For complex mesh volume mode, ``use_outer_shell`` also
        strips nested MC sheets via ``_outer_shell_trimesh`` after extraction.
        """
        _, ref_metadata = self._load_metadata(directory, reference)
        if is_preserved_mesh_metadata(ref_metadata):
            mesh_path = self._source_mesh_path(directory, reference, "latest", ref_metadata)
            _mesh_elastic_log(f"Loading preserved-mesh reference surface from {mesh_path}")
            reference_mesh = self._load_mesh(mesh_path)
            reference_mesh = mesh_elastic_reference_in_registration_frame(
                reference_mesh, ref_metadata, directory
            )
            _mesh_elastic_log(
                f"{_mesh_elastic_mesh_summary(reference_mesh, 'Reference mesh')} "
                f"(prism corner-origin shared mm)"
            )
        else:
            nifti_path, voxel_size = self._latest_reference_nifti_for_mesh_elastic(directory, reference, ref_metadata)
            _mesh_elastic_log(
                f"Extracting voxel reference surface via marching cubes from {nifti_path} "
                f"(voxel_size={voxel_size:.4f} mm, smoothing_iterations={smoothing_iterations})"
            )
            t0 = time.perf_counter()
            volume = nib.load(nifti_path).get_fdata()
            # Match _load_voxel_subject_display_mesh / aligned preserved-mesh PLY basis:
            # swap NIfTI axes 0↔2 so marching-cubes mm coords use display [X,Y,Z].
            volume = np.swapaxes(volume, 0, 2)
            _mesh_elastic_log(f"Loaded reference volume shape {volume.shape} (display axes), dtype {volume.dtype}")
            volume = gaussian_filter(volume, sigma=1.0)
            threshold = ref_metadata.get("threshold")
            if threshold is None:
                threshold = float(np.mean(volume))
            _mesh_elastic_log(f"Marching cubes at threshold={threshold}")
            verts, faces, _, _ = measure.marching_cubes(volume, level=float(threshold), spacing=(voxel_size,) * 3)
            faces = faces[:, ::-1]
            reference_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            _mesh_elastic_log(
                f"Marching cubes finished in {time.perf_counter() - t0:.1f}s — "
                f"{_mesh_elastic_mesh_summary(reference_mesh, 'Extracted reference mesh')}"
            )

        if smoothing_iterations > 0:
            smooth_t0 = time.perf_counter()
            trimesh.smoothing.filter_laplacian(reference_mesh, iterations=int(smoothing_iterations))
            _mesh_elastic_log(
                f"Laplacian smoothing ({smoothing_iterations} iterations) finished in "
                f"{time.perf_counter() - smooth_t0:.1f}s"
            )
        if use_outer_shell:
            reference_mesh = _outer_shell_trimesh(reference_mesh, label="Reference mesh")
        return reference_mesh

    def _mesh_elastic_error_threshold_mm(self, ref_metadata, reference_mesh, directory=None):
        if directory is not None and is_preserved_mesh_metadata(ref_metadata):
            voxel_size = resolve_mesh_reference_voxel_size(
                directory,
                reference_mesh.vertices,
                ref_metadata.get("mesh_metadata"),
            )
        else:
            voxel_size = ref_metadata.get("voxel_size")
            if not isinstance(voxel_size, (int, float)) or voxel_size <= 0:
                voxel_size = derive_mesh_reference_voxel_size(
                    reference_mesh.vertices, ref_metadata
                )
        return float(voxel_size) * 6.0

    def _persist_mesh_elastic_reference_surface(self, directory, reference, reference_mesh, ref_metadata):
        """
        Record the exact reference surface used for NR-ICP so later heatmap/metric reloads use the
        same geometry. Voxel references are written as a dedicated PLY; preserved-mesh references
        store the basename of the PLY that was loaded.
        """
        ref_dir = self._subject_dir(directory, reference)
        ref_json_path, ref_metadata = self._load_metadata(directory, reference)

        if is_preserved_mesh_metadata(ref_metadata):
            ref_mesh_path = self._source_mesh_path(directory, reference, "latest", ref_metadata)
            ref_metadata["mesh_elastic_reference_ply"] = os.path.basename(ref_mesh_path)
            _mesh_elastic_log(
                f"Reference surface for mesh elastic metrics: preserved mesh "
                f"{ref_metadata['mesh_elastic_reference_ply']}"
            )
        else:
            ref_ply_name = mesh_elastic_reference_ply_basename(reference)
            ref_ply_path = os.path.join(ref_dir, ref_ply_name)
            self._save_mesh_edit(reference_mesh, ref_ply_path)
            ref_metadata["mesh_elastic_reference_ply"] = ref_ply_name
            _mesh_elastic_log(
                f"Saved voxel-derived mesh-elastic reference surface to {ref_ply_path} "
                f"({len(reference_mesh.vertices):,} vertices)"
            )

        with open(ref_json_path, "w") as jf:
            json.dump(ref_metadata, jf, indent=4)
        return ref_metadata

    def _load_mesh_elastic_reference_surface(self, directory, reference, ref_metadata=None):
        if ref_metadata is None:
            _, ref_metadata = self._load_metadata(directory, reference)
        ref_dir = self._subject_dir(directory, reference)
        ref_ply_name = ref_metadata.get("mesh_elastic_reference_ply")
        if ref_ply_name:
            ref_ply_path = os.path.join(ref_dir, os.path.basename(ref_ply_name))
            if os.path.isfile(ref_ply_path):
                mesh = self._load_mesh(ref_ply_path)
                return mesh_elastic_reference_in_registration_frame(mesh, ref_metadata, directory)

        if is_preserved_mesh_metadata(ref_metadata):
            ref_mesh_path = self._source_mesh_path(directory, reference, "latest", ref_metadata)
        else:
            ref_mesh_path = os.path.join(ref_dir, mesh_elastic_reference_ply_basename(reference))
        if not os.path.isfile(ref_mesh_path):
            raise FileNotFoundError(f"Mesh elastic reference surface not found for {reference}: {ref_mesh_path}")
        mesh = self._load_mesh(ref_mesh_path)
        return mesh_elastic_reference_in_registration_frame(mesh, ref_metadata, directory)

    def _compute_mesh_elastic_surface_metrics(
        self, elastic_mesh, reference_mesh, ref_metadata, subject_label, directory,
    ):
        # Distances here are for the registration accuracy score written to metadata only.
        # The per-subject display heatmap is computed view-time in MarchingCubesView
        # (_load_preserved_mesh_response) from the elastic PLY against the reference.
        # Elastic always deforms the subject onto the reference, so the metric is
        # deformed-subject vertices → reference surface.
        # Always evaluate on outer shells only (matches volumetric elastic
        # calculate_surface_distance), so nested MC sheets do not inflate the outlier rate.
        error_threshold_mm = self._mesh_elastic_error_threshold_mm(
            ref_metadata, reference_mesh, directory,
        )
        voxel_size_mm = float(error_threshold_mm) / 6.0
        metric_note = "outer-shell distance to reference"
        query_shell = _outer_shell_trimesh(
            elastic_mesh, label=f"{subject_label} surface-metric query",
        )
        reference_shell = _outer_shell_trimesh(
            reference_mesh, label=f"{subject_label} surface-metric reference",
        )
        distances = compute_mesh_to_reference_surface_distances(
            query_shell.vertices, reference_shell, label=subject_label,
        )
        above_fraction, surface_score, mean_distance_mm = mesh_elastic_surface_metric(
            distances, error_threshold_mm,
        )
        above_fraction_1, surface_score_1, _ = mesh_elastic_surface_metric(
            distances, voxel_size_mm,
        )
        _mesh_elastic_log(
            f"{subject_label}: surface metric ({metric_note}) — mean={mean_distance_mm:.4f} mm, "
            f"within {error_threshold_mm:.3f} mm (6×voxel) on {surface_score * 100:.1f}% of vertices, "
            f"within {voxel_size_mm:.3f} mm (1×voxel) on {surface_score_1 * 100:.1f}%"
        )
        return {
            "elastic_surface_distance": surface_score,
            "elastic_surface_distance_1vox": surface_score_1,
            "elastic_error": mean_distance_mm,
            "surface_error_threshold_mm": error_threshold_mm,
            "surface_above_threshold_fraction": above_fraction,
            "surface_above_1vox_fraction": above_fraction_1,
        }

    def _load_guidepoints_array(self, path):
        if not path or not os.path.isfile(path):
            return None
        try:
            with open(path, "r") as jf:
                data = np.asarray(json.load(jf), dtype=np.float64)
        except (OSError, ValueError, TypeError):
            return None
        if data.ndim != 2 or data.shape[1] != 3 or len(data) == 0:
            return None
        return data

    @staticmethod
    def _similarity_transform_from_points(source_points, target_points):
        """
        Umeyama similarity (scale + rotation + translation) mapping source→target.
        Matches MeshMonk ``computeTransform(..., true)`` for guidepoint initialization.
        """
        src = np.asarray(source_points, dtype=np.float64)
        dst = np.asarray(target_points, dtype=np.float64)
        if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3 or len(src) < 3:
            raise ValueError("Need at least 3 paired Nx3 guidepoints for similarity init.")
        src_c = src.mean(axis=0)
        dst_c = dst.mean(axis=0)
        src_h = src - src_c
        dst_h = dst - dst_c
        ss_src = float(np.sum(src_h * src_h))
        if ss_src <= 0:
            raise ValueError("Degenerate guidepoint configuration (zero source variance).")
        H = src_h.T @ dst_h
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt = Vt.copy()
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        scale = float(np.sum(S) / ss_src)
        t = dst_c - scale * (R @ src_c)
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = scale * R
        matrix[:3, 3] = t
        return matrix

    def _gather_mesh_elastic_guidepoints_for_init(
        self, directory, filename, reference, subject_metadata,
    ):
        """
        Optional propagated manual guidepoints for template Procrustes init.

        Homologous pairing is enforced when guidepoints are saved in the UI; load
        ``{latest_stem}_guidepoints.json`` on subject and reference only.

        Returns (reference_positions, subject_positions, origin) or (None, None, None)
        when either scan has no guidepoints. Raises ValueError when both exist but
        cannot be used for Procrustes (count mismatch or fewer than 3 points).
        """
        subject_dir = self._subject_dir(directory, filename)
        subject_stem = self._mesh_stem(self._source_mesh_path(directory, filename, "latest", subject_metadata))
        subject_gp_path = os.path.join(subject_dir, f"{subject_stem}_guidepoints.json")
        subject_positions = self._load_guidepoints_array(subject_gp_path)
        if subject_positions is None:
            return None, None, None

        _, ref_metadata = self._load_metadata(directory, reference)
        if not is_preserved_mesh_metadata(ref_metadata):
            return None, None, None
        ref_dir = self._subject_dir(directory, reference)
        ref_stem = self._mesh_stem(self._source_mesh_path(directory, reference, "latest", ref_metadata))
        reference_positions = self._load_guidepoints_array(
            os.path.join(ref_dir, f"{ref_stem}_guidepoints.json")
        )
        if reference_positions is None:
            return None, None, None

        if len(subject_positions) != len(reference_positions):
            raise ValueError(
                f"Guidepoint count mismatch for {filename}: subject has "
                f"{len(subject_positions)}, reference has {len(reference_positions)}"
            )
        if len(subject_positions) < 3:
            raise ValueError(
                f"At least 3 homologous guidepoints are required for Procrustes init "
                f"({filename}: {len(subject_positions)})"
            )

        ref_mesh_path = self._source_mesh_path(directory, reference, "latest", ref_metadata)
        local_template = self._load_mesh(ref_mesh_path)
        reference_positions = mesh_local_to_shared_mm(
            reference_positions,
            ref_metadata.get("mesh_metadata"),
            prism_vertices=local_template.vertices,
            directory=directory,
        )
        return reference_positions, subject_positions, "manual_guidepoints"

    @staticmethod
    def _mesh_centroid_size(mesh):
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        if len(vertices) == 0:
            return 0.0
        centered = vertices - vertices.mean(axis=0)
        return float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))

    def _keep_largest_connected_component(self, mesh):
        """Drop disconnected islands; keep the component with the largest surface area."""
        edges = mesh.edges_unique
        vertex_count = len(mesh.vertices)
        if len(edges) == 0 or vertex_count == 0:
            return mesh
        data = np.ones(len(edges), dtype=bool)
        adj = sp.coo_matrix((data, (edges[:, 0], edges[:, 1])), shape=(vertex_count, vertex_count))
        adj = adj + adj.T
        n_components, vertex_labels = sp.csgraph.connected_components(adj, directed=False)
        if n_components <= 1:
            return mesh
        face_labels = vertex_labels[mesh.faces[:, 0]]
        face_areas = mesh.area_faces
        best_label = None
        best_area = -1.0
        for i in range(n_components):
            mask = face_labels == i
            area = float(np.sum(face_areas[mask])) if np.any(mask) else 0.0
            if area > best_area:
                best_area = area
                best_label = i
        if best_label is None:
            return mesh
        kept = trimesh.Trimesh(
            vertices=mesh.vertices.copy(),
            faces=mesh.faces[face_labels == best_label],
            process=False,
        )
        kept.remove_unreferenced_vertices()
        return kept

    def _clip_polygon_by_distance(self, polygon, distances, max_distance_mm, epsilon=1e-9):
        """
        Keep the portion of a polygon where surface distance <= ``max_distance_mm``.

        Same Sutherland–Hodgman edge intersection as ``_clip_polygon_to_halfspace``
        (crop / slicer clean cuts), but the halfspace is the distance isosurface:
        signed = pad - distance, inside when signed >= 0. Crossing edges get a new
        vertex exactly on the pad boundary so the cut is a clean polyline, not a
        jagged stair of whole triangles.
        """
        if not polygon:
            return []

        signed = [float(max_distance_mm) - float(distance) for distance in distances]
        clipped = []
        previous = np.asarray(polygon[-1], dtype=np.float64)
        previous_signed = signed[-1]
        previous_inside = previous_signed >= -epsilon

        for current_point, current_signed in zip(polygon, signed):
            current = np.asarray(current_point, dtype=np.float64)
            current_inside = current_signed >= -epsilon

            if current_inside != previous_inside:
                denominator = previous_signed - current_signed
                if abs(denominator) > epsilon:
                    t = previous_signed / denominator
                    t = float(np.clip(t, 0.0, 1.0))
                    clipped.append(previous + t * (current - previous))

            if current_inside:
                clipped.append(current)

            previous = current
            previous_signed = current_signed
            previous_inside = current_inside

        deduped = []
        for point in clipped:
            if not deduped or not np.allclose(point, deduped[-1], atol=epsilon):
                deduped.append(point)
        if len(deduped) > 1 and np.allclose(deduped[0], deduped[-1], atol=epsilon):
            deduped.pop()
        return deduped

    def _crop_mesh_by_surface_distance(self, subject_mesh, guide_mesh, max_distance_mm, label="subject"):
        """
        Keep subject geometry within a parallel-to-surface pad of the guide mesh.

        Pad uses the separable normal/tangent slab
        ``max(|n·(q−c)|, ‖(q−c)_t‖) ≤ pad``: the footprint expands smoothly by
        ``pad`` along the surface, with the same seating thickness along the
        normal — without the rounded isotropic rim tube. Boundary faces are
        edge-interpolated on the pad isosurface (same clean cut as crop/slicer).
        """
        subject_vertices = np.asarray(subject_mesh.vertices, dtype=np.float64)
        subject_faces = np.asarray(subject_mesh.faces, dtype=np.int64)
        if len(guide_mesh.vertices) == 0 or len(subject_vertices) == 0 or subject_faces.size == 0:
            raise ValueError(f"{label}: cannot crop — empty subject or guide mesh")

        distances = compute_mesh_to_reference_parallel_surface_distances(
            subject_vertices, guide_mesh, label=f"{label} mask-crop",
        )
        max_d = float(max_distance_mm)
        epsilon = 1e-9
        # signed >= 0 ⇒ inside the pad (same convention as plane halfspaces).
        signed = max_d - distances
        face_signed = signed[subject_faces]
        fully_inside = (face_signed >= -epsilon).all(axis=1)
        fully_outside = (face_signed < -epsilon).all(axis=1)
        needs_clip = ~fully_inside & ~fully_outside

        _mesh_elastic_log(
            f"{label}: template mask clip (parallel-to-surface pad) — {len(subject_faces):,} faces "
            f"({int(np.sum(fully_inside)):,} inside, {int(np.sum(fully_outside)):,} outside, "
            f"{int(np.sum(needs_clip)):,} boundary)"
        )

        polygons = []
        if np.any(fully_inside):
            polygons.extend(subject_vertices[subject_faces[fully_inside]])

        for face in subject_faces[needs_clip]:
            polygon = [subject_vertices[int(index)] for index in face]
            face_distances = [distances[int(index)] for index in face]
            clipped = self._clip_polygon_by_distance(polygon, face_distances, max_d, epsilon=epsilon)
            if len(clipped) >= 3:
                polygons.append(clipped)

        if not polygons:
            raise ValueError(
                f"{label}: template mask removed all faces "
                f"(pad={max_d:.4f} mm)"
            )

        cropped = self._mesh_from_polygons(
            polygons,
            empty_error=f"{label}: template mask produced an empty mesh",
        )
        cropped = self._keep_largest_connected_component(cropped)
        if len(cropped.faces) == 0 or len(cropped.vertices) == 0:
            raise ValueError(f"{label}: template mask produced an empty mesh")

        keep_vertex = distances <= max_d
        _mesh_elastic_log(
            f"{label}: template mask crop — kept {len(cropped.vertices):,}/{len(subject_vertices):,} "
            f"vertices, {len(cropped.faces):,}/{len(subject_faces):,} faces "
            f"(parallel-to-surface pad={max_d:.4f} mm, clean isosurface clip)"
        )
        return cropped, distances, keep_vertex

    def _template_mask_settings(self, settings, max_distance_mm):
        """
        NR-ICP settings for the internal template→subject warp used only as an ROI guide.

        ``distance_threshold`` is the crop pad in mm: trimesh rejects farther matches, so
        the solve is driven only by correspondences that will survive the hard crop.
        """
        steps3 = self.MESH_ELASTIC_TEMPLATE_MASK_STEPS
        return {
            **settings,
            "steps": [[ws, 0.0, wn, max_iter] for ws, wn, max_iter in steps3],
            "distance_threshold": float(max_distance_mm),
            "eps": min(float(settings.get("eps", 1e-4)), 1e-5),
            "use_vertex_normals": True,
            "homology_mode": True,
        }

    def _mask_subject_with_warped_template(
        self,
        subject_mesh,
        reference_mesh,
        settings,
        subject_label=None,
        init_reference_guidepoints=None,
        init_subject_guidepoints=None,
    ):
        """
        Warp the reference template onto the subject, then hard-crop subject geometry
        outside a parallel-to-surface pad of the warped template.

        Pad thickness is ``mask_pad_centroid_fraction`` × reference centroid size,
        applied as ``max(|normal offset|, tangential exterior)`` so the ROI expands
        smoothly along the surface without an isotropic rim tube.
        Returns (masked_mesh, mask_details).
        """
        label = subject_label or "subject"
        pad_fraction = float(settings.get(
            "mask_pad_centroid_fraction",
            self.MESH_ELASTIC_TEMPLATE_MASK_PAD_CENTROID_FRACTION,
        ))
        centroid_size = self._mesh_centroid_size(reference_mesh)
        if centroid_size <= 0:
            raise ValueError(f"{label}: reference centroid size is zero; cannot build template mask")
        max_distance_mm = centroid_size * pad_fraction

        _mesh_elastic_log(
            f"{label}: template-mask pass — warping reference onto subject "
            f"(parallel-to-surface pad={pad_fraction * 100:.2f}% CS = {max_distance_mm:.4f} mm, "
            f"reference CS={centroid_size:.4f} mm; NR-ICP distance_threshold={max_distance_mm:.4f} mm)"
        )
        mask_settings = self._template_mask_settings(settings, max_distance_mm)
        warped_template, _ = self._run_mesh_nricp(
            subject_mesh,
            reference_mesh,
            mask_settings,
            subject_label=f"{label} template-mask",
            init_reference_guidepoints=init_reference_guidepoints,
            init_subject_guidepoints=init_subject_guidepoints,
        )

        # Fit check: warped template → subject surface. Must sit inside the pad or the
        # crop will remove valid face regions and leave holes.
        template_fit = compute_mesh_to_reference_surface_distances(
            warped_template.vertices, subject_mesh, label=f"{label} template-fit",
        )
        fit_mean = float(np.mean(template_fit))
        fit_max = float(np.max(template_fit))
        fit_within = float(np.mean(template_fit <= max_distance_mm))
        _mesh_elastic_log(
            f"{label}: template→subject fit — mean={fit_mean:.4f} mm, max={fit_max:.4f} mm, "
            f"{fit_within * 100:.1f}% of template vertices within pad {max_distance_mm:.4f} mm"
        )

        masked_mesh, distances, keep_vertex = self._crop_mesh_by_surface_distance(
            subject_mesh, warped_template, max_distance_mm, label=label,
        )
        details = {
            "mask_pad_centroid_fraction": pad_fraction,
            "mask_pad_mm": max_distance_mm,
            "mask_pad_kind": "parallel_to_surface",
            "reference_centroid_size_mm": centroid_size,
            "vertices_before": int(len(subject_mesh.vertices)),
            "faces_before": int(len(subject_mesh.faces)),
            "vertices_after": int(len(masked_mesh.vertices)),
            "faces_after": int(len(masked_mesh.faces)),
            "vertices_kept_fraction": float(np.mean(keep_vertex)),
            "mean_distance_to_template_mm": float(np.mean(distances)),
            "template_fit_mean_mm": fit_mean,
            "template_fit_max_mm": fit_max,
            "template_fit_within_pad_fraction": fit_within,
        }
        return masked_mesh, details

    def _create_subject_outer_shell_edit(
        self, directory, filename, subject_dir, json_path, metadata, subject_label=None,
    ):
        """
        Complex mesh volume — optional pre-registration step.

        When ``auto_extract_subject_outer_shell`` is requested, extract the outer shell from
        each subject's latest mesh and persist ``{name}_edit_{n}_shell.ply`` (same convention
        as ApplyMeshShellView) before NR-ICP. Landmarks on the shell surface are kept; others
        are dropped. Returns the in-memory shell mesh plus updated metadata paths.
        """
        label = subject_label or filename
        source_path = self._source_mesh_path(directory, filename, "latest", metadata)
        _mesh_elastic_log(f"{label}: auto-extracting outer shell from {source_path}")
        mesh = self._load_mesh(source_path)
        shell_mesh = _outer_shell_trimesh(mesh, label=label)
        if len(shell_mesh.vertices) == 0 or len(shell_mesh.faces) == 0:
            raise ValueError(f"{label}: outer shell extraction produced an empty mesh")

        edit_number = self._next_edit_number(subject_dir, filename)
        edit_filename = f"{filename}_edit_{edit_number}_shell.ply"
        output_path = os.path.join(subject_dir, edit_filename)
        self._save_mesh_edit(shell_mesh, output_path)
        _mesh_elastic_log(f"{label}: saved outer shell edit to {output_path}")

        edit_stem = self._mesh_stem(edit_filename)
        source_stem = self._mesh_stem(source_path)
        _, source_landmarks = self._load_landmark_rows_for_stem(subject_dir, source_stem)
        landmark_details = {}
        if source_landmarks is not None:
            shell_landmarks, kept_landmark_indices = _filter_landmarks_to_shell_mesh(
                source_landmarks, shell_mesh,
            )
            landmarks_path = self._write_landmark_rows_for_edit(subject_dir, edit_stem, shell_landmarks)
            distances_path = self._copy_landmark_distances_for_edit(
                subject_dir,
                source_stem,
                edit_stem,
                kept_indices=kept_landmark_indices,
            )
            landmark_details = {
                "source_landmarks": os.path.basename(
                    os.path.join(subject_dir, f"{source_stem}_landmarks.json")
                ),
                "landmarks_file": os.path.basename(landmarks_path),
                "landmark_distances_file": os.path.basename(distances_path) if distances_path else None,
                "landmark_count": len(shell_landmarks),
            }

        self._append_edit_metadata(json_path, metadata, edit_filename, "shell", {
            "source": os.path.basename(source_path),
            "shell_mode": "alpaca_outer_mesh",
            "auto_extracted_for_mesh_elastic": True,
            **landmark_details,
            "mesh_summary": self._mesh_summary(shell_mesh),
        })
        json_path, metadata = self._load_metadata(directory, filename)
        return shell_mesh, edit_filename, json_path, metadata

    def _run_mesh_nricp(
        self,
        subject_mesh,
        reference_mesh,
        settings,
        subject_label=None,
        init_reference_guidepoints=None,
        init_subject_guidepoints=None,
    ):
        label = subject_label or "subject"
        vertex_cap = MESH_ELASTIC_NRICP_VERTEX_CAP
        homology_mode = bool(settings.get("homology_mode", False))

        # homology_mode is internal only (template-mask pass): deform the template onto the
        # subject as an ROI guide. Default / final elastic: deform the subject onto the
        # reference (subject topology preserved).
        if homology_mode:
            full_source_mesh = reference_mesh
            full_target_mesh = subject_mesh
            source_role, target_role = "template", "subject"
        else:
            full_source_mesh = subject_mesh
            full_target_mesh = reference_mesh
            source_role, target_role = "subject", "reference"

        use_outer_shell = bool(settings.get("use_outer_shell", False))
        outer_shell_only = bool(settings.get("outer_shell_only", False))
        # Output topology always matches the registration input mesh (the edit saved
        # immediately before elastic). Shell extraction / decimation apply only to the
        # NR-ICP solver meshes; displacements are propagated back when topologies differ.
        output_mesh = full_source_mesh
        nricp_source_mesh = full_source_mesh
        nricp_target_mesh = full_target_mesh
        if outer_shell_only and not homology_mode:
            shell_source = _outer_shell_trimesh(full_source_mesh, f"{label} {source_role}")
            nricp_source_mesh = shell_source
            _mesh_elastic_log(
                f"{label}: outer-shell-only — NR-ICP solver uses outer shell "
                f"({len(shell_source.vertices):,} vertices); elastic output preserves "
                f"registration input ({len(output_mesh.vertices):,} vertices)"
            )
        elif use_outer_shell:
            nricp_source_mesh = _outer_shell_trimesh(full_source_mesh, f"{label} {source_role}")
            nricp_target_mesh = _outer_shell_trimesh(full_target_mesh, f"{label} {target_role}")
            _mesh_elastic_log(
                f"{label}: NR-ICP solver uses outer shell(s) "
                f"({len(nricp_source_mesh.vertices):,} subject, "
                f"{len(nricp_target_mesh.vertices):,} reference vertices); "
                f"elastic output preserves registration input "
                f"({len(output_mesh.vertices):,} vertices)"
            )

        reg_source, source_decimated = _decimate_mesh_for_nricp(
            nricp_source_mesh, vertex_cap, f"{label} {source_role}",
        )
        reg_target, target_decimated = _decimate_mesh_for_nricp(
            nricp_target_mesh, vertex_cap, f"{label} {target_role}",
        )

        # Decimation is what introduces the debris/singular-matrix risk (see _sanitize_mesh_for_nricp
        # docstring), so only sanitize when it actually ran — meshes already within the cap skip
        # this destructive step entirely, preserving their exact vertex count/order so the saved
        # elastic edit still lines up index-for-index with its pre-elastic predecessor edit.
        if source_decimated:
            reg_source = _sanitize_mesh_for_nricp(reg_source, f"{label} {source_role}")
        if target_decimated:
            reg_target = _sanitize_mesh_for_nricp(reg_target, f"{label} {target_role}")

        working = reg_source.copy()
        rigid_matrix = np.eye(4, dtype=np.float64)

        # MeshMonk-style: optional project-wide manual guidepoints only initialize a similarity
        # transform of the floating mesh. They are not soft NR-ICP anchors. If absent, assume
        # ALPACA (or another rigid method) already placed the subject; skip extra ICP unless
        # rigid_prealignment is explicitly enabled.
        used_guidepoint_init = False
        if (
            init_reference_guidepoints is not None
            and init_subject_guidepoints is not None
            and len(init_reference_guidepoints) >= 3
            and len(init_subject_guidepoints) >= 3
        ):
            if homology_mode:
                # Template guidepoints → subject guidepoints.
                rigid_matrix = self._similarity_transform_from_points(
                    init_reference_guidepoints, init_subject_guidepoints,
                )
            else:
                # Subject guidepoints → reference guidepoints.
                rigid_matrix = self._similarity_transform_from_points(
                    init_subject_guidepoints, init_reference_guidepoints,
                )
            working.vertices = ALPACA.apply_rigid_transform(working.vertices, rigid_matrix)
            used_guidepoint_init = True
            _mesh_elastic_log(
                f"{label}: similarity Procrustes init from {len(init_reference_guidepoints)} "
                f"guidepoints ({source_role} → {target_role})"
            )

        if not used_guidepoint_init and settings["rigid_prealignment"]:
            _mesh_elastic_log(
                f"{label}: starting scaled-rigid ICP ({source_role} → {target_role}, max 30 iterations)..."
            )
            icp_t0 = time.perf_counter()
            # Pass target vertices (a plain array) rather than the Trimesh itself: trimesh's
            # icp() routes mesh targets through proximity.closest_point, which allocates one
            # unchunked (query_point, candidate_triangle) array per call and has no size limit —
            # see the MESH_ELASTIC_NRICP_VERTEX_CAP comment above for why that blows up memory.
            # A plain point array instead uses a cKDTree nearest-vertex query, which is memory-safe.
            rigid_matrix, transformed, icp_cost = trimesh.registration.icp(
                working.vertices, reg_target.vertices, max_iterations=30,
            )
            working.vertices = transformed
            _mesh_elastic_log(
                f"{label}: scaled-rigid ICP finished in {time.perf_counter() - icp_t0:.1f}s "
                f"(final cost={icp_cost:.6f})"
            )
        elif not used_guidepoint_init:
            _mesh_elastic_log(
                f"{label}: skipping rigid pre-alignment (assuming prior ALPACA / rigid alignment)"
            )

        reg_pre = working.copy()
        stage_count = len(settings["steps"])
        total_max_iters = sum(int(step[3]) for step in settings["steps"])
        _mesh_elastic_log(
            f"{label}: starting NR-ICP "
            f"({'homology: template → subject' if homology_mode else 'subject → reference'}) "
            f"({_mesh_elastic_mesh_summary(working, f'registration {source_role}')}) — "
            f"{stage_count} stage(s), up to {total_max_iters} total iterations, "
            f"distance_threshold={settings['distance_threshold']:.3f}, "
            f"use_vertex_normals={settings['use_vertex_normals']}, "
            f"guidepoint_init={'yes' if used_guidepoint_init else 'no'}"
        )

        nricp_t0 = time.perf_counter()
        for stage_idx, step in enumerate(settings["steps"], start=1):
            ws, wl, wn, max_iter = step
            _mesh_elastic_log(
                f"{label}: NR-ICP stage {stage_idx}/{stage_count} running — "
                f"stiffness={ws}, landmark_weight={wl}, normal_weight={wn}, max_iter={max_iter}"
            )
            stage_t0 = time.perf_counter()
            registered_vertices = trimesh.registration.nricp_amberg(
                working,
                reg_target,
                source_landmarks=None,
                target_positions=None,
                steps=[step],
                eps=settings["eps"],
                gamma=settings["gamma"],
                distance_threshold=settings["distance_threshold"],
                use_vertex_normals=settings["use_vertex_normals"],
                # use_faces=False forces nearest-*vertex* correspondence search (cKDTree) instead
                # of nearest-point-on-triangle (proximity.closest_point). At the vertex density
                # enforced by MESH_ELASTIC_NRICP_VERTEX_CAP the two are nearly equivalent in
                # practice, but closest_point has no chunking and is the main cause of the memory
                # blowups/system freezes seen on dense meshes — see the cap comment above.
                use_faces=False,
            )
            if isinstance(registered_vertices, list):
                registered_vertices = registered_vertices[0]
            working = trimesh.Trimesh(vertices=registered_vertices, faces=working.faces, process=False)
            _mesh_elastic_log(
                f"{label}: stage {stage_idx}/{stage_count} finished in {time.perf_counter() - stage_t0:.1f}s"
            )

        _mesh_elastic_log(f"{label}: NR-ICP finished in {time.perf_counter() - nricp_t0:.1f}s total")

        solver_matches_output = _meshes_share_topology(working, output_mesh)
        if solver_matches_output and not source_decimated:
            registered_mesh = trimesh.Trimesh(
                vertices=working.vertices, faces=output_mesh.faces, process=False,
            )
            propagation_note = ""
        elif outer_shell_only:
            registered_mesh = _nn_displacement_propagate_to_full_mesh(
                output_mesh, reg_pre, working, rigid_matrix, label,
            )
            propagation_note = " (k-NN displacement propagation onto registration input)"
        else:
            registered_mesh = _tps_propagate_to_full_mesh(
                output_mesh,
                reg_pre,
                working,
                rigid_matrix,
                label,
                tps_alpha_spacing_fraction=settings.get("tps_alpha_spacing_fraction"),
            )
            propagation_note = (
                " (TPS propagation onto registration input)"
                if source_decimated
                else " (TPS propagation from solver shell onto registration input)"
            )

        if not homology_mode:
            _assert_elastic_preserves_source_topology(output_mesh, registered_mesh, label)

        baseline = output_mesh.vertices
        mean_disp = float(np.mean(np.linalg.norm(registered_mesh.vertices - baseline, axis=1)))
        max_disp = float(np.max(np.linalg.norm(registered_mesh.vertices - baseline, axis=1)))
        _mesh_elastic_log(
            f"{label}: {source_role} vertex displacement mean={mean_disp:.4f} mm, max={max_disp:.4f} mm"
            + propagation_note
        )
        # Second return value: whether the *saved* mesh topology was decimated-then-TPS'd.
        # Homology saves reference topology; default saves subject topology.
        return registered_mesh, source_decimated

    def _latest_elastic_mesh_edit(self, subject_dir, filename):
        return self._latest_edit_file_matching(subject_dir, filename, "elastic")

    def _latest_edit_file_matching(self, subject_dir, filename, suffix):
        matches = glob.glob(os.path.join(subject_dir, f"{filename}_edit_*_{suffix}.ply"))
        if not matches:
            return None

        def edit_number(path):
            basename = os.path.basename(path)
            try:
                return int(basename.split("_edit_")[1].split("_")[0])
            except (IndexError, ValueError):
                return -1

        return max(matches, key=edit_number)

    def _edit_number_from_mesh_path(self, path):
        basename = os.path.basename(path)
        try:
            return int(basename.split("_edit_")[1].split("_")[0])
        except (IndexError, ValueError):
            return -1

    def _mesh_edit_path_by_number(self, subject_dir, filename, edit_number):
        matches = glob.glob(os.path.join(subject_dir, f"{filename}_edit_{edit_number}_*.ply"))
        return matches[0] if matches else None

    def _load_latest_mesh_landmarks(self, directory, filename):
        """Landmark rows (list of {position, landmark_type}) for a preserved-mesh subject's latest edit, or None."""
        _, metadata = self._load_metadata(directory, filename)
        subject_dir = self._subject_dir(directory, filename)
        mesh_path = self._source_mesh_path(directory, filename, "latest", metadata)
        stem = self._mesh_stem(mesh_path)
        _, landmarks = self._load_landmark_rows_for_stem(subject_dir, stem)
        return landmarks

    def _resolve_pre_elastic_mesh_path(self, subject_dir, filename, elastic_edit_path, metadata=None):
        """
        Resolve the mesh edit whose topology matches ``elastic_edit_path``.

        Uses the PLY edit numbered immediately before elastic (mask-then-elastic keeps
        the masked edit at N-1).
        """
        edit_number = self._edit_number_from_mesh_path(elastic_edit_path)
        if edit_number > 0:
            pre_elastic_path = self._mesh_edit_path_by_number(subject_dir, filename, edit_number - 1)
            if pre_elastic_path and os.path.isfile(pre_elastic_path):
                return pre_elastic_path

        base_path = os.path.join(subject_dir, f"{filename}.ply")
        return base_path if os.path.isfile(base_path) else None

    def _reference_landmark_positions_for_mesh_subject_transfer(
        self, directory, reference, ref_metadata, subject_metadata, reference_landmarks,
    ):
        """
        Express reference landmark positions in the same physical frame as preserved-mesh
        subject PLYs used for mesh elastic registration / transfer.

        - Preserved-mesh reference: landmarks are saved in mesh-local mm (UI display
          undo of center/scale on the reference PLY). Subjects aligned to that reference
          are stored in prism corner-origin shared mm — convert with mesh_local_to_shared_mm.
        - Voxel reference: landmarks are saved in voxel display-basis physical mm (UI axis
          mapping). Subjects aligned to a voxel reference receive the same display basis on
          save (see rigidAlignment._swap_xz_for_voxel_display_basis).
        """
        positions = np.asarray([lm["position"] for lm in reference_landmarks], dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("Reference landmarks must be Nx3 positions.")

        if is_preserved_mesh_metadata(ref_metadata):
            ref_mesh_path = self._source_mesh_path(directory, reference, "latest", ref_metadata)
            ref_mesh = self._load_mesh(ref_mesh_path)
            shared = mesh_local_to_shared_mm(
                positions,
                ref_metadata.get("mesh_metadata"),
                prism_vertices=ref_mesh.vertices,
                directory=directory,
            )
            _mesh_elastic_log(
                f"Reference {reference} (preserved mesh): mapped {len(positions)} landmark(s) "
                f"from mesh-local to shared registration mm "
                f"(median |Δ|={float(np.median(np.linalg.norm(shared - positions, axis=1))):.2f} mm)"
            )
            return shared

        # Voxel reference — landmarks already in display-basis mm from the UI.
        alignment_to = subject_metadata.get("alignment_to") or subject_metadata.get("elastic_to")
        if alignment_to and alignment_to != reference:
            _mesh_elastic_log(
                f"WARNING: subject alignment/elastic target is {alignment_to!r} but transfer "
                f"reference is {reference!r} — landmark frame may be inconsistent"
            )
        return positions

    def _nearest_vertex_indices_on_elastic_mesh(self, elastic_mesh, query_positions):
        """
        For each query point, return the index of the closest vertex on the subject's
        *elastic* edit (reference mesh topology may differ entirely).
        """
        query_positions = np.asarray(query_positions, dtype=np.float64)
        tree = cKDTree(np.asarray(elastic_mesh.vertices, dtype=np.float64))
        distances, nearest_vertex = tree.query(query_positions)
        return np.asarray(nearest_vertex, dtype=np.int64), np.asarray(distances, dtype=np.float64)

    def transfer_reference_landmarks_to_mesh_subject(
        self, directory, filename, reference, reference_landmarks, overwrite_existing=True,
    ):
        """
        Nearest-vertex landmark transfer for preserved-mesh subjects:

        1. Map each reference landmark into the subject registration frame (shared mm for
           preserved-mesh references, voxel display mm for voxel references).
        2. Find the closest vertex on the subject's *elastic* edit to each mapped position.
           The reference mesh may have completely different vertices — only the deformed
           subject elastic surface is queried.
        3. Copy the coordinate of that same vertex index from the pre-elastic edit (identical
           topology to the registration input).

        Returns the edit stem the landmarks were written to, or None if the subject has no
        elastic mesh edit yet.
        """
        subject_dir = self._subject_dir(directory, filename)
        elastic_edit = self._latest_elastic_mesh_edit(subject_dir, filename)
        if not elastic_edit:
            return None

        _, subject_metadata = self._load_metadata(directory, filename)
        _, ref_metadata = self._load_metadata(directory, reference)
        pre_elastic_path = self._resolve_pre_elastic_mesh_path(
            subject_dir, filename, elastic_edit, metadata=subject_metadata,
        )
        if not pre_elastic_path or not os.path.isfile(pre_elastic_path):
            return None

        edit_stem = self._mesh_stem(pre_elastic_path)
        if not overwrite_existing:
            landmarks_check_path = os.path.join(subject_dir, f"{edit_stem}_landmarks.json")
            if os.path.exists(landmarks_check_path):
                return edit_stem

        elastic_mesh = self._load_mesh(elastic_edit)
        pre_elastic_mesh = self._load_mesh(pre_elastic_path)
        _assert_elastic_preserves_source_topology(pre_elastic_mesh, elastic_mesh, filename)

        query_positions = self._reference_landmark_positions_for_mesh_subject_transfer(
            directory, reference, ref_metadata, subject_metadata, reference_landmarks,
        )
        if not is_preserved_mesh_metadata(ref_metadata):
            raw_positions = np.asarray([lm["position"] for lm in reference_landmarks], dtype=np.float64)
            swapped_positions = _voxel_display_basis_xyz(raw_positions)
            _, dist_raw = self._nearest_vertex_indices_on_elastic_mesh(elastic_mesh, raw_positions)
            _, dist_swap = self._nearest_vertex_indices_on_elastic_mesh(elastic_mesh, swapped_positions)
            mean_raw = float(np.mean(dist_raw))
            mean_swap = float(np.mean(dist_swap))
            if mean_swap + 1.0 < mean_raw:
                query_positions = swapped_positions
                _mesh_elastic_log(
                    f"{filename}: voxel reference landmarks fit better after axis 0↔2 swap "
                    f"(mean query {mean_raw:.2f} → {mean_swap:.2f} mm) — using display basis"
                )
            elif mean_raw + 1.0 < mean_swap:
                _mesh_elastic_log(
                    f"{filename}: voxel reference landmarks already in display basis "
                    f"(mean query {mean_raw:.2f} mm vs {mean_swap:.2f} mm swapped)"
                )

        nearest_vertex, transfer_distances = self._nearest_vertex_indices_on_elastic_mesh(
            elastic_mesh, query_positions,
        )

        mean_transfer_dist = float(np.mean(transfer_distances))
        max_transfer_dist = float(np.max(transfer_distances))
        _mesh_elastic_log(
            f"{filename}: landmark transfer query on elastic edit — "
            f"mean={mean_transfer_dist:.3f} mm, max={max_transfer_dist:.3f} mm "
            f"(reference={reference}, elastic={os.path.basename(elastic_edit)}, "
            f"pre_elastic={os.path.basename(pre_elastic_path)})"
        )
        if mean_transfer_dist > 15.0:
            _mesh_elastic_log(
                f"WARNING: {filename}: large mean landmark transfer distance ({mean_transfer_dist:.1f} mm) "
                f"suggests a coordinate-frame mismatch between reference landmarks and the subject elastic mesh"
            )

        new_landmarks = [
            {
                "position": pre_elastic_mesh.vertices[int(idx)].tolist(),
                "landmark_type": lm.get("landmark_type", "main"),
            }
            for lm, idx in zip(reference_landmarks, nearest_vertex)
        ]

        self._write_landmark_rows_for_edit(subject_dir, edit_stem, new_landmarks)

        elastic_stem = self._mesh_stem(elastic_edit)
        if elastic_stem != edit_stem:
            elastic_landmarks = [
                {
                    "position": elastic_mesh.vertices[int(idx)].tolist(),
                    "landmark_type": lm.get("landmark_type", "main"),
                }
                for lm, idx in zip(reference_landmarks, nearest_vertex)
            ]
            self._write_landmark_rows_for_edit(subject_dir, elastic_stem, elastic_landmarks)

        distances_path = os.path.join(subject_dir, f"{edit_stem}_landmark_distances.json")
        with open(distances_path, "w") as jf:
            json.dump(
                {
                    "distances": transfer_distances.tolist(),
                    "snap_distance": None,
                    "transfer_query_mean_mm": mean_transfer_dist,
                    "transfer_query_max_mm": max_transfer_dist,
                },
                jf,
                indent=4,
            )

        return edit_stem

    def transfer_mesh_subject_landmarks_to_reference(
        self, directory, filename, reference,
    ):
        """
        Inverse of :meth:`transfer_reference_landmarks_to_mesh_subject`.

        1. Load landmarks from the subject's *pre-elastic* edit stem.
        2. Nearest vertex on the pre-elastic mesh → copy that index from the elastic mesh
           (registration / reference-aligned frame).
        3. Map registration-frame positions into the reference landmark storage frame:
           - preserved-mesh reference: shared mm → mesh-local mm
           - voxel reference: elastic verts already in display-basis mm

        Returns ``(landmarks, transfer_distances)`` without writing files. Returns
        ``(None, None)`` when the subject has no elastic mesh edit or missing
        pre-elastic mesh. Raises ``ValueError`` when pre-elastic landmarks are
        missing or landmark positions are malformed.
        """
        subject_dir = self._subject_dir(directory, filename)
        elastic_edit = self._latest_elastic_mesh_edit(subject_dir, filename)
        if not elastic_edit:
            return None, None

        _, subject_metadata = self._load_metadata(directory, filename)
        _, ref_metadata = self._load_metadata(directory, reference)
        pre_elastic_path = self._resolve_pre_elastic_mesh_path(
            subject_dir, filename, elastic_edit, metadata=subject_metadata,
        )
        if not pre_elastic_path or not os.path.isfile(pre_elastic_path):
            return None, None

        edit_stem = self._mesh_stem(pre_elastic_path)
        _, subject_landmarks = self._load_landmark_rows_for_stem(subject_dir, edit_stem)
        if not subject_landmarks:
            raise ValueError(
                f"No landmarks on pre-elastic edit for {filename} "
                f"(expected {edit_stem}_landmarks.json). "
                f"Place or transfer landmarks onto the pre-elastic edit before "
                f"subject→reference transfer."
            )

        elastic_mesh = self._load_mesh(elastic_edit)
        pre_elastic_mesh = self._load_mesh(pre_elastic_path)
        _assert_elastic_preserves_source_topology(pre_elastic_mesh, elastic_mesh, filename)

        query_positions = np.asarray(
            [lm["position"] for lm in subject_landmarks], dtype=np.float64,
        )
        if query_positions.ndim != 2 or query_positions.shape[1] != 3:
            raise ValueError(f"{filename}: subject landmarks must be Nx3 positions.")

        tree = cKDTree(np.asarray(pre_elastic_mesh.vertices, dtype=np.float64))
        transfer_distances, nearest_vertex = tree.query(query_positions)
        transfer_distances = np.asarray(transfer_distances, dtype=np.float64)
        nearest_vertex = np.asarray(nearest_vertex, dtype=np.int64)

        registration_positions = np.asarray(
            elastic_mesh.vertices[nearest_vertex], dtype=np.float64,
        )

        mean_transfer_dist = float(np.mean(transfer_distances))
        max_transfer_dist = float(np.max(transfer_distances))
        _mesh_elastic_log(
            f"{filename}: subject→reference landmark transfer — "
            f"mean pre-elastic NN={mean_transfer_dist:.3f} mm, "
            f"max={max_transfer_dist:.3f} mm "
            f"(reference={reference}, elastic={os.path.basename(elastic_edit)}, "
            f"pre_elastic={os.path.basename(pre_elastic_path)})"
        )
        if mean_transfer_dist > 15.0:
            _mesh_elastic_log(
                f"WARNING: {filename}: large mean pre-elastic NN distance "
                f"({mean_transfer_dist:.1f} mm) suggests landmarks may not lie on "
                f"the pre-elastic mesh surface"
            )

        # Map registration-frame positions into reference landmark storage frame.
        if is_preserved_mesh_metadata(ref_metadata):
            ref_mesh_path = self._source_mesh_path(directory, reference, "latest", ref_metadata)
            ref_mesh = self._load_mesh(ref_mesh_path)
            corner = mesh_reference_prism_corner_mm(
                ref_mesh.vertices,
                ref_metadata.get("mesh_metadata"),
                directory=directory,
            )
            # Inverse of mesh_local_to_shared_mm: shared = local - corner → local = shared + corner
            ref_positions = registration_positions + corner
            _mesh_elastic_log(
                f"Reference {reference} (preserved mesh): mapped {len(ref_positions)} "
                f"landmark(s) from shared registration mm to mesh-local "
                f"(median |Δ|={float(np.median(np.linalg.norm(ref_positions - registration_positions, axis=1))):.2f} mm)"
            )
        else:
            # Voxel reference: elastic mesh verts are already in display-basis mm.
            # Forward path optionally swaps 0↔2 when *querying*; reverse starts from
            # elastic verts so no swap is needed for storage on the voxel reference.
            alignment_to = subject_metadata.get("alignment_to") or subject_metadata.get("elastic_to")
            if alignment_to and alignment_to != reference:
                _mesh_elastic_log(
                    f"WARNING: subject alignment/elastic target is {alignment_to!r} but "
                    f"transfer reference is {reference!r} — landmark frame may be inconsistent"
                )
            ref_positions = registration_positions

        new_landmarks = [
            {
                "position": ref_positions[i].tolist(),
                "landmark_type": lm.get("landmark_type", "main"),
            }
            for i, lm in enumerate(subject_landmarks)
        ]
        return new_landmarks, transfer_distances


class MeshElasticRegistrationView(MeshElasticRegistrationMixin, APIView):
    """
    Batch Non-Rigid ICP elastic registration for preserved PLY subjects, mirroring
    ElasticRegistrationView's voxel/ANTs SyN pathway (faulty exclusion, flag filter,
    WebSocket progress) but operating purely on mesh vertices/faces.
    """

    def _send_progress(self, total_scans, scan_name, custom_message, current):
        if total_scans <= 0:
            return
        _mesh_elastic_log(f"progress {current}/{total_scans}: {scan_name} — {custom_message}")
        channel_layer = get_channel_layer()
        if channel_layer is None:
            _mesh_elastic_log("WARNING: WebSocket channel_layer is None — UI progress updates will not be sent")
            return
        async_to_sync(channel_layer.group_send)(
            'progress_group',
            {
                'type': 'send_progress',
                'progress': current / total_scans,
                'scan_name': scan_name,
                'custom_message': custom_message,
                'total': total_scans,
                'current': current,
            }
        )

    def post(self, request):
        batch_t0 = time.perf_counter()
        directory = request.data.get("directory")
        reference = request.data.get("reference")
        if not directory or not reference:
            return Response({"error": "directory and reference are required."}, status=status.HTTP_400_BAD_REQUEST)

        flag_filter = normalize_flag_filter_value(request.data.get("flagFilter", "off"), only_current_scan=False)
        try:
            settings = self._resolve_mesh_elastic_settings(request.data)
        except ValueError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        _mesh_elastic_log(f"=== mesh elastic registration (NR-ICP) started ===")
        _mesh_elastic_log(f"directory={directory}")
        _mesh_elastic_log(f"reference={reference}")
        _mesh_elastic_log(f"mode={settings['mode']}, flag_filter={flag_filter}")
        _mesh_elastic_log(f"settings={settings}")
        if settings.get("auto_extract_subject_outer_shell") and not settings.get("outer_shell_only"):
            return Response(
                {"error": "auto_extract_subject_outer_shell requires outer_shell_only mode."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        extracted_dir = os.path.join(directory, "extracted")
        try:
            all_folders = [
                name for name in os.listdir(extracted_dir)
                if name != "project_settings.json" and os.path.isdir(os.path.join(extracted_dir, name))
            ]
        except FileNotFoundError:
            return Response({"error": f"No extracted directory found in {directory}."}, status=status.HTTP_400_BAD_REQUEST)

        _, preserved_mesh_scans = partition_voxel_and_preserved_mesh_scans(directory, all_folders)
        _mesh_elastic_log(f"found {len(preserved_mesh_scans)} preserved-mesh scan(s) in project: {preserved_mesh_scans}")

        faulty_names = set()
        for name in preserved_mesh_scans:
            if name == reference:
                continue
            try:
                _, metadata = self._load_metadata(directory, name)
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            if metadata.get("faulty", False):
                faulty_names.add(name)

        subjects = [name for name in preserved_mesh_scans if name != reference and name not in faulty_names]
        already_elastic = [
            name for name in subjects
            if self._latest_elastic_mesh_edit(self._subject_dir(directory, name), name)
        ]
        subjects = [name for name in subjects if name not in already_elastic]

        flagged_set = load_flagged_subject_names(directory)
        subjects = apply_flag_filter(subjects, flagged_set, flag_filter)
        subjects = filter_out_linked_children(directory, subjects)
        _mesh_elastic_log(f"eligible subjects after exclusions: {subjects}")
        if faulty_names:
            _mesh_elastic_log(f"skipped faulty: {sorted(faulty_names)}")
        if already_elastic:
            _mesh_elastic_log(f"skipped already elastically registered: {already_elastic}")

        total_scans = len(subjects)
        if total_scans == 0:
            _mesh_elastic_log("no eligible subjects — exiting")
            return Response(
                {"message": "No eligible preserved-mesh subjects found for mesh elastic registration."},
                status=status.HTTP_200_OK,
            )

        self._send_progress(total_scans, reference, "Loading reference surface for NR-ICP...", 0)
        try:
            ref_t0 = time.perf_counter()
            _, ref_metadata = self._load_metadata(directory, reference)
            reference_mesh = self._load_reference_surface_for_mesh_elastic(
                directory,
                reference,
                settings["reference_smoothing_iterations"],
                use_outer_shell=bool(
                    settings.get("use_outer_shell") or settings.get("outer_shell_only")
                ),
            )
            ref_metadata = self._persist_mesh_elastic_reference_surface(
                directory, reference, reference_mesh, ref_metadata,
            )
            _mesh_elastic_log(
                f"Reference surface ready in {time.perf_counter() - ref_t0:.1f}s — "
                f"{_mesh_elastic_mesh_summary(reference_mesh)}"
            )
        except Exception as exc:
            _mesh_elastic_log(f"ERROR loading reference surface: {exc}")
            return Response(
                {"error": f"Failed to load reference surface for mesh elastic registration: {exc}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        registered_count = 0
        errors = []
        for i, filename in enumerate(subjects):
            subject_t0 = time.perf_counter()
            _mesh_elastic_log(f"--- subject {i + 1}/{total_scans}: {filename} ---")
            self._send_progress(total_scans, filename, f"Loading mesh for {filename}...", i)
            try:
                subject_dir = self._subject_dir(directory, filename)
                json_path, metadata = self._load_metadata(directory, filename)
                source_path = self._source_mesh_path(directory, filename, "latest", metadata)
                _mesh_elastic_log(f"{filename}: loading subject mesh from {source_path}")
                subject_mesh = self._load_mesh(source_path)
                _mesh_elastic_log(_mesh_elastic_mesh_summary(subject_mesh, f"{filename} subject"))

                registration_subject = subject_mesh
                registration_source_path = source_path
                shell_edit_filename = None
                # --- Complex mesh volume: optional persisted shell edit ---
                if settings.get("outer_shell_only") and settings.get("auto_extract_subject_outer_shell"):
                    self._send_progress(
                        total_scans, filename,
                        f"Extracting outer shell edit for {filename}...", i,
                    )
                    registration_subject, shell_edit_filename, json_path, metadata = (
                        self._create_subject_outer_shell_edit(
                            directory, filename, subject_dir, json_path, metadata,
                            subject_label=filename,
                        )
                    )
                    registration_source_path = os.path.join(subject_dir, shell_edit_filename)
                elif settings.get("outer_shell_only"):
                    _mesh_elastic_log(
                        f"{filename}: outer-shell-only — using latest subject mesh as the "
                        f"registration surface (must already be an outer shell)"
                    )

                init_ref_gp, init_subj_gp, gp_origin = self._gather_mesh_elastic_guidepoints_for_init(
                    directory, filename, reference, metadata,
                )
                if init_ref_gp is not None:
                    _mesh_elastic_log(
                        f"{filename}: found {len(init_ref_gp)} manual guidepoints "
                        f"({gp_origin}) for optional Procrustes init"
                    )
                else:
                    _mesh_elastic_log(
                        f"{filename}: no project guidepoints — proceeding without Procrustes init"
                    )

                mask_details = None
                if settings.get("mask_subject_with_template"):
                    self._send_progress(
                        total_scans, filename,
                        f"Masking {filename} with warped reference template...", i,
                    )
                    pre_mask_subject = registration_subject
                    pre_mask_source_path = registration_source_path
                    registration_subject, mask_details = self._mask_subject_with_warped_template(
                        registration_subject,
                        reference_mesh,
                        settings,
                        subject_label=filename,
                        init_reference_guidepoints=init_ref_gp,
                        init_subject_guidepoints=init_subj_gp,
                    )
                    mask_edit_number = self._next_edit_number(subject_dir, filename)
                    mask_edit_filename = f"{filename}_edit_{mask_edit_number}_masked.ply"
                    mask_output_path = os.path.join(subject_dir, mask_edit_filename)
                    _mesh_elastic_log(f"{filename}: saving template-masked edit to {mask_output_path}")
                    self._save_mesh_edit(registration_subject, mask_output_path)
                    registration_source_path = mask_output_path

                    source_stem = self._mesh_stem(pre_mask_source_path)
                    mask_stem = self._mesh_stem(mask_edit_filename)
                    _, source_landmarks = self._load_landmark_rows_for_stem(subject_dir, source_stem)
                    if source_landmarks is not None:
                        # Keep landmarks whose nearest pre-mask vertex survived the crop.
                        pre_mask_vertices = np.asarray(pre_mask_subject.vertices, dtype=np.float64)
                        orig_tree = cKDTree(pre_mask_vertices)
                        crop_tree = cKDTree(np.asarray(registration_subject.vertices, dtype=np.float64))
                        kept_landmarks = []
                        for landmark in source_landmarks:
                            position = np.asarray(landmark.get("position"), dtype=np.float64)
                            if position.shape != (3,):
                                continue
                            _, orig_idx = orig_tree.query(position)
                            d_crop, _ = crop_tree.query(pre_mask_vertices[int(orig_idx)])
                            if d_crop <= 1e-6:
                                kept_landmarks.append({
                                    "position": position.tolist(),
                                    "landmark_type": landmark.get("landmark_type", "main"),
                                })
                        self._write_landmark_rows_for_edit(subject_dir, mask_stem, kept_landmarks)
                        mask_details["landmark_count"] = len(kept_landmarks)
                    else:
                        mask_details["landmark_count"] = 0

                    self._append_edit_metadata(json_path, metadata, mask_edit_filename, "masked", {
                        "reference": reference,
                        "mask_source": "warped_reference_template",
                        "mesh_summary": self._mesh_summary(registration_subject),
                        **mask_details,
                    })
                    # Reload metadata path state after append (edits list updated on disk).
                    json_path, metadata = self._load_metadata(directory, filename)

                self._send_progress(total_scans, filename, f"Running NR-ICP on {filename}...", i)
                # Final elastic always deforms the (optionally masked) subject onto the reference.
                elastic_settings = {**settings, "homology_mode": False}
                elastic_mesh, subject_decimated_for_nricp = self._run_mesh_nricp(
                    registration_subject, reference_mesh, elastic_settings,
                    subject_label=filename,
                    init_reference_guidepoints=init_ref_gp,
                    init_subject_guidepoints=init_subj_gp,
                )
                _assert_elastic_preserves_source_topology(
                    registration_subject, elastic_mesh, filename,
                )

                edit_number = self._next_edit_number(subject_dir, filename)
                edit_filename = f"{filename}_edit_{edit_number}_elastic.ply"
                output_path = os.path.join(subject_dir, edit_filename)
                self._send_progress(total_scans, filename, f"Saving elastic mesh edit for {filename}...", i + 1)
                _mesh_elastic_log(f"{filename}: saving elastic edit to {output_path}")
                save_t0 = time.perf_counter()
                self._save_mesh_edit(elastic_mesh, output_path)
                _mesh_elastic_log(f"{filename}: saved in {time.perf_counter() - save_t0:.1f}s")

                metric_t0 = time.perf_counter()
                self._send_progress(total_scans, filename, f"Computing surface metric for {filename}...", i + 1)
                surface_metrics = self._compute_mesh_elastic_surface_metrics(
                    elastic_mesh, reference_mesh, ref_metadata, filename, directory,
                )
                _mesh_elastic_log(
                    f"{filename}: surface metrics finished in {time.perf_counter() - metric_t0:.1f}s"
                )

                metadata["elastic_to"] = reference
                metadata["elastic_surface_distance"] = surface_metrics["elastic_surface_distance"]
                if surface_metrics.get("elastic_surface_distance_1vox") is not None:
                    metadata["elastic_surface_distance_1vox"] = surface_metrics[
                        "elastic_surface_distance_1vox"
                    ]
                if surface_metrics.get("surface_above_threshold_fraction") is not None:
                    metadata["surface_above_threshold_fraction"] = surface_metrics[
                        "surface_above_threshold_fraction"
                    ]
                if surface_metrics.get("surface_above_1vox_fraction") is not None:
                    metadata["surface_above_1vox_fraction"] = surface_metrics[
                        "surface_above_1vox_fraction"
                    ]
                metadata["elastic_error"] = surface_metrics["elastic_error"]
                self._append_edit_metadata(json_path, metadata, edit_filename, "elastic", {
                    "reference": reference,
                    "registration_source_ply": os.path.basename(registration_source_path),
                    "mesh_elastic_reference_ply": ref_metadata.get("mesh_elastic_reference_ply"),
                    "mesh_elastic_settings": {
                        "mode": settings["mode"],
                        "steps": settings["steps"],
                        "gamma": settings["gamma"],
                        "eps": settings["eps"],
                        "distance_threshold": settings["distance_threshold"],
                        "use_vertex_normals": settings["use_vertex_normals"],
                        "rigid_prealignment": settings["rigid_prealignment"],
                        "use_outer_shell": bool(settings.get("use_outer_shell")),
                        "outer_shell_only": bool(settings.get("outer_shell_only")),
                        "auto_extract_subject_outer_shell": bool(
                            settings.get("auto_extract_subject_outer_shell")
                        ),
                        "subject_shell_edit": shell_edit_filename,
                        "tps_alpha_spacing_fraction": settings.get("tps_alpha_spacing_fraction"),
                        "reference_smoothing_iterations": settings.get("reference_smoothing_iterations"),
                        "mask_subject_with_template": bool(settings.get("mask_subject_with_template")),
                        "mask_pad_centroid_fraction": settings.get("mask_pad_centroid_fraction"),
                        "guidepoint_procrustes_init": bool(init_ref_gp is not None),
                        "guidepoint_init_origin": gp_origin,
                        "nricp_vertex_cap": MESH_ELASTIC_NRICP_VERTEX_CAP,
                        "tps_alpha_spacing_fraction": MESH_ELASTIC_TPS_ALPHA_SPACING_FRACTION,
                        "subject_decimated_for_nricp": subject_decimated_for_nricp,
                        "displacement_propagation": (
                            "none"
                            if _meshes_share_topology(registration_subject, elastic_mesh)
                            and not subject_decimated_for_nricp
                            else (
                                "knn" if settings.get("outer_shell_only")
                                else ("tps" if subject_decimated_for_nricp else "tps")
                            )
                        ),
                    },
                    "mesh_summary": self._mesh_summary(elastic_mesh),
                    **surface_metrics,
                })
                registered_count += 1
                _mesh_elastic_log(
                    f"{filename}: completed successfully in {time.perf_counter() - subject_t0:.1f}s "
                    f"({registered_count}/{total_scans} done)"
                )
            except Exception as exc:
                _mesh_elastic_log(f"ERROR registering {filename} after {time.perf_counter() - subject_t0:.1f}s: {exc}")
                traceback.print_exc()
                errors.append(f"{filename}: {exc}")

        self._send_progress(total_scans, reference, "Mesh elastic registration complete", total_scans)
        _mesh_elastic_log(
            f"=== mesh elastic registration finished in {time.perf_counter() - batch_t0:.1f}s — "
            f"{registered_count}/{total_scans} succeeded ==="
        )

        if registered_count == 0:
            return Response(
                {"error": "Mesh elastic registration failed for all subjects.", "details": errors},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        response_payload = {
            "message": f"Mesh elastic registration complete for {registered_count}/{total_scans} subject(s).",
        }
        if errors:
            response_payload["warnings"] = errors
        return Response(response_payload, status=status.HTTP_200_OK)


class ApplyMeshCropView(MeshEditMixin, APIView):
    def post(self, request):
        directory = request.data.get("directory")
        filename = request.data.get("filename")
        edit = request.data.get("edit")
        crop = request.data.get("crop")
        padding = float(request.data.get("padding", 0) or 0)

        if not directory or not filename or not isinstance(crop, list) or len(crop) != 6:
            return Response({"error": "directory, filename, and six-value crop are required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            json_path, metadata = self._load_metadata(directory, filename)
            source_path = self._source_mesh_path(directory, filename, edit, metadata)
            mesh = self._load_mesh(source_path)

            bounds = np.asarray(crop, dtype=np.float64)
            mins = np.minimum(bounds[:3], bounds[3:]) - padding
            maxs = np.maximum(bounds[:3], bounds[3:]) + padding

            cropped_mesh = self._clip_mesh_to_bounds(mesh, mins, maxs)
            crop_reorigin_offset = mesh_reorigin_offset_after_mm_crop(mins, padding)
            cropped_mesh.vertices = reorigin_mesh_coords_after_mm_crop(
                cropped_mesh.vertices, mins, padding,
            )
            subject_dir = self._subject_dir(directory, filename)
            edit_number = self._next_edit_number(subject_dir, filename)
            edit_filename = f"{filename}_edit_{edit_number}_cropped.ply"
            output_path = os.path.join(subject_dir, edit_filename)
            self._save_mesh_edit(
                cropped_mesh,
                output_path,
                source_mesh=mesh,
                source_ply_path=source_path,
                uv_lookup_vertex_offset=crop_reorigin_offset,
            )
            edit_stem = self._mesh_stem(edit_filename)

            source_landmarks_path, source_landmarks = self._load_landmark_rows_for_stem(
                subject_dir,
                self._mesh_stem(source_path),
            )
            landmark_details = {}
            if source_landmarks is not None:
                cropped_landmarks = []
                kept_landmark_indices = []
                for landmark_index, landmark in enumerate(source_landmarks):
                    position = np.asarray(landmark.get("position"), dtype=np.float64)
                    if position.shape == (3,) and np.all((position >= mins) & (position <= maxs)):
                        kept_landmark_indices.append(landmark_index)
                        cropped_landmarks.append({
                            **landmark,
                            "position": reorigin_mesh_coords_after_mm_crop(
                                position, mins, padding,
                            ).tolist(),
                        })

                landmarks_path = self._write_landmark_rows_for_edit(
                    subject_dir,
                    edit_stem,
                    cropped_landmarks,
                )
                distances_path = self._copy_landmark_distances_for_edit(
                    subject_dir,
                    self._mesh_stem(source_path),
                    edit_stem,
                    kept_indices=kept_landmark_indices,
                )
                landmark_details = {
                    "source_landmarks": os.path.basename(source_landmarks_path),
                    "landmarks_file": os.path.basename(landmarks_path),
                    "landmark_distances_file": os.path.basename(distances_path) if distances_path else None,
                    "landmark_count": len(cropped_landmarks),
                }

            self._append_edit_metadata(json_path, metadata, edit_filename, "cropped", {
                "mesh_summary": self._mesh_summary(cropped_mesh),
            })

            return Response({
                "message": "Mesh crop applied successfully",
                "edit": edit_filename,
                "mesh_summary": self._mesh_summary(cropped_mesh),
            }, status=status.HTTP_200_OK)
        except Exception as exc:
            return Response({"error": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ApplyMeshSliceView(MeshEditMixin, APIView):
    def _truthy_request_flag(self, value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on", "save", "commit")
        return False

    def _debug_slice_request(self, request, *, save, operations):
        op_count = len(operations) if isinstance(operations, list) else None
        print(
            "[apply-mesh-slice] "
            f"method={request.method} "
            f"directory={request.data.get('directory')!r} "
            f"filename={request.data.get('filename')!r} "
            f"edit={request.data.get('edit')!r} "
            f"save={save} "
            f"save_raw={request.data.get('save')!r} "
            f"commit_raw={request.data.get('commit')!r} "
            f"op_count={op_count}",
            flush=True,
        )
        if isinstance(operations, list):
            for index, operation in enumerate(operations):
                if not isinstance(operation, dict):
                    print(f"[apply-mesh-slice]   op[{index}] invalid type={type(operation)!r}", flush=True)
                    continue
                print(
                    "[apply-mesh-slice]   "
                    f"op[{index}] action={operation.get('action')!r} "
                    f"volumes={len(operation.get('volumes') or [])} "
                    f"outline_pts={len(operation.get('outline_points') or [])} "
                    f"outline_rays={len(operation.get('outline_rays') or [])}",
                    flush=True,
                )

    def options(self, request, *args, **kwargs):
        print("[apply-mesh-slice] OPTIONS preflight received", flush=True)
        return super().options(request, *args, **kwargs)

    def post(self, request):
        directory = request.data.get("directory")
        filename = request.data.get("filename")
        edit = request.data.get("edit")
        operations = request.data.get("operations")
        save = self._truthy_request_flag(request.data.get("save", False)) or self._truthy_request_flag(request.data.get("commit", False))
        self._debug_slice_request(request, save=save, operations=operations)

        if not directory or not filename or not isinstance(operations, list) or not operations:
            print("[apply-mesh-slice] validation failed: missing directory/filename/operations", flush=True)
            return Response(
                {"error": "directory, filename, and at least one slicer operation are required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            json_path, metadata = self._load_metadata(directory, filename)
            if is_preserved_mesh_metadata(metadata):
                return self._post_mesh_slice(
                    directory=directory,
                    filename=filename,
                    edit=edit,
                    operations=operations,
                    save=save,
                    json_path=json_path,
                    metadata=metadata,
                )
            return self._post_voxel_slice(
                directory=directory,
                filename=filename,
                edit=edit,
                operations=operations,
                save=save,
                json_path=json_path,
                metadata=metadata,
            )
        except Exception as exc:
            print(f"[apply-mesh-slice] ERROR: {type(exc).__name__}: {exc}", flush=True)
            return Response({"error": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def _post_mesh_slice(self, *, directory, filename, edit, operations, save, json_path, metadata):
        source_path = self._source_mesh_path(directory, filename, edit, metadata)
        print(f"[apply-mesh-slice] source mesh: {source_path}", flush=True)
        source_mesh = self._load_mesh(source_path)
        print(
            "[apply-mesh-slice] "
            f"source vertices={len(source_mesh.vertices)} faces={len(source_mesh.faces)}",
            flush=True,
        )
        sliced_mesh = self._apply_slice_operations_to_mesh(source_mesh, operations)
        print(
            "[apply-mesh-slice] "
            f"result vertices={len(sliced_mesh.vertices)} faces={len(sliced_mesh.faces)} save={save}",
            flush=True,
        )

        if not save:
            print("[apply-mesh-slice] returning preview payload (save=False)", flush=True)
            preview_payload = self._mesh_preview_payload(sliced_mesh)
            return Response({
                "message": "Mesh slice preview generated successfully",
                **preview_payload,
            }, status=status.HTTP_200_OK)

        subject_dir = self._subject_dir(directory, filename)
        edit_number = self._next_edit_number(subject_dir, filename)
        edit_filename = f"{filename}_edit_{edit_number}_sliced.ply"
        output_path = os.path.join(subject_dir, edit_filename)
        print(f"[apply-mesh-slice] writing sliced edit: {output_path}", flush=True)
        self._save_mesh_edit(
            sliced_mesh,
            output_path,
            source_mesh=source_mesh,
            source_ply_path=source_path,
        )
        edit_stem = self._mesh_stem(edit_filename)
        source_stem = self._mesh_stem(source_path)

        source_landmarks_path, source_landmarks = self._load_landmark_rows_for_stem(subject_dir, source_stem)
        landmark_details = {}
        if source_landmarks is not None:
            sliced_landmarks, kept_landmark_indices = self._apply_slice_operations_to_landmarks(source_landmarks, operations)
            landmarks_path = self._write_landmark_rows_for_edit(subject_dir, edit_stem, sliced_landmarks)
            distances_path = self._copy_landmark_distances_for_edit(
                subject_dir,
                source_stem,
                edit_stem,
                kept_indices=kept_landmark_indices,
            )
            landmark_details = {
                "source_landmarks": os.path.basename(source_landmarks_path),
                "landmarks_file": os.path.basename(landmarks_path),
                "landmark_distances_file": os.path.basename(distances_path) if distances_path else None,
                "landmark_count": len(sliced_landmarks),
            }

        operation_summary = self._slice_operation_summary(operations)
        self._append_edit_metadata(json_path, metadata, edit_filename, "sliced", {
            "source": os.path.basename(source_path),
            "slice_mode": "view_projected_plane_clip",
            "slice_operations": operation_summary,
            **landmark_details,
            "mesh_summary": self._mesh_summary(sliced_mesh),
        })

        print(f"[apply-mesh-slice] saved successfully: {edit_filename}", flush=True)
        return Response({
            "message": "Mesh slice saved successfully",
            "saved": True,
            "edit": edit_filename,
            "mesh_summary": self._mesh_summary(sliced_mesh),
        }, status=status.HTTP_200_OK)

    def _post_voxel_slice(self, *, directory, filename, edit, operations, save, json_path, metadata):
        if not save:
            return Response(
                {"error": "Voxel slice preview is handled in the client; save is required for volume edits."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        registration_tools = RegistrationTools()
        subject_dir = self._subject_dir(directory, filename)
        source_path = self._source_nifti_path(directory, filename, edit)
        print(f"[apply-mesh-slice] source volume: {source_path}", flush=True)
        if not os.path.isfile(source_path):
            raise ValueError(f"Source NIfTI not found: {source_path}")

        nifti_img = nib.load(source_path)
        original_dtype = nifti_img.get_data_dtype()
        nifti_data = nifti_img.get_fdata().astype(original_dtype)
        voxel_size = float(metadata.get("voxel_size", 1.0))
        threshold = metadata.get("threshold")
        background_value = registration_tools.get_background_value(
            nifti_data,
            mode="border",
            threshold=threshold,
        )

        sliced_data = self._apply_slice_operations_to_volume(
            nifti_data,
            voxel_size,
            operations,
            background_value,
            threshold=threshold,
        )
        print(
            "[apply-mesh-slice] "
            f"result shape={sliced_data.shape} save={save}",
            flush=True,
        )

        edit_number = self._next_nifti_edit_number(subject_dir, filename)
        edit_filename = f"{filename}_edit_{edit_number}_sliced.nii.gz"
        output_path = os.path.join(subject_dir, edit_filename)
        print(f"[apply-mesh-slice] writing sliced volume edit: {output_path}", flush=True)
        nib.save(nib.Nifti1Image(sliced_data.astype(original_dtype), nifti_img.affine), output_path)

        # Record exactly which voxels this slice cleared. Linked siblings replay this mask
        # instead of re-running the wedge test, whose candidate set depends on each scan's
        # own threshold and border background value.
        removal_mask_file = None
        try:
            removal_mask = (sliced_data != nifti_data)
            removal_mask_file = f"{filename}_edit_{edit_number}_sliced.nii.removal_mask.gz"
            removal_mask_path = os.path.join(subject_dir, removal_mask_file)
            removal_mask_temp = os.path.join(subject_dir, f"{filename}_edit_{edit_number}_sliced_removal_mask.nii.gz")
            nib.save(
                nib.Nifti1Image(removal_mask.astype(np.uint8), nifti_img.affine),
                removal_mask_temp,
            )
            os.replace(removal_mask_temp, removal_mask_path)
        except Exception as removal_error:
            removal_mask_file = None
            print(f"[apply-mesh-slice] removal mask write failed: {removal_error}", flush=True)

        lossy_edit_filename = f"{filename}_lossy_edit_{edit_number}_sliced.nii.gz"
        lossy_output_path = os.path.join(subject_dir, lossy_edit_filename)
        registration_tools.save_as_lossy_nifti(
            sliced_data.astype(original_dtype),
            voxel_size,
            json_path,
            lossy_output_path,
        )

        source_stem = self._nifti_stem(source_path)
        mask_file = None
        try:
            mask_file = self._propagate_sliced_mask(
                subject_dir,
                source_stem,
                edit_number,
                filename,
                operations,
                voxel_size,
                nifti_img.affine,
            )
        except Exception as mask_error:
            print(f"[apply-mesh-slice] mask propagation failed: {mask_error}", flush=True)

        edit_stem = self._nifti_stem(edit_filename)
        source_landmarks_path, source_landmarks = self._load_landmark_rows_for_stem(subject_dir, source_stem)
        landmark_details = {}
        if source_landmarks is not None:
            sliced_landmarks, kept_landmark_indices = self._apply_slice_operations_to_landmarks(source_landmarks, operations)
            landmarks_path = self._write_landmark_rows_for_edit(subject_dir, edit_stem, sliced_landmarks)
            distances_path = self._copy_landmark_distances_for_edit(
                subject_dir,
                source_stem,
                edit_stem,
                kept_indices=kept_landmark_indices,
            )
            landmark_details = {
                "source_landmarks": os.path.basename(source_landmarks_path),
                "landmarks_file": os.path.basename(landmarks_path),
                "landmark_distances_file": os.path.basename(distances_path) if distances_path else None,
                "landmark_count": len(sliced_landmarks),
            }

        operation_summary = self._slice_operation_summary(operations)
        volume_summary = self._volume_summary(sliced_data)
        self._append_edit_metadata(json_path, metadata, edit_filename, "sliced", {
            "source": os.path.basename(source_path),
            "slice_mode": "view_projected_plane_clip",
            "slice_operations": operation_summary,
            "lossy_edit": lossy_edit_filename,
            "mask_file": mask_file,
            "removal_mask_file": removal_mask_file,
            **landmark_details,
            "volume_summary": volume_summary,
        })

        print(f"[apply-mesh-slice] saved successfully: {lossy_edit_filename}", flush=True)
        return Response({
            "message": "Volume slice saved successfully",
            "saved": True,
            "edit": lossy_edit_filename,
            "edit_number": edit_number,
            "removal_mask": removal_mask_file,
            "volume_summary": volume_summary,
        }, status=status.HTTP_200_OK)

    def _slice_operation_summary(self, operations):
        return [
            {
                "action": operation.get("action"),
                "volume_count": len(operation.get("volumes", [])) if isinstance(operation, dict) else 0,
            }
            for operation in operations
        ]


def _scale_uniform_about_center(positions, center, scale_ratio):
    positions = np.asarray(positions, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    return center + (positions - center) * float(scale_ratio)


def _scale_mesh_uniform_about_center(mesh, center, scale_ratio):
    scaled_vertices = _scale_uniform_about_center(mesh.vertices, center, scale_ratio)
    return trimesh.Trimesh(
        vertices=scaled_vertices,
        faces=np.asarray(mesh.faces, dtype=np.int64),
        process=False,
    )


def _scale_landmark_positions(landmarks, center, scale_ratio):
    scaled_landmarks = []
    for landmark in landmarks:
        position = np.asarray(landmark.get("position"), dtype=np.float64)
        if position.shape == (3,):
            scaled_position = _scale_uniform_about_center(position.reshape(1, 3), center, scale_ratio)[0]
            scaled_landmarks.append({
                **landmark,
                "position": scaled_position.tolist(),
            })
        else:
            scaled_landmarks.append(landmark)
    return scaled_landmarks


class ApplyMeshRotationView(MeshEditMixin, APIView):
    def _rotation_matrix_from_babylon(self, rotation_data):
        m = rotation_data.get("_m") if isinstance(rotation_data, dict) else None
        if not isinstance(m, dict):
            raise ValueError("rotation._m is required.")

        # Babylon Matrix.m is column-major. For column rotation R that the viewer
        # applies as R @ v, the layout below is R.T — the form needed for the
        # row-vector bake `(v - c) @ R.T + c` used in post().
        #
        # Do NOT conjugate by diag(1,-1,-1) here. That equals (up to sign) the
        # glTF RH→LH __root__ transform G=diag(-1,1,1) that Babylon already
        # applies around the mesh. Conjugating double-applies that basis and
        # bakes R@G instead of the previewed G@R (voxel volumes still need
        # their own axis remap in ApplyRotationView — that path stays separate).
        return np.array([
            [m["0"], m["1"], m["2"]],
            [m["4"], m["5"], m["6"]],
            [m["8"], m["9"], m["10"]],
        ], dtype=np.float64)

    def _rotation_center(self, metadata, vertices):
        quick_mesh_properties = metadata.get("quick_mesh_properties")
        if isinstance(quick_mesh_properties, dict):
            center = quick_mesh_properties.get("center")
            if isinstance(center, list) and len(center) == 3:
                return np.asarray(center, dtype=np.float64)
        return np.mean(vertices, axis=0)

    def post(self, request):
        directory = request.data.get("directory")
        filename = request.data.get("filename")
        edit = request.data.get("edit")
        rotation_data = request.data.get("rotation")

        if not directory or not filename or not rotation_data:
            return Response({"error": "directory, filename, and rotation are required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            rotation_matrix = self._rotation_matrix_from_babylon(rotation_data)
            json_path, metadata = self._load_metadata(directory, filename)
            source_path = self._source_mesh_path(directory, filename, edit, metadata)
            mesh = self._load_mesh(source_path)

            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            center = self._rotation_center(metadata, vertices)
            rotated_vertices = (vertices - center) @ rotation_matrix + center
            rotated_mesh = trimesh.Trimesh(vertices=rotated_vertices, faces=np.asarray(mesh.faces, dtype=np.int64), process=False)

            subject_dir = self._subject_dir(directory, filename)
            edit_number = self._next_edit_number(subject_dir, filename)
            edit_filename = f"{filename}_edit_{edit_number}_rotated.ply"
            output_path = os.path.join(subject_dir, edit_filename)
            self._save_mesh_edit(
                rotated_mesh,
                output_path,
                source_mesh=mesh,
                source_ply_path=source_path,
            )
            edit_stem = self._mesh_stem(edit_filename)

            source_stem = self._mesh_stem(source_path)
            source_landmarks_path, source_landmarks = self._load_landmark_rows_for_stem(
                subject_dir,
                source_stem,
            )
            landmark_details = {}
            if source_landmarks is not None:
                rotated_landmarks = []
                for landmark in source_landmarks:
                    position = np.asarray(landmark.get("position"), dtype=np.float64)
                    if position.shape == (3,):
                        rotated_position = (position - center) @ rotation_matrix + center
                        rotated_landmarks.append({
                            **landmark,
                            "position": rotated_position.tolist(),
                        })

                landmarks_path = self._write_landmark_rows_for_edit(
                    subject_dir,
                    edit_stem,
                    rotated_landmarks,
                )
                distances_path = self._copy_landmark_distances_for_edit(subject_dir, source_stem, edit_stem)
                landmark_details = {
                    "source_landmarks": os.path.basename(source_landmarks_path),
                    "landmarks_file": os.path.basename(landmarks_path),
                    "landmark_distances_file": os.path.basename(distances_path) if distances_path else None,
                    "landmark_count": len(rotated_landmarks),
                }

            self._append_edit_metadata(json_path, metadata, edit_filename, "rotated", {
                "source": os.path.basename(source_path),
                "rotation_matrix": rotation_matrix.tolist(),
                "rotation_center": center.tolist(),
                **landmark_details,
                "mesh_summary": self._mesh_summary(rotated_mesh),
            })

            return Response({
                "message": "Mesh rotation applied successfully",
                "edit": edit_filename,
                "mesh_summary": self._mesh_summary(rotated_mesh),
            }, status=status.HTTP_200_OK)
        except Exception as exc:
            return Response({"error": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ApplyMeshScaleView(MeshEditMixin, APIView):
    MEASUREMENT_TOLERANCE = 0.01
    MAX_SCALE_RATIO = 1e6

    def _mesh_transform_center(self, metadata, vertices):
        quick_mesh_properties = metadata.get("quick_mesh_properties")
        if isinstance(quick_mesh_properties, dict):
            center = quick_mesh_properties.get("center")
            if isinstance(center, list) and len(center) == 3:
                return np.asarray(center, dtype=np.float64)
        return np.mean(vertices, axis=0)

    def _parse_measurement_mm(self, measurement):
        if not isinstance(measurement, dict):
            return None, None
        start_mm = measurement.get("start_mm")
        end_mm = measurement.get("end_mm")
        if not (isinstance(start_mm, list) and isinstance(end_mm, list) and len(start_mm) == 3 and len(end_mm) == 3):
            return None, None
        start = np.asarray(start_mm, dtype=np.float64)
        end = np.asarray(end_mm, dtype=np.float64)
        return start, end

    def post(self, request):
        directory = request.data.get("directory")
        filename = request.data.get("filename")
        edit = request.data.get("edit")
        measured_distance_mm = request.data.get("measured_distance_mm")
        target_distance_mm = request.data.get("target_distance_mm")
        measurement = request.data.get("measurement")

        if not directory or not filename:
            return Response({"error": "directory and filename are required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            measured_distance_mm = float(measured_distance_mm)
            target_distance_mm = float(target_distance_mm)
        except (TypeError, ValueError):
            return Response({"error": "measured_distance_mm and target_distance_mm must be numbers."}, status=status.HTTP_400_BAD_REQUEST)

        if measured_distance_mm <= 0:
            return Response(
                {"error": "Measured distance must be greater than 0 mm."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if target_distance_mm <= 0:
            return Response({"error": "Target distance must be greater than 0 mm."}, status=status.HTTP_400_BAD_REQUEST)

        scale_ratio = target_distance_mm / measured_distance_mm
        if not np.isfinite(scale_ratio) or scale_ratio <= 0 or scale_ratio > self.MAX_SCALE_RATIO:
            return Response(
                {"error": f"Scale ratio must be between 0 and {self.MAX_SCALE_RATIO:g}."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        start_mm, end_mm = self._parse_measurement_mm(measurement)
        if start_mm is None or end_mm is None:
            return Response({"error": "measurement.start_mm and measurement.end_mm are required."}, status=status.HTTP_400_BAD_REQUEST)

        recomputed_distance = float(np.linalg.norm(end_mm - start_mm))
        distance_tolerance = max(1e-9, measured_distance_mm * self.MEASUREMENT_TOLERANCE)
        if abs(recomputed_distance - measured_distance_mm) > distance_tolerance:
            return Response(
                {
                    "error": (
                        "Measurement endpoints do not match measured_distance_mm "
                        f"(expected ~{measured_distance_mm:.4f} mm, got {recomputed_distance:.4f} mm)."
                    )
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            json_path, metadata = self._load_metadata(directory, filename)
            if not is_preserved_mesh_metadata(metadata):
                return Response({"error": "Scale calibration is only supported for preserved mesh scans."}, status=status.HTTP_400_BAD_REQUEST)

            source_path = self._source_mesh_path(directory, filename, edit, metadata)
            mesh = self._load_mesh(source_path)

            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            center = self._mesh_transform_center(metadata, vertices)
            scaled_mesh = _scale_mesh_uniform_about_center(mesh, center, scale_ratio)

            subject_dir = self._subject_dir(directory, filename)
            edit_number = self._next_edit_number(subject_dir, filename)
            edit_filename = f"{filename}_edit_{edit_number}_scaled.ply"
            output_path = os.path.join(subject_dir, edit_filename)
            self._save_mesh_edit(
                scaled_mesh,
                output_path,
                source_mesh=mesh,
                source_ply_path=source_path,
            )
            edit_stem = self._mesh_stem(edit_filename)

            source_stem = self._mesh_stem(source_path)
            source_landmarks_path, source_landmarks = self._load_landmark_rows_for_stem(
                subject_dir,
                source_stem,
            )
            landmark_details = {}
            if source_landmarks is not None:
                scaled_landmarks = _scale_landmark_positions(source_landmarks, center, scale_ratio)
                landmarks_path = self._write_landmark_rows_for_edit(
                    subject_dir,
                    edit_stem,
                    scaled_landmarks,
                )
                distances_path = self._copy_landmark_distances_for_edit(subject_dir, source_stem, edit_stem)
                landmark_details = {
                    "source_landmarks": os.path.basename(source_landmarks_path),
                    "landmarks_file": os.path.basename(landmarks_path),
                    "landmark_distances_file": os.path.basename(distances_path) if distances_path else None,
                    "landmark_count": len(scaled_landmarks),
                }

            existing_mesh_scale = metadata.get("mesh_scale")
            if not isinstance(existing_mesh_scale, dict):
                existing_mesh_scale = {}
            prior_user_factor = float(existing_mesh_scale.get("user_calibration_factor", 1.0) or 1.0)
            updated_mesh_scale = {
                **existing_mesh_scale,
                "assumed_units_after_scaling": existing_mesh_scale.get("assumed_units_after_scaling", "mm"),
                "user_calibration_factor": prior_user_factor * scale_ratio,
            }
            mesh_scale_calibration = {
                "measured_distance_mm": measured_distance_mm,
                "target_distance_mm": target_distance_mm,
                "scale_ratio": scale_ratio,
                "scale_center": center.tolist(),
                "measurement": {
                    "start_mm": start_mm.tolist(),
                    "end_mm": end_mm.tolist(),
                },
                "source_mesh": os.path.basename(source_path),
            }

            self._append_edit_metadata(json_path, metadata, edit_filename, "scaled", {
                "source": os.path.basename(source_path),
                "scale_ratio": scale_ratio,
                "scale_center": center.tolist(),
                **landmark_details,
                "mesh_summary": self._mesh_summary(scaled_mesh),
                "mesh_scale_calibration": mesh_scale_calibration,
                "mesh_scale": updated_mesh_scale,
            })

            return Response({
                "message": "Mesh scale calibration applied successfully",
                "edit": edit_filename,
                "scale_ratio": scale_ratio,
                "mesh_summary": self._mesh_summary(scaled_mesh),
                "mesh_scale_calibration": mesh_scale_calibration,
            }, status=status.HTTP_200_OK)
        except Exception as exc:
            return Response({"error": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ApplyMeshSnapToMeshView(MeshEditMixin, APIView):
    def _landmarks_path(self, directory, filename, edit):
        subject_dir = self._subject_dir(directory, filename)
        if edit == "latest":
            latest = self._latest_edit_file(subject_dir, filename)
            edit_stem = self._mesh_stem(latest) if latest else filename
        else:
            edit_stem = self._edit_stem(filename, edit)
        preferred = os.path.join(subject_dir, f"{edit_stem}_landmarks.json")
        if edit_stem != filename:
            return preferred, edit_stem

        fallback = os.path.join(subject_dir, f"{filename}_landmarks.json")
        return fallback, edit_stem

    def _load_landmarks(self, path):
        with open(path, "r") as jf:
            landmarks_data = json.load(jf)

        if isinstance(landmarks_data, list) and landmarks_data and isinstance(landmarks_data[0], dict):
            landmarks_list = landmarks_data
            landmarks_positions = np.asarray([lm["position"] for lm in landmarks_list], dtype=np.float64)
        else:
            landmarks_positions = np.asarray(landmarks_data, dtype=np.float64)
            landmarks_list = [
                {"position": np.asarray(pos, dtype=float).tolist(), "landmark_type": "main"}
                for pos in landmarks_positions
            ]

        if landmarks_positions.ndim != 2 or landmarks_positions.shape[1] != 3:
            raise ValueError("Landmarks must be a list of 3D positions.")

        return landmarks_list, landmarks_positions

    def _distance_analysis(self, directory, filename, edit_stem, landmarks, vertices, faces, snap_distance, use_outer_shell):
        json_path, metadata = self._load_metadata(directory, filename)
        distances = alpaca.calculate_landmark_distances(landmarks, vertices, faces)
        voxel_size = metadata.get("voxel_size")

        if isinstance(voxel_size, (int, float)) and voxel_size > 0:
            outliers = int(np.sum(distances > voxel_size))
            outliers_six = int(np.sum(distances > voxel_size * 6))
            snapped_outliers = int(np.sum(snap_distance > voxel_size))
            snapped_outliers_six = int(np.sum(snap_distance > voxel_size * 6))
        else:
            outliers = None
            outliers_six = None
            snapped_outliers = None
            snapped_outliers_six = None

        distances_path = os.path.join(self._subject_dir(directory, filename), f"{edit_stem}_landmark_distances.json")
        with open(distances_path, "w") as jf:
            json.dump({
                "distances": distances.tolist(),
                "snap_distance": snap_distance.tolist(),
            }, jf, indent=4)

        metadata["landmarks"] = {
            "num_landmarks": int(len(landmarks)),
            "mean_distance": float(np.mean(distances)),
            "std_distance": float(np.std(distances)),
            "outliers": outliers,
            "outliers_six": outliers_six,
            "snapped_outliers": snapped_outliers,
            "snapped_outliers_six": snapped_outliers_six,
            "snapped_mean_distance": float(np.mean(snap_distance)),
            "snapped_std_distance": float(np.std(snap_distance)),
            "edit_name": edit_stem,
            "snap_source": "preserved_mesh_ply_outer_shell" if use_outer_shell else "preserved_mesh_ply",
        }
        with open(json_path, "w") as jf:
            json.dump(metadata, jf, indent=4)

    def post(self, request):
        directory = request.data.get("directory")
        filename = request.data.get("filename")
        edit = request.data.get("edit")
        use_outer_shell = _parse_bool_request_flag(request.data.get("use_outer_shell"), default=True)

        if not directory or not filename:
            return Response({"error": "directory and filename are required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            _, metadata = self._load_metadata(directory, filename)
            source_path = self._source_mesh_path(directory, filename, edit, metadata)
            mesh = self._load_mesh(source_path)
            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            faces = np.asarray(mesh.faces, dtype=np.int64)
            if use_outer_shell:
                vertices, faces = alpaca.get_outer_mesh(vertices, faces)
                print(
                    "[apply-mesh-snap-to-mesh] "
                    f"using outer shell from loaded PLY vertices={len(vertices)} faces={len(faces)}",
                    flush=True,
                )
            else:
                print(
                    "[apply-mesh-snap-to-mesh] "
                    f"using full loaded PLY vertices={len(vertices)} faces={len(faces)}",
                    flush=True,
                )

            landmarks_path, edit_stem = self._landmarks_path(directory, filename, edit)
            if not os.path.exists(landmarks_path):
                return Response({"error": "No landmark file found for this mesh edit."}, status=status.HTTP_404_NOT_FOUND)

            landmarks_list, landmarks_positions = self._load_landmarks(landmarks_path)
            closest_points = alpaca.calculate_landmarks_closest_points(landmarks_positions, vertices, faces)

            if landmarks_positions.shape != closest_points.shape:
                return Response(
                    {"error": "Landmarks and snapped mesh points do not have the same shape."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            snap_distance = np.linalg.norm(landmarks_positions - closest_points, axis=1)
            for i, landmark in enumerate(landmarks_list):
                landmark["position"] = closest_points[i].tolist()

            with open(landmarks_path, "w") as jf:
                json.dump(landmarks_list, jf, indent=4)

            self._distance_analysis(
                directory,
                filename,
                edit_stem,
                closest_points,
                vertices,
                faces,
                snap_distance,
                use_outer_shell,
            )

            return Response({
                "message": "Landmarks snapped to preserved mesh successfully",
                "source": os.path.basename(source_path),
                "use_outer_shell": use_outer_shell,
                "landmarks": int(len(landmarks_list)),
                "mean_snap_distance": float(np.mean(snap_distance)),
            }, status=status.HTTP_200_OK)
        except Exception as exc:
            return Response({"error": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ApplyMeshCleanupView(MeshEditMixin, APIView):
    def post(self, request):
        directory = request.data.get("directory")
        filename = request.data.get("filename")
        edit = request.data.get("edit")
        use_island_volume_threshold = bool(request.data.get("use_island_volume_threshold", False))
        min_island_volume_percent = float(request.data.get("min_island_volume_percent", 1.0) or 1.0)

        if not directory or not filename:
            return Response({"error": "directory and filename are required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            result = self.cleanup_preserved_mesh(
                directory,
                filename,
                edit=edit or "latest",
                use_island_volume_threshold=use_island_volume_threshold,
                min_island_volume_percent=min_island_volume_percent,
            )
            if result["status"] == "error":
                return Response({"error": result.get("message", "Mesh cleanup failed")}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            if result["status"] == "skipped":
                return Response({
                    "message": result.get("message", "No cleanup performed"),
                    "edit": edit,
                    "component_count": result.get("component_count"),
                    "component_summary": result.get("component_summary"),
                }, status=status.HTTP_200_OK)
            return Response({
                "message": result.get("message", "Mesh cleanup applied successfully"),
                "edit": result.get("edit"),
                "component_summary": result.get("component_summary"),
                "mesh_summary": result.get("mesh_summary"),
            }, status=status.HTTP_200_OK)
        except Exception as exc:
            return Response({"error": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ApplyMeshWeldView(MeshEditMixin, APIView):
    def post(self, request):
        directory = request.data.get("directory")
        filename = request.data.get("filename")
        edit = request.data.get("edit")
        distance_factor = float(request.data.get("distance_factor", 1.0) or 1.0)

        if not directory or not filename:
            return Response({"error": "directory and filename are required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            result = self.weld_preserved_mesh(
                directory,
                filename,
                edit=edit or "latest",
                distance_factor=distance_factor,
            )
            if result["status"] == "error":
                return Response({"error": result.get("message", "Mesh weld failed"), "status": "error"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            if result["status"] == "skipped":
                return Response({
                    "status": "skipped",
                    "message": result.get("message", "No weld performed"),
                    "edit": edit,
                    "mesh_weld_settings": result.get("mesh_weld_settings"),
                    "reason": result.get("reason"),
                }, status=status.HTTP_200_OK)
            return Response({
                "status": "success",
                "message": result.get("message", "Mesh welded successfully"),
                "edit": result.get("edit"),
                "mesh_weld_settings": result.get("mesh_weld_settings"),
                "mesh_summary": result.get("mesh_summary"),
                "warning": result.get("warning"),
            }, status=status.HTTP_200_OK)
        except Exception as exc:
            return Response({"error": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class BatchWeldMeshView(MeshEditMixin, APIView):
    def post(self, request):
        directory = request.data.get("directory")
        only_current_scan = bool(request.data.get("onlyCurrentScan", False))
        selected_scan = request.data.get("selectedScan")
        distance_factor = float(request.data.get("distance_factor", 1.0) or 1.0)
        flag_filter = normalize_flag_filter_value(request.data.get("flagFilter", "off"))

        if not directory:
            return Response({"error": "directory is required."}, status=status.HTTP_400_BAD_REQUEST)

        extracted_root = os.path.join(directory, "extracted")
        if not os.path.isdir(extracted_root):
            return Response({"error": "No extracted scans found in directory."}, status=status.HTTP_400_BAD_REQUEST)

        if only_current_scan and selected_scan:
            scan_names = [selected_scan]
        else:
            scan_names = [
                name for name in os.listdir(extracted_root)
                if os.path.isdir(os.path.join(extracted_root, name)) and name != "atlas"
            ]
            scan_names = filter_out_linked_children(scan_names, directory)
            flagged_names = load_flagged_subject_names(directory)
            scan_names = apply_flag_filter(scan_names, flagged_names, flag_filter)

        preserved_scans = []
        for scan_name in scan_names:
            json_path = os.path.join(extracted_root, scan_name, f"{scan_name}.json")
            if not os.path.isfile(json_path):
                continue
            try:
                with open(json_path, "r") as jf:
                    metadata = json.load(jf)
            except (OSError, json.JSONDecodeError):
                continue
            if is_preserved_mesh_metadata(metadata):
                preserved_scans.append(scan_name)

        print(
            f"Batch weld mesh: {len(preserved_scans)} preserved mesh subjects "
            f"(flag={flag_filter}, only_current={only_current_scan}, "
            f"selected={selected_scan!r}, directory={directory!r}, "
            f"distance_factor={distance_factor:g})",
            flush=True,
        )
        if preserved_scans:
            print(
                f"Batch weld mesh subjects: {[repr(n) for n in preserved_scans]}",
                flush=True,
            )
        if not preserved_scans:
            return Response({
                "status": "error",
                "message": "No preserved mesh subjects found for welding",
                "success_count": 0,
                "skipped_count": 0,
                "error_count": 0,
                "results": [],
            }, status=status.HTTP_400_BAD_REQUEST)

        total_scans = len(preserved_scans)
        channel_layer = get_channel_layer()
        if channel_layer is not None:
            async_to_sync(channel_layer.group_send)(
                "progress_group",
                {
                    "type": "send_progress",
                    "progress": 0,
                    "scan_name": preserved_scans[0],
                    "custom_message": "Starting batch mesh weld...",
                    "total": total_scans,
                    "current": 0,
                },
            )

        results = []
        for idx, scan_name in enumerate(preserved_scans):
            if channel_layer is not None:
                async_to_sync(channel_layer.group_send)(
                    "progress_group",
                    {
                        "type": "send_progress",
                        "progress": idx / total_scans,
                        "scan_name": scan_name,
                        "custom_message": f"Welding mesh for {scan_name}...",
                        "total": total_scans,
                        "current": idx + 1,
                    },
                )
            try:
                result = self.weld_preserved_mesh(
                    directory,
                    scan_name,
                    edit="latest",
                    distance_factor=distance_factor,
                )
                results.append({"scan_name": scan_name, "mode": "preserved_mesh", **result})
                print(
                    f"Batch weld mesh [{scan_name}]: {result.get('status')} — {result.get('message')}",
                    flush=True,
                )
            except Exception as exc:
                traceback.print_exc()
                message = self._format_permission_error(exc)
                results.append({
                    "scan_name": scan_name,
                    "status": "error",
                    "mode": "preserved_mesh",
                    "message": message,
                })
                print(f"Batch weld mesh [{scan_name}]: error — {message}", flush=True)

        if channel_layer is not None:
            async_to_sync(channel_layer.group_send)(
                "progress_group",
                {
                    "type": "send_progress",
                    "progress": 1.0,
                    "scan_name": preserved_scans[-1] if preserved_scans else "",
                    "custom_message": "Batch mesh weld complete",
                    "total": total_scans,
                    "current": total_scans,
                },
            )

        success_count = sum(1 for row in results if row.get("status") == "success")
        skipped_count = sum(1 for row in results if row.get("status") == "skipped")
        error_count = sum(1 for row in results if row.get("status") == "error")
        detail_parts = []
        for row in results:
            if row.get("status") in ("skipped", "error") and row.get("message"):
                detail_parts.append(f"{row.get('scan_name')}: {row.get('message')}")
        detail_suffix = f" — {detail_parts[0]}" if detail_parts else ""
        if len(detail_parts) > 1:
            detail_suffix += f" (+{len(detail_parts) - 1} more)"

        if error_count and not success_count:
            outcome = "error"
        elif skipped_count and not success_count and not error_count:
            outcome = "skipped"
        elif (skipped_count or error_count) and success_count:
            outcome = "partial"
        else:
            outcome = "success"

        return Response({
            "status": outcome,
            "message": (
                f"Batch weld finished: {success_count} welded, "
                f"{skipped_count} skipped, {error_count} errors"
                f"{detail_suffix}"
            ),
            "success_count": success_count,
            "skipped_count": skipped_count,
            "error_count": error_count,
            "results": results,
        }, status=status.HTTP_200_OK)


class ApplyMeshShellView(MeshEditMixin, APIView):
    def _filter_landmarks_to_shell(self, landmarks, shell_mesh):
        if landmarks is None:
            return None, None

        vertices = np.asarray(shell_mesh.vertices, dtype=np.float64)
        faces = np.asarray(shell_mesh.faces, dtype=np.int64)
        if len(vertices) == 0 or len(faces) == 0:
            return [], []

        bounds = np.asarray(shell_mesh.bounds, dtype=np.float64)
        diagonal = float(np.linalg.norm(bounds[1] - bounds[0])) if bounds.shape == (2, 3) else 1.0
        tolerance = max(diagonal * 0.005, 1e-6)

        valid_rows = []
        positions = []
        for landmark_index, landmark in enumerate(landmarks):
            try:
                position = np.asarray(landmark.get("position"), dtype=np.float64)
            except (TypeError, ValueError):
                continue
            if position.shape != (3,) or not np.all(np.isfinite(position)):
                continue
            valid_rows.append((landmark_index, landmark, position))
            positions.append(position)

        if not positions:
            return [], []

        distances = alpaca.calculate_landmark_distances(
            np.asarray(positions, dtype=np.float64),
            vertices,
            faces,
        )

        kept_landmarks = []
        kept_indices = []
        for row_index, (landmark_index, landmark, position) in enumerate(valid_rows):
            if float(distances[row_index]) <= tolerance:
                kept_indices.append(landmark_index)
                kept_landmarks.append({
                    **landmark,
                    "position": position.tolist(),
                })

        return kept_landmarks, kept_indices

    def _post_mesh_shell(self, *, directory, filename, edit, json_path, metadata, started_at):
        source_path = self._source_mesh_path(directory, filename, edit, metadata)
        mesh = self._load_mesh(source_path)
        print(
            "[apply-mesh-shell] "
            f"source={source_path} vertices={len(mesh.vertices)} faces={len(mesh.faces)}",
            flush=True,
        )

        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        shell_started = time.time()
        shell_vertices, shell_faces = alpaca.get_outer_mesh(vertices, faces)
        print(
            "[apply-mesh-shell] "
            f"outer shell extracted in {time.time() - shell_started:.2f}s "
            f"vertices={len(shell_vertices)} faces={len(shell_faces)}",
            flush=True,
        )
        if len(shell_vertices) == 0 or len(shell_faces) == 0:
            raise ValueError("Outer shell extraction produced an empty mesh.")

        build_started = time.time()
        shell_mesh = trimesh.Trimesh(
            vertices=np.asarray(shell_vertices, dtype=np.float64),
            faces=np.asarray(shell_faces, dtype=np.int64),
            process=False,
        )
        shell_mesh.remove_unreferenced_vertices()
        print(
            "[apply-mesh-shell] "
            f"shell mesh built in {time.time() - build_started:.2f}s "
            f"vertices={len(shell_mesh.vertices)} faces={len(shell_mesh.faces)}",
            flush=True,
        )

        subject_dir = self._subject_dir(directory, filename)
        edit_number = self._next_edit_number(subject_dir, filename)
        edit_filename = f"{filename}_edit_{edit_number}_shell.ply"
        output_path = os.path.join(subject_dir, edit_filename)
        export_started = time.time()
        self._save_mesh_edit(shell_mesh, output_path)
        print(
            f"[apply-mesh-shell] exported {output_path} in {time.time() - export_started:.2f}s",
            flush=True,
        )
        edit_stem = self._mesh_stem(edit_filename)
        source_stem = self._mesh_stem(source_path)

        source_landmarks_path, source_landmarks = self._load_landmark_rows_for_stem(subject_dir, source_stem)
        landmark_details = {}
        if source_landmarks is not None:
            landmarks_started = time.time()
            shell_landmarks, kept_landmark_indices = self._filter_landmarks_to_shell(source_landmarks, shell_mesh)
            print(
                "[apply-mesh-shell] "
                f"filtered landmarks in {time.time() - landmarks_started:.2f}s "
                f"kept={len(shell_landmarks)}/{len(source_landmarks)}",
                flush=True,
            )
            landmarks_path = self._write_landmark_rows_for_edit(subject_dir, edit_stem, shell_landmarks)
            distances_path = self._copy_landmark_distances_for_edit(
                subject_dir,
                source_stem,
                edit_stem,
                kept_indices=kept_landmark_indices,
            )
            landmark_details = {
                "source_landmarks": os.path.basename(source_landmarks_path),
                "landmarks_file": os.path.basename(landmarks_path),
                "landmark_distances_file": os.path.basename(distances_path) if distances_path else None,
                "landmark_count": len(shell_landmarks),
            }

        self._append_edit_metadata(json_path, metadata, edit_filename, "shell", {
            "source": os.path.basename(source_path),
            "shell_mode": "alpaca_outer_mesh",
            **landmark_details,
            "mesh_summary": self._mesh_summary(shell_mesh),
        })
        print(f"[apply-mesh-shell] completed in {time.time() - started_at:.2f}s", flush=True)

        return Response({
            "message": "Mesh outer shell edit created successfully",
            "edit": edit_filename,
            "mesh_summary": self._mesh_summary(shell_mesh),
        }, status=status.HTTP_200_OK)

    def _post_voxel_shell(self, *, directory, filename, edit, json_path, metadata, started_at, gaussian_blur):
        registration_tools = RegistrationTools()
        subject_dir = self._subject_dir(directory, filename)
        source_path = self._source_nifti_path(directory, filename, edit)
        print(f"[apply-mesh-shell] source volume: {source_path}", flush=True)
        if not os.path.isfile(source_path):
            raise ValueError(f"Source NIfTI not found: {source_path}")

        nifti_img = nib.load(source_path)
        original_dtype = nifti_img.get_data_dtype()
        nifti_data = nifti_img.get_fdata().astype(original_dtype)
        voxel_size = float(metadata.get("voxel_size", 1.0))
        threshold = self._resolve_fullres_volume_threshold(metadata, nifti_data)
        background_value = registration_tools.get_background_value(
            nifti_data,
            mode="border",
            threshold=threshold,
        )

        mc_volume = np.asarray(nifti_data, dtype=np.float64)
        if gaussian_blur is not None:
            print(f"[apply-mesh-shell] applying gaussian blur sigma={gaussian_blur}", flush=True)
            mc_volume = gaussian_filter(mc_volume, sigma=gaussian_blur)

        ratio = np.sum(mc_volume >= threshold) / max(mc_volume.size, 1)
        if not (0.00005 < ratio < 0.99995):
            raise ValueError("Unable to extract outer shell. Adjust the threshold and try again.")

        marching_started = time.time()
        vertices, faces, _, _ = measure.marching_cubes(
            mc_volume,
            level=threshold,
            spacing=(voxel_size, voxel_size, voxel_size),
        )
        print(
            "[apply-mesh-shell] "
            f"marching cubes completed in {time.time() - marching_started:.2f}s "
            f"vertices={len(vertices)} faces={len(faces)}",
            flush=True,
        )

        shell_started = time.time()
        shell_vertices, shell_faces = alpaca.get_outer_mesh(vertices, faces[:, ::-1])
        print(
            "[apply-mesh-shell] "
            f"outer shell extracted in {time.time() - shell_started:.2f}s "
            f"vertices={len(shell_vertices)} faces={len(shell_faces)}",
            flush=True,
        )
        if len(shell_vertices) == 0 or len(shell_faces) == 0:
            raise ValueError("Outer shell extraction produced an empty mesh.")

        shell_mesh = trimesh.Trimesh(
            vertices=np.asarray(shell_vertices, dtype=np.float64),
            faces=np.asarray(shell_faces, dtype=np.int64),
            process=False,
        )
        shell_mesh.remove_unreferenced_vertices()

        raster_started = time.time()
        shell_mask = _rasterize_mesh_to_volume_mask(
            shell_mesh.vertices,
            shell_mesh.faces,
            nifti_data.shape,
            voxel_size,
            fill_watertight=False,
        )
        # Expand the rasterized surface band by ~10 voxels in each direction so the
        # saved volume keeps original intensities in a thicker shell, not a 1-voxel sheet.
        shell_mask = binary_dilation(shell_mask, iterations=10)
        print(
            "[apply-mesh-shell] "
            f"shell rasterized in {time.time() - raster_started:.2f}s "
            f"voxels={int(np.sum(shell_mask))}",
            flush=True,
        )

        shell_data = np.where(shell_mask, nifti_data, background_value).astype(original_dtype)

        edit_number = self._next_nifti_edit_number(subject_dir, filename)
        edit_filename = f"{filename}_edit_{edit_number}_shell.nii.gz"
        output_path = os.path.join(subject_dir, edit_filename)
        print(f"[apply-mesh-shell] writing shell volume edit: {output_path}", flush=True)
        nib.save(nib.Nifti1Image(shell_data, nifti_img.affine), output_path)

        lossy_edit_filename = f"{filename}_lossy_edit_{edit_number}_shell.nii.gz"
        lossy_output_path = os.path.join(subject_dir, lossy_edit_filename)
        registration_tools.save_as_lossy_nifti(
            shell_data.astype(original_dtype),
            voxel_size,
            json_path,
            lossy_output_path,
        )

        source_stem = self._nifti_stem(source_path)
        mask_file = None
        try:
            mask_file = self._propagate_shell_mask(
                subject_dir,
                source_stem,
                edit_number,
                filename,
                shell_mask,
                nifti_img.affine,
                json_path,
            )
        except Exception as mask_error:
            print(f"[apply-mesh-shell] mask propagation failed: {mask_error}", flush=True)

        edit_stem = self._nifti_stem(edit_filename)
        source_landmarks_path, source_landmarks = self._load_landmark_rows_for_stem(subject_dir, source_stem)
        landmark_details = {}
        if source_landmarks is not None:
            landmarks_started = time.time()
            shell_landmarks, kept_landmark_indices = self._filter_landmarks_to_shell(source_landmarks, shell_mesh)
            print(
                "[apply-mesh-shell] "
                f"filtered landmarks in {time.time() - landmarks_started:.2f}s "
                f"kept={len(shell_landmarks)}/{len(source_landmarks)}",
                flush=True,
            )
            landmarks_path = self._write_landmark_rows_for_edit(subject_dir, edit_stem, shell_landmarks)
            distances_path = self._copy_landmark_distances_for_edit(
                subject_dir,
                source_stem,
                edit_stem,
                kept_indices=kept_landmark_indices,
            )
            landmark_details = {
                "source_landmarks": os.path.basename(source_landmarks_path),
                "landmarks_file": os.path.basename(landmarks_path),
                "landmark_distances_file": os.path.basename(distances_path) if distances_path else None,
                "landmark_count": len(shell_landmarks),
            }

        volume_summary = self._volume_summary(shell_data)
        self._append_edit_metadata(json_path, metadata, edit_filename, "shell", {
            "source": os.path.basename(source_path),
            "shell_mode": "alpaca_outer_mesh_voxel",
            "gaussian_blur": gaussian_blur,
            "lossy_edit": lossy_edit_filename,
            "mask_file": mask_file,
            **landmark_details,
            "volume_summary": volume_summary,
        })
        print(f"[apply-mesh-shell] completed in {time.time() - started_at:.2f}s", flush=True)

        return Response({
            "message": "Volume outer shell edit created successfully",
            "edit": lossy_edit_filename,
            "volume_summary": volume_summary,
        }, status=status.HTTP_200_OK)

    def post(self, request):
        directory = request.data.get("directory")
        filename = request.data.get("filename")
        edit = request.data.get("edit")

        if not directory or not filename:
            return Response({"error": "directory and filename are required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            started_at = time.time()
            json_path, metadata = self._load_metadata(directory, filename)
            gaussian_blur = self._parse_optional_gaussian_blur(request.data.get("gaussian_blur"))
            if is_preserved_mesh_metadata(metadata):
                return self._post_mesh_shell(
                    directory=directory,
                    filename=filename,
                    edit=edit,
                    json_path=json_path,
                    metadata=metadata,
                    started_at=started_at,
                )
            return self._post_voxel_shell(
                directory=directory,
                filename=filename,
                edit=edit,
                json_path=json_path,
                metadata=metadata,
                started_at=started_at,
                gaussian_blur=gaussian_blur,
            )
        except Exception as exc:
            print(f"[apply-mesh-shell] ERROR: {type(exc).__name__}: {exc}", flush=True)
            return Response({"error": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
