"""
Population-average registration heatmap: reference-vertex metric values averaged across
elastically registered subjects. Supports voxel and preserved-mesh references plus
mixed cohorts (voxel and mesh elastic edits targeting the same reference).

Per-vertex scalars follow the shared heatmap contract (higher = worse) and are
metric-tagged via elasticHeatmapMetrics (surface_distance, ncc, …).
"""

from __future__ import annotations

import base64
import glob
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Optional

import nibabel as nib
import numpy as np
import open3d as o3d
import trimesh
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from skimage import measure

from .ALPACA import ALPACA
from .atlas_paths import REFERENCE_ATLAS_SUBJECT, resolve_atlas_dir
from .coordinateFrames import is_preserved_mesh_metadata
from .elasticHeatmapMetrics import (
    FALLBACK_HEATMAP_METRIC,
    SURFACE_DISTANCE,
    get_heatmap_metric_spec,
    normalize_heatmap_metric,
    resolve_heatmap_metric,
    try_compute_intensity_heatmap,
)
from .rigidAlignment import voxel_size_for_edit_stem
from .builtinTextureTools import load_uv_sidecar
from .meshBasedTools import (
    _decimate_trimesh_to_vertex_cap,
    build_and_cache_preserved_mesh_display_geometry,
    load_mesh_elastic_reference_surface,
    mesh_elastic_reference_in_registration_frame,
    preserved_mesh_reference_in_shared_mm,
    try_load_cached_preserved_mesh_display_geometry,
)
# Crash-safe protocol (atomic writes, read-only mask loads, subject transactions): see crashSafety.py.
from .crashSafety import (
    atomic_write_json,
    atomic_np_savez,
    atomic_np_save,
    atomic_write_text,
)

alpaca = ALPACA()

DEFAULT_VERTEX_CAP = 250_000
FULL_RES_VERTEX_CAP = 500_000


def display_mesh_stale_oversize_threshold(vertex_cap: int) -> int:
    """
    Treat a persisted display NPZ as stale full-res only above this count.

    Marching-cubes decimation targets ``vertex_cap`` but often lands slightly
    above it (e.g. 250026 vs 250000). Those meshes are the canonical QuickMesh
    display geometry and must not be re-decimated for population heatmaps.
    """
    cap = int(vertex_cap)
    return max(int(cap * 1.10), cap + 50_000)


class RegistrationHeatmapError(Exception):
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass
class ElasticSubject:
    name: str
    modality: str  # "voxel" | "mesh"
    edit_path: str
    metadata: dict


@dataclass
class ReferenceContext:
    directory: str
    filename: str
    sub_path: str
    metadata: dict
    is_preserved_mesh: bool
    display_vertices: np.ndarray
    display_faces: np.ndarray
    reference_outer_vertices: np.ndarray
    reference_vertex_mapping: np.ndarray
    ref_query_vertices: np.ndarray
    display_to_query_nn: Optional[np.ndarray] = None
    cache_file_path: str = ""
    heatmap_metric: str = SURFACE_DISTANCE
    full_resolution: bool = False


def registration_heatmap_cache_path(
    directory: str,
    sub_path: str,
    filename: str,
    metric_id: str,
) -> str:
    """
    Surface distance keeps the legacy filename for backward compatibility.
    Other metrics use a metric-suffixed cache so they never collide.
    """
    metric = normalize_heatmap_metric(metric_id)
    if metric == SURFACE_DISTANCE:
        return os.path.join(directory, sub_path, f"{filename}_registration_error.json")
    return os.path.join(
        directory, sub_path, f"{filename}_registration_error_{metric}.json"
    )


def strip_nifti_basename(filename: str) -> str:
    basename = os.path.basename(filename)
    for suffix in (".nii.gz", ".nii.mask.gz", ".nii", ".ply"):
        if basename.endswith(suffix):
            return basename[: -len(suffix)]
    return basename


def vertex_cap_from_full_resolution(full_resolution: bool) -> int:
    return FULL_RES_VERTEX_CAP if full_resolution else DEFAULT_VERTEX_CAP


def resolve_sub_path(filename: str) -> str:
    return "atlas" if filename == "atlas" else f"extracted/{filename}"


def resolve_json_path(directory: str, filename: str, sub_path: str) -> str:
    if filename == "atlas":
        return os.path.join(resolve_atlas_dir(directory), f"{filename}.json")
    return os.path.join(directory, sub_path, f"{filename}.json")


def normalize_reference_token(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    if name == "atlas":
        return REFERENCE_ATLAS_SUBJECT
    return name


def elastic_to_matches_reference(elastic_to: Optional[str], reference_filename: str) -> bool:
    if not elastic_to:
        return True
    ref_norm = normalize_reference_token(reference_filename)
    elastic_norm = normalize_reference_token(elastic_to)
    return (
        elastic_to == reference_filename
        or elastic_norm == ref_norm
        or elastic_to == ref_norm
        or elastic_norm == reference_filename
    )


def get_outer_mesh_with_mapping(vertices, faces):
    """
    Extract outer mesh while tracking which original vertices are retained.
    Returns: outer_vertices, outer_faces, vertex_mapping
    where vertex_mapping[i] gives the original index of outer_vertices[i]
    """
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_triangle_normals()

    normals = np.asarray(mesh.triangle_normals)
    triangles = np.asarray(mesh.triangles)
    verts = np.asarray(mesh.vertices)

    centers = np.mean(verts[triangles], axis=1)
    epsilon = 1e-3
    ray_origins = centers + epsilon * normals
    ray_directions = normals
    rays = np.hstack((ray_origins, ray_directions)).astype(np.float32)

    scene = o3d.t.geometry.RaycastingScene()
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene.add_triangles(mesh_t)

    result = scene.cast_rays(rays)
    hit_distances = result["t_hit"].numpy()

    outer_indices = np.where(np.isinf(hit_distances))[0]
    outer_faces = faces[outer_indices]
    unique_vertex_indices = np.unique(outer_faces.flatten())
    vertex_mapping = unique_vertex_indices
    outer_vertices = vertices[unique_vertex_indices]

    inverse_mapping = {orig_idx: new_idx for new_idx, orig_idx in enumerate(unique_vertex_indices)}
    remapped_outer_faces = np.array(
        [[inverse_mapping[orig_idx] for orig_idx in face] for face in outer_faces]
    )

    print(
        f"get_outer_mesh_with_mapping: {len(vertices)} -> {len(outer_vertices)} vertices, "
        f"{len(faces)} -> {len(remapped_outer_faces)} faces"
    )
    return outer_vertices, remapped_outer_faces, vertex_mapping


def _load_trimesh_as_single_mesh(mesh_path: str) -> trimesh.Trimesh:
    mesh_obj = trimesh.load(mesh_path, process=False)
    if isinstance(mesh_obj, trimesh.Scene):
        geometries = [
            geom
            for geom in mesh_obj.geometry.values()
            if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) > 0
        ]
        if not geometries:
            raise ValueError("No mesh geometry found in preserved PLY")
        mesh_obj = trimesh.util.concatenate(geometries)
    if not isinstance(mesh_obj, trimesh.Trimesh) or len(mesh_obj.vertices) == 0 or len(mesh_obj.faces) == 0:
        raise ValueError("Preserved mesh has no usable vertices or faces")
    return mesh_obj


def _preserved_mesh_path(directory: str, sub_path: str, filename: str, metadata: dict, edit=None) -> str:
    if edit and str(edit).lower().endswith(".ply"):
        mesh_filename = os.path.basename(str(edit))
    else:
        mesh_filename = metadata.get("mesh_file") or f"{filename}.ply"
    mesh_path = os.path.join(directory, sub_path, os.path.basename(mesh_filename))
    if not os.path.isfile(mesh_path):
        raise FileNotFoundError(f"Preserved mesh file not found: {mesh_path}")
    return mesh_path


def _load_preserved_mesh_display_geometry(
    directory: str,
    sub_path: str,
    filename: str,
    metadata: dict,
    edit,
    vertex_cap: int,
):
    ply_path = _preserved_mesh_path(directory, sub_path, filename, metadata, edit)
    cached = try_load_cached_preserved_mesh_display_geometry(ply_path, vertex_cap)
    if cached is not None:
        vertices, faces, _original_count, _face_uv = cached
        return vertices, faces

    mesh_obj = _load_trimesh_as_single_mesh(ply_path)
    full_vertices = np.asarray(mesh_obj.vertices, dtype=np.float64)
    full_faces = np.asarray(mesh_obj.faces, dtype=np.uint32)
    full_face_uv = load_uv_sidecar(ply_path)
    vertices, faces, _display_face_uv = build_and_cache_preserved_mesh_display_geometry(
        ply_path, full_vertices, full_faces, vertex_cap, full_face_uv=full_face_uv,
    )
    return vertices, faces


def _rebuild_capped_voxel_display_from_volume(
    directory: str,
    sub_path: str,
    filename: str,
    metadata: dict,
    edit,
    vertex_cap: int,
):
    """
    Rebuild a display-budget mesh from a (preferably lossy) NIfTI instead of
    loading a stale multi-million-vertex ``*_vertices.npz``.
    """
    voxel_size = float(metadata.get("voxel_size") or 1.0)
    nifti_path = None
    is_lossy = False

    if edit:
        candidate = os.path.join(directory, sub_path, os.path.basename(str(edit)))
        if os.path.isfile(candidate):
            if "_lossy" in os.path.basename(candidate):
                nifti_path = candidate
                is_lossy = True
            else:
                lossy = _lossy_companion_path(candidate)
                if lossy:
                    nifti_path = lossy
                    is_lossy = True
                else:
                    nifti_path = candidate

    if nifti_path is None:
        lossy_glob = [
            p
            for p in glob.glob(
                os.path.join(directory, sub_path, f"{filename}_lossy_edit_*.nii.gz")
            )
            if "_elastic" not in p and "_inv" not in p and "_fwd" not in p
        ]
        if lossy_glob:

            def _edit_num(p):
                try:
                    rest = os.path.basename(p).split("_edit_", 1)[1]
                    match = re.match(r"(\d+)", rest)
                    return int(match.group(1)) if match else -1
                except Exception:
                    return -1

            nifti_path = max(lossy_glob, key=_edit_num)
            is_lossy = True
        else:
            lossy = os.path.join(directory, sub_path, f"{filename}_lossy.nii.gz")
            if os.path.isfile(lossy):
                nifti_path = lossy
                is_lossy = True
            else:
                nifti_path = os.path.join(directory, sub_path, f"{filename}.nii.gz")

    if not os.path.isfile(nifti_path):
        raise FileNotFoundError(f"No volume available to rebuild display mesh: {nifti_path}")

    spacing = voxel_size
    if is_lossy:
        resolution_factor = _lossy_resolution_factor(
            metadata, os.path.basename(nifti_path)
        )
        spacing *= resolution_factor

    print(
        f"Rebuilding capped display mesh from {os.path.basename(nifti_path)} "
        f"(spacing={spacing:.4f}, cap={vertex_cap})"
    )
    volume = nib.load(nifti_path).get_fdata()
    volume = np.swapaxes(volume, 0, 2)
    threshold = np.clip(metadata["threshold"], np.min(volume), np.max(volume))
    vertices, faces, _, _ = measure.marching_cubes(
        volume,
        level=threshold,
        spacing=(spacing, spacing, spacing),
    )
    faces = faces[:, ::-1]
    del volume
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    return _decimate_trimesh_to_vertex_cap(mesh, int(vertex_cap))


# CRASH-SAFE PROTOCOL (see pipeline/crashSafety.py): every file this writes goes through
# an atomic_* helper, so a crash never leaves a half-written file. Subject data written
# here is covered by the transaction that the calling view or batch loop opens. Adding a
# new derivative file? Use an atomic_* helper and keep it inside that transaction.
def _persist_voxel_display_geometry(vertices_path, faces_path, vertices, faces):
    try:
        atomic_np_savez(vertices_path, vertices=vertices)
        atomic_np_savez(faces_path, faces=faces)
        print(f"Wrote capped display mesh NPZ: {vertices_path}")
    except Exception as exc:
        print(f"Warning: could not write display mesh NPZ: {exc}")


def _load_voxel_display_geometry(
    directory: str,
    sub_path: str,
    filename: str,
    vertex_cap: int,
    metadata: dict,
    edit=None,
):
    """
    Load the persisted display mesh NPZ. If missing, stale full-res (far above
    ``vertex_cap``), or unreadable, rebuild/cap from a lossy/display volume and
    rewrite the NPZ. Meshes only slightly above ``vertex_cap`` (normal MC
    decimation overshoot) are kept as-is so population heatmaps match QuickMesh.
    """
    if filename == "atlas":
        base = resolve_atlas_dir(directory)
    else:
        base = os.path.join(directory, sub_path)
    vertices_path = os.path.join(base, f"{filename}_vertices.npz")
    faces_path = os.path.join(base, f"{filename}_faces.npz")
    cap = int(vertex_cap)

    npz_missing = not (os.path.isfile(vertices_path) and os.path.isfile(faces_path))
    n_vertices = None
    if not npz_missing:
        try:
            with np.load(vertices_path, mmap_mode="r") as cached:
                n_vertices = int(cached["vertices"].shape[0])
        except Exception as exc:
            print(f"WARNING: could not read display mesh {vertices_path}: {exc}")
            npz_missing = True

    stale_threshold = display_mesh_stale_oversize_threshold(cap)
    if npz_missing or (n_vertices is not None and n_vertices > stale_threshold):
        reason = (
            "missing after cache wipe / first-time load"
            if npz_missing
            else f"stale full-res ({n_vertices} > {stale_threshold})"
        )
        print(
            f"Display mesh NPZ {reason}; rebuilding capped mesh from "
            f"lossy/display volume (cap={cap})."
        )
        try:
            vertices, faces = _rebuild_capped_voxel_display_from_volume(
                directory, sub_path, filename, metadata, edit, cap
            )
        except Exception as exc:
            if npz_missing:
                raise FileNotFoundError(
                    f"Display mesh NPZ missing and volume rebuild failed: {exc}"
                ) from exc
            print(
                f"Volume rebuild failed ({exc}); falling back to load+decimate "
                f"of oversized NPZ (may be slow / memory-heavy)."
            )
            vertices = np.load(vertices_path)["vertices"]
            faces = np.load(faces_path)["faces"]
            mesh = trimesh.Trimesh(
                vertices=np.asarray(vertices, dtype=np.float64),
                faces=np.asarray(faces, dtype=np.int64),
                process=False,
            )
            vertices, faces = _decimate_trimesh_to_vertex_cap(mesh, cap)

        print(
            f"Capped display mesh to {len(vertices)} vertices, {len(faces)} faces"
        )
        _persist_voxel_display_geometry(vertices_path, faces_path, vertices, faces)
        return vertices, faces

    vertices = np.load(vertices_path)["vertices"]
    faces = np.load(faces_path)["faces"]
    return vertices, faces


def _build_voxel_dense_query(
    directory: str,
    filename: str,
    ref_sub_path: str,
    metadata: dict,
    reference_outer_vertices: np.ndarray,
):
    voxel_size = metadata["voxel_size"]
    ref_edit_glob = os.path.join(directory, ref_sub_path, f"{filename}_edit_*.nii.gz")
    ref_edit_paths = [
        p
        for p in glob.glob(ref_edit_glob)
        if "_lossy" not in p and "_elastic" not in p and "_inv" not in p and "_fwd" not in p
    ]
    if ref_edit_paths:

        def _ref_edit_num(p):
            try:
                rest = os.path.basename(p).split("_edit_", 1)[1]
                match = re.match(r"(\d+)", rest)
                return int(match.group(1)) if match else -1
            except Exception:
                return -1

        ref_nifti_path = max(ref_edit_paths, key=_ref_edit_num)
    else:
        ref_nifti_path = os.path.join(directory, ref_sub_path, f"{filename}.nii.gz")

    print(f"Building dense full-res reference mesh from: {ref_nifti_path}")
    ref_dense_volume = nib.load(ref_nifti_path).get_fdata()
    ref_dense_volume = np.swapaxes(ref_dense_volume, 0, 2)
    ref_dense_threshold = np.clip(
        metadata["threshold"], np.min(ref_dense_volume), np.max(ref_dense_volume)
    )
    ref_dense_vertices, ref_dense_faces, _, _ = measure.marching_cubes(
        ref_dense_volume,
        level=ref_dense_threshold,
        spacing=(voxel_size, voxel_size, voxel_size),
    )
    ref_dense_faces = ref_dense_faces[:, ::-1]
    del ref_dense_volume

    ref_dense_outer_vertices, _, _ = get_outer_mesh_with_mapping(ref_dense_vertices, ref_dense_faces)
    dense_outer_tree = cKDTree(ref_dense_outer_vertices)
    _, display_to_dense_nn = dense_outer_tree.query(reference_outer_vertices)
    display_to_dense_nn = np.asarray(display_to_dense_nn, dtype=np.int64)
    return ref_dense_outer_vertices, display_to_dense_nn


def _build_mesh_dense_query(
    directory: str,
    filename: str,
    metadata: dict,
    reference_outer_vertices: np.ndarray,
):
    try:
        dense_mesh = load_mesh_elastic_reference_surface(directory, filename)
        dense_mesh = mesh_elastic_reference_in_registration_frame(dense_mesh, metadata, directory)
        dense_verts = np.asarray(dense_mesh.vertices, dtype=np.float64)
        dense_faces = np.asarray(dense_mesh.faces, dtype=np.int64)
        dense_outer_vertices, _, _ = get_outer_mesh_with_mapping(dense_verts, dense_faces)
        dense_outer_tree = cKDTree(dense_outer_vertices)
        _, display_to_query_nn = dense_outer_tree.query(reference_outer_vertices)
        return dense_outer_vertices, np.asarray(display_to_query_nn, dtype=np.int64)
    except Exception as exc:
        print(f"Warning: could not build dense mesh reference query ({exc}); using display outer shell")
        return reference_outer_vertices, None


def resolve_reference_context(
    directory: str,
    filename: str,
    edit,
    vertex_cap: int,
    heatmap_metric: str = SURFACE_DISTANCE,
    full_resolution: bool = False,
) -> ReferenceContext:
    sub_path = resolve_sub_path(filename)
    json_path = resolve_json_path(directory, filename, sub_path)
    with open(json_path, "r") as jf:
        metadata = json.load(jf)

    is_mesh = is_preserved_mesh_metadata(metadata)
    resolved_metric = resolve_heatmap_metric(heatmap_metric, is_mesh=is_mesh)
    if resolved_metric != normalize_heatmap_metric(heatmap_metric):
        print(
            f"Population heatmap: '{heatmap_metric}' unavailable for this reference "
            f"(mesh={is_mesh}); using '{resolved_metric}'."
        )

    if is_mesh:
        vertices, faces = _load_preserved_mesh_display_geometry(
            directory, sub_path, filename, metadata, edit, vertex_cap
        )
    else:
        vertices, faces = _load_voxel_display_geometry(
            directory, sub_path, filename, vertex_cap, metadata, edit
        )

    reference_outer_vertices, _, reference_vertex_mapping = get_outer_mesh_with_mapping(
        vertices, faces
    )

    # Display-budget / lossy path: query on the same outer shell as the display
    # mesh. Skip full-res dense MC — that is what made report heatmaps crawl.
    if full_resolution:
        if is_mesh:
            display_mesh = trimesh.Trimesh(
                vertices=np.asarray(vertices, dtype=np.float64),
                faces=np.asarray(faces, dtype=np.int64),
                process=False,
            )
            reg_mesh = preserved_mesh_reference_in_shared_mm(
                display_mesh, metadata, directory
            )
            reg_vertices = np.asarray(reg_mesh.vertices, dtype=np.float64)
            reference_outer_reg = reg_vertices[reference_vertex_mapping]
            ref_query_vertices, display_to_query_nn = _build_mesh_dense_query(
                directory, filename, metadata, reference_outer_reg
            )
        else:
            ref_query_vertices, display_to_query_nn = _build_voxel_dense_query(
                directory, filename, sub_path, metadata, reference_outer_vertices
            )
    else:
        if is_mesh:
            display_mesh = trimesh.Trimesh(
                vertices=np.asarray(vertices, dtype=np.float64),
                faces=np.asarray(faces, dtype=np.int64),
                process=False,
            )
            reg_mesh = preserved_mesh_reference_in_shared_mm(
                display_mesh, metadata, directory
            )
            reg_vertices = np.asarray(reg_mesh.vertices, dtype=np.float64)
            ref_query_vertices = reg_vertices[reference_vertex_mapping]
        else:
            ref_query_vertices = np.asarray(reference_outer_vertices, dtype=np.float64)
        display_to_query_nn = None
        print(
            f"Lossy/display heatmap: querying {len(ref_query_vertices)} outer-shell "
            f"vertices (no dense full-res mesh)."
        )

    cache_file_path = registration_heatmap_cache_path(
        directory, sub_path, filename, resolved_metric
    )
    return ReferenceContext(
        directory=directory,
        filename=filename,
        sub_path=sub_path,
        metadata=metadata,
        is_preserved_mesh=is_mesh,
        display_vertices=np.asarray(vertices, dtype=np.float64),
        display_faces=np.asarray(faces, dtype=np.int64),
        reference_outer_vertices=np.asarray(reference_outer_vertices, dtype=np.float64),
        reference_vertex_mapping=np.asarray(reference_vertex_mapping, dtype=np.int64),
        ref_query_vertices=np.asarray(ref_query_vertices, dtype=np.float64),
        display_to_query_nn=display_to_query_nn,
        cache_file_path=cache_file_path,
        heatmap_metric=resolved_metric,
        full_resolution=bool(full_resolution),
    )


def _latest_voxel_elastic_edit(subject_dir: str, subject_name: str) -> Optional[str]:
    edit_number = -1
    try:
        subject_files = os.listdir(subject_dir)
    except OSError:
        return None
    for file in subject_files:
        if "edit" in file:
            try:
                current_edit_number = int(file.split("_edit_")[1].split("_")[0])
                edit_number = max(edit_number, current_edit_number)
            except (IndexError, ValueError):
                continue
    if edit_number < 0:
        return None
    pattern = os.path.join(subject_dir, f"{subject_name}_edit_{edit_number}*.nii.gz")
    edit_files = glob.glob(pattern)
    if not edit_files:
        return None
    edit_files = [f for f in edit_files if "_inv" not in f and "_fwd" not in f]
    if not edit_files:
        return None
    edit_name = strip_nifti_basename(edit_files[0])
    if "elastic" not in edit_name:
        return None
    path = os.path.join(subject_dir, f"{edit_name}.nii.gz")
    return path if os.path.isfile(path) else None


def _latest_mesh_elastic_edit(subject_dir: str, subject_name: str) -> Optional[str]:
    matches = glob.glob(os.path.join(subject_dir, f"{subject_name}_edit_*_elastic.ply"))
    if not matches:

        def _has_elastic(name: str) -> bool:
            lower = name.lower()
            return "_edit_" in lower and "elastic" in lower

        matches = [
            os.path.join(subject_dir, f)
            for f in os.listdir(subject_dir)
            if f.endswith(".ply") and _has_elastic(f)
        ]
    if not matches:
        return None

    def edit_number(path: str) -> int:
        basename = os.path.basename(path)
        try:
            return int(basename.split("_edit_")[1].split("_")[0])
        except (IndexError, ValueError):
            return -1

    return max(matches, key=edit_number)


def discover_elastic_subjects(directory: str, reference_filename: str) -> list[ElasticSubject]:
    extracted_path = os.path.join(directory, "extracted")
    if not os.path.exists(extracted_path):
        return []

    subjects = [
        d
        for d in os.listdir(extracted_path)
        if os.path.isdir(os.path.join(extracted_path, d))
    ]
    subjects = [s for s in subjects if s != reference_filename and s != "atlas"]

    elastic_subjects: list[ElasticSubject] = []
    for subject in subjects:
        subject_path = os.path.join(extracted_path, subject)
        subject_json_path = os.path.join(subject_path, f"{subject}.json")
        if not os.path.exists(subject_json_path):
            continue
        try:
            with open(subject_json_path, "r") as jf:
                subject_metadata = json.load(jf)
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if subject_metadata.get("faulty", False):
            continue
        if not elastic_to_matches_reference(
            subject_metadata.get("elastic_to"), reference_filename
        ):
            continue

        mesh_edit = _latest_mesh_elastic_edit(subject_path, subject)
        voxel_edit = _latest_voxel_elastic_edit(subject_path, subject)

        if mesh_edit and voxel_edit:
            mesh_num = edit_number_from_path(mesh_edit)
            voxel_num = edit_number_from_path(voxel_edit)
            if mesh_num >= voxel_num:
                elastic_subjects.append(
                    ElasticSubject(subject, "mesh", mesh_edit, subject_metadata)
                )
            else:
                elastic_subjects.append(
                    ElasticSubject(subject, "voxel", voxel_edit, subject_metadata)
                )
        elif mesh_edit:
            elastic_subjects.append(ElasticSubject(subject, "mesh", mesh_edit, subject_metadata))
        elif voxel_edit:
            elastic_subjects.append(ElasticSubject(subject, "voxel", voxel_edit, subject_metadata))

    return elastic_subjects


def edit_number_from_path(path: str) -> int:
    basename = os.path.basename(path)
    try:
        return int(basename.split("_edit_")[1].split("_")[0])
    except (IndexError, ValueError):
        return -1


def _scatter_outer_distances_to_full(
    ctx: ReferenceContext,
    distances_outer: np.ndarray,
) -> np.ndarray:
    distances_full = np.zeros(len(ctx.display_vertices))
    for i, orig_idx in enumerate(ctx.reference_vertex_mapping):
        distances_full[int(orig_idx)] = distances_outer[i]
    return distances_full


def _project_query_distances_to_display_outer(
    ctx: ReferenceContext,
    distances_on_query: np.ndarray,
) -> np.ndarray:
    if ctx.display_to_query_nn is not None:
        return np.asarray(distances_on_query)[ctx.display_to_query_nn]
    if len(distances_on_query) == len(ctx.reference_outer_vertices):
        return np.asarray(distances_on_query)
    raise ValueError(
        f"Cannot project query distances ({len(distances_on_query)}) to display outer shell "
        f"({len(ctx.reference_outer_vertices)})"
    )


def _lossy_companion_path(nifti_path: str) -> Optional[str]:
    """Map a full-res edit NIfTI to its lossy sibling when present."""
    if not nifti_path or "_lossy" in os.path.basename(nifti_path):
        return nifti_path if nifti_path and os.path.isfile(nifti_path) else None
    base = os.path.basename(nifti_path)
    directory = os.path.dirname(nifti_path)
    if "_edit_" in base:
        lossy_name = base.replace("_edit_", "_lossy_edit_", 1)
    else:
        stem = base
        for suffix in (".nii.gz", ".nii"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        lossy_name = f"{stem}_lossy.nii.gz"
    lossy_path = os.path.join(directory, lossy_name)
    return lossy_path if os.path.isfile(lossy_path) else None


def _lossy_params_entry(metadata: dict, nifti_basename: Optional[str] = None) -> dict:
    """Pick the matching (or best) ``lossy_compression`` entry; never return a list."""
    lossy_params = metadata.get("lossy_compression", []) if metadata else []
    if isinstance(lossy_params, dict):
        return lossy_params
    if not isinstance(lossy_params, list) or not lossy_params:
        return {}

    entries = [p for p in lossy_params if isinstance(p, dict)]
    if not entries:
        return {}
    if not nifti_basename:
        return entries[-1]

    matching = [p for p in entries if p.get("filename") == nifti_basename]
    if matching:
        return matching[0]

    # Elastic lossy warps often have no own compression entry; reuse the latest
    # non-elastic lossy edit params (same intensity scale as the warped volume).
    stem = nifti_basename
    for suffix in (".nii.gz", ".nii"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    if "_elastic" in stem:
        non_elastic = [
            p
            for p in entries
            if p.get("filename")
            and "_elastic" not in str(p.get("filename"))
            and "_fwd" not in str(p.get("filename"))
            and "_inv" not in str(p.get("filename"))
        ]
        if non_elastic:
            return non_elastic[-1]

    return entries[-1]


def _lossy_resolution_factor(metadata: dict, nifti_basename: Optional[str] = None) -> float:
    """
    Resolve lossy grid downsampling factor.

    ``lossy_compression`` is normally a list of per-file dicts (legacy: a single
    dict). Never call ``.get`` on the list itself.
    """
    entry = _lossy_params_entry(metadata, nifti_basename)
    try:
        rf = float(
            entry.get("resolution_factor")
            or (metadata or {}).get("resolution_factor")
            or 2.0
        )
    except (TypeError, ValueError):
        rf = 2.0
    return rf if np.isfinite(rf) and rf > 0 else 2.0


def _iso_threshold_for_volume(
    metadata: dict,
    volume: np.ndarray,
    nifti_basename: Optional[str] = None,
    is_lossy: bool = False,
) -> float:
    """
    Iso-value for marching cubes, matching /marchingcubes/ lossy remapping.

    Full-res ``threshold`` must be mapped through scale_factor / zero_point_shift
    before use on lossy volumes. Clamping a full-res threshold onto uint8 data
    often lands on vmax and skimage raises "No surface found at the given iso value."
    """
    try:
        raw = float(metadata.get("threshold") or 0.0)
    except (TypeError, ValueError):
        raw = 0.0

    if is_lossy:
        entry = _lossy_params_entry(metadata, nifti_basename)
        try:
            scale = float(entry.get("scale_factor", 1.0) or 1.0)
        except (TypeError, ValueError):
            scale = 1.0
        try:
            zero = float(entry.get("zero_point_shift", 0.0) or 0.0)
        except (TypeError, ValueError):
            zero = 0.0
        if abs(scale) > 1e-8:
            raw = (raw - zero) / scale
        else:
            raw = raw - zero

    vmin = float(np.min(volume))
    vmax = float(np.max(volume))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        raise ValueError(
            f"Volume has no intensity range for MC (min={vmin}, max={vmax})"
        )

    # Iso must be strictly inside (min, max) or Lewiner reports no surface.
    lo = np.nextafter(vmin, vmax)
    hi = np.nextafter(vmax, vmin)
    iso = float(np.clip(raw, lo, hi))
    if iso <= vmin or iso >= vmax:
        iso = 0.5 * (vmin + vmax)
    return iso


def compute_voxel_subject_distances(ctx: ReferenceContext, subject: ElasticSubject) -> np.ndarray:
    edit_path = subject.edit_path
    spacing = float(subject.metadata.get("voxel_size") or 1.0)
    using_lossy = False
    if not ctx.full_resolution:
        lossy = _lossy_companion_path(edit_path)
        if lossy:
            edit_path = lossy
            using_lossy = True
            # Lossy volumes are on the compressed grid.
            resolution_factor = _lossy_resolution_factor(
                subject.metadata, os.path.basename(edit_path)
            )
            spacing = spacing * resolution_factor
            print(
                f"Subject {subject.name}: lossy MC at spacing={spacing:.4f} "
                f"({os.path.basename(edit_path)})"
            )

    edit_nifti_data = nib.load(edit_path).get_fdata()
    edit_nifti_data = np.swapaxes(edit_nifti_data, 0, 2)

    subject_threshold = _iso_threshold_for_volume(
        subject.metadata,
        edit_nifti_data,
        nifti_basename=os.path.basename(edit_path),
        is_lossy=using_lossy or "_lossy" in os.path.basename(edit_path),
    )
    print(
        f"Subject {subject.name}: MC iso={subject_threshold:.4f} "
        f"(volume range {float(np.min(edit_nifti_data)):.4f}–{float(np.max(edit_nifti_data)):.4f})"
    )

    subject_vertices, subject_faces, _, _ = measure.marching_cubes(
        edit_nifti_data,
        level=subject_threshold,
        spacing=(spacing, spacing, spacing),
    )
    subject_faces = subject_faces[:, ::-1]
    subject_outer_vertices, subject_outer_faces, _ = get_outer_mesh_with_mapping(
        subject_vertices, subject_faces
    )

    distances_on_query = alpaca.calculate_landmark_distances(
        ctx.ref_query_vertices, subject_outer_vertices, subject_outer_faces
    )
    distances_outer = _project_query_distances_to_display_outer(ctx, distances_on_query)
    return _scatter_outer_distances_to_full(ctx, distances_outer)


def compute_mesh_subject_distances(ctx: ReferenceContext, subject: ElasticSubject) -> np.ndarray:
    subject_mesh = _load_trimesh_as_single_mesh(subject.edit_path)
    subject_vertices = np.asarray(subject_mesh.vertices, dtype=np.float64)
    subject_faces = np.asarray(subject_mesh.faces, dtype=np.int64)
    subject_outer_vertices, subject_outer_faces, _ = get_outer_mesh_with_mapping(
        subject_vertices, subject_faces
    )

    distances_on_query = alpaca.calculate_landmark_distances(
        ctx.ref_query_vertices, subject_outer_vertices, subject_outer_faces
    )
    distances_outer = _project_query_distances_to_display_outer(ctx, distances_on_query)
    return _scatter_outer_distances_to_full(ctx, distances_outer)


def compute_subject_distances(ctx: ReferenceContext, subject: ElasticSubject) -> np.ndarray:
    if subject.modality == "voxel":
        return compute_voxel_subject_distances(ctx, subject)
    return compute_mesh_subject_distances(ctx, subject)


def _resolve_reference_nifti_path(ctx: ReferenceContext) -> str:
    directory = ctx.directory
    filename = ctx.filename
    ref_sub_path = ctx.sub_path
    prefer_lossy = not ctx.full_resolution
    ref_edit_glob = os.path.join(directory, ref_sub_path, f"{filename}_edit_*.nii.gz")
    if prefer_lossy:
        lossy_glob = os.path.join(
            directory, ref_sub_path, f"{filename}_lossy_edit_*.nii.gz"
        )
        lossy_paths = [
            p
            for p in glob.glob(lossy_glob)
            if "_elastic" not in p and "_inv" not in p and "_fwd" not in p
        ]
        if lossy_paths:

            def _ref_edit_num(p):
                try:
                    rest = os.path.basename(p).split("_edit_", 1)[1]
                    match = re.match(r"(\d+)", rest)
                    return int(match.group(1)) if match else -1
                except Exception:
                    return -1

            return max(lossy_paths, key=_ref_edit_num)

    ref_edit_paths = [
        p
        for p in glob.glob(ref_edit_glob)
        if "_lossy" not in p and "_elastic" not in p and "_inv" not in p and "_fwd" not in p
    ]
    if ref_edit_paths:

        def _ref_edit_num(p):
            try:
                rest = os.path.basename(p).split("_edit_", 1)[1]
                match = re.match(r"(\d+)", rest)
                return int(match.group(1)) if match else -1
            except Exception:
                return -1

        return max(ref_edit_paths, key=_ref_edit_num)

    if filename == "atlas":
        atlas_dir = resolve_atlas_dir(directory)
        if prefer_lossy:
            lossy_atlas = os.path.join(atlas_dir, f"{filename}_lossy.nii.gz")
            if os.path.isfile(lossy_atlas):
                return lossy_atlas
        return os.path.join(atlas_dir, f"{filename}.nii.gz")
    if prefer_lossy:
        lossy = os.path.join(directory, ref_sub_path, f"{filename}_lossy.nii.gz")
        if os.path.isfile(lossy):
            return lossy
    return os.path.join(directory, ref_sub_path, f"{filename}.nii.gz")


def _load_volume_swapped(nifti_path: str) -> np.ndarray:
    volume = nib.load(nifti_path).get_fdata()
    return np.swapaxes(volume, 0, 2)


def compute_subject_intensity_metric(
    ctx: ReferenceContext,
    subject: ElasticSubject,
    metric_id: str,
    fixed_volume: np.ndarray,
    spacing_per_voxel: float,
) -> np.ndarray:
    """Intensity metrics (NCC, …) at reference query vertices, scattered to display mesh."""
    if subject.modality != "voxel":
        raise ValueError(
            f"{metric_id} population heatmap skips mesh subject {subject.name}"
        )
    moving_path = subject.edit_path
    if not ctx.full_resolution:
        lossy = _lossy_companion_path(subject.edit_path)
        if lossy:
            moving_path = lossy
    moving_volume = _load_volume_swapped(moving_path)
    values_on_query = try_compute_intensity_heatmap(
        metric_id,
        fixed_volume,
        moving_volume,
        ctx.ref_query_vertices,
        spacing_per_voxel,
    )
    if values_on_query is None:
        raise ValueError(f"Metric '{metric_id}' is not an intensity heatmap")
    values_outer = _project_query_distances_to_display_outer(ctx, values_on_query)
    return _scatter_outer_distances_to_full(ctx, values_outer)


def compute_subject_metric_values(
    ctx: ReferenceContext,
    subject: ElasticSubject,
    metric_id: str,
    *,
    fixed_volume: Optional[np.ndarray] = None,
    spacing_per_voxel: Optional[float] = None,
) -> np.ndarray:
    """Dispatch one subject's per-display-vertex heatmap values for the active metric."""
    spec = get_heatmap_metric_spec(metric_id)
    if spec.compute_kind == "intensity":
        if fixed_volume is None or spacing_per_voxel is None:
            raise ValueError("Intensity population heatmaps require a fixed volume")
        return compute_subject_intensity_metric(
            ctx, subject, metric_id, fixed_volume, spacing_per_voxel
        )
    return compute_subject_distances(ctx, subject)


def convert_to_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    return obj


def strip_response_payload(full_data: dict) -> dict:
    return {
        "distances": full_data["distances"],
        "subject_distances": {
            "average": full_data["subject_distances"]["average"]
        }
        if "subject_distances" in full_data
        else {},
        "coordinates": full_data["coordinates"],
        "n_subjects_processed": full_data.get("n_subjects_processed", 0),
        "n_vertices_total": full_data["n_vertices_total"],
        "n_vertices_outer_shell": full_data.get("n_vertices_outer_shell", 0),
        "has_full_subject_data": True,
        "metric": full_data.get("metric", FALLBACK_HEATMAP_METRIC),
        "dense_distances_metric": full_data.get(
            "metric", FALLBACK_HEATMAP_METRIC
        ),  # alias matching individual heatmap naming
    }


# CRASH-SAFE PROTOCOL (see pipeline/crashSafety.py): every file this writes goes through
# an atomic_* helper, so a crash never leaves a half-written file. Subject data written
# here is covered by the transaction that the calling view or batch loop opens. Adding a
# new derivative file? Use an atomic_* helper and keep it inside that transaction.
def remap_cached_data_to_new_topology(
    cached_data: dict,
    vertices: np.ndarray,
    faces: np.ndarray,
    cache_file_path: str,
) -> dict:
    print(f"Cached data has {cached_data['n_vertices_total']} vertices, but {len(vertices)} were provided.")
    print("Different resolution detected - performing topology-aware adjustment...")

    new_outer_vertices, _, new_vertex_mapping = get_outer_mesh_with_mapping(vertices, faces)

    if cached_data["coordinates"] and isinstance(cached_data["coordinates"][0], dict):
        cached_coords = np.array(
            [coord["position"] for coord in cached_data["coordinates"]], dtype=np.float64
        )
    else:
        cached_coords = np.array(cached_data["coordinates"], dtype=np.float64)

    tree = cKDTree(cached_coords)
    _, closest_indices = tree.query(new_outer_vertices.astype(np.float64))
    closest_indices = np.asarray(closest_indices, dtype=np.int32)

    new_distances_full = np.zeros(len(vertices))
    for i, new_outer_vertex_idx in enumerate(new_vertex_mapping):
        old_closest_vertex_idx = int(closest_indices[i])
        new_distances_full[new_outer_vertex_idx] = cached_data["distances"][old_closest_vertex_idx]

    cached_data["distances"] = new_distances_full.tolist()
    cached_data["coordinates"] = vertices.tolist()
    cached_data["n_vertices_total"] = len(vertices)

    if "subject_distances" in cached_data:
        for subject_name in list(cached_data["subject_distances"].keys()):
            if subject_name == "average":
                continue
            old_subject_distances = np.array(cached_data["subject_distances"][subject_name])
            new_subject_distances = np.zeros(len(vertices))
            for i, new_outer_vertex_idx in enumerate(new_vertex_mapping):
                old_closest_vertex_idx = int(closest_indices[i])
                new_subject_distances[new_outer_vertex_idx] = old_subject_distances[
                    old_closest_vertex_idx
                ]
            cached_data["subject_distances"][subject_name] = new_subject_distances.tolist()

        subject_arrays = [
            np.array(cached_data["subject_distances"][s])
            for s in cached_data["subject_distances"]
            if s != "average"
        ]
        if subject_arrays:
            avg = np.mean(subject_arrays, axis=0)
            avg = np.nan_to_num(avg, nan=0, posinf=0, neginf=0)
            cached_data["subject_distances"]["average"] = avg.tolist()

    cached_data = convert_to_json_serializable(cached_data)
    atomic_write_text(cache_file_path, json.dumps(cached_data))
    save_display_sidecar_cache(cache_file_path, cached_data)
    return cached_data


def display_sidecar_path(cache_file_path: str) -> str:
    """Compact NPZ cache for display-budget heatmaps (average only, no per-subject JSON)."""
    return cache_file_path.replace(".json", "_display.npz")


# CRASH-SAFE PROTOCOL (see pipeline/crashSafety.py): every file this writes goes through
# an atomic_* helper, so a crash never leaves a half-written file. Subject data written
# here is covered by the transaction that the calling view or batch loop opens. Adding a
# new derivative file? Use an atomic_* helper and keep it inside that transaction.
def save_display_sidecar_cache(cache_file_path: str, response_data_full: dict) -> None:
    sidecar = display_sidecar_path(cache_file_path)
    distances = np.asarray(response_data_full["distances"], dtype=np.float32)
    coordinates = np.asarray(response_data_full["coordinates"], dtype=np.float32)
    metric_id = str(response_data_full.get("metric", FALLBACK_HEATMAP_METRIC))
    atomic_np_savez(
        sidecar,
        compressed=True,
        distances=distances,
        coordinates=coordinates,
        n_vertices_total=int(response_data_full["n_vertices_total"]),
        n_subjects_processed=int(response_data_full.get("n_subjects_processed", 0)),
        metric=np.array(metric_id),
    )


def try_load_display_sidecar_cache(ctx: ReferenceContext) -> Optional[dict]:
    if ctx.full_resolution:
        return None
    sidecar = display_sidecar_path(ctx.cache_file_path)
    if not os.path.isfile(sidecar):
        return None
    try:
        with np.load(sidecar, allow_pickle=False) as data:
            n_cached = int(data["n_vertices_total"])
            metric_raw = data["metric"]
            metric_id = normalize_heatmap_metric(
                metric_raw.item() if hasattr(metric_raw, "item") else str(metric_raw)
            )
            if metric_id != ctx.heatmap_metric:
                return None
            n_vertices = len(ctx.display_vertices)
            if n_cached != n_vertices:
                return None
            if n_vertices > DEFAULT_VERTEX_CAP * 2:
                return None
            distances = np.asarray(data["distances"], dtype=np.float64)
            coordinates = np.asarray(data["coordinates"], dtype=np.float64)
            n_subjects = int(data.get("n_subjects_processed", 0))
    except Exception as exc:
        print(f"Could not load display heatmap sidecar {sidecar}: {exc}")
        return None
    payload = {
        "distances": distances.tolist(),
        "coordinates": coordinates.tolist(),
        "n_vertices_total": n_vertices,
        "n_subjects_processed": n_subjects,
        "metric": metric_id,
        "subject_distances": {"average": distances.tolist()},
    }
    print(f"Loaded display heatmap sidecar: {sidecar}")
    return strip_response_payload(payload)


def try_migrate_oversized_json_to_sidecar(
    ctx: ReferenceContext,
    cache_file_path: str,
) -> Optional[dict]:
    """
    One-time migration: legacy full JSON caches (with all per-subject arrays)
    are too large to load on every report open. Extract the averaged display
    payload once and persist a compact NPZ sidecar.
    """
    sidecar = display_sidecar_path(cache_file_path)
    if os.path.isfile(sidecar):
        return try_load_display_sidecar_cache(ctx)
    print(
        f"Migrating oversized heatmap cache to display sidecar (one-time): "
        f"{cache_file_path}"
    )
    with open(cache_file_path, "r") as cf:
        cached_data = json.load(cf)
    cached_metric = normalize_heatmap_metric(
        cached_data.get("metric") or cached_data.get("dense_distances_metric")
    )
    if "metric" not in cached_data and "dense_distances_metric" not in cached_data:
        cached_metric = SURFACE_DISTANCE
        cached_data["metric"] = SURFACE_DISTANCE
    if cached_metric != ctx.heatmap_metric:
        return None
    n_vertices = len(ctx.display_vertices)
    if cached_data.get("n_vertices_total") != n_vertices:
        return None
    if n_vertices > DEFAULT_VERTEX_CAP * 2:
        return None
    save_display_sidecar_cache(cache_file_path, cached_data)
    return strip_response_payload(cached_data)


def try_load_cached_heatmap(
    ctx: ReferenceContext,
    recalculate: bool,
) -> Optional[dict]:
    cache_file_path = ctx.cache_file_path
    sidecar_path = display_sidecar_path(cache_file_path)
    if recalculate:
        if os.path.exists(cache_file_path):
            print(f"Recalculate requested, deleting existing cache: {cache_file_path}")
            os.remove(cache_file_path)
        if os.path.exists(sidecar_path):
            os.remove(sidecar_path)

    if recalculate:
        return None

    sidecar_cached = try_load_display_sidecar_cache(ctx)
    if sidecar_cached is not None:
        return sidecar_cached

    if not os.path.exists(cache_file_path):
        return None

    # Oversized JSON caches (from stale full-res display meshes) can be multi-GB
    # and hang/OOM the process before we even check vertex counts.
    cache_bytes = os.path.getsize(cache_file_path)
    max_display_cache_bytes = 80 * 1024 * 1024
    if (not ctx.full_resolution) and cache_bytes > max_display_cache_bytes:
        migrated = try_migrate_oversized_json_to_sidecar(ctx, cache_file_path)
        if migrated is not None:
            return migrated
        print(
            f"Cache {cache_file_path} is {cache_bytes / (1024 ** 2):.1f} MB — too large "
            f"for display-budget heatmap (likely stale full-res). Ignoring cache."
        )
        return None

    print(f"Loading cached registration heatmap from: {cache_file_path}")
    with open(cache_file_path, "r") as cf:
        cached_data = json.load(cf)

    cached_metric = normalize_heatmap_metric(
        cached_data.get("metric") or cached_data.get("dense_distances_metric")
    )
    # Legacy surface caches omit metric — treat as surface_distance.
    if "metric" not in cached_data and "dense_distances_metric" not in cached_data:
        cached_metric = SURFACE_DISTANCE
        cached_data["metric"] = SURFACE_DISTANCE
    if cached_metric != ctx.heatmap_metric:
        print(
            f"Cache metric '{cached_metric}' != requested '{ctx.heatmap_metric}'; "
            f"ignoring cache."
        )
        return None

    n_vertices = len(ctx.display_vertices)
    if cached_data["n_vertices_total"] != n_vertices:
        # Do not nearest-neighbour remap onto a different display mesh (shell /
        # blur / resolution). That is what scrambled report heatmaps.
        print(
            f"Cache has {cached_data['n_vertices_total']} vertices, display mesh has "
            f"{n_vertices}; ignoring cache so the heatmap can be recomputed."
        )
        return None

    if (not ctx.full_resolution) and n_vertices > DEFAULT_VERTEX_CAP * 2:
        print(
            f"Display mesh still has {n_vertices} vertices after load; refusing "
            f"cached heatmap payload (would crash the browser)."
        )
        return None

    return strip_response_payload(cached_data)


def _send_progress(total: int, scan_name: str, message: str, current: int):
    channel_layer = get_channel_layer()
    if channel_layer is None or total <= 0:
        return
    progress = current / total if total else 0
    async_to_sync(channel_layer.group_send)(
        "progress_group",
        {
            "type": "send_progress",
            "progress": progress,
            "scan_name": scan_name,
            "custom_message": message,
            "total": total,
            "current": current,
        },
    )


# CRASH-SAFE PROTOCOL (see pipeline/crashSafety.py): every file this writes goes through
# an atomic_* helper, so a crash never leaves a half-written file. Subject data written
# here is covered by the transaction that the calling view or batch loop opens. Adding a
# new derivative file? Use an atomic_* helper and keep it inside that transaction.
def compute_registration_heatmap(
    directory: str,
    filename: str,
    edit=None,
    recalculate: bool = False,
    full_resolution: bool = False,
    heatmap_metric: str = SURFACE_DISTANCE,
) -> dict:
    vertex_cap = vertex_cap_from_full_resolution(full_resolution)
    ctx = resolve_reference_context(
        directory,
        filename,
        edit,
        vertex_cap,
        heatmap_metric=heatmap_metric,
        full_resolution=full_resolution,
    )
    metric_id = ctx.heatmap_metric
    metric_spec = get_heatmap_metric_spec(metric_id)

    print(
        f"Creating registration heatmap for {filename}, edit: {edit}, "
        f"metric: {metric_id}, recalculate: {recalculate}, vertex_cap: {vertex_cap}"
    )
    print(
        f"Loaded display mesh: {len(ctx.display_vertices)} vertices, "
        f"{len(ctx.display_faces)} faces (preserved_mesh={ctx.is_preserved_mesh})"
    )

    cached = try_load_cached_heatmap(ctx, recalculate)
    if cached is not None:
        return cached

    print(
        f"No cache found or recalculation requested, computing registration heatmap "
        f"({metric_id})..."
    )
    elastic_subjects = discover_elastic_subjects(directory, filename)
    if not elastic_subjects:
        raise RegistrationHeatmapError(
            "No valid elastic edits found in any subjects", status_code=404
        )

    fixed_volume = None
    spacing_per_voxel = None
    if metric_spec.compute_kind == "intensity":
        ref_nifti = _resolve_reference_nifti_path(ctx)
        if not os.path.isfile(ref_nifti):
            raise RegistrationHeatmapError(
                f"Reference volume not found for {metric_id} heatmap: {ref_nifti}",
                status_code=404,
            )
        fixed_volume = _load_volume_swapped(ref_nifti)
        spacing_per_voxel = float(ctx.metadata.get("voxel_size") or 1.0)
        if not ctx.full_resolution and "_lossy" in os.path.basename(ref_nifti):
            rf = _lossy_resolution_factor(
                ctx.metadata, os.path.basename(ref_nifti)
            )
            spacing_per_voxel *= rf
        print(
            f"Intensity population heatmap fixed volume: {ref_nifti} "
            f"shape={fixed_volume.shape}, spacing={spacing_per_voxel}"
        )

    total_subjects = len(elastic_subjects)
    _send_progress(
        total_subjects,
        elastic_subjects[0].name if elastic_subjects else "",
        f"Computing {metric_id} registration heatmap for {total_subjects} subjects...",
        0,
    )

    subject_distances: dict[str, list[float]] = {}
    processed_subjects = 0

    for idx, subject in enumerate(elastic_subjects):
        _send_progress(
            total_subjects,
            subject.name,
            f"Processing {subject.name} ({idx + 1}/{total_subjects}) [{metric_id}]...",
            idx + 1,
        )
        try:
            print(
                f"Processing subject {subject.name} ({subject.modality}) "
                f"edit={os.path.basename(subject.edit_path)} metric={metric_id}"
            )
            if (
                metric_spec.compute_kind == "intensity"
                and subject.modality != "voxel"
            ):
                print(
                    f"Skipping mesh subject {subject.name} for intensity metric "
                    f"'{metric_id}'"
                )
                continue
            distances_full = compute_subject_metric_values(
                ctx,
                subject,
                metric_id,
                fixed_volume=fixed_volume,
                spacing_per_voxel=spacing_per_voxel,
            )
            subject_distances[subject.name] = distances_full.tolist()
            processed_subjects += 1
            outer_vals = distances_full[distances_full > 0]
            if len(outer_vals) > 0:
                print(
                    f"Calculated {metric_id} for {subject.name}, "
                    f"mean (outer shell): {np.mean(outer_vals):.4f}"
                )
        except Exception as exc:
            print(f"Error processing subject {subject.name}: {exc}")
            import traceback

            print(traceback.format_exc())
            continue

    if processed_subjects == 0:
        raise RegistrationHeatmapError(
            f"No valid elastic subjects could be processed for metric '{metric_id}'",
            status_code=404,
        )

    print(f"Averaging {metric_id} across {processed_subjects} subjects...")
    all_distances_array = np.array([subject_distances[s] for s in subject_distances])
    averaged_distances = np.mean(all_distances_array, axis=0)
    averaged_distances = np.nan_to_num(averaged_distances, nan=0, posinf=0, neginf=0)

    non_zero_distances = averaged_distances[averaged_distances > 0]
    if len(non_zero_distances) > 0:
        print(
            f"Final averaged {metric_id} - outer shell mean: {np.mean(non_zero_distances):.4f}, "
            f"std: {np.std(non_zero_distances):.4f}"
        )
    else:
        print("WARNING: No non-zero metric values found! All values are 0 or NaN.")

    subject_distances["average"] = averaged_distances.tolist()
    response_data_full = {
        "distances": averaged_distances.tolist(),
        "subject_distances": subject_distances,
        "coordinates": ctx.display_vertices.tolist(),
        "n_subjects_processed": processed_subjects,
        "n_vertices_total": len(ctx.display_vertices),
        "n_vertices_outer_shell": int(len(non_zero_distances)),
        "metric": metric_id,
    }

    print(f"Saving FULL registration heatmap cache to: {ctx.cache_file_path}")
    os.makedirs(os.path.dirname(ctx.cache_file_path), exist_ok=True)
    atomic_write_json(ctx.cache_file_path, response_data_full, indent=2)
    if not ctx.full_resolution:
        save_display_sidecar_cache(ctx.cache_file_path, response_data_full)

    return strip_response_payload(response_data_full)


# ---------------------------------------------------------------------------
# Settings-based population heatmaps (QuickMesh viewer)
#
# The population error is computed on a *calculation surface* defined only by the
# scientific mesh settings the user chose in the refresh menu (edit, threshold,
# full/half resolution, Gaussian sigma, outer shell). The reference and every
# subject are meshed with exactly those settings and never decimated, so two runs
# with the same settings give the same numbers. The display mesh (decimated, and
# not identical from run to run) only *receives* those values: each displayed vertex
# takes the value of its nearest calculation-surface vertex on every load.
#
# Caches are stored per settings combination (stamp + key), so a heatmap is never
# silently reused for a mesh built with different settings; the viewer is told and
# asks the user to recalculate or reload the mesh with the stored settings.
# High quality vs regular refresh only changes display decimation and is not part
# of the stamp.
# ---------------------------------------------------------------------------

SETTINGS_CACHES_KEPT = 3  # per reference + metric; oldest settings caches are pruned
MAPPING_WARN_SPACINGS = 1.0  # warn when a displayed vertex is > 1 voxel from the calc surface


def _canonical_reference_edit(edit) -> Optional[str]:
    """Full-resolution edit stem (no ``_lossy``, no extension); None for the raw scan."""
    if edit is None:
        return None
    text = os.path.basename(str(edit)).strip()
    if not text or text.lower() in ("raw", "none", "null", "latest"):
        return None
    if text.lower().endswith(".ply"):
        return text
    return strip_nifti_basename(text).replace("_lossy", "")


def _round_sig(value, digits: int = 6) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return float(f"{value:.{digits}g}")


def normalize_calculation_settings(raw: dict, metric_id: str, is_preserved_mesh: bool) -> dict:
    """
    Canonical settings stamp. Only fields that change the calculated numbers are kept;
    voxel-only fields are dropped for preserved-mesh references (they have no volume).
    """
    raw = raw if isinstance(raw, dict) else {}
    sigma = _round_sig(raw.get("gaussian_blur"), 4)
    settings = {
        "metric": normalize_heatmap_metric(metric_id),
        "edit": _canonical_reference_edit(raw.get("edit")),
        "shell": bool(raw.get("shell_mode") or raw.get("shell")),
    }
    if not is_preserved_mesh:
        settings["threshold"] = _round_sig(raw.get("threshold"))
        settings["full_resolution"] = bool(raw.get("full_resolution"))
        settings["gaussian_blur"] = sigma if sigma and sigma > 0 else 0.0
    return settings


def calculation_settings_key(settings: dict) -> str:
    payload = json.dumps(settings, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


_SETTING_LABELS = {
    "edit": "edit",
    "threshold": "threshold",
    "full_resolution": "resolution",
    "gaussian_blur": "Gaussian σ",
    "shell": "shell",
    "metric": "metric",
}


def _format_setting(name: str, value) -> str:
    if name == "full_resolution":
        return "full" if value else "half"
    if name == "shell":
        return "on" if value else "off"
    if name == "edit":
        return value or "raw"
    return "—" if value is None else str(value)


def describe_settings_differences(stored: dict, current: dict) -> list:
    """Human-readable rows {setting, stored, current} for the viewer prompt."""
    rows = []
    for name in ("edit", "threshold", "full_resolution", "gaussian_blur", "shell", "metric"):
        if name not in stored and name not in current:
            continue
        if stored.get(name) != current.get(name):
            rows.append({
                "setting": _SETTING_LABELS[name],
                "stored": _format_setting(name, stored.get(name)),
                "current": _format_setting(name, current.get(name)),
            })
    return rows


def _settings_cache_base(directory: str, sub_path: str, filename: str, metric_id: str) -> str:
    legacy = registration_heatmap_cache_path(directory, sub_path, filename, metric_id)
    return legacy[:-5] if legacy.endswith(".json") else legacy


def _settings_cache_paths(base: str, key: str) -> dict:
    return {
        "meta": f"{base}_calc_{key}.json",
        "surface": f"{base}_calc_{key}.npz",
        "subjects": f"{base}_calc_{key}_subjects.npy",
    }


def _list_settings_caches(base: str) -> list:
    """Stored settings caches for this reference+metric, newest first: [(key, meta)]."""
    found = []
    for meta_path in glob.glob(f"{base}_calc_*.json"):
        try:
            with open(meta_path, "r") as jf:
                meta = json.load(jf)
            found.append((os.path.getmtime(meta_path), meta.get("key"), meta))
        except (OSError, ValueError):
            continue
    found.sort(key=lambda row: row[0], reverse=True)
    return [(key, meta) for _, key, meta in found if key]


def _prune_settings_caches(base: str, keep: int = SETTINGS_CACHES_KEPT) -> None:
    for key, _meta in _list_settings_caches(base)[keep:]:
        for path in _settings_cache_paths(base, key).values():
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError as exc:
                print(f"Warning: could not prune heatmap cache {path}: {exc}")


def _resolve_voxel_reference_volume(directory, sub_path, filename, metadata, settings):
    """
    Volume, iso-value and spacing exactly as /marchingcubes/ builds the reference display
    mesh for these settings (same file choice, lossy threshold mapping and spacing).
    """
    edit = settings.get("edit")
    full_res = settings.get("full_resolution")
    if full_res:
        name = f"{edit}.nii.gz" if edit else f"{filename}.nii.gz"
    elif edit and "_edit_" in edit:
        name = f"{edit.replace('_edit_', '_lossy_edit_', 1)}.nii.gz"
    elif edit:
        name = f"{edit}_lossy.nii.gz"
    else:
        name = f"{filename}_lossy.nii.gz"
    nifti_path = os.path.join(directory, sub_path, name)
    if not os.path.isfile(nifti_path):
        raise RegistrationHeatmapError(f"Reference volume not found: {name}", status_code=404)

    threshold = settings.get("threshold")
    if threshold is None:
        threshold = float(metadata.get("threshold") or 0.0)
    volume = _load_volume_swapped(nifti_path)

    resolution_factor = 1.0
    if not full_res:
        entry = _lossy_params_entry(metadata, name)
        scale = float(entry.get("scale_factor", 1.0) or 1.0)
        zero = float(entry.get("zero_point_shift", 0.0) or 0.0)
        resolution_factor = float(entry.get("resolution_factor", 1) or 1)
        threshold = (float(threshold) - zero) / scale
    iso = float(np.clip(float(threshold), np.min(volume), np.max(volume)))

    edit_voxel_size = voxel_size_for_edit_stem(
        metadata, name, filename, scan_dir=os.path.join(directory, sub_path)
    )
    spacing = float(edit_voxel_size) * resolution_factor
    return nifti_path, volume, iso, spacing


def _voxel_surface(volume, iso, spacing, sigma, shell):
    """Marching cubes with the display pipeline's blur/shell steps, without decimation."""
    if sigma and sigma > 0:
        volume = gaussian_filter(volume, sigma=sigma)
    vertices, faces, _, _ = measure.marching_cubes(
        volume, level=iso, spacing=(spacing, spacing, spacing)
    )
    faces = faces[:, ::-1]
    if shell:
        vertices, faces = alpaca.get_outer_mesh(vertices, faces)
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def build_reference_calculation_surface(directory, filename, sub_path, metadata, settings):
    """
    Returns dict with:
      coords        calculation-surface vertices in the display frame (for mapping)
      query         the same vertices in the frame subjects are measured in
      spacing       nominal sample spacing (mm), for the mapping quality check
      fixed_volume / fixed_spacing  reference intensities for intensity metrics (voxel only)
    """
    shell = bool(settings.get("shell"))
    if is_preserved_mesh_metadata(metadata):
        ply_path = _preserved_mesh_path(directory, sub_path, filename, metadata, settings.get("edit"))
        mesh_obj = _load_trimesh_as_single_mesh(ply_path)
        vertices = np.asarray(mesh_obj.vertices, dtype=np.float64)
        faces = np.asarray(mesh_obj.faces, dtype=np.int64)
        if shell:
            vertices, faces, _ = get_outer_mesh_with_mapping(vertices, faces)
        reg_mesh = preserved_mesh_reference_in_shared_mm(
            trimesh.Trimesh(vertices=vertices, faces=faces, process=False), metadata, directory
        )
        edges = vertices[faces[:, 1]] - vertices[faces[:, 0]] if len(faces) else np.zeros((1, 3))
        return {
            "coords": vertices,
            "query": np.asarray(reg_mesh.vertices, dtype=np.float64),
            "spacing": float(np.mean(np.linalg.norm(edges, axis=1))) or 1.0,
            "fixed_volume": None,
            "fixed_spacing": None,
        }

    nifti_path, volume, iso, spacing = _resolve_voxel_reference_volume(
        directory, sub_path, filename, metadata, settings
    )
    vertices, _faces = _voxel_surface(
        volume, iso, spacing, settings.get("gaussian_blur"), shell
    )
    print(
        f"Calculation surface for {filename}: {len(vertices):,} vertices from "
        f"{os.path.basename(nifti_path)} (iso={iso:.4f}, spacing={spacing:.4f}, "
        f"sigma={settings.get('gaussian_blur')}, shell={shell})"
    )
    return {
        "coords": vertices,
        "query": vertices,
        "spacing": spacing,
        # Intensity metrics sample unblurred intensities on the reference grid.
        "fixed_volume": volume,
        "fixed_spacing": spacing,
    }


def _subject_surface_for_settings(subject: ElasticSubject, settings: dict):
    """Subject surface meshed with the same resolution / blur / shell as the reference."""
    shell = bool(settings.get("shell"))
    if subject.modality != "voxel":
        mesh_obj = _load_trimesh_as_single_mesh(subject.edit_path)
        vertices = np.asarray(mesh_obj.vertices, dtype=np.float64)
        faces = np.asarray(mesh_obj.faces, dtype=np.int64)
        if shell:
            vertices, faces, _ = get_outer_mesh_with_mapping(vertices, faces)
        return vertices, faces

    edit_path = subject.edit_path
    spacing = float(subject.metadata.get("voxel_size") or 1.0)
    using_lossy = False
    if settings.get("full_resolution") is False:
        lossy = _lossy_companion_path(edit_path)
        if lossy:
            edit_path = lossy
            using_lossy = True
            spacing *= _lossy_resolution_factor(subject.metadata, os.path.basename(edit_path))
    volume = _load_volume_swapped(edit_path)
    # Each subject keeps its own stored threshold (intensities differ per scan).
    iso = _iso_threshold_for_volume(
        subject.metadata, volume, nifti_basename=os.path.basename(edit_path), is_lossy=using_lossy
    )
    return _voxel_surface(volume, iso, spacing, settings.get("gaussian_blur"), shell)


def _subject_values_for_settings(surface: dict, subject: ElasticSubject, settings: dict, metric_id: str):
    spec = get_heatmap_metric_spec(metric_id)
    if spec.compute_kind == "intensity":
        if subject.modality != "voxel" or surface["fixed_volume"] is None:
            raise ValueError(f"{metric_id} population heatmap skips mesh subject {subject.name}")
        moving_path = subject.edit_path
        if settings.get("full_resolution") is False:
            moving_path = _lossy_companion_path(subject.edit_path) or moving_path
        values = try_compute_intensity_heatmap(
            metric_id,
            surface["fixed_volume"],
            _load_volume_swapped(moving_path),
            surface["query"],
            surface["fixed_spacing"],
        )
        if values is None:
            raise ValueError(f"Metric '{metric_id}' is not an intensity heatmap")
        return np.asarray(values, dtype=np.float64)

    subject_vertices, subject_faces = _subject_surface_for_settings(subject, settings)
    return np.asarray(
        alpaca.calculate_landmark_distances(surface["query"], subject_vertices, subject_faces),
        dtype=np.float64,
    )


def decode_display_vertices(b64: str, center, scale_factor) -> np.ndarray:
    """Viewer positions (normalized GLB buffer) back to physical display-frame mm."""
    flat = np.frombuffer(base64.b64decode(b64), dtype="<f4").astype(np.float64)
    if flat.size % 3:
        raise RegistrationHeatmapError("Displayed vertex buffer is not a multiple of 3", 400)
    center = np.asarray(center if center is not None else [0, 0, 0], dtype=np.float64).reshape(1, 3)
    scale = float(scale_factor) if scale_factor else 1.0
    return flat.reshape(-1, 3) * scale + center


def _encode_int32(values: np.ndarray) -> str:
    return base64.b64encode(np.asarray(values, dtype="<i4").tobytes()).decode("ascii")


def _map_to_display(base, key, meta, display_vertices):
    paths = _settings_cache_paths(base, key)
    with np.load(paths["surface"], allow_pickle=False) as data:
        coords = np.asarray(data["coords"], dtype=np.float64)
        average = np.asarray(data["average"], dtype=np.float64)
    nn_dist, nn_idx = cKDTree(coords).query(display_vertices)
    spacing = float(meta.get("spacing_mm") or 1.0)
    max_dist = float(np.max(nn_dist)) if len(nn_dist) else 0.0
    mean_dist = float(np.mean(nn_dist)) if len(nn_dist) else 0.0
    warning = None
    if max_dist > MAPPING_WARN_SPACINGS * spacing:
        warning = (
            f"Some displayed vertices are up to {max_dist:.3f} mm from the calculation surface "
            f"(sample spacing {spacing:.3f} mm); their colours come from the nearest surface point."
        )
    print(
        f"Mapped {len(display_vertices):,} displayed vertices onto {len(coords):,} calculation "
        f"vertices (mean {mean_dist:.4f} mm, max {max_dist:.4f} mm)"
    )
    values = average[nn_idx]
    return {
        "distances": values.tolist(),
        "subject_distances": {"average": values.tolist()},
        "n_subjects_processed": int(meta.get("n_subjects_processed", 0)),
        "n_vertices_total": int(len(display_vertices)),
        "n_vertices_calculation": int(len(coords)),
        "has_full_subject_data": True,
        "metric": meta.get("metric", FALLBACK_HEATMAP_METRIC),
        "dense_distances_metric": meta.get("metric", FALLBACK_HEATMAP_METRIC),
        "cache_key": key,
        "calculation_settings": meta.get("settings"),
        "display_to_calc": _encode_int32(nn_idx),
        "mapping_mean_mm": mean_dist,
        "mapping_max_mm": max_dist,
        "mapping_warning": warning,
    }


# CRASH-SAFE PROTOCOL (see pipeline/crashSafety.py): every file this writes goes through
# an atomic_* helper, so a crash never leaves a half-written file. Subject data written
# here is covered by the transaction that the calling view or batch loop opens. Adding a
# new derivative file? Use an atomic_* helper and keep it inside that transaction.
def compute_settings_registration_heatmap(
    directory: str,
    filename: str,
    raw_settings: dict,
    display_vertices: np.ndarray,
    recalculate: bool = False,
    heatmap_metric: str = SURFACE_DISTANCE,
) -> dict:
    """
    Settings-stamped population heatmap mapped onto the displayed mesh.

    Without ``recalculate``: a cache for these exact settings is mapped and returned;
    if only caches for *other* settings exist, nothing is computed and a
    ``settings_mismatch`` payload lets the viewer ask the user what to do.
    """
    sub_path = resolve_sub_path(filename)
    with open(resolve_json_path(directory, filename, sub_path), "r") as jf:
        metadata = json.load(jf)
    is_mesh = is_preserved_mesh_metadata(metadata)
    metric_id = resolve_heatmap_metric(heatmap_metric, is_mesh=is_mesh)
    settings = normalize_calculation_settings(raw_settings, metric_id, is_mesh)
    if not is_mesh and str(settings.get("edit") or "").lower().endswith(".ply"):
        raise RegistrationHeatmapError(
            "Population heatmaps for a .ply edit of a voxel reference are not supported; "
            "select a volume edit of the reference.",
            status_code=400,
        )
    key = calculation_settings_key(settings)
    base = _settings_cache_base(directory, sub_path, filename, metric_id)
    paths = _settings_cache_paths(base, key)

    if not recalculate:
        if os.path.isfile(paths["meta"]) and os.path.isfile(paths["surface"]):
            with open(paths["meta"], "r") as jf:
                meta = json.load(jf)
            os.utime(paths["meta"], None)  # most recently used survives pruning
            print(f"Loaded settings heatmap cache {key} for {filename}")
            return _map_to_display(base, key, meta, display_vertices)
        others = _list_settings_caches(base)
        if others:
            stored = others[0][1].get("settings") or {}
            return {
                "settings_mismatch": True,
                "stored_settings": stored,
                "current_settings": settings,
                "differences": describe_settings_differences(stored, settings),
                "metric": metric_id,
            }

    elastic_subjects = discover_elastic_subjects(directory, filename)
    if not elastic_subjects:
        raise RegistrationHeatmapError("No valid elastic edits found in any subjects", status_code=404)

    started = time.time()
    surface = build_reference_calculation_surface(directory, filename, sub_path, metadata, settings)
    total = len(elastic_subjects)
    _send_progress(total, elastic_subjects[0].name, f"Computing {metric_id} population heatmap for {total} subjects...", 0)

    names, columns = [], []
    for idx, subject in enumerate(elastic_subjects):
        _send_progress(total, subject.name, f"Processing {subject.name} ({idx + 1}/{total}) [{metric_id}]...", idx + 1)
        try:
            values = _subject_values_for_settings(surface, subject, settings, metric_id)
        except Exception as exc:
            print(f"Error processing subject {subject.name}: {exc}")
            continue
        names.append(subject.name)
        columns.append(values.astype(np.float32))
        print(f"{metric_id} for {subject.name}: mean {float(np.mean(values)):.4f}")

    if not columns:
        raise RegistrationHeatmapError(
            f"No valid elastic subjects could be processed for metric '{metric_id}'", status_code=404
        )

    per_subject = np.stack(columns, axis=1)  # (n_calc, n_subjects), row = calc vertex
    average = np.nan_to_num(np.mean(per_subject.astype(np.float64), axis=1), nan=0, posinf=0, neginf=0)

    os.makedirs(os.path.dirname(paths["meta"]), exist_ok=True)
    atomic_np_savez(paths["surface"], coords=surface["coords"].astype(np.float32), average=average)
    atomic_np_save(paths["subjects"], per_subject)
    meta = {
        "key": key,
        "settings": settings,
        "metric": metric_id,
        "subjects": names,
        "n_subjects_processed": len(names),
        "n_vertices_calculation": int(len(average)),
        "spacing_mm": float(surface["spacing"]),
        "computed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "compute_seconds": round(time.time() - started, 1),
    }
    atomic_write_json(paths["meta"], meta, indent=2)
    _prune_settings_caches(base)
    print(f"Saved settings heatmap cache {key}: {len(names)} subjects, {len(average):,} vertices")
    return _map_to_display(base, key, meta, display_vertices)


def load_settings_subject_values(directory: str, filename: str, metric_id: str, key: str, calc_indices) -> dict:
    """Per-subject values at calculation-surface indices (histograms), via a memory map."""
    sub_path = resolve_sub_path(filename)
    base = _settings_cache_base(directory, sub_path, filename, normalize_heatmap_metric(metric_id))
    paths = _settings_cache_paths(base, key)
    if not (os.path.isfile(paths["meta"]) and os.path.isfile(paths["subjects"])):
        raise RegistrationHeatmapError("Registration cache not found", status_code=404)
    with open(paths["meta"], "r") as jf:
        meta = json.load(jf)
    matrix = np.load(paths["subjects"], mmap_mode="r")
    idx = np.asarray(calc_indices, dtype=np.int64)
    if idx.size and (idx.min() < 0 or idx.max() >= matrix.shape[0]):
        raise RegistrationHeatmapError("Calculation index out of range", status_code=400)
    rows = np.asarray(matrix[idx, :], dtype=np.float64)
    return {
        "subjects": meta.get("subjects", []),
        "values": rows,  # (len(idx), n_subjects)
        "metric": meta.get("metric", FALLBACK_HEATMAP_METRIC),
    }
