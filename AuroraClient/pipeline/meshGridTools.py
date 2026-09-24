"""Grid preview generation for preserved PLY meshes (regular + quick grid)."""

from __future__ import annotations

import gc
import glob
import io
import json
import os
import re
import tempfile
import time

import numpy as np
import trimesh
from PIL import Image

from .registrationTools import (
    PROJECTION_VTK_UP_SIGN,
    PROJECTION_VTK_VIEW_SIGN,
    _compose_projection_rgba_from_render,
    _ensure_vtk_headless,
    _isometric_preview_rotation_matrix,
    _projection_back_file,
    save_projection_png,
)

QUADRANT_RENDER_SIZE = 512
PROJECTION_RENDER_SIZE = 768
GRID_PREVIEW_JPEG_QUALITY = 80

# Keys populated at bundle-load time from on-disk edit files — never persist in scan JSON.
EPHEMERAL_SCAN_METADATA_KEYS = ("edits", "edits_masks", "last_mesh_edit")


def strip_ephemeral_scan_metadata(metadata):
    """Remove bundle-only fields before writing scan JSON to disk."""
    if isinstance(metadata, dict):
        for key in EPHEMERAL_SCAN_METADATA_KEYS:
            metadata.pop(key, None)
    return metadata


def load_subject_json(json_path, retries=10, delay=0.03):
    """
    Load a subject JSON object, retrying empty/truncated files.

    Concurrent ``patch_subject_json`` used to truncate with ``open(..., 'w')``
    before dump finished; readers then saw ``Expecting value: line 1 column 1``.
    """
    last_err = None
    for attempt in range(max(1, int(retries))):
        try:
            if not os.path.isfile(json_path):
                return {}
            if os.path.getsize(json_path) == 0:
                last_err = json.JSONDecodeError('Expecting value', '', 0)
                time.sleep(delay * (attempt + 1))
                continue
            with open(json_path, 'r') as jf:
                data = json.load(jf)
            if isinstance(data, dict):
                return data
            last_err = ValueError('JSON root is not an object')
        except (json.JSONDecodeError, OSError) as exc:
            last_err = exc
            time.sleep(delay * (attempt + 1))
    if last_err:
        raise last_err
    return {}


def _format_os_error(exc):
    parts = [str(exc)]
    winerror = getattr(exc, 'winerror', None)
    errno_val = getattr(exc, 'errno', None)
    if winerror is not None:
        parts.append(f"winerror={winerror}")
    if errno_val is not None:
        parts.append(f"errno={errno_val}")
    return '; '.join(parts)


def _atomic_write_json(json_path, payload, retries=5, delay=0.05):
    directory = os.path.dirname(json_path) or '.'
    os.makedirs(directory, exist_ok=True)
    last_exc = None
    for attempt in range(max(1, int(retries))):
        fd, tmp_path = tempfile.mkstemp(
            prefix='.jsonwrite-',
            suffix='.tmp',
            dir=directory,
        )
        try:
            with os.fdopen(fd, 'w') as jf:
                json.dump(payload, jf, indent=4)
                jf.flush()
                try:
                    os.fsync(jf.fileno())
                except OSError:
                    pass
            try:
                os.replace(tmp_path, json_path)
                tmp_path = None
                return
            except OSError as exc:
                last_exc = exc
                print(
                    f"[jsonwrite] os.replace failed attempt {attempt + 1}/{retries}: "
                    f"{tmp_path!r} -> {json_path!r} ({_format_os_error(exc)})",
                    flush=True,
                )
                time.sleep(delay * (attempt + 1))
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
    if last_exc is not None:
        raise last_exc


def patch_subject_json(json_path, updates, metadata=None):
    """
    Reload subject JSON, apply ``updates``, and write atomically.

    The write always starts from disk so a later dump cannot clobber keys that
    another writer (notably ``save_as_lossy_nifti``) appended after ``metadata``
    was loaded. Optional ``metadata`` receives the same updates in memory and
    is synced to the on-disk ``lossy_compression`` list when present.
    """
    try:
        on_disk = load_subject_json(json_path)
    except (json.JSONDecodeError, OSError, ValueError):
        on_disk = {}
    if not isinstance(on_disk, dict):
        on_disk = {}

    if updates:
        on_disk.update(updates)
    strip_ephemeral_scan_metadata(on_disk)
    _atomic_write_json(json_path, on_disk)

    if isinstance(metadata, dict):
        if updates:
            metadata.update(updates)
        if 'lossy_compression' in on_disk:
            metadata['lossy_compression'] = on_disk['lossy_compression']
    return on_disk


def sort_edit_filenames(filenames):
    """Sort edit filenames by numeric edit index (edit_2 before edit_11)."""

    def sort_key(fn):
        match = re.search(r"_(?:lossy_)?edit_(\d+)_", fn)
        return (int(match.group(1)), fn) if match else (0, fn)

    if not filenames:
        return filenames
    return sorted(filenames, key=sort_key)


def list_scan_edit_filenames(scan_dir, scan_name, is_preserved_mesh):
    """Discover edit payloads on disk (PLY for meshes, lossy NIfTI for voxels)."""
    if not os.path.isdir(scan_dir):
        return []
    if is_preserved_mesh:
        return [
            name for name in os.listdir(scan_dir)
            if name.startswith(f"{scan_name}_edit_") and name.endswith(".ply")
        ]
    return [
        name for name in os.listdir(scan_dir)
        if name.startswith(f"{scan_name}_lossy_edit_") and name.endswith(".nii.gz")
    ]


def list_scan_edit_mask_filenames(scan_dir, scan_name):
    """Discover lossy mask sidecars for voxel edit files."""
    if not os.path.isdir(scan_dir):
        return []
    return [
        name for name in os.listdir(scan_dir)
        if name.startswith(f"{scan_name}_lossy_edit_") and name.endswith(".nii.mask.gz")
    ]


def scan_has_edit_substring(scan_dir, scan_name, is_preserved_mesh, substring):
    """True when any on-disk edit filename contains ``substring`` (e.g. ``backfixed``)."""
    return any(
        substring in name
        for name in list_scan_edit_filenames(scan_dir, scan_name, is_preserved_mesh)
    )


EDIT_PROVENANCE_CLEANUP = (
    (('n4corrected', 'n3corrected'), 'n4_bias_settings'),
    (('denoised',), 'denoise_settings'),
    (('restored',), 'restore_settings'),
    (('cleaned',), 'mesh_cleanup_settings'),
    (('welded',), 'mesh_weld_settings'),
    (('histmatched',), 'intensity_norm_settings'),
    (('backfixed',), 'background_offset_settings'),
)


def clear_orphaned_edit_provenance(metadata, scan_dir, scan_name, is_preserved_mesh):
    """Drop subject-level settings when no on-disk edit of that type remains."""
    if not isinstance(metadata, dict):
        return metadata
    for substrings, key in EDIT_PROVENANCE_CLEANUP:
        if any(scan_has_edit_substring(scan_dir, scan_name, is_preserved_mesh, sub) for sub in substrings):
            continue
        metadata[key] = None
    return metadata


def _load_trimesh_as_single_mesh(mesh_path):
    mesh_obj = trimesh.load(mesh_path, process=False)
    if isinstance(mesh_obj, trimesh.Scene):
        geometries = [
            geom for geom in mesh_obj.geometry.values()
            if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) > 0
        ]
        if not geometries:
            raise ValueError("No mesh geometry found in preserved PLY")
        mesh_obj = trimesh.util.concatenate(geometries)

    if not isinstance(mesh_obj, trimesh.Trimesh) or len(mesh_obj.vertices) == 0 or len(mesh_obj.faces) == 0:
        raise ValueError("Preserved mesh has no usable vertices or faces")
    return mesh_obj


def mesh_appears_unwelded(mesh):
    """
    True when faces look like a triangle soup (no shared vertex indices).

    Closed/shared-vertex surfaces have V ≈ F/2; per-face vertex triples have V ≈ 3F.
    Face projects treat each scan as one surface, so soups should be welded on save.
    """
    n_vertices = len(mesh.vertices)
    n_faces = len(mesh.faces)
    return n_faces > 0 and n_vertices >= n_faces * 2.5


def sanitize_trimesh_faces(mesh):
    """
    Drop duplicate/degenerate faces on both trimesh 4.x and 5.x.

    ``remove_duplicate_faces`` / ``remove_degenerate_faces`` were removed after
    the March 2024 deprecation; Windows installs may already be on 5.x while
    Linux still pins 4.5.1.
    """
    if mesh is None or len(getattr(mesh, "faces", [])) == 0:
        return mesh
    if hasattr(mesh, "remove_duplicate_faces"):
        mesh.remove_duplicate_faces()
    else:
        mesh.update_faces(mesh.unique_faces())
    if hasattr(mesh, "remove_degenerate_faces"):
        mesh.remove_degenerate_faces()
    else:
        mesh.update_faces(mesh.nondegenerate_faces())
    return mesh


def ensure_welded_trimesh(mesh):
    """
    Merge coincident vertices when ``mesh`` is stored as an unwelded triangle soup.

    Exported face PLYs often use one unique vertex triple per face. Connectivity-
    dependent tools (outer shell, hole healing, elastic registration) need shared
    indices along edges. Already-welded meshes are returned unchanged.
    """
    if mesh is None or len(getattr(mesh, "vertices", [])) == 0 or len(getattr(mesh, "faces", [])) == 0:
        return mesh
    if not mesh_appears_unwelded(mesh):
        return mesh

    n_before = int(len(mesh.vertices))
    n_faces_before = int(len(mesh.faces))
    welded = mesh.copy()
    welded.merge_vertices()
    sanitize_trimesh_faces(welded)
    welded.remove_unreferenced_vertices()
    welded = trimesh.Trimesh(
        vertices=np.asarray(welded.vertices, dtype=np.float64),
        faces=np.asarray(welded.faces, dtype=np.int64),
        process=False,
    )
    print(
        f"Welded coincident mesh vertices: {n_before} -> {len(welded.vertices)} "
        f"({n_faces_before} -> {len(welded.faces)} faces)"
    )
    return welded


def _mesh_connected_component_count(vertices, faces):
    """Number of face-connected components via undirected edge adjacency."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0:
        return 0
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    if len(edges) == 0:
        return int(len(vertices))
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    graph = csr_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, cols)),
        shape=(len(vertices), len(vertices)),
    )
    n_components, _ = connected_components(graph, directed=False)
    return int(n_components)


def proximity_weld_trimesh(mesh, distance_factor=1.0):
    """
    Merge vertices within ``distance_factor * median_edge_length`` **across different
    connected components** to stitch near-touching mesh fragments.

    Intra-component neighbors are left alone (they are already edge-connected).
    Returns ``(welded_mesh, stats_dict)``.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    if mesh is None or len(getattr(mesh, "vertices", [])) == 0 or len(getattr(mesh, "faces", [])) == 0:
        raise ValueError("Mesh has no vertices or faces")

    distance_factor = float(distance_factor)
    if not np.isfinite(distance_factor) or distance_factor <= 0:
        raise ValueError("distance_factor must be a positive number")

    working = ensure_welded_trimesh(mesh)
    vertices = np.asarray(working.vertices, dtype=np.float64)
    faces = np.asarray(working.faces, dtype=np.int64)
    n_vertices_before = int(len(vertices))
    n_faces_before = int(len(faces))

    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    edge_lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    median_edge = float(np.median(edge_lengths)) if len(edge_lengths) else 0.0
    epsilon = max(median_edge * distance_factor, 1e-12)

    if len(edges) == 0:
        labels = np.arange(n_vertices_before, dtype=np.int32)
        components_before = n_vertices_before
    else:
        rows = np.concatenate([edges[:, 0], edges[:, 1]])
        cols = np.concatenate([edges[:, 1], edges[:, 0]])
        graph = csr_matrix(
            (np.ones(len(rows), dtype=np.uint8), (rows, cols)),
            shape=(n_vertices_before, n_vertices_before),
        )
        components_before, labels = connected_components(graph, directed=False)
        components_before = int(components_before)

    empty_stats = {
        "weld_performed": False,
        "reason": "single_component",
        "distance_factor": distance_factor,
        "median_edge_length": median_edge,
        "epsilon": epsilon,
        "vertices_before": n_vertices_before,
        "vertices_after": n_vertices_before,
        "faces_before": n_faces_before,
        "faces_after": n_faces_before,
        "components_before": components_before,
        "components_after": components_before,
        "pairs_merged": 0,
    }
    if components_before <= 1:
        empty_stats["reason"] = "single_component"
        print(
            f"Proximity weld (factor={distance_factor:.3f}, eps={epsilon:.6g}): "
            f"no-op single_component (vertices={n_vertices_before}, faces={n_faces_before})",
            flush=True,
        )
        return working, empty_stats

    tree = cKDTree(vertices)
    pairs = tree.query_pairs(r=epsilon, output_type="ndarray")
    if pairs is None or len(pairs) == 0:
        empty_stats["reason"] = "no_vertices_within_threshold"
        print(
            f"Proximity weld (factor={distance_factor:.3f}, eps={epsilon:.6g}): "
            f"no-op no_vertices_within_threshold "
            f"(components={components_before}, vertices={n_vertices_before})",
            flush=True,
        )
        return working, empty_stats

    # Only bridge vertices that currently sit on different components.
    cross_mask = labels[pairs[:, 0]] != labels[pairs[:, 1]]
    cross_pairs = pairs[cross_mask]
    if len(cross_pairs) == 0:
        empty_stats["reason"] = "no_cross_component_pairs"
        print(
            f"Proximity weld (factor={distance_factor:.3f}, eps={epsilon:.6g}): "
            f"no-op no_cross_component_pairs "
            f"(components={components_before}, pairs={len(pairs)})",
            flush=True,
        )
        return working, empty_stats

    parent = np.arange(n_vertices_before, dtype=np.int64)

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left, right in cross_pairs:
        union(int(left), int(right))

    roots = np.fromiter((find(i) for i in range(n_vertices_before)), dtype=np.int64, count=n_vertices_before)
    unique_roots, inverse = np.unique(roots, return_inverse=True)
    new_vertices = np.zeros((len(unique_roots), 3), dtype=np.float64)
    counts = np.zeros(len(unique_roots), dtype=np.int64)
    np.add.at(new_vertices, inverse, vertices)
    np.add.at(counts, inverse, 1)
    new_vertices /= counts[:, None]

    new_faces = inverse[faces]
    non_degenerate = (
        (new_faces[:, 0] != new_faces[:, 1])
        & (new_faces[:, 1] != new_faces[:, 2])
        & (new_faces[:, 0] != new_faces[:, 2])
    )
    new_faces = new_faces[non_degenerate]

    welded = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
    sanitize_trimesh_faces(welded)
    welded.remove_unreferenced_vertices()
    welded = trimesh.Trimesh(
        vertices=np.asarray(welded.vertices, dtype=np.float64),
        faces=np.asarray(welded.faces, dtype=np.int64),
        process=False,
    )

    components_after = _mesh_connected_component_count(welded.vertices, welded.faces)
    weld_performed = components_after < components_before or int(len(welded.vertices)) < n_vertices_before
    print(
        f"Proximity weld (factor={distance_factor:.3f}, eps={epsilon:.6g}): "
        f"vertices {n_vertices_before} -> {len(welded.vertices)}, "
        f"faces {n_faces_before} -> {len(welded.faces)}, "
        f"components {components_before} -> {components_after}, "
        f"cross-pairs={len(cross_pairs)}",
        flush=True,
    )
    return welded, {
        "weld_performed": bool(weld_performed),
        "reason": "components_bridged" if weld_performed else "no_topology_change",
        "distance_factor": distance_factor,
        "median_edge_length": median_edge,
        "epsilon": epsilon,
        "vertices_before": n_vertices_before,
        "vertices_after": int(len(welded.vertices)),
        "faces_before": n_faces_before,
        "faces_after": int(len(welded.faces)),
        "components_before": components_before,
        "components_after": components_after,
        "pairs_merged": int(len(cross_pairs)),
    }


def load_centered_preserved_mesh(ply_path):
    mesh_obj = _load_trimesh_as_single_mesh(ply_path)
    vertices = np.asarray(mesh_obj.vertices, dtype=np.float64)
    faces = np.asarray(mesh_obj.faces, dtype=np.int64)
    center = vertices.mean(axis=0)
    vertices = vertices - center
    max_extent = float(np.max(np.abs(vertices)))
    if max_extent <= 0:
        max_extent = 1.0
    vertices = vertices / max_extent
    return vertices, faces, center, max_extent


def is_preserved_mesh_scan_dir(scan_dir, file_stem):
    json_path = os.path.join(scan_dir, f"{file_stem}.json")
    if not os.path.isfile(json_path):
        return False
    try:
        import json
        with open(json_path, 'r') as json_file:
            metadata = json.load(json_file)
        return metadata.get('is_mesh') is True and metadata.get('voxelized') is False
    except (OSError, ValueError, TypeError):
        return False


def _trimesh_to_pyvista(vertices, faces):
    import pyvista as pv

    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]).ravel()
    return pv.PolyData(vertices, faces_pv)


def _mesh_max_extent(vertices):
    return float(max(np.ptp(vertices, axis=0).max(), 0.001))


def _isometric_camera(plotter, center, max_extent, view_sign=PROJECTION_VTK_VIEW_SIGN):
    rotation_matrix = _isometric_preview_rotation_matrix()
    view_dir = rotation_matrix[:, 2]
    screen_up = rotation_matrix[:, 1]
    camera_distance = float(max(max_extent * 3.0, 2.5))
    plotter.camera.position = center + view_dir * camera_distance * view_sign
    plotter.camera.focal_point = center
    plotter.camera.up = screen_up * PROJECTION_VTK_UP_SIGN
    plotter.enable_parallel_projection()


def _orthographic_camera(plotter, center, max_extent, view_axis, up_vector):
    view_axis = np.asarray(view_axis, dtype=np.float64)
    up_vector = np.asarray(up_vector, dtype=np.float64)
    distance = float(max(max_extent * 3.0, 2.5))
    plotter.camera.position = center + view_axis * distance
    plotter.camera.focal_point = center
    plotter.camera.up = up_vector
    plotter.enable_parallel_projection()
    plotter.camera.parallel_scale = max_extent * 0.62


def _orient_rendered_image(rendered, orientation='rotate180'):
    """Correct PyVista off-screen screenshot orientation per view type."""
    if orientation == 'none':
        return rendered
    if orientation == 'flipud':
        return np.flipud(rendered)
    if orientation == 'fliplr':
        return np.fliplr(rendered)
    return np.rot90(rendered, 2)


def _render_mesh_screenshot(
    vertices,
    faces,
    camera_setup,
    window_size,
    transparent=False,
    image_orientation='rotate180',
):
    _ensure_vtk_headless()
    import pyvista as pv

    pv.OFF_SCREEN = True
    poly = _trimesh_to_pyvista(vertices, faces)
    plotter = pv.Plotter(off_screen=True, window_size=[window_size, window_size])
    if transparent:
        plotter.set_background([0.0, 0.0, 0.0, 0.0])
    else:
        plotter.set_background([0.0, 0.0, 0.0])

    plotter.add_mesh(
        poly,
        color='white',
        smooth_shading=True,
        ambient=0.35,
        diffuse=0.65,
        specular=0.1,
        show_scalar_bar=False,
    )

    center = np.zeros(3, dtype=np.float64)
    max_extent = _mesh_max_extent(vertices)
    camera_setup(plotter, center, max_extent)
    plotter.hide_axes()
    rendered = plotter.screenshot(return_img=True, transparent_background=transparent)
    plotter.close()
    return _orient_rendered_image(rendered, orientation=image_orientation)


def _resize_with_aspect_ratio(img, target_size, bg_color='black'):
    img_aspect = img.width / img.height
    target_aspect = target_size[0] / target_size[1]
    if img_aspect > target_aspect:
        new_width = target_size[0]
        new_height = int(new_width / img_aspect)
    else:
        new_height = target_size[1]
        new_width = int(new_height * img_aspect)
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    new_img = Image.new('RGB', target_size, bg_color)
    paste_x = (target_size[0] - new_width) // 2
    paste_y = (target_size[1] - new_height) // 2
    new_img.paste(img_resized, (paste_x, paste_y))
    return new_img


def _rgba_render_to_pil(rendered, transparent=False):
    if transparent:
        rgba = _compose_projection_rgba_from_render(rendered, flip_vertical=False)
        return Image.fromarray(rgba, 'RGBA')
    rgb = rendered[..., :3]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, 'RGB')


def mesh_projection_paths(ply_path):
    projection_path = ply_path.replace('.ply', '_projection.png')
    return projection_path, _projection_back_file(projection_path)


def mesh_projection_files_exist(ply_path):
    projection_path, back_projection_path = mesh_projection_paths(ply_path)
    return os.path.isfile(projection_path) and os.path.isfile(back_projection_path)


def resolve_mesh_projection_path(scan_dir, file_stem, edit=None, side='front'):
    """Locate an on-disk preserved-mesh projection PNG for the requested side."""
    projection_suffix = '_projection_back.png' if side == 'back' else '_projection.png'
    candidates = []

    if edit and edit.endswith('.ply'):
        candidates.append(os.path.join(scan_dir, edit.replace('.ply', projection_suffix)))

    projection_edit = resolve_preserved_mesh_projection_edit(scan_dir, file_stem)
    if projection_edit:
        candidates.append(os.path.join(scan_dir, projection_edit.replace('.ply', projection_suffix)))

    _, ply_edit_path = resolve_latest_preserved_mesh_edit(scan_dir, file_stem)
    if ply_edit_path:
        front_path, back_path = mesh_projection_paths(ply_edit_path)
        candidates.append(back_path if side == 'back' else front_path)

    return next((path for path in candidates if os.path.isfile(path)), None)




def mesh_projection_pair_up_to_date(ply_path):
    projection_path, back_projection_path = mesh_projection_paths(ply_path)
    if not os.path.isfile(projection_path) or not os.path.isfile(back_projection_path):
        return False
    try:
        ply_mtime = os.path.getmtime(ply_path)
        front_mtime = os.path.getmtime(projection_path)
        back_mtime = os.path.getmtime(back_projection_path)
    except OSError:
        return False
    return front_mtime >= ply_mtime and back_mtime >= ply_mtime


def ensure_mesh_projection_pngs(ply_path, *, force=False):
    if not force and mesh_projection_pair_up_to_date(ply_path):
        return False

    projection_path, back_projection_path = mesh_projection_paths(ply_path)
    vertices, faces, _, _ = load_centered_preserved_mesh(ply_path)
    try:
        front_render = _render_mesh_screenshot(
            vertices,
            faces,
            lambda plotter, center, max_extent: _isometric_camera(
                plotter, center, max_extent, view_sign=PROJECTION_VTK_VIEW_SIGN
            ),
            PROJECTION_RENDER_SIZE,
            transparent=True,
        )
        save_projection_png(
            _compose_projection_rgba_from_render(front_render, flip_vertical=False),
            projection_path,
        )

        back_render = _render_mesh_screenshot(
            vertices,
            faces,
            lambda plotter, center, max_extent: _isometric_camera(
                plotter, center, max_extent, view_sign=-PROJECTION_VTK_VIEW_SIGN
            ),
            PROJECTION_RENDER_SIZE,
            transparent=True,
        )
        save_projection_png(
            _compose_projection_rgba_from_render(back_render, flip_vertical=False),
            back_projection_path,
        )
    finally:
        vertices = faces = None
        gc.collect()

    return True


class PreservedMeshEditWriter:
    """Write a preserved-mesh PLY edit and regenerate its quick-grid projection PNG pair."""

    def __init__(self, mesh, output_path):
        self.mesh = mesh
        self.output_path = output_path

    @classmethod
    def save(cls, mesh, output_path):
        """Export ``mesh`` to ``output_path`` and write front/back projection PNGs."""
        return cls(mesh, output_path).write()

    def write(self):
        output_path = os.path.abspath(self.output_path)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        # Face / surface PLYs must share vertex indices along edges so downstream
        # tools see one connected surface rather than a triangle soup.
        self.mesh = ensure_welded_trimesh(self.mesh)
        self.mesh.export(output_path, file_type='ply')
        ensure_mesh_projection_pngs(output_path, force=True)
        return output_path


def generate_mesh_grid_preview_jpeg(ply_path):
    vertices, faces, _, _ = load_centered_preserved_mesh(ply_path)
    target_size = (QUADRANT_RENDER_SIZE, QUADRANT_RENDER_SIZE)

    try:
        axial_render = _render_mesh_screenshot(
            vertices,
            faces,
            lambda plotter, center, max_extent: _orthographic_camera(
                plotter, center, max_extent, [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]
            ),
            QUADRANT_RENDER_SIZE,
            transparent=False,
            image_orientation='fliplr',
        )
        isometric_render = _render_mesh_screenshot(
            vertices,
            faces,
            lambda plotter, center, max_extent: _isometric_camera(
                plotter, center, max_extent, view_sign=PROJECTION_VTK_VIEW_SIGN
            ),
            QUADRANT_RENDER_SIZE,
            transparent=False,
        )
        sagittal_render = _render_mesh_screenshot(
            vertices,
            faces,
            lambda plotter, center, max_extent: _orthographic_camera(
                plotter, center, max_extent, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]
            ),
            QUADRANT_RENDER_SIZE,
            transparent=False,
        )
        coronal_render = _render_mesh_screenshot(
            vertices,
            faces,
            lambda plotter, center, max_extent: _orthographic_camera(
                plotter, center, max_extent, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]
            ),
            QUADRANT_RENDER_SIZE,
            transparent=False,
        )

        img_q1 = _resize_with_aspect_ratio(_rgba_render_to_pil(axial_render), target_size)
        img_q2 = _resize_with_aspect_ratio(_rgba_render_to_pil(isometric_render), target_size)
        img_q3 = _resize_with_aspect_ratio(_rgba_render_to_pil(sagittal_render), target_size)
        img_q4 = _resize_with_aspect_ratio(_rgba_render_to_pil(coronal_render), target_size)

        grid_img = Image.new('RGB', (target_size[0] * 2, target_size[1] * 2))
        grid_img.paste(img_q1, (0, 0))
        grid_img.paste(img_q2, (target_size[0], 0))
        grid_img.paste(img_q3, (0, target_size[1]))
        grid_img.paste(img_q4, (target_size[0], target_size[1]))

        buf = io.BytesIO()
        grid_img.save(buf, format='JPEG', quality=GRID_PREVIEW_JPEG_QUALITY)
        buf.seek(0)
        return buf.read()
    finally:
        vertices = faces = None
        gc.collect()


def _is_elastic_file(filename):
    return 'elastic' in os.path.basename(filename).lower()


def resolve_latest_preserved_mesh_edit(scan_dir, file_stem, ignore_elastic=False):
    edit_paths = glob.glob(os.path.join(scan_dir, f"{file_stem}_edit_*_*.ply"))
    candidates = []
    for path in edit_paths:
        if ignore_elastic and _is_elastic_file(path):
            continue
        basename = os.path.basename(path)
        try:
            edit_number = int(basename.split('_edit_')[1].split('_')[0])
        except (IndexError, ValueError):
            continue
        candidates.append((edit_number, path))

    if candidates:
        latest_path = max(candidates, key=lambda item: item[0])[1]
        return os.path.basename(latest_path), latest_path

    original_path = os.path.join(scan_dir, f"{file_stem}.ply")
    if os.path.isfile(original_path):
        return f"{file_stem}.ply", original_path

    return '', None


def mesh_projection_pair_exists(scan_dir, edit):
    if not edit or not edit.endswith('.ply'):
        return False
    ply_path = os.path.join(scan_dir, edit)
    if not os.path.isfile(ply_path):
        return False
    return mesh_projection_files_exist(ply_path)


def resolve_preserved_mesh_projection_edit(scan_dir, file_stem):
    edit_paths = glob.glob(os.path.join(scan_dir, f"{file_stem}_edit_*_*.ply"))
    candidates = []
    for ply_path in edit_paths:
        if _is_elastic_file(ply_path):
            continue
        projection_path = ply_path.replace('.ply', '_projection.png')
        if not os.path.isfile(projection_path):
            continue
        basename = os.path.basename(ply_path)
        try:
            edit_number = int(basename.split('_edit_')[1].split('_')[0])
        except (IndexError, ValueError):
            continue
        candidates.append((edit_number, basename))

    if candidates:
        return max(candidates, key=lambda item: item[0])[1]

    base_ply = f"{file_stem}.ply"
    base_projection = os.path.join(scan_dir, f"{file_stem}_projection.png")
    if os.path.isfile(base_projection):
        return base_ply

    return ''
