"""Shared coordinate frames for mixed mesh / voxel projects."""

from __future__ import annotations

import json
import os
import tempfile
import time

import numpy as np

MESH_MAX_VOXEL_CUBE = 512
MESH_REFERENCE_PRISM_PADDING_FRACTION = 0.1
PLY_REFERENCE_VOXEL_SIZE_KEY = "ply_reference_voxel_size"


def mesh_reference_prism_padding_mm(extent_mm):
    """Per-axis padding (mm) added on every side of a mesh-reference prism."""
    extent_mm = np.asarray(extent_mm, dtype=np.float64)
    return MESH_REFERENCE_PRISM_PADDING_FRACTION * np.maximum(extent_mm, 0.0)


def mesh_reference_prism_padding_scale():
    """Total extent multiplier after padding both sides of each axis."""
    return 1.0 + (2.0 * MESH_REFERENCE_PRISM_PADDING_FRACTION)


def swap_voxel_volume_xz(volume):
    """Reorient array axes 0↔2 between NIfTI storage (Z,Y,X) and display (X,Y,Z) layout."""
    return np.swapaxes(np.asarray(volume), 0, 2)


def project_settings_path(directory):
    return os.path.join(directory, "extracted", "project_settings.json")


def atomic_write_json(json_path, payload, retries=5, delay=0.05):
    """Write JSON via temp file + fsync + os.replace so a crash cannot truncate the target."""
    directory = os.path.dirname(json_path) or "."
    os.makedirs(directory, exist_ok=True)
    last_exc = None
    for attempt in range(max(1, int(retries))):
        fd, tmp_path = tempfile.mkstemp(
            prefix=".jsonwrite-",
            suffix=".tmp",
            dir=directory,
        )
        try:
            with os.fdopen(fd, "w") as jf:
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
                time.sleep(delay * (attempt + 1))
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
    if last_exc is not None:
        raise last_exc


def load_project_settings(directory):
    path = project_settings_path(directory)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r") as jf:
            settings = json.load(jf)
        return settings if isinstance(settings, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_project_settings(directory, settings):
    """Atomically persist the full project_settings.json payload."""
    path = project_settings_path(directory)
    payload = settings if isinstance(settings, dict) else {}
    atomic_write_json(path, payload)
    return path


PROJECT_LABEL_NAMES_KEY = "label_names"


def load_project_label_names(directory):
    """Return the project-wide { "<label_id>": "<custom name>" } map.

    The mapping lives in extracted/project_settings.json so it is shared by
    every subject/scan in the project.
    """
    value = load_project_settings(directory).get(PROJECT_LABEL_NAMES_KEY)
    return value if isinstance(value, dict) else {}


def save_project_label_name(directory, label, name):
    """Set (or clear) the project-wide custom name for a single label id.

    An empty/blank name removes the mapping so the label falls back to the
    default "Label <n>" display name. Returns the full updated mapping.
    """
    label_key = str(label)
    cleaned = (name or "").strip()

    settings = load_project_settings(directory)

    label_names = settings.get(PROJECT_LABEL_NAMES_KEY)
    if not isinstance(label_names, dict):
        label_names = {}

    if cleaned:
        label_names[label_key] = cleaned
    else:
        label_names.pop(label_key, None)

    settings[PROJECT_LABEL_NAMES_KEY] = label_names
    save_project_settings(directory, settings)
    return label_names


def load_ply_reference_voxel_size(directory):
    """Return persisted PLY-reference prism spacing from project_settings.json."""
    value = load_project_settings(directory).get(PLY_REFERENCE_VOXEL_SIZE_KEY)
    if isinstance(value, (int, float)) and float(value) > 0:
        return float(value)
    return None


def save_ply_reference_voxel_size(directory, voxel_size):
    """Persist PLY-reference prism spacing for future mesh-reference alignments."""
    spacing = float(voxel_size)
    if spacing <= 0:
        raise ValueError(f"ply_reference_voxel_size must be positive, got {spacing}")

    settings = load_project_settings(directory)
    settings[PLY_REFERENCE_VOXEL_SIZE_KEY] = spacing
    return save_project_settings(directory, settings)

def is_preserved_mesh_metadata(metadata):
    return metadata.get("is_mesh") is True and metadata.get("voxelized") is False


def is_voxel_based_metadata(metadata):
    return not is_preserved_mesh_metadata(metadata)


def load_extracted_scan_metadata(directory, scan_name):
    json_path = os.path.join(directory, "extracted", scan_name, f"{scan_name}.json")
    with open(json_path, "r") as jf:
        return json.load(jf)


def partition_voxel_and_preserved_mesh_scans(directory, scan_names):
    """Split subject names into voxel-eligible scans and preserved PLY meshes."""
    voxel_scan_names = []
    preserved_mesh_scans = []
    for scan_name in scan_names:
        try:
            metadata = load_extracted_scan_metadata(directory, scan_name)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            print(f"Skipping {scan_name}: could not read metadata ({exc})")
            continue
        if is_preserved_mesh_metadata(metadata):
            preserved_mesh_scans.append(scan_name)
        else:
            voxel_scan_names.append(scan_name)
    return voxel_scan_names, preserved_mesh_scans


def voxel_only_all_meshes_message(tool_name):
    return (
        f"{tool_name} only works with voxel-based volumes. "
        "This project currently contains only preserved PLY meshes."
    )


def voxel_only_no_eligible_targets_message(tool_name):
    return (
        f"No eligible voxel-based scans were found for {tool_name}. "
        "Preserved PLY meshes are skipped."
    )


def list_extracted_scan_names(directory):
    extracted_dir = os.path.join(directory, "extracted")
    if not os.path.isdir(extracted_dir):
        return []
    return sorted(
        name for name in os.listdir(extracted_dir)
        if os.path.isdir(os.path.join(extracted_dir, name))
    )


def native_voxel_size_mm(metadata):
    """
    Acquisition spacing for a voxel subject before mesh-reference resampling.

    Prefers ``alignment_previous_voxel_size`` so already-aligned subjects still
    report their native spacing, then scanner metadata, then current ``voxel_size``.
    """
    previous = metadata.get("alignment_previous_voxel_size")
    if isinstance(previous, (int, float)) and float(previous) > 0:
        return float(previous)

    scanner = metadata.get("scanner_metadata") or {}
    if isinstance(scanner, dict):
        resampling = scanner.get("anisotropic_resampling") or {}
        if isinstance(resampling, dict):
            voxel_sizes = resampling.get("voxel_sizes")
            if isinstance(voxel_sizes, (list, tuple)) and len(voxel_sizes) == 3:
                values = [float(value) for value in voxel_sizes if float(value) > 0]
                if values:
                    return float(np.mean(values))

    voxel_size = metadata.get("voxel_size")
    if isinstance(voxel_size, (int, float)) and float(voxel_size) > 0:
        return float(voxel_size)
    return None


def collect_project_native_voxel_sizes(directory, scan_names=None):
    """Native isotropic spacing (mm) for every non-faulty voxel subject in the project."""
    if scan_names is None:
        scan_names = list_extracted_scan_names(directory)

    sizes = []
    for scan_name in scan_names:
        try:
            metadata = load_extracted_scan_metadata(directory, scan_name)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            print(f"Skipping {scan_name} for cohort voxel size: {exc}")
            continue
        if metadata.get("faulty"):
            continue
        if is_preserved_mesh_metadata(metadata):
            continue
        native = native_voxel_size_mm(metadata)
        if native is not None and native > 0:
            sizes.append(native)
    return sizes


def cohort_median_voxel_size_mm(sizes, *, round_decimals=4):
    if not sizes:
        return None
    rounded = np.round(np.asarray(sizes, dtype=np.float64), int(round_decimals))
    return float(np.median(rounded))


def derive_mesh_reference_voxel_size(vertices, mesh_metadata=None):
    """
    Geometry-derived spacing for mesh voxelization at extract time.

    Fits the mesh inside ``MESH_MAX_VOXEL_CUBE`` with prism padding. Used when
    inventing a grid from PLY geometry alone, not for mixed-project registration.
    """
    mesh_metadata = mesh_metadata if isinstance(mesh_metadata, dict) else {}
    vertices = np.asarray(vertices, dtype=np.float64)
    if len(vertices) == 0:
        return 1.0

    extents = np.ptp(vertices, axis=0)
    max_extent = float(np.max(extents)) if len(extents) else 1.0
    centroid_size = mesh_metadata.get("centroid_size")
    if isinstance(centroid_size, (int, float)) and centroid_size > 0:
        avg_centroid_size = float(centroid_size)
    else:
        avg_centroid_size = max_extent / 2.0

    usable_dim = max(
        MESH_MAX_VOXEL_CUBE / mesh_reference_prism_padding_scale(),
        1.0,
    )
    return float(
        max(
            (avg_centroid_size * 2.0) / MESH_MAX_VOXEL_CUBE,
            max_extent / usable_dim,
            1e-6,
        )
    )


def resolve_mesh_reference_voxel_size(
    directory,
    vertices,
    mesh_metadata=None,
    scan_names=None,
):
    """
    Spacing for the mesh-reference registration prism in mixed projects.

    Resolution order:
    1. ``ply_reference_voxel_size`` in extracted/project_settings.json
    2. Median native voxel size across voxel subjects (persisted on first use)
    3. ``derive_mesh_reference_voxel_size`` when no voxel cohort exists yet
    """
    persisted = load_ply_reference_voxel_size(directory)
    if persisted is not None:
        print(
            f"Mesh-reference prism spacing {persisted:.6f} mm from "
            f"project_settings.{PLY_REFERENCE_VOXEL_SIZE_KEY}"
        )
        return persisted

    cohort = collect_project_native_voxel_sizes(directory, scan_names)
    if cohort:
        spacing = cohort_median_voxel_size_mm(cohort)
        if spacing is not None and spacing > 0:
            unique = sorted({round(float(value), 6) for value in cohort})
            save_ply_reference_voxel_size(directory, spacing)
            print(
                f"Mesh-reference prism spacing {spacing:.6f} mm from median of "
                f"{len(cohort)} voxel subject(s): {unique}; saved to project_settings"
            )
            return spacing

    spacing = derive_mesh_reference_voxel_size(vertices, mesh_metadata)
    print(
        f"Mesh-reference prism spacing {spacing:.6f} mm from mesh geometry "
        "(no voxel cohort in project)"
    )
    return spacing


def resolve_mesh_reference_prism(
    vertices,
    mesh_metadata,
    directory,
    scan_names=None,
):
    """Build the mesh-reference prism using cohort or geometry-derived spacing."""
    spacing = resolve_mesh_reference_voxel_size(
        directory,
        vertices,
        mesh_metadata,
        scan_names,
    )
    return build_mesh_reference_prism(vertices, mesh_metadata, voxel_size=spacing)


def build_mesh_reference_prism(vertices, mesh_metadata=None, voxel_size=None):
    """Imaginary voxel reference prism derived deterministically from mesh geometry.

    Pads by ``MESH_REFERENCE_PRISM_PADDING_FRACTION`` of each axis extent on every
    side so subjects that slightly exceed the reference bounds still land inside the
    shared grid.

    When ``voxel_size`` is omitted, spacing falls back to
    ``derive_mesh_reference_voxel_size`` (extract / geometry-only projects).
    Mixed mesh-reference registration should pass spacing from
    ``resolve_mesh_reference_voxel_size`` instead.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    mesh_metadata = mesh_metadata if isinstance(mesh_metadata, dict) else {}
    if voxel_size is None:
        voxel_size = derive_mesh_reference_voxel_size(vertices, mesh_metadata)

    voxel_size = float(voxel_size)

    if len(vertices) == 0:
        fallback_extent = np.full(3, voxel_size * 10.0, dtype=np.float64)
        padding_mm = mesh_reference_prism_padding_mm(fallback_extent)
        corner_origin_mm_local = -padding_mm
        padded_extent_mm = fallback_extent + (2.0 * padding_mm)
        shape_voxels = np.maximum(
            np.ceil(padded_extent_mm / voxel_size).astype(np.int64),
            1,
        )
    else:
        bounds_min = vertices.min(axis=0)
        bounds_max = vertices.max(axis=0)
        extent_mm = bounds_max - bounds_min
        padding_mm = mesh_reference_prism_padding_mm(extent_mm)
        corner_origin_mm_local = bounds_min - padding_mm
        padded_extent_mm = extent_mm + (2.0 * padding_mm)
        shape_voxels = np.maximum(
            np.ceil(padded_extent_mm / voxel_size).astype(np.int64),
            1,
        )

    return {
        "voxel_size": voxel_size,
        "padding_fraction": MESH_REFERENCE_PRISM_PADDING_FRACTION,
        "padding_mm": np.asarray(padding_mm, dtype=np.float64).tolist(),
        "corner_origin_mm_local": corner_origin_mm_local.tolist(),
        "shape_voxels": shape_voxels.tolist(),
    }


def mesh_reference_prism_corner_mm(
    vertices,
    mesh_metadata=None,
    *,
    directory=None,
    scan_names=None,
    voxel_size=None,
):
    if voxel_size is not None:
        prism = build_mesh_reference_prism(
            vertices, mesh_metadata, voxel_size=float(voxel_size)
        )
    elif directory is not None:
        prism = resolve_mesh_reference_prism(
            vertices, mesh_metadata, directory, scan_names
        )
    else:
        prism = build_mesh_reference_prism(vertices, mesh_metadata)
    return np.asarray(prism["corner_origin_mm_local"], dtype=np.float64)


def mesh_local_to_shared_mm(
    vertices,
    mesh_metadata=None,
    *,
    prism_vertices=None,
    directory=None,
    scan_names=None,
    voxel_size=None,
):
    """Map centroid-centered mesh-local mm into prism corner-origin mm."""
    corner = mesh_reference_prism_corner_mm(
        prism_vertices if prism_vertices is not None else vertices,
        mesh_metadata,
        directory=directory,
        scan_names=scan_names,
        voxel_size=voxel_size,
    )
    return np.asarray(vertices, dtype=np.float64) - corner


def mesh_centroid_local_to_reference_prism_mm(
    vertices,
    mesh_metadata,
    *,
    mesh_vertices_for_prism,
    reference_prism_corner_mm,
    directory=None,
    scan_names=None,
    voxel_size=None,
):
    """Map centroid-local mesh coordinates into another mesh reference's prism frame."""
    subj_corner = mesh_reference_prism_corner_mm(
        mesh_vertices_for_prism,
        mesh_metadata,
        directory=directory,
        scan_names=scan_names,
        voxel_size=voxel_size,
    )
    ref_corner = np.asarray(reference_prism_corner_mm, dtype=np.float64)
    return np.asarray(vertices, dtype=np.float64) + (subj_corner - ref_corner)


def overlay_vertices_in_reference_shared_mm(
    vertices,
    subject_metadata,
    reference_metadata,
    *,
    subject_name=None,
    reference_name=None,
    reference_prism_corner_mm=None,
    subject_mesh_vertices=None,
    directory=None,
    scan_names=None,
    voxel_size=None,
):
    """
    Express overlay vertices in the reference subject's shared corner-origin mm frame.

    Voxel subjects already live in corner-origin mm. After rigid alignment to the
    current reference, preserved-mesh PLY vertices are already stored in that same
    frame and are returned unchanged.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    reference_name = reference_name or reference_metadata.get("name")
    subject_name = subject_name or subject_metadata.get("name")

    if not is_preserved_mesh_metadata(subject_metadata):
        return vertices

    if subject_metadata.get("alignment_to") == reference_name:
        return vertices

    if not is_preserved_mesh_metadata(reference_metadata):
        return vertices

    ref_corner = np.asarray(reference_prism_corner_mm, dtype=np.float64)
    mesh_metadata = subject_metadata.get("mesh_metadata")
    prism_vertices = (
        subject_mesh_vertices if subject_mesh_vertices is not None else vertices
    )

    if subject_name == reference_name:
        return mesh_local_to_shared_mm(
            vertices,
            mesh_metadata,
            prism_vertices=prism_vertices,
            directory=directory,
            scan_names=scan_names,
            voxel_size=voxel_size,
        )

    return mesh_centroid_local_to_reference_prism_mm(
        vertices,
        mesh_metadata,
        mesh_vertices_for_prism=prism_vertices,
        reference_prism_corner_mm=ref_corner,
        directory=directory,
        scan_names=scan_names,
        voxel_size=voxel_size,
    )


def mesh_reorigin_offset_after_mm_crop(mins, padding_mm=0.0):
    """
    Translation that maps the clip-box minimum corner to ``padding_mm`` on each axis.

    After crop-all, preserved meshes and cropped voxel volumes both use this
    crop-relative frame so mixed overlays do not need a stored shared crop box.
    """
    padding_vec = np.full(3, float(padding_mm), dtype=np.float64)
    return padding_vec - np.asarray(mins, dtype=np.float64).reshape(3)


def reorigin_mesh_coords_after_mm_crop(positions_mm, mins, padding_mm=0.0):
    """Apply :func:`mesh_reorigin_offset_after_mm_crop` to Nx3 or length-3 mm coordinates."""
    positions_mm = np.asarray(positions_mm, dtype=np.float64)
    offset = mesh_reorigin_offset_after_mm_crop(mins, padding_mm)
    return positions_mm + offset
