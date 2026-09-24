"""
Built-in photogrammetry texture support for preserved meshes.

Texture images are stored once per subject; per-PLY-edit UV sidecars carry
face-corner UV coordinates that survive mesh edits via propagation.
"""

import base64
import glob
import os
import shutil
import tempfile

import numpy as np
import pygltflib
import trimesh
from scipy.spatial import cKDTree

BUILTIN_TEXTURE_SUFFIX = "_builtin_texture"
BUILTIN_UV_SUFFIX = "_builtin_uv"
TEXTURE_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp")
PLACEHOLDER_TEXTURE_MAX_PIXELS = 16


def _load_single_trimesh(mesh_path):
    from .openUsdMeshTools import is_openusd_mesh_source, load_trimesh_from_openusd

    if is_openusd_mesh_source(mesh_path):
        return load_trimesh_from_openusd(mesh_path)

    loaded = trimesh.load(mesh_path, process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = [
            geom for geom in loaded.geometry.values()
            if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) > 0
        ]
        if not geometries:
            return None
        loaded = trimesh.util.concatenate(geometries) if len(geometries) > 1 else geometries[0]
    if not isinstance(loaded, trimesh.Trimesh):
        return None
    return loaded


def _parse_obj_mtllib(mesh_path):
    mtllib_names = []
    try:
        with open(mesh_path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.startswith("mtllib "):
                    mtllib_names.extend(line.strip().split()[1:])
    except OSError:
        return []
    return mtllib_names


def _parse_mtl_map_kd(mtl_path):
    if not os.path.isfile(mtl_path):
        return None
    try:
        with open(mtl_path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped.lower().startswith("map_kd"):
                    parts = stripped.split(maxsplit=1)
                    if len(parts) == 2:
                        return parts[1].strip().strip('"').strip("'")
    except OSError:
        return None
    return None


def _resolve_texture_image_path(mesh_path, mesh_obj=None):
    from .openUsdMeshTools import is_openusd_mesh_source, resolve_openusd_texture_path

    if is_openusd_mesh_source(mesh_path):
        return resolve_openusd_texture_path(mesh_path)

    mesh_dir = os.path.dirname(os.path.abspath(mesh_path))
    candidates = []

    for mtllib_name in _parse_obj_mtllib(mesh_path):
        map_kd = _parse_mtl_map_kd(os.path.join(mesh_dir, mtllib_name))
        if map_kd:
            candidates.append(os.path.normpath(os.path.join(mesh_dir, map_kd)))

    if mesh_obj is not None and getattr(mesh_obj.visual, "kind", None) == "texture":
        material = mesh_obj.visual.material
        for attr in ("image_path", "file_path", "path"):
            value = getattr(material, attr, None)
            if isinstance(value, str) and value.strip():
                candidates.append(os.path.normpath(os.path.join(mesh_dir, value.strip())))

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if os.path.isfile(candidate):
            return candidate
    return None


def _texture_image_is_placeholder(image):
    if image is None:
        return True
    try:
        width, height = image.size
    except Exception:
        return True
    return width * height <= PLACEHOLDER_TEXTURE_MAX_PIXELS


def _mesh_has_uvs(mesh_obj):
    if mesh_obj is None or getattr(mesh_obj.visual, "kind", None) != "texture":
        return False
    uvs = getattr(mesh_obj.visual, "uv", None)
    if uvs is None:
        return False
    uvs = np.asarray(uvs, dtype=np.float64)
    return uvs.ndim == 2 and uvs.shape[1] == 2 and len(uvs) > 0


def builtin_texture_filename(scan_name, source_path):
    ext = os.path.splitext(source_path)[1].lower()
    if ext not in TEXTURE_IMAGE_EXTENSIONS:
        ext = ".jpg"
    return f"{scan_name}{BUILTIN_TEXTURE_SUFFIX}{ext}"


def uv_sidecar_path(ply_path):
    stem, _ext = os.path.splitext(os.path.abspath(ply_path))
    return f"{stem}{BUILTIN_UV_SUFFIX}.npz"


def extract_face_corner_uvs(mesh_obj):
    """Return (n_faces, 3, 2) UV array aligned with mesh.faces."""
    if not _mesh_has_uvs(mesh_obj):
        return None
    faces = np.asarray(mesh_obj.faces, dtype=np.int64)
    uvs = np.asarray(mesh_obj.visual.uv, dtype=np.float64)
    if faces.size == 0:
        return None
    if uvs.shape[0] <= int(faces.max()):
        return None
    return uvs[faces]


def save_uv_sidecar(ply_path, face_uvs):
    face_uvs = np.asarray(face_uvs, dtype=np.float32)
    if face_uvs.ndim != 3 or face_uvs.shape[1] != 3 or face_uvs.shape[2] != 2:
        raise ValueError("face_uvs must have shape (n_faces, 3, 2)")
    output_path = uv_sidecar_path(ply_path)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    np.savez(output_path, face_uv=face_uvs)
    return output_path


def load_uv_sidecar(ply_path):
    sidecar_path = uv_sidecar_path(ply_path)
    if not os.path.isfile(sidecar_path):
        return None
    try:
        data = np.load(sidecar_path)
        face_uv = np.asarray(data["face_uv"], dtype=np.float32)
        if face_uv.ndim != 3 or face_uv.shape[1] != 3 or face_uv.shape[2] != 2:
            return None
        return face_uv
    except Exception as exc:
        print(f"Warning: could not read UV sidecar {sidecar_path}: {exc}")
        return None


def copy_uv_sidecar(source_ply_path, target_ply_path):
    face_uv = load_uv_sidecar(source_ply_path)
    if face_uv is None:
        return None
    return save_uv_sidecar(target_ply_path, face_uv)


def invalidate_display_mesh_cache(ply_path):
    stem, _ext = os.path.splitext(os.path.abspath(ply_path))
    for pattern in (f"{stem}_display_mesh_vertices.npz", f"{stem}_display_mesh_faces.npz"):
        try:
            if os.path.isfile(pattern):
                os.remove(pattern)
        except OSError as exc:
            print(f"Warning: could not remove display cache {pattern}: {exc}")


def detect_and_extract_builtin_texture(mesh_path, output_dir, scan_name):
    """
    Detect a built-in diffuse texture on a source mesh and copy it into the extracted folder.

    Returns metadata dict when both UVs and a real texture image are available, else None.
    """
    mesh_path = os.path.abspath(mesh_path)
    if not (os.path.isfile(mesh_path) or os.path.isdir(mesh_path)):
        return None

    mesh_obj = _load_single_trimesh(mesh_path)
    if mesh_obj is None or not _mesh_has_uvs(mesh_obj):
        return None

    texture_source = _resolve_texture_image_path(mesh_path, mesh_obj)
    if texture_source is None:
        print(
            f"Built-in texture skipped for {scan_name}: UVs present but no texture image "
            f"resolved next to {mesh_path}"
        )
        return None

    try:
        from PIL import Image
        with Image.open(texture_source) as image:
            if _texture_image_is_placeholder(image):
                print(
                    f"Built-in texture skipped for {scan_name}: resolved image appears to be a "
                    f"placeholder ({image.size[0]}x{image.size[1]})"
                )
                return None
    except Exception as exc:
        print(f"Built-in texture skipped for {scan_name}: could not open texture image: {exc}")
        return None

    os.makedirs(output_dir, exist_ok=True)
    texture_filename = builtin_texture_filename(scan_name, texture_source)
    texture_dest = os.path.join(output_dir, texture_filename)
    shutil.copy2(texture_source, texture_dest)

    source_mtl = None
    source_format = None
    from .openUsdMeshTools import is_openusd_mesh_source

    if is_openusd_mesh_source(mesh_path):
        source_format = "openusd"
    elif os.path.isfile(mesh_path):
        mtllib_names = _parse_obj_mtllib(mesh_path)
        if mtllib_names:
            source_mtl = mtllib_names[0]

    metadata = {
        "available": True,
        "texture_file": texture_filename,
        "source_texture": os.path.basename(texture_source),
        "source_mtl": source_mtl,
    }
    if source_format:
        metadata["source_format"] = source_format
    return metadata


def enrich_builtin_texture_metadata(json_data, scan_dir, scan_name):
    """Fill builtin_texture metadata from on-disk assets when JSON is missing it."""
    if not isinstance(json_data, dict):
        return json_data
    existing = json_data.get("builtin_texture")
    if isinstance(existing, dict) and existing.get("available") is True:
        return json_data

    texture_candidates = glob.glob(
        os.path.join(scan_dir, f"{scan_name}{BUILTIN_TEXTURE_SUFFIX}.*")
    )
    texture_candidates = [
        path for path in texture_candidates
        if os.path.splitext(path)[1].lower() in TEXTURE_IMAGE_EXTENSIONS
    ]
    if not texture_candidates:
        return json_data

    base_ply = os.path.join(scan_dir, f"{scan_name}.ply")
    if load_uv_sidecar(base_ply) is None:
        return json_data

    texture_file = os.path.basename(texture_candidates[0])
    json_data["builtin_texture"] = {
        "available": True,
        "texture_file": texture_file,
    }
    return json_data


def _barycentric_weights(point, triangle):
    a, b, c = triangle
    v0 = b - a
    v1 = c - a
    v2 = point - a
    denom = np.dot(v0, v0) * np.dot(v1, v1) - np.dot(v0, v1) ** 2
    if abs(denom) < 1e-18:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    inv_denom = 1.0 / denom
    u = (np.dot(v1, v1) * np.dot(v2, v0) - np.dot(v0, v1) * np.dot(v2, v1)) * inv_denom
    v = (np.dot(v0, v0) * np.dot(v2, v1) - np.dot(v0, v1) * np.dot(v2, v0)) * inv_denom
    w = 1.0 - u - v
    return np.array([w, u, v], dtype=np.float64)


def _interpolate_face_uv(point, triangle, face_uv):
    weights = _barycentric_weights(point, triangle)
    return np.sum(face_uv * weights[:, np.newaxis], axis=0)


def propagate_uvs_after_edit(
    source_mesh,
    source_ply_path,
    target_mesh,
    target_ply_path,
    target_vertex_offset=None,
):
    """
    Transfer face-corner UVs from source mesh edit to target mesh edit.

    Uses closest-triangle barycentric interpolation so slice/crop topology changes keep
    a usable UV field. When topology is unchanged, UV corners are copied directly.

    ``target_vertex_offset`` undoes a rigid translation applied to ``target_mesh``
    vertices after the edit (for example crop re-origin) so UV lookup uses the same
    coordinate frame as ``source_mesh``.
    """
    source_face_uv = load_uv_sidecar(source_ply_path)
    if source_face_uv is None:
        return None

    source_faces = np.asarray(source_mesh.faces, dtype=np.int64)
    target_faces = np.asarray(target_mesh.faces, dtype=np.int64)
    source_vertices = np.asarray(source_mesh.vertices, dtype=np.float64)
    target_vertices = np.asarray(target_mesh.vertices, dtype=np.float64)
    if target_vertex_offset is not None:
        target_vertices = target_vertices - np.asarray(target_vertex_offset, dtype=np.float64).reshape(3)

    if source_faces.shape == target_faces.shape and np.array_equal(source_faces, target_faces):
        return save_uv_sidecar(target_ply_path, source_face_uv)

    source_triangles = source_vertices[source_faces]
    source_centers = source_triangles.mean(axis=1)
    tree = cKDTree(source_centers)

    target_triangles = target_vertices[target_faces]
    target_centers = target_triangles.mean(axis=1)
    _, nearest_face_indices = tree.query(target_centers)

    new_face_uv = np.zeros((len(target_faces), 3, 2), dtype=np.float32)
    for face_index, target_face in enumerate(target_faces):
        source_index = int(nearest_face_indices[face_index])
        source_triangle = source_vertices[source_faces[source_index]]
        source_uv = source_face_uv[source_index]
        target_triangle = target_vertices[target_face]
        for corner in range(3):
            new_face_uv[face_index, corner] = _interpolate_face_uv(
                target_triangle[corner],
                source_triangle,
                source_uv,
            )

    return save_uv_sidecar(target_ply_path, new_face_uv)


def remap_face_uvs_for_decimated_mesh(orig_vertices, orig_faces, face_uv, new_vertices, new_faces):
    """Build per-face-corner UVs for a decimated mesh."""
    orig_vertices = np.asarray(orig_vertices, dtype=np.float64)
    orig_faces = np.asarray(orig_faces, dtype=np.int64)
    face_uv = np.asarray(face_uv, dtype=np.float64)
    new_vertices = np.asarray(new_vertices, dtype=np.float64)
    new_faces = np.asarray(new_faces, dtype=np.int64)

    if orig_faces.shape[0] != face_uv.shape[0]:
        return None

    orig_triangles = orig_vertices[orig_faces]
    centers = orig_triangles.mean(axis=1)
    tree = cKDTree(centers)

    new_face_uv = np.zeros((len(new_faces), 3, 2), dtype=np.float32)
    for face_index, face in enumerate(new_faces):
        target_triangle = new_vertices[face]
        _, nearest_face = tree.query(target_triangle.mean(axis=0))
        source_triangle = orig_vertices[orig_faces[int(nearest_face)]]
        source_uv = face_uv[int(nearest_face)]
        for corner in range(3):
            new_face_uv[face_index, corner] = _interpolate_face_uv(
                target_triangle[corner],
                source_triangle,
                source_uv,
            )
    return new_face_uv


def remap_uvs_for_decimated_mesh(orig_vertices, orig_faces, face_uv, new_vertices, new_faces):
    """Map display-resolution vertices to per-vertex UVs via closest-triangle lookup."""
    orig_vertices = np.asarray(orig_vertices, dtype=np.float64)
    orig_faces = np.asarray(orig_faces, dtype=np.int64)
    face_uv = np.asarray(face_uv, dtype=np.float64)
    new_vertices = np.asarray(new_vertices, dtype=np.float64)
    new_faces = np.asarray(new_faces, dtype=np.int64)

    if orig_faces.shape[0] != face_uv.shape[0]:
        return None

    orig_triangles = orig_vertices[orig_faces]
    centers = orig_triangles.mean(axis=1)
    tree = cKDTree(centers)

    vertex_uvs = np.zeros((len(new_vertices), 2), dtype=np.float32)
    for vertex_index, point in enumerate(new_vertices):
        _, nearest_face = tree.query(point)
        source_triangle = orig_vertices[orig_faces[int(nearest_face)]]
        vertex_uvs[vertex_index] = _interpolate_face_uv(
            point,
            source_triangle,
            face_uv[int(nearest_face)],
        )
    return vertex_uvs


def _compute_unwelded_normals(vertices, faces):
    """Flat per-face normals expanded to unwelded corner vertices."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-12)
    normals = (normals / lengths).astype(np.float32)
    corner_normals = np.empty((len(vertices), 3), dtype=np.float32)
    corner_normals[faces[:, 0]] = normals
    corner_normals[faces[:, 1]] = normals
    corner_normals[faces[:, 2]] = normals
    return corner_normals


def face_uvs_to_unwelded_glb_arrays(vertices, faces, face_uv, flip_v=True):
    """
    Expand face-corner UVs into GLB-friendly unwelded vertex/index buffers.

    Returns (vertices, faces, uvs) where each face corner is a unique vertex so UV seams
    are preserved.
    """
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    face_uv = np.asarray(face_uv, dtype=np.float32)
    if faces.shape[0] != face_uv.shape[0]:
        raise ValueError("faces and face_uv must have the same number of faces")

    corner_vertices = vertices[faces].reshape(-1, 3)
    corner_uvs = face_uv.reshape(-1, 2).astype(np.float32, copy=True)
    if flip_v:
        corner_uvs[:, 1] = 1.0 - corner_uvs[:, 1]

    corner_faces = np.arange(len(corner_vertices), dtype=np.uint32).reshape(-1, 3)
    return corner_vertices, corner_faces, corner_uvs


def build_glb_with_uvs(vertices, faces, uvs, flip_v=True):
    """Build a base64-encoded GLB with POSITION, NORMAL, TEXCOORD_0, and indices."""
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.uint32)
    uvs = np.asarray(uvs, dtype=np.float32)
    if flip_v:
        uvs = uvs.copy()
        uvs[:, 1] = 1.0 - uvs[:, 1]

    normals = _compute_unwelded_normals(vertices, faces)

    vertex_data = vertices.tobytes()
    normal_data = normals.tobytes()
    uv_data = uvs.astype(np.float32).tobytes()
    face_data = faces.tobytes()
    buffer_data = vertex_data + normal_data + uv_data + face_data

    gltf = pygltflib.GLTF2()
    gltf.buffers.append(pygltflib.Buffer(byteLength=len(buffer_data), uri=None))
    gltf.set_binary_blob(buffer_data)

    normal_offset = len(vertex_data)
    uv_offset = normal_offset + len(normal_data)
    face_offset = uv_offset + len(uv_data)

    gltf.bufferViews.extend([
        pygltflib.BufferView(
            buffer=0, byteOffset=0, byteLength=len(vertex_data), target=pygltflib.ARRAY_BUFFER
        ),
        pygltflib.BufferView(
            buffer=0, byteOffset=normal_offset, byteLength=len(normal_data), target=pygltflib.ARRAY_BUFFER
        ),
        pygltflib.BufferView(
            buffer=0, byteOffset=uv_offset, byteLength=len(uv_data), target=pygltflib.ARRAY_BUFFER
        ),
        pygltflib.BufferView(
            buffer=0, byteOffset=face_offset, byteLength=len(face_data), target=pygltflib.ELEMENT_ARRAY_BUFFER
        ),
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
            componentType=pygltflib.FLOAT,
            count=len(normals),
            type=pygltflib.VEC3,
            max=normals.max(axis=0).tolist(),
            min=normals.min(axis=0).tolist(),
        ),
        pygltflib.Accessor(
            bufferView=2,
            componentType=pygltflib.FLOAT,
            count=len(uvs),
            type=pygltflib.VEC2,
            max=uvs.max(axis=0).tolist(),
            min=uvs.min(axis=0).tolist(),
        ),
        pygltflib.Accessor(
            bufferView=3,
            componentType=pygltflib.UNSIGNED_INT,
            count=len(faces.flatten()),
            type=pygltflib.SCALAR,
        ),
    ])
    gltf.meshes.append(pygltflib.Mesh(
        primitives=[pygltflib.Primitive(
            attributes=pygltflib.Attributes(POSITION=0, NORMAL=1, TEXCOORD_0=2),
            indices=3,
        )]
    ))
    gltf.nodes.append(pygltflib.Node(mesh=0))
    gltf.scenes.append(pygltflib.Scene(nodes=[0]))
    gltf.scene = 0

    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
        gltf.save_binary(tmp.name)
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as handle:
            return base64.b64encode(handle.read()).decode("utf-8")
    finally:
        try:
            os.unlink(tmp_path)
        except (OSError, PermissionError):
            pass


def build_glb_from_face_uvs(vertices, faces, face_uv, flip_v=True):
    """Unweld UV seams and emit a textured GLB payload."""
    unwelded_vertices, unwelded_faces, unwelded_uvs = face_uvs_to_unwelded_glb_arrays(
        vertices, faces, face_uv, flip_v=flip_v
    )
    return build_glb_with_uvs(unwelded_vertices, unwelded_faces, unwelded_uvs, flip_v=False)
