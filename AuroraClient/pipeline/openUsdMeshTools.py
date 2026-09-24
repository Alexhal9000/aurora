"""
Load textured meshes from OpenUSD archives and extracted USD scene folders.

OpenUSD supports several on-disk layouts, including .usdz zip packages and
standalone .usdc/.usd/.usda stage files. Archive utilities often expand .usdz
into a folder with the same basename; both layouts are supported.
"""

import os
import shutil
import tempfile
import zipfile

import numpy as np
import trimesh

OPENUSD_ARCHIVE_EXTENSIONS = (".usdz",)
OPENUSD_STAGE_EXTENSIONS = (".usdc", ".usd", ".usda")


def is_openusd_mesh_source(path):
    if not path:
        return False
    path = os.path.abspath(path)
    if os.path.isfile(path):
        return path.lower().endswith(OPENUSD_ARCHIVE_EXTENSIONS + OPENUSD_STAGE_EXTENSIONS)
    if os.path.isdir(path):
        return find_usd_stage_in_directory(path) is not None
    return False


def find_usd_stage_in_directory(directory):
    directory = os.path.abspath(directory)
    if not os.path.isdir(directory):
        return None
    preferred = []
    fallback = []
    for entry in os.listdir(directory):
        lower = entry.lower()
        if lower.endswith(".usdc"):
            preferred.append(os.path.join(directory, entry))
        elif lower.endswith((".usd", ".usda")):
            fallback.append(os.path.join(directory, entry))
    if preferred:
        preferred.sort(key=len)
        return preferred[0]
    if fallback:
        fallback.sort(key=len)
        return fallback[0]
    return None


def resolve_openusd_asset_root(mesh_path):
    """
    Return (stage_path, asset_root) for an OpenUSD archive, stage file, or extracted folder.
    """
    mesh_path = os.path.abspath(mesh_path)
    if os.path.isdir(mesh_path):
        stage_path = find_usd_stage_in_directory(mesh_path)
        if stage_path is None:
            raise ValueError(f"No OpenUSD stage file found in directory: {mesh_path}")
        return stage_path, mesh_path
    if mesh_path.lower().endswith(OPENUSD_ARCHIVE_EXTENSIONS):
        return mesh_path, os.path.dirname(mesh_path)
    if mesh_path.lower().endswith(OPENUSD_STAGE_EXTENSIONS):
        return mesh_path, os.path.dirname(mesh_path)
    raise ValueError(f"Unsupported OpenUSD mesh source: {mesh_path}")


def _triangulate_faces(counts, indices):
    faces = []
    cursor = 0
    for count in counts:
        count = int(count)
        polygon = indices[cursor:cursor + count]
        cursor += count
        if count < 3:
            continue
        if count == 3:
            faces.append(polygon)
            continue
        for tri_index in range(1, count - 1):
            faces.append([polygon[0], polygon[tri_index], polygon[tri_index + 1]])
    if not faces:
        return np.zeros((0, 3), dtype=np.int64)
    return np.asarray(faces, dtype=np.int64)


def _triangulate_face_corner_values(counts, indices, corner_values):
    """Triangulate per-face-corner attributes in USD face-vertex order."""
    corner_values = np.asarray(corner_values, dtype=np.float64)
    face_values = []
    cursor = 0
    corner_cursor = 0
    for count in counts:
        count = int(count)
        polygon_values = corner_values[corner_cursor:corner_cursor + count]
        corner_cursor += count
        cursor += count
        if count < 3:
            continue
        if count == 3:
            face_values.append(polygon_values)
            continue
        for tri_index in range(1, count - 1):
            face_values.append([
                polygon_values[0],
                polygon_values[tri_index],
                polygon_values[tri_index + 1],
            ])
    if not face_values:
        return np.zeros((0, 3, corner_values.shape[-1]), dtype=np.float64)
    return np.stack(face_values, axis=0)


def _extract_openusd_face_corner_uvs(mesh_prim, counts, indices):
    """Return (n_faces, 3, 2) UVs aligned with triangulated mesh faces."""
    from pxr import UsdGeom

    primvars = UsdGeom.PrimvarsAPI(mesh_prim)
    st_primvar = primvars.GetPrimvar("st")
    if not st_primvar or not st_primvar.Get():
        return None

    values = np.asarray(st_primvar.Get(), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        return None

    prim_indices = st_primvar.GetIndices()
    if prim_indices:
        values = values[np.asarray(prim_indices, dtype=np.int64)]

    interpolation = st_primvar.GetInterpolation()
    counts = [int(count) for count in counts]
    indices = list(indices)
    expected_corners = int(sum(counts))

    if interpolation == UsdGeom.Tokens.faceVarying:
        if len(values) != expected_corners:
            return None
        corner_uvs = values
    else:
        corner_uvs = values[np.asarray(indices, dtype=np.int64)]
        if len(corner_uvs) != expected_corners:
            return None

    return _triangulate_face_corner_values(counts, indices, corner_uvs)


def load_openusd_face_corner_uvs(mesh_path, scale_factor=1.0):
    """
    Load OpenUSD geometry and per-triangle face-corner UVs without welding or centering.

    Returns (vertices, faces, face_uv) where face_uv has shape (n_faces, 3, 2).
    """
    from pxr import Gf, Usd, UsdGeom

    stage_path, _asset_root = resolve_openusd_asset_root(mesh_path)
    stage = Usd.Stage.Open(stage_path)

    vertices_parts = []
    faces_parts = []
    face_uv_parts = []
    vertex_offset = 0

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue

        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not points or not counts or not indices:
            continue

        local_vertices = np.asarray(points, dtype=np.float64)
        xformable = UsdGeom.Xformable(prim)
        local_to_world = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        local_vertices = np.asarray(
            [
                local_to_world.Transform(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
                for point in local_vertices
            ],
            dtype=np.float64,
        )

        faces = _triangulate_faces(counts, indices)
        if len(faces) == 0:
            continue

        face_uv = _extract_openusd_face_corner_uvs(mesh, counts, indices)

        vertices_parts.append(local_vertices)
        faces_parts.append(faces + vertex_offset)
        if face_uv is not None:
            face_uv_parts.append(face_uv)
        vertex_offset += len(local_vertices)

    if not vertices_parts:
        raise ValueError(f"No mesh geometry found in OpenUSD source: {mesh_path}")

    vertices = np.vstack(vertices_parts)
    faces = np.vstack(faces_parts)
    if scale_factor != 1.0:
        vertices = vertices * float(scale_factor)

    face_uv = None
    if face_uv_parts and len(face_uv_parts) == len(vertices_parts):
        face_uv = np.vstack(face_uv_parts).astype(np.float32)

    return vertices, faces, face_uv


def _usd_asset_path_to_relative(asset_path):
    raw = str(asset_path.path if hasattr(asset_path, "path") else asset_path).strip()
    return raw.strip("@")


def _find_usd_diffuse_texture_asset(stage):
    from pxr import UsdShade

    for prim in stage.Traverse():
        if not prim.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(prim)
        if shader.GetIdAttr().Get() != "UsdUVTexture":
            continue
        for shader_input in shader.GetInputs():
            if shader_input.GetBaseName() != "file":
                continue
            value = shader_input.Get()
            if value is None:
                continue
            relative = _usd_asset_path_to_relative(value)
            if relative:
                return relative
    return None


def resolve_openusd_texture_path(mesh_path, texture_asset=None, extract_dir=None):
    stage_path, asset_root = resolve_openusd_asset_root(mesh_path)
    if texture_asset is None:
        from pxr import Usd
        stage = Usd.Stage.Open(stage_path)
        texture_asset = _find_usd_diffuse_texture_asset(stage)
    if not texture_asset:
        return None

    relative = texture_asset.replace("\\", "/").lstrip("/")
    if os.path.isfile(stage_path) and stage_path.lower().endswith(".usdz"):
        with zipfile.ZipFile(stage_path) as archive:
            for member in archive.namelist():
                member_norm = member.replace("\\", "/").lstrip("/")
                if (
                    member_norm == relative
                    or member_norm.endswith("/" + relative)
                    or os.path.basename(member_norm) == os.path.basename(relative)
                ):
                    target_dir = extract_dir or tempfile.mkdtemp(prefix="aurora_openusd_texture_")
                    os.makedirs(target_dir, exist_ok=True)
                    target_path = os.path.join(target_dir, os.path.basename(member_norm))
                    if not os.path.isfile(target_path):
                        with archive.open(member) as source, open(target_path, "wb") as dest:
                            shutil.copyfileobj(source, dest)
                    return target_path
        return None

    candidate = os.path.join(asset_root, relative)
    if os.path.isfile(candidate):
        return candidate

    basename = os.path.basename(relative)
    for root, _dirs, files in os.walk(asset_root):
        for filename in files:
            if filename == basename:
                return os.path.join(root, filename)
    return None


def load_trimesh_from_openusd(mesh_path):
    vertices, faces, face_uv = load_openusd_face_corner_uvs(mesh_path)
    visual = None
    if face_uv is not None:
        texture_path = resolve_openusd_texture_path(mesh_path)
        material = trimesh.visual.material.SimpleMaterial(image=texture_path) if texture_path else None
        vertex_uv = np.zeros((len(vertices), 2), dtype=np.float64)
        for face_index, face in enumerate(faces):
            for corner, vertex_index in enumerate(face):
                vertex_uv[int(vertex_index)] = face_uv[face_index, corner]
        visual = trimesh.visual.TextureVisuals(uv=vertex_uv, material=material)

    return trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)
