import os
import tempfile
import unittest

import numpy as np
import trimesh

from pipeline.builtinTextureTools import (
    load_uv_sidecar,
    propagate_uvs_after_edit,
    save_uv_sidecar,
)
from pipeline.coordinateFrames import (
    mesh_reorigin_offset_after_mm_crop,
    reorigin_mesh_coords_after_mm_crop,
)


class PropagateUvsAfterEditTests(unittest.TestCase):
    def _make_textured_square(self):
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.0, 10.0, 0.0],
                [0.0, 10.0, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        face_uv = np.array(
            [
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            ],
            dtype=np.float32,
        )
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False), face_uv

    def test_crop_reorigin_offset_preserves_uvs(self):
        source_mesh, face_uv = self._make_textured_square()
        mins = np.array([2.0, 2.0, -1.0], dtype=np.float64)
        maxs = np.array([8.0, 8.0, 1.0], dtype=np.float64)
        padding = 0.0

        inside = np.all(
            (source_mesh.vertices[source_mesh.faces] >= mins)
            & (source_mesh.vertices[source_mesh.faces] <= maxs),
            axis=(1, 2),
        )
        cropped_faces = source_mesh.faces[inside]
        used_vertices = np.unique(cropped_faces.ravel())
        index_map = np.full(len(source_mesh.vertices), -1, dtype=np.int64)
        index_map[used_vertices] = np.arange(len(used_vertices))
        cropped_mesh = trimesh.Trimesh(
            vertices=source_mesh.vertices[used_vertices],
            faces=index_map[cropped_faces],
            process=False,
        )

        reorigin_offset = mesh_reorigin_offset_after_mm_crop(mins, padding)
        cropped_mesh.vertices = reorigin_mesh_coords_after_mm_crop(
            cropped_mesh.vertices,
            mins,
            padding,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = os.path.join(tmpdir, "source.ply")
            target_path = os.path.join(tmpdir, "target.ply")
            source_mesh.export(source_path)
            cropped_mesh.export(target_path)
            save_uv_sidecar(source_path, face_uv)

            propagate_uvs_after_edit(
                source_mesh,
                source_path,
                cropped_mesh,
                target_path,
                target_vertex_offset=reorigin_offset,
            )
            propagated = load_uv_sidecar(target_path)

        expected = face_uv[inside]
        self.assertIsNotNone(propagated)
        np.testing.assert_allclose(propagated, expected, rtol=1e-5, atol=1e-5)

    def test_rotation_keeps_uvs_when_topology_is_unchanged(self):
        source_mesh, face_uv = self._make_textured_square()
        rotation = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        center = source_mesh.vertices.mean(axis=0)
        rotated_vertices = (source_mesh.vertices - center) @ rotation + center
        rotated_mesh = trimesh.Trimesh(
            vertices=rotated_vertices,
            faces=np.asarray(source_mesh.faces, dtype=np.int64),
            process=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = os.path.join(tmpdir, "source.ply")
            target_path = os.path.join(tmpdir, "target.ply")
            source_mesh.export(source_path)
            rotated_mesh.export(target_path)
            save_uv_sidecar(source_path, face_uv)

            propagate_uvs_after_edit(source_mesh, source_path, rotated_mesh, target_path)
            propagated = load_uv_sidecar(target_path)

        np.testing.assert_allclose(propagated, face_uv, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
